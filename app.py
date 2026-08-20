"""
VectorDB — FastAPI Server (v4.0.0 — Phase 5)
=============================================
Python vector database engine & introspection studio.
Serves the web UI at http://localhost:8080 and exposes a full REST API.

Phase 5 additions:
  - GET /items/3d                   — 3D PCA coordinates for interactive visualizer
  - GET /hnsw/trace                 — step-by-step greedy search trajectory across layers
  - GET /hnsw/topology              — full multi-layer HNSW graph topology
  - GET /analytics/clusters         — 3D cluster centroids, radii & variances
  - GET /analytics/metric-compare   — Cosine vs Euclidean vs Manhattan ranking comparison
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from vectordb.chunker import chunk_text
from vectordb.demo_data import DEMO_VECTORS
from vectordb.document_db import DocumentDB
from vectordb.grounding import ground_answer
from vectordb.metrics import cosine, get_dist_fn
from vectordb.ollama_client import OllamaClient
from vectordb.persistence import (
    load_document_db,
    load_vector_db,
    save_document_db,
    save_vector_db,
    snapshot_info,
)
from vectordb.reranker import rerank_candidates
from vectordb.semantic_chunker import preview_semantic_chunks, semantic_chunk_text
from vectordb.vector_db import VectorDB

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Constants & Singletons ─────────────────────────────────────────────────────

DIMS = 16
DATA_DIR = Path(__file__).parent / "data"
VECTOR_SNAPSHOT = DATA_DIR / "vectordb_index.json"
DOCUMENT_SNAPSHOT = DATA_DIR / "document_index.json"

db = VectorDB(dims=DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()


# ── Lifespan (startup + shutdown) ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Startup:  load persisted snapshots (if they exist) then seed demo vectors.
    Shutdown: auto-save both indexes to disk.
    """
    # ── Startup ───────────────────────────────────────────────────────────────

    loaded_docs = load_document_db(doc_db, DOCUMENT_SNAPSHOT)

    dist_fn = get_dist_fn("cosine")
    for metadata, category, emb in DEMO_VECTORS:
        db.insert(metadata, category, emb, dist_fn)

    if VECTOR_SNAPSHOT.exists():
        import json
        try:
            payload = json.loads(VECTOR_SNAPSHOT.read_text(encoding="utf-8"))
            for item in payload.get("items", []):
                demo_metas = {d[0] for d in DEMO_VECTORS}
                if item["metadata"] not in demo_metas:
                    db.insert(item["metadata"], item["category"], item["emb"], dist_fn)
                    logger.info("[Startup] Restored user demo vector: %s", item["metadata"])
        except Exception as exc:
            logger.warning("[Startup] Could not restore extra demo vectors: %s", exc)

    ollama_status = "ONLINE" if ollama.is_available() else "OFFLINE"
    print("\n=== VectorDB Engine (Phase 5 — Visualization & Introspection) ===")
    print("http://localhost:8080")
    print(f"{len(db)} demo vectors | {DIMS} dims | HNSW + KD-Tree + BruteForce + SQ8")
    print(f"Documents loaded: {loaded_docs} chunks | Hybrid Search (BM25 + HNSW + RRF)")
    print("Introspection: 3D Manifold | Stacked HNSW Graph | Trajectory Tracer | Metric Geometry")
    print(f"Ollama: {ollama_status}")
    if ollama_status == "ONLINE":
        print(f"  embed: {ollama.embed_model}  gen: {ollama.gen_model}")
    print()

    yield  # ── server is running ─────────────────────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────────────
    print("\n[Shutdown] Saving indexes to disk…")
    n_vec = save_vector_db(db, VECTOR_SNAPSHOT)
    n_doc = save_document_db(doc_db, DOCUMENT_SNAPSHOT)
    print(f"[Shutdown] Saved {n_vec} demo vectors and {n_doc} document chunks.")


# ── App & Middleware ───────────────────────────────────────────────────────────

app = FastAPI(title="VectorDB", version="4.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ── Static UI ──────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_ui():
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(html_path, media_type="text/html")


# ── Pydantic Request Models ────────────────────────────────────────────────────

class InsertBody(BaseModel):
    metadata: str
    category: str = ""
    embedding: List[float]


class DocInsertBody(BaseModel):
    title: str
    text: str
    chunking_strategy: str = "fixed"       # "fixed" or "semantic"
    threshold_percentile: float = 75.0


class DocAskBody(BaseModel):
    question: str
    k: int = 3


class HybridSearchBody(BaseModel):
    question: str
    k: int = 3
    rrf_k: int = 60


class AdvancedAskBody(BaseModel):
    question: str
    k: int = 3
    pipeline: str = "vector"               # "vector", "hybrid", "rerank", "hyde"
    rerank_strategy: str = "cross"         # "cross" or "llm"
    grounding: bool = True


class SemanticChunkPreviewBody(BaseModel):
    text: str
    threshold_percentile: float = 75.0


class RerankTestBody(BaseModel):
    query: str
    k: int = 5
    strategy: str = "cross"


# ══════════════════════════════════════════════════════════════════════════════
# DEMO VECTOR ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/search")
def search(
    v: str = Query(..., description="Comma-separated floats, must be 16D"),
    k: int = Query(5),
    metric: str = Query("cosine"),
    algo: str = Query("hnsw"),
    category: Optional[str] = Query(None, description="Filter results to this category"),
):
    query = _parse_vec(v)
    if len(query) != DIMS:
        raise HTTPException(400, f"Expected {DIMS}-dimensional vector, got {len(query)}D")

    result = db.search_filtered(query, k, metric, algo, category_filter=category)
    return {
        "results": [
            {
                "id": h.id,
                "metadata": h.metadata,
                "category": h.category,
                "distance": round(h.dist, 6),
                "embedding": h.emb,
            }
            for h in result.hits
        ],
        "latencyUs": result.latency_us,
        "algo": result.algo,
        "metric": result.metric,
        "categoryFilter": category,
    }


@app.post("/insert")
def insert(body: InsertBody):
    if len(body.embedding) != DIMS:
        raise HTTPException(400, f"Expected {DIMS}-dimensional vector")
    id_ = db.insert(body.metadata, body.category, body.embedding, cosine)
    return {"id": id_}


@app.delete("/delete/{item_id}")
def delete(item_id: int):
    ok = db.remove(item_id)
    return {"ok": ok}


@app.get("/items")
def items():
    return [
        {
            "id": v.id,
            "metadata": v.metadata,
            "category": v.category,
            "embedding": v.emb,
        }
        for v in db.all_items()
    ]


@app.get("/items/3d")
def items_3d():
    """Phase 5: Returns all items with 3D PCA coordinates."""
    return db.items_3d()


@app.get("/categories")
def categories():
    return {"categories": db.categories()}


@app.get("/benchmark")
def benchmark(
    v: str = Query(...),
    k: int = Query(5),
    metric: str = Query("cosine"),
):
    query = _parse_vec(v)
    if len(query) != DIMS:
        raise HTTPException(400, f"Expected {DIMS}-dimensional vector")
    b = db.benchmark(query, k, metric)
    return {
        "bruteforceUs": b.bruteforce_us,
        "kdtreeUs": b.kdtree_us,
        "hnswUs": b.hnsw_us,
        "itemCount": b.item_count,
    }


@app.get("/hnsw-info")
def hnsw_info():
    return db.hnsw_info()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: INTROSPECTION & TRAJECTORY TRACING ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/hnsw/trace")
def hnsw_trace(
    v: str = Query(...),
    k: int = Query(5),
    metric: str = Query("cosine"),
    ef: int = Query(50),
):
    """
    Phase 5: Records the step-by-step traversal path of HNSW search
    from the entry point through all graph layers down to Layer 0.
    """
    query = _parse_vec(v)
    if len(query) != DIMS:
        raise HTTPException(400, f"Expected {DIMS}-dimensional vector")
    return db.search_with_trace(query, k, metric=metric, ef=ef)


@app.get("/hnsw/topology")
def hnsw_topology():
    """
    Phase 5: Returns full multi-layer HNSW graph topology with 3D coordinates.
    """
    info = db.hnsw_info()
    items_3d_data = db.items_3d()
    coords_map = {it["id"]: it["coords3d"] for it in items_3d_data}

    # Enrich nodes with 3D coords
    for n in info.get("nodes", []):
        n["coords3d"] = coords_map.get(n["id"], [0.0, 0.0, 0.0])

    return info


@app.get("/analytics/clusters")
def analytics_clusters():
    """
    Phase 5: Computes 3D cluster centroids, radii, and intra-cluster variances.
    """
    return db.cluster_analytics()


@app.get("/analytics/metric-compare")
def analytics_metric_compare(
    v: str = Query(...),
    k: int = Query(5),
):
    """
    Phase 5: Compares neighborhood ranking across Cosine, Euclidean, and Manhattan metrics.
    """
    query = _parse_vec(v)
    if len(query) != DIMS:
        raise HTTPException(400, f"Expected {DIMS}-dimensional vector")
    return db.compare_metrics(query, k=k)


@app.get("/stats")
def stats():
    """Extended statistics including Phase 3, 4 & 5 capabilities."""
    sq8 = db.sq8_stats()
    doc_sq8 = doc_db.sq8_stats()
    bm25 = doc_db.bm25_stats()
    return {
        "count": len(db),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
        "sq8": {
            "compressionRatio": sq8.get("compressionRatio", 4.0),
            "savedBytes": sq8.get("savedBytes", 0),
            "float32Bytes": sq8.get("float32Bytes", 0),
            "int8Bytes": sq8.get("int8Bytes", 0),
        },
        "docIndex": {
            "chunkCount": len(doc_db),
            "bm25DocCount": bm25["docCount"],
            "bm25TokenCount": bm25["tokenCount"],
            "sq8CompressionRatio": doc_sq8.get("compressionRatio", 4.0),
            "sq8SavedBytes": doc_sq8.get("savedBytes", 0),
        },
        "phase5": {
            "visualizer3D": True,
            "hnswTrajectoryTracer": True,
            "stackedGraphInspector": True,
            "metricSpaceGeometry": True,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/persist/save")
def persist_save():
    n_vec = save_vector_db(db, VECTOR_SNAPSHOT)
    n_doc = save_document_db(doc_db, DOCUMENT_SNAPSHOT)
    return {
        "ok": True,
        "savedVectors": n_vec,
        "savedDocs": n_doc,
        "vectorSnapshot": str(VECTOR_SNAPSHOT),
        "documentSnapshot": str(DOCUMENT_SNAPSHOT),
    }


@app.get("/persist/status")
def persist_status():
    return {
        "vectorSnapshot": snapshot_info(VECTOR_SNAPSHOT),
        "documentSnapshot": snapshot_info(DOCUMENT_SNAPSHOT),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT & ADVANCED RAG ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/doc/insert")
def doc_insert(body: DocInsertBody):
    if not body.title or not body.text:
        raise HTTPException(400, "title and text are required")

    if body.chunking_strategy == "semantic":
        chunks = semantic_chunk_text(
            body.text,
            embed_fn=ollama.embed,
            threshold_percentile=body.threshold_percentile,
        )
    else:
        chunks = chunk_text(body.text, chunk_words=250, overlap_words=30)

    if not chunks:
        raise HTTPException(400, "Could not generate chunks from provided text")

    ids = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            raise HTTPException(
                503,
                "Ollama unavailable. Run: ollama serve && ollama pull nomic-embed-text",
            )
        suffix = f" [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else ""
        strategy_tag = f" ({body.chunking_strategy})"
        title = f"{body.title}{suffix}{strategy_tag}"
        ids.append(doc_db.insert(title, chunk, emb))

    return {
        "ids": ids,
        "chunks": len(chunks),
        "strategy": body.chunking_strategy,
        "dims": doc_db.get_dims(),
    }


@app.post("/doc/semantic-chunk-preview")
def doc_semantic_chunk_preview(body: SemanticChunkPreviewBody):
    if not body.text:
        raise HTTPException(400, "text is required")
    if not ollama.is_available():
        raise HTTPException(503, "Ollama is required for semantic chunking embeddings.")

    preview = preview_semantic_chunks(
        body.text,
        embed_fn=ollama.embed,
        threshold_percentile=body.threshold_percentile,
    )
    return preview


@app.get("/doc/list")
def doc_list():
    docs = doc_db.all_items()
    return [
        {
            "id": d.id,
            "title": d.title,
            "preview": d.text[:120] + ("…" if len(d.text) > 120 else ""),
            "words": len(d.text.split()),
        }
        for d in docs
    ]


@app.delete("/doc/delete/{item_id}")
def doc_delete(item_id: int):
    ok = doc_db.remove(item_id)
    return {"ok": ok}


@app.post("/doc/search")
async def doc_search(request: Request):
    body = await request.json()
    question = body.get("question", "")
    k = int(body.get("k", 3))
    if not question:
        raise HTTPException(400, "question is required")
    q_emb = ollama.embed(question)
    if not q_emb:
        raise HTTPException(503, "Ollama unavailable")
    hits = doc_db.search(q_emb, k)
    return {
        "contexts": [
            {"id": d.id, "title": d.title, "distance": round(dist, 4)}
            for dist, d in hits
        ]
    }


@app.post("/doc/hybrid-search")
def doc_hybrid_search(body: HybridSearchBody):
    if not body.question:
        raise HTTPException(400, "question is required")

    q_emb = ollama.embed(body.question)
    if not q_emb:
        raise HTTPException(503, "Ollama unavailable. Run: ollama serve")

    hits = doc_db.hybrid_search(
        query_text=body.question,
        query_emb=q_emb,
        k=body.k,
        rrf_k=body.rrf_k,
    )

    return {
        "results": [
            {
                "id": doc.id,
                "title": doc.title,
                "rrfScore": round(score, 6),
                "preview": doc.text[:200] + ("…" if len(doc.text) > 200 else ""),
            }
            for score, doc in hits
        ],
        "count": len(hits),
        "searchType": "hybrid (BM25 + HNSW + RRF)",
        "rrfK": body.rrf_k,
    }


@app.post("/doc/rerank")
def doc_rerank(body: RerankTestBody):
    if not body.query:
        raise HTTPException(400, "query is required")
    if len(doc_db) == 0:
        return {"items": [], "count": 0}

    q_emb = ollama.embed(body.query)
    candidates = doc_db.hybrid_search(body.query, q_emb, k=min(body.k * 3, len(doc_db)))

    reranked = rerank_candidates(
        query=body.query,
        candidates=candidates,
        strategy=body.strategy,
        ollama_generate_fn=ollama.generate if body.strategy == "llm" else None,
        top_k=body.k,
    )

    return {
        "query": body.query,
        "strategy": body.strategy,
        "items": [
            {
                "id": item.id,
                "title": item.title,
                "originalRank": item.original_rank,
                "newRank": item.new_rank,
                "score": item.rerank_score,
                "reasoning": item.reasoning,
                "preview": item.text[:150] + ("…" if len(item.text) > 150 else ""),
            }
            for item in reranked
        ],
    }


@app.post("/doc/advanced-ask")
def doc_advanced_ask(body: AdvancedAskBody):
    if not body.question:
        raise HTTPException(400, "question is required")
    if not ollama.is_available():
        raise HTTPException(503, "Ollama unavailable. Run: ollama serve")

    t_start = time.perf_counter()

    t_ret_0 = time.perf_counter()
    adv_res = doc_db.advanced_search(
        query_text=body.question,
        query_emb_fn=ollama.embed,
        k=body.k,
        pipeline=body.pipeline,
        rerank_strategy=body.rerank_strategy,
        ollama_generate_fn=ollama.generate,
    )
    t_ret_ms = round((time.perf_counter() - t_ret_0) * 1000, 1)

    retrieved_items = [doc for _, doc in adv_res.results]

    t_gen_0 = time.perf_counter()
    if retrieved_items:
        context_str = "\n\n".join(
            f"[{i + 1}] {doc.title}:\n{doc.text}" for i, doc in enumerate(retrieved_items)
        )
        prompt = (
            "You are a helpful and precise assistant. Answer the user's question directly. "
            "Base your answer strictly on the provided context if relevant information is present. "
            "If the context does not contain the answer, use your general knowledge. "
            "IMPORTANT: Do NOT say 'according to the text' or 'the context mentions'. Just answer naturally.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {body.question}\n\nAnswer:"
        )
    else:
        prompt = f"Answer this question directly:\n{body.question}\n\nAnswer:"

    answer = ollama.generate(prompt)
    t_gen_ms = round((time.perf_counter() - t_gen_0) * 1000, 1)

    t_gnd_0 = time.perf_counter()
    grounding_data = None
    if body.grounding and retrieved_items and not answer.startswith("ERROR"):
        report = ground_answer(answer, retrieved_items, embed_fn=ollama.embed)
        grounding_data = {
            "overallConfidence": report.overall_confidence,
            "groundingRate": report.grounding_rate,
            "citedDocs": report.cited_docs,
            "annotatedAnswer": report.annotated_answer,
            "sentences": [
                {
                    "sentence": s.sentence,
                    "citationId": s.citation_id,
                    "sourceDocId": s.source_doc_id,
                    "sourceDocTitle": s.source_doc_title,
                    "sourceSentence": s.source_sentence,
                    "confidence": s.confidence,
                    "status": s.status,
                }
                for s in report.grounded_sentences
            ],
        }
    t_gnd_ms = round((time.perf_counter() - t_gnd_0) * 1000, 1)
    t_total_ms = round((time.perf_counter() - t_start) * 1000, 1)

    return {
        "answer": answer,
        "annotatedAnswer": grounding_data["annotatedAnswer"] if grounding_data else answer,
        "model": ollama.gen_model,
        "pipeline": body.pipeline,
        "contexts": [
            {
                "id": doc.id,
                "title": doc.title,
                "text": doc.text,
                "score": round(score, 4),
            }
            for score, doc in adv_res.results
        ],
        "grounding": grounding_data,
        "rerankItems": [
            {
                "id": item.id,
                "title": item.title,
                "originalRank": item.original_rank,
                "newRank": item.new_rank,
                "score": item.rerank_score,
                "reasoning": item.reasoning,
            }
            for item in adv_res.rerank_items
        ],
        "hydeDoc": adv_res.hyde_doc,
        "latencies": {
            "retrievalMs": t_ret_ms,
            "generationMs": t_gen_ms,
            "groundingMs": t_gnd_ms,
            "totalMs": t_total_ms,
        },
        "docCount": len(doc_db),
    }


@app.post("/doc/ask")
def doc_ask(body: DocAskBody):
    adv_body = AdvancedAskBody(question=body.question, k=body.k, pipeline="vector", grounding=False)
    return doc_advanced_ask(adv_body)


@app.get("/status")
def status():
    up = ollama.is_available()
    return {
        "ollamaAvailable": up,
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docCount": len(doc_db),
        "docDims": doc_db.get_dims(),
        "demoDims": DIMS,
        "demoCount": len(db),
    }


# ── Helper ─────────────────────────────────────────────────────────────────────

def _parse_vec(s: str) -> List[float]:
    try:
        return [float(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError:
        return []


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
