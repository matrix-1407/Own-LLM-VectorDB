"""
VectorDB — FastAPI Server (v2.0.0 — Phase 3)
=============================================
Python vector database engine.
Serves the web UI at http://localhost:8080 and exposes a full REST API.

Phase 3 additions:
  - Auto-save / auto-load on startup and shutdown (disk persistence)
  - POST /persist/save          — manual snapshot trigger
  - GET  /persist/status        — snapshot file info
  - GET  /search?category=...   — category-filtered vector search
  - POST /doc/hybrid-search     — BM25 + HNSW + RRF hybrid search
  - GET  /stats                 — extended with SQ8 + BM25 stats
  - GET  /categories            — list unique demo vector categories
"""

import logging
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
from vectordb.metrics import cosine, get_dist_fn
from vectordb.ollama_client import OllamaClient
from vectordb.persistence import (
    load_document_db,
    load_vector_db,
    save_document_db,
    save_vector_db,
    snapshot_info,
)
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

    # Restore persisted document index (RAG chunks survive restarts)
    loaded_docs = load_document_db(doc_db, DOCUMENT_SNAPSHOT)

    # Seed demo vectors — always start fresh from DEMO_VECTORS so the demo
    # experience is consistent, but also try to restore any user-inserted demos.
    dist_fn = get_dist_fn("cosine")
    for metadata, category, emb in DEMO_VECTORS:
        db.insert(metadata, category, emb, dist_fn)

    # Restore any extra demo vectors the user inserted during a previous session.
    # (We load *after* seeding so user items get IDs above the demo range.)
    if VECTOR_SNAPSHOT.exists():
        import json
        try:
            payload = json.loads(VECTOR_SNAPSHOT.read_text(encoding="utf-8"))
            for item in payload.get("items", []):
                # Skip if it looks like a built-in demo item (metadata in DEMO_VECTORS)
                demo_metas = {d[0] for d in DEMO_VECTORS}
                if item["metadata"] not in demo_metas:
                    db.insert(item["metadata"], item["category"], item["emb"], dist_fn)
                    logger.info("[Startup] Restored user demo vector: %s", item["metadata"])
        except Exception as exc:
            logger.warning("[Startup] Could not restore extra demo vectors: %s", exc)

    ollama_status = "ONLINE" if ollama.is_available() else "OFFLINE"
    print("\n=== VectorDB Engine (Phase 3) ===")
    print("http://localhost:8080")
    print(f"{len(db)} demo vectors | {DIMS} dims | HNSW + KD-Tree + BruteForce + SQ8")
    print(f"Documents loaded: {loaded_docs} chunks | Hybrid Search (BM25 + HNSW + RRF)")
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

app = FastAPI(title="VectorDB", version="2.0.0", lifespan=lifespan)

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


class DocAskBody(BaseModel):
    question: str
    k: int = 3


class HybridSearchBody(BaseModel):
    question: str
    k: int = 3
    rrf_k: int = 60


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
    """
    K-nearest-neighbour search over the demo vector index.

    Phase 3: pass ?category=cs to restrict results to a single category.
    """
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


@app.get("/categories")
def categories():
    """Return the unique category labels present in the demo vector index."""
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


@app.get("/stats")
def stats():
    """Extended statistics including Phase 3 SQ8 and BM25 info."""
    sq8 = db.sq8_stats()
    doc_sq8 = doc_db.sq8_stats()
    bm25 = doc_db.bm25_stats()
    return {
        # Core counts
        "count": len(db),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
        # Phase 3 — SQ8 demo index
        "sq8": {
            "compressionRatio": sq8.get("compressionRatio", 4.0),
            "savedBytes": sq8.get("savedBytes", 0),
            "float32Bytes": sq8.get("float32Bytes", 0),
            "int8Bytes": sq8.get("int8Bytes", 0),
        },
        # Phase 3 — Document index
        "docIndex": {
            "chunkCount": len(doc_db),
            "bm25DocCount": bm25["docCount"],
            "bm25TokenCount": bm25["tokenCount"],
            "sq8CompressionRatio": doc_sq8.get("compressionRatio", 4.0),
            "sq8SavedBytes": doc_sq8.get("savedBytes", 0),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE ENDPOINTS  (Phase 3)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/persist/save")
def persist_save():
    """Manually trigger saving both indexes to disk."""
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
    """Return info about the on-disk snapshot files."""
    return {
        "vectorSnapshot": snapshot_info(VECTOR_SNAPSHOT),
        "documentSnapshot": snapshot_info(DOCUMENT_SNAPSHOT),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT & RAG ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/doc/insert")
def doc_insert(body: DocInsertBody):
    if not body.title or not body.text:
        raise HTTPException(400, "title and text are required")
    chunks = chunk_text(body.text, chunk_words=250, overlap_words=30)
    ids = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            raise HTTPException(
                503,
                "Ollama unavailable. Run: ollama serve && ollama pull nomic-embed-text",
            )
        title = f"{body.title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else body.title
        ids.append(doc_db.insert(title, chunk, emb))
    return {"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()}


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
    """
    Phase 3: Hybrid BM25 + HNSW search via Reciprocal Rank Fusion.

    Combines semantic vector similarity (HNSW) and keyword relevance (BM25).
    Useful when the query contains exact product names, IDs, or rare terms
    that semantic search alone might miss.
    """
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


@app.post("/doc/ask")
def doc_ask(body: DocAskBody):
    if not body.question:
        raise HTTPException(400, "question is required")

    # Step 1: Embed the question
    q_emb = ollama.embed(body.question)
    if not q_emb:
        raise HTTPException(503, "Ollama unavailable. Run: ollama serve")

    # Step 2: Retrieve top-k semantically relevant chunks
    hits = doc_db.search(q_emb, body.k)

    # Step 3: Build grounded prompt
    context_str = "\n\n".join(
        f"[{i + 1}] {doc.title}:\n{doc.text}" for i, (_, doc) in enumerate(hits)
    )
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {body.question}\n\nAnswer:"
    )

    # Step 4: Generate answer via local LLM
    answer = ollama.generate(prompt)

    return {
        "answer": answer,
        "model": ollama.gen_model,
        "contexts": [
            {"id": d.id, "title": d.title, "text": d.text, "distance": round(dist, 4)}
            for dist, d in hits
        ],
        "docCount": len(doc_db),
    }


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
