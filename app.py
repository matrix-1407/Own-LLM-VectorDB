"""
VectorDB — FastAPI Server
=========================
Python vector database engine.
Serves the web UI at http://localhost:8080 and exposes a full REST API.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

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
from vectordb.vector_db import VectorDB

# ── Constants & Singletons ─────────────────────────────────────────────────────

DIMS = 16
db = VectorDB(dims=DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()


# ── Lifespan (replaces deprecated on_event("startup")) ────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Seed demo vectors on startup."""
    dist_fn = get_dist_fn("cosine")
    for metadata, category, emb in DEMO_VECTORS:
        db.insert(metadata, category, emb, dist_fn)
    ollama_status = "ONLINE" if ollama.is_available() else "OFFLINE"
    print("=== VectorDB Engine ===")
    print("http://localhost:8080")
    print(f"{len(db)} demo vectors | {DIMS} dims | HNSW + KD-Tree + BruteForce")
    print(f"Ollama: {ollama_status}")
    if ollama_status == "ONLINE":
        print(f"  embed: {ollama.embed_model}  gen: {ollama.gen_model}")
    yield  # server runs here


# ── App & Middleware ───────────────────────────────────────────────────────────

app = FastAPI(title="VectorDB", version="1.0.0", lifespan=lifespan)

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


# ══════════════════════════════════════════════════════════════════════════════
# DEMO VECTOR ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/search")
def search(
    v: str = Query(..., description="Comma-separated floats, must be 16D"),
    k: int = Query(5),
    metric: str = Query("cosine"),
    algo: str = Query("hnsw"),
):
    query = _parse_vec(v)
    if len(query) != DIMS:
        raise HTTPException(400, f"Expected {DIMS}-dimensional vector, got {len(query)}D")
    result = db.search(query, k, metric, algo)
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
    return {
        "count": len(db),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
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
            "preview": d.text[:120] + ("\u2026" if len(d.text) > 120 else ""),
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
