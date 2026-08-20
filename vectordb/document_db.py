"""
vectordb/document_db.py
=======================
768-dimensional HNSW-backed document store for RAG.

Phase 3 additions
-----------------
- BM25Index integration for lexical keyword search.
- hybrid_search() — merges HNSW vector scores and BM25 keyword scores
  via Reciprocal Rank Fusion (RRF).
- SQ8Index for memory-compressed flat search (used when n < 10).

Phase 4 additions
-----------------
- advanced_search() — orchestrates multi-stage retrieval pipelines:
  * "vector"  : Standard HNSW vector search
  * "hybrid"  : BM25 + HNSW with Reciprocal Rank Fusion
  * "rerank"  : 2-stage retrieval (Oversample candidates -> Cross/LLM Rerank)
  * "hyde"    : Hypothetical Document Embeddings query expansion
"""

import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from vectordb.metrics import cosine
from vectordb.algorithms.brute_force import BruteForce, VectorItem
from vectordb.algorithms.hnsw import HNSW
from vectordb.bm25 import BM25Index
from vectordb.quantization import SQ8Index
from vectordb.reranker import RerankItem, rerank_candidates
from vectordb.hyde import generate_hypothetical_doc


@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: List[float]


@dataclass
class AdvancedSearchResult:
    pipeline: str
    results: List[Tuple[float, DocItem]]  # (score_or_dist, DocItem)
    rerank_items: List[RerankItem]        # Populated if pipeline == "rerank"
    hyde_doc: Optional[str]               # Populated if pipeline == "hyde"
    candidate_count: int


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def _rrf_merge(
    vector_hits: List[Tuple[float, int]],
    bm25_hits: List[Tuple[float, int]],
    k: int,
    rrf_k: int = 60,
) -> List[Tuple[float, int]]:
    """
    Merge two ranked lists via Reciprocal Rank Fusion.

    RRF score = Σ  1 / (rrf_k + rank_m(d))   for each method m.
    Higher RRF score = better combined rank.

    Both input lists are (score_or_distance, id) tuples.
    *vector_hits* is sorted ascending by distance (lower = better).
    *bm25_hits*   is sorted descending by BM25 score (higher = better).
    """
    rrf_scores: Dict[int, float] = {}

    for rank, (_, id_) in enumerate(vector_hits, start=1):
        rrf_scores[id_] = rrf_scores.get(id_, 0.0) + 1.0 / (rrf_k + rank)

    for rank, (_, id_) in enumerate(bm25_hits, start=1):
        rrf_scores[id_] = rrf_scores.get(id_, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return [(score, id_) for id_, score in ranked[:k]]


# ── DocumentDB ────────────────────────────────────────────────────────────────

class DocumentDB:
    """
    768-dimensional HNSW-backed document store for RAG
    (Retrieval-Augmented Generation).

    Falls back to SQ8 compressed flat search for very small collections
    (< 10 items) — avoids HNSW graph instability at tiny scale.

    Thread-safe via threading.Lock.
    """

    def __init__(self) -> None:
        self._store: Dict[int, DocItem] = {}
        self._hnsw = HNSW(M=16, ef_construction=200)
        self._sq8 = SQ8Index()          # Phase 3: compressed flat fallback
        self._bm25 = BM25Index()        # Phase 3: keyword search index
        self._lock = threading.Lock()
        self._next_id = 1
        self._dims: int = 0

    # ── Insert ────────────────────────────────────────────────────────────────

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self._lock:
            if self._dims == 0:
                self._dims = len(emb)
            item = DocItem(id=self._next_id, title=title, text=text, emb=emb)
            self._next_id += 1
            self._store[item.id] = item

            # HNSW + SQ8 for vector search
            self._hnsw.insert(item.id, title, "doc", emb, cosine)
            self._sq8.insert(item.id, emb)

            # BM25 for keyword search — index title + text
            self._bm25.insert(item.id, f"{title} {text}")

            return item.id

    # ── Vector-only search (original API — backward compatible) ───────────────

    def search(
        self,
        query: List[float],
        k: int,
        max_dist: float = 0.9,
    ) -> List[Tuple[float, DocItem]]:
        """
        Pure vector (HNSW / SQ8) semantic search.
        Returns list of (distance, DocItem) sorted by ascending distance.
        """
        with self._lock:
            if not self._store:
                return []
            raw = self._raw_vector_search(query, k * 2)
            results = []
            for d, id_ in raw:
                if id_ in self._store and d <= max_dist:
                    results.append((d, self._store[id_]))
            return results[:k]

    def _raw_vector_search(
        self,
        query: List[float],
        k: int,
    ) -> List[Tuple[float, int]]:
        """Internal: run HNSW or SQ8 depending on collection size."""
        n = len(self._store)
        if n == 0:
            return []
        if n < 10:
            return self._sq8.knn(query, k, cosine)
        return self._hnsw.knn(query, k, ef=50, dist_fn=cosine)

    # ── Hybrid search (Phase 3) ───────────────────────────────────────────────

    def hybrid_search(
        self,
        query_text: str,
        query_emb: List[float],
        k: int,
        rrf_k: int = 60,
    ) -> List[Tuple[float, DocItem]]:
        """
        Hybrid BM25 + HNSW search merged via Reciprocal Rank Fusion.
        Returns list of (rrf_score, DocItem) sorted descending by RRF score.
        """
        with self._lock:
            if not self._store:
                return []

            oversample = min(k * 4, len(self._store))

            # Stage 1: vector retrieval
            vector_hits = self._raw_vector_search(query_emb, oversample)

            # Stage 2: BM25 keyword retrieval
            bm25_hits = self._bm25.score(query_text, oversample)

            # Stage 3: Merge via RRF
            merged = _rrf_merge(vector_hits, bm25_hits, k=k, rrf_k=rrf_k)

            results = []
            for score, id_ in merged:
                if id_ in self._store:
                    results.append((score, self._store[id_]))
            return results

    # ── Advanced Multi-Pipeline Search (Phase 4) ──────────────────────────────

    def advanced_search(
        self,
        query_text: str,
        query_emb_fn: Callable[[str], List[float]],
        k: int = 3,
        pipeline: str = "vector",            # "vector", "hybrid", "rerank", "hyde"
        rerank_strategy: str = "cross",       # "cross" or "llm"
        ollama_generate_fn: Optional[Callable[[str], str]] = None,
        candidate_k: int = 10,
    ) -> AdvancedSearchResult:
        """
        Executes an advanced multi-stage retrieval pipeline.

        Pipelines:
        - "vector" : Standard HNSW vector search.
        - "hybrid" : BM25 + HNSW merged via Reciprocal Rank Fusion.
        - "rerank" : Stage 1 oversampling -> Stage 2 Cross/LLM Re-ranking.
        - "hyde"   : Generate hypothetical document -> embed -> HNSW search.
        """
        with self._lock:
            if not self._store:
                return AdvancedSearchResult(
                    pipeline=pipeline,
                    results=[],
                    rerank_items=[],
                    hyde_doc=None,
                    candidate_count=0,
                )

        hyde_doc: Optional[str] = None

        if pipeline == "hyde":
            # 1. Generate hypothetical doc
            if ollama_generate_fn:
                hyde_doc = generate_hypothetical_doc(query_text, ollama_generate_fn)
            else:
                hyde_doc = query_text

            # 2. Embed hypothetical doc
            h_emb = query_emb_fn(hyde_doc)
            if not h_emb:
                h_emb = query_emb_fn(query_text)

            # 3. Vector search using hypothetical vector
            hits = self.search(h_emb, k=k)
            return AdvancedSearchResult(
                pipeline="hyde",
                results=hits,
                rerank_items=[],
                hyde_doc=hyde_doc,
                candidate_count=len(hits),
            )

        elif pipeline == "hybrid":
            q_emb = query_emb_fn(query_text)
            hits = self.hybrid_search(query_text, q_emb, k=k)
            return AdvancedSearchResult(
                pipeline="hybrid",
                results=hits,
                rerank_items=[],
                hyde_doc=None,
                candidate_count=len(hits),
            )

        elif pipeline == "rerank":
            # Stage 1: Oversample candidates using hybrid search
            q_emb = query_emb_fn(query_text)
            oversample_k = min(max(candidate_k, k * 3), len(self._store))
            candidates = self.hybrid_search(query_text, q_emb, k=oversample_k)

            # Stage 2: Cross/LLM Rerank
            rerank_items = rerank_candidates(
                query=query_text,
                candidates=candidates,
                strategy=rerank_strategy,
                ollama_generate_fn=ollama_generate_fn,
                top_k=k,
            )

            # Map rerank items back to results
            with self._lock:
                final_results = []
                for item in rerank_items:
                    if item.id in self._store:
                        final_results.append((item.rerank_score, self._store[item.id]))

            return AdvancedSearchResult(
                pipeline="rerank",
                results=final_results,
                rerank_items=rerank_items,
                hyde_doc=None,
                candidate_count=len(candidates),
            )

        else:  # "vector"
            q_emb = query_emb_fn(query_text)
            hits = self.search(q_emb, k=k)
            return AdvancedSearchResult(
                pipeline="vector",
                results=hits,
                rerank_items=[],
                hyde_doc=None,
                candidate_count=len(hits),
            )

    # ── Remove ────────────────────────────────────────────────────────────────

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._hnsw.remove(item_id)
            self._sq8.remove(item_id)
            self._bm25.remove(item_id)
            return True

    # ── Accessors ─────────────────────────────────────────────────────────────

    def all_items(self) -> List[DocItem]:
        with self._lock:
            return list(self._store.values())

    def get_dims(self) -> int:
        return self._dims

    def bm25_stats(self) -> dict:
        """Return BM25 index statistics."""
        return {
            "docCount": len(self._bm25),
            "tokenCount": self._bm25.token_count,
        }

    def sq8_stats(self) -> dict:
        """Return SQ8 compression statistics."""
        return self._sq8.stats()

    def __len__(self) -> int:
        return len(self._store)
