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
"""

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from vectordb.metrics import cosine
from vectordb.algorithms.brute_force import BruteForce, VectorItem
from vectordb.algorithms.hnsw import HNSW
from vectordb.bm25 import BM25Index
from vectordb.quantization import SQ8Index


@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: List[float]


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

        Parameters
        ----------
        query_text : str    — original question text for BM25 keyword scoring
        query_emb  : list   — 768-D embedding of the question for HNSW scoring
        k          : int    — number of results to return
        rrf_k      : int    — RRF smoothing constant (default 60)

        Returns
        -------
        List of (rrf_score, DocItem) sorted descending by RRF score.
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
