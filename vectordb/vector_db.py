"""
vectordb/vector_db.py
=====================
Unified 16-dimensional demo vector database.

Maintains all 3 indexes simultaneously: BruteForce, KD-Tree, and HNSW.
Thread-safe via threading.Lock.

Phase 3 additions
-----------------
- search_filtered()  — HNSW/KDTree/BruteForce search with optional
                       category-string metadata filter (single-stage).
- sq8_stats()        — Returns SQ8Index compression stats for the demo store.

Phase 5 additions
-----------------
- search_with_trace() — HNSW search returning step-by-step traversal hops.
- items_3d()          — 3D PCA coordinates for interactive 3D visualizer.
- cluster_analytics() — Category centroids, radii, and intra-cluster variances.
- compare_metrics()   — Neighborhood comparison across Cosine, Euclidean, and Manhattan.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from vectordb.algorithms.brute_force import BruteForce, VectorItem
from vectordb.algorithms.hnsw import HNSW
from vectordb.algorithms.kd_tree import KDTree
from vectordb.manifold import (
    compare_metric_neighborhoods,
    compute_cluster_centroids,
    pca_3d,
)
from vectordb.metrics import DistFn, cosine, get_dist_fn
from vectordb.quantization import SQ8Index


@dataclass
class SearchHit:
    id: int
    metadata: str
    category: str
    emb: List[float]
    dist: float


@dataclass
class SearchResult:
    hits: List[SearchHit]
    latency_us: int
    algo: str
    metric: str


@dataclass
class BenchResult:
    bruteforce_us: int
    kdtree_us: int
    hnsw_us: int
    item_count: int


class VectorDB:
    """
    Unified 16-dimensional demo vector database.
    Maintains all 3 indexes simultaneously: BruteForce, KD-Tree, and HNSW.
    Also maintains an SQ8Index for compressed stats reporting.
    Thread-safe via threading.Lock.
    """

    def __init__(self, dims: int = 16):
        self.dims = dims
        self._store: Dict[int, VectorItem] = {}
        self._bf = BruteForce()
        self._kdt = KDTree(dims)
        self._hnsw = HNSW(M=16, ef_construction=200)
        self._sq8 = SQ8Index()          # Phase 3: compressed mirror for stats
        self._lock = threading.Lock()
        self._next_id = 1

    # ── Insert ────────────────────────────────────────────────────────────────

    def insert(
        self,
        metadata: str,
        category: str,
        emb: List[float],
        dist_fn: Optional[DistFn] = None,
    ) -> int:
        dist_fn = dist_fn or cosine
        with self._lock:
            item = VectorItem(
                id=self._next_id, metadata=metadata, category=category, emb=emb
            )
            self._next_id += 1
            self._store[item.id] = item
            self._bf.insert(item)
            self._kdt.insert(item.id, metadata, category, emb)
            self._hnsw.insert(item.id, metadata, category, emb, dist_fn)
            self._sq8.insert(item.id, emb)
            return item.id

    # ── Remove ────────────────────────────────────────────────────────────────

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._bf.remove(item_id)
            self._hnsw.remove(item_id)
            self._sq8.remove(item_id)
            # KDTree requires a full rebuild after deletion
            self._kdt.rebuild(list(self._store.values()))
            return True

    # ── Standard Search (original API — fully backward compatible) ────────────

    def search(
        self,
        query: List[float],
        k: int,
        metric: str,
        algo: str,
    ) -> SearchResult:
        with self._lock:
            dist_fn = get_dist_fn(metric)
            t0 = time.perf_counter_ns()
            if algo == "bruteforce":
                raw = self._bf.knn(query, k, dist_fn)
            elif algo == "kdtree":
                raw = self._kdt.knn(query, k, dist_fn)
            else:
                raw = self._hnsw.knn(query, k, ef=50, dist_fn=dist_fn)
            latency_us = (time.perf_counter_ns() - t0) // 1000

            hits = []
            for d, id_ in raw:
                if id_ in self._store:
                    item = self._store[id_]
                    hits.append(
                        SearchHit(
                            id=id_,
                            metadata=item.metadata,
                            category=item.category,
                            emb=item.emb,
                            dist=d,
                        )
                    )
            return SearchResult(hits=hits, latency_us=latency_us, algo=algo, metric=metric)

    # ── Search with Trajectory Trace (Phase 5) ────────────────────────────────

    def search_with_trace(
        self,
        query: List[float],
        k: int,
        metric: str = "cosine",
        ef: int = 50,
    ) -> dict:
        """
        Executes HNSW search while recording the step-by-step traversal path
        across all graph layers for interactive animation.
        """
        with self._lock:
            dist_fn = get_dist_fn(metric)
            t0 = time.perf_counter_ns()
            raw, trace = self._hnsw.search_with_trace(query, k, ef=ef, dist_fn=dist_fn)
            latency_us = (time.perf_counter_ns() - t0) // 1000

            hits = []
            for d, id_ in raw:
                if id_ in self._store:
                    item = self._store[id_]
                    hits.append({
                        "id": id_,
                        "metadata": item.metadata,
                        "category": item.category,
                        "distance": round(d, 5),
                    })

            return {
                "results": hits,
                "trace": trace,
                "latencyUs": latency_us,
                "metric": metric,
                "topLayer": self._hnsw._top_layer,
                "entryPoint": self._hnsw._entry,
            }

    # ── Filtered Search (Phase 3) ─────────────────────────────────────────────

    def search_filtered(
        self,
        query: List[float],
        k: int,
        metric: str,
        algo: str,
        category_filter: Optional[str] = None,
    ) -> SearchResult:
        if not category_filter:
            return self.search(query, k, metric, algo)

        with self._lock:
            dist_fn = get_dist_fn(metric)
            oversample = min(k * 5, len(self._store))
            oversample = max(oversample, k)

            t0 = time.perf_counter_ns()
            if algo == "bruteforce":
                raw = self._bf.knn(query, oversample, dist_fn)
            elif algo == "kdtree":
                raw = self._kdt.knn(query, oversample, dist_fn)
            else:
                raw = self._hnsw.knn(query, oversample, ef=max(50, oversample), dist_fn=dist_fn)
            latency_us = (time.perf_counter_ns() - t0) // 1000

            hits = []
            cat_lower = category_filter.lower().strip()
            for d, id_ in raw:
                if len(hits) >= k:
                    break
                if id_ not in self._store:
                    continue
                item = self._store[id_]
                if cat_lower and item.category.lower() != cat_lower:
                    continue
                hits.append(
                    SearchHit(
                        id=id_,
                        metadata=item.metadata,
                        category=item.category,
                        emb=item.emb,
                        dist=d,
                    )
                )
            return SearchResult(hits=hits, latency_us=latency_us, algo=algo, metric=metric)

    # ── Benchmark ─────────────────────────────────────────────────────────────

    def benchmark(self, query: List[float], k: int, metric: str) -> BenchResult:
        with self._lock:
            dist_fn = get_dist_fn(metric)

            def timed(fn):
                t = time.perf_counter_ns()
                fn()
                return (time.perf_counter_ns() - t) // 1000

            return BenchResult(
                bruteforce_us=timed(lambda: self._bf.knn(query, k, dist_fn)),
                kdtree_us=timed(lambda: self._kdt.knn(query, k, dist_fn)),
                hnsw_us=timed(lambda: self._hnsw.knn(query, k, ef=50, dist_fn=dist_fn)),
                item_count=len(self._store),
            )

    # ── 3D Manifold & Introspection (Phase 5) ─────────────────────────────────

    def items_3d(self) -> List[dict]:
        """
        Returns all vector items enriched with 3D PCA coordinates [x, y, z]
        for the interactive 3D visualizer.
        """
        with self._lock:
            items = list(self._store.values())
            if not items:
                return []
            embs = [it.emb for it in items]
            coords = pca_3d(embs)

            result = []
            for i, it in enumerate(items):
                result.append({
                    "id": it.id,
                    "metadata": it.metadata,
                    "category": it.category,
                    "embedding": it.emb,
                    "coords3d": coords[i] if i < len(coords) else [0.0, 0.0, 0.0],
                })
            return result

    def cluster_analytics(self) -> dict:
        """Computes 3D centroids, radii, and intra-cluster variances."""
        items_with_coords = self.items_3d()
        return compute_cluster_centroids(items_with_coords)

    def compare_metrics(self, query: List[float], k: int = 5) -> dict:
        """Compares top-K neighborhoods under Cosine, Euclidean, and Manhattan metrics."""
        with self._lock:
            items = [
                {"id": it.id, "metadata": it.metadata, "embedding": it.emb}
                for it in self._store.values()
            ]
            return compare_metric_neighborhoods(query, items, k=k)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def all_items(self) -> List[VectorItem]:
        with self._lock:
            return list(self._store.values())

    def hnsw_info(self) -> dict:
        with self._lock:
            return self._hnsw.get_info()

    def sq8_stats(self) -> dict:
        with self._lock:
            return self._sq8.stats()

    def categories(self) -> List[str]:
        with self._lock:
            return sorted({item.category for item in self._store.values()})

    def __len__(self) -> int:
        return len(self._store)
