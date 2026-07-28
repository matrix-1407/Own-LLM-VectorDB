import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from vectordb.metrics import get_dist_fn, DistFn, cosine
from vectordb.algorithms.brute_force import BruteForce, VectorItem
from vectordb.algorithms.kd_tree import KDTree
from vectordb.algorithms.hnsw import HNSW


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
    Thread-safe via threading.Lock.
    """

    def __init__(self, dims: int = 16):
        self.dims = dims
        self._store: Dict[int, VectorItem] = {}
        self._bf = BruteForce()
        self._kdt = KDTree(dims)
        self._hnsw = HNSW(M=16, ef_construction=200)
        self._lock = threading.Lock()
        self._next_id = 1

    def insert(self, metadata: str, category: str, emb: List[float],
               dist_fn: Optional[DistFn] = None) -> int:
        dist_fn = dist_fn or cosine
        with self._lock:
            item = VectorItem(id=self._next_id, metadata=metadata, category=category, emb=emb)
            self._next_id += 1
            self._store[item.id] = item
            self._bf.insert(item)
            self._kdt.insert(item.id, metadata, category, emb)
            self._hnsw.insert(item.id, metadata, category, emb, dist_fn)
            return item.id

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._bf.remove(item_id)
            self._hnsw.remove(item_id)
            # KDTree requires a full rebuild after deletion
            self._kdt.rebuild(list(self._store.values()))
            return True

    def search(self, query: List[float], k: int, metric: str, algo: str) -> SearchResult:
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
                    hits.append(SearchHit(id=id_, metadata=item.metadata,
                                         category=item.category, emb=item.emb, dist=d))
            return SearchResult(hits=hits, latency_us=latency_us, algo=algo, metric=metric)

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

    def all_items(self) -> List[VectorItem]:
        with self._lock:
            return list(self._store.values())

    def hnsw_info(self) -> dict:
        with self._lock:
            return self._hnsw.get_info()

    def __len__(self) -> int:
        return len(self._store)
