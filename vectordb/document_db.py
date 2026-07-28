import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from vectordb.metrics import cosine
from vectordb.algorithms.brute_force import BruteForce, VectorItem
from vectordb.algorithms.hnsw import HNSW


@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: List[float]


class DocumentDB:
    """
    768-dimensional HNSW-backed document store for RAG (Retrieval-Augmented Generation).
    Falls back to BruteForce for very small collections (< 10 items).
    Thread-safe via threading.Lock.
    """

    def __init__(self):
        self._store: Dict[int, DocItem] = {}
        self._hnsw = HNSW(M=16, ef_construction=200)
        self._bf = BruteForce()
        self._lock = threading.Lock()
        self._next_id = 1
        self._dims: int = 0

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self._lock:
            if self._dims == 0:
                self._dims = len(emb)
            item = DocItem(id=self._next_id, title=title, text=text, emb=emb)
            self._next_id += 1
            self._store[item.id] = item
            vi = VectorItem(id=item.id, metadata=title, category="doc", emb=emb)
            self._hnsw.insert(item.id, title, "doc", emb, cosine)
            self._bf.insert(vi)
            return item.id

    def search(self, query: List[float], k: int,
               max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        with self._lock:
            if not self._store:
                return []
            if len(self._store) < 10:
                raw = self._bf.knn(query, k, cosine)
            else:
                raw = self._hnsw.knn(query, k, ef=50, dist_fn=cosine)
            results = []
            for d, id_ in raw:
                if id_ in self._store and d <= max_dist:
                    results.append((d, self._store[id_]))
            return results

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._hnsw.remove(item_id)
            self._bf.remove(item_id)
            return True

    def all_items(self) -> List[DocItem]:
        with self._lock:
            return list(self._store.values())

    def get_dims(self) -> int:
        return self._dims

    def __len__(self) -> int:
        return len(self._store)
