import heapq
from typing import List, Tuple, Optional, Callable

DistFn = Callable[[List[float], List[float]], float]


class _KDNode:
    __slots__ = ("id", "metadata", "category", "emb", "left", "right")

    def __init__(self, id_: int, metadata: str, category: str, emb: List[float]):
        self.id = id_
        self.metadata = metadata
        self.category = category
        self.emb = emb
        self.left: Optional["_KDNode"] = None
        self.right: Optional["_KDNode"] = None


class KDTree:
    """
    K-Dimensional Tree spatial partitioning index.
    Achieves O(log N) search in low dimensions; degrades in high-dim (curse of dimensionality).
    Cycles through splitting axes: depth % dims.
    """

    def __init__(self, dims: int):
        self.dims = dims
        self._root: Optional[_KDNode] = None

    # ── Insert ─────────────────────────────────────────────────────────────────

    def insert(self, id_: int, metadata: str, category: str, emb: List[float]) -> None:
        self._root = self._insert(self._root, id_, metadata, category, emb, 0)

    def _insert(self, node: Optional[_KDNode], id_: int, meta: str, cat: str,
                emb: List[float], depth: int) -> _KDNode:
        if node is None:
            return _KDNode(id_, meta, cat, emb)
        ax = depth % self.dims
        if emb[ax] < node.emb[ax]:
            node.left = self._insert(node.left, id_, meta, cat, emb, depth + 1)
        else:
            node.right = self._insert(node.right, id_, meta, cat, emb, depth + 1)
        return node

    # ── K-NN Search ────────────────────────────────────────────────────────────

    def knn(self, query: List[float], k: int, dist_fn: DistFn) -> List[Tuple[float, int]]:
        # Max-heap stores (-dist, id) to efficiently track k closest
        heap: List[Tuple[float, int]] = []
        self._knn(self._root, query, k, 0, dist_fn, heap)
        result = [(-d, id_) for d, id_ in heap]
        result.sort(key=lambda x: x[0])
        return result

    def _knn(self, node: Optional[_KDNode], query: List[float], k: int,
             depth: int, dist_fn: DistFn, heap: List[Tuple[float, int]]) -> None:
        if node is None:
            return

        d = dist_fn(query, node.emb)
        if len(heap) < k:
            heapq.heappush(heap, (-d, node.id))
        elif d < -heap[0][0]:
            heapq.heapreplace(heap, (-d, node.id))

        ax = depth % self.dims
        diff = query[ax] - node.emb[ax]
        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer, query, k, depth + 1, dist_fn, heap)

        # Prune: only visit farther subtree if its plane might have closer points
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, query, k, depth + 1, dist_fn, heap)

    # ── Rebuild (used after deletion) ──────────────────────────────────────────

    def rebuild(self, items: list) -> None:
        self._root = None
        for item in items:
            self.insert(item.id, item.metadata, item.category, item.emb)
