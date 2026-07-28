from typing import List, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: List[float]


DistFn = Callable[[List[float], List[float]], float]


class BruteForce:
    """Brute-force exact nearest neighbor search — O(N*d) baseline."""

    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, item: VectorItem) -> None:
        self.items.append(item)

    def knn(self, query: List[float], k: int, dist_fn: DistFn) -> List[Tuple[float, int]]:
        results = [(dist_fn(query, item.emb), item.id) for item in self.items]
        results.sort(key=lambda x: x[0])
        return results[:k]

    def remove(self, item_id: int) -> None:
        self.items = [item for item in self.items if item.id != item_id]
