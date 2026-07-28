import math
from typing import List, Callable
import numpy as np

DistFn = Callable[[List[float], List[float]], float]


def euclidean(a: List[float], b: List[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(va - vb))


def cosine(a: List[float], b: List[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    dot = float(np.dot(va, vb))
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (na * nb)


def manhattan(a: List[float], b: List[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    return float(np.sum(np.abs(va - vb)))


def get_dist_fn(metric_name: str) -> DistFn:
    name = (metric_name or "cosine").lower()
    if name == "cosine":
        return cosine
    if name == "manhattan":
        return manhattan
    return euclidean
