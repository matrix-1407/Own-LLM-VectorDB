"""
vectordb/manifold.py
====================
Dimensionality Reduction, 3D Manifold Projections & Cluster Analytics (Phase 5).

Provides pure-Python geometric projections for visualizing high-dimensional
vector spaces (16D demo vectors and 768D document embeddings) in 3D:
1. 3D Principal Component Analysis (PCA via orthogonal deflation).
2. Category cluster centroid & variance analytics.
3. Multi-metric neighborhood comparison (Cosine vs Euclidean vs Manhattan).
"""

import math
import random
from typing import Dict, List, Optional, Tuple

from vectordb.metrics import cosine, euclidean, manhattan


def pca_3d(embs: List[List[float]]) -> List[List[float]]:
    """
    Projects N high-dimensional embedding vectors into 3D coordinate space [x, y, z]
    using Principal Component Analysis (power iteration with Gram-Schmidt orthogonalization).

    Parameters
    ----------
    embs : List[List[float]]
        List of D-dimensional vectors.

    Returns
    -------
    List[List[float]]
        List of 3D coordinates [x, y, z] for each vector.
    """
    n = len(embs)
    if n == 0:
        return []
    d = len(embs[0])
    if n < 3:
        # Fallback for fewer than 3 vectors
        res = []
        for i, e in enumerate(embs):
            x = e[0] if d > 0 else 0.0
            y = e[1] if d > 1 else 0.0
            z = e[2] if d > 2 else float(i)
            res.append([round(x, 4), round(y, 4), round(z, 4)])
        return res

    # 1. Mean-center the data
    mean = [0.0] * d
    for e in embs:
        for j in range(d):
            mean[j] += e[j] / n

    X = [[e[j] - mean[j] for j in range(d)] for e in embs]

    # 2. Power iteration to find principal components
    rng = random.Random(42)

    def power_iteration(X_mat: List[List[float]], prev_pcs: List[List[float]]) -> List[float]:
        v = [rng.uniform(-1.0, 1.0) for _ in range(d)]
        # Orthogonalize against previously found components
        for pc in prev_pcs:
            dot = sum(v[j] * pc[j] for j in range(d))
            v = [v[j] - dot * pc[j] for j in range(d)]
        norm = math.sqrt(sum(x * x for x in v))
        if norm < 1e-10:
            return [1.0 if j == 0 else 0.0 for j in range(d)]
        v = [x / norm for x in v]

        for _ in range(120):
            # Compute X^T * X * v
            # Step a: X * v (length n)
            Xv = [sum(X_mat[i][j] * v[j] for j in range(d)) for i in range(n)]
            # Step b: X^T * (Xv) (length d)
            nv = [sum(X_mat[i][j] * Xv[i] for i in range(n)) for j in range(d)]

            # Orthogonalize
            for pc in prev_pcs:
                dot = sum(nv[j] * pc[j] for j in range(d))
                nv = [nv[j] - dot * pc[j] for j in range(d)]

            norm = math.sqrt(sum(x * x for x in nv))
            if norm < 1e-10:
                break
            new_v = [x / norm for x in nv]
            diff = sum((new_v[j] - v[j]) ** 2 for j in range(d))
            v = new_v
            if diff < 1e-12:
                break

        return v

    pc1 = power_iteration(X, [])
    pc2 = power_iteration(X, [pc1])
    pc3 = power_iteration(X, [pc1, pc2])

    # 3. Project each point onto the 3 principal axes
    coords_3d = []
    for x in X:
        x_proj = sum(x[j] * pc1[j] for j in range(d))
        y_proj = sum(x[j] * pc2[j] for j in range(d))
        z_proj = sum(x[j] * pc3[j] for j in range(d))
        coords_3d.append([round(x_proj, 4), round(y_proj, 4), round(z_proj, 4)])

    return coords_3d


def compute_cluster_centroids(
    items: List[dict],
) -> Dict[str, dict]:
    """
    Computes 3D centroids, radii, and intra-cluster variances for categories.

    Parameters
    ----------
    items : List[dict]
        Items containing 'category' and 'coords3d' ([x, y, z]).

    Returns
    -------
    Dict[str, dict]
        Category -> {centroid: [x, y, z], radius: float, count: int, variance: float}
    """
    cat_points: Dict[str, List[List[float]]] = {}
    for item in items:
        cat = item.get("category", "default")
        coords = item.get("coords3d", [0.0, 0.0, 0.0])
        if cat not in cat_points:
            cat_points[cat] = []
        cat_points[cat].append(coords)

    analytics: Dict[str, dict] = {}
    for cat, pts in cat_points.items():
        k = len(pts)
        if k == 0:
            continue
        cx = sum(p[0] for p in pts) / k
        cy = sum(p[1] for p in pts) / k
        cz = sum(p[2] for p in pts) / k
        centroid = [round(cx, 4), round(cy, 4), round(cz, 4)]

        # Variance and max radius
        dists = [math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2) for p in pts]
        variance = sum(d * d for d in dists) / k if k > 0 else 0.0
        radius = max(dists) if dists else 0.0

        analytics[cat] = {
            "centroid": centroid,
            "radius": round(radius, 4),
            "variance": round(variance, 4),
            "count": k,
        }

    return analytics


def compare_metric_neighborhoods(
    query_emb: List[float],
    items: List[dict],
    k: int = 5,
) -> dict:
    """
    Computes top-K neighbors across Cosine, Euclidean, and Manhattan metrics
    to visualize metric space distortion and ranking shifts.
    """
    metrics = {
        "cosine": cosine,
        "euclidean": euclidean,
        "manhattan": manhattan,
    }

    results = {}
    for name, fn in metrics.items():
        scored = []
        for it in items:
            emb = it.get("embedding", [])
            if emb:
                d = fn(query_emb, emb)
                scored.append({"id": it.get("id"), "metadata": it.get("metadata"), "dist": round(d, 5)})
        scored.sort(key=lambda x: x["dist"])
        results[name] = scored[:k]

    return results
