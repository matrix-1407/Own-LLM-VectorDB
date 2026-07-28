import math
import heapq
import random
from typing import Dict, List, Tuple, Optional, Callable

DistFn = Callable[[List[float], List[float]], float]


class HNSW:
    """
    Hierarchical Navigable Small World (HNSW) approximate nearest-neighbor graph index.

    Parameters:
        M            = 16  (max neighbors per node per layer, except layer 0)
        M0           = 32  (max neighbors at layer 0 = 2*M)
        ef_construct = 200 (beam width during construction)
        mL           = 1 / ln(M) (level generation factor)
    """

    def __init__(self, M: int = 16, ef_construction: int = 200, seed: int = 42):
        self.M = M
        self.M0 = 2 * M
        self.ef_build = ef_construction
        self.mL = 1.0 / math.log(M)
        self._rng = random.Random(seed)

        # Graph: {id: {"emb": [...], "meta": "...", "cat": "...", "max_lyr": int, "nbrs": [[...], [...], ...]}}
        self._G: Dict[int, dict] = {}
        self._entry: Optional[int] = None
        self._top_layer: int = -1

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(self._rng.random()) * self.mL))

    def _search_layer(self, query: List[float], ep: int, ef: int,
                      layer: int, dist_fn: DistFn) -> List[Tuple[float, int]]:
        visited = {ep}
        d0 = dist_fn(query, self._G[ep]["emb"])
        # min-heap for candidates, max-heap (negated) for found
        cands = [(d0, ep)]
        found = [(-d0, ep)]

        while cands:
            cd, cid = heapq.heappop(cands)
            worst = -found[0][0]
            if cd > worst and len(found) >= ef:
                break
            node = self._G.get(cid)
            if node is None or layer >= len(node["nbrs"]):
                continue
            for nid in node["nbrs"][layer]:
                if nid in visited or nid not in self._G:
                    continue
                visited.add(nid)
                nd = dist_fn(query, self._G[nid]["emb"])
                worst = -found[0][0]
                if len(found) < ef or nd < worst:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = [(-d, id_) for d, id_ in found]
        result.sort(key=lambda x: x[0])
        return result

    def _select_neighbors(self, candidates: List[Tuple[float, int]], max_m: int) -> List[int]:
        return [id_ for _, id_ in candidates[:max_m]]

    # ── Insert ─────────────────────────────────────────────────────────────────

    def insert(self, id_: int, metadata: str, category: str, emb: List[float], dist_fn: DistFn) -> None:
        lvl = self._rand_level()
        self._G[id_] = {
            "emb": emb, "meta": metadata, "cat": category,
            "max_lyr": lvl, "nbrs": [[] for _ in range(lvl + 1)]
        }

        if self._entry is None:
            self._entry = id_
            self._top_layer = lvl
            return

        ep = self._entry
        for lc in range(self._top_layer, lvl, -1):
            if lc < len(self._G[ep]["nbrs"]):
                W = self._search_layer(emb, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]

        for lc in range(min(self._top_layer, lvl), -1, -1):
            W = self._search_layer(emb, ep, self.ef_build, lc, dist_fn)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_neighbors(W, max_m)
            self._G[id_]["nbrs"][lc] = sel

            for nid in sel:
                if nid not in self._G:
                    continue
                node = self._G[nid]
                if lc >= len(node["nbrs"]):
                    node["nbrs"].extend([] for _ in range(lc - len(node["nbrs"]) + 1))
                conn = node["nbrs"][lc]
                conn.append(id_)
                if len(conn) > max_m:
                    # Trim to max_m nearest neighbors
                    scored = [(dist_fn(node["emb"], self._G[c]["emb"]), c)
                              for c in conn if c in self._G]
                    scored.sort(key=lambda x: x[0])
                    node["nbrs"][lc] = [c for _, c in scored[:max_m]]

            if W:
                ep = W[0][1]

        if lvl > self._top_layer:
            self._top_layer = lvl
            self._entry = id_

    # ── K-NN Search ────────────────────────────────────────────────────────────

    def knn(self, query: List[float], k: int, ef: int, dist_fn: DistFn) -> List[Tuple[float, int]]:
        if self._entry is None:
            return []
        ep = self._entry
        for lc in range(self._top_layer, 0, -1):
            if ep in self._G and lc < len(self._G[ep]["nbrs"]):
                W = self._search_layer(query, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]
        W = self._search_layer(query, ep, max(ef, k), 0, dist_fn)
        return W[:k]

    # ── Delete ─────────────────────────────────────────────────────────────────

    def remove(self, id_: int) -> None:
        if id_ not in self._G:
            return
        for node in self._G.values():
            for layer in node["nbrs"]:
                if id_ in layer:
                    layer.remove(id_)
        if self._entry == id_:
            remaining = [nid for nid in self._G if nid != id_]
            self._entry = remaining[0] if remaining else None
        del self._G[id_]

    # ── Graph Introspection (for /hnsw-info API) ────────────────────────────────

    def get_info(self) -> dict:
        max_l = max(self._top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes_out = []
        edges_out = []

        for id_, node in self._G.items():
            nodes_out.append({
                "id": id_, "metadata": node["meta"],
                "category": node["cat"], "maxLyr": node["max_lyr"]
            })
            for lc in range(min(node["max_lyr"] + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(node["nbrs"]):
                    for nid in node["nbrs"][lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges_out.append({"src": id_, "dst": nid, "lyr": lc})

        return {
            "topLayer": self._top_layer,
            "nodeCount": len(self._G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes_out,
            "edges": edges_out,
        }

    def __len__(self) -> int:
        return len(self._G)
