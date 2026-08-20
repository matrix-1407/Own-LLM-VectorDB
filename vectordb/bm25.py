"""
vectordb/bm25.py
================
BM25 (Best Match 25) lexical keyword search index.

Used alongside the HNSW vector index in DocumentDB to enable
Hybrid Search via Reciprocal Rank Fusion (RRF).

Algorithm parameters (Okapi BM25):
    k1 = 1.5  — term frequency saturation
    b  = 0.75 — length normalisation factor

Public API
----------
BM25Index.insert(id, text)             — Add a document
BM25Index.remove(id)                   — Remove a document
BM25Index.score(query_text, k)         — Return top-k (score, id) pairs
BM25Index.token_count                  — Property: total tokens indexed
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple


# ── Tokeniser ─────────────────────────────────────────────────────────────────

_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "as", "is", "it", "its", "this",
    "that", "was", "are", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "i", "we", "you", "he", "she", "they", "not", "no", "so", "if",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stopwords, return token list."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


# ── BM25Index ─────────────────────────────────────────────────────────────────

class BM25Index:
    """
    Inverted-index BM25 retrieval over document text.

    Each document is stored as a token-frequency Counter.
    IDF is recomputed lazily on each score() call if the corpus has changed
    (``_dirty`` flag).

    Thread safety: not included — callers hold their own locks.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

        # {id: Counter(token -> freq)}
        self._docs: Dict[int, Counter] = {}
        # {id: doc_length}
        self._doc_lengths: Dict[int, int] = {}
        # {token: set of doc ids containing it}
        self._inverted: Dict[str, Set[int]] = defaultdict(set)

        self._avg_dl: float = 0.0
        self._idf_cache: Dict[str, float] = {}
        self._dirty: bool = False

    # ── Insert ────────────────────────────────────────────────────────────────

    def insert(self, id_: int, text: str) -> None:
        """Tokenise *text* and add document *id_* to the index."""
        tokens = _tokenize(text)
        tf = Counter(tokens)
        self._docs[id_] = tf
        self._doc_lengths[id_] = len(tokens)
        for token in tf:
            self._inverted[token].add(id_)
        self._dirty = True

    # ── Remove ────────────────────────────────────────────────────────────────

    def remove(self, id_: int) -> None:
        """Remove document *id_* from the index."""
        if id_ not in self._docs:
            return
        for token in self._docs[id_]:
            self._inverted[token].discard(id_)
            if not self._inverted[token]:
                del self._inverted[token]
        del self._docs[id_]
        del self._doc_lengths[id_]
        self._dirty = True

    # ── IDF recomputation ─────────────────────────────────────────────────────

    def _recompute_stats(self) -> None:
        N = len(self._docs)
        if N == 0:
            self._avg_dl = 0.0
            self._idf_cache = {}
            self._dirty = False
            return

        total_len = sum(self._doc_lengths.values())
        self._avg_dl = total_len / N

        # Robertson-Sparck Jones IDF (smoothed)
        self._idf_cache = {}
        for token, doc_set in self._inverted.items():
            df = len(doc_set)
            self._idf_cache[token] = math.log(
                (N - df + 0.5) / (df + 0.5) + 1
            )
        self._dirty = False

    # ── Score ─────────────────────────────────────────────────────────────────

    def score(self, query_text: str, k: int) -> List[Tuple[float, int]]:
        """
        Return top-k BM25-scored (score, doc_id) pairs for *query_text*.

        Higher scores = better keyword match.
        """
        if self._dirty:
            self._recompute_stats()

        if not self._docs:
            return []

        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []

        # Candidate documents: only those containing at least one query token
        candidate_ids: Set[int] = set()
        for token in query_tokens:
            candidate_ids.update(self._inverted.get(token, set()))

        scores: Dict[int, float] = {}
        k1 = self.k1
        b = self.b
        avg_dl = self._avg_dl

        for id_ in candidate_ids:
            dl = self._doc_lengths[id_]
            tf_map = self._docs[id_]
            s = 0.0
            for token in query_tokens:
                if token not in tf_map:
                    continue
                idf = self._idf_cache.get(token, 0.0)
                tf = tf_map[token]
                # BM25 term score
                s += idf * (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * dl / max(avg_dl, 1))
                )
            scores[id_] = s

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [(score, id_) for id_, score in ranked[:k]]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def token_count(self) -> int:
        """Total number of tokens across all indexed documents."""
        return sum(self._doc_lengths.values())

    def __len__(self) -> int:
        return len(self._docs)
