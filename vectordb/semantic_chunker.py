"""
vectordb/semantic_chunker.py
============================
Semantic Chunking Engine (Phase 4).

Instead of slicing text blindly at arbitrary word/token counts, Semantic
Chunking identifies natural topic transitions by measuring cosine distance
between consecutive sentence embeddings.

How It Works (Industry / Production Standard)
---------------------------------------------
1. Sentence Tokenization:
   Split document into individual grammatical sentences using abbreviation-safe
   tokenization.
2. Sentence Embeddings:
   Embed each sentence using the embedding model (nomic-embed-text via Ollama).
3. Distance Curve Computation:
   Calculate cosine distance between sentence i and sentence i+1.
   High distance = semantic divergence (topic shift).
4. Dynamic Thresholding:
   Calculate a split threshold (e.g., 75th percentile or mean + 0.5*std).
5. Chunk Assembly:
   Group sentences into coherent chunks, splitting when distance > threshold
   while respecting min_chunk_words and max_chunk_words constraints.
"""

import math
import re
from typing import Callable, Dict, List, Tuple


def _split_into_sentences(text: str) -> List[str]:
    """
    Splits text into natural sentences while protecting common abbreviations,
    decimal numbers, quotes, and punctuation.
    """
    if not text or not text.strip():
        return []

    cleaned = re.sub(r"\r\n|\r", "\n", text.strip())

    abbreviations = [
        "e.g.", "i.e.", "etc.", "vs.", "dr.", "mr.", "mrs.", "ms.",
        "prof.", "fig.", "vol.", "no.", "p.", "pp.", "al.", "approx.", "est.", "inc.", "corp."
    ]

    temp_text = cleaned
    for abbr in abbreviations:
        pattern = re.compile(re.escape(abbr), re.IGNORECASE)
        temp_text = pattern.sub(lambda m: m.group(0).replace(".", "@@DOT@@"), temp_text)

    # Protect decimal numbers (e.g., 3.14, 0.05)
    temp_text = re.sub(r"(\d+)\.(\d+)", r"\1@@DOT@@\2", temp_text)

    # Split on sentence boundaries: (. ! ?) followed by whitespace or newline
    raw_splits = re.split(r"(?<=[.!?])\s+|\n+", temp_text)

    sentences: List[str] = []
    for s in raw_splits:
        restored = s.replace("@@DOT@@", ".").strip()
        if not restored:
            continue
        if len(restored.split()) >= 2:
            sentences.append(restored)
        elif sentences:
            sentences[-1] += " " + restored
        else:
            sentences.append(restored)

    return sentences if sentences else [text.strip()]


def _cosine_dist(a: List[float], b: List[float]) -> float:
    """Computes cosine distance = 1 - cosine_similarity."""
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    sim = dot / (na * nb)
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def semantic_chunk_text(
    text: str,
    embed_fn: Callable[[str], List[float]],
    threshold_percentile: float = 75.0,
    min_chunk_words: int = 25,
    max_chunk_words: int = 350,
) -> List[str]:
    """
    Splits text into semantically cohesive chunks by detecting embedding
    distance spikes between consecutive sentences.

    Parameters
    ----------
    text : str
        Input document text.
    embed_fn : Callable[[str], List[float]]
        Function taking a string and returning its embedding vector.
    threshold_percentile : float
        Percentile (0-100) of distance curve used as split threshold.
    min_chunk_words : int
        Minimum word count before a split is allowed (prevents tiny chunks).
    max_chunk_words : int
        Maximum word count after which a split is forced (prevents oversized chunks).

    Returns
    -------
    List[str]
        List of semantically coherent chunk strings.
    """
    sentences = _split_into_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    total_words = len(text.split())
    if total_words <= min_chunk_words:
        return [" ".join(sentences)]

    # Step 1: Embed all sentences
    embs: List[List[float]] = []
    for s in sentences:
        emb = embed_fn(s)
        if not emb:
            from vectordb.chunker import chunk_text
            return chunk_text(text, chunk_words=250, overlap_words=30)
        embs.append(emb)

    # Step 2: Compute pairwise distances between adjacent sentences
    distances: List[float] = []
    for i in range(len(embs) - 1):
        distances.append(_cosine_dist(embs[i], embs[i + 1]))

    # Step 3: Compute dynamic threshold
    if not distances:
        return [" ".join(sentences)]

    sorted_dists = sorted(distances)
    idx = int(len(sorted_dists) * (threshold_percentile / 100.0))
    idx = min(idx, len(sorted_dists) - 1)
    threshold = sorted_dists[idx]

    # Step 4: Assemble chunks
    chunks: List[str] = []
    curr_sentences: List[str] = [sentences[0]]
    curr_word_count = len(sentences[0].split())

    for i in range(len(distances)):
        next_sent = sentences[i + 1]
        next_word_count = len(next_sent.split())
        dist = distances[i]

        should_split = (dist >= threshold and curr_word_count >= min_chunk_words) or (
            curr_word_count + next_word_count > max_chunk_words
        )

        if should_split:
            chunks.append(" ".join(curr_sentences))
            curr_sentences = [next_sent]
            curr_word_count = next_word_count
        else:
            curr_sentences.append(next_sent)
            curr_word_count += next_word_count

    if curr_sentences:
        chunks.append(" ".join(curr_sentences))

    return chunks


def preview_semantic_chunks(
    text: str,
    embed_fn: Callable[[str], List[float]],
    threshold_percentile: float = 75.0,
    min_chunk_words: int = 25,
    max_chunk_words: int = 350,
) -> Dict:
    """
    Detailed diagnostic preview of the semantic chunking process.
    Used by the Web UI to visualize the sentence distance curve and split boundaries.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return {
            "sentences": [],
            "distances": [],
            "threshold": 0.0,
            "splitIndices": [],
            "chunks": [],
            "sentenceCount": 0,
            "chunkCount": 0,
        }

    if len(sentences) == 1:
        return {
            "sentences": sentences,
            "distances": [],
            "threshold": 0.0,
            "splitIndices": [],
            "chunks": [{"text": sentences[0], "words": len(sentences[0].split()), "sentenceRange": [0, 0]}],
            "sentenceCount": 1,
            "chunkCount": 1,
        }

    embs: List[List[float]] = []
    for s in sentences:
        embs.append(embed_fn(s))

    distances: List[float] = []
    for i in range(len(embs) - 1):
        distances.append(round(_cosine_dist(embs[i], embs[i + 1]), 4))

    sorted_dists = sorted(distances)
    idx = int(len(sorted_dists) * (threshold_percentile / 100.0))
    idx = min(idx, len(sorted_dists) - 1)
    threshold = round(sorted_dists[idx], 4)

    # Calculate splits
    split_indices: List[int] = []
    chunks: List[Dict] = []
    curr_sentences: List[str] = [sentences[0]]
    curr_start_idx = 0
    curr_words = len(sentences[0].split())

    for i in range(len(distances)):
        next_sent = sentences[i + 1]
        next_words = len(next_sent.split())
        dist = distances[i]

        should_split = (dist >= threshold and curr_words >= min_chunk_words) or (
            curr_words + next_words > max_chunk_words
        )

        if should_split:
            split_indices.append(i)
            chunks.append({
                "text": " ".join(curr_sentences),
                "words": curr_words,
                "sentenceRange": [curr_start_idx, i],
            })
            curr_sentences = [next_sent]
            curr_start_idx = i + 1
            curr_words = next_words
        else:
            curr_sentences.append(next_sent)
            curr_words += next_words

    if curr_sentences:
        chunks.append({
            "text": " ".join(curr_sentences),
            "words": curr_words,
            "sentenceRange": [curr_start_idx, len(sentences) - 1],
        })

    return {
        "sentences": sentences,
        "distances": distances,
        "threshold": threshold,
        "splitIndices": split_indices,
        "chunks": chunks,
        "sentenceCount": len(sentences),
        "chunkCount": len(chunks),
    }
