from typing import List


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    """
    Splits text into overlapping word-boundary chunks.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    step = chunk_words - overlap_words
    if step <= 0:
        step = chunk_words

    chunks = []
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks
