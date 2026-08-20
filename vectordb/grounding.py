"""
vectordb/grounding.py
=====================
Sentence-Level Grounding & Citation Extraction Engine (Phase 4).

Why Grounding is Essential in Production RAG:
---------------------------------------------
LLMs are prone to hallucinations or extrapolating beyond the provided context.
Grounding performs post-generation verification:
1. Deconstructs the generated response into individual claims / sentences.
2. Performs sentence-to-sentence semantic verification against all retrieved documents.
3. Quantifies factuality confidence (0.0 to 1.0) and identifies exact source citations.
4. Detects ungrounded or speculative statements.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class GroundedSentence:
    sentence: str
    citation_id: Optional[int]         # 1-indexed citation number (e.g., [1])
    source_doc_id: Optional[int]
    source_doc_title: Optional[str]
    source_sentence: Optional[str]
    confidence: float                  # 0.0 - 1.0
    status: str                        # "grounded", "partial", "ungrounded"


@dataclass
class GroundingReport:
    grounded_sentences: List[GroundedSentence]
    overall_confidence: float          # Average confidence across all sentences
    grounding_rate: float              # Percentage of sentences classified as grounded (>= 0.70)
    cited_docs: List[Dict]             # Unique list of cited documents
    annotated_answer: str              # Answer with inline [1], [2] citation markers


def _extract_sentences(text: str) -> List[str]:
    """Extracts clean sentences from markdown / LLM output."""
    if not text:
        return []
    # Split on punctuation followed by space or newline
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|(?<=[.!?])\n+", text.strip())
    sentences = []
    for s in raw:
        s_clean = s.strip()
        if len(s_clean.split()) >= 3:  # Only meaningful sentences
            sentences.append(s_clean)
        elif sentences and s_clean:
            sentences[-1] += " " + s_clean
        elif s_clean:
            sentences.append(s_clean)
    return sentences if sentences else [text.strip()]


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Computes cosine similarity = 1.0 (identical) to -1.0."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def ground_answer(
    answer: str,
    retrieved_docs: List[any],           # List of DocItem or dict with .title, .text, .id
    embed_fn: Callable[[str], List[float]],
    grounded_threshold: float = 0.68,
    partial_threshold: float = 0.50,
) -> GroundingReport:
    """
    Evaluates each sentence of the LLM answer against retrieved document chunks.

    Parameters
    ----------
    answer : str
        The full generated text from the LLM.
    retrieved_docs : list
        The context documents that were provided to the LLM.
    embed_fn : Callable
        Function returning embedding vector for a string.
    grounded_threshold : float
        Similarity threshold for "grounded" (default 0.68).
    partial_threshold : float
        Similarity threshold for "partial" (default 0.50).

    Returns
    -------
    GroundingReport
    """
    answer_sentences = _extract_sentences(answer)
    if not answer_sentences or not retrieved_docs:
        return GroundingReport(
            grounded_sentences=[],
            overall_confidence=0.0,
            grounding_rate=0.0,
            cited_docs=[],
            annotated_answer=answer,
        )

    # 1. Break retrieved docs into sentences and embed them
    doc_sentence_pool: List[Tuple[int, str, str, List[float]]] = []  # (doc_id, doc_title, sent_text, sent_emb)
    doc_id_to_citation: Dict[int, int] = {}
    citation_counter = 1

    for doc in retrieved_docs:
        doc_id = getattr(doc, "id", doc.get("id") if isinstance(doc, dict) else 0)
        doc_title = getattr(doc, "title", doc.get("title") if isinstance(doc, dict) else "")
        doc_text = getattr(doc, "text", doc.get("text") if isinstance(doc, dict) else "")

        if doc_id not in doc_id_to_citation:
            doc_id_to_citation[doc_id] = citation_counter
            citation_counter += 1

        doc_sentences = _extract_sentences(doc_text)
        for d_sent in doc_sentences:
            emb = embed_fn(d_sent)
            if emb:
                doc_sentence_pool.append((doc_id, doc_title, d_sent, emb))

    # 2. Compare each answer sentence against the pool
    grounded_results: List[GroundedSentence] = []
    annotated_parts: List[str] = []
    total_conf = 0.0
    grounded_count = 0

    for a_sent in answer_sentences:
        a_emb = embed_fn(a_sent)
        best_sim = -1.0
        best_match: Optional[Tuple[int, str, str]] = None

        if a_emb and doc_sentence_pool:
            for doc_id, doc_title, d_sent, d_emb in doc_sentence_pool:
                sim = _cosine_sim(a_emb, d_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_match = (doc_id, doc_title, d_sent)

        conf = round(max(0.0, best_sim), 4) if best_sim > 0 else 0.0
        total_conf += conf

        if conf >= grounded_threshold:
            status = "grounded"
            grounded_count += 1
        elif conf >= partial_threshold:
            status = "partial"
        else:
            status = "ungrounded"

        cit_id = doc_id_to_citation.get(best_match[0]) if (best_match and status != "ungrounded") else None

        grounded_item = GroundedSentence(
            sentence=a_sent,
            citation_id=cit_id,
            source_doc_id=best_match[0] if best_match else None,
            source_doc_title=best_match[1] if best_match else None,
            source_sentence=best_match[2] if best_match else None,
            confidence=conf,
            status=status,
        )
        grounded_results.append(grounded_item)

        # Annotate inline citation
        if cit_id:
            annotated_parts.append(f"{a_sent} [{cit_id}]")
        else:
            annotated_parts.append(a_sent)

    num_sents = len(answer_sentences)
    avg_conf = round(total_conf / num_sents, 4) if num_sents > 0 else 0.0
    ground_rate = round((grounded_count / num_sents) * 100.0, 1) if num_sents > 0 else 0.0

    # Build unique cited docs
    cited_docs_list = []
    seen_doc_ids = set()
    for item in grounded_results:
        if item.source_doc_id and item.source_doc_id not in seen_doc_ids and item.status != "ungrounded":
            seen_doc_ids.add(item.source_doc_id)
            cited_docs_list.append({
                "citationId": item.citation_id,
                "docId": item.source_doc_id,
                "title": item.source_doc_title,
                "confidence": item.confidence,
            })

    return GroundingReport(
        grounded_sentences=grounded_results,
        overall_confidence=avg_conf,
        grounding_rate=ground_rate,
        cited_docs=cited_docs_list,
        annotated_answer=" ".join(annotated_parts),
    )
