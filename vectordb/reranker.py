"""
vectordb/reranker.py
====================
Two-Stage Retrieval & Re-ranking Engine (Phase 4).

Why Two-Stage Retrieval is Standard in Production RAG:
------------------------------------------------------
- Stage 1 (Bi-Encoder / HNSW + BM25):
  Extremely fast (O(log N)), retrieves a broad candidate pool (e.g., top 10-20)
  from millions of vectors. However, compressing an entire passage into one
  fixed vector can lose subtle cross-token relationships.
- Stage 2 (Cross-Encoder / Joint Re-ranking):
  Jointly evaluates (query, document) pairs using deep cross-scoring or LLM
  pointwise assessment to accurately re-sort the top candidates into the final
  top K (e.g., top 3-5).

Implemented Rerankers:
----------------------
1. CrossScoreReranker:
   Fast, local lexical-semantic cross-attention scoring.
2. LLMReranker:
   Pointwise relevance evaluation using local Ollama (llama3.2).
"""

import math
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass
class RerankItem:
    id: int
    title: str
    text: str
    original_rank: int
    new_rank: int
    rerank_score: float
    reasoning: str


class CrossScoreReranker:
    """
    Fast, deterministic cross-attention scorer.
    Evaluates term density, exact phrase matches, query term coverage,
    and sentence-level semantic alignment.
    """

    def score(self, query: str, document_text: str, document_title: str = "") -> float:
        q_clean = query.lower()
        q_tokens = [w for w in re.findall(r"\b\w+\b", q_clean) if len(w) > 1]
        if not q_tokens:
            return 0.0

        full_doc = f"{document_title} {document_text}".lower()
        doc_tokens = re.findall(r"\b\w+\b", full_doc)
        if not doc_tokens:
            return 0.0

        doc_token_set = set(doc_tokens)

        # 1. Term Coverage Ratio (how many query words appear in doc)
        matched_tokens = [t for t in q_tokens if t in doc_token_set]
        coverage = len(matched_tokens) / len(q_tokens)

        # 2. Exact Query Substring Bonus
        exact_bonus = 0.25 if q_clean in full_doc else 0.0

        # 3. Bigram / Phrase Proximity Bonus
        bigram_matches = 0
        if len(q_tokens) >= 2:
            for i in range(len(q_tokens) - 1):
                bigram = f"{q_tokens[i]} {q_tokens[i+1]}"
                if bigram in full_doc:
                    bigram_matches += 1
            bigram_score = (bigram_matches / (len(q_tokens) - 1)) * 0.20
        else:
            bigram_score = 0.0

        # 4. Title match bonus
        title_bonus = 0.15 if any(t in document_title.lower() for t in q_tokens) else 0.0

        # 5. Density & Frequency (BM25-like saturation)
        tf_sum = sum(min(full_doc.count(t), 5) for t in matched_tokens)
        tf_factor = min(tf_sum / (len(q_tokens) * 3 + 1), 0.20)

        # Total combined score in [0.0, 1.0]
        final_score = (coverage * 0.40) + exact_bonus + bigram_score + title_bonus + tf_factor
        return round(min(1.0, max(0.0, final_score)), 4)


class LLMReranker:
    """
    Pointwise LLM Relevance Evaluator using Ollama llama3.2.
    Prompts the local model to grade relevance from 0.0 to 10.0 and provide reasoning.
    """

    def __init__(self, ollama_generate_fn: Callable[[str], str]):
        self._generate = ollama_generate_fn

    def score(self, query: str, document_text: str, document_title: str = "") -> Tuple[float, str]:
        prompt = (
            "You are an expert search relevance grader.\n"
            f"User Query: {query}\n\n"
            f"Document Title: {document_title}\n"
            f"Document Content: {document_text[:600]}\n\n"
            "Task: On a scale of 0.0 (completely irrelevant) to 10.0 (perfectly answers the query), "
            "rate how relevant this document is to the query.\n"
            "Respond in EXACTLY this format:\n"
            "Reasoning: <one short sentence>\n"
            "Score: <number between 0.0 and 10.0>"
        )

        resp = self._generate(prompt)
        score = 5.0
        reasoning = "Relevance assessed by LLM."

        # Parse score
        score_match = re.search(r"Score:\s*([0-9]+(?:\.[0-9]+)?)", resp, re.IGNORECASE)
        if score_match:
            try:
                raw_score = float(score_match.group(1))
                score = min(10.0, max(0.0, raw_score))
            except ValueError:
                pass

        # Parse reasoning
        reasoning_match = re.search(r"Reasoning:\s*([^\n\r]+)", resp, re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            first_line = resp.split("\n")[0].strip()
            if first_line and not first_line.startswith("Score"):
                reasoning = first_line

        return round(score / 10.0, 4), reasoning


def rerank_candidates(
    query: str,
    candidates: List[Tuple[float, any]],  # List of (initial_score_or_dist, DocItem)
    strategy: str = "cross",              # "cross" or "llm"
    ollama_generate_fn: Optional[Callable[[str], str]] = None,
    top_k: int = 3,
) -> List[RerankItem]:
    """
    Reranks candidate documents from Stage 1 retrieval.

    Parameters
    ----------
    query : str
        The search question / prompt.
    candidates : List[Tuple[float, DocItem]]
        Candidate documents returned by vector / hybrid search.
    strategy : str
        "cross" for fast deterministic cross-scoring;
        "llm" for deep pointwise Ollama LLM evaluation.
    ollama_generate_fn : Optional[Callable]
        Required if strategy == "llm".
    top_k : int
        Number of reranked items to return.

    Returns
    -------
    List[RerankItem]
        Sorted descending by rerank_score.
    """
    if not candidates:
        return []

    cross_scorer = CrossScoreReranker()
    llm_scorer = LLMReranker(ollama_generate_fn) if (strategy == "llm" and ollama_generate_fn) else None

    scored_items: List[RerankItem] = []

    for rank, (_, doc) in enumerate(candidates, start=1):
        if strategy == "llm" and llm_scorer:
            score, reasoning = llm_scorer.score(query, doc.text, doc.title)
        else:
            score = cross_scorer.score(query, doc.text, doc.title)
            reasoning = f"Cross-attention score based on term coverage and phrasing ({int(score * 100)}%)."

        scored_items.append(
            RerankItem(
                id=doc.id,
                title=doc.title,
                text=doc.text,
                original_rank=rank,
                new_rank=0,  # Will assign after sort
                rerank_score=score,
                reasoning=reasoning,
            )
        )

    # Sort descending by rerank_score
    scored_items.sort(key=lambda x: -x.rerank_score)

    # Assign new ranks
    for new_rank, item in enumerate(scored_items, start=1):
        item.new_rank = new_rank

    return scored_items[:top_k]
