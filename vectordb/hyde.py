"""
vectordb/hyde.py
================
Hypothetical Document Embeddings (HyDE) Engine (Phase 4).

Why HyDE Improves Retrieval Quality:
------------------------------------
Traditional Bi-Encoder vector retrieval suffers from an inherent "Question vs. Document"
semantic mismatch:
- User Query: Short, interrogative, 5-10 words ("why does SQ8 save 4x memory?")
- Target Document: Long, declarative, technical explanation with formulas and details.

HyDE (Gao et al., 2022) bridges this gap:
1. Prompts the LLM (llama3.2) to generate a hypothetical, ideal passage that
   directly answers the user's question.
2. Even if the hypothetical passage has minor factual inaccuracies, its *embedding*
   resides in the same semantic and linguistic manifold as true technical documents.
3. The hypothetical passage embedding is used to search the HNSW vector database.
"""

import re
from typing import Callable, Dict, List, Tuple


def generate_hypothetical_doc(
    query: str,
    ollama_generate_fn: Callable[[str], str],
) -> str:
    """
    Generates a hypothetical document passage answering the user's query.

    Parameters
    ----------
    query : str
        The user's search query or question.
    ollama_generate_fn : Callable[[str], str]
        Function calling the local LLM (e.g., OllamaClient.generate).

    Returns
    -------
    str
        The generated hypothetical passage.
    """
    prompt = (
        "You are an expert technical writer. Write a clear, factual, and informative "
        f"paragraph that directly answers the following question: '{query}'.\n"
        "Do not include introductory greetings or meta-commentary. "
        "Write only the technical explanation passage."
    )
    raw = ollama_generate_fn(prompt)
    if not raw or raw.startswith("ERROR"):
        # Fallback to query itself if LLM is unavailable
        return query

    # Clean any conversational lead-in
    cleaned = re.sub(r"^(Here is a paragraph|Sure!|Here's an explanation):\s*", "", raw.strip(), flags=re.IGNORECASE)
    return cleaned if cleaned else query
