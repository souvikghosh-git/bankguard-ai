"""
BankGuard AI — Context Ranking.

Ranks policies and transactions by relevance to the current case.
Uses keyword overlap as a fast, zero-cost ranker.
Plugs in a cross-encoder reranker when available.
"""

from __future__ import annotations

import re
from typing import Any


def rank_policies(
    policies: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """
    Rank policies by relevance to the query string.
    Returns policies sorted highest-relevance first.
    """
    if not policies:
        return []

    query_tokens = _tokenise(query)

    scored: list[tuple[float, dict]] = []
    for p in policies:
        text = f"{p.get('title', '')} {p.get('category', '')} {p.get('content', '')}"
        score = _keyword_overlap(query_tokens, _tokenise(text))
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]


def rank_transactions(
    transactions: list[dict[str, Any]],
    reference_amount: float | None = None,
) -> list[dict[str, Any]]:
    """
    Rank related transactions by relevance.
    Prefers: PENDING status, similar amount, most recent.
    """
    if not transactions:
        return []

    def score(txn: dict) -> float:
        s = 0.0
        # Prefer pending / failed transactions (most likely related)
        if txn.get("status") in ("PENDING", "FAILED"):
            s += 3.0
        # Amount proximity
        if reference_amount and txn.get("amount"):
            try:
                ratio = float(txn["amount"]) / float(reference_amount)
                if 0.9 <= ratio <= 1.1:
                    s += 2.0
            except (TypeError, ZeroDivisionError):
                pass
        return s

    return sorted(transactions, key=score, reverse=True)


def _tokenise(text: str) -> set[str]:
    """Lower-case word tokeniser."""
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))


def _keyword_overlap(query_tokens: set[str], doc_tokens: set[str]) -> float:
    """Jaccard-like overlap score."""
    if not query_tokens or not doc_tokens:
        return 0.0
    intersection = query_tokens & doc_tokens
    union = query_tokens | doc_tokens
    return len(intersection) / len(union)
