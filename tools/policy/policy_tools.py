"""
BankGuard AI — Policy tools.

Tools:
  - search_policy   (semantic search via pgvector embeddings)
  - get_case_history
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from tools.schemas import (
    GetCaseHistoryInput,
    PolicyResult,
    SearchPolicyInput,
    ToolResponse,
)

log = structlog.get_logger(__name__)


def policy_tools(
    db: Any,
) -> dict[str, Callable[[dict], Awaitable[ToolResponse]]]:

    async def search_policy(raw: dict[str, Any]) -> ToolResponse:
        """
        Semantic policy search.
        Falls back to keyword search if embeddings aren't initialised yet.
        """
        inp = SearchPolicyInput(**raw)
        try:
            async with db.acquire() as conn:
                # Try vector similarity first
                try:
                    from memory.semantic.vector_store import get_embedding

                    query_vec = await get_embedding(inp.query)
                    # pgvector cosine similarity
                    rows = await conn.fetch(
                        """
                        SELECT ss.source_ref, ss.content, ss.metadata,
                               1 - (ss.embedding <=> $1::vector) AS score
                        FROM memory.semantic_store ss
                        WHERE ss.source_type = 'POLICY'
                          AND ($2::text IS NULL OR ss.metadata->>'category' = $2)
                        ORDER BY ss.embedding <=> $1::vector
                        LIMIT $3
                        """,
                        f"[{','.join(str(x) for x in query_vec)}]",
                        inp.category,
                        inp.top_k,
                    )
                    policies = []
                    for r in rows:
                        meta = json.loads(r["metadata"] or "{}")
                        policies.append(
                            PolicyResult(
                                policy_ref=r["source_ref"] or meta.get("policy_ref", ""),
                                title=meta.get("title", ""),
                                category=meta.get("category", ""),
                                content=r["content"],
                                relevance_score=float(r["score"]),
                            ).model_dump(mode="json")
                        )
                    if policies:
                        return ToolResponse.success("search_policy", {"policies": policies})
                except Exception as emb_err:
                    log.warning("embedding_search_failed_falling_back", error=str(emb_err))

                # Keyword fallback
                where_clause = "WHERE active = TRUE"
                params: list[Any] = []
                if inp.category:
                    where_clause += " AND category = $1"
                    params.append(inp.category)

                keyword_sql = f"""
                    SELECT policy_ref, title, category, content
                    FROM banking.policies
                    {where_clause}
                    AND (
                        to_tsvector('english', content) @@ plainto_tsquery('english', ${len(params) + 1})
                        OR title ILIKE '%' || ${len(params) + 2} || '%'
                    )
                    LIMIT ${len(params) + 3}
                """
                params.extend([inp.query, inp.query, inp.top_k])
                rows = await conn.fetch(keyword_sql, *params)
                policies = [
                    PolicyResult(
                        policy_ref=r["policy_ref"],
                        title=r["title"],
                        category=r["category"],
                        content=r["content"],
                        relevance_score=None,
                    ).model_dump(mode="json")
                    for r in rows
                ]
                if not policies:
                    # Last resort: return top k active policies by category
                    all_rows = await conn.fetch(
                        """
                        SELECT policy_ref, title, category, content
                        FROM banking.policies
                        WHERE active = TRUE
                        ORDER BY created_at DESC
                        LIMIT $1
                        """,
                        inp.top_k,
                    )
                    policies = [
                        PolicyResult(
                            policy_ref=r["policy_ref"],
                            title=r["title"],
                            category=r["category"],
                            content=r["content"],
                        ).model_dump(mode="json")
                        for r in all_rows
                    ]
                return ToolResponse.success("search_policy", {"policies": policies})
        except Exception as exc:
            log.exception("search_policy_error", error=str(exc))
            return ToolResponse.error("search_policy", "DB_ERROR", str(exc), retryable=True)

    async def get_case_history(raw: dict[str, Any]) -> ToolResponse:
        inp = GetCaseHistoryInput(**raw)
        try:
            async with db.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT c.case_ref, c.title, c.status, c.priority,
                           c.root_cause, c.resolution, c.confidence,
                           c.created_at, c.closed_at
                    FROM agent.cases c
                    JOIN banking.customers cu ON cu.id = c.customer_id
                    WHERE cu.customer_ref = $1
                    ORDER BY c.created_at DESC
                    LIMIT $2
                    """,
                    inp.customer_ref,
                    inp.limit,
                )
                cases = [
                    {
                        "case_ref": r["case_ref"],
                        "title": r["title"],
                        "status": r["status"],
                        "priority": r["priority"],
                        "root_cause": r["root_cause"],
                        "resolution": r["resolution"],
                        "confidence": r["confidence"],
                        "created_at": str(r["created_at"]),
                        "closed_at": str(r["closed_at"]) if r["closed_at"] else None,
                    }
                    for r in rows
                ]
                return ToolResponse.success("get_case_history", {"cases": cases})
        except Exception as exc:
            return ToolResponse.error("get_case_history", "DB_ERROR", str(exc), retryable=True)

    return {
        "search_policy": search_policy,
        "get_case_history": get_case_history,
    }
