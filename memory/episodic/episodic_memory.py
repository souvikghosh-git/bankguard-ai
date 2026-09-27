"""
BankGuard AI — Episodic Memory (PostgreSQL).

Records what happened during investigations so future agents can learn
from past cases. Each episode captures:
  - What evidence was found
  - What root cause was identified
  - What action was taken
  - What the outcome was

This is NOT the full audit log — that goes to agent.tool_calls.
Episodic memory is the compressed, searchable narrative.

Schema: memory.episodic
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class EpisodicMemory:
    """
    Persistent episodic memory for case investigations.

    Usage:
        ep = EpisodicMemory(db_pool)
        await ep.record(case_id, run_id, event_type, summary, evidence, outcome)
        episodes = await ep.recall(case_id)
        similar  = await ep.find_similar_cases(root_cause="BENEFICIARY_BANK_TIMEOUT", limit=3)
    """

    def __init__(self, db: Any) -> None:
        self.db = db

    async def record(
        self,
        case_id: str,
        run_id: str,
        event_type: str,
        summary: str,
        evidence: list[Any] | None = None,
        outcome: str | None = None,
    ) -> str:
        """Persist an episode. Returns the episode ID."""
        episode_id = str(uuid.uuid4())
        try:
            async with self.db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO memory.episodic
                        (id, case_id, run_id, event_type, summary, evidence, outcome)
                    VALUES ($1,
                            (SELECT id FROM agent.cases WHERE case_ref = $2 OR id::text = $2 LIMIT 1),
                            (SELECT id FROM agent.agent_runs WHERE run_ref = $3 OR id::text = $3 LIMIT 1),
                            $4, $5, $6::jsonb, $7)
                    """,
                    episode_id,
                    case_id,
                    run_id,
                    event_type,
                    summary,
                    __import__("json").dumps(evidence or []),
                    outcome,
                )
                log.debug(
                    "episodic_recorded",
                    episode_id=episode_id,
                    event_type=event_type,
                    outcome=outcome,
                )
        except Exception as exc:
            log.warning("episodic_record_error", error=str(exc))
        return episode_id

    async def recall(self, case_id: str, limit: int = 10) -> list[dict]:
        """Retrieve all episodes for a case (chronological)."""
        try:
            async with self.db.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT e.id, e.event_type, e.summary, e.evidence,
                           e.outcome, e.occurred_at
                    FROM memory.episodic e
                    JOIN agent.cases c ON c.id = e.case_id
                    WHERE c.case_ref = $1 OR c.id::text = $1
                    ORDER BY e.occurred_at ASC
                    LIMIT $2
                    """,
                    case_id,
                    limit,
                )
                return [
                    {
                        "id": str(r["id"]),
                        "event_type": r["event_type"],
                        "summary": r["summary"],
                        "evidence": r["evidence"],
                        "outcome": r["outcome"],
                        "occurred_at": str(r["occurred_at"]),
                    }
                    for r in rows
                ]
        except Exception as exc:
            log.warning("episodic_recall_error", error=str(exc))
            return []

    async def find_similar_cases(
        self,
        root_cause: str,
        limit: int = 3,
    ) -> list[dict]:
        """
        Find past episodes where the same root cause was identified.
        Useful for suggesting resolution patterns.
        """
        try:
            async with self.db.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT e.summary, e.outcome, e.occurred_at,
                           c.case_ref, c.resolution
                    FROM memory.episodic e
                    JOIN agent.cases c ON c.id = e.case_id
                    WHERE e.event_type = 'ROOT_CAUSE_IDENTIFIED'
                      AND e.summary ILIKE '%' || $1 || '%'
                    ORDER BY e.occurred_at DESC
                    LIMIT $2
                    """,
                    root_cause,
                    limit,
                )
                return [
                    {
                        "case_ref": r["case_ref"],
                        "summary": r["summary"],
                        "outcome": r["outcome"],
                        "resolution": r["resolution"],
                        "occurred_at": str(r["occurred_at"]),
                    }
                    for r in rows
                ]
        except Exception as exc:
            log.warning("episodic_find_similar_error", error=str(exc))
            return []
