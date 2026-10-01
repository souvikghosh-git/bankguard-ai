"""
BankGuard AI — Cases API router.

Endpoints:
  GET  /api/cases                  list cases
  GET  /api/cases/{case_ref}       get case details
  POST /api/cases                  create a case
  GET  /api/cases/{case_ref}/notes get case notes
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from apps.api.dependencies import DBDep, IdentDep

router = APIRouter(prefix="/api/cases", tags=["Cases"])


# ── Models ────────────────────────────────────────────────────────────────────


class CreateCaseRequest(BaseModel):
    title: str = Field(..., min_length=5)
    description: str = Field(..., min_length=10)
    priority: str = Field("MEDIUM", pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    customer_ref: str | None = None
    transaction_ref: str | None = None


class CaseResponse(BaseModel):
    case_ref: str
    title: str
    description: str
    status: str
    priority: str
    customer_ref: str | None = None
    transaction_ref: str | None = None
    root_cause: str | None = None
    resolution: str | None = None
    confidence: float | None = None
    # Fields written by the agent after investigation
    requires_human_approval: bool = False
    action_risk_level: str | None = None
    created_at: str
    updated_at: str


# ── SQL helpers ───────────────────────────────────────────────────────────────

_CASE_SELECT = """
    SELECT c.case_ref, c.title, c.description, c.status, c.priority,
           cu.customer_ref, t.transaction_ref,
           c.root_cause, c.resolution, c.confidence,
           COALESCE(c.requires_human_approval, false) AS requires_human_approval,
           c.action_risk_level,
           c.created_at::text, c.updated_at::text
    FROM agent.cases c
    LEFT JOIN banking.customers cu ON cu.id = c.customer_id
    LEFT JOIN banking.transactions t ON t.id = c.transaction_id
"""


def _row_to_dict(row: Any) -> dict[str, Any]:
    d = dict(row)
    # Ensure bool cast (asyncpg may return None)
    d["requires_human_approval"] = bool(d.get("requires_human_approval") or False)
    return d


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("", response_model=list[CaseResponse])
async def list_cases(
    db: DBDep,
    identity: IdentDep,
    status_filter: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    async with db.acquire() as conn:
        params: list[Any] = []
        where = "WHERE 1=1"
        if status_filter:
            params.append(status_filter)
            where += f" AND c.status = ${len(params)}"
        params.append(limit)
        rows = await conn.fetch(
            f"{_CASE_SELECT} {where} ORDER BY c.created_at DESC LIMIT ${len(params)}",
            *params,
        )
    return [_row_to_dict(r) for r in rows]


@router.get("/{case_ref}", response_model=CaseResponse)
async def get_case(case_ref: str, db: DBDep, identity: IdentDep) -> dict[str, Any]:
    async with db.acquire() as conn:
        row = await conn.fetchrow(f"{_CASE_SELECT} WHERE c.case_ref = $1", case_ref)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Case {case_ref} not found")
    return _row_to_dict(row)


@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(req: CreateCaseRequest, db: DBDep, identity: IdentDep) -> dict[str, Any]:
    case_ref = f"CASE-{uuid.uuid4().hex[:6].upper()}"
    case_id = str(uuid.uuid4())

    async with db.acquire() as conn:
        customer_id = None
        if req.customer_ref:
            row = await conn.fetchrow("SELECT id FROM banking.customers WHERE customer_ref = $1", req.customer_ref)
            customer_id = row["id"] if row else None

        transaction_id = None
        if req.transaction_ref:
            row = await conn.fetchrow(
                "SELECT id FROM banking.transactions WHERE transaction_ref = $1",
                req.transaction_ref,
            )
            transaction_id = row["id"] if row else None

        await conn.execute(
            """
            INSERT INTO agent.cases
                (id, case_ref, title, description, status, priority, customer_id, transaction_id)
            VALUES ($1, $2, $3, $4, 'OPEN', $5, $6, $7)
            """,
            case_id,
            case_ref,
            req.title,
            req.description,
            req.priority,
            customer_id,
            transaction_id,
        )

        # Fetch back the real row with actual timestamps
        row = await conn.fetchrow(f"{_CASE_SELECT} WHERE c.case_ref = $1", case_ref)

    return _row_to_dict(row)


@router.get("/{case_ref}/notes")
async def get_case_notes(case_ref: str, db: DBDep, identity: IdentDep) -> list[dict]:
    async with db.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT cn.note_type, cn.content, cn.created_by, cn.created_at::text
            FROM agent.case_notes cn
            JOIN agent.cases c ON c.id = cn.case_id
            WHERE c.case_ref = $1
            ORDER BY cn.created_at DESC
            """,
            case_ref,
        )
    return [dict(r) for r in rows]
