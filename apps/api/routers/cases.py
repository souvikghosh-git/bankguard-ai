"""
BankGuard AI — Cases API router.

Endpoints:
  GET  /api/cases                  list cases
  GET  /api/cases/{case_ref}       get case details
  POST /api/cases                  create a case
  POST /api/cases/{case_ref}/investigate   trigger investigation
  GET  /api/cases/{case_ref}/notes         get case notes
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from apps.api.dependencies import DBDep, ValDep, IdentDep

router = APIRouter(prefix="/api/cases", tags=["Cases"])


# ── Request / Response models ─────────────────────────────────────────────────

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
    created_at: str
    updated_at: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[CaseResponse])
async def list_cases(
    db: DBDep,
    identity: IdentDep,
    status_filter: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List investigation cases."""
    async with db.acquire() as conn:
        where = "WHERE 1=1"
        params: list[Any] = []
        if status_filter:
            where += f" AND c.status = ${len(params)+1}"
            params.append(status_filter)
        params.append(limit)

        rows = await conn.fetch(
            f"""
            SELECT c.case_ref, c.title, c.description, c.status, c.priority,
                   cu.customer_ref, t.transaction_ref,
                   c.root_cause, c.resolution, c.confidence,
                   c.created_at::text, c.updated_at::text
            FROM agent.cases c
            LEFT JOIN banking.customers cu ON cu.id = c.customer_id
            LEFT JOIN banking.transactions t ON t.id = c.transaction_id
            {where}
            ORDER BY c.created_at DESC
            LIMIT ${len(params)}
            """,
            *params,
        )
    return [dict(r) for r in rows]


@router.get("/{case_ref}", response_model=CaseResponse)
async def get_case(case_ref: str, db: DBDep, identity: IdentDep) -> dict[str, Any]:
    """Get a single case with full details."""
    async with db.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT c.case_ref, c.title, c.description, c.status, c.priority,
                   cu.customer_ref, t.transaction_ref,
                   c.root_cause, c.resolution, c.confidence,
                   c.created_at::text, c.updated_at::text
            FROM agent.cases c
            LEFT JOIN banking.customers cu ON cu.id = c.customer_id
            LEFT JOIN banking.transactions t ON t.id = c.transaction_id
            WHERE c.case_ref = $1
            """,
            case_ref,
        )
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Case {case_ref} not found")
    return dict(row)


@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(
    req: CreateCaseRequest,
    db: DBDep,
    identity: IdentDep,
) -> dict[str, Any]:
    """Create a new investigation case."""
    case_ref = f"CASE-{uuid.uuid4().hex[:6].upper()}"
    case_id  = str(uuid.uuid4())

    async with db.acquire() as conn:
        # Resolve customer_id
        customer_id = None
        if req.customer_ref:
            row = await conn.fetchrow(
                "SELECT id FROM banking.customers WHERE customer_ref = $1", req.customer_ref
            )
            customer_id = row["id"] if row else None

        # Resolve transaction_id
        transaction_id = None
        if req.transaction_ref:
            row = await conn.fetchrow(
                "SELECT id FROM banking.transactions WHERE transaction_ref = $1", req.transaction_ref
            )
            transaction_id = row["id"] if row else None

        await conn.execute(
            """
            INSERT INTO agent.cases
                (id, case_ref, title, description, status, priority, customer_id, transaction_id)
            VALUES ($1, $2, $3, $4, 'OPEN', $5, $6, $7)
            """,
            case_id, case_ref, req.title, req.description,
            req.priority, customer_id, transaction_id,
        )

    return {
        "case_ref": case_ref,
        "title": req.title,
        "description": req.description,
        "status": "OPEN",
        "priority": req.priority,
        "customer_ref": req.customer_ref,
        "transaction_ref": req.transaction_ref,
        "root_cause": None,
        "resolution": None,
        "confidence": None,
        "created_at": str(uuid.uuid4()),  # placeholder; real app returns actual timestamp
        "updated_at": str(uuid.uuid4()),
    }


@router.get("/{case_ref}/notes")
async def get_case_notes(case_ref: str, db: DBDep, identity: IdentDep) -> list[dict]:
    """Get all notes for a case."""
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
