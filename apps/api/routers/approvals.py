"""
BankGuard AI — Approvals API router.

Endpoints:
  GET  /api/approvals           list pending approvals
  GET  /api/approvals/{ref}     get approval details
  POST /api/approvals/{ref}/approve
  POST /api/approvals/{ref}/reject
  POST /api/approvals/{ref}/escalate
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from apps.api.dependencies import DBDep, IdentDep
from approvals.service import ApprovalService

router = APIRouter(prefix="/api/approvals", tags=["Approvals"])


class DecisionRequest(BaseModel):
    notes: str = ""


@router.get("")
async def list_pending_approvals(
    db: DBDep,
    identity: IdentDep,
    risk_level: str | None = None,
) -> list[dict]:
    """List all pending approval requests."""
    svc = ApprovalService(db)
    return await svc.list_pending(risk_level)


@router.get("/{approval_ref}")
async def get_approval(approval_ref: str, db: DBDep, identity: IdentDep) -> dict:
    """Get approval request details."""
    svc = ApprovalService(db)
    approval = await svc.get_status(approval_ref)
    if not approval:
        raise HTTPException(status_code=404, detail=f"Approval {approval_ref} not found")
    return approval


@router.post("/{approval_ref}/approve", status_code=status.HTTP_200_OK)
async def approve_action(
    approval_ref: str,
    req: DecisionRequest,
    db: DBDep,
    identity: IdentDep,
) -> dict:
    """Approve a pending action."""
    # Only RISK_OFFICER and above can approve HIGH/CRITICAL risk
    role = identity.get("role", "")
    if role not in ("RISK_OFFICER", "SENIOR_ANALYST", "ADMIN"):
        raise HTTPException(status_code=403, detail="Insufficient role to approve")

    svc = ApprovalService(db)
    updated = await svc.approve(
        approval_ref=approval_ref,
        reviewed_by=identity.get("user_id", "unknown"),
        notes=req.notes,
    )
    if not updated:
        raise HTTPException(status_code=400, detail="Approval not found or already decided")
    return {"status": "APPROVED", "approval_ref": approval_ref}


@router.post("/{approval_ref}/reject", status_code=status.HTTP_200_OK)
async def reject_action(
    approval_ref: str,
    req: DecisionRequest,
    db: DBDep,
    identity: IdentDep,
) -> dict:
    """Reject a pending action."""
    svc = ApprovalService(db)
    updated = await svc.reject(
        approval_ref=approval_ref,
        reviewed_by=identity.get("user_id", "unknown"),
        notes=req.notes,
    )
    if not updated:
        raise HTTPException(status_code=400, detail="Approval not found or already decided")
    return {"status": "REJECTED", "approval_ref": approval_ref}


@router.post("/{approval_ref}/escalate", status_code=status.HTTP_200_OK)
async def escalate_action(
    approval_ref: str,
    req: DecisionRequest,
    db: DBDep,
    identity: IdentDep,
) -> dict:
    """Escalate an approval to the next level."""
    svc = ApprovalService(db)
    updated = await svc.escalate(
        approval_ref=approval_ref,
        escalated_by=identity.get("user_id", "unknown"),
        notes=req.notes,
    )
    if not updated:
        raise HTTPException(status_code=400, detail="Approval not found or already decided")
    return {"status": "ESCALATED", "approval_ref": approval_ref}
