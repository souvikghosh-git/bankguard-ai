"""
BankGuard AI — Approvals API router.

After a human approves or rejects, this router:
  1. Updates agent.approvals in the DB (ApprovalService)
  2. Sends a Temporal signal to the waiting ApprovalWorkflow so it
     resumes immediately instead of waiting for the next poll cycle.

Endpoints:
  GET  /api/approvals                    list pending approvals
  GET  /api/approvals/{ref}              get approval details
  POST /api/approvals/{ref}/approve
  POST /api/approvals/{ref}/reject
  POST /api/approvals/{ref}/escalate
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from apps.api.dependencies import DBDep, IdentDep
from approvals.service import ApprovalService

router = APIRouter(prefix="/api/approvals", tags=["Approvals"])
log = structlog.get_logger(__name__)


class DecisionRequest(BaseModel):
    notes: str = ""
    # Optional: Temporal workflow ID to signal directly.
    # Set this when the approval was created via submit_approval_workflow().
    temporal_workflow_id: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _signal_temporal(workflow_id: str, decision: str) -> None:
    """Best-effort Temporal signal — never fail the HTTP response."""
    try:
        from workflows.temporal.worker import signal_approval_decision
        await signal_approval_decision(workflow_id, decision)
    except Exception as exc:
        log.warning(
            "temporal_signal_failed_approval_still_recorded",
            workflow_id=workflow_id,
            decision=decision,
            error=str(exc),
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("")
async def list_pending_approvals(
    db: DBDep,
    identity: IdentDep,
    risk_level: str | None = None,
) -> list[dict]:
    return await ApprovalService(db).list_pending(risk_level)


@router.get("/{approval_ref}")
async def get_approval(approval_ref: str, db: DBDep, identity: IdentDep) -> dict:
    svc = ApprovalService(db)
    approval = await svc.get_status(approval_ref)
    if not approval:
        raise HTTPException(status_code=404, detail=f"Approval {approval_ref} not found")
    return approval


@router.post("/{approval_ref}/approve")
async def approve_action(
    approval_ref: str,
    req: DecisionRequest,
    db: DBDep,
    identity: IdentDep,
) -> dict:
    """
    Approve a pending action.
    Requires RISK_OFFICER or ADMIN role for HIGH/CRITICAL actions.
    Signals the Temporal workflow so execution resumes within seconds.
    """
    role = identity.get("role", "")
    if role not in ("RISK_OFFICER", "SENIOR_ANALYST", "ADMIN"):
        raise HTTPException(status_code=403, detail="Insufficient role to approve HIGH/CRITICAL actions")

    svc = ApprovalService(db)
    updated = await svc.approve(
        approval_ref=approval_ref,
        reviewed_by=identity.get("user_id", "unknown"),
        notes=req.notes,
    )
    if not updated:
        raise HTTPException(status_code=400, detail="Approval not found or already decided")

    # Signal Temporal so the waiting workflow resumes immediately
    if req.temporal_workflow_id:
        await _signal_temporal(req.temporal_workflow_id, "APPROVED")

    log.info("approval_approved", ref=approval_ref, by=identity.get("user_id"))
    return {"status": "APPROVED", "approval_ref": approval_ref}


@router.post("/{approval_ref}/reject")
async def reject_action(
    approval_ref: str,
    req: DecisionRequest,
    db: DBDep,
    identity: IdentDep,
) -> dict:
    svc = ApprovalService(db)
    updated = await svc.reject(
        approval_ref=approval_ref,
        reviewed_by=identity.get("user_id", "unknown"),
        notes=req.notes,
    )
    if not updated:
        raise HTTPException(status_code=400, detail="Approval not found or already decided")

    if req.temporal_workflow_id:
        await _signal_temporal(req.temporal_workflow_id, "REJECTED")

    log.info("approval_rejected", ref=approval_ref, by=identity.get("user_id"))
    return {"status": "REJECTED", "approval_ref": approval_ref}


@router.post("/{approval_ref}/escalate")
async def escalate_action(
    approval_ref: str,
    req: DecisionRequest,
    db: DBDep,
    identity: IdentDep,
) -> dict:
    svc = ApprovalService(db)
    updated = await svc.escalate(
        approval_ref=approval_ref,
        escalated_by=identity.get("user_id", "unknown"),
        notes=req.notes,
    )
    if not updated:
        raise HTTPException(status_code=400, detail="Approval not found or already decided")

    log.info("approval_escalated", ref=approval_ref, by=identity.get("user_id"))
    return {"status": "ESCALATED", "approval_ref": approval_ref}
