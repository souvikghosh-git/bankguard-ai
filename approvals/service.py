"""
BankGuard AI — Human-in-the-Loop (HITL) Approval Service.

Risk matrix:
  READ tools           → AUTO (no approval)
  LOW-risk writes      → AUTO
  MEDIUM (refund ≤500) → AUTO
  MEDIUM (refund >500) → APPROVAL from Senior Analyst
  HIGH                 → APPROVAL from Risk Officer
  CRITICAL             → MANDATORY human approval (Risk Officer + Branch Manager)

Approval workflow:
  1. Agent recommends high-risk action
  2. Approval request created in agent.approvals (status=PENDING)
  3. Notification sent to approver (webhook / UI poll)
  4. Approver reviews evidence + action via Operations Portal
  5. Approver: APPROVE / REJECT / MODIFY / ESCALATE
  6. If approved: Temporal workflow resumes tool execution
  7. Full audit trail written to DB

Durable wait is handled by Temporal (workflows/temporal/).
This module handles the DB-side approval lifecycle.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# How long an approval request stays valid before auto-expiry
APPROVAL_TTL_HOURS = 24


class ApprovalService:
    """
    Manages approval request lifecycle.

    Usage:
        svc = ApprovalService(db_pool)
        approval_ref = await svc.request_approval(case_ref, run_id, action_type, payload, risk_level)
        status = await svc.get_status(approval_ref)
        await svc.approve(approval_ref, reviewer_id, notes)
        await svc.reject(approval_ref, reviewer_id, notes)
    """

    def __init__(self, db: Any) -> None:
        self.db = db

    # ── Risk classification ───────────────────────────────────────────────────

    @staticmethod
    def classify_risk(tool_name: str, payload: dict[str, Any]) -> str:
        """Return risk level for a given tool + payload."""
        risk_map = {
            "get_transaction_details":     "LOW",
            "get_account_summary":         "LOW",
            "get_customer_profile":        "LOW",
            "get_recent_transactions":     "LOW",
            "get_payment_status":          "LOW",
            "search_policy":               "LOW",
            "get_case_history":            "LOW",
            "check_transaction_limit":     "LOW",
            "get_payment_rail_status":     "LOW",
            "create_case_note":            "LOW",
            "create_operations_ticket":    "LOW",
            "draft_customer_notification": "LOW",
            "retry_payment":               "HIGH",
            "block_card":                  "HIGH",
            "freeze_account":              "CRITICAL",
            "reverse_transaction":         "CRITICAL",
        }
        risk = risk_map.get(tool_name, "MEDIUM")

        # Override: refund_fee is threshold-based
        if tool_name == "refund_fee":
            amount = payload.get("fee_amount", 0)
            risk = "MEDIUM" if float(amount) <= 500 else "HIGH"

        return risk

    @staticmethod
    def requires_approval(risk_level: str) -> bool:
        return risk_level in ("MEDIUM", "HIGH", "CRITICAL")

    # ── Request ───────────────────────────────────────────────────────────────

    async def request_approval(
        self,
        case_ref: str,
        run_id: str,
        action_type: str,
        action_payload: dict[str, Any],
        risk_level: str,
        requested_by: str = "agent",
    ) -> str:
        """Create an approval request. Returns approval_ref."""
        approval_ref = f"APR-{uuid.uuid4().hex[:10].upper()}"
        expires_at = datetime.now(timezone.utc) + timedelta(hours=APPROVAL_TTL_HOURS)

        try:
            async with self.db.acquire() as conn:
                # Resolve case_id and run_id
                case_row = await conn.fetchrow(
                    "SELECT id FROM agent.cases WHERE case_ref = $1", case_ref
                )
                run_row = await conn.fetchrow(
                    "SELECT id FROM agent.agent_runs WHERE run_ref = $1", run_id
                )

                import json
                await conn.execute(
                    """
                    INSERT INTO agent.approvals
                        (id, approval_ref, case_id, run_id, action_type,
                         action_payload, risk_level, status,
                         requested_by, expires_at)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, 'PENDING', $8, $9)
                    """,
                    str(uuid.uuid4()),
                    approval_ref,
                    case_row["id"] if case_row else None,
                    run_row["id"] if run_row else None,
                    action_type,
                    json.dumps(action_payload),
                    risk_level,
                    requested_by,
                    expires_at,
                )

            log.info(
                "approval_requested",
                approval_ref=approval_ref,
                action_type=action_type,
                risk_level=risk_level,
                case_ref=case_ref,
            )
        except Exception as exc:
            log.error("approval_request_error", error=str(exc))
            raise

        return approval_ref

    # ── Status ────────────────────────────────────────────────────────────────

    async def get_status(self, approval_ref: str) -> dict[str, Any] | None:
        try:
            async with self.db.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT approval_ref, action_type, action_payload, risk_level,
                           status, requested_by, reviewed_by, review_notes,
                           requested_at, reviewed_at, expires_at
                    FROM agent.approvals
                    WHERE approval_ref = $1
                    """,
                    approval_ref,
                )
                if not row:
                    return None
                return {
                    "approval_ref": row["approval_ref"],
                    "action_type": row["action_type"],
                    "action_payload": row["action_payload"],
                    "risk_level": row["risk_level"],
                    "status": row["status"],
                    "requested_by": row["requested_by"],
                    "reviewed_by": row["reviewed_by"],
                    "review_notes": row["review_notes"],
                    "requested_at": str(row["requested_at"]),
                    "reviewed_at": str(row["reviewed_at"]) if row["reviewed_at"] else None,
                    "expires_at": str(row["expires_at"]),
                    "is_expired": row["expires_at"] < datetime.now(timezone.utc),
                }
        except Exception as exc:
            log.error("approval_get_status_error", error=str(exc))
            return None

    # ── Approve ───────────────────────────────────────────────────────────────

    async def approve(
        self,
        approval_ref: str,
        reviewed_by: str,
        notes: str = "",
    ) -> bool:
        return await self._update_decision(approval_ref, "APPROVED", reviewed_by, notes)

    async def reject(
        self,
        approval_ref: str,
        reviewed_by: str,
        notes: str = "",
    ) -> bool:
        return await self._update_decision(approval_ref, "REJECTED", reviewed_by, notes)

    async def escalate(
        self,
        approval_ref: str,
        escalated_by: str,
        notes: str = "",
    ) -> bool:
        return await self._update_decision(approval_ref, "ESCALATED", escalated_by, notes)

    async def _update_decision(
        self,
        approval_ref: str,
        new_status: str,
        reviewed_by: str,
        notes: str,
    ) -> bool:
        try:
            async with self.db.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE agent.approvals
                    SET status      = $2,
                        reviewed_by = $3,
                        review_notes = $4,
                        reviewed_at = NOW()
                    WHERE approval_ref = $1
                      AND status = 'PENDING'
                    """,
                    approval_ref,
                    new_status,
                    reviewed_by,
                    notes,
                )
            updated = result != "UPDATE 0"
            log.info(
                "approval_decision",
                approval_ref=approval_ref,
                status=new_status,
                reviewer=reviewed_by,
                updated=updated,
            )
            return updated
        except Exception as exc:
            log.error("approval_decision_error", error=str(exc))
            return False

    # ── List pending ──────────────────────────────────────────────────────────

    async def list_pending(self, risk_level: str | None = None) -> list[dict[str, Any]]:
        """Return all pending approvals, optionally filtered by risk level."""
        try:
            async with self.db.acquire() as conn:
                if risk_level:
                    rows = await conn.fetch(
                        """
                        SELECT a.approval_ref, a.action_type, a.risk_level,
                               a.requested_at, a.expires_at, c.case_ref
                        FROM agent.approvals a
                        LEFT JOIN agent.cases c ON c.id = a.case_id
                        WHERE a.status = 'PENDING' AND a.risk_level = $1
                        ORDER BY a.requested_at ASC
                        """,
                        risk_level,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT a.approval_ref, a.action_type, a.risk_level,
                               a.requested_at, a.expires_at, c.case_ref
                        FROM agent.approvals a
                        LEFT JOIN agent.cases c ON c.id = a.case_id
                        WHERE a.status = 'PENDING'
                        ORDER BY a.requested_at ASC
                        """,
                    )
                return [
                    {
                        "approval_ref": r["approval_ref"],
                        "action_type":  r["action_type"],
                        "risk_level":   r["risk_level"],
                        "case_ref":     r["case_ref"],
                        "requested_at": str(r["requested_at"]),
                        "expires_at":   str(r["expires_at"]),
                    }
                    for r in rows
                ]
        except Exception as exc:
            log.error("approval_list_pending_error", error=str(exc))
            return []
