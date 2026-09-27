"""
BankGuard AI — Temporal Activities.

Activities are the side-effectful units inside a Temporal workflow.
Each activity can be retried independently; they are the boundary
between durable Temporal state and the outside world (DB, tools).

Activities defined here:
  request_approval_activity    — creates approval record in PostgreSQL
  poll_approval_decision       — queries DB for current decision
  execute_approved_tool        — calls the tool after approval
  expire_approval_activity     — marks timed-out approval as EXPIRED
  notify_approver_activity     — sends notification to reviewer (stub)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import asyncpg
import structlog
from temporalio import activity

from config import settings

log = structlog.get_logger(__name__)


# ── Input / output dataclasses ────────────────────────────────────────────────

@dataclass
class ApprovalRequestInput:
    case_ref: str
    run_id: str
    action_type: str           # tool name, e.g. "retry_payment"
    action_payload: dict       # tool input dict (JSON-serialisable)
    risk_level: str            # HIGH | CRITICAL
    requested_by: str = "agent"


@dataclass
class ApprovalDecision:
    approval_ref: str
    status: str                # PENDING | APPROVED | REJECTED | EXPIRED
    reviewed_by: str | None
    review_notes: str | None


@dataclass
class ToolExecutionInput:
    tool_name: str
    tool_input: dict
    run_id: str
    case_id: str
    identity: dict


@dataclass
class ToolExecutionResult:
    success: bool
    status: str
    data: Any
    error: str | None


# ── DB pool helper — activities share a module-level pool ─────────────────────

_pool: asyncpg.Pool | None = None


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url.replace("postgresql+asyncpg://", "postgresql://"),
            min_size=1,
            max_size=3,
            command_timeout=10,
        )
    return _pool


# ─────────────────────────────────────────────────────────────────────────────
# Activity: create approval record
# ─────────────────────────────────────────────────────────────────────────────

@activity.defn(name="request_approval")
async def request_approval_activity(inp: ApprovalRequestInput) -> str:
    """
    Writes a PENDING approval record to agent.approvals.
    Returns the approval_ref (e.g. APR-XXXXXXXXXXXX).
    """
    approval_ref = f"APR-{uuid.uuid4().hex[:10].upper()}"
    db = await _get_pool()

    async with db.acquire() as conn:
        case_row = await conn.fetchrow(
            "SELECT id FROM agent.cases WHERE case_ref = $1", inp.case_ref
        )
        run_row = await conn.fetchrow(
            "SELECT id FROM agent.agent_runs WHERE run_ref = $1", inp.run_id
        )
        await conn.execute(
            """
            INSERT INTO agent.approvals
                (id, approval_ref, case_id, run_id, action_type,
                 action_payload, risk_level, status, requested_by,
                 expires_at)
            VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,'PENDING',$8,
                    NOW() + INTERVAL '24 hours')
            """,
            str(uuid.uuid4()),
            approval_ref,
            case_row["id"] if case_row else None,
            run_row["id"] if run_row else None,
            inp.action_type,
            json.dumps(inp.action_payload),
            inp.risk_level,
            inp.requested_by,
        )

    log.info(
        "approval_record_created",
        approval_ref=approval_ref,
        case_ref=inp.case_ref,
        action=inp.action_type,
        risk=inp.risk_level,
    )
    activity.heartbeat(f"approval_ref={approval_ref}")
    return approval_ref


# ─────────────────────────────────────────────────────────────────────────────
# Activity: poll for decision (called on a timer inside the workflow)
# ─────────────────────────────────────────────────────────────────────────────

@activity.defn(name="poll_approval_decision")
async def poll_approval_decision(approval_ref: str) -> ApprovalDecision:
    """
    Reads the current status of an approval from the DB.
    The workflow calls this on a short-sleep/poll loop.
    """
    db = await _get_pool()
    async with db.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT status, reviewed_by, review_notes, expires_at
            FROM agent.approvals WHERE approval_ref = $1
            """,
            approval_ref,
        )

    if not row:
        return ApprovalDecision(
            approval_ref=approval_ref,
            status="NOT_FOUND",
            reviewed_by=None,
            review_notes=None,
        )

    from datetime import datetime, timezone
    status = row["status"]
    if status == "PENDING" and row["expires_at"] < datetime.now(timezone.utc):
        status = "EXPIRED"

    return ApprovalDecision(
        approval_ref=approval_ref,
        status=status,
        reviewed_by=row["reviewed_by"],
        review_notes=row["review_notes"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Activity: execute the tool after approval
# ─────────────────────────────────────────────────────────────────────────────

@activity.defn(name="execute_approved_tool")
async def execute_approved_tool(inp: ToolExecutionInput) -> ToolExecutionResult:
    """
    Runs the actual tool now that human approval has been received.
    Uses the Tool Gateway so all enforcement (idempotency, audit) still fires.
    """
    import redis.asyncio as aioredis

    db = await _get_pool()
    valkey = aioredis.from_url(settings.valkey_url, decode_responses=True)

    try:
        from tools.gateway import ToolGateway
        gw = ToolGateway(
            db_pool=db,
            valkey=valkey,
            run_id=inp.run_id,
            case_id=inp.case_id,
            identity=inp.identity,
        )

        # Temporarily elevate identity to RISK_OFFICER for the approved action
        # (the approval record IS the authorisation)
        elevated_identity = {**inp.identity, "role": "RISK_OFFICER", "_approved_via_temporal": True}
        gw.identity = elevated_identity

        # Bypass the approval gate now — temporarily remove from APPROVAL_REQUIRED
        from tools import gateway as gw_module
        saved = frozenset(gw_module.APPROVAL_REQUIRED)
        gw_module.APPROVAL_REQUIRED -= {inp.tool_name}
        try:
            result = await gw.call(inp.tool_name, inp.tool_input)
        finally:
            gw_module.APPROVAL_REQUIRED = set(saved)

        log.info(
            "approved_tool_executed",
            tool=inp.tool_name,
            status=str(result.status),
            run_id=inp.run_id,
        )
        return ToolExecutionResult(
            success=result.status.value == "SUCCESS",
            status=str(result.status),
            data=result.data,
            error=result.error_message,
        )
    finally:
        await valkey.aclose()


# ─────────────────────────────────────────────────────────────────────────────
# Activity: mark expired approvals
# ─────────────────────────────────────────────────────────────────────────────

@activity.defn(name="expire_approval")
async def expire_approval_activity(approval_ref: str) -> None:
    """Mark a timed-out approval as EXPIRED in the DB."""
    db = await _get_pool()
    async with db.acquire() as conn:
        await conn.execute(
            """
            UPDATE agent.approvals
            SET status = 'EXPIRED', reviewed_at = NOW()
            WHERE approval_ref = $1 AND status = 'PENDING'
            """,
            approval_ref,
        )
    log.info("approval_expired", approval_ref=approval_ref)


# ─────────────────────────────────────────────────────────────────────────────
# Activity: notify approver (stub — extend with SNS / email / Slack)
# ─────────────────────────────────────────────────────────────────────────────

@activity.defn(name="notify_approver")
async def notify_approver_activity(
    approval_ref: str, action_type: str, risk_level: str, case_ref: str
) -> None:
    """
    Send a notification to the approver queue.
    Currently logs only; extend with AWS SNS / SES / Slack webhook.
    """
    log.info(
        "approver_notification",
        approval_ref=approval_ref,
        action=action_type,
        risk=risk_level,
        case_ref=case_ref,
        message=(
            f"Action '{action_type}' (risk={risk_level}) for case {case_ref} "
            f"requires your approval. Ref: {approval_ref}"
        ),
    )
    # TODO: aws_sns.publish(TopicArn=..., Message=...) or httpx.post(slack_webhook, ...)
