"""
BankGuard AI — Temporal Workflow: ApprovalWorkflow.

This is a durable workflow — it survives process restarts, EC2 reboots,
and network outages. Temporal replays the event history to restore state.

Approval lifecycle inside this workflow:
  1. Create approval record in DB (request_approval activity)
  2. Notify approver (notify_approver activity)
  3. Poll DB every 30 s for up to 24 h (poll_approval_decision activity)
  4a. APPROVED  → execute_approved_tool activity → return result
  4b. REJECTED  → return rejection result (no tool call)
  4c. EXPIRED   → expire_approval activity → return timeout result

Process-restart safety:
  Temporal replays workflow history on restart. The wait loop uses
  workflow.sleep() which is durable — the workflow resumes exactly
  where it paused even after an EC2 reboot during the wait.

Usage from agent code:
    from temporalio.client import Client
    from workflows.temporal.workflows import ApprovalWorkflow
    from workflows.temporal.activities import ApprovalRequestInput

    client = await Client.connect(settings.temporal_host,
                                  namespace=settings.temporal_namespace)
    result = await client.execute_workflow(
        ApprovalWorkflow.run,
        ApprovalWorkflowInput(
            request=ApprovalRequestInput(...),
            tool_name="retry_payment",
            tool_input={...},
            run_id=run_id,
            case_id=case_id,
            identity=identity,
        ),
        id=f"approval-{case_ref}-{tool_name}",
        task_queue=APPROVAL_TASK_QUEUE,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import structlog
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from workflows.temporal.activities import (
        ApprovalDecision,
        ApprovalRequestInput,
        ToolExecutionInput,
        ToolExecutionResult,
        execute_approved_tool,
        expire_approval_activity,
        notify_approver_activity,
        poll_approval_decision,
        request_approval_activity,
    )

log = structlog.get_logger(__name__)

APPROVAL_TASK_QUEUE = "bankguard-approval"

# How long to wait for a human decision before auto-expiry
APPROVAL_TIMEOUT_HOURS = 24
# How often to poll the DB for a decision
POLL_INTERVAL_SECONDS = 30


@dataclass
class ApprovalWorkflowInput:
    request: ApprovalRequestInput
    tool_name: str
    tool_input: dict
    run_id: str
    case_id: str
    identity: dict


@dataclass
class ApprovalWorkflowResult:
    approval_ref: str
    decision: str  # APPROVED | REJECTED | EXPIRED
    tool_executed: bool
    tool_result: ToolExecutionResult | None
    reviewed_by: str | None
    review_notes: str | None


@workflow.defn(name="ApprovalWorkflow")
class ApprovalWorkflow:
    """
    Durable HITL approval workflow.
    Waits up to APPROVAL_TIMEOUT_HOURS for a human decision,
    then either executes the tool or returns the rejection/expiry result.
    """

    # Allow external signals to fast-path the poll loop
    _decision_signal: str | None = None

    @workflow.signal(name="approval_decision")
    async def receive_decision_signal(self, decision: str) -> None:
        """
        Optionally signal the workflow directly from the approval API
        instead of waiting for the next poll cycle.
        decision: "APPROVED" | "REJECTED"
        """
        self._decision_signal = decision

    @workflow.run
    async def run(self, inp: ApprovalWorkflowInput) -> ApprovalWorkflowResult:
        workflow.logger.info(
            "approval_workflow_started",
            case_ref=inp.request.case_ref,
            tool=inp.tool_name,
            risk=inp.request.risk_level,
        )

        # ── Step 1: Create approval record ───────────────────────────────────
        approval_ref = await workflow.execute_activity(
            request_approval_activity,
            inp.request,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # ── Step 2: Notify approver ───────────────────────────────────────────
        await workflow.execute_activity(
            notify_approver_activity,
            args=[
                approval_ref,
                inp.request.action_type,
                inp.request.risk_level,
                inp.request.case_ref,
            ],
            start_to_close_timeout=timedelta(seconds=15),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        # ── Step 3: Poll loop — up to APPROVAL_TIMEOUT_HOURS ─────────────────
        max_polls = (APPROVAL_TIMEOUT_HOURS * 3600) // POLL_INTERVAL_SECONDS
        decision: ApprovalDecision | None = None

        for _ in range(int(max_polls)):
            # Fast-path: check if a signal arrived while we slept
            if self._decision_signal is not None:
                break

            await workflow.sleep(timedelta(seconds=POLL_INTERVAL_SECONDS))

            decision = await workflow.execute_activity(
                poll_approval_decision,
                approval_ref,
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            if decision.status in ("APPROVED", "REJECTED", "EXPIRED"):
                break

        # Resolve final status (signal override or polled result)
        if self._decision_signal:
            final_status = self._decision_signal
            reviewed_by = None
            review_notes = "Received via direct workflow signal"
        elif decision:
            final_status = decision.status
            reviewed_by = decision.reviewed_by
            review_notes = decision.review_notes
        else:
            final_status = "EXPIRED"
            reviewed_by = None
            review_notes = "Poll loop exhausted without decision"

        # ── Step 4a: Approved → execute the tool ─────────────────────────────
        if final_status == "APPROVED":
            tool_result = await workflow.execute_activity(
                execute_approved_tool,
                ToolExecutionInput(
                    tool_name=inp.tool_name,
                    tool_input=inp.tool_input,
                    run_id=inp.run_id,
                    case_id=inp.case_id,
                    identity=inp.identity,
                ),
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(
                    maximum_attempts=2,
                    non_retryable_error_types=["ValueError", "PermissionError"],
                ),
            )
            workflow.logger.info(
                "approval_workflow_approved_and_executed",
                approval_ref=approval_ref,
                tool=inp.tool_name,
                success=tool_result.success,
            )
            return ApprovalWorkflowResult(
                approval_ref=approval_ref,
                decision="APPROVED",
                tool_executed=True,
                tool_result=tool_result,
                reviewed_by=reviewed_by,
                review_notes=review_notes,
            )

        # ── Step 4b/c: Rejected or Expired — clean up, no tool call ──────────
        if final_status == "EXPIRED":
            await workflow.execute_activity(
                expire_approval_activity,
                approval_ref,
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )

        workflow.logger.info(
            "approval_workflow_not_approved",
            approval_ref=approval_ref,
            decision=final_status,
        )
        return ApprovalWorkflowResult(
            approval_ref=approval_ref,
            decision=final_status,
            tool_executed=False,
            tool_result=None,
            reviewed_by=reviewed_by,
            review_notes=review_notes,
        )
