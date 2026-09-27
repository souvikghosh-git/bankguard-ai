"""
BankGuard AI — Temporal Worker.

Runs a Temporal worker that processes approval workflows and activities.
Start this alongside the FastAPI server on the same EC2 instance.

Run:
    python -m workflows.temporal.worker
    # or via the start_local.sh script

The worker registers:
  - ApprovalWorkflow
  - All 5 activities (request, poll, execute, expire, notify)

It uses the same PostgreSQL + Valkey as the main application.
"""

from __future__ import annotations

import asyncio
import signal

import structlog
from temporalio.client import Client
from temporalio.worker import Worker

from config import settings
from observability.telemetry import setup_logging
from workflows.temporal.activities import (
    execute_approved_tool,
    expire_approval_activity,
    notify_approver_activity,
    poll_approval_decision,
    request_approval_activity,
)
from workflows.temporal.workflows import APPROVAL_TASK_QUEUE, ApprovalWorkflow

log = structlog.get_logger(__name__)


async def run_worker() -> None:
    setup_logging(settings.log_level)
    log.info(
        "temporal_worker_starting",
        host=settings.temporal_host,
        namespace=settings.temporal_namespace,
        queue=APPROVAL_TASK_QUEUE,
    )

    client = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
    )

    worker = Worker(
        client,
        task_queue=APPROVAL_TASK_QUEUE,
        workflows=[ApprovalWorkflow],
        activities=[
            request_approval_activity,
            poll_approval_decision,
            execute_approved_tool,
            expire_approval_activity,
            notify_approver_activity,
        ],
        # Tune for a t4g.xlarge: 4 vCPU, shared with other processes
        max_concurrent_workflow_tasks=20,
        max_concurrent_activities=10,
    )

    # Graceful shutdown on SIGTERM/SIGINT (Docker stop sends SIGTERM)
    stop_event = asyncio.Event()

    def _handle_signal(*_) -> None:
        log.info("temporal_worker_shutdown_signal_received")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handle_signal)

    log.info("temporal_worker_started", queue=APPROVAL_TASK_QUEUE)
    async with worker:
        await stop_event.wait()

    log.info("temporal_worker_stopped")


async def submit_approval_workflow(
    case_ref: str,
    run_id: str,
    action_type: str,
    action_payload: dict,
    risk_level: str,
    tool_name: str,
    tool_input: dict,
    case_id: str,
    identity: dict,
) -> str:
    """
    Helper used by the FastAPI layer to submit a new approval workflow.
    Returns the Temporal workflow ID.

    The caller can then poll the workflow result or wait on it with
    `client.get_workflow_handle(workflow_id).result()`.
    """
    from workflows.temporal.activities import ApprovalRequestInput
    from workflows.temporal.workflows import ApprovalWorkflowInput

    client = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
    )

    workflow_id = f"approval-{case_ref}-{action_type}-{run_id[:8]}"

    await client.start_workflow(
        ApprovalWorkflow.run,
        ApprovalWorkflowInput(
            request=ApprovalRequestInput(
                case_ref=case_ref,
                run_id=run_id,
                action_type=action_type,
                action_payload=action_payload,
                risk_level=risk_level,
            ),
            tool_name=tool_name,
            tool_input=tool_input,
            run_id=run_id,
            case_id=case_id,
            identity=identity,
        ),
        id=workflow_id,
        task_queue=APPROVAL_TASK_QUEUE,
    )

    log.info(
        "approval_workflow_submitted",
        workflow_id=workflow_id,
        case_ref=case_ref,
        action=action_type,
        risk=risk_level,
    )
    return workflow_id


async def signal_approval_decision(workflow_id: str, decision: str) -> None:
    """
    Send an APPROVED or REJECTED signal directly to a running workflow.
    Called from the approvals API router after a human clicks Approve/Reject.
    This fast-paths the poll loop so execution resumes within seconds.
    """
    client = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
    )
    handle = client.get_workflow_handle(workflow_id)
    await handle.signal(ApprovalWorkflow.receive_decision_signal, decision)
    log.info("approval_signal_sent", workflow_id=workflow_id, decision=decision)


if __name__ == "__main__":
    asyncio.run(run_worker())
