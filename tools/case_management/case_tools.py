"""
BankGuard AI — Case management tools (write).

Tools:
  - create_case_note
  - create_operations_ticket
  - draft_customer_notification
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from tools.schemas import (
    CreateCaseNoteInput,
    CreateOperationsTicketInput,
    DraftCustomerNotificationInput,
    ToolResponse,
)

log = structlog.get_logger(__name__)


def case_tools(
    db: Any,
) -> dict[str, Callable[[dict], Awaitable[ToolResponse]]]:

    async def create_case_note(raw: dict[str, Any]) -> ToolResponse:
        inp = CreateCaseNoteInput(**raw)
        try:
            async with db.acquire() as conn:
                case_row = await conn.fetchrow(
                    "SELECT id FROM agent.cases WHERE case_ref = $1",
                    inp.case_ref,
                )
                if not case_row:
                    return ToolResponse.not_found("create_case_note", inp.case_ref)

                note_id = str(uuid.uuid4())
                await conn.execute(
                    """
                    INSERT INTO agent.case_notes
                        (id, case_id, note_type, content, created_by)
                    VALUES ($1, $2, $3, $4, 'agent')
                    """,
                    note_id,
                    case_row["id"],
                    inp.note_type,
                    inp.content,
                )
                # Update case updated_at
                await conn.execute(
                    "UPDATE agent.cases SET updated_at = NOW() WHERE id = $1",
                    case_row["id"],
                )
                return ToolResponse.success(
                    "create_case_note",
                    {"note_id": note_id, "case_ref": inp.case_ref, "note_type": inp.note_type},
                )
        except Exception as exc:
            log.exception("create_case_note_error", error=str(exc))
            return ToolResponse.error("create_case_note", "DB_ERROR", str(exc), retryable=True)

    async def create_operations_ticket(raw: dict[str, Any]) -> ToolResponse:
        inp = CreateOperationsTicketInput(**raw)
        # In production this integrates with ServiceNow/Jira
        # Simulator: store as a case note with TICKET type
        ticket_ref = f"TKT-{uuid.uuid4().hex[:8].upper()}"
        try:
            async with db.acquire() as conn:
                case_row = await conn.fetchrow("SELECT id FROM agent.cases WHERE case_ref = $1", inp.case_ref)
                if not case_row:
                    return ToolResponse.not_found("create_operations_ticket", inp.case_ref)

                note_content = (
                    f"OPERATIONS TICKET: {ticket_ref}\n"
                    f"Title: {inp.title}\n"
                    f"Priority: {inp.priority}\n"
                    f"Description: {inp.description}"
                )
                await conn.execute(
                    """
                    INSERT INTO agent.case_notes
                        (id, case_id, note_type, content, created_by)
                    VALUES ($1, $2, 'ACTION', $3, 'agent')
                    """,
                    str(uuid.uuid4()),
                    case_row["id"],
                    note_content,
                )
                return ToolResponse.success(
                    "create_operations_ticket",
                    {
                        "ticket_ref": ticket_ref,
                        "case_ref": inp.case_ref,
                        "title": inp.title,
                        "priority": inp.priority,
                        "status": "OPEN",
                    },
                )
        except Exception as exc:
            return ToolResponse.error("create_operations_ticket", "DB_ERROR", str(exc), retryable=True)

    async def draft_customer_notification(raw: dict[str, Any]) -> ToolResponse:
        inp = DraftCustomerNotificationInput(**raw)
        # Draft only — does NOT send. Requires human review before dispatch.
        draft_id = f"NOTIF-DRAFT-{uuid.uuid4().hex[:8].upper()}"
        try:
            async with db.acquire() as conn:
                case_row = await conn.fetchrow("SELECT id FROM agent.cases WHERE case_ref = $1", inp.case_ref)
                if not case_row:
                    return ToolResponse.not_found("draft_customer_notification", inp.case_ref)

                draft_content = (
                    f"NOTIFICATION DRAFT: {draft_id}\n"
                    f"Customer: {inp.customer_ref}\n"
                    f"Type: {inp.notification_type}\n"
                    f"Message: {inp.message}\n"
                    f"[PENDING HUMAN REVIEW — NOT SENT]"
                )
                await conn.execute(
                    """
                    INSERT INTO agent.case_notes
                        (id, case_id, note_type, content, created_by)
                    VALUES ($1, $2, 'ACTION', $3, 'agent')
                    """,
                    str(uuid.uuid4()),
                    case_row["id"],
                    draft_content,
                )
                return ToolResponse.success(
                    "draft_customer_notification",
                    {
                        "draft_id": draft_id,
                        "status": "DRAFT_PENDING_REVIEW",
                        "warning": "Notification has NOT been sent. Requires human review.",
                    },
                )
        except Exception as exc:
            return ToolResponse.error("draft_customer_notification", "DB_ERROR", str(exc), retryable=True)

    return {
        "create_case_note": create_case_note,
        "create_operations_ticket": create_operations_ticket,
        "draft_customer_notification": draft_customer_notification,
    }
