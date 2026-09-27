"""
BankGuard AI — Transaction tools (read-only).

Tools:
  - get_transaction_details
  - get_payment_status
  - get_recent_transactions
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from tools.schemas import (
    GetPaymentStatusInput,
    GetRecentTransactionsInput,
    GetTransactionInput,
    PaymentEvent,
    ToolResponse,
    TransactionDetails,
    TransactionStatus,
)

log = structlog.get_logger(__name__)


def transaction_tools(
    db: Any,
) -> dict[str, Callable[[dict], Awaitable[ToolResponse]]]:
    """Return a dict of tool_name → async handler bound to the db pool."""

    async def get_transaction_details(raw: dict[str, Any]) -> ToolResponse:
        inp = GetTransactionInput(**raw)
        try:
            async with db.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT
                        t.transaction_ref,
                        a.account_ref AS debit_account_ref,
                        t.amount, t.currency, t.transaction_type, t.status,
                        t.reference_number, t.description,
                        t.initiated_at, t.completed_at,
                        t.metadata
                    FROM banking.transactions t
                    JOIN banking.accounts a ON a.id = t.debit_account_id
                    WHERE t.transaction_ref = $1
                    """,
                    inp.transaction_ref,
                )
                if not row:
                    return ToolResponse.not_found("get_transaction_details", inp.transaction_ref)

                meta = json.loads(row["metadata"] or "{}")
                events: list[PaymentEvent] = []

                if inp.include_events:
                    event_rows = await conn.fetch(
                        """
                        SELECT event_type, event_code, event_message,
                               payment_rail, occurred_at, source_system
                        FROM banking.payment_events pe
                        JOIN banking.transactions t ON t.id = pe.transaction_id
                        WHERE t.transaction_ref = $1
                        ORDER BY occurred_at ASC
                        """,
                        inp.transaction_ref,
                    )
                    events = [
                        PaymentEvent(
                            event_type=e["event_type"],
                            event_code=e["event_code"],
                            event_message=e["event_message"],
                            payment_rail=e["payment_rail"],
                            occurred_at=e["occurred_at"],
                            source_system=e["source_system"],
                        )
                        for e in event_rows
                    ]

                details = TransactionDetails(
                    transaction_ref=row["transaction_ref"],
                    debit_account_ref=row["debit_account_ref"],
                    amount=float(row["amount"]),
                    currency=row["currency"],
                    transaction_type=row["transaction_type"],
                    status=TransactionStatus(row["status"]),
                    payment_rail=meta.get("payment_rail"),
                    reference_number=row["reference_number"],
                    description=row["description"],
                    initiated_at=row["initiated_at"],
                    completed_at=row["completed_at"],
                    beneficiary_bank=meta.get("beneficiary_bank"),
                    beneficiary_ifsc=meta.get("beneficiary_ifsc"),
                    events=events,
                    last_event_code=events[-1].event_code if events else None,
                )
                return ToolResponse.success(
                    tool_name="get_transaction_details",
                    data=details.model_dump(mode="json"),
                )
        except Exception as exc:
            log.exception("get_transaction_details_error", error=str(exc))
            return ToolResponse.error(
                "get_transaction_details",
                "DB_ERROR",
                str(exc),
                retryable=True,
            )

    async def get_payment_status(raw: dict[str, Any]) -> ToolResponse:
        inp = GetPaymentStatusInput(**raw)
        try:
            async with db.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT t.transaction_ref, t.status, t.metadata,
                           pe.event_code AS last_event_code,
                           pe.event_message AS last_event_message,
                           pe.occurred_at AS last_event_at
                    FROM banking.transactions t
                    LEFT JOIN LATERAL (
                        SELECT event_code, event_message, occurred_at
                        FROM banking.payment_events
                        WHERE transaction_id = t.id
                        ORDER BY occurred_at DESC
                        LIMIT 1
                    ) pe ON TRUE
                    WHERE t.transaction_ref = $1
                    """,
                    inp.transaction_ref,
                )
                if not row:
                    return ToolResponse.not_found("get_payment_status", inp.transaction_ref)

                meta = json.loads(row["metadata"] or "{}")
                return ToolResponse.success(
                    tool_name="get_payment_status",
                    data={
                        "transaction_ref": row["transaction_ref"],
                        "status": row["status"],
                        "payment_rail": meta.get("payment_rail"),
                        "last_event_code": row["last_event_code"],
                        "last_event_message": row["last_event_message"],
                        "last_event_at": str(row["last_event_at"]) if row["last_event_at"] else None,
                        "is_pending_reconciliation": row["status"] == "PENDING"
                        and row["last_event_code"] == "BENEFICIARY_BANK_TIMEOUT",
                    },
                )
        except Exception as exc:
            return ToolResponse.error("get_payment_status", "DB_ERROR", str(exc), retryable=True)

    async def get_recent_transactions(raw: dict[str, Any]) -> ToolResponse:
        inp = GetRecentTransactionsInput(**raw)
        try:
            async with db.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT t.transaction_ref, t.amount, t.currency,
                           t.transaction_type, t.status, t.initiated_at,
                           t.metadata
                    FROM banking.transactions t
                    JOIN banking.accounts a ON a.id = t.debit_account_id
                    WHERE a.account_ref = $1
                      AND t.initiated_at >= NOW() - ($2 || ' days')::INTERVAL
                    ORDER BY t.initiated_at DESC
                    LIMIT $3
                    """,
                    inp.account_ref,
                    str(inp.days_back),
                    inp.limit,
                )
                txns = []
                for r in rows:
                    meta = json.loads(r["metadata"] or "{}")
                    txns.append(
                        {
                            "transaction_ref": r["transaction_ref"],
                            "amount": float(r["amount"]),
                            "currency": r["currency"],
                            "type": r["transaction_type"],
                            "status": r["status"],
                            "payment_rail": meta.get("payment_rail"),
                            "initiated_at": str(r["initiated_at"]),
                        }
                    )
                return ToolResponse.success("get_recent_transactions", {"transactions": txns})
        except Exception as exc:
            return ToolResponse.error("get_recent_transactions", "DB_ERROR", str(exc), retryable=True)

    return {
        "get_transaction_details": get_transaction_details,
        "get_payment_status": get_payment_status,
        "get_recent_transactions": get_recent_transactions,
    }
