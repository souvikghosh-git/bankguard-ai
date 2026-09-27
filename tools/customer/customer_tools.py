"""
BankGuard AI — Customer tools (read-only).

Tools:
  - get_customer_profile
  - get_account_summary
  - check_transaction_limit
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from tools.schemas import (
    AccountStatus,
    AccountSummary,
    CheckTransactionLimitInput,
    CustomerProfile,
    GetAccountSummaryInput,
    GetCustomerProfileInput,
    LimitCheckResult,
    ToolResponse,
)

log = structlog.get_logger(__name__)

# Per-rail limits in INR (from PAY-LIM-301)
RAIL_LIMITS: dict[str, dict[str, float]] = {
    "IMPS": {"per_txn": 500_000, "per_day": 1_000_000},
    "NEFT": {"per_txn": 1_000_000, "per_day": 2_000_000},
    "UPI": {"per_txn": 100_000, "per_day": 200_000},
    "RTGS": {"per_txn": 10_000_000, "per_day": 50_000_000},
}


def customer_tools(
    db: Any,
) -> dict[str, Callable[[dict], Awaitable[ToolResponse]]]:

    async def get_customer_profile(raw: dict[str, Any]) -> ToolResponse:
        inp = GetCustomerProfileInput(**raw)
        try:
            async with db.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT customer_ref, full_name, kyc_status, risk_category
                    FROM banking.customers
                    WHERE customer_ref = $1
                    """,
                    inp.customer_ref,
                )
                if not row:
                    return ToolResponse.not_found("get_customer_profile", inp.customer_ref)

                accounts: list[AccountSummary] = []
                if inp.include_accounts:
                    acc_rows = await conn.fetch(
                        """
                        SELECT a.account_ref, a.account_type, a.balance, a.currency,
                               a.status, a.bank_name, a.ifsc_code
                        FROM banking.accounts a
                        JOIN banking.customers c ON c.id = a.customer_id
                        WHERE c.customer_ref = $1
                        ORDER BY a.created_at
                        """,
                        inp.customer_ref,
                    )
                    accounts = [
                        AccountSummary(
                            account_ref=a["account_ref"],
                            account_type=a["account_type"],
                            balance=float(a["balance"]),
                            currency=a["currency"],
                            status=AccountStatus(a["status"]),
                            bank_name=a["bank_name"],
                            ifsc_code=a["ifsc_code"],
                        )
                        for a in acc_rows
                    ]

                profile = CustomerProfile(
                    customer_ref=row["customer_ref"],
                    full_name=row["full_name"],
                    kyc_status=row["kyc_status"],
                    risk_category=row["risk_category"],
                    accounts=accounts,
                )
                return ToolResponse.success(
                    "get_customer_profile",
                    profile.model_dump(mode="json"),
                )
        except Exception as exc:
            log.exception("get_customer_profile_error", error=str(exc))
            return ToolResponse.error("get_customer_profile", "DB_ERROR", str(exc), retryable=True)

    async def get_account_summary(raw: dict[str, Any]) -> ToolResponse:
        inp = GetAccountSummaryInput(**raw)
        try:
            async with db.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT account_ref, account_type, balance, currency,
                           status, bank_name, ifsc_code
                    FROM banking.accounts
                    WHERE account_ref = $1
                    """,
                    inp.account_ref,
                )
                if not row:
                    return ToolResponse.not_found("get_account_summary", inp.account_ref)

                summary = AccountSummary(
                    account_ref=row["account_ref"],
                    account_type=row["account_type"],
                    balance=float(row["balance"]),
                    currency=row["currency"],
                    status=AccountStatus(row["status"]),
                    bank_name=row["bank_name"],
                    ifsc_code=row["ifsc_code"],
                )
                return ToolResponse.success(
                    "get_account_summary",
                    summary.model_dump(mode="json"),
                )
        except Exception as exc:
            return ToolResponse.error("get_account_summary", "DB_ERROR", str(exc), retryable=True)

    async def check_transaction_limit(raw: dict[str, Any]) -> ToolResponse:
        inp = CheckTransactionLimitInput(**raw)
        try:
            limits = RAIL_LIMITS.get(inp.payment_rail.value, {"per_txn": 500_000, "per_day": 1_000_000})
            async with db.acquire() as conn:
                # Sum today's transactions on this rail from this account
                daily_used_row = await conn.fetchrow(
                    """
                    SELECT COALESCE(SUM(t.amount), 0) AS daily_used
                    FROM banking.transactions t
                    JOIN banking.accounts a ON a.id = t.debit_account_id
                    WHERE a.account_ref = $1
                      AND t.initiated_at >= CURRENT_DATE
                      AND t.status NOT IN ('FAILED', 'REVERSED')
                      AND t.metadata->>'payment_rail' = $2
                    """,
                    inp.account_ref,
                    inp.payment_rail.value,
                )
                daily_used = float(daily_used_row["daily_used"])
                within_per_txn = inp.amount <= limits["per_txn"]
                within_daily = (daily_used + inp.amount) <= limits["per_day"]

                breach_reason = None
                if not within_per_txn:
                    breach_reason = (
                        f"Amount ₹{inp.amount:,.2f} exceeds {inp.payment_rail.value} "
                        f"per-transaction limit of ₹{limits['per_txn']:,.2f}"
                    )
                elif not within_daily:
                    breach_reason = (
                        f"Adding ₹{inp.amount:,.2f} to today's ₹{daily_used:,.2f} "
                        f"exceeds {inp.payment_rail.value} daily limit of ₹{limits['per_day']:,.2f}"
                    )

                result = LimitCheckResult(
                    account_ref=inp.account_ref,
                    amount=inp.amount,
                    payment_rail=inp.payment_rail.value,
                    within_limit=within_per_txn and within_daily,
                    limit_amount=limits["per_txn"],
                    daily_used=daily_used,
                    daily_limit=limits["per_day"],
                    breach_reason=breach_reason,
                )
                return ToolResponse.success(
                    "check_transaction_limit",
                    result.model_dump(mode="json"),
                )
        except Exception as exc:
            return ToolResponse.error("check_transaction_limit", "DB_ERROR", str(exc), retryable=True)

    return {
        "get_customer_profile": get_customer_profile,
        "get_account_summary": get_account_summary,
        "check_transaction_limit": check_transaction_limit,
    }
