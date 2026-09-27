"""
BankGuard AI — Tool Gateway: Pydantic schemas.

Every tool input/output is strongly typed here.
The agent never sees raw dicts — only validated contracts.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────


class ToolStatus(str, Enum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    BLOCKED = "BLOCKED"
    TIMEOUT = "TIMEOUT"
    NOT_FOUND = "NOT_FOUND"


class TransactionStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REVERSED = "REVERSED"
    CANCELLED = "CANCELLED"


class PaymentRail(str, Enum):
    IMPS = "IMPS"
    NEFT = "NEFT"
    UPI = "UPI"
    RTGS = "RTGS"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AccountStatus(str, Enum):
    ACTIVE = "ACTIVE"
    FROZEN = "FROZEN"
    CLOSED = "CLOSED"


class CaseStatus(str, Enum):
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    ESCALATED = "ESCALATED"


# ──────────────────────────────────────────────────────────────────────────────
# Base response wrapper — every tool returns this
# ──────────────────────────────────────────────────────────────────────────────


class ToolResponse(BaseModel):
    """Standard envelope for all tool responses."""

    status: ToolStatus
    tool_name: str
    idempotency_key: str | None = None
    data: Any = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def success(
        cls,
        tool_name: str,
        data: Any,
        idempotency_key: str | None = None,
        **meta: Any,
    ) -> "ToolResponse":
        return cls(
            status=ToolStatus.SUCCESS,
            tool_name=tool_name,
            data=data,
            idempotency_key=idempotency_key,
            metadata=meta,
        )

    @classmethod
    def error(
        cls,
        tool_name: str,
        error_code: str,
        error_message: str,
        retryable: bool = False,
    ) -> "ToolResponse":
        return cls(
            status=ToolStatus.ERROR,
            tool_name=tool_name,
            error_code=error_code,
            error_message=error_message,
            retryable=retryable,
        )

    @classmethod
    def blocked(cls, tool_name: str, reason: str) -> "ToolResponse":
        return cls(
            status=ToolStatus.BLOCKED,
            tool_name=tool_name,
            error_code="PERMISSION_DENIED",
            error_message=reason,
            retryable=False,
        )

    @classmethod
    def not_found(cls, tool_name: str, resource: str) -> "ToolResponse":
        return cls(
            status=ToolStatus.NOT_FOUND,
            tool_name=tool_name,
            error_code="NOT_FOUND",
            error_message=f"{resource} not found",
            retryable=False,
        )


# ──────────────────────────────────────────────────────────────────────────────
# READ tool inputs
# ──────────────────────────────────────────────────────────────────────────────


class GetTransactionInput(BaseModel):
    transaction_ref: str = Field(..., description="Transaction reference e.g. TXN-30001")
    include_events: bool = Field(True, description="Include payment event timeline")


class GetAccountSummaryInput(BaseModel):
    account_ref: str = Field(..., description="Account reference e.g. ACC-20001")


class GetCustomerProfileInput(BaseModel):
    customer_ref: str = Field(..., description="Customer reference e.g. CUST-10001")
    include_accounts: bool = Field(True, description="Include linked accounts")


class GetRecentTransactionsInput(BaseModel):
    account_ref: str = Field(..., description="Account reference")
    limit: int = Field(10, ge=1, le=50, description="Max transactions to return")
    days_back: int = Field(30, ge=1, le=365, description="Look-back window in days")


class GetPaymentStatusInput(BaseModel):
    transaction_ref: str = Field(..., description="Transaction reference")


class SearchPolicyInput(BaseModel):
    query: str = Field(..., description="Natural-language policy query")
    category: str | None = Field(None, description="Filter by category: PAYMENT|REFUND|LIMIT|COMPLIANCE")
    top_k: int = Field(3, ge=1, le=10, description="Number of policies to return")


class GetCaseHistoryInput(BaseModel):
    customer_ref: str = Field(..., description="Customer reference")
    limit: int = Field(5, ge=1, le=20, description="Max cases to return")


class CheckTransactionLimitInput(BaseModel):
    account_ref: str = Field(..., description="Account reference")
    amount: float = Field(..., gt=0, description="Amount in INR")
    payment_rail: PaymentRail = Field(..., description="Payment rail")


# ──────────────────────────────────────────────────────────────────────────────
# WRITE tool inputs
# ──────────────────────────────────────────────────────────────────────────────


class CreateCaseNoteInput(BaseModel):
    case_ref: str = Field(..., description="Case reference e.g. CASE-9845")
    content: str = Field(..., min_length=10, description="Note content")
    note_type: str = Field(
        "INVESTIGATION",
        description="Note type: INVESTIGATION|ACTION|RESOLUTION|ESCALATION",
    )


class CreateOperationsTicketInput(BaseModel):
    case_ref: str = Field(..., description="Case reference")
    title: str = Field(..., description="Ticket title")
    description: str = Field(..., description="Detailed description")
    priority: str = Field("MEDIUM", description="Priority: LOW|MEDIUM|HIGH|CRITICAL")


class DraftCustomerNotificationInput(BaseModel):
    case_ref: str = Field(..., description="Case reference")
    customer_ref: str = Field(..., description="Customer reference")
    notification_type: str = Field(
        ..., description="Type: TRANSACTION_UPDATE|CASE_OPENED|RESOLUTION|DELAY"
    )
    message: str = Field(..., description="Draft message content (will be reviewed before send)")


class RetryPaymentInput(BaseModel):
    """HIGH-RISK: Requires human approval before execution."""
    transaction_ref: str = Field(..., description="Original transaction reference")
    reason: str = Field(..., description="Documented reason for retry")
    idempotency_key: str = Field(..., description="Unique key to prevent duplicate execution")


class RefundFeeInput(BaseModel):
    """MEDIUM-RISK: May require approval above threshold."""
    transaction_ref: str = Field(..., description="Transaction reference")
    fee_amount: float = Field(..., gt=0, description="Fee amount in INR to refund")
    reason: str = Field(..., description="Documented reason for refund")
    idempotency_key: str = Field(..., description="Unique key to prevent duplicate refund")


# ──────────────────────────────────────────────────────────────────────────────
# Rich output data models
# ──────────────────────────────────────────────────────────────────────────────


class PaymentEvent(BaseModel):
    event_type: str
    event_code: str
    event_message: str
    payment_rail: str | None = None
    occurred_at: datetime
    source_system: str | None = None


class TransactionDetails(BaseModel):
    transaction_ref: str
    debit_account_ref: str
    amount: float
    currency: str
    transaction_type: str
    status: TransactionStatus
    payment_rail: str | None = None
    reference_number: str | None = None
    description: str | None = None
    initiated_at: datetime
    completed_at: datetime | None = None
    beneficiary_bank: str | None = None
    beneficiary_ifsc: str | None = None
    events: list[PaymentEvent] = Field(default_factory=list)
    last_event_code: str | None = None


class AccountSummary(BaseModel):
    account_ref: str
    account_type: str
    balance: float
    currency: str
    status: AccountStatus
    bank_name: str | None = None
    ifsc_code: str | None = None


class CustomerProfile(BaseModel):
    customer_ref: str
    full_name: str
    kyc_status: str
    risk_category: str
    accounts: list[AccountSummary] = Field(default_factory=list)


class PolicyResult(BaseModel):
    policy_ref: str
    title: str
    category: str
    content: str
    relevance_score: float | None = None


class LimitCheckResult(BaseModel):
    account_ref: str
    amount: float
    payment_rail: str
    within_limit: bool
    limit_amount: float
    daily_used: float
    daily_limit: float
    breach_reason: str | None = None
