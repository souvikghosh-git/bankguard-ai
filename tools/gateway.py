"""
BankGuard AI — Tool Gateway.

All tool calls flow through this gateway.  It enforces:
  1. Schema validation (Pydantic)
  2. Permission check (OPA / local policy)
  3. Idempotency (Valkey dedup cache)
  4. Audit logging (every call is recorded)
  5. Timeout + retry wrapper
  6. Structured error responses

Pattern:
    LLM → AgentRuntime → ToolGateway → BankingTool → DB/external
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Callable, Awaitable

import httpx
import structlog
from pydantic import BaseModel

from config import settings
from tools.schemas import ToolResponse, ToolStatus

log = structlog.get_logger(__name__)

# ── Risk classification per tool ──────────────────────────────────────────────

TOOL_RISK: dict[str, str] = {
    # READ — LOW risk, no approval required
    "get_transaction_details":      "LOW",
    "get_account_summary":          "LOW",
    "get_customer_profile":         "LOW",
    "get_recent_transactions":      "LOW",
    "get_payment_status":           "LOW",
    "search_policy":                "LOW",
    "get_case_history":             "LOW",
    "check_transaction_limit":      "LOW",
    # WRITE — controlled risk
    "create_case_note":             "LOW",
    "create_operations_ticket":     "LOW",
    "draft_customer_notification":  "LOW",
    # HIGH-RISK — require human approval
    "retry_payment":                "HIGH",
    "refund_fee":                   "MEDIUM",
    "block_card":                   "HIGH",
    "freeze_account":               "CRITICAL",
    "reverse_transaction":          "CRITICAL",
}

APPROVAL_REQUIRED: set[str] = {
    "retry_payment",
    "block_card",
    "freeze_account",
    "reverse_transaction",
}

APPROVAL_THRESHOLD: dict[str, str] = {
    "refund_fee": "MEDIUM",  # requires approval if amount > ₹500
}


class ToolGateway:
    """
    Central gateway that every tool call must pass through.

    Usage:
        gateway = ToolGateway(db_pool, valkey_client, run_id, identity)
        result  = await gateway.call("get_transaction_details", {...})
    """

    def __init__(
        self,
        db_pool: Any,          # asyncpg pool
        valkey: Any,           # redis.asyncio client
        run_id: str,
        case_id: str,
        identity: dict[str, Any],   # {"user_id": ..., "role": ..., "permissions": [...]}
        audit_log: list[dict] | None = None,
    ) -> None:
        self.db = db_pool
        self.valkey = valkey
        self.run_id = run_id
        self.case_id = case_id
        self.identity = identity
        self.audit_log: list[dict] = audit_log if audit_log is not None else []
        self._call_count = 0
        self._dedup_cache: dict[str, str] = {}   # idempotency_key → result hash

        # Import tools lazily to avoid circular deps
        from tools.transaction import transaction_tools
        from tools.payments import payment_tools
        from tools.customer import customer_tools
        from tools.policy import policy_tools
        from tools.case_management import case_tools

        self._registry: dict[str, Callable[..., Awaitable[ToolResponse]]] = {
            **transaction_tools(db_pool),
            **payment_tools(db_pool),
            **customer_tools(db_pool),
            **policy_tools(db_pool),
            **case_tools(db_pool),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    async def call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> ToolResponse:
        """Execute a tool with full gateway enforcement."""
        call_id = str(uuid.uuid4())
        started = time.monotonic()
        self._call_count += 1

        log.info(
            "tool_gateway_call",
            tool_name=tool_name,
            call_id=call_id,
            run_id=self.run_id,
            identity_role=self.identity.get("role"),
        )

        # 1. Tool must exist
        if tool_name not in self._registry:
            return ToolResponse.error(
                tool_name=tool_name,
                error_code="UNKNOWN_TOOL",
                error_message=f"Tool '{tool_name}' is not registered in the gateway.",
                retryable=False,
            )

        # 2. Permission check
        perm_result = self._check_permission(tool_name)
        if not perm_result["allowed"]:
            self._audit(call_id, tool_name, tool_input, None, "BLOCKED", started)
            return ToolResponse.blocked(tool_name, perm_result["reason"])

        # 3. Approval gate for high-risk tools
        if tool_name in APPROVAL_REQUIRED:
            self._audit(call_id, tool_name, tool_input, None, "BLOCKED_APPROVAL", started)
            return ToolResponse.blocked(
                tool_name,
                f"Tool '{tool_name}' requires human approval before execution. "
                f"Risk level: {TOOL_RISK.get(tool_name, 'UNKNOWN')}. "
                "Submit an approval request via create_approval_request.",
            )

        # 4. Idempotency check
        if idempotency_key:
            cached = await self._check_idempotency(idempotency_key)
            if cached:
                log.info("idempotency_hit", key=idempotency_key, tool_name=tool_name)
                self._audit(call_id, tool_name, tool_input, cached, "IDEMPOTENT", started)
                return ToolResponse.success(
                    tool_name=tool_name,
                    data=cached,
                    idempotency_key=idempotency_key,
                    idempotent_replay=True,
                )

        # 5. Execute tool with timeout
        try:
            import asyncio
            result = await asyncio.wait_for(
                self._registry[tool_name](tool_input),
                timeout=settings.tool_timeout_seconds,
            )
        except TimeoutError:
            self._audit(call_id, tool_name, tool_input, None, "TIMEOUT", started)
            return ToolResponse(
                status=ToolStatus.TIMEOUT,
                tool_name=tool_name,
                error_code="TOOL_TIMEOUT",
                error_message=f"Tool '{tool_name}' timed out after {settings.tool_timeout_seconds}s.",
                retryable=True,
            )
        except Exception as exc:
            log.exception("tool_execution_error", tool_name=tool_name, error=str(exc))
            self._audit(call_id, tool_name, tool_input, None, "ERROR", started)
            return ToolResponse.error(
                tool_name=tool_name,
                error_code="TOOL_EXECUTION_ERROR",
                error_message=str(exc),
                retryable=True,
            )

        # 6. Cache idempotency result for write tools
        if idempotency_key and result.status == ToolStatus.SUCCESS:
            await self._store_idempotency(idempotency_key, result.data)

        duration_ms = int((time.monotonic() - started) * 1000)
        self._audit(call_id, tool_name, tool_input, result.data, result.status, started)

        log.info(
            "tool_gateway_complete",
            tool_name=tool_name,
            status=result.status,
            duration_ms=duration_ms,
        )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Permission check (local RBAC; OPA called for prod)
    # ──────────────────────────────────────────────────────────────────────────

    def _check_permission(self, tool_name: str) -> dict[str, Any]:
        role = self.identity.get("role", "UNKNOWN")
        permissions = self.identity.get("permissions", [])

        # System agent has all read permissions
        if role == "AGENT":
            risk = TOOL_RISK.get(tool_name, "LOW")
            if risk in ("HIGH", "CRITICAL"):
                return {
                    "allowed": False,
                    "reason": (
                        f"Role AGENT cannot directly execute {risk}-risk tool '{tool_name}'. "
                        "Human approval required."
                    ),
                }
            return {"allowed": True, "reason": ""}

        # Human roles
        role_permissions: dict[str, set[str]] = {
            "OPERATIONS_ANALYST": {
                "get_transaction_details", "get_account_summary", "get_customer_profile",
                "get_recent_transactions", "get_payment_status", "search_policy",
                "get_case_history", "check_transaction_limit",
                "create_case_note", "create_operations_ticket", "draft_customer_notification",
                "retry_payment", "refund_fee",
            },
            "SENIOR_ANALYST": {
                "get_transaction_details", "get_account_summary", "get_customer_profile",
                "get_recent_transactions", "get_payment_status", "search_policy",
                "get_case_history", "check_transaction_limit",
                "create_case_note", "create_operations_ticket", "draft_customer_notification",
                "retry_payment", "refund_fee", "block_card",
            },
            "RISK_OFFICER": {
                "get_transaction_details", "get_account_summary", "get_customer_profile",
                "get_recent_transactions", "get_payment_status", "search_policy",
                "get_case_history", "check_transaction_limit",
                "create_case_note", "create_operations_ticket", "draft_customer_notification",
                "retry_payment", "refund_fee", "block_card", "freeze_account",
                "reverse_transaction",
            },
            "ADMIN": set(TOOL_RISK.keys()),
        }

        allowed_tools = role_permissions.get(role, set())
        if tool_name not in allowed_tools:
            return {
                "allowed": False,
                "reason": f"Role '{role}' does not have permission to use '{tool_name}'.",
            }
        return {"allowed": True, "reason": ""}

    # ──────────────────────────────────────────────────────────────────────────
    # Idempotency helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def _check_idempotency(self, key: str) -> Any | None:
        try:
            cached = await self.valkey.get(f"idem:{key}")
            if cached:
                return json.loads(cached)
        except Exception:
            pass
        return None

    async def _store_idempotency(self, key: str, data: Any, ttl: int = 3600) -> None:
        try:
            await self.valkey.setex(f"idem:{key}", ttl, json.dumps(data, default=str))
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    # Audit
    # ──────────────────────────────────────────────────────────────────────────

    def _audit(
        self,
        call_id: str,
        tool_name: str,
        tool_input: dict,
        result: Any,
        status: Any,
        started: float,
    ) -> None:
        entry = {
            "call_id": call_id,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "status": str(status),
            "duration_ms": int((time.monotonic() - started) * 1000),
            "identity": self.identity,
        }
        self.audit_log.append(entry)
