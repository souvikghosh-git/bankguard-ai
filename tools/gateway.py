"""
BankGuard AI — Tool Gateway.

Every tool call must pass through this gateway, which enforces:
  1. Schema validation (Pydantic — in each tool module)
  2. Permission check  (local RBAC + live OPA call)
  3. Approval gate    (CRITICAL/HIGH tools blocked until Temporal approval)
  4. Idempotency      (Valkey — auto-key generated for write tools)
  5. Audit persistence (every call written to agent.tool_calls in DB)
  6. Timeout + retry
  7. Structured error responses

Fixed from audit:
  - OPA is now called at runtime for every tool invocation (not just as comment)
  - Audit log is persisted to agent.tool_calls table in PostgreSQL
  - Write tools get an auto-generated idempotency key if caller omits one
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Awaitable, Callable

import httpx
import structlog
from pydantic import BaseModel

from config import settings
from tools.schemas import ToolResponse, ToolStatus

log = structlog.get_logger(__name__)

# ── Risk / approval tables ────────────────────────────────────────────────────

TOOL_RISK: dict[str, str] = {
    "get_transaction_details":      "LOW",
    "get_account_summary":          "LOW",
    "get_customer_profile":         "LOW",
    "get_recent_transactions":      "LOW",
    "get_payment_status":           "LOW",
    "search_policy":                "LOW",
    "get_case_history":             "LOW",
    "check_transaction_limit":      "LOW",
    "get_payment_rail_status":      "LOW",
    "create_case_note":             "LOW",
    "create_operations_ticket":     "LOW",
    "draft_customer_notification":  "LOW",
    "retry_payment":                "HIGH",
    "refund_fee":                   "MEDIUM",
    "block_card":                   "HIGH",
    "freeze_account":               "CRITICAL",
    "reverse_transaction":          "CRITICAL",
}

# Tools that require a human-approved Temporal workflow before execution
APPROVAL_REQUIRED: set[str] = {
    "retry_payment",
    "block_card",
    "freeze_account",
    "reverse_transaction",
}

# Write tools that mutate state — these always get an idempotency key
WRITE_TOOLS: set[str] = {
    "create_case_note",
    "create_operations_ticket",
    "draft_customer_notification",
    "retry_payment",
    "refund_fee",
    "block_card",
    "freeze_account",
    "reverse_transaction",
}


class ToolGateway:
    """
    Central tool gateway — every agent tool call passes through here.

    Usage:
        gw = ToolGateway(db_pool, valkey, run_id, case_id, identity)
        result = await gw.call("get_transaction_details", {"transaction_ref": "TXN-1"})
    """

    def __init__(
        self,
        db_pool: Any,
        valkey: Any,
        run_id: str,
        case_id: str,
        identity: dict[str, Any],
        audit_log: list[dict] | None = None,
    ) -> None:
        self.db = db_pool
        self.valkey = valkey
        self.run_id = run_id
        self.case_id = case_id
        self.identity = identity
        self.audit_log: list[dict] = audit_log if audit_log is not None else []
        self._call_count = 0

        # Lazy-import tool registries to avoid circular deps at module load
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

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    async def call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> ToolResponse:
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

        # ── 1. Tool must be registered ────────────────────────────────────────
        if tool_name not in self._registry:
            result = ToolResponse.error(
                tool_name=tool_name,
                error_code="UNKNOWN_TOOL",
                error_message=f"Tool '{tool_name}' is not registered.",
                retryable=False,
            )
            await self._persist_audit(call_id, tool_name, tool_input, result, started)
            return result

        # ── 2. Local RBAC ─────────────────────────────────────────────────────
        perm = self._check_local_permission(tool_name)
        if not perm["allowed"]:
            result = ToolResponse.blocked(tool_name, perm["reason"])
            await self._persist_audit(call_id, tool_name, tool_input, result, started)
            return result

        # ── 3. OPA (live HTTP call) ───────────────────────────────────────────
        opa_ok, opa_reason = await self._check_opa(tool_name, tool_input)
        if not opa_ok:
            result = ToolResponse.blocked(tool_name, f"OPA denied: {opa_reason}")
            await self._persist_audit(call_id, tool_name, tool_input, result, started)
            return result

        # ── 4. Approval gate for HIGH / CRITICAL tools ────────────────────────
        if tool_name in APPROVAL_REQUIRED:
            result = ToolResponse.blocked(
                tool_name,
                f"Tool '{tool_name}' (risk={TOOL_RISK[tool_name]}) requires human "
                "approval via Temporal workflow before execution. "
                "The caller should invoke ApprovalService.request_approval() and "
                "await the Temporal signal before retrying.",
            )
            await self._persist_audit(call_id, tool_name, tool_input, result, started)
            return result

        # ── 5. Idempotency — auto-generate key for write tools ────────────────
        if idempotency_key is None and tool_name in WRITE_TOOLS:
            # Deterministic key: sha256(run_id + tool_name + sorted_input)
            idempotency_key = self._derive_idempotency_key(tool_name, tool_input)

        if idempotency_key:
            cached = await self._check_idempotency(idempotency_key)
            if cached is not None:
                log.info("idempotency_hit", key=idempotency_key, tool_name=tool_name)
                result = ToolResponse.success(
                    tool_name=tool_name,
                    data=cached,
                    idempotency_key=idempotency_key,
                    idempotent_replay=True,
                )
                await self._persist_audit(call_id, tool_name, tool_input, result, started)
                return result

        # ── 6. Execute with timeout ───────────────────────────────────────────
        import asyncio
        try:
            result = await asyncio.wait_for(
                self._registry[tool_name](tool_input),
                timeout=settings.tool_timeout_seconds,
            )
        except TimeoutError:
            result = ToolResponse(
                status=ToolStatus.TIMEOUT,
                tool_name=tool_name,
                error_code="TOOL_TIMEOUT",
                error_message=f"Tool timed out after {settings.tool_timeout_seconds}s.",
                retryable=True,
            )
        except Exception as exc:
            log.exception("tool_execution_error", tool_name=tool_name, error=str(exc))
            result = ToolResponse.error(
                tool_name=tool_name,
                error_code="TOOL_EXECUTION_ERROR",
                error_message=str(exc),
                retryable=True,
            )

        # ── 7. Cache successful write results ─────────────────────────────────
        if idempotency_key and result.status == ToolStatus.SUCCESS:
            await self._store_idempotency(idempotency_key, result.data)

        # ── 8. Persist audit to DB ────────────────────────────────────────────
        await self._persist_audit(call_id, tool_name, tool_input, result, started)

        log.info(
            "tool_gateway_complete",
            tool_name=tool_name,
            status=result.status,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Local RBAC
    # ─────────────────────────────────────────────────────────────────────────

    def _check_local_permission(self, tool_name: str) -> dict[str, Any]:
        role = self.identity.get("role", "UNKNOWN")

        if role == "AGENT":
            risk = TOOL_RISK.get(tool_name, "LOW")
            if risk in ("HIGH", "CRITICAL"):
                return {
                    "allowed": False,
                    "reason": (
                        f"AGENT role cannot directly execute {risk}-risk tool "
                        f"'{tool_name}'. Human approval required."
                    ),
                }
            return {"allowed": True, "reason": ""}

        role_permissions: dict[str, set[str]] = {
            "OPERATIONS_ANALYST": {
                "get_transaction_details", "get_account_summary", "get_customer_profile",
                "get_recent_transactions", "get_payment_status", "search_policy",
                "get_case_history", "check_transaction_limit", "get_payment_rail_status",
                "create_case_note", "create_operations_ticket", "draft_customer_notification",
                "retry_payment", "refund_fee",
            },
            "SENIOR_ANALYST": {
                "get_transaction_details", "get_account_summary", "get_customer_profile",
                "get_recent_transactions", "get_payment_status", "search_policy",
                "get_case_history", "check_transaction_limit", "get_payment_rail_status",
                "create_case_note", "create_operations_ticket", "draft_customer_notification",
                "retry_payment", "refund_fee", "block_card",
            },
            "RISK_OFFICER": {
                "get_transaction_details", "get_account_summary", "get_customer_profile",
                "get_recent_transactions", "get_payment_status", "search_policy",
                "get_case_history", "check_transaction_limit", "get_payment_rail_status",
                "create_case_note", "create_operations_ticket", "draft_customer_notification",
                "retry_payment", "refund_fee", "block_card",
                "freeze_account", "reverse_transaction",
            },
            "ADMIN": set(TOOL_RISK.keys()),
        }

        allowed = role_permissions.get(role, set())
        if tool_name not in allowed:
            return {
                "allowed": False,
                "reason": f"Role '{role}' cannot use tool '{tool_name}'.",
            }
        return {"allowed": True, "reason": ""}

    # ─────────────────────────────────────────────────────────────────────────
    # OPA live call
    # ─────────────────────────────────────────────────────────────────────────

    async def _check_opa(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Call OPA synchronously.  Returns (allowed: bool, reason: str).
        Falls back to ALLOW on connection error (OPA is a second layer;
        local RBAC is the first).  Logs the fallback so it is visible.
        """
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.post(
                    f"{settings.opa_url}/v1/data/bankguard/tool_access",
                    json={
                        "input": {
                            "tool": tool_name,
                            "role": self.identity.get("role"),
                            "user_id": self.identity.get("user_id"),
                            "context": {
                                "fee_amount": tool_input.get("fee_amount", 0),
                                "amount": tool_input.get("amount", 0),
                            },
                        }
                    },
                )
            data = resp.json()
            result = data.get("result", {})
            # OPA Rego returns `allow` or `final_allow`
            allowed = result.get("final_allow", result.get("allow", True))
            deny_reasons = result.get("deny_reason", {})
            reason = ", ".join(deny_reasons.keys()) if deny_reasons else ""
            return bool(allowed), reason
        except httpx.ConnectError:
            log.warning(
                "opa_unreachable_fallback_allow",
                tool=tool_name,
                opa_url=settings.opa_url,
            )
            return True, ""
        except Exception as exc:
            log.warning("opa_check_error_fallback_allow", error=str(exc))
            return True, ""

    # ─────────────────────────────────────────────────────────────────────────
    # Idempotency (Valkey)
    # ─────────────────────────────────────────────────────────────────────────

    def _derive_idempotency_key(self, tool_name: str, tool_input: dict) -> str:
        """
        Deterministic idempotency key scoped to (run_id, tool_name, sorted input).
        Same run + same tool + same inputs always maps to the same key.
        """
        payload = json.dumps(
            {"run": self.run_id, "tool": tool_name, "input": tool_input},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    async def _check_idempotency(self, key: str) -> Any | None:
        try:
            raw = await self.valkey.get(f"idem:{key}")
            return json.loads(raw) if raw else None
        except Exception:
            return None

    async def _store_idempotency(self, key: str, data: Any, ttl: int = 3600) -> None:
        try:
            await self.valkey.setex(
                f"idem:{key}", ttl, json.dumps(data, default=str)
            )
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Audit persistence (PostgreSQL + in-memory list)
    # ─────────────────────────────────────────────────────────────────────────

    async def _persist_audit(
        self,
        call_id: str,
        tool_name: str,
        tool_input: dict,
        result: ToolResponse,
        started: float,
    ) -> None:
        duration_ms = int((time.monotonic() - started) * 1000)
        entry = {
            "call_id": call_id,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "status": str(result.status),
            "error_code": result.error_code,
            "duration_ms": duration_ms,
            "identity_role": self.identity.get("role"),
        }
        # Always append to in-memory log (returned in RunResult.audit_log)
        self.audit_log.append(entry)

        # Persist to DB (best-effort — never fail the tool call on audit error)
        if self.db is None:
            return
        try:
            # Resolve agent_run UUID from run_ref string
            async with self.db.acquire() as conn:
                run_row = await conn.fetchrow(
                    "SELECT id FROM agent.agent_runs WHERE run_ref = $1",
                    self.run_id,
                )
                run_uuid = run_row["id"] if run_row else None

                await conn.execute(
                    """
                    INSERT INTO agent.tool_calls
                        (id, run_id, tool_name, tool_input, tool_output,
                         status, error_code, duration_ms, attempt_number, called_at)
                    VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7, $8, 1, NOW())
                    ON CONFLICT DO NOTHING
                    """,
                    call_id,
                    run_uuid,
                    tool_name,
                    json.dumps(tool_input, default=str),
                    json.dumps(
                        result.data if result.data is not None else result.error_message,
                        default=str,
                    ),
                    str(result.status),
                    result.error_code,
                    duration_ms,
                )
        except Exception as exc:
            # Non-fatal: log and continue
            log.warning("audit_persist_failed", error=str(exc), tool=tool_name)
