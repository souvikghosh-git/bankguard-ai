"""
BankGuard AI — RBAC / ABAC Permission Engine.

Implements role-based access control for all agent and human actions.
Optionally delegates to OPA for policy-as-code evaluation.

Roles:
  AGENT            → autonomous agent identity (read + low-risk writes)
  OPERATIONS_ANALYST → human ops staff
  SENIOR_ANALYST   → senior ops
  RISK_OFFICER     → can approve HIGH-risk actions
  COMPLIANCE       → AML / compliance actions
  ADMIN            → all permissions

Permission check flow:
  1. Local role table (fast, no network)
  2. OPA (if configured) for complex ABAC rules
  3. Hard-coded overrides for ALWAYS_BLOCKED tools
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from config import settings

log = structlog.get_logger(__name__)


# ── Permission definitions ────────────────────────────────────────────────────


@dataclass
class PermissionDecision:
    allowed: bool
    reason: str
    requires_approval: bool = False
    approval_level: str | None = None


# Role → allowed tool set
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "AGENT": {
        # Read tools — always allowed
        "get_transaction_details",
        "get_account_summary",
        "get_customer_profile",
        "get_recent_transactions",
        "get_payment_status",
        "search_policy",
        "get_case_history",
        "check_transaction_limit",
        "get_payment_rail_status",
        # Low-risk writes
        "create_case_note",
        "create_operations_ticket",
        "draft_customer_notification",
    },
    "OPERATIONS_ANALYST": {
        "get_transaction_details",
        "get_account_summary",
        "get_customer_profile",
        "get_recent_transactions",
        "get_payment_status",
        "search_policy",
        "get_case_history",
        "check_transaction_limit",
        "get_payment_rail_status",
        "create_case_note",
        "create_operations_ticket",
        "draft_customer_notification",
        "retry_payment",
        "refund_fee",
    },
    "SENIOR_ANALYST": {
        "get_transaction_details",
        "get_account_summary",
        "get_customer_profile",
        "get_recent_transactions",
        "get_payment_status",
        "search_policy",
        "get_case_history",
        "check_transaction_limit",
        "get_payment_rail_status",
        "create_case_note",
        "create_operations_ticket",
        "draft_customer_notification",
        "retry_payment",
        "refund_fee",
        "block_card",
    },
    "RISK_OFFICER": {
        "get_transaction_details",
        "get_account_summary",
        "get_customer_profile",
        "get_recent_transactions",
        "get_payment_status",
        "search_policy",
        "get_case_history",
        "check_transaction_limit",
        "get_payment_rail_status",
        "create_case_note",
        "create_operations_ticket",
        "draft_customer_notification",
        "retry_payment",
        "refund_fee",
        "block_card",
        "freeze_account",
        "reverse_transaction",
    },
    "COMPLIANCE": {
        "get_transaction_details",
        "get_account_summary",
        "get_customer_profile",
        "get_recent_transactions",
        "get_payment_status",
        "search_policy",
        "get_case_history",
        "check_transaction_limit",
        "create_case_note",
        "create_operations_ticket",
        "freeze_account",
    },
    "ADMIN": set(),  # populated below
}

# ADMIN gets all tools
ALL_TOOLS = set().union(*ROLE_PERMISSIONS.values()) | {
    "freeze_account",
    "reverse_transaction",
    "block_card",
    "retry_payment",
    "refund_fee",
}
ROLE_PERMISSIONS["ADMIN"] = ALL_TOOLS

# Tools that ALWAYS require explicit human approval (regardless of role)
ALWAYS_REQUIRES_APPROVAL: set[str] = {
    "freeze_account",
    "reverse_transaction",
    "block_card",
    "retry_payment",
}

# Tools requiring approval above a monetary threshold
THRESHOLD_APPROVAL: dict[str, dict] = {
    "refund_fee": {"threshold_inr": 500, "required_role": "SENIOR_ANALYST"},
}


class PermissionEngine:
    """
    Centralised permission engine.

    Usage:
        engine = PermissionEngine()
        decision = await engine.check(
            tool_name="retry_payment",
            identity={"role": "OPERATIONS_ANALYST", "user_id": "u-001"},
            context={"amount": 25000}
        )
    """

    def __init__(self, opa_url: str | None = None) -> None:
        self.opa_url = opa_url or settings.opa_url

    async def check(
        self,
        tool_name: str,
        identity: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> PermissionDecision:
        """
        Check whether the identity can execute tool_name.
        Returns a PermissionDecision.
        """
        role = identity.get("role", "UNKNOWN")

        # 1. Role permission check
        allowed_tools = ROLE_PERMISSIONS.get(role, set())
        if tool_name not in allowed_tools:
            return PermissionDecision(
                allowed=False,
                reason=f"Role '{role}' does not have permission to use '{tool_name}'.",
            )

        # 2. Always-requires-approval check
        if tool_name in ALWAYS_REQUIRES_APPROVAL:
            return PermissionDecision(
                allowed=True,
                reason=f"'{tool_name}' is allowed for role '{role}' but requires explicit approval.",
                requires_approval=True,
                approval_level="RISK_OFFICER",
            )

        # 3. Threshold-based approval
        if tool_name in THRESHOLD_APPROVAL and context:
            rule = THRESHOLD_APPROVAL[tool_name]
            amount = context.get("amount", 0) or context.get("fee_amount", 0)
            if amount > rule["threshold_inr"]:
                return PermissionDecision(
                    allowed=True,
                    reason=(
                        f"Amount ₹{amount:,.0f} exceeds threshold ₹{rule['threshold_inr']:,.0f}. "
                        f"Requires {rule['required_role']} approval."
                    ),
                    requires_approval=True,
                    approval_level=rule["required_role"],
                )

        # 4. Optional OPA check for complex ABAC
        if self.opa_url:
            try:
                opa_result = await self._check_opa(tool_name, identity, context or {})
                if not opa_result["allow"]:
                    return PermissionDecision(
                        allowed=False,
                        reason=f"OPA policy denied: {opa_result.get('reason', 'policy violation')}",
                    )
            except Exception as exc:
                log.warning("opa_check_failed_falling_back", error=str(exc))

        return PermissionDecision(allowed=True, reason="Permitted.")

    async def _check_opa(
        self,
        tool_name: str,
        identity: dict,
        context: dict,
    ) -> dict[str, Any]:
        """Call OPA policy server for ABAC decision."""
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.post(
                f"{self.opa_url}/v1/data/bankguard/tool_access",
                json={
                    "input": {
                        "tool": tool_name,
                        "role": identity.get("role"),
                        "user_id": identity.get("user_id"),
                        "context": context,
                    }
                },
            )
            data = resp.json()
            return data.get("result", {"allow": True})


# ── Singleton ─────────────────────────────────────────────────────────────────
permission_engine = PermissionEngine()
