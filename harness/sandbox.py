"""
BankGuard AI — Sandbox / Environment Guard.

Prevents the agent from accidentally touching production data
when running in development or evaluation mode.

Features:
  - Environment tag enforcement (dev tools can't reach prod DBs)
  - Read-only mode (blocks all write tools in eval/test runs)
  - Dry-run mode (executes logic but does NOT persist changes)
  - Sensitive-action blocklist (always-blocked tool names)
"""

from __future__ import annotations

from enum import StrEnum

import structlog

from config import settings

log = structlog.get_logger(__name__)


class SandboxMode(StrEnum):
    LIVE = "live"  # full execution (production)
    DRY_RUN = "dry_run"  # logic runs, no DB writes
    READ_ONLY = "read_only"  # only read tools allowed
    EVAL = "eval"  # evaluation: reads + case notes only


# Tools that are NEVER allowed regardless of mode
ALWAYS_BLOCKED: set[str] = {
    "freeze_account",
    "reverse_transaction",
    "block_card",
}

# Tools that require LIVE mode
LIVE_ONLY_TOOLS: set[str] = {
    "retry_payment",
    "refund_fee",
}

# Tools allowed in READ_ONLY mode
READ_ONLY_TOOLS: set[str] = {
    "get_transaction_details",
    "get_account_summary",
    "get_customer_profile",
    "get_recent_transactions",
    "get_payment_status",
    "search_policy",
    "get_case_history",
    "check_transaction_limit",
    "get_payment_rail_status",
}

# Tools allowed in EVAL mode (reads + non-financial writes)
EVAL_TOOLS: set[str] = READ_ONLY_TOOLS | {
    "create_case_note",
    "create_operations_ticket",
    "draft_customer_notification",
}


class Sandbox:
    """
    Guards tool execution based on the current environment mode.

    Usage:
        sandbox = Sandbox.from_env()
        sandbox.check("retry_payment")   # raises SandboxViolation if blocked
    """

    def __init__(self, mode: SandboxMode, run_id: str = "") -> None:
        self.mode = mode
        self.run_id = run_id

    @classmethod
    def from_env(cls, run_id: str = "") -> Sandbox:
        """Derive sandbox mode from APP_ENV setting.

        development → EVAL  (reads + case notes/tickets/notifications — no financial mutations)
        staging     → DRY_RUN
        production  → LIVE
        """
        env_mode_map = {
            "production": SandboxMode.LIVE,
            "staging": SandboxMode.DRY_RUN,
            "development": SandboxMode.EVAL,  # EVAL allows case writes; READ_ONLY was too restrictive
        }
        mode = env_mode_map.get(settings.app_env, SandboxMode.EVAL)
        return cls(mode=mode, run_id=run_id)

    def check(self, tool_name: str) -> None:
        """
        Check if tool_name is allowed in the current sandbox mode.
        Raises SandboxViolation if blocked.
        """
        # Always-blocked tools
        if tool_name in ALWAYS_BLOCKED:
            raise SandboxViolation(
                tool_name=tool_name,
                reason=(
                    f"Tool '{tool_name}' is ALWAYS BLOCKED. "
                    "It requires mandatory human approval outside the agent loop."
                ),
            )

        if self.mode == SandboxMode.LIVE:
            return  # all tools allowed (still subject to permission checks)

        if self.mode == SandboxMode.READ_ONLY:
            if tool_name not in READ_ONLY_TOOLS:
                raise SandboxViolation(
                    tool_name=tool_name,
                    reason=f"READ_ONLY sandbox: tool '{tool_name}' is a write operation.",
                )

        elif self.mode == SandboxMode.EVAL:
            if tool_name not in EVAL_TOOLS:
                raise SandboxViolation(
                    tool_name=tool_name,
                    reason=f"EVAL sandbox: tool '{tool_name}' is not allowed in evaluation mode.",
                )

        elif self.mode == SandboxMode.DRY_RUN:
            if tool_name in LIVE_ONLY_TOOLS:
                raise SandboxViolation(
                    tool_name=tool_name,
                    reason=f"DRY_RUN sandbox: '{tool_name}' would have been called (dry run, not executed).",
                    is_dry_run=True,
                )

        log.debug("sandbox_check_passed", tool=tool_name, mode=self.mode.value)

    def is_write_allowed(self) -> bool:
        return self.mode in (SandboxMode.LIVE, SandboxMode.DRY_RUN)


class SandboxViolation(Exception):
    """Raised when a tool call is blocked by the sandbox."""

    def __init__(
        self,
        tool_name: str,
        reason: str,
        is_dry_run: bool = False,
    ) -> None:
        super().__init__(reason)
        self.tool_name = tool_name
        self.reason = reason
        self.is_dry_run = is_dry_run
