"""
BankGuard AI — Token Budget Manager for context assembly.

Allocates tokens to context sections in priority order.
Higher-priority sections are always included; lower-priority sections
are skipped when the budget is exhausted.

Uses a simple character ÷ 4 approximation (good enough for planning).
For production, swap in tiktoken or the model's own tokeniser.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)


class TokenBudgetManager:
    """
    Manages context token allocation across named sections.

    Usage:
        mgr = TokenBudgetManager(total_budget=8000)
        included = mgr.allocate("transaction", json_str, priority=9)  # bool
        report   = mgr.usage_report()
    """

    def __init__(self, total_budget: int = 8_000) -> None:
        self.total_budget = total_budget
        self.total_used = 0
        self.section_tokens: dict[str, int] = {}
        self._sections: list[tuple[str, str, int]] = []  # (name, content, priority)

    def reset(self) -> None:
        self.total_used = 0
        self.section_tokens = {}
        self._sections = []

    def allocate(self, name: str, content: str, priority: int = 5) -> str:
        """
        Try to include `content` within the remaining token budget.
        Returns the (possibly truncated) content that was actually included,
        or an empty string if the section couldn't fit at all.
        """
        est_tokens = self._estimate_tokens(content)

        if self.total_used + est_tokens <= self.total_budget:
            self.total_used += est_tokens
            self.section_tokens[name] = est_tokens
            return content

        # Try truncation: allocate remaining budget to this section
        remaining = self.total_budget - self.total_used
        if remaining > 100:  # worth including a partial section
            chars = remaining * 4
            truncated = content[:chars] + "...[truncated]"
            actual_tokens = self._estimate_tokens(truncated)
            self.total_used += actual_tokens
            self.section_tokens[name] = actual_tokens
            log.debug(
                "context_section_truncated",
                section=name,
                original_tokens=est_tokens,
                allocated_tokens=actual_tokens,
                priority=priority,
            )
            return truncated

        log.debug(
            "context_section_skipped",
            section=name,
            needed_tokens=est_tokens,
            remaining=self.total_budget - self.total_used,
            priority=priority,
        )
        self.section_tokens[name] = 0
        return ""

    def remaining(self) -> int:
        return max(0, self.total_budget - self.total_used)

    def usage_report(self) -> dict:
        return {
            "total_budget": self.total_budget,
            "total_used": self.total_used,
            "utilisation_pct": round(self.total_used / self.total_budget * 100, 1),
            "sections": self.section_tokens,
        }

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Approximate token count: ~4 chars per token."""
        return max(1, len(text) // 4)
