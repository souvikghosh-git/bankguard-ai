"""
BankGuard AI — Payment tools (simulated payment rail interactions).
"""

from __future__ import annotations

from typing import Any, Callable, Awaitable

import structlog

from tools.schemas import ToolResponse

log = structlog.get_logger(__name__)


def payment_tools(
    db: Any,
) -> dict[str, Callable[[dict], Awaitable[ToolResponse]]]:
    """Payment rail tools — these interact with the payment system."""

    async def get_payment_rail_status(raw: dict[str, Any]) -> ToolResponse:
        """Check if a payment rail is operational."""
        rail = raw.get("payment_rail", "IMPS").upper()
        # In production this would call NPCI/RBI APIs
        # Simulator: all rails operational in dev
        return ToolResponse.success(
            "get_payment_rail_status",
            {
                "rail": rail,
                "status": "OPERATIONAL",
                "last_checked": "2024-01-01T00:00:00Z",
                "avg_settlement_minutes": {"IMPS": 2, "NEFT": 60, "UPI": 1, "RTGS": 30}.get(rail, 5),
            },
        )

    return {
        "get_payment_rail_status": get_payment_rail_status,
    }
