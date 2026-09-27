"""
Unit tests for tools/gateway.py — permission and approval gate logic.

Uses a mock DB and Valkey so no real services are needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.gateway import APPROVAL_REQUIRED, TOOL_RISK, ToolGateway
from tools.schemas import ToolStatus

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_gateway(role: str = "AGENT", extra: dict | None = None) -> ToolGateway:
    """Return a ToolGateway with mocked DB and Valkey."""
    # Mock connection — acquire() returns an async context manager
    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.execute = AsyncMock()
    # Make the conn itself usable as an async context manager
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    # acquire() must return an async context manager that yields mock_conn
    mock_acquire_cm = AsyncMock()
    mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire_cm.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock(return_value=mock_acquire_cm)

    # Mock Valkey
    mock_valkey = AsyncMock()
    mock_valkey.get = AsyncMock(return_value=None)
    mock_valkey.setex = AsyncMock()

    identity = {"role": role, "user_id": "test-user", **(extra or {})}
    gw = ToolGateway(
        db_pool=mock_pool,
        valkey=mock_valkey,
        run_id="RUN-TEST001",
        case_id="case-uuid-001",
        identity=identity,
    )
    return gw


# ─────────────────────────────────────────────────────────────────────────────
# Local RBAC
# ─────────────────────────────────────────────────────────────────────────────


def test_agent_can_call_read_tools():
    gw = _make_gateway(role="AGENT")
    result = gw._check_local_permission("get_transaction_details")
    assert result["allowed"] is True


def test_agent_blocked_from_high_risk():
    gw = _make_gateway(role="AGENT")
    for tool in ("retry_payment", "block_card", "freeze_account", "reverse_transaction"):
        result = gw._check_local_permission(tool)
        assert result["allowed"] is False, f"AGENT should not be allowed: {tool}"


def test_operations_analyst_can_call_write_tools():
    gw = _make_gateway(role="OPERATIONS_ANALYST")
    for tool in ("create_case_note", "create_operations_ticket", "draft_customer_notification"):
        result = gw._check_local_permission(tool)
        assert result["allowed"] is True, f"Expected OPERATIONS_ANALYST to access: {tool}"


def test_operations_analyst_blocked_from_freeze():
    gw = _make_gateway(role="OPERATIONS_ANALYST")
    result = gw._check_local_permission("freeze_account")
    assert result["allowed"] is False


def test_risk_officer_can_see_all_tools():
    gw = _make_gateway(role="RISK_OFFICER")
    for tool in TOOL_RISK:
        result = gw._check_local_permission(tool)
        assert result["allowed"] is True, f"RISK_OFFICER should access: {tool}"


def test_unknown_role_blocked():
    gw = _make_gateway(role="INTERN")
    result = gw._check_local_permission("get_transaction_details")
    assert result["allowed"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Approval gate
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_approval_gate_blocks_retry_payment():
    gw = _make_gateway(role="RISK_OFFICER")  # role permits, but gate still blocks
    with patch.object(gw, "_check_opa", AsyncMock(return_value=(True, ""))):
        result = await gw.call("retry_payment", {"transaction_ref": "TXN-001", "reason": "test"})
    assert result.status == ToolStatus.BLOCKED
    assert "approval" in result.error_message.lower()


@pytest.mark.asyncio
async def test_approval_gate_blocks_freeze_account():
    gw = _make_gateway(role="ADMIN")
    with patch.object(gw, "_check_opa", AsyncMock(return_value=(True, ""))):
        result = await gw.call("freeze_account", {"account_ref": "ACC-001"})
    assert result.status == ToolStatus.BLOCKED


@pytest.mark.asyncio
async def test_approval_gate_blocks_reverse_transaction():
    gw = _make_gateway(role="ADMIN")
    with patch.object(gw, "_check_opa", AsyncMock(return_value=(True, ""))):
        result = await gw.call("reverse_transaction", {"transaction_ref": "TXN-001"})
    assert result.status == ToolStatus.BLOCKED


def test_all_critical_tools_in_approval_required():
    """Every CRITICAL-risk tool must be in APPROVAL_REQUIRED."""
    critical_tools = {t for t, r in TOOL_RISK.items() if r == "CRITICAL"}
    for tool in critical_tools:
        assert tool in APPROVAL_REQUIRED, f"CRITICAL tool not in APPROVAL_REQUIRED: {tool}"


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency key derivation
# ─────────────────────────────────────────────────────────────────────────────


def test_idempotency_key_is_deterministic():
    gw = _make_gateway()
    key1 = gw._derive_idempotency_key("create_case_note", {"case_ref": "CASE-001", "content": "hi"})
    key2 = gw._derive_idempotency_key("create_case_note", {"content": "hi", "case_ref": "CASE-001"})
    assert key1 == key2  # order-independent due to sort_keys=True


def test_idempotency_key_differs_for_different_inputs():
    gw = _make_gateway()
    key1 = gw._derive_idempotency_key("create_case_note", {"case_ref": "CASE-001", "content": "note A"})
    key2 = gw._derive_idempotency_key("create_case_note", {"case_ref": "CASE-001", "content": "note B"})
    assert key1 != key2


def test_idempotency_key_differs_for_different_tools():
    gw = _make_gateway()
    key1 = gw._derive_idempotency_key("create_case_note", {"case_ref": "X"})
    key2 = gw._derive_idempotency_key("create_operations_ticket", {"case_ref": "X"})
    assert key1 != key2


# ─────────────────────────────────────────────────────────────────────────────
# OPA fallback behaviour
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_opa_connection_error_falls_back_to_allow():
    """If OPA is unreachable the gateway should ALLOW (local RBAC is the primary gate)."""
    import httpx

    gw = _make_gateway(role="AGENT")
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client
        allowed, reason = await gw._check_opa("get_transaction_details", {})
    assert allowed is True


@pytest.mark.asyncio
async def test_opa_explicit_deny_blocks_tool():
    gw = _make_gateway(role="OPERATIONS_ANALYST")
    # OPA returns deny
    with patch.object(gw, "_check_opa", AsyncMock(return_value=(False, "policy_violation"))):
        result = await gw.call("create_case_note", {"case_ref": "CASE-001", "content": "x" * 15})
    assert result.status == ToolStatus.BLOCKED
    assert "OPA denied" in result.error_message
