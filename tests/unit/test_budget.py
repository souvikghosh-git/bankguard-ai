"""
Unit tests for harness/budget.py

All tests are pure-Python, no DB, no network.
"""

from __future__ import annotations

import pytest
from harness.budget import BudgetExceeded, BudgetManager, BudgetSnapshot


# ─────────────────────────────────────────────────────────────────────────────
# BudgetManager — basic accounting
# ─────────────────────────────────────────────────────────────────────────────

def test_initial_snapshot_is_zero():
    bm = BudgetManager(max_iterations=8, max_tool_calls=15, max_cost_usd=0.15)
    s = bm.snapshot()
    assert s.iterations_used == 0
    assert s.tool_calls_used == 0
    assert s.cost_usd == 0.0
    assert not s.is_exhausted


def test_record_llm_call_accumulates():
    bm = BudgetManager(max_cost_usd=1.0)
    cost1 = bm.record_llm_call(10_000, 2_000)
    cost2 = bm.record_llm_call(5_000, 1_000)
    s = bm.snapshot()
    assert s.input_tokens_used == 15_000
    assert s.output_tokens_used == 3_000
    assert s.cost_usd == pytest.approx(cost1 + cost2, rel=1e-5)


def test_cost_budget_exceeded():
    bm = BudgetManager(max_cost_usd=0.0001)  # very tight budget
    with pytest.raises(BudgetExceeded) as exc_info:
        bm.record_llm_call(10_000, 10_000)
    assert exc_info.value.budget_type in (
        "COST_BUDGET", "INPUT_TOKEN_BUDGET", "OUTPUT_TOKEN_BUDGET"
    )


def test_iteration_limit():
    bm = BudgetManager(max_iterations=3)
    bm.increment_iteration()
    bm.increment_iteration()
    bm.increment_iteration()
    with pytest.raises(BudgetExceeded) as exc_info:
        bm.increment_iteration()
    assert exc_info.value.budget_type == "MAX_ITERATIONS"


def test_tool_call_limit():
    bm = BudgetManager(max_tool_calls=2)
    bm.record_tool_call()
    bm.record_tool_call()
    with pytest.raises(BudgetExceeded) as exc_info:
        bm.record_tool_call()
    assert exc_info.value.budget_type == "MAX_TOOL_CALLS"


def test_parallel_slot_limit():
    bm = BudgetManager(max_parallel_tools=2)
    bm.acquire_tool_slot()
    bm.acquire_tool_slot()
    with pytest.raises(BudgetExceeded) as exc_info:
        bm.acquire_tool_slot()
    assert exc_info.value.budget_type == "MAX_PARALLEL_TOOLS"
    bm.release_tool_slot()
    # Now slot is free — should not raise
    bm.acquire_tool_slot()


def test_release_slot_allows_reacquire():
    bm = BudgetManager(max_parallel_tools=1)
    bm.acquire_tool_slot()
    bm.release_tool_slot()
    bm.acquire_tool_slot()  # no exception


def test_snapshot_remaining():
    bm = BudgetManager(max_iterations=8, max_tool_calls=15, max_cost_usd=0.15)
    bm.increment_iteration()
    bm.increment_iteration()
    bm.record_tool_call()
    s = bm.snapshot()
    assert s.iterations_remaining == 6
    assert s.tool_calls_remaining == 14
    assert s.cost_remaining_usd == pytest.approx(0.15, rel=1e-5)
    assert not s.is_exhausted


def test_is_exhausted_when_iterations_hit():
    bm = BudgetManager(max_iterations=1)
    bm.increment_iteration()
    s = bm.snapshot()
    assert s.iterations_remaining == 0
    assert s.is_exhausted


def test_token_budget_input():
    bm = BudgetManager(token_budget_input=1000, token_budget_output=100_000, max_cost_usd=100)
    with pytest.raises(BudgetExceeded) as exc_info:
        bm.record_llm_call(1001, 1)
    assert exc_info.value.budget_type == "INPUT_TOKEN_BUDGET"


def test_to_dict_shape():
    bm = BudgetManager()
    d = bm.to_dict()
    assert "iterations" in d
    assert "tool_calls" in d
    assert "cost_usd" in d
    assert "exhausted" in d
    assert d["exhausted"] is False
