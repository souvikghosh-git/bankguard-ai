"""
Unit tests for agents/supervisor/supervisor.py

Tests the routing logic (_should_stop, _route_after_review)
and the NO_NEW_EVIDENCE delta check without running LangGraph.
"""

from __future__ import annotations

import pytest
from langgraph.graph import END

from agents.state import LoopState, StopReason
from agents.supervisor.supervisor import (
    _should_stop,
    _route_after_review,
    supervisor_evaluate_node,
    supervisor_plan_node,
    MAX_ITERATIONS,
    CONFIDENCE_FLOOR,
    MIN_EVIDENCE_ITEMS,
)


def _state(**overrides) -> dict:
    """Build a minimal AgentState dict for routing tests."""
    base = {
        "run_id": "RUN-TEST",
        "case_ref": "CASE-TEST",
        "iteration": 1,
        "confidence": 0.0,
        "evidence": [],
        "evidence_count_at_last_cycle": 0,
        "stop_reason": None,
        "errors": [],
        "loop_state": LoopState.EVALUATE,
        "hypotheses": [],
        "reviewer_approved": False,
        "reviewer_concerns": [],
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# _should_stop routing
# ─────────────────────────────────────────────────────────────────────────────

def test_routes_to_reviewer_when_confidence_high_and_enough_evidence():
    state = _state(
        confidence=CONFIDENCE_FLOOR,
        evidence=[{"s": "e"}] * MIN_EVIDENCE_ITEMS,
    )
    assert _should_stop(state) == "reviewer"


def test_routes_to_transaction_agent_when_more_evidence_needed():
    state = _state(confidence=0.4, evidence=[{"s": "e"}])
    assert _should_stop(state) == "transaction_agent"


def test_routes_to_reviewer_at_max_iterations():
    state = _state(iteration=MAX_ITERATIONS, confidence=0.0)
    assert _should_stop(state) == "reviewer"


def test_routes_to_escalate_on_human_escalation():
    state = _state(stop_reason=StopReason.HUMAN_ESCALATION)
    assert _should_stop(state) == "escalate"


def test_routes_to_escalate_on_budget_exceeded():
    state = _state(stop_reason=StopReason.BUDGET_EXCEEDED)
    assert _should_stop(state) == "escalate"


def test_routes_to_escalate_on_too_many_errors():
    state = _state(errors=["e1", "e2", "e3"])
    assert _should_stop(state) == "escalate"


def test_routes_to_reviewer_on_case_resolved():
    state = _state(stop_reason=StopReason.CASE_RESOLVED)
    assert _should_stop(state) == "reviewer"


def test_routes_to_reviewer_on_confidence_reached():
    state = _state(stop_reason=StopReason.CONFIDENCE_REACHED)
    assert _should_stop(state) == "reviewer"


def test_does_not_route_to_transaction_agent_via_loop_state():
    """
    Critical regression: previously checked loop_state == ACT/OBSERVE
    to decide loop-back, but loop_state is always EVALUATE at this point.
    Now we always route to transaction_agent when more evidence is needed.
    """
    state = _state(
        confidence=0.3,
        evidence=[],
        loop_state=LoopState.EVALUATE,  # always EVALUATE at this routing point
    )
    # Should still loop back even though loop_state is EVALUATE
    assert _should_stop(state) == "transaction_agent"


# ─────────────────────────────────────────────────────────────────────────────
# _route_after_review
# ─────────────────────────────────────────────────────────────────────────────

def test_ends_when_reviewer_approved():
    state = _state(reviewer_approved=True)
    assert _route_after_review(state) == END


def test_escalates_when_unsafe_concern():
    state = _state(reviewer_concerns=["unsafe action detected"])
    assert _route_after_review(state) == "escalate"


def test_escalates_when_escalation_keyword_in_concern():
    state = _state(reviewer_concerns=["requires escalation to compliance"])
    assert _route_after_review(state) == "escalate"


def test_loops_back_when_reviewer_rejected_and_budget_remains():
    state = _state(
        reviewer_approved=False,
        reviewer_concerns=["insufficient evidence"],
        iteration=3,
    )
    assert _route_after_review(state) == "transaction_agent"


def test_ends_when_reviewer_rejected_and_max_iterations():
    state = _state(
        reviewer_approved=False,
        reviewer_concerns=["insufficient evidence"],
        iteration=MAX_ITERATIONS - 1,
    )
    assert _route_after_review(state) == END


# ─────────────────────────────────────────────────────────────────────────────
# supervisor_evaluate_node — NO_NEW_EVIDENCE delta fix
# ─────────────────────────────────────────────────────────────────────────────

def test_no_new_evidence_fires_on_zero_delta():
    """
    Delta = len(evidence) - evidence_count_at_last_cycle
    If delta == 0 AND iteration >= 2, stop reason should be set.
    """
    evidence = [{"source": "tool_a", "content": "x"}] * 5
    state = _state(
        iteration=3,
        evidence=evidence,
        evidence_count_at_last_cycle=5,  # same as current — delta = 0
        hypotheses=[],
    )
    result = supervisor_evaluate_node(state)
    assert result.get("stop_reason") == StopReason.NO_NEW_EVIDENCE


def test_no_new_evidence_does_not_fire_when_delta_positive():
    """Evidence grew this cycle — should NOT set NO_NEW_EVIDENCE."""
    evidence = [{"source": "tool_a", "content": "x"}] * 5
    state = _state(
        iteration=3,
        evidence=evidence,
        evidence_count_at_last_cycle=3,  # delta = 2 → evidence was added
        hypotheses=[],
    )
    result = supervisor_evaluate_node(state)
    assert result.get("stop_reason") != StopReason.NO_NEW_EVIDENCE


def test_no_new_evidence_does_not_fire_on_first_iteration():
    """Must not fire on iteration < 2 (first cycle might genuinely have no evidence yet)."""
    state = _state(
        iteration=1,
        evidence=[],
        evidence_count_at_last_cycle=0,  # delta = 0 but iteration < 2
        hypotheses=[],
    )
    result = supervisor_evaluate_node(state)
    assert result.get("stop_reason") != StopReason.NO_NEW_EVIDENCE


def test_evaluate_updates_evidence_count_snapshot():
    """supervisor_evaluate_node must update evidence_count_at_last_cycle for next cycle."""
    evidence = [{"source": "t", "content": "x"}] * 4
    state = _state(
        iteration=2,
        evidence=evidence,
        evidence_count_at_last_cycle=2,
        hypotheses=[],
    )
    result = supervisor_evaluate_node(state)
    assert result["evidence_count_at_last_cycle"] == 4


def test_evaluate_extracts_best_hypothesis():
    hypotheses = [
        {"root_cause": "WEAK",   "confidence": 0.40},
        {"root_cause": "STRONG", "confidence": 0.87},
        {"root_cause": "MID",    "confidence": 0.65},
    ]
    state = _state(
        iteration=2,
        evidence=[{"s": "e"}] * 3,
        evidence_count_at_last_cycle=1,
        hypotheses=hypotheses,
    )
    result = supervisor_evaluate_node(state)
    assert result["root_cause"] == "STRONG"
    assert result["confidence"] == pytest.approx(0.87)


# ─────────────────────────────────────────────────────────────────────────────
# supervisor_plan_node
# ─────────────────────────────────────────────────────────────────────────────

def test_plan_node_increments_iteration():
    state = _state(iteration=2)
    result = supervisor_plan_node(state)
    assert result["iteration"] == 3


def test_plan_node_sets_evidence_count_snapshot():
    evidence = [{"s": "e"}] * 3
    state = _state(iteration=0, evidence=evidence)
    result = supervisor_plan_node(state)
    assert result["evidence_count_at_last_cycle"] == 3


def test_plan_node_returns_non_empty_plan():
    state = _state(iteration=0)
    result = supervisor_plan_node(state)
    assert isinstance(result["investigation_plan"], list)
    assert len(result["investigation_plan"]) >= 5
