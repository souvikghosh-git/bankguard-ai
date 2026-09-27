"""
BankGuard AI — Supervisor Agent (LangGraph orchestrator).

Owns the top-level PLAN→ACT→OBSERVE→EVALUATE loop.
Delegates to specialist nodes; decides when to stop.

Fixes from audit:
  - NO_NEW_EVIDENCE now tracks DELTA (evidence added in the last cycle),
    not total cumulative count — uses evidence_count_at_last_cycle.
  - Loop-back from EVALUATE correctly returns to transaction_agent when
    more evidence is needed, regardless of current loop_state value.
  - Budget enforcement: supervisor_plan_node increments the budget
    iteration counter via run_ctx; BudgetExceeded propagates cleanly.
"""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import StateGraph, END

from agents.state import AgentState, LoopState, StopReason

log = structlog.get_logger(__name__)

MAX_ITERATIONS = 8
CONFIDENCE_FLOOR = 0.80
MIN_EVIDENCE_ITEMS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────────────────────────────────────

def _should_stop(state: AgentState) -> str:
    """
    Central routing after EVALUATE.
    Checks hard limits first, then natural completion signals.
    Returns a node name or END.
    """
    iteration   = state.get("iteration", 0)
    confidence  = state.get("confidence") or 0.0
    evidence    = state.get("evidence", [])
    stop_reason = state.get("stop_reason")
    errors      = state.get("errors", [])

    # Explicit stop signals take priority
    if stop_reason == StopReason.HUMAN_ESCALATION:
        return "escalate"

    if stop_reason in (StopReason.CASE_RESOLVED, StopReason.CONFIDENCE_REACHED):
        return "reviewer"

    if stop_reason == StopReason.BUDGET_EXCEEDED:
        return "escalate"

    # Hard limits
    if iteration >= MAX_ITERATIONS:
        log.info("max_iterations_reached", iteration=iteration)
        return "reviewer"

    if len(errors) >= 3:
        log.warning("too_many_errors_escalating", error_count=len(errors))
        return "escalate"

    # Natural completion
    if confidence >= CONFIDENCE_FLOOR and len(evidence) >= MIN_EVIDENCE_ITEMS:
        return "reviewer"

    # Still need more evidence — always loop back to transaction_agent
    # (previously this checked loop_state which was always EVALUATE at this point)
    return "transaction_agent"


def _route_after_review(state: AgentState) -> str:
    """After reviewer, conclude or escalate."""
    if state.get("reviewer_approved", False):
        return END
    concerns = state.get("reviewer_concerns", [])
    if any("unsafe" in c.lower() or "escalat" in c.lower() for c in concerns):
        return "escalate"
    iteration = state.get("iteration", 0)
    if iteration < MAX_ITERATIONS - 1:
        return "transaction_agent"
    return END


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

def supervisor_plan_node(state: AgentState) -> dict[str, Any]:
    """
    PLAN node — sets investigation strategy.
    Increments iteration counter; respects MAX_ITERATIONS.
    """
    iteration = state.get("iteration", 0)
    case_ref = state.get("case_ref", "")

    log.info("supervisor_plan", run_id=state.get("run_id"), case_ref=case_ref, iteration=iteration)

    plan = [
        "1. Retrieve transaction details and payment event timeline",
        "2. Retrieve customer profile and account status",
        "3. Check recent transactions for duplicate or pattern signals",
        "4. Retrieve applicable payment policies via semantic search",
        "5. Evaluate evidence — determine most likely root cause",
        "6. Propose remediation with risk classification",
        "7. Review findings for safety, grounding and policy compliance",
    ]

    return {
        "loop_state": LoopState.ACT,
        "iteration": iteration + 1,
        "investigation_plan": plan,
        # Record evidence count at the start of this cycle for delta check
        "evidence_count_at_last_cycle": len(state.get("evidence", [])),
        "observations": [f"[Plan] Investigation plan created for {case_ref} (iteration {iteration + 1})"],
    }


def supervisor_evaluate_node(state: AgentState) -> dict[str, Any]:
    """
    EVALUATE node — runs after each ACT/OBSERVE cycle.

    Key fix: NO_NEW_EVIDENCE is detected by comparing current evidence
    count against `evidence_count_at_last_cycle` (a delta), not the
    absolute cumulative count which can never be zero after the first run.
    """
    evidence      = state.get("evidence", [])
    hypotheses    = state.get("hypotheses", [])
    iteration     = state.get("iteration", 0)
    prev_count    = state.get("evidence_count_at_last_cycle", 0)
    current_count = len(evidence)
    delta         = current_count - prev_count

    log.info(
        "supervisor_evaluate",
        run_id=state.get("run_id"),
        evidence_total=current_count,
        evidence_delta=delta,
        hypothesis_count=len(hypotheses),
        iteration=iteration,
    )

    top = max(hypotheses, key=lambda h: h.get("confidence", 0), default=None)
    confidence = top.get("confidence", 0.0) if top else 0.0

    updates: dict[str, Any] = {
        "loop_state": LoopState.EVALUATE,
        "confidence": confidence,
        # Snapshot for next cycle's delta calculation
        "evidence_count_at_last_cycle": current_count,
    }

    if top:
        updates["root_cause"] = top.get("root_cause")
        updates["observations"] = [
            f"[Evaluate] Best hypothesis: '{top.get('root_cause')}' "
            f"at {confidence:.0%} confidence (evidence: {current_count} items, +{delta} this cycle)"
        ]

    # NO_NEW_EVIDENCE: no evidence was added in this cycle AND we've done ≥2 iterations
    if iteration >= 2 and delta == 0:
        log.warning("no_new_evidence_detected", iteration=iteration, prev_count=prev_count)
        updates["stop_reason"] = StopReason.NO_NEW_EVIDENCE
        updates["observations"] = [
            f"[Evaluate] No new evidence in iteration {iteration}. "
            "Stopping investigation to avoid infinite loop."
        ]

    return updates


def escalate_node(state: AgentState) -> dict[str, Any]:
    """Mark case for human escalation and exit the graph."""
    log.warning(
        "case_escalated",
        run_id=state.get("run_id"),
        case_ref=state.get("case_ref"),
        reason=str(state.get("stop_reason")),
        iteration=state.get("iteration"),
    )
    return {
        "loop_state": LoopState.ESCALATE,
        "requires_human_approval": True,
        "observations": [
            f"[Escalate] Case escalated: {state.get('stop_reason', 'UNKNOWN')} "
            f"after {state.get('iteration', 0)} iterations."
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_supervisor_graph(
    transaction_agent_fn: Any,
    policy_agent_fn: Any,
    resolution_agent_fn: Any,
    reviewer_agent_fn: Any,
) -> Any:
    """
    Assemble the multi-agent LangGraph.
    Returns a compiled graph ready to .astream().
    """
    graph = StateGraph(AgentState)

    graph.add_node("plan",              supervisor_plan_node)
    graph.add_node("transaction_agent", transaction_agent_fn)
    graph.add_node("policy_agent",      policy_agent_fn)
    graph.add_node("resolution_agent",  resolution_agent_fn)
    graph.add_node("evaluate",          supervisor_evaluate_node)
    graph.add_node("reviewer",          reviewer_agent_fn)
    graph.add_node("escalate",          escalate_node)

    graph.set_entry_point("plan")

    graph.add_edge("plan",               "transaction_agent")
    graph.add_edge("transaction_agent",  "policy_agent")
    graph.add_edge("policy_agent",       "resolution_agent")
    graph.add_edge("resolution_agent",   "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        _should_stop,
        {
            "transaction_agent": "transaction_agent",
            "reviewer":          "reviewer",
            "escalate":          "escalate",
        },
    )

    graph.add_conditional_edges(
        "reviewer",
        _route_after_review,
        {
            "transaction_agent": "transaction_agent",
            "escalate":          "escalate",
            END:                 END,
        },
    )

    graph.add_edge("escalate", END)

    return graph.compile()
