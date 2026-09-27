"""
BankGuard AI — Supervisor Agent (LangGraph orchestrator).

The supervisor owns the top-level PLAN→ACT→OBSERVE→EVALUATE loop.
It delegates to specialist nodes and decides when to stop.

Graph topology:
                    [supervisor]
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
  [transaction_agent] [policy_agent] [resolution_agent]
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                   [reviewer_agent]
                         │
                    (stop / escalate)
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from agents.state import AgentState, LoopState, StopReason

log = structlog.get_logger(__name__)

# ── Stopping conditions ───────────────────────────────────────────────────────

MAX_ITERATIONS    = 8
CONFIDENCE_FLOOR  = 0.80   # stop when confidence ≥ this
MIN_EVIDENCE_ITEMS = 3     # need at least this many evidence items before concluding


def _should_stop(state: AgentState) -> str:
    """
    Central routing function — decides the next node after EVALUATE.
    Returns a node name (str) or END.
    """
    iteration    = state.get("iteration", 0)
    confidence   = state.get("confidence") or 0.0
    evidence     = state.get("evidence", [])
    stop_reason  = state.get("stop_reason")
    errors       = state.get("errors", [])
    loop_state   = state.get("loop_state", LoopState.PLAN)

    # Explicit stop signals
    if stop_reason in (StopReason.HUMAN_ESCALATION,):
        return "escalate"

    if stop_reason in (
        StopReason.CASE_RESOLVED,
        StopReason.CONFIDENCE_REACHED,
    ):
        return "reviewer"

    # Hard limits
    if iteration >= MAX_ITERATIONS:
        return "reviewer"

    if len(errors) >= 3:
        return "escalate"

    # Natural completion
    if confidence >= CONFIDENCE_FLOOR and len(evidence) >= MIN_EVIDENCE_ITEMS:
        return "reviewer"

    # Still gathering evidence
    if loop_state in (LoopState.PLAN, LoopState.ACT, LoopState.OBSERVE):
        return "transaction_agent"

    return "resolution_agent"


def _route_after_review(state: AgentState) -> str:
    """After reviewer runs, conclude or escalate."""
    if state.get("reviewer_approved", False):
        return END
    concerns = state.get("reviewer_concerns", [])
    if any("unsafe" in c.lower() or "escalat" in c.lower() for c in concerns):
        return "escalate"
    # Reviewer found issues — do one more evidence pass
    iteration = state.get("iteration", 0)
    if iteration < MAX_ITERATIONS - 1:
        return "transaction_agent"
    return END


# ── Node: entry / plan ────────────────────────────────────────────────────────

def supervisor_plan_node(state: AgentState) -> dict[str, Any]:
    """
    Initial PLAN node.
    Sets up the investigation plan and moves to ACT.
    """
    case = state.get("case_data", {})
    description = case.get("description", "")

    log.info(
        "supervisor_plan",
        run_id=state.get("run_id"),
        case_ref=state.get("case_ref"),
        iteration=state.get("iteration", 0),
    )

    plan = [
        "1. Retrieve transaction details and payment events",
        "2. Retrieve customer profile and account status",
        "3. Check related transactions for duplicates or patterns",
        "4. Search applicable payment policies",
        "5. Evaluate evidence and identify root cause",
        "6. Propose remediation action with risk classification",
        "7. Review for policy compliance and safety",
    ]

    return {
        "loop_state": LoopState.ACT,
        "iteration": state.get("iteration", 0) + 1,
        "investigation_plan": plan,
        "observations": [f"Investigation plan created for case {state.get('case_ref')}"],
    }


# ── Node: evaluate ────────────────────────────────────────────────────────────

def supervisor_evaluate_node(state: AgentState) -> dict[str, Any]:
    """
    EVALUATE node — runs after each ACT/OBSERVE cycle.
    Determines: have we gathered enough evidence?
    """
    evidence     = state.get("evidence", [])
    hypotheses   = state.get("hypotheses", [])
    iteration    = state.get("iteration", 0)

    log.info(
        "supervisor_evaluate",
        run_id=state.get("run_id"),
        evidence_count=len(evidence),
        hypothesis_count=len(hypotheses),
        iteration=iteration,
    )

    # Check for a dominant hypothesis
    top_hypothesis = None
    if hypotheses:
        top_hypothesis = max(hypotheses, key=lambda h: h.get("confidence", 0))

    confidence = top_hypothesis.get("confidence", 0.0) if top_hypothesis else 0.0

    updates: dict[str, Any] = {
        "loop_state": LoopState.EVALUATE,
        "confidence": confidence,
    }

    if top_hypothesis:
        updates["root_cause"] = top_hypothesis.get("root_cause")
        updates["observations"] = [
            f"Best hypothesis: {top_hypothesis.get('root_cause')} "
            f"(confidence: {confidence:.0%})"
        ]

    # No new evidence from last ACT cycle
    last_tools = state.get("tool_calls_made", [])
    if iteration > 2 and len(evidence) == 0:
        updates["stop_reason"] = StopReason.NO_NEW_EVIDENCE
        updates["observations"] = ["No new evidence gathered. Stopping investigation."]

    return updates


# ── Node: escalate ────────────────────────────────────────────────────────────

def escalate_node(state: AgentState) -> dict[str, Any]:
    """Mark case for human escalation."""
    log.warning(
        "case_escalated",
        run_id=state.get("run_id"),
        case_ref=state.get("case_ref"),
        reason=str(state.get("stop_reason")),
    )
    return {
        "loop_state": LoopState.ESCALATE,
        "requires_human_approval": True,
        "observations": [
            f"Case escalated to human: {state.get('stop_reason', 'UNKNOWN')}"
        ],
    }


# ── Build the graph ───────────────────────────────────────────────────────────

def build_supervisor_graph(
    transaction_agent_fn: Any,
    policy_agent_fn: Any,
    resolution_agent_fn: Any,
    reviewer_agent_fn: Any,
) -> StateGraph:
    """
    Assemble the full multi-agent LangGraph.

    Returns a compiled graph ready to invoke.
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("plan",               supervisor_plan_node)
    graph.add_node("transaction_agent",  transaction_agent_fn)
    graph.add_node("policy_agent",       policy_agent_fn)
    graph.add_node("resolution_agent",   resolution_agent_fn)
    graph.add_node("evaluate",           supervisor_evaluate_node)
    graph.add_node("reviewer",           reviewer_agent_fn)
    graph.add_node("escalate",           escalate_node)

    # Entry point
    graph.set_entry_point("plan")

    # plan → transaction_agent
    graph.add_edge("plan", "transaction_agent")

    # After transaction investigation: run policy agent
    graph.add_edge("transaction_agent", "policy_agent")

    # After policy: run resolution
    graph.add_edge("policy_agent", "resolution_agent")

    # After resolution: evaluate
    graph.add_edge("resolution_agent", "evaluate")

    # Conditional routing after evaluate
    graph.add_conditional_edges(
        "evaluate",
        _should_stop,
        {
            "transaction_agent": "transaction_agent",
            "resolution_agent":  "resolution_agent",
            "reviewer":          "reviewer",
            "escalate":          "escalate",
        },
    )

    # After reviewer: conclude or re-investigate
    graph.add_conditional_edges(
        "reviewer",
        _route_after_review,
        {
            "transaction_agent": "transaction_agent",
            "escalate":          "escalate",
            END:                 END,
        },
    )

    # Escalate → END
    graph.add_edge("escalate", END)

    return graph.compile()
