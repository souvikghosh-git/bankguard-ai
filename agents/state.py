"""
BankGuard AI — Shared Agent State (LangGraph TypedDict).

This is the single source of truth that flows through every node
in the LangGraph graph.  Every agent reads from and writes to this.

Loop states:
  PLAN      → understand the case and build an investigation plan
  ACT       → execute tools to gather evidence
  OBSERVE   → review what the tools returned
  EVALUATE  → decide: enough evidence? root cause identified? action needed?
  CONCLUDE  → produce final output
  ESCALATE  → hand off to human
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Annotated
import operator

from langgraph.graph import MessagesState


class LoopState(str, Enum):
    PLAN     = "PLAN"
    ACT      = "ACT"
    OBSERVE  = "OBSERVE"
    EVALUATE = "EVALUATE"
    CONCLUDE = "CONCLUDE"
    ESCALATE = "ESCALATE"


class StopReason(str, Enum):
    CASE_RESOLVED         = "CASE_RESOLVED"
    CONFIDENCE_REACHED    = "CONFIDENCE_REACHED"
    NO_NEW_EVIDENCE       = "NO_NEW_EVIDENCE"
    MAX_ITERATIONS        = "MAX_ITERATIONS"
    BUDGET_EXCEEDED       = "BUDGET_EXCEEDED"
    REPEATED_TOOL_CALL    = "REPEATED_TOOL_CALL"
    CRITICAL_TOOL_FAILURE = "CRITICAL_TOOL_FAILURE"
    HUMAN_ESCALATION      = "HUMAN_ESCALATION"


class AgentState(MessagesState):
    """
    Complete mutable state for one BankGuard investigation run.
    Flows through all LangGraph nodes.
    """

    # ── Identity / run context ─────────────────────────────────────────────
    run_id: str
    case_id: str
    case_ref: str
    identity: dict[str, Any]

    # ── Loop control ───────────────────────────────────────────────────────
    loop_state: LoopState
    iteration: int
    stop_reason: StopReason | None

    # ── Investigation data ─────────────────────────────────────────────────
    case_data: dict[str, Any]
    customer_data: dict[str, Any] | None
    transaction_data: dict[str, Any] | None
    related_transactions: list[dict[str, Any]]

    # ── Evidence accumulator (append-only via reducer) ─────────────────────
    evidence: Annotated[list[dict[str, Any]], operator.add]

    # ── Agent outputs ──────────────────────────────────────────────────────
    investigation_plan: list[str]          # steps planned in PLAN phase
    observations: Annotated[list[str], operator.add]  # running narrative
    hypotheses: list[dict[str, Any]]       # [{root_cause, confidence, evidence}]

    # ── Policy retrieval results ───────────────────────────────────────────
    retrieved_policies: list[dict[str, Any]]

    # ── Final output ──────────────────────────────────────────────────────
    root_cause: str | None
    confidence: float | None
    recommended_action: str | None
    action_risk_level: str | None
    requires_human_approval: bool
    resolution_notes: str | None

    # ── Review flags ──────────────────────────────────────────────────────
    reviewer_approved: bool
    reviewer_concerns: list[str]

    # ── Tool call tracking ────────────────────────────────────────────────
    tool_calls_made: Annotated[list[str], operator.add]  # ["tool_name:hash", ...]
    last_tool_results: list[dict[str, Any]]

    # ── Budget snapshot ───────────────────────────────────────────────────
    budget_snapshot: dict[str, Any] | None

    # ── Error state ───────────────────────────────────────────────────────
    errors: Annotated[list[str], operator.add]
