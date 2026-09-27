"""
BankGuard AI — Agent Runner.

Ties together:
  - AgentRuntime (harness)
  - WorkingMemory
  - EpisodicMemory
  - ContextBuilder
  - LangGraph supervisor graph

Entry point:  runner.investigate(case_ref, identity)
"""

from __future__ import annotations

import uuid
from typing import Any, AsyncGenerator

import structlog

from agents.state import AgentState, LoopState
from agents.supervisor import build_supervisor_graph
from agents.transaction_agent import make_transaction_agent
from agents.policy_agent import make_policy_agent
from agents.resolution_agent import make_resolution_agent
from agents.reviewer_agent import make_reviewer_agent
from harness.runtime import AgentRuntime, StoppingReason
from harness.sandbox import SandboxMode
from memory.working import WorkingMemory
from memory.episodic import EpisodicMemory

log = structlog.get_logger(__name__)


class AgentRunner:
    """
    High-level runner that orchestrates a full case investigation.

    Usage:
        runner = AgentRunner(db_pool, valkey)
        async for event in runner.stream_investigation(case_ref, identity):
            print(event)
    """

    def __init__(
        self,
        db: Any,
        valkey: Any,
        sandbox_mode: SandboxMode | None = None,
    ) -> None:
        self.db = db
        self.valkey = valkey
        self.runtime = AgentRuntime(db, valkey, sandbox_mode)
        self.episodic = EpisodicMemory(db)

    async def investigate(
        self,
        case_ref: str,
        identity: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run a full case investigation synchronously.
        Returns the final AgentState as a dict.
        """
        result = None
        async for event in self.stream_investigation(case_ref, identity):
            if event.get("type") == "final":
                result = event.get("state")
        return result or {}

    async def stream_investigation(
        self,
        case_ref: str,
        identity: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Run a full case investigation, yielding events as the graph progresses.
        Each yielded dict has: {"type": "step"|"final"|"error", "node": str, "state": ...}
        """
        # Load case from DB
        case_data = await self._load_case(case_ref)
        if not case_data:
            yield {"type": "error", "message": f"Case {case_ref} not found"}
            return

        async with self.runtime.run(case_data["id"], identity) as run_ctx:
            wm = WorkingMemory(self.valkey, run_ctx.run_id)
            await wm.set_state("PLAN")

            # Build agent nodes bound to this run context
            txn_fn    = make_transaction_agent(run_ctx)
            pol_fn    = make_policy_agent(run_ctx)
            res_fn    = make_resolution_agent(run_ctx)
            rev_fn    = make_reviewer_agent(run_ctx)

            graph = build_supervisor_graph(txn_fn, pol_fn, res_fn, rev_fn)

            # Build initial state
            initial_state: dict[str, Any] = {
                "messages":               [],
                "run_id":                 run_ctx.run_id,
                "case_id":                case_data["id"],
                "case_ref":               case_ref,
                "identity":               identity,
                "loop_state":             LoopState.PLAN,
                "iteration":              0,
                "stop_reason":            None,
                "case_data":              case_data,
                "customer_data":          None,
                "transaction_data":       None,
                "related_transactions":   [],
                "evidence":               [],
                "investigation_plan":     [],
                "observations":           [],
                "hypotheses":             [],
                "retrieved_policies":     [],
                "root_cause":             None,
                "confidence":             None,
                "recommended_action":     None,
                "action_risk_level":      None,
                "requires_human_approval": False,
                "resolution_notes":       None,
                "reviewer_approved":      False,
                "reviewer_concerns":      [],
                "tool_calls_made":        [],
                "last_tool_results":      [],
                "budget_snapshot":        None,
                "errors":                 [],
            }

            yield {"type": "step", "node": "start", "run_id": run_ctx.run_id}

            # Stream through the graph
            async for chunk in graph.astream(initial_state):
                node_name = list(chunk.keys())[0] if chunk else "unknown"
                node_state = chunk.get(node_name, {})

                # Update working memory observations
                for obs in node_state.get("observations", []):
                    await wm.add_observation(obs)
                for ev in node_state.get("evidence", []):
                    await wm.add_evidence(ev.get("source", ""), ev.get("content"))

                yield {
                    "type": "step",
                    "node": node_name,
                    "observations": node_state.get("observations", []),
                    "evidence_added": len(node_state.get("evidence", [])),
                    "loop_state": str(node_state.get("loop_state", "")),
                }

            # Get final state from graph
            final = await graph.aget_state({"configurable": {}})
            final_vals = final.values if hasattr(final, "values") else initial_state

            # Persist episode
            await self.episodic.record(
                case_id=case_ref,
                run_id=run_ctx.run_id,
                event_type="INVESTIGATION_COMPLETE",
                summary=(
                    f"Root cause: {final_vals.get('root_cause')}. "
                    f"Action: {final_vals.get('recommended_action', '')[:200]}"
                ),
                outcome="RESOLVED" if final_vals.get("reviewer_approved") else "ESCALATED",
            )

            # Update case status in DB
            await self._update_case_status(
                case_ref=case_ref,
                root_cause=final_vals.get("root_cause"),
                resolution=final_vals.get("recommended_action"),
                confidence=final_vals.get("confidence"),
                requires_approval=final_vals.get("requires_human_approval", False),
            )

            run_ctx.stop(StoppingReason.CASE_RESOLVED, final_output=dict(final_vals))

            yield {
                "type": "final",
                "run_id": run_ctx.run_id,
                "state": dict(final_vals),
            }

    # ── DB helpers ────────────────────────────────────────────────────────────

    async def _load_case(self, case_ref: str) -> dict | None:
        try:
            async with self.db.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT c.id::text, c.case_ref, c.title, c.description,
                           c.status, c.priority, c.created_at,
                           cu.customer_ref,
                           t.transaction_ref
                    FROM agent.cases c
                    LEFT JOIN banking.customers cu ON cu.id = c.customer_id
                    LEFT JOIN banking.transactions t ON t.id = c.transaction_id
                    WHERE c.case_ref = $1
                    """,
                    case_ref,
                )
                if not row:
                    return None
                return dict(row)
        except Exception as exc:
            log.error("load_case_error", case_ref=case_ref, error=str(exc))
            return None

    async def _update_case_status(
        self,
        case_ref: str,
        root_cause: str | None,
        resolution: str | None,
        confidence: float | None,
        requires_approval: bool,
    ) -> None:
        new_status = "PENDING_APPROVAL" if requires_approval else "RESOLVED"
        try:
            async with self.db.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE agent.cases
                    SET status     = $2,
                        root_cause = $3,
                        resolution = $4,
                        confidence = $5,
                        updated_at = NOW()
                    WHERE case_ref = $1
                    """,
                    case_ref,
                    new_status,
                    root_cause,
                    resolution,
                    confidence,
                )
        except Exception as exc:
            log.warning("update_case_status_error", error=str(exc))
