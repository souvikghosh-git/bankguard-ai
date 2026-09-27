"""
BankGuard AI — Agent Runner.

Orchestrates a full case investigation:
  - AgentRuntime (harness: budget, retry, circuit-breaker, sandbox)
  - WorkingMemory (Valkey — per-run state)
  - EpisodicMemory (PostgreSQL — case narrative)
  - ContextBuilder (token-budgeted context assembly)
  - LangGraph supervisor graph (multi-agent loop)
  - Observability (OTel trace, Prometheus, Langfuse — all wired here)

Observability contract:
  Every run calls trace_agent_run(run_id, case_ref) which:
    1. Derives a stable OTel trace_id from run_id
    2. Binds run_id + case_ref + trace_id to structlog context-vars
       → every log line inside the run carries these automatically
    3. Emits Langfuse trace for LLM calls
    4. Increments Prometheus counters/gauges on every lifecycle event
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import structlog

from agents.policy_agent import make_policy_agent
from agents.resolution_agent import make_resolution_agent
from agents.reviewer_agent import make_reviewer_agent
from agents.state import LoopState
from agents.supervisor import build_supervisor_graph
from agents.transaction_agent import make_transaction_agent
from harness.runtime import AgentRuntime, StoppingReason
from harness.sandbox import SandboxMode
from memory.episodic import EpisodicMemory
from memory.working import WorkingMemory
from observability.telemetry import (
    record_tool_call,
    setup_telemetry,
    trace_agent_run,
)

log = structlog.get_logger(__name__)


class AgentRunner:
    """
    High-level runner — entry point for all case investigations.

    Usage:
        runner = AgentRunner(db_pool, valkey)
        async for event in runner.stream_investigation(case_ref, identity):
            yield event
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
        setup_telemetry()  # idempotent — safe to call multiple times

    async def investigate(
        self,
        case_ref: str,
        identity: dict[str, Any],
    ) -> dict[str, Any]:
        """Synchronous wrapper around stream_investigation."""
        result: dict[str, Any] = {}
        async for event in self.stream_investigation(case_ref, identity):
            if event.get("type") == "final":
                result = event.get("state", {})
        return result

    async def stream_investigation(
        self,
        case_ref: str,
        identity: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Run a full case investigation, yielding events as the graph progresses.

        Each yielded dict has:
            {"type": "step"|"final"|"error", "node": str, ...}

        Observability fires at:
            - Run start → OTel span opened, structlog vars bound, Langfuse trace created
            - Each LLM call → record_llm_call() → Prometheus + Langfuse
            - Each tool call → record_tool_call() → Prometheus + OTel event
            - Run end → Prometheus counters incremented, OTel span closed
        """
        case_data = await self._load_case(case_ref)
        if not case_data:
            yield {"type": "error", "message": f"Case {case_ref} not found"}
            return

        sandbox_str = "development"  # resolved properly below

        async with self.runtime.run(case_data["id"], identity) as run_ctx:
            sandbox_str = run_ctx.sandbox.mode.value

            async with trace_agent_run(
                run_id=run_ctx.run_id,
                case_ref=case_ref,
                sandbox_mode=sandbox_str,
            ) as tctx:
                # Attach the Langfuse trace to run_ctx so agent nodes can
                # call record_llm_call() with it.
                run_ctx._langfuse_trace = tctx.get("langfuse_trace")
                run_ctx._otel_span = tctx.get("otel_span")

                # Working memory
                wm = WorkingMemory(self.valkey, run_ctx.run_id)
                await wm.set_state("PLAN")

                # Persist run record so tool audit FK works
                await self._create_run_record(run_ctx.run_id, case_data["id"], identity)

                # Build agent nodes bound to this run context
                txn_fn = make_transaction_agent(run_ctx)
                pol_fn = make_policy_agent(run_ctx)
                res_fn = make_resolution_agent(run_ctx)
                rev_fn = make_reviewer_agent(run_ctx)

                graph = build_supervisor_graph(txn_fn, pol_fn, res_fn, rev_fn)

                initial_state: dict[str, Any] = {
                    "messages": [],
                    "run_id": run_ctx.run_id,
                    "case_id": case_data["id"],
                    "case_ref": case_ref,
                    "identity": identity,
                    "loop_state": LoopState.PLAN,
                    "iteration": 0,
                    "stop_reason": None,
                    "case_data": case_data,
                    "customer_data": None,
                    "transaction_data": None,
                    "related_transactions": [],
                    "evidence": [],
                    "evidence_count_at_last_cycle": 0,
                    "investigation_plan": [],
                    "observations": [],
                    "hypotheses": [],
                    "retrieved_policies": [],
                    "root_cause": None,
                    "confidence": None,
                    "recommended_action": None,
                    "action_risk_level": None,
                    "requires_human_approval": False,
                    "resolution_notes": None,
                    "reviewer_approved": False,
                    "reviewer_concerns": [],
                    "tool_calls_made": [],
                    "last_tool_results": [],
                    "budget_snapshot": None,
                    "errors": [],
                }

                yield {
                    "type": "step",
                    "node": "start",
                    "run_id": run_ctx.run_id,
                    "trace_id": tctx["trace_id"],
                }

                final_state = initial_state
                try:
                    async for chunk in graph.astream(initial_state):
                        node_name = next(iter(chunk), "unknown")
                        node_state = chunk.get(node_name, {})

                        # Update working memory
                        for obs in node_state.get("observations", []):
                            await wm.add_observation(obs)
                        for ev in node_state.get("evidence", []):
                            await wm.add_evidence(ev.get("source", ""), ev.get("content"))

                        # Emit Prometheus for tool calls made this step
                        for tool_ref in node_state.get("tool_calls_made", []):
                            record_tool_call(
                                tool_name=str(tool_ref).split(":")[0],
                                status="SUCCESS",
                                duration_ms=0,
                                otel_span=run_ctx._otel_span,
                            )

                        # Merge node_state into final_state for end-of-run use
                        final_state = {**final_state, **node_state}

                        yield {
                            "type": "step",
                            "node": node_name,
                            "observations": node_state.get("observations", []),
                            "evidence_added": len(node_state.get("evidence", [])),
                            "loop_state": str(node_state.get("loop_state", "")),
                            "iteration": node_state.get("iteration", 0),
                            "trace_id": tctx["trace_id"],
                        }

                except Exception as exc:
                    log.exception("graph_stream_error", error=str(exc))
                    tctx["final_status"] = "error"
                    yield {"type": "error", "message": str(exc)}
                    return

                # Populate telemetry context for cleanup in trace_agent_run
                tctx["final_status"] = "success" if final_state.get("reviewer_approved") else "escalated"
                tctx["iterations"] = final_state.get("iteration", 0)
                tctx["root_cause"] = final_state.get("root_cause")
                tctx["stop_reason"] = str(final_state.get("stop_reason", ""))

                # Record cost metrics from the budget
                snap = run_ctx.budget.snapshot()
                from config import settings
                from observability.telemetry import llm_tokens_total

                model = settings.bedrock_default_model
                if snap.input_tokens_used > 0:
                    llm_tokens_total.labels(model=model, direction="input").inc(snap.input_tokens_used)
                if snap.output_tokens_used > 0:
                    llm_tokens_total.labels(model=model, direction="output").inc(snap.output_tokens_used)

                # Persist episode and update case
                await self.episodic.record(
                    case_id=case_ref,
                    run_id=run_ctx.run_id,
                    event_type="INVESTIGATION_COMPLETE",
                    summary=(
                        f"Root cause: {final_state.get('root_cause')}. "
                        f"Action: {str(final_state.get('recommended_action', ''))[:200]}"
                    ),
                    outcome="RESOLVED" if final_state.get("reviewer_approved") else "ESCALATED",
                )

                await self._update_case_status(
                    case_ref=case_ref,
                    root_cause=final_state.get("root_cause"),
                    resolution=final_state.get("recommended_action"),
                    confidence=final_state.get("confidence"),
                    requires_approval=final_state.get("requires_human_approval", False),
                )

                await self._update_run_record(
                    run_id=run_ctx.run_id,
                    snap=snap,
                    stopping_reason=str(final_state.get("stop_reason", "")),
                    final_output=final_state,
                )

                run_ctx.stop(StoppingReason.CASE_RESOLVED, final_output=dict(final_state))

                yield {
                    "type": "final",
                    "run_id": run_ctx.run_id,
                    "trace_id": tctx["trace_id"],
                    "state": dict(final_state),
                }

    # ─────────────────────────────────────────────────────────────────────────
    # DB helpers
    # ─────────────────────────────────────────────────────────────────────────

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
                return dict(row) if row else None
        except Exception as exc:
            log.error("load_case_error", case_ref=case_ref, error=str(exc))
            return None

    async def _create_run_record(self, run_id: str, case_id: str, identity: dict) -> None:
        """Insert an agent_runs row so tool_calls FK can resolve."""
        try:
            async with self.db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO agent.agent_runs
                        (id, run_ref, case_id, agent_id, model_id, status)
                    VALUES (gen_random_uuid(), $1,
                            (SELECT id FROM agent.cases WHERE id::text = $2 LIMIT 1),
                            $3, $4, 'RUNNING')
                    ON CONFLICT DO NOTHING
                    """,
                    run_id,
                    case_id,
                    identity.get("user_id", "agent"),
                    "amazon.nova-micro-v1:0",
                )
        except Exception as exc:
            log.warning("create_run_record_error", error=str(exc))

    async def _update_run_record(self, run_id: str, snap: Any, stopping_reason: str, final_output: dict) -> None:
        import json

        try:
            async with self.db.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE agent.agent_runs
                    SET status          = 'COMPLETED',
                        iteration_count = $2,
                        tool_call_count = $3,
                        input_tokens    = $4,
                        output_tokens   = $5,
                        cost_usd        = $6,
                        stopping_reason = $7,
                        final_output    = $8::jsonb,
                        completed_at    = NOW()
                    WHERE run_ref = $1
                    """,
                    run_id,
                    snap.iterations_used,
                    snap.tool_calls_used,
                    snap.input_tokens_used,
                    snap.output_tokens_used,
                    snap.cost_usd,
                    stopping_reason,
                    json.dumps(
                        {
                            k: str(v)
                            for k, v in final_output.items()
                            if k
                            in (
                                "root_cause",
                                "confidence",
                                "recommended_action",
                                "action_risk_level",
                                "requires_human_approval",
                            )
                        },
                    ),
                )
        except Exception as exc:
            log.warning("update_run_record_error", error=str(exc))

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
