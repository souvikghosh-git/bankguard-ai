"""
BankGuard AI — Policy Agent.

Retrieves applicable banking policies based on:
  - Transaction type and payment rail
  - Current root cause hypothesis
  - Case description keywords

Adds policy constraints to the agent state so the
Resolution Agent can propose compliant actions.
"""

from __future__ import annotations

from typing import Any

import structlog

from agents.state import AgentState, LoopState

log = structlog.get_logger(__name__)


def make_policy_agent(run_ctx: Any) -> Any:
    """Factory: returns a LangGraph node function."""

    async def policy_agent_node(state: AgentState) -> dict[str, Any]:
        log.info(
            "policy_agent_start",
            run_id=state.get("run_id"),
            case_ref=state.get("case_ref"),
        )

        evidence: list[dict] = []
        observations: list[str] = []
        policies: list[dict] = []

        hypotheses = state.get("hypotheses", [])
        transaction_data = state.get("transaction_data", {}) or {}
        case_data = state.get("case_data", {})

        # Build search queries from context
        queries = [case_data.get("description", "payment failure")]
        if hypotheses:
            top = max(hypotheses, key=lambda h: h.get("confidence", 0))
            root_cause = top.get("root_cause", "")
            queries.append(f"{root_cause} payment policy procedure")

        payment_rail = transaction_data.get("payment_rail")
        if payment_rail:
            queries.append(f"{payment_rail} payment policy reconciliation")

        # Search for policies with each query (deduplicate by policy_ref)
        seen_refs: set[str] = set()
        for query in queries[:3]:   # cap at 3 search calls
            result = await run_ctx.invoke_tool(
                "search_policy",
                {"query": query, "top_k": 3},
            )
            if result.status == "SUCCESS":
                for p in result.data.get("policies", []):
                    ref = p.get("policy_ref", "")
                    if ref not in seen_refs:
                        seen_refs.add(ref)
                        policies.append(p)
                        observations.append(
                            f"Retrieved policy {ref}: {p.get('title')}"
                        )

        # Also get case history to spot patterns
        customer_ref = case_data.get("customer_ref")
        if customer_ref:
            hist_result = await run_ctx.invoke_tool(
                "get_case_history",
                {"customer_ref": customer_ref, "limit": 3},
            )
            if hist_result.status == "SUCCESS":
                prev_cases = hist_result.data.get("cases", [])
                if prev_cases:
                    observations.append(
                        f"Customer has {len(prev_cases)} previous cases. "
                        f"Latest: {prev_cases[0].get('case_ref')} — "
                        f"status={prev_cases[0].get('status')}"
                    )
                    evidence.append({"source": "get_case_history", "content": prev_cases})

        log.info(
            "policy_agent_complete",
            run_id=state.get("run_id"),
            policies_found=len(policies),
        )

        return {
            "loop_state": LoopState.OBSERVE,
            "retrieved_policies": policies,
            "evidence": evidence,
            "observations": observations,
        }

    return policy_agent_node
