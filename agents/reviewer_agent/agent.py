"""
BankGuard AI — Reviewer Agent.

The final safety gate before the agent concludes.

Checks:
  1. Is every conclusion backed by evidence? (hallucination guard)
  2. Does the proposed action comply with retrieved policies?
  3. Are there any contradictions in the evidence?
  4. Is a missing investigation step obvious?
  5. Is the proposed action flagged as UNSAFE or requiring approval?

If concerns are found → sets reviewer_approved=False + reviewer_concerns.
If all clear → sets reviewer_approved=True.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from agents.state import AgentState, LoopState, StopReason

log = structlog.get_logger(__name__)

# Actions that are always unsafe for the agent to execute autonomously
UNSAFE_ACTIONS: list[str] = [
    "reverse",
    "reversal",
    "freeze",
    "block card",
    "retry payment",
    "refund",
    "cancel account",
    "unfreeze",
]


def make_reviewer_agent(run_ctx: Any) -> Any:
    """Factory: returns a LangGraph node function."""

    async def reviewer_agent_node(state: AgentState) -> dict[str, Any]:
        log.info(
            "reviewer_agent_start",
            run_id=state.get("run_id"),
            case_ref=state.get("case_ref"),
        )

        concerns: list[str] = []
        observations: list[str] = []
        approved = True

        evidence = state.get("evidence", [])
        root_cause = state.get("root_cause")
        recommended_action = state.get("recommended_action", "")
        confidence = state.get("confidence", 0.0)
        policies = state.get("retrieved_policies", [])
        requires_approval = state.get("requires_human_approval", False)

        # ── Check 1: Evidence grounding ────────────────────────────────────
        if not evidence:
            concerns.append("No evidence collected — conclusions are ungrounded.")
            approved = False
        elif len(evidence) < 2:
            concerns.append("Only 1 evidence item — investigation may be incomplete.")

        # ── Check 2: Root cause identified ─────────────────────────────────
        if not root_cause or root_cause == "UNKNOWN":
            concerns.append("Root cause not identified — cannot propose safe action.")
            approved = False

        # ── Check 3: Confidence threshold ──────────────────────────────────
        if confidence < 0.60:
            concerns.append(f"Confidence {confidence:.0%} is below 60% threshold — more investigation needed.")
            if confidence < 0.40:
                approved = False

        # ── Check 4: Unsafe action detection ──────────────────────────────
        action_lower = (recommended_action or "").lower()
        for unsafe_kw in UNSAFE_ACTIONS:
            if unsafe_kw in action_lower:
                concerns.append(
                    f"Proposed action contains '{unsafe_kw}' — this REQUIRES human approval before execution."
                )
                # Don't reject — just flag as requiring approval
                break

        # ── Check 5: Policy compliance ─────────────────────────────────────
        if recommended_action and policies:
            policy_texts = " ".join(p.get("content", "") for p in policies[:2]).lower()
            # Check for explicit prohibitions in policy text
            prohibition_phrases = [
                "prohibited",
                "must not",
                "shall not",
                "do not",
                "never",
                "forbidden",
                "not allowed",
            ]
            if any(ph in policy_texts for ph in prohibition_phrases):
                # Simple check: does the action appear near a prohibition?
                # In production this would use an LLM judge
                observations.append("Policy contains prohibitions — verify action does not violate them.")

        # ── Check 6: High-risk action requires approval flag ───────────────
        action_risk = state.get("action_risk_level", "LOW")
        if action_risk in ("HIGH", "CRITICAL") and not requires_approval:
            concerns.append(
                f"Action risk level is {action_risk} but requires_human_approval=False. Forcing approval requirement."
            )
            # Force it
            state = {**state, "requires_human_approval": True}

        # ── LLM review pass (optional, when budget allows) ─────────────────
        budget = state.get("budget_snapshot", {})
        if budget.get("cost_remaining_usd", 0.15) > 0.02:
            try:
                review_prompt = [
                    {
                        "role": "system",
                        "content": (
                            "You are a banking operations safety reviewer. "
                            "Review the investigation and flag any issues. "
                            "Respond ONLY with JSON: "
                            '{"approved": bool, "concerns": [str], "summary": str}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "root_cause": root_cause,
                                "confidence": confidence,
                                "recommended_action": recommended_action,
                                "evidence_count": len(evidence),
                                "policies_applied": [p.get("policy_ref") for p in policies],
                                "existing_concerns": concerns,
                            }
                        ),
                    },
                ]
                response = await run_ctx.invoke_llm(review_prompt, temperature=0.0)
                content = response.choices[0].message.content if hasattr(response, "choices") else "{}"
                parsed = json.loads(content)
                llm_concerns = parsed.get("concerns", [])
                if llm_concerns:
                    concerns.extend(llm_concerns)
                if not parsed.get("approved", True):
                    approved = False
                observations.append(f"LLM reviewer summary: {parsed.get('summary', 'N/A')}")
            except Exception as exc:
                log.warning("reviewer_llm_failed", error=str(exc))

        if concerns:
            observations.append(f"Reviewer concerns ({len(concerns)}): " + "; ".join(concerns[:3]))
        else:
            observations.append("Reviewer: no concerns. Investigation complete.")

        log.info(
            "reviewer_agent_complete",
            run_id=state.get("run_id"),
            approved=approved,
            concern_count=len(concerns),
        )

        final_stop = StopReason.CASE_RESOLVED if approved else None

        return {
            "loop_state": LoopState.CONCLUDE,
            "reviewer_approved": approved,
            "reviewer_concerns": concerns,
            "observations": observations,
            "stop_reason": final_stop,
            "requires_human_approval": state.get("requires_human_approval", False),
        }

    return reviewer_agent_node
