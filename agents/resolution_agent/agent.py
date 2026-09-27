"""
BankGuard AI — Resolution Agent.

Given evidence + policies, this agent:
  1. Determines the most likely root cause
  2. Proposes the appropriate remediation action
  3. Classifies the risk level of that action
  4. Flags whether human approval is required

Uses the LLM (via RunContext) to reason across evidence.
Falls back to rule-based resolution when LLM is unavailable.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from agents.state import AgentState, LoopState, StopReason

log = structlog.get_logger(__name__)

# ── Rule-based resolution table (no LLM required) ────────────────────────────
# root_cause → (action, risk_level, requires_approval, notes)

RESOLUTION_RULES: dict[str, tuple[str, str, bool, str]] = {
    "BENEFICIARY_BANK_TIMEOUT": (
        "Wait for reconciliation window (24–48h). Create ops ticket to monitor. Do NOT retry payment.",
        "LOW",
        False,
        "Per PAY-REC-101: do not retry during reconciliation window.",
    ),
    "DUPLICATE_TRANSACTION": (
        "Cancel pending duplicate transaction. Notify customer.",
        "MEDIUM",
        True,
        "Requires Risk Officer confirmation before cancellation.",
    ),
    "PAYMENT_RAIL_UNAVAILABLE": (
        "Retry payment after rail restoration. Check NPCI status. Notify customer of delay.",
        "LOW",
        False,
        "Rail outage — retry when operational.",
    ),
    "INSUFFICIENT_FUNDS": (
        "Notify customer of insufficient funds. No bank action required.",
        "LOW",
        False,
        "Customer-side issue. No financial action needed.",
    ),
    "DEBIT_FAILURE": (
        "Investigate account status. Check for freezes or holds.",
        "MEDIUM",
        False,
        "May indicate account freeze or compliance hold.",
    ),
    "ACCOUNT_FROZEN": (
        "Escalate to compliance officer. Do not unfreeze without authorization.",
        "HIGH",
        True,
        "Per OPS-FRZ-501: unfreeze always requires human approval.",
    ),
    "PAYMENT_COMPLETED_OK": (
        "No action required. Payment completed successfully.",
        "LOW",
        False,
        "Transaction confirmed complete. Case can be closed.",
    ),
    "PAYMENT_REJECTED": (
        "Review rejection reason. Retry may be applicable if system error.",
        "MEDIUM",
        True,
        "Retry payment requires Risk Officer approval per PAY-REV-601.",
    ),
}

# Risk level → whether human approval is always required
RISK_APPROVAL_MAP = {
    "LOW": False,
    "MEDIUM": True,
    "HIGH": True,
    "CRITICAL": True,
}


def make_resolution_agent(run_ctx: Any) -> Any:
    """Factory: returns a LangGraph node function."""

    async def resolution_agent_node(state: AgentState) -> dict[str, Any]:
        log.info(
            "resolution_agent_start",
            run_id=state.get("run_id"),
            case_ref=state.get("case_ref"),
        )

        hypotheses = state.get("hypotheses", [])
        evidence = state.get("evidence", [])
        policies = state.get("retrieved_policies", [])
        case_data = state.get("case_data", {})
        observations: list[str] = []
        errors: list[str] = []

        # Pick the top hypothesis
        top_hypothesis: dict | None = None
        if hypotheses:
            top_hypothesis = max(hypotheses, key=lambda h: h.get("confidence", 0))

        root_cause = top_hypothesis["root_cause"] if top_hypothesis else "UNKNOWN"
        confidence = top_hypothesis.get("confidence", 0.0) if top_hypothesis else 0.0

        # ── LLM-based resolution (primary path) ───────────────────────────
        recommended_action = None
        action_risk_level = "MEDIUM"
        requires_approval = True
        resolution_notes = None

        try:
            evidence_summary = json.dumps(
                [{"source": e["source"], "summary": str(e["content"])[:400]} for e in evidence[:6]],
                indent=2,
            )
            policy_summary = json.dumps(
                [
                    {"ref": p.get("policy_ref"), "title": p.get("title"), "excerpt": p.get("content", "")[:500]}
                    for p in policies[:3]
                ],
                indent=2,
            )

            prompt_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a banking operations resolution specialist. "
                        "Given evidence and policies, propose the safest remediation. "
                        "Respond ONLY with valid JSON matching this schema:\n"
                        '{"recommended_action": str, "risk_level": "LOW|MEDIUM|HIGH|CRITICAL", '
                        '"requires_human_approval": bool, "resolution_notes": str, "confidence": float}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Case: {case_data.get('description', '')}\n"
                        f"Root cause identified: {root_cause} (confidence: {confidence:.0%})\n\n"
                        f"Evidence:\n{evidence_summary}\n\n"
                        f"Applicable policies:\n{policy_summary}\n\n"
                        "Propose the appropriate remediation action."
                    ),
                },
            ]

            response = await run_ctx.invoke_llm(prompt_messages, temperature=0.0)
            content = response.choices[0].message.content if hasattr(response, "choices") else ""

            # Parse JSON response
            parsed = json.loads(content)
            recommended_action = parsed.get("recommended_action")
            action_risk_level = parsed.get("risk_level", "MEDIUM")
            requires_approval = parsed.get("requires_human_approval", True)
            resolution_notes = parsed.get("resolution_notes")
            if parsed.get("confidence"):
                confidence = max(confidence, float(parsed["confidence"]))

            observations.append(f"LLM resolution: {recommended_action}")

        except Exception as exc:
            log.warning("resolution_llm_failed_using_rules", error=str(exc))
            errors.append(f"LLM resolution failed: {exc}")

            # Fall back to rule-based
            if root_cause in RESOLUTION_RULES:
                action, risk, approval, notes = RESOLUTION_RULES[root_cause]
                recommended_action = action
                action_risk_level = risk
                requires_approval = RISK_APPROVAL_MAP.get(risk, True)
                resolution_notes = notes
                observations.append(f"Rule-based resolution for '{root_cause}': {action[:100]}...")
            else:
                recommended_action = "Escalate to human operator — root cause undetermined."
                action_risk_level = "HIGH"
                requires_approval = True
                observations.append("Unknown root cause — escalating to human.")

        # ── Create case note ───────────────────────────────────────────────
        case_ref = state.get("case_ref", "")
        if case_ref and recommended_action:
            note_result = await run_ctx.invoke_tool(
                "create_case_note",
                {
                    "case_ref": case_ref,
                    "content": (
                        f"Root cause: {root_cause}\n"
                        f"Confidence: {confidence:.0%}\n"
                        f"Recommended action: {recommended_action}\n"
                        f"Risk level: {action_risk_level}\n"
                        f"Requires approval: {requires_approval}"
                    ),
                    "note_type": "RESOLUTION",
                },
            )
            if note_result.status == "SUCCESS":
                observations.append(f"Resolution note created: {note_result.data.get('note_id')}")

        log.info(
            "resolution_agent_complete",
            run_id=state.get("run_id"),
            root_cause=root_cause,
            risk_level=action_risk_level,
            requires_approval=requires_approval,
        )

        stop_reason = None
        if confidence >= 0.80:
            stop_reason = StopReason.CONFIDENCE_REACHED

        return {
            "loop_state": LoopState.EVALUATE,
            "root_cause": root_cause,
            "confidence": confidence,
            "recommended_action": recommended_action,
            "action_risk_level": action_risk_level,
            "requires_human_approval": requires_approval or RISK_APPROVAL_MAP.get(action_risk_level, True),
            "resolution_notes": resolution_notes,
            "observations": observations,
            "errors": errors,
            "stop_reason": stop_reason,
        }

    return resolution_agent_node
