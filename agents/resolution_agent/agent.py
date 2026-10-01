"""
BankGuard AI — Resolution Agent.

Resolution strategy (in priority order):
  1. Rule-based table  — authoritative for all known root causes.
     The table encodes bank policy; LLM cannot override it.
  2. LLM               — only called when root_cause == "UNKNOWN" and
     enough evidence exists to reason from.
  3. Default escalation — when both above fail.

This prevents the LLM from hallucinating incorrect actions (e.g.
suggesting a reversal for a reconciliation-window timeout).
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from agents.state import AgentState, LoopState, StopReason

log = structlog.get_logger(__name__)

# ── Rule-based resolution table ───────────────────────────────────────────────
# root_cause → (action, risk_level, requires_approval, policy_note)
# These are AUTHORITATIVE — the LLM cannot override them.

RESOLUTION_RULES: dict[str, tuple[str, str, bool, str]] = {
    "BENEFICIARY_BANK_TIMEOUT": (
        "Wait for reconciliation window (24–48h). Create an ops ticket to monitor. "
        "Do NOT retry the payment — per PAY-REC-101 initiating another transfer during "
        "the reconciliation window risks a duplicate debit.",
        "LOW",
        False,
        "Per PAY-REC-101: never retry during reconciliation window.",
    ),
    "DUPLICATE_TRANSACTION": (
        "A duplicate payment was detected. Place a hold on the second transaction and notify the customer. "
        "Requires Risk Officer confirmation before any cancellation.",
        "MEDIUM",
        True,
        "Requires Risk Officer confirmation before cancellation.",
    ),
    "PAYMENT_RAIL_UNAVAILABLE": (
        "Payment rail is currently unavailable. Check NPCI / RBI status page. "
        "Notify the customer of the delay. Retry automatically when rail is restored — "
        "no manual retry needed.",
        "LOW",
        False,
        "Rail outage — retry when operational.",
    ),
    "INSUFFICIENT_FUNDS": (
        "Payment failed due to insufficient funds — this is a customer-side issue. "
        "Notify the customer of the failure reason. No bank action is required.",
        "LOW",
        False,
        "Customer-side issue. No financial action needed.",
    ),
    "DEBIT_FAILURE": (
        "Debit failed — investigate the account status for freezes, holds, or compliance flags. "
        "Do not retry until the root cause is confirmed.",
        "MEDIUM",
        False,
        "May indicate account freeze or compliance hold.",
    ),
    "ACCOUNT_FROZEN": (
        "Account is frozen. Escalate to the compliance officer immediately. "
        "Do NOT unfreeze the account without explicit authorization — per OPS-FRZ-501.",
        "HIGH",
        True,
        "Per OPS-FRZ-501: unfreeze always requires human approval.",
    ),
    "PAYMENT_COMPLETED_OK": (
        "Payment completed successfully. No further action is required. "
        "Provide the transaction confirmation to the customer and close the case.",
        "LOW",
        False,
        "Transaction confirmed complete. Case can be closed.",
    ),
    "PAYMENT_REJECTED": (
        "Payment was rejected. Review the rejection reason code. "
        "If caused by a system error, a retry may be applicable with Risk Officer approval. "
        "If caused by customer data error, notify the customer.",
        "MEDIUM",
        True,
        "Retry requires Risk Officer approval per PAY-REV-601.",
    ),
    "PAYMENT_FAILURE": (
        "Payment failed. Check the payment event log for a specific failure code. "
        "Notify the customer. Escalate to ops if no clear failure reason.",
        "MEDIUM",
        False,
        "Investigate specific failure code before any action.",
    ),
    "TRANSACTION_LIMIT_EXCEEDED": (
        "Transaction exceeds the allowed limit for this payment rail. "
        "Notify the customer of the applicable limit. The customer may apply for a "
        "temporary limit enhancement with Risk Officer approval.",
        "LOW",
        False,
        "Per PAY-LIM-301: limits apply per rail per day.",
    ),
    "AML_TRIGGERED": (
        "AML monitoring has flagged this transaction. Escalate to the Compliance Officer "
        "within 4 hours. Do NOT notify the customer. Do NOT release any holds.",
        "CRITICAL",
        True,
        "Per COMP-AML-401: mandatory escalation, no customer notification.",
    ),
    "SUSPICIOUS_TRANSACTION_PATTERN": (
        "Suspicious transaction pattern detected. Create an investigation note and escalate "
        "to the compliance team. Do NOT freeze the account without authorization.",
        "HIGH",
        True,
        "Per COMP-AML-401 and OPS-FRZ-501.",
    ),
    "WRONG_BENEFICIARY_CREDITED": (
        "Wrong beneficiary was credited. Create an urgent ops ticket for reversal review. "
        "A reversal requires Risk Officer + Branch Manager approval per PAY-REV-601. "
        "Do NOT initiate a reversal autonomously.",
        "CRITICAL",
        True,
        "Per PAY-REV-601: reversal always requires dual human approval.",
    ),
}

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

        # Top hypothesis
        top = max(hypotheses, key=lambda h: h.get("confidence", 0), default=None)
        root_cause = top["root_cause"] if top else "UNKNOWN"
        confidence = top.get("confidence", 0.0) if top else 0.0

        recommended_action: str | None = None
        action_risk_level = "MEDIUM"
        requires_approval = True
        resolution_notes: str | None = None

        # ── Strategy 1: Rule-based (authoritative for known root causes) ──
        if root_cause in RESOLUTION_RULES:
            action, risk, approval, notes = RESOLUTION_RULES[root_cause]
            recommended_action = action
            action_risk_level = risk
            requires_approval = RISK_APPROVAL_MAP.get(risk, True)
            resolution_notes = notes
            observations.append(f"[Resolution] Rule-based answer for '{root_cause}': {action[:120]}...")
            log.info(
                "resolution_rule_applied",
                root_cause=root_cause,
                risk=risk,
                requires_approval=requires_approval,
            )

        # ── Strategy 2: LLM — only for UNKNOWN root cause with evidence ───
        elif root_cause == "UNKNOWN" and len(evidence) >= 2:
            try:
                evidence_summary = json.dumps(
                    [{"source": e["source"], "summary": str(e["content"])[:300]} for e in evidence[:4]],
                    indent=2,
                )
                policy_summary = json.dumps(
                    [
                        {
                            "ref": p.get("policy_ref"),
                            "title": p.get("title"),
                            "excerpt": p.get("content", "")[:400],
                        }
                        for p in policies[:2]
                    ],
                    indent=2,
                )

                prompt_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a conservative banking operations specialist. "
                            "The root cause of this case is UNKNOWN. "
                            "Based on the evidence, suggest ONLY safe, low-risk actions. "
                            "NEVER suggest: reversal, freeze, retry payment, or block card. "
                            "Prefer: escalation, investigation ticket, notification draft. "
                            "Respond ONLY with valid JSON:\n"
                            '{"recommended_action": str, "risk_level": "LOW|MEDIUM", '
                            '"requires_human_approval": bool, "resolution_notes": str}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Case description: {case_data.get('description', '')}\n\n"
                            f"Evidence gathered:\n{evidence_summary}\n\n"
                            f"Applicable policies:\n{policy_summary}\n\n"
                            "Root cause is unclear. What is the safest next step?"
                        ),
                    },
                ]

                response = await run_ctx.invoke_llm(prompt_messages, temperature=0.0)
                raw = response.choices[0].message.content if hasattr(response, "choices") else ""

                # Strip markdown code fences if present
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()

                parsed = json.loads(raw)
                recommended_action = parsed.get("recommended_action")
                # Cap LLM risk at MEDIUM — it cannot recommend HIGH/CRITICAL actions
                llm_risk = parsed.get("risk_level", "MEDIUM")
                action_risk_level = llm_risk if llm_risk in ("LOW", "MEDIUM") else "MEDIUM"
                requires_approval = parsed.get("requires_human_approval", True)
                resolution_notes = parsed.get("resolution_notes", "")
                observations.append(f"[Resolution] LLM (unknown root cause): {recommended_action}")

            except Exception as exc:
                log.warning("resolution_llm_failed", error=str(exc))
                errors.append(f"LLM resolution failed: {exc}")
                recommended_action = (
                    "Root cause could not be determined. "
                    "Create an investigation ticket and escalate to a senior analyst."
                )
                action_risk_level = "MEDIUM"
                requires_approval = True

        # ── Strategy 3: Default escalation ────────────────────────────────
        else:
            recommended_action = (
                "Root cause undetermined — insufficient evidence. "
                "Escalate to a senior analyst for manual investigation."
            )
            action_risk_level = "MEDIUM"
            requires_approval = True
            observations.append("[Resolution] Insufficient evidence — escalating to human.")

        # ── Create case note (final resolution only) ──────────────────────
        case_ref = state.get("case_ref", "")
        is_final = confidence >= 0.70 or root_cause in RESOLUTION_RULES or state.get("stop_reason") is not None
        if case_ref and recommended_action and is_final:
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
            strategy="rule" if root_cause in RESOLUTION_RULES else "llm_or_default",
        )

        stop_reason = StopReason.CONFIDENCE_REACHED if confidence >= 0.80 else None

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
