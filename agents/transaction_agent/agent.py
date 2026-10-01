"""
BankGuard AI — Transaction Investigator Agent.

Responsibilities:
  - Retrieve transaction details + payment event timeline
  - Retrieve customer profile and account status
  - Fetch recent related transactions
  - Identify patterns: duplicate, timeout, rail failure, etc.
  - Populate evidence list in AgentState

This node uses the RunContext (harness) for all tool calls,
so budget + permission enforcement is automatic.
"""

from __future__ import annotations

from typing import Any

import structlog

from agents.state import AgentState, LoopState, StopReason

log = structlog.get_logger(__name__)

# Root-cause detection heuristics (event_code → root_cause, confidence)
_EVENT_HEURISTICS: dict[str, tuple[str, float]] = {
    "BENEFICIARY_BANK_TIMEOUT": ("BENEFICIARY_BANK_TIMEOUT", 0.85),
    "PENDING_RECONCILIATION": ("BENEFICIARY_BANK_TIMEOUT", 0.85),  # same root cause
    "DUPLICATE_CHECK_FAILED": ("DUPLICATE_TRANSACTION", 0.92),
    "RAIL_UNAVAILABLE": ("PAYMENT_RAIL_UNAVAILABLE", 0.88),
    "FALLBACK_FAILED": ("PAYMENT_RAIL_UNAVAILABLE", 0.85),
    "INSUFFICIENT_FUNDS": ("INSUFFICIENT_FUNDS", 0.95),
    "DEBIT_FAILED": ("DEBIT_FAILURE", 0.90),
    "CREDIT_CONFIRMED": ("PAYMENT_COMPLETED_OK", 0.98),
    "PAYMENT_COMPLETED": ("PAYMENT_COMPLETED_OK", 0.98),
    "PAYMENT_REJECTED": ("PAYMENT_REJECTED", 0.80),
    "PAYMENT_FAILED": ("PAYMENT_FAILURE", 0.82),
}


def make_transaction_agent(run_ctx: Any) -> Any:
    """
    Factory: returns a LangGraph node function bound to a RunContext.
    The run_ctx gives access to invoke_tool() through the harness.
    """

    async def transaction_agent_node(state: AgentState) -> dict[str, Any]:
        log.info(
            "transaction_agent_start",
            run_id=state.get("run_id"),
            case_ref=state.get("case_ref"),
            iteration=state.get("iteration"),
        )

        evidence: list[dict] = []
        observations: list[str] = []
        hypotheses: list[dict] = []
        errors: list[str] = []

        case_data = state.get("case_data", {})
        transaction_ref = case_data.get("transaction_ref") or state.get("case_data", {}).get("transaction_ref")
        customer_ref = case_data.get("customer_ref")
        account_ref = None

        # If neither transaction_ref nor customer_ref is linked, we cannot investigate.
        # Signal NO_NEW_EVIDENCE immediately rather than looping.
        if not transaction_ref and not customer_ref:
            observations.append(
                "[TransactionAgent] No transaction_ref or customer_ref linked to this case. "
                "Cannot gather evidence. Please re-create the case with a valid Transaction Ref or Customer Ref."
            )
            errors.append("NO_REFS: Case has no transaction_ref or customer_ref — investigation cannot proceed.")
            return {
                "loop_state": LoopState.OBSERVE,
                "evidence": [],
                "observations": observations,
                "hypotheses": [],
                "errors": errors,
                "stop_reason": StopReason.NO_NEW_EVIDENCE,
            }

        # ── 1. Get transaction details ─────────────────────────────────────
        if transaction_ref:
            txn_result = await run_ctx.invoke_tool(
                "get_transaction_details",
                {"transaction_ref": transaction_ref, "include_events": True},
            )
            if txn_result.status == "SUCCESS":
                txn_data = txn_result.data
                evidence.append({"source": "get_transaction_details", "content": txn_data})
                observations.append(
                    f"Transaction {transaction_ref}: status={txn_data.get('status')}, "
                    f"amount=₹{txn_data.get('amount', 0):,.2f}, "
                    f"rail={txn_data.get('payment_rail')}, "
                    f"last_event={txn_data.get('last_event_code')}"
                )
                account_ref = txn_data.get("debit_account_ref")

                # Apply event heuristics
                last_event = txn_data.get("last_event_code", "")
                if last_event in _EVENT_HEURISTICS:
                    root_cause, conf = _EVENT_HEURISTICS[last_event]
                    hypotheses.append(
                        {
                            "root_cause": root_cause,
                            "confidence": conf,
                            "evidence": [f"last_event={last_event}"],
                            "source": "transaction_heuristic",
                        }
                    )
                    observations.append(
                        f"Heuristic match: last event '{last_event}' → "
                        f"root cause '{root_cause}' (confidence {conf:.0%})"
                    )
            else:
                err = f"get_transaction_details failed: {txn_result.error_code} — {txn_result.error_message}"
                errors.append(err)
                observations.append(err)

            # ── 2. Get payment status ──────────────────────────────────────
            status_result = await run_ctx.invoke_tool(
                "get_payment_status",
                {"transaction_ref": transaction_ref},
            )
            if status_result.status == "SUCCESS":
                evidence.append({"source": "get_payment_status", "content": status_result.data})
                if status_result.data.get("is_pending_reconciliation"):
                    observations.append(
                        "IMPORTANT: Transaction is in PENDING_RECONCILIATION state. "
                        "Do NOT initiate another payment — wait for reconciliation window."
                    )

        # ── 3. Get customer profile ────────────────────────────────────────
        if customer_ref:
            cust_result = await run_ctx.invoke_tool(
                "get_customer_profile",
                {"customer_ref": customer_ref, "include_accounts": True},
            )
            if cust_result.status == "SUCCESS":
                cust_data = cust_result.data
                evidence.append({"source": "get_customer_profile", "content": cust_data})
                observations.append(
                    f"Customer {customer_ref}: KYC={cust_data.get('kyc_status')}, "
                    f"risk={cust_data.get('risk_category')}, "
                    f"accounts={len(cust_data.get('accounts', []))}"
                )
                # Get primary account ref if not already known
                if not account_ref and cust_data.get("accounts"):
                    account_ref = cust_data["accounts"][0]["account_ref"]

        # ── 4. Get account status ──────────────────────────────────────────
        if account_ref:
            acc_result = await run_ctx.invoke_tool(
                "get_account_summary",
                {"account_ref": account_ref},
            )
            if acc_result.status == "SUCCESS":
                acc_data = acc_result.data
                evidence.append({"source": "get_account_summary", "content": acc_data})
                if acc_data.get("status") == "FROZEN":
                    observations.append(
                        f"ALERT: Account {account_ref} is FROZEN. This may be the root cause of the payment failure."
                    )
                    hypotheses.append(
                        {
                            "root_cause": "ACCOUNT_FROZEN",
                            "confidence": 0.90,
                            "evidence": ["account_status=FROZEN"],
                            "source": "account_check",
                        }
                    )

            # ── 5. Recent transactions ─────────────────────────────────────
            recent_result = await run_ctx.invoke_tool(
                "get_recent_transactions",
                {"account_ref": account_ref, "limit": 10, "days_back": 7},
            )
            if recent_result.status == "SUCCESS":
                recent_txns = recent_result.data.get("transactions", [])
                evidence.append({"source": "get_recent_transactions", "content": recent_txns})
                pending_count = sum(1 for t in recent_txns if t["status"] == "PENDING")
                observations.append(f"Recent transactions (7 days): {len(recent_txns)} total, {pending_count} pending")

        updates: dict[str, Any] = {
            "loop_state": LoopState.OBSERVE,
            "transaction_data": evidence[0]["content"] if evidence else None,
            "evidence": evidence,
            "observations": observations,
            "hypotheses": hypotheses if hypotheses else state.get("hypotheses", []),
            "errors": errors,
        }
        if account_ref:
            updates["case_data"] = {**state.get("case_data", {}), "account_ref": account_ref}

        log.info(
            "transaction_agent_complete",
            run_id=state.get("run_id"),
            evidence_items=len(evidence),
            hypotheses=len(hypotheses),
        )
        return updates

    return transaction_agent_node
