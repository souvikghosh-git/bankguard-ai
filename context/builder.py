"""
BankGuard AI — Context Builder.

Builds a structured, token-budgeted context for each agent invocation
instead of dumping all data into the prompt.

Context contract:
    {
      system_instructions: str,
      case:                CaseSummary,
      customer_summary:    CustomerSummary,
      transaction:         TransactionDetails,
      related_transactions: [TransactionSummary, ...],
      policies:            [PolicySummary, ...],
      previous_cases:      [CaseSummary, ...],
      current_evidence:    [EvidenceItem, ...],
      agent_observations:  [str, ...],
      remaining_budget:    BudgetInfo,
    }

Token budget is enforced at each section.
PII is masked before anything reaches the LLM.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import structlog

from context.ranking import rank_policies, rank_transactions
from context.token_budget import TokenBudgetManager
from guardrails.pii.presidio_filter import mask_pii

log = structlog.get_logger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are BankGuard AI, a specialized banking operations agent.
Your role is to investigate payment issues, determine root causes, retrieve
applicable policies, and recommend or execute appropriate remediation actions.

CORE PRINCIPLES:
1. Base every conclusion on evidence — never speculate without data.
2. Follow bank policy — never bypass policy constraints.
3. Escalate when uncertain — do NOT guess on financial actions.
4. Flag high-risk actions — these require human approval before execution.
5. Be concise and factual — output structured reasoning.

INVESTIGATION APPROACH:
- Gather transaction evidence first
- Retrieve relevant payment policies
- Check account and customer status
- Correlate evidence to determine root cause
- Propose remediation with clear risk classification

PROHIBITED ACTIONS (require human approval):
- retry_payment, refund_fee (above threshold), block_card, freeze_account, reverse_transaction

When you need to take an action, call the appropriate tool.
When you have reached a conclusion, output your finding in the specified JSON format."""


# ── Data classes for context contract ────────────────────────────────────────


@dataclass
class CaseContext:
    case_ref: str
    title: str
    description: str
    status: str
    priority: str
    created_at: str


@dataclass
class CustomerContext:
    customer_ref: str
    full_name: str  # will be masked → "Customer C-1001"
    kyc_status: str
    risk_category: str
    account_count: int


@dataclass
class EvidenceItem:
    source: str  # tool that produced this
    content: Any
    timestamp: str | None = None
    relevance: float | None = None


@dataclass
class BudgetInfo:
    iterations_remaining: int
    tool_calls_remaining: int
    cost_remaining_usd: float


@dataclass
class ContextContract:
    """The full structured context handed to the agent."""

    system_instructions: str
    case: CaseContext
    customer_summary: CustomerContext | None
    transaction: dict | None
    related_transactions: list[dict]
    policies: list[dict]
    previous_cases: list[dict]
    current_evidence: list[EvidenceItem]
    agent_observations: list[str]
    remaining_budget: BudgetInfo
    token_usage: dict = field(default_factory=dict)

    def to_messages(self) -> list[dict[str, str]]:
        """Convert to LLM message list (system + user)."""
        user_content = json.dumps(self._to_payload(), indent=2, default=str)
        return [
            {"role": "system", "content": self.system_instructions},
            {"role": "user", "content": user_content},
        ]

    def _to_payload(self) -> dict:
        return {
            "case": asdict(self.case),
            "customer_summary": asdict(self.customer_summary) if self.customer_summary else None,
            "transaction": self.transaction,
            "related_transactions": self.related_transactions[:5],  # cap at 5
            "policies": self.policies[:3],  # cap at 3
            "previous_cases": self.previous_cases[:2],  # cap at 2
            "current_evidence": [asdict(e) for e in self.current_evidence],
            "agent_observations": self.agent_observations[-10:],  # last 10
            "remaining_budget": asdict(self.remaining_budget),
        }


# ── Context Builder ───────────────────────────────────────────────────────────


class ContextBuilder:
    """
    Builds a ContextContract from raw banking data.

    Usage:
        builder = ContextBuilder(token_budget=8000)
        context = await builder.build(
            case_data=...,
            customer_data=...,
            transaction_data=...,
            observations=[...],
            budget_snapshot=...,
        )
        messages = context.to_messages()
    """

    def __init__(self, token_budget: int = 8_000, mask_pii_enabled: bool = True) -> None:
        self.token_mgr = TokenBudgetManager(total_budget=token_budget)
        self.mask_pii_enabled = mask_pii_enabled

    async def build(
        self,
        case_data: dict[str, Any],
        customer_data: dict[str, Any] | None = None,
        transaction_data: dict[str, Any] | None = None,
        related_transactions: list[dict] | None = None,
        raw_policies: list[dict] | None = None,
        previous_cases: list[dict] | None = None,
        current_evidence: list[EvidenceItem] | None = None,
        observations: list[str] | None = None,
        budget_snapshot: Any = None,
    ) -> ContextContract:
        """
        Build a token-budgeted context contract.
        Sections are added in priority order; lower-priority sections are
        truncated if the token budget is exhausted.
        """
        self.token_mgr.reset()

        # 1. System prompt (always included, high priority)
        system_instructions = self.token_mgr.allocate("system", SYSTEM_PROMPT, priority=10)

        # 2. Case (always included)
        case_ctx = CaseContext(
            case_ref=case_data.get("case_ref", ""),
            title=case_data.get("title", ""),
            description=case_data.get("description", ""),
            status=case_data.get("status", ""),
            priority=case_data.get("priority", "MEDIUM"),
            created_at=str(case_data.get("created_at", "")),
        )
        self.token_mgr.allocate("case", json.dumps(asdict(case_ctx)), priority=9)

        # 3. Customer summary (PII-masked)
        customer_ctx: CustomerContext | None = None
        if customer_data:
            full_name = customer_data.get("full_name", "Customer")
            if self.mask_pii_enabled:
                full_name = await mask_pii(full_name) or f"Customer {customer_data.get('customer_ref', '')}"
            customer_ctx = CustomerContext(
                customer_ref=customer_data.get("customer_ref", ""),
                full_name=full_name,
                kyc_status=customer_data.get("kyc_status", ""),
                risk_category=customer_data.get("risk_category", ""),
                account_count=len(customer_data.get("accounts", [])),
            )
            self.token_mgr.allocate("customer", json.dumps(asdict(customer_ctx)), priority=8)

        # 4. Transaction details
        txn: dict | None = None
        if transaction_data:
            txn = self._trim_transaction(transaction_data)
            self.token_mgr.allocate("transaction", json.dumps(txn, default=str), priority=9)

        # 5. Related transactions (ranked by relevance)
        ranked_related: list[dict] = []
        if related_transactions:
            ranked = rank_transactions(
                related_transactions,
                reference_amount=transaction_data.get("amount") if transaction_data else None,
            )
            for rt in ranked[:5]:
                trimmed = {k: rt[k] for k in ("transaction_ref", "amount", "status", "initiated_at") if k in rt}
                token_ok = self.token_mgr.allocate(
                    f"related_txn_{rt.get('transaction_ref', '')}", json.dumps(trimmed, default=str), priority=6
                )
                if token_ok:
                    ranked_related.append(trimmed)

        # 6. Policies (ranked by relevance to case description)
        selected_policies: list[dict] = []
        if raw_policies:
            ranked_policies = rank_policies(
                raw_policies,
                query=case_data.get("description", ""),
            )
            for p in ranked_policies[:3]:
                # Truncate policy content to save tokens
                trimmed = {
                    "policy_ref": p.get("policy_ref"),
                    "title": p.get("title"),
                    "category": p.get("category"),
                    "content": p.get("content", "")[:1500],  # cap policy content
                }
                token_ok = self.token_mgr.allocate(f"policy_{p.get('policy_ref', '')}", json.dumps(trimmed), priority=7)
                if token_ok:
                    selected_policies.append(trimmed)

        # 7. Previous cases (low priority — most useful for patterns)
        prev_cases_trimmed: list[dict] = []
        if previous_cases:
            for pc in previous_cases[:2]:
                mini = {
                    "case_ref": pc.get("case_ref"),
                    "root_cause": pc.get("root_cause"),
                    "resolution": pc.get("resolution"),
                    "status": pc.get("status"),
                }
                token_ok = self.token_mgr.allocate(f"prev_case_{pc.get('case_ref', '')}", json.dumps(mini), priority=4)
                if token_ok:
                    prev_cases_trimmed.append(mini)

        # 8. Current evidence (from tool call results)
        evidence_items = current_evidence or []

        # 9. Agent observations
        obs = (observations or [])[-10:]
        obs_text = "\n".join(obs)
        self.token_mgr.allocate("observations", obs_text, priority=5)

        # 10. Budget remaining
        if budget_snapshot:
            budget_info = BudgetInfo(
                iterations_remaining=budget_snapshot.iterations_remaining,
                tool_calls_remaining=budget_snapshot.tool_calls_remaining,
                cost_remaining_usd=budget_snapshot.cost_remaining_usd,
            )
        else:
            budget_info = BudgetInfo(
                iterations_remaining=8,
                tool_calls_remaining=15,
                cost_remaining_usd=0.15,
            )

        ctx = ContextContract(
            system_instructions=system_instructions,
            case=case_ctx,
            customer_summary=customer_ctx,
            transaction=txn,
            related_transactions=ranked_related,
            policies=selected_policies,
            previous_cases=prev_cases_trimmed,
            current_evidence=evidence_items,
            agent_observations=obs,
            remaining_budget=budget_info,
            token_usage=self.token_mgr.usage_report(),
        )

        log.debug(
            "context_built",
            case_ref=case_data.get("case_ref"),
            total_tokens=self.token_mgr.total_used,
            sections=list(self.token_mgr.section_tokens.keys()),
        )
        return ctx

    @staticmethod
    def _trim_transaction(txn: dict) -> dict:
        """Keep only the fields the agent needs; drop raw internals."""
        keep = {
            "transaction_ref",
            "debit_account_ref",
            "amount",
            "currency",
            "transaction_type",
            "status",
            "payment_rail",
            "reference_number",
            "description",
            "initiated_at",
            "completed_at",
            "beneficiary_bank",
            "beneficiary_ifsc",
            "last_event_code",
        }
        trimmed = {k: v for k, v in txn.items() if k in keep}
        # Keep only last 5 events to save tokens
        if "events" in txn:
            trimmed["events"] = txn["events"][-5:]
        return trimmed
