# BankGuard AI — OPA Policy: Tool Access Control
# Package: bankguard.tool_access
#
# Evaluated by PermissionEngine._check_opa()
# Input: {tool, role, user_id, context}

package bankguard.tool_access

import future.keywords.if
import future.keywords.in

# Default: deny
default allow := false

# ── READ tools — allowed for all recognised roles ─────────────────────────────

read_tools := {
    "get_transaction_details",
    "get_account_summary",
    "get_customer_profile",
    "get_recent_transactions",
    "get_payment_status",
    "search_policy",
    "get_case_history",
    "check_transaction_limit",
    "get_payment_rail_status",
}

known_roles := {"AGENT", "OPERATIONS_ANALYST", "SENIOR_ANALYST", "RISK_OFFICER", "COMPLIANCE", "ADMIN"}

allow if {
    input.tool in read_tools
    input.role in known_roles
}

# ── Low-risk writes ───────────────────────────────────────────────────────────

low_risk_write_tools := {
    "create_case_note",
    "create_operations_ticket",
    "draft_customer_notification",
}

low_risk_roles := {"OPERATIONS_ANALYST", "SENIOR_ANALYST", "RISK_OFFICER", "ADMIN"}

allow if {
    input.tool in low_risk_write_tools
    input.role in low_risk_roles
}

# Agents can also create notes and tickets
allow if {
    input.tool in low_risk_write_tools
    input.role == "AGENT"
}

# ── Medium-risk: retry_payment, refund_fee ────────────────────────────────────

medium_risk_roles := {"OPERATIONS_ANALYST", "SENIOR_ANALYST", "RISK_OFFICER", "ADMIN"}

allow if {
    input.tool in {"retry_payment", "refund_fee"}
    input.role in medium_risk_roles
}

# refund_fee threshold: amount > 500 requires SENIOR_ANALYST+
deny_reason["refund_fee_threshold"] if {
    input.tool == "refund_fee"
    input.role == "OPERATIONS_ANALYST"
    to_number(input.context.fee_amount) > 500
}

# ── High-risk: block_card ─────────────────────────────────────────────────────

allow if {
    input.tool == "block_card"
    input.role in {"SENIOR_ANALYST", "RISK_OFFICER", "ADMIN"}
}

# ── Critical-risk: freeze_account, reverse_transaction ───────────────────────

allow if {
    input.tool in {"freeze_account", "reverse_transaction"}
    input.role in {"RISK_OFFICER", "ADMIN"}
}

# ── ADMIN bypass ─────────────────────────────────────────────────────────────

allow if {
    input.role == "ADMIN"
}

# ── Explicit denials override allows ─────────────────────────────────────────

final_allow := allow if {
    count(deny_reason) == 0
}

final_allow := false if {
    count(deny_reason) > 0
}
