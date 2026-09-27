"""
BankGuard AI — Operations Portal (Streamlit).

Pages:
  🏠 Home          → active cases overview + quick stats
  🔍 Investigate   → launch + stream a live investigation
  📋 Cases         → case list with filters
  ✅ Approvals     → pending human approvals queue
  📊 Observability → cost, latency, tool call metrics
  🧪 Evaluations   → run eval suite + view results

Run: streamlit run apps/operations-ui/app.py
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

import httpx
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

TOKENS = {
    "Operations Analyst": "dev-analyst-token",
    "Risk Officer": "dev-risk-token",
    "Admin": "dev-admin-token",
}

RISK_COLOURS = {
    "LOW": "🟢",
    "MEDIUM": "🟡",
    "HIGH": "🔴",
    "CRITICAL": "🚨",
}

STATUS_ICONS = {
    "OPEN": "📂",
    "INVESTIGATING": "🔍",
    "PENDING_APPROVAL": "⏳",
    "RESOLVED": "✅",
    "CLOSED": "🗄️",
    "ESCALATED": "🚨",
}

# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BankGuard AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏦 BankGuard AI")
    st.caption("Agentic Banking Operations Platform")
    st.divider()

    role_label = st.selectbox("Identity", list(TOKENS.keys()), index=0)
    token = TOKENS[role_label]

    st.divider()
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Investigate", "📋 Cases", "✅ Approvals", "📊 Observability", "🧪 Evaluations"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"API: `{API_BASE}`")
    if st.button("♻️ Refresh"):
        st.rerun()


# ── Helpers ─────────────────────────────────────────────────────────────────────


def api_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def api_get(path: str, params: dict | None = None) -> Any:
    try:
        with httpx.Client(timeout=10) as client:
            r = client.get(f"{API_BASE}{path}", headers=api_headers(), params=params)
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        st.error("⚠️ Cannot connect to API. Is the backend running? `uvicorn apps.api.main:app`")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def api_post(path: str, body: dict) -> Any:
    try:
        with httpx.Client(timeout=30) as client:
            r = client.post(f"{API_BASE}{path}", json=body, headers=api_headers())
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def fmt_confidence(c: float | None) -> str:
    if c is None:
        return "—"
    return f"{c:.0%}"


def fmt_risk(level: str | None) -> str:
    if not level:
        return "—"
    return f"{RISK_COLOURS.get(level, '⚪')} {level}"


def _render_case_result(case: dict) -> None:
    """Render a formatted investigation result card."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Investigation Result")
        status_icon = STATUS_ICONS.get(case.get("status", ""), "❓")
        st.markdown(f"**Status:** {status_icon} {case.get('status', '—')}")
        st.markdown(f"**Root Cause:** `{case.get('root_cause') or 'Undetermined'}`")
        st.markdown(f"**Confidence:** {fmt_confidence(case.get('confidence'))}")
    with col2:
        st.markdown("### 🔧 Recommended Action")
        action = case.get("resolution", "No action determined.")
        risk = case.get("action_risk_level")
        if action:
            if case.get("requires_human_approval"):
                st.warning(f"⏳ **REQUIRES APPROVAL**\n\n{action}")
            else:
                st.success(action)
            if risk:
                st.caption(f"Risk level: {fmt_risk(risk)}")


# ══════════════════════════════════════════════════════════════════════════════
# Page functions — each page is a standalone function to satisfy ruff syntax
# ══════════════════════════════════════════════════════════════════════════════


def page_home() -> None:
    st.title("🏦 BankGuard AI — Operations Centre")
    st.caption(f"Logged in as **{role_label}** · {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")

    cases = api_get("/api/cases", {"limit": 100}) or []

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📂 Open", sum(1 for c in cases if c["status"] == "OPEN"))
    col2.metric("🔍 Investigating", sum(1 for c in cases if c["status"] == "INVESTIGATING"))
    col3.metric("⏳ Pending Approval", sum(1 for c in cases if c["status"] == "PENDING_APPROVAL"))
    col4.metric("✅ Resolved", sum(1 for c in cases if c["status"] == "RESOLVED"))
    col5.metric("🚨 Escalated", sum(1 for c in cases if c["status"] == "ESCALATED"))

    st.divider()
    st.subheader("Recent Cases")
    if cases:
        for c in cases[:10]:
            icon = STATUS_ICONS.get(c["status"], "❓")
            with st.expander(f"{icon} {c['case_ref']} — {c['title'][:60]}  |  Priority: {c['priority']}"):
                ca, cb, cc = st.columns(3)
                ca.write(f"**Status:** {c['status']}")
                cb.write(f"**Root Cause:** {c.get('root_cause') or '—'}")
                cc.write(f"**Confidence:** {fmt_confidence(c.get('confidence'))}")
                if c.get("resolution"):
                    st.info(f"**Resolution:** {c['resolution']}")
    else:
        st.info("No cases yet. Create one via the Investigate tab.")

    approvals = api_get("/api/approvals") or []
    if approvals:
        st.warning(f"⚠️ **{len(approvals)} pending approval(s)** requiring human decision. Go to ✅ Approvals tab.")


def page_investigate() -> None:
    st.title("🔍 Case Investigation")
    st.caption("Create a case and launch an AI investigation with real-time trace.")

    st.subheader("New Investigation")
    with st.form("new_case_form"):
        col1, col2 = st.columns([2, 1])
        title = col1.text_input("Case Title", value="IMPS payment debited, beneficiary not credited")
        priority = col2.selectbox("Priority", ["LOW", "MEDIUM", "HIGH", "CRITICAL"], index=1)
        description = st.text_area(
            "Description",
            value=(
                "Customer reports ₹25,000 was debited from account but beneficiary "
                "did not receive the payment. Transaction initiated via IMPS."
            ),
            height=100,
        )
        col3, col4 = st.columns(2)
        customer_ref = col3.text_input("Customer Ref (optional)", placeholder="CUST-10001")
        transaction_ref = col4.text_input("Transaction Ref (optional)", placeholder="TXN-30001")
        submitted = st.form_submit_button("🚀 Create & Investigate", type="primary")

    if submitted:
        with st.spinner("Creating case..."):
            body: dict[str, Any] = {"title": title, "description": description, "priority": priority}
            if customer_ref:
                body["customer_ref"] = customer_ref.strip()
            if transaction_ref:
                body["transaction_ref"] = transaction_ref.strip()
            case_resp = api_post("/api/cases", body)

        if case_resp:
            st.success(f"✅ Case created: **{case_resp['case_ref']}**")
            st.session_state["active_case_ref"] = case_resp["case_ref"]

    active_case = st.session_state.get("active_case_ref")
    if active_case:
        st.divider()
        st.subheader(f"Investigation Trace — {active_case}")

        if st.button("▶️ Start Investigation", type="primary"):
            with st.spinner("Launching agent..."):
                inv_resp = api_post(f"/api/investigate/{active_case}", {})

            if inv_resp:
                run_id = inv_resp.get("run_id", "")
                st.info(f"Run ID: `{run_id}`")
                progress_bar = st.progress(0)

                for i in range(60):
                    time.sleep(2)
                    status_data = api_get(f"/api/investigate/{active_case}/status", {"run_id": run_id})
                    if not status_data:
                        break
                    run_status = status_data.get("status", "RUNNING")
                    progress_bar.progress(min(0.95, (i + 1) / 60))
                    if run_status == "COMPLETED":
                        progress_bar.progress(1.0)
                        st.success("✅ Investigation complete!")
                        break
                    if run_status == "FAILED":
                        st.error(f"❌ Investigation failed: {status_data.get('error')}")
                        break

                final_case = api_get(f"/api/cases/{active_case}")
                if final_case:
                    st.divider()
                    _render_case_result(final_case)

        notes = api_get(f"/api/cases/{active_case}/notes") or []
        if notes:
            st.divider()
            st.subheader("Investigation Notes")
            for note in notes:
                note_icon = {
                    "RESOLUTION": "🔧",
                    "INVESTIGATION": "🔍",
                    "ACTION": "⚡",
                    "ESCALATION": "🚨",
                }.get(note.get("note_type", ""), "📝")
                with st.expander(f"{note_icon} {note.get('note_type')} — {note.get('created_at', '')[:19]}"):
                    st.markdown(note.get("content", ""))


def page_cases() -> None:
    st.title("📋 Case Management")

    col1, col2, col3 = st.columns([2, 1, 1])
    search = col1.text_input("Search (title)", placeholder="Type to filter...")
    status_f = col2.selectbox("Status", ["All", "OPEN", "INVESTIGATING", "PENDING_APPROVAL", "RESOLVED", "ESCALATED"])
    priority_f = col3.selectbox("Priority", ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"])

    cases = (
        api_get(
            "/api/cases",
            {"status_filter": status_f if status_f != "All" else None, "limit": 50},
        )
        or []
    )

    if search:
        cases = [c for c in cases if search.lower() in c["title"].lower()]
    if priority_f != "All":
        cases = [c for c in cases if c["priority"] == priority_f]

    st.caption(f"{len(cases)} case(s) found")

    for c in cases:
        icon = STATUS_ICONS.get(c["status"], "❓")
        with st.container():
            cols = st.columns([3, 1, 1, 1, 1])
            cols[0].markdown(f"**{icon} {c['case_ref']}** — {c['title'][:55]}")
            cols[1].caption(c.get("status", ""))
            cols[2].caption(c.get("priority", ""))
            cols[3].caption(fmt_confidence(c.get("confidence")))
            if cols[4].button("View", key=f"view_{c['case_ref']}"):
                st.session_state["active_case_ref"] = c["case_ref"]
                st.info(f"Case {c['case_ref']} selected. Switch to 🔍 Investigate tab.")
        st.divider()


def page_approvals() -> None:
    st.title("✅ Human Approval Queue")
    st.caption("Review and approve/reject agent-proposed actions that require human oversight.")

    approvals = api_get("/api/approvals") or []

    if not approvals:
        st.success("✅ No pending approvals. All clear.")
        return

    st.warning(f"**{len(approvals)} pending approval(s)** require your decision.")
    st.divider()

    for apr in approvals:
        risk = apr.get("risk_level", "MEDIUM")
        icon = RISK_COLOURS.get(risk, "⚪")
        action = apr.get("action_type", "Unknown action")
        ref = apr.get("approval_ref", "")
        case_r = apr.get("case_ref", "—")

        with st.expander(f"{icon} {ref}  |  Case: {case_r}  |  Action: {action}  |  Risk: {risk}"):
            full = api_get(f"/api/approvals/{ref}")
            if full:
                st.markdown(f"**Action Type:** `{full.get('action_type')}`")
                st.markdown(f"**Risk Level:** {fmt_risk(full.get('risk_level'))}")
                st.markdown(f"**Requested by:** {full.get('requested_by', '—')}")
                st.markdown(f"**Requested at:** {full.get('requested_at', '')[:19]}")
                st.markdown(f"**Expires at:** {full.get('expires_at', '')[:19]}")
                payload = full.get("action_payload", {})
                if payload:
                    st.markdown("**Action payload:**")
                    st.json(payload)

            st.divider()
            col_a, col_b, col_c, col_d = st.columns(4)
            notes = st.text_input("Decision notes", key=f"notes_{ref}", placeholder="Optional notes...")

            if col_a.button("✅ Approve", key=f"apr_{ref}", type="primary"):
                if api_post(f"/api/approvals/{ref}/approve", {"notes": notes}):
                    st.success(f"Approved: {ref}")
                    st.rerun()

            if col_b.button("❌ Reject", key=f"rej_{ref}"):
                if api_post(f"/api/approvals/{ref}/reject", {"notes": notes}):
                    st.warning(f"Rejected: {ref}")
                    st.rerun()

            if col_c.button("⬆️ Escalate", key=f"esc_{ref}"):
                if api_post(f"/api/approvals/{ref}/escalate", {"notes": notes}):
                    st.info(f"Escalated: {ref}")
                    st.rerun()

            col_d.caption(f"Expires: {apr.get('expires_at', '')[:10]}")


def page_observability() -> None:
    st.title("📊 Observability & Metrics")
    st.info(
        "Live metrics are served from Prometheus. "
        "Open **Grafana** at [http://localhost:3002](http://localhost:3002) for full dashboards."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🔗 Service Links")
        st.markdown("- 📈 [Grafana](http://localhost:3002)")
        st.markdown("- 🔍 [Langfuse Traces](http://localhost:3000)")
        st.markdown("- ⏱️ [Temporal UI](http://localhost:8088)")
        st.markdown("- 📊 [Prometheus](http://localhost:9090)")

    with col2:
        st.markdown("#### 📐 Agent Limits (config)")
        from config import settings  # noqa: PLC0415

        st.table(
            {
                "Limit": [
                    "Max iterations",
                    "Max tool calls",
                    "Max cost/case",
                    "Tool timeout",
                    "Model timeout",
                    "Max parallel tools",
                ],
                "Value": [
                    settings.max_agent_iterations,
                    settings.max_tool_calls,
                    f"${settings.max_cost_per_case_usd:.2f}",
                    f"{settings.tool_timeout_seconds}s",
                    f"{settings.model_timeout_seconds}s",
                    settings.max_parallel_tools,
                ],
            }
        )

    with col3:
        st.markdown("#### 💰 LLM Pricing (Nova Micro)")
        st.table({"Direction": ["Input", "Output"], "Price": ["$0.035 / 1M tokens", "$0.140 / 1M tokens"]})
        st.caption("Budget: ~$1.75 / 1,000 investigations at 30K input + 5K output each.")

    st.divider()
    st.subheader("Raw Prometheus Metrics")
    try:
        with httpx.Client(timeout=5) as client:
            r = client.get(f"{API_BASE}/metrics")
            lines = [ln for ln in r.text.split("\n") if ln.startswith("bankguard_")]
            if lines:
                st.code("\n".join(lines[:40]))
            else:
                st.caption("No bankguard_ metrics yet (start some investigations first).")
    except Exception:
        st.caption("Metrics endpoint not reachable.")


def page_evaluations() -> None:
    st.title("🧪 Evaluation Framework")
    st.caption("Run the agent evaluation suite and view scored results.")

    from evals.regression.run_evals import load_test_cases, run_mock_eval  # noqa: PLC0415

    test_cases = load_test_cases()
    st.metric("Test cases loaded", len(test_cases))

    cats: dict[str, int] = {}
    for tc in test_cases:
        key = tc.get("category", "unknown")
        cats[key] = cats.get(key, 0) + 1

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Categories:**")
        for cat, count in sorted(cats.items()):
            st.markdown(f"- `{cat}`: {count}")
    with col2:
        st.markdown("**Test case tags:**")
        all_tags: set[str] = set()
        for tc in test_cases:
            all_tags.update(tc.get("tags", []))
        st.markdown(", ".join(sorted(f"`{t}`" for t in all_tags)))

    st.divider()

    col_run1, col_run2 = st.columns([1, 3])
    category_filter = col_run1.selectbox("Filter category", ["All"] + sorted(cats.keys()))

    if col_run2.button("▶️ Run Dry-Run Evaluation", type="primary"):
        filtered = load_test_cases(category=category_filter if category_filter != "All" else None)
        with st.spinner(f"Running {len(filtered)} test cases (mock agent)..."):
            results = asyncio.run(run_mock_eval(filtered))

        total = len(results)
        passed = sum(1 for r in results if r.outcome_score >= 0.70)
        unsafe = sum(1 for r in results if r.unsafe_action_taken)
        avg_s = sum(r.outcome_score for r in results) / total if total else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total cases", total)
        c2.metric("Passed (≥70%)", passed)
        c3.metric("🚨 Unsafe", unsafe, delta_color="inverse")
        c4.metric("Avg score", f"{avg_s:.0%}")

        if unsafe > 0:
            st.error(f"🚨 {unsafe} UNSAFE ACTION(S) detected — critical failures!")

        st.subheader("Per-Case Results")
        rows = [
            {
                "Test": r.test_ref,
                "Score": f"{r.outcome_score:.0%}",
                "RC ✓": "✓" if r.root_cause_correct else "✗",
                "Act ✓": "✓" if r.action_correct else "✗",
                "Pol ✓": "✓" if r.policy_applied_correctly else "✗",
                "Evid ✓": "✓" if r.evidence_grounded else "✗",
                "Unsafe": "🚨" if r.unsafe_action_taken else "✅",
                "Cost": f"${r.cost_usd:.4f}",
            }
            for r in sorted(results, key=lambda x: x.outcome_score, reverse=True)
        ]
        st.dataframe(rows, use_container_width=True)

        failures = [r for r in results if r.outcome_score < 0.70 or r.unsafe_action_taken]
        if failures:
            st.subheader("⚠️ Failures & Concerns")
            for r in failures:
                badge = "🚨" if r.unsafe_action_taken else "⚠️"
                with st.expander(f"{badge} {r.test_ref} — {r.outcome_score:.0%}"):
                    for note in r.grader_notes:
                        st.markdown(f"- {note}")


# ── Dispatch ────────────────────────────────────────────────────────────────────

_PAGES = {
    "🏠 Home": page_home,
    "🔍 Investigate": page_investigate,
    "📋 Cases": page_cases,
    "✅ Approvals": page_approvals,
    "📊 Observability": page_observability,
    "🧪 Evaluations": page_evaluations,
}

_PAGES[page]()
