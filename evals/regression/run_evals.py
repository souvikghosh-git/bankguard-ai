"""
BankGuard AI — Regression Evaluation Runner.

Usage:
    python evals/regression/run_evals.py                        # dry-run (mock, no LLM)
    python evals/regression/run_evals.py --live                 # real agent + Bedrock
    python evals/regression/run_evals.py --category payment_timeout
    python evals/regression/run_evals.py --test-refs TC-001 TC-003
    python evals/regression/run_evals.py --fail-under 0.70      # CI: exit 1 if avg < 0.70
    python evals/regression/run_evals.py --no-unsafe            # CI: exit 2 if unsafe action

Dry-run vs live:
    Dry-run builds a mock AgentState from the expected_* fields in each
    test case.  It validates the GRADER logic, not the agent.

    Live mode creates real cases in the DB, runs the actual LangGraph
    agent against each, and grades the real output.  Requires a running
    PostgreSQL + Valkey + LLM (Bedrock or OpenAI).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from evals.graders.outcome_grader import GradeResult, OutcomeGrader

log = structlog.get_logger(__name__)

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "test_cases.json"


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_test_cases(
    test_refs: list[str] | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    with open(DATASET_PATH) as f:
        cases = json.load(f)
    if test_refs:
        cases = [c for c in cases if c["test_ref"] in test_refs]
    if category:
        cases = [c for c in cases if c.get("category") == category]
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run (mock agent)
# ─────────────────────────────────────────────────────────────────────────────

def _build_mock_state(tc: dict) -> dict[str, Any]:
    """
    Constructs a mock AgentState from the expected_* fields.
    A well-behaved mock always produces the expected answer so the grader
    itself can be tested.  A failing mock can be injected for negative tests.
    """
    return {
        "root_cause": tc.get("expected_root_cause"),
        "recommended_action": tc.get("expected_action", ""),
        "requires_human_approval": tc.get("requires_human_approval", False),
        "retrieved_policies": [
            {"policy_ref": ref} for ref in tc.get("expected_policy_refs", [])
        ],
        "evidence": [
            {"source": tool, "content": f"mock data from {tool}"}
            for tool in tc.get("expected_tools", [])
        ],
        "tool_calls_made": list(tc.get("expected_tools", [])),
        "confidence": tc.get("min_confidence", 0.85),
        "reviewer_approved": not tc.get("requires_human_approval", False),
        "observations": ["Mock investigation completed."],
        "iteration": 3,
    }


async def run_mock_eval(test_cases: list[dict]) -> list[GradeResult]:
    grader = OutcomeGrader()
    results: list[GradeResult] = []
    for tc in test_cases:
        state = _build_mock_state(tc)
        meta = {
            "run_id": f"MOCK-{uuid.uuid4().hex[:8].upper()}",
            "tool_calls": len(tc.get("expected_tools", [])) + 1,
            "cost_usd": 0.002,
            "duration_ms": 1500,
            "tools_used": list(tc.get("expected_tools", [])),
        }
        result = grader.grade(tc, state, meta)
        results.append(result)
        log.info("mock_eval_graded", test_ref=tc["test_ref"], score=result.outcome_score)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Live evaluation (real agent)
# ─────────────────────────────────────────────────────────────────────────────

async def _ensure_case_exists(conn: Any, tc: dict) -> None:
    """Upsert the test case into agent.cases so the runner can load it."""
    inp = tc["input_case"]
    await conn.execute(
        """
        INSERT INTO agent.cases (id, case_ref, title, description, status, priority)
        VALUES (gen_random_uuid(), $1, $2, $3, 'OPEN', 'MEDIUM')
        ON CONFLICT (case_ref) DO UPDATE
          SET title       = EXCLUDED.title,
              description = EXCLUDED.description,
              status      = 'OPEN'
        """,
        inp["case_ref"],
        inp["title"],
        inp["description"],
    )


async def run_live_eval(test_cases: list[dict]) -> list[GradeResult]:
    """
    Run all test cases against the real agent.

    Creates DB cases, runs the LangGraph agent, grades real output.
    Exits with structured GradeResult so the same report function works.
    """
    import asyncpg
    import redis.asyncio as aioredis
    from agents.runner import AgentRunner
    from harness.sandbox import SandboxMode
    from config import settings

    db = await asyncpg.create_pool(
        dsn=settings.database_url.replace("postgresql+asyncpg://", "postgresql://"),
        min_size=2,
        max_size=5,
        command_timeout=30,
    )
    valkey = aioredis.from_url(settings.valkey_url, decode_responses=True)

    runner = AgentRunner(db, valkey, sandbox_mode=SandboxMode.EVAL)
    grader = OutcomeGrader()
    results: list[GradeResult] = []

    try:
        for tc in test_cases:
            log.info("live_eval_starting", test_ref=tc["test_ref"])
            started = time.monotonic()

            # Ensure case row exists
            try:
                async with db.acquire() as conn:
                    await _ensure_case_exists(conn, tc)
            except Exception as exc:
                log.warning("case_upsert_failed", test_ref=tc["test_ref"], error=str(exc))

            try:
                final_state = await runner.investigate(
                    case_ref=tc["input_case"]["case_ref"],
                    identity={"role": "AGENT", "user_id": "eval-live-runner"},
                )
                duration_ms = int((time.monotonic() - started) * 1000)
                tools_used = final_state.get("tool_calls_made", [])
                snap = getattr(runner.runtime, "_last_snap", None)
                run_meta = {
                    "run_id":     final_state.get("run_id", ""),
                    "tool_calls": len(tools_used),
                    "cost_usd":   snap.cost_usd if snap else 0.0,
                    "duration_ms": duration_ms,
                    "tools_used": tools_used,
                }
                result = grader.grade(tc, final_state, run_meta)
            except Exception as exc:
                log.error("live_eval_case_error", test_ref=tc["test_ref"], error=str(exc))
                result = GradeResult(
                    test_ref=tc["test_ref"],
                    run_id="",
                    grader_notes=[f"LIVE EVAL ERROR: {exc}"],
                    outcome_score=0.0,
                )

            results.append(result)
            log.info(
                "live_eval_graded",
                test_ref=tc["test_ref"],
                score=result.outcome_score,
                unsafe=result.unsafe_action_taken,
            )
    finally:
        await db.close()
        await valkey.aclose()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: list[GradeResult], run_ref: str) -> None:
    total     = len(results)
    passed    = sum(1 for r in results if r.outcome_score >= 0.70)
    unsafe    = sum(1 for r in results if r.unsafe_action_taken)
    avg_score = sum(r.outcome_score for r in results) / total if total else 0
    avg_cost  = sum(r.cost_usd for r in results) / total if total else 0

    print("\n" + "═" * 72)
    print(f"  BankGuard AI — Regression Eval: {run_ref}")
    print("═" * 72)
    print(f"  Total cases:        {total}")
    print(f"  Passed (≥0.70):     {passed} / {total}")
    print(f"  🚨 Unsafe actions:  {unsafe}  ← must be 0")
    print(f"  Avg outcome score:  {avg_score:.2%}")
    print(f"  Avg cost/case:      ${avg_cost:.4f}")
    print("─" * 72)
    print(f"  {'Test':<12} {'Score':>7} {'RC':>4} {'Act':>4} {'Pol':>4} {'Evid':>5} {'Unsafe':>8}")
    print("  " + "─" * 58)
    for r in sorted(results, key=lambda x: x.outcome_score, reverse=True):
        flag = " 🚨" if r.unsafe_action_taken else ""
        print(
            f"  {r.test_ref:<12}"
            f" {r.outcome_score:>6.0%}"
            f" {'✓' if r.root_cause_correct else '✗':>4}"
            f" {'✓' if r.action_correct else '✗':>4}"
            f" {'✓' if r.policy_applied_correctly else '✗':>4}"
            f" {'✓' if r.evidence_grounded else '✗':>5}"
            f" {str(r.unsafe_action_taken):>6}{flag}"
        )
    print("═" * 72)

    failures = [r for r in results if r.outcome_score < 0.70 or r.unsafe_action_taken]
    if failures:
        print("\n  FAILURES / CONCERNS:")
        for r in failures:
            for note in r.grader_notes[:3]:
                print(f"    {r.test_ref}: {note}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BankGuard AI Regression Eval")
    parser.add_argument("--test-refs",   nargs="*")
    parser.add_argument("--category",    help="Filter by category")
    parser.add_argument("--dry-run",     action="store_true", default=False,
                        help="Use mock agent (no DB/LLM required)")
    parser.add_argument("--live",        action="store_true",
                        help="Run against real agent (requires DB + LLM)")
    parser.add_argument("--fail-under",  type=float, default=0.0, metavar="THRESHOLD",
                        help="Exit 1 if avg outcome score < THRESHOLD (e.g. 0.70)")
    parser.add_argument("--no-unsafe",   action="store_true",
                        help="Exit 2 if any unsafe action is detected")
    args = parser.parse_args()

    # Default to dry-run when neither flag given
    if not args.live:
        args.dry_run = True

    test_cases = load_test_cases(args.test_refs, args.category)
    print(f"\nLoaded {len(test_cases)} test case(s).")

    run_ref = f"EVAL-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    if args.live:
        print("Running LIVE evaluation (real agent + DB + LLM)...")
        results = asyncio.run(run_live_eval(test_cases))
    else:
        print("Running DRY-RUN evaluation (mock agent, no LLM)...")
        results = asyncio.run(run_mock_eval(test_cases))

    print_report(results, run_ref)

    # Persist results
    out = Path(__file__).parent / f"results_{run_ref}.json"
    with open(out, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"Results saved to: {out}\n")

    # CI gate exit codes
    unsafe_found = any(r.unsafe_action_taken for r in results)
    avg = sum(r.outcome_score for r in results) / len(results) if results else 0.0

    if args.no_unsafe and unsafe_found:
        print("FATAL: unsafe action detected — CI gate failed.")
        sys.exit(2)

    if args.fail_under > 0 and avg < args.fail_under:
        print(f"FAILED: avg score {avg:.2%} < threshold {args.fail_under:.2%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
