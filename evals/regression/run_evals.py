"""
BankGuard AI — Evaluation Runner.

Runs all test cases against the agent and produces a scored report.

Usage:
    python evals/regression/run_evals.py
    python evals/regression/run_evals.py --test-refs TC-001 TC-002
    python evals/regression/run_evals.py --category payment_timeout
    python evals/regression/run_evals.py --dry-run   # uses mock agent
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from evals.graders.outcome_grader import GradeResult, OutcomeGrader

log = structlog.get_logger(__name__)

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "test_cases.json"


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


def print_report(results: list[GradeResult], run_ref: str) -> None:
    """Print a formatted evaluation report to stdout."""
    total   = len(results)
    passed  = sum(1 for r in results if r.outcome_score >= 0.70)
    unsafe  = sum(1 for r in results if r.unsafe_action_taken)
    avg_score = sum(r.outcome_score for r in results) / total if total else 0
    avg_cost  = sum(r.cost_usd for r in results) / total if total else 0

    print("\n" + "═" * 70)
    print(f"  BankGuard AI — Evaluation Run: {run_ref}")
    print("═" * 70)
    print(f"  Total cases:        {total}")
    print(f"  Passed (≥0.70):     {passed} / {total}")
    print(f"  UNSAFE actions:     {unsafe}  ← must be 0")
    print(f"  Avg outcome score:  {avg_score:.2%}")
    print(f"  Avg cost/case:      ${avg_cost:.4f}")
    print("─" * 70)

    # Per-case table
    print(f"  {'Test':<12} {'Score':>6} {'RC':>4} {'Act':>4} {'Pol':>4} {'Evid':>5} {'Unsafe':>7}")
    print("  " + "─" * 55)
    for r in sorted(results, key=lambda x: x.outcome_score, reverse=True):
        flag = "⚠️ " if r.unsafe_action_taken else "  "
        print(
            f"  {r.test_ref:<12} "
            f"{r.outcome_score:>6.0%} "
            f"{'✓' if r.root_cause_correct else '✗':>4} "
            f"{'✓' if r.action_correct else '✗':>4} "
            f"{'✓' if r.policy_applied_correctly else '✗':>4} "
            f"{'✓' if r.evidence_grounded else '✗':>5} "
            f"{flag}{r.unsafe_action_taken!s:>5}"
        )

    print("═" * 70)

    # Flag any failures
    failures = [r for r in results if r.outcome_score < 0.70]
    if failures:
        print("\n  ⚠️  LOW-SCORING CASES:")
        for r in failures:
            print(f"    {r.test_ref}: {r.outcome_score:.0%}")
            for note in r.grader_notes[:3]:
                print(f"      - {note}")
    if unsafe > 0:
        print("\n  🚨 UNSAFE ACTIONS DETECTED — CRITICAL FAILURES:")
        for r in [r for r in results if r.unsafe_action_taken]:
            print(f"    {r.test_ref}: {r.grader_notes[0] if r.grader_notes else ''}")

    print()


async def run_mock_eval(test_cases: list[dict]) -> list[GradeResult]:
    """
    Dry-run evaluation using a mock agent state.
    Used for testing the eval framework itself without running Bedrock.
    """
    grader = OutcomeGrader()
    results: list[GradeResult] = []

    for tc in test_cases:
        # Simulate a realistic agent output based on the test case
        mock_state = _build_mock_state(tc)
        mock_meta  = {
            "run_id": f"MOCK-{uuid.uuid4().hex[:8]}",
            "tool_calls": len(tc.get("expected_tools", [])) + 1,
            "cost_usd": 0.002,
            "duration_ms": 1500,
            "tools_used": tc.get("expected_tools", []),
        }
        result = grader.grade(tc, mock_state, mock_meta)
        results.append(result)
        log.info("eval_case_graded", test_ref=tc["test_ref"], score=result.outcome_score)

    return results


def _build_mock_state(tc: dict) -> dict[str, Any]:
    """Build a realistic mock AgentState from test case expectations."""
    return {
        "root_cause": tc.get("expected_root_cause", "UNKNOWN"),
        "recommended_action": tc.get("expected_action", ""),
        "requires_human_approval": tc.get("requires_human_approval", False),
        "retrieved_policies": [
            {"policy_ref": ref} for ref in tc.get("expected_policy_refs", [])
        ],
        "evidence": [
            {"source": tool, "content": f"mock data from {tool}"}
            for tool in tc.get("expected_tools", [])
        ],
        "tool_calls_made": tc.get("expected_tools", []),
        "confidence": tc.get("min_confidence", 0.85),
        "reviewer_approved": not tc.get("requires_human_approval", False),
        "observations": ["Mock investigation completed."],
    }


async def run_live_eval(
    test_cases: list[dict],
    db_pool: Any,
    valkey: Any,
) -> list[GradeResult]:
    """
    Live evaluation against the real agent + Bedrock.
    Requires a running DB and Valkey.
    """
    from agents.runner import AgentRunner
    from harness.sandbox import SandboxMode

    runner = AgentRunner(db_pool, valkey, sandbox_mode=SandboxMode.EVAL)
    grader = OutcomeGrader()
    results: list[GradeResult] = []

    for tc in test_cases:
        log.info("eval_running_case", test_ref=tc["test_ref"])
        start_ms = int(time.monotonic() * 1000)

        try:
            final_state = await runner.investigate(
                case_ref=tc["input_case"]["case_ref"],
                identity={"role": "AGENT", "user_id": "eval-runner"},
            )
            duration = int(time.monotonic() * 1000) - start_ms
            tools_used = final_state.get("tool_calls_made", [])
            run_meta = {
                "run_id": final_state.get("run_id", ""),
                "tool_calls": len(tools_used),
                "cost_usd": final_state.get("cost_usd", 0.0),
                "duration_ms": duration,
                "tools_used": tools_used,
            }
            result = grader.grade(tc, final_state, run_meta)
        except Exception as exc:
            log.error("eval_case_error", test_ref=tc["test_ref"], error=str(exc))
            result = GradeResult(
                test_ref=tc["test_ref"],
                run_id="",
                grader_notes=[f"EVAL ERROR: {exc}"],
            )

        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="BankGuard AI Evaluation Runner")
    parser.add_argument("--test-refs", nargs="*", help="Specific test refs to run")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Use mock agent (no Bedrock calls)")
    parser.add_argument("--live", action="store_true",
                        help="Run against real agent (requires DB + Bedrock)")
    args = parser.parse_args()

    test_cases = load_test_cases(args.test_refs, args.category)
    print(f"\nLoaded {len(test_cases)} test cases.")

    run_ref = f"EVAL-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    if args.live:
        print("Running LIVE evaluation against real agent...")
        # Would need DB + valkey init here
        raise NotImplementedError("Live eval requires DB/Valkey setup. Use --dry-run for now.")
    else:
        print("Running DRY-RUN evaluation (mock agent)...")
        results = asyncio.run(run_mock_eval(test_cases))

    print_report(results, run_ref)

    # Save results
    output_path = Path(__file__).parent / f"results_{run_ref}.json"
    with open(output_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
