"""
BankGuard AI — Budget Manager.

Tracks and enforces per-run limits:
  - Token budget (input + output)
  - Cost budget (USD)
  - Tool-call count
  - Iteration count
  - Parallel tool slots

Bedrock Nova Micro pricing (ap-south-1):
  Input:  $0.000035 / 1K tokens
  Output: $0.000140 / 1K tokens
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import structlog

from config import settings

log = structlog.get_logger(__name__)

# Nova Micro pricing (per 1K tokens, USD)
COST_PER_1K_INPUT = 0.000035
COST_PER_1K_OUTPUT = 0.000140


class BudgetExceeded(Exception):
    """Raised when any budget limit is hit."""

    def __init__(self, reason: str, budget_type: str) -> None:
        super().__init__(reason)
        self.budget_type = budget_type


@dataclass
class BudgetSnapshot:
    """Point-in-time budget state — returned to the loop controller."""

    iterations_used: int
    iterations_limit: int
    tool_calls_used: int
    tool_calls_limit: int
    input_tokens_used: int
    output_tokens_used: int
    token_budget_input: int
    token_budget_output: int
    cost_usd: float
    cost_limit_usd: float
    parallel_slots_available: int

    @property
    def iterations_remaining(self) -> int:
        return self.iterations_limit - self.iterations_used

    @property
    def tool_calls_remaining(self) -> int:
        return self.tool_calls_limit - self.tool_calls_used

    @property
    def cost_remaining_usd(self) -> float:
        return self.cost_limit_usd - self.cost_usd

    @property
    def is_exhausted(self) -> bool:
        return self.iterations_remaining <= 0 or self.tool_calls_remaining <= 0 or self.cost_remaining_usd <= 0


class BudgetManager:
    """
    Thread-safe budget tracker for a single agent run.

    Usage:
        budget = BudgetManager()
        budget.record_llm_call(input_tokens=1500, output_tokens=300)
        budget.increment_iteration()       # raises BudgetExceeded if limit hit
        budget.acquire_tool_slot()         # raises BudgetExceeded if at capacity
        budget.record_tool_call()
    """

    def __init__(
        self,
        max_iterations: int | None = None,
        max_tool_calls: int | None = None,
        max_cost_usd: float | None = None,
        token_budget_input: int | None = None,
        token_budget_output: int | None = None,
        max_parallel_tools: int | None = None,
    ) -> None:
        self._lock = threading.Lock()

        self.max_iterations = max_iterations or settings.max_agent_iterations
        self.max_tool_calls = max_tool_calls or settings.max_tool_calls
        self.max_cost_usd = max_cost_usd or settings.max_cost_per_case_usd
        self.token_budget_input = token_budget_input or settings.token_budget_input
        self.token_budget_output = token_budget_output or settings.token_budget_output
        self.max_parallel_tools = max_parallel_tools or settings.max_parallel_tools

        # Counters
        self._iterations = 0
        self._tool_calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost_usd = 0.0
        self._active_tools = 0  # current parallel slots in use

    # ── LLM usage ─────────────────────────────────────────────────────────────

    def record_llm_call(self, input_tokens: int, output_tokens: int) -> float:
        """Record an LLM invocation and return the incremental cost."""
        cost = (input_tokens / 1000) * COST_PER_1K_INPUT + (output_tokens / 1000) * COST_PER_1K_OUTPUT
        with self._lock:
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._cost_usd += cost

            if self._input_tokens > self.token_budget_input:
                raise BudgetExceeded(
                    f"Input token budget exceeded: {self._input_tokens} > {self.token_budget_input}",
                    "INPUT_TOKEN_BUDGET",
                )
            if self._output_tokens > self.token_budget_output:
                raise BudgetExceeded(
                    f"Output token budget exceeded: {self._output_tokens} > {self.token_budget_output}",
                    "OUTPUT_TOKEN_BUDGET",
                )
            if self._cost_usd > self.max_cost_usd:
                raise BudgetExceeded(
                    f"Cost budget exceeded: ${self._cost_usd:.4f} > ${self.max_cost_usd:.4f}",
                    "COST_BUDGET",
                )

        log.debug(
            "budget_llm_recorded",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
            total_cost_usd=round(self._cost_usd, 6),
        )
        return cost

    # ── Iterations ────────────────────────────────────────────────────────────

    def increment_iteration(self) -> int:
        with self._lock:
            self._iterations += 1
            if self._iterations > self.max_iterations:
                raise BudgetExceeded(
                    f"Max iterations exceeded: {self._iterations} > {self.max_iterations}",
                    "MAX_ITERATIONS",
                )
        log.debug("budget_iteration", count=self._iterations, limit=self.max_iterations)
        return self._iterations

    # ── Tool calls ────────────────────────────────────────────────────────────

    def record_tool_call(self) -> int:
        with self._lock:
            self._tool_calls += 1
            if self._tool_calls > self.max_tool_calls:
                raise BudgetExceeded(
                    f"Max tool calls exceeded: {self._tool_calls} > {self.max_tool_calls}",
                    "MAX_TOOL_CALLS",
                )
        return self._tool_calls

    def acquire_tool_slot(self) -> None:
        """Acquire a parallel tool execution slot (blocking check)."""
        with self._lock:
            if self._active_tools >= self.max_parallel_tools:
                raise BudgetExceeded(
                    f"Max parallel tools reached: {self._active_tools}",
                    "MAX_PARALLEL_TOOLS",
                )
            self._active_tools += 1

    def release_tool_slot(self) -> None:
        with self._lock:
            self._active_tools = max(0, self._active_tools - 1)

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                iterations_used=self._iterations,
                iterations_limit=self.max_iterations,
                tool_calls_used=self._tool_calls,
                tool_calls_limit=self.max_tool_calls,
                input_tokens_used=self._input_tokens,
                output_tokens_used=self._output_tokens,
                token_budget_input=self.token_budget_input,
                token_budget_output=self.token_budget_output,
                cost_usd=round(self._cost_usd, 6),
                cost_limit_usd=self.max_cost_usd,
                parallel_slots_available=self.max_parallel_tools - self._active_tools,
            )

    def to_dict(self) -> dict:
        s = self.snapshot()
        return {
            "iterations": f"{s.iterations_used}/{s.iterations_limit}",
            "tool_calls": f"{s.tool_calls_used}/{s.tool_calls_limit}",
            "input_tokens": f"{s.input_tokens_used}/{s.token_budget_input}",
            "output_tokens": f"{s.output_tokens_used}/{s.token_budget_output}",
            "cost_usd": s.cost_usd,
            "cost_limit_usd": s.cost_limit_usd,
            "exhausted": s.is_exhausted,
        }
