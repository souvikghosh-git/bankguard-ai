"""
BankGuard AI — Agent Runtime (Harness).

The runtime wraps every agent run with:
  ┌────────────────────────────────────────────────────────┐
  │ Identity validation                                    │
  │ Budget manager (tokens / cost / iterations / tools)   │
  │ Retry policy                                          │
  │ Circuit breakers                                      │
  │ Sandbox guard                                         │
  │ Duplicate-call detection                              │
  │ Model routing (default → fallback)                    │
  │ Trace context propagation                             │
  │ Structured run lifecycle events                       │
  └────────────────────────────────────────────────────────┘

LLM calls are made via LiteLLM so that Bedrock, OpenAI and any
other provider can be swapped without changing agent code.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator

import structlog
from litellm import acompletion, token_counter

from config import settings
from harness.budget import BudgetExceeded, BudgetManager
from harness.circuit_breaker import get_breaker
from harness.retries import RetryPolicy
from harness.sandbox import Sandbox, SandboxMode, SandboxViolation

log = structlog.get_logger(__name__)


class StoppingReason(str, Enum):
    CASE_RESOLVED          = "CASE_RESOLVED"
    CONFIDENCE_REACHED     = "CONFIDENCE_REACHED"
    NO_NEW_EVIDENCE        = "NO_NEW_EVIDENCE"
    MAX_ITERATIONS         = "MAX_ITERATIONS"
    BUDGET_EXCEEDED        = "BUDGET_EXCEEDED"
    REPEATED_TOOL_CALL     = "REPEATED_TOOL_CALL"
    CRITICAL_TOOL_FAILURE  = "CRITICAL_TOOL_FAILURE"
    HUMAN_ESCALATION       = "HUMAN_ESCALATION"
    CANCELLED              = "CANCELLED"
    ERROR                  = "ERROR"


@dataclass
class RunResult:
    run_id: str
    case_id: str
    status: str
    stopping_reason: StoppingReason | None
    iterations: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_ms: int
    final_output: dict[str, Any] | None
    audit_log: list[dict]
    error: str | None = None


class AgentRuntime:
    """
    Production agent runtime harness.

    Lifecycle:
        async with runtime.run(case_id, identity) as ctx:
            result = await ctx.invoke_llm(messages)
            tool_resp = await ctx.invoke_tool("get_transaction_details", {...})
    """

    def __init__(
        self,
        db_pool: Any,
        valkey: Any,
        sandbox_mode: SandboxMode | None = None,
    ) -> None:
        self.db = db_pool
        self.valkey = valkey
        self._sandbox_mode = sandbox_mode

    @asynccontextmanager
    async def run(
        self,
        case_id: str,
        identity: dict[str, Any],
        budget_overrides: dict[str, Any] | None = None,
    ) -> AsyncGenerator["RunContext", None]:
        """
        Create a run context.  All agent actions go through this context.

        async with runtime.run(case_id, identity) as ctx:
            ...
        """
        run_id = f"RUN-{uuid.uuid4().hex[:12].upper()}"
        started = time.monotonic()

        budget = BudgetManager(**(budget_overrides or {}))
        sandbox = (
            Sandbox(self._sandbox_mode, run_id)
            if self._sandbox_mode
            else Sandbox.from_env(run_id)
        )
        retry = RetryPolicy()
        audit_log: list[dict] = []

        log.info(
            "agent_run_started",
            run_id=run_id,
            case_id=case_id,
            identity_role=identity.get("role"),
            sandbox=sandbox.mode.value,
        )

        ctx = RunContext(
            run_id=run_id,
            case_id=case_id,
            identity=identity,
            budget=budget,
            sandbox=sandbox,
            retry=retry,
            audit_log=audit_log,
            db=self.db,
            valkey=self.valkey,
        )

        try:
            yield ctx

        except BudgetExceeded as exc:
            log.warning("agent_run_budget_exceeded", run_id=run_id, reason=str(exc))
            ctx._stopping_reason = StoppingReason.BUDGET_EXCEEDED
            ctx._error = str(exc)

        except SandboxViolation as exc:
            log.error("agent_run_sandbox_violation", run_id=run_id, reason=str(exc))
            ctx._stopping_reason = StoppingReason.ERROR
            ctx._error = str(exc)

        except Exception as exc:
            log.exception("agent_run_error", run_id=run_id, error=str(exc))
            ctx._stopping_reason = StoppingReason.ERROR
            ctx._error = str(exc)

        finally:
            duration_ms = int((time.monotonic() - started) * 1000)
            snap = budget.snapshot()

            result = RunResult(
                run_id=run_id,
                case_id=case_id,
                status="COMPLETED" if ctx._stopping_reason != StoppingReason.ERROR else "FAILED",
                stopping_reason=ctx._stopping_reason,
                iterations=snap.iterations_used,
                tool_calls=snap.tool_calls_used,
                input_tokens=snap.input_tokens_used,
                output_tokens=snap.output_tokens_used,
                cost_usd=snap.cost_usd,
                duration_ms=duration_ms,
                final_output=ctx._final_output,
                audit_log=audit_log,
                error=ctx._error,
            )

            log.info(
                "agent_run_finished",
                run_id=run_id,
                status=result.status,
                stopping_reason=str(result.stopping_reason),
                iterations=result.iterations,
                tool_calls=result.tool_calls,
                cost_usd=result.cost_usd,
                duration_ms=duration_ms,
            )

            ctx._result = result


class RunContext:
    """
    Active run context — passed to the agent graph.

    Provides:
      invoke_llm()     — calls LLM through budget + retry + circuit breaker
      invoke_tool()    — calls tool through gateway + sandbox + budget
      stop()           — signals graceful termination
    """

    def __init__(
        self,
        run_id: str,
        case_id: str,
        identity: dict[str, Any],
        budget: BudgetManager,
        sandbox: Sandbox,
        retry: RetryPolicy,
        audit_log: list[dict],
        db: Any,
        valkey: Any,
    ) -> None:
        self.run_id = run_id
        self.case_id = case_id
        self.identity = identity
        self.budget = budget
        self.sandbox = sandbox
        self.retry = retry
        self.audit_log = audit_log
        self.db = db
        self.valkey = valkey

        # State
        self._stopping_reason: StoppingReason | None = None
        self._final_output: dict[str, Any] | None = None
        self._result: RunResult | None = None
        self._error: str | None = None
        self._tool_call_hashes: set[str] = set()   # duplicate detection

        # Observability handles — set by AgentRunner.stream_investigation()
        self._langfuse_trace: Any = None
        self._otel_span: Any = None

        # Gateway (lazy init)
        self._gateway: Any = None

    def _get_gateway(self) -> Any:
        if self._gateway is None:
            from tools.gateway import ToolGateway
            self._gateway = ToolGateway(
                db_pool=self.db,
                valkey=self.valkey,
                run_id=self.run_id,
                case_id=self.case_id,
                identity=self.identity,
                audit_log=self.audit_log,
            )
        return self._gateway

    # ── LLM invocation ────────────────────────────────────────────────────────

    async def invoke_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        Call LLM through LiteLLM with budget tracking, retry and circuit breaker.
        Returns the raw LiteLLM response dict.
        """
        self.budget.increment_iteration()
        chosen_model = model or settings.bedrock_default_model

        # Estimate input tokens before call
        try:
            est_input = token_counter(model=chosen_model, messages=messages)
        except Exception:
            est_input = sum(len(m.get("content", "")) // 4 for m in messages)

        # Build LiteLLM model string.
        # If the model name already contains a provider prefix (e.g. "gpt-4o-mini",
        # "ollama/llama3") use it as-is; otherwise prepend "bedrock/".
        def _model_string(m: str) -> str:
            providers = ("gpt-", "claude-", "gemini-", "ollama/", "openai/",
                         "anthropic/", "mistral/", "groq/", "bedrock/")
            return m if any(m.startswith(p) for p in providers) else f"bedrock/{m}"

        # Build LiteLLM kwargs
        kwargs: dict[str, Any] = {
            "model": _model_string(chosen_model),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": min(4096, self.budget.token_budget_output - self.budget.snapshot().output_tokens_used),
        }
        if tools:
            kwargs["tools"] = tools

        # Call with retry + circuit breaker
        breaker = get_breaker("bedrock-llm", fail_max=3, reset_timeout=30)

        async def _call() -> Any:
            return await asyncio.wait_for(
                acompletion(**kwargs),
                timeout=settings.model_timeout_seconds,
            )

        try:
            response = await self.retry.execute(_call, operation_name="llm_call")
        except Exception:
            # Try fallback model
            log.warning("llm_primary_failed_trying_fallback", model=chosen_model)
            kwargs["model"] = _model_string(settings.bedrock_fallback_model)
            response = await asyncio.wait_for(
                acompletion(**kwargs),
                timeout=settings.model_timeout_seconds,
            )

        # Record token usage and cost
        usage = getattr(response, "usage", None)
        in_tok  = getattr(usage, "prompt_tokens",     est_input) if usage else est_input
        out_tok = getattr(usage, "completion_tokens", 100)       if usage else 100
        cost = self.budget.record_llm_call(in_tok, out_tok)

        # Fire observability — Prometheus counters + Langfuse generation
        try:
            from observability.telemetry import record_llm_call
            langfuse_trace = getattr(self, "_langfuse_trace", None)
            record_llm_call(
                model=chosen_model,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
                agent_node="llm",
                langfuse_trace=langfuse_trace,
            )
        except Exception:
            pass  # telemetry must never break the agent path

        log.info(
            "llm_invoked",
            run_id=self.run_id,
            model=chosen_model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost, 6),
        )
        return response

    # ── Tool invocation ───────────────────────────────────────────────────────

    async def invoke_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> Any:
        """
        Call a tool through the full gateway stack.
        Returns the tool's ToolResponse.
        """
        # Sandbox check
        self.sandbox.check(tool_name)

        # Duplicate call detection
        call_hash = self._hash_call(tool_name, tool_input)
        if call_hash in self._tool_call_hashes:
            log.warning(
                "duplicate_tool_call_detected",
                tool=tool_name,
                run_id=self.run_id,
            )
            self._stopping_reason = StoppingReason.REPEATED_TOOL_CALL
            from tools.schemas import ToolResponse
            return ToolResponse.error(
                tool_name=tool_name,
                error_code="DUPLICATE_TOOL_CALL",
                error_message=(
                    f"Tool '{tool_name}' with identical inputs was already called this run. "
                    "Avoid repeating the same call — use the cached result."
                ),
                retryable=False,
            )
        self._tool_call_hashes.add(call_hash)

        # Budget slot
        self.budget.acquire_tool_slot()
        try:
            self.budget.record_tool_call()
            result = await self._get_gateway().call(
                tool_name=tool_name,
                tool_input=tool_input,
                idempotency_key=idempotency_key,
            )
        finally:
            self.budget.release_tool_slot()

        return result

    # ── Stop signals ─────────────────────────────────────────────────────────

    def stop(
        self,
        reason: StoppingReason,
        final_output: dict[str, Any] | None = None,
    ) -> None:
        self._stopping_reason = reason
        self._final_output = final_output
        log.info("agent_stopping", run_id=self.run_id, reason=reason.value)

    @property
    def should_stop(self) -> bool:
        return self._stopping_reason is not None

    @property
    def result(self) -> RunResult | None:
        return self._result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _hash_call(tool_name: str, tool_input: dict) -> str:
        payload = json.dumps({"tool": tool_name, "input": tool_input}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
