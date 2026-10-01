"""
BankGuard AI — Observability / Telemetry.

Wires a single trace_id through the entire execution path:

    agent run start
        │  run_id injected into structlog context-vars
        │  OTel span "agent_run" started — trace_id derived from run_id
        │
        ├── LLM call → Langfuse generation logged with run_id
        ├── tool call → OTel child span + Prometheus tool_calls_total.inc()
        ├── approval  → Prometheus approval_requests_total.inc()
        └── run end   → Prometheus agent_runs_total/duration/cost inc()

All log lines emitted within a run automatically carry run_id, case_ref
and trace_id via structlog context-vars, so Loki queries can correlate
logs ↔ OTel spans ↔ Langfuse traces using the same run_id string.

Exports (imported by runner.py and gateway.py):
    setup_logging()
    setup_telemetry()
    get_tracer()
    trace_agent_run()         ← async context manager, links run_id to OTel
    record_tool_call()        ← Prometheus + OTel child span for a tool call
    record_llm_call()         ← Langfuse + Prometheus for one LLM invocation
    langfuse_tracer           ← singleton LangfuseTracer
    Prometheus counters/gauges (imported directly where needed)
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from prometheus_client import Counter, Gauge, Histogram

# ─────────────────────────────────────────────────────────────────────────────
# Prometheus metrics
# ─────────────────────────────────────────────────────────────────────────────

agent_runs_total = Counter(
    "bankguard_agent_runs_total",
    "Total agent investigation runs",
    ["status", "sandbox_mode"],
)
agent_run_duration_seconds = Histogram(
    "bankguard_agent_run_duration_seconds",
    "Duration of agent investigation runs",
    ["status"],
    buckets=[1, 5, 10, 30, 60, 120, 300],
)
agent_iterations_histogram = Histogram(
    "bankguard_agent_iterations",
    "Loop iterations per investigation run",
    buckets=[1, 2, 3, 4, 5, 6, 7, 8],
)
tool_calls_total = Counter(
    "bankguard_tool_calls_total",
    "Total tool calls through the gateway",
    ["tool_name", "status"],
)
tool_call_duration_seconds = Histogram(
    "bankguard_tool_call_duration_seconds",
    "Duration of individual tool calls",
    ["tool_name"],
    buckets=[0.05, 0.1, 0.5, 1, 2, 5, 10],
)
llm_invocations_total = Counter(
    "bankguard_llm_invocations_total",
    "Total LLM model invocations",
    ["model", "agent_node"],
)
llm_tokens_total = Counter(
    "bankguard_llm_tokens_total",
    "Total LLM tokens consumed",
    ["model", "direction"],  # direction: input | output
)
llm_cost_usd_total = Counter(
    "bankguard_llm_cost_usd_total",
    "Cumulative LLM cost in USD",
    ["model"],
)
cases_investigated_total = Counter(
    "bankguard_cases_investigated_total",
    "Cases investigated by outcome",
    ["outcome"],  # RESOLVED | ESCALATED | FAILED
)
approval_requests_total = Counter(
    "bankguard_approval_requests_total",
    "Approval requests created",
    ["risk_level", "action_type"],
)
approval_decisions_total = Counter(
    "bankguard_approval_decisions_total",
    "Approval decisions made",
    ["decision"],
)
active_investigations = Gauge(
    "bankguard_active_investigations",
    "Number of currently running agent investigations",
)
root_cause_total = Counter(
    "bankguard_root_cause_total",
    "Distribution of identified root causes",
    ["root_cause"],
)
unsafe_actions_blocked_total = Counter(
    "bankguard_unsafe_actions_blocked_total",
    "High/critical tool calls blocked by gateway",
    ["tool_name", "reason"],
)


# ─────────────────────────────────────────────────────────────────────────────
# structlog setup
# ─────────────────────────────────────────────────────────────────────────────


_logging_configured = False


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog for JSON output. Idempotent — safe to call multiple times.
    Uses stdlib.LoggerFactory so add_logger_name processor works correctly.
    """
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, log_level.upper(), logging.INFO)),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# OpenTelemetry setup
# ─────────────────────────────────────────────────────────────────────────────

_otel_initialized = False


def setup_telemetry(service_name: str | None = None) -> None:
    """Initialise the OTel SDK. Call once at application startup.
    Always calls setup_logging() first so structlog is configured before we log.
    """
    global _otel_initialized
    if _otel_initialized:
        return

    # Ensure structlog is configured before we try to use it
    setup_logging()

    from config import settings

    svc = service_name or settings.otel_service_name

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import DEPLOYMENT_ENVIRONMENT, SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {
                SERVICE_NAME: svc,
                DEPLOYMENT_ENVIRONMENT: settings.otel_environment,
            }
        )
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            insecure=True,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _otel_initialized = True
        structlog.get_logger(__name__).info(
            "otel_initialized", endpoint=settings.otel_exporter_otlp_endpoint
        )
    except Exception as exc:
        _otel_initialized = True  # don't retry — proceed without tracing
        structlog.get_logger(__name__).warning(
            "otel_setup_failed_proceeding_without_tracing", error=str(exc)
        )


def get_tracer(name: str) -> Any:
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except Exception:
        return _NoOpTracer()


def _run_id_to_trace_id(run_id: str) -> int:
    """
    Convert a BankGuard run_id (e.g. RUN-ABC123DEF456) into a 128-bit
    integer suitable for use as an OTel trace ID.
    We SHA-256 the run_id and take the first 16 bytes.
    This means every OTel span created within trace_agent_run() carries
    a trace_id that is deterministically derived from the run_id —
    so Langfuse, Loki and Prometheus all share the same correlation key.
    """
    import hashlib

    digest = hashlib.sha256(run_id.encode()).digest()
    return int.from_bytes(digest[:16], "big")


# ─────────────────────────────────────────────────────────────────────────────
# Main context manager — call this once per agent run
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def trace_agent_run(
    run_id: str,
    case_ref: str,
    sandbox_mode: str = "development",
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Async context manager that ties all observability together for one run:
      - Binds run_id + case_ref + trace_id to structlog context-vars
        (every log line inside the `async with` block carries these)
      - Creates a root OTel span whose trace_id is derived from run_id
      - Manages active_investigations Prometheus gauge
      - Records agent_runs_total / duration / iterations on exit

    Usage in runner.py:
        async with trace_agent_run(run_id, case_ref, sandbox_mode) as tctx:
            tctx["langfuse_trace"]  # Langfuse trace handle
            tctx["otel_span"]       # root OTel span
    """
    tracer = get_tracer("bankguard.agent_runner")
    start_ts = time.monotonic()
    active_investigations.inc()

    # Derive a stable OTel trace_id from run_id
    trace_id_int = _run_id_to_trace_id(run_id)
    trace_id_hex = format(trace_id_int, "032x")

    # Bind to structlog context-vars — ALL log lines in this scope carry these
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        run_id=run_id,
        case_ref=case_ref,
        trace_id=trace_id_hex,
        sandbox=sandbox_mode,
    )

    # Create the Langfuse trace
    lf_trace = langfuse_tracer.trace_run(
        run_id=run_id,
        case_ref=case_ref,
        metadata={"sandbox_mode": sandbox_mode, "trace_id": trace_id_hex},
    )

    ctx: dict[str, Any] = {
        "run_id": run_id,
        "case_ref": case_ref,
        "trace_id": trace_id_hex,
        "langfuse_trace": lf_trace,
        "otel_span": None,
        "final_status": "unknown",
        "iterations": 0,
        "root_cause": None,
    }

    # Start a root OTel span, injecting the derived trace_id via NonRecordingSpan
    # This ensures the same trace_id appears in Langfuse AND OTel backends.
    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

        span_ctx = SpanContext(
            trace_id=trace_id_int,
            span_id=int.from_bytes(uuid.uuid4().bytes[:8], "big"),
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        root_span_ctx = otel_trace.use_span(NonRecordingSpan(span_ctx), end_on_exit=False)
        root_span_ctx.__enter__()
    except Exception:
        root_span_ctx = None

    with tracer.start_as_current_span(
        "agent_run",
        attributes={
            "run.id": run_id,
            "case.ref": case_ref,
            "sandbox.mode": sandbox_mode,
            "trace.id": trace_id_hex,
        },
    ) as span:
        ctx["otel_span"] = span
        try:
            yield ctx
            agent_runs_total.labels(status="success", sandbox_mode=sandbox_mode).inc()
        except Exception as exc:
            span.record_exception(exc)
            ctx["final_status"] = "error"
            agent_runs_total.labels(status="error", sandbox_mode=sandbox_mode).inc()
            raise
        finally:
            if root_span_ctx:
                try:
                    root_span_ctx.__exit__(None, None, None)
                except Exception:
                    pass

            duration = time.monotonic() - start_ts
            status = ctx.get("final_status", "unknown")
            iters = ctx.get("iterations", 0)

            agent_run_duration_seconds.labels(status=status).observe(duration)
            agent_iterations_histogram.observe(iters)

            root_cause = ctx.get("root_cause")
            if root_cause:
                root_cause_total.labels(root_cause=root_cause).inc()

            outcome = "RESOLVED" if status == "success" else "FAILED"
            if "escalat" in str(ctx.get("stop_reason", "")).lower():
                outcome = "ESCALATED"
            cases_investigated_total.labels(outcome=outcome).inc()

            active_investigations.dec()
            structlog.contextvars.clear_contextvars()
            langfuse_tracer.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Per-tool and per-LLM recording helpers
# ─────────────────────────────────────────────────────────────────────────────


def record_tool_call(
    tool_name: str,
    status: str,
    duration_ms: int,
    otel_span: Any | None = None,
) -> None:
    """
    Increment Prometheus tool_calls_total and add a child OTel span attribute.
    Called by the Tool Gateway after every call.
    """
    tool_calls_total.labels(tool_name=tool_name, status=status).inc()
    tool_call_duration_seconds.labels(tool_name=tool_name).observe(duration_ms / 1000)

    if otel_span is not None:
        try:
            otel_span.add_event(
                "tool_call",
                attributes={"tool": tool_name, "status": status, "duration_ms": duration_ms},
            )
        except Exception:
            pass


def record_llm_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    agent_node: str = "unknown",
    langfuse_trace: Any = None,
) -> None:
    """
    Increment Prometheus LLM counters and log to Langfuse.
    Called by RunContext.invoke_llm() after every model call.
    """
    llm_invocations_total.labels(model=model, agent_node=agent_node).inc()
    llm_tokens_total.labels(model=model, direction="input").inc(input_tokens)
    llm_tokens_total.labels(model=model, direction="output").inc(output_tokens)
    llm_cost_usd_total.labels(model=model).inc(cost_usd)

    if langfuse_trace is not None:
        langfuse_tracer.log_llm_call(
            trace=langfuse_trace,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            agent_name=agent_node,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Langfuse tracer
# ─────────────────────────────────────────────────────────────────────────────


class LangfuseTracer:
    """Thin wrapper around the Langfuse SDK. Fails gracefully if unavailable."""

    def __init__(self) -> None:
        self._client: Any = None
        self._available = False
        self._init()

    def _init(self) -> None:
        try:
            from langfuse import Langfuse

            from config import settings

            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            self._available = True
        except Exception as exc:
            structlog.get_logger(__name__).warning("langfuse_unavailable", error=str(exc))

    def trace_run(self, run_id: str, case_ref: str, metadata: dict | None = None) -> Any:
        if not self._available:
            return _NoOpTrace()
        try:
            return self._client.trace(
                id=run_id,
                name=f"investigate:{case_ref}",
                metadata=metadata or {},
            )
        except Exception:
            return _NoOpTrace()

    def log_llm_call(
        self,
        trace: Any,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        agent_name: str = "unknown",
    ) -> None:
        if not self._available or isinstance(trace, _NoOpTrace):
            return
        try:
            trace.generation(
                name=f"llm:{agent_name}",
                model=model,
                usage={"input": input_tokens, "output": output_tokens},
                metadata={"cost_usd": cost_usd},
            )
        except Exception:
            pass

    def flush(self) -> None:
        if self._available:
            try:
                self._client.flush()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# No-op stubs (used when OTel / Langfuse are unavailable)
# ─────────────────────────────────────────────────────────────────────────────


class _NoOpTracer:
    def start_as_current_span(self, name: str, **_: Any) -> Any:
        return _NoOpSpan()


class _NoOpSpan:
    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def set_attribute(self, *_: Any, **__: Any) -> None:
        pass

    def add_event(self, *_: Any, **__: Any) -> None:
        pass

    def record_exception(self, *_: Any, **__: Any) -> None:
        pass


class _NoOpTrace:
    def generation(self, *_: Any, **__: Any) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────────────────────

langfuse_tracer = LangfuseTracer()
