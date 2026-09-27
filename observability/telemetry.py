"""
BankGuard AI — Observability / Telemetry.

Sets up:
  - OpenTelemetry SDK (traces + metrics)
  - OTLP exporter → OTel Collector → Langfuse + Prometheus
  - Structured logging via structlog → JSON → Loki
  - Prometheus metrics for agent operations
  - Cost tracking per run

Usage:
    from observability.telemetry import setup_telemetry, get_tracer, agent_metrics

    setup_telemetry()  # call once at app startup

    tracer = get_tracer("bankguard.agent")
    with tracer.start_as_current_span("investigate_case") as span:
        span.set_attribute("case.ref", case_ref)
        ...
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary

# ── Prometheus metrics ────────────────────────────────────────────────────────

agent_runs_total = Counter(
    "bankguard_agent_runs_total",
    "Total agent investigation runs",
    ["status", "sandbox_mode"],
)

agent_run_duration = Histogram(
    "bankguard_agent_run_duration_seconds",
    "Duration of agent investigation runs",
    ["status"],
    buckets=[1, 5, 10, 30, 60, 120, 300],
)

tool_calls_total = Counter(
    "bankguard_tool_calls_total",
    "Total tool calls through the gateway",
    ["tool_name", "status"],
)

tool_call_duration = Histogram(
    "bankguard_tool_call_duration_seconds",
    "Duration of individual tool calls",
    ["tool_name"],
    buckets=[0.1, 0.5, 1, 2, 5, 10],
)

llm_invocations_total = Counter(
    "bankguard_llm_invocations_total",
    "Total LLM model invocations",
    ["model", "agent"],
)

llm_tokens_total = Counter(
    "bankguard_llm_tokens_total",
    "Total LLM tokens consumed",
    ["model", "direction"],   # direction: input | output
)

llm_cost_usd_total = Counter(
    "bankguard_llm_cost_usd_total",
    "Total LLM cost in USD",
    ["model"],
)

cases_investigated = Counter(
    "bankguard_cases_investigated_total",
    "Cases investigated",
    ["outcome"],   # RESOLVED | ESCALATED | FAILED
)

approval_requests_total = Counter(
    "bankguard_approval_requests_total",
    "Approval requests created",
    ["risk_level", "action_type"],
)

approval_decisions_total = Counter(
    "bankguard_approval_decisions_total",
    "Approval decisions made",
    ["decision"],   # APPROVED | REJECTED | ESCALATED
)

active_investigations = Gauge(
    "bankguard_active_investigations",
    "Number of currently running agent investigations",
)

root_cause_distribution = Counter(
    "bankguard_root_cause_total",
    "Distribution of identified root causes",
    ["root_cause"],
)


# ── Structlog setup ───────────────────────────────────────────────────────────

def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog for JSON output (shipped to Loki via docker log driver)."""
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
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ── OpenTelemetry setup ───────────────────────────────────────────────────────

_otel_initialized = False


def setup_telemetry(service_name: str | None = None) -> None:
    """Initialize OpenTelemetry SDK. Call once at application startup."""
    global _otel_initialized
    if _otel_initialized:
        return

    from config import settings

    svc = service_name or settings.otel_service_name

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME, DEPLOYMENT_ENVIRONMENT

        resource = Resource.create({
            SERVICE_NAME: svc,
            DEPLOYMENT_ENVIRONMENT: settings.otel_environment,
        })

        provider = TracerProvider(resource=resource)

        # OTLP exporter → OTel Collector
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
        structlog.get_logger(__name__).warning(
            "otel_setup_failed_proceeding_without_tracing", error=str(exc)
        )


def get_tracer(name: str) -> Any:
    """Return an OpenTelemetry tracer (or no-op if OTel unavailable)."""
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except Exception:
        return _NoOpTracer()


# ── Langfuse trace helper ─────────────────────────────────────────────────────

class LangfuseTracer:
    """
    Thin wrapper around the Langfuse SDK for LLM-specific tracing.
    Falls back gracefully if Langfuse is not configured.
    """

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
            structlog.get_logger(__name__).info("langfuse_initialized")
        except Exception as exc:
            structlog.get_logger(__name__).warning(
                "langfuse_unavailable", error=str(exc)
            )

    def trace_run(
        self,
        run_id: str,
        case_ref: str,
        metadata: dict | None = None,
    ) -> Any:
        """Create a Langfuse trace for an agent run."""
        if not self._available:
            return _NoOpTrace()
        return self._client.trace(
            id=run_id,
            name=f"investigate:{case_ref}",
            metadata=metadata or {},
        )

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


# ── No-op stubs ───────────────────────────────────────────────────────────────

class _NoOpTracer:
    @contextmanager
    def start_as_current_span(self, name: str, **_: Any) -> Generator:
        yield _NoOpSpan()


class _NoOpSpan:
    def set_attribute(self, *_: Any, **__: Any) -> None: pass
    def set_status(self, *_: Any, **__: Any) -> None: pass
    def record_exception(self, *_: Any, **__: Any) -> None: pass
    def __enter__(self) -> "_NoOpSpan": return self
    def __exit__(self, *_: Any) -> None: pass


class _NoOpTrace:
    def generation(self, *_: Any, **__: Any) -> None: pass


# ── Context manager for run-level tracing ─────────────────────────────────────

@contextmanager
def trace_agent_run(
    run_id: str,
    case_ref: str,
    sandbox_mode: str = "development",
) -> Generator[dict[str, Any], None, None]:
    """
    Context manager that:
    - Starts an OTel span
    - Tracks Prometheus active_investigations gauge
    - Records run outcome on exit

    Usage:
        with trace_agent_run(run_id, case_ref) as ctx:
            ctx["span"].set_attribute("case.priority", "HIGH")
    """
    tracer = get_tracer("bankguard.agent_runner")
    start_time = time.monotonic()
    active_investigations.inc()

    ctx: dict[str, Any] = {"run_id": run_id, "case_ref": case_ref}

    with tracer.start_as_current_span(
        "agent_run",
        attributes={
            "run.id": run_id,
            "case.ref": case_ref,
            "sandbox.mode": sandbox_mode,
        },
    ) as span:
        ctx["span"] = span
        try:
            yield ctx
            agent_runs_total.labels(status="success", sandbox_mode=sandbox_mode).inc()
        except Exception as exc:
            span.record_exception(exc)
            agent_runs_total.labels(status="error", sandbox_mode=sandbox_mode).inc()
            raise
        finally:
            active_investigations.dec()
            duration = time.monotonic() - start_time
            agent_run_duration.labels(
                status=ctx.get("final_status", "unknown")
            ).observe(duration)


# ── Singletons ────────────────────────────────────────────────────────────────

langfuse_tracer = LangfuseTracer()
