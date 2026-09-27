"""
BankGuard AI — FastAPI application entry point.

Routes:
  /api/cases             → cases.router
  /api/investigate       → investigations.router
  /api/approvals         → approvals.router
  /ws/investigate        → investigations.router (WebSocket)
  /health                → health check
  /metrics               → Prometheus metrics
  /docs                  → Swagger UI
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from config import settings
from observability.telemetry import setup_logging, setup_telemetry
from apps.api.routers.cases import router as cases_router
from apps.api.routers.investigations import router as investigations_router
from apps.api.routers.approvals import router as approvals_router

log = structlog.get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    setup_logging(settings.log_level)
    setup_telemetry()
    log.info(
        "bankguard_api_starting",
        env=settings.app_env,
        port=settings.api_port,
    )
    yield
    log.info("bankguard_api_shutting_down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BankGuard AI — Operations API",
    description=(
        "Production-grade agentic banking operations platform. "
        "Investigates payment issues, determines root causes, enforces policies, "
        "and proposes/executes remediation with human oversight."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request logging middleware ────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = int((time.monotonic() - start) * 1000)
    log.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
    )
    return response

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(cases_router)
app.include_router(investigations_router)
app.include_router(approvals_router)

# ── Prometheus metrics endpoint ───────────────────────────────────────────────

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health() -> dict:
    return {
        "status": "ok",
        "service": "bankguard-api",
        "version": "0.1.0",
        "environment": settings.app_env,
    }


@app.get("/health/ready", tags=["System"])
async def readiness() -> dict:
    """Deep health check — verifies DB and Valkey connectivity."""
    checks: dict[str, str] = {}

    # DB check
    try:
        from apps.api.dependencies import get_db
        pool = await get_db()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = f"error: {exc}"

    # Valkey check
    try:
        from apps.api.dependencies import get_valkey
        val = await get_valkey()
        await val.ping()
        checks["valkey"] = "ok"
    except Exception as exc:
        checks["valkey"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ready" if all_ok else "degraded", "checks": checks},
    )
