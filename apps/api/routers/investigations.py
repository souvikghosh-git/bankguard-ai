"""
BankGuard AI — Investigations API router.

Endpoints:
  POST /api/investigate/{case_ref}         start investigation (returns run_id)
  GET  /api/investigate/{case_ref}/status  poll run status
  WS   /ws/investigate/{case_ref}          stream investigation events in real time
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from apps.api.dependencies import DBDep, IdentDep, ValDep

router = APIRouter(tags=["Investigations"])
log = structlog.get_logger(__name__)


class InvestigateResponse(BaseModel):
    run_id: str
    case_ref: str
    status: str
    message: str


# ── REST: Start investigation ─────────────────────────────────────────────────


@router.post("/api/investigate/{case_ref}", response_model=InvestigateResponse)
async def start_investigation(
    case_ref: str,
    db: DBDep,
    valkey: ValDep,
    identity: IdentDep,
) -> dict[str, Any]:
    """
    Kick off an agent investigation for the given case.
    Returns immediately with run_id; use the WebSocket or status endpoint
    to track progress.
    """
    from agents.runner import AgentRunner
    from config import settings
    from harness.sandbox import SandboxMode

    # Verify case exists
    async with db.acquire() as conn:
        row = await conn.fetchrow("SELECT case_ref, status FROM agent.cases WHERE case_ref = $1", case_ref)
    if not row:
        raise HTTPException(status_code=404, detail=f"Case {case_ref} not found")
    if row["status"] in ("RESOLVED", "CLOSED"):
        raise HTTPException(
            status_code=400,
            detail=f"Case {case_ref} is already {row['status']}",
        )

    sandbox = SandboxMode.READ_ONLY if settings.is_development else SandboxMode.LIVE
    runner = AgentRunner(db, valkey, sandbox_mode=sandbox)

    # Run investigation in background task; store progress in Valkey
    import uuid

    run_id = f"RUN-{uuid.uuid4().hex[:12].upper()}"

    async def _run_in_background() -> None:
        try:
            await valkey.setex(f"run:{run_id}:status", 3600, json.dumps({"status": "RUNNING"}))
            got_error = False
            async for event in runner.stream_investigation(case_ref, identity):
                await valkey.setex(f"run:{run_id}:latest_event", 3600, json.dumps(event, default=str))
                if event.get("type") == "error":
                    got_error = True
            # If the generator yielded an error event, mark as FAILED
            if got_error:
                await valkey.setex(
                    f"run:{run_id}:status",
                    3600,
                    json.dumps({"status": "FAILED", "error": "See case notes for details"}),
                )
            else:
                await valkey.setex(f"run:{run_id}:status", 3600, json.dumps({"status": "COMPLETED"}))
        except Exception as exc:
            log.exception("investigation_background_error", run_id=run_id, error=str(exc))
            await valkey.setex(
                f"run:{run_id}:status",
                3600,
                json.dumps({"status": "FAILED", "error": str(exc)}),
            )

    asyncio.create_task(_run_in_background())

    return {
        "run_id": run_id,
        "case_ref": case_ref,
        "status": "STARTED",
        "message": f"Investigation started. Connect to WS /ws/investigate/{case_ref}?run_id={run_id} for live updates.",
    }


# ── REST: Poll status ─────────────────────────────────────────────────────────


@router.get("/api/investigate/{case_ref}/status")
async def get_investigation_status(
    case_ref: str,
    run_id: str,
    valkey: ValDep,
    identity: IdentDep,
) -> dict[str, Any]:
    """Poll the current status of a running investigation."""
    raw = await valkey.get(f"run:{run_id}:status")
    if not raw:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found or expired")
    return json.loads(raw)


# ── WebSocket: Real-time stream ───────────────────────────────────────────────


@router.websocket("/ws/investigate/{case_ref}")
async def ws_investigate(
    websocket: WebSocket,
    case_ref: str,
    db: DBDep,
    valkey: ValDep,
) -> None:
    """
    WebSocket endpoint that streams investigation events in real time.

    Client connects → receives a stream of JSON events:
      {"type": "step",  "node": "transaction_agent", "observations": [...], ...}
      {"type": "final", "run_id": "...", "state": {...}}
      {"type": "error", "message": "..."}

    The identity is extracted from query param token= for WebSocket auth.
    """
    await websocket.accept()
    log.info("ws_connected", case_ref=case_ref)

    # Extract identity from query params (WS can't use headers easily)
    token = websocket.query_params.get("token", "dev-analyst-token")
    from apps.api.dependencies import DEV_IDENTITIES
    from config import settings

    identity = DEV_IDENTITIES.get(token, DEV_IDENTITIES["dev-analyst-token"])

    from agents.runner import AgentRunner
    from harness.sandbox import SandboxMode

    sandbox = SandboxMode.READ_ONLY if settings.is_development else SandboxMode.LIVE
    runner = AgentRunner(db, valkey, sandbox_mode=sandbox)

    try:
        await websocket.send_json({"type": "connected", "case_ref": case_ref})

        async for event in runner.stream_investigation(case_ref, identity):
            # Filter out non-serialisable objects
            safe_event = _make_serialisable(event)
            await websocket.send_json(safe_event)

            # Small yield to allow other coroutines
            await asyncio.sleep(0)

        await websocket.send_json({"type": "done", "case_ref": case_ref})

    except WebSocketDisconnect:
        log.info("ws_disconnected", case_ref=case_ref)
    except Exception as exc:
        log.exception("ws_investigation_error", case_ref=case_ref, error=str(exc))
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def _make_serialisable(obj: Any) -> Any:
    """Recursively make an object JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(i) for i in obj]
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if hasattr(obj, "isoformat"):  # datetime
        return obj.isoformat()
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
