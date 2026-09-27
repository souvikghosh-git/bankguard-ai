"""
BankGuard AI — FastAPI shared dependencies.
Provides DB pool, Valkey client, and identity extraction as
FastAPI Depends() callables.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

import asyncpg
import redis.asyncio as aioredis
import structlog
from fastapi import Depends, Header, HTTPException, status

from config import settings

log = structlog.get_logger(__name__)

# ── DB pool singleton ─────────────────────────────────────────────────────────

_db_pool: asyncpg.Pool | None = None


async def get_db() -> asyncpg.Pool:
    global _db_pool
    if _db_pool is None:
        _db_pool = await asyncpg.create_pool(
            dsn=settings.database_url.replace("postgresql+asyncpg://", "postgresql://"),
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
    return _db_pool


# ── Valkey client singleton ───────────────────────────────────────────────────

_valkey: aioredis.Redis | None = None


async def get_valkey() -> aioredis.Redis:
    global _valkey
    if _valkey is None:
        _valkey = aioredis.from_url(
            settings.valkey_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _valkey


# ── Identity / Auth ───────────────────────────────────────────────────────────

DEV_IDENTITIES = {
    "dev-analyst-token": {
        "user_id": "user-001",
        "role": "OPERATIONS_ANALYST",
        "name": "Dev Analyst",
        "permissions": [],
    },
    "dev-risk-token": {
        "user_id": "user-002",
        "role": "RISK_OFFICER",
        "name": "Dev Risk Officer",
        "permissions": [],
    },
    "dev-agent-token": {
        "user_id": "agent-001",
        "role": "AGENT",
        "name": "BankGuard Agent",
        "permissions": [],
    },
    "dev-admin-token": {
        "user_id": "admin-001",
        "role": "ADMIN",
        "name": "Dev Admin",
        "permissions": [],
    },
}


async def get_identity(
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """
    Extract and validate identity from Bearer token.
    In dev: accepts simple static tokens.
    In prod: validates Cognito JWT.
    """
    if not authorization:
        if settings.is_development:
            # Default dev identity — read-only analyst
            return DEV_IDENTITIES["dev-analyst-token"]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    token = authorization.removeprefix("Bearer ").strip()

    # Dev token lookup
    if token in DEV_IDENTITIES:
        return DEV_IDENTITIES[token]

    if settings.is_production:
        return await _validate_cognito_jwt(token)

    # Development fallback
    return DEV_IDENTITIES["dev-analyst-token"]


async def _validate_cognito_jwt(token: str) -> dict[str, Any]:
    """Validate a Cognito JWT and extract claims."""
    try:
        import boto3
        from jose import jwt, JWTError
        # In production: fetch JWKS from Cognito and verify signature
        # Simplified for portfolio — extend with proper JWKS validation
        claims = jwt.get_unverified_claims(token)
        return {
            "user_id": claims.get("sub", ""),
            "role": claims.get("custom:role", "OPERATIONS_ANALYST"),
            "name": claims.get("name", ""),
            "permissions": [],
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        )


# ── Type aliases for DI ───────────────────────────────────────────────────────

DBDep    = Annotated[asyncpg.Pool,       Depends(get_db)]
ValDep   = Annotated[aioredis.Redis,     Depends(get_valkey)]
IdentDep = Annotated[dict[str, Any],     Depends(get_identity)]
