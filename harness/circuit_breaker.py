"""
BankGuard AI — Circuit Breaker.

Prevents cascading failures when a tool or external service is repeatedly
failing.  Uses pybreaker with a per-tool breaker registry.

States:
  CLOSED   → normal operation
  OPEN     → failing fast, not calling backend
  HALF_OPEN → probe request to test if backend recovered
"""

from __future__ import annotations

import asyncio
import functools
from datetime import datetime
from typing import Any, Callable, Awaitable

import pybreaker
import structlog

log = structlog.get_logger(__name__)


class BankGuardCircuitListener(pybreaker.CircuitBreakerListener):
    """Log all state transitions through structlog."""

    def state_change(
        self, cb: pybreaker.CircuitBreaker, old_state: Any, new_state: Any
    ) -> None:
        log.warning(
            "circuit_breaker_state_change",
            name=cb.name,
            old_state=str(old_state),
            new_state=str(new_state),
            fail_counter=cb.fail_counter,
        )

    def failure(self, cb: pybreaker.CircuitBreaker, exc: Exception) -> None:
        log.warning(
            "circuit_breaker_failure",
            name=cb.name,
            error=str(exc),
            fail_counter=cb.fail_counter,
        )

    def success(self, cb: pybreaker.CircuitBreaker) -> None:
        log.debug("circuit_breaker_success", name=cb.name)


_listener = BankGuardCircuitListener()

# Global registry: tool_name → CircuitBreaker
_registry: dict[str, pybreaker.CircuitBreaker] = {}


def get_breaker(
    name: str,
    fail_max: int = 3,
    reset_timeout: int = 30,
) -> pybreaker.CircuitBreaker:
    """Get or create a CircuitBreaker for the named service."""
    if name not in _registry:
        _registry[name] = pybreaker.CircuitBreaker(
            fail_max=fail_max,
            reset_timeout=reset_timeout,
            name=name,
            listeners=[_listener],
        )
    return _registry[name]


async def call_with_breaker(
    name: str,
    fn: Callable[..., Awaitable[Any]],
    *args: Any,
    fail_max: int = 3,
    reset_timeout: int = 30,
    **kwargs: Any,
) -> Any:
    """
    Execute an async function through a named circuit breaker.

    Raises pybreaker.CircuitBreakerError if the circuit is OPEN.
    """
    breaker = get_breaker(name, fail_max=fail_max, reset_timeout=reset_timeout)

    # pybreaker is sync; wrap for async
    wrapped = functools.partial(fn, *args, **kwargs)
    try:
        # Use call() which applies the breaker logic
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: breaker.call(asyncio.get_event_loop().run_until_complete, wrapped()),
        )
        return result
    except pybreaker.CircuitBreakerError as exc:
        log.error(
            "circuit_open",
            name=name,
            error=str(exc),
        )
        raise


def get_all_breaker_states() -> dict[str, dict]:
    """Return current state of all circuit breakers (for observability)."""
    return {
        name: {
            "state": str(cb.current_state),
            "fail_counter": cb.fail_counter,
            "name": cb.name,
        }
        for name, cb in _registry.items()
    }
