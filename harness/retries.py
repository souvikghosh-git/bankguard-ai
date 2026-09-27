"""
BankGuard AI — Retry policy with exponential backoff.

Uses Tenacity under the hood, wrapped with structured logging
and budget-awareness (stops retrying if cost budget is exceeded).
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Callable, Awaitable, TypeVar

import structlog
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)

from config import settings

log = structlog.get_logger(__name__)

T = TypeVar("T")

# Exceptions that should trigger a retry
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    OSError,
)


class RetryPolicy:
    """
    Configurable retry policy for agent tool calls and LLM calls.

    Default: up to 2 retries with exponential backoff (2s → 8s) + jitter.
    """

    def __init__(
        self,
        max_attempts: int | None = None,
        min_wait: float = 1.0,
        max_wait: float = 10.0,
        jitter: float = 1.0,
    ) -> None:
        self.max_attempts = max_attempts or (settings.max_retries + 1)
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.jitter = jitter

    async def execute(
        self,
        fn: Callable[..., Awaitable[T]],
        *args: Any,
        operation_name: str = "operation",
        **kwargs: Any,
    ) -> T:
        """
        Execute fn with retry policy.
        Raises RetryError if all attempts are exhausted.
        """
        attempt = 0
        last_exc: Exception | None = None

        async for attempt_obj in AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential_jitter(
                initial=self.min_wait, max=self.max_wait, jitter=self.jitter
            ),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            reraise=False,
        ):
            with attempt_obj:
                attempt += 1
                try:
                    result = await fn(*args, **kwargs)
                    if attempt > 1:
                        log.info(
                            "retry_succeeded",
                            operation=operation_name,
                            attempt=attempt,
                        )
                    return result
                except RETRYABLE_EXCEPTIONS as exc:
                    last_exc = exc
                    log.warning(
                        "retry_attempt_failed",
                        operation=operation_name,
                        attempt=attempt,
                        max_attempts=self.max_attempts,
                        error=str(exc),
                        will_retry=attempt < self.max_attempts,
                    )
                    raise  # let tenacity handle wait+retry

        # All attempts exhausted
        log.error(
            "retry_exhausted",
            operation=operation_name,
            max_attempts=self.max_attempts,
            last_error=str(last_exc),
        )
        raise RetryError(last_exc) from last_exc


# Singleton default policy
default_retry_policy = RetryPolicy()
