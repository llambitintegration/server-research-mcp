"""Simple token-bucket rate limiter with optional exponential-back-off.

Designed to protect outbound calls (LLM and MCP servers) while keeping the
implementation dependency-free so the lightweight test environment can import
it safely.
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable, TypeVar, Awaitable, Any

T = TypeVar("T")

class RateLimitError(RuntimeError):
    """Raised when a caller exhausts the maximum retry budget."""

class TokenBucketRateLimiter:
    """Token-bucket rate limiter supporting sync and async acquisition."""

    def __init__(self, rate: float, capacity: int | None = None):
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self.rate = rate                      # Tokens added per second
        self.capacity = capacity or rate      # Bucket size defaults to rate
        self._tokens = float(self.capacity)
        self._last_check = time.perf_counter()
        # Lock for async concurrency
        self._lock = asyncio.Lock()

    def _add_new_tokens(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_check
        self._last_check = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

    async def acquire(self) -> None:
        """Asynchronously wait until a token is available and then consume it."""
        while True:
            async with self._lock:
                self._add_new_tokens()
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
            # Wait a bit before re-checking (granularity 100 ms)
            await asyncio.sleep(0.1)

    def acquire_sync(self) -> None:
        """Blocking version for synchronous contexts."""
        while True:
            self._add_new_tokens()
            if self._tokens >= 1:
                self._tokens -= 1
                return
            time.sleep(0.1)

# ---------------------------------------------------------------------------
# Helper decorator – works for async or sync callables
# ---------------------------------------------------------------------------

def rate_limited(
    limiter: TokenBucketRateLimiter,
    max_retries: int = 3,
    base_delay: float = 0.5,
    backoff_factor: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that enforces *limiter* and retries on *RateLimitError*.

    The wrapped callable can be sync or async.  If it raises *RateLimitError*
    we sleep using exponential back-off and retry up to *max_retries* times.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(func):

            async def _async_wrapper(*args, **kwargs):  # type: ignore[override]
                retries = 0
                while True:
                    await limiter.acquire()
                    try:
                        return await func(*args, **kwargs)
                    except RateLimitError:
                        if retries >= max_retries:
                            raise
                        await asyncio.sleep(base_delay * (backoff_factor ** retries))
                        retries += 1

            return _async_wrapper  # type: ignore[return-value]

        else:

            def _sync_wrapper(*args, **kwargs):  # type: ignore[override]
                retries = 0
                while True:
                    limiter.acquire_sync()
                    try:
                        return func(*args, **kwargs)
                    except RateLimitError:
                        if retries >= max_retries:
                            raise
                        time.sleep(base_delay * (backoff_factor ** retries))
                        retries += 1

            return _sync_wrapper  # type: ignore[return-value]

    return decorator