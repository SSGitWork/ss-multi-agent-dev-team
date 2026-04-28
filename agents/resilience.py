"""
Resilience infrastructure: retry with exponential backoff + jitter,
and a circuit breaker with configurable cooldown.

Design:
  - retry_with_backoff wraps individual LLM/tool calls (not entire loops).
  - CircuitBreaker tracks consecutive failures per named service.
  - State transitions are logged at WARNING level.
  - When open, returns a structured error - never crashes.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from agents.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Circuit Breaker
class CircuitState(str, Enum):
    """Enumeration representing the state of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when the circuit is open and calls are rejected."""
    pass


class CircuitBreaker:
    """Per-service circuit breaker with configurable threshold and cooldown.

    State machine:
        CLOSED  - (threshold consecutive failures) - OPEN
        OPEN    - (cooldown elapsed)               - HALF_OPEN
        HALF_OPEN - (one success)                  - CLOSED
        HALF_OPEN - (one failure)                  - OPEN
    """

    _instances: dict[str, "CircuitBreaker"] = {}

    def __init__(self, name: str, threshold: int = 3, cooldown: int = 60) -> None:
        self.name = name
        self.threshold = threshold
        self.cooldown = cooldown
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._last_cached_result: Optional[Any] = None

    @classmethod
    def get(cls, name: str) -> "CircuitBreaker":
        """Retrieve or create a circuit breaker instance for a given service name."""
        if name not in cls._instances:
            settings = get_settings()
            cls._instances[name] = cls(
                name=name,
                threshold=settings.resilience.circuit_breaker_threshold,
                cooldown=settings.resilience.circuit_breaker_cooldown,
            )
        return cls._instances[name]

    @classmethod
    def reset_all(cls) -> None:
        """Reset and remove all registered circuit breaker instances."""
        cls._instances.clear()

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state.

        If the breaker is OPEN and the cooldown period has elapsed,
        the state automatically transitions to HALF_OPEN.
        """
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.cooldown:
                self._transition(CircuitState.HALF_OPEN)
        return self._state

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a function through the circuit breaker.

        If the breaker is open, the call is rejected or a cached result
        is returned if available. Failures update breaker state.
        """
        current = self.state

        if current == CircuitState.OPEN:
            logger.warning(
                "Circuit breaker '%s' is OPEN — rejecting call. "
                "Returning cached result if available.",
                self.name,
            )
            if self._last_cached_result is not None:
                return self._last_cached_result
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is open. No cached result available."
            )

        try:
            result = fn(*args, **kwargs)
            self._on_success(result)
            return result
        except Exception as exc:
            self._on_failure()
            raise

    def _on_success(self, result: Any) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._transition(CircuitState.CLOSED)
        self._failure_count = 0
        self._last_cached_result = result

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.threshold:
            self._transition(CircuitState.OPEN)

    def _transition(self, new_state: CircuitState) -> None:
        old = self._state
        self._state = new_state
        if old != new_state:
            logger.warning(
                "Circuit breaker '%s': %s , %s (failures=%d)",
                self.name, old.value, new_state.value, self._failure_count,
            )
            if new_state == CircuitState.CLOSED:
                self._failure_count = 0


# Retry decorator with exponential backoff + jitter
def retry_with_backoff(
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    exceptions: tuple = (Exception,),
    circuit_breaker_name: Optional[str] = None,
):
    """Decorator: retry with exponential backoff + jitter.

    Wraps individual calls (not entire agent loops).
    Optionally integrates with a named circuit breaker.
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        """Create a retry decorator with exponential backoff and jitter."""
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            """Execute the wrapped function with retry and optional circuit breaker."""
            settings = get_settings()
            retries = max_retries if max_retries is not None else settings.resilience.max_retries
            delay = base_delay if base_delay is not None else settings.resilience.retry_base_delay

            cb = CircuitBreaker.get(circuit_breaker_name) if circuit_breaker_name else None

            last_exc = None
            for attempt in range(1, retries + 1):
                try:
                    if cb:
                        return cb.call(fn, *args, **kwargs)
                    return fn(*args, **kwargs)
                except CircuitBreakerError:
                    raise
                except exceptions as exc:
                    last_exc = exc
                    if attempt < retries:
                        jitter = random.uniform(0, delay * 0.5)
                        wait = (delay * (2 ** (attempt - 1))) + jitter
                        logger.warning(
                            "Retry %d/%d for %s after %.1fs: %s",
                            attempt, retries, fn.__name__, wait, exc,
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            "All %d retries exhausted for %s: %s",
                            retries, fn.__name__, exc,
                        )

            raise last_exc  # type: ignore[misc]

        return wrapper
    return decorator
