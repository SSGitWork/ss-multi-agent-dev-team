"""
Unit tests for retry logic and circuit breaker.
"""

from __future__ import annotations

import time

import pytest

from agents.resilience import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    retry_with_backoff,
)


class TestRetryWithBackoff:
    def test_succeeds_on_first_try(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count == 1

    def test_retries_on_failure_then_succeeds(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient error")
            return "ok"

        assert fail_twice() == "ok"
        assert call_count == 3

    def test_exhausts_retries(self):
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fail():
            raise ValueError("permanent error")

        with pytest.raises(ValueError, match="permanent error"):
            always_fail()

    def test_only_catches_specified_exceptions(self):
        @retry_with_backoff(max_retries=3, base_delay=0.01, exceptions=(ValueError,))
        def raise_type_error():
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            raise_type_error()


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker("test_closed", threshold=3, cooldown=60)
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker("test_open", threshold=2, cooldown=60)

        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(self._failing_fn)

        assert cb.state == CircuitState.OPEN

    def test_rejects_calls_when_open(self):
        cb = CircuitBreaker("test_reject", threshold=1, cooldown=60)

        with pytest.raises(ValueError):
            cb.call(self._failing_fn)

        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerError):
            cb.call(self._succeeding_fn)

    def test_returns_cached_result_when_open(self):
        cb = CircuitBreaker("test_cache", threshold=2, cooldown=60)

        # First call succeeds — caches result
        result = cb.call(self._succeeding_fn)
        assert result == "success"

        # Two failures to open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(self._failing_fn)

        assert cb.state == CircuitState.OPEN

        # Should return cached result
        result = cb.call(self._succeeding_fn)
        assert result == "success"

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker("test_half_open", threshold=1, cooldown=0.1)

        with pytest.raises(ValueError):
            cb.call(self._failing_fn)

        assert cb.state == CircuitState.OPEN

        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_success_in_half_open(self):
        cb = CircuitBreaker("test_close_again", threshold=1, cooldown=0.1)

        with pytest.raises(ValueError):
            cb.call(self._failing_fn)

        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        result = cb.call(self._succeeding_fn)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        cb = CircuitBreaker("test_reopen", threshold=1, cooldown=0.1)

        with pytest.raises(ValueError):
            cb.call(self._failing_fn)

        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        with pytest.raises(ValueError):
            cb.call(self._failing_fn)

        assert cb.state == CircuitState.OPEN

    def test_reset_all(self):
        CircuitBreaker.get("svc_a")
        CircuitBreaker.get("svc_b")
        assert len(CircuitBreaker._instances) >= 2
        CircuitBreaker.reset_all()
        assert len(CircuitBreaker._instances) == 0

    @staticmethod
    def _failing_fn():
        raise ValueError("fail")

    @staticmethod
    def _succeeding_fn():
        return "success"
