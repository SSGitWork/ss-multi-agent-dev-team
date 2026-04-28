"""
Resilient LLM wrapper.

Wraps CrewAI's LLM construction with:
  - Retry with exponential backoff + jitter
  - Circuit breaker integration
  - Token tracking after every call
  - Prompt caching
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from crewai import LLM

from agents.config import get_settings
from agents.cost_tracker import get_current_tracker
from agents.prompt_cache import get_prompt_cache
from agents.resilience import CircuitBreaker, retry_with_backoff

logger = logging.getLogger(__name__)


def build_llm(
    model_name: str,
    deployment: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> LLM:
    """Build a CrewAI LLM object with Azure configuration."""
    settings = get_settings()
    return LLM(
        model=model_name,
        api_key=settings.azure.api_key,
        api_base=settings.azure.api_base,
        api_version=settings.azure.api_version,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_pm_llm() -> LLM:
    """Create the LLM configuration used by the Product Manager agent."""
    settings = get_settings()
    return build_llm(
        model_name=settings.models.pm_model,
        deployment=settings.models.pm_deployment,
        temperature=0.3,
    )


def build_coder_llm() -> LLM:
    """Create the LLM configuration used by the Coder agent."""
    settings = get_settings()
    return build_llm(
        model_name=settings.models.coder_model,
        deployment=settings.models.coder_deployment,
        temperature=0.2,
    )


def build_qa_llm() -> LLM:
    """Create the LLM configuration used by the QA agent."""
    settings = get_settings()
    return build_llm(
        model_name=settings.models.qa_model,
        deployment=settings.models.qa_deployment,
        temperature=0.2,
    )


@retry_with_backoff(circuit_breaker_name="crewai_kickoff")
def resilient_crew_kickoff(crew, agent_name: str = "unknown", model_name: str = "unknown"):
    """Execute crew.kickoff() with retry + circuit breaker + token tracking."""
    start = time.time()
    result = crew.kickoff()
    duration_ms = (time.time() - start) * 1000

    # Track tokens
    tracker = get_current_tracker()
    if tracker:
        usage = getattr(result, "token_usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        else:
            # Estimate from output length if usage not available
            raw = str(result)
            prompt_tokens = 0
            completion_tokens = len(raw) // 4  # rough estimate
        tracker.record_llm_call(
            agent_name=agent_name,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=duration_ms,
        )

    return result
