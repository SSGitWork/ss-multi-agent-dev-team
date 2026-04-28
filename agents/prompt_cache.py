"""
Prompt caching with configurable TTL.

Caches LLM responses keyed by a hash of the prompt + model.
Reduces cost for repeated or similar prompts within a session.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Optional

from agents.config import get_settings

logger = logging.getLogger(__name__)


class PromptCache:
    """LRU prompt cache with TTL expiration."""

    def __init__(self, max_size: int = 100, ttl: int = 3600) -> None:
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl

    @staticmethod
    def _make_key(prompt: str, model: str) -> str:
        content = f"{model}::{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, prompt: str, model: str) -> Optional[str]:
        """Retrieve a cached response for a prompt/model pair.

        Returns the cached response if it exists and has not expired,
        otherwise returns None.
        """
        key = self._make_key(prompt, model)
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self._ttl:
                self._cache.move_to_end(key)
                logger.debug("Prompt cache HIT for model=%s", model)
                return entry["response"]
            else:
                del self._cache[key]
                logger.debug("Prompt cache EXPIRED for model=%s", model)
        return None

    def put(self, prompt: str, model: str, response: str) -> None:
        """Store a prompt response in the cache.

        Maintains LRU ordering and evicts the oldest entries if the cache
        exceeds the configured maximum size.
        """
        key = self._make_key(prompt, model)
        self._cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "model": model,
        }
        self._cache.move_to_end(key)

        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Remove all cached prompt responses."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Return the current number of cached entries."""
        return len(self._cache)


# Global cache instance
_global_cache: Optional[PromptCache] = None


def get_prompt_cache() -> PromptCache:
    """Return the global prompt cache instance.

    Initializes the cache using application settings if it does not yet exist.
    """
    global _global_cache
    if _global_cache is None:
        settings = get_settings()
        _global_cache = PromptCache(
            max_size=settings.cache.max_size,
            ttl=settings.cache.ttl,
        )
    return _global_cache
