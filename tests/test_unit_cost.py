"""
Unit tests for cost tracking and prompt cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.cost_tracker import CostTracker
from agents.prompt_cache import PromptCache


class TestCostTracker:
    def test_record_llm_call(self):
        tracker = CostTracker(session_id="test")
        tracker.record_llm_call("pm_agent", "azure/gpt-4o", 500, 200, 1000.0)
        record = tracker._records["pm_agent"]
        assert record.prompt_tokens == 500
        assert record.completion_tokens == 200
        assert record.total_tokens == 700
        assert record.llm_calls == 1
        assert record.estimated_cost_usd > 0

    def test_record_multiple_calls(self):
        tracker = CostTracker(session_id="test")
        tracker.record_llm_call("coder", "azure/gpt-4o-mini", 100, 50)
        tracker.record_llm_call("coder", "azure/gpt-4o-mini", 200, 100)
        record = tracker._records["coder"]
        assert record.prompt_tokens == 300
        assert record.completion_tokens == 150
        assert record.llm_calls == 2

    def test_record_tool_call(self):
        tracker = CostTracker(session_id="test")
        tracker.record_tool_call("coder")
        tracker.record_tool_call("coder")
        record = tracker._records["coder"]
        assert record.tool_calls == 2

    def test_record_cache_hit(self):
        tracker = CostTracker(session_id="test")
        tracker.record_cache_hit("pm_agent")
        record = tracker._records["pm_agent"]
        assert record.cache_hits == 1

    def test_generate_report(self):
        tracker = CostTracker(session_id="test")
        tracker.record_llm_call("pm_agent", "azure/gpt-4o", 500, 200)
        tracker.record_llm_call("coder", "azure/gpt-4o-mini", 300, 150)
        report = tracker.generate_report()
        assert report.session_id == "test"
        assert report.total_tokens == 1150
        assert report.total_cost_usd > 0
        assert len(report.agents) == 2

    def test_save_report(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COST_REPORT_DIR", str(tmp_path))
        # Need to reset Settings singleton to pick up new env
        from agents.config import Settings
        Settings._instance = None

        tracker = CostTracker(session_id="save_test")
        tracker.record_llm_call("pm_agent", "azure/gpt-4o", 100, 50)
        filepath = tracker.save_report()
        assert Path(filepath).exists()

        data = json.loads(Path(filepath).read_text())
        assert data["session_id"] == "save_test"
        assert data["total_tokens"] == 150

        # Reset singleton
        Settings._instance = None

    def test_mini_model_costs_less(self):
        tracker = CostTracker(session_id="test")
        tracker.record_llm_call("pm", "azure/gpt-4o", 1000, 1000)
        tracker.record_llm_call("coder", "azure/gpt-4o-mini", 1000, 1000)
        pm_cost = tracker._records["pm"].estimated_cost_usd
        coder_cost = tracker._records["coder"].estimated_cost_usd
        assert pm_cost > coder_cost, "GPT-4o should cost more than GPT-4o-mini"


class TestPromptCache:
    def test_put_and_get(self):
        cache = PromptCache(max_size=10, ttl=3600)
        cache.put("hello", "gpt-4o", "world")
        assert cache.get("hello", "gpt-4o") == "world"

    def test_cache_miss(self):
        cache = PromptCache(max_size=10, ttl=3600)
        assert cache.get("nonexistent", "gpt-4o") is None

    def test_different_models_different_keys(self):
        cache = PromptCache(max_size=10, ttl=3600)
        cache.put("hello", "gpt-4o", "response_4o")
        cache.put("hello", "gpt-4o-mini", "response_mini")
        assert cache.get("hello", "gpt-4o") == "response_4o"
        assert cache.get("hello", "gpt-4o-mini") == "response_mini"

    def test_ttl_expiration(self):
        cache = PromptCache(max_size=10, ttl=0)  # immediate expiry
        cache.put("hello", "gpt-4o", "world")
        assert cache.get("hello", "gpt-4o") is None

    def test_max_size_eviction(self):
        cache = PromptCache(max_size=2, ttl=3600)
        cache.put("a", "m", "1")
        cache.put("b", "m", "2")
        cache.put("c", "m", "3")
        assert cache.size == 2
        assert cache.get("a", "m") is None  # evicted
        assert cache.get("b", "m") == "2"
        assert cache.get("c", "m") == "3"

    def test_clear(self):
        cache = PromptCache(max_size=10, ttl=3600)
        cache.put("a", "m", "1")
        cache.clear()
        assert cache.size == 0
