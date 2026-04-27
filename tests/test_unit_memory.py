"""
Unit tests for the dual-layer memory system.
"""

from __future__ import annotations

import uuid

import pytest

from agents.memory import AgentMemory, MemoryTurn, SemanticMemory, SlidingWindowBuffer


class TestSlidingWindowBuffer:
    def test_add_and_retrieve(self):
        buf = SlidingWindowBuffer(max_size=3)
        for i in range(5):
            buf.add(MemoryTurn(role="user", content=f"msg-{i}"))
        assert buf.size == 3
        contents = [t.content for t in buf.get_recent()]
        assert contents == ["msg-2", "msg-3", "msg-4"]

    def test_get_recent_n(self):
        buf = SlidingWindowBuffer(max_size=10)
        for i in range(5):
            buf.add(MemoryTurn(role="user", content=f"msg-{i}"))
        recent = buf.get_recent(2)
        assert len(recent) == 2
        assert recent[0].content == "msg-3"

    def test_clear(self):
        buf = SlidingWindowBuffer()
        buf.add(MemoryTurn(role="user", content="hello"))
        buf.clear()
        assert buf.size == 0


class TestSemanticMemory:
    def test_store_and_retrieve(self, tmp_path):
        sm = SemanticMemory(
            collection_name=f"test_{uuid.uuid4().hex[:8]}",
            persist_dir=str(tmp_path / "chroma"),
        )
        sm.store(MemoryTurn(role="user", content="Write a fibonacci function"))
        sm.store(MemoryTurn(role="user", content="Deploy to AWS Lambda"))
        results = sm.retrieve("fibonacci sequence", top_k=1)
        assert len(results) >= 1
        assert "fibonacci" in results[0]["content"].lower()

    def test_empty_collection_returns_empty(self, tmp_path):
        sm = SemanticMemory(
            collection_name=f"test_{uuid.uuid4().hex[:8]}",
            persist_dir=str(tmp_path / "chroma"),
        )
        results = sm.retrieve("anything", top_k=5)
        assert results == []

    def test_count(self, tmp_path):
        sm = SemanticMemory(
            collection_name=f"test_{uuid.uuid4().hex[:8]}",
            persist_dir=str(tmp_path / "chroma"),
        )
        assert sm.count == 0
        sm.store(MemoryTurn(role="user", content="hello"))
        assert sm.count == 1


class TestAgentMemory:
    def test_build_context(self, tmp_path):
        mem = AgentMemory()
        mem.semantic = SemanticMemory(
            collection_name=f"test_{uuid.uuid4().hex[:8]}",
            persist_dir=str(tmp_path / "chroma"),
        )
        mem.add_turn("user", "Create a REST API with Flask")
        mem.add_turn("assistant", "Here is a Flask REST API...")
        ctx = mem.build_context("Flask API")
        assert "Flask" in ctx

    def test_empty_context(self):
        mem = AgentMemory()
        ctx = mem.build_context("anything")
        # May or may not be empty depending on chroma state, but should not crash
        assert isinstance(ctx, str)
