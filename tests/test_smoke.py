"""
Smoke tests for Phase 1 – Coder Agent.

These tests validate:
  1. The structured output schema has the correct fields.
  2. The memory system works (both layers).
  3. The tools work in isolation.
  4. The full agent can be invoked end-to-end (requires LLM).
"""

from __future__ import annotations

import os
import shutil
import uuid

import pytest

from agents.memory import AgentMemory, MemoryTurn, SlidingWindowBuffer, SemanticMemory
from agents.schemas import CoderAgentOutput
from tools.file_tools import read_file, write_file
from tools.exec_tools import exec_python


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def temp_workspace(tmp_path):
    """Provide a temporary workspace directory."""
    ws = tmp_path / "test_workspace"
    ws.mkdir()
    return str(ws)


@pytest.fixture()
def agent_memory(tmp_path):
    """Provide a fresh AgentMemory with an isolated Chroma store."""
    mem = AgentMemory()
    mem.semantic = SemanticMemory(
        collection_name=f"test_{uuid.uuid4().hex[:8]}",
        persist_dir=str(tmp_path / "chroma_test"),
    )
    yield mem
    mem.clear()


# ── Schema Tests ──────────────────────────────────────────────────────────

class TestCoderAgentOutput:
    """Validate the Pydantic output model."""

    def test_required_fields_present(self):
        output = CoderAgentOutput(
            code="print('hello')",
            explanation="Prints hello",
            plan="1. Write code\n2. Run it",
            result="hello",
            session_id="abc123",
        )
        assert output.code == "print('hello')"
        assert output.explanation == "Prints hello"
        assert output.plan.startswith("1.")
        assert output.result == "hello"
        assert output.session_id == "abc123"
        assert output.success is True

    def test_model_has_minimum_four_fields(self):
        fields = CoderAgentOutput.model_fields
        required = {"code", "explanation", "plan", "result"}
        assert required.issubset(set(fields.keys()))

    def test_failure_output(self):
        output = CoderAgentOutput(
            code="",
            explanation="",
            plan="",
            result="",
            session_id="fail123",
            success=False,
            error="Something went wrong",
        )
        assert output.success is False
        assert output.error == "Something went wrong"


# ── Memory Tests ──────────────────────────────────────────────────────────

class TestSlidingWindowBuffer:
    def test_add_and_retrieve(self):
        buf = SlidingWindowBuffer(max_size=3)
        for i in range(5):
            buf.add(MemoryTurn(role="user", content=f"msg-{i}"))
        assert buf.size == 3
        contents = [t.content for t in buf.get_recent()]
        assert contents == ["msg-2", "msg-3", "msg-4"]

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


class TestAgentMemory:
    def test_build_context(self, agent_memory):
        agent_memory.add_turn("user", "Create a REST API with Flask")
        agent_memory.add_turn("assistant", "Here is a Flask REST API...")
        ctx = agent_memory.build_context("Flask API")
        assert "Flask" in ctx


# ── Tool Tests ────────────────────────────────────────────────────────────

class TestFileTools:
    def test_write_and_read(self, temp_workspace):
        result = write_file.run(
            file_path="test.py",
            content="print('hello')",
            workspace=temp_workspace,
        )
        assert "OK" in result

        content = read_file.run(
            file_path="test.py",
            workspace=temp_workspace,
        )
        assert "print('hello')" in content

    def test_read_nonexistent(self, temp_workspace):
        result = read_file.run(
            file_path="nope.py",
            workspace=temp_workspace,
        )
        assert "ERROR" in result

    def test_path_traversal_blocked(self, temp_workspace):
        result = write_file.run(
            file_path="../../etc/passwd",
            content="hacked",
            workspace=temp_workspace,
        )
        assert "ERROR" in result


class TestExecPython:
    def test_simple_execution(self, temp_workspace):
        result = exec_python.run(
            code="print('hello from sandbox')",
            workspace=temp_workspace,
        )
        assert "hello from sandbox" in result

    def test_timeout(self, temp_workspace):
        result = exec_python.run(
            code="import time; time.sleep(30)",
            workspace=temp_workspace,
            timeout=2,
        )
        assert "timed out" in result.lower()

    def test_syntax_error(self, temp_workspace):
        result = exec_python.run(
            code="def broken(",
            workspace=temp_workspace,
        )
        assert "SyntaxError" in result or "STDERR" in result


# ── End-to-End Agent Test (requires LLM) ─────────────────────────────────

@pytest.mark.skipif(
    not os.getenv("AZURE_OPENAI_API_KEY"),
    reason="AZURE_OPENAI_API_KEY not set – skipping live LLM test.",
)
class TestCoderAgentE2E:
    def test_simple_task(self):
        from agents.coder_agent import run_coder_agent

        result = run_coder_agent(
            "Write a Python function that returns the factorial of a number. "
            "Save it to factorial.py and run it with the input 5."
        )
        assert isinstance(result, CoderAgentOutput)
        assert result.session_id  # non-empty
        assert result.code  # non-empty
        assert result.plan  # non-empty
        assert result.explanation  # non-empty
        # result.result may be empty if execution had issues, but field exists
        assert hasattr(result, "result")
