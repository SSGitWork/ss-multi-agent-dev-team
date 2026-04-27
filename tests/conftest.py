"""
Shared test fixtures and mock LLM infrastructure.

Provides:
  • Automatic circuit breaker reset between tests.
  • Temporary workspace directories.
  • Pre-built SharedState, TaskItem, and A2A message fixtures.
  • A mock CrewAI kickoff helper that returns configurable output
    without calling any real LLM.
  • A cost tracker fixture wired into each test.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.config import Settings
from agents.cost_tracker import CostTracker, set_current_tracker
from agents.message_bus import A2AMessageBus
from agents.resilience import CircuitBreaker
from agents.schemas_a2a import (
    A2AIntent,
    A2AMessage,
    AgentRole,
    AllTestsPassedPayload,
    FixInstruction,
    FixInstructionsPayload,
    ReviewRequestPayload,
    TestFailure,
)
from agents.schemas_shared import (
    AgentCostRecord,
    PipelineCostReport,
    SharedState,
    TaskItem,
    TaskPriority,
    TaskStatus,
    TechnicalSpec,
)


# ══════════════════════════════════════════════════════════════════════════
# Auto-use fixtures
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Reset all circuit breakers between tests."""
    CircuitBreaker.reset_all()
    yield
    CircuitBreaker.reset_all()


@pytest.fixture(autouse=True)
def setup_cost_tracker():
    """Provide a fresh cost tracker for every test."""
    tracker = CostTracker(session_id="test_session")
    set_current_tracker(tracker)
    yield tracker
    set_current_tracker(None)


# ══════════════════════════════════════════════════════════════════════════
# Workspace fixtures
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def temp_workspace(tmp_path):
    """Provide a temporary workspace directory."""
    ws = tmp_path / "test_workspace"
    ws.mkdir()
    return str(ws)


@pytest.fixture()
def temp_cost_report_dir(tmp_path):
    """Provide a temporary directory for cost reports."""
    d = tmp_path / "cost_reports"
    d.mkdir()
    return str(d)


# ══════════════════════════════════════════════════════════════════════════
# SharedState fixtures
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def sample_requirement() -> str:
    return "Build a Python CLI calculator that supports add, subtract, multiply, and divide."


@pytest.fixture()
def sample_task_item() -> TaskItem:
    return TaskItem(
        task_id="TASK-001",
        title="Core arithmetic functions",
        description="Implement add, subtract, multiply, divide functions.",
        acceptance_criteria=[
            "add(2, 3) returns 5",
            "subtract(5, 3) returns 2",
            "multiply(4, 3) returns 12",
            "divide(10, 2) returns 5.0",
            "divide(1, 0) raises ValueError",
        ],
        priority=TaskPriority.HIGH,
        status=TaskStatus.PENDING,
    )


@pytest.fixture()
def sample_tasks() -> list[TaskItem]:
    return [
        TaskItem(
            task_id=f"TASK-{i:03d}",
            title=f"Task {i}",
            description=f"Description for task {i}",
            acceptance_criteria=[f"Criterion {i}.1", f"Criterion {i}.2"],
            priority=TaskPriority.HIGH if i <= 2 else TaskPriority.MEDIUM,
        )
        for i in range(1, 4)
    ]


@pytest.fixture()
def sample_state(sample_requirement, temp_workspace) -> SharedState:
    return SharedState(
        original_requirement=sample_requirement,
        session_id="test_abc123",
        workspace_path=temp_workspace,
    )


@pytest.fixture()
def pm_completed_state(sample_state, sample_tasks) -> SharedState:
    """A SharedState after the PM has run (simulated)."""
    sample_state.technical_spec = TechnicalSpec(
        name="CLI Calculator",
        description="A command-line calculator with four operations.",
        acceptance_criteria=["All four operations work", "Division by zero handled"],
        technical_approach="Python functions with argparse CLI",
        constraints=["No external dependencies"],
    )
    sample_state.tasks = sample_tasks
    sample_state.phase = "pm_complete"
    return sample_state


# ══════════════════════════════════════════════════════════════════════════
# A2A message fixtures
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def correlation_id() -> str:
    return uuid.uuid4().hex[:16]


@pytest.fixture()
def message_bus() -> A2AMessageBus:
    return A2AMessageBus()


@pytest.fixture()
def sample_review_request(correlation_id) -> A2AMessage:
    payload = ReviewRequestPayload(
        task_id="TASK-001",
        code='def add(a, b):\n    return a + b\n',
        file_path="task_001.py",
        acceptance_criteria=["add(2,3) returns 5", "add(-1,1) returns 0"],
        workspace_path="./workspace/test",
        iteration=1,
    )
    return A2AMessage(
        correlation_id=correlation_id,
        sender=AgentRole.CODER,
        receiver=AgentRole.QA,
        intent=A2AIntent.REVIEW_REQUEST,
        payload=payload.model_dump(),
    )


@pytest.fixture()
def sample_fix_response(correlation_id) -> A2AMessage:
    payload = FixInstructionsPayload(
        task_id="TASK-001",
        iteration=1,
        test_file="test_task_001.py",
        total_tests=3,
        passed=1,
        failed=2,
        failures=[
            TestFailure(
                test_name="test_add_negative",
                error_message="AssertionError: expected 0, got -2",
            ),
        ],
        fix_instructions=[
            FixInstruction(
                issue_id="FIX-001",
                description="add function does not handle negative numbers correctly",
                severity="high",
                suggested_fix="Check the logic for negative inputs",
                related_test="test_add_negative",
            ),
        ],
        raw_output="FAILED test_task_001.py::test_add_negative",
    )
    return A2AMessage(
        correlation_id=correlation_id,
        sender=AgentRole.QA,
        receiver=AgentRole.CODER,
        intent=A2AIntent.FIX_INSTRUCTIONS,
        payload=payload.model_dump(),
    )


@pytest.fixture()
def sample_all_passed(correlation_id) -> A2AMessage:
    payload = AllTestsPassedPayload(
        task_id="TASK-001",
        iteration=2,
        test_file="test_task_001.py",
        total_tests=3,
        passed=3,
        raw_output="3 passed",
    )
    return A2AMessage(
        correlation_id=correlation_id,
        sender=AgentRole.QA,
        receiver=AgentRole.CODER,
        intent=A2AIntent.ALL_TESTS_PASSED,
        payload=payload.model_dump(),
    )


# ══════════════════════════════════════════════════════════════════════════
# Mock LLM helpers
# ══════════════════════════════════════════════════════════════════════════

class MockCrewOutput:
    """Mimics the object returned by crew.kickoff()."""

    def __init__(self, raw_output: str, prompt_tokens: int = 100, completion_tokens: int = 200):
        self.raw = raw_output
        self.token_usage = MagicMock()
        self.token_usage.prompt_tokens = prompt_tokens
        self.token_usage.completion_tokens = completion_tokens

    def __str__(self) -> str:
        return self.raw


def make_mock_crew_output(raw: str, prompt_tokens: int = 100, completion_tokens: int = 200) -> MockCrewOutput:
    """Factory for creating mock crew outputs."""
    return MockCrewOutput(raw, prompt_tokens, completion_tokens)


# Standard mock outputs for each agent type

MOCK_PM_OUTPUT = json.dumps({
    "technical_spec": {
        "name": "Test Project",
        "description": "A test project for unit testing.",
        "acceptance_criteria": ["It works correctly", "Handles edge cases"],
        "technical_approach": "Python with standard library",
        "constraints": ["No external dependencies"],
    },
    "tasks": [
        {
            "task_id": "TASK-001",
            "title": "Core implementation",
            "description": "Implement the core functionality.",
            "acceptance_criteria": ["Function returns correct results", "Handles edge cases"],
            "priority": "high",
        },
        {
            "task_id": "TASK-002",
            "title": "CLI interface",
            "description": "Add command-line interface.",
            "acceptance_criteria": ["Accepts user input", "Displays output"],
            "priority": "medium",
        },
    ],
    "pm_notes": "Keep it simple for the MVP.",
})

MOCK_CODER_OUTPUT = """
CODE:
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

EXPLANATION:
Implemented four arithmetic functions with division-by-zero protection.

RESULT:
All functions tested and working correctly.
"""

MOCK_SELF_REFLECTION_OUTPUT = """
ISSUES_FOUND:
1. No type hints on function parameters.
2. No docstrings.
3. divide() should return float explicitly.

REVISED_CODE:
def add(a: float, b: float) -> float:
    \"\"\"Add two numbers.\"\"\"
    return a + b

def subtract(a: float, b: float) -> float:
    \"\"\"Subtract b from a.\"\"\"
    return a - b

def multiply(a: float, b: float) -> float:
    \"\"\"Multiply two numbers.\"\"\"
    return a * b

def divide(a: float, b: float) -> float:
    \"\"\"Divide a by b. Raises ValueError if b is zero.\"\"\"
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return float(a / b)
"""

MOCK_QA_TEST_OUTPUT = """
import pytest
from task_001 import add, subtract, multiply, divide

def test_add():
    assert add(2, 3) == 5

def test_subtract():
    assert subtract(5, 3) == 2

def test_multiply():
    assert multiply(4, 3) == 12

def test_divide():
    assert divide(10, 2) == 5.0

def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(1, 0)
"""

MOCK_QA_FIX_OUTPUT = json.dumps([
    {
        "issue_id": "FIX-001",
        "description": "test_add_negative failed: expected 0 got -2",
        "severity": "high",
        "suggested_fix": "Check addition logic for negative numbers",
        "related_test": "test_add_negative",
    }
])

MOCK_CODER_REVISION_OUTPUT = """
def add(a: float, b: float) -> float:
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b

def multiply(a: float, b: float) -> float:
    return a * b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return float(a / b)
"""
