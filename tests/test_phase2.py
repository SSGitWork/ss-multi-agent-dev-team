"""
Phase 2 Tests – PM → Coder Handoff.

Test categories:
  1. SharedState schema validation.
  2. TaskItem and handoff protocol.
  3. Task consolidation logic.
  4. PM output parsing.
  5. Orchestration graph structure.
  6. End-to-end PM → Coder (requires LLM).
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

from agents.schemas_shared import (
    MAX_TASKS_PER_REQUIREMENT,
    SharedState,
    TaskItem,
    TaskPriority,
    TaskStatus,
    TechnicalSpec,
)
from agents.pm_agent import _consolidate_tasks, _extract_json
from orchestration.graph import OrchestrationGraph, build_default_pipeline


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_state() -> SharedState:
    """A SharedState initialized with a sample requirement."""
    return SharedState(
        original_requirement="Build a REST API with Flask that has CRUD endpoints for a todo list.",
        session_id="test123",
        workspace_path="./workspace/test123",
    )


@pytest.fixture()
def sample_tasks() -> list[TaskItem]:
    """A list of sample TaskItems."""
    return [
        TaskItem(
            task_id=f"TASK-{i:03d}",
            title=f"Task {i}",
            description=f"Description for task {i}",
            acceptance_criteria=[f"Criterion {i}.1", f"Criterion {i}.2"],
            priority=TaskPriority.HIGH if i <= 2 else TaskPriority.MEDIUM,
        )
        for i in range(1, 6)
    ]


@pytest.fixture()
def pm_completed_state(sample_state, sample_tasks) -> SharedState:
    """A SharedState after the PM has run (simulated)."""
    sample_state.technical_spec = TechnicalSpec(
        name="Todo REST API",
        description="A Flask-based REST API with CRUD for todos.",
        acceptance_criteria=["All CRUD endpoints work", "Returns JSON"],
        technical_approach="Flask + in-memory storage",
        constraints=["No database required for MVP"],
    )
    sample_state.tasks = sample_tasks
    sample_state.phase = "pm_complete"
    return sample_state


# ── SharedState Schema Tests ──────────────────────────────────────────────

class TestSharedStateSchema:
    """Validate the SharedState Pydantic model."""

    def test_minimal_creation(self):
        state = SharedState(original_requirement="Build something")
        assert state.original_requirement == "Build something"
        assert state.technical_spec is None
        assert state.tasks == []
        assert state.phase == "initialized"
        assert state.success is True

    def test_full_creation(self, pm_completed_state):
        state = pm_completed_state
        assert state.technical_spec is not None
        assert state.technical_spec.name == "Todo REST API"
        assert len(state.tasks) == 5
        assert state.phase == "pm_complete"

    def test_json_serialization_roundtrip(self, pm_completed_state):
        """SharedState must be JSON-serializable (handoff protocol requirement)."""
        json_str = pm_completed_state.to_json()
        restored = SharedState.from_json(json_str)
        assert restored.original_requirement == pm_completed_state.original_requirement
        assert restored.technical_spec.name == pm_completed_state.technical_spec.name
        assert len(restored.tasks) == len(pm_completed_state.tasks)
        assert restored.tasks[0].task_id == pm_completed_state.tasks[0].task_id

    def test_json_is_valid(self, pm_completed_state):
        """The serialized JSON must be parseable."""
        json_str = pm_completed_state.to_json()
        parsed = json.loads(json_str)
        assert "original_requirement" in parsed
        assert "tasks" in parsed
        assert isinstance(parsed["tasks"], list)


# ── TaskItem & Handoff Protocol Tests ─────────────────────────────────────

class TestTaskItem:
    """Validate the handoff protocol record."""

    def test_required_fields(self):
        task = TaskItem(
            task_id="TASK-001",
            title="Implement login",
            description="Create a login endpoint",
        )
        assert task.task_id == "TASK-001"
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.MEDIUM
        assert task.code_output == ""

    def test_handoff_protocol_fields(self):
        """The handoff protocol must include: task_id, description,
        acceptance_criteria, and priority."""
        task = TaskItem(
            task_id="TASK-001",
            title="Test task",
            description="Do something",
            acceptance_criteria=["It works"],
            priority=TaskPriority.HIGH,
        )
        # Verify all protocol fields exist and are correct types
        assert isinstance(task.task_id, str)
        assert isinstance(task.description, str)
        assert isinstance(task.acceptance_criteria, list)
        assert isinstance(task.priority, TaskPriority)

    def test_status_transitions(self):
        task = TaskItem(
            task_id="TASK-001",
            title="Test",
            description="Test task",
        )
        assert task.status == TaskStatus.PENDING

        task.status = TaskStatus.IN_PROGRESS
        assert task.status == TaskStatus.IN_PROGRESS

        task.status = TaskStatus.COMPLETED
        assert task.status == TaskStatus.COMPLETED

    def test_json_serializable(self):
        task = TaskItem(
            task_id="TASK-001",
            title="Test",
            description="Test task",
            acceptance_criteria=["Works correctly"],
            priority=TaskPriority.HIGH,
        )
        json_str = task.model_dump_json()
        restored = TaskItem.model_validate_json(json_str)
        assert restored.task_id == task.task_id
        assert restored.priority == TaskPriority.HIGH


# ── Pending / Completed Task Helpers ──────────────────────────────────────

class TestSharedStateHelpers:
    def test_get_pending_tasks(self, pm_completed_state):
        pending = pm_completed_state.get_pending_tasks()
        assert len(pending) == 5
        assert all(t.status == TaskStatus.PENDING for t in pending)

    def test_get_completed_tasks(self, pm_completed_state):
        pm_completed_state.tasks[0].status = TaskStatus.COMPLETED
        pm_completed_state.tasks[1].status = TaskStatus.COMPLETED
        completed = pm_completed_state.get_completed_tasks()
        assert len(completed) == 2

    def test_all_tasks_done(self, pm_completed_state):
        assert not pm_completed_state.all_tasks_done()

        for t in pm_completed_state.tasks:
            t.status = TaskStatus.COMPLETED
        assert pm_completed_state.all_tasks_done()

    def test_all_tasks_done_with_failures(self, pm_completed_state):
        for t in pm_completed_state.tasks:
            t.status = TaskStatus.COMPLETED
        pm_completed_state.tasks[-1].status = TaskStatus.FAILED
        # Failed also counts as "done"
        assert pm_completed_state.all_tasks_done()


# ── Task Consolidation Tests ──────────────────────────────────────────────

class TestTaskConsolidation:
    def test_no_consolidation_needed(self, sample_tasks):
        """5 tasks < 8 max → no change."""
        result = _consolidate_tasks(sample_tasks)
        assert len(result) == 5

    def test_consolidation_at_limit(self):
        """Exactly 8 tasks → no change."""
        tasks = [
            TaskItem(
                task_id=f"TASK-{i:03d}",
                title=f"Task {i}",
                description=f"Desc {i}",
            )
            for i in range(1, 9)
        ]
        result = _consolidate_tasks(tasks)
        assert len(result) == 8

    def test_consolidation_over_limit(self):
        """12 tasks → consolidated to 8."""
        tasks = [
            TaskItem(
                task_id=f"TASK-{i:03d}",
                title=f"Task {i}",
                description=f"Description for task {i}",
                acceptance_criteria=[f"Criterion {i}"],
                priority=TaskPriority.LOW,
            )
            for i in range(1, 13)
        ]
        result = _consolidate_tasks(tasks)
        assert len(result) == MAX_TASKS_PER_REQUIREMENT  # 8

        # Last task should be the consolidated one
        last = result[-1]
        assert "Consolidated" in last.title
        assert "TASK-009" in last.description or "TASK-009" in last.title or last.task_id == "TASK-008"

    def test_consolidation_preserves_first_tasks(self):
        """The first (MAX-1) tasks should be unchanged."""
        tasks = [
            TaskItem(
                task_id=f"TASK-{i:03d}",
                title=f"Task {i}",
                description=f"Desc {i}",
            )
            for i in range(1, 12)
        ]
        result = _consolidate_tasks(tasks)
        for i in range(MAX_TASKS_PER_REQUIREMENT - 1):
            assert result[i].task_id == tasks[i].task_id


# ── JSON Extraction Tests ─────────────────────────────────────────────────

class TestJsonExtraction:
    def test_clean_json(self):
        raw = '{"technical_spec": {"name": "Test"}, "tasks": [], "pm_notes": ""}'
        result = _extract_json(raw)
        assert result["technical_spec"]["name"] == "Test"

    def test_json_with_surrounding_text(self):
        raw = 'Here is the output:\n{"technical_spec": {"name": "Test"}, "tasks": [], "pm_notes": ""}\nDone!'
        result = _extract_json(raw)
        assert result["technical_spec"]["name"] == "Test"

    def test_json_in_code_fence(self):
        raw = '```json\n{"technical_spec": {"name": "Test"}, "tasks": [], "pm_notes": ""}\n```'
        result = _extract_json(raw)
        assert result["technical_spec"]["name"] == "Test"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            _extract_json("This is not JSON at all")


# ── Orchestration Graph Tests ─────────────────────────────────────────────

class TestOrchestrationGraph:
    def test_graph_node_execution(self):
        """A simple node should transform state."""
        def mock_pm(state: SharedState) -> SharedState:
            state.phase = "pm_complete"
            state.tasks = [
                TaskItem(
                    task_id="TASK-001",
                    title="Mock task",
                    description="Do something",
                )
            ]
            return state

        graph = OrchestrationGraph()
        graph.add_node("MockPM", mock_pm)

        state = graph.run("Test requirement")
        assert state.phase == "pm_complete"
        assert len(state.tasks) == 1

    def test_graph_stops_on_failure(self):
        """If a node fails, subsequent nodes should not run."""
        def failing_node(state: SharedState) -> SharedState:
            state.success = False
            state.error = "Intentional failure"
            return state

        def should_not_run(state: SharedState) -> SharedState:
            state.phase = "should_not_reach"
            return state

        graph = OrchestrationGraph()
        graph.add_node("Failing", failing_node)
        graph.add_node("After", should_not_run)

        state = graph.run("Test")
        assert state.success is False
        assert state.phase != "should_not_reach"

    def test_graph_passes_state_between_nodes(self):
        """State mutations in node 1 must be visible in node 2."""
        def node1(state: SharedState) -> SharedState:
            state.pm_notes = "from_node1"
            return state

        def node2(state: SharedState) -> SharedState:
            assert state.pm_notes == "from_node1"
            state.phase = "node2_complete"
            return state

        graph = OrchestrationGraph()
        graph.add_node("Node1", node1)
        graph.add_node("Node2", node2)

        state = graph.run("Test")
        assert state.phase == "node2_complete"

    def test_default_pipeline_has_two_nodes(self):
        pipeline = build_default_pipeline()
        assert len(pipeline._nodes) == 2
        assert pipeline._nodes[0].name == "Product Manager"
        assert pipeline._nodes[1].name == "Coder"


# ── PM → Coder Handoff Contract Tests ─────────────────────────────────────

class TestPMCoderHandoff:
    """Assert that after the PM node runs, SharedState has the correct shape."""

    def test_simulated_pm_output_has_tasks(self):
        """Simulate PM output and verify the Coder can read it."""
        state = SharedState(original_requirement="Build a calculator")

        # Simulate PM output
        state.technical_spec = TechnicalSpec(
            name="Calculator",
            description="A CLI calculator",
            acceptance_criteria=["Supports +, -, *, /"],
            technical_approach="Python with argparse",
        )
        state.tasks = [
            TaskItem(
                task_id="TASK-001",
                title="Core arithmetic",
                description="Implement add, subtract, multiply, divide functions",
                acceptance_criteria=["All four operations work", "Division by zero handled"],
                priority=TaskPriority.HIGH,
            ),
            TaskItem(
                task_id="TASK-002",
                title="CLI interface",
                description="Add argparse-based CLI",
                acceptance_criteria=["Accepts two numbers and an operator"],
                priority=TaskPriority.MEDIUM,
            ),
        ]
        state.phase = "pm_complete"

        # Verify the Coder can read the handoff
        pending = state.get_pending_tasks()
        assert len(pending) == 2
        assert pending[0].task_id == "TASK-001"
        assert pending[0].status == TaskStatus.PENDING
        assert len(pending[0].acceptance_criteria) == 2
        assert pending[0].priority == TaskPriority.HIGH

    def test_task_list_schema_after_pm(self):
        """Every task must have the required handoff protocol fields."""
        state = SharedState(original_requirement="Build something")
        state.tasks = [
            TaskItem(
                task_id="TASK-001",
                title="First task",
                description="Do the first thing",
                acceptance_criteria=["It works"],
                priority=TaskPriority.HIGH,
            ),
        ]

        for task in state.tasks:
            # Required handoff protocol fields
            assert task.task_id, "task_id must be non-empty"
            assert task.description, "description must be non-empty"
            assert isinstance(task.acceptance_criteria, list), "acceptance_criteria must be a list"
            assert isinstance(task.priority, TaskPriority), "priority must be a TaskPriority enum"
            assert task.status == TaskStatus.PENDING, "initial status must be PENDING"

    def test_state_is_json_serializable_after_pm(self, pm_completed_state):
        """The full state after PM must serialize to valid JSON."""
        json_str = pm_completed_state.to_json()
        data = json.loads(json_str)

        assert data["phase"] == "pm_complete"
        assert len(data["tasks"]) > 0
        assert data["technical_spec"]["name"] == "Todo REST API"

        # Verify each task in JSON has protocol fields
        for task_data in data["tasks"]:
            assert "task_id" in task_data
            assert "description" in task_data
            assert "acceptance_criteria" in task_data
            assert "priority" in task_data


# ── End-to-End Pipeline Test (requires LLM) ──────────────────────────────

@pytest.mark.skipif(
    not os.getenv("AZURE_API_KEY"),
    reason="AZURE_API_KEY not set – skipping live LLM test.",
)
class TestPhase2E2E:
    def test_pm_produces_valid_state(self):
        """Run only the PM agent and verify output structure."""
        from agents.pm_agent import run_pm_agent

        state = SharedState(
            original_requirement="Create a Python script that converts CSV files to JSON format.",
            session_id="e2e_test",
            workspace_path="./workspace/e2e_test",
        )

        result = run_pm_agent(state)

        assert result.phase == "pm_complete"
        assert result.technical_spec is not None
        assert result.technical_spec.name, "Spec must have a name"
        assert len(result.tasks) > 0, "PM must produce at least one task"
        assert len(result.tasks) <= MAX_TASKS_PER_REQUIREMENT, \
            f"PM must not exceed {MAX_TASKS_PER_REQUIREMENT} tasks"

        for task in result.tasks:
            assert task.task_id.startswith("TASK-")
            assert task.description
            assert task.status == TaskStatus.PENDING

    def test_full_pipeline(self):
        """Run the complete PM → Coder pipeline."""
        pipeline = build_default_pipeline()
        state = pipeline.run(
            "Write a Python function that takes a list of numbers and returns "
            "a dictionary with the mean, median, and mode. Save it to stats.py "
            "and test it."
        )

        assert state.session_id, "Must have a session ID"
        assert state.technical_spec is not None, "PM must produce a spec"
        assert len(state.tasks) > 0, "Must have tasks"

        # At least some tasks should be completed
        completed = state.get_completed_tasks()
        assert len(completed) > 0, "Coder must complete at least one task"

        # Accumulated code should be non-empty
        assert state.accumulated_code.strip(), "Must have accumulated code"
