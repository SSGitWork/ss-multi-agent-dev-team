"""
Integration tests for PM + Coder agent pair.

Uses fully mocked Crew to avoid Pydantic validation of Agent/Task.
Verifies the handoff protocol works correctly:
  - PM produces tasks → Coder reads them from SharedState.
  - Task status transitions are correct.
  - State is JSON-serializable after PM runs.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.schemas_shared import SharedState, TaskStatus
from tests.conftest import (
    MOCK_CODER_OUTPUT,
    MOCK_PM_OUTPUT,
    MOCK_SELF_REFLECTION_OUTPUT,
    MockCrewOutput,
)


def _make_mock_crew(raw_output: str):
    """Create a mock Crew whose kickoff() returns the given raw text."""
    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = MockCrewOutput(raw_output)
    return mock_crew_instance


class TestPMCoderIntegration:
    """PM, Coder handoff integration tests."""

    @patch("agents.coder_agent.Crew")
    @patch("agents.coder_agent.Task")
    @patch("agents.coder_agent.build_coder_agent")
    @patch("agents.pm_agent.Crew")
    @patch("agents.pm_agent.Task")
    @patch("agents.pm_agent.build_pm_agent")
    def test_pm_to_coder_handoff(
        self,
        mock_pm_build, mock_pm_task_cls, mock_pm_crew_cls,
        mock_coder_build, mock_coder_task_cls, mock_coder_crew_cls,
        sample_state,
    ):
        """After PM runs, Coder should receive tasks and produce code."""
        # PM mocks
        mock_pm_build.return_value = MagicMock()
        mock_pm_task_cls.return_value = MagicMock()
        mock_pm_crew_cls.return_value = _make_mock_crew(MOCK_PM_OUTPUT)

        # Coder mocks — kickoff is called multiple times (code gen + self-reflection per task)
        mock_coder_build.return_value = MagicMock()
        mock_coder_task_cls.return_value = MagicMock()
        mock_coder_crew_cls.return_value = _make_mock_crew(MOCK_CODER_OUTPUT)

        from agents.pm_agent import run_pm_agent
        from agents.coder_agent import run_coder_from_state

        # Step 1: PM runs
        state = run_pm_agent(sample_state)
        assert state.phase == "pm_complete"
        assert len(state.tasks) > 0

        # Step 2: Coder runs
        state = run_coder_from_state(state)
        assert state.phase == "coder_complete"

        # Verify tasks were processed
        completed = state.get_completed_tasks()
        assert len(completed) > 0, "Coder should complete at least one task"

    @patch("agents.pm_agent.Crew")
    @patch("agents.pm_agent.Task")
    @patch("agents.pm_agent.build_pm_agent")
    def test_pm_output_has_correct_task_schema(
        self, mock_build, mock_task_cls, mock_crew_cls, sample_state
    ):
        """Every task from PM must have the handoff protocol fields."""
        mock_build.return_value = MagicMock()
        mock_task_cls.return_value = MagicMock()
        mock_crew_cls.return_value = _make_mock_crew(MOCK_PM_OUTPUT)

        from agents.pm_agent import run_pm_agent
        state = run_pm_agent(sample_state)

        assert state.phase == "pm_complete"
        for task in state.tasks:
            assert task.task_id, "task_id must be non-empty"
            assert task.description, "description must be non-empty"
            assert isinstance(task.acceptance_criteria, list)
            assert task.status == TaskStatus.PENDING

    @patch("agents.pm_agent.Crew")
    @patch("agents.pm_agent.Task")
    @patch("agents.pm_agent.build_pm_agent")
    def test_state_is_json_serializable_after_pm(
        self, mock_build, mock_task_cls, mock_crew_cls, sample_state
    ):
        """The full state after PM must serialize to valid JSON."""
        mock_build.return_value = MagicMock()
        mock_task_cls.return_value = MagicMock()
        mock_crew_cls.return_value = _make_mock_crew(MOCK_PM_OUTPUT)

        from agents.pm_agent import run_pm_agent
        state = run_pm_agent(sample_state)

        json_str = state.to_json()
        data = json.loads(json_str)
        assert data["phase"] == "pm_complete"
        assert len(data["tasks"]) > 0
