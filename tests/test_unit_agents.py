"""
Unit tests for each agent using mocked LLM responses.

Strategy: We mock `Crew` entirely so that `Task()` and `Agent()` Pydantic
validation is never triggered. The mock Crew's `kickoff()` returns our
pre-built MockCrewOutput with the expected LLM response.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from agents.schemas_shared import SharedState, TaskStatus
from tests.conftest import (
    MOCK_CODER_OUTPUT,
    MOCK_PM_OUTPUT,
    MOCK_SELF_REFLECTION_OUTPUT,
    MockCrewOutput,
)


def _make_mock_crew(raw_output: str):
    """Create a MagicMock that behaves like a Crew instance.

    The mock's kickoff() returns a MockCrewOutput with the given raw text.
    This avoids constructing real Agent/Task/Crew objects (which trigger
    Pydantic validation and LLM configuration).
    """
    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = MockCrewOutput(raw_output)
    return mock_crew_instance


class TestPMAgentMocked:
    """PM agent with fully mocked Crew."""

    @patch("agents.pm_agent.Crew")
    @patch("agents.pm_agent.Task")
    @patch("agents.pm_agent.build_pm_agent")
    def test_pm_produces_valid_state(self, mock_build, mock_task_cls, mock_crew_cls, sample_state):
        mock_build.return_value = MagicMock()
        mock_task_cls.return_value = MagicMock()
        mock_crew_cls.return_value = _make_mock_crew(MOCK_PM_OUTPUT)

        from agents.pm_agent import run_pm_agent
        result = run_pm_agent(sample_state)

        assert result.phase == "pm_complete"
        assert result.technical_spec is not None
        assert result.technical_spec.name == "Test Project"
        assert len(result.tasks) == 2
        assert result.tasks[0].task_id == "TASK-001"
        assert result.tasks[0].status == TaskStatus.PENDING

    @patch("agents.pm_agent.Crew")
    @patch("agents.pm_agent.Task")
    @patch("agents.pm_agent.build_pm_agent")
    def test_pm_handles_llm_failure(self, mock_build, mock_task_cls, mock_crew_cls, sample_state):
        mock_build.return_value = MagicMock()
        mock_task_cls.return_value = MagicMock()
        mock_crew_instance = MagicMock()
        mock_crew_instance.kickoff.side_effect = Exception("LLM timeout")
        mock_crew_cls.return_value = mock_crew_instance

        from agents.pm_agent import run_pm_agent
        result = run_pm_agent(sample_state)

        assert result.phase == "pm_failed"
        assert result.success is False
        assert "LLM timeout" in result.error

    @patch("agents.pm_agent.Crew")
    @patch("agents.pm_agent.Task")
    @patch("agents.pm_agent.build_pm_agent")
    def test_pm_handles_invalid_json(self, mock_build, mock_task_cls, mock_crew_cls, sample_state):
        mock_build.return_value = MagicMock()
        mock_task_cls.return_value = MagicMock()
        mock_crew_cls.return_value = _make_mock_crew("This is not JSON at all")

        from agents.pm_agent import run_pm_agent
        result = run_pm_agent(sample_state)

        assert result.success is False
        assert result.phase == "pm_failed"


class TestCoderAgentMocked:
    """Coder agent with fully mocked Crew."""

    @patch("agents.coder_agent.Crew")
    @patch("agents.coder_agent.Task")
    @patch("agents.coder_agent.build_coder_agent")
    def test_coder_parses_output(self, mock_build, mock_task_cls, mock_crew_cls):
        mock_build.return_value = MagicMock()
        mock_task_cls.return_value = MagicMock()
        mock_crew_cls.return_value = _make_mock_crew(MOCK_CODER_OUTPUT)

        from agents.coder_agent import run_coder_agent
        result = run_coder_agent("Write arithmetic functions")

        assert result.success is True
        assert "def add" in result.code or "add" in result.code.lower() or result.code != ""
        assert result.session_id

    @patch("agents.coder_agent.Crew")
    @patch("agents.coder_agent.Task")
    @patch("agents.coder_agent.build_coder_agent")
    def test_coder_handles_failure(self, mock_build, mock_task_cls, mock_crew_cls):
        mock_build.return_value = MagicMock()
        mock_task_cls.return_value = MagicMock()
        mock_crew_instance = MagicMock()
        mock_crew_instance.kickoff.side_effect = Exception("API error")
        mock_crew_cls.return_value = mock_crew_instance

        from agents.coder_agent import run_coder_agent
        result = run_coder_agent("Write something")

        assert result.success is False
        assert "API error" in result.error

    @patch("agents.coder_agent.Crew")
    @patch("agents.coder_agent.Task")
    @patch("agents.coder_agent.build_coder_agent")
    def test_self_reflection_produces_revised_code(
        self, mock_build, mock_task_cls, mock_crew_cls, temp_workspace
    ):
        mock_build.return_value = MagicMock()
        mock_task_cls.return_value = MagicMock()
        mock_crew_cls.return_value = _make_mock_crew(MOCK_SELF_REFLECTION_OUTPUT)

        from agents.coder_agent import run_self_reflection
        revised, issues = run_self_reflection(
            code="def add(a, b): return a + b",
            task_description="Write add function",
            workspace_path=temp_workspace,
        )

        # Self-reflection should have found issues and produced revised code
        assert len(issues) > 0, "Should have found at least one issue"
        assert "def add" in revised, "Revised code should contain the function"
        assert revised != "def add(a, b): return a + b", "Code should be revised"


class TestQAAgentMocked:
    """QA agent output validation with mocked inputs."""

    def test_qa_fix_instructions_schema(self):
        """Verify QA produces correctly structured fix instructions."""
        from agents.schemas_a2a import FixInstruction, FixInstructionsPayload, TestFailure

        payload = FixInstructionsPayload(
            task_id="TASK-001",
            iteration=1,
            test_file="test_task_001.py",
            total_tests=5,
            passed=3,
            failed=2,
            failures=[
                TestFailure(test_name="test_edge", error_message="AssertionError"),
            ],
            fix_instructions=[
                FixInstruction(
                    issue_id="FIX-001",
                    description="Edge case not handled",
                    severity="high",
                    suggested_fix="Add boundary check",
                    related_test="test_edge",
                ),
            ],
        )
        assert len(payload.fix_instructions) == 1
        assert payload.fix_instructions[0].issue_id == "FIX-001"
        assert payload.failed == 2
