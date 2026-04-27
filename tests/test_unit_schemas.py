"""
Unit tests for all Pydantic schemas.
"""

from __future__ import annotations

import json

import pytest

from agents.schemas import CoderAgentOutput
from agents.schemas_shared import (
    AgentCostRecord,
    PipelineCostReport,
    SharedState,
    TaskItem,
    TaskPriority,
    TaskStatus,
    TechnicalSpec,
)
from agents.schemas_a2a import (
    A2AIntent,
    A2AMessage,
    AgentRole,
    AllTestsPassedPayload,
    FinalQAReport,
    FixInstruction,
    FixInstructionsPayload,
    ReviewRequestPayload,
    TestFailure,
    validate_message_route,
)


class TestCoderAgentOutput:
    def test_required_fields_present(self):
        output = CoderAgentOutput(
            code="print('hello')",
            explanation="Prints hello",
            plan="1. Write code\n2. Run it",
            result="hello",
            session_id="abc123",
        )
        assert output.code == "print('hello')"
        assert output.success is True

    def test_model_has_minimum_four_fields(self):
        fields = CoderAgentOutput.model_fields
        required = {"code", "explanation", "plan", "result"}
        assert required.issubset(set(fields.keys()))

    def test_failure_output(self):
        output = CoderAgentOutput(
            code="", explanation="", plan="", result="",
            session_id="fail123", success=False, error="Something went wrong",
        )
        assert output.success is False
        assert output.error == "Something went wrong"


class TestSharedState:
    def test_minimal_creation(self):
        state = SharedState(original_requirement="Build something")
        assert state.technical_spec is None
        assert state.tasks == []
        assert state.phase == "initialized"
        assert state.success is True

    def test_json_roundtrip(self, pm_completed_state):
        json_str = pm_completed_state.to_json()
        restored = SharedState.from_json(json_str)
        assert restored.original_requirement == pm_completed_state.original_requirement
        assert len(restored.tasks) == len(pm_completed_state.tasks)

    def test_get_pending_tasks(self, pm_completed_state):
        pending = pm_completed_state.get_pending_tasks()
        assert len(pending) == 3
        assert all(t.status == TaskStatus.PENDING for t in pending)

    def test_get_completed_tasks(self, pm_completed_state):
        pm_completed_state.tasks[0].status = TaskStatus.COMPLETED
        assert len(pm_completed_state.get_completed_tasks()) == 1

    def test_all_tasks_done(self, pm_completed_state):
        assert not pm_completed_state.all_tasks_done()
        for t in pm_completed_state.tasks:
            t.status = TaskStatus.COMPLETED
        assert pm_completed_state.all_tasks_done()

    def test_all_tasks_done_with_failures(self, pm_completed_state):
        for t in pm_completed_state.tasks:
            t.status = TaskStatus.COMPLETED
        pm_completed_state.tasks[-1].status = TaskStatus.FAILED
        assert pm_completed_state.all_tasks_done()


class TestTaskItem:
    def test_defaults(self):
        task = TaskItem(task_id="TASK-001", title="Test", description="Do something")
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.MEDIUM
        assert task.code_output == ""

    def test_handoff_protocol_fields(self):
        task = TaskItem(
            task_id="TASK-001", title="Test", description="Do something",
            acceptance_criteria=["It works"], priority=TaskPriority.HIGH,
        )
        assert isinstance(task.task_id, str)
        assert isinstance(task.description, str)
        assert isinstance(task.acceptance_criteria, list)
        assert isinstance(task.priority, TaskPriority)

    def test_json_serializable(self):
        task = TaskItem(task_id="TASK-001", title="Test", description="Do something")
        json_str = task.model_dump_json()
        restored = TaskItem.model_validate_json(json_str)
        assert restored.task_id == task.task_id


class TestCostSchemas:
    def test_agent_cost_record(self):
        record = AgentCostRecord(
            agent_name="pm_agent", model_name="azure/gpt-4o",
            prompt_tokens=500, completion_tokens=200, total_tokens=700,
            estimated_cost_usd=0.0032, llm_calls=1,
        )
        assert record.total_tokens == 700

    def test_pipeline_cost_report(self):
        report = PipelineCostReport(
            session_id="test123",
            agents={"pm": AgentCostRecord(agent_name="pm")},
            total_tokens=1000, total_cost_usd=0.005,
        )
        json_str = report.model_dump_json()
        restored = PipelineCostReport.model_validate_json(json_str)
        assert restored.session_id == "test123"


class TestA2ASchemas:
    def test_message_required_fields(self, sample_review_request):
        msg = sample_review_request
        assert msg.message_id
        assert msg.correlation_id
        assert isinstance(msg.sender, AgentRole)
        assert isinstance(msg.receiver, AgentRole)
        assert isinstance(msg.intent, A2AIntent)

    def test_message_json_roundtrip(self, sample_review_request):
        json_str = sample_review_request.to_json()
        restored = A2AMessage.from_json(json_str)
        assert restored.correlation_id == sample_review_request.correlation_id

    def test_correlation_id_cannot_be_empty(self):
        with pytest.raises(Exception):
            A2AMessage(
                correlation_id="", sender=AgentRole.CODER,
                receiver=AgentRole.QA, intent=A2AIntent.REVIEW_REQUEST,
            )

    def test_valid_routes(self, sample_review_request, sample_fix_response, sample_all_passed):
        assert validate_message_route(sample_review_request) is True
        assert validate_message_route(sample_fix_response) is True
        assert validate_message_route(sample_all_passed) is True

    def test_invalid_route(self, correlation_id):
        msg = A2AMessage(
            correlation_id=correlation_id, sender=AgentRole.CODER,
            receiver=AgentRole.QA, intent=A2AIntent.FIX_INSTRUCTIONS,
        )
        assert validate_message_route(msg) is False

    def test_payload_schemas(self):
        rr = ReviewRequestPayload(
            task_id="TASK-001", code="pass", file_path="t.py",
            acceptance_criteria=["works"], workspace_path="./ws",
        )
        assert rr.iteration == 1

        tf = TestFailure(test_name="test_x", error_message="fail")
        assert tf.test_name == "test_x"

        fi = FixInstruction(issue_id="FIX-001", description="bug")
        assert fi.severity == "medium"

        ap = AllTestsPassedPayload(
            task_id="T", iteration=1, test_file="t.py", total_tests=3, passed=3,
        )
        assert ap.passed == ap.total_tests

    def test_final_qa_report(self):
        report = FinalQAReport(
            task_id="TASK-001", total_iterations=5,
            final_test_results="2 failed",
            unresolved_issues=[FixInstruction(issue_id="FIX-001", description="bug")],
            resolved_issues=["FIX-002"],
            recommendation="Manual review needed",
        )
        json_str = report.model_dump_json()
        restored = FinalQAReport.model_validate_json(json_str)
        assert len(restored.unresolved_issues) == 1
