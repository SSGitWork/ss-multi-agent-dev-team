"""
Phase 3 Tests – QA Agent, Review Loop, A2A Protocol.

Test categories:
  1. A2A message schema and validation.
  2. A2A message bus routing.
  3. QA agent output schema (mocked code input).
  4. Review loop termination (success and max-iteration).
  5. Self-reflection produces revised output.
  6. Final QA report structure.
  7. End-to-end Phase 3 pipeline (requires LLM).
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

from agents.message_bus import A2AMessageBus
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
from agents.schemas_shared import (
    SharedState,
    TaskItem,
    TaskPriority,
    TaskStatus,
    TechnicalSpec,
)
from agents.qa_agent import generate_final_report


# ── Fixtures ──────────────────────────────────────────────────────────────

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
# 1. A2A Message Schema Tests
# ══════════════════════════════════════════════════════════════════════════

class TestA2AMessageSchema:
    """Validate the A2A message structure."""

    def test_message_has_required_fields(self, sample_review_request):
        msg = sample_review_request
        assert msg.message_id, "message_id must be non-empty"
        assert msg.correlation_id, "correlation_id must be non-empty"
        assert isinstance(msg.sender, AgentRole)
        assert isinstance(msg.receiver, AgentRole)
        assert isinstance(msg.intent, A2AIntent)
        assert isinstance(msg.payload, dict)
        assert msg.timestamp, "timestamp must be non-empty"

    def test_message_json_serialization(self, sample_review_request):
        json_str = sample_review_request.to_json()
        restored = A2AMessage.from_json(json_str)
        assert restored.correlation_id == sample_review_request.correlation_id
        assert restored.sender == sample_review_request.sender
        assert restored.intent == sample_review_request.intent

    def test_correlation_id_cannot_be_empty(self):
        with pytest.raises(Exception):
            A2AMessage(
                correlation_id="",
                sender=AgentRole.CODER,
                receiver=AgentRole.QA,
                intent=A2AIntent.REVIEW_REQUEST,
            )

    def test_parent_message_id_optional(self, correlation_id):
        msg = A2AMessage(
            correlation_id=correlation_id,
            sender=AgentRole.CODER,
            receiver=AgentRole.QA,
            intent=A2AIntent.REVIEW_REQUEST,
            parent_message_id="parent123",
        )
        assert msg.parent_message_id == "parent123"


# ══════════════════════════════════════════════════════════════════════════
# 2. A2A Intent Validation Tests
# ══════════════════════════════════════════════════════════════════════════

class TestA2AIntentValidation:
    """Validate that intents are only allowed on correct routes."""

    def test_coder_to_qa_review_request_valid(self, sample_review_request):
        assert validate_message_route(sample_review_request) is True

    def test_qa_to_coder_fix_instructions_valid(self, sample_fix_response):
        assert validate_message_route(sample_fix_response) is True

    def test_qa_to_coder_all_passed_valid(self, sample_all_passed):
        assert validate_message_route(sample_all_passed) is True

    def test_invalid_route_coder_to_qa_fix(self, correlation_id):
        """Coder cannot send FIX_INSTRUCTIONS to QA."""
        msg = A2AMessage(
            correlation_id=correlation_id,
            sender=AgentRole.CODER,
            receiver=AgentRole.QA,
            intent=A2AIntent.FIX_INSTRUCTIONS,
        )
        assert validate_message_route(msg) is False

    def test_invalid_route_qa_to_coder_review(self, correlation_id):
        """QA cannot send REVIEW_REQUEST to Coder."""
        msg = A2AMessage(
            correlation_id=correlation_id,
            sender=AgentRole.QA,
            receiver=AgentRole.CODER,
            intent=A2AIntent.REVIEW_REQUEST,
        )
        assert validate_message_route(msg) is False


# ══════════════════════════════════════════════════════════════════════════
# 3. A2A Message Bus Tests
# ══════════════════════════════════════════════════════════════════════════

class TestA2AMessageBus:
    """Validate the message bus routing and retrieval."""

    def test_send_and_retrieve(self, message_bus, sample_review_request):
        message_bus.send(sample_review_request)
        msgs = message_bus.get_messages_for(AgentRole.QA)
        assert len(msgs) == 1
        assert msgs[0].intent == A2AIntent.REVIEW_REQUEST

    def test_filter_by_correlation_id(
        self, message_bus, sample_review_request, sample_fix_response
    ):
        message_bus.send(sample_review_request)
        message_bus.send(sample_fix_response)

        corr = sample_review_request.correlation_id
        msgs = message_bus.get_messages_for(
            AgentRole.CODER, correlation_id=corr
        )
        # Only the fix_response is addressed to CODER
        assert len(msgs) == 1
        assert msgs[0].intent == A2AIntent.FIX_INSTRUCTIONS

    def test_filter_by_intent(self, message_bus, sample_fix_response, sample_all_passed):
        message_bus.send(sample_fix_response)
        message_bus.send(sample_all_passed)

        msgs = message_bus.get_messages_for(
            AgentRole.CODER, intent=A2AIntent.ALL_TESTS_PASSED
        )
        assert len(msgs) == 1

    def test_get_latest(self, message_bus, correlation_id):
        for i in range(3):
            payload = ReviewRequestPayload(
                task_id="TASK-001",
                code=f"version_{i}",
                file_path="test.py",
                acceptance_criteria=[],
                workspace_path="./workspace/test",
                iteration=i + 1,
            )
            msg = A2AMessage(
                correlation_id=correlation_id,
                sender=AgentRole.CODER,
                receiver=AgentRole.QA,
                intent=A2AIntent.REVIEW_REQUEST,
                payload=payload.model_dump(),
            )
            message_bus.send(msg)

        latest = message_bus.get_latest_for(AgentRole.QA, correlation_id)
        assert latest is not None
        assert latest.payload["iteration"] == 3

    def test_get_conversation(
        self, message_bus, sample_review_request, sample_fix_response
    ):
        message_bus.send(sample_review_request)
        message_bus.send(sample_fix_response)

        corr = sample_review_request.correlation_id
        convo = message_bus.get_conversation(corr)
        assert len(convo) == 2
        assert convo[0].sender == AgentRole.CODER
        assert convo[1].sender == AgentRole.QA

    def test_invalid_route_raises(self, message_bus, correlation_id):
        bad_msg = A2AMessage(
            correlation_id=correlation_id,
            sender=AgentRole.CODER,
            receiver=AgentRole.QA,
            intent=A2AIntent.FIX_INSTRUCTIONS,
        )
        with pytest.raises(ValueError, match="Invalid A2A route"):
            message_bus.send(bad_msg)

    def test_message_count_and_clear(self, message_bus, sample_review_request):
        message_bus.send(sample_review_request)
        assert message_bus.message_count == 1
        message_bus.clear()
        assert message_bus.message_count == 0

    def test_summary(self, message_bus, sample_review_request):
        message_bus.send(sample_review_request)
        summary = message_bus.summary()
        assert "coder" in summary.lower()
        assert "review_request" in summary


# ══════════════════════════════════════════════════════════════════════════
# 4. QA Agent Output Schema Tests (mocked code input)
# ══════════════════════════════════════════════════════════════════════════

class TestQAOutputSchema:
    """Validate QA produces correctly structured output."""

    def test_fix_instructions_payload_schema(self):
        payload = FixInstructionsPayload(
            task_id="TASK-001",
            iteration=1,
            test_file="test_task_001.py",
            total_tests=5,
            passed=3,
            failed=2,
            failures=[
                TestFailure(
                    test_name="test_edge_case",
                    error_message="AssertionError",
                ),
            ],
            fix_instructions=[
                FixInstruction(
                    issue_id="FIX-001",
                    description="Edge case not handled",
                    severity="high",
                    suggested_fix="Add boundary check",
                    related_test="test_edge_case",
                ),
            ],
        )
        assert payload.task_id == "TASK-001"
        assert len(payload.failures) == 1
        assert len(payload.fix_instructions) == 1
        assert payload.fix_instructions[0].issue_id == "FIX-001"

    def test_fix_instruction_has_required_fields(self):
        fix = FixInstruction(
            issue_id="FIX-001",
            description="Something is wrong",
            severity="high",
            suggested_fix="Fix it",
            related_test="test_something",
        )
        assert fix.issue_id
        assert fix.description
        assert fix.severity in ("high", "medium", "low")

    def test_all_tests_passed_payload(self):
        payload = AllTestsPassedPayload(
            task_id="TASK-001",
            iteration=2,
            test_file="test_task_001.py",
            total_tests=5,
            passed=5,
        )
        assert payload.passed == payload.total_tests

    def test_fix_payload_json_serializable(self):
        payload = FixInstructionsPayload(
            task_id="TASK-001",
            iteration=1,
            test_file="test.py",
            fix_instructions=[
                FixInstruction(
                    issue_id="FIX-001",
                    description="Bug found",
                ),
            ],
        )
        json_str = payload.model_dump_json()
        restored = FixInstructionsPayload.model_validate_json(json_str)
        assert restored.fix_instructions[0].issue_id == "FIX-001"


# ══════════════════════════════════════════════════════════════════════════
# 5. Final QA Report Tests
# ══════════════════════════════════════════════════════════════════════════

class TestFinalQAReport:
    """Validate the convergence safeguard report."""

    def test_final_report_with_unresolved(self):
        report = generate_final_report(
            task_id="TASK-001",
            total_iterations=5,
            last_test_output="2 failed, 3 passed",
            unresolved=[
                FixInstruction(
                    issue_id="FIX-003",
                    description="Timeout not handled",
                    severity="high",
                ),
            ],
            resolved=["FIX-001", "FIX-002"],
        )
        assert report.task_id == "TASK-001"
        assert report.total_iterations == 5
        assert len(report.unresolved_issues) == 1
        assert len(report.resolved_issues) == 2
        assert "FIX-003" in report.recommendation

    def test_final_report_all_resolved(self):
        report = generate_final_report(
            task_id="TASK-001",
            total_iterations=3,
            last_test_output="5 passed",
            unresolved=[],
            resolved=["FIX-001", "FIX-002"],
        )
        assert len(report.unresolved_issues) == 0
        assert "resolved" in report.recommendation.lower()

    def test_final_report_json_serializable(self):
        report = generate_final_report(
            task_id="TASK-001",
            total_iterations=5,
            last_test_output="output",
            unresolved=[],
            resolved=[],
        )
        json_str = report.model_dump_json()
        restored = FinalQAReport.model_validate_json(json_str)
        assert restored.task_id == "TASK-001"


# ══════════════════════════════════════════════════════════════════════════
# 6. Review Loop Termination Tests (mocked)
# ══════════════════════════════════════════════════════════════════════════

class TestReviewLoopTermination:
    """Test that the review loop terminates correctly in both scenarios."""

    def test_loop_terminates_on_success(self, message_bus, correlation_id):
        """Simulate: QA sends ALL_TESTS_PASSED on first iteration."""
        # Create a task
        task = TaskItem(
            task_id="TASK-001",
            title="Test task",
            description="Write a function",
            acceptance_criteria=["It works"],
            priority=TaskPriority.HIGH,
        )

        # Simulate the success message
        payload = AllTestsPassedPayload(
            task_id="TASK-001",
            iteration=1,
            test_file="test_task_001.py",
            total_tests=3,
            passed=3,
        )
        msg = A2AMessage(
            correlation_id=correlation_id,
            sender=AgentRole.QA,
            receiver=AgentRole.CODER,
            intent=A2AIntent.ALL_TESTS_PASSED,
            payload=payload.model_dump(),
        )
        message_bus.send(msg)

        # Verify the message is retrievable
        latest = message_bus.get_latest_for(
            AgentRole.CODER, correlation_id, A2AIntent.ALL_TESTS_PASSED
        )
        assert latest is not None
        assert latest.payload["passed"] == 3

    def test_loop_terminates_on_max_iterations(self, message_bus, correlation_id):
        """Simulate: QA sends FIX_INSTRUCTIONS for 5 iterations."""
        for i in range(1, 6):
            payload = FixInstructionsPayload(
                task_id="TASK-001",
                iteration=i,
                test_file="test_task_001.py",
                total_tests=3,
                passed=1,
                failed=2,
                fix_instructions=[
                    FixInstruction(
                        issue_id=f"FIX-{i:03d}",
                        description=f"Issue from iteration {i}",
                    ),
                ],
            )
            msg = A2AMessage(
                correlation_id=correlation_id,
                sender=AgentRole.QA,
                receiver=AgentRole.CODER,
                intent=A2AIntent.FIX_INSTRUCTIONS,
                payload=payload.model_dump(),
            )
            message_bus.send(msg)

        # Verify all 5 messages exist
        msgs = message_bus.get_messages_for(
            AgentRole.CODER, correlation_id=correlation_id
        )
        assert len(msgs) == 5

        # Simulate max-iterations-reached
        report = generate_final_report(
            task_id="TASK-001",
            total_iterations=5,
            last_test_output="2 failed",
            unresolved=[
                FixInstruction(issue_id="FIX-005", description="Still broken"),
            ],
            resolved=["FIX-001", "FIX-002", "FIX-003", "FIX-004"],
        )
        max_msg = A2AMessage(
            correlation_id=correlation_id,
            sender=AgentRole.QA,
            receiver=AgentRole.CODER,
            intent=A2AIntent.MAX_ITERATIONS_REACHED,
            payload=report.model_dump(),
        )
        message_bus.send(max_msg)

        final = message_bus.get_latest_for(
            AgentRole.CODER, correlation_id, A2AIntent.MAX_ITERATIONS_REACHED
        )
        assert final is not None
        assert final.payload["total_iterations"] == 5
        assert len(final.payload["unresolved_issues"]) == 1

    def test_shared_state_after_review_complete(self):
        """Verify SharedState shape after review loop completes."""
        state = SharedState(original_requirement="Build something")
        state.tasks = [
            TaskItem(
                task_id="TASK-001",
                title="Task 1",
                description="Do something",
                acceptance_criteria=["It works"],
                status=TaskStatus.COMPLETED,
                code_output="print('hello')",
                execution_result="All 3 tests passed on iteration 2.",
            ),
        ]
        state.phase = "review_complete"
        state.accumulated_code = "print('hello')"

        assert state.phase == "review_complete"
        assert state.all_tasks_done()
        assert state.accumulated_code.strip()


# ══════════════════════════════════════════════════════════════════════════
# 7. Review Request / Response Payload Tests
# ══════════════════════════════════════════════════════════════════════════

class TestPayloadSchemas:
    """Validate all A2A payload schemas."""

    def test_review_request_payload(self):
        p = ReviewRequestPayload(
            task_id="TASK-001",
            code="def foo(): pass",
            file_path="foo.py",
            acceptance_criteria=["foo() returns None"],
            workspace_path="./workspace/test",
            iteration=1,
        )
        assert p.task_id == "TASK-001"
        assert p.iteration == 1
        data = p.model_dump()
        assert "code" in data
        assert "acceptance_criteria" in data

    def test_test_failure_schema(self):
        f = TestFailure(
            test_name="test_foo",
            error_message="AssertionError",
            expected="None",
            actual="42",
        )
        assert f.test_name == "test_foo"
        assert f.expected == "None"

    def test_fix_instruction_schema(self):
        fix = FixInstruction(
            issue_id="FIX-001",
            description="Bug in foo",
            severity="high",
            suggested_fix="Return None explicitly",
            related_test="test_foo",
        )
        data = fix.model_dump()
        assert all(
            k in data
            for k in ["issue_id", "description", "severity", "suggested_fix", "related_test"]
        )


# ══════════════════════════════════════════════════════════════════════════
# 8. End-to-End Phase 3 Test (requires LLM)
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not os.getenv("AZURE_API_KEY"),
    reason="AZURE_API_KEY not set – skipping live LLM test.",
)
class TestPhase3E2E:
    def test_full_pipeline_with_qa(self):
        """Run the complete PM → Coder ↔ QA pipeline."""
        from orchestration.graph import build_default_pipeline

        pipeline = build_default_pipeline()
        state = pipeline.run(
            "Write a Python function called 'is_palindrome' that checks if a "
            "string is a palindrome. It should ignore case and spaces. "
            "Save it to palindrome.py."
        )

        assert state.session_id, "Must have a session ID"
        assert state.technical_spec is not None, "PM must produce a spec"
        assert len(state.tasks) > 0, "Must have tasks"
        assert state.accumulated_code.strip(), "Must have accumulated code"

        # Check that A2A messages were logged
        assert "A2A Message" in state.pm_notes or state.phase == "review_complete"