"""
Integration tests for Coder + QA agent pair.

Uses mocked LLM to verify:
  - Coder produces code, QA receives it via A2A.
  - QA fix instructions flow back to Coder.
  - Review loop terminates correctly.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from agents.message_bus import A2AMessageBus
from agents.schemas_a2a import (
    A2AIntent,
    A2AMessage,
    AgentRole,
    AllTestsPassedPayload,
    FixInstruction,
    FixInstructionsPayload,
    ReviewRequestPayload,
)
from agents.schemas_shared import TaskItem, TaskPriority, TaskStatus
from agents.qa_agent import generate_final_report


class TestCoderQAIntegration:
    """Coder, QA integration via A2A protocol."""

    def test_review_request_to_fix_instructions_flow(self, message_bus, correlation_id):
        """Simulate full Coder,QA,Coder message exchange."""
        # Coder sends review request
        review_payload = ReviewRequestPayload(
            task_id="TASK-001",
            code="def add(a, b): return a + b",
            file_path="task_001.py",
            acceptance_criteria=["add(2,3)==5"],
            workspace_path="./workspace/test",
            iteration=1,
        )
        review_msg = A2AMessage(
            correlation_id=correlation_id,
            sender=AgentRole.CODER,
            receiver=AgentRole.QA,
            intent=A2AIntent.REVIEW_REQUEST,
            payload=review_payload.model_dump(),
        )
        message_bus.send(review_msg)

        # QA sends fix instructions
        fix_payload = FixInstructionsPayload(
            task_id="TASK-001",
            iteration=1,
            test_file="test_task_001.py",
            total_tests=2,
            passed=1,
            failed=1,
            fix_instructions=[
                FixInstruction(
                    issue_id="FIX-001",
                    description="Missing type hints",
                    severity="medium",
                    suggested_fix="Add type annotations",
                    related_test="test_types",
                ),
            ],
        )
        fix_msg = A2AMessage(
            correlation_id=correlation_id,
            sender=AgentRole.QA,
            receiver=AgentRole.CODER,
            intent=A2AIntent.FIX_INSTRUCTIONS,
            payload=fix_payload.model_dump(),
        )
        message_bus.send(fix_msg)

        # Verify conversation
        convo = message_bus.get_conversation(correlation_id)
        assert len(convo) == 2
        assert convo[0].intent == A2AIntent.REVIEW_REQUEST
        assert convo[1].intent == A2AIntent.FIX_INSTRUCTIONS

        # Verify Coder can read the fix instructions
        coder_msgs = message_bus.get_messages_for(AgentRole.CODER, correlation_id=correlation_id)
        assert len(coder_msgs) == 1
        assert len(coder_msgs[0].payload["fix_instructions"]) == 1

    def test_review_loop_success_scenario(self, message_bus, correlation_id):
        """Simulate: Coder sends code, QA passes all tests."""
        review_msg = A2AMessage(
            correlation_id=correlation_id,
            sender=AgentRole.CODER,
            receiver=AgentRole.QA,
            intent=A2AIntent.REVIEW_REQUEST,
            payload=ReviewRequestPayload(
                task_id="TASK-001", code="pass", file_path="t.py",
                acceptance_criteria=[], workspace_path="./ws",
            ).model_dump(),
        )
        message_bus.send(review_msg)

        pass_msg = A2AMessage(
            correlation_id=correlation_id,
            sender=AgentRole.QA,
            receiver=AgentRole.CODER,
            intent=A2AIntent.ALL_TESTS_PASSED,
            payload=AllTestsPassedPayload(
                task_id="TASK-001", iteration=1,
                test_file="test.py", total_tests=5, passed=5,
            ).model_dump(),
        )
        message_bus.send(pass_msg)

        latest = message_bus.get_latest_for(
            AgentRole.CODER, correlation_id, A2AIntent.ALL_TESTS_PASSED
        )
        assert latest is not None
        assert latest.payload["passed"] == 5

    def test_review_loop_max_iterations_scenario(self, message_bus, correlation_id):
        """Simulate 5 iterations of failures, final report."""
        for i in range(1, 6):
            message_bus.send(A2AMessage(
                correlation_id=correlation_id,
                sender=AgentRole.CODER,
                receiver=AgentRole.QA,
                intent=A2AIntent.REVIEW_REQUEST,
                payload=ReviewRequestPayload(
                    task_id="TASK-001", code=f"v{i}", file_path="t.py",
                    acceptance_criteria=[], workspace_path="./ws", iteration=i,
                ).model_dump(),
            ))
            message_bus.send(A2AMessage(
                correlation_id=correlation_id,
                sender=AgentRole.QA,
                receiver=AgentRole.CODER,
                intent=A2AIntent.FIX_INSTRUCTIONS,
                payload=FixInstructionsPayload(
                    task_id="TASK-001", iteration=i, test_file="t.py",
                    total_tests=3, passed=1, failed=2,
                    fix_instructions=[
                        FixInstruction(issue_id=f"FIX-{i:03d}", description=f"Issue {i}"),
                    ],
                ).model_dump(),
            ))

        # Final report
        report = generate_final_report(
            task_id="TASK-001", total_iterations=5,
            last_test_output="2 failed",
            unresolved=[FixInstruction(issue_id="FIX-005", description="Still broken")],
            resolved=["FIX-001", "FIX-002", "FIX-003", "FIX-004"],
        )

        message_bus.send(A2AMessage(
            correlation_id=correlation_id,
            sender=AgentRole.QA,
            receiver=AgentRole.CODER,
            intent=A2AIntent.MAX_ITERATIONS_REACHED,
            payload=report.model_dump(),
        ))

        final = message_bus.get_latest_for(
            AgentRole.CODER, correlation_id, A2AIntent.MAX_ITERATIONS_REACHED
        )
        assert final is not None
        assert final.payload["total_iterations"] == 5
        assert len(final.payload["unresolved_issues"]) == 1
        assert len(final.payload["resolved_issues"]) == 4
