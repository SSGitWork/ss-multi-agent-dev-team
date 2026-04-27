"""
Unit tests for A2A message bus.
"""

from __future__ import annotations

import uuid

import pytest

from agents.message_bus import A2AMessageBus
from agents.schemas_a2a import (
    A2AIntent,
    A2AMessage,
    AgentRole,
    ReviewRequestPayload,
)


class TestA2AMessageBus:
    def test_send_and_retrieve(self, message_bus, sample_review_request):
        message_bus.send(sample_review_request)
        msgs = message_bus.get_messages_for(AgentRole.QA)
        assert len(msgs) == 1
        assert msgs[0].intent == A2AIntent.REVIEW_REQUEST

    def test_filter_by_correlation_id(self, message_bus, sample_review_request, sample_fix_response):
        message_bus.send(sample_review_request)
        message_bus.send(sample_fix_response)
        corr = sample_review_request.correlation_id
        msgs = message_bus.get_messages_for(AgentRole.CODER, correlation_id=corr)
        assert len(msgs) == 1
        assert msgs[0].intent == A2AIntent.FIX_INSTRUCTIONS

    def test_filter_by_intent(self, message_bus, sample_fix_response, sample_all_passed):
        message_bus.send(sample_fix_response)
        message_bus.send(sample_all_passed)
        msgs = message_bus.get_messages_for(AgentRole.CODER, intent=A2AIntent.ALL_TESTS_PASSED)
        assert len(msgs) == 1

    def test_get_latest(self, message_bus, correlation_id):
        for i in range(3):
            payload = ReviewRequestPayload(
                task_id="TASK-001", code=f"v{i}", file_path="t.py",
                acceptance_criteria=[], workspace_path="./ws", iteration=i + 1,
            )
            msg = A2AMessage(
                correlation_id=correlation_id, sender=AgentRole.CODER,
                receiver=AgentRole.QA, intent=A2AIntent.REVIEW_REQUEST,
                payload=payload.model_dump(),
            )
            message_bus.send(msg)
        latest = message_bus.get_latest_for(AgentRole.QA, correlation_id)
        assert latest is not None
        assert latest.payload["iteration"] == 3

    def test_get_conversation(self, message_bus, sample_review_request, sample_fix_response):
        message_bus.send(sample_review_request)
        message_bus.send(sample_fix_response)
        convo = message_bus.get_conversation(sample_review_request.correlation_id)
        assert len(convo) == 2

    def test_invalid_route_raises(self, message_bus, correlation_id):
        bad_msg = A2AMessage(
            correlation_id=correlation_id, sender=AgentRole.CODER,
            receiver=AgentRole.QA, intent=A2AIntent.FIX_INSTRUCTIONS,
        )
        with pytest.raises(ValueError, match="Invalid A2A route"):
            message_bus.send(bad_msg)

    def test_clear(self, message_bus, sample_review_request):
        message_bus.send(sample_review_request)
        assert message_bus.message_count == 1
        message_bus.clear()
        assert message_bus.message_count == 0

    def test_summary(self, message_bus, sample_review_request):
        message_bus.send(sample_review_request)
        summary = message_bus.summary()
        assert "coder" in summary.lower()
