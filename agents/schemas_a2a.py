"""
Phase 3 – Agent-to-Agent (A2A) Protocol Schema.

Implements a structured message format for inter-agent communication.
Every message is:
  • Typed with an intent enum (what the sender wants).
  • Correlated via a correlation_id (links request → response chains).
  • Validated on send and receive (intent must match expectations).
  • Fully JSON-serializable for logging, replay, and debugging.

Design decisions:
  • A2A was chosen over MCP because our agents are peers that exchange
    work products (code, test results, fix instructions) rather than
    a client calling a tool server.  A2A's intent-based routing maps
    naturally to the Coder↔QA review cycle.
  • correlation_id lets us trace an entire review loop iteration as a
    single logical conversation even across multiple message hops.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class A2AIntent(str, Enum):
    """The purpose of an A2A message."""

    # Coder → QA
    REVIEW_REQUEST = "review_request"

    # QA → Coder
    FIX_INSTRUCTIONS = "fix_instructions"
    ALL_TESTS_PASSED = "all_tests_passed"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"

    # Generic
    ACKNOWLEDGEMENT = "acknowledgement"
    ERROR = "error"


class AgentRole(str, Enum):
    """Known agent identities in the system."""
    PM = "product_manager"
    CODER = "coder"
    QA = "qa_debugger"
    ORCHESTRATOR = "orchestrator"


# ---------------------------------------------------------------------------
# A2A Message — the core protocol record
# ---------------------------------------------------------------------------
class A2AMessage(BaseModel):
    """A single message in the A2A protocol.

    Required fields for A2A compliance:
      • sender        – who sent the message
      • receiver      – who should process it
      • intent        – what the sender wants
      • payload       – the actual data
      • correlation_id – links related messages in a conversation
    """

    message_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for this message.",
    )
    correlation_id: str = Field(
        ...,
        description="Links all messages in a single review cycle.",
    )
    sender: AgentRole = Field(
        ...,
        description="The agent that created this message.",
    )
    receiver: AgentRole = Field(
        ...,
        description="The agent that should process this message.",
    )
    intent: A2AIntent = Field(
        ...,
        description="The purpose / action requested.",
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Intent-specific data.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of message creation.",
    )
    parent_message_id: Optional[str] = Field(
        default=None,
        description="ID of the message this is replying to.",
    )

    # -- Validation --------------------------------------------------------
    @field_validator("correlation_id")
    @classmethod
    def correlation_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("correlation_id must not be empty")
        return v

    # -- Helpers -----------------------------------------------------------
    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> "A2AMessage":
        return cls.model_validate_json(data)


# ---------------------------------------------------------------------------
# Payload schemas for specific intents
# ---------------------------------------------------------------------------
class ReviewRequestPayload(BaseModel):
    """Payload for REVIEW_REQUEST: Coder → QA."""
    task_id: str
    code: str
    file_path: str
    acceptance_criteria: List[str]
    workspace_path: str
    iteration: int = 1


class TestFailure(BaseModel):
    """A single test failure reported by QA."""
    test_name: str
    error_message: str
    expected: str = ""
    actual: str = ""


class FixInstruction(BaseModel):
    """A single fix instruction from QA to Coder."""
    issue_id: str
    description: str
    severity: str = "medium"  # high, medium, low
    suggested_fix: str = ""
    related_test: str = ""


class FixInstructionsPayload(BaseModel):
    """Payload for FIX_INSTRUCTIONS: QA → Coder."""
    task_id: str
    iteration: int
    test_file: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    failures: List[TestFailure] = Field(default_factory=list)
    fix_instructions: List[FixInstruction] = Field(default_factory=list)
    raw_output: str = ""


class AllTestsPassedPayload(BaseModel):
    """Payload for ALL_TESTS_PASSED: QA → Coder."""
    task_id: str
    iteration: int
    test_file: str
    total_tests: int
    passed: int
    raw_output: str = ""


class FinalQAReport(BaseModel):
    """Final report when max iterations reached without full pass."""
    task_id: str
    total_iterations: int
    final_test_results: str
    unresolved_issues: List[FixInstruction] = Field(default_factory=list)
    resolved_issues: List[str] = Field(default_factory=list)
    recommendation: str = ""


# ---------------------------------------------------------------------------
# Intent validation helper
# ---------------------------------------------------------------------------
VALID_INTENT_ROUTES: Dict[tuple, list] = {
    (AgentRole.CODER, AgentRole.QA): [A2AIntent.REVIEW_REQUEST],
    (AgentRole.QA, AgentRole.CODER): [
        A2AIntent.FIX_INSTRUCTIONS,
        A2AIntent.ALL_TESTS_PASSED,
        A2AIntent.MAX_ITERATIONS_REACHED,
    ],
}


def validate_message_route(message: A2AMessage) -> bool:
    """Check that the intent is valid for the sender→receiver pair."""
    route = (message.sender, message.receiver)
    allowed = VALID_INTENT_ROUTES.get(route, [])
    return message.intent in allowed
