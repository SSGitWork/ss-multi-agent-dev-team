"""
Phase 2 – Shared State & Handoff Protocol Schemas.

Defines the Pydantic models that form the contract between the
Product Manager and Coder agents.  Every field is typed and
JSON-serializable so the handoff protocol can be inspected,
logged, and tested deterministically.

Key design decisions:
  • SharedState is the single source of truth — no ad-hoc dicts.
  • TaskItem carries its own acceptance_criteria so the Coder can
    self-validate without referring back to the spec.
  • TaskStatus is an enum to prevent typos in status strings.
  • The PM must consolidate if it produces > MAX_TASKS (8).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_TASKS_PER_REQUIREMENT: int = 8


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class TaskStatus(str, Enum):
    """Lifecycle states for a single task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(str, Enum):
    """Priority levels for task ordering."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Handoff Protocol — individual task
# ---------------------------------------------------------------------------
class TaskItem(BaseModel):
    """A single discrete task handed from the PM to the Coder.

    This is the *handoff protocol record*.  The Coder reads tasks
    whose status == PENDING and processes them in priority / order.
    """
    task_id: str = Field(
        ...,
        description="Unique identifier for this task (e.g. 'TASK-001').",
    )
    title: str = Field(
        ...,
        description="Short human-readable title.",
    )
    description: str = Field(
        ...,
        description="Detailed description of what the Coder must implement.",
    )
    acceptance_criteria: List[str] = Field(
        default_factory=list,
        description="List of criteria that must be met for the task to pass.",
    )
    priority: TaskPriority = Field(
        default=TaskPriority.MEDIUM,
        description="Execution priority.",
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current lifecycle status.",
    )
    code_output: str = Field(
        default="",
        description="Code produced by the Coder for this task.",
    )
    execution_result: str = Field(
        default="",
        description="stdout/stderr from running the code.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the task failed.",
    )
    completed_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp when the task was completed.",
    )


# ---------------------------------------------------------------------------
# Technical Specification
# ---------------------------------------------------------------------------
class TechnicalSpec(BaseModel):
    """Structured technical specification produced by the PM agent."""
    name: str = Field(
        ...,
        description="Project / feature name.",
    )
    description: str = Field(
        ...,
        description="High-level description of what will be built.",
    )
    acceptance_criteria: List[str] = Field(
        default_factory=list,
        description="Overall acceptance criteria for the entire requirement.",
    )
    technical_approach: str = Field(
        default="",
        description="Recommended technical approach / architecture notes.",
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Any constraints or limitations to observe.",
    )


# ---------------------------------------------------------------------------
# Shared State — the single source of truth
# ---------------------------------------------------------------------------
class SharedState(BaseModel):
    """The shared state object that both PM and Coder read/write.

    Lifecycle:
        1. User fills `original_requirement`.
        2. PM fills `technical_spec` and `tasks`.
        3. Coder iterates over `tasks`, updating status + code_output.
        4. Orchestrator reads the final state and returns it.
    """

    # -- Input -------------------------------------------------------------
    original_requirement: str = Field(
        ...,
        description="The raw user requirement text.",
    )

    # -- PM output ---------------------------------------------------------
    technical_spec: Optional[TechnicalSpec] = Field(
        default=None,
        description="Structured spec produced by the PM agent.",
    )
    tasks: List[TaskItem] = Field(
        default_factory=list,
        description="Ordered list of tasks for the Coder.",
    )
    pm_notes: str = Field(
        default="",
        description="Any additional notes from the PM.",
    )

    # -- Coder output ------------------------------------------------------
    accumulated_code: str = Field(
        default="",
        description="All code produced across all tasks, concatenated.",
    )
    final_explanation: str = Field(
        default="",
        description="Coder's summary explanation of the full implementation.",
    )

    # -- Metadata ----------------------------------------------------------
    session_id: str = Field(
        default="",
        description="Unique session identifier for workspace isolation.",
    )
    workspace_path: str = Field(
        default="",
        description="Absolute path to the session workspace.",
    )
    phase: str = Field(
        default="initialized",
        description="Current pipeline phase: initialized | pm_complete | coder_complete.",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of state creation.",
    )
    success: bool = Field(
        default=True,
        description="Overall success flag.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Top-level error if the pipeline failed.",
    )

    # -- Helpers -----------------------------------------------------------
    def get_pending_tasks(self) -> List[TaskItem]:
        """Return tasks that are ready for the Coder."""
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def get_completed_tasks(self) -> List[TaskItem]:
        """Return tasks the Coder has finished."""
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]

    def all_tasks_done(self) -> bool:
        """Check if every task is completed or failed."""
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for t in self.tasks
        )

    def to_json(self) -> str:
        """Serialize to JSON (handoff protocol requirement)."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> "SharedState":
        """Deserialize from JSON."""
        return cls.model_validate_json(data)
