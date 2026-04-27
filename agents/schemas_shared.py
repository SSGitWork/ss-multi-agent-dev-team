"""
Shared State & Handoff Protocol Schemas.

And cost tracking fields.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

MAX_TASKS_PER_REQUIREMENT: int = 8


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskItem(BaseModel):
    task_id: str = Field(..., description="Unique task identifier.")
    title: str = Field(..., description="Short title.")
    description: str = Field(..., description="Detailed description.")
    acceptance_criteria: List[str] = Field(default_factory=list)
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    code_output: str = Field(default="")
    execution_result: str = Field(default="")
    error: Optional[str] = Field(default=None)
    completed_at: Optional[str] = Field(default=None)


class TechnicalSpec(BaseModel):
    name: str = Field(..., description="Project name.")
    description: str = Field(..., description="High-level description.")
    acceptance_criteria: List[str] = Field(default_factory=list)
    technical_approach: str = Field(default="")
    constraints: List[str] = Field(default_factory=list)


class AgentCostRecord(BaseModel):
    """Per-agent cost tracking record."""
    agent_name: str = ""
    model_name: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    duration_ms: float = 0.0
    cache_hits: int = 0


class PipelineCostReport(BaseModel):
    """Full pipeline cost report written to docs/cost_reports/."""
    session_id: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agents: Dict[str, AgentCostRecord] = Field(default_factory=dict)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: float = 0.0


class SharedState(BaseModel):
    original_requirement: str = Field(..., description="Raw user requirement.")
    technical_spec: Optional[TechnicalSpec] = Field(default=None)
    tasks: List[TaskItem] = Field(default_factory=list)
    pm_notes: str = Field(default="")
    accumulated_code: str = Field(default="")
    final_explanation: str = Field(default="")
    session_id: str = Field(default="")
    workspace_path: str = Field(default="")
    phase: str = Field(default="initialized")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    success: bool = Field(default=True)
    error: Optional[str] = Field(default=None)
    cost_report: Optional[PipelineCostReport] = Field(default=None)

    def get_pending_tasks(self) -> List[TaskItem]:
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def get_completed_tasks(self) -> List[TaskItem]:
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]

    def all_tasks_done(self) -> bool:
        return all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) for t in self.tasks)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> "SharedState":
        return cls.model_validate_json(data)
