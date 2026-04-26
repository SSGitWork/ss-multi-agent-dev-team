"""
Structured output schemas for the Coder Agent.

Defines the Pydantic models that enforce a consistent, validated
response shape from every agent invocation.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CoderAgentOutput(BaseModel):
    """Structured output returned by the Coder Agent after task completion.

    Attributes:
        code:        The generated source code (may span multiple files).
        explanation: A natural-language summary of *what* the code does and
                     *why* certain design choices were made.
        plan:        The step-by-step execution plan the agent followed
                     (derived from the ReACT reasoning trace).
        result:      stdout / stderr captured from running the code, or an
                     error message if execution failed.
        session_id:  Unique workspace session identifier so artefacts are
                     isolated per run.
        success:     Whether the overall task completed without errors.
    """

    code: str = Field(
        ...,
        description="The final generated source code.",
    )
    explanation: str = Field(
        ...,
        description="Natural-language explanation of the code and design decisions.",
    )
    plan: str = Field(
        ...,
        description="The step-by-step execution plan the agent followed.",
    )
    result: str = Field(
        ...,
        description="Execution output (stdout/stderr) or error message.",
    )
    session_id: str = Field(
        ...,
        description="Unique session identifier for the workspace folder.",
    )
    success: bool = Field(
        default=True,
        description="Whether the task completed successfully.",
    )
    iterations_used: int = Field(
        default=0,
        description="Number of ReACT loop iterations consumed.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error details if the task failed.",
    )
