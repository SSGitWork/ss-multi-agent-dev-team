"""
Phase 1 + Phase 2 – Coder Agent.

Phase 1: run_coder_agent()       — standalone, takes a raw task string.
Phase 2: run_coder_from_state()  — reads tasks from SharedState, updates
                                    status and code after each task.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

from agents.memory import AgentMemory
from agents.schemas import CoderAgentOutput
from agents.schemas_shared import SharedState, TaskItem, TaskStatus
from tools.exec_tools import exec_python
from tools.file_tools import read_file, write_file

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_ITERATIONS: int = int(os.getenv("MAX_REACT_ITERATIONS", "10"))

AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_API_BASE = os.getenv("AZURE_API_BASE", os.getenv("AZURE_OPENAI_ENDPOINT", ""))
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")


def _build_azure_llm() -> LLM:
    """Construct the CrewAI LLM object for Azure OpenAI."""
    return LLM(
        model=f"azure/{AZURE_DEPLOYMENT}",
        api_key=AZURE_API_KEY,
        api_base=AZURE_API_BASE,
        api_version=AZURE_API_VERSION,
        temperature=0.2,
        max_tokens=4096,
    )


def _build_session_workspace() -> tuple[str, str]:
    """Create a unique workspace directory and return (session_id, path)."""
    session_id = uuid.uuid4().hex[:12]
    ws = Path("./workspace") / session_id
    ws.mkdir(parents=True, exist_ok=True)
    return session_id, str(ws.resolve())


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def build_coder_agent() -> Agent:
    """Construct the CrewAI Coder Agent with tools attached."""
    return Agent(
        role="Senior Python Developer",
        goal=(
            "Accept a coding task, plan a step-by-step approach, write clean "
            "production-quality Python code, save it to the workspace, execute "
            "it, and return the structured result."
        ),
        backstory=(
            "You are an expert Python developer with 15 years of experience. "
            "You always plan before coding. You write clean, well-documented "
            "code. You test your code by executing it and verifying the output. "
            "You never use eval() — you always write code to files and run "
            "them via the exec_python tool."
        ),
        tools=[read_file, write_file, exec_python],
        llm=_build_azure_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=MAX_ITERATIONS,
        max_retry_limit=2,
    )


# ---------------------------------------------------------------------------
# Phase 2 — Task-level prompt builder
# ---------------------------------------------------------------------------
def _build_task_prompt(
    task_item: TaskItem,
    workspace_path: str,
    context: str = "",
    accumulated_code: str = "",
) -> str:
    """Build a ReACT prompt for a single task from the PM's task list."""
    criteria_str = "\n".join(
        f"  - {c}" for c in task_item.acceptance_criteria
    ) or "  - (none specified)"

    prev_code_section = ""
    if accumulated_code.strip():
        prev_code_section = f"""
## CODE FROM PREVIOUS TASKS (already in workspace)
{accumulated_code}
"""

    memory_section = ""
    if context.strip():
        memory_section = f"""
--- MEMORY CONTEXT (use if relevant) ---
{context}
--- END MEMORY CONTEXT ---
"""

    return f"""
{memory_section}
{prev_code_section}

## CURRENT TASK
Task ID: {task_item.task_id}
Title: {task_item.title}
Priority: {task_item.priority.value}

Description:
{task_item.description}

Acceptance Criteria:
{criteria_str}

## WORKSPACE
All files MUST be read from / written to the workspace directory.
Workspace path: {workspace_path}
When calling write_file or read_file, always pass workspace="{workspace_path}".
When calling exec_python, always pass workspace="{workspace_path}".

## INSTRUCTIONS — ReACT Loop
Follow this loop strictly:

1. **Thought**: Analyse the task. Break it into numbered steps.
2. **Action**: Execute ONE step using a tool (write_file, read_file, exec_python).
3. **Observation**: Read the tool output. Decide if the step succeeded.
4. Repeat until the task is complete or you hit {MAX_ITERATIONS} iterations.

## FINAL ANSWER FORMAT
When done, respond with EXACTLY this format:

CODE:
<the complete source code you wrote for THIS task>

EXPLANATION:
<what the code does and why you made these design choices>

RESULT:
<the execution output from running the code>
"""


# ---------------------------------------------------------------------------
# Phase 2 — Public entry point (SharedState-based)
# ---------------------------------------------------------------------------
def run_coder_from_state(state: SharedState) -> SharedState:
    """Execute the Coder agent on all pending tasks in SharedState.

    For each pending task:
      1. Build a task-specific prompt with memory context.
      2. Run the CrewAI agent.
      3. Parse the output and update the task in SharedState.
      4. Append code to accumulated_code.

    Returns the mutated SharedState.
    """
    # -- Set up workspace if not already done ------------------------------
    if not state.session_id or not state.workspace_path:
        sid, wpath = _build_session_workspace()
        state.session_id = sid
        state.workspace_path = wpath
    else:
        Path(state.workspace_path).mkdir(parents=True, exist_ok=True)

    memory = AgentMemory()
    all_explanations: list[str] = []

    pending_tasks = state.get_pending_tasks()
    if not pending_tasks:
        state.phase = "coder_complete"
        state.final_explanation = "No pending tasks to process."
        return state

    for task_item in pending_tasks:
        # Mark as in-progress
        task_item.status = TaskStatus.IN_PROGRESS

        # Build context
        memory_context = memory.build_context(task_item.description)

        prompt = _build_task_prompt(
            task_item=task_item,
            workspace_path=state.workspace_path,
            context=memory_context,
            accumulated_code=state.accumulated_code,
        )

        # Build and run CrewAI
        coder_agent = build_coder_agent()

        crew_task = Task(
            description=prompt,
            expected_output=(
                "A response with three labelled sections: CODE, EXPLANATION, RESULT."
            ),
            agent=coder_agent,
        )

        crew = Crew(
            agents=[coder_agent],
            tasks=[crew_task],
            process=Process.sequential,
            verbose=True,
        )

        try:
            crew_output = crew.kickoff()
            raw = str(crew_output)

            parsed = _parse_sections(raw)

            code = parsed.get("CODE", "")
            explanation = parsed.get("EXPLANATION", "")
            result = parsed.get("RESULT", "")

            # Update the task item
            task_item.code_output = code
            task_item.execution_result = result
            task_item.status = TaskStatus.COMPLETED
            task_item.completed_at = datetime.now(timezone.utc).isoformat()

            # Accumulate
            if code.strip():
                header = f"\n# === {task_item.task_id}: {task_item.title} ===\n"
                state.accumulated_code += header + code + "\n"

            all_explanations.append(
                f"**{task_item.task_id} — {task_item.title}**: {explanation}"
            )

            # Persist to memory
            memory.add_turn("user", f"Task: {task_item.description}")
            memory.add_turn("assistant", f"Code:\n{code}\n\n{explanation}")

        except Exception as exc:
            task_item.status = TaskStatus.FAILED
            task_item.error = str(exc)
            task_item.completed_at = datetime.now(timezone.utc).isoformat()

    # -- Finalize state ----------------------------------------------------
    state.final_explanation = "\n\n".join(all_explanations)
    state.phase = "coder_complete"

    # Check if any task failed
    failed = [t for t in state.tasks if t.status == TaskStatus.FAILED]
    if failed:
        state.success = False
        state.error = f"{len(failed)} task(s) failed: {[t.task_id for t in failed]}"

    return state


# ===========================================================================
# Phase 1 — Original standalone entry point (preserved for backward compat)
# ===========================================================================
def run_coder_agent(task_description: str) -> CoderAgentOutput:
    """Execute the Coder Agent on a given task and return structured output.

    Steps:
        1. Create an isolated session workspace.
        2. Build memory context from past interactions.
        3. Construct a CrewAI Task with a ReACT-style prompt.
        4. Kick off the Crew and parse the result.
        5. Persist the interaction in memory.
    """
    session_id, workspace_path = _build_session_workspace()
    memory = AgentMemory()

    # -- Retrieve relevant memory context ----------------------------------
    memory_context = memory.build_context(task_description)
    memory_preamble = ""
    if memory_context:
        memory_preamble = (
            f"\n\n--- MEMORY CONTEXT (use if relevant) ---\n"
            f"{memory_context}\n"
            f"--- END MEMORY CONTEXT ---\n\n"
        )

    # -- Build the ReACT-style prompt -------------------------------------
    react_prompt = f"""
{memory_preamble}
## YOUR TASK
{task_description}

## WORKSPACE
All files MUST be read from / written to the workspace directory.
Workspace path: {workspace_path}
When calling write_file or read_file, always pass workspace="{workspace_path}".
When calling exec_python, always pass workspace="{workspace_path}".

## INSTRUCTIONS — ReACT Loop
Follow this loop strictly:

1. **Thought**: Analyse the task. Break it into numbered steps (your plan).
2. **Action**: Execute ONE step using a tool (write_file, read_file, exec_python).
3. **Observation**: Read the tool output. Decide if the step succeeded.
4. Repeat steps 1-3 until the task is complete or you hit {MAX_ITERATIONS} iterations.

## FINAL ANSWER FORMAT
When done, you MUST respond with EXACTLY this format (keep the labels):

PLAN:
<your numbered step-by-step plan>

CODE:
<the final complete source code you wrote>

EXPLANATION:
<natural-language explanation of the code and design decisions>

RESULT:
<the execution output from running the code>
"""

    # -- Build CrewAI Task & Crew ------------------------------------------
    coder_agent = build_coder_agent()

    coding_task = Task(
        description=react_prompt,
        expected_output=(
            "A response containing four clearly labelled sections: "
            "PLAN, CODE, EXPLANATION, and RESULT."
        ),
        agent=coder_agent,
    )

    crew = Crew(
        agents=[coder_agent],
        tasks=[coding_task],
        process=Process.sequential,
        verbose=True,
    )

    # -- Execute -----------------------------------------------------------
    try:
        crew_output = crew.kickoff()
        raw_result = str(crew_output)

        # -- Parse the structured sections ---------------------------------
        parsed = _parse_sections(raw_result)

        output = CoderAgentOutput(
            code=parsed.get("CODE", ""),
            explanation=parsed.get("EXPLANATION", ""),
            plan=parsed.get("PLAN", ""),
            result=parsed.get("RESULT", ""),
            session_id=session_id,
            success=True,
            iterations_used=MAX_ITERATIONS,  # CrewAI doesn't expose count
        )

    except Exception as exc:
        output = CoderAgentOutput(
            code="",
            explanation="",
            plan="",
            result="",
            session_id=session_id,
            success=False,
            error=str(exc),
            iterations_used=0,
        )

    # -- Persist to memory -------------------------------------------------
    memory.add_turn("user", task_description)
    memory.add_turn(
        "assistant",
        f"[code]\n{output.code}\n[/code]\n{output.explanation}",
    )

    return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_sections(text: str) -> dict[str, str]:
    """Extract labelled sections (PLAN, CODE, EXPLANATION, RESULT) from text."""
    sections: dict[str, str] = {}
    labels = ["PLAN", "CODE", "EXPLANATION", "RESULT"]

    for i, label in enumerate(labels):
        start_marker = f"{label}:"
        start_idx = text.find(start_marker)
        if start_idx == -1:
            continue

        content_start = start_idx + len(start_marker)

        # Find the start of the next section (or end of text)
        end_idx = len(text)
        for next_label in labels[i + 1:]:
            next_marker = f"{next_label}:"
            next_idx = text.find(next_marker, content_start)
            if next_idx != -1:
                end_idx = next_idx
                break

        sections[label] = text[content_start:end_idx].strip()

    return sections
