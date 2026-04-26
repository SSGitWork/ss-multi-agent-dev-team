"""
Phase 1 + Phase 2 + Phase 3 – Coder Agent.

Phase 1: run_coder_agent()         — standalone, takes a raw task string.
Phase 2: run_coder_from_state()    — reads tasks from SharedState.
Phase 3: run_coder_task_with_reflection() — single task with self-reflection.
         revise_code_from_fixes()  — revise code based on QA fix instructions.
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
# Phase 3 — Self-Reflection
# ---------------------------------------------------------------------------
def _build_self_reflection_prompt(code: str, task_description: str) -> str:
    """Prompt the Coder to critique its own code before QA review."""
    return f"""
## SELF-REFLECTION TASK
You just wrote the following code. Before it goes to QA review, critically
analyse it for potential issues.

## THE CODE
```python
{code}
```

## ORIGINAL TASK
{task_description}

## INSTRUCTIONS
1. List ALL potential bugs, edge cases, and issues you can find.
2. For each issue, describe the fix.
3. Then output the REVISED code that addresses all issues.

## OUTPUT FORMAT
Respond with EXACTLY this format:

ISSUES_FOUND:
<numbered list of issues and their fixes>

REVISED_CODE:
<the complete revised Python code>
"""


def run_self_reflection(
    code: str,
    task_description: str,
    workspace_path: str,
) -> tuple[str, str]:
    """Run the Coder's self-reflection step.

    Returns:
        (revised_code, issues_found)
    """
    coder = build_coder_agent()
    prompt = _build_self_reflection_prompt(code, task_description)

    reflection_task = Task(
        description=prompt,
        expected_output="ISSUES_FOUND section and REVISED_CODE section.",
        agent=coder,
    )

    crew = Crew(
        agents=[coder],
        tasks=[reflection_task],
        process=Process.sequential,
        verbose=True,
    )

    crew_output = crew.kickoff()
    raw = str(crew_output)

    # Parse sections
    issues = ""
    revised_code = code  # fallback to original

    issues_idx = raw.find("ISSUES_FOUND:")
    code_idx = raw.find("REVISED_CODE:")

    if issues_idx != -1 and code_idx != -1:
        issues = raw[issues_idx + len("ISSUES_FOUND:"):code_idx].strip()
        revised_raw = raw[code_idx + len("REVISED_CODE:"):].strip()
        # Clean markdown fences if present
        revised_code = _clean_code_block(revised_raw) or revised_raw
    elif code_idx != -1:
        revised_raw = raw[code_idx + len("REVISED_CODE:"):].strip()
        revised_code = _clean_code_block(revised_raw) or revised_raw

    # Write revised code to workspace
    if revised_code.strip():
        ws = Path(workspace_path)
        ws.mkdir(parents=True, exist_ok=True)

    return revised_code, issues


# ---------------------------------------------------------------------------
# Phase 3 — Revise code from QA fix instructions
# ---------------------------------------------------------------------------
def _build_revision_prompt(
    code: str,
    fix_instructions: list[dict],
    task_description: str,
    test_output: str = "",
) -> str:
    fixes_str = "\n".join(
        f"  {i+1}. [{f.get('issue_id', 'N/A')}] {f.get('description', '')} "
        f"(Severity: {f.get('severity', 'medium')})\n"
        f"     Suggested fix: {f.get('suggested_fix', 'N/A')}\n"
        f"     Related test: {f.get('related_test', 'N/A')}"
        for i, f in enumerate(fix_instructions)
    )

    test_section = ""
    if test_output:
        test_section = f"""
## TEST OUTPUT (from QA)
{test_output[:2000]}
"""

    return f"""
## ORIGINAL TASK
{task_description}

## CURRENT CODE (has issues)
```python
{code}
```

{test_section}

## FIX INSTRUCTIONS FROM QA
{fixes_str}

## YOUR TASK
Revise the code to fix ALL the issues listed above. Make sure:
1. Every fix instruction is addressed.
2. The code still fulfils the original task.
3. No new bugs are introduced.

## OUTPUT FORMAT
Output ONLY the complete revised Python code. No markdown fences. No explanation.
Start directly with the code.
"""


def revise_code_from_fixes(
    code: str,
    fix_instructions: list[dict],
    task_description: str,
    workspace_path: str,
    file_path: str,
    test_output: str = "",
) -> str:
    """Revise code based on QA fix instructions.

    Returns the revised code string.
    """
    coder = build_coder_agent()
    prompt = _build_revision_prompt(code, fix_instructions, task_description, test_output)

    revision_task = Task(
        description=prompt,
        expected_output="Complete revised Python code.",
        agent=coder,
    )

    crew = Crew(
        agents=[coder],
        tasks=[revision_task],
        process=Process.sequential,
        verbose=True,
    )

    crew_output = crew.kickoff()
    raw = str(crew_output)

    revised = _clean_code_block(raw) or raw.strip()

    # Write revised code to workspace
    ws = Path(workspace_path)
    target = ws / file_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(revised, encoding="utf-8")

    return revised


# ---------------------------------------------------------------------------
# Phase 3 — Single task with self-reflection (used by review loop)
# ---------------------------------------------------------------------------
def run_coder_task_with_reflection(
    task_item: TaskItem,
    workspace_path: str,
    accumulated_code: str = "",
    memory: AgentMemory | None = None,
) -> tuple[str, str, str]:
    """Execute a single task with self-reflection.

    Steps:
        1. Generate initial code via ReACT loop.
        2. Run self-reflection to critique and revise.
        3. Write final code to workspace.

    Returns:
        (final_code, explanation, file_path)
    """
    if memory is None:
        memory = AgentMemory()

    memory_context = memory.build_context(task_item.description)

    # -- Step 1: Generate initial code -------------------------------------
    prompt = _build_task_prompt(
        task_item=task_item,
        workspace_path=workspace_path,
        context=memory_context,
        accumulated_code=accumulated_code,
    )

    coder = build_coder_agent()

    crew_task = Task(
        description=prompt,
        expected_output="CODE, EXPLANATION, and RESULT sections.",
        agent=coder,
    )

    crew = Crew(
        agents=[coder],
        tasks=[crew_task],
        process=Process.sequential,
        verbose=True,
    )

    crew_output = crew.kickoff()
    raw = str(crew_output)
    parsed = _parse_sections(raw)

    initial_code = parsed.get("CODE", "")
    explanation = parsed.get("EXPLANATION", "")

    # Determine file path
    file_path = f"{task_item.task_id.lower().replace('-', '_')}.py"

    # Write initial code
    ws = Path(workspace_path)
    ws.mkdir(parents=True, exist_ok=True)
    (ws / file_path).write_text(initial_code, encoding="utf-8")

    # -- Step 2: Self-reflection -------------------------------------------
    print(f"\n{'─'*40}")
    print(f"  SELF-REFLECTION for {task_item.task_id}")
    print(f"{'─'*40}\n")

    revised_code, issues_found = run_self_reflection(
        code=initial_code,
        task_description=task_item.description,
        workspace_path=workspace_path,
    )

    if issues_found:
        print(f"  Issues found during self-reflection:\n{issues_found}")

    # Write revised code
    final_code = revised_code if revised_code.strip() else initial_code
    (ws / file_path).write_text(final_code, encoding="utf-8")

    # Persist to memory
    memory.add_turn("user", f"Task: {task_item.description}")
    memory.add_turn("assistant", f"Code:\n{final_code}\n\n{explanation}")

    return final_code, explanation, file_path


# ---------------------------------------------------------------------------
# Phase 2 — SharedState-based entry point (updated for Phase 3)
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
        try:
            final_code, explanation, file_path = run_coder_task_with_reflection(
                task_item=task_item,
                workspace_path=state.workspace_path,
                accumulated_code=state.accumulated_code,
                memory=memory,
            )

            task_item.code_output = final_code
            task_item.execution_result = f"Code written to {file_path}"
            task_item.status = TaskStatus.COMPLETED
            task_item.completed_at = datetime.now(timezone.utc).isoformat()

            if final_code.strip():
                header = f"\n# === {task_item.task_id}: {task_item.title} ===\n"
                state.accumulated_code += header + final_code + "\n"

            all_explanations.append(
                f"**{task_item.task_id} — {task_item.title}**: {explanation}"
            )

        except Exception as exc:
            task_item.status = TaskStatus.FAILED
            task_item.error = str(exc)
            task_item.completed_at = datetime.now(timezone.utc).isoformat()

    state.final_explanation = "\n\n".join(all_explanations)
    state.phase = "coder_complete"

    failed = [t for t in state.tasks if t.status == TaskStatus.FAILED]
    if failed:
        state.success = False
        state.error = f"{len(failed)} task(s) failed: {[t.task_id for t in failed]}"

    return state


# ===========================================================================
# Phase 1 — Original standalone entry point (preserved)
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
def _build_task_prompt(
    task_item: TaskItem,
    workspace_path: str,
    context: str = "",
    accumulated_code: str = "",
) -> str:
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
1. **Thought**: Analyse the task. Break it into numbered steps.
2. **Action**: Execute ONE step using a tool (write_file, read_file, exec_python).
3. **Observation**: Read the tool output. Decide if the step succeeded.
4. Repeat until complete or {MAX_ITERATIONS} iterations.

## FINAL ANSWER FORMAT
CODE:
<the complete source code for THIS task>

EXPLANATION:
<what the code does and design choices>

RESULT:
<execution output>
"""


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


def _clean_code_block(text: str) -> str:
    """Remove markdown fences from code output."""
    import re
    pattern = r"```(?:python)?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
