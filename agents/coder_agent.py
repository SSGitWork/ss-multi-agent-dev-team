"""
Phase 1 – Coder Agent v1.0

A single autonomous agent built on CrewAI that:
  • Accepts a natural-language coding task.
  • Retrieves relevant context from dual-layer memory.
  • Plans its approach using a ReACT loop (Thought → Action → Observation).
  • Uses file I/O and sandboxed execution tools.
  • Returns structured output (CoderAgentOutput).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

from agents.memory import AgentMemory
from agents.schemas import CoderAgentOutput
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
    """Construct the CrewAI LLM object for Azure OpenAI.

    Uses the native Azure AI Inference provider with explicit
    configuration to avoid environment variable mismatches.
    """

    # ── Build the endpoint URL ──────────────────────────────────────────
    # CrewAI's native Azure provider expects the full deployment endpoint:
    #   https://<resource>.openai.azure.com/openai/deployments/<deployment>
    #
    # If your AZURE_API_BASE is just the resource URL, we construct the
    # full deployment endpoint. If it already contains /openai/deployments/,
    # we use it as-is.

    base = AZURE_API_BASE.rstrip("/")
    if "/openai/deployments/" not in base:
        endpoint = f"{base}/openai/deployments/{AZURE_DEPLOYMENT}"
    else:
        endpoint = base

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
        llm=_build_azure_llm(),          # ← LLM object, not a string
        verbose=True,
        allow_delegation=False,
        max_iter=MAX_ITERATIONS,
        max_retry_limit=2,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
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
