"""
Phase 2 – Product Manager Agent.

Responsibilities:
  1. Parse a free-text user requirement.
  2. Generate a structured TechnicalSpec.
  3. Break the spec into discrete TaskItems (max 8 — consolidate if more).
  4. Write everything into SharedState for the Coder to consume.

The PM does NOT write code — it only plans and specifies.
"""

from __future__ import annotations

import json
import os
import re
from typing import List

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

from agents.schemas_shared import (
    MAX_TASKS_PER_REQUIREMENT,
    SharedState,
    TaskItem,
    TaskPriority,
    TaskStatus,
    TechnicalSpec,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Azure LLM configuration (reuse pattern from Phase 1)
# ---------------------------------------------------------------------------
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
        temperature=0.3,
        max_tokens=4096,
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def build_pm_agent() -> Agent:
    """Construct the Product Manager CrewAI agent."""
    return Agent(
        role="Senior Product Manager",
        goal=(
            "Analyse the user's requirement, produce a detailed technical "
            "specification, and break it into a prioritised list of discrete "
            "coding tasks. Each task must have a unique ID, clear description, "
            "and acceptance criteria. Never produce more than "
            f"{MAX_TASKS_PER_REQUIREMENT} tasks — consolidate if needed."
        ),
        backstory=(
            "You are a seasoned Product Manager with deep technical knowledge. "
            "You excel at translating vague requirements into precise, "
            "actionable specifications. You always think about edge cases, "
            "acceptance criteria, and task dependencies. You communicate "
            "in structured formats that developers can immediately act on."
        ),
        llm=_build_azure_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=5,
        max_retry_limit=2,
    )


# ---------------------------------------------------------------------------
# PM prompt builder
# ---------------------------------------------------------------------------
def _build_pm_prompt(requirement: str) -> str:
    """Build the prompt that instructs the PM to produce structured output."""
    return f"""
## USER REQUIREMENT
{requirement}

## YOUR TASK
Analyse the above requirement and produce a structured technical specification
with a task breakdown.

## RULES
1. Maximum {MAX_TASKS_PER_REQUIREMENT} tasks. If you identify more, consolidate
   related work into fewer tasks.
2. Each task must be independently implementable by a Python developer.
3. Tasks should be ordered by dependency (foundational tasks first).
4. Every task needs clear acceptance criteria.

## OUTPUT FORMAT
You MUST respond with EXACTLY this JSON structure (no markdown fences, no extra text):

{{
  "technical_spec": {{
    "name": "<project/feature name>",
    "description": "<high-level description>",
    "acceptance_criteria": ["<criterion 1>", "<criterion 2>"],
    "technical_approach": "<recommended approach>",
    "constraints": ["<constraint 1>"]
  }},
  "tasks": [
    {{
      "task_id": "TASK-001",
      "title": "<short title>",
      "description": "<detailed description of what to implement>",
      "acceptance_criteria": ["<criterion 1>", "<criterion 2>"],
      "priority": "high"
    }},
    {{
      "task_id": "TASK-002",
      "title": "<short title>",
      "description": "<detailed description>",
      "acceptance_criteria": ["<criterion 1>"],
      "priority": "medium"
    }}
  ],
  "pm_notes": "<any additional notes or considerations>"
}}

IMPORTANT:
- Output ONLY valid JSON. No markdown code fences. No explanation before or after.
- task_id must follow the pattern TASK-001, TASK-002, etc.
- priority must be one of: high, medium, low
- Maximum {MAX_TASKS_PER_REQUIREMENT} tasks.
"""


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    """Extract a JSON object from LLM output that may contain extra text.

    Tries multiple strategies:
      1. Direct parse of the full text.
      2. Find the first { ... } block via brace matching.
      3. Regex for ```json ... ``` fenced blocks.
    """
    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find outermost { ... }
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    # Strategy 3: fenced code block
    pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from PM output:\n{text[:500]}")


# ---------------------------------------------------------------------------
# Task consolidation
# ---------------------------------------------------------------------------
def _consolidate_tasks(tasks: List[TaskItem]) -> List[TaskItem]:
    """If there are more than MAX_TASKS, merge the lowest-priority ones.

    Strategy: keep the first (MAX - 1) tasks, merge the rest into one
    consolidated task.
    """
    if len(tasks) <= MAX_TASKS_PER_REQUIREMENT:
        return tasks

    keep = tasks[:MAX_TASKS_PER_REQUIREMENT - 1]
    merge = tasks[MAX_TASKS_PER_REQUIREMENT - 1:]

    merged_descriptions = "\n".join(
        f"- [{t.task_id}] {t.title}: {t.description}" for t in merge
    )
    merged_criteria = []
    for t in merge:
        merged_criteria.extend(t.acceptance_criteria)

    consolidated = TaskItem(
        task_id=f"TASK-{MAX_TASKS_PER_REQUIREMENT:03d}",
        title="Consolidated remaining tasks",
        description=(
            f"The following tasks have been consolidated into one:\n"
            f"{merged_descriptions}"
        ),
        acceptance_criteria=merged_criteria,
        priority=TaskPriority.MEDIUM,
        status=TaskStatus.PENDING,
    )

    keep.append(consolidated)
    return keep


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_pm_agent(state: SharedState) -> SharedState:
    """Execute the PM agent: parse requirement → spec → task list.

    Mutates and returns the SharedState with:
      - technical_spec populated
      - tasks list populated (max 8, consolidated if needed)
      - phase set to 'pm_complete'
    """
    pm_agent = build_pm_agent()
    prompt = _build_pm_prompt(state.original_requirement)

    pm_task = Task(
        description=prompt,
        expected_output="A valid JSON object with technical_spec, tasks, and pm_notes.",
        agent=pm_agent,
    )

    crew = Crew(
        agents=[pm_agent],
        tasks=[pm_task],
        process=Process.sequential,
        verbose=True,
    )

    try:
        crew_output = crew.kickoff()
        raw = str(crew_output)

        # Parse the JSON output
        parsed = _extract_json(raw)

        # Build TechnicalSpec
        spec_data = parsed.get("technical_spec", {})
        state.technical_spec = TechnicalSpec(
            name=spec_data.get("name", "Unnamed Project"),
            description=spec_data.get("description", ""),
            acceptance_criteria=spec_data.get("acceptance_criteria", []),
            technical_approach=spec_data.get("technical_approach", ""),
            constraints=spec_data.get("constraints", []),
        )

        # Build TaskItems
        raw_tasks = parsed.get("tasks", [])
        task_items: List[TaskItem] = []
        for t in raw_tasks:
            priority_str = t.get("priority", "medium").lower()
            try:
                priority = TaskPriority(priority_str)
            except ValueError:
                priority = TaskPriority.MEDIUM

            task_items.append(TaskItem(
                task_id=t.get("task_id", f"TASK-{len(task_items)+1:03d}"),
                title=t.get("title", "Untitled Task"),
                description=t.get("description", ""),
                acceptance_criteria=t.get("acceptance_criteria", []),
                priority=priority,
                status=TaskStatus.PENDING,
            ))

        # Consolidate if > MAX_TASKS
        state.tasks = _consolidate_tasks(task_items)
        state.pm_notes = parsed.get("pm_notes", "")
        state.phase = "pm_complete"

    except Exception as exc:
        state.success = False
        state.error = f"PM Agent failed: {exc}"
        state.phase = "pm_failed"

    return state
