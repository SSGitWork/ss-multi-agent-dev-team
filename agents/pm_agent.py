"""
Product Manager Agent — uses GPT-4o (larger model) with tracing & resilience.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List

from crewai import Agent, Crew, Process, Task

from agents.config import get_settings
from agents.llm_wrapper import build_pm_llm, resilient_crew_kickoff
from agents.schemas_shared import (
    MAX_TASKS_PER_REQUIREMENT,
    SharedState,
    TaskItem,
    TaskPriority,
    TaskStatus,
    TechnicalSpec,
)
from agents.tracing import agent_span, record_span_metadata

logger = logging.getLogger(__name__)


def build_pm_agent() -> Agent:
    return Agent(
        role="Senior Product Manager",
        goal=(
            "Analyse the user's requirement, produce a detailed technical "
            "specification, and break it into a prioritised list of discrete "
            f"coding tasks. Never produce more than {MAX_TASKS_PER_REQUIREMENT} tasks."
        ),
        backstory=(
            "You are a seasoned Product Manager with deep technical knowledge. "
            "You excel at translating vague requirements into precise, "
            "actionable specifications."
        ),
        llm=build_pm_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=5,
        max_retry_limit=2,
    )


def _build_pm_prompt(requirement: str) -> str:
    return f"""
## USER REQUIREMENT
{requirement}

## YOUR TASK
Analyse the above requirement and produce a structured technical specification
with a task breakdown.

## RULES
1. Maximum {MAX_TASKS_PER_REQUIREMENT} tasks. If you identify more, consolidate.
2. Each task must be independently implementable by a Python developer.
3. Tasks should be ordered by dependency.
4. Every task needs clear acceptance criteria.

## OUTPUT FORMAT
Respond with EXACTLY this JSON structure (no markdown fences, no extra text):

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
      "description": "<detailed description>",
      "acceptance_criteria": ["<criterion 1>", "<criterion 2>"],
      "priority": "high"
    }}
  ],
  "pm_notes": "<any additional notes>"
}}

IMPORTANT: Output ONLY valid JSON. No markdown code fences.
"""


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

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

    pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from PM output:\n{text[:500]}")


def _consolidate_tasks(tasks: List[TaskItem]) -> List[TaskItem]:
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
        description=f"Consolidated:\n{merged_descriptions}",
        acceptance_criteria=merged_criteria,
        priority=TaskPriority.MEDIUM,
        status=TaskStatus.PENDING,
    )
    keep.append(consolidated)
    return keep


def run_pm_agent(state: SharedState) -> SharedState:
    """Execute the PM agent with tracing and resilience."""
    settings = get_settings()

    with agent_span("pm_agent") as span:
        record_span_metadata(span, agent_name="pm_agent", model=settings.models.pm_model)

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
            crew_output = resilient_crew_kickoff(
                crew, agent_name="pm_agent", model_name=settings.models.pm_model
            )
            raw = str(crew_output)
            parsed = _extract_json(raw)

            spec_data = parsed.get("technical_spec", {})
            state.technical_spec = TechnicalSpec(
                name=spec_data.get("name", "Unnamed Project"),
                description=spec_data.get("description", ""),
                acceptance_criteria=spec_data.get("acceptance_criteria", []),
                technical_approach=spec_data.get("technical_approach", ""),
                constraints=spec_data.get("constraints", []),
            )

            raw_tasks = parsed.get("tasks", [])
            task_items: List[TaskItem] = []
            for t in raw_tasks:
                try:
                    priority = TaskPriority(t.get("priority", "medium").lower())
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

            state.tasks = _consolidate_tasks(task_items)
            state.pm_notes = parsed.get("pm_notes", "")
            state.phase = "pm_complete"

            record_span_metadata(span, task_count=len(state.tasks))

        except Exception as exc:
            logger.error("PM Agent failed: %s", exc)
            state.success = False
            state.error = f"PM Agent failed: {exc}"
            state.phase = "pm_failed"
            record_span_metadata(span, error=str(exc))

    return state
