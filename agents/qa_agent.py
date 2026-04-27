"""
QA & Debugger Agent — uses GPT-4o-mini with tracing, resilience, and cost tracking.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import List

from crewai import Agent, Crew, Process, Task

from agents.config import get_settings
from agents.llm_wrapper import build_qa_llm, resilient_crew_kickoff
from agents.schemas_a2a import (
    A2AIntent,
    A2AMessage,
    AgentRole,
    AllTestsPassedPayload,
    FinalQAReport,
    FixInstruction,
    FixInstructionsPayload,
    ReviewRequestPayload,
    TestFailure,
)
from agents.tracing import agent_span, record_span_metadata

logger = logging.getLogger(__name__)


def build_qa_agent() -> Agent:
    return Agent(
        role="Senior QA Engineer & Debugger",
        goal=(
            "Write comprehensive pytest tests for the given code based on "
            "acceptance criteria. Execute the tests, analyse failures, and "
            "produce clear, actionable fix instructions."
        ),
        backstory=(
            "You are a meticulous QA engineer with 12 years of experience. "
            "You write thorough tests covering happy paths, edge cases, "
            "and error conditions."
        ),
        llm=build_qa_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=5,
        max_retry_limit=2,
    )


def _build_test_gen_prompt(
    code: str,
    file_path: str,
    acceptance_criteria: list[str],
    workspace_path: str,
) -> str:
    criteria_str = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(acceptance_criteria))
    return f"""
## CODE UNDER TEST
File: {file_path}
```python
{code}
```

## ACCEPTANCE CRITERIA
{criteria_str}

## YOUR TASK
Write a pytest test file that tests the code above against ALL acceptance criteria.

## RULES
1. Import the code using the correct module path relative to workspace: {workspace_path}/{file_path}
2. Write at least one test per acceptance criterion.
3. Include edge case tests.
4. Use descriptive test names: test_<what_it_tests>.
5. Use only pytest. Each test must have a clear assert.

## OUTPUT FORMAT
Output ONLY the Python test code. No markdown fences. No explanation.
"""


def _execute_tests(test_file_path: str, workspace_path: str, timeout: int = 30) -> dict:
    """Run pytest in a sandboxed subprocess and return parsed results."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", test_file_path, "-v", "--tb=short", "--no-header"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workspace_path,
        )

        stdout = result.stdout
        stderr = result.stderr

        passed = len(re.findall(r" PASSED", stdout))
        failed = len(re.findall(r" FAILED", stdout))
        errors = len(re.findall(r" ERROR", stdout))
        total = passed + failed + errors

        failures: list[dict] = []
        fail_pattern = re.findall(r"FAILED\s+[\w./]+::(\w+)(?:\s*-\s*(.+))?", stdout)
        for test_name, error_msg in fail_pattern:
            failures.append({
                "test_name": test_name,
                "error_message": error_msg.strip() if error_msg else "See traceback",
            })

        return {
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "passed": passed,
            "failed": failed + errors,
            "total": total,
            "failures": failures,
        }

    except subprocess.TimeoutExpired:
        return {
            "returncode": -1, "stdout": "", "stderr": f"Timed out after {timeout}s",
            "passed": 0, "failed": 0, "total": 0,
            "failures": [{"test_name": "TIMEOUT", "error_message": f"Timed out after {timeout}s"}],
        }
    except Exception as exc:
        return {
            "returncode": -1, "stdout": "", "stderr": str(exc),
            "passed": 0, "failed": 0, "total": 0,
            "failures": [{"test_name": "EXECUTION_ERROR", "error_message": str(exc)}],
        }


def run_qa_review(
    review_request: ReviewRequestPayload,
    workspace_path: str,
    correlation_id: str,
) -> tuple[A2AMessage, str]:
    """Execute a full QA review cycle for one piece of code."""
    settings = get_settings()

    with agent_span("qa_agent") as span:
        record_span_metadata(
            span,
            agent_name="qa_agent",
            task_id=review_request.task_id,
            iteration=review_request.iteration,
            model=settings.models.qa_model,
        )

        ws = Path(workspace_path)
        ws.mkdir(parents=True, exist_ok=True)

        # -- Generate tests ------------------------------------------------
        qa_agent = build_qa_agent()
        test_gen_prompt = _build_test_gen_prompt(
            code=review_request.code,
            file_path=review_request.file_path,
            acceptance_criteria=review_request.acceptance_criteria,
            workspace_path=workspace_path,
        )

        gen_task = Task(
            description=test_gen_prompt,
            expected_output="Python pytest test code.",
            agent=qa_agent,
        )
        crew = Crew(agents=[qa_agent], tasks=[gen_task], process=Process.sequential, verbose=True)

        crew_output = resilient_crew_kickoff(
            crew, agent_name="qa_agent", model_name=settings.models.qa_model
        )
        test_code = _clean_code_output(str(crew_output))

        # -- Write test file -----------------------------------------------
        test_filename = f"test_{review_request.task_id.lower().replace('-', '_')}.py"
        test_file_path = ws / test_filename
        test_file_path.write_text(test_code, encoding="utf-8")

        # -- Execute tests -------------------------------------------------
        test_results = _execute_tests(
            test_file_path=str(test_file_path),
            workspace_path=workspace_path,
            timeout=settings.agent.qa_test_timeout,
        )

        record_span_metadata(
            span,
            tests_total=test_results["total"],
            tests_passed=test_results["passed"],
            tests_failed=test_results["failed"],
        )

        # -- Build response ------------------------------------------------
        if test_results["failed"] == 0 and test_results["passed"] > 0:
            payload = AllTestsPassedPayload(
                task_id=review_request.task_id,
                iteration=review_request.iteration,
                test_file=test_filename,
                total_tests=test_results["total"],
                passed=test_results["passed"],
                raw_output=test_results["stdout"][:2000],
            )
            message = A2AMessage(
                correlation_id=correlation_id,
                sender=AgentRole.QA,
                receiver=AgentRole.CODER,
                intent=A2AIntent.ALL_TESTS_PASSED,
                payload=payload.model_dump(),
            )
        else:
            fix_instructions = _generate_fix_instructions(
                code=review_request.code,
                test_code=test_code,
                test_output=test_results["stdout"] + "\n" + test_results["stderr"],
                failures=test_results["failures"],
            )
            payload = FixInstructionsPayload(
                task_id=review_request.task_id,
                iteration=review_request.iteration,
                test_file=test_filename,
                total_tests=test_results["total"],
                passed=test_results["passed"],
                failed=test_results["failed"],
                failures=[
                    TestFailure(test_name=f["test_name"], error_message=f["error_message"])
                    for f in test_results["failures"]
                ],
                fix_instructions=fix_instructions,
                raw_output=(test_results["stdout"] + "\n" + test_results["stderr"])[:2000],
            )
            message = A2AMessage(
                correlation_id=correlation_id,
                sender=AgentRole.QA,
                receiver=AgentRole.CODER,
                intent=A2AIntent.FIX_INSTRUCTIONS,
                payload=payload.model_dump(),
            )

    return message, test_code


def generate_final_report(
    task_id: str,
    total_iterations: int,
    last_test_output: str,
    unresolved: list[FixInstruction],
    resolved: list[str],
) -> FinalQAReport:
    return FinalQAReport(
        task_id=task_id,
        total_iterations=total_iterations,
        final_test_results=last_test_output[:3000],
        unresolved_issues=unresolved,
        resolved_issues=resolved,
        recommendation=(
            f"After {total_iterations} iterations, {len(unresolved)} issue(s) "
            f"remain unresolved. Manual review recommended for: "
            + ", ".join(i.issue_id for i in unresolved)
            if unresolved
            else f"All issues resolved in {total_iterations} iterations."
        ),
    )


def _generate_fix_instructions(
    code: str, test_code: str, test_output: str, failures: list[dict],
) -> list[FixInstruction]:
    if not failures:
        return []

    settings = get_settings()
    failures_str = "\n".join(f"  - {f['test_name']}: {f['error_message']}" for f in failures)

    prompt = f"""
## ORIGINAL CODE
```python
{code}
```

## TEST CODE
```python
{test_code}
```

## TEST OUTPUT
{test_output}

## FAILURES
{failures_str}

## YOUR TASK
Produce fix instructions for the developer.

## OUTPUT FORMAT
Output ONLY a valid JSON array:
[
  {{
    "issue_id": "FIX-001",
    "description": "What is wrong",
    "severity": "high",
    "suggested_fix": "Specific code change",
    "related_test": "test_name"
  }}
]
"""

    qa_agent = build_qa_agent()
    fix_task = Task(description=prompt, expected_output="JSON array of fix instructions.", agent=qa_agent)
    crew = Crew(agents=[qa_agent], tasks=[fix_task], process=Process.sequential, verbose=True)

    try:
        crew_output = resilient_crew_kickoff(
            crew, agent_name="qa_agent", model_name=settings.models.qa_model
        )
        raw = str(crew_output)
        instructions_data = _extract_json_array(raw)
        return [
            FixInstruction(
                issue_id=item.get("issue_id", f"FIX-{i+1:03d}"),
                description=item.get("description", ""),
                severity=item.get("severity", "medium"),
                suggested_fix=item.get("suggested_fix", ""),
                related_test=item.get("related_test", ""),
            )
            for i, item in enumerate(instructions_data)
        ]
    except Exception:
        return [
            FixInstruction(
                issue_id=f"FIX-{i+1:03d}",
                description=f"Test '{f['test_name']}' failed: {f['error_message']}",
                severity="high",
                suggested_fix="Review the failing test and fix the code.",
                related_test=f["test_name"],
            )
            for i, f in enumerate(failures)
        ]


def _extract_json_array(text: str) -> list[dict]:
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(text[start:i + 1])
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        break
    return []


def _clean_code_output(text: str) -> str:
    pattern = r"```(?:python)?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    lines = text.strip().split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "def ", "class ", "#", "@")):
            return "\n".join(lines[i:]).strip()

    return text.strip()