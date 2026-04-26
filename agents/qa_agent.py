"""
Phase 3 – QA & Debugger Agent.

Responsibilities:
  1. Receive code from the Coder (via A2A REVIEW_REQUEST).
  2. Write real pytest tests based on acceptance criteria.
  3. Execute tests in a sandboxed subprocess.
  4. Parse failures and produce structured fix instructions.
  5. Send results back via A2A (FIX_INSTRUCTIONS or ALL_TESTS_PASSED).
  6. If max iterations reached, produce a final report of unresolved issues.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

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

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_API_BASE = os.getenv("AZURE_API_BASE", os.getenv("AZURE_OPENAI_ENDPOINT", ""))
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
QA_TEST_TIMEOUT = int(os.getenv("QA_TEST_TIMEOUT", "30"))


def _build_azure_llm() -> LLM:
    return LLM(
        model=f"azure/{AZURE_DEPLOYMENT}",
        api_key=AZURE_API_KEY,
        api_base=AZURE_API_BASE,
        api_version=AZURE_API_VERSION,
        temperature=0.2,
        max_tokens=4096,
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def build_qa_agent() -> Agent:
    """Construct the QA & Debugger CrewAI agent."""
    return Agent(
        role="Senior QA Engineer & Debugger",
        goal=(
            "Write comprehensive pytest tests for the given code based on "
            "acceptance criteria. Execute the tests, analyse failures, and "
            "produce clear, actionable fix instructions for the developer."
        ),
        backstory=(
            "You are a meticulous QA engineer with 12 years of experience. "
            "You write thorough tests that cover happy paths, edge cases, "
            "and error conditions. You are expert at reading test output "
            "and diagnosing root causes. You communicate fixes clearly."
        ),
        llm=_build_azure_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=5,
        max_retry_limit=2,
    )


# ---------------------------------------------------------------------------
# Test generation prompt
# ---------------------------------------------------------------------------
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
Write a pytest test file that tests the code above against ALL the acceptance
criteria. The tests must be real, executable pytest tests.

## RULES
1. Import the code under test using the correct module path relative to the
   workspace. The code file is at: {workspace_path}/{file_path}
2. Write at least one test per acceptance criterion.
3. Include edge case tests where appropriate.
4. Use descriptive test names: test_<what_it_tests>.
5. Use only pytest (no unittest).
6. Each test must have a clear assert statement.

## OUTPUT FORMAT
Output ONLY the Python test code. No markdown fences. No explanation.
Start directly with the import statements.
"""


# ---------------------------------------------------------------------------
# Test execution (sandboxed subprocess)
# ---------------------------------------------------------------------------
def _execute_tests(
    test_file_path: str,
    workspace_path: str,
    timeout: int = QA_TEST_TIMEOUT,
) -> dict:
    """Run pytest on the test file and return parsed results.

    Returns:
        dict with keys: returncode, stdout, stderr, passed, failed, total, failures
    """
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
        output = stdout + "\n" + stderr

        # Parse results from pytest output
        passed = len(re.findall(r" PASSED", stdout))
        failed = len(re.findall(r" FAILED", stdout))
        errors = len(re.findall(r" ERROR", stdout))
        total = passed + failed + errors

        # Extract failure details
        failures: list[dict] = []
        # Match patterns like "FAILED test_file.py::test_name - AssertionError: ..."
        fail_pattern = re.findall(
            r"FAILED\s+[\w./]+::(\w+)(?:\s*-\s*(.+))?", stdout
        )
        for test_name, error_msg in fail_pattern:
            failures.append({
                "test_name": test_name,
                "error_message": error_msg.strip() if error_msg else "See traceback above",
            })

        # Also capture short tracebacks for more detail
        tb_sections = re.findall(
            r"_{5,}\s+([\w.]+::[\w]+)\s+_{5,}\s*\n(.*?)(?=_{5,}|FAILED|$)",
            stdout,
            re.DOTALL,
        )
        tb_map = {}
        for tb_name, tb_body in tb_sections:
            short_name = tb_name.split("::")[-1] if "::" in tb_name else tb_name
            tb_map[short_name] = tb_body.strip()

        # Enrich failures with traceback info
        for f in failures:
            if f["test_name"] in tb_map:
                f["error_message"] += f"\nTraceback:\n{tb_map[f['test_name']]}"

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
            "returncode": -1,
            "stdout": "",
            "stderr": f"Tests timed out after {timeout} seconds",
            "passed": 0,
            "failed": 0,
            "total": 0,
            "failures": [{"test_name": "TIMEOUT", "error_message": f"Timed out after {timeout}s"}],
        }
    except Exception as exc:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "passed": 0,
            "failed": 0,
            "total": 0,
            "failures": [{"test_name": "EXECUTION_ERROR", "error_message": str(exc)}],
        }


# ---------------------------------------------------------------------------
# Fix instruction generation
# ---------------------------------------------------------------------------
def _build_fix_prompt(
    code: str,
    test_code: str,
    test_output: str,
    failures: list[dict],
) -> str:
    failures_str = "\n".join(
        f"  - {f['test_name']}: {f['error_message']}" for f in failures
    )

    return f"""
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
Analyse the test failures and produce fix instructions for the developer.

## OUTPUT FORMAT
Output ONLY a valid JSON array of fix instructions. No markdown fences.
Each instruction must have these fields:

[
  {{
    "issue_id": "FIX-001",
    "description": "Clear description of what is wrong",
    "severity": "high",
    "suggested_fix": "Specific code change to make",
    "related_test": "test_name_that_failed"
  }}
]
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_qa_review(
    review_request: ReviewRequestPayload,
    workspace_path: str,
    correlation_id: str,
) -> tuple[A2AMessage, str]:
    """Execute a full QA review cycle for one piece of code.

    Steps:
        1. Generate pytest tests from acceptance criteria.
        2. Write tests to workspace.
        3. Execute tests via subprocess.
        4. If all pass → return ALL_TESTS_PASSED message.
        5. If failures → generate fix instructions → return FIX_INSTRUCTIONS message.

    Returns:
        (A2AMessage to send back to Coder, test_file_content)
    """
    ws = Path(workspace_path)
    ws.mkdir(parents=True, exist_ok=True)

    # -- Step 1: Generate tests --------------------------------------------
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

    crew = Crew(
        agents=[qa_agent],
        tasks=[gen_task],
        process=Process.sequential,
        verbose=True,
    )

    crew_output = crew.kickoff()
    test_code = _clean_code_output(str(crew_output))

    # -- Step 2: Write test file -------------------------------------------
    test_filename = f"test_{review_request.task_id.lower().replace('-', '_')}.py"
    test_file_path = ws / test_filename
    test_file_path.write_text(test_code, encoding="utf-8")

    # -- Step 3: Execute tests ---------------------------------------------
    test_results = _execute_tests(
        test_file_path=str(test_file_path),
        workspace_path=workspace_path,
    )

    # -- Step 4/5: Build response message ----------------------------------
    if test_results["failed"] == 0 and test_results["passed"] > 0:
        # All tests passed
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
        # Tests failed — generate fix instructions
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
                TestFailure(
                    test_name=f["test_name"],
                    error_message=f["error_message"],
                )
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
    """Produce a final QA report when max iterations are reached."""
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _generate_fix_instructions(
    code: str,
    test_code: str,
    test_output: str,
    failures: list[dict],
) -> list[FixInstruction]:
    """Use the LLM to generate structured fix instructions from failures."""
    if not failures:
        return []

    qa_agent = build_qa_agent()
    prompt = _build_fix_prompt(code, test_code, test_output, failures)

    fix_task = Task(
        description=prompt,
        expected_output="A JSON array of fix instructions.",
        agent=qa_agent,
    )

    crew = Crew(
        agents=[qa_agent],
        tasks=[fix_task],
        process=Process.sequential,
        verbose=True,
    )

    crew_output = crew.kickoff()
    raw = str(crew_output)

    # Parse the JSON array
    try:
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
        # Fallback: create one instruction per failure
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
    """Extract a JSON array from LLM output."""
    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Find [ ... ]
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

    # Fenced code block
    pattern = r"```(?:json)?\s*(\[.*?\])\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []


def _clean_code_output(text: str) -> str:
    """Remove markdown fences and extra text from LLM code output."""
    # Remove ```python ... ``` fences
    pattern = r"```(?:python)?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no fences, check if it starts with import/from/def
    lines = text.strip().split("\n")
    code_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "def ", "class ", "#", "@")):
            code_start = i
            break

    return "\n".join(lines[code_start:]).strip()