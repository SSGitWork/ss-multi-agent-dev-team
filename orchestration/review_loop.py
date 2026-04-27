"""
Iterative Coder ↔ QA Review Loop with tracing and cost tracking.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agents.coder_agent import revise_code_from_fixes, run_coder_task_with_reflection
from agents.memory import AgentMemory
from agents.message_bus import A2AMessageBus
from agents.qa_agent import generate_final_report, run_qa_review
from agents.schemas_a2a import (
    A2AIntent,
    A2AMessage,
    AgentRole,
    FixInstruction,
    ReviewRequestPayload,
)
from agents.schemas_shared import SharedState, TaskItem, TaskStatus
from agents.tracing import agent_span, record_span_metadata

load_dotenv()
logger = logging.getLogger(__name__)

MAX_QA_ITERATIONS: int = int(os.getenv("MAX_QA_ITERATIONS", "5"))


def run_review_loop_for_task(
    task_item: TaskItem,
    workspace_path: str,
    accumulated_code: str,
    bus: A2AMessageBus,
    memory: AgentMemory,
    max_iterations: int = MAX_QA_ITERATIONS,
) -> tuple[TaskItem, str, list[dict]]:
    """Run the full Coder → QA → Coder loop for one task."""
    correlation_id = uuid.uuid4().hex[:16]
    qa_reports: list[dict] = []
    last_test_output = ""
    all_fix_instructions: list[FixInstruction] = []
    resolved_issues: list[str] = []

    # -- Step 1: Coder generates code with self-reflection -----------------
    logger.info("Coder generating code for %s", task_item.task_id)
    task_item.status = TaskStatus.IN_PROGRESS

    try:
        current_code, explanation, file_path = run_coder_task_with_reflection(
            task_item=task_item,
            workspace_path=workspace_path,
            accumulated_code=accumulated_code,
            memory=memory,
        )
    except Exception as exc:
        logger.error("Coder failed for %s: %s", task_item.task_id, exc)
        task_item.status = TaskStatus.FAILED
        task_item.error = f"Coder failed: {exc}"
        task_item.completed_at = datetime.now(timezone.utc).isoformat()
        return task_item, "", qa_reports

    # -- Step 2: Iterative QA loop -----------------------------------------
    for iteration in range(1, max_iterations + 1):
        logger.info("QA review %s — iteration %d/%d", task_item.task_id, iteration, max_iterations)

        # Coder → QA via A2A
        review_payload = ReviewRequestPayload(
            task_id=task_item.task_id,
            code=current_code,
            file_path=file_path,
            acceptance_criteria=task_item.acceptance_criteria,
            workspace_path=workspace_path,
            iteration=iteration,
        )

        review_msg = A2AMessage(
            correlation_id=correlation_id,
            sender=AgentRole.CODER,
            receiver=AgentRole.QA,
            intent=A2AIntent.REVIEW_REQUEST,
            payload=review_payload.model_dump(),
        )
        bus.send(review_msg)

        # QA reviews
        try:
            qa_response, test_code = run_qa_review(
                review_request=review_payload,
                workspace_path=workspace_path,
                correlation_id=correlation_id,
            )
            bus.send(qa_response)
        except Exception as exc:
            logger.error("QA review failed: %s", exc)
            qa_reports.append({"iteration": iteration, "status": "error", "error": str(exc)})
            continue

        # Process response
        if qa_response.intent == A2AIntent.ALL_TESTS_PASSED:
            logger.info("ALL TESTS PASSED for %s on iteration %d", task_item.task_id, iteration)
            payload_data = qa_response.payload
            qa_reports.append({
                "iteration": iteration, "status": "all_passed",
                "total_tests": payload_data.get("total_tests", 0),
                "passed": payload_data.get("passed", 0),
            })

            task_item.code_output = current_code
            task_item.execution_result = (
                f"All {payload_data.get('passed', 0)} tests passed on iteration {iteration}."
            )
            task_item.status = TaskStatus.COMPLETED
            task_item.completed_at = datetime.now(timezone.utc).isoformat()
            return task_item, current_code, qa_reports

        elif qa_response.intent == A2AIntent.FIX_INSTRUCTIONS:
            payload_data = qa_response.payload
            fix_list = payload_data.get("fix_instructions", [])
            last_test_output = payload_data.get("raw_output", "")

            logger.info("%d test(s) failed, %d fix instruction(s)", payload_data.get("failed", 0), len(fix_list))

            qa_reports.append({
                "iteration": iteration, "status": "fixes_needed",
                "total_tests": payload_data.get("total_tests", 0),
                "passed": payload_data.get("passed", 0),
                "failed": payload_data.get("failed", 0),
                "fix_count": len(fix_list),
            })

            for fix in fix_list:
                if isinstance(fix, dict):
                    all_fix_instructions.append(FixInstruction(**fix))
                else:
                    all_fix_instructions.append(fix)

            if iteration >= max_iterations:
                logger.warning("Max iterations reached for %s", task_item.task_id)
                break

            # Coder revises
            try:
                current_code = revise_code_from_fixes(
                    code=current_code,
                    fix_instructions=fix_list,
                    task_description=task_item.description,
                    workspace_path=workspace_path,
                    file_path=file_path,
                    test_output=last_test_output,
                )
                for fix in fix_list:
                    fix_id = fix.get("issue_id", "unknown") if isinstance(fix, dict) else fix.issue_id
                    resolved_issues.append(fix_id)
            except Exception as exc:
                logger.error("Coder revision failed: %s", exc)
                qa_reports.append({"iteration": iteration, "status": "revision_error", "error": str(exc)})
                continue

    # -- Max iterations reached — final report -----------------------------
    last_iteration_fixes: list[FixInstruction] = []
    last_qa_msg = bus.get_latest_for(
        receiver=AgentRole.CODER,
        correlation_id=correlation_id,
        intent=A2AIntent.FIX_INSTRUCTIONS,
    )
    if last_qa_msg:
        raw_fixes = last_qa_msg.payload.get("fix_instructions", [])
        for fix in raw_fixes:
            if isinstance(fix, dict):
                last_iteration_fixes.append(FixInstruction(**fix))
            else:
                last_iteration_fixes.append(fix)

    final_report = generate_final_report(
        task_id=task_item.task_id,
        total_iterations=max_iterations,
        last_test_output=last_test_output,
        unresolved=last_iteration_fixes,
        resolved=resolved_issues,
    )

    max_iter_msg = A2AMessage(
        correlation_id=correlation_id,
        sender=AgentRole.QA,
        receiver=AgentRole.CODER,
        intent=A2AIntent.MAX_ITERATIONS_REACHED,
        payload=final_report.model_dump(),
    )
    bus.send(max_iter_msg)

    task_item.code_output = current_code
    task_item.execution_result = (
        f"Max iterations ({max_iterations}) reached. "
        f"{len(last_iteration_fixes)} unresolved issue(s)."
    )
    task_item.status = TaskStatus.COMPLETED
    task_item.completed_at = datetime.now(timezone.utc).isoformat()

    qa_reports.append({
        "iteration": "final_report",
        "status": "max_iterations_reached",
        "unresolved_count": len(last_iteration_fixes),
        "resolved_count": len(resolved_issues),
        "recommendation": final_report.recommendation,
    })

    return task_item, current_code, qa_reports


def run_review_loop(state: SharedState) -> SharedState:
    """Run the Coder ↔ QA review loop for every pending task."""
    if not state.session_id or not state.workspace_path:
        session_id = uuid.uuid4().hex[:12]
        workspace_path = str((Path("./workspace") / session_id).resolve())
        state.session_id = session_id
        state.workspace_path = workspace_path
    Path(state.workspace_path).mkdir(parents=True, exist_ok=True)

    bus = A2AMessageBus()
    memory = AgentMemory()
    all_explanations: list[str] = []

    pending_tasks = state.get_pending_tasks()
    if not pending_tasks:
        state.phase = "review_complete"
        state.final_explanation = "No pending tasks to process."
        return state

    for task_item in pending_tasks:
        logger.info("Processing task: %s — %s", task_item.task_id, task_item.title)

        updated_task, final_code, qa_reports = run_review_loop_for_task(
            task_item=task_item,
            workspace_path=state.workspace_path,
            accumulated_code=state.accumulated_code,
            bus=bus,
            memory=memory,
        )

        if final_code.strip():
            header = f"\n# === {task_item.task_id}: {task_item.title} ===\n"
            state.accumulated_code += header + final_code + "\n"

        status_emoji = "✅" if updated_task.status == TaskStatus.COMPLETED else "❌"
        all_explanations.append(
            f"{status_emoji} **{task_item.task_id} — {task_item.title}**: "
            f"{updated_task.execution_result}"
        )

    state.final_explanation = "\n\n".join(all_explanations)
    state.phase = "review_complete"
    state.pm_notes += f"\n\n--- A2A Message Log ---\n{bus.summary()}"

    failed = [t for t in state.tasks if t.status == TaskStatus.FAILED]
    if failed:
        state.success = False
        state.error = f"{len(failed)} task(s) failed: {[t.task_id for t in failed]}"

    return state
