"""
Orchestration entry point for the multi-agent dev team.

Phase 1: --phase1  → Coder Agent only.
Phase 2: --phase2  → PM → Coder (no QA).
Phase 3: (default) → PM → Coder↔QA Review Loop with A2A.
"""

from __future__ import annotations

import argparse
import json
import sys

from agents.coder_agent import run_coder_agent
from orchestration.graph import build_default_pipeline, build_phase2_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Dev Team – Orchestration Runner",
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="The requirement / coding task in natural language.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode.",
    )
    parser.add_argument(
        "--phase1",
        action="store_true",
        help="Phase 1: Coder agent only.",
    )
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="Phase 2: PM → Coder (no QA).",
    )
    args = parser.parse_args()

    if args.interactive:
        _interactive_loop(phase1=args.phase1, phase2=args.phase2)
    elif args.task:
        if args.phase1:
            _run_phase1(args.task)
        elif args.phase2:
            _run_phase2(args.task)
        else:
            _run_phase3(args.task)
    else:
        print("Provide a task or use --interactive mode.")
        print('Phase 3 (default): python -m orchestration.runner "Build a calculator"')
        print('Phase 2:           python -m orchestration.runner --phase2 "Build a calculator"')
        print('Phase 1:           python -m orchestration.runner --phase1 "Write hello world"')
        sys.exit(1)


def _run_phase1(task: str) -> None:
    print(f"\n{'='*60}")
    print(f"  PHASE 1 — CODER AGENT ONLY")
    print(f"  TASK: {task}")
    print(f"{'='*60}\n")

    result = run_coder_agent(task)

    print(f"\n{'='*60}")
    print("  RESULT")
    print(f"{'='*60}")
    print(json.dumps(result.model_dump(), indent=2))


def _run_phase2(requirement: str) -> None:
    print(f"\n{'='*60}")
    print(f"  PHASE 2 — PM → CODER PIPELINE")
    print(f"  REQUIREMENT: {requirement}")
    print(f"{'='*60}\n")

    pipeline = build_phase2_pipeline()
    state = pipeline.run(requirement)
    _print_state_summary(state)


def _run_phase3(requirement: str) -> None:
    print(f"\n{'='*60}")
    print(f"  PHASE 3 — PM → CODER ↔ QA REVIEW LOOP (A2A)")
    print(f"  REQUIREMENT: {requirement}")
    print(f"{'='*60}\n")

    pipeline = build_default_pipeline()
    state = pipeline.run(requirement)
    _print_state_summary(state)


def _print_state_summary(state) -> None:
    print(f"\n{'='*60}")
    print("  FINAL STATE (JSON)")
    print(f"{'='*60}")
    print(state.to_json())

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Session ID:    {state.session_id}")
    print(f"  Phase:         {state.phase}")
    print(f"  Success:       {state.success}")

    if state.technical_spec:
        print(f"  Project Name:  {state.technical_spec.name}")

    total = len(state.tasks)
    completed = len(state.get_completed_tasks())
    failed_tasks = [t for t in state.tasks if t.status.value == "failed"]
    print(f"  Tasks:         {completed}/{total} completed, {len(failed_tasks)} failed")

    if state.error:
        print(f"  Error:         {state.error}")

    print(f"\n  Workspace:     {state.workspace_path}")
    print(f"{'='*60}\n")


def _interactive_loop(phase1: bool = False, phase2: bool = False) -> None:
    if phase1:
        mode = "Phase 1 (Coder only)"
    elif phase2:
        mode = "Phase 2 (PM → Coder)"
    else:
        mode = "Phase 3 (PM → Coder ↔ QA)"

    print(f"\n🤖 Multi-Agent Dev Team — {mode}")
    print("   Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            task = input("📝 Enter requirement: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not task:
            continue

        if phase1:
            _run_phase1(task)
        elif phase2:
            _run_phase2(task)
        else:
            _run_phase3(task)

        print()


if __name__ == "__main__":
    main()