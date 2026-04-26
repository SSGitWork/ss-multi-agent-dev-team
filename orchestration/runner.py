"""
Orchestration entry point for the multi-agent dev team.

Phase 1: Runs only the Coder Agent.
Future phases will add PM → Coder → QA handoff graphs here.
"""

from __future__ import annotations

import argparse
import json
import sys

from agents.coder_agent import run_coder_agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Dev Team - Phase 1: Coder Agent",
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="The coding task in natural language.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode (prompt for tasks in a loop).",
    )
    args = parser.parse_args()

    if args.interactive:
        _interactive_loop()
    elif args.task:
        _run_once(args.task)
    else:
        print("Provide a task or use --interactive mode.")
        print('Example: python -m orchestration.runner "Write a fibonacci function"')
        sys.exit(1)


def _run_once(task: str) -> None:
    print(f"\n{'='*60}")
    print(f"  TASK: {task}")
    print(f"{'='*60}\n")

    result = run_coder_agent(task)

    print(f"\n{'='*60}")
    print("  RESULT")
    print(f"{'='*60}")
    print(json.dumps(result.model_dump(), indent=2))


def _interactive_loop() -> None:
    print("\n Multi-Agent Dev Team - Coder Agent v1.0")
    print("   Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            task = input("Enter coding task: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not task:
            continue

        _run_once(task)
        print()


if __name__ == "__main__":
    main()
