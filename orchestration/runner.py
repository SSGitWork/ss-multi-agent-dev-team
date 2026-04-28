"""
CLI entry point - production pipeline only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from orchestration.graph import build_default_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for running the multi-agent development pipeline.

    Parses command line arguments and executes the pipeline in either
    single-task mode or interactive mode.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Agent Dev Team — Production Pipeline",
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="The requirement in natural language.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.interactive:
        _interactive_loop()
    elif args.task:
        _run_pipeline(args.task)
    else:
        print("Provide a task or use --interactive mode.")
        print('Example: python -m orchestration.runner "Build a calculator"')
        sys.exit(1)


def _run_pipeline(requirement: str) -> None:
    print(f"Production pipeline - PM, CODER, QA")
    print(f"Requirement: {requirement}")

    pipeline = build_default_pipeline()
    state = pipeline.run(requirement)

    # Print final state
    print("Final sate :\n",state.to_json())

    # Print summary
    print(f"Summary \n Session Id: {state.session_id} \n Phase: {state.phase} \n Sucess: {state.success}")

    if state.technical_spec:
        print(f"Project Name: {state.technical_spec.name}")

    total = len(state.tasks)
    completed = len(state.get_completed_tasks())
    failed_tasks = [t for t in state.tasks if t.status.value == "failed"]
    print(f"Tasks:  {completed}/{total} completed, {len(failed_tasks)} failed")

    if state.error:
        print(f"Error: {state.error}")

    # Print cost report
    if state.cost_report:
        cr = state.cost_report
        print(f"\nCost Report")
        print(f"Total Tokens: {cr.total_tokens}")
        print(f"Total Cost: ${cr.total_cost_usd:.6f}")
        print(f"Duration: {cr.total_duration_ms:.0f}ms")
        for name, record in cr.agents.items():
            print(f"{name}: {record.total_tokens} tokens, ${record.estimated_cost_usd:.6f}, {record.llm_calls} calls")

    print(f"\nWorkspace: {state.workspace_path}")


def _interactive_loop() -> None:
    print("\nMulti-Agent Dev Team - Production Pipeline")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            task = input("Enter requirement: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not task:
            continue

        _run_pipeline(task)
        print()


if __name__ == "__main__":
    main()
