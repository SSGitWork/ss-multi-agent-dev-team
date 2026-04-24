from __future__ import annotations

from agents.coder_agent import CoderAgent


def run_coder_agent(task: str):
    agent = CoderAgent()
    result = agent.solve(task)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Coder Agent on a task.")
    parser.add_argument("task", type=str, help="Natural language coding task.")
    args = parser.parse_args()

    outcome = run_coder_agent(args.task)
    print("=== PLAN ===")
    print(outcome.plan)
    print("\n=== CODE ===")
    print(outcome.code)
    print("\n=== EXPLANATION ===")
    print(outcome.explanation)
    print("\n=== RESULT ===")
    print(outcome.result)
