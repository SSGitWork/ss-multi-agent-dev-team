"""
Phase 2 + Phase 3 – Orchestration Graph.

Phase 2: PM → Coder (sequential, no QA).
Phase 3: PM → Coder↔QA Review Loop (with A2A).

Both pipelines operate on SharedState within a single process.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable, List

from agents.coder_agent import run_coder_from_state
from agents.pm_agent import run_pm_agent
from agents.schemas_shared import SharedState
from orchestration.review_loop import run_review_loop


# ---------------------------------------------------------------------------
# Graph node type
# ---------------------------------------------------------------------------
class GraphNode:
    """A named step in the orchestration pipeline."""

    def __init__(self, name: str, fn: Callable[[SharedState], SharedState]) -> None:
        self.name = name
        self.fn = fn

    def execute(self, state: SharedState) -> SharedState:
        print(f"\n{'='*60}")
        print(f"  GRAPH NODE: {self.name}")
        print(f"{'='*60}\n")
        return self.fn(state)


# ---------------------------------------------------------------------------
# Orchestration Graph
# ---------------------------------------------------------------------------
class OrchestrationGraph:
    """Sequential pipeline of GraphNodes sharing a single SharedState."""

    def __init__(self) -> None:
        self._nodes: List[GraphNode] = []

    def add_node(
        self, name: str, fn: Callable[[SharedState], SharedState]
    ) -> "OrchestrationGraph":
        self._nodes.append(GraphNode(name=name, fn=fn))
        return self

    def run(self, requirement: str) -> SharedState:
        session_id = uuid.uuid4().hex[:12]
        workspace_path = str((Path("./workspace") / session_id).resolve())
        Path(workspace_path).mkdir(parents=True, exist_ok=True)

        state = SharedState(
            original_requirement=requirement,
            session_id=session_id,
            workspace_path=workspace_path,
            phase="initialized",
        )

        print(f"\n{'#'*60}")
        print(f"  MULTI-AGENT PIPELINE")
        print(f"  Session: {session_id}")
        print(f"  Nodes:   {' → '.join(n.name for n in self._nodes)}")
        print(f"{'#'*60}\n")

        for node in self._nodes:
            state = node.execute(state)

            if not state.success:
                print(f"\n⚠️  Pipeline stopped: {node.name} failed.")
                print(f"   Error: {state.error}")
                break

            if state.phase.endswith("_failed"):
                print(f"\n⚠️  Pipeline stopped at phase: {state.phase}")
                break

        return state


# ---------------------------------------------------------------------------
# Pre-built pipeline factories
# ---------------------------------------------------------------------------
def build_phase2_pipeline() -> OrchestrationGraph:
    """Phase 2 pipeline: PM → Coder (no QA)."""
    graph = OrchestrationGraph()
    graph.add_node("Product Manager", run_pm_agent)
    graph.add_node("Coder", run_coder_from_state)
    return graph


def build_default_pipeline() -> OrchestrationGraph:
    """Phase 3 pipeline: PM → Coder↔QA Review Loop."""
    graph = OrchestrationGraph()
    graph.add_node("Product Manager", run_pm_agent)
    graph.add_node("Coder + QA Review Loop", run_review_loop)
    return graph
