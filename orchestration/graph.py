"""
Phase 2 – Orchestration Graph.

Implements a PM → Coder pipeline within a single process.
Both agents operate on the same SharedState object.

The graph is a simple sequential pipeline for Phase 2:
    [User Requirement] → PM Node → Coder Node → [Final Output]

Phase 3 will add a QA node and iterative review loops.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable, List

from agents.coder_agent import run_coder_from_state
from agents.pm_agent import run_pm_agent
from agents.schemas_shared import SharedState


# ---------------------------------------------------------------------------
# Graph node type
# ---------------------------------------------------------------------------
class GraphNode:
    """A named step in the orchestration pipeline."""

    def __init__(self, name: str, fn: Callable[[SharedState], SharedState]) -> None:
        self.name = name
        self.fn = fn

    def execute(self, state: SharedState) -> SharedState:
        """Run this node's function, passing and returning SharedState."""
        print(f"\n{'='*60}")
        print(f"  GRAPH NODE: {self.name}")
        print(f"{'='*60}\n")
        return self.fn(state)


# ---------------------------------------------------------------------------
# Orchestration Graph
# ---------------------------------------------------------------------------
class OrchestrationGraph:
    """Sequential pipeline of GraphNodes sharing a single SharedState.

    Usage:
        graph = OrchestrationGraph()
        graph.add_node("PM", run_pm_agent)
        graph.add_node("Coder", run_coder_from_state)
        result = graph.run("Build a REST API")
    """

    def __init__(self) -> None:
        self._nodes: List[GraphNode] = []

    def add_node(
        self, name: str, fn: Callable[[SharedState], SharedState]
    ) -> "OrchestrationGraph":
        """Add a node to the pipeline. Returns self for chaining."""
        self._nodes.append(GraphNode(name=name, fn=fn))
        return self

    def run(self, requirement: str) -> SharedState:
        """Execute the full pipeline on a user requirement.

        1. Initialize SharedState with the requirement + workspace.
        2. Pass state through each node sequentially.
        3. Return the final state.
        """
        # -- Initialize state ----------------------------------------------
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

        # -- Execute nodes -------------------------------------------------
        for node in self._nodes:
            state = node.execute(state)

            # Stop early if a node failed
            if not state.success:
                print(f"\n⚠️  Pipeline stopped: {node.name} failed.")
                print(f"   Error: {state.error}")
                break

            # Stop if phase indicates failure
            if state.phase.endswith("_failed"):
                print(f"\n⚠️  Pipeline stopped at phase: {state.phase}")
                break

        return state


# ---------------------------------------------------------------------------
# Pre-built pipeline factory
# ---------------------------------------------------------------------------
def build_default_pipeline() -> OrchestrationGraph:
    """Build the standard Phase 2 pipeline: PM → Coder."""
    graph = OrchestrationGraph()
    graph.add_node("Product Manager", run_pm_agent)
    graph.add_node("Coder", run_coder_from_state)
    return graph
