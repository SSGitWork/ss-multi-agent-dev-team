"""
Orchestration Graph with single root trace and cost tracking.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Callable, List

from agents.cost_tracker import CostTracker, set_current_tracker
from agents.pm_agent import run_pm_agent
from agents.schemas_shared import SharedState
from agents.tracing import record_span_metadata, root_span
from orchestration.review_loop import run_review_loop

logger = logging.getLogger(__name__)


class GraphNode:
    """Represents a single execution node in the orchestration graph."""

    def __init__(self, name: str, fn: Callable[[SharedState], SharedState]) -> None:
        self.name = name
        self.fn = fn

    def execute(self, state: SharedState) -> SharedState:
        """Execute the node's function using the provided shared state."""
        logger.info("Executing graph node: %s", self.name)
        print(f"Graph node: {self.name}")
        return self.fn(state)


class OrchestrationGraph:
    """Pipeline graph that executes agents sequentially with shared state."""

    def __init__(self) -> None:
        self._nodes: List[GraphNode] = []

    def add_node(self, name: str, fn: Callable[[SharedState], SharedState]) -> "OrchestrationGraph":
        """Add a new node to the orchestration pipeline."""
        self._nodes.append(GraphNode(name=name, fn=fn))
        return self

    def run(self, requirement: str) -> SharedState:
        """Execute the orchestration pipeline for a given requirement.

        Initializes the session, workspace, tracing root span, and cost
        tracking, then runs each graph node sequentially until completion
        or failure.
        """
        session_id = uuid.uuid4().hex[:12]
        workspace_path = str((Path("./workspace") / session_id).resolve())
        Path(workspace_path).mkdir(parents=True, exist_ok=True)

        state = SharedState(
            original_requirement=requirement,
            session_id=session_id,
            workspace_path=workspace_path,
            phase="initialized",
        )

        # Initialize cost tracker
        tracker = CostTracker(session_id=session_id)
        set_current_tracker(tracker)

        print(f"Multi agent pipeline - Session: {session_id}")
        print(f"Nodes: {' '.join(n.name for n in self._nodes)}")

        # Single root trace spanning all agents
        with root_span("pipeline_run", {"session_id": session_id}) as span:
            record_span_metadata(span, requirement=requirement[:200])

            for node in self._nodes:
                state = node.execute(state)

                if not state.success:
                    logger.warning("Pipeline stopped: %s failed - %s", node.name, state.error)
                    break

                if state.phase.endswith("_failed"):
                    logger.warning("Pipeline stopped at phase: %s", state.phase)
                    break

            # Generate and attach cost report
            cost_report = tracker.generate_report()
            state.cost_report = cost_report

            record_span_metadata(
                span,
                total_tokens=cost_report.total_tokens,
                total_cost_usd=cost_report.total_cost_usd,
                total_duration_ms=cost_report.total_duration_ms,
            )

        # Save cost report to file
        try:
            report_path = tracker.save_report(cost_report)
            logger.info("Cost report saved: %s", report_path)
        except Exception as exc:
            logger.warning("Failed to save cost report: %s", exc)

        return state


def build_default_pipeline() -> OrchestrationGraph:
    """Production pipeline: PM, Coder, QA Review Loop."""
    graph = OrchestrationGraph()
    graph.add_node("Product Manager", run_pm_agent)
    graph.add_node("Coder + QA Review Loop", run_review_loop)
    return graph
