"""
Per-agent token and cost tracking.

Records prompt_tokens, completion_tokens, and estimated USD cost
after every LLM call. Produces a PipelineCostReport at the end.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

from agents.config import get_settings
from agents.schemas_shared import AgentCostRecord, PipelineCostReport

logger = logging.getLogger(__name__)


class CostTracker:
    """Tracks token usage and cost per agent across a pipeline run."""

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id
        self._records: Dict[str, AgentCostRecord] = {}
        self._start_time = time.time()

    def get_or_create_record(self, agent_name: str, model_name: str = "") -> AgentCostRecord:
        """Retrieve or create a cost record for an agent.

        Args:
            agent_name: Name of the agent whose costs are being tracked.
            model_name: Model used by the agent.

        Returns:
            The AgentCostRecord associated with the agent.
        """
        if agent_name not in self._records:
            self._records[agent_name] = AgentCostRecord(
                agent_name=agent_name,
                model_name=model_name,
            )
        return self._records[agent_name]

    def record_llm_call(
        self,
        agent_name: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: float = 0.0,
    ) -> None:
        """Record token usage from a single LLM call."""
        record = self.get_or_create_record(agent_name, model_name)
        record.prompt_tokens += prompt_tokens
        record.completion_tokens += completion_tokens
        record.total_tokens += prompt_tokens + completion_tokens
        record.llm_calls += 1
        record.duration_ms += duration_ms

        # Calculate cost
        settings = get_settings()
        if "mini" in model_name.lower():
            input_cost = (prompt_tokens / 1000) * settings.cost.gpt4o_mini_input_per_1k
            output_cost = (completion_tokens / 1000) * settings.cost.gpt4o_mini_output_per_1k
        else:
            input_cost = (prompt_tokens / 1000) * settings.cost.gpt4o_input_per_1k
            output_cost = (completion_tokens / 1000) * settings.cost.gpt4o_output_per_1k

        record.estimated_cost_usd += input_cost + output_cost

        logger.debug(
            "LLM call recorded: agent=%s model=%s tokens=%d+%d cost=$%.6f",
            agent_name, model_name, prompt_tokens, completion_tokens,
            input_cost + output_cost,
        )

    def record_tool_call(self, agent_name: str) -> None:
        """Record a tool invocation performed by an agent."""
        record = self.get_or_create_record(agent_name)
        record.tool_calls += 1

    def record_cache_hit(self, agent_name: str) -> None:
        """Record a prompt cache hit for the specified agent."""
        record = self.get_or_create_record(agent_name)
        record.cache_hits += 1

    def generate_report(self) -> PipelineCostReport:
        """Generate the final pipeline cost report.

        Aggregates token usage, tool calls, cache hits, and estimated cost
        for all agents participating in the pipeline.
        """
        total_duration = (time.time() - self._start_time) * 1000
        report = PipelineCostReport(
            session_id=self.session_id,
            agents={name: record for name, record in self._records.items()},
            total_prompt_tokens=sum(r.prompt_tokens for r in self._records.values()),
            total_completion_tokens=sum(r.completion_tokens for r in self._records.values()),
            total_tokens=sum(r.total_tokens for r in self._records.values()),
            total_cost_usd=sum(r.estimated_cost_usd for r in self._records.values()),
            total_duration_ms=total_duration,
        )
        return report

    def save_report(self, report: Optional[PipelineCostReport] = None) -> str:
        """Save cost report to docs/cost_reports/ and return the file path."""
        if report is None:
            report = self.generate_report()

        settings = get_settings()
        report_dir = Path(settings.cost.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        filename = f"cost_report_{self.session_id}.json"
        filepath = report_dir / filename

        filepath.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Cost report saved to %s", filepath)
        return str(filepath)


# Global tracker instance (set per pipeline run)
_current_tracker: Optional[CostTracker] = None


def set_current_tracker(tracker: CostTracker) -> None:
    """Set the active CostTracker for the current pipeline session."""
    global _current_tracker
    _current_tracker = tracker


def get_current_tracker() -> Optional[CostTracker]:
    """Return the currently active CostTracker instance."""
    return _current_tracker
