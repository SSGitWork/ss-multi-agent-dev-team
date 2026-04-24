from agents.coder_agent import CoderAgent
from agents.models import CoderResult


def test_coder_agent_smoke():
    agent = CoderAgent()
    task = "Write a simple Python function add(a, b) that returns a + b and test it."
    result = agent.solve(task)

    # Basic field existence & types
    assert isinstance(result, CoderResult)
    assert isinstance(result.code, str)
    assert isinstance(result.explanation, str)
    assert isinstance(result.plan, str)
    assert isinstance(result.result, str)

    # Non-empty sanity checks (may fail if model is misconfigured)
    assert result.code.strip() != ""
    assert "add" in result.code or "sum" in result.code
