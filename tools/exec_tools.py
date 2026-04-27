"""
Sandboxed Python execution tool with retry support.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from crewai.tools import tool

from agents.config import get_settings
from agents.cost_tracker import get_current_tracker

logger = logging.getLogger(__name__)


@tool("exec_python")
def exec_python(code: str, workspace: str = "./workspace/default", timeout: int = 0) -> str:
    """Execute Python code in a sandboxed subprocess."""
    try:
        tracker = get_current_tracker()
        if tracker:
            tracker.record_tool_call("coder")

        settings = get_settings()
        if timeout <= 0:
            timeout = settings.agent.exec_timeout

        ws = Path(workspace).resolve()
        ws.mkdir(parents=True, exist_ok=True)

        tmp_file = ws / f"_exec_{os.getpid()}.py"
        try:
            tmp_file.write_text(code, encoding="utf-8")

            result = subprocess.run(
                ["python", str(tmp_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(ws),
            )

            output_parts: list[str] = []
            if result.stdout:
                output_parts.append(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr}")
            if not output_parts:
                output_parts.append("(no output)")
            output_parts.append(f"Return code: {result.returncode}")
            return "\n".join(output_parts)

        except subprocess.TimeoutExpired:
            return f"ERROR: Execution timed out after {timeout} seconds."
        except Exception as exc:
            return f"ERROR executing code: {exc}"
        finally:
            if tmp_file.exists():
                tmp_file.unlink()

    except Exception as exc:
        logger.error("exec_python failed: %s", exc)
        return f"ERROR: {exc}"
