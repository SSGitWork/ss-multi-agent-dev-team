"""
Sandboxed Python execution tool.

Runs code via *subprocess* in a child process with a configurable
timeout.  **Never** uses eval() or exec().
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

EXEC_TIMEOUT: int = int(os.getenv("EXEC_TIMEOUT_SECONDS", "10"))


@tool("exec_python")
def exec_python(
    code: str,
    workspace: str = "./workspace/default",
    timeout: int = EXEC_TIMEOUT,
) -> str:
    """Execute a Python code snippet in a sandboxed subprocess.

    The code is written to a temporary file inside the workspace and
    executed with the system Python interpreter.  stdout and stderr
    are captured and returned.

    Args:
        code:      The Python source code to execute.
        workspace: Session workspace directory.
        timeout:   Maximum execution time in seconds (default 10).

    Returns:
        Combined stdout + stderr output, or a timeout / error message.
    """
    ws = Path(workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)

    # Write code to a temp file inside the workspace
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
