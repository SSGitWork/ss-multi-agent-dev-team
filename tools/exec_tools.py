import os
import subprocess
import textwrap
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()


def exec_python(code: str, timeout: int | None = None) -> Tuple[int, str, str]:
    """
    Execute Python code in a subprocess.
    Returns (exit_code, stdout, stderr).
    """
    if timeout is None:
        timeout = int(os.getenv("EXEC_TIMEOUT_SECONDS", "10"))

    # Dedent to reduce indentation issues
    dedented = textwrap.dedent(code)

    try:
        proc = subprocess.run(
            ["python", "-c", dedented],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return 124, "", f"Execution timed out after {timeout} seconds."

    return proc.returncode, proc.stdout, proc.stderr
