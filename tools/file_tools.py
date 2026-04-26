"""
File I/O tools for the Coder Agent.

Provides read_file and write_file operations scoped to the
session workspace directory so generated artefacts are isolated.
"""

from __future__ import annotations

import os
from pathlib import Path

from crewai.tools import tool


def _resolve_path(workspace: str, relative_path: str) -> Path:
    """Resolve and validate that the target path stays inside the workspace."""
    base = Path(workspace).resolve()
    target = (base / relative_path).resolve()
    if not str(target).startswith(str(base)):
        raise PermissionError(
            f"Path traversal blocked: {relative_path!r} escapes workspace."
        )
    return target


@tool("read_file")
def read_file(file_path: str, workspace: str = "./workspace/default") -> str:
    """Read the contents of a file inside the workspace.

    Args:
        file_path: Relative path within the workspace.
        workspace: Absolute or relative path to the session workspace.

    Returns:
        The file contents as a string, or an error message.
    """
    try:
        target = _resolve_path(workspace, file_path)
        if not target.exists():
            return f"ERROR: File not found – {file_path}"
        return target.read_text(encoding="utf-8")
    except Exception as exc:
        return f"ERROR reading file: {exc}"


@tool("write_file")
def write_file(
    file_path: str,
    content: str,
    workspace: str = "./workspace/default",
) -> str:
    """Write (or overwrite) a file inside the workspace.

    Args:
        file_path: Relative path within the workspace.
        content:   The text content to write.
        workspace: Absolute or relative path to the session workspace.

    Returns:
        A confirmation message or an error string.
    """
    try:
        target = _resolve_path(workspace, file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"OK: Wrote {len(content)} chars to {file_path}"
    except Exception as exc:
        return f"ERROR writing file: {exc}"
