"""
File I/O tools with retry wrapper and path traversal protection.
"""

from __future__ import annotations

import logging
from pathlib import Path

from crewai.tools import tool

from agents.cost_tracker import get_current_tracker

logger = logging.getLogger(__name__)


def _resolve_path(workspace: str, relative_path: str) -> Path:
    base = Path(workspace).resolve()
    target = (base / relative_path).resolve()
    if not str(target).startswith(str(base)):
        raise PermissionError(f"Path traversal blocked: {relative_path!r} escapes workspace.")
    return target


@tool("read_file")
def read_file(file_path: str, workspace: str = "./workspace/default") -> str:
    """Read the contents of a file inside the workspace."""
    try:
        tracker = get_current_tracker()
        if tracker:
            tracker.record_tool_call("coder")

        target = _resolve_path(workspace, file_path)
        if not target.exists():
            return f"ERROR: File not found – {file_path}"
        return target.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("read_file failed: %s", exc)
        return f"ERROR reading file: {exc}"


@tool("write_file")
def write_file(file_path: str, content: str, workspace: str = "./workspace/default") -> str:
    """Write (or overwrite) a file inside the workspace."""
    try:
        tracker = get_current_tracker()
        if tracker:
            tracker.record_tool_call("coder")

        target = _resolve_path(workspace, file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"OK: Wrote {len(content)} chars to {file_path}"
    except Exception as exc:
        logger.error("write_file failed: %s", exc)
        return f"ERROR writing file: {exc}"
