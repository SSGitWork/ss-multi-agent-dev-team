import os
from typing import Optional


BASE_DIR = os.path.abspath(os.getcwd())


def _safe_path(path: str) -> str:
    """Resolve a safe absolute path under the project directory."""
    abs_path = os.path.abspath(os.path.join(BASE_DIR, path))
    if not abs_path.startswith(BASE_DIR):
        raise ValueError("Attempted to access path outside project directory.")
    return abs_path


def read_file(path: str, encoding: str = "utf-8") -> str:
    """Read the contents of a text file."""
    file_path = _safe_path(path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


def write_file(path: str, content: str, encoding: str = "utf-8", overwrite: bool = True) -> str:
    """Write content to a text file, optionally preventing overwrite."""
    file_path = _safe_path(path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not overwrite and os.path.exists(file_path):
        raise FileExistsError(f"File already exists: {path}")

    with open(file_path, "w", encoding=encoding) as f:
        f.write(content)

    return f"Wrote {len(content)} characters to {path}"