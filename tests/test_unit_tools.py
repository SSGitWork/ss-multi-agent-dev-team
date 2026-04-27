"""
Unit tests for file I/O and sandboxed execution tools.
"""

from __future__ import annotations

import pytest

from tools.file_tools import read_file, write_file
from tools.exec_tools import exec_python


class TestFileTools:
    def test_write_and_read(self, temp_workspace):
        result = write_file.run(file_path="test.py", content="print('hello')", workspace=temp_workspace)
        assert "OK" in result

        content = read_file.run(file_path="test.py", workspace=temp_workspace)
        assert "print('hello')" in content

    def test_read_nonexistent(self, temp_workspace):
        result = read_file.run(file_path="nope.py", workspace=temp_workspace)
        assert "ERROR" in result

    def test_path_traversal_blocked(self, temp_workspace):
        result = write_file.run(file_path="../../etc/passwd", content="hacked", workspace=temp_workspace)
        assert "ERROR" in result

    def test_nested_directory_creation(self, temp_workspace):
        result = write_file.run(file_path="sub/dir/test.py", content="pass", workspace=temp_workspace)
        assert "OK" in result
        content = read_file.run(file_path="sub/dir/test.py", workspace=temp_workspace)
        assert "pass" in content


class TestExecPython:
    def test_simple_execution(self, temp_workspace):
        result = exec_python.run(code="print('hello from sandbox')", workspace=temp_workspace)
        assert "hello from sandbox" in result

    def test_timeout(self, temp_workspace):
        result = exec_python.run(code="import time; time.sleep(30)", workspace=temp_workspace, timeout=2)
        assert "timed out" in result.lower()

    def test_syntax_error(self, temp_workspace):
        result = exec_python.run(code="def broken(", workspace=temp_workspace)
        assert "SyntaxError" in result or "STDERR" in result

    def test_return_code(self, temp_workspace):
        result = exec_python.run(code="import sys; sys.exit(0)", workspace=temp_workspace)
        assert "Return code: 0" in result
