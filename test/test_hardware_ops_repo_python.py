from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "hardware" / "ops"))

import _repo_python  # noqa: E402


class HardwareOpsRepoPythonTests(unittest.TestCase):
    def test_ensure_repo_python_noops_when_already_running_in_repo_venv(self) -> None:
        repo_python = (_repo_python.PROJECT_ROOT / ".venv" / "bin" / "python").absolute()
        repo_prefix = str((_repo_python.PROJECT_ROOT / ".venv").resolve(strict=False))
        with (
            patch.object(_repo_python.sys, "executable", str(repo_python)),
            patch.object(_repo_python.sys, "prefix", repo_prefix),
            patch.object(_repo_python.sys, "base_prefix", "/usr"),
        ):
            _repo_python.ensure_repo_python()

    def test_ensure_repo_python_execves_repo_venv_when_host_python_differs(self) -> None:
        repo_python = (_repo_python.PROJECT_ROOT / ".venv" / "bin" / "python").absolute()
        with (
            patch.object(_repo_python.sys, "executable", "/usr/bin/python3"),
            patch.object(_repo_python.sys, "prefix", "/usr"),
            patch.object(_repo_python.sys, "base_prefix", "/usr"),
            patch.object(_repo_python.sys, "argv", [str(PROJECT_ROOT / "hardware" / "ops" / "repair_remote_chonkydb.py"), "--no-restart"]),
            patch.dict(_repo_python.os.environ, {}, clear=True),
            patch.object(_repo_python.os, "execve") as execve_mock,
        ):
            _repo_python.ensure_repo_python()

        execve_mock.assert_called_once()
        called_executable, called_argv, called_env = execve_mock.call_args.args
        self.assertEqual(called_executable, str(repo_python))
        self.assertEqual(
            called_argv,
            [
                str(repo_python),
                str((PROJECT_ROOT / "hardware" / "ops" / "repair_remote_chonkydb.py").resolve(strict=False)),
                "--no-restart",
            ],
        )
        self.assertEqual(called_env["TWINR_HARDWARE_OPS_REEXEC"], "1")

    def test_ensure_repo_python_fails_closed_on_reexec_loop(self) -> None:
        with (
            patch.object(_repo_python.sys, "executable", "/usr/bin/python3"),
            patch.object(_repo_python.sys, "prefix", "/usr"),
            patch.object(_repo_python.sys, "base_prefix", "/usr"),
            patch.dict(_repo_python.os.environ, {"TWINR_HARDWARE_OPS_REEXEC": "1"}, clear=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "repo-local Python runtime"):
                _repo_python.ensure_repo_python()


if __name__ == "__main__":
    unittest.main()
