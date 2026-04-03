"""Re-exec hardware/ops entrypoints into the repo-local Python runtime.

Purpose
-------
Keep documented operator commands such as ``python3 hardware/ops/...`` stable
even on hosts where the ambient interpreter does not have Twinr's repo-local
dependencies installed. The helper re-launches the current script via
``/home/thh/twinr/.venv/bin/python`` unless it is already running there.

Usage
-----
Call ``ensure_repo_python()`` near the top of a small hardware/ops wrapper
before importing Twinr modules from ``src/``.

Outputs
-------
- No direct output on success.
- Raises ``RuntimeError`` when the repo venv is missing or when a re-exec loop
  would otherwise hide the real problem.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REEXEC_ENV = "TWINR_HARDWARE_OPS_REEXEC"


def _running_inside_repo_venv(*, repo_root: Path) -> bool:
    """Return whether the current interpreter is already the repo-local venv.

    Python's venv documentation recommends checking ``sys.prefix`` against
    ``sys.base_prefix`` because the venv executable may be a symlink to the
    base interpreter binary.
    """

    repo_venv = (repo_root / ".venv").resolve(strict=False)
    current_prefix = Path(getattr(sys, "prefix", "") or "").expanduser().resolve(strict=False)
    if current_prefix == repo_venv and getattr(sys, "prefix", "") != getattr(sys, "base_prefix", ""):
        return True
    current_executable = Path(sys.executable or "").expanduser().absolute()
    try:
        return current_executable.is_relative_to(repo_venv)
    except ValueError:
        return False


def ensure_repo_python(*, marker_env: str = _DEFAULT_REEXEC_ENV) -> None:
    """Re-exec the current hardware/ops script through the repo-local venv."""

    if _running_inside_repo_venv(repo_root=PROJECT_ROOT):
        return
    repo_python = (PROJECT_ROOT / ".venv" / "bin" / "python").absolute()
    if os.environ.get(marker_env) == "1":
        raise RuntimeError(
            "hardware/ops wrappers require the repo-local Python runtime at .venv/bin/python"
        )
    if not repo_python.exists():
        raise RuntimeError(
            "hardware/ops wrappers require the repo-local Python runtime at .venv/bin/python"
        )
    env = dict(os.environ)
    env[marker_env] = "1"
    os.execve(
        str(repo_python),
        [str(repo_python), str(Path(sys.argv[0]).resolve(strict=False)), *sys.argv[1:]],
        env,
    )


__all__ = ["PROJECT_ROOT", "ensure_repo_python"]
