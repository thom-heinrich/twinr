#!/usr/bin/env python3
"""Diagnose and optionally repair the dedicated remote ChonkyDB backend.

Purpose
-------
Use this operator script when Twinr's public remote-memory endpoint is
unhealthy and you need to prove whether the fault sits at the public URL, the
dedicated backend systemd unit, or the backend's local loopback instance on
`127.0.0.1:3044`. The command avoids blind restarts by only restarting the
backend service when the backend itself is the proven failing layer.

Usage
-----
Command-line invocation examples::

    python3 hardware/ops/repair_remote_chonkydb.py
    python3 hardware/ops/repair_remote_chonkydb.py --no-restart
    python3 hardware/ops/repair_remote_chonkydb.py --wait-ready-s 180

Outputs
-------
- One compact JSON object describing the public probe, backend service state,
  backend loopback probe, chosen repair plan, and final status.
- Exit code 0 when the public endpoint is healthy at the end of the run,
  otherwise 1.
"""

from __future__ import annotations

import sys

from _repo_python import PROJECT_ROOT, ensure_repo_python

ensure_repo_python()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from twinr.ops.remote_chonkydb_repair import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
