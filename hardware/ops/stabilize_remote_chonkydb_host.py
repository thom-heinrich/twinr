#!/usr/bin/env python3
"""Quiesce shared-host jobs that destabilize Twinr's dedicated ChonkyDB host.

Purpose
-------
Use this operator script when `https://tessairact.com:2149` stays reachable but
becomes slow, flaky, or freeze-prone because the backend host is contended by
unrelated CAIA timers, services, or path-triggered jobs. The command raises the
dedicated Twinr backend CPU/IO weights and disables a curated set of known
conflict units on the remote host.

Usage
-----
Command-line invocation examples::

    python3 hardware/ops/stabilize_remote_chonkydb_host.py
    python3 hardware/ops/stabilize_remote_chonkydb_host.py --settle-s 12

Outputs
-------
- One compact JSON object describing the public probe before/after, the backend
  service weights, and the before/after state for the conflict-unit set.
- Exit code 0 when the public endpoint is healthy after stabilization,
  otherwise 1.
"""

from __future__ import annotations

import sys

from _repo_python import PROJECT_ROOT, ensure_repo_python

ensure_repo_python()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from twinr.ops.remote_chonkydb_host_stabilizer import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
