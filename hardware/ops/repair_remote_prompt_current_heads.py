#!/usr/bin/env python3
"""Force-repair unreadable prompt current heads on one remote ChonkyDB namespace.

Usage
-----
Typical operator invocation against a direct backend SSH tunnel::

    ssh -L 43044:127.0.0.1:3044 thh@thh1986.ddns.net
    python3 hardware/ops/repair_remote_prompt_current_heads.py \
        --namespace twinr_longterm_v1:twinr:a7f1ed265838 \
        --base-url http://127.0.0.1:43044 \
        --force

The script prints one compact JSON result and exits with code 0 only when all
requested prompt current heads are healthy at the end of the run.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from twinr.ops.remote_prompt_current_head_repair import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
