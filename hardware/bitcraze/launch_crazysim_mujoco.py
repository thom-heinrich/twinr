#!/usr/bin/env python3
"""Launch one CrazySim MuJoCo session through Twinr's repo-owned bridge."""

from __future__ import annotations

from pathlib import Path
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from twinr.hardware.crazysim_mujoco_bridge import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
