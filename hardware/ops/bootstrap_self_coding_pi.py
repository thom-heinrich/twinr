#!/usr/bin/env python3
"""Bootstrap the Pi-side self_coding Codex runtime from the leading repo.

Usage:
    python3 hardware/ops/bootstrap_self_coding_pi.py

The script reads `.env.pi`, installs the Pi-side Node/npm/Codex CLI
prerequisites, syncs the pinned SDK bridge and local Codex auth/config, and
then runs the explicit Twinr self-test remotely.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = PROJECT_ROOT / "src" / "twinr" / "ops" / "self_coding_pi.py"
_SPEC = importlib.util.spec_from_file_location("twinr_ops_self_coding_pi", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load bootstrap module from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
bootstrap_self_coding_pi = _MODULE.bootstrap_self_coding_pi


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Pi bootstrap helper."""

    parser = argparse.ArgumentParser(description="Bootstrap the Pi self_coding Codex runtime.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Leading Twinr repo root that owns the bridge files and .env.pi.",
    )
    parser.add_argument(
        "--pi-env-file",
        type=Path,
        default=PROJECT_ROOT / ".env.pi",
        help="Path to the Pi SSH credential env file.",
    )
    parser.add_argument(
        "--remote-root",
        default="/twinr",
        help="Runtime checkout root on the Raspberry Pi.",
    )
    parser.add_argument(
        "--local-codex-home",
        type=Path,
        default=None,
        help="Optional override for the local Codex home that provides auth.json and config.toml.",
    )
    return parser


def main() -> int:
    """Run the reproducible Pi bootstrap and print a compact JSON result."""

    args = build_parser().parse_args()
    result = bootstrap_self_coding_pi(
        project_root=args.project_root,
        pi_env_path=args.pi_env_file,
        remote_root=args.remote_root,
        local_codex_home=args.local_codex_home,
    )
    print(
        json.dumps(
            {
                "ready": result.ready,
                "host": result.host,
                "remote_root": result.remote_root,
                "codex_cli_version": result.codex_cli_version,
                "self_test_output": result.self_test_output,
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.ready else 1


if __name__ == "__main__":
    sys.exit(main())
