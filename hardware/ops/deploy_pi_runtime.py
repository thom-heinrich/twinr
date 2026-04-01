#!/usr/bin/env python3
"""Deploy the authoritative Twinr repo state to the Raspberry Pi.

Purpose
-------
Use this operator script when the current leading-repo state in
``/home/thh/twinr`` must be pushed to the Pi acceptance checkout under
``/twinr``. The command mirrors the repo, optionally overwrites the Pi
runtime ``.env`` from the local repo, refreshes the editable install, installs
optional mirrored browser-automation runtime manifests when present, installs
the productive base systemd units plus any repo-backed Pi runtime units that
are already enabled on the host, supports explicit first-rollout activation for
optional Pi units, restarts them, and verifies that the services and env
contract came back healthy.

Usage
-----
Command-line invocation examples::

    python3 hardware/ops/deploy_pi_runtime.py
    python3 hardware/ops/deploy_pi_runtime.py --live-text "Antworte nur mit: ok."
    python3 hardware/ops/deploy_pi_runtime.py --skip-env-sync --service twinr-runtime-supervisor

Outputs
-------
- Live structured progress JSON lines on stderr while long-running phases are active.
- One compact JSON object describing the completed deploy and verification run on stdout.
- Exit code 0 on success, otherwise 1 with a phase-specific JSON error payload.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _reexec_repo_python_if_needed() -> None:
    """Re-exec into the repo venv when the ambient Python is too old."""

    if sys.version_info >= (3, 11):
        return
    if os.environ.get("TWINR_DEPLOY_REEXEC") == "1":
        raise RuntimeError("deploy_pi_runtime.py requires Python 3.11+")
    repo_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if not repo_python.exists():
        raise RuntimeError(
            "deploy_pi_runtime.py requires Python 3.11+ or the repo venv at .venv/bin/python"
        )
    env = dict(os.environ)
    env["TWINR_DEPLOY_REEXEC"] = "1"
    os.execve(
        str(repo_python),
        [str(repo_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        env,
    )


_reexec_repo_python_if_needed()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from twinr.ops.pi_runtime_deploy import DEFAULT_DEPLOY_SERVICES, PiRuntimeDeployError, deploy_pi_runtime  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Pi deploy helper."""

    parser = argparse.ArgumentParser(
        description="Deploy /home/thh/twinr to the Pi acceptance runtime and verify restart health.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Leading Twinr repo root that is treated as authoritative.",
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
        help="Twinr runtime checkout root on the Raspberry Pi.",
    )
    parser.add_argument(
        "--env-source",
        type=Path,
        default=PROJECT_ROOT / ".env",
        help="Authoritative local Twinr env file to copy onto the Pi.",
    )
    parser.add_argument(
        "--remote-env-path",
        default=None,
        help="Optional override for the runtime env target on the Pi.",
    )
    parser.add_argument(
        "--service",
        action="append",
        default=[],
        help=(
            "Explicit systemd unit to install/restart/verify. Repeat as needed. "
            "Without this flag the deploy always manages the base services "
            f"({', '.join(DEFAULT_DEPLOY_SERVICES)}) and also picks up any "
            "repo-backed Pi runtime units that are already enabled on the Pi."
            " Use this flag when you intentionally want to replace that "
            "automatic target set."
        ),
    )
    parser.add_argument(
        "--rollout-service",
        action="append",
        default=[],
        help=(
            "Add one repo-backed Pi runtime unit to the deploy target set and "
            "enable it even if it is not enabled on the Pi yet. Repeat as "
            "needed. This is the explicit first-rollout path for optional "
            "services such as twinr-whatsapp-channel.service."
        ),
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=180.0,
        help="Per-SSH/SCP command timeout in seconds.",
    )
    parser.add_argument(
        "--service-wait-s",
        type=float,
        default=30.0,
        help="Maximum wait time for restarted services to report healthy.",
    )
    parser.add_argument(
        "--skip-env-sync",
        action="store_true",
        help="Keep the existing Pi runtime env file instead of overwriting it from the leading repo.",
    )
    parser.add_argument(
        "--skip-editable-install",
        action="store_true",
        help="Skip the Pi-side `pip install -e /twinr` refresh.",
    )
    parser.add_argument(
        "--install-with-deps",
        action="store_true",
        help="Let the editable refresh also resolve runtime dependencies instead of the default `--no-deps` install.",
    )
    parser.add_argument(
        "--skip-systemd-install",
        action="store_true",
        help="Skip copying unit files into /etc/systemd/system and daemon-reload.",
    )
    parser.add_argument(
        "--skip-env-contract-check",
        action="store_true",
        help="Skip the bounded Pi env-contract verification after restart.",
    )
    parser.add_argument(
        "--skip-retention-canary",
        action="store_true",
        help="Skip the bounded remote-memory retention canary after the normal deploy health checks.",
    )
    probe_group = parser.add_mutually_exclusive_group()
    probe_group.add_argument(
        "--live-text",
        default=None,
        help="Run one bounded non-search OpenAI text probe during env-contract verification.",
    )
    probe_group.add_argument(
        "--live-search",
        default=None,
        help="Run one bounded search-backed OpenAI probe during env-contract verification.",
    )
    return parser


def main() -> int:
    """Run the Pi deploy helper and print one JSON payload."""

    args = build_parser().parse_args()
    services = tuple(args.service) if args.service else None

    def _emit_progress(payload: dict[str, object]) -> None:
        """Write live deploy progress to stderr without polluting stdout JSON."""

        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)

    try:
        result = deploy_pi_runtime(
            project_root=args.project_root,
            pi_env_path=args.pi_env_file,
            remote_root=args.remote_root,
            services=services,
            rollout_services=tuple(args.rollout_service),
            env_source_path=args.env_source,
            remote_env_path=args.remote_env_path,
            timeout_s=args.timeout_s,
            service_wait_s=args.service_wait_s,
            sync_env=not args.skip_env_sync,
            install_editable=not args.skip_editable_install,
            install_with_deps=args.install_with_deps,
            install_systemd_units=not args.skip_systemd_install,
            verify_env_contract=not args.skip_env_contract_check,
            verify_retention_canary=not args.skip_retention_canary,
            live_text=args.live_text,
            live_search=args.live_search,
            progress_callback=_emit_progress,
        )
    except PiRuntimeDeployError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "phase": exc.phase,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "phase": "bootstrap",
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 1

    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
