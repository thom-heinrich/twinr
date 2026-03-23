#!/usr/bin/env python3
"""Check the Pi OpenAI env contract and optionally run one live provider probe.

Purpose
-------
Use this operator script to prove that `/twinr/.env` is sufficient for direct
OpenAI-backed acceptance runs on the Raspberry Pi. It validates the local env
file fail-closed and can optionally execute a bounded real provider request
without any manual `OPENAI_API_KEY` shell injection.

Usage
-----
Command-line invocation examples::

    python3 hardware/ops/check_pi_openai_env_contract.py
    python3 hardware/ops/check_pi_openai_env_contract.py --env-file /twinr/.env --live-search "Was ist aktuell in der Hamburger Lokalpolitik los?"
    python3 hardware/ops/check_pi_openai_env_contract.py --env-file /twinr/.env --live-text "Antworte nur mit: ok."

Outputs
-------
- One compact JSON object describing the env contract result.
- Optional nested live-probe result when `--live-text` or `--live-search` is set.
- Exit code 0 when the env contract (and optional live probe) passes, otherwise 1.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.openai_env_contract import check_openai_env_contract
from twinr.providers.openai.api.backend import OpenAIBackend


@dataclass(frozen=True, slots=True)
class LiveProbeResult:
    """Describe one optional bounded live provider probe."""

    kind: str
    ok: bool
    elapsed_s: float
    requested_model: str | None
    actual_model: str | None
    answer_head: str
    error: str | None = None


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the env-contract probe."""

    parser = argparse.ArgumentParser(
        description="Fail-closed check for Twinr's Pi-side OpenAI env contract.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("/twinr/.env"),
        help="Twinr env file to validate.",
    )
    probe_group = parser.add_mutually_exclusive_group()
    probe_group.add_argument(
        "--live-text",
        default=None,
        help="Run one bounded non-search OpenAI text response after the env contract check passes.",
    )
    probe_group.add_argument(
        "--live-search",
        default=None,
        help="Run one bounded OpenAI live-search response after the env contract check passes.",
    )
    return parser


def main() -> int:
    """Run the env-contract check and print one JSON payload."""

    args = build_parser().parse_args()
    status = check_openai_env_contract(args.env_file)
    payload: dict[str, object] = status.to_dict()
    exit_code = 0 if status.ok else 1

    if status.ok and (args.live_text or args.live_search):
        live_probe = _run_live_probe(
            env_file=args.env_file,
            live_text=args.live_text,
            live_search=args.live_search,
        )
        payload["live_probe"] = asdict(live_probe)
        if not live_probe.ok:
            exit_code = 1

    print(json.dumps(payload, ensure_ascii=False))
    return exit_code


def _run_live_probe(
    *,
    env_file: Path,
    live_text: str | None,
    live_search: str | None,
) -> LiveProbeResult:
    """Run one optional real OpenAI probe using the env-backed Twinr config."""

    started = time.perf_counter()
    config = TwinrConfig.from_env(env_file)
    backend = OpenAIBackend(config=config)

    try:
        if live_search is not None:
            result = backend.search_live_info_with_metadata(live_search)
            answer = str(result.answer or "")
            return LiveProbeResult(
                kind="search",
                ok=bool(answer.strip()),
                elapsed_s=round(time.perf_counter() - started, 3),
                requested_model=_clean_text(config.openai_search_model),
                actual_model=_clean_text(result.model),
                answer_head=_answer_head(answer),
            )

        text_prompt = str(live_text or "").strip()
        response = backend.respond_with_metadata(text_prompt, allow_web_search=False)
        answer = str(response.text or "")
        return LiveProbeResult(
            kind="text",
            ok=bool(answer.strip()),
            elapsed_s=round(time.perf_counter() - started, 3),
            requested_model=_clean_text(config.default_model),
            actual_model=_clean_text(response.model),
            answer_head=_answer_head(answer),
        )
    except Exception as exc:
        return LiveProbeResult(
            kind="search" if live_search is not None else "text",
            ok=False,
            elapsed_s=round(time.perf_counter() - started, 3),
            requested_model=None,
            actual_model=None,
            answer_head="",
            error=str(exc),
        )


def _answer_head(value: str) -> str:
    """Return a compact single-line answer preview."""

    return " ".join(value.split())[:220]


def _clean_text(value: object) -> str | None:
    """Normalize metadata values to short strings for JSON output."""

    text = str(value or "").strip()
    return text or None


if __name__ == "__main__":
    raise SystemExit(main())
