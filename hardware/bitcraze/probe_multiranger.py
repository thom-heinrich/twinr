#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cflib>=0.1.31",
# ]
# ///
"""Probe Crazyflie deck presence and Multi-ranger distance data.

Purpose
-------
Connect to one Crazyflie over the configured radio URI, read the current deck
presence flags, and sample the Multi-ranger distances in all supported
directions. This gives Twinr an immediate acceptance check when a Multi-ranger
deck or new deck stack arrives.

Usage
-----
Command-line examples::

    /twinr/bitcraze/.venv/bin/python hardware/bitcraze/probe_multiranger.py
    /twinr/bitcraze/.venv/bin/python hardware/bitcraze/probe_multiranger.py --json
    /twinr/bitcraze/.venv/bin/python hardware/bitcraze/probe_multiranger.py --require-deck multiranger --require-deck flow2

Outputs
-------
- Human-readable status lines by default
- JSON report with ``--json``
- Exit code 0 on success and 1 when required decks/ranges are missing
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Iterable


DEFAULT_URI = "radio://0/80/2M"
DECK_PARAM_NAMES = ("bcMultiranger", "bcFlow2", "bcZRanger2", "bcAI")
DECK_NAME_ALIASES = {
    "aideck": "bcAI",
    "ai": "bcAI",
    "bcai": "bcAI",
    "flow2": "bcFlow2",
    "flow": "bcFlow2",
    "bcflow2": "bcFlow2",
    "multiranger": "bcMultiranger",
    "multi-ranger": "bcMultiranger",
    "bcmultiranger": "bcMultiranger",
    "zranger2": "bcZRanger2",
    "zranger": "bcZRanger2",
    "bczranger2": "bcZRanger2",
}
RANGE_DIRECTIONS = ("front", "back", "left", "right", "up", "down")


@dataclass(frozen=True)
class RangeSummary:
    """Summarize one directional distance stream in meters."""

    latest_m: float | None
    minimum_m: float | None
    maximum_m: float | None
    valid_samples: int
    missing_samples: int


@dataclass(frozen=True)
class MultirangerProbeReport:
    """Represent one bounded Multi-ranger probe run."""

    uri: str
    workspace: str
    duration_s: float
    sample_period_s: float
    sample_count: int
    deck_flags: dict[str, int | None]
    ranges_m: dict[str, RangeSummary]
    recommendations: tuple[str, ...]


def normalize_required_deck_name(value: str) -> str:
    """Normalize a user-facing deck selector into the firmware deck flag name."""

    normalized = str(value or "").strip().lower()
    if not normalized:
        raise ValueError("required deck names must not be empty")
    try:
        return DECK_NAME_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(dict.fromkeys(DECK_NAME_ALIASES)))
        raise ValueError(f"unsupported deck name `{value}`; choose one of: {allowed}") from exc


def summarize_samples(samples: Iterable[float | None]) -> RangeSummary:
    """Reduce one sequence of range readings into min/max/latest statistics."""

    observed = list(samples)
    latest = observed[-1] if observed else None
    valid = [value for value in observed if value is not None]
    return RangeSummary(
        latest_m=latest,
        minimum_m=min(valid) if valid else None,
        maximum_m=max(valid) if valid else None,
        valid_samples=len(valid),
        missing_samples=len(observed) - len(valid),
    )


def recommendations_for_report(report: MultirangerProbeReport) -> tuple[str, ...]:
    """Return the next concrete operator actions for the sampled state."""

    recommendations: list[str] = []
    if report.deck_flags.get("bcMultiranger") != 1:
        recommendations.append("Multi-ranger deck not detected; reseat the deck and reboot the Crazyflie.")
    else:
        readable_directions = [
            direction
            for direction, summary in report.ranges_m.items()
            if summary.valid_samples > 0 and direction != "down"
        ]
        if not readable_directions:
            recommendations.append(
                "Multi-ranger deck is present but no directional ranges were readable; power-cycle the drone and recheck."
            )
    if report.deck_flags.get("bcFlow2") != 1:
        recommendations.append("Pair the Multi-ranger deck with the Flow deck for stable ground/down sensing.")
    if report.deck_flags.get("bcZRanger2") != 1:
        recommendations.append("Downward z-range is unavailable; expect `down` to stay empty without a Z-ranger/Flow deck.")
    if not recommendations:
        recommendations.append("Multi-ranger and supporting decks look ready for bounded obstacle-awareness experiments.")
    return tuple(recommendations)


def validate_report(
    report: MultirangerProbeReport,
    *,
    required_decks: tuple[str, ...],
    require_readable_ranges: bool,
) -> list[str]:
    """Return validation failures for the requested deck/range expectations."""

    failures: list[str] = []
    for deck_name in required_decks:
        if report.deck_flags.get(deck_name) != 1:
            failures.append(f"required deck {deck_name} is not detected")
    if require_readable_ranges:
        readable = any(summary.valid_samples > 0 for direction, summary in report.ranges_m.items() if direction != "down")
        if not readable:
            failures.append("no readable Multi-ranger directions were observed")
    return failures


def _import_cflib():
    """Import cflib lazily so unit tests can load this script without the workspace venv.

    Twinr's repo-local ``.venv`` stays focused on Twinr itself. The Bitcraze
    runtime dependencies live in ``/twinr/bitcraze/.venv`` because current
    ``cflib`` releases want a different ``numpy`` line than Twinr.
    """

    import cflib.crtp  # type: ignore[import-not-found]  # pylint: disable=import-error
    from cflib.crazyflie import Crazyflie  # type: ignore[import-not-found]  # pylint: disable=import-error
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie  # type: ignore[import-not-found]  # pylint: disable=import-error
    from cflib.utils.multiranger import Multiranger  # type: ignore[import-not-found]  # pylint: disable=import-error

    return cflib.crtp, Crazyflie, SyncCrazyflie, Multiranger


def _read_deck_flags(sync_cf, deck_names: Iterable[str]) -> dict[str, int | None]:
    """Read the firmware deck presence flags and normalize them to integers."""

    flags: dict[str, int | None] = {}
    for deck_name in deck_names:
        value: int | None
        try:
            raw = sync_cf.cf.param.get_value(f"deck.{deck_name}")
        except Exception:
            value = None
        else:
            try:
                value = int(str(raw).strip())
            except (TypeError, ValueError):
                value = None
        flags[deck_name] = value
    return flags


def probe_multiranger(
    *,
    uri: str,
    workspace: Path,
    duration_s: float,
    sample_period_s: float,
    connect_settle_s: float,
) -> MultirangerProbeReport:
    """Connect to one Crazyflie and sample Multi-ranger readings."""

    crtp, Crazyflie, SyncCrazyflie, Multiranger = _import_cflib()
    workspace = workspace.expanduser().resolve()
    cache_dir = workspace / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    crtp.init_drivers()
    crazyflie = Crazyflie(rw_cache=str(cache_dir))
    readings: dict[str, list[float | None]] = {direction: [] for direction in RANGE_DIRECTIONS}

    with SyncCrazyflie(uri, cf=crazyflie) as sync_cf:
        if connect_settle_s > 0:
            time.sleep(connect_settle_s)
        deck_flags = _read_deck_flags(sync_cf, DECK_PARAM_NAMES)

        start = time.monotonic()
        with Multiranger(sync_cf) as multiranger:
            while time.monotonic() - start < duration_s:
                readings["front"].append(multiranger.front)
                readings["back"].append(multiranger.back)
                readings["left"].append(multiranger.left)
                readings["right"].append(multiranger.right)
                readings["up"].append(multiranger.up)
                readings["down"].append(multiranger.down)
                time.sleep(sample_period_s)

    report = MultirangerProbeReport(
        uri=uri,
        workspace=str(workspace),
        duration_s=duration_s,
        sample_period_s=sample_period_s,
        sample_count=len(readings["front"]),
        deck_flags=deck_flags,
        ranges_m={direction: summarize_samples(samples) for direction, samples in readings.items()},
        recommendations=(),
    )
    return MultirangerProbeReport(
        uri=report.uri,
        workspace=report.workspace,
        duration_s=report.duration_s,
        sample_period_s=report.sample_period_s,
        sample_count=report.sample_count,
        deck_flags=report.deck_flags,
        ranges_m=report.ranges_m,
        recommendations=recommendations_for_report(report),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe Crazyflie deck flags and Multi-ranger data.")
    parser.add_argument("--uri", default=DEFAULT_URI, help="Crazyflie radio URI (default: radio://0/80/2M)")
    parser.add_argument(
        "--workspace",
        default="/twinr/bitcraze",
        help="Bitcraze workspace root used for cflib cache files",
    )
    parser.add_argument("--duration-s", type=float, default=2.0, help="How long to sample ranges (default: 2.0)")
    parser.add_argument(
        "--sample-period-s",
        type=float,
        default=0.1,
        help="Delay between range samples in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--connect-settle-s",
        type=float,
        default=1.0,
        help="Initial wait after connect before reading params/ranges (default: 1.0)",
    )
    parser.add_argument(
        "--require-deck",
        action="append",
        default=[],
        help="Require one deck flag (examples: multiranger, flow2, zranger2, aideck); repeat as needed",
    )
    parser.add_argument(
        "--require-readable-ranges",
        action="store_true",
        help="Fail if no readable directional Multi-ranger samples were observed",
    )
    parser.add_argument("--json", action="store_true", help="Emit the full report as JSON")
    return parser


def _print_human_report(report: MultirangerProbeReport, failures: Iterable[str]) -> None:
    print(f"uri={report.uri}")
    print(f"workspace={report.workspace}")
    print(f"sample_count={report.sample_count}")
    for deck_name, flag in sorted(report.deck_flags.items()):
        print(f"deck.{deck_name}={flag if flag is not None else 'unknown'}")
    for direction in RANGE_DIRECTIONS:
        summary = report.ranges_m[direction]
        print(
            f"range.{direction}.latest_m={summary.latest_m} "
            f"min_m={summary.minimum_m} max_m={summary.maximum_m} "
            f"valid={summary.valid_samples} missing={summary.missing_samples}"
        )
    for recommendation in report.recommendations:
        print(f"recommendation={recommendation}")
    for failure in failures:
        print(f"failure={failure}")


def main() -> int:
    args = _build_parser().parse_args()
    required_decks = tuple(normalize_required_deck_name(name) for name in args.require_deck)
    report = probe_multiranger(
        uri=str(args.uri).strip() or DEFAULT_URI,
        workspace=Path(args.workspace),
        duration_s=max(0.1, float(args.duration_s)),
        sample_period_s=max(0.05, float(args.sample_period_s)),
        connect_settle_s=max(0.0, float(args.connect_settle_s)),
    )
    failures = validate_report(
        report,
        required_decks=required_decks,
        require_readable_ranges=bool(args.require_readable_ranges),
    )
    if args.json:
        print(json.dumps({"report": asdict(report), "failures": failures}, indent=2, sort_keys=True))
    else:
        _print_human_report(report, failures)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
