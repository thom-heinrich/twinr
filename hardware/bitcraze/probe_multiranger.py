#!/usr/bin/env python3
# CHANGELOG: 2026-03-27
# BUG-1: Wait for the Crazyflie `fully_connected` state before reading deck flags; removed the heuristic post-connect race.
# BUG-2: `--sample-period-s` now configures the Crazyflie log period itself; `sample_count` is actual log packets, not host-side polls.
# BUG-3: Fixed downward-range deck detection and recommendations for Flow/Z-ranger v1/v2 stacks.
# SEC-1: Non-radio URIs are rejected by default to prevent unintended TCP/serial link access from this radio-only probe.
# IMP-1: Switched to packet-driven `LogConfig` acquisition with bounded startup/runtime checks and true 10 ms cadence quantization.
# IMP-2: Added cflib link diagnostics (latency, link quality, RSSI, packet rate, congestion) to separate sensor faults from transport faults.
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
import os
from pathlib import Path
import time
from threading import Event, Lock
from typing import Any, Iterable


DEFAULT_URI_ENV = "CFLIB_URI"
DEFAULT_URI_FALLBACK = "radio://0/80/2M/E7E7E7E7E7"
DECK_PARAM_NAMES = (
    "bcMultiranger",
    "bcFlow",
    "bcFlow2",
    "bcZRanger",
    "bcZRanger2",
    "bcAI",
)
DECK_NAME_ALIASES = {
    "aideck": "bcAI",
    "ai": "bcAI",
    "bcai": "bcAI",
    "flow": "bcFlow2",
    "flow1": "bcFlow",
    "flow2": "bcFlow2",
    "bcflow": "bcFlow",
    "bcflow2": "bcFlow2",
    "multiranger": "bcMultiranger",
    "multi-ranger": "bcMultiranger",
    "bcmultiranger": "bcMultiranger",
    "zranger": "bcZRanger2",
    "zranger1": "bcZRanger",
    "zranger2": "bcZRanger2",
    "bczranger": "bcZRanger",
    "bczranger2": "bcZRanger2",
}
RANGE_DIRECTIONS = ("front", "back", "left", "right", "up", "down")
RANGE_LOG_VARIABLES = {
    "front": "range.front",
    "back": "range.back",
    "left": "range.left",
    "right": "range.right",
    "up": "range.up",
    "down": "range.zrange",
}
DOWNWARD_RANGE_DECKS = ("bcFlow", "bcFlow2", "bcZRanger", "bcZRanger2")


class ProbeError(RuntimeError):
    """Represent a bounded probe failure with a human-meaningful message."""


@dataclass(frozen=True)
class RangeSummary:
    """Summarize one directional distance stream in meters."""

    latest_m: float | None
    minimum_m: float | None
    maximum_m: float | None
    valid_samples: int
    missing_samples: int


@dataclass(frozen=True)
class LinkStatisticsSnapshot:
    """Capture the latest cflib link-health metrics observed during the probe."""

    latency_p95_ms: float | None
    link_quality: float | None
    uplink_rssi: float | None
    uplink_rate_packets_s: float | None
    downlink_rate_packets_s: float | None
    uplink_congestion: float | None
    downlink_congestion: float | None


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
    link_stats: LinkStatisticsSnapshot
    recommendations: tuple[str, ...]


@dataclass
class _ConnectionState:
    fully_connected: Event
    connection_failed: Event
    disconnected: Event
    failure_message: str | None = None
    disconnect_message: str | None = None


def default_uri_from_env() -> str:
    """Return the URI configured through CFLIB_URI or the documented default."""

    return str(os.getenv(DEFAULT_URI_ENV, DEFAULT_URI_FALLBACK)).strip() or DEFAULT_URI_FALLBACK


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


def normalize_log_period_s(requested_s: float) -> float:
    """Quantize the requested period to the Crazyflie logging granularity."""

    period_ms = max(10, int(round(float(requested_s) * 1000.0 / 10.0)) * 10)
    return period_ms / 1000.0


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


def _has_downward_range_deck(deck_flags: dict[str, int | None]) -> bool:
    return any(deck_flags.get(deck_name) == 1 for deck_name in DOWNWARD_RANGE_DECKS)


def recommendations_for_report(report: MultirangerProbeReport) -> tuple[str, ...]:
    """Return the next concrete operator actions for the sampled state."""

    recommendations: list[str] = []

    if report.sample_count == 0:
        recommendations.append(
            "No log packets were received; verify URI, radio link, and that the Crazyflie finished booting before retrying."
        )

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

    if not any(report.deck_flags.get(deck_name) == 1 for deck_name in ("bcFlow", "bcFlow2")):
        recommendations.append(
            "Flow deck not detected; lateral optical-flow stabilization will be unavailable during close-range experiments."
        )

    if not _has_downward_range_deck(report.deck_flags):
        recommendations.append(
            "Downward z-range is unavailable; expect `down` to stay empty without a Flow or Z-ranger deck."
        )
    elif report.ranges_m["down"].valid_samples == 0:
        recommendations.append(
            "A downward range deck is detected but `down` stayed empty; inspect the bottom-facing sensor and deck stack order."
        )

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
        readable = any(
            summary.valid_samples > 0
            for direction, summary in report.ranges_m.items()
            if direction != "down"
        )
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
    from cflib.crazyflie.log import LogConfig  # type: ignore[import-not-found]  # pylint: disable=import-error

    return cflib.crtp, Crazyflie, LogConfig


def _convert_range_mm_to_m(value: Any) -> float | None:
    """Mirror the official Multiranger helper conversion semantics."""

    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    if numeric >= 8000:
        return None
    return numeric / 1000.0


def _read_deck_flags(cf, deck_names: Iterable[str]) -> dict[str, int | None]:
    """Read the firmware deck presence flags and normalize them to integers."""

    flags: dict[str, int | None] = {}
    for deck_name in deck_names:
        value: int | None
        try:
            raw = cf.param.get_value(f"deck.{deck_name}")
        except Exception:
            value = None
        else:
            try:
                value = int(str(raw).strip())
            except (TypeError, ValueError):
                value = None
        flags[deck_name] = value
    return flags


def _ensure_secure_cache_dir(cache_dir: Path) -> None:
    """Create the cache directory and keep it private to the current account."""

    cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(cache_dir, 0o700)
    except OSError:
        pass


# BREAKING: This probe now rejects non-radio URIs by default. Pass
# --allow-non-radio-uri only if you intentionally want to use a non-radio link.
def _validate_uri(uri: str, *, allow_non_radio_uri: bool) -> str:
    normalized = str(uri or "").strip() or default_uri_from_env()
    if not allow_non_radio_uri and not normalized.startswith("radio://"):
        raise ValueError(
            "non-radio URI blocked by default; this probe is intended for radio:// links. "
            "Pass --allow-non-radio-uri only for an explicitly trusted non-radio transport."
        )
    return normalized


def _attach_connection_watchers(cf) -> _ConnectionState:
    state = _ConnectionState(
        fully_connected=Event(),
        connection_failed=Event(),
        disconnected=Event(),
    )

    def on_fully_connected(_uri: str) -> None:
        state.fully_connected.set()

    def on_connection_failed(_uri: str, message: str) -> None:
        state.failure_message = str(message)
        state.connection_failed.set()

    def on_connection_lost(_uri: str, message: str) -> None:
        state.disconnect_message = str(message)
        state.disconnected.set()

    def on_disconnected(_uri: str) -> None:
        state.disconnected.set()

    cf.fully_connected.add_callback(on_fully_connected)
    cf.connection_failed.add_callback(on_connection_failed)
    cf.connection_lost.add_callback(on_connection_lost)
    cf.disconnected.add_callback(on_disconnected)
    return state


def _wait_for_fully_connected(state: _ConnectionState, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if state.fully_connected.is_set():
            return
        if state.connection_failed.is_set():
            message = state.failure_message or "connection failed"
            raise ProbeError(message)
        if state.disconnected.is_set() and not state.fully_connected.is_set():
            message = state.disconnect_message or "disconnected during connection setup"
            raise ProbeError(message)
        time.sleep(0.01)
    raise ProbeError(f"timed out after {timeout_s:.1f}s waiting for Crazyflie fully_connected")


def probe_multiranger(
    *,
    uri: str,
    workspace: Path,
    duration_s: float,
    sample_period_s: float,
    connect_settle_s: float,
    connect_timeout_s: float,
    first_packet_timeout_s: float,
    allow_non_radio_uri: bool,
) -> MultirangerProbeReport:
    """Connect to one Crazyflie and sample Multi-ranger readings."""

    crtp, Crazyflie, LogConfig = _import_cflib()
    uri = _validate_uri(uri, allow_non_radio_uri=allow_non_radio_uri)
    workspace = workspace.expanduser().resolve()
    cache_dir = workspace / "cache"
    _ensure_secure_cache_dir(cache_dir)

    actual_sample_period_s = normalize_log_period_s(sample_period_s)
    sample_period_ms = int(round(actual_sample_period_s * 1000.0))

    crtp.init_drivers()
    crazyflie = Crazyflie(rw_cache=str(cache_dir))
    connection_state = _attach_connection_watchers(crazyflie)

    readings: dict[str, list[float | None]] = {direction: [] for direction in RANGE_DIRECTIONS}
    packet_timestamps_ms: list[int] = []
    readings_lock = Lock()
    first_packet_event = Event()
    log_error_lock = Lock()
    log_error_message: str | None = None

    link_stats_lock = Lock()
    latest_link_stats: dict[str, float | None] = {
        "latency_p95_ms": None,
        "link_quality": None,
        "uplink_rssi": None,
        "uplink_rate_packets_s": None,
        "downlink_rate_packets_s": None,
        "uplink_congestion": None,
        "downlink_congestion": None,
    }

    def set_link_stat(name: str, value: float) -> None:
        with link_stats_lock:
            latest_link_stats[name] = float(value)

    crazyflie.link_statistics.latency_updated.add_callback(
        lambda value: set_link_stat("latency_p95_ms", value)
    )
    crazyflie.link_statistics.link_quality_updated.add_callback(
        lambda value: set_link_stat("link_quality", value)
    )
    crazyflie.link_statistics.uplink_rssi_updated.add_callback(
        lambda value: set_link_stat("uplink_rssi", value)
    )
    crazyflie.link_statistics.uplink_rate_updated.add_callback(
        lambda value: set_link_stat("uplink_rate_packets_s", value)
    )
    crazyflie.link_statistics.downlink_rate_updated.add_callback(
        lambda value: set_link_stat("downlink_rate_packets_s", value)
    )
    crazyflie.link_statistics.uplink_congestion_updated.add_callback(
        lambda value: set_link_stat("uplink_congestion", value)
    )
    crazyflie.link_statistics.downlink_congestion_updated.add_callback(
        lambda value: set_link_stat("downlink_congestion", value)
    )

    log_config = LogConfig(name="multiranger_probe", period_in_ms=sample_period_ms)
    for variable_name in RANGE_LOG_VARIABLES.values():
        log_config.add_variable(variable_name)

    def on_log_error(_log_conf, message: str) -> None:
        nonlocal log_error_message
        with log_error_lock:
            log_error_message = str(message)

    def on_range_packet(timestamp_ms: int, data: dict[str, Any], _log_conf) -> None:
        converted = {
            direction: _convert_range_mm_to_m(data.get(log_variable))
            for direction, log_variable in RANGE_LOG_VARIABLES.items()
        }
        with readings_lock:
            packet_timestamps_ms.append(int(timestamp_ms))
            for direction, value in converted.items():
                readings[direction].append(value)
        first_packet_event.set()

    log_config.error_cb.add_callback(on_log_error)
    log_config.data_received_cb.add_callback(on_range_packet)

    try:
        crazyflie.open_link(uri)
        _wait_for_fully_connected(connection_state, connect_timeout_s)
        if connect_settle_s > 0:
            time.sleep(connect_settle_s)

        deck_flags = _read_deck_flags(crazyflie, DECK_PARAM_NAMES)

        crazyflie.log.add_config(log_config)
        if not log_config.valid:
            raise ProbeError("multiranger probe log configuration is invalid for this firmware")
        log_config.start()

        if not first_packet_event.wait(first_packet_timeout_s):
            raise ProbeError(
                f"timed out after {first_packet_timeout_s:.1f}s waiting for first range log packet"
            )

        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            if connection_state.connection_failed.is_set():
                raise ProbeError(connection_state.failure_message or "connection failed during probe")
            if connection_state.disconnected.is_set():
                raise ProbeError(connection_state.disconnect_message or "Crazyflie disconnected during probe")
            with log_error_lock:
                if log_error_message is not None:
                    raise ProbeError(f"Crazyflie logging error: {log_error_message}")
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    finally:
        try:
            if log_config.started:
                log_config.stop()
        except Exception:
            pass
        try:
            log_config.delete()
        except Exception:
            pass
        try:
            crazyflie.close_link()
        except Exception:
            pass

    with readings_lock:
        readings_snapshot = {direction: list(samples) for direction, samples in readings.items()}
        packet_count = len(packet_timestamps_ms)
    with link_stats_lock:
        link_stats_snapshot = LinkStatisticsSnapshot(**latest_link_stats)

    report = MultirangerProbeReport(
        uri=uri,
        workspace=str(workspace),
        duration_s=duration_s,
        sample_period_s=actual_sample_period_s,
        sample_count=packet_count,
        deck_flags=deck_flags,
        ranges_m={direction: summarize_samples(samples) for direction, samples in readings_snapshot.items()},
        link_stats=link_stats_snapshot,
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
        link_stats=report.link_stats,
        recommendations=recommendations_for_report(report),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe Crazyflie deck flags and Multi-ranger data.")
    parser.add_argument(
        "--uri",
        default=default_uri_from_env(),
        help=(
            "Crazyflie URI (default: CFLIB_URI or radio://0/80/2M/E7E7E7E7E7). "
            "Non-radio URIs are blocked unless --allow-non-radio-uri is passed."
        ),
    )
    parser.add_argument(
        "--allow-non-radio-uri",
        action="store_true",
        help="Allow trusted non-radio URIs such as tcp:// or serial:// for this probe",
    )
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
        help="Target Crazyflie log period in seconds; rounded to 10 ms steps (default: 0.1)",
    )
    parser.add_argument(
        "--connect-settle-s",
        type=float,
        default=1.0,
        help="Optional extra wait after fully_connected before probing (default: 1.0)",
    )
    parser.add_argument(
        "--connect-timeout-s",
        type=float,
        default=10.0,
        help="Fail if the Crazyflie does not reach fully_connected in time (default: 10.0)",
    )
    parser.add_argument(
        "--first-packet-timeout-s",
        type=float,
        default=1.0,
        help="Fail if no range log packet arrives after logging starts (default: 1.0)",
    )
    parser.add_argument(
        "--require-deck",
        action="append",
        default=[],
        help=(
            "Require one deck flag (examples: multiranger, flow1, flow2, zranger1, zranger2, aideck); "
            "repeat as needed"
        ),
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
    print(f"duration_s={report.duration_s}")
    print(f"sample_period_s={report.sample_period_s}")
    print(f"sample_count={report.sample_count}")
    for deck_name, flag in sorted(report.deck_flags.items()):
        print(f"deck.{deck_name}={flag if flag is not None else 'unknown'}")
    print(f"link.latency_p95_ms={report.link_stats.latency_p95_ms}")
    print(f"link.link_quality={report.link_stats.link_quality}")
    print(f"link.uplink_rssi={report.link_stats.uplink_rssi}")
    print(f"link.uplink_rate_packets_s={report.link_stats.uplink_rate_packets_s}")
    print(f"link.downlink_rate_packets_s={report.link_stats.downlink_rate_packets_s}")
    print(f"link.uplink_congestion={report.link_stats.uplink_congestion}")
    print(f"link.downlink_congestion={report.link_stats.downlink_congestion}")
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
    try:
        required_decks = tuple(normalize_required_deck_name(name) for name in args.require_deck)
        report = probe_multiranger(
            uri=str(args.uri).strip() or default_uri_from_env(),
            workspace=Path(args.workspace),
            duration_s=max(0.1, float(args.duration_s)),
            sample_period_s=max(0.01, float(args.sample_period_s)),
            connect_settle_s=max(0.0, float(args.connect_settle_s)),
            connect_timeout_s=max(0.5, float(args.connect_timeout_s)),
            first_packet_timeout_s=max(0.1, float(args.first_packet_timeout_s)),
            allow_non_radio_uri=bool(args.allow_non_radio_uri),
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
    except Exception as exc:  # pylint: disable=broad-except
        message = str(exc) or exc.__class__.__name__
        if args.json:
            print(json.dumps({"error": message, "failures": [message]}, indent=2, sort_keys=True))
        else:
            print(f"failure={message}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())