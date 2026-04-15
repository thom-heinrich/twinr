#!/usr/bin/env python3
# CHANGELOG: 2026-04-15
# BUG-1: Add one bounded non-rotor probe for the proven STM32/I2C1 deck-bus failure mode so we can separate "platform handshake fixed" from "deck stack still broken".
# BUG-2: Power-cycle the STM32/deck rail before probing by default on radio URIs so startup-only deck discovery does not misreport hot-reconfigured deck stacks as missing.
# IMP-1: Reuse the existing Bitcraze radio connect/deck-flag helpers from probe_multiranger.py to keep one probe lane for Crazyflie session setup.
# IMP-2: Capture firmware console output, explicit memory TOC state, and deck-memory enumeration in one report so deck-bus regressions stop hiding behind a single missing deck flag.
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cflib>=0.1.31",
# ]
# ///
"""Probe Crazyflie deck-bus health without spinning motors.

Purpose
-------
Connect to one Crazyflie over the normal radio URI, capture bounded startup
console output, read the firmware ``deck.*`` flags, refresh the memory TOC, and
query any deck-memory managers that survived boot. This is the non-rotor probe
for the proven ``I2C1`` deck-bus failure mode where the STM32 firmware reaches
``commInit()`` again but the deck stack itself is still electrically broken.

Usage
-----
Command-line examples::

    /twinr/bitcraze/.venv/bin/python hardware/bitcraze/probe_deck_bus.py
    /twinr/bitcraze/.venv/bin/python hardware/bitcraze/probe_deck_bus.py --json
    /twinr/bitcraze/.venv/bin/python hardware/bitcraze/probe_deck_bus.py --console-capture-s 2.5

Outputs
-------
- Human-readable probe evidence by default
- JSON report with ``--json``
- Exit code 0 on success and 1 when the bounded probe itself fails

Notes
-----
Crazyflie deck discovery happens at startup. If the physical deck stack changed
after boot, reading ``deck.*`` and the deck memory TOC without a reboot only
returns stale topology. This probe therefore power-cycles the STM32 and deck
rail by default on ``radio://`` URIs before it reconnects and reads the
firmware truth surface.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from threading import Event, Lock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from probe_multiranger import (  # noqa: E402
    DECK_PARAM_NAMES,
    ProbeError,
    _attach_connection_watchers,
    _ensure_secure_cache_dir,
    _import_cflib,
    _read_deck_flags,
    _validate_uri,
    _wait_for_fully_connected,
    default_uri_from_env,
)


DEFAULT_URI_FALLBACK = default_uri_from_env()
CONSOLE_EVIDENCE_MARKERS = (
    "i2c1 [FAIL]",
    "EEPROM: I2C connection [FAIL]",
    "SYS: configblock [FAIL]",
    "STORAGE:",
    "DECK_CORE:",
)


@dataclass(frozen=True)
class ConsoleLine:
    """Represent one bounded firmware console line."""

    timestamp_s: float
    text: str


@dataclass(frozen=True)
class MemorySummary:
    """Summarize one Crazyflie memory element after refresh."""

    id: int
    type_code: int
    type_name: str
    size: int
    valid: bool | None
    vid: int | None
    pid: int | None
    name: str | None
    revision: str | None
    address: int | str | None
    elements: dict[str, str]


@dataclass(frozen=True)
class DeckBusProbeReport:
    """Represent one bounded Crazyflie deck-bus probe run."""

    uri: str
    workspace: str
    stm_power_cycle_before_probe: bool
    post_power_cycle_settle_s: float
    connect_settle_s: float
    console_capture_s: float
    deck_flags: dict[str, int | None]
    console_lines: tuple[ConsoleLine, ...]
    console_markers: tuple[str, ...]
    memories: tuple[MemorySummary, ...]


class _ConsoleCollector:
    """Collect firmware console lines with relative timestamps."""

    def __init__(self) -> None:
        self._start = time.monotonic()
        self._buffer = ""
        self._lines: list[ConsoleLine] = []
        self._lock = Lock()

    def on_text(self, text: str) -> None:
        """Accumulate incoming console chunks into newline-terminated lines."""

        timestamp_s = time.monotonic() - self._start
        with self._lock:
            self._buffer += str(text)
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                stripped = line.strip()
                if stripped:
                    self._lines.append(ConsoleLine(timestamp_s=timestamp_s, text=stripped))

    def finalize(self) -> tuple[ConsoleLine, ...]:
        """Return all completed lines plus one bounded trailing fragment."""

        with self._lock:
            lines = list(self._lines)
            trailing = self._buffer.strip()
            if trailing:
                lines.append(ConsoleLine(timestamp_s=time.monotonic() - self._start, text=trailing))
            return tuple(lines)


def _import_memory_types():
    """Import the cflib memory type definitions lazily."""

    from cflib.crazyflie.mem import MemoryElement  # type: ignore[import-not-found]  # pylint: disable=import-error

    return MemoryElement


def _wait_for_event(event: Event, timeout_s: float, *, failure_message: str) -> None:
    """Fail closed if one asynchronous cflib operation does not finish in time."""

    if not event.wait(timeout_s):
        raise ProbeError(failure_message)


def _refresh_memory_catalog(cf, *, timeout_s: float) -> None:
    """Refresh the firmware memory catalog and fail closed on timeout/failure."""

    finished = Event()
    failure: list[str] = []

    def on_done() -> None:
        finished.set()

    def on_failed() -> None:
        failure.append("memory refresh failed")
        finished.set()

    cf.mem.refresh(on_done, on_failed)
    _wait_for_event(finished, timeout_s, failure_message=f"timed out after {timeout_s:.1f}s waiting for memory refresh")
    if failure:
        raise ProbeError(failure[0])


def _should_stm_power_cycle(uri: str, *, skip_stm_power_cycle: bool) -> bool:
    """Return whether the bounded probe should restart the STM32/deck rail."""

    return uri.startswith("radio://") and not skip_stm_power_cycle


def _stm_power_cycle(uri: str, *, settle_s: float) -> None:
    """Restart the STM32 and all expansion decks over Crazyradio."""

    from cflib.utils.power_switch import PowerSwitch  # type: ignore[import-not-found]  # pylint: disable=import-error

    power_switch = PowerSwitch(uri)
    try:
        power_switch.stm_power_cycle()
    except Exception as exc:  # pragma: no cover - thin wrapper over cflib
        raise ProbeError(f"failed to power-cycle Crazyflie STM32/decks before deck probe: {exc}") from exc
    finally:
        power_switch.close()

    if settle_s > 0:
        time.sleep(settle_s)


def _update_memory(mem, *, timeout_s: float, description: str) -> None:
    """Request one additional memory-element update and fail closed on timeout."""

    finished = Event()

    def on_done(_mem) -> None:
        finished.set()

    mem.update(on_done)
    _wait_for_event(finished, timeout_s, failure_message=f"timed out after {timeout_s:.1f}s waiting for {description}")


def _summarize_memory(mem, MemoryElement) -> MemorySummary:
    """Convert one cflib memory object into JSON-safe evidence."""

    elements = getattr(mem, "elements", {})
    summarized_elements = {str(key): str(value) for key, value in dict(elements).items()}
    revision = getattr(mem, "revision", None)
    raw_address = getattr(mem, "addr", None)
    if raw_address is None:
        address: int | str | None = None
    elif isinstance(raw_address, int):
        address = raw_address
    else:
        address = str(raw_address)
    return MemorySummary(
        id=int(mem.id),
        type_code=int(mem.type),
        type_name=str(MemoryElement.type_to_string(mem.type)),
        size=int(mem.size),
        valid=bool(getattr(mem, "valid")) if hasattr(mem, "valid") else None,
        vid=int(getattr(mem, "vid")) if getattr(mem, "vid", None) is not None else None,
        pid=int(getattr(mem, "pid")) if getattr(mem, "pid", None) is not None else None,
        name=str(getattr(mem, "name")) if getattr(mem, "name", None) is not None else None,
        revision=str(revision) if revision not in (None, "") else None,
        address=address,
        elements=summarized_elements,
    )
def _capture_console_markers(console_lines: tuple[ConsoleLine, ...]) -> tuple[str, ...]:
    """Extract exact known firmware evidence markers from the captured console."""

    observed: list[str] = []
    for line in console_lines:
        for marker in CONSOLE_EVIDENCE_MARKERS:
            if marker in line.text and marker not in observed:
                observed.append(marker)
    return tuple(observed)


def probe_deck_bus(
    *,
    uri: str,
    workspace: Path,
    skip_stm_power_cycle: bool,
    post_power_cycle_settle_s: float,
    connect_settle_s: float,
    connect_timeout_s: float,
    mem_timeout_s: float,
    console_capture_s: float,
    allow_non_radio_uri: bool,
) -> DeckBusProbeReport:
    """Connect to one Crazyflie and collect bounded deck-bus evidence."""

    crtp, Crazyflie, _LogConfig = _import_cflib()
    MemoryElement = _import_memory_types()
    uri = _validate_uri(uri, allow_non_radio_uri=allow_non_radio_uri)
    workspace = workspace.expanduser().resolve()
    cache_dir = workspace / "cache"
    _ensure_secure_cache_dir(cache_dir)

    crtp.init_drivers()
    stm_power_cycle_before_probe = _should_stm_power_cycle(
        uri,
        skip_stm_power_cycle=skip_stm_power_cycle,
    )
    if stm_power_cycle_before_probe:
        _stm_power_cycle(uri, settle_s=post_power_cycle_settle_s)
    crazyflie = Crazyflie(rw_cache=str(cache_dir))
    connection_state = _attach_connection_watchers(crazyflie)
    console_collector = _ConsoleCollector()
    crazyflie.console.receivedChar.add_callback(console_collector.on_text)

    try:
        crazyflie.open_link(uri)
        _wait_for_fully_connected(connection_state, connect_timeout_s)
        if connect_settle_s > 0:
            time.sleep(connect_settle_s)

        deck_flags = _read_deck_flags(crazyflie, DECK_PARAM_NAMES)
        _refresh_memory_catalog(crazyflie, timeout_s=mem_timeout_s)

        for mem in crazyflie.mem.get_mems(MemoryElement.TYPE_DECKCTRL):
            _update_memory(mem, timeout_s=mem_timeout_s, description=f"deckctrl memory {mem.id}")

        if console_capture_s > 0:
            time.sleep(console_capture_s)

        memory_summaries = tuple(
            _summarize_memory(mem, MemoryElement)
            for mem in sorted(crazyflie.mem.mems, key=lambda current: (int(current.type), int(current.id)))
        )
        console_lines = console_collector.finalize()
        return DeckBusProbeReport(
            uri=uri,
            workspace=str(workspace),
            stm_power_cycle_before_probe=stm_power_cycle_before_probe,
            post_power_cycle_settle_s=post_power_cycle_settle_s if stm_power_cycle_before_probe else 0.0,
            connect_settle_s=connect_settle_s,
            console_capture_s=console_capture_s,
            deck_flags=deck_flags,
            console_lines=console_lines,
            console_markers=_capture_console_markers(console_lines),
            memories=memory_summaries,
        )
    finally:
        try:
            crazyflie.close_link()
        except Exception:
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe Crazyflie deck-bus health without flying.")
    parser.add_argument(
        "--uri",
        default=DEFAULT_URI_FALLBACK,
        help="Crazyflie URI (default: CFLIB_URI or radio://0/80/2M/E7E7E7E7E7).",
    )
    parser.add_argument(
        "--allow-non-radio-uri",
        action="store_true",
        help="Allow trusted non-radio URIs such as tcp:// or serial:// for this probe.",
    )
    parser.add_argument(
        "--workspace",
        default="/twinr/bitcraze",
        help="Bitcraze workspace root used for cflib cache files.",
    )
    parser.add_argument(
        "--skip-stm-power-cycle",
        action="store_true",
        help=(
            "Do not restart the STM32/deck rail before probing. By default radio URIs are "
            "power-cycled because Crazyflie deck discovery is startup-only."
        ),
    )
    parser.add_argument(
        "--post-power-cycle-settle-s",
        type=float,
        default=3.0,
        help="Extra wait after the bounded STM/deck power-cycle before reconnecting (default: 3.0).",
    )
    parser.add_argument(
        "--connect-settle-s",
        type=float,
        default=1.0,
        help="Optional extra wait after fully_connected before probing (default: 1.0).",
    )
    parser.add_argument(
        "--connect-timeout-s",
        type=float,
        default=10.0,
        help="Fail if the Crazyflie does not reach fully_connected in time (default: 10.0).",
    )
    parser.add_argument(
        "--mem-timeout-s",
        type=float,
        default=3.0,
        help="Fail if bounded memory refresh/update/query steps do not finish in time (default: 3.0).",
    )
    parser.add_argument(
        "--console-capture-s",
        type=float,
        default=1.0,
        help="How long to keep collecting post-connect console output before closing the link (default: 1.0).",
    )
    parser.add_argument("--json", action="store_true", help="Emit the full report as JSON.")
    return parser


def _print_human_report(report: DeckBusProbeReport) -> None:
    print(f"uri={report.uri}")
    print(f"workspace={report.workspace}")
    print(f"stm_power_cycle_before_probe={report.stm_power_cycle_before_probe}")
    print(f"post_power_cycle_settle_s={report.post_power_cycle_settle_s}")
    print(f"connect_settle_s={report.connect_settle_s}")
    print(f"console_capture_s={report.console_capture_s}")
    for deck_name, flag in sorted(report.deck_flags.items()):
        print(f"deck.{deck_name}={flag if flag is not None else 'unknown'}")
    print(f"console_marker_count={len(report.console_markers)}")
    for marker in report.console_markers:
        print(f"console.marker={marker}")
    print(f"memory_count={len(report.memories)}")
    for memory in report.memories:
        print(
            "memory="
            f"id:{memory.id} "
            f"type:{memory.type_name}({memory.type_code}) "
            f"size:{memory.size} "
            f"valid:{memory.valid} "
            f"name:{memory.name} "
            f"revision:{memory.revision} "
            f"vid:{memory.vid} pid:{memory.pid} "
            f"address:{memory.address} "
            f"elements:{memory.elements}"
        )
    print(f"console_line_count={len(report.console_lines)}")
    for line in report.console_lines:
        print(f"console[{line.timestamp_s:.3f}]={line.text}")


def main() -> int:
    """Run the CLI and return a process exit code."""

    parser = _build_parser()
    args = parser.parse_args()
    try:
        report = probe_deck_bus(
            uri=str(args.uri),
            workspace=Path(args.workspace),
            skip_stm_power_cycle=bool(args.skip_stm_power_cycle),
            post_power_cycle_settle_s=float(args.post_power_cycle_settle_s),
            connect_settle_s=float(args.connect_settle_s),
            connect_timeout_s=float(args.connect_timeout_s),
            mem_timeout_s=float(args.mem_timeout_s),
            console_capture_s=float(args.console_capture_s),
            allow_non_radio_uri=bool(args.allow_non_radio_uri),
        )
    except (ProbeError, ValueError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"error={exc}")
        return 1

    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        _print_human_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
