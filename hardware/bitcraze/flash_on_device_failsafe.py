#!/usr/bin/env python3
"""Flash and verify the Twinr on-device Crazyflie failsafe app.

Purpose
-------
Keep the firmware rollout for the on-device STM32 failsafe deterministic and
operator-friendly. This script flashes the already-built OOT firmware image
through ``cfloader`` and then proves that the new ``twinrFs`` app-layer param
surface is visible over the normal radio link.

Usage
-----
Command-line examples::

    /twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py
    /twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py --skip-probe
    /twinr/bitcraze/.venv/bin/python hardware/bitcraze/flash_on_device_failsafe.py --boot-mode cold --json

Inputs
------
- A reachable Crazyflie on ``radio://0/80/2M`` by default
- One built STM32 image at
  ``hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.bin``
- A working Bitcraze workspace under ``/twinr/bitcraze``

Outputs
-------
- A bounded flash/probe report on stdout
- Exit code ``0`` when flashing succeeded and the ``twinrFs`` app is visible
- Exit code ``1`` when flashing or probing failed
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from on_device_failsafe import OnDeviceFailsafeAvailability, probe_on_device_failsafe  # noqa: E402


DEFAULT_URI = "radio://0/80/2M"
DEFAULT_WORKSPACE = Path("/twinr/bitcraze")
DEFAULT_BINARY = SCRIPT_DIR / "twinr_on_device_failsafe" / "build" / "twinr_on_device_failsafe.bin"
DEFAULT_CFLOADER = DEFAULT_WORKSPACE / ".venv" / "bin" / "cfloader"
DEFAULT_FLASH_TIMEOUT_S = 120.0
DEFAULT_PROBE_SETTLE_S = 3.0


@dataclass(frozen=True)
class OnDeviceFailsafeFlashReport:
    """Persist the bounded outcome of one firmware flash/probe run."""

    uri: str
    workspace: str
    binary_path: str
    cfloader_executable: str
    boot_mode: str
    flash_command: tuple[str, ...]
    flashed: bool
    probe_attempted: bool
    availability: OnDeviceFailsafeAvailability | None
    flash_stdout_tail: tuple[str, ...]
    flash_stderr_tail: tuple[str, ...]
    failures: tuple[str, ...]


def _tail_lines(text: str, *, limit: int = 8) -> tuple[str, ...]:
    """Return only the last few non-empty output lines for durable reports."""

    lines = [line.rstrip() for line in str(text).splitlines() if line.strip()]
    if len(lines) <= limit:
        return tuple(lines)
    return tuple(lines[-limit:])


def build_cfloader_command(
    *,
    cfloader_executable: Path,
    binary_path: Path,
    uri: str,
    boot_mode: str,
) -> tuple[str, ...]:
    """Build the exact ``cfloader`` command for one firmware flash attempt."""

    command: list[str] = [str(cfloader_executable)]
    normalized_boot_mode = str(boot_mode).strip().lower()
    if normalized_boot_mode == "warm":
        command.extend(["-w", str(uri).strip() or DEFAULT_URI])
    elif normalized_boot_mode != "cold":
        raise ValueError(f"unsupported boot mode: {boot_mode}")
    command.extend(["flash", str(binary_path), "stm32-fw"])
    return tuple(command)


def _import_cflib() -> tuple[Any, Any, Any]:
    """Import the small cflib slice needed for post-flash probing."""

    import cflib.crtp as crtp  # pylint: disable=import-error
    from cflib.crazyflie import Crazyflie  # pylint: disable=import-error
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie  # pylint: disable=import-error

    return crtp, Crazyflie, SyncCrazyflie


def _run_cfloader(
    command: Sequence[str],
    *,
    timeout_s: float,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    """Execute one bounded ``cfloader`` process."""

    return runner(
        list(command),
        capture_output=True,
        text=True,
        timeout=max(1.0, float(timeout_s)),
        check=False,
    )


def _normalize_process_output(text: bytes | str | None) -> str:
    """Normalize subprocess output into text for durable tail capture."""

    if text is None:
        return ""
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    return text


def _probe_loaded_failsafe(
    *,
    uri: str,
    workspace: Path,
    settle_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> OnDeviceFailsafeAvailability:
    """Reconnect over the normal radio link and prove that ``twinrFs`` is live."""

    if settle_s > 0.0:
        sleep(float(settle_s))
    crtp, crazyflie_cls, sync_crazyflie_cls = _import_cflib()
    crtp.init_drivers()
    cache_dir = workspace / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sync_context = sync_crazyflie_cls(str(uri).strip() or DEFAULT_URI, cf=crazyflie_cls(rw_cache=str(cache_dir)))
    with sync_context as sync_cf:
        return probe_on_device_failsafe(sync_cf)


def flash_on_device_failsafe(
    *,
    uri: str = DEFAULT_URI,
    workspace: Path = DEFAULT_WORKSPACE,
    binary_path: Path = DEFAULT_BINARY,
    cfloader_executable: Path = DEFAULT_CFLOADER,
    boot_mode: str = "warm",
    flash_timeout_s: float = DEFAULT_FLASH_TIMEOUT_S,
    probe_settle_s: float = DEFAULT_PROBE_SETTLE_S,
    skip_probe: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
) -> OnDeviceFailsafeFlashReport:
    """Flash the firmware app and optionally verify the ``twinrFs`` param surface."""

    normalized_workspace = Path(workspace)
    normalized_binary = Path(binary_path)
    normalized_cfloader = Path(cfloader_executable)
    failures: list[str] = []
    stdout_tail: tuple[str, ...] = ()
    stderr_tail: tuple[str, ...] = ()
    availability: OnDeviceFailsafeAvailability | None = None
    flashed = False
    probe_attempted = False

    if not normalized_binary.is_file():
        failures.append(f"binary not found: {normalized_binary}")
    if not normalized_cfloader.is_file():
        failures.append(f"cfloader not found: {normalized_cfloader}")
    try:
        command = build_cfloader_command(
            cfloader_executable=normalized_cfloader,
            binary_path=normalized_binary,
            uri=uri,
            boot_mode=boot_mode,
        )
    except ValueError as exc:
        command = ()
        failures.append(str(exc))
    if failures:
        return OnDeviceFailsafeFlashReport(
            uri=str(uri),
            workspace=str(normalized_workspace),
            binary_path=str(normalized_binary),
            cfloader_executable=str(normalized_cfloader),
            boot_mode=str(boot_mode),
            flash_command=tuple(command),
            flashed=False,
            probe_attempted=False,
            availability=None,
            flash_stdout_tail=stdout_tail,
            flash_stderr_tail=stderr_tail,
            failures=tuple(failures),
        )

    try:
        completed = _run_cfloader(command, timeout_s=flash_timeout_s, runner=runner)
    except subprocess.TimeoutExpired as exc:
        failures.append(f"cfloader timeout after {float(exc.timeout):.1f}s")
        stdout_tail = _tail_lines(_normalize_process_output(exc.stdout))
        stderr_tail = _tail_lines(_normalize_process_output(exc.stderr))
        return OnDeviceFailsafeFlashReport(
            uri=str(uri),
            workspace=str(normalized_workspace),
            binary_path=str(normalized_binary),
            cfloader_executable=str(normalized_cfloader),
            boot_mode=str(boot_mode),
            flash_command=tuple(command),
            flashed=False,
            probe_attempted=False,
            availability=None,
            flash_stdout_tail=stdout_tail,
            flash_stderr_tail=stderr_tail,
            failures=tuple(failures),
        )

    stdout_tail = _tail_lines(completed.stdout)
    stderr_tail = _tail_lines(completed.stderr)
    if completed.returncode != 0:
        failures.append(f"cfloader exited with {completed.returncode}")
        return OnDeviceFailsafeFlashReport(
            uri=str(uri),
            workspace=str(normalized_workspace),
            binary_path=str(normalized_binary),
            cfloader_executable=str(normalized_cfloader),
            boot_mode=str(boot_mode),
            flash_command=tuple(command),
            flashed=False,
            probe_attempted=False,
            availability=None,
            flash_stdout_tail=stdout_tail,
            flash_stderr_tail=stderr_tail,
            failures=tuple(failures),
        )
    flashed = True

    if not skip_probe:
        probe_attempted = True
        try:
            availability = _probe_loaded_failsafe(
                uri=uri,
                workspace=normalized_workspace,
                settle_s=probe_settle_s,
                sleep=sleep,
            )
        except Exception as exc:
            failures.append(f"probe failed: {exc.__class__.__name__}:{exc}")
        else:
            failures.extend(availability.failures)
            if not availability.loaded:
                failures.append("post-flash probe did not see the `twinrFs` firmware app")

    return OnDeviceFailsafeFlashReport(
        uri=str(uri),
        workspace=str(normalized_workspace),
        binary_path=str(normalized_binary),
        cfloader_executable=str(normalized_cfloader),
        boot_mode=str(boot_mode),
        flash_command=tuple(command),
        flashed=flashed,
        probe_attempted=probe_attempted,
        availability=availability,
        flash_stdout_tail=stdout_tail,
        flash_stderr_tail=stderr_tail,
        failures=tuple(failures),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flash and verify the Twinr on-device Crazyflie failsafe app.")
    parser.add_argument("--uri", default=DEFAULT_URI, help="Crazyflie radio URI for warm-boot flashing and probing.")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="Bitcraze workspace root (default: /twinr/bitcraze).")
    parser.add_argument(
        "--binary",
        default=str(DEFAULT_BINARY),
        help="Built STM32 firmware image to flash (default: hardware/bitcraze/twinr_on_device_failsafe/build/twinr_on_device_failsafe.bin).",
    )
    parser.add_argument(
        "--cfloader",
        default=str(DEFAULT_CFLOADER),
        help="cfloader executable to use for flashing (default: /twinr/bitcraze/.venv/bin/cfloader).",
    )
    parser.add_argument(
        "--boot-mode",
        choices=("warm", "cold"),
        default="warm",
        help="Warm boot reboots the current URI into bootloader; cold boot waits for a manual restart (default: warm).",
    )
    parser.add_argument(
        "--flash-timeout-s",
        type=float,
        default=DEFAULT_FLASH_TIMEOUT_S,
        help="Maximum seconds to wait for cfloader before failing (default: 120).",
    )
    parser.add_argument(
        "--probe-settle-s",
        type=float,
        default=DEFAULT_PROBE_SETTLE_S,
        help="Seconds to wait after flash/reset before reconnecting for the post-flash probe (default: 3).",
    )
    parser.add_argument("--skip-probe", action="store_true", help="Only flash the binary and skip the post-flash twinrFs probe.")
    parser.add_argument("--json", action="store_true", help="Emit the full bounded report as JSON.")
    return parser


def _print_human_report(report: OnDeviceFailsafeFlashReport) -> None:
    """Print one compact operator-facing flash/probe report."""

    print(f"flashed={str(report.flashed).lower()}")
    print(f"uri={report.uri}")
    print(f"binary_path={report.binary_path}")
    print(f"boot_mode={report.boot_mode}")
    print(f"probe_attempted={str(report.probe_attempted).lower()}")
    print(f"flash_command={' '.join(report.flash_command)}")
    if report.availability is not None:
        print(f"twinrFs.loaded={str(report.availability.loaded).lower()}")
        print(f"twinrFs.protocol_version={report.availability.protocol_version}")
        print(f"twinrFs.state={report.availability.state_name}")
        print(f"twinrFs.reason={report.availability.reason_name}")
    for line in report.flash_stdout_tail:
        print(f"cfloader.stdout={line}")
    for line in report.flash_stderr_tail:
        print(f"cfloader.stderr={line}")
    for failure in report.failures:
        print(f"failure={failure}")


def main() -> int:
    """Parse args, flash one firmware image, and emit the bounded report."""

    args = _build_parser().parse_args()
    report = flash_on_device_failsafe(
        uri=str(args.uri).strip() or DEFAULT_URI,
        workspace=Path(str(args.workspace).strip() or str(DEFAULT_WORKSPACE)),
        binary_path=Path(str(args.binary).strip() or str(DEFAULT_BINARY)),
        cfloader_executable=Path(str(args.cfloader).strip() or str(DEFAULT_CFLOADER)),
        boot_mode=str(args.boot_mode).strip() or "warm",
        flash_timeout_s=max(5.0, float(args.flash_timeout_s)),
        probe_settle_s=max(0.0, float(args.probe_settle_s)),
        skip_probe=bool(args.skip_probe),
    )
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        _print_human_report(report)
    return 0 if report.flashed and not report.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
