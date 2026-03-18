#!/usr/bin/env python3
"""Probe Waveshare BUSY/RST/SPI full-refresh paths on the Raspberry Pi.

Purpose
-------
Run a bounded, instrumented display sequence outside the main Twinr runtime so
operators can prove which full-refresh stage stalls the Waveshare 4.2" panel.

Usage
-----
Command-line invocation examples::

    python3 hardware/display/probe_busy_path.py --env-file .env
    python3 hardware/display/probe_busy_path.py --env-file .env --steady-renders 6

Inputs
------
- ``--env-file`` path to the Twinr environment file with display wiring
- ``--steady-renders`` number of repeated full-refresh renders after the first
- ``--sleep-between-s`` pause between repeated full-refresh renders
- ``--busy-timeout-s`` optional override for the bounded BUSY timeout

Outputs
-------
- Emits JSON-line telemetry for GPIO, BUSY, SPI-command, and phase timing data
- Exits 0 when all requested phases succeed
- Exits 1 when any phase hits a bounded timeout or other runtime error
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.display import WaveshareEPD4In2V2


_BUSY_POLL_DELAY_MS = 20
_BUSY_SAMPLE_LIMIT = 24
_GPIO_SNAPSHOT_TIMEOUT_S = 3.0
_SUPPLY_SNAPSHOT_TIMEOUT_S = 3.0
_SPI_COMMAND_SAMPLE_LIMIT = 8


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the BUSY-path probe."""

    parser = argparse.ArgumentParser(description="Probe bounded Waveshare BUSY-path behaviour on the Pi")
    parser.add_argument("--env-file", default=Path(__file__).resolve().parents[2] / ".env")
    parser.add_argument(
        "--steady-renders",
        type=int,
        default=6,
        help="Number of additional full-refresh renders after the initial render",
    )
    parser.add_argument(
        "--sleep-between-s",
        type=float,
        default=1.0,
        help="Pause between repeated full-refresh renders",
    )
    parser.add_argument(
        "--busy-timeout-s",
        type=float,
        help="Optional override for the adapter's bounded BUSY timeout",
    )
    return parser


def _now_utc() -> str:
    """Return the current UTC timestamp as ISO-8601 text."""

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(value: object) -> object:
    """Serialize simple non-JSON-native probe values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return repr(value)


def emit_event(kind: str, **details: object) -> None:
    """Emit one structured probe event as compact JSON."""

    payload = {
        "ts_utc": _now_utc(),
        "ts_mono_s": round(time.monotonic(), 6),
        "kind": kind,
        "details": details,
    }
    print(json.dumps(payload, sort_keys=True, default=_json_default), flush=True)


def _run_command(command: list[str], *, timeout_s: float) -> dict[str, object]:
    """Run one bounded shell command and return a compact result record."""

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:
        return {
            "command": command,
            "error": type(exc).__name__,
            "detail": str(exc),
        }

    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


class ProbeTimeout(RuntimeError):
    """Raised when the bounded BUSY probe exceeds the configured timeout."""


class VendorBusyProbeSession:
    """Instrument one vendor-driver session for bounded Waveshare probing."""

    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.adapter = WaveshareEPD4In2V2.from_config(config)
        self.driver_module: Any | None = None
        self.epdconfig_module: Any | None = None
        self.epd: Any | None = None
        self.buffer: Any | None = None
        self.current_phase = "idle"
        self.last_command: str | None = None
        self.busy_last_value: int | None = None
        self.spi_write_calls = 0
        self.spi_write_bytes = 0
        self.gpio_levels: dict[int, int] = {}

    def open(self) -> None:
        """Load the vendor package, create the driver instance, and instrument it."""

        self.driver_module = self.adapter._load_driver_module()
        self.epdconfig_module = self.adapter._epdconfig_module
        if self.epdconfig_module is None:
            raise RuntimeError("Display adapter did not expose the loaded epdconfig module.")
        self.epd = self.driver_module.EPD()
        self.adapter._driver_module = self.driver_module
        self.adapter._epdconfig_module = self.epdconfig_module
        self.adapter._epd = self.epd
        self._instrument_epdconfig()
        self._instrument_epd()
        image = self.adapter.render_test_image()
        prepared_image = self.adapter._prepare_image(image)
        self.adapter._validate_prepared_image(prepared_image)
        self.buffer = self.epd.getbuffer(prepared_image)
        emit_event(
            "probe_session_opened",
            busy_gpio=self.config.display_busy_gpio,
            reset_gpio=self.config.display_reset_gpio,
            dc_gpio=self.config.display_dc_gpio,
            cs_gpio=self.config.display_cs_gpio,
            timeout_s=self.adapter.busy_timeout_s,
            supply=self._supply_snapshot(),
            gpio=self._gpio_snapshot(),
        )

    def close(self) -> None:
        """Release driver resources and emit the final GPIO/supply snapshot."""

        emit_event(
            "probe_session_closing",
            phase=self.current_phase,
            supply=self._supply_snapshot(),
            gpio=self._gpio_snapshot(),
        )
        self.adapter.close()

    def run_initial_full_render(self) -> None:
        """Run the initial full-refresh render path."""

        self._run_phase("initial_full_render", self._full_render)

    def run_steady_full_render(self, iteration: int) -> None:
        """Run one repeated full-refresh render on the same driver session."""

        self._run_phase(f"steady_full_render_{iteration}", self._full_render)

    def run_clear_recovery(self) -> None:
        """Run the clear-first recovery path on a fresh driver session."""

        self._run_phase("clear_recovery_render", self._clear_render)

    def _run_phase(self, phase_name: str, action: Any) -> None:
        started_at = time.monotonic()
        self.current_phase = phase_name
        self.last_command = None
        emit_event(
            "phase_start",
            phase=phase_name,
            supply=self._supply_snapshot(),
            gpio=self._gpio_snapshot(),
        )
        try:
            action()
        except Exception as exc:
            emit_event(
                "phase_error",
                phase=phase_name,
                elapsed_s=round(time.monotonic() - started_at, 3),
                error=type(exc).__name__,
                detail=str(exc),
                last_command=self.last_command,
                spi_write_calls=self.spi_write_calls,
                spi_write_bytes=self.spi_write_bytes,
                supply=self._supply_snapshot(),
                gpio=self._gpio_snapshot(),
                traceback=traceback.format_exc().strip().splitlines()[-6:],
            )
            raise
        emit_event(
            "phase_ok",
            phase=phase_name,
            elapsed_s=round(time.monotonic() - started_at, 3),
            last_command=self.last_command,
            spi_write_calls=self.spi_write_calls,
            spi_write_bytes=self.spi_write_bytes,
            supply=self._supply_snapshot(),
            gpio=self._gpio_snapshot(),
        )

    def _full_render(self) -> None:
        self.epd.init()
        self.epd.display(self.buffer)

    def _clear_render(self) -> None:
        self.epd.init()
        self.epd.Clear()
        self.epd.display(self.buffer)

    def _instrument_epdconfig(self) -> None:
        module = self.epdconfig_module
        original_digital_write = module.digital_write
        original_digital_read = module.digital_read
        original_delay_ms = module.delay_ms
        original_module_init = module.module_init
        original_module_exit = module.module_exit
        pwr_pin = int(getattr(module, "PWR_PIN", 18))
        watched_pins = {
            int(self.config.display_reset_gpio): "rst",
            int(self.config.display_busy_gpio): "busy",
            int(self.config.display_dc_gpio): "dc",
            int(self.config.display_cs_gpio): "cs",
            pwr_pin: "pwr",
        }

        def wrapped_module_init(*args: object, **kwargs: object) -> object:
            emit_event("module_init_start", phase=self.current_phase)
            result = original_module_init(*args, **kwargs)
            implementation = getattr(module, "implementation", None)
            spi = getattr(implementation, "SPI", None)
            emit_event(
                "module_init_end",
                phase=self.current_phase,
                result=result,
                spi_bus=getattr(self.config, "display_spi_bus", None),
                spi_device=getattr(self.config, "display_spi_device", None),
                spi_mode=getattr(spi, "mode", None),
                spi_max_speed_hz=getattr(spi, "max_speed_hz", None),
            )
            return result

        def wrapped_module_exit(*args: object, **kwargs: object) -> object:
            emit_event("module_exit_start", phase=self.current_phase)
            result = original_module_exit(*args, **kwargs)
            emit_event("module_exit_end", phase=self.current_phase, result=result)
            return result

        def wrapped_digital_write(pin: object, value: object) -> None:
            original_digital_write(pin, value)
            pin_int = int(pin)
            value_int = int(value)
            previous = self.gpio_levels.get(pin_int)
            self.gpio_levels[pin_int] = value_int
            if previous != value_int and pin_int in watched_pins:
                emit_event(
                    "gpio_write",
                    phase=self.current_phase,
                    pin=pin_int,
                    pin_name=watched_pins[pin_int],
                    value=value_int,
                )

        def wrapped_digital_read(pin: object) -> object:
            value = int(original_digital_read(pin))
            pin_int = int(pin)
            if pin_int in watched_pins:
                previous = self.gpio_levels.get(pin_int)
                self.gpio_levels[pin_int] = value
                if previous != value:
                    emit_event(
                        "gpio_read_transition",
                        phase=self.current_phase,
                        pin=pin_int,
                        pin_name=watched_pins[pin_int],
                        value=value,
                    )
            return value

        module.module_init = wrapped_module_init
        module.module_exit = wrapped_module_exit
        module.digital_write = wrapped_digital_write
        module.digital_read = wrapped_digital_read
        module.delay_ms = original_delay_ms
        self._original_digital_read = original_digital_read
        self._original_delay_ms = original_delay_ms

    def _instrument_epd(self) -> None:
        epd = self.epd
        original_init = epd.init
        original_reset = epd.reset
        original_display = epd.display
        original_clear = epd.Clear
        original_turn_on = epd.TurnOnDisplay
        original_send_command = epd.send_command
        original_send_data = epd.send_data
        original_send_data2 = epd.send_data2

        def wrap_call(method_name: str, original: Any) -> Any:
            def _wrapped(*args: object, **kwargs: object) -> object:
                started_at = time.monotonic()
                emit_event(
                    "epd_call_start",
                    phase=self.current_phase,
                    method=method_name,
                )
                try:
                    result = original(*args, **kwargs)
                except Exception as exc:
                    emit_event(
                        "epd_call_error",
                        phase=self.current_phase,
                        method=method_name,
                        elapsed_s=round(time.monotonic() - started_at, 3),
                        error=type(exc).__name__,
                        detail=str(exc),
                    )
                    raise
                emit_event(
                    "epd_call_end",
                    phase=self.current_phase,
                    method=method_name,
                    elapsed_s=round(time.monotonic() - started_at, 3),
                    result=result,
                )
                return result

            return _wrapped

        def wrapped_send_command(command: object) -> object:
            self.last_command = f"0x{int(command):02X}"
            emit_event(
                "epd_command",
                phase=self.current_phase,
                command=self.last_command,
            )
            return original_send_command(command)

        def wrapped_send_data(data: object) -> object:
            emit_event(
                "epd_data_byte",
                phase=self.current_phase,
                command=self.last_command,
                value=int(data),
            )
            return original_send_data(data)

        def wrapped_send_data2(data: object) -> object:
            try:
                length = len(data)
            except Exception:
                length = None
            self.spi_write_calls += 1
            if length is not None:
                self.spi_write_bytes += int(length)
            sample = []
            if length is not None:
                try:
                    sample = [int(value) for value in list(data[:_SPI_COMMAND_SAMPLE_LIMIT])]
                except Exception:
                    sample = []
            emit_event(
                "spi_bulk_write",
                phase=self.current_phase,
                command=self.last_command,
                bytes=length,
                sample=sample,
                call_index=self.spi_write_calls,
            )
            return original_send_data2(data)

        def wrapped_readbusy() -> None:
            caller = inspect.stack()[1].function
            busy_pin = int(epd.busy_pin)
            started_at = time.monotonic()
            sampled_states: list[dict[str, object]] = []
            emit_event(
                "busy_wait_start",
                phase=self.current_phase,
                caller=caller,
                command=self.last_command,
                gpio=busy_pin,
            )
            while True:
                value = int(self._original_digital_read(busy_pin))
                if value != self.busy_last_value:
                    self.busy_last_value = value
                    emit_event(
                        "busy_wait_transition",
                        phase=self.current_phase,
                        caller=caller,
                        command=self.last_command,
                        gpio=busy_pin,
                        value=value,
                        elapsed_s=round(time.monotonic() - started_at, 3),
                    )
                if len(sampled_states) < _BUSY_SAMPLE_LIMIT:
                    sampled_states.append(
                        {
                            "elapsed_s": round(time.monotonic() - started_at, 3),
                            "value": value,
                        }
                    )
                if value == 0:
                    emit_event(
                        "busy_wait_end",
                        phase=self.current_phase,
                        caller=caller,
                        command=self.last_command,
                        gpio=busy_pin,
                        elapsed_s=round(time.monotonic() - started_at, 3),
                        samples=sampled_states,
                    )
                    return
                age_s = time.monotonic() - started_at
                if age_s >= float(self.adapter.busy_timeout_s):
                    emit_event(
                        "busy_wait_timeout",
                        phase=self.current_phase,
                        caller=caller,
                        command=self.last_command,
                        gpio=busy_pin,
                        elapsed_s=round(age_s, 3),
                        samples=sampled_states,
                        supply=self._supply_snapshot(),
                        gpio_snapshot=self._gpio_snapshot(),
                    )
                    raise ProbeTimeout(
                        "BUSY stayed active during "
                        f"{self.current_phase} caller={caller} command={self.last_command} "
                        f"for {age_s:.1f}s on GPIO {busy_pin}; samples={sampled_states}"
                    )
                self._original_delay_ms(_BUSY_POLL_DELAY_MS)

        epd.init = wrap_call("init", original_init)
        epd.reset = wrap_call("reset", original_reset)
        epd.display = wrap_call("display", original_display)
        epd.Clear = wrap_call("Clear", original_clear)
        epd.TurnOnDisplay = wrap_call("TurnOnDisplay", original_turn_on)
        epd.send_command = wrapped_send_command
        epd.send_data = wrapped_send_data
        epd.send_data2 = wrapped_send_data2
        epd.ReadBusy = wrapped_readbusy

    def _gpio_snapshot(self) -> dict[str, object]:
        pins = (
            str(self.config.display_reset_gpio),
            str(getattr(self.epdconfig_module, "PWR_PIN", 18)),
            str(self.config.display_busy_gpio),
            str(self.config.display_dc_gpio),
            str(self.config.display_cs_gpio),
        )
        command = [
            "bash",
            "-lc",
            "; ".join(f"pinctrl get {pin}" for pin in pins),
        ]
        return _run_command(command, timeout_s=_GPIO_SNAPSHOT_TIMEOUT_S)

    def _supply_snapshot(self) -> dict[str, object]:
        return _run_command(["vcgencmd", "get_throttled"], timeout_s=_SUPPLY_SNAPSHOT_TIMEOUT_S)


def main() -> int:
    """Run the bounded BUSY-path probe."""

    args = build_parser().parse_args()
    config = TwinrConfig.from_env(Path(args.env_file))
    if args.busy_timeout_s is not None:
        config.display_busy_timeout_s = float(args.busy_timeout_s)
    if args.steady_renders < 0:
        raise SystemExit("--steady-renders must be >= 0")
    if args.sleep_between_s < 0:
        raise SystemExit("--sleep-between-s must be >= 0")

    summary: list[dict[str, object]] = []
    steady_failure = False

    session = VendorBusyProbeSession(config)
    try:
        session.open()
        session.run_initial_full_render()
        summary.append({"phase": "initial_full_render", "status": "ok"})
        for iteration in range(1, args.steady_renders + 1):
            try:
                session.run_steady_full_render(iteration)
            except Exception as exc:
                summary.append(
                    {
                        "phase": f"steady_full_render_{iteration}",
                        "status": "error",
                        "error": type(exc).__name__,
                        "detail": str(exc),
                    }
                )
                steady_failure = True
                break
            summary.append({"phase": f"steady_full_render_{iteration}", "status": "ok"})
            if args.sleep_between_s > 0:
                time.sleep(args.sleep_between_s)
    finally:
        session.close()

    recovery_status = "skipped"
    recovery_error: dict[str, object] | None = None
    if steady_failure:
        recovery = VendorBusyProbeSession(config)
        try:
            recovery.open()
            recovery.run_clear_recovery()
            recovery_status = "ok"
        except Exception as exc:
            recovery_status = "error"
            recovery_error = {
                "error": type(exc).__name__,
                "detail": str(exc),
            }
        finally:
            recovery.close()
        entry: dict[str, object] = {"phase": "clear_recovery_render", "status": recovery_status}
        if recovery_error is not None:
            entry.update(recovery_error)
        summary.append(entry)

    emit_event("probe_summary", phases=summary)
    return 1 if any(entry["status"] == "error" for entry in summary) else 0


if __name__ == "__main__":
    raise SystemExit(main())
