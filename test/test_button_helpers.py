from pathlib import Path
import subprocess
import sys
import time
from typing import Any, cast
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.hardware import buttons as buttons_module
from twinr.hardware.buttons import (
    ButtonAction,
    ButtonBinding,
    GpioButtonMonitor,
    build_button_bindings,
    build_probe_bindings,
    edge_to_action,
)


class _FakeLine:
    def __init__(self, state: dict[str, int]) -> None:
        self._state = state

    def get_value(self) -> int:
        return self._state["value"]

    def release(self) -> None:
        return None


class ButtonHelperTests(unittest.TestCase):
    def test_build_button_bindings_uses_configured_lines(self) -> None:
        config = TwinrConfig(green_button_gpio=17, yellow_button_gpio=27)

        bindings = build_button_bindings(config)

        self.assertEqual(bindings[0].name, "green")
        self.assertEqual(bindings[0].line_offset, 17)
        self.assertEqual(bindings[1].name, "yellow")
        self.assertEqual(bindings[1].line_offset, 27)

    def test_build_probe_bindings_sorts_and_deduplicates_lines(self) -> None:
        bindings = build_probe_bindings([27, 17, 27, 22])

        self.assertEqual([binding.line_offset for binding in bindings], [17, 22, 27])
        self.assertEqual([binding.name for binding in bindings], ["gpio17", "gpio22", "gpio27"])

    def test_active_low_edge_mapping_matches_button_press(self) -> None:
        self.assertEqual(
            edge_to_action(0, active_low=True),
            ButtonAction.PRESSED,
        )
        self.assertEqual(
            edge_to_action(1, active_low=True),
            ButtonAction.RELEASED,
        )

    def test_legacy_monitor_buffers_presses_in_background(self) -> None:
        state = {"value": 1}
        monitor = GpioButtonMonitor(
            "gpiochip0",
            bindings=(ButtonBinding(name="yellow", line_offset=22),),
            active_low=True,
            debounce_ms=40,
        )
        monitor._line_by_offset = {22: _FakeLine(state)}
        monitor._last_values = {22: 1}
        monitor._start_background_sampler()
        self.addCleanup(monitor.close)

        time.sleep(0.05)
        state["value"] = 0
        time.sleep(0.12)
        state["value"] = 1

        pressed = monitor.poll(timeout=0.5)
        released = monitor.poll(timeout=0.5)

        self.assertIsNotNone(pressed)
        self.assertIsNotNone(released)
        self.assertEqual(pressed.name, "yellow")
        self.assertEqual(pressed.action, ButtonAction.PRESSED)
        self.assertEqual(released.action, ButtonAction.RELEASED)

    def test_cli_monitor_uses_legacy_gpioget_syntax_when_help_lacks_named_flags(self) -> None:
        monitor = GpioButtonMonitor(
            "gpiochip0",
            bindings=(ButtonBinding(name="green", line_offset=23),),
        )
        monitor._cli_tools["gpioget"] = "/usr/bin/gpioget"
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            commands.append(command)
            if command == ["/usr/bin/gpioget", "--help"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=(
                        "Usage: gpioget [OPTIONS] <chip name/number> <offset 1>\n"
                        "Options:\n"
                        "  -B, --bias=[as-is|disable|pull-down|pull-up]\n"
                    ),
                    stderr="",
                )
            return subprocess.CompletedProcess(command, 0, stdout="0\n", stderr="")

        with patch.object(buttons_module, "_GPIOGET_CLI_SPECS", {}, create=True):
            with patch("twinr.hardware.buttons.subprocess.run", side_effect=fake_run):
                values = monitor._read_cli_values()

        self.assertEqual(values, {23: 0})
        self.assertEqual(
            commands[1],
            ["/usr/bin/gpioget", "--bias", "pull-up", "gpiochip0", "23"],
        )

    def test_cli_monitor_uses_named_gpioget_flags_when_help_advertises_them(self) -> None:
        monitor = GpioButtonMonitor(
            "gpiochip0",
            bindings=(
                ButtonBinding(name="green", line_offset=17),
                ButtonBinding(name="yellow", line_offset=27),
            ),
        )
        monitor._cli_tools["gpioget"] = "/usr/bin/gpioget"
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            commands.append(command)
            if command == ["/usr/bin/gpioget", "--help"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=(
                        "Usage: gpioget [OPTIONS] <offset 1>\n"
                        "Options:\n"
                        "  --chip <chip>\n"
                        "  --numeric\n"
                        "  -B, --bias=[as-is|disable|pull-down|pull-up]\n"
                    ),
                    stderr="",
                )
            return subprocess.CompletedProcess(command, 0, stdout="1 0\n", stderr="")

        with patch.object(buttons_module, "_GPIOGET_CLI_SPECS", {}, create=True):
            with patch("twinr.hardware.buttons.subprocess.run", side_effect=fake_run):
                values = monitor._read_cli_values()

        self.assertEqual(values, {17: 1, 27: 0})
        self.assertEqual(
            commands[1],
            ["/usr/bin/gpioget", "--bias", "pull-up", "--chip", "gpiochip0", "--numeric", "17", "27"],
        )

    def test_cli_monitor_retries_with_legacy_syntax_when_named_flags_are_rejected(self) -> None:
        monitor = GpioButtonMonitor(
            "gpiochip0",
            bindings=(ButtonBinding(name="green", line_offset=23),),
        )
        monitor._cli_tools["gpioget"] = "/usr/bin/gpioget"
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            commands.append(command)
            if command == ["/usr/bin/gpioget", "--help"]:
                raise OSError("help unavailable")
            if "--chip" in command or "--numeric" in command:
                raise subprocess.CalledProcessError(
                    1,
                    command,
                    stderr="/usr/bin/gpioget: unrecognized option '--chip'\n",
                )
            return subprocess.CompletedProcess(command, 0, stdout="1\n", stderr="")

        with patch.object(buttons_module, "_GPIOGET_CLI_SPECS", {}, create=True):
            with patch("twinr.hardware.buttons.subprocess.run", side_effect=fake_run):
                values = monitor._read_cli_values()

        self.assertEqual(values, {23: 1})
        self.assertEqual(
            commands[-1],
            ["/usr/bin/gpioget", "--bias", "pull-up", "gpiochip0", "23"],
        )

    def test_gpiomon_legacy_command_places_options_before_chip_and_offsets(self) -> None:
        spec = buttons_module._legacy_gpiomon_cli_spec()

        with patch.object(buttons_module, "_GPIOMON_CLI_SPECS", {"/usr/bin/gpiomon": spec}, create=True):
            command = buttons_module._build_gpiomon_command(
                "/usr/bin/gpiomon",
                chip_name="gpiochip0",
                bindings=(ButtonBinding(name="green", line_offset=17),),
                bias="pull-up",
                consumer="twinr-buttons",
                debounce_ms=40,
            )

        self.assertEqual(
            command,
            [
                "/usr/bin/gpiomon",
                "--bias",
                "pull-up",
                "--line-buffered",
                "--format",
                "%o\t%e",
                "gpiochip0",
                "17",
            ],
        )

    def test_gpiomon_named_command_keeps_chip_flag_before_offsets(self) -> None:
        spec = buttons_module._GpiomonCliSpec(
            supports_chip_option=True,
            supports_consumer_option=True,
            supports_edges_option=True,
            supports_event_clock_option=True,
            supports_debounce_option=True,
            supports_line_buffered_option=False,
            disabled_bias_keyword="disabled",
        )

        with patch.object(buttons_module, "_GPIOMON_CLI_SPECS", {"/usr/bin/gpiomon": spec}, create=True):
            command = buttons_module._build_gpiomon_command(
                "/usr/bin/gpiomon",
                chip_name="gpiochip0",
                bindings=(
                    ButtonBinding(name="green", line_offset=17),
                    ButtonBinding(name="yellow", line_offset=27),
                ),
                bias="pull-up",
                consumer="twinr-buttons",
                debounce_ms=40,
            )

        self.assertEqual(
            command,
            [
                "/usr/bin/gpiomon",
                "--bias",
                "pull-up",
                "--chip",
                "gpiochip0",
                "--consumer",
                "twinr-buttons",
                "--edges",
                "both",
                "--event-clock",
                "monotonic",
                "--debounce-period",
                "40ms",
                "--format",
                "%o\t%e",
                "17",
                "27",
            ],
        )

    def test_snapshot_values_uses_cached_state_while_gpiomon_backend_owns_lines(self) -> None:
        monitor = GpioButtonMonitor(
            "gpiochip0",
            bindings=(
                ButtonBinding(name="green", line_offset=17),
                ButtonBinding(name="yellow", line_offset=27),
            ),
        )
        monitor._gpiomon_process = cast(Any, object())
        monitor._last_values = {17: 1, 27: 0}

        with patch.object(monitor, "_read_cli_values", side_effect=AssertionError("gpioget must not run")):
            values = monitor.snapshot_values()

        self.assertEqual(values, {17: 1, 27: 0})
