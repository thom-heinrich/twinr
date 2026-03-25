from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import cast
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware import servo_follow
from twinr.hardware.servo_follow import (
    AttentionServoConfig,
    AttentionServoController,
    LGPIOPWMServoPulseWriter,
    LGPIOServoPulseWriter,
    PigpioServoPulseWriter,
    SysfsPWMServoPulseWriter,
    TwinrKernelServoPulseWriter,
)


class FakeServoPulseWriter:
    def __init__(self) -> None:
        self.writes: list[tuple[str, int, int]] = []
        self.disables: list[tuple[str, int]] = []
        self.closed = False
        self.current_pulse_width_us_value: int | None = None

    def probe(self, gpio: int) -> None:
        del gpio

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        self.writes.append((gpio_chip, gpio, pulse_width_us))

    def disable(self, *, gpio_chip: str, gpio: int) -> None:
        self.disables.append((gpio_chip, gpio))

    def current_pulse_width_us(self, *, gpio_chip: str, gpio: int) -> int | None:
        del gpio_chip, gpio
        return self.current_pulse_width_us_value

    def close(self) -> None:
        self.closed = True


class RecoveringFakeServoPulseWriter(FakeServoPulseWriter):
    def __init__(self) -> None:
        super().__init__()
        self.probes: list[int] = []
        self.fail_next_write = True

    def probe(self, gpio: int) -> None:
        self.probes.append(gpio)

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        if self.fail_next_write:
            self.fail_next_write = False
            raise RuntimeError("maestro command port missing")
        super().write(gpio_chip=gpio_chip, gpio=gpio, pulse_width_us=pulse_width_us)


class FakeLGPIOModule:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def gpiochip_open(self, chip_index: int) -> int:
        self.calls.append(("gpiochip_open", chip_index))
        return 100 + chip_index

    def gpio_claim_output(self, handle: int, gpio: int, level: int) -> None:
        self.calls.append(("gpio_claim_output", handle, gpio, level))

    def tx_servo(self, handle: int, gpio: int, pulse_width_us: int, *, servo_frequency: int) -> None:
        self.calls.append(("tx_servo", handle, gpio, pulse_width_us, servo_frequency))

    def tx_pwm(self, handle: int, gpio: int, pwm_frequency: float, pwm_duty_cycle: float) -> int:
        self.calls.append(("tx_pwm", handle, gpio, pwm_frequency, pwm_duty_cycle))
        return 0

    def gpio_claim_input(self, handle: int, gpio: int) -> None:
        self.calls.append(("gpio_claim_input", handle, gpio))

    def gpiochip_close(self, handle: int) -> None:
        self.calls.append(("gpiochip_close", handle))


class FakePigpioConnection:
    def __init__(self, *, connected: bool = True) -> None:
        self.connected = connected
        self.calls: list[tuple[object, ...]] = []

    def set_mode(self, gpio: int, mode: int) -> int:
        self.calls.append(("set_mode", gpio, mode))
        return 0

    def set_servo_pulsewidth(self, gpio: int, pulse_width_us: int) -> int:
        self.calls.append(("set_servo_pulsewidth", gpio, pulse_width_us))
        return 0

    def stop(self) -> None:
        self.calls.append(("stop",))


class FakePigpioModule:
    INPUT = 0
    OUTPUT = 1

    def __init__(self, connection: FakePigpioConnection) -> None:
        self.connection = connection

    def pi(self) -> FakePigpioConnection:
        return self.connection


class LGPIOServoPulseWriterTests(unittest.TestCase):
    def test_explicit_pololu_maestro_driver_builds_maestro_writer(self) -> None:
        config = AttentionServoConfig(
            enabled=True,
            driver="pololu_maestro",
            maestro_device="/dev/serial/by-id/maestro-if00",
            gpio=0,
        )

        with mock.patch.object(servo_follow, "PololuMaestroServoPulseWriter") as writer_class:
            writer = writer_class.return_value
            selected = servo_follow._default_pulse_writer_for_config(config)

        writer_class.assert_called_once_with(device_path="/dev/serial/by-id/maestro-if00")
        writer.probe.assert_not_called()
        self.assertIs(selected, writer)

    def test_explicit_peer_pololu_maestro_driver_builds_peer_writer(self) -> None:
        config = AttentionServoConfig(
            enabled=True,
            driver="peer_pololu_maestro",
            peer_base_url="http://10.42.0.2:8768",
            peer_timeout_s=2.5,
            gpio=1,
        )

        with mock.patch.object(servo_follow, "PeerPololuMaestroServoPulseWriter") as writer_class:
            writer = writer_class.return_value
            selected = servo_follow._default_pulse_writer_for_config(config)

        writer_class.assert_called_once_with(
            base_url="http://10.42.0.2:8768",
            timeout_s=2.5,
        )
        self.assertIs(selected, writer)

    def test_write_claims_output_before_first_servo_pulse(self) -> None:
        fake_module = FakeLGPIOModule()
        writer = LGPIOServoPulseWriter()
        writer._lgpio = fake_module

        writer.write(gpio_chip="gpiochip0", gpio=18, pulse_width_us=1500)
        writer.write(gpio_chip="gpiochip0", gpio=18, pulse_width_us=1510)

        self.assertEqual(
            fake_module.calls,
            [
                ("gpiochip_open", 0),
                ("gpio_claim_output", 100, 18, 0),
                ("tx_servo", 100, 18, 1500, 50),
                ("tx_servo", 100, 18, 1510, 50),
            ],
        )

    def test_disable_releases_claim_and_next_write_reclaims_output(self) -> None:
        fake_module = FakeLGPIOModule()
        writer = LGPIOServoPulseWriter()
        writer._lgpio = fake_module

        writer.write(gpio_chip="gpiochip0", gpio=18, pulse_width_us=1500)
        writer.disable(gpio_chip="gpiochip0", gpio=18)
        writer.write(gpio_chip="gpiochip0", gpio=18, pulse_width_us=1490)

        self.assertEqual(
            fake_module.calls,
            [
                ("gpiochip_open", 0),
                ("gpio_claim_output", 100, 18, 0),
                ("tx_servo", 100, 18, 1500, 50),
                ("tx_servo", 100, 18, 0, 50),
                ("gpio_claim_input", 100, 18),
                ("gpio_claim_output", 100, 18, 0),
                ("tx_servo", 100, 18, 1490, 50),
            ],
        )

    def test_auto_driver_falls_back_to_lgpio_when_pigpio_probe_fails(self) -> None:
        config = AttentionServoConfig(enabled=True, driver="auto", gpio_chip="gpiochip0", gpio=18)

        with (
            mock.patch.object(
                servo_follow.TwinrKernelServoPulseWriter,
                "probe",
                side_effect=RuntimeError("kernel servo unavailable"),
            ),
            mock.patch.object(
                servo_follow,
                "_detect_conflicting_servo_gpio_environment",
                return_value=[],
            ),
            mock.patch.object(
                servo_follow.SysfsPWMServoPulseWriter,
                "probe",
                side_effect=RuntimeError("sysfs pwm unavailable"),
            ),
            mock.patch.object(
                servo_follow.PigpioServoPulseWriter,
                "_connection",
                side_effect=RuntimeError("pigpiod is required"),
            ),
        ):
            writer = servo_follow._default_pulse_writer_for_config(config)

        self.assertIsInstance(writer, LGPIOPWMServoPulseWriter)

    def test_auto_driver_prefers_sysfs_pwm_when_probe_succeeds(self) -> None:
        config = AttentionServoConfig(enabled=True, driver="auto", gpio_chip="gpiochip0", gpio=18)

        with (
            mock.patch.object(
                servo_follow.TwinrKernelServoPulseWriter,
                "probe",
                side_effect=RuntimeError("kernel servo unavailable"),
            ),
            mock.patch.object(
                servo_follow,
                "_detect_conflicting_servo_gpio_environment",
                return_value=[],
            ),
            mock.patch.object(
                servo_follow.SysfsPWMServoPulseWriter,
                "probe",
                return_value=None,
            ),
        ):
            writer = servo_follow._default_pulse_writer_for_config(config)

        self.assertIsInstance(writer, SysfsPWMServoPulseWriter)


class LGPIOPWMServoPulseWriterTests(unittest.TestCase):
    def test_write_claims_output_before_first_pwm_update(self) -> None:
        fake_module = FakeLGPIOModule()
        writer = LGPIOPWMServoPulseWriter()
        writer._lgpio = fake_module

        writer.write(gpio_chip="gpiochip0", gpio=18, pulse_width_us=1500)
        writer.write(gpio_chip="gpiochip0", gpio=18, pulse_width_us=1600)

        self.assertEqual(
            fake_module.calls,
            [
                ("gpiochip_open", 0),
                ("gpio_claim_output", 100, 18, 0),
                ("tx_pwm", 100, 18, 50.0, 7.5),
                ("tx_pwm", 100, 18, 50.0, 8.0),
            ],
        )

    def test_disable_stops_pwm_and_releases_claim(self) -> None:
        fake_module = FakeLGPIOModule()
        writer = LGPIOPWMServoPulseWriter()
        writer._lgpio = fake_module

        writer.write(gpio_chip="gpiochip0", gpio=18, pulse_width_us=1500)
        writer.disable(gpio_chip="gpiochip0", gpio=18)
        writer.write(gpio_chip="gpiochip0", gpio=18, pulse_width_us=1450)

        self.assertEqual(
            fake_module.calls,
            [
                ("gpiochip_open", 0),
                ("gpio_claim_output", 100, 18, 0),
                ("tx_pwm", 100, 18, 50.0, 7.5),
                ("tx_pwm", 100, 18, 0.0, 0.0),
                ("gpio_claim_input", 100, 18),
                ("gpio_claim_output", 100, 18, 0),
                ("tx_pwm", 100, 18, 50.0, 7.25),
            ],
        )


class PigpioServoPulseWriterTests(unittest.TestCase):
    def test_write_sets_output_once_and_updates_servo_pulsewidth(self) -> None:
        connection = FakePigpioConnection()
        writer = PigpioServoPulseWriter()
        writer._pigpio = FakePigpioModule(connection)

        writer.write(gpio_chip="gpiochip0", gpio=18, pulse_width_us=1500)
        writer.write(gpio_chip="gpiochip0", gpio=18, pulse_width_us=1512)

        self.assertEqual(
            connection.calls,
            [
                ("set_mode", 18, 1),
                ("set_servo_pulsewidth", 18, 1500),
                ("set_servo_pulsewidth", 18, 1512),
            ],
        )

    def test_disable_releases_servo_and_returns_gpio_to_input(self) -> None:
        connection = FakePigpioConnection()
        writer = PigpioServoPulseWriter()
        writer._pigpio = FakePigpioModule(connection)

        writer.write(gpio_chip="gpiochip0", gpio=18, pulse_width_us=1500)
        writer.disable(gpio_chip="gpiochip0", gpio=18)
        writer.close()

        self.assertEqual(
            connection.calls,
            [
                ("set_mode", 18, 1),
                ("set_servo_pulsewidth", 18, 1500),
                ("set_servo_pulsewidth", 18, 0),
                ("set_mode", 18, 0),
                ("stop",),
            ],
        )

    def test_write_requires_connected_pigpiod(self) -> None:
        connection = FakePigpioConnection(connected=False)
        writer = PigpioServoPulseWriter()
        writer._pigpio = FakePigpioModule(connection)

        with self.assertRaisesRegex(RuntimeError, "pigpiod is required"):
            writer.write(gpio_chip="gpiochip0", gpio=18, pulse_width_us=1500)


class SysfsPWMServoPulseWriterTests(unittest.TestCase):
    def test_probe_validates_gpio_pwm_binding_and_chip_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "pwmchip0").mkdir()
            writer = SysfsPWMServoPulseWriter(sysfs_root=root)
            writer._run_pinctrl = mock.Mock(  # type: ignore[method-assign]
                return_value="18, GPIO18, SPI1_CE0, DPI_D14, I2S0_SCLK, PWM0_CHAN2"
            )

            writer.probe(18)

            self.assertEqual(
                writer._descriptors[18],
                servo_follow._SysfsPWMDescriptor(
                    pwm_chip_index=0,
                    pwm_channel_index=2,
                    pinctrl_alt_mode="a3",
                ),
            )

    def test_write_and_disable_program_sysfs_pwm_and_pinctrl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pwmdir = root / "pwmchip0" / "pwm2"
            pwmdir.mkdir(parents=True)
            for name, value in {
                "period": "0",
                "duty_cycle": "0",
                "enable": "0",
                "polarity": "normal",
            }.items():
                (pwmdir / name).write_text(value, encoding="utf-8")

            calls: list[tuple[str, ...]] = []

            def fake_run_pinctrl(*args: str) -> str:
                calls.append(tuple(args))
                if args[:2] == ("funcs", "18"):
                    return "18, GPIO18, SPI1_CE0, DPI_D14, I2S0_SCLK, PWM0_CHAN2"
                return ""

            writer = SysfsPWMServoPulseWriter(sysfs_root=root)
            writer._run_pinctrl = fake_run_pinctrl  # type: ignore[method-assign]

            writer.write(gpio_chip="gpiochip0", gpio=18, pulse_width_us=1500)

            self.assertEqual((pwmdir / "period").read_text(encoding="utf-8"), "20000000")
            self.assertEqual((pwmdir / "duty_cycle").read_text(encoding="utf-8"), "1500000")
            self.assertEqual((pwmdir / "enable").read_text(encoding="utf-8"), "1")
            self.assertEqual(
                calls,
                [
                    ("funcs", "18"),
                    ("set", "18", "a3", "pn"),
                ],
            )

            writer.disable(gpio_chip="gpiochip0", gpio=18)

            self.assertEqual((pwmdir / "enable").read_text(encoding="utf-8"), "0")
            self.assertEqual(
                calls,
                [
                    ("funcs", "18"),
                    ("set", "18", "a3", "pn"),
                    ("set", "18", "ip", "pd"),
                ],
            )


class TwinrKernelServoPulseWriterTests(unittest.TestCase):
    def test_probe_requires_kernel_sysfs_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = TwinrKernelServoPulseWriter(sysfs_root=Path(temp_dir))

            with self.assertRaisesRegex(RuntimeError, "Twinr kernel servo module"):
                writer.probe(18)

    def test_write_and_disable_program_kernel_sysfs_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, value in {
                "gpio": "-1\n",
                "period_us": "20000\n",
                "pulse_width_us": "1500\n",
                "enabled": "0\n",
            }.items():
                (root / name).write_text(value, encoding="utf-8")

            writer = TwinrKernelServoPulseWriter(sysfs_root=root)

            writer.write(gpio_chip="gpiochip0", gpio=18, pulse_width_us=1498)
            self.assertEqual((root / "gpio").read_text(encoding="utf-8").strip(), "18")
            self.assertEqual((root / "pulse_width_us").read_text(encoding="utf-8").strip(), "1498")
            self.assertEqual((root / "enabled").read_text(encoding="utf-8").strip(), "1")

            writer.write(gpio_chip="gpiochip0", gpio=18, pulse_width_us=1481)
            self.assertEqual((root / "gpio").read_text(encoding="utf-8").strip(), "18")
            self.assertEqual((root / "pulse_width_us").read_text(encoding="utf-8").strip(), "1481")
            self.assertEqual((root / "enabled").read_text(encoding="utf-8").strip(), "1")

            writer.disable(gpio_chip="gpiochip0", gpio=18)
            self.assertEqual((root / "enabled").read_text(encoding="utf-8").strip(), "0")
            self.assertEqual((root / "gpio").read_text(encoding="utf-8").strip(), "-1")


class ServoEnvironmentGuardTests(unittest.TestCase):
    def test_detect_conflicting_servo_gpio_environment_reports_pwm_pio_overlay_and_remove_process(self) -> None:
        def fake_run_text_command(args: list[str]) -> str | None:
            if args == ["dtoverlay", "-l"]:
                return "Overlays (in load order):\n0: pwm pin=18 func=2\n1: pwm-pio gpio=18\n"
            if args == ["ps", "-eo", "pid=,stat=,args="]:
                return "504086 D dtoverlay pwm-pio gpio=18\n532933 D dtoverlay -r 1\n"
            return None

        conflicts = servo_follow._detect_conflicting_servo_gpio_environment(
            gpio=18,
            run_text_command=fake_run_text_command,
        )

        self.assertIn("overlay pwm-pio gpio=18", conflicts)
        self.assertIn("process pid=504086 stat=D args=dtoverlay pwm-pio gpio=18", conflicts)
        self.assertIn("process pid=532933 stat=D args=dtoverlay -r 1", conflicts)

    def test_from_config_fail_closes_servo_when_startup_preflight_fails(self) -> None:
        config = SimpleNamespace(
            attention_servo_enabled=True,
            attention_servo_driver="lgpio_pwm",
            attention_servo_gpio=18,
            attention_servo_invert_direction=True,
            gpio_chip="gpiochip0",
        )

        with mock.patch.object(
            servo_follow,
            "_default_pulse_writer_for_config",
            side_effect=RuntimeError("conflicting overlay"),
        ):
            controller = AttentionServoController.from_config(cast(TwinrConfig, config))

        decision = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.65,
            confidence=0.9,
        )
        debug = controller.debug_snapshot(observed_at=10.0)

        self.assertFalse(controller.config.enabled)
        self.assertEqual(decision.reason, "faulted")
        self.assertTrue(debug["disabled_due_to_fault"])
        self.assertIn("conflicting overlay", str(debug["fault_reason"]))


class AttentionServoControllerTests(unittest.TestCase):
    def _build_controller(
        self,
        *,
        invert_direction: bool = False,
        target_hold_s: float = 1.1,
        loss_extrapolation_s: float = 0.8,
        loss_extrapolation_gain: float = 0.65,
        max_step_us: int = 45,
        target_smoothing_s: float = 0.0,
        max_velocity_us_per_s: float = 100000.0,
        max_acceleration_us_per_s2: float = 100000.0,
        max_jerk_us_per_s3: float = 1000000.0,
        rest_max_velocity_us_per_s: float = 35.0,
        rest_max_acceleration_us_per_s2: float = 120.0,
        rest_max_jerk_us_per_s3: float = 450.0,
        min_command_delta_us: int = 1,
        visible_retarget_tolerance_us: int = 0,
        reference_interval_s: float = 0.2,
        soft_limit_margin_us: int = 70,
        idle_release_s: float = 1.0,
        settled_release_s: float = 0.0,
        follow_exit_only: bool = False,
        visible_recenter_interval_s: float = 30.0,
        visible_recenter_center_tolerance: float = 0.12,
        mechanical_range_degrees: float = 270.0,
        exit_follow_max_degrees: float = 60.0,
        exit_activation_delay_s: float = 0.75,
        exit_settle_hold_s: float = 0.6,
        exit_reacquire_center_tolerance: float = 0.08,
        exit_visible_edge_threshold: float = 0.74,
        exit_visible_box_edge_threshold: float = 0.92,
        exit_cooldown_s: float = 30.0,
        control_mode: str = "position",
        continuous_max_speed_degrees_per_s: float = 120.0,
        continuous_slow_zone_degrees: float = 45.0,
        continuous_stop_tolerance_degrees: float = 4.0,
        continuous_min_speed_pulse_delta_us: int = 70,
        continuous_max_speed_pulse_delta_us: int = 160,
    ) -> tuple[AttentionServoController, FakeServoPulseWriter]:
        writer = FakeServoPulseWriter()
        controller = AttentionServoController(
            config=AttentionServoConfig(
                enabled=True,
                driver="lgpio",
                control_mode=control_mode,
                gpio_chip="gpiochip0",
                gpio=18,
                invert_direction=invert_direction,
                target_hold_s=target_hold_s,
                loss_extrapolation_s=loss_extrapolation_s,
                loss_extrapolation_gain=loss_extrapolation_gain,
                min_confidence=0.58,
                deadband=0.045,
                min_pulse_width_us=1050,
                center_pulse_width_us=1500,
                max_pulse_width_us=1950,
                max_step_us=max_step_us,
                target_smoothing_s=target_smoothing_s,
                max_velocity_us_per_s=max_velocity_us_per_s,
                max_acceleration_us_per_s2=max_acceleration_us_per_s2,
                max_jerk_us_per_s3=max_jerk_us_per_s3,
                rest_max_velocity_us_per_s=rest_max_velocity_us_per_s,
                rest_max_acceleration_us_per_s2=rest_max_acceleration_us_per_s2,
                rest_max_jerk_us_per_s3=rest_max_jerk_us_per_s3,
                min_command_delta_us=min_command_delta_us,
                visible_retarget_tolerance_us=visible_retarget_tolerance_us,
                reference_interval_s=reference_interval_s,
                soft_limit_margin_us=soft_limit_margin_us,
                idle_release_s=idle_release_s,
                settled_release_s=settled_release_s,
                follow_exit_only=follow_exit_only,
                visible_recenter_interval_s=visible_recenter_interval_s,
                visible_recenter_center_tolerance=visible_recenter_center_tolerance,
                mechanical_range_degrees=mechanical_range_degrees,
                exit_follow_max_degrees=exit_follow_max_degrees,
                exit_activation_delay_s=exit_activation_delay_s,
                exit_settle_hold_s=exit_settle_hold_s,
                exit_reacquire_center_tolerance=exit_reacquire_center_tolerance,
                exit_visible_edge_threshold=exit_visible_edge_threshold,
                exit_visible_box_edge_threshold=exit_visible_box_edge_threshold,
                exit_cooldown_s=exit_cooldown_s,
                continuous_max_speed_degrees_per_s=continuous_max_speed_degrees_per_s,
                continuous_slow_zone_degrees=continuous_slow_zone_degrees,
                continuous_stop_tolerance_degrees=continuous_stop_tolerance_degrees,
                continuous_min_speed_pulse_delta_us=continuous_min_speed_pulse_delta_us,
                continuous_max_speed_pulse_delta_us=continuous_max_speed_pulse_delta_us,
            ),
            pulse_writer=writer,
        )
        return controller, writer

    def test_update_moves_toward_active_target_with_rate_limit(self) -> None:
        controller, writer = self._build_controller(max_step_us=40)

        decision = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.9,
            confidence=0.9,
        )

        self.assertEqual(decision.reason, "following_target")
        self.assertTrue(decision.active)
        self.assertEqual(decision.target_pulse_width_us, 1804)
        self.assertEqual(decision.commanded_pulse_width_us, 1540)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1540)])

    def test_follow_exit_only_clamps_three_sixty_degree_servo_to_ninety_degree_offset(self) -> None:
        controller, _ = self._build_controller(
            follow_exit_only=True,
            mechanical_range_degrees=360.0,
            exit_follow_max_degrees=90.0,
        )

        self.assertEqual(controller._pulse_width_for_center_x(1.0), 1690)
        self.assertEqual(controller._pulse_width_for_center_x(0.0), 1310)

    def test_continuous_rotation_mode_uses_bounded_speed_commands_for_three_sixty_servo(self) -> None:
        controller, writer = self._build_controller(
            control_mode="continuous_rotation",
            follow_exit_only=False,
            mechanical_range_degrees=360.0,
            max_step_us=250,
            min_command_delta_us=1,
        )

        left = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=1.0,
            confidence=0.95,
        )
        right = controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=0.0,
            confidence=0.95,
        )

        self.assertEqual(left.target_pulse_width_us, 1660)
        self.assertEqual(left.commanded_pulse_width_us, 1660)
        self.assertEqual(right.target_pulse_width_us, 1340)
        self.assertEqual(right.commanded_pulse_width_us, 1410)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1660), ("gpiochip0", 18, 1410)])

    def test_continuous_rotation_mode_recenters_after_virtual_heading_reaches_target(self) -> None:
        controller, writer = self._build_controller(
            control_mode="continuous_rotation",
            follow_exit_only=False,
            mechanical_range_degrees=360.0,
            max_step_us=250,
            min_command_delta_us=1,
            continuous_max_speed_degrees_per_s=180.0,
            continuous_slow_zone_degrees=35.0,
            continuous_stop_tolerance_degrees=5.0,
            continuous_min_speed_pulse_delta_us=70,
            continuous_max_speed_pulse_delta_us=160,
        )

        decisions = [
            controller.update(
                observed_at=10.0 + (0.2 * step),
                active=True,
                target_center_x=0.75,
                confidence=0.95,
            )
            for step in range(7)
        ]
        debug = controller.debug_snapshot(observed_at=11.4)

        self.assertEqual(decisions[0].commanded_pulse_width_us, 1660)
        self.assertEqual(decisions[-1].target_pulse_width_us, 1500)
        self.assertEqual(decisions[-1].commanded_pulse_width_us, 1500)
        self.assertEqual(writer.writes[-1], ("gpiochip0", 18, 1500))
        self.assertIsNotNone(debug["continuous_estimated_heading_degrees"])
        self.assertEqual(debug["config"]["control_mode"], "continuous_rotation")

    def test_pololu_fault_recovers_on_later_update_when_writer_probe_succeeds(self) -> None:
        writer = RecoveringFakeServoPulseWriter()
        controller = AttentionServoController(
            config=AttentionServoConfig(
                enabled=True,
                driver="pololu_maestro",
                control_mode="continuous_rotation",
                gpio_chip="gpiochip0",
                gpio=0,
                follow_exit_only=False,
                max_step_us=250,
                min_command_delta_us=1,
                mechanical_range_degrees=360.0,
            ),
            pulse_writer=writer,
        )

        with self.assertRaisesRegex(RuntimeError, "maestro command port missing"):
            controller.update(
                observed_at=10.0,
                active=True,
                target_center_x=1.0,
                confidence=0.95,
            )

        decision = controller.update(
            observed_at=11.5,
            active=True,
            target_center_x=1.0,
            confidence=0.95,
        )

        self.assertEqual(writer.probes, [0])
        self.assertTrue(writer.closed)
        self.assertEqual(decision.reason, "following_target")
        self.assertIsNone(controller.debug_snapshot(observed_at=11.5)["fault_reason"])
        self.assertEqual(writer.writes, [("gpiochip0", 0, decision.commanded_pulse_width_us or 0)])
        self.assertIsNotNone(decision.commanded_pulse_width_us)
        self.assertGreater(decision.commanded_pulse_width_us or 0, 1500)

    def test_disabled_controller_releases_stale_output_on_startup(self) -> None:
        writer = FakeServoPulseWriter()

        AttentionServoController(
            config=AttentionServoConfig(
                enabled=False,
                driver="twinr_kernel",
                gpio_chip="gpiochip0",
                gpio=18,
            ),
            pulse_writer=writer,
        )

        self.assertEqual(writer.disables, [("gpiochip0", 18)])

    def test_from_config_uses_dedicated_maestro_channel_for_pololu_driver(self) -> None:
        config = TwinrConfig(
            attention_servo_enabled=True,
            attention_servo_driver="pololu_maestro",
            attention_servo_maestro_device="/dev/serial/by-id/maestro-if00",
            attention_servo_maestro_channel=0,
            attention_servo_gpio=18,
        )

        servo_config = AttentionServoConfig.from_config(cast(TwinrConfig, config))

        self.assertEqual(servo_config.driver, "pololu_maestro")
        self.assertEqual(servo_config.maestro_device, "/dev/serial/by-id/maestro-if00")
        self.assertEqual(servo_config.gpio, 0)

    def test_from_config_uses_dedicated_maestro_channel_for_peer_pololu_driver(self) -> None:
        config = TwinrConfig(
            attention_servo_enabled=True,
            attention_servo_driver="peer_pololu_maestro",
            attention_servo_peer_base_url="http://10.42.0.2:8768/",
            attention_servo_peer_timeout_s=2.0,
            attention_servo_maestro_channel=1,
            attention_servo_gpio=18,
        )

        servo_config = AttentionServoConfig.from_config(cast(TwinrConfig, config))

        self.assertEqual(servo_config.driver, "peer_pololu_maestro")
        self.assertEqual(servo_config.peer_base_url, "http://10.42.0.2:8768")
        self.assertEqual(servo_config.peer_timeout_s, 2.0)
        self.assertEqual(servo_config.gpio, 1)

    def test_from_config_reads_continuous_rotation_servo_tuning(self) -> None:
        config = TwinrConfig(
            attention_servo_enabled=True,
            attention_servo_control_mode="continuous_rotation",
            attention_servo_continuous_max_speed_degrees_per_s=140.0,
            attention_servo_continuous_slow_zone_degrees=28.0,
            attention_servo_continuous_stop_tolerance_degrees=3.5,
            attention_servo_continuous_min_speed_pulse_delta_us=72,
            attention_servo_continuous_max_speed_pulse_delta_us=165,
        )

        servo_config = AttentionServoConfig.from_config(cast(TwinrConfig, config))

        self.assertEqual(servo_config.control_mode, "continuous_rotation")
        self.assertTrue(servo_config.uses_continuous_rotation)
        self.assertEqual(servo_config.continuous_max_speed_degrees_per_s, 140.0)
        self.assertEqual(servo_config.continuous_slow_zone_degrees, 28.0)
        self.assertEqual(servo_config.continuous_stop_tolerance_degrees, 3.5)
        self.assertEqual(servo_config.continuous_min_speed_pulse_delta_us, 72)
        self.assertEqual(servo_config.continuous_max_speed_pulse_delta_us, 165)

    def test_exit_only_recentering_starts_from_writer_seed_instead_of_blind_center(self) -> None:
        writer = FakeServoPulseWriter()
        writer.current_pulse_width_us_value = 1337
        controller = AttentionServoController(
            config=AttentionServoConfig(
                enabled=True,
                driver="twinr_kernel",
                gpio_chip="gpiochip0",
                gpio=18,
                follow_exit_only=True,
                max_velocity_us_per_s=100000.0,
                max_acceleration_us_per_s2=100000.0,
                max_jerk_us_per_s3=1000000.0,
                rest_max_velocity_us_per_s=35.0,
                rest_max_acceleration_us_per_s2=120.0,
                rest_max_jerk_us_per_s3=450.0,
                min_command_delta_us=1,
            ),
            pulse_writer=writer,
        )

        decisions = [
            controller.update(
                observed_at=10.0 + (0.2 * step),
                active=False,
                target_center_x=None,
                confidence=0.0,
            )
            for step in range(5)
        ]

        self.assertEqual(decisions[0].reason, "recentering")
        self.assertIsNotNone(decisions[0].commanded_pulse_width_us)
        self.assertGreater(decisions[0].commanded_pulse_width_us or 0, 1337)
        self.assertLess(decisions[0].commanded_pulse_width_us or 0, 1500)
        self.assertTrue(all(decision.reason == "recentering" for decision in decisions))
        self.assertGreater(len(writer.writes), 1)
        self.assertLess(writer.writes[-1][2], 1500)

    def test_exit_only_visible_target_first_clears_stale_startup_offset_before_waiting(self) -> None:
        writer = FakeServoPulseWriter()
        writer.current_pulse_width_us_value = 1331
        controller = AttentionServoController(
            config=AttentionServoConfig(
                enabled=True,
                driver="twinr_kernel",
                gpio_chip="gpiochip0",
                gpio=18,
                follow_exit_only=True,
                max_velocity_us_per_s=100000.0,
                max_acceleration_us_per_s2=100000.0,
                max_jerk_us_per_s3=1000000.0,
                rest_max_velocity_us_per_s=35.0,
                rest_max_acceleration_us_per_s2=120.0,
                rest_max_jerk_us_per_s3=450.0,
                min_command_delta_us=1,
            ),
            pulse_writer=writer,
        )

        decision = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.62,
            confidence=0.95,
            visible_target_present=True,
        )

        self.assertEqual(decision.reason, "startup_recentering")
        self.assertIsNotNone(decision.commanded_pulse_width_us)
        self.assertGreater(decision.commanded_pulse_width_us or 0, 1331)
        self.assertLess(decision.commanded_pulse_width_us or 0, 1500)

    def test_exit_only_visible_target_recenters_near_center_stale_kernel_seed_before_waiting(self) -> None:
        writer = FakeServoPulseWriter()
        writer.current_pulse_width_us_value = 1476
        controller = AttentionServoController(
            config=AttentionServoConfig(
                enabled=True,
                driver="twinr_kernel",
                gpio_chip="gpiochip0",
                gpio=18,
                follow_exit_only=True,
                max_velocity_us_per_s=100000.0,
                max_acceleration_us_per_s2=100000.0,
                max_jerk_us_per_s3=1000000.0,
                rest_max_velocity_us_per_s=35.0,
                rest_max_acceleration_us_per_s2=120.0,
                rest_max_jerk_us_per_s3=450.0,
                min_command_delta_us=10,
                max_step_us=20,
            ),
            pulse_writer=writer,
        )

        decision = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.62,
            confidence=0.95,
            visible_target_present=True,
        )

        self.assertEqual(decision.reason, "startup_recentering")
        self.assertEqual(decision.commanded_pulse_width_us, 1500)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1500)])

    def test_update_holds_recent_target_then_recenters(self) -> None:
        controller, writer = self._build_controller(
            target_hold_s=1.0,
            loss_extrapolation_s=0.0,
            max_step_us=40,
        )

        first = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.85,
            confidence=0.9,
        )
        held = controller.update(
            observed_at=10.4,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        recentered = controller.update(
            observed_at=11.5,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )

        self.assertEqual(first.reason, "following_target")
        self.assertEqual(held.reason, "holding_recent_target")
        self.assertTrue(held.active)
        self.assertEqual(held.applied_center_x, first.applied_center_x)
        self.assertEqual(recentered.reason, "recentering")
        self.assertFalse(recentered.active)
        self.assertEqual(recentered.applied_center_x, 0.5)
        self.assertGreater(len(writer.writes), 1)
        self.assertLess(recentered.commanded_pulse_width_us or 0, held.commanded_pulse_width_us or 0)

    def test_update_projects_recent_target_trajectory_on_loss(self) -> None:
        controller, writer = self._build_controller(
            target_hold_s=1.2,
            loss_extrapolation_s=0.8,
            loss_extrapolation_gain=1.0,
            max_step_us=400,
            target_smoothing_s=0.0,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.6,
            confidence=0.95,
        )
        controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=0.8,
            confidence=0.95,
        )
        projected = controller.update(
            observed_at=10.4,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )

        self.assertEqual(projected.reason, "projecting_recent_trajectory")
        self.assertTrue(projected.active)
        self.assertGreater(projected.applied_center_x or 0.0, 0.8)
        self.assertGreater(projected.commanded_pulse_width_us or 0, 1728)
        self.assertEqual(writer.writes[-1][2], projected.commanded_pulse_width_us)

    def test_update_holds_projected_exit_point_before_recentering(self) -> None:
        controller, writer = self._build_controller(
            target_hold_s=1.2,
            loss_extrapolation_s=0.6,
            loss_extrapolation_gain=1.0,
            max_step_us=400,
            target_smoothing_s=0.0,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.6,
            confidence=0.95,
        )
        controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=0.8,
            confidence=0.95,
        )
        held_projection = controller.update(
            observed_at=10.9,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        recentered = controller.update(
            observed_at=11.5,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )

        self.assertEqual(held_projection.reason, "holding_projected_trajectory")
        self.assertTrue(held_projection.active)
        self.assertGreater(held_projection.applied_center_x or 0.0, 0.8)
        self.assertEqual(recentered.reason, "recentering")
        self.assertFalse(recentered.active)
        self.assertLess(recentered.commanded_pulse_width_us or 0, held_projection.commanded_pulse_width_us or 0)

    def test_update_respects_direction_inversion(self) -> None:
        controller, writer = self._build_controller(invert_direction=True, max_step_us=200)

        decision = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.8,
            confidence=0.9,
        )

        self.assertEqual(decision.target_pulse_width_us, 1272)
        self.assertEqual(decision.commanded_pulse_width_us, 1300)
        self.assertEqual(writer.writes[-1], ("gpiochip0", 18, 1300))

    def test_update_applies_soft_limit_margin_before_mapping(self) -> None:
        controller, writer = self._build_controller(max_step_us=400, soft_limit_margin_us=90)

        decision = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=1.0,
            confidence=0.95,
        )

        self.assertEqual(decision.target_pulse_width_us, 1860)
        self.assertEqual(decision.commanded_pulse_width_us, 1860)
        self.assertEqual(writer.writes[-1], ("gpiochip0", 18, 1860))

    def test_update_smooths_large_target_jump_before_mapping(self) -> None:
        controller, writer = self._build_controller(
            max_step_us=400,
            target_smoothing_s=1.0,
            max_velocity_us_per_s=100000.0,
        )

        decision = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=1.0,
            confidence=0.95,
        )

        self.assertAlmostEqual(decision.applied_center_x or 0.0, 0.6, places=3)
        self.assertEqual(decision.target_pulse_width_us, 1576)
        self.assertEqual(decision.commanded_pulse_width_us, 1576)
        self.assertEqual(writer.writes[-1], ("gpiochip0", 18, 1576))

    def test_update_rate_limits_by_velocity_per_second(self) -> None:
        controller, writer = self._build_controller(
            max_step_us=80,
            target_smoothing_s=0.0,
            max_velocity_us_per_s=60.0,
            reference_interval_s=0.2,
        )

        first = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=1.0,
            confidence=0.95,
        )
        second = controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=1.0,
            confidence=0.95,
        )

        self.assertEqual(first.commanded_pulse_width_us, 1506)
        self.assertEqual(second.commanded_pulse_width_us, 1518)
        self.assertEqual(
            writer.writes,
            [
                ("gpiochip0", 18, 1506),
                ("gpiochip0", 18, 1518),
            ],
        )

    def test_update_limits_initial_motion_by_acceleration_and_jerk(self) -> None:
        controller, writer = self._build_controller(
            max_step_us=400,
            target_smoothing_s=0.0,
            max_velocity_us_per_s=80.0,
            max_acceleration_us_per_s2=220.0,
            max_jerk_us_per_s3=900.0,
            min_command_delta_us=1,
            reference_interval_s=0.2,
        )

        first = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=1.0,
            confidence=0.95,
        )
        second = controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=1.0,
            confidence=0.95,
        )

        self.assertEqual(first.commanded_pulse_width_us, 1504)
        self.assertEqual(second.commanded_pulse_width_us, 1515)
        self.assertEqual(
            writer.writes,
            [
                ("gpiochip0", 18, 1504),
                ("gpiochip0", 18, 1515),
            ],
        )

    def test_update_holds_sub_threshold_commands_until_motion_accumulates(self) -> None:
        controller, writer = self._build_controller(
            max_step_us=400,
            target_smoothing_s=0.0,
            max_velocity_us_per_s=20.0,
            max_acceleration_us_per_s2=100000.0,
            max_jerk_us_per_s3=1000000.0,
            min_command_delta_us=8,
            reference_interval_s=0.2,
        )

        first = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=1.0,
            confidence=0.95,
        )
        second = controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=1.0,
            confidence=0.95,
        )
        third = controller.update(
            observed_at=10.4,
            active=True,
            target_center_x=1.0,
            confidence=0.95,
        )

        self.assertEqual(first.commanded_pulse_width_us, 1502)
        self.assertEqual(second.commanded_pulse_width_us, 1502)
        self.assertEqual(third.commanded_pulse_width_us, 1510)
        self.assertEqual(
            writer.writes,
            [
                ("gpiochip0", 18, 1502),
                ("gpiochip0", 18, 1510),
            ],
        )

    def test_update_holds_visible_target_micro_retargets_within_tolerance(self) -> None:
        controller, writer = self._build_controller(
            max_step_us=400,
            target_smoothing_s=0.0,
            visible_retarget_tolerance_us=50,
        )

        first = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.80,
            confidence=0.95,
        )
        second = controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=0.82,
            confidence=0.95,
        )

        self.assertEqual(first.commanded_pulse_width_us, 1728)
        self.assertEqual(second.target_pulse_width_us, 1728)
        self.assertEqual(second.commanded_pulse_width_us, 1728)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1728)])

    def test_update_allows_loss_projection_after_visible_target_hysteresis(self) -> None:
        controller, writer = self._build_controller(
            target_hold_s=1.2,
            loss_extrapolation_s=0.8,
            loss_extrapolation_gain=1.0,
            max_step_us=400,
            target_smoothing_s=0.0,
            visible_retarget_tolerance_us=50,
        )

        first = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.80,
            confidence=0.95,
        )
        held_visible = controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=0.82,
            confidence=0.95,
        )
        projected = controller.update(
            observed_at=10.4,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )

        self.assertEqual(first.commanded_pulse_width_us, 1728)
        self.assertEqual(held_visible.commanded_pulse_width_us, 1728)
        self.assertEqual(held_visible.reason, "following_target")
        self.assertEqual(projected.reason, "projecting_recent_trajectory")
        self.assertGreater(projected.commanded_pulse_width_us or 0, held_visible.commanded_pulse_width_us or 0)
        self.assertEqual(
            writer.writes,
            [
                ("gpiochip0", 18, 1728),
                ("gpiochip0", 18, projected.commanded_pulse_width_us),
            ],
        )

    def test_update_releases_idle_output_after_recentering(self) -> None:
        controller, writer = self._build_controller(target_hold_s=0.0, max_step_us=300, idle_release_s=0.6)
        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.85,
            confidence=0.9,
        )

        recentering = controller.update(
            observed_at=10.2,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        released = controller.update(
            observed_at=10.9,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        released_confirmed = controller.update(
            observed_at=11.6,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )

        self.assertEqual(recentering.reason, "recentering")
        self.assertEqual(recentering.commanded_pulse_width_us, 1500)
        self.assertEqual(released.reason, "recentering")
        self.assertEqual(released.commanded_pulse_width_us, 1500)
        self.assertEqual(released_confirmed.reason, "idle_released")
        self.assertIsNone(released_confirmed.commanded_pulse_width_us)
        self.assertEqual(writer.disables, [("gpiochip0", 18)])

    def test_recentering_snaps_to_exact_center_before_idle_release(self) -> None:
        writer = FakeServoPulseWriter()
        writer.current_pulse_width_us_value = 1476
        controller = AttentionServoController(
            config=AttentionServoConfig(
                enabled=True,
                driver="twinr_kernel",
                gpio_chip="gpiochip0",
                gpio=18,
                follow_exit_only=True,
                idle_release_s=0.6,
                max_velocity_us_per_s=100000.0,
                max_acceleration_us_per_s2=100000.0,
                max_jerk_us_per_s3=1000000.0,
                rest_max_velocity_us_per_s=35.0,
                rest_max_acceleration_us_per_s2=120.0,
                rest_max_jerk_us_per_s3=450.0,
                min_command_delta_us=10,
                max_step_us=20,
            ),
            pulse_writer=writer,
        )

        first = controller.update(
            observed_at=10.0,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        released = controller.update(
            observed_at=10.7,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        released_confirmed = controller.update(
            observed_at=11.4,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )

        self.assertEqual(first.reason, "recentering")
        self.assertEqual(first.commanded_pulse_width_us, 1500)
        self.assertEqual(released.reason, "recentering")
        self.assertEqual(released.commanded_pulse_width_us, 1500)
        self.assertEqual(released_confirmed.reason, "idle_released")
        self.assertIsNone(released_confirmed.commanded_pulse_width_us)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1500)])
        self.assertEqual(writer.disables, [("gpiochip0", 18)])

    def test_update_releases_stable_off_center_target_after_settle(self) -> None:
        controller, writer = self._build_controller(
            max_step_us=300,
            settled_release_s=0.6,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.85,
            confidence=0.9,
        )
        controller.update(
            observed_at=10.4,
            active=True,
            target_center_x=0.85,
            confidence=0.9,
        )
        released = controller.update(
            observed_at=11.1,
            active=True,
            target_center_x=0.85,
            confidence=0.9,
        )
        held_released = controller.update(
            observed_at=11.3,
            active=True,
            target_center_x=0.85,
            confidence=0.9,
        )

        self.assertEqual(released.reason, "settled_released")
        self.assertIsNone(released.commanded_pulse_width_us)
        self.assertEqual(held_released.reason, "settled_released")
        self.assertIsNone(held_released.commanded_pulse_width_us)
        self.assertEqual(writer.disables, [("gpiochip0", 18)])
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1766)])

    def test_update_resumes_from_released_target_instead_of_center(self) -> None:
        controller, writer = self._build_controller(
            max_step_us=100,
            settled_release_s=0.6,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.85,
            confidence=0.9,
        )
        controller.update(
            observed_at=10.4,
            active=True,
            target_center_x=0.85,
            confidence=0.9,
        )
        controller.update(
            observed_at=11.1,
            active=True,
            target_center_x=0.85,
            confidence=0.9,
        )
        controller.update(
            observed_at=11.8,
            active=True,
            target_center_x=0.85,
            confidence=0.9,
        )
        controller.update(
            observed_at=12.5,
            active=True,
            target_center_x=0.85,
            confidence=0.9,
        )
        resumed = controller.update(
            observed_at=12.7,
            active=True,
            target_center_x=0.2,
            confidence=0.9,
        )

        self.assertEqual(resumed.commanded_pulse_width_us, 1666)
        self.assertEqual(writer.writes[-1], ("gpiochip0", 18, 1666))
        self.assertEqual(writer.disables, [("gpiochip0", 18)])

    def test_exit_only_visible_target_does_not_command_servo(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            max_step_us=400,
            target_smoothing_s=0.0,
        )

        decision = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.9,
            confidence=0.95,
        )

        self.assertEqual(decision.reason, "waiting_for_exit")
        self.assertIsNone(decision.commanded_pulse_width_us)
        self.assertEqual(writer.writes, [])
        self.assertEqual(writer.disables, [])

    def test_exit_only_visible_target_ignores_low_confidence_and_stays_still(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            max_step_us=400,
            target_smoothing_s=0.0,
        )

        decision = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.9,
            confidence=0.15,
        )

        self.assertEqual(decision.reason, "waiting_for_exit")
        self.assertIsNone(decision.commanded_pulse_width_us)
        self.assertEqual(writer.writes, [])
        self.assertEqual(writer.disables, [])

    def test_exit_only_visible_target_waits_for_visible_recenter_interval_before_moving(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            max_step_us=400,
            target_smoothing_s=0.0,
            visible_recenter_interval_s=30.0,
            visible_recenter_center_tolerance=0.12,
        )

        first = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.68,
            confidence=0.95,
        )
        second = controller.update(
            observed_at=39.9,
            active=True,
            target_center_x=0.68,
            confidence=0.95,
        )

        self.assertEqual(first.reason, "waiting_for_exit")
        self.assertEqual(second.reason, "waiting_for_exit")
        self.assertIsNone(second.commanded_pulse_width_us)
        self.assertEqual(writer.writes, [])
        self.assertEqual(writer.disables, [])

    def test_exit_only_visible_target_periodically_recenters_after_interval(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            max_step_us=400,
            target_smoothing_s=0.0,
            visible_recenter_interval_s=30.0,
            visible_recenter_center_tolerance=0.12,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.68,
            confidence=0.95,
        )
        pursuing = controller.update(
            observed_at=40.0,
            active=True,
            target_center_x=0.68,
            confidence=0.95,
        )
        centered = controller.update(
            observed_at=40.2,
            active=True,
            target_center_x=0.56,
            confidence=0.95,
        )

        self.assertEqual(pursuing.reason, "pursuing_visible_recenter")
        self.assertEqual(pursuing.target_pulse_width_us, 1637)
        self.assertEqual(pursuing.commanded_pulse_width_us, 1637)
        self.assertEqual(centered.reason, "waiting_for_exit")
        self.assertIsNone(centered.commanded_pulse_width_us)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1637)])
        self.assertEqual(writer.disables, [("gpiochip0", 18)])

    def test_exit_only_visible_target_recenter_timer_resets_once_user_is_centered(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            max_step_us=400,
            target_smoothing_s=0.0,
            visible_recenter_interval_s=30.0,
            visible_recenter_center_tolerance=0.12,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.68,
            confidence=0.95,
        )
        controller.update(
            observed_at=20.0,
            active=True,
            target_center_x=0.56,
            confidence=0.95,
        )
        retry = controller.update(
            observed_at=49.9,
            active=True,
            target_center_x=0.68,
            confidence=0.95,
        )

        self.assertEqual(retry.reason, "waiting_for_exit")
        self.assertIsNone(retry.commanded_pulse_width_us)
        self.assertEqual(writer.writes, [])
        self.assertEqual(writer.disables, [])

    def test_exit_only_visible_target_recenter_releases_after_settle_when_still_off_center(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            max_step_us=400,
            target_smoothing_s=0.0,
            visible_recenter_interval_s=0.0,
            visible_recenter_center_tolerance=0.12,
            settled_release_s=0.3,
        )

        pursuing = controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.68,
            confidence=0.95,
        )
        released = controller.update(
            observed_at=10.4,
            active=True,
            target_center_x=0.68,
            confidence=0.95,
        )

        self.assertEqual(pursuing.reason, "pursuing_visible_recenter")
        self.assertEqual(pursuing.commanded_pulse_width_us, 1637)
        self.assertEqual(released.reason, "waiting_for_exit")
        self.assertIsNone(released.commanded_pulse_width_us)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1637)])
        self.assertEqual(writer.disables, [("gpiochip0", 18)])

    def test_exit_only_visible_edge_departure_can_start_monotone_pursuit_without_full_loss(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            target_hold_s=1.2,
            max_step_us=400,
            target_smoothing_s=0.0,
            exit_activation_delay_s=0.4,
            exit_visible_edge_threshold=0.62,
            exit_visible_box_edge_threshold=0.92,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.63,
            confidence=0.95,
            visible_target_present=True,
            visible_target_box_left=0.84,
            visible_target_box_right=0.94,
        )
        waiting = controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=0.64,
            confidence=0.95,
            visible_target_present=True,
            visible_target_box_left=0.85,
            visible_target_box_right=0.95,
        )
        pursuing = controller.update(
            observed_at=10.5,
            active=True,
            target_center_x=0.64,
            confidence=0.95,
            visible_target_present=True,
            visible_target_box_left=0.85,
            visible_target_box_right=0.95,
        )

        self.assertEqual(waiting.reason, "waiting_for_exit")
        self.assertEqual(pursuing.reason, "pursuing_edge_departure")
        self.assertEqual(pursuing.commanded_pulse_width_us, 1669)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1669)])

    def test_exit_only_recent_side_departure_anchor_can_start_pursuit_after_target_snaps_inward(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            target_hold_s=1.2,
            max_step_us=400,
            target_smoothing_s=0.0,
            exit_activation_delay_s=0.4,
            exit_visible_edge_threshold=0.62,
            exit_visible_box_edge_threshold=0.92,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.66,
            confidence=0.95,
            visible_target_present=True,
            visible_target_box_left=0.84,
            visible_target_box_right=0.95,
        )
        waiting = controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=0.54,
            confidence=0.95,
            visible_target_present=True,
            visible_target_box_left=0.83,
            visible_target_box_right=0.94,
        )
        pursuing = controller.update(
            observed_at=10.5,
            active=True,
            target_center_x=0.54,
            confidence=0.95,
            visible_target_present=True,
            visible_target_box_left=0.83,
            visible_target_box_right=0.94,
        )

        self.assertEqual(waiting.reason, "waiting_for_exit")
        self.assertEqual(pursuing.reason, "pursuing_edge_departure")

    def test_exit_only_visible_edge_departure_waits_for_real_frame_edge_geometry(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            target_hold_s=1.2,
            max_step_us=400,
            target_smoothing_s=0.0,
            exit_activation_delay_s=0.4,
            exit_visible_edge_threshold=0.62,
            exit_visible_box_edge_threshold=0.92,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.63,
            confidence=0.95,
            visible_target_present=True,
            visible_target_box_left=0.58,
            visible_target_box_right=0.80,
        )
        waiting = controller.update(
            observed_at=10.5,
            active=True,
            target_center_x=0.64,
            confidence=0.95,
            visible_target_present=True,
            visible_target_box_left=0.59,
            visible_target_box_right=0.81,
        )

        self.assertEqual(waiting.reason, "waiting_for_exit")
        self.assertEqual(writer.writes, [])

    def test_exit_only_waits_for_loss_confirmation_before_projecting(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            target_hold_s=1.2,
            loss_extrapolation_s=0.8,
            loss_extrapolation_gain=1.0,
            max_step_us=400,
            target_smoothing_s=0.0,
            exit_activation_delay_s=0.5,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.6,
            confidence=0.95,
        )
        controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=0.8,
            confidence=0.95,
        )
        waiting = controller.update(
            observed_at=10.4,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        projected = controller.update(
            observed_at=10.8,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )

        self.assertEqual(waiting.reason, "awaiting_exit_confirmation")
        self.assertIsNone(waiting.commanded_pulse_width_us)
        self.assertEqual(projected.reason, "pursuing_exit_direction")
        self.assertEqual(projected.commanded_pulse_width_us, 1669)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1669)])

    def test_exit_only_uses_explicit_camera_visibility_instead_of_active_hold_state(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            target_hold_s=1.2,
            loss_extrapolation_s=0.8,
            loss_extrapolation_gain=1.0,
            max_step_us=400,
            target_smoothing_s=0.0,
            exit_activation_delay_s=0.5,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.6,
            confidence=0.95,
            visible_target_present=True,
        )
        controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=0.8,
            confidence=0.95,
            visible_target_present=True,
        )
        waiting = controller.update(
            observed_at=10.4,
            active=True,
            target_center_x=0.8,
            confidence=0.95,
            visible_target_present=False,
        )
        projected = controller.update(
            observed_at=10.8,
            active=True,
            target_center_x=0.8,
            confidence=0.95,
            visible_target_present=False,
        )

        self.assertEqual(waiting.reason, "awaiting_exit_confirmation")
        self.assertIsNone(waiting.commanded_pulse_width_us)
        self.assertEqual(projected.reason, "pursuing_exit_direction")
        self.assertEqual(projected.commanded_pulse_width_us, 1669)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1669)])

    def test_exit_only_projects_after_visibility_loss_and_clamps_to_sixty_degrees(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            target_hold_s=1.2,
            loss_extrapolation_s=0.8,
            loss_extrapolation_gain=1.0,
            max_step_us=400,
            target_smoothing_s=0.0,
            exit_activation_delay_s=0.0,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.6,
            confidence=0.95,
        )
        controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=0.8,
            confidence=0.95,
        )
        projected = controller.update(
            observed_at=10.4,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )

        self.assertEqual(projected.reason, "pursuing_exit_direction")
        self.assertEqual(projected.target_pulse_width_us, 1669)
        self.assertEqual(projected.commanded_pulse_width_us, 1669)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1669)])

    def test_exit_only_prefers_recent_outward_anchor_over_last_inward_visible_sample(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            target_hold_s=1.2,
            loss_extrapolation_s=0.8,
            loss_extrapolation_gain=0.0,
            max_step_us=400,
            target_smoothing_s=0.0,
            exit_activation_delay_s=0.0,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.94,
            confidence=0.95,
        )
        controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=0.64,
            confidence=0.95,
        )
        projected = controller.update(
            observed_at=10.4,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )

        self.assertEqual(projected.reason, "pursuing_exit_direction")
        self.assertEqual(projected.target_pulse_width_us, 1669)
        self.assertEqual(projected.commanded_pulse_width_us, 1669)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1669)])

    def test_exit_only_visible_reentry_starts_cooldown_while_exit_motion_is_still_in_flight(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            target_hold_s=1.2,
            loss_extrapolation_s=0.8,
            loss_extrapolation_gain=1.0,
            max_step_us=80,
            target_smoothing_s=0.0,
            exit_activation_delay_s=0.0,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.6,
            confidence=0.95,
        )
        controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=0.8,
            confidence=0.95,
        )
        projected = controller.update(
            observed_at=10.4,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        returned_visible = controller.update(
            observed_at=10.5,
            active=True,
            target_center_x=0.52,
            confidence=0.12,
        )

        self.assertEqual(projected.reason, "pursuing_exit_direction")
        self.assertEqual(returned_visible.reason, "reacquired_visible_cooldown")
        self.assertEqual(projected.commanded_pulse_width_us, 1580)
        self.assertIsNone(returned_visible.commanded_pulse_width_us)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1580)])
        self.assertEqual(writer.disables, [("gpiochip0", 18)])

    def test_exit_only_visible_reentry_after_exit_limit_stays_still_even_with_low_confidence(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            target_hold_s=1.2,
            loss_extrapolation_s=0.8,
            loss_extrapolation_gain=1.0,
            max_step_us=400,
            target_smoothing_s=0.0,
            exit_activation_delay_s=0.0,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.6,
            confidence=0.95,
        )
        controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=0.8,
            confidence=0.95,
        )
        controller.update(
            observed_at=10.4,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        settled = controller.update(
            observed_at=11.1,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        returned_visible = controller.update(
            observed_at=11.2,
            active=True,
            target_center_x=0.52,
            confidence=0.12,
        )

        self.assertEqual(settled.reason, "holding_exit_limit")
        self.assertIsNone(settled.commanded_pulse_width_us)
        self.assertEqual(returned_visible.reason, "reacquired_visible_cooldown")
        self.assertIsNone(returned_visible.commanded_pulse_width_us)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1669)])
        self.assertEqual(writer.disables, [("gpiochip0", 18)])

    def test_exit_only_holds_output_briefly_after_reaching_exit_limit_before_release(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            target_hold_s=1.2,
            loss_extrapolation_s=0.8,
            loss_extrapolation_gain=1.0,
            max_step_us=400,
            target_smoothing_s=0.0,
            exit_activation_delay_s=0.0,
            exit_settle_hold_s=0.6,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.6,
            confidence=0.95,
        )
        pursuing = controller.update(
            observed_at=10.2,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        held = controller.update(
            observed_at=10.6,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        released = controller.update(
            observed_at=10.9,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )

        self.assertEqual(pursuing.reason, "pursuing_exit_direction")
        self.assertEqual(held.reason, "pursuing_exit_direction")
        self.assertEqual(held.commanded_pulse_width_us, 1669)
        self.assertEqual(released.reason, "holding_exit_limit")
        self.assertIsNone(released.commanded_pulse_width_us)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1669)])
        self.assertEqual(writer.disables, [("gpiochip0", 18)])

    def test_exit_only_visible_edge_reentry_keeps_pursuing_same_direction_until_centered(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            target_hold_s=1.2,
            max_step_us=400,
            target_smoothing_s=0.0,
            exit_activation_delay_s=0.0,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.85,
            confidence=0.95,
        )
        pursuing = controller.update(
            observed_at=10.2,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        still_off_center = controller.update(
            observed_at=10.3,
            active=True,
            target_center_x=0.78,
            confidence=0.95,
        )

        self.assertEqual(pursuing.reason, "pursuing_exit_direction")
        self.assertEqual(still_off_center.reason, "pursuing_edge_departure")
        self.assertEqual(still_off_center.commanded_pulse_width_us, 1669)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1669)])
        self.assertEqual(writer.disables, [])

    def test_exit_only_visible_reacquire_starts_cooldown_without_reversing(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            target_hold_s=1.2,
            max_step_us=80,
            target_smoothing_s=0.0,
            exit_activation_delay_s=0.0,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.85,
            confidence=0.95,
        )
        pursuing = controller.update(
            observed_at=10.2,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        visible_again = controller.update(
            observed_at=10.3,
            active=True,
            target_center_x=0.52,
            confidence=0.95,
        )

        self.assertEqual(pursuing.reason, "pursuing_exit_direction")
        self.assertEqual(pursuing.commanded_pulse_width_us, 1580)
        self.assertEqual(visible_again.reason, "reacquired_visible_cooldown")
        self.assertIsNone(visible_again.commanded_pulse_width_us)
        self.assertEqual(writer.disables, [("gpiochip0", 18)])

    def test_exit_only_cooldown_blocks_new_motion_then_recenters_after_absence(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            target_hold_s=1.2,
            max_step_us=400,
            target_smoothing_s=0.0,
            exit_activation_delay_s=0.0,
            exit_cooldown_s=30.0,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.85,
            confidence=0.95,
        )
        controller.update(
            observed_at=10.2,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        controller.update(
            observed_at=10.4,
            active=True,
            target_center_x=0.52,
            confidence=0.95,
        )
        cooldown = controller.update(
            observed_at=15.0,
            active=True,
            target_center_x=0.9,
            confidence=0.95,
        )
        recentered = controller.update(
            observed_at=40.6,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )

        self.assertEqual(cooldown.reason, "exit_cooldown")
        self.assertIsNone(cooldown.commanded_pulse_width_us)
        self.assertEqual(recentered.reason, "recentering")
        self.assertIsNotNone(recentered.commanded_pulse_width_us)
        self.assertGreater(recentered.commanded_pulse_width_us or 0, 1500)
        self.assertLess(recentered.commanded_pulse_width_us or 0, 1669)
        self.assertEqual(writer.writes[-1], ("gpiochip0", 18, recentered.commanded_pulse_width_us or 0))

    def test_exit_only_holds_exit_limit_without_new_impulses_until_visible_reacquire_or_rest_return(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            target_hold_s=1.2,
            loss_extrapolation_s=0.2,
            loss_extrapolation_gain=1.0,
            max_step_us=400,
            target_smoothing_s=0.0,
            exit_activation_delay_s=0.0,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.75,
            confidence=0.95,
        )
        controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=0.9,
            confidence=0.95,
        )
        controller.update(
            observed_at=10.4,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        released = controller.update(
            observed_at=11.1,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        held_visible = controller.update(
            observed_at=11.2,
            active=True,
            target_center_x=0.72,
            confidence=0.95,
        )

        self.assertEqual(released.reason, "holding_exit_limit")
        self.assertIsNone(released.commanded_pulse_width_us)
        self.assertEqual(held_visible.reason, "holding_exit_limit")
        self.assertIsNone(held_visible.commanded_pulse_width_us)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1669)])
        self.assertEqual(writer.disables, [("gpiochip0", 18)])

    def test_exit_only_centered_visible_reentry_after_exit_limit_starts_cooldown(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            target_hold_s=1.2,
            loss_extrapolation_s=0.2,
            loss_extrapolation_gain=1.0,
            max_step_us=400,
            target_smoothing_s=0.0,
            exit_activation_delay_s=0.0,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.75,
            confidence=0.95,
        )
        controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=0.9,
            confidence=0.95,
        )
        controller.update(
            observed_at=10.4,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        controller.update(
            observed_at=11.1,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        centered_visible = controller.update(
            observed_at=11.2,
            active=True,
            target_center_x=0.51,
            confidence=0.95,
        )

        self.assertEqual(centered_visible.reason, "reacquired_visible_cooldown")
        self.assertIsNone(centered_visible.commanded_pulse_width_us)
        self.assertEqual(writer.writes, [("gpiochip0", 18, 1669)])
        self.assertEqual(writer.disables, [("gpiochip0", 18)])

    def test_exit_only_recenters_after_long_absence_from_exit_limit(self) -> None:
        controller, writer = self._build_controller(
            follow_exit_only=True,
            target_hold_s=1.2,
            loss_extrapolation_s=0.2,
            loss_extrapolation_gain=1.0,
            max_step_us=400,
            target_smoothing_s=0.0,
            exit_activation_delay_s=0.0,
            exit_cooldown_s=30.0,
        )

        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.75,
            confidence=0.95,
        )
        controller.update(
            observed_at=10.2,
            active=True,
            target_center_x=0.9,
            confidence=0.95,
        )
        controller.update(
            observed_at=10.4,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        controller.update(
            observed_at=10.7,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        controller.update(
            observed_at=11.1,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )
        recentered = controller.update(
            observed_at=40.3,
            active=False,
            target_center_x=None,
            confidence=0.0,
        )

        self.assertEqual(recentered.reason, "recentering")
        self.assertIsNotNone(recentered.commanded_pulse_width_us)
        self.assertLess(recentered.commanded_pulse_width_us or 0, 1669)
        self.assertGreater(recentered.commanded_pulse_width_us or 0, 1500)
        self.assertEqual(
            writer.writes,
            [("gpiochip0", 18, 1669), ("gpiochip0", 18, recentered.commanded_pulse_width_us or 0)],
        )
        self.assertEqual(writer.disables, [("gpiochip0", 18)])

    def test_close_disables_active_output(self) -> None:
        controller, writer = self._build_controller()
        controller.update(
            observed_at=10.0,
            active=True,
            target_center_x=0.8,
            confidence=0.9,
        )

        controller.close()

        self.assertEqual(writer.disables, [("gpiochip0", 18)])
        self.assertTrue(writer.closed)


if __name__ == "__main__":
    unittest.main()
