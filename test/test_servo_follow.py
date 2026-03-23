from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware import servo_follow
from twinr.hardware.servo_follow import (
    AttentionServoConfig,
    AttentionServoController,
    LGPIOPWMServoPulseWriter,
    LGPIOServoPulseWriter,
    PigpioServoPulseWriter,
    SysfsPWMServoPulseWriter,
)


class FakeServoPulseWriter:
    def __init__(self) -> None:
        self.writes: list[tuple[str, int, int]] = []
        self.disables: list[tuple[str, int]] = []
        self.closed = False

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        self.writes.append((gpio_chip, gpio, pulse_width_us))

    def disable(self, *, gpio_chip: str, gpio: int) -> None:
        self.disables.append((gpio_chip, gpio))

    def close(self) -> None:
        self.closed = True


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

        with mock.patch.object(
            servo_follow.SysfsPWMServoPulseWriter,
            "probe",
            return_value=None,
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
            writer._run_pinctrl = mock.Mock(return_value="18, GPIO18, SPI1_CE0, DPI_D14, I2S0_SCLK, PWM0_CHAN2")

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
            writer._run_pinctrl = fake_run_pinctrl

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
        min_command_delta_us: int = 1,
        reference_interval_s: float = 0.2,
        soft_limit_margin_us: int = 70,
        idle_release_s: float = 1.0,
    ) -> tuple[AttentionServoController, FakeServoPulseWriter]:
        writer = FakeServoPulseWriter()
        controller = AttentionServoController(
            config=AttentionServoConfig(
                enabled=True,
                driver="lgpio",
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
                min_command_delta_us=min_command_delta_us,
                reference_interval_s=reference_interval_s,
                soft_limit_margin_us=soft_limit_margin_us,
                idle_release_s=idle_release_s,
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

        self.assertEqual(recentering.reason, "recentering")
        self.assertEqual(recentering.commanded_pulse_width_us, 1500)
        self.assertEqual(released.reason, "idle_released")
        self.assertIsNone(released.commanded_pulse_width_us)
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
