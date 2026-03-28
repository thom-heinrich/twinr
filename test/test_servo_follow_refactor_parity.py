from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, cast
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware import servo_follow_impl
from twinr.hardware.servo_follow import AttentionServoConfig, AttentionServoController
from twinr.hardware.servo_state import AttentionServoRuntimeState, AttentionServoStateStore

_EXPECTED_GOLDEN_DIGESTS = {
    "config": "db1e84faa8c0268376c29f618315333dfb349964f328a12e4a56ef4c37bf2d69",
    "position": "c62bc928bd197cf2d0dcb9fa99252f2ec8d199a22ebfe8ad82101da85f21d256",
    "exit": "32cfc8b2dfa6905c93a7c9c3be05c20a3761787d22c6b1cbf564105c8294980e",
    "continuous": "bbced7797d771936817a8ddbf0025e7b81f7636c6da24e59c0a9fa977a37c618",
}


class FakeServoPulseWriter:
    """Capture servo writer side effects for deterministic controller tests."""

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
        self.current_pulse_width_us_value = None

    def current_pulse_width_us(self, *, gpio_chip: str, gpio: int) -> int | None:
        del gpio_chip, gpio
        return self.current_pulse_width_us_value

    def close(self) -> None:
        self.closed = True


class ServoFollowRefactorGoldenMasterTests(unittest.TestCase):
    """Freeze representative servo-follow behavior before module extraction."""

    def _build_controller(
        self,
        **overrides: Any,
    ) -> tuple[tempfile.TemporaryDirectory[str], AttentionServoController, FakeServoPulseWriter]:
        writer = FakeServoPulseWriter()
        temporary_state_dir = tempfile.TemporaryDirectory()
        resolved_state_path = str(Path(temporary_state_dir.name) / "attention_servo_state.json")
        if "state_path" in overrides and overrides["state_path"]:
            resolved_state_path = str(overrides.pop("state_path"))
        controller = AttentionServoController(
            config=AttentionServoConfig(
                enabled=True,
                driver="lgpio",
                control_mode=str(overrides.pop("control_mode", "position")),
                state_path=resolved_state_path,
                gpio_chip="gpiochip0",
                gpio=18,
                invert_direction=bool(overrides.pop("invert_direction", False)),
                target_hold_s=float(overrides.pop("target_hold_s", 1.1)),
                loss_extrapolation_s=float(overrides.pop("loss_extrapolation_s", 0.8)),
                loss_extrapolation_gain=float(overrides.pop("loss_extrapolation_gain", 0.65)),
                min_confidence=0.58,
                hold_min_confidence=float(overrides.pop("hold_min_confidence", 0.58)),
                deadband=0.045,
                min_pulse_width_us=1050,
                center_pulse_width_us=1500,
                max_pulse_width_us=1950,
                max_step_us=int(overrides.pop("max_step_us", 45)),
                target_smoothing_s=float(overrides.pop("target_smoothing_s", 0.0)),
                max_velocity_us_per_s=float(overrides.pop("max_velocity_us_per_s", 100000.0)),
                max_acceleration_us_per_s2=float(
                    overrides.pop("max_acceleration_us_per_s2", 100000.0)
                ),
                max_jerk_us_per_s3=float(overrides.pop("max_jerk_us_per_s3", 1000000.0)),
                rest_max_velocity_us_per_s=float(
                    overrides.pop("rest_max_velocity_us_per_s", 35.0)
                ),
                rest_max_acceleration_us_per_s2=float(
                    overrides.pop("rest_max_acceleration_us_per_s2", 120.0)
                ),
                rest_max_jerk_us_per_s3=float(overrides.pop("rest_max_jerk_us_per_s3", 450.0)),
                min_command_delta_us=int(overrides.pop("min_command_delta_us", 1)),
                visible_retarget_tolerance_us=int(
                    overrides.pop("visible_retarget_tolerance_us", 0)
                ),
                reference_interval_s=float(overrides.pop("reference_interval_s", 0.2)),
                soft_limit_margin_us=int(overrides.pop("soft_limit_margin_us", 70)),
                idle_release_s=float(overrides.pop("idle_release_s", 1.0)),
                settled_release_s=float(overrides.pop("settled_release_s", 0.0)),
                follow_exit_only=bool(overrides.pop("follow_exit_only", False)),
                visible_recenter_interval_s=float(
                    overrides.pop("visible_recenter_interval_s", 30.0)
                ),
                visible_recenter_center_tolerance=float(
                    overrides.pop("visible_recenter_center_tolerance", 0.12)
                ),
                mechanical_range_degrees=float(
                    overrides.pop("mechanical_range_degrees", 270.0)
                ),
                exit_follow_max_degrees=float(overrides.pop("exit_follow_max_degrees", 60.0)),
                exit_activation_delay_s=float(
                    overrides.pop("exit_activation_delay_s", 0.75)
                ),
                exit_settle_hold_s=float(overrides.pop("exit_settle_hold_s", 0.6)),
                exit_reacquire_center_tolerance=float(
                    overrides.pop("exit_reacquire_center_tolerance", 0.08)
                ),
                exit_visible_edge_threshold=float(
                    overrides.pop("exit_visible_edge_threshold", 0.74)
                ),
                exit_visible_box_edge_threshold=float(
                    overrides.pop("exit_visible_box_edge_threshold", 0.92)
                ),
                exit_cooldown_s=float(overrides.pop("exit_cooldown_s", 30.0)),
                continuous_max_speed_degrees_per_s=float(
                    overrides.pop("continuous_max_speed_degrees_per_s", 120.0)
                ),
                continuous_slow_zone_degrees=float(
                    overrides.pop("continuous_slow_zone_degrees", 45.0)
                ),
                continuous_stop_tolerance_degrees=float(
                    overrides.pop("continuous_stop_tolerance_degrees", 4.0)
                ),
                estimated_zero_settle_tolerance_degrees=float(
                    overrides.pop("estimated_zero_settle_tolerance_degrees", 1.0)
                ),
                estimated_zero_speed_scale=float(
                    overrides.pop("estimated_zero_speed_scale", 1.0)
                ),
                continuous_min_speed_pulse_delta_us=int(
                    overrides.pop("continuous_min_speed_pulse_delta_us", 70)
                ),
                continuous_max_speed_pulse_delta_us=int(
                    overrides.pop("continuous_max_speed_pulse_delta_us", 160)
                ),
                estimated_zero_move_pulse_delta_us=int(
                    overrides.pop("estimated_zero_move_pulse_delta_us", 70)
                ),
                estimated_zero_move_period_s=float(
                    overrides.pop("estimated_zero_move_period_s", 0.8)
                ),
                estimated_zero_move_duty_cycle=float(
                    overrides.pop("estimated_zero_move_duty_cycle", 0.2)
                ),
                continuous_return_to_zero_after_s=float(
                    overrides.pop("continuous_return_to_zero_after_s", 0.0)
                ),
                **overrides,
            ),
            pulse_writer=writer,
        )
        return temporary_state_dir, controller, writer

    def _normalize_payload(
        self,
        value: object,
        *,
        replacements: dict[str, str],
    ) -> object:
        if isinstance(value, dict):
            return {
                key: self._normalize_payload(item, replacements=replacements)
                for key, item in sorted(value.items())
            }
        if isinstance(value, tuple):
            return [self._normalize_payload(item, replacements=replacements) for item in value]
        if isinstance(value, list):
            return [self._normalize_payload(item, replacements=replacements) for item in value]
        if isinstance(value, str):
            return replacements.get(value, value)
        if isinstance(value, float):
            return round(value, 6)
        return value

    def _payload_digest(
        self,
        payload: object,
        *,
        replacements: dict[str, str] | None = None,
    ) -> str:
        normalized = self._normalize_payload(payload, replacements=replacements or {})
        rendered = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return sha256(rendered.encode("utf-8")).hexdigest()

    def _config_payload(self) -> tuple[object, dict[str, str]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimpleNamespace(
                project_root=temp_dir,
                attention_servo_enabled=True,
                attention_servo_driver="peer_pololu_maestro",
                attention_servo_control_mode="continuous_rotation",
                attention_servo_peer_base_url="http://10.42.0.2:8768/",
                attention_servo_peer_timeout_s=2.5,
                attention_servo_state_path="state/custom_servo_state.json",
                attention_servo_gpio=18,
                attention_servo_maestro_channel=4,
                attention_servo_invert_direction=True,
                attention_servo_target_hold_s=1.7,
                attention_servo_loss_extrapolation_s=0.9,
                attention_servo_loss_extrapolation_gain=1.1,
                attention_servo_min_confidence=0.61,
                attention_servo_hold_min_confidence=0.55,
                attention_servo_deadband=0.08,
                attention_servo_min_pulse_width_us=1020,
                attention_servo_center_pulse_width_us=1495,
                attention_servo_max_pulse_width_us=1988,
                attention_servo_max_step_us=66,
                attention_servo_target_smoothing_s=0.4,
                attention_servo_max_velocity_us_per_s=180.0,
                attention_servo_max_acceleration_us_per_s2=400.0,
                attention_servo_max_jerk_us_per_s3=1200.0,
                attention_servo_rest_max_velocity_us_per_s=40.0,
                attention_servo_rest_max_acceleration_us_per_s2=100.0,
                attention_servo_rest_max_jerk_us_per_s3=500.0,
                attention_servo_min_command_delta_us=6,
                attention_servo_visible_retarget_tolerance_us=30,
                display_attention_refresh_interval_s=0.15,
                attention_servo_soft_limit_margin_us=50,
                attention_servo_idle_release_s=0.9,
                attention_servo_settled_release_s=0.25,
                attention_servo_follow_exit_only=True,
                attention_servo_visible_recenter_interval_s=45.0,
                attention_servo_visible_recenter_center_tolerance=0.11,
                attention_servo_mechanical_range_degrees=360.0,
                attention_servo_exit_follow_max_degrees=90.0,
                attention_servo_exit_activation_delay_s=0.6,
                attention_servo_exit_settle_hold_s=1.2,
                attention_servo_exit_reacquire_center_tolerance=0.07,
                attention_servo_exit_visible_edge_threshold=0.71,
                attention_servo_exit_visible_box_edge_threshold=0.93,
                attention_servo_exit_cooldown_s=25.0,
                attention_servo_continuous_max_speed_degrees_per_s=140.0,
                attention_servo_continuous_slow_zone_degrees=28.0,
                attention_servo_continuous_stop_tolerance_degrees=3.5,
                attention_servo_continuous_min_speed_pulse_delta_us=72,
                attention_servo_continuous_max_speed_pulse_delta_us=165,
                attention_servo_estimated_zero_max_uncertainty_degrees=11.5,
                attention_servo_estimated_zero_settle_tolerance_degrees=0.9,
                attention_servo_estimated_zero_speed_scale=0.4,
                attention_servo_estimated_zero_move_pulse_delta_us=72,
                attention_servo_estimated_zero_move_period_s=1.2,
                attention_servo_estimated_zero_move_duty_cycle=0.15,
                attention_servo_continuous_return_to_zero_after_s=60.0,
                gpio_chip="gpiochip0",
            )
            servo_config = AttentionServoConfig.from_config(cast(Any, config))
            payload = asdict(servo_config)
        return payload, {
            str(payload["state_path"]): "<PROJECT_ROOT>/state/custom_servo_state.json",
        }

    def _position_payload(self) -> tuple[object, dict[str, str]]:
        temporary_state_dir, controller, writer = self._build_controller(
            max_step_us=80,
            target_smoothing_s=0.0,
            idle_release_s=0.4,
            rest_max_velocity_us_per_s=100000.0,
            rest_max_acceleration_us_per_s2=100000.0,
            rest_max_jerk_us_per_s3=1000000.0,
        )
        try:
            decisions = []
            for observed_at, active, center_x, confidence in (
                (10.0, True, 0.9, 0.95),
                (10.2, True, 0.82, 0.92),
                (10.4, False, None, 0.0),
                (10.6, False, None, 0.0),
                (10.8, False, None, 0.0),
                (11.0, False, None, 0.0),
                (11.2, False, None, 0.0),
            ):
                decisions.append(
                    asdict(
                        controller.update(
                            observed_at=observed_at,
                            active=active,
                            target_center_x=center_x,
                            confidence=confidence,
                        )
                    )
                )
            payload = {
                "decisions": decisions,
                "writes": writer.writes,
                "disables": writer.disables,
                "debug": controller.debug_snapshot(observed_at=11.2),
            }
            replacements = {
                controller.config.state_path: "<TMP>/attention_servo_state.json",
            }
        finally:
            temporary_state_dir.cleanup()
        return payload, replacements

    def _exit_payload(self) -> tuple[object, dict[str, str]]:
        temporary_state_dir, controller, writer = self._build_controller(
            follow_exit_only=True,
            target_hold_s=1.2,
            max_step_us=80,
            target_smoothing_s=0.0,
            exit_activation_delay_s=0.0,
        )
        try:
            decisions = []
            sequence: tuple[dict[str, Any], ...] = (
                {
                    "observed_at": 10.0,
                    "active": True,
                    "target_center_x": 0.85,
                    "confidence": 0.95,
                },
                {
                    "observed_at": 10.2,
                    "active": False,
                    "target_center_x": None,
                    "confidence": 0.0,
                },
                {
                    "observed_at": 10.3,
                    "active": True,
                    "target_center_x": 0.52,
                    "confidence": 0.95,
                },
                {
                    "observed_at": 10.4,
                    "active": False,
                    "target_center_x": None,
                    "confidence": 0.0,
                },
            )
            for kwargs in sequence:
                decisions.append(asdict(controller.update(**kwargs)))
            payload = {
                "decisions": decisions,
                "writes": writer.writes,
                "disables": writer.disables,
                "debug": controller.debug_snapshot(observed_at=10.4),
            }
            replacements = {
                controller.config.state_path: "<TMP>/attention_servo_state.json",
            }
        finally:
            temporary_state_dir.cleanup()
        return payload, replacements

    def _continuous_payload(self) -> tuple[object, dict[str, str]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "attention_servo_state.json"
            store = AttentionServoStateStore(state_path)
            store.save(
                AttentionServoRuntimeState(
                    heading_degrees=18.0,
                    heading_uncertainty_degrees=4.0,
                    hold_until_armed=False,
                    return_to_zero_requested=True,
                    zero_reference_confirmed=True,
                    updated_at=100.0,
                )
            )
            writer = FakeServoPulseWriter()
            controller = AttentionServoController(
                config=AttentionServoConfig(
                    enabled=True,
                    driver="lgpio",
                    control_mode="continuous_rotation",
                    state_path=str(state_path),
                    estimated_zero_max_uncertainty_degrees=15.0,
                    estimated_zero_move_pulse_delta_us=70,
                    estimated_zero_move_period_s=1.0,
                    estimated_zero_move_duty_cycle=0.25,
                    gpio_chip="gpiochip0",
                    gpio=18,
                    follow_exit_only=False,
                    mechanical_range_degrees=360.0,
                    exit_follow_max_degrees=90.0,
                    max_step_us=250,
                    min_command_delta_us=1,
                    rest_max_velocity_us_per_s=100000.0,
                    rest_max_acceleration_us_per_s2=100000.0,
                    rest_max_jerk_us_per_s3=1000000.0,
                ),
                pulse_writer=writer,
            )
            decisions = []
            sequence: tuple[dict[str, Any], ...] = (
                {
                    "observed_at": 10.0,
                    "active": False,
                    "target_center_x": None,
                    "confidence": 0.0,
                },
                {
                    "observed_at": 10.26,
                    "active": False,
                    "target_center_x": None,
                    "confidence": 0.0,
                },
                {
                    "observed_at": 10.75,
                    "active": False,
                    "target_center_x": None,
                    "confidence": 0.0,
                },
            )
            for kwargs in sequence:
                decisions.append(asdict(controller.update(**kwargs)))
            persisted_state = store.load()
            if persisted_state is None:
                self.fail("expected a persisted state after the continuous scenario")
            payload = {
                "decisions": decisions,
                "writes": writer.writes,
                "disables": writer.disables,
                "debug": controller.debug_snapshot(observed_at=10.75),
                "persisted_state": asdict(persisted_state),
            }
        return payload, {str(state_path): "<TMP>/attention_servo_state.json"}

    def test_golden_master_hashes_remain_stable(self) -> None:
        scenarios = {
            "config": self._config_payload(),
            "position": self._position_payload(),
            "exit": self._exit_payload(),
            "continuous": self._continuous_payload(),
        }
        for name, (payload, replacements) in scenarios.items():
            with self.subTest(case=name):
                digest = self._payload_digest(payload, replacements=replacements)
                self.assertEqual(digest, _EXPECTED_GOLDEN_DIGESTS[name])

    def test_public_wrapper_preserves_class_modules(self) -> None:
        self.assertEqual(AttentionServoConfig.__module__, "twinr.hardware.servo_follow")
        self.assertEqual(AttentionServoController.__module__, "twinr.hardware.servo_follow")
        self.assertEqual(
            servo_follow_impl.AttentionServoDecision.__module__,
            "twinr.hardware.servo_follow",
        )
        self.assertEqual(
            servo_follow_impl.LGPIOPWMServoPulseWriter.__module__,
            "twinr.hardware.servo_follow",
        )

    def test_public_wrapper_matches_internal_package_for_config_and_controller(self) -> None:
        config_payload, config_replacements = self._config_payload()
        self.assertEqual(
            self._payload_digest(config_payload, replacements=config_replacements),
            self._payload_digest(
                asdict(
                    servo_follow_impl.AttentionServoConfig.from_config(
                        cast(
                            Any,
                            SimpleNamespace(
                            project_root="/tmp/project-root",
                            attention_servo_enabled=True,
                            attention_servo_driver="peer_pololu_maestro",
                            attention_servo_control_mode="continuous_rotation",
                            attention_servo_peer_base_url="http://10.42.0.2:8768/",
                            attention_servo_peer_timeout_s=2.5,
                            attention_servo_state_path="state/custom_servo_state.json",
                            attention_servo_gpio=18,
                            attention_servo_maestro_channel=4,
                            attention_servo_invert_direction=True,
                            attention_servo_target_hold_s=1.7,
                            attention_servo_loss_extrapolation_s=0.9,
                            attention_servo_loss_extrapolation_gain=1.1,
                            attention_servo_min_confidence=0.61,
                            attention_servo_hold_min_confidence=0.55,
                            attention_servo_deadband=0.08,
                            attention_servo_min_pulse_width_us=1020,
                            attention_servo_center_pulse_width_us=1495,
                            attention_servo_max_pulse_width_us=1988,
                            attention_servo_max_step_us=66,
                            attention_servo_target_smoothing_s=0.4,
                            attention_servo_max_velocity_us_per_s=180.0,
                            attention_servo_max_acceleration_us_per_s2=400.0,
                            attention_servo_max_jerk_us_per_s3=1200.0,
                            attention_servo_rest_max_velocity_us_per_s=40.0,
                            attention_servo_rest_max_acceleration_us_per_s2=100.0,
                            attention_servo_rest_max_jerk_us_per_s3=500.0,
                            attention_servo_min_command_delta_us=6,
                            attention_servo_visible_retarget_tolerance_us=30,
                            display_attention_refresh_interval_s=0.15,
                            attention_servo_soft_limit_margin_us=50,
                            attention_servo_idle_release_s=0.9,
                            attention_servo_settled_release_s=0.25,
                            attention_servo_follow_exit_only=True,
                            attention_servo_visible_recenter_interval_s=45.0,
                            attention_servo_visible_recenter_center_tolerance=0.11,
                            attention_servo_mechanical_range_degrees=360.0,
                            attention_servo_exit_follow_max_degrees=90.0,
                            attention_servo_exit_activation_delay_s=0.6,
                            attention_servo_exit_settle_hold_s=1.2,
                            attention_servo_exit_reacquire_center_tolerance=0.07,
                            attention_servo_exit_visible_edge_threshold=0.71,
                            attention_servo_exit_visible_box_edge_threshold=0.93,
                            attention_servo_exit_cooldown_s=25.0,
                            attention_servo_continuous_max_speed_degrees_per_s=140.0,
                            attention_servo_continuous_slow_zone_degrees=28.0,
                            attention_servo_continuous_stop_tolerance_degrees=3.5,
                            attention_servo_continuous_min_speed_pulse_delta_us=72,
                            attention_servo_continuous_max_speed_pulse_delta_us=165,
                            attention_servo_estimated_zero_max_uncertainty_degrees=11.5,
                            attention_servo_estimated_zero_settle_tolerance_degrees=0.9,
                            attention_servo_estimated_zero_speed_scale=0.4,
                            attention_servo_estimated_zero_move_pulse_delta_us=72,
                            attention_servo_estimated_zero_move_period_s=1.2,
                            attention_servo_estimated_zero_move_duty_cycle=0.15,
                            attention_servo_continuous_return_to_zero_after_s=60.0,
                            gpio_chip="gpiochip0",
                            ),
                        )
                    )
                ),
                replacements={
                    "/tmp/project-root/state/custom_servo_state.json": (
                        "<PROJECT_ROOT>/state/custom_servo_state.json"
                    )
                },
            ),
        )

        public_writer = FakeServoPulseWriter()
        internal_writer = FakeServoPulseWriter()
        public = AttentionServoController(
            config=AttentionServoConfig(
                enabled=True,
                driver="lgpio",
                gpio_chip="gpiochip0",
                gpio=18,
                max_step_us=80,
                target_smoothing_s=0.0,
                target_hold_s=1.2,
                exit_activation_delay_s=0.0,
                follow_exit_only=True,
            ),
            pulse_writer=public_writer,
        )
        internal = servo_follow_impl.AttentionServoController(
            config=servo_follow_impl.AttentionServoConfig(
                enabled=True,
                driver="lgpio",
                gpio_chip="gpiochip0",
                gpio=18,
                max_step_us=80,
                target_smoothing_s=0.0,
                target_hold_s=1.2,
                exit_activation_delay_s=0.0,
                follow_exit_only=True,
            ),
            pulse_writer=internal_writer,
        )
        sequence: tuple[dict[str, Any], ...] = (
            {
                "observed_at": 10.0,
                "active": True,
                "target_center_x": 0.85,
                "confidence": 0.95,
            },
            {
                "observed_at": 10.2,
                "active": False,
                "target_center_x": None,
                "confidence": 0.0,
            },
            {
                "observed_at": 10.3,
                "active": True,
                "target_center_x": 0.52,
                "confidence": 0.95,
            },
        )
        public_payload = {
            "decisions": [asdict(public.update(**kwargs)) for kwargs in sequence],
            "writes": public_writer.writes,
            "disables": public_writer.disables,
        }
        internal_payload = {
            "decisions": [asdict(internal.update(**kwargs)) for kwargs in sequence],
            "writes": internal_writer.writes,
            "disables": internal_writer.disables,
        }
        self.assertEqual(public_payload, internal_payload)
