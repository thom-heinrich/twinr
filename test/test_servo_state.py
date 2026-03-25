from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.servo_state import AttentionServoRuntimeState, AttentionServoStateStore


class AttentionServoStateStoreTests(unittest.TestCase):
    def test_load_returns_none_when_state_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "attention_servo_state.json"

            loaded_state = AttentionServoStateStore(state_path).load()

        self.assertIsNone(loaded_state)

    def test_save_and_load_round_trip_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "attention_servo_state.json"
            store = AttentionServoStateStore(state_path)
            expected_state = AttentionServoRuntimeState(
                heading_degrees=12.5,
                hold_until_armed=True,
                zero_reference_confirmed=True,
                updated_at=42.0,
            )

            store.save(expected_state)
            loaded_state = store.load()

        self.assertEqual(loaded_state, expected_state)

    def test_runtime_state_normalizes_invalid_payload_values(self) -> None:
        state = AttentionServoRuntimeState.from_payload(
            {
                "heading_degrees": "999",
                "hold_until_armed": 1,
                "zero_reference_confirmed": "yes",
                "updated_at": "nan",
            }
        )

        self.assertEqual(state.heading_degrees, 180.0)
        self.assertTrue(state.hold_until_armed)
        self.assertTrue(state.zero_reference_confirmed)
        self.assertIsNone(state.updated_at)

    def test_hold_current_heading_preserves_heading_and_sets_hold(self) -> None:
        current_state = AttentionServoRuntimeState(
            heading_degrees=21.5,
            hold_until_armed=False,
            zero_reference_confirmed=True,
            updated_at=10.0,
        )

        held_state = current_state.hold_current_heading(updated_at=22.0)

        self.assertEqual(held_state.heading_degrees, 21.5)
        self.assertTrue(held_state.hold_until_armed)
        self.assertTrue(held_state.zero_reference_confirmed)
        self.assertEqual(held_state.updated_at, 22.0)

    def test_adopt_current_as_zero_resets_heading_and_confirms_reference(self) -> None:
        current_state = AttentionServoRuntimeState(
            heading_degrees=-33.0,
            hold_until_armed=False,
            zero_reference_confirmed=False,
            updated_at=10.0,
        )

        zero_state = current_state.adopt_current_as_zero(updated_at=42.0)

        self.assertEqual(zero_state.heading_degrees, 0.0)
        self.assertTrue(zero_state.hold_until_armed)
        self.assertTrue(zero_state.zero_reference_confirmed)
        self.assertEqual(zero_state.updated_at, 42.0)

    def test_arm_follow_preserves_heading_and_disables_hold(self) -> None:
        current_state = AttentionServoRuntimeState(
            heading_degrees=9.75,
            hold_until_armed=True,
            zero_reference_confirmed=True,
            updated_at=10.0,
        )

        armed_state = current_state.arm_follow(updated_at=55.0)

        self.assertEqual(armed_state.heading_degrees, 9.75)
        self.assertFalse(armed_state.hold_until_armed)
        self.assertTrue(armed_state.zero_reference_confirmed)
        self.assertEqual(armed_state.updated_at, 55.0)

    def test_load_or_default_returns_default_state_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "attention_servo_state.json"

            loaded_state = AttentionServoStateStore(state_path).load_or_default()

        self.assertEqual(loaded_state, AttentionServoRuntimeState())
