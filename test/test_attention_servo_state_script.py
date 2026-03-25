"""Regression coverage for the operator-facing attention-servo state helper."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.servo_state import AttentionServoRuntimeState, AttentionServoStateStore


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "hardware" / "servo" / "attention_servo_state.py"
_SPEC = importlib.util.spec_from_file_location("attention_servo_state_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class AttentionServoStateScriptTests(unittest.TestCase):
    def test_hold_current_zero_persists_confirmed_manual_hold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "attention_servo_state.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = _MODULE.main(["--state-path", str(state_path), "hold-current-zero"])
            saved_state = AttentionServoStateStore(state_path).load()

        self.assertEqual(exit_code, 0)
        self.assertIn("attention_servo_state=manual_hold", stdout.getvalue())
        self.assertIsNotNone(saved_state)
        if saved_state is None:
            self.fail("expected hold-current-zero to create a persisted state file")
        self.assertEqual(saved_state.heading_degrees, 0.0)
        self.assertTrue(saved_state.hold_until_armed)
        self.assertTrue(saved_state.zero_reference_confirmed)

    def test_arm_preserves_heading_and_disables_manual_hold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "attention_servo_state.json"
            store = AttentionServoStateStore(state_path)
            store.save(
                AttentionServoRuntimeState(
                    heading_degrees=14.5,
                    hold_until_armed=True,
                    zero_reference_confirmed=True,
                    updated_at=10.0,
                )
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = _MODULE.main(["--state-path", str(state_path), "arm"])
            saved_state = store.load()

        self.assertEqual(exit_code, 0)
        self.assertIn("attention_servo_state=armed", stdout.getvalue())
        self.assertIsNotNone(saved_state)
        if saved_state is None:
            self.fail("expected arm to keep a persisted state file")
        self.assertEqual(saved_state.heading_degrees, 14.5)
        self.assertFalse(saved_state.hold_until_armed)
        self.assertTrue(saved_state.zero_reference_confirmed)

    def test_arm_fails_closed_without_confirmed_zero_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "attention_servo_state.json"
            store = AttentionServoStateStore(state_path)
            store.save(
                AttentionServoRuntimeState(
                    heading_degrees=0.0,
                    hold_until_armed=True,
                    zero_reference_confirmed=False,
                    updated_at=10.0,
                )
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = _MODULE.main(["--state-path", str(state_path), "arm"])
            saved_state = store.load()

        self.assertEqual(exit_code, 2)
        self.assertIn("zero reference is confirmed", stderr.getvalue())
        self.assertIsNotNone(saved_state)
        if saved_state is None:
            self.fail("expected arm failure to leave the state file intact")
        self.assertTrue(saved_state.hold_until_armed)
