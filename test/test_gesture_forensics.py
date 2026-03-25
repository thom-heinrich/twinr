import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.workflows.forensics import workflow_event
from twinr.hardware.camera_ai.gesture_forensics import GestureForensics


class GestureForensicsTests(unittest.TestCase):
    def test_bind_refresh_writes_runpack_and_scoped_branch_edges(self) -> None:
        with TemporaryDirectory() as temp_dir:
            module_path = Path(temp_dir) / "branch_target.py"
            module_path.write_text(
                "\n".join(
                    [
                        "def branch_probe(flag: bool) -> int:",
                        "    total = 0",
                        "    if flag:",
                        "        total += 1",
                        "    else:",
                        "        total -= 1",
                        "    return total",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            spec = importlib.util.spec_from_file_location("gesture_forensics_branch_target", module_path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            trace_dir = Path(temp_dir) / "state" / "forensics" / "gesture"
            previous_env = {
                key: os.environ.get(key)
                for key in (
                    "TWINR_GESTURE_FORENSICS_ENABLED",
                    "TWINR_GESTURE_FORENSICS_MODE",
                    "TWINR_GESTURE_FORENSICS_DIR",
                    "TWINR_GESTURE_FORENSICS_SCOPE",
                )
            }
            os.environ["TWINR_GESTURE_FORENSICS_ENABLED"] = "1"
            os.environ["TWINR_GESTURE_FORENSICS_MODE"] = "deep-exec"
            os.environ["TWINR_GESTURE_FORENSICS_DIR"] = str(trace_dir)
            os.environ["TWINR_GESTURE_FORENSICS_SCOPE"] = str(module_path.resolve())
            try:
                tracer = GestureForensics.from_env(project_root=Path(temp_dir), service="gesture-test")
                with tracer.bind_refresh(
                    observed_at=1.25,
                    runtime_status_value="waiting",
                    vision_mode="test",
                    refresh_interval_s=0.2,
                ):
                    module.branch_probe(True)
                tracer.close()

                run_id = (trace_dir / "LATEST").read_text(encoding="utf-8").strip()
                run_dir = trace_dir / run_id
                self.assertTrue((run_dir / "run.jsonl").exists())
                self.assertTrue((run_dir / "run.trace").exists())
                self.assertTrue((run_dir / "run.metrics.json").exists())
                self.assertTrue((run_dir / "run.summary.json").exists())
                self.assertTrue((run_dir / "run.repro" / "runtime.json").exists())

                records = [
                    json.loads(line)
                    for line in (run_dir / "run.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                msgs = {record["msg"] for record in records}
                self.assertIn("gesture_refresh_started", msgs)
                self.assertIn("gesture_refresh_completed", msgs)
                self.assertIn("gesture_deep_exec_started", msgs)
                self.assertIn("gesture_deep_exec_stopped", msgs)
                self.assertTrue(any(record["msg"] == "gesture_deep_exec_edge" for record in records))
            finally:
                sys.modules.pop("gesture_forensics_branch_target", None)
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_bind_refresh_redacts_secret_fields(self) -> None:
        with TemporaryDirectory() as temp_dir:
            trace_dir = Path(temp_dir) / "state" / "forensics" / "gesture"
            previous_env = {
                key: os.environ.get(key)
                for key in (
                    "TWINR_GESTURE_FORENSICS_ENABLED",
                    "TWINR_GESTURE_FORENSICS_MODE",
                    "TWINR_GESTURE_FORENSICS_DIR",
                )
            }
            os.environ["TWINR_GESTURE_FORENSICS_ENABLED"] = "1"
            os.environ["TWINR_GESTURE_FORENSICS_MODE"] = "forensic"
            os.environ["TWINR_GESTURE_FORENSICS_DIR"] = str(trace_dir)
            try:
                tracer = GestureForensics.from_env(project_root=Path(temp_dir), service="gesture-test")
                with tracer.bind_refresh(
                    observed_at=2.5,
                    runtime_status_value="waiting",
                    vision_mode="test",
                    refresh_interval_s=0.2,
                ):
                    workflow_event(
                        kind="io",
                        msg="secret_probe",
                        details={"api_key": "sk-super-secret", "prompt": "a raw prompt"},
                    )
                tracer.close()

                run_id = (trace_dir / "LATEST").read_text(encoding="utf-8").strip()
                records = [
                    json.loads(line)
                    for line in (trace_dir / run_id / "run.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                secret_record = next(record for record in records if record["msg"] == "secret_probe")
                details = secret_record["details"]
                self.assertEqual(details["api_key"], "[redacted]")
                self.assertIn("[redacted_text", details["prompt"])
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_from_env_accepts_larger_runtime_event_budget(self) -> None:
        with TemporaryDirectory() as temp_dir:
            trace_dir = Path(temp_dir) / "state" / "forensics" / "gesture"
            previous_env = {
                key: os.environ.get(key)
                for key in (
                    "TWINR_GESTURE_FORENSICS_ENABLED",
                    "TWINR_GESTURE_FORENSICS_MODE",
                    "TWINR_GESTURE_FORENSICS_DIR",
                    "TWINR_GESTURE_FORENSICS_MAX_EVENTS",
                )
            }
            os.environ["TWINR_GESTURE_FORENSICS_ENABLED"] = "1"
            os.environ["TWINR_GESTURE_FORENSICS_MODE"] = "forensic"
            os.environ["TWINR_GESTURE_FORENSICS_DIR"] = str(trace_dir)
            os.environ["TWINR_GESTURE_FORENSICS_MAX_EVENTS"] = "300000"
            try:
                tracer = GestureForensics.from_env(project_root=Path(temp_dir), service="gesture-test")
                self.assertEqual(tracer.config.max_events, 300000)
                tracer.close()
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_deep_exec_defaults_to_one_refresh_per_runtime(self) -> None:
        with TemporaryDirectory() as temp_dir:
            trace_dir = Path(temp_dir) / "state" / "forensics" / "gesture"
            previous_env = {
                key: os.environ.get(key)
                for key in (
                    "TWINR_GESTURE_FORENSICS_ENABLED",
                    "TWINR_GESTURE_FORENSICS_MODE",
                    "TWINR_GESTURE_FORENSICS_DIR",
                    "TWINR_GESTURE_FORENSICS_UNSAFE_CONTINUOUS_DEEP_EXEC",
                    "TWINR_GESTURE_FORENSICS_DEEP_EXEC_MAX_REFRESHES",
                )
            }
            os.environ["TWINR_GESTURE_FORENSICS_ENABLED"] = "1"
            os.environ["TWINR_GESTURE_FORENSICS_MODE"] = "deep-exec"
            os.environ["TWINR_GESTURE_FORENSICS_DIR"] = str(trace_dir)
            try:
                tracer = GestureForensics.from_env(project_root=Path(temp_dir), service="gesture-test")
                with patch("twinr.hardware.camera_ai.gesture_forensics._DeepExecMonitor") as monitor_cls:
                    monitor = monitor_cls.return_value
                    monitor.__enter__.return_value = monitor
                    monitor.__exit__.return_value = None
                    for _ in range(2):
                        with tracer.bind_refresh(
                            observed_at=1.25,
                            runtime_status_value="waiting",
                            vision_mode="test",
                            refresh_interval_s=0.2,
                        ):
                            pass
                tracer.close()
                self.assertEqual(monitor_cls.call_count, 1)
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_deep_exec_can_stay_continuous_with_explicit_unsafe_override(self) -> None:
        with TemporaryDirectory() as temp_dir:
            trace_dir = Path(temp_dir) / "state" / "forensics" / "gesture"
            previous_env = {
                key: os.environ.get(key)
                for key in (
                    "TWINR_GESTURE_FORENSICS_ENABLED",
                    "TWINR_GESTURE_FORENSICS_MODE",
                    "TWINR_GESTURE_FORENSICS_DIR",
                    "TWINR_GESTURE_FORENSICS_UNSAFE_CONTINUOUS_DEEP_EXEC",
                )
            }
            os.environ["TWINR_GESTURE_FORENSICS_ENABLED"] = "1"
            os.environ["TWINR_GESTURE_FORENSICS_MODE"] = "deep-exec"
            os.environ["TWINR_GESTURE_FORENSICS_DIR"] = str(trace_dir)
            os.environ["TWINR_GESTURE_FORENSICS_UNSAFE_CONTINUOUS_DEEP_EXEC"] = "1"
            try:
                tracer = GestureForensics.from_env(project_root=Path(temp_dir), service="gesture-test")
                with patch("twinr.hardware.camera_ai.gesture_forensics._DeepExecMonitor") as monitor_cls:
                    monitor = monitor_cls.return_value
                    monitor.__enter__.return_value = monitor
                    monitor.__exit__.return_value = None
                    for _ in range(2):
                        with tracer.bind_refresh(
                            observed_at=1.25,
                            runtime_status_value="waiting",
                            vision_mode="test",
                            refresh_interval_s=0.2,
                        ):
                            pass
                tracer.close()
                self.assertEqual(monitor_cls.call_count, 2)
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_deep_exec_skips_when_underlying_tracer_is_saturated(self) -> None:
        with TemporaryDirectory() as temp_dir:
            trace_dir = Path(temp_dir) / "state" / "forensics" / "gesture"
            previous_env = {
                key: os.environ.get(key)
                for key in (
                    "TWINR_GESTURE_FORENSICS_ENABLED",
                    "TWINR_GESTURE_FORENSICS_MODE",
                    "TWINR_GESTURE_FORENSICS_DIR",
                )
            }
            os.environ["TWINR_GESTURE_FORENSICS_ENABLED"] = "1"
            os.environ["TWINR_GESTURE_FORENSICS_MODE"] = "deep-exec"
            os.environ["TWINR_GESTURE_FORENSICS_DIR"] = str(trace_dir)
            try:
                tracer = GestureForensics.from_env(project_root=Path(temp_dir), service="gesture-test")
                tracer.tracer._trace_truncated = True
                with patch("twinr.hardware.camera_ai.gesture_forensics._DeepExecMonitor") as monitor_cls:
                    with tracer.bind_refresh(
                        observed_at=1.25,
                        runtime_status_value="waiting",
                        vision_mode="test",
                        refresh_interval_s=0.2,
                    ):
                        pass
                tracer.close()
                monitor_cls.assert_not_called()
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

if __name__ == "__main__":
    unittest.main()
