import json
import os
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_FORENSICS_PATH = Path(__file__).resolve().parents[1] / "src" / "twinr" / "agent" / "workflows" / "forensics.py"
_SPEC = importlib.util.spec_from_file_location("twinr_workflow_forensics_test_module", _FORENSICS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
WorkflowForensics = _MODULE.WorkflowForensics


class WorkflowForensicsTests(unittest.TestCase):
    def test_forensics_runpack_contains_events_decisions_and_spans(self) -> None:
        with TemporaryDirectory() as temp_dir:
            trace_dir = Path(temp_dir) / "state" / "forensics" / "workflow"
            previous_env = {
                key: os.environ.get(key)
                for key in (
                    "TWINR_WORKFLOW_TRACE_ENABLED",
                    "TWINR_WORKFLOW_TRACE_MODE",
                    "TWINR_WORKFLOW_TRACE_DIR",
                )
            }
            os.environ["TWINR_WORKFLOW_TRACE_ENABLED"] = "1"
            os.environ["TWINR_WORKFLOW_TRACE_MODE"] = "forensic"
            os.environ["TWINR_WORKFLOW_TRACE_DIR"] = str(trace_dir)
            try:
                tracer = WorkflowForensics.from_env(project_root=Path(temp_dir), service="workflow-test")
                tracer.event(kind="workflow", msg="probe_event", details={"value": 1})
                tracer.decision(
                    msg="probe_decision",
                    question="Which path should run?",
                    selected={"id": "path_a", "summary": "Choose path A"},
                    options=[
                        {"id": "path_a", "summary": "Path A"},
                        {"id": "path_b", "summary": "Path B"},
                    ],
                    context={"state": "ready"},
                )
                with tracer.span(name="probe_span", kind="workflow", details={"scope": "test"}):
                    tracer.event(kind="metric", msg="inside_span", details={"scope": "test"})
                tracer.close()

                run_id = (trace_dir / "LATEST").read_text(encoding="utf-8").strip()
                run_dir = trace_dir / run_id
                self.assertTrue((run_dir / "run.jsonl").exists())
                self.assertTrue((run_dir / "run.trace").exists())
                self.assertTrue((run_dir / "run.metrics.json").exists())
                self.assertTrue((run_dir / "run.summary.json").exists())
                self.assertTrue((run_dir / "run.repro" / "runtime.json").exists())
                self.assertTrue((run_dir / "run.repro" / "env.json").exists())

                records = [
                    json.loads(line)
                    for line in (run_dir / "run.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                msgs = {record["msg"] for record in records}
                self.assertIn("workflow_trace_started", msgs)
                self.assertIn("probe_event", msgs)
                self.assertIn("probe_decision", msgs)
                self.assertIn("probe_span", msgs)
                self.assertIn("workflow_trace_stopped", msgs)
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_forensics_reads_trace_flags_from_project_dotenv(self) -> None:
        with TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            trace_dir = project_root / "trace-out"
            (project_root / ".env").write_text(
                "\n".join(
                    [
                        'TWINR_WORKFLOW_TRACE_ENABLED="1"',
                        'TWINR_WORKFLOW_TRACE_MODE="forensic"',
                        f'TWINR_WORKFLOW_TRACE_DIR="{trace_dir}"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            previous_env = {
                key: os.environ.get(key)
                for key in (
                    "TWINR_WORKFLOW_TRACE_ENABLED",
                    "TWINR_WORKFLOW_TRACE_MODE",
                    "TWINR_WORKFLOW_TRACE_DIR",
                )
            }
            for key in previous_env:
                os.environ.pop(key, None)
            try:
                tracer = WorkflowForensics.from_env(project_root=project_root, service="workflow-test")
                tracer.event(kind="workflow", msg="dotenv_probe", details={})
                tracer.close()
                run_id = (trace_dir / "LATEST").read_text(encoding="utf-8").strip()
                self.assertTrue((trace_dir / run_id / "run.jsonl").exists())
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_forensics_handles_string_exception_payload_without_crashing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            trace_dir = Path(temp_dir) / "state" / "forensics" / "workflow"
            previous_env = {
                key: os.environ.get(key)
                for key in (
                    "TWINR_WORKFLOW_TRACE_ENABLED",
                    "TWINR_WORKFLOW_TRACE_MODE",
                    "TWINR_WORKFLOW_TRACE_DIR",
                )
            }
            os.environ["TWINR_WORKFLOW_TRACE_ENABLED"] = "1"
            os.environ["TWINR_WORKFLOW_TRACE_MODE"] = "forensic"
            os.environ["TWINR_WORKFLOW_TRACE_DIR"] = str(trace_dir)
            try:
                tracer = WorkflowForensics.from_env(project_root=Path(temp_dir), service="workflow-test")
                tracer.event(kind="exception", msg="legacy_exception", details={"exception": "LegacyError: boom"})
                tracer.close()

                run_id = (trace_dir / "LATEST").read_text(encoding="utf-8").strip()
                summary = json.loads((trace_dir / run_id / "run.summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["exception_counts"].get("LegacyError"), 1)
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
