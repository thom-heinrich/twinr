import json
import os
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from threading import Event, Thread
import unittest
from contextvars import copy_context

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_FORENSICS_PATH = Path(__file__).resolve().parents[1] / "src" / "twinr" / "agent" / "workflows" / "forensics.py"
_SPEC = importlib.util.spec_from_file_location("twinr_workflow_forensics_test_module", _FORENSICS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
WorkflowForensics = _MODULE.WorkflowForensics
bind_workflow_forensics = _MODULE.bind_workflow_forensics
capture_thread_snapshot = _MODULE.capture_thread_snapshot
workflow_event = _MODULE.workflow_event
workflow_span = _MODULE.workflow_span


class WorkflowForensicsTests(unittest.TestCase):
    def test_capture_thread_snapshot_reports_live_top_frame(self) -> None:
        started = Event()
        release = Event()

        def _worker() -> None:
            started.set()
            release.wait(timeout=0.5)

        thread = Thread(target=_worker, name="snapshot-worker", daemon=True)
        thread.start()
        self.assertTrue(started.wait(timeout=1.0))
        try:
            snapshot = capture_thread_snapshot(thread)
        finally:
            release.set()
            thread.join(timeout=1.0)

        self.assertTrue(snapshot["present"])
        self.assertTrue(snapshot["alive"])
        self.assertEqual(snapshot["name"], "snapshot-worker")
        self.assertTrue(snapshot["stack_present"])
        self.assertIsNotNone(snapshot["top_frame"])
        self.assertIn(
            "_worker",
            [frame["func"] for frame in snapshot["stack"]],
        )

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

    def test_bound_workflow_context_propagates_trace_and_parent_span(self) -> None:
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
                with bind_workflow_forensics(tracer, trace_id="trace-under-test"):
                    workflow_event(kind="workflow", msg="outside_span", details={})
                    with workflow_span(name="outer_span", kind="workflow") as outer:
                        workflow_event(kind="workflow", msg="inside_outer", details={})
                        copy_context().run(self._emit_child_span_event)
                tracer.close()

                run_id = (trace_dir / "LATEST").read_text(encoding="utf-8").strip()
                records = [
                    json.loads(line)
                    for line in (trace_dir / run_id / "run.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                outside_event = next(record for record in records if record["msg"] == "outside_span")
                outer_span = next(
                    record
                    for record in records
                    if record["msg"] == "outer_span" and record["kind"] == "span_start"
                )
                child_span = next(
                    record
                    for record in records
                    if record["msg"] == "child_span" and record["kind"] == "span_start"
                )
                child_event = next(record for record in records if record["msg"] == "inside_child")

                self.assertEqual(outside_event["trace_id"], "trace-under-test")
                self.assertEqual(outer_span["trace_id"], "trace-under-test")
                self.assertEqual(child_span["trace_id"], "trace-under-test")
                self.assertEqual(child_event["trace_id"], "trace-under-test")
                self.assertEqual(child_span["parent_span_id"], outer.span_id)
                self.assertEqual(child_event["span_id"], child_span["span_id"])
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    @staticmethod
    def _emit_child_span_event() -> None:
        with workflow_span(name="child_span", kind="workflow"):
            workflow_event(kind="workflow", msg="inside_child", details={})


if __name__ == "__main__":
    unittest.main()
