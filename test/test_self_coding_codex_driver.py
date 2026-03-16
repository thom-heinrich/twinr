import json
import importlib
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.self_coding import (
    CompileJobRecord,
    CompileJobStatus,
    CompileTarget,
    FeasibilityOutcome,
    FeasibilityResult,
    RequirementsDialogueSession,
    RequirementsDialogueStatus,
)
from twinr.agent.self_coding.codex_driver import (
    CodexCompileProgress,
    CodexCompileRequest,
    CodexDriverUnavailableError,
    compile_output_schema,
)
from twinr.agent.self_coding.codex_driver.app_server import CodexAppServerDriver, CodexAppServerRunCollector
from twinr.agent.self_coding.codex_driver.exec_fallback import CodexExecRunCollector
from twinr.agent.self_coding.codex_driver.sdk import CodexSdkDriver
from twinr.agent.self_coding.worker import LocalCodexCompileDriver


def _ready_session() -> RequirementsDialogueSession:
    return RequirementsDialogueSession(
        session_id="dialogue_ready123",
        request_summary="Read new email aloud.",
        skill_name="Read Emails",
        action="Read new email aloud",
        capabilities=("email", "speaker", "safety"),
        feasibility=FeasibilityResult(
            outcome=FeasibilityOutcome.GREEN,
            summary="Fits the automation-first path.",
            suggested_target=CompileTarget.AUTOMATION_MANIFEST,
        ),
        status=RequirementsDialogueStatus.READY_FOR_COMPILE,
        trigger_mode="push",
        trigger_conditions=("new_email",),
        scope={"channel": "email"},
        constraints=("ask_first",),
    )


def _compile_request(root: Path) -> CodexCompileRequest:
    request_path = root / "REQUEST.md"
    output_schema_path = root / "output_schema.json"
    request_path.write_text("compile", encoding="utf-8")
    output_schema_path.write_text("{}", encoding="utf-8")
    return CodexCompileRequest(
        job=CompileJobRecord(
            job_id="job_sdk123",
            skill_id="sdk_skill",
            skill_name="SDK Skill",
            status=CompileJobStatus.QUEUED,
            requested_target=CompileTarget.AUTOMATION_MANIFEST,
            spec_hash="sdk-spec",
        ),
        session=_ready_session(),
        prompt="compile",
        output_schema={},
        workspace_root=str(root),
        request_path=str(request_path),
        output_schema_path=str(output_schema_path),
    )


def _write_fake_bridge(temp_dir: str, script_body: str) -> Path:
    script_path = Path(temp_dir) / "fake_bridge.py"
    script_path.write_text(
        "import json\nimport sys\n\n" + textwrap.dedent(script_body).lstrip(),
        encoding="utf-8",
    )
    return script_path


class CodexExecRunCollectorTests(unittest.TestCase):
    def test_exec_collector_tracks_final_message_and_events(self) -> None:
        collector = CodexExecRunCollector()

        collector.consume({"type": "thread.started", "thread_id": "thread-123"})
        collector.consume({"type": "turn.started"})
        collector.consume({"type": "item.completed", "item": {"id": "item_0", "type": "agent_message", "text": "{\"status\":\"ok\"}"}})
        collector.consume({"type": "turn.completed", "usage": {"output_tokens": 12}})

        result = collector.build_result()

        self.assertEqual(result.thread_id, "thread-123")
        self.assertEqual(result.final_message, "{\"status\":\"ok\"}")
        self.assertEqual(result.events[-1].kind, "turn_completed")
        self.assertEqual(result.events[-1].metadata["output_tokens"], 12)

    def test_compile_output_schema_marks_nested_metadata_objects_closed(self) -> None:
        schema = compile_output_schema()
        metadata_schema = schema["properties"]["artifacts"]["items"]["properties"]["metadata"]

        self.assertFalse(metadata_schema["additionalProperties"])


class CodexAppServerRunCollectorTests(unittest.TestCase):
    def test_app_server_collector_tracks_agent_message_deltas_and_completion(self) -> None:
        collector = CodexAppServerRunCollector()

        collector.consume({"method": "thread/started", "params": {"thread": {"id": "thread-123"}}})
        collector.consume({"method": "turn/started", "params": {"threadId": "thread-123", "turn": {"id": "turn-123", "status": "inProgress", "items": []}}})
        collector.consume(
            {
                "method": "item/started",
                "params": {
                    "threadId": "thread-123",
                    "turnId": "turn-123",
                    "item": {"type": "agentMessage", "id": "msg-1", "text": "", "phase": "final_answer"},
                },
            }
        )
        collector.consume(
            {
                "method": "item/agentMessage/delta",
                "params": {"threadId": "thread-123", "turnId": "turn-123", "itemId": "msg-1", "delta": "{\"status\":\"ok\"}"},
            }
        )
        collector.consume(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thread-123",
                    "turnId": "turn-123",
                    "item": {"type": "agentMessage", "id": "msg-1", "text": "{\"status\":\"ok\"}", "phase": "final_answer"},
                },
            }
        )
        collector.consume({"method": "turn/completed", "params": {"threadId": "thread-123", "turn": {"id": "turn-123", "status": "completed", "items": [], "error": None}}})

        result = collector.build_result()

        self.assertEqual(result.thread_id, "thread-123")
        self.assertEqual(result.turn_id, "turn-123")
        self.assertEqual(result.final_message, "{\"status\":\"ok\"}")
        self.assertEqual(result.events[-1].kind, "turn_completed")

    def test_app_server_collector_exposes_generic_progress_snapshot(self) -> None:
        collector = CodexAppServerRunCollector()

        emitted = collector.consume(
            {
                "method": "item/reasoning/delta",
                "params": {
                    "threadId": "thread-123",
                    "turnId": "turn-123",
                    "itemId": "reason-1",
                    "delta": "checking capabilities",
                },
            }
        )
        snapshot = collector.snapshot()

        self.assertEqual(emitted[-1].kind, "item_reasoning_delta")
        self.assertIsInstance(snapshot, CodexCompileProgress)
        self.assertEqual(snapshot.event_count, 1)
        self.assertEqual(snapshot.last_event_kind, "item_reasoning_delta")
        self.assertEqual(snapshot.thread_id, "thread-123")
        self.assertEqual(snapshot.turn_id, "turn-123")

    def test_driver_accepts_inferred_completion_after_final_message_grace_period(self) -> None:
        collector = CodexAppServerRunCollector()
        collector.consume(
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thread-123",
                    "turnId": "turn-123",
                    "item": {"type": "agentMessage", "id": "msg-1", "text": "{\"status\":\"ok\"}", "phase": "final_answer"},
                },
            }
        )
        snapshot = collector.snapshot()

        self.assertTrue(
            CodexAppServerDriver._should_infer_completion(
                progress=snapshot,
                idle_seconds=1.6,
                final_message_grace_seconds=1.0,
            )
        )
        self.assertFalse(
            CodexAppServerDriver._should_infer_completion(
                progress=snapshot,
                idle_seconds=0.2,
                final_message_grace_seconds=1.0,
            )
        )


class CodexSdkDriverTests(unittest.TestCase):
    def test_productive_codex_driver_surface_excludes_app_server_exports(self) -> None:
        driver_module = importlib.import_module("twinr.agent.self_coding.codex_driver")

        self.assertFalse(hasattr(driver_module, "CodexAppServerDriver"))
        self.assertFalse(hasattr(driver_module, "CodexAppServerRunCollector"))

    def test_sdk_driver_streams_events_and_parses_compile_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request = _compile_request(root)
            bridge_script = _write_fake_bridge(
                temp_dir,
                """
if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
    sys.stdout.write(json.dumps({"ok": True, "node_version": "test"}) + "\\n")
    sys.exit(0)
payload = json.loads(sys.stdin.read())
assert payload["workspaceRoot"]
events = [
    {"type": "thread.started", "thread_id": "thread-sdk-123"},
    {"type": "turn.started"},
    {
        "type": "item.completed",
        "item": {
            "id": "item-1",
            "type": "agent_message",
            "text": json.dumps(
                {
                    "status": "ok",
                    "summary": "Compiled through SDK.",
                    "review": "Ready for review.",
                    "artifacts": [],
                }
            ),
        },
    },
    {"type": "turn.completed", "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 2}},
]
for event in events:
    sys.stdout.write(json.dumps(event) + "\\n")
    sys.stdout.flush()
                """,
            )
            driver = CodexSdkDriver(command=(sys.executable, "-u"), bridge_script=bridge_script, timeout_seconds=5.0)
            progress: list[CodexCompileProgress] = []

            result = driver.run_compile(request, event_sink=lambda event, snapshot: progress.append(snapshot))

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.summary, "Compiled through SDK.")
        self.assertEqual(result.events[-1].kind, "turn_completed")
        self.assertEqual(progress[-1].thread_id, "thread-sdk-123")
        self.assertTrue(progress[-1].turn_completed)

    def test_sdk_driver_raises_unavailable_for_failed_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request = _compile_request(root)
            bridge_script = _write_fake_bridge(
                temp_dir,
                """
if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
    sys.stdout.write(json.dumps({"ok": True, "node_version": "test"}) + "\\n")
    sys.exit(0)
json.loads(sys.stdin.read())
sys.stdout.write(json.dumps({"type": "turn.failed", "error": {"message": "sdk compile failed"}}) + "\\n")
sys.stdout.flush()
                """,
            )
            driver = CodexSdkDriver(command=(sys.executable, "-u"), bridge_script=bridge_script, timeout_seconds=5.0)

            with self.assertRaises(CodexDriverUnavailableError) as ctx:
                driver.run_compile(request)

        self.assertIn("sdk compile failed", str(ctx.exception))

    def test_sdk_driver_runs_startup_self_test_before_compile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request = _compile_request(root)
            marker_path = Path(temp_dir) / "compile-called.txt"
            bridge_script = _write_fake_bridge(
                temp_dir,
                f"""
from pathlib import Path
marker_path = Path({str(marker_path)!r})
if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
    sys.stderr.write("broken bridge install")
    sys.exit(7)
marker_path.write_text("compile was invoked", encoding="utf-8")
sys.stdout.write(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}}}}) + "\\n")
                """,
            )
            driver = CodexSdkDriver(command=(sys.executable, "-u"), bridge_script=bridge_script, timeout_seconds=5.0)

            with self.assertRaises(CodexDriverUnavailableError) as ctx:
                driver.run_compile(request)

        self.assertIn("startup self-test failed", str(ctx.exception))
        self.assertIn("broken bridge install", str(ctx.exception))
        self.assertFalse(marker_path.exists())

    def test_local_compile_driver_prefers_sdk_primary_by_default(self) -> None:
        driver = LocalCodexCompileDriver()

        self.assertIsInstance(driver.primary, CodexSdkDriver)


if __name__ == "__main__":
    unittest.main()
