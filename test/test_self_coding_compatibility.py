from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test.self_coding_test_utils import stable_sha256
from twinr.agent.self_coding import (
    CompileJobRecord,
    CompileJobStatus,
    CompileTarget,
    FeasibilityOutcome,
    FeasibilityResult,
    RequirementsDialogueSession,
    RequirementsDialogueStatus,
)
from twinr.agent.self_coding.codex_driver import CodexSdkDriver, CodexCompileRequest
from twinr.agent.self_coding.sandbox.broker import SkillBrokerPolicy
from twinr.agent.self_coding.sandbox.loader_process import sandbox_loader_child_main
from twinr.agent.self_coding.sandbox.os_hardening import SandboxHardeningReport
from twinr.agent.self_coding.sandbox.policy import build_capability_broker_manifest
from twinr.agent.self_coding.sandbox.skill_runner import SelfCodingSandboxLimits, SelfCodingSandboxRunner


def _ready_session() -> RequirementsDialogueSession:
    return RequirementsDialogueSession(
        session_id="dialogue_compatibility",
        request_summary="Compile a compatibility probe skill.",
        skill_name="Compatibility Probe",
        action="Run one compatibility probe handler.",
        capabilities=("memory", "speaker", "web_search"),
        feasibility=FeasibilityResult(
            outcome=FeasibilityOutcome.YELLOW,
            summary="Needs the skill-package path.",
            suggested_target=CompileTarget.SKILL_PACKAGE,
        ),
        status=RequirementsDialogueStatus.READY_FOR_COMPILE,
        trigger_mode="push",
        trigger_conditions=("daily_0800",),
        scope={"channel": "voice"},
        constraints=(),
    )


@dataclass
class _Limits:
    cpu_seconds: int = 5
    address_space_bytes: int = 64 * 1024 * 1024
    max_open_files: int = 16


class _LoaderConnection:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []
        self.closed = False

    def fileno(self) -> int:
        return 1

    def send_bytes(self, payload: bytes) -> None:
        self.messages.append(pickle.loads(payload))

    def send(self, payload: object) -> None:
        raise AssertionError(f"unexpected rpc send: {payload!r}")

    def recv(self) -> object:
        raise AssertionError("unexpected rpc recv")

    def poll(self, timeout: float | None = None) -> bool:
        del timeout
        return False

    def close(self) -> None:
        self.closed = True


class _ParentContext:
    def __init__(self) -> None:
        self.payloads: dict[str, object] = {}
        self.spoken: list[str] = []
        self.spoken_count = 0

    def store_json(self, key: str, value: object) -> None:
        self.payloads[str(key)] = value

    def say(self, text: str) -> None:
        self.spoken.append(str(text))
        self.spoken_count = len(self.spoken)


class SelfCodingCompatibilityTests(unittest.TestCase):
    def test_broker_policy_matches_capability_manifest_allowed_methods(self) -> None:
        manifest = build_capability_broker_manifest(("memory", "speaker", "web_search"))

        policy = SkillBrokerPolicy.from_manifest(manifest)

        self.assertEqual(policy.allowed_methods, frozenset(manifest.allowed_methods))
        self.assertIn("store_json", policy.allowed_methods)
        self.assertIn("say", policy.allowed_methods)
        self.assertIn("search_web", policy.allowed_methods)

    def test_loader_child_emits_ready_and_completed_messages(self) -> None:
        from unittest.mock import patch
        from twinr.agent.self_coding.sandbox import loader_process

        connection = _LoaderConnection()
        baseline_report = SandboxHardeningReport(no_new_privs="enabled", seccomp="filter", landlock="pending")
        completed_report = SandboxHardeningReport(no_new_privs="enabled", seccomp="filter", landlock="enforced")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.object(loader_process, "apply_baseline_os_hardening", return_value=baseline_report), patch.object(
                loader_process,
                "apply_post_load_landlock",
                return_value=completed_report,
            ):
                sandbox_loader_child_main(
                    connection=connection,
                    source_text="from __future__ import annotations\n\ndef refresh(ctx):\n    return None\n",
                    entry_module="skill_main.py",
                    handler_name="refresh",
                    event_name=None,
                    materialized_root=str(root),
                    limits=_Limits(),
                )

        self.assertTrue(connection.closed)
        self.assertEqual([message["kind"] for message in connection.messages], ["loader_ready", "completed"])
        self.assertEqual(connection.messages[-1]["hardening"]["landlock"], "enforced")

    def test_sandbox_runner_executes_with_manifest_derived_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            skill_path = root / "skill_main.py"
            skill_path.write_text(
                textwrap.dedent(
                    """
                    from __future__ import annotations

                    def refresh(ctx):
                        ctx.store_json("status", {"ok": True})
                        ctx.say("compatibility-ok")
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            context = _ParentContext()
            manifest = build_capability_broker_manifest(("memory", "speaker"))
            runner = SelfCodingSandboxRunner(
                limits=SelfCodingSandboxLimits(
                    timeout_seconds=10.0,
                    cpu_seconds=5,
                    address_space_bytes=128 * 1024 * 1024,
                    max_open_files=32,
                )
            )

            result = runner.run_handler(
                owner=object(),
                context=context,
                materialized_root=root,
                entry_module="skill_main.py",
                handler_name="refresh",
                policy=SkillBrokerPolicy.from_manifest(manifest),
            )

        self.assertEqual(context.payloads["status"], {"ok": True})
        self.assertEqual(context.spoken, ["compatibility-ok"])
        self.assertEqual(result.spoken_count, 1)
        self.assertIn("landlock", result.hardening)

    def test_sdk_driver_relativizes_workspace_internal_paths_for_bridge_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request_path = root / "REQUEST.md"
            output_schema_path = root / "output_schema.json"
            marker_path = root / "sdk_payload.json"
            request_path.write_text("compile", encoding="utf-8")
            output_schema_path.write_text("{}", encoding="utf-8")
            bridge_script = root / "fake_bridge.py"
            bridge_script.write_text(
                textwrap.dedent(
                    f"""
                    import json
                    import sys
                    from pathlib import Path

                    marker_path = Path({str(marker_path)!r})
                    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
                        sys.stdout.write(json.dumps({{"ok": True, "node_version": "test"}}) + "\\n")
                        sys.exit(0)
                    payload = json.loads(sys.stdin.read())
                    marker_path.write_text(json.dumps({{
                        "workspaceRoot": payload.get("workspaceRoot"),
                        "requestPath": payload.get("requestPath"),
                        "outputSchemaPath": payload.get("outputSchemaPath"),
                    }}), encoding="utf-8")
                    events = [
                        {{"type": "thread.started", "thread_id": "thread-sdk-compat"}},
                        {{"type": "turn.started"}},
                        {{
                            "type": "item.completed",
                            "item": {{
                                "id": "item-1",
                                "type": "agent_message",
                                "text": json.dumps({{
                                    "status": "ok",
                                    "summary": "Compiled through SDK.",
                                    "review": "Ready for review.",
                                    "artifacts": [],
                                }}),
                            }},
                        }},
                        {{"type": "turn.completed", "usage": {{"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 2}}}},
                    ]
                    for event in events:
                        sys.stdout.write(json.dumps(event) + "\\n")
                        sys.stdout.flush()
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            request = CodexCompileRequest(
                job=CompileJobRecord(
                    job_id="job_sdk_compatibility",
                    skill_id="compatibility_probe",
                    skill_name="Compatibility Probe",
                    status=CompileJobStatus.QUEUED,
                    requested_target=CompileTarget.SKILL_PACKAGE,
                    spec_hash=stable_sha256("sdk-compatibility"),
                ),
                session=_ready_session(),
                prompt="compile",
                output_schema={},
                workspace_root=str(root),
                request_path=str(request_path),
                output_schema_path=str(output_schema_path),
            )
            driver = CodexSdkDriver(command=(sys.executable, "-u"), bridge_script=bridge_script, timeout_seconds=5.0)

            result = driver.run_compile(request)
            payload = json.loads(marker_path.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "ok")
        self.assertEqual(payload["workspaceRoot"], str(root))
        self.assertEqual(payload["requestPath"], "REQUEST.md")
        self.assertEqual(payload["outputSchemaPath"], "output_schema.json")


if __name__ == "__main__":
    unittest.main()
