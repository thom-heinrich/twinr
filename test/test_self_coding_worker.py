from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test.self_coding_test_utils import stable_sha256
from twinr.agent.self_coding import (
    ArtifactKind,
    CompileJobRecord,
    CompileJobStatus,
    CompileTarget,
    FeasibilityOutcome,
    FeasibilityResult,
    LocalCodexCompileDriver,
    RequirementsDialogueSession,
    RequirementsDialogueStatus,
    SelfCodingStore,
)
from twinr.agent.self_coding.codex_driver import (
    CodexCompileArtifact,
    CodexCompileEvent,
    CodexCompileProgress,
    CodexCompileRequest,
    CodexCompileResult,
    CodexDriverUnavailableError,
    CodexExecFallbackDriver,
    CodexSdkDriver,
)
from twinr.agent.self_coding.worker import SelfCodingCompileWorker


class _FakeCompileDriver:
    def __init__(self, result: CodexCompileResult) -> None:
        self.result = result
        self.requests: list[object] = []
        self.workspace_checks: list[dict[str, bool]] = []

    def run_compile(self, request, *, event_sink=None) -> CodexCompileResult:
        self.requests.append(request)
        workspace_root = Path(request.workspace_root)
        self.workspace_checks.append(
            {
                "request_md": (workspace_root / "REQUEST.md").exists(),
                "skill_spec_json": (workspace_root / "skill_spec.json").exists(),
                "dialogue_session_json": (workspace_root / "dialogue_session.json").exists(),
                "compile_job_json": (workspace_root / "compile_job.json").exists(),
                "output_schema_json": (workspace_root / "output_schema.json").exists(),
            }
        )
        if event_sink is not None:
            event_sink(
                CodexCompileEvent(kind="turn_started"),
                CodexCompileProgress(
                    driver_name=type(self).__name__,
                    event_count=1,
                    last_event_kind="turn_started",
                ),
            )
        return self.result


class _UnavailableCompileDriver:
    def run_compile(self, request, *, event_sink=None) -> CodexCompileResult:
        raise CodexDriverUnavailableError("primary compile driver unavailable")


class _DelayedUnavailableCompileDriver:
    def __init__(self) -> None:
        self.saved_sink = None

    def run_compile(self, request, *, event_sink=None) -> CodexCompileResult:
        del request
        self.saved_sink = event_sink
        raise CodexDriverUnavailableError("primary compile driver unavailable")


class _ReplayPrimaryEventFallbackDriver:
    def __init__(self, primary: _DelayedUnavailableCompileDriver) -> None:
        self.primary = primary

    def run_compile(self, request, *, event_sink=None) -> CodexCompileResult:
        del request
        del event_sink
        if self.primary.saved_sink is not None:
            self.primary.saved_sink(
                CodexCompileEvent(kind="assistant_delta", message="late primary event"),
                CodexCompileProgress(
                    driver_name=type(self.primary).__name__,
                    event_count=1,
                    last_event_kind="assistant_delta",
                ),
            )
        return CodexCompileResult(status="ok", summary="Fallback worked.")


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


def _ready_skill_package_session() -> RequirementsDialogueSession:
    return RequirementsDialogueSession(
        session_id="dialogue_briefing123",
        request_summary="Every day at 08:00 research three topics, write a short German abstract, and read it aloud when I enter the room.",
        skill_name="Morning Briefing",
        action="Research three topics, write a short German abstract, and read it aloud when I enter the room.",
        capabilities=("web_search", "llm_call", "memory", "speaker", "camera", "scheduler", "safety"),
        feasibility=FeasibilityResult(
            outcome=FeasibilityOutcome.YELLOW,
            summary="Needs the skill-package path.",
            suggested_target=CompileTarget.SKILL_PACKAGE,
        ),
        status=RequirementsDialogueStatus.READY_FOR_COMPILE,
        trigger_mode="push",
        trigger_conditions=("camera_person_visible", "daily_0800"),
        scope={"channel": "voice", "time_of_day": "08:00", "query_count": 3},
        constraints=("read_once_per_morning", "quiet_at_night"),
    )


class SelfCodingCompileWorkerTests(unittest.TestCase):
    def test_local_codex_driver_defaults_to_sdk_primary_and_exec_fallback(self) -> None:
        driver = LocalCodexCompileDriver()

        self.assertIsInstance(driver.primary, CodexSdkDriver)
        self.assertIsInstance(driver.fallback, CodexExecFallbackDriver)

    def test_local_codex_driver_falls_back_when_primary_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request_path = root / "REQUEST.md"
            schema_path = root / "output_schema.json"
            request_path.write_text("compile", encoding="utf-8")
            schema_path.write_text("{}", encoding="utf-8")
            request = CodexCompileRequest(
                job=CompileJobRecord(
                    job_id="job_fallback123",
                    skill_id="fallback_skill",
                    skill_name="Fallback Skill",
                    status=CompileJobStatus.QUEUED,
                    requested_target=CompileTarget.AUTOMATION_MANIFEST,
                    spec_hash=stable_sha256("fallback-spec"),
                ),
                session=_ready_session(),
                prompt="compile",
                output_schema={},
                workspace_root=str(root),
                request_path=str(request_path),
                output_schema_path=str(schema_path),
            )
            fallback_driver = _FakeCompileDriver(CodexCompileResult(status="ok", summary="Fallback worked."))
            driver = LocalCodexCompileDriver(primary=_UnavailableCompileDriver(), fallback=fallback_driver)

            result = driver.run_compile(request)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.summary, "Fallback worked.")
        self.assertEqual(len(fallback_driver.requests), 1)

    def test_local_codex_driver_binds_event_metadata_to_original_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request_path = root / "REQUEST.md"
            schema_path = root / "output_schema.json"
            request_path.write_text("compile", encoding="utf-8")
            schema_path.write_text("{}", encoding="utf-8")
            request = CodexCompileRequest(
                job=CompileJobRecord(
                    job_id="job_late_event123",
                    skill_id="late_event_skill",
                    skill_name="Late Event Skill",
                    status=CompileJobStatus.QUEUED,
                    requested_target=CompileTarget.AUTOMATION_MANIFEST,
                    spec_hash=stable_sha256("late-event-spec"),
                ),
                session=_ready_session(),
                prompt="compile",
                output_schema={},
                workspace_root=str(root),
                request_path=str(request_path),
                output_schema_path=str(schema_path),
            )
            primary_driver = _DelayedUnavailableCompileDriver()
            fallback_driver = _ReplayPrimaryEventFallbackDriver(primary_driver)
            driver = LocalCodexCompileDriver(primary=primary_driver, fallback=fallback_driver)

            result = driver.run_compile(request)

        delayed_event = next(event for event in result.events if event.message == "late primary event")
        self.assertEqual(delayed_event.metadata["driver_name"], "_DelayedUnavailableCompileDriver")
        self.assertEqual(delayed_event.metadata["driver_attempt"], 1)

    def test_worker_runs_job_and_persists_artifacts(self) -> None:
        compile_result = CodexCompileResult(
            status="ok",
            summary="Compiled an automation draft.",
            review="Looks safe for soft launch review.",
            events=(
                CodexCompileEvent(kind="driver_started", message="compile started"),
                CodexCompileEvent(kind="assistant_delta", message="drafting"),
            ),
            artifacts=(
                CodexCompileArtifact(
                    kind=ArtifactKind.AUTOMATION_MANIFEST,
                    artifact_name="automation_manifest.json",
                    media_type="application/json",
                    content=json.dumps(
                        {
                            "automation": {
                                "name": "Read Emails",
                                "description": "Read new mail aloud after it arrives.",
                                "trigger": {
                                    "kind": "if_then",
                                    "event_name": "new_email",
                                    "all_conditions": [],
                                    "any_conditions": [],
                                    "cooldown_seconds": 45,
                                },
                                "actions": [{"kind": "say", "text": "You have a new email."}],
                            }
                        },
                        indent=2,
                    ),
                    summary="Draft automation manifest.",
                    metadata={"target": "automation_manifest"},
                ),
                CodexCompileArtifact(
                    kind=ArtifactKind.TEST_SUITE,
                    artifact_name="tests.md",
                    media_type="text/markdown",
                    content="Test checklist",
                    summary="Generated test checklist.",
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            store = SelfCodingStore.from_project_root(temp_dir)
            driver = _FakeCompileDriver(compile_result)
            worker = SelfCodingCompileWorker(store=store, driver=driver)

            job = worker.ensure_job_for_session(_ready_session())
            completed = worker.run_job(job.job_id)
            artifacts = store.list_artifacts(job_id=job.job_id)
            manifest = next(artifact for artifact in artifacts if artifact.kind == ArtifactKind.AUTOMATION_MANIFEST)
            manifest_payload = json.loads(store.read_text_artifact(manifest.artifact_id))
            compile_status = store.load_compile_status(job.job_id)

        self.assertEqual(completed.status, CompileJobStatus.SOFT_LAUNCH_READY)
        self.assertEqual(len(driver.requests), 1)
        self.assertEqual(
            driver.workspace_checks,
            [
                {
                    "request_md": True,
                    "skill_spec_json": True,
                    "dialogue_session_json": True,
                    "compile_job_json": True,
                    "output_schema_json": True,
                }
            ],
        )
        self.assertEqual(len(artifacts), 4)
        kinds = {artifact.kind for artifact in artifacts}
        self.assertIn(ArtifactKind.AUTOMATION_MANIFEST, kinds)
        self.assertIn(ArtifactKind.TEST_SUITE, kinds)
        self.assertIn(ArtifactKind.REVIEW, kinds)
        self.assertIn(ArtifactKind.LOG, kinds)
        self.assertEqual(manifest_payload["schema"], "twinr_self_coding_automation_manifest_v1")
        self.assertEqual(manifest_payload["automation"]["name"], "Read Emails")
        self.assertFalse(manifest_payload["automation"]["enabled"])
        self.assertEqual(compile_status.phase, "completed")
        self.assertEqual(compile_status.driver_name, "_FakeCompileDriver")
        self.assertGreaterEqual(compile_status.event_count, 1)

    def test_worker_marks_job_failed_for_unsupported_compile_result(self) -> None:
        compile_result = CodexCompileResult(
            status="unsupported",
            summary="Needs the future skill-package pipeline.",
            review="Current MVP target is insufficient.",
            events=(CodexCompileEvent(kind="driver_started", message="compile started"),),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            store = SelfCodingStore.from_project_root(temp_dir)
            worker = SelfCodingCompileWorker(store=store, driver=_FakeCompileDriver(compile_result))

            job = worker.ensure_job_for_session(_ready_session())
            failed = worker.run_job(job.job_id)
            artifacts = store.list_artifacts(job_id=job.job_id)

        self.assertEqual(failed.status, CompileJobStatus.FAILED)
        self.assertIn("future skill-package pipeline", failed.last_error or "")
        self.assertTrue(any(artifact.kind == ArtifactKind.REVIEW for artifact in artifacts))
        self.assertTrue(any(artifact.kind == ArtifactKind.LOG for artifact in artifacts))

    def test_worker_marks_job_failed_for_invalid_automation_manifest(self) -> None:
        compile_result = CodexCompileResult(
            status="ok",
            summary="Produced a broken manifest.",
            artifacts=(
                CodexCompileArtifact(
                    kind=ArtifactKind.AUTOMATION_MANIFEST,
                    artifact_name="automation_manifest.json",
                    media_type="application/json",
                    content=json.dumps({"automation": {"name": "Read Emails", "actions": []}}),
                    summary="Broken manifest.",
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            store = SelfCodingStore.from_project_root(temp_dir)
            worker = SelfCodingCompileWorker(store=store, driver=_FakeCompileDriver(compile_result))

            job = worker.ensure_job_for_session(_ready_session())
            failed = worker.run_job(job.job_id)
            artifacts = store.list_artifacts(job_id=job.job_id)
            compile_status = store.load_compile_status(job.job_id)

        self.assertEqual(failed.status, CompileJobStatus.FAILED)
        self.assertIn("automation manifest", failed.last_error or "")
        self.assertFalse(any(artifact.kind == ArtifactKind.AUTOMATION_MANIFEST for artifact in artifacts))
        self.assertEqual(compile_status.phase, "failed")

    def test_worker_reuses_existing_job_for_same_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SelfCodingStore.from_project_root(temp_dir)
            worker = SelfCodingCompileWorker(
                store=store,
                driver=_FakeCompileDriver(CodexCompileResult(status="ok", summary="done")),
            )
            session = _ready_session()

            first = worker.ensure_job_for_session(session)
            second = worker.ensure_job_for_session(replace(session, answer_summaries={"when": "immediately"}))

        self.assertEqual(first.job_id, second.job_id)

    def test_worker_reuses_existing_job_when_only_created_at_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SelfCodingStore.from_project_root(temp_dir)
            worker = SelfCodingCompileWorker(
                store=store,
                driver=_FakeCompileDriver(CodexCompileResult(status="ok", summary="done")),
            )
            session = _ready_session()

            first = worker.ensure_job_for_session(session)
            second = worker.ensure_job_for_session(
                replace(
                    session,
                    created_at=session.created_at + timedelta(minutes=5),
                    updated_at=session.updated_at + timedelta(minutes=5),
                )
            )

        self.assertEqual(first.job_id, second.job_id)

    def test_worker_build_prompt_anchors_twinr_automation_contract(self) -> None:
        session = _ready_session()
        job = CompileJobRecord(
            job_id="job_prompt123",
            skill_id="read_new_emails",
            skill_name="Read Emails",
            status=CompileJobStatus.QUEUED,
            requested_target=CompileTarget.AUTOMATION_MANIFEST,
            spec_hash=stable_sha256("prompt-spec"),
        )

        prompt = SelfCodingCompileWorker._build_prompt(job, session)

        self.assertIn("Do not invent alternate manifest schemas", prompt)
        self.assertIn('"automation"', prompt)
        self.assertIn('"kind": "if_then"', prompt)
        self.assertIn('"event_name": "new_email"', prompt)
        self.assertIn('"kind": "time"', prompt)
        self.assertIn('"time_of_day": "08:00"', prompt)
        self.assertIn('"kind": "say"', prompt)
        self.assertIn("Relevant Twinr module APIs:", prompt)
        self.assertIn("ask_and_wait(question: str) -> str", prompt)
        self.assertIn("`mode`", prompt)
        self.assertIn("`conditions`", prompt)

    def test_worker_build_prompt_keeps_skill_package_context_on_ctx_api_surface(self) -> None:
        session = _ready_skill_package_session()
        job = CompileJobRecord(
            job_id="job_skill_prompt123",
            skill_id="morning_briefing",
            skill_name="Morning Briefing",
            status=CompileJobStatus.QUEUED,
            requested_target=CompileTarget.SKILL_PACKAGE,
            spec_hash=stable_sha256("skill-prompt-spec"),
        )

        prompt = SelfCodingCompileWorker._build_prompt(job, session)

        self.assertIn("ctx.search_web", prompt)
        self.assertIn("ctx.store_json", prompt)
        self.assertIn("Relevant capability modules:", prompt)
        self.assertNotIn("Relevant Twinr module APIs:", prompt)


if __name__ == "__main__":
    unittest.main()
