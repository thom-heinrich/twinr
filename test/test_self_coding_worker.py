from dataclasses import replace
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.self_coding import (
    ArtifactKind,
    CompileJobStatus,
    CompileTarget,
    FeasibilityOutcome,
    FeasibilityResult,
    RequirementsDialogueSession,
    RequirementsDialogueStatus,
    SelfCodingStore,
)
from twinr.agent.self_coding.codex_driver import (
    CodexCompileArtifact,
    CodexCompileEvent,
    CodexCompileResult,
)
from twinr.agent.self_coding.worker import SelfCodingCompileWorker


class _FakeCompileDriver:
    def __init__(self, result: CodexCompileResult) -> None:
        self.result = result
        self.requests: list[object] = []

    def run_compile(self, request) -> CodexCompileResult:
        self.requests.append(request)
        return self.result


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


class SelfCodingCompileWorkerTests(unittest.TestCase):
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
                    content=json.dumps({"automation": {"name": "Read Emails"}}, indent=2),
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

        self.assertEqual(completed.status, CompileJobStatus.SOFT_LAUNCH_READY)
        self.assertEqual(len(driver.requests), 1)
        request = driver.requests[0]
        self.assertTrue((Path(request.workspace_root) / "REQUEST.md").exists())
        self.assertTrue((Path(request.workspace_root) / "skill_spec.json").exists())
        self.assertEqual(len(artifacts), 4)
        kinds = {artifact.kind for artifact in artifacts}
        self.assertIn(ArtifactKind.AUTOMATION_MANIFEST, kinds)
        self.assertIn(ArtifactKind.TEST_SUITE, kinds)
        self.assertIn(ArtifactKind.REVIEW, kinds)
        self.assertIn(ArtifactKind.LOG, kinds)
        manifest = next(artifact for artifact in artifacts if artifact.kind == ArtifactKind.AUTOMATION_MANIFEST)
        self.assertEqual(json.loads(store.read_text_artifact(manifest.artifact_id))["automation"]["name"], "Read Emails")

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

    def test_worker_reuses_existing_job_for_same_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SelfCodingStore.from_project_root(temp_dir)
            worker = SelfCodingCompileWorker(
                store=store,
                driver=_FakeCompileDriver(CodexCompileResult(status="ok", summary="done")),
            )
            session = _ready_session()

            first = worker.ensure_job_for_session(session)
            second = worker.ensure_job_for_session(replace(session, action="Read new email aloud slowly"))

        self.assertEqual(first.job_id, second.job_id)


if __name__ == "__main__":
    unittest.main()
