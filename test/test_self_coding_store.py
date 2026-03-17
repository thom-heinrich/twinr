from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test.self_coding_test_utils import stable_sha256
from twinr.agent.self_coding import (
    ActivationRecord,
    ArtifactKind,
    CompileJobRecord,
    CompileJobStatus,
    CompileTarget,
    FeasibilityOutcome,
    FeasibilityResult,
    LearnedSkillStatus,
    RequirementsDialogueSession,
    RequirementsDialogueStatus,
    SelfCodingStore,
    self_coding_store_root,
)


class SelfCodingStoreTests(unittest.TestCase):
    def test_store_root_lives_under_state_self_coding(self) -> None:
        root = self_coding_store_root("/tmp/twinr-project")
        self.assertEqual(root, Path("/tmp/twinr-project/state/self_coding"))

    def test_save_and_list_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SelfCodingStore.from_project_root(temp_dir)
            older = CompileJobRecord(
                job_id="job_older123",
                skill_id="read_messages",
                skill_name="Read Messages",
                status=CompileJobStatus.QUEUED,
                requested_target=CompileTarget.AUTOMATION_MANIFEST,
                spec_hash=stable_sha256("old-spec"),
                created_at=datetime(2026, 3, 16, 14, 0, tzinfo=UTC),
                updated_at=datetime(2026, 3, 16, 14, 1, tzinfo=UTC),
            )
            newer = replace(
                older,
                job_id="job_newer123",
                spec_hash=stable_sha256("new-spec"),
                updated_at=datetime(2026, 3, 16, 14, 2, tzinfo=UTC),
            )

            store.save_job(older)
            store.save_job(newer)

            loaded = store.load_job("job_newer123")
            all_jobs = store.list_jobs()

        self.assertEqual(loaded.spec_hash, stable_sha256("new-spec"))
        self.assertEqual(tuple(job.job_id for job in all_jobs), ("job_newer123", "job_older123"))

    def test_write_text_artifact_and_attach_it_to_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SelfCodingStore.from_project_root(temp_dir)
            job = CompileJobRecord(
                job_id="job_artifact123",
                skill_id="read_calendar",
                skill_name="Read Calendar",
                status=CompileJobStatus.COMPILING,
                requested_target=CompileTarget.AUTOMATION_MANIFEST,
                spec_hash=stable_sha256("artifact-spec"),
            )
            store.save_job(job)

            artifact = store.write_text_artifact(
                job_id="job_artifact123",
                kind=ArtifactKind.LOG,
                text="compile started",
                summary="Compile worker log",
                metadata={"phase": "compile"},
                artifact_id="artifact_log123",
            )
            updated_job = store.append_artifact_to_job("job_artifact123", "artifact_log123")
            loaded_text = store.read_text_artifact("artifact_log123")

        self.assertEqual(artifact.artifact_id, "artifact_log123")
        self.assertEqual(loaded_text, "compile started")
        self.assertEqual(updated_job.artifact_ids, ("artifact_log123",))
        self.assertEqual(artifact.content_path, "contents/artifact_log123.txt")

    def test_save_and_list_dialogue_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SelfCodingStore.from_project_root(temp_dir)
            earlier = RequirementsDialogueSession(
                session_id="dialogue_alpha123",
                request_summary="Read new emails.",
                skill_name="Read Emails",
                action="Read new email aloud",
                capabilities=("email", "speaker"),
                feasibility=FeasibilityResult(
                    outcome=FeasibilityOutcome.GREEN,
                    summary="Fits the automation-first path.",
                    suggested_target=CompileTarget.AUTOMATION_MANIFEST,
                ),
                status=RequirementsDialogueStatus.QUESTIONING,
                current_question_id="when",
                created_at=datetime(2026, 3, 16, 14, 0, tzinfo=UTC),
                updated_at=datetime(2026, 3, 16, 14, 1, tzinfo=UTC),
            )
            later = replace(
                earlier,
                session_id="dialogue_beta123",
                updated_at=datetime(2026, 3, 16, 14, 2, tzinfo=UTC),
                current_question_id="what",
                answered_question_ids=("when",),
            )

            store.save_dialogue_session(earlier)
            store.save_dialogue_session(later)

            loaded = store.load_dialogue_session("dialogue_beta123")
            listed = store.list_dialogue_sessions()

        self.assertEqual(loaded.current_question_id, "what")
        self.assertEqual(tuple(session.session_id for session in listed), ("dialogue_beta123", "dialogue_alpha123"))

    def test_save_and_list_activation_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SelfCodingStore.from_project_root(temp_dir)
            older = ActivationRecord(
                skill_id="read_emails",
                skill_name="Read Emails",
                version=1,
                status=LearnedSkillStatus.PAUSED,
                job_id="job_read_emails_v1",
                artifact_id="artifact_read_emails_v1",
                updated_at=datetime(2026, 3, 16, 14, 5, tzinfo=UTC),
                metadata={"automation_id": "ase_read_emails_v1"},
            )
            newer = replace(
                older,
                version=2,
                status=LearnedSkillStatus.ACTIVE,
                job_id="job_read_emails_v2",
                artifact_id="artifact_read_emails_v2",
                updated_at=datetime(2026, 3, 16, 14, 6, tzinfo=UTC),
                metadata={"automation_id": "ase_read_emails_v2"},
            )

            store.save_activation(older)
            store.save_activation(newer)

            loaded = store.load_activation("read_emails", version=2)
            listed = store.list_activations(skill_id="read_emails")
            by_job = store.find_activation_for_job("job_read_emails_v2")

        self.assertEqual(loaded.status, LearnedSkillStatus.ACTIVE)
        self.assertEqual(tuple(record.version for record in listed), (2, 1))
        self.assertEqual(by_job.version, 2)


if __name__ == "__main__":
    unittest.main()
