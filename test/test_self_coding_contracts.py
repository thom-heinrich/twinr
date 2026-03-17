from datetime import UTC, datetime
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test.self_coding_test_utils import stable_sha256
from twinr.agent.self_coding import (
    ActivationRecord,
    ArtifactKind,
    CapabilityAvailability,
    CapabilityRiskClass,
    CapabilityStatus,
    CompileArtifactRecord,
    CompileJobRecord,
    CompileJobStatus,
    CompileTarget,
    FeasibilityOutcome,
    FeasibilityResult,
    LearnedSkillStatus,
    RequirementsDialogueSession,
    RequirementsDialogueStatus,
    SkillSpec,
    SkillTriggerSpec,
)


class SkillContractTests(unittest.TestCase):
    def test_skill_spec_round_trips_with_generated_skill_id(self) -> None:
        trigger = SkillTriggerSpec(mode="push", conditions=("user_visible", "new_email"))
        spec = SkillSpec(
            name="Read New Emails",
            action="Read new email aloud",
            trigger=trigger,
            scope={"channel": "email", "contacts": ["family"]},
            constraints=("not_during_night_mode",),
            capabilities=("email", "speaker"),
            created_at=datetime(2026, 3, 16, 14, 0, tzinfo=UTC),
        )

        payload = spec.to_payload()
        restored = SkillSpec.from_payload(payload)

        self.assertEqual(restored.skill_id, "read_new_emails")
        self.assertEqual(restored.trigger.conditions, ("user_visible", "new_email"))
        self.assertEqual(restored.scope["channel"], "email")
        self.assertEqual(restored.capabilities, ("email", "speaker"))

    def test_feasibility_result_round_trips(self) -> None:
        result = FeasibilityResult(
            outcome=FeasibilityOutcome.YELLOW,
            summary="Capability combination is novel but still buildable.",
            reasons=("No close template found.", "Required capabilities are present."),
            missing_capabilities=(),
            suggested_target=CompileTarget.AUTOMATION_MANIFEST,
            checked_at=datetime(2026, 3, 16, 14, 5, tzinfo=UTC),
        )

        restored = FeasibilityResult.from_payload(result.to_payload())

        self.assertEqual(restored.outcome, FeasibilityOutcome.YELLOW)
        self.assertEqual(restored.suggested_target, CompileTarget.AUTOMATION_MANIFEST)
        self.assertIn("Required capabilities are present.", restored.reasons)

    def test_requirements_dialogue_session_round_trips_and_builds_skill_spec(self) -> None:
        session = RequirementsDialogueSession(
            session_id="dialogue_demo123",
            request_summary="Read new email aloud.",
            skill_name="Read Emails",
            action="Read new email aloud",
            capabilities=("email", "speaker"),
            feasibility=FeasibilityResult(
                outcome=FeasibilityOutcome.GREEN,
                summary="Fits the automation-first path.",
                suggested_target=CompileTarget.AUTOMATION_MANIFEST,
            ),
            status=RequirementsDialogueStatus.CONFIRMING,
            trigger_mode="push",
            trigger_conditions=("new_email", "user_visible"),
            scope={"channel": "email", "contacts": ["family"]},
            constraints=("not_during_night_mode",),
            answered_question_ids=("when", "what", "how"),
            answer_summaries={"what": "Only from family."},
            created_at=datetime(2026, 3, 16, 14, 10, tzinfo=UTC),
            updated_at=datetime(2026, 3, 16, 14, 12, tzinfo=UTC),
        )

        restored = RequirementsDialogueSession.from_payload(session.to_payload())
        skill_spec = restored.to_skill_spec()

        self.assertEqual(restored.status, RequirementsDialogueStatus.CONFIRMING)
        self.assertEqual(restored.current_question_id, "confirm")
        self.assertEqual(skill_spec.skill_id, "read_emails")
        self.assertEqual(skill_spec.scope["contacts"], ["family"])
        self.assertEqual(skill_spec.trigger.conditions, ("new_email", "user_visible"))

    def test_job_artifact_and_activation_round_trip(self) -> None:
        job = CompileJobRecord(
            job_id="job_demo123",
            skill_id="read_new_emails",
            skill_name="Read New Emails",
            status=CompileJobStatus.VALIDATING,
            requested_target=CompileTarget.AUTOMATION_MANIFEST,
            spec_hash=stable_sha256("spec-hash-123"),
            required_capabilities=("email", "speaker"),
            artifact_ids=("artifact_demo123",),
            created_at=datetime(2026, 3, 16, 14, 0, tzinfo=UTC),
            updated_at=datetime(2026, 3, 16, 14, 7, tzinfo=UTC),
            attempt_count=2,
            last_error="",
            metadata={"worker": "local_codex"},
        )
        artifact = CompileArtifactRecord(
            artifact_id="artifact_demo123",
            job_id="job_demo123",
            kind=ArtifactKind.AUTOMATION_MANIFEST,
            media_type="application/json",
            content_path="contents/artifact_demo123.json",
            sha256="a" * 64,
            size_bytes=128,
            summary="Compiled automation manifest.",
            created_at=datetime(2026, 3, 16, 14, 8, tzinfo=UTC),
            metadata={"target": "automation_manifest"},
        )
        activation = ActivationRecord(
            skill_id="read_new_emails",
            skill_name="Read New Emails",
            version=1,
            status=LearnedSkillStatus.ACTIVE,
            job_id="job_demo123",
            artifact_id="artifact_demo123",
            updated_at=datetime(2026, 3, 16, 14, 9, tzinfo=UTC),
            activated_at=datetime(2026, 3, 16, 14, 9, tzinfo=UTC),
            feedback_due_at=datetime(2026, 3, 19, 14, 9, tzinfo=UTC),
            metadata={"source": "soft_launch"},
        )

        restored_job = CompileJobRecord.from_payload(job.to_payload())
        restored_artifact = CompileArtifactRecord.from_payload(artifact.to_payload())
        restored_activation = ActivationRecord.from_payload(activation.to_payload())

        self.assertEqual(restored_job.status, CompileJobStatus.VALIDATING)
        self.assertEqual(restored_job.attempt_count, 2)
        self.assertEqual(restored_artifact.kind, ArtifactKind.AUTOMATION_MANIFEST)
        self.assertEqual(restored_artifact.content_path, "contents/artifact_demo123.json")
        self.assertEqual(restored_activation.status, LearnedSkillStatus.ACTIVE)
        self.assertEqual(restored_activation.feedback_due_at.day, 19)

    def test_capability_availability_properties_match_status(self) -> None:
        ready = CapabilityAvailability(
            capability_id="email",
            status=CapabilityStatus.READY,
            detail="Configured and ready.",
            metadata={"risk": CapabilityRiskClass.HIGH.value},
        )
        unconfigured = CapabilityAvailability(
            capability_id="calendar",
            status=CapabilityStatus.UNCONFIGURED,
            detail="Missing URL.",
        )

        self.assertTrue(ready.available)
        self.assertTrue(ready.configured)
        self.assertTrue(unconfigured.available)
        self.assertFalse(unconfigured.configured)


if __name__ == "__main__":
    unittest.main()
