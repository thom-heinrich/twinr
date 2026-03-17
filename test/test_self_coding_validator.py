import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.self_coding import (
    ArtifactKind,
    CompileJobRecord,
    CompileJobStatus,
    CompileTarget,
    FeasibilityOutcome,
    FeasibilityResult,
    RequirementsDialogueSession,
    RequirementsDialogueStatus,
)
from twinr.agent.self_coding.codex_driver import CodexCompileArtifact
from twinr.agent.self_coding.compiler import CompileArtifactValidationError, validate_compile_artifact


def _automation_session() -> RequirementsDialogueSession:
    return RequirementsDialogueSession(
        session_id="dialogue_validator_automation",
        request_summary="Read new email aloud.",
        skill_name="Read Emails",
        action="Read new email aloud.",
        capabilities=("email", "speaker", "safety"),
        feasibility=FeasibilityResult(
            outcome=FeasibilityOutcome.GREEN,
            summary="Fits the automation target.",
            suggested_target=CompileTarget.AUTOMATION_MANIFEST,
        ),
        status=RequirementsDialogueStatus.READY_FOR_COMPILE,
        trigger_mode="push",
        trigger_conditions=("new_email",),
        scope={"channel": "email"},
        constraints=("ask_first",),
    )


def _skill_session() -> RequirementsDialogueSession:
    return RequirementsDialogueSession(
        session_id="dialogue_validator_skill",
        request_summary="Research three topics at 08:00 and read the abstract aloud when I enter the room.",
        skill_name="Morning Briefing",
        action="Research three topics, write an abstract, and read it aloud when I enter the room.",
        capabilities=("web_search", "llm_call", "memory", "speaker", "camera", "scheduler", "safety"),
        feasibility=FeasibilityResult(
            outcome=FeasibilityOutcome.YELLOW,
            summary="Needs the skill-package path.",
            suggested_target=CompileTarget.SKILL_PACKAGE,
        ),
        status=RequirementsDialogueStatus.READY_FOR_COMPILE,
        trigger_mode="push",
        trigger_conditions=("camera_person_visible", "daily_0800"),
        scope={"channel": "voice", "time_of_day": "08:00"},
        constraints=("read_once_per_morning", "quiet_at_night"),
    )


class SelfCodingCompileValidatorTests(unittest.TestCase):
    def test_validator_rejects_target_artifact_kind_mismatch(self) -> None:
        job = CompileJobRecord(
            job_id="job_validator_mismatch",
            skill_id="read_emails",
            skill_name="Read Emails",
            status=CompileJobStatus.VALIDATING,
            requested_target=CompileTarget.AUTOMATION_MANIFEST,
            spec_hash="validator-mismatch",
        )
        artifact = CodexCompileArtifact(
            kind=ArtifactKind.SKILL_PACKAGE,
            artifact_name="skill_package.json",
            media_type="application/json",
            content=json.dumps(
                {
                    "skill_package": {
                        "name": "Morning Briefing",
                        "description": "desc",
                        "entry_module": "skill_main.py",
                        "scheduled_triggers": [
                            {
                                "trigger_id": "refresh",
                                "schedule": "daily",
                                "time_of_day": "08:00",
                                "timezone_name": "Europe/Berlin",
                                "handler": "refresh",
                            }
                        ],
                        "files": [{"path": "skill_main.py", "content": "def refresh(ctx):\n    return None\n"}],
                    }
                }
            ),
            summary="wrong kind",
        )

        with self.assertRaises(CompileArtifactValidationError):
            validate_compile_artifact(job=job, session=_automation_session(), artifact=artifact)

    def test_validator_canonicalizes_skill_package_target_artifact(self) -> None:
        job = CompileJobRecord(
            job_id="job_validator_skill",
            skill_id="morning_briefing",
            skill_name="Morning Briefing",
            status=CompileJobStatus.VALIDATING,
            requested_target=CompileTarget.SKILL_PACKAGE,
            spec_hash="validator-skill",
        )
        artifact = CodexCompileArtifact(
            kind=ArtifactKind.SKILL_PACKAGE,
            artifact_name="skill_package.json",
            media_type="application/json",
            content=json.dumps(
                {
                    "skill_package": {
                        "name": "Morning Briefing",
                        "description": "Research three topics and read the abstract aloud later.",
                        "entry_module": "skill_main.py",
                        "scheduled_triggers": [
                            {
                                "trigger_id": "refresh",
                                "schedule": "daily",
                                "time_of_day": "08:00",
                                "timezone_name": "Europe/Berlin",
                                "handler": "refresh",
                            }
                        ],
                        "sensor_triggers": [
                            {
                                "trigger_id": "deliver",
                                "sensor_trigger_kind": "camera_person_visible",
                                "cooldown_seconds": 30,
                                "handler": "deliver",
                            }
                        ],
                        "files": [
                            {
                                "path": "skill_main.py",
                                "content": (
                                    "def refresh(ctx):\n"
                                    "    ctx.store_json('briefing', {'summary': 'Hallo'})\n\n"
                                    "def deliver(ctx, event_name=None):\n"
                                    "    payload = ctx.load_json('briefing', {}) or {}\n"
                                    "    if payload.get('summary'):\n"
                                    "        ctx.say(payload['summary'])\n"
                                ),
                            }
                        ],
                    }
                }
            ),
            summary="skill package",
        )

        validated = validate_compile_artifact(job=job, session=_skill_session(), artifact=artifact)
        payload = json.loads(validated.content)

        self.assertEqual(payload["schema"], "twinr_self_coding_skill_package_v1")
        self.assertEqual(validated.kind, ArtifactKind.SKILL_PACKAGE)
        self.assertEqual(validated.media_type, "application/json")
        self.assertEqual(validated.metadata["artifact_kind"], "skill_package")


if __name__ == "__main__":
    unittest.main()
