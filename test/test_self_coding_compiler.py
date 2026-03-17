from pathlib import Path
import json
import sys
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
from twinr.agent.self_coding.compiler import (
    AutomationManifestCompilerError,
    compile_automation_manifest_content,
)


def _ready_session() -> RequirementsDialogueSession:
    return RequirementsDialogueSession(
        session_id="dialogue_compile123",
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


class AutomationManifestCompilerTests(unittest.TestCase):
    def test_compile_manifest_wraps_valid_automation_in_activation_ready_document(self) -> None:
        job = CompileJobRecord(
            job_id="job_manifest123",
            skill_id="read_emails",
            skill_name="Read Emails",
            status=CompileJobStatus.COMPILING,
            requested_target=CompileTarget.AUTOMATION_MANIFEST,
            spec_hash=stable_sha256("manifest-spec"),
        )

        compiled = compile_automation_manifest_content(
            job=job,
            session=_ready_session(),
            raw_content=json.dumps(
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
                        "actions": [
                            {
                                "kind": "say",
                                "text": "You have a new email.",
                            }
                        ],
                    }
                }
            ),
        )

        payload = json.loads(compiled.content)

        self.assertEqual(payload["schema"], "twinr_self_coding_automation_manifest_v1")
        self.assertEqual(payload["job_id"], "job_manifest123")
        self.assertEqual(payload["skill_id"], "read_emails")
        self.assertTrue(payload["activation_policy"]["requires_confirmation"])
        self.assertFalse(payload["activation_policy"]["initial_enabled"])
        self.assertFalse(payload["automation"]["enabled"])
        self.assertEqual(payload["automation"]["automation_id"], "ase_read_emails")
        self.assertEqual(payload["automation"]["trigger"]["event_name"], "new_email")
        self.assertEqual(compiled.metadata["automation_id"], "ase_read_emails")
        self.assertEqual(compiled.metadata["trigger_kind"], "if_then")

    def test_compile_manifest_rejects_invalid_automation_payload(self) -> None:
        job = CompileJobRecord(
            job_id="job_manifestbad123",
            skill_id="read_emails",
            skill_name="Read Emails",
            status=CompileJobStatus.COMPILING,
            requested_target=CompileTarget.AUTOMATION_MANIFEST,
            spec_hash=stable_sha256("manifest-spec"),
        )

        with self.assertRaises(AutomationManifestCompilerError):
            compile_automation_manifest_content(
                job=job,
                session=_ready_session(),
                raw_content=json.dumps(
                    {
                        "automation": {
                            "name": "Read Emails",
                            "actions": [],
                        }
                    }
                ),
            )

    def test_compile_manifest_accepts_trigger_shorthand_and_action_type_alias(self) -> None:
        job = CompileJobRecord(
            job_id="job_manifestalias123",
            skill_id="probe_skill",
            skill_name="Probe Skill",
            status=CompileJobStatus.COMPILING,
            requested_target=CompileTarget.AUTOMATION_MANIFEST,
            spec_hash=stable_sha256("manifest-spec"),
        )

        compiled = compile_automation_manifest_content(
            job=job,
            session=_ready_session(),
            raw_content=json.dumps(
                {
                    "event_name": "probe",
                    "actions": [
                        {
                            "type": "say",
                            "text": "Hello",
                        }
                    ],
                }
            ),
        )

        payload = json.loads(compiled.content)

        self.assertEqual(payload["automation"]["trigger"]["kind"], "if_then")
        self.assertEqual(payload["automation"]["trigger"]["event_name"], "probe")
        self.assertEqual(payload["automation"]["actions"][0]["kind"], "say")

    def test_compile_manifest_accepts_single_action_and_message_alias(self) -> None:
        job = CompileJobRecord(
            job_id="job_manifestsingle123",
            skill_id="probe_skill",
            skill_name="Probe Skill",
            status=CompileJobStatus.COMPILING,
            requested_target=CompileTarget.AUTOMATION_MANIFEST,
            spec_hash=stable_sha256("manifest-spec"),
        )

        compiled = compile_automation_manifest_content(
            job=job,
            session=_ready_session(),
            raw_content=json.dumps(
                {
                    "automation": {
                        "trigger": {"type": "probe"},
                        "action": {"type": "say", "message": "Hello"},
                    }
                }
            ),
        )

        payload = json.loads(compiled.content)

        self.assertEqual(payload["automation"]["trigger"]["kind"], "if_then")
        self.assertEqual(payload["automation"]["trigger"]["event_name"], "probe")
        self.assertEqual(payload["automation"]["actions"][0]["kind"], "say")
        self.assertEqual(payload["automation"]["actions"][0]["text"], "Hello")


if __name__ == "__main__":
    unittest.main()
