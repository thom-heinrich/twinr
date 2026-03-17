from datetime import UTC, datetime, timedelta
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.self_coding import (
    ArtifactKind,
    CompileTarget,
    FeasibilityOutcome,
    FeasibilityResult,
    LearnedSkillStatus,
    RequirementsDialogueSession,
    RequirementsDialogueStatus,
    SelfCodingStore,
)
from twinr.agent.self_coding.activation import SelfCodingActivationService
from twinr.agent.self_coding.codex_driver import (
    CodexCompileArtifact,
    CodexCompileEvent,
    CodexCompileProgress,
    CodexCompileResult,
)
from twinr.agent.self_coding.runtime import SelfCodingSkillExecutionService, SelfCodingSkillRuntimeStore
from twinr.agent.self_coding.worker import SelfCodingCompileWorker
from twinr.automations import AutomationStore


class _FakeCompileDriver:
    def __init__(self, *, event_name: str, spoken_text: str) -> None:
        self.event_name = event_name
        self.spoken_text = spoken_text

    def run_compile(self, request, *, event_sink=None) -> CodexCompileResult:
        if event_sink is not None:
            event_sink(
                CodexCompileEvent(kind="turn_started"),
                CodexCompileProgress(
                    driver_name=type(self).__name__,
                    event_count=1,
                    last_event_kind="turn_started",
                ),
            )
        return CodexCompileResult(
            status="ok",
            summary="Compiled an automation draft.",
            review="Looks safe for soft launch review.",
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
                                    "event_name": self.event_name,
                                    "all_conditions": [],
                                    "any_conditions": [],
                                    "cooldown_seconds": 45,
                                },
                                "actions": [{"kind": "say", "text": self.spoken_text}],
                            }
                        }
                    ),
                    summary="Draft automation manifest.",
                ),
            ),
        )


def _ready_session(*, session_id: str, request_summary: str) -> RequirementsDialogueSession:
    return RequirementsDialogueSession(
        session_id=session_id,
        request_summary=request_summary,
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


class SelfCodingActivationServiceTests(unittest.TestCase):
    def test_prepare_confirm_and_rollback_activation_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SelfCodingStore.from_project_root(root)
            automation_store = AutomationStore(root / "state" / "automations.json", timezone_name="Europe/Berlin")
            activation = SelfCodingActivationService(store=store, automation_store=automation_store)

            worker_v1 = SelfCodingCompileWorker(
                store=store,
                driver=_FakeCompileDriver(event_name="new_email", spoken_text="You have a new email."),
            )
            job_v1 = worker_v1.ensure_job_for_session(
                _ready_session(session_id="dialogue_reademails_v1", request_summary="Read new email aloud.")
            )
            completed_v1 = worker_v1.run_job(job_v1.job_id)

            soft_launch_v1 = activation.prepare_soft_launch(completed_v1.job_id)
            staged_v1 = automation_store.get(str(soft_launch_v1.metadata["automation_id"]))

            self.assertEqual(soft_launch_v1.version, 1)
            self.assertEqual(soft_launch_v1.status, LearnedSkillStatus.SOFT_LAUNCH_READY)
            self.assertIsNotNone(staged_v1)
            assert staged_v1 is not None
            self.assertFalse(staged_v1.enabled)

            active_v1 = activation.confirm_activation(job_id=completed_v1.job_id, confirmed=True)
            active_entry_v1 = automation_store.get(str(active_v1.metadata["automation_id"]))

            self.assertEqual(active_v1.status, LearnedSkillStatus.ACTIVE)
            self.assertTrue(active_entry_v1.enabled)
            assert active_v1.feedback_due_at is not None
            self.assertGreaterEqual(active_v1.feedback_due_at, active_v1.activated_at + timedelta(hours=71))

            worker_v2 = SelfCodingCompileWorker(
                store=store,
                driver=_FakeCompileDriver(event_name="vip_email", spoken_text="A family email arrived."),
            )
            job_v2 = worker_v2.ensure_job_for_session(
                _ready_session(session_id="dialogue_reademails_v2", request_summary="Read family email aloud.")
            )
            completed_v2 = worker_v2.run_job(job_v2.job_id)
            soft_launch_v2 = activation.prepare_soft_launch(completed_v2.job_id)
            active_v2 = activation.confirm_activation(job_id=completed_v2.job_id, confirmed=True)
            rolled_back = activation.rollback_activation(skill_id="read_emails")

            paused_v2 = activation.load_activation(skill_id="read_emails", version=2)
            restored_v1 = activation.load_activation(skill_id="read_emails", version=1)
            restored_entry_v1 = automation_store.get(str(restored_v1.metadata["automation_id"]))
            paused_entry_v2 = automation_store.get(str(paused_v2.metadata["automation_id"]))

        self.assertEqual(soft_launch_v2.version, 2)
        self.assertEqual(active_v2.status, LearnedSkillStatus.ACTIVE)
        self.assertEqual(paused_v2.status, LearnedSkillStatus.PAUSED)
        self.assertEqual(rolled_back.version, 1)
        self.assertEqual(rolled_back.status, LearnedSkillStatus.ACTIVE)
        self.assertTrue(restored_entry_v1.enabled)
        self.assertFalse(paused_entry_v2.enabled)

    def test_pause_and_reactivate_activation_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SelfCodingStore.from_project_root(root)
            automation_store = AutomationStore(root / "state" / "automations.json", timezone_name="Europe/Berlin")
            activation = SelfCodingActivationService(store=store, automation_store=automation_store)

            worker = SelfCodingCompileWorker(
                store=store,
                driver=_FakeCompileDriver(event_name="new_email", spoken_text="You have a new email."),
            )
            job = worker.ensure_job_for_session(
                _ready_session(session_id="dialogue_pause_v1", request_summary="Read new email aloud.")
            )
            completed = worker.run_job(job.job_id)
            active = activation.confirm_activation(job_id=completed.job_id, confirmed=True)

            paused = activation.pause_activation(skill_id=active.skill_id, version=active.version, reason="operator_pause")
            reactivated = activation.reactivate_activation(skill_id=active.skill_id, version=active.version)
            entry = automation_store.get(str(active.metadata["automation_id"]))

        self.assertEqual(paused.status, LearnedSkillStatus.PAUSED)
        self.assertEqual(paused.metadata["pause_reason"], "operator_pause")
        self.assertEqual(reactivated.status, LearnedSkillStatus.ACTIVE)
        self.assertTrue(entry.enabled)

    def test_cleanup_retires_paused_skill_package_and_removes_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SelfCodingStore.from_project_root(root)
            runtime_store = SelfCodingSkillRuntimeStore(store.root)
            automation_store = AutomationStore(root / "state" / "automations.json", timezone_name="Europe/Berlin")
            activation = SelfCodingActivationService(store=store, automation_store=automation_store)
            runtime = SelfCodingSkillExecutionService(store=store, runtime_store=runtime_store)

            class _SkillPackageDriver:
                def run_compile(self, request, *, event_sink=None) -> CodexCompileResult:
                    del request, event_sink
                    return CodexCompileResult(
                        status="ok",
                        summary="Compiled a skill package.",
                        artifacts=(
                            CodexCompileArtifact(
                                kind=ArtifactKind.SKILL_PACKAGE,
                                artifact_name="skill_package.json",
                                media_type="application/json",
                                content=json.dumps(
                                    {
                                        "skill_package": {
                                            "name": "Cleanup Probe",
                                            "description": "Persist a bit of runtime state.",
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
                                            "sensor_triggers": [],
                                            "files": [
                                                {
                                                    "path": "skill_main.py",
                                                    "content": (
                                                        "from __future__ import annotations\n\n"
                                                        "def refresh(ctx):\n"
                                                        "    ctx.store_json('value', {'ok': True})\n"
                                                    ),
                                                }
                                            ],
                                        }
                                    }
                                ),
                                summary="Cleanup probe package.",
                            ),
                        ),
                    )

            class _Owner:
                def __init__(self) -> None:
                    self.config = type("_Config", (), {"local_timezone_name": "Europe/Berlin"})()
                    self.runtime = type("_Runtime", (), {"search_provider_conversation_context": lambda self: ()})()

            worker = SelfCodingCompileWorker(store=store, driver=_SkillPackageDriver())
            job = worker.ensure_job_for_session(
                RequirementsDialogueSession(
                    session_id="dialogue_cleanup_probe",
                    request_summary="Persist some skill state and then retire the version.",
                    skill_name="Cleanup Probe",
                    action="Store state once per day",
                    capabilities=("memory", "scheduler"),
                    feasibility=FeasibilityResult(
                        outcome=FeasibilityOutcome.YELLOW,
                        summary="Needs skill_package.",
                        suggested_target=CompileTarget.SKILL_PACKAGE,
                    ),
                    status=RequirementsDialogueStatus.READY_FOR_COMPILE,
                    trigger_mode="push",
                    trigger_conditions=("daily_0800",),
                    scope={"channel": "voice"},
                    constraints=(),
                )
            )
            completed = worker.run_job(job.job_id)
            active = activation.confirm_activation(job_id=completed.job_id, confirmed=True)
            runtime.execute_scheduled(
                _Owner(),
                skill_id=active.skill_id,
                version=active.version,
                trigger_id="refresh",
                now=datetime(2026, 3, 17, 8, 0, tzinfo=UTC),
            )
            paused = activation.pause_activation(skill_id=active.skill_id, version=active.version)

            materialized_root = runtime_store.materialized_dir / active.skill_id / f"v{active.version}"
            state_path = runtime_store.state_dir / active.skill_id / f"v{active.version}.json"
            self.assertTrue(materialized_root.exists())
            self.assertTrue(state_path.exists())

            cleaned = activation.cleanup_activation(
                skill_id=paused.skill_id,
                version=paused.version,
                runtime_store=runtime_store,
            )

            self.assertEqual(cleaned.status, LearnedSkillStatus.RETIRED)
            self.assertFalse(materialized_root.exists())
            self.assertFalse(state_path.exists())
            for automation_id in cleaned.metadata["automation_ids"]:
                self.assertIsNone(automation_store.get(str(automation_id)))


if __name__ == "__main__":
    unittest.main()
