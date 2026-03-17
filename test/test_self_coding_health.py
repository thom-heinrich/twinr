from datetime import UTC, datetime
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
from twinr.agent.self_coding.activation import SelfCodingActivationService
from twinr.agent.self_coding.health import SelfCodingHealthService
from twinr.agent.self_coding.runtime import SelfCodingSkillExecutionService
from twinr.agent.self_coding.worker import SelfCodingCompileWorker
from twinr.agent.self_coding.codex_driver import CodexCompileArtifact, CodexCompileResult
from twinr.automations import AutomationStore


def _ready_skill_package_session() -> RequirementsDialogueSession:
    return RequirementsDialogueSession(
        session_id="dialogue_health_briefing",
        request_summary="Research three topics at 08:00, prepare an abstract, and read it aloud when I enter the room.",
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


class _SkillPackageCompileDriver:
    def __init__(self, *, entry_code: str) -> None:
        self.entry_code = entry_code

    def run_compile(self, request, *, event_sink=None) -> CodexCompileResult:
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
                                "files": [{"path": "skill_main.py", "content": self.entry_code}],
                            }
                        }
                    ),
                    summary="Skill package artifact.",
                ),
            ),
        )


class _Owner:
    def __init__(self) -> None:
        self.config = type("Cfg", (), {"local_timezone_name": "Europe/Berlin"})()
        self.runtime = type("Runtime", (), {"search_provider_conversation_context": staticmethod(lambda: ())})()
        self._latest_sensor_observation_facts = {"camera": {"person_visible": True, "count_persons": 1}}
        self._night_mode = False
        self._presence_session_id = 1

    def _current_presence_session_id(self) -> int | None:
        return self._presence_session_id

    def _speak_automation_text(self, entry, text: str) -> None:
        del entry, text


class SelfCodingHealthServiceTests(unittest.TestCase):
    def test_skill_health_records_successful_runs_and_delivery(self) -> None:
        entry_code = """
from __future__ import annotations

def refresh(ctx):
    ctx.store_json('briefing', {'summary': 'Hallo'})

def deliver(ctx, event_name=None):
    payload = ctx.load_json('briefing', {}) or {}
    if payload.get('summary'):
        ctx.say(payload['summary'])
""".strip()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SelfCodingStore.from_project_root(root)
            automation_store = AutomationStore(root / "state" / "automations.json", timezone_name="Europe/Berlin")
            activation = SelfCodingActivationService(store=store, automation_store=automation_store)
            health = SelfCodingHealthService(store=store, activation_service=activation, auto_pause_failure_threshold=2)
            runtime = SelfCodingSkillExecutionService(store=store, health_service=health)
            worker = SelfCodingCompileWorker(store=store, driver=_SkillPackageCompileDriver(entry_code=entry_code))
            owner = _Owner()

            job = worker.ensure_job_for_session(_ready_skill_package_session())
            completed = worker.run_job(job.job_id)
            active = activation.confirm_activation(job_id=completed.job_id, confirmed=True)

            runtime.execute_scheduled(
                owner,
                skill_id=active.skill_id,
                version=active.version,
                trigger_id="refresh",
                now=datetime(2026, 3, 16, 8, 0, tzinfo=UTC),
            )
            runtime.execute_sensor_event(
                owner,
                skill_id=active.skill_id,
                version=active.version,
                trigger_id="deliver",
                event_name="camera.person_visible",
                now=datetime(2026, 3, 16, 8, 5, tzinfo=UTC),
            )
            snapshot = store.load_skill_health(skill_id=active.skill_id, version=active.version)

        self.assertEqual(snapshot.trigger_count, 2)
        self.assertEqual(snapshot.delivered_count, 1)
        self.assertEqual(snapshot.error_count, 0)
        self.assertEqual(snapshot.status, "healthy")

    def test_skill_health_auto_pauses_after_consecutive_failures(self) -> None:
        entry_code = """
from __future__ import annotations

def refresh(ctx):
    raise RuntimeError('search backend unavailable')

def deliver(ctx, event_name=None):
    return None
""".strip()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SelfCodingStore.from_project_root(root)
            automation_store = AutomationStore(root / "state" / "automations.json", timezone_name="Europe/Berlin")
            activation = SelfCodingActivationService(store=store, automation_store=automation_store)
            health = SelfCodingHealthService(store=store, activation_service=activation, auto_pause_failure_threshold=2)
            runtime = SelfCodingSkillExecutionService(store=store, health_service=health)
            worker = SelfCodingCompileWorker(store=store, driver=_SkillPackageCompileDriver(entry_code=entry_code))
            owner = _Owner()

            job = worker.ensure_job_for_session(_ready_skill_package_session())
            completed = worker.run_job(job.job_id)
            active = activation.confirm_activation(job_id=completed.job_id, confirmed=True)

            with self.assertRaises(RuntimeError):
                runtime.execute_scheduled(
                    owner,
                    skill_id=active.skill_id,
                    version=active.version,
                    trigger_id="refresh",
                    now=datetime(2026, 3, 16, 8, 0, tzinfo=UTC),
                )
            with self.assertRaises(RuntimeError):
                runtime.execute_scheduled(
                    owner,
                    skill_id=active.skill_id,
                    version=active.version,
                    trigger_id="refresh",
                    now=datetime(2026, 3, 16, 8, 5, tzinfo=UTC),
                )

            paused = activation.load_activation(skill_id=active.skill_id, version=active.version)
            snapshot = store.load_skill_health(skill_id=active.skill_id, version=active.version)

        self.assertEqual(paused.status.value, "paused")
        self.assertEqual(snapshot.error_count, 2)
        self.assertEqual(snapshot.consecutive_error_count, 2)
        self.assertEqual(snapshot.auto_pause_count, 1)
        self.assertEqual(snapshot.status, "auto_paused")

    def test_live_e2e_status_record_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SelfCodingStore.from_project_root(temp_dir)
            health = SelfCodingHealthService(store=store)

            health.record_live_e2e_status(
                suite_id="morning_briefing",
                environment="local",
                status="passed",
                duration_seconds=67.4,
                model="gpt-5-codex",
                reasoning_effort="high",
                details="26 passed",
            )
            latest = store.load_live_e2e_status("morning_briefing", environment="local")

        self.assertEqual(latest.status, "passed")
        self.assertEqual(latest.model, "gpt-5-codex")
        self.assertEqual(latest.reasoning_effort, "high")


if __name__ == "__main__":
    unittest.main()
