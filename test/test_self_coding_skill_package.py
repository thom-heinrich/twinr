from datetime import UTC, datetime
from pathlib import Path
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.self_coding import (
    ArtifactKind,
    CompileJobStatus,
    CompileTarget,
    FeasibilityOutcome,
    FeasibilityResult,
    LearnedSkillStatus,
    RequirementsDialogueSession,
    RequirementsDialogueStatus,
    SelfCodingStore,
)
from twinr.agent.self_coding.activation import SelfCodingActivationService
from twinr.agent.self_coding.codex_driver import CodexCompileArtifact, CodexCompileResult
from twinr.agent.self_coding.runtime import SelfCodingSkillExecutionService
from twinr.agent.self_coding.worker import SelfCodingCompileWorker
from twinr.automations import AutomationStore


def _briefing_skill_code() -> str:
    return """
from __future__ import annotations


SEARCH_TERMS = (
    "seniorenpolitik deutschland",
    "pflege gesundheit deutschland",
    "lokale nachrichten schleswig-holstein",
)


def refresh_briefing(ctx):
    research_lines = []
    sources = []
    for search_term in SEARCH_TERMS:
        result = ctx.search_web(search_term)
        research_lines.append(f"{search_term}: {result.answer}")
        sources.extend(result.sources)
    abstract = ctx.summarize_text(
        "\\n".join(research_lines),
        "Schreibe einen kurzen deutschen Morgenabstract mit drei klaren Saetzen.",
    )
    key = f"briefing:{ctx.today_local_date()}"
    ctx.store_json(
        key,
        {
            "search_terms": list(SEARCH_TERMS),
            "research_lines": research_lines,
            "sources": sources,
            "abstract": abstract,
            "delivered_at": None,
            "delivered_date": None,
        },
    )


def deliver_briefing(ctx, *, event_name=None):
    key = f"briefing:{ctx.today_local_date()}"
    payload = ctx.load_json(key, {}) or {}
    abstract = str(payload.get("abstract") or "").strip()
    if not abstract:
        return
    if payload.get("delivered_date") == ctx.today_local_date():
        return
    if ctx.is_night_mode():
        return
    if not ctx.is_private_for_speech():
        return
    ctx.say(abstract)
    payload["delivered_at"] = ctx.now_iso()
    payload["delivered_date"] = ctx.today_local_date()
    payload["delivered_event_name"] = event_name
    ctx.store_json(key, payload)
""".strip()


def _raw_skill_package_payload(*, time_of_day: str = "08:00") -> str:
    return json.dumps(
        {
            "skill_package": {
                "name": "Morning Briefing",
                "description": "Research three topics every morning, write an abstract, and read it aloud when the user enters the room.",
                "entry_module": "skill_main.py",
                "scheduled_triggers": [
                    {
                        "trigger_id": "refresh_briefing",
                        "schedule": "daily",
                        "time_of_day": time_of_day,
                        "timezone_name": "Europe/Berlin",
                        "handler": "refresh_briefing",
                    }
                ],
                "sensor_triggers": [
                    {
                        "trigger_id": "deliver_briefing",
                        "sensor_trigger_kind": "camera_person_visible",
                        "cooldown_seconds": 30,
                        "handler": "deliver_briefing",
                    }
                ],
                "files": [
                    {
                        "path": "skill_main.py",
                        "content": _briefing_skill_code(),
                    }
                ],
            }
        }
    )


class _SkillPackageCompileDriver:
    def run_compile(self, request, *, event_sink=None) -> CodexCompileResult:
        return CodexCompileResult(
            status="ok",
            summary="Compiled a stateful skill package.",
            review="The package uses the skill runtime context and bounded triggers.",
            artifacts=(
                CodexCompileArtifact(
                    kind=ArtifactKind.SKILL_PACKAGE,
                    artifact_name="skill_package.json",
                    media_type="application/json",
                    content=_raw_skill_package_payload(),
                    summary="Briefing skill package.",
                ),
            ),
        )


def _ready_skill_package_session(session_id: str = "dialogue_morning_briefing") -> RequirementsDialogueSession:
    return RequirementsDialogueSession(
        session_id=session_id,
        request_summary="Research three morning topics at 08:00, prepare an abstract, and read it aloud when I enter the room.",
        skill_name="Morning Briefing",
        action="Research three morning topics, write an abstract, and read it aloud when I enter the room.",
        capabilities=("web_search", "llm_call", "memory", "speaker", "camera", "scheduler", "safety"),
        feasibility=FeasibilityResult(
            outcome=FeasibilityOutcome.YELLOW,
            summary="This request needs the later skill-package path.",
            suggested_target=CompileTarget.SKILL_PACKAGE,
        ),
        status=RequirementsDialogueStatus.READY_FOR_COMPILE,
        trigger_mode="push",
        trigger_conditions=("camera_person_visible", "daily_0800"),
        scope={"channel": "voice", "time_of_day": "08:00"},
        constraints=("read_once_per_morning", "quiet_at_night"),
    )


class _FakeSearchBackend:
    def __init__(self) -> None:
        self.search_calls: list[str] = []
        self.summary_calls: list[str] = []

    def search_live_info_with_metadata(
        self,
        question: str,
        *,
        conversation=None,
        location_hint=None,
        date_context=None,
    ):
        del conversation, location_hint, date_context
        self.search_calls.append(question)
        return SimpleNamespace(
            answer=f"Aktuelle Zusammenfassung fuer {question}",
            sources=(f"https://example.com/{len(self.search_calls)}",),
            response_id=f"resp_search_{len(self.search_calls)}",
            request_id=f"req_search_{len(self.search_calls)}",
            used_web_search=True,
            model="gpt-5.2-chat-latest",
            token_usage=None,
        )

    def respond_with_metadata(self, prompt: str, *, instructions=None, allow_web_search=False):
        del instructions, allow_web_search
        self.summary_calls.append(prompt)
        return SimpleNamespace(
            text="Guten Morgen. Hier ist dein kurzer Morgenabstract mit den wichtigsten drei Entwicklungen.",
            response_id="resp_summary_1",
            request_id="req_summary_1",
            used_web_search=False,
            model="gpt-5.2-chat-latest",
            token_usage=None,
        )


class _FakeSkillOwner:
    def __init__(self, backend: _FakeSearchBackend) -> None:
        self.config = SimpleNamespace(local_timezone_name="Europe/Berlin")
        self.print_backend = backend
        self.agent_provider = backend
        self.runtime = SimpleNamespace(search_provider_conversation_context=lambda: ())
        self.spoken: list[str] = []
        self._presence_session_id = 1
        self._night_mode = False
        self._latest_sensor_observation_facts = {
            "camera": {"person_visible": True, "count_persons": 1},
            "pir": {"motion_detected": True},
        }

    def _current_presence_session_id(self) -> int | None:
        return self._presence_session_id

    def _speak_automation_text(self, entry, text: str) -> None:
        del entry
        self.spoken.append(text)


class SelfCodingSkillPackageTests(unittest.TestCase):
    def test_worker_persists_canonical_skill_package_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SelfCodingStore.from_project_root(temp_dir)
            worker = SelfCodingCompileWorker(store=store, driver=_SkillPackageCompileDriver())

            job = worker.ensure_job_for_session(_ready_skill_package_session())
            completed = worker.run_job(job.job_id)
            artifacts = store.list_artifacts(job_id=job.job_id)
            package = next(artifact for artifact in artifacts if artifact.kind == ArtifactKind.SKILL_PACKAGE)
            package_payload = json.loads(store.read_text_artifact(package.artifact_id))

        self.assertEqual(completed.status, CompileJobStatus.SOFT_LAUNCH_READY)
        self.assertEqual(package_payload["schema"], "twinr_self_coding_skill_package_v1")
        self.assertEqual(package_payload["target"], "skill_package")
        self.assertEqual(package_payload["package"]["entry_module"], "skill_main.py")
        self.assertEqual(len(package_payload["package"]["scheduled_triggers"]), 1)
        self.assertEqual(len(package_payload["package"]["sensor_triggers"]), 1)
        self.assertEqual(package_payload["package"]["files"][0]["path"], "skill_main.py")

    def test_activation_materializes_and_confirms_skill_package_automations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SelfCodingStore.from_project_root(root)
            automation_store = AutomationStore(root / "state" / "automations.json", timezone_name="Europe/Berlin")
            worker = SelfCodingCompileWorker(store=store, driver=_SkillPackageCompileDriver())
            activation = SelfCodingActivationService(store=store, automation_store=automation_store)

            job = worker.ensure_job_for_session(_ready_skill_package_session())
            completed = worker.run_job(job.job_id)
            staged = activation.prepare_soft_launch(completed.job_id)
            active = activation.confirm_activation(job_id=completed.job_id, confirmed=True)

            automation_ids = tuple(staged.metadata["automation_ids"])
            staged_entries = tuple(automation_store.get(item) for item in automation_ids)
            active_enabled = tuple(bool(automation_store.get(item).enabled) for item in automation_ids if automation_store.get(item) is not None)

        self.assertEqual(staged.status, LearnedSkillStatus.SOFT_LAUNCH_READY)
        self.assertEqual(staged.metadata["artifact_kind"], "skill_package")
        self.assertEqual(len(automation_ids), 2)
        self.assertTrue(all(entry is not None for entry in staged_entries))
        self.assertTrue(any(getattr(entry.trigger, "kind", None) == "time" for entry in staged_entries if entry is not None))
        self.assertTrue(any(getattr(entry.trigger, "kind", None) == "if_then" for entry in staged_entries if entry is not None))
        self.assertEqual(active.status, LearnedSkillStatus.ACTIVE)
        self.assertEqual(active_enabled, (True, True))

    def test_skill_runtime_executes_schedule_and_delivers_once_per_day(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SelfCodingStore.from_project_root(root)
            automation_store = AutomationStore(root / "state" / "automations.json", timezone_name="Europe/Berlin")
            worker = SelfCodingCompileWorker(store=store, driver=_SkillPackageCompileDriver())
            activation = SelfCodingActivationService(store=store, automation_store=automation_store)
            runtime = SelfCodingSkillExecutionService(store=store)
            backend = _FakeSearchBackend()
            owner = _FakeSkillOwner(backend)

            job = worker.ensure_job_for_session(_ready_skill_package_session())
            completed = worker.run_job(job.job_id)
            active = activation.confirm_activation(job_id=completed.job_id, confirmed=True)

            day_one = datetime(2026, 3, 16, 8, 0, tzinfo=UTC)
            refresh = runtime.execute_scheduled(
                owner,
                skill_id=active.skill_id,
                version=active.version,
                trigger_id="refresh_briefing",
                now=day_one,
            )
            first_delivery = runtime.execute_sensor_event(
                owner,
                skill_id=active.skill_id,
                version=active.version,
                trigger_id="deliver_briefing",
                event_name="camera.person_visible",
                now=day_one,
            )
            owner._presence_session_id = 2
            second_delivery_same_day = runtime.execute_sensor_event(
                owner,
                skill_id=active.skill_id,
                version=active.version,
                trigger_id="deliver_briefing",
                event_name="camera.person_visible",
                now=day_one,
            )

            day_two = datetime(2026, 3, 17, 8, 0, tzinfo=UTC)
            runtime.execute_scheduled(
                owner,
                skill_id=active.skill_id,
                version=active.version,
                trigger_id="refresh_briefing",
                now=day_two,
            )
            third_delivery_next_day = runtime.execute_sensor_event(
                owner,
                skill_id=active.skill_id,
                version=active.version,
                trigger_id="deliver_briefing",
                event_name="camera.person_visible",
                now=day_two,
            )

        self.assertEqual(refresh["status"], "ok")
        self.assertEqual(len(backend.search_calls), 6)
        self.assertEqual(len(backend.summary_calls), 2)
        self.assertTrue(first_delivery["delivered"])
        self.assertFalse(second_delivery_same_day["delivered"])
        self.assertTrue(third_delivery_next_day["delivered"])
        self.assertEqual(len(owner.spoken), 2)
        self.assertIn("Morgenabstract", owner.spoken[0])


if __name__ == "__main__":
    unittest.main()
