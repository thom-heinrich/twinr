from __future__ import annotations

from datetime import UTC, datetime
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
    RequirementsDialogueSession,
    RequirementsDialogueStatus,
    SelfCodingActivationService,
    SelfCodingHealthService,
    SelfCodingStore,
)
from twinr.agent.self_coding.activation import SelfCodingActivationService
from twinr.agent.self_coding.codex_driver import CodexCompileArtifact, CodexCompileResult
from twinr.agent.self_coding.live_acceptance import (
    CountingBackendProxy,
    MemorySpeechOutput,
    MorningBriefingAcceptanceOwner,
    build_morning_briefing_ready_session,
    run_morning_briefing_acceptance,
)
from twinr.agent.self_coding.runtime import SelfCodingSkillExecutionService
from twinr.agent.self_coding.worker import SelfCodingCompileWorker
from twinr.automations import AutomationStore


class _SkillPackageCompileDriver:
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
                                            "    research = []\n"
                                            "    for index in range(3):\n"
                                            "        result = ctx.search_web(f'topic {index + 1}')\n"
                                            "        research.append(result.answer)\n"
                                            "    summary = ctx.summarize_text('\\n'.join(research), instructions='Write one short German abstract.')\n"
                                            "    ctx.store_json('briefing', {'summary': summary})\n\n"
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
                    summary="Skill package artifact.",
                ),
            ),
        )


class _ExplodingCompileDriver:
    def run_compile(self, request, *, event_sink=None) -> CodexCompileResult:
        del request, event_sink
        raise AssertionError("acceptance runner should have reused the active skill")


class _Backend:
    def __init__(self) -> None:
        self.search_calls: list[str] = []
        self.summary_prompts: list[str] = []

    def search_live_info_with_metadata(self, question: str, *, conversation=None, location_hint=None, date_context=None):
        del conversation, location_hint, date_context
        self.search_calls.append(question)
        return type(
            "SearchResponse",
            (),
            {
                "answer": f"answer for {question}",
                "sources": ("https://example.com/1",),
                "response_id": "resp_search",
                "request_id": "req_search",
                "model": "gpt-5.2-chat-latest",
            },
        )()

    def respond_with_metadata(self, prompt: str, *, instructions=None, allow_web_search=None):
        del instructions, allow_web_search
        self.summary_prompts.append(prompt)
        return type(
            "TextResponse",
            (),
            {
                "text": "Guten Morgen. Hier ist dein Morgenbriefing.",
                "response_id": "resp_summary",
                "request_id": "req_summary",
                "model": "gpt-5.2-chat-latest",
            },
        )()


class MorningBriefingAcceptanceTests(unittest.TestCase):
    def test_acceptance_runner_compiles_executes_and_records_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SelfCodingStore.from_project_root(root)
            automation_store = AutomationStore(root / "state" / "automations.json", timezone_name="Europe/Berlin")
            activation = SelfCodingActivationService(store=store, automation_store=automation_store)
            health = SelfCodingHealthService(store=store)
            runtime = SelfCodingSkillExecutionService(store=store, health_service=health)
            worker = SelfCodingCompileWorker(store=store, driver=_SkillPackageCompileDriver())
            backend = CountingBackendProxy(_Backend())
            speech = MemorySpeechOutput()
            owner = MorningBriefingAcceptanceOwner.with_backend(
                backend,
                speech_output=speech,
                timezone_name="Europe/Berlin",
            )

            result = run_morning_briefing_acceptance(
                store=store,
                automation_store=automation_store,
                compile_worker=worker,
                activation_service=activation,
                runtime_service=runtime,
                owner=owner,
                session=build_morning_briefing_ready_session(session_id="dialogue_acceptance"),
                refresh_now=datetime(2026, 3, 16, 8, 0, tzinfo=UTC),
                delivery_now=datetime(2026, 3, 16, 8, 5, tzinfo=UTC),
                live_e2e_environment="local",
            )

            latest = store.load_live_e2e_status("morning_briefing", environment="local")

        self.assertEqual(result.job_status, "soft_launch_ready")
        self.assertEqual(result.activation_status, "active")
        self.assertEqual(result.search_call_count, 3)
        self.assertEqual(result.summary_call_count, 1)
        self.assertTrue(result.delivery_delivered)
        self.assertEqual(speech.spoken_texts, ("Guten Morgen. Hier ist dein Morgenbriefing.",))
        self.assertEqual(latest.status, "passed")
        self.assertEqual(latest.environment, "local")

    def test_acceptance_runner_reuses_active_skill_without_recompile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SelfCodingStore.from_project_root(root)
            automation_store = AutomationStore(root / "state" / "automations.json", timezone_name="Europe/Berlin")
            activation = SelfCodingActivationService(store=store, automation_store=automation_store)
            health = SelfCodingHealthService(store=store)
            runtime = SelfCodingSkillExecutionService(store=store, health_service=health)
            compile_worker = SelfCodingCompileWorker(store=store, driver=_SkillPackageCompileDriver())
            backend = CountingBackendProxy(_Backend())
            speech = MemorySpeechOutput()
            owner = MorningBriefingAcceptanceOwner.with_backend(
                backend,
                speech_output=speech,
                timezone_name="Europe/Berlin",
            )

            initial = run_morning_briefing_acceptance(
                store=store,
                automation_store=automation_store,
                compile_worker=compile_worker,
                activation_service=activation,
                runtime_service=runtime,
                owner=owner,
                session=build_morning_briefing_ready_session(session_id="dialogue_acceptance"),
                refresh_now=datetime(2026, 3, 16, 8, 0, tzinfo=UTC),
                delivery_now=datetime(2026, 3, 16, 8, 5, tzinfo=UTC),
                live_e2e_environment="local",
            )

            reuse_owner = MorningBriefingAcceptanceOwner.with_backend(
                CountingBackendProxy(_Backend()),
                speech_output=MemorySpeechOutput(),
                timezone_name="Europe/Berlin",
            )
            exploding_worker = SelfCodingCompileWorker(store=store, driver=_ExplodingCompileDriver())
            reused = run_morning_briefing_acceptance(
                store=store,
                automation_store=automation_store,
                compile_worker=exploding_worker,
                activation_service=activation,
                runtime_service=runtime,
                owner=reuse_owner,
                session=build_morning_briefing_ready_session(session_id="dialogue_acceptance"),
                refresh_now=datetime(2026, 3, 17, 8, 0, tzinfo=UTC),
                delivery_now=datetime(2026, 3, 17, 8, 5, tzinfo=UTC),
                live_e2e_environment="local",
            )

        self.assertEqual(initial.skill_id, reused.skill_id)
        self.assertEqual(initial.version, reused.version)
        self.assertTrue(reused.delivery_delivered)
        self.assertEqual(reused.search_call_count, 3)
        self.assertEqual(reused.summary_call_count, 1)


if __name__ == "__main__":
    unittest.main()
