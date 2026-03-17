from __future__ import annotations

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
    SelfCodingStore,
)
from twinr.agent.self_coding.activation import SelfCodingActivationService
from twinr.agent.self_coding.codex_driver import CodexCompileArtifact, CodexCompileResult
from twinr.agent.self_coding.retest import run_self_coding_skill_retest
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
                                "name": "Retest Probe",
                                "description": "Research three topics and speak the summary later.",
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
                                            "from __future__ import annotations\n\n"
                                            "def refresh(ctx):\n"
                                            "    lines = []\n"
                                            "    for index in range(3):\n"
                                            "        result = ctx.search_web(f'topic {index + 1}')\n"
                                            "        lines.append(result.answer)\n"
                                            "    summary = ctx.summarize_text('\\n'.join(lines), instructions='Write one short German abstract.')\n"
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
                    summary="Retest probe package.",
                ),
            ),
        )


def _ready_session() -> RequirementsDialogueSession:
    return RequirementsDialogueSession(
        session_id="dialogue_retest_probe",
        request_summary="Research three topics and read the summary aloud.",
        skill_name="Retest Probe",
        action="Research and read aloud",
        capabilities=("web_search", "llm_call", "memory", "speaker", "camera", "scheduler"),
        feasibility=FeasibilityResult(
            outcome=FeasibilityOutcome.YELLOW,
            summary="Needs skill_package.",
            suggested_target=CompileTarget.SKILL_PACKAGE,
        ),
        status=RequirementsDialogueStatus.READY_FOR_COMPILE,
        trigger_mode="push",
        trigger_conditions=("camera_person_visible", "daily_0800"),
        scope={"channel": "voice"},
        constraints=(),
    )


class _Backend:
    def search_live_info_with_metadata(self, question: str, *, conversation=None, location_hint=None, date_context=None):
        del conversation, location_hint, date_context
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
        del prompt, instructions, allow_web_search
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


class SelfCodingRetestTests(unittest.TestCase):
    def test_retest_runs_active_skill_package_and_records_live_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=sk-test-1234",
                        "OPENAI_MODEL=gpt-5-codex",
                        f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            store = SelfCodingStore.from_project_root(root)
            automation_store = AutomationStore(root / "state" / "automations.json", timezone_name="Europe/Berlin")
            worker = SelfCodingCompileWorker(store=store, driver=_SkillPackageCompileDriver())
            activation = SelfCodingActivationService(store=store, automation_store=automation_store)

            job = worker.ensure_job_for_session(_ready_session())
            completed = worker.run_job(job.job_id)
            active = activation.confirm_activation(job_id=completed.job_id, confirmed=True)

            result = run_self_coding_skill_retest(
                project_root=root,
                env_file=env_path,
                skill_id=active.skill_id,
                version=active.version,
                environment="web",
                backend_factory=lambda config: _Backend(),
            )

            latest = store.load_live_e2e_status(active.skill_id, environment="web")

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.search_call_count, 3)
        self.assertEqual(result.summary_call_count, 1)
        self.assertEqual(result.spoken_count, 1)
        self.assertTrue(result.delivered)
        self.assertEqual(latest.status, "passed")
        self.assertEqual(latest.suite_id, active.skill_id)


if __name__ == "__main__":
    unittest.main()
