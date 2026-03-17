"""Opt-in live end-to-end test for the real codex-sdk skill-package path."""

from __future__ import annotations

from datetime import UTC, datetime
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.self_coding import (
    CompileJobStatus,
    CompileTarget,
    FeasibilityOutcome,
    FeasibilityResult,
    RequirementsDialogueSession,
    RequirementsDialogueStatus,
    SelfCodingActivationService,
    SelfCodingCompileWorker,
    SelfCodingHealthService,
    SelfCodingStore,
)
from twinr.agent.self_coding.live_acceptance import (
    CountingBackendProxy,
    MemorySpeechOutput,
    MorningBriefingAcceptanceOwner,
    build_morning_briefing_ready_session,
    run_morning_briefing_acceptance,
)
from twinr.agent.self_coding.runtime import SelfCodingSkillExecutionService
from twinr.automations import AutomationStore


class _LiveSkillBackend:
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
            answer=f"Live search answer for {question}",
            sources=(f"https://example.com/{len(self.search_calls)}",),
            response_id=f"resp_live_search_{len(self.search_calls)}",
            request_id=f"req_live_search_{len(self.search_calls)}",
            used_web_search=True,
            model="gpt-5.2-chat-latest",
            token_usage=None,
        )

    def respond_with_metadata(self, prompt: str, *, instructions=None, allow_web_search=None):
        del instructions, allow_web_search
        self.summary_calls.append(prompt)
        return SimpleNamespace(
            text="Guten Morgen. Hier ist dein kurzer Morgenabstract aus den drei Recherchen.",
            response_id="resp_live_summary_1",
            request_id="req_live_summary_1",
            used_web_search=False,
            model="gpt-5.2-chat-latest",
            token_usage=None,
        )


@unittest.skipUnless(
    os.environ.get("TWINR_RUN_LIVE_CODEX_E2E") == "1",
    "Set TWINR_RUN_LIVE_CODEX_E2E=1 to run the live codex-sdk morning briefing test.",
)
class SelfCodingLiveBriefingE2ETests(unittest.TestCase):
    def test_real_codex_sdk_compiles_and_executes_morning_briefing_skill_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SelfCodingStore.from_project_root(root)
            automation_store = AutomationStore(root / "state" / "automations.json", timezone_name="Europe/Berlin")
            worker = SelfCodingCompileWorker(store=store)
            activation = SelfCodingActivationService(store=store, automation_store=automation_store)
            runtime = SelfCodingSkillExecutionService(store=store)
            backend = CountingBackendProxy(_LiveSkillBackend())
            speech_output = MemorySpeechOutput()
            owner = MorningBriefingAcceptanceOwner.with_backend(
                backend,
                speech_output=speech_output,
                timezone_name="Europe/Berlin",
            )

            result = run_morning_briefing_acceptance(
                store=store,
                automation_store=automation_store,
                compile_worker=worker,
                activation_service=activation,
                runtime_service=runtime,
                owner=owner,
                session=build_morning_briefing_ready_session(),
                refresh_now=datetime(2026, 3, 16, 8, 0, tzinfo=UTC),
                delivery_now=datetime(2026, 3, 16, 8, 5, tzinfo=UTC),
            )

            self.assertEqual(result.job_status, CompileJobStatus.SOFT_LAUNCH_READY.value)
            self.assertEqual(result.refresh_status, "ok")
            self.assertEqual(result.delivery_status, "ok")
            self.assertTrue(result.delivery_delivered)
            self.assertEqual(result.search_call_count, 3)
            self.assertGreaterEqual(result.summary_call_count, 1)
            self.assertEqual(result.spoken_count, 1)
            _record_live_e2e_success(
                suite_id="morning_briefing",
                status="passed",
                duration_seconds=None,
            )


def _record_live_e2e_success(*, suite_id: str, status: str, duration_seconds: float | None) -> None:
    record_root = os.environ.get("TWINR_SELF_CODING_LIVE_E2E_RECORD_ROOT", "").strip()
    if not record_root:
        return
    store = SelfCodingStore.from_project_root(record_root)
    environment = os.environ.get("TWINR_SELF_CODING_LIVE_E2E_ENV", "local").strip() or "local"
    SelfCodingHealthService(store=store).record_live_e2e_status(
        suite_id=suite_id,
        environment=environment,
        status=status,
        duration_seconds=duration_seconds,
        model="gpt-5-codex",
        reasoning_effort="high",
        details="Live morning briefing e2e passed.",
    )


if __name__ == "__main__":
    unittest.main()
