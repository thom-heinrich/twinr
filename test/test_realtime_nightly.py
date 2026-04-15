from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.agent.personality.evolution import PersonalityEvolutionResult
from twinr.agent.personality.learning import PersonalityHistoryProbe
from twinr.agent.personality.intelligence.models import (
    SituationalAwarenessThread,
    WorldFeedSubscription,
    WorldIntelligenceRefreshResult,
)
from twinr.agent.personality.models import ContinuityThread, PersonalityDelta, PersonalitySnapshot
from twinr.agent.workflows.realtime_runtime.nightly import TwinrNightlyOrchestrator
from twinr.agent.workflows.realtime_runtime.nightly_digest_refinement import (
    build_nightly_digest_refinement,
)
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermReflectionResultV1, LongTermSourceRefV1
from twinr.memory.reminders import ReminderEntry
from twinr.proactive.runtime.display_reserve_companion_planner import DisplayReserveCompanionPlanner
from test.test_display_reserve_companion_planner import _candidate


class _FakeNightlyBackend:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, str | None, str | None]] = []
        self.respond_calls: list[str] = []
        self.print_calls: list[str] = []

    def search_live_info_with_metadata(
        self,
        question: str,
        *,
        conversation=None,
        location_hint=None,
        date_context=None,
    ):
        del conversation
        self.search_calls.append((question, location_hint, date_context))
        if "weather" in question.lower():
            return SimpleNamespace(
                answer="Mild und trocken.",
                sources=("https://weather.example/forecast",),
            )
        return SimpleNamespace(
            answer="Lokal bleibt es ruhig. Weltweit steht Energiepolitik im Fokus.",
            sources=("https://news.example/top",),
        )

    def respond_with_metadata(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions=None,
        allow_web_search=None,
    ):
        del conversation, instructions, allow_web_search
        self.respond_calls.append(prompt)
        if "Prepare Twinr's morning news lines." in prompt:
            return SimpleNamespace(
                text="Bruecken-Sanierung bleibt in Berlin wichtig.\nHalbmarathon bringt am Wochenende ruhige Umleitungen.",
            )
        if "Prepare Twinr's calm morning news summary." in prompt:
            return SimpleNamespace(
                text="In Berlin bleiben heute vor allem Verkehrs- und Stadtservice-Themen wichtig. Fuer den Alltag relevant sind ruhigere Umleitungen und laufende Sanierungen.",
            )
        if "Select supporting URLs for Twinr's morning digest." in prompt:
            return SimpleNamespace(
                text="https://www.tagesschau.de/index~rss2.xml\nhttps://rss.dw.com/rdf/rss-en-world",
            )
        if "Prepare Twinr's morning-visible overnight insight lines." in prompt:
            return SimpleNamespace(
                text="Heute darf Twinr etwas aktiver freundlich vorausdenken.\nHeute lieber kurz und warm antworten.",
            )
        if "Prepare Twinr's morning continuity lines." in prompt:
            return SimpleNamespace(
                text="Janina und der Termin in der Augenklinik bleiben heute wichtig.\nDie ruhigen Vorlesestunden in der Stadtbibliothek koennen im Blick bleiben.",
            )
        if "spoken morning briefing" in prompt:
            return SimpleNamespace(
                text="Guten Morgen. Heute ist 2026-03-23. Das Wetter wird mild und trocken. Um 09:00 steht Blutdruck messen an.",
            )
        return SimpleNamespace(
            text="Morgenbriefing 2026-03-23\nWetter: Mild und trocken.\nTermin: 09:00 Blutdruck messen.",
        )

    def compose_print_job_with_metadata(
        self,
        *,
        conversation=None,
        focus_hint=None,
        direct_text=None,
        request_source="button",
    ):
        del conversation, focus_hint, request_source
        self.print_calls.append(direct_text or "")
        return SimpleNamespace(text=direct_text or "")


class _MissingDigestBackend(_FakeNightlyBackend):
    def respond_with_metadata(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions=None,
        allow_web_search=None,
    ):
        del prompt, conversation, instructions, allow_web_search
        raise RuntimeError("digest backend unavailable")


class _FakePersonalityLearning:
    def __init__(
        self,
        *,
        refresh_result: WorldIntelligenceRefreshResult | None,
        flush_results: tuple[PersonalityEvolutionResult | None, ...] | None = None,
        history_probe: PersonalityHistoryProbe | None = None,
    ) -> None:
        self.refresh_result = refresh_result
        self.flush_results = flush_results or (
            PersonalityEvolutionResult(snapshot=PersonalitySnapshot()),
            None,
        )
        self.flush_calls = 0
        self.history_probe = history_probe or PersonalityHistoryProbe(
            history_read_status="loaded",
            snapshot_head_status="loaded",
            delta_head_status="loaded",
            last_commit_at="2026-03-22T20:00:00Z",
        )

    def flush_pending(self) -> PersonalityEvolutionResult | None:
        self.flush_calls += 1
        index = self.flush_calls - 1
        if index >= len(self.flush_results):
            return None
        return self.flush_results[index]

    def maybe_refresh_world_intelligence(
        self,
        *,
        force: bool = False,
        search_backend: object | None = None,
    ) -> WorldIntelligenceRefreshResult | None:
        del force, search_backend
        return self.refresh_result

    def probe_history_status(self) -> PersonalityHistoryProbe:
        return self.history_probe


@dataclass
class _FakeLongTermMemory:
    reflection: LongTermReflectionResultV1
    personality_learning: _FakePersonalityLearning
    flush_ok: bool = True
    flush_calls: int = 0
    reflection_calls: int = 0
    writer: object | None = None
    multimodal_writer: object | None = None

    def flush(self, *, timeout_s: float = 2.0) -> bool:
        del timeout_s
        self.flush_calls += 1
        return self.flush_ok

    def run_reflection(
        self,
        *,
        search_backend: object | None = None,
    ) -> LongTermReflectionResultV1:
        del search_backend
        self.reflection_calls += 1
        return self.reflection


class _FakeReminderStore:
    def __init__(self, entries: tuple[ReminderEntry, ...]) -> None:
        self._entries = entries

    def load_entries(self) -> tuple[ReminderEntry, ...]:
        return self._entries


class _RecordingCompose:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, *, prompt: str, max_len: int) -> str:
        del max_len
        self.prompts.append(prompt)
        return ""


@dataclass
class _FakeWriterState:
    worker_name: str = "writer"
    pending_count: int = 0
    inflight_count: int = 0
    dropped_count: int = 0
    last_error_message: str | None = None
    accepting: bool = True
    worker_alive: bool = True


class _FakeWriter:
    def __init__(self, state: _FakeWriterState) -> None:
        self._state = state

    def snapshot_state(self) -> _FakeWriterState:
        return self._state


def _make_orchestrator(
    *,
    config: TwinrConfig,
    runtime: _FakeRuntime,
    backend: object,
    planner: DisplayReserveCompanionPlanner | None = None,
    remote_ready=lambda: True,
    background_allowed=lambda: True,
) -> TwinrNightlyOrchestrator:
    return TwinrNightlyOrchestrator(
        config=config,
        runtime=cast(Any, runtime),
        text_backend=cast(Any, backend),
        search_backend=cast(Any, backend),
        print_backend=cast(Any, backend),
        display_planner=planner,
        remote_ready=remote_ready,
        background_allowed=background_allowed,
    )


def _digest_store(orchestrator: TwinrNightlyOrchestrator):
    assert orchestrator.digest_store is not None
    return orchestrator.digest_store


def _summary_store(orchestrator: TwinrNightlyOrchestrator):
    assert orchestrator.summary_store is not None
    return orchestrator.summary_store


def _state_store(orchestrator: TwinrNightlyOrchestrator):
    assert orchestrator.state_store is not None
    return orchestrator.state_store


def _personality_status_store(orchestrator: TwinrNightlyOrchestrator):
    assert orchestrator.personality_status_store is not None
    return orchestrator.personality_status_store


@dataclass
class _FakeRuntime:
    long_term_memory: _FakeLongTermMemory
    reminder_store: _FakeReminderStore
    due_reminders: tuple[ReminderEntry, ...]

    def peek_due_reminders(self, *, limit: int = 1) -> tuple[ReminderEntry, ...]:
        return self.due_reminders[:limit]


def _reflection_result() -> LongTermReflectionResultV1:
    return LongTermReflectionResultV1(
        reflected_objects=(
            LongTermMemoryObjectV1(
                memory_id="reflection:1",
                kind="summary",
                summary="Reflection object.",
                source=LongTermSourceRefV1(source_type="reflection", event_ids=("turn:reflection",)),
                status="active",
                confidence=0.8,
                sensitivity="normal",
            ),
        ),
        created_summaries=(
            LongTermMemoryObjectV1(
                memory_id="summary:1",
                kind="summary",
                summary="Created summary.",
                source=LongTermSourceRefV1(source_type="reflection", event_ids=("turn:summary",)),
                status="active",
                confidence=0.82,
                sensitivity="normal",
            ),
        ),
        midterm_packets=(),
    )


def _awareness_refresh() -> WorldIntelligenceRefreshResult:
    return WorldIntelligenceRefreshResult(
        status="refreshed",
        refreshed=True,
        subscriptions=(
            WorldFeedSubscription(
                subscription_id="feed:tagesschau",
                label="Tagesschau",
                feed_url="https://www.tagesschau.de/index~rss2.xml",
                scope="national",
                region="DE",
                topics=("news",),
                source_page_url="https://www.tagesschau.de/",
                created_by="test",
            ),
            WorldFeedSubscription(
                subscription_id="feed:dw",
                label="DW",
                feed_url="https://rss.dw.com/rdf/rss-en-world",
                scope="global",
                region="world",
                topics=("world",),
                source_page_url="https://www.dw.com/",
                created_by="test",
            ),
        ),
        continuity_threads=(
            ContinuityThread(
                title="Janina und die Augenklinik",
                summary="Der Termin in der Augenklinik bleibt ein aktiver Begleitfaden fuer den naechsten Tag.",
                salience=0.86,
                updated_at="2026-03-22T20:10:00Z",
            ),
        ),
        awareness_threads=(
            SituationalAwarenessThread(
                thread_id="thread:local",
                title="Stadtbibliothek bekommt neue Vorlesestunden",
                summary="Ab nächster Woche gibt es zusätzliche ruhige Vormittagstermine.",
                topic="community",
                region="local",
                scope="local",
                salience=0.88,
                update_count=2,
                updated_at="2026-03-22T20:00:00Z",
            ),
        ),
        checked_at="2026-03-22T20:00:00Z",
    )


def _reminder(summary: str, *, due_at: datetime, reminder_id: str) -> ReminderEntry:
    return ReminderEntry(
        reminder_id=reminder_id,
        kind="reminder",
        summary=summary,
        due_at=due_at,
    )


def _accepted_delta_result() -> PersonalityEvolutionResult:
    return PersonalityEvolutionResult(
        snapshot=PersonalitySnapshot(),
        accepted_deltas=(
            PersonalityDelta(
                delta_id="delta:initiative:test",
                target="style.initiative",
                summary="The user explicitly asked Twinr to take a bit more initiative.",
                rationale="Explicit user-requested preference with sufficient confidence.",
                delta_value=0.11,
                confidence=0.92,
                support_count=1,
                source_signal_ids=("turn:init",),
                status="accepted",
                explicit_user_requested=True,
                review_at="2026-03-22T20:00:00Z",
            ),
        ),
    )


class TwinrNightlyOrchestratorTests(unittest.TestCase):
    def _make_config(self, root: str) -> TwinrConfig:
        return TwinrConfig(
            project_root=root,
            openai_realtime_language="de",
            nightly_orchestration_after_local="00:30",
            display_reserve_bus_nightly_after_local="00:30",
            display_reserve_bus_refresh_after_local="05:30",
        )

    def _make_planner(self, config: TwinrConfig) -> DisplayReserveCompanionPlanner:
        planner = DisplayReserveCompanionPlanner.from_config(config)
        planner.candidate_loader = lambda _config, *, local_now, max_items: (
            _candidate("ai companions"),
            _candidate("world politics"),
        )[:max_items]
        return planner

    def test_orchestrator_prepares_digest_summary_and_display_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._make_config(temp_dir)
            backend = _FakeNightlyBackend()
            runtime = _FakeRuntime(
                long_term_memory=_FakeLongTermMemory(
                    reflection=_reflection_result(),
                    personality_learning=_FakePersonalityLearning(
                        refresh_result=_awareness_refresh(),
                        flush_results=(_accepted_delta_result(), None),
                    ),
                ),
                reminder_store=_FakeReminderStore(
                    (
                        _reminder(
                            "Blutdruck messen",
                            due_at=datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
                            reminder_id="r1",
                        ),
                    )
                ),
                due_reminders=(),
            )
            planner = self._make_planner(config)
            orchestrator = _make_orchestrator(
                config=config,
                runtime=runtime,
                backend=backend,
                planner=planner,
                remote_ready=lambda: True,
                background_allowed=lambda: True,
            )

            result = orchestrator.maybe_run(
                local_now=datetime(2026, 3, 23, 1, 0, tzinfo=timezone.utc)
            )

            prepared_digest = _digest_store(orchestrator).load()
            summary = _summary_store(orchestrator).load()
            prepared_plan = planner.prepared_store.load()

        self.assertEqual(result.action, "prepared")
        self.assertIsNotNone(result.state)
        assert result.state is not None
        self.assertEqual(result.state.last_status, "ready")
        self.assertIsNotNone(prepared_digest)
        self.assertIsNotNone(summary)
        self.assertIsNotNone(prepared_plan)
        assert prepared_digest is not None
        assert summary is not None
        assert prepared_plan is not None
        self.assertEqual(prepared_digest.target_local_day, "2026-03-23")
        self.assertIn("Mild und trocken", prepared_digest.weather_summary or "")
        self.assertIn("Blutdruck messen", prepared_digest.print_text)
        self.assertEqual(summary.reflection_reflected_object_count, 1)
        self.assertEqual(summary.reflection_created_summary_count, 1)
        self.assertEqual(summary.accepted_personality_delta_count, 1)
        self.assertEqual(summary.personality_flush_status, "committed")
        self.assertEqual(summary.personality_history_read_status, "loaded")
        self.assertEqual(summary.last_personality_commit_at, "2026-03-22T20:00:00Z")
        self.assertEqual(summary.target_day_reminder_count, 1)
        self.assertEqual(summary.live_search_queries, 1)
        self.assertTrue(summary.new_insights)
        self.assertTrue(summary.continuity_shifts)
        self.assertTrue(prepared_digest.new_insights)
        self.assertTrue(prepared_digest.continuity_shifts)
        self.assertIn("heute", " ".join(summary.new_insights).casefold())
        self.assertIn("augenklinik", " ".join(summary.continuity_shifts).casefold())
        self.assertTrue(summary.operator_new_insights)
        self.assertTrue(summary.operator_continuity_shifts)
        self.assertNotIn("the user explicitly", " ".join(prepared_digest.new_insights).casefold())
        self.assertEqual(
            prepared_digest.news_sources,
            (
                "https://www.tagesschau.de/index~rss2.xml",
                "https://rss.dw.com/rdf/rss-en-world",
            ),
        )
        self.assertEqual(prepared_plan.local_day, "2026-03-23")
        self.assertEqual(runtime.long_term_memory.flush_calls, 1)
        self.assertEqual(runtime.long_term_memory.reflection_calls, 1)
        self.assertEqual(runtime.long_term_memory.personality_learning.flush_calls, 2)
        self.assertEqual(len(backend.search_calls), 1)

    def test_orchestrator_passes_configured_location_hint_into_live_search(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._make_config(temp_dir),
                openai_web_search_country="DE",
                openai_web_search_region="Berlin",
                openai_web_search_city="Berlin",
            )
            backend = _FakeNightlyBackend()
            runtime = _FakeRuntime(
                long_term_memory=_FakeLongTermMemory(
                    reflection=_reflection_result(),
                    personality_learning=_FakePersonalityLearning(
                        refresh_result=_awareness_refresh()
                    ),
                ),
                reminder_store=_FakeReminderStore(()),
                due_reminders=(),
            )
            orchestrator = _make_orchestrator(
                config=config,
                runtime=runtime,
                backend=backend,
                planner=self._make_planner(config),
                remote_ready=lambda: True,
                background_allowed=lambda: True,
            )

            orchestrator.maybe_run(
                local_now=datetime(2026, 3, 23, 1, 0, tzinfo=timezone.utc)
            )

        self.assertEqual(len(backend.search_calls), 1)
        weather_question, weather_location_hint, weather_date_context = backend.search_calls[0]
        self.assertIn("Berlin, DE", weather_question)
        self.assertEqual(weather_location_hint, "Berlin, DE")
        self.assertEqual(weather_date_context, "2026-03-23")

    def test_orchestrator_blocks_when_remote_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._make_config(temp_dir)
            backend = _FakeNightlyBackend()
            runtime = _FakeRuntime(
                long_term_memory=_FakeLongTermMemory(
                    reflection=_reflection_result(),
                    personality_learning=_FakePersonalityLearning(refresh_result=None),
                ),
                reminder_store=_FakeReminderStore(()),
                due_reminders=(),
            )
            orchestrator = _make_orchestrator(
                config=config,
                runtime=runtime,
                backend=backend,
                remote_ready=lambda: False,
                background_allowed=lambda: True,
            )

            result = orchestrator.maybe_run(
                local_now=datetime(2026, 3, 23, 1, 0, tzinfo=timezone.utc)
            )

            state = _state_store(orchestrator).load()
            personality_status = _personality_status_store(orchestrator).load()

        self.assertEqual(result.action, "blocked")
        self.assertEqual(result.reason, "remote_not_ready")
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.last_status, "blocked_remote_not_ready")
        self.assertIsNone(_digest_store(orchestrator).load())

        self.assertIsNotNone(personality_status)
        assert personality_status is not None
        self.assertEqual(personality_status.failure_stage, "required_remote_not_ready")
        self.assertEqual(personality_status.error, "required_remote_not_ready")

    def test_orchestrator_degrades_to_fallback_digest_when_text_backend_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._make_config(temp_dir)
            backend = _MissingDigestBackend()
            runtime = _FakeRuntime(
                long_term_memory=_FakeLongTermMemory(
                    reflection=_reflection_result(),
                    personality_learning=_FakePersonalityLearning(refresh_result=None),
                ),
                reminder_store=_FakeReminderStore(
                    (
                        _reminder(
                            "Medikament nehmen",
                            due_at=datetime(2026, 3, 23, 8, 30, tzinfo=timezone.utc),
                            reminder_id="r2",
                        ),
                    )
                ),
                due_reminders=(),
            )
            orchestrator = _make_orchestrator(
                config=config,
                runtime=runtime,
                backend=backend,
                planner=self._make_planner(config),
                remote_ready=lambda: True,
                background_allowed=lambda: True,
            )

            result = orchestrator.maybe_run(
                local_now=datetime(2026, 3, 23, 1, 0, tzinfo=timezone.utc)
            )

            digest = _digest_store(orchestrator).load()
            summary = _summary_store(orchestrator).load()

        self.assertEqual(result.action, "prepared")
        self.assertIsNotNone(result.state)
        assert result.state is not None
        self.assertEqual(result.state.last_status, "degraded")
        self.assertIsNotNone(digest)
        self.assertIsNotNone(summary)
        assert digest is not None
        assert summary is not None
        self.assertIn("Guten Morgen", digest.spoken_text)
        self.assertIn("Medikament nehmen", digest.print_text)
        self.assertIn("spoken_digest:provider_unavailable", summary.errors)
        self.assertIn("print_digest:provider_unavailable", summary.errors)

    def test_digest_refinement_does_not_invent_personality_or_continuity_lines_without_inputs(self) -> None:
        compose = _RecordingCompose()

        refinement = build_nightly_digest_refinement(
            compose=compose,
            language="de",
            target_day_text="2026-03-23",
            location_hint="Berlin, DE",
            raw_headline_lines=("Stadtservice bleibt ruhig.",),
            raw_live_news_summary=None,
            candidate_news_sources=("https://news.example/top",),
            raw_new_insights=(),
            raw_continuity_shifts=(),
        )

        self.assertEqual(refinement.new_insights, ())
        self.assertEqual(refinement.continuity_shifts, ())
        self.assertFalse(
            any("morning-visible overnight insight lines" in prompt for prompt in compose.prompts)
        )
        self.assertFalse(
            any("Prepare Twinr's morning continuity lines." in prompt for prompt in compose.prompts)
        )

    def test_orchestrator_marks_history_blindness_in_summary_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._make_config(temp_dir)
            backend = _FakeNightlyBackend()
            runtime = _FakeRuntime(
                long_term_memory=_FakeLongTermMemory(
                    reflection=_reflection_result(),
                    personality_learning=_FakePersonalityLearning(
                        refresh_result=None,
                        history_probe=PersonalityHistoryProbe(
                            history_read_status="remote_unavailable",
                            snapshot_head_status="loaded",
                            delta_head_status="error",
                            failure_stage="delta_payload",
                            error="LongTermRemoteUnavailableError: GET /v1/external/documents/full 503",
                        ),
                    ),
                ),
                reminder_store=_FakeReminderStore(()),
                due_reminders=(),
            )
            orchestrator = _make_orchestrator(
                config=config,
                runtime=runtime,
                backend=backend,
                planner=self._make_planner(config),
                remote_ready=lambda: True,
                background_allowed=lambda: True,
            )

            result = orchestrator.maybe_run(
                local_now=datetime(2026, 3, 23, 1, 0, tzinfo=timezone.utc)
            )

            summary = _summary_store(orchestrator).load()
            state = _state_store(orchestrator).load()
            personality_status = _personality_status_store(orchestrator).load()

        self.assertEqual(result.action, "prepared")
        self.assertEqual(result.reason, "prepared_degraded")
        self.assertIsNotNone(summary)
        self.assertIsNotNone(state)
        self.assertIsNotNone(personality_status)
        assert summary is not None
        assert state is not None
        assert personality_status is not None
        self.assertEqual(summary.failure_stage, "delta_payload")
        self.assertEqual(summary.personality_history_read_status, "remote_unavailable")
        self.assertEqual(
            summary.personality_history_error,
            "LongTermRemoteUnavailableError: GET /v1/external/documents/full 503",
        )
        self.assertIn(
            "personality_history:delta_payload:LongTermRemoteUnavailableError: GET /v1/external/documents/full 503",
            summary.errors,
        )
        self.assertEqual(state.last_status, "degraded")
        self.assertEqual(state.personality_history_read_status, "remote_unavailable")
        self.assertEqual(personality_status.history_read_status, "remote_unavailable")
        self.assertEqual(personality_status.failure_stage, "delta_payload")

    def test_orchestrator_records_explicit_long_term_flush_failure_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._make_config(temp_dir)
            backend = _FakeNightlyBackend()
            runtime = _FakeRuntime(
                long_term_memory=_FakeLongTermMemory(
                    reflection=_reflection_result(),
                    personality_learning=_FakePersonalityLearning(refresh_result=None),
                    flush_ok=False,
                    writer=_FakeWriter(_FakeWriterState(pending_count=3)),
                ),
                reminder_store=_FakeReminderStore(()),
                due_reminders=(),
            )
            orchestrator = _make_orchestrator(
                config=config,
                runtime=runtime,
                backend=backend,
                planner=self._make_planner(config),
                remote_ready=lambda: True,
                background_allowed=lambda: True,
            )

            result = orchestrator.maybe_run(
                local_now=datetime(2026, 3, 23, 1, 0, tzinfo=timezone.utc)
            )

            summary = _summary_store(orchestrator).load()

        self.assertEqual(result.action, "prepared")
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertFalse(summary.long_term_flush_ok)
        self.assertEqual(summary.failure_stage, "long_term_flush_conversation_writer_pending")
        self.assertIn(
            "long_term_flush:long_term_flush_conversation_writer_pending:pending_count=3",
            summary.errors,
        )


if __name__ == "__main__":
    unittest.main()
