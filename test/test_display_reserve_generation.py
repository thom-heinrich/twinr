from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.agent.personality.models import (
    ConversationStyleProfile,
    HumorProfile,
    PersonalitySnapshot,
    PersonalityTrait,
    WorldSignal,
)
from twinr.providers.openai import OpenAIBackend
from twinr.proactive.runtime.display_reserve_generation import (
    DisplayReserveCopyGenerator,
    DisplayReserveGenerationError,
)


class _FakeResponsesAPI:
    def __init__(self, response: object | Exception | list[object | Exception]) -> None:
        if isinstance(response, list):
            self._responses = list(response)
        else:
            self._responses = [response]
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeBackend:
    def __init__(self, *, response: object | Exception | list[object | Exception], config: TwinrConfig) -> None:
        self.config = config
        self.responses_api = _FakeResponsesAPI(response)
        self._client = SimpleNamespace(responses=self.responses_api)

    def _build_response_request(
        self,
        prompt: str,
        *,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int | None = None,
        prompt_cache_scope: str | None = None,
    ) -> dict[str, object]:
        return {
            "model": model,
            "input": prompt,
            "instructions": instructions,
            "allow_web_search": allow_web_search,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            "prompt_cache_scope": prompt_cache_scope,
        }

    def _extract_output_text(self, response: object) -> str:
        return str(getattr(response, "output_text", "") or "")

    def _create_response(self, request: dict[str, object], *, operation: str) -> object:
        del operation
        return self._client.responses.create(**request)


def _candidate() -> AmbientDisplayImpulseCandidate:
    return AmbientDisplayImpulseCandidate(
        topic_key="ai companions",
        title="AI companions",
        source="world",
        action="invite_follow_up",
        attention_state="shared_thread",
        salience=0.84,
        eyebrow="",
        headline="Was meinst du dazu?",
        body="Es geht um AI companions.",
        symbol="question",
        accent="warm",
        reason="shared_thread_invite",
        generation_context={
            "candidate_family": "mindshare",
            "display_anchor": "AI companions",
            "hook_hint": "Die Systeme wirken alltagstauglicher und persoenlicher.",
            "topic_summary": "Persoenliche KI-Begleiter mit mehr Alltagsnaehe.",
            "card_intent": {
                "topic_semantics": "oeffentliches Thema zu KI-Begleitern",
                "statement_intent": "Twinr soll eine konkrete Beobachtung zu KI-Begleitern machen.",
                "cta_intent": "Zu einer echten Meinung oder Einordnung einladen.",
                "relationship_stance": "ruhig beobachtend mit leichter Haltung",
            },
            "display_goal": "open_positive_conversation",
            "ambient_learning": {
                "topic_state": "pulling",
                "topic_score": 0.68,
                "topic_repetition_pressure": 0.16,
                "family_state": "pulling",
                "family_score": 0.31,
                "action_score": 0.24,
            },
        },
    )


class DisplayReserveCopyGeneratorTests(unittest.TestCase):
    def test_rewrites_candidates_with_llm_generated_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                openai_api_key="test-key",
                display_reserve_generation_model="gpt-4o-mini",
                display_reserve_generation_timeout_seconds=5.5,
                display_reserve_generation_variants_per_candidate=3,
            )
            backend = _FakeBackend(
                config=config,
                response=[
                    SimpleNamespace(
                        output_parsed={
                            "items": [
                                {
                                    "topic_key": "ai companions",
                                    "variants": [
                                        {
                                            "headline": "Ich habe heute etwas zu KI-Begleitern gelesen.",
                                            "body": "Was denkst du darueber?",
                                        },
                                        {
                                            "headline": "KI-Begleiter tauchen gerade wieder oft auf.",
                                            "body": "Ist das fuer dich eher spannend oder komisch?",
                                        },
                                        {
                                            "headline": "Bei KI-Begleitern wird es gerade alltagsnah.",
                                            "body": "Soll ich dir sagen, was daran haengen bleibt?",
                                        },
                                    ],
                                }
                            ]
                        }
                    ),
                    SimpleNamespace(
                        output_parsed={
                            "items": [
                                {
                                    "topic_key": "ai companions",
                                    "headline": "Ich habe heute etwas zu KI-Begleitern gelesen.",
                                    "body": "Was denkst du darueber?",
                                }
                            ]
                        }
                    ),
                ],
            )
            generator = DisplayReserveCopyGenerator(
                backend_factory=lambda _config: cast(OpenAIBackend, backend)
            )
            snapshot = PersonalitySnapshot(
                core_traits=(
                    PersonalityTrait(
                        name="calm",
                        summary="ruhig, aufmerksam und unaufgeregt",
                        weight=0.92,
                    ),
                ),
                style_profile=ConversationStyleProfile(verbosity=0.36, initiative=0.67),
                humor_profile=HumorProfile(
                    style="trocken",
                    summary="leichter trockener Humor",
                    intensity=0.32,
                ),
                world_signals=(
                    WorldSignal(
                        topic="AI companions",
                        summary="Neue Bewegung rund um persoenliche KI-Begleiter.",
                        salience=0.88,
                    ),
                ),
            )

            rewritten = generator.rewrite_candidates(
                config=config,
                snapshot=snapshot,
                candidates=(_candidate(),),
                local_now=datetime(2026, 3, 22, 10, 30, tzinfo=timezone.utc),
            )

        self.assertEqual(rewritten[0].headline, "Ich habe heute etwas zu KI-Begleitern gelesen.")
        self.assertEqual(rewritten[0].body, "Was denkst du darueber?")
        self.assertEqual(len(backend.responses_api.calls), 2)
        variant_request = backend.responses_api.calls[0]
        selection_request = backend.responses_api.calls[1]
        variant_instructions = cast(str, variant_request["instructions"])
        variant_input = cast(str, variant_request["input"])
        selection_input = cast(str, selection_request["input"])
        self.assertEqual(variant_request["model"], "gpt-4o-mini")
        self.assertEqual(variant_request["timeout"], 5.5)
        self.assertEqual(variant_request["prompt_cache_scope"], "display_reserve_generation_variants")
        self.assertEqual(selection_request["prompt_cache_scope"], "display_reserve_generation_selection")
        self.assertIn("rechte Display-Spalte", variant_instructions)
        self.assertIn("kleine Karte auf einem Screen", variant_instructions)
        self.assertIn("sofort verstehen worum es geht", variant_instructions)
        self.assertIn("Lust haben, mit Twinr in Interaktion zu gehen", variant_instructions)
        self.assertIn("deutscher Muttersprachler", variant_instructions)
        self.assertIn("echten Reaktionen auf fruehere angezeigte Karten", variant_instructions)
        self.assertIn("Ich halte den Faden kurz offen.", variant_instructions)
        self.assertIn("Wollen wir das kurz aufziehen?", variant_instructions)
        self.assertIn("Ich habe heute etwas zu X gelesen.", variant_instructions)
        self.assertIn("In X ist heute wohl etwas passiert.", variant_instructions)
        self.assertIn("Bei Weltpolitik bin ich noch nicht fertig mit dem Staunen.", variant_instructions)
        self.assertIn("Was denkst du darueber?", variant_instructions)
        self.assertIn("Hast du davon schon gehoert?", variant_instructions)
        self.assertIn("Willst du?.", variant_instructions)
        self.assertIn("Die body-Zeile muss als vollstaendiger, grammatisch sauberer Satz", variant_instructions)
        self.assertIn("X taucht bei dir oefter auf", variant_instructions)
        self.assertIn("klaren Themenanker", variant_instructions)
        self.assertIn("wie Kundenservice", variant_instructions)
        self.assertIn("card_intent als semantische Kartenspezifikation", variant_input)
        self.assertIn("beschreibt statement_intent die Aussage der Headline", variant_input)
        self.assertIn("family_examples", variant_input)
        self.assertIn("quality_rubric", variant_input)
        self.assertIn("copy_family", variant_input)
        self.assertIn("Nutze family_examples als positive Gold-Beispiele", variant_instructions)
        self.assertIn("Nutze quality_rubric schon im Writer-Pass", variant_instructions)
        self.assertIn("display_goal invite_user_discovery", variant_instructions)
        self.assertIn(
            "Basisinfos, Ansprache, Interessen, Routinen, No-Gos, Gelerntes oder Einrichtung",
            variant_instructions,
        )
        self.assertIn("display_goal call_back_to_earlier_conversation", variant_instructions)
        self.assertIn("nicht wie Stoerungsmeldung, Support-Ticket oder technische Diagnose", variant_instructions)
        self.assertIn("Die headline muss fuer sich stehend als klare Aussage funktionieren.", variant_instructions)
        self.assertIn("vollstaendiger, natuerlicher deutscher Aussagesatz mit finitem Verb", variant_instructions)
        self.assertIn("Wie ich dich ansprechen soll.", variant_instructions)
        self.assertIn("Mich interessiert, wer dir wichtig ist.", variant_instructions)
        self.assertIn("Die body-Zeile ist der Call to Action.", variant_instructions)
        self.assertIn("Erzeuge fuer jeden Kandidaten genau 3 unterschiedliche Varianten.", variant_instructions)
        self.assertIn("\"variants_per_candidate\":3", variant_input)
        self.assertIn("\"context_summary\":", variant_input)
        self.assertIn("\"hook_hint\":", variant_input)
        self.assertIn("\"card_intent\":", variant_input)
        self.assertIn("\"topic_anchor\":", variant_input)
        self.assertIn("\"pickup_signal\":", variant_input)
        self.assertIn("\"voice_de\":", variant_input)
        self.assertNotIn("\"context\":", variant_input)
        self.assertNotIn("fallback_headline", variant_input)
        self.assertNotIn("fallback_body", variant_input)
        self.assertIn("Waehle fuer jeden Kandidaten aus mehreren Vorschlaegen", selection_input)
        self.assertIn("quality_rubric", selection_input)
        self.assertIn("family_examples", selection_input)
        self.assertIn("Ich habe heute etwas zu KI-Begleitern gelesen.", selection_input)
        self.assertIn("Ist das fuer dich eher spannend oder komisch?", selection_input)

    def test_rewrite_candidates_with_trace_reports_stage_attempts_and_final_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                openai_api_key="test-key",
                display_reserve_generation_model="gpt-4o-mini",
                display_reserve_generation_variants_per_candidate=3,
            )
            backend = _FakeBackend(
                config=config,
                response=[
                    SimpleNamespace(
                        output_parsed={
                            "items": [
                                {
                                    "topic_key": "ai companions",
                                    "variants": [
                                        {
                                            "headline": "Ich habe heute etwas zu KI-Begleitern gelesen.",
                                            "body": "Was denkst du darueber?",
                                        },
                                        {
                                            "headline": "KI-Begleiter tauchen gerade wieder auf.",
                                            "body": "Ist das fuer dich eher praktisch oder seltsam?",
                                        },
                                        {
                                            "headline": "Heute geht es bei KI oft um Begleiter.",
                                            "body": "Hast du davon schon gehoert?",
                                        },
                                    ],
                                }
                            ]
                        }
                    ),
                    SimpleNamespace(
                        output_parsed={
                            "items": [
                                {
                                    "topic_key": "ai companions",
                                    "headline": "Ich habe heute etwas zu KI-Begleitern gelesen.",
                                    "body": "Was denkst du darueber?",
                                }
                            ]
                        }
                    ),
                ],
            )
            generator = DisplayReserveCopyGenerator(
                backend_factory=lambda _config: cast(OpenAIBackend, backend)
            )

            rewritten, trace = generator.rewrite_candidates_with_trace(
                config=config,
                snapshot=None,
                candidates=(_candidate(),),
                local_now=datetime(2026, 3, 22, 10, 30, tzinfo=timezone.utc),
            )

        self.assertEqual(rewritten[0].headline, "Ich habe heute etwas zu KI-Begleitern gelesen.")
        self.assertFalse(trace.bypassed)
        self.assertEqual(trace.model, "gpt-4o-mini")
        self.assertEqual(trace.variants_per_candidate, 3)
        self.assertEqual(len(trace.stage_attempts), 2)
        self.assertTrue(all(attempt.succeeded for attempt in trace.stage_attempts))
        self.assertEqual(trace.stage_attempts[0].stage_name, "variant generation")
        self.assertEqual(trace.stage_attempts[1].stage_name, "selection")
        self.assertEqual(len(trace.selections), 1)
        self.assertEqual(trace.selections[0].copy_family, "world")
        self.assertEqual(len(trace.selections[0].variants), 3)
        self.assertEqual(
            trace.selections[0].final_copy.headline,
            "Ich habe heute etwas zu KI-Begleitern gelesen.",
        )

    def test_raises_when_generation_fails_instead_of_falling_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                openai_api_key="test-key",
                display_reserve_generation_variants_per_candidate=3,
            )
            generator = DisplayReserveCopyGenerator(
                backend_factory=lambda _config: cast(
                    OpenAIBackend,
                    _FakeBackend(
                    config=config,
                    response=RuntimeError("boom"),
                    ),
                )
            )
            candidate = _candidate()

            with self.assertRaises(DisplayReserveGenerationError) as raised:
                generator.rewrite_candidates(
                    config=config,
                    snapshot=None,
                    candidates=(candidate,),
                    local_now=datetime(2026, 3, 22, 10, 30, tzinfo=timezone.utc),
                )

        self.assertEqual(raised.exception.batch_index, 1)
        self.assertEqual(raised.exception.model, config.default_model)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)

    def test_accepts_top_level_array_text_when_sdk_parsed_output_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                openai_api_key="test-key",
                display_reserve_generation_variants_per_candidate=3,
            )
            backend = _FakeBackend(
                config=config,
                response=[
                    SimpleNamespace(
                        output_parsed={
                            "items": [
                                {
                                    "topic_key": "ai companions",
                                    "variants": [
                                        {
                                            "headline": "Ich habe heute etwas zu KI-Begleitern gelesen.",
                                            "body": "Was denkst du darueber?",
                                        },
                                        {
                                            "headline": "KI-Begleiter kommen gerade wieder hoch.",
                                            "body": "Hast du das auch mitbekommen?",
                                        },
                                        {
                                            "headline": "KI-Begleiter wirken gerade alltagsnah.",
                                            "body": "Soll ich dir kurz sagen, was daran haengt?",
                                        },
                                    ],
                                }
                            ]
                        }
                    ),
                    SimpleNamespace(
                        output_parsed=None,
                        output_text='[{"topic_key":"ai companions","headline":"KI-Begleiter: was tut sich da?","body":"Ich habe das ruhig im Blick."}]',
                    ),
                ],
            )
            generator = DisplayReserveCopyGenerator(
                backend_factory=lambda _config: cast(OpenAIBackend, backend)
            )
            candidate = _candidate()

            rewritten = generator.rewrite_candidates(
                config=config,
                snapshot=None,
                candidates=(candidate,),
                local_now=datetime(2026, 3, 22, 10, 30, tzinfo=timezone.utc),
            )

        self.assertEqual(rewritten[0].headline, "KI-Begleiter: was tut sich da?")
        self.assertEqual(rewritten[0].body, "Ich habe das ruhig im Blick.")

    def test_accepts_content_level_parsed_output_when_top_level_parsed_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                openai_api_key="test-key",
                display_reserve_generation_variants_per_candidate=3,
            )
            backend = _FakeBackend(
                config=config,
                response=[
                    SimpleNamespace(
                        output_parsed={
                            "items": [
                                {
                                    "topic_key": "ai companions",
                                    "variants": [
                                        {
                                            "headline": "Ich habe heute etwas zu KI-Begleitern gelesen.",
                                            "body": "Was denkst du darueber?",
                                        },
                                        {
                                            "headline": "KI-Begleiter tauchen gerade wieder auf.",
                                            "body": "Ist das fuer dich eher praktisch oder seltsam?",
                                        },
                                        {
                                            "headline": "Bei KI-Begleitern tut sich gerade etwas.",
                                            "body": "Soll ich dir sagen, was daran spannend ist?",
                                        },
                                    ],
                                }
                            ]
                        }
                    ),
                    SimpleNamespace(
                        output_parsed=None,
                        output=[
                            SimpleNamespace(
                                content=[
                                    SimpleNamespace(
                                        parsed={
                                            "items": [
                                                {
                                                    "topic_key": "ai companions",
                                                    "headline": "KI-Begleiter werden alltagstauglicher.",
                                                    "body": "Wollen wir kurz darueber reden?",
                                                }
                                            ]
                                        }
                                    )
                                ]
                            )
                        ],
                        output_text='{"items":[]}',
                    ),
                ],
            )
            generator = DisplayReserveCopyGenerator(
                backend_factory=lambda _config: cast(OpenAIBackend, backend)
            )
            candidate = _candidate()

            rewritten = generator.rewrite_candidates(
                config=config,
                snapshot=None,
                candidates=(candidate,),
                local_now=datetime(2026, 3, 22, 10, 30, tzinfo=timezone.utc),
            )

        self.assertEqual(rewritten[0].headline, "KI-Begleiter werden alltagstauglicher.")
        self.assertEqual(rewritten[0].body, "Wollen wir kurz darueber reden?")

    def test_retries_once_with_more_budget_and_no_reasoning_after_token_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                openai_api_key="test-key",
                display_reserve_generation_timeout_seconds=6.0,
                display_reserve_generation_max_output_tokens=900,
                display_reserve_generation_variants_per_candidate=3,
            )
            candidates = (
                _candidate(),
                replace(
                    _candidate(),
                    topic_key="world politics",
                    title="world politics",
                    headline="Fallback world",
                    body="Body world",
                    generation_context={
                        "candidate_family": "relationship",
                        "display_anchor": "world politics",
                        "hook_hint": "Da ist gerade einiges in Bewegung.",
                        "topic_summary": "Weltpolitik mit mehreren offenen Linien.",
                    },
                ),
            )
            backend = _FakeBackend(
                config=config,
                response=[
                    SimpleNamespace(
                        output_parsed=None,
                        output_text='{"items":[{"topic_key":"ai companions","variants":[{"headline":"Ich habe heute etwas zu KI-B',
                        status="incomplete",
                        incomplete_details={"reason": "max_output_tokens"},
                    ),
                    SimpleNamespace(
                        output_parsed={
                            "items": [
                                {
                                    "topic_key": "ai companions",
                                    "variants": [
                                        {
                                            "headline": "Ich habe heute etwas zu KI-Begleitern gelesen.",
                                            "body": "Was denkst du darueber?",
                                        },
                                        {
                                            "headline": "KI-Begleiter werden alltagstauglicher.",
                                            "body": "Wollen wir kurz darueber reden?",
                                        },
                                        {
                                            "headline": "KI-Begleiter tauchen gerade wieder auf.",
                                            "body": "Ist das fuer dich eher praktisch oder seltsam?",
                                        },
                                    ],
                                },
                                {
                                    "topic_key": "world politics",
                                    "variants": [
                                        {
                                            "headline": "In der Weltpolitik ist heute wieder einiges los.",
                                            "body": "Hast du davon schon gehoert?",
                                        },
                                        {
                                            "headline": "Weltpolitik ist gerade nichts fuer nebenbei.",
                                            "body": "Wenn du magst, ordnen wir das kurz ein.",
                                        },
                                        {
                                            "headline": "Ich habe heute wieder viel Weltpolitik gesehen.",
                                            "body": "Was davon beschaeftigt dich am ehesten?",
                                        },
                                    ],
                                },
                            ]
                        }
                    ),
                    SimpleNamespace(
                        output_parsed={
                            "items": [
                                {
                                    "topic_key": "ai companions",
                                    "headline": "KI-Begleiter werden alltagstauglicher.",
                                    "body": "Wollen wir kurz darueber reden?",
                                },
                                {
                                    "topic_key": "world politics",
                                    "headline": "Weltpolitik ist gerade nichts fuer nebenbei.",
                                    "body": "Wenn du magst, ordnen wir das kurz ein.",
                                },
                            ]
                        }
                    ),
                ],
            )
            generator = DisplayReserveCopyGenerator(
                backend_factory=lambda _config: cast(OpenAIBackend, backend)
            )

            rewritten = generator.rewrite_candidates(
                config=config,
                snapshot=None,
                candidates=candidates,
                local_now=datetime(2026, 3, 22, 10, 30, tzinfo=timezone.utc),
            )

        self.assertEqual(len(backend.responses_api.calls), 3)
        self.assertEqual(backend.responses_api.calls[0]["max_output_tokens"], 640)
        self.assertEqual(backend.responses_api.calls[0]["reasoning_effort"], "low")
        self.assertEqual(backend.responses_api.calls[1]["max_output_tokens"], 900)
        self.assertEqual(backend.responses_api.calls[1]["reasoning_effort"], "none")
        self.assertEqual(backend.responses_api.calls[2]["max_output_tokens"], 320)
        self.assertEqual(backend.responses_api.calls[2]["reasoning_effort"], "low")
        self.assertEqual(rewritten[0].headline, "KI-Begleiter werden alltagstauglicher.")
        self.assertEqual(rewritten[1].body, "Wenn du magst, ordnen wir das kurz ein.")

    def test_uses_configured_batch_size_and_variant_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                openai_api_key="test-key",
                display_reserve_generation_batch_size=1,
                display_reserve_generation_variants_per_candidate=2,
            )
            candidates = (
                _candidate(),
                replace(
                    _candidate(),
                    topic_key="world politics",
                    title="world politics",
                    headline="Fallback world",
                    body="Body world",
                    generation_context={
                        "candidate_family": "relationship",
                        "display_anchor": "world politics",
                        "hook_hint": "Da ist gerade einiges in Bewegung.",
                        "topic_summary": "Weltpolitik mit mehreren offenen Linien.",
                    },
                ),
            )
            backend = _FakeBackend(
                config=config,
                response=[
                    SimpleNamespace(
                        output_parsed={
                            "items": [
                                {
                                    "topic_key": "ai companions",
                                    "variants": [
                                        {
                                            "headline": "Ich habe heute etwas zu KI-Begleitern gelesen.",
                                            "body": "Was denkst du darueber?",
                                        },
                                        {
                                            "headline": "KI-Begleiter tauchen gerade wieder auf.",
                                            "body": "Hast du davon schon gehoert?",
                                        },
                                    ],
                                }
                            ]
                        }
                    ),
                    SimpleNamespace(
                        output_parsed={
                            "items": [
                                {
                                    "topic_key": "ai companions",
                                    "headline": "Ich habe heute etwas zu KI-Begleitern gelesen.",
                                    "body": "Was denkst du darueber?",
                                }
                            ]
                        }
                    ),
                    SimpleNamespace(
                        output_parsed={
                            "items": [
                                {
                                    "topic_key": "world politics",
                                    "variants": [
                                        {
                                            "headline": "In der Weltpolitik ist heute wieder einiges los.",
                                            "body": "Hast du davon schon gehoert?",
                                        },
                                        {
                                            "headline": "Weltpolitik ist heute ein grosses Thema.",
                                            "body": "Was denkst du dazu?",
                                        },
                                    ],
                                }
                            ]
                        }
                    ),
                    SimpleNamespace(
                        output_parsed={
                            "items": [
                                {
                                    "topic_key": "world politics",
                                    "headline": "Weltpolitik ist heute ein grosses Thema.",
                                    "body": "Was denkst du dazu?",
                                }
                            ]
                        }
                    ),
                ],
            )
            generator = DisplayReserveCopyGenerator(
                backend_factory=lambda _config: cast(OpenAIBackend, backend)
            )

            rewritten = generator.rewrite_candidates(
                config=config,
                snapshot=None,
                candidates=candidates,
                local_now=datetime(2026, 3, 22, 10, 30, tzinfo=timezone.utc),
            )

        self.assertEqual(len(backend.responses_api.calls), 4)
        self.assertEqual(
            [call["max_output_tokens"] for call in backend.responses_api.calls],
            [320, 160, 320, 160],
        )
        self.assertIn("\"variants_per_candidate\":2", cast(str, backend.responses_api.calls[0]["input"]))
        self.assertIn("\"variants_per_candidate\":2", cast(str, backend.responses_api.calls[2]["input"]))
        self.assertEqual(rewritten[0].headline, "Ich habe heute etwas zu KI-Begleitern gelesen.")
        self.assertEqual(rewritten[1].headline, "Weltpolitik ist heute ein grosses Thema.")

    def test_raises_when_any_batch_fails_instead_of_returning_partial_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                openai_api_key="test-key",
                display_reserve_generation_timeout_seconds=6.0,
                display_reserve_generation_max_output_tokens=900,
                display_reserve_generation_variants_per_candidate=3,
            )
            candidates = tuple(
                replace(
                    _candidate(),
                    topic_key=f"topic-{index}",
                    title=f"Thema {index}",
                    headline=f"Fallback {index}",
                    body=f"Body {index}",
                    generation_context={
                        "candidate_family": "mindshare",
                        "display_anchor": f"Thema {index}",
                        "hook_hint": f"Aufhaenger {index}",
                        "topic_summary": f"Zusammenfassung {index}",
                    },
                )
                for index in range(5)
            )
            backend = _FakeBackend(
                config=config,
                response=[
                    SimpleNamespace(
                        output_parsed={
                            "items": [
                                {
                                    "topic_key": f"topic-{index}",
                                    "variants": [
                                        {
                                            "headline": f"Rewrite {index}A",
                                            "body": f"Neu {index}A",
                                        },
                                        {
                                            "headline": f"Rewrite {index}B",
                                            "body": f"Neu {index}B",
                                        },
                                        {
                                            "headline": f"Rewrite {index}C",
                                            "body": f"Neu {index}C",
                                        },
                                    ],
                                }
                                for index in range(2)
                            ]
                        }
                    ),
                    SimpleNamespace(
                        output_parsed={
                            "items": [
                                {
                                    "topic_key": f"topic-{index}",
                                    "headline": f"Rewrite {index}",
                                    "body": f"Neu {index}",
                                }
                                for index in range(2)
                            ]
                        }
                    ),
                    RuntimeError("boom"),
                    SimpleNamespace(
                        output_parsed={
                            "items": [
                                {
                                    "topic_key": "topic-4",
                                    "variants": [
                                        {
                                            "headline": "Rewrite 4A",
                                            "body": "Neu 4A",
                                        },
                                        {
                                            "headline": "Rewrite 4B",
                                            "body": "Neu 4B",
                                        },
                                        {
                                            "headline": "Rewrite 4C",
                                            "body": "Neu 4C",
                                        },
                                    ],
                                }
                            ]
                        }
                    ),
                ],
            )
            generator = DisplayReserveCopyGenerator(
                backend_factory=lambda _config: cast(OpenAIBackend, backend)
            )

            with self.assertRaises(DisplayReserveGenerationError) as raised:
                generator.rewrite_candidates(
                    config=config,
                    snapshot=None,
                    candidates=candidates,
                    local_now=datetime(2026, 3, 22, 10, 30, tzinfo=timezone.utc),
                )

        self.assertEqual(len(backend.responses_api.calls), 3)
        self.assertEqual(backend.responses_api.calls[0]["max_output_tokens"], 640)
        self.assertEqual(backend.responses_api.calls[1]["max_output_tokens"], 320)
        self.assertEqual(backend.responses_api.calls[2]["max_output_tokens"], 640)
        self.assertEqual(raised.exception.batch_index, 2)
        self.assertEqual(raised.exception.topic_keys, ("topic-2", "topic-3"))
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)

    def test_raises_when_response_omits_copy_for_a_topic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                openai_api_key="test-key",
                display_reserve_generation_timeout_seconds=6.0,
                display_reserve_generation_max_output_tokens=900,
                display_reserve_generation_variants_per_candidate=3,
            )
            candidates = (
                _candidate(),
                replace(
                    _candidate(),
                    topic_key="world politics",
                    title="world politics",
                    headline="Fallback world",
                    body="Body world",
                    generation_context={
                        "candidate_family": "relationship",
                        "display_anchor": "world politics",
                        "hook_hint": "Da ist gerade einiges in Bewegung.",
                        "topic_summary": "Weltpolitik mit mehreren offenen Linien.",
                    },
                ),
            )
            backend = _FakeBackend(
                config=config,
                response=[
                    SimpleNamespace(
                        output_parsed={
                            "items": [
                                {
                                    "topic_key": "ai companions",
                                    "variants": [
                                        {
                                            "headline": "Ich habe heute etwas zu KI-Begleitern gelesen.",
                                            "body": "Was denkst du darueber?",
                                        },
                                        {
                                            "headline": "KI-Begleiter sind heute wieder Thema.",
                                            "body": "Willst du kurz darauf schauen?",
                                        },
                                        {
                                            "headline": "KI-Begleiter wirken heute wieder alltagsnah.",
                                            "body": "Soll ich dir sagen, was daran haengen bleibt?",
                                        },
                                    ],
                                },
                                {
                                    "topic_key": "world politics",
                                    "variants": [
                                        {
                                            "headline": "In der Weltpolitik ist heute wieder einiges los.",
                                            "body": "Hast du davon schon gehoert?",
                                        },
                                        {
                                            "headline": "Weltpolitik ist gerade nichts fuer nebenbei.",
                                            "body": "Wenn du magst, ordnen wir das kurz ein.",
                                        },
                                        {
                                            "headline": "Ich habe heute wieder viel Weltpolitik gesehen.",
                                            "body": "Was davon beschaeftigt dich am ehesten?",
                                        },
                                    ],
                                },
                            ]
                        }
                    ),
                    SimpleNamespace(
                        output_parsed={
                            "items": [
                                {
                                    "topic_key": "ai companions",
                                    "headline": "KI-Begleiter sind heute wieder Thema.",
                                    "body": "Was denkst du darueber?",
                                }
                            ]
                        }
                    ),
                ],
            )
            generator = DisplayReserveCopyGenerator(
                backend_factory=lambda _config: cast(OpenAIBackend, backend)
            )

            with self.assertRaises(DisplayReserveGenerationError) as raised:
                generator.rewrite_candidates(
                    config=config,
                    snapshot=None,
                    candidates=candidates,
                    local_now=datetime(2026, 3, 22, 10, 30, tzinfo=timezone.utc),
                )

        self.assertEqual(len(backend.responses_api.calls), 2)
        self.assertEqual(raised.exception.batch_index, 1)
        self.assertIsInstance(raised.exception.__cause__, ValueError)


if __name__ == "__main__":
    unittest.main()
