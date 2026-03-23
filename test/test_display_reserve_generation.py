from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.agent.personality.models import (
    ConversationStyleProfile,
    HumorProfile,
    PersonalitySnapshot,
    PersonalityTrait,
    WorldSignal,
)
from twinr.proactive.runtime.display_reserve_generation import DisplayReserveCopyGenerator


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
            "display_goal": "open_positive_conversation",
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
            )
            backend = _FakeBackend(
                config=config,
                response=SimpleNamespace(
                    output_parsed={
                        "items": [
                            {
                                "topic_key": "ai companions",
                                "headline": "Denkst du, dass das heute kippt?",
                                "body": "Ich wuerde da gern kurz mit dir draufschauen.",
                            }
                        ]
                    }
                ),
            )
            generator = DisplayReserveCopyGenerator(backend_factory=lambda _config: backend)
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

        self.assertEqual(rewritten[0].headline, "Denkst du, dass das heute kippt?")
        self.assertEqual(rewritten[0].body, "Ich wuerde da gern kurz mit dir draufschauen.")
        self.assertEqual(len(backend.responses_api.calls), 1)
        request = backend.responses_api.calls[0]
        self.assertEqual(request["model"], "gpt-4o-mini")
        self.assertEqual(request["timeout"], 5.5)
        self.assertEqual(request["prompt_cache_scope"], "display_reserve_generation")
        self.assertIn("rechte Display-Spalte", request["instructions"])
        self.assertIn("klaren Themenanker", request["instructions"])
        self.assertIn("wie Kundenservice", request["instructions"])
        self.assertIn("\"context_summary\":", request["input"])
        self.assertIn("\"hook_hint\":", request["input"])
        self.assertIn("\"topic_anchor\":", request["input"])
        self.assertIn("\"voice_de\":", request["input"])
        self.assertNotIn("\"context\":", request["input"])
        self.assertNotIn("fallback_headline", request["input"])
        self.assertNotIn("fallback_body", request["input"])

    def test_falls_back_to_existing_copy_when_generation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                openai_api_key="test-key",
            )
            generator = DisplayReserveCopyGenerator(
                backend_factory=lambda _config: _FakeBackend(
                    config=config,
                    response=RuntimeError("boom"),
                )
            )
            candidate = _candidate()

            rewritten = generator.rewrite_candidates(
                config=config,
                snapshot=None,
                candidates=(candidate,),
                local_now=datetime(2026, 3, 22, 10, 30, tzinfo=timezone.utc),
            )

        self.assertEqual(rewritten, (candidate,))

    def test_accepts_top_level_array_text_when_sdk_parsed_output_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                openai_api_key="test-key",
            )
            backend = _FakeBackend(
                config=config,
                response=SimpleNamespace(
                    output_parsed=None,
                    output_text='[{"topic_key":"ai companions","headline":"KI-Begleiter: was tut sich da?","body":"Ich habe das ruhig im Blick."}]',
                ),
            )
            generator = DisplayReserveCopyGenerator(backend_factory=lambda _config: backend)
            candidate = _candidate()

            rewritten = generator.rewrite_candidates(
                config=config,
                snapshot=None,
                candidates=(candidate,),
                local_now=datetime(2026, 3, 22, 10, 30, tzinfo=timezone.utc),
            )

        self.assertEqual(rewritten[0].headline, "KI-Begleiter: was tut sich da?")
        self.assertEqual(rewritten[0].body, "Ich habe das ruhig im Blick.")

    def test_batches_requests_and_isolates_failed_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                openai_api_key="test-key",
                display_reserve_generation_timeout_seconds=6.0,
                display_reserve_generation_max_output_tokens=900,
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
                                    "headline": "Rewrite 4",
                                    "body": "Neu 4",
                                }
                            ]
                        }
                    ),
                ],
            )
            generator = DisplayReserveCopyGenerator(backend_factory=lambda _config: backend)

            rewritten = generator.rewrite_candidates(
                config=config,
                snapshot=None,
                candidates=candidates,
                local_now=datetime(2026, 3, 22, 10, 30, tzinfo=timezone.utc),
            )

        self.assertEqual(len(backend.responses_api.calls), 3)
        self.assertEqual(backend.responses_api.calls[0]["max_output_tokens"], 256)
        self.assertEqual(backend.responses_api.calls[1]["max_output_tokens"], 256)
        self.assertEqual(backend.responses_api.calls[2]["max_output_tokens"], 160)
        self.assertEqual(rewritten[0].headline, "Rewrite 0")
        self.assertEqual(rewritten[1].body, "Neu 1")
        self.assertEqual(rewritten[2].body, "Body 2")
        self.assertEqual(rewritten[3].body, "Body 3")
        self.assertEqual(rewritten[4].headline, "Rewrite 4")
        self.assertEqual(rewritten[4].body, "Neu 4")


if __name__ == "__main__":
    unittest.main()
