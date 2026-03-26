from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.proactive.runtime.display_reserve_prompting import (
    build_candidate_prompt_payload,
    build_generation_prompt,
    build_selection_prompt,
)


class DisplayReservePromptingTests(unittest.TestCase):
    def test_candidate_prompt_payload_prefers_display_anchor_and_compacts_context(self) -> None:
        candidate = AmbientDisplayImpulseCandidate(
            topic_key="doctor-appointment",
            title="Donald Trump heute",
            source="reflection_midterm",
            action="ask_one",
            attention_state="shared_thread",
            salience=0.82,
            eyebrow="",
            headline="Wollen wir da kurz anknuepfen?",
            body="Da ist noch etwas offen.",
            symbol="question",
            accent="warm",
            reason="reflection_midterm",
            candidate_family="reflection_thread",
            generation_context={
                "display_anchor": "Arzttermin gestern",
                "hook_hint": "Du meintest, dass du da noch auf das Ergebnis wartest.",
                "card_intent": {
                    "topic_semantics": "frueherer Gespraechsfaden zu Arzttermin gestern",
                    "statement_intent": "Twinr soll ruhig an den frueheren Gespraechsfaden zum Arzttermin anknuepfen.",
                    "cta_intent": "Zu einem kurzen Update oder Weiterreden einladen.",
                    "relationship_stance": "ruhiger Rueckbezug statt Diagnose",
                },
                "summary": "Internal raw summary that should not dominate the prompt payload.",
                "details": "A very long internal details field that should stay compressed in the outgoing prompt payload.",
                "recent_titles": ("Hausarzt", "Blutwerte", "Rueckruf"),
            },
        )

        payload = build_candidate_prompt_payload(candidate)
        context_summary = cast(str, payload["context_summary"])
        topic_anchor = cast(str, payload["topic_anchor"])

        self.assertEqual(topic_anchor, "Arzttermin gestern")
        self.assertEqual(payload["copy_family"], "reflection")
        self.assertEqual(payload["hook_hint"], "Du meintest, dass du da noch auf das Ergebnis wartest.")
        self.assertEqual(
            payload["card_intent"],
            {
                "topic_semantics": "frueherer Gespraechsfaden zu Arzttermin gestern",
                "statement_intent": "Twinr soll ruhig an den frueheren Gespraechsfaden zum Arzttermin anknuepfen.",
                "cta_intent": "Zu einem kurzen Update oder Weiterreden einladen.",
                "relationship_stance": "ruhiger Rueckbezug statt Diagnose",
            },
        )
        self.assertNotIn("family_examples", payload)
        self.assertIn("Hausarzt", context_summary)
        self.assertNotIn("Donald Trump heute", topic_anchor)
        self.assertNotIn("fallback_headline", payload)
        self.assertNotIn("fallback_body", payload)

    def test_generation_prompt_uses_compact_candidate_payloads(self) -> None:
        candidate = AmbientDisplayImpulseCandidate(
            topic_key="ai-companions",
            title="AI companions",
            source="world",
            action="brief_update",
            attention_state="growing",
            salience=0.74,
            eyebrow="",
            headline="Was meinst du dazu?",
            body="Da bleibe ich dran.",
            symbol="sparkles",
            accent="info",
            reason="world",
            candidate_family="world_awareness",
            generation_context={
                "display_anchor": "AI companions",
                "hook_hint": "Die Systeme werden alltagstauglicher.",
                "topic_summary": "Persoenliche KI-Begleiter im Alltag.",
                "card_intent": {
                    "topic_semantics": "oeffentliches Thema zu KI-Begleitern",
                    "statement_intent": "Twinr soll eine konkrete Beobachtung zu KI-Begleitern machen.",
                    "cta_intent": "Zu einer echten Meinung oder Einordnung einladen.",
                    "relationship_stance": "ruhig beobachtend mit leichter Haltung",
                },
                "ambient_learning": {
                    "topic_state": "pulling",
                    "topic_score": 0.71,
                    "topic_repetition_pressure": 0.12,
                    "family_state": "pulling",
                    "family_score": 0.42,
                    "action_score": 0.28,
                },
            },
        )

        prompt = build_generation_prompt(
            snapshot=None,
            candidates=(candidate,),
            local_now=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
            variants_per_candidate=3,
        )

        self.assertIn("\"topic_anchor\":\"AI companions\"", prompt)
        self.assertIn("\"hook_hint\":\"Die Systeme werden alltagstauglicher.\"", prompt)
        self.assertIn("\"context_summary\":", prompt)
        self.assertIn("\"copy_family\":\"world\"", prompt)
        self.assertIn("\"card_intent\":", prompt)
        self.assertIn("\"statement_intent\":\"Twinr soll eine konkrete Beobachtung zu KI-Begleitern machen.\"", prompt)
        self.assertIn("\"pickup_signal\":", prompt)
        self.assertIn("\"topic_state\":\"pulling\"", prompt)
        self.assertIn("\"quality_rubric\":", prompt)
        self.assertIn("\"family_examples\":", prompt)
        self.assertIn("Ich habe heute etwas zu KI-Begleitern gelesen.", prompt)
        self.assertIn("Klingt die Karte wie normales, spontanes Deutsch?", prompt)
        self.assertIn("\"variants_per_candidate\":3", prompt)
        self.assertNotIn("\"context\":", prompt)
        self.assertNotIn("fallback_headline", prompt)
        self.assertNotIn("fallback_body", prompt)

    def test_candidate_prompt_payload_does_not_backfill_raw_visible_copy_without_anchor(self) -> None:
        candidate = AmbientDisplayImpulseCandidate(
            topic_key="green button",
            title="Green button",
            source="reflection_midterm",
            action="hint",
            attention_state="forming",
            salience=0.41,
            eyebrow="",
            headline="Zu Green button wuerde ich gern kurz anknuepfen.",
            body="Ein kleiner Nachtrag dazu waere schon hilfreich.",
            symbol="sparkles",
            accent="info",
            reason="reflection_midterm",
            candidate_family="reflection",
            generation_context={
                "summary": "Internal packet summary only.",
                "details": "Internal packet details only.",
            },
        )

        payload = build_candidate_prompt_payload(candidate)
        context_summary = cast(str, payload["context_summary"])

        self.assertEqual(payload["topic_anchor"], "")
        self.assertEqual(payload["hook_hint"], "")
        self.assertNotIn("Green button", context_summary)
        self.assertNotIn("Zu Green button", context_summary)
        self.assertNotIn("Ein kleiner Nachtrag", context_summary)

    def test_candidate_prompt_payload_includes_compact_pickup_signal_from_real_outcomes(self) -> None:
        candidate = AmbientDisplayImpulseCandidate(
            topic_key="ai-companions",
            title="AI companions",
            source="relationship",
            action="hint",
            attention_state="growing",
            salience=0.81,
            eyebrow="",
            headline="Bei AI companions ist fuer mich noch etwas offen.",
            body="Wollen wir kurz darueber reden?",
            symbol="heart",
            accent="warm",
            reason="light_interest_hint",
            candidate_family="memory_thread",
            generation_context={
                "display_anchor": "AI companions",
                "hook_hint": "Persoenliche KI-Begleiter werden alltagstauglicher.",
                "ambient_learning": {
                    "topic_state": "pulling",
                    "topic_score": 0.63,
                    "topic_repetition_pressure": 0.18,
                    "family_state": "neutral",
                    "family_score": 0.14,
                    "action_score": 0.22,
                },
            },
        )

        payload = build_candidate_prompt_payload(candidate)

        self.assertEqual(
            payload["pickup_signal"],
            {
                "topic_state": "pulling",
                "topic_score": 0.63,
                "topic_repetition_pressure": 0.18,
                "family_state": "neutral",
                "family_score": 0.14,
                "action_score": 0.22,
            },
        )

    def test_selection_prompt_includes_family_examples_and_quality_rubric(self) -> None:
        candidate = AmbientDisplayImpulseCandidate(
            topic_key="discovery-name",
            title="Name",
            source="user_discovery",
            action="ask_one",
            attention_state="forming",
            salience=0.68,
            eyebrow="",
            headline="Ich moechte wissen, wie ich dich ansprechen soll.",
            body="Wie soll ich dich nennen?",
            symbol="question",
            accent="warm",
            reason="invite_user_discovery",
            candidate_family="user_discovery",
            generation_context={
                "display_goal": "invite_user_discovery",
                "display_anchor": "Ansprache",
                "hook_hint": "Twinr soll den Namen oder die gewuenschte Ansprache lernen.",
            },
        )

        prompt = build_selection_prompt(
            snapshot=None,
            candidates=(candidate,),
            variants_by_topic={
                "discovery-name": (
                    {
                        "headline": "Ich moechte wissen, wie ich dich ansprechen soll.",
                        "body": "Wie soll ich dich nennen?",
                    },
                    {
                        "headline": "Mich interessiert, wie ich dich nennen soll.",
                        "body": "Wie ist es dir lieber?",
                    },
                )
            },
            local_now=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
        )

        self.assertIn("\"copy_family\":\"discovery\"", prompt)
        self.assertIn("\"family_examples\":", prompt)
        self.assertIn("Mich interessiert, was dir morgens gut tut.", prompt)
        self.assertIn("\"quality_rubric\":", prompt)
        self.assertIn("Klingt die Karte wie normales, spontanes Deutsch?", prompt)
        self.assertIn("Wie soll ich dich nennen?", prompt)

    def test_reflection_prompt_examples_stay_topic_grounded(self) -> None:
        candidate = AmbientDisplayImpulseCandidate(
            topic_key="doctor-appointment",
            title="Arzttermin",
            source="reflection_midterm",
            action="ask_one",
            attention_state="shared_thread",
            salience=0.72,
            eyebrow="",
            headline="Ich denke noch an den Arzttermin.",
            body="Wollen wir da kurz weitermachen?",
            symbol="question",
            accent="warm",
            reason="reflection_midterm",
            candidate_family="reflection_thread",
            generation_context={
                "display_goal": "call_back_to_earlier_conversation",
                "display_anchor": "Arzttermin gestern",
                "hook_hint": "Du meintest, dass da noch etwas offen ist.",
            },
        )

        prompt = build_generation_prompt(
            snapshot=None,
            candidates=(candidate,),
            local_now=datetime(2026, 3, 26, 9, 0, tzinfo=timezone.utc),
            variants_per_candidate=2,
        )

        self.assertIn("Dein Gedanke zum Arzttermin ist mir geblieben.", prompt)
        self.assertNotIn("Dein Hallo von vorhin ist mir geblieben.", prompt)


if __name__ == "__main__":
    unittest.main()
