from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.proactive.runtime.display_reserve_prompting import (
    build_candidate_prompt_payload,
    build_generation_prompt,
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
                "summary": "Internal raw summary that should not dominate the prompt payload.",
                "details": "A very long internal details field that should stay compressed in the outgoing prompt payload.",
                "recent_titles": ("Hausarzt", "Blutwerte", "Rueckruf"),
            },
        )

        payload = build_candidate_prompt_payload(candidate)

        self.assertEqual(payload["topic_anchor"], "Arzttermin gestern")
        self.assertEqual(payload["hook_hint"], "Du meintest, dass du da noch auf das Ergebnis wartest.")
        self.assertIn("Hausarzt", payload["context_summary"])
        self.assertNotIn("Donald Trump heute", payload["topic_anchor"])
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
            },
        )

        prompt = build_generation_prompt(
            snapshot=None,
            candidates=(candidate,),
            local_now=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
        )

        self.assertIn("\"topic_anchor\":\"AI companions\"", prompt)
        self.assertIn("\"hook_hint\":\"Die Systeme werden alltagstauglicher.\"", prompt)
        self.assertIn("\"context_summary\":", prompt)
        self.assertNotIn("\"context\":", prompt)
        self.assertNotIn("fallback_headline", prompt)
        self.assertNotIn("fallback_body", prompt)


if __name__ == "__main__":
    unittest.main()
