from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.proactive.runtime.display_reserve_diversity import (
    display_reserve_seed_profile,
    reserve_seed_family,
    select_diverse_candidates,
)


def _candidate(
    *,
    topic_key: str,
    source: str,
    candidate_family: str,
    salience: float,
    attention_state: str = "forming",
    generation_context: dict[str, object] | None = None,
) -> AmbientDisplayImpulseCandidate:
    return AmbientDisplayImpulseCandidate(
        topic_key=topic_key,
        title=topic_key.title(),
        source=source,
        action="ask_one",
        attention_state=attention_state,
        salience=salience,
        eyebrow="",
        headline=topic_key.title(),
        body="Da ist etwas offen.",
        symbol="sparkles",
        accent="info",
        reason=f"seed:{topic_key}",
        candidate_family=candidate_family,
        generation_context=generation_context or {},
    )


class DisplayReserveDiversityTests(unittest.TestCase):
    def test_seed_profile_normalizes_existing_candidate_shapes(self) -> None:
        identity = _candidate(
            topic_key="name",
            source="user_discovery",
            candidate_family="user_discovery",
            salience=0.7,
            generation_context={"topic_id": "basics", "invite_kind": "topic"},
        )
        local_world = _candidate(
            topic_key="hamburg",
            source="regional_news",
            candidate_family="world_subscription",
            salience=0.8,
            generation_context={"scope": "regional"},
        )
        shared_thread = _candidate(
            topic_key="janina",
            source="reflection_summary",
            candidate_family="reflection",
            salience=0.75,
            attention_state="shared_thread",
            generation_context={"reflection_kind": "thread", "memory_domain": "thread"},
        )

        self.assertEqual(display_reserve_seed_profile(identity).family, "identity_setup")
        self.assertEqual(display_reserve_seed_profile(identity).axis, "setup")
        self.assertEqual(display_reserve_seed_profile(local_world).family, "local_world")
        self.assertEqual(display_reserve_seed_profile(local_world).axis, "public")
        self.assertEqual(display_reserve_seed_profile(shared_thread).family, "shared_thread")
        self.assertEqual(display_reserve_seed_profile(shared_thread).axis, "personal")

    def test_select_diverse_candidates_prefers_broader_mix(self) -> None:
        candidates = (
            _candidate(
                topic_key="world-1",
                source="world",
                candidate_family="world_awareness",
                salience=0.96,
            ),
            _candidate(
                topic_key="world-2",
                source="world",
                candidate_family="world_subscription",
                salience=0.91,
            ),
            _candidate(
                topic_key="world-3",
                source="world",
                candidate_family="world_subscription",
                salience=0.88,
            ),
            _candidate(
                topic_key="name",
                source="user_discovery",
                candidate_family="user_discovery",
                salience=0.73,
                generation_context={"topic_id": "basics", "invite_kind": "topic"},
            ),
            _candidate(
                topic_key="doctor",
                source="memory_follow_up",
                candidate_family="memory_follow_up",
                salience=0.72,
                generation_context={"memory_goal": "gentle_follow_up"},
            ),
        )

        selected = select_diverse_candidates(candidates, max_items=4)
        families = [reserve_seed_family(candidate) for candidate in selected]

        self.assertIn("public_world", families)
        self.assertIn("identity_setup", families)
        self.assertIn("memory_follow_up", families)
        self.assertLessEqual(families.count("identity_setup"), 1)


if __name__ == "__main__":
    unittest.main()
