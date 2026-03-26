from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.proactive.runtime.display_reserve_expansion import (
    bundle_display_reserve_candidates,
    expand_display_reserve_candidates,
)


def _candidate(
    *,
    topic_key: str,
    source: str,
    candidate_family: str,
    salience: float,
    semantic_topic_key: str = "",
) -> AmbientDisplayImpulseCandidate:
    return AmbientDisplayImpulseCandidate(
        topic_key=topic_key,
        semantic_topic_key=semantic_topic_key,
        title=topic_key.title(),
        source=source,
        action="brief_update",
        attention_state="growing",
        salience=salience,
        eyebrow="IM BLICK",
        headline=topic_key.title(),
        body="Da tut sich gerade etwas.",
        symbol="sparkles",
        accent="info",
        reason=f"seed:{topic_key}",
        candidate_family=candidate_family,
    )


class DisplayReserveExpansionTests(unittest.TestCase):
    def test_bundle_groups_candidates_by_semantic_topic(self) -> None:
        bundles = bundle_display_reserve_candidates(
            (
                _candidate(
                    topic_key="world politics::primary",
                    semantic_topic_key="world politics",
                    source="world",
                    candidate_family="world",
                    salience=0.81,
                ),
                _candidate(
                    topic_key="world politics::follow_up",
                    semantic_topic_key="world politics",
                    source="memory_follow_up",
                    candidate_family="memory_follow_up",
                    salience=0.76,
                ),
                _candidate(
                    topic_key="janina::primary",
                    semantic_topic_key="janina",
                    source="memory_follow_up",
                    candidate_family="memory_follow_up",
                    salience=0.70,
                ),
            )
        )

        self.assertEqual(len(bundles), 2)
        world_bundle = next(bundle for bundle in bundles if bundle.semantic_topic_key == "world politics")
        self.assertEqual(world_bundle.candidate_count, 2)
        self.assertEqual(world_bundle.support_sources, ("world", "memory_follow_up"))
        self.assertEqual(world_bundle.support_families, ("world", "memory_follow_up"))

    def test_expand_generates_unique_card_surfaces_for_same_semantic_topic(self) -> None:
        expanded = expand_display_reserve_candidates(
            (
                _candidate(
                    topic_key="world politics::seed",
                    semantic_topic_key="world politics",
                    source="world",
                    candidate_family="world",
                    salience=0.88,
                ),
                _candidate(
                    topic_key="janina::seed",
                    semantic_topic_key="janina",
                    source="memory_follow_up",
                    candidate_family="memory_follow_up",
                    salience=0.77,
                ),
                _candidate(
                    topic_key="setup::seed",
                    semantic_topic_key="user_discovery:initial_setup:basics",
                    source="user_discovery",
                    candidate_family="user_discovery",
                    salience=0.40,
                ),
            ),
            target_cards=7,
        )

        self.assertEqual(len(expanded), 7)
        self.assertEqual(len({candidate.topic_key for candidate in expanded}), len(expanded))
        world_cards = [
            candidate
            for candidate in expanded
            if candidate.semantic_key() == "world politics"
        ]
        self.assertEqual([candidate.expansion_angle for candidate in world_cards], ["primary", "public_reaction", "broader_view"])
        self.assertTrue(all(candidate.support_sources == ("world",) for candidate in world_cards))
        self.assertTrue(all(candidate.topic_key.startswith("world politics::") for candidate in world_cards))


if __name__ == "__main__":
    unittest.main()
