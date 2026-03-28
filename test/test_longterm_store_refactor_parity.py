from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.core.models import (  # noqa: E402
    LongTermConflictResolutionV1,
    LongTermConsolidationResultV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.storage.store import LongTermStructuredStore  # noqa: E402


def _source_ref(event_id: str) -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation",
        event_ids=(event_id,),
        speaker="user",
        modality="voice",
    )


def _capture_primary_flow(store_cls: type[LongTermStructuredStore]) -> dict[str, object]:
    base = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    objects = (
        LongTermMemoryObjectV1(
            memory_id="durable-jam",
            kind="preference",
            summary="Corinna mag Aprikosenmarmelade",
            details="Zum Fruehstueck",
            source=_source_ref("evt-1"),
            status="active",
            confirmed_by_user=True,
            slot_key="food/jam",
            value_key="apricot",
            created_at=base,
            updated_at=base + timedelta(minutes=1),
            attributes={"room": "kitchen"},
        ),
        LongTermMemoryObjectV1(
            memory_id="durable-thermos",
            kind="fact",
            summary="Die Thermoskanne steht im Flur",
            details="Neben der Garderobe",
            source=_source_ref("evt-2"),
            status="candidate",
            slot_key="object/thermos/location",
            value_key="hallway",
            created_at=base,
            updated_at=base + timedelta(minutes=2),
        ),
        LongTermMemoryObjectV1(
            memory_id="episode-breakfast",
            kind="episode",
            summary="Beim Fruehstueck gab es Aprikosenmarmelade",
            details="Corinna fragte nach Nachkauf",
            source=_source_ref("evt-3"),
            status="active",
            created_at=base,
            updated_at=base + timedelta(minutes=3),
        ),
        LongTermMemoryObjectV1(
            memory_id="durable-rainbow",
            kind="fact",
            summary="Ein Regenbogen entsteht durch Lichtbrechung",
            details="Allgemeinwissen",
            source=_source_ref("evt-4"),
            status="active",
            created_at=base,
            updated_at=base + timedelta(minutes=4),
        ),
    )
    conflicts = (
        LongTermMemoryConflictV1(
            slot_key="food/jam",
            candidate_memory_id="durable-jam",
            existing_memory_ids=("durable-thermos",),
            question="Welche Marmelade mag Corinna?",
            reason="Neue Aussage kollidiert mit altem Fakt",
        ),
    )
    archived_objects = (
        LongTermMemoryObjectV1(
            memory_id="archive-old-jam",
            kind="preference",
            summary="Frueher mochte Corinna Erdbeermarmelade",
            details="Nicht mehr aktuell",
            source=_source_ref("evt-5"),
            status="superseded",
            archived_at=(base + timedelta(days=1)).isoformat(),
            created_at=base,
            updated_at=base + timedelta(minutes=5),
        ),
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        store = store_cls(base_path=Path(temp_dir))
        store.write_snapshot(
            objects=objects,
            conflicts=conflicts,
            archived_objects=archived_objects,
        )
        return {
            "load_ids": [item.memory_id for item in store.load_objects()],
            "review": [
                item.memory_id
                for item in store.review_objects(
                    query_text="Aprikosenmarmelade",
                    include_episodes=True,
                    limit=5,
                ).items
            ],
            "relevant": [
                item.memory_id
                for item in store.select_relevant_objects(
                    query_text="Aprikosenmarmelade",
                    limit=4,
                )
            ],
            "episodic": [
                item.memory_id
                for item in store.select_relevant_episodic_objects(
                    query_text="Aprikosenmarmelade",
                    limit=4,
                )
            ],
            "fast_topic": [
                item.memory_id
                for item in store.select_fast_topic_objects(
                    query_text="Aprikosenmarmelade",
                    limit=3,
                )
            ],
            "context": [
                [item.memory_id for item in section]
                for section in store.select_relevant_context_objects(
                    query_text="Aprikosenmarmelade",
                    episodic_limit=2,
                    durable_limit=2,
                )
            ],
            "conflicts": [
                f"{item.slot_key}|{item.candidate_memory_id}"
                for item in store.select_open_conflicts(
                    query_text="Marmelade",
                    limit=3,
                )
            ],
            "confirm_status": store.confirm_object("durable-thermos").updated_objects[0].status,
            "invalidate_ids": [
                item.memory_id
                for item in store.invalidate_object(
                    "durable-thermos",
                    reason="Falsch",
                ).updated_objects
            ],
            "delete_deleted": list(store.delete_object("durable-thermos").deleted_memory_ids),
        }


def _capture_merge_flow(store_cls: type[LongTermStructuredStore]) -> dict[str, object]:
    base = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    seed = LongTermMemoryObjectV1(
        memory_id="fact-seed",
        kind="fact",
        summary="Corinna hat eine rote Tasse",
        source=_source_ref("evt-0"),
        status="candidate",
        slot_key="cup/color",
        value_key="red",
        created_at=base,
        updated_at=base,
        attributes={"support_count": 2, "seen": "yesterday"},
    )
    consolidated = LongTermMemoryObjectV1(
        memory_id="fact-seed",
        kind="fact",
        summary="Corinna hat eine rote Tasse",
        source=_source_ref("evt-1"),
        status="active",
        confirmed_by_user=True,
        slot_key="cup/color",
        value_key="red",
        created_at=base,
        updated_at=base + timedelta(minutes=1),
        attributes={"note": "seen twice"},
    )
    replacement = LongTermMemoryObjectV1(
        memory_id="fact-new",
        kind="fact",
        summary="Die Tasse ist jetzt gruen",
        source=_source_ref("evt-2"),
        status="active",
        confirmed_by_user=True,
        slot_key="cup/color",
        value_key="green",
        created_at=base,
        updated_at=base + timedelta(minutes=2),
        supersedes=("fact-seed",),
    )
    conflict = LongTermMemoryConflictV1(
        slot_key="cup/color",
        candidate_memory_id="fact-new",
        existing_memory_ids=("fact-seed",),
        question="Welche Farbe hat die Tasse?",
        reason="Neue Beobachtung widerspricht alter Erinnerung",
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        store = store_cls(base_path=Path(temp_dir))
        store.write_snapshot(objects=(seed,), conflicts=(), archived_objects=())
        store.apply_consolidation(
            LongTermConsolidationResultV1(
                turn_id="turn-1",
                occurred_at=base,
                episodic_objects=(),
                durable_objects=(consolidated, replacement),
                deferred_objects=(),
                conflicts=(conflict,),
                graph_edges=(),
            )
        )
        merged = store.get_object("fact-seed")
        store.apply_conflict_resolution(
            LongTermConflictResolutionV1(
                slot_key="cup/color",
                selected_memory_id="fact-new",
                updated_objects=(replacement,),
                remaining_conflicts=(),
            )
        )
        return {
            "merged_status": None if merged is None else merged.status,
            "merged_confirmed": None if merged is None else merged.confirmed_by_user,
            "merged_support_count": None if merged is None or merged.attributes is None else merged.attributes.get("support_count"),
            "merged_event_ids": None if merged is None else list(merged.source.event_ids),
            "post_resolution_conflicts": [item.candidate_memory_id for item in store.load_conflicts()],
            "post_resolution_objects": [item.memory_id for item in store.load_objects()],
        }


class LongTermStructuredStoreRefactorParityTests(unittest.TestCase):
    def test_public_store_matches_primary_golden_master(self) -> None:
        self.assertEqual(
            _capture_primary_flow(LongTermStructuredStore),
            {
                "load_ids": [
                    "durable-jam",
                    "durable-rainbow",
                    "durable-thermos",
                    "episode-breakfast",
                ],
                "review": [
                    "episode-breakfast",
                    "durable-jam",
                ],
                "relevant": ["durable-jam"],
                "episodic": ["episode-breakfast"],
                "fast_topic": [
                    "episode-breakfast",
                    "durable-jam",
                ],
                "context": [
                    ["episode-breakfast"],
                    ["durable-jam"],
                ],
                "conflicts": ["food/jam|durable-jam"],
                "confirm_status": "active",
                "invalidate_ids": ["durable-thermos"],
                "delete_deleted": ["durable-thermos"],
            },
        )

    def test_public_store_matches_merge_golden_master(self) -> None:
        self.assertEqual(
            _capture_merge_flow(LongTermStructuredStore),
            {
                "merged_status": "active",
                "merged_confirmed": True,
                "merged_support_count": 3,
                "merged_event_ids": ["evt-0", "evt-1"],
                "post_resolution_conflicts": [],
                "post_resolution_objects": ["fact-new", "fact-seed"],
            },
        )

    def test_public_wrapper_matches_internal_base_when_available(self) -> None:
        try:
            from twinr.memory.longterm.storage._structured_store import LongTermStructuredStoreBase
        except ImportError:
            self.skipTest("internal structured-store package not available before refactor")
        self.assertEqual(
            _capture_primary_flow(LongTermStructuredStore),
            _capture_primary_flow(LongTermStructuredStoreBase),
        )
        self.assertEqual(
            _capture_merge_flow(LongTermStructuredStore),
            _capture_merge_flow(LongTermStructuredStoreBase),
        )


if __name__ == "__main__":
    unittest.main()
