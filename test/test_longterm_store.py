from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
from threading import Thread
from types import SimpleNamespace
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test.longterm_test_program import make_test_extractor
from twinr.config import TwinrConfig
from twinr.memory.longterm.reasoning.consolidator import LongTermMemoryConsolidator
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermGraphEdgeCandidateV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.memory.longterm.storage.store import LongTermStructuredStore, _write_json_atomic
from twinr.memory.longterm.reasoning.truth import LongTermTruthMaintainer


class _FakeRemoteState:
    def __init__(self) -> None:
        self.client = _FakeChonkyClient()
        self.enabled = True
        self.required = False
        self.namespace = "test-namespace"
        self.read_client = self.client
        self.write_client = self.client
        self.config = SimpleNamespace(
            long_term_memory_migration_enabled=False,
            long_term_memory_remote_shard_max_content_chars=1000,
            long_term_memory_remote_max_content_chars=2_000_000,
        )
        self.snapshots: dict[str, dict[str, object]] = {}

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        del local_path
        payload = self.snapshots.get(snapshot_kind)
        return dict(payload) if isinstance(payload, dict) else None

    def save_snapshot(self, *, snapshot_kind: str, payload):
        self.snapshots[snapshot_kind] = dict(payload)


class _FakeChonkyClient:
    def __init__(self, *, max_items_per_bulk: int | None = None) -> None:
        self._next_document_id = 1
        self.max_items_per_bulk = max_items_per_bulk
        self.bulk_calls = 0
        self.records_by_document_id: dict[str, dict[str, object]] = {}
        self.records_by_uri: dict[str, dict[str, object]] = {}

    def store_records_bulk(self, request):
        items = tuple(getattr(request, "items", ()))
        self.bulk_calls += 1
        if self.max_items_per_bulk is not None and len(items) > self.max_items_per_bulk:
            raise LongTermRemoteUnavailableError("bulk request too large")
        response_items = []
        for item in items:
            document_id = f"doc-{self._next_document_id}"
            self._next_document_id += 1
            record = {
                "document_id": document_id,
                "payload": dict(getattr(item, "payload", {}) or {}),
                "metadata": dict(getattr(item, "metadata", {}) or {}),
                "content": getattr(item, "content", None),
                "uri": getattr(item, "uri", None),
            }
            self.records_by_document_id[document_id] = record
            uri = record.get("uri")
            if isinstance(uri, str) and uri:
                self.records_by_uri[uri] = record
            response_items.append({"document_id": document_id})
        return {"items": response_items}

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        del include_content
        del max_content_chars
        if isinstance(document_id, str) and document_id:
            record = self.records_by_document_id.get(document_id)
            if record is not None:
                return dict(record)
        if isinstance(origin_uri, str) and origin_uri:
            record = self.records_by_uri.get(origin_uri)
            if record is not None:
                return dict(record)
        raise LongTermRemoteUnavailableError("remote document unavailable")

    def retrieve(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        query_text = str(payload.get("query_text") or "").lower()
        allowed = set(str(value) for value in payload.get("allowed_doc_ids", ()) if str(value))
        ranked = []
        for document_id, record in self.records_by_document_id.items():
            if allowed and document_id not in allowed:
                continue
            content = str(record.get("content") or "").lower()
            if query_text and query_text not in content:
                continue
            ranked.append(
                {
                    "payload_id": document_id,
                    "relevance_score": 1.0,
                    "metadata": dict(record.get("metadata") or {}),
                    "source_index": "fulltext",
                    "candidate_origin": "fulltext",
                }
            )
        return SimpleNamespace(success=True, mode="advanced", results=tuple(SimpleNamespace(**item) for item in ranked), indexes_used=("fulltext",))


class _FailingRemoteState(_FakeRemoteState):
    def __init__(self) -> None:
        super().__init__()
        self.required = True

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        del snapshot_kind
        del local_path
        raise LongTermRemoteUnavailableError("remote unavailable")


def _config(root: str) -> TwinrConfig:
    return TwinrConfig(
        project_root=root,
        personality_dir="personality",
        memory_markdown_path=str(Path(root) / "state" / "MEMORY.md"),
        long_term_memory_path=str(Path(root) / "state" / "chonkydb"),
    )


def _source() -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=("turn:test",),
        speaker="user",
        modality="voice",
    )


class LongTermStructuredStoreTests(unittest.TestCase):
    def test_atomic_json_write_survives_concurrent_writers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "race.json"
            errors: list[BaseException] = []

            def worker(index: int) -> None:
                try:
                    _write_json_atomic(target, {"writer": index})
                except BaseException as exc:
                    errors.append(exc)

            threads = [Thread(target=worker, args=(index,)) for index in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            payload = json.loads(target.read_text(encoding="utf-8"))

        self.assertIn(payload["writer"], range(8))

    def test_apply_consolidation_persists_objects_and_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermStructuredStore.from_config(_config(temp_dir))
            result = LongTermConsolidationResultV1(
                turn_id="turn:test",
                occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                episodic_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="episode:turn_test",
                        kind="episode",
                        summary="Conversation turn recorded for long-term memory.",
                        source=_source(),
                        status="active",
                        confidence=1.0,
                        slot_key="episode:turn:test",
                        value_key="turn:test",
                    ),
                ),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                        slot_key="relationship:user:main:wife",
                        value_key="person:janina",
                        attributes={"person_ref": "person:janina"},
                    ),
                ),
                deferred_objects=(),
                conflicts=(
                    LongTermMemoryConflictV1(
                        slot_key="contact:person:corinna_maier:phone",
                        candidate_memory_id="fact:corinna_phone_new",
                        existing_memory_ids=("fact:corinna_phone_old",),
                        question="I have more than one contact detail for this person. Which one should I use?",
                        reason="Conflicting active memories exist for slot contact:person:corinna_maier:phone.",
                    ),
                ),
                graph_edges=(
                    LongTermGraphEdgeCandidateV1(
                        source_ref="user:main",
                        edge_type="social_family_of",
                        target_ref="person:janina",
                        confidence=0.98,
                    ),
                ),
            )

            store.apply_consolidation(result)

            objects = store.load_objects()
            conflicts = store.load_conflicts()

        self.assertEqual(len(objects), 2)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].slot_key, "contact:person:corinna_maier:phone")

    def test_remote_primary_store_keeps_snapshots_off_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            result = LongTermConsolidationResultV1(
                turn_id="turn:test",
                occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                episodic_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="episode:turn_test",
                        kind="episode",
                        summary="Conversation turn recorded for long-term memory.",
                        source=_source(),
                        status="active",
                        confidence=1.0,
                    ),
                ),
                durable_objects=(),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            )

            store.apply_consolidation(result)
            loaded = store.load_objects()

        self.assertFalse(store.objects_path.exists())
        self.assertEqual(len(loaded), 1)
        self.assertEqual(remote_state.snapshots["objects"]["schema"], "twinr_memory_object_catalog_v2")
        self.assertEqual(len(remote_state.snapshots["objects"]["items"]), 1)
        self.assertEqual(len(remote_state.client.records_by_document_id), 1)

    def test_ensure_remote_snapshots_seeds_empty_remote_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)

            ensured = store.ensure_remote_snapshots()

        self.assertEqual(set(ensured), {"objects", "conflicts", "archive"})
        self.assertEqual(remote_state.snapshots["conflicts"]["schema"], "twinr_memory_conflict_catalog_v2")
        self.assertEqual(remote_state.snapshots["archive"]["schema"], "twinr_memory_archive_catalog_v2")
        self.assertEqual(remote_state.snapshots["objects"]["schema"], "twinr_memory_object_catalog_v2")
        self.assertEqual(remote_state.snapshots["conflicts"]["version"], 2)
        self.assertEqual(remote_state.snapshots["archive"]["version"], 2)
        self.assertEqual(remote_state.snapshots["objects"]["version"], 2)
        self.assertEqual(remote_state.snapshots["conflicts"]["items"], [])
        self.assertEqual(remote_state.snapshots["archive"]["items"], [])
        self.assertEqual(remote_state.snapshots["objects"]["items"], [])
        self.assertIn("written_at", remote_state.snapshots["conflicts"])
        self.assertIn("written_at", remote_state.snapshots["archive"])
        self.assertIn("written_at", remote_state.snapshots["objects"])

    def test_ensure_remote_snapshots_fails_closed_when_required_remote_snapshot_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)

            with self.assertRaises(LongTermRemoteUnavailableError):
                store.ensure_remote_snapshots()

    def test_remote_primary_store_persists_one_document_per_object(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            objects = tuple(
                LongTermMemoryObjectV1(
                    memory_id=f"fact:{index}",
                    kind="fact",
                    summary=f"Fact number {index} " + ("x" * 400),
                    source=_source(),
                    status="active",
                    confidence=0.9,
                )
                for index in range(4)
            )

            store.write_snapshot(objects=objects)
            loaded = store.load_objects()

        self.assertEqual(len(loaded), 4)
        self.assertEqual(len(remote_state.snapshots["objects"]["items"]), 4)
        self.assertEqual(len(remote_state.client.records_by_document_id), 4)

    def test_remote_primary_store_batches_fine_grained_remote_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _FakeChonkyClient(max_items_per_bulk=2)
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            remote_state.config.long_term_memory_migration_batch_size = 2
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            objects = tuple(
                LongTermMemoryObjectV1(
                    memory_id=f"fact:{index}",
                    kind="fact",
                    summary=f"Fact number {index}",
                    source=_source(),
                    status="active",
                    confidence=0.9,
                )
                for index in range(5)
            )

            store.write_snapshot(objects=objects)

        self.assertEqual(len(remote_state.snapshots["objects"]["items"]), 5)
        self.assertEqual(len(remote_state.client.records_by_document_id), 5)
        self.assertEqual(remote_state.client.bulk_calls, 3)

    def test_remote_primary_store_does_not_fallback_to_local_objects_when_remote_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "state" / "chonkydb"
            base_path.mkdir(parents=True, exist_ok=True)
            objects_path = base_path / "twinr_memory_objects_v1.json"
            objects_path.write_text(
                json.dumps({"schema": "twinr_memory_object_store", "version": 1, "objects": [{"memory_id": "fact:local"}]}),
                encoding="utf-8",
            )
            store = LongTermStructuredStore(base_path=base_path, remote_state=_FailingRemoteState())

            with self.assertRaises(LongTermRemoteUnavailableError):
                store.load_objects()

    def test_remote_primary_store_selects_relevant_objects_via_fine_grained_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:tea_preference",
                        kind="preference_fact",
                        summary="The user likes black tea.",
                        source=_source(),
                        status="active",
                        confidence=0.9,
                    ),
                )
            )

            relevant = store.select_relevant_objects(query_text="Janina", limit=1)

        self.assertEqual(tuple(item.memory_id for item in relevant), ("fact:janina_spouse",))

    def test_select_relevant_objects_prefers_query_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            store = LongTermStructuredStore.from_config(config)
            extractor = make_test_extractor()
            consolidator = LongTermMemoryConsolidator(truth_maintainer=LongTermTruthMaintainer())
            extraction = extractor.extract_conversation_turn(
                transcript=(
                    "Today is a beautiful Sunday, it is really warm. "
                    "My wife Janina is at the eye doctor and is getting eye laser treatment."
                ),
                response="I hope Janina's appointment goes smoothly.",
                occurred_at=datetime(2026, 3, 14, 10, 30, tzinfo=ZoneInfo("Europe/Berlin")),
                turn_id="turn:test",
            )
            result = consolidator.consolidate(extraction=extraction)
            store.apply_consolidation(result)

            relevant = store.select_relevant_objects(query_text="How is Janina today?", limit=3)

        summaries = [item.summary for item in relevant]
        self.assertIn("Janina is the user's wife.", summaries)
        self.assertTrue(any("eye laser treatment" in summary for summary in summaries))

    def test_store_merges_repeated_memory_ids_into_support_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermStructuredStore.from_config(_config(temp_dir))
            first = LongTermConsolidationResultV1(
                turn_id="turn:1",
                occurred_at=datetime(2026, 3, 14, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                episodic_objects=(),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_wife",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                        slot_key="relationship:user:main:wife",
                        value_key="person:janina",
                        attributes={"person_ref": "person:janina", "person_name": "Janina", "relation": "wife"},
                    ),
                ),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            )
            second = LongTermConsolidationResultV1(
                turn_id="turn:2",
                occurred_at=datetime(2026, 3, 14, 11, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                episodic_objects=(),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_wife",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=LongTermSourceRefV1(
                            source_type="conversation_turn",
                            event_ids=("turn:2",),
                            speaker="user",
                            modality="voice",
                        ),
                        status="active",
                        confidence=0.98,
                        slot_key="relationship:user:main:wife",
                        value_key="person:janina",
                        attributes={"person_ref": "person:janina", "person_name": "Janina", "relation": "wife"},
                    ),
                ),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            )

            store.apply_consolidation(first)
            store.apply_consolidation(second)

            objects = store.load_objects()

        relation = next(item for item in objects if item.memory_id == "fact:janina_wife")
        self.assertEqual((relation.attributes or {}).get("support_count"), 2)

    def test_select_relevant_objects_ignores_internal_numeric_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermStructuredStore.from_config(_config(temp_dir))
            result = LongTermConsolidationResultV1(
                turn_id="turn:test",
                occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                episodic_objects=(),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="pattern:print:button:afternoon",
                        kind="interaction_pattern_fact",
                        summary="Printed Twinr output was used in the afternoon.",
                        details="Low-confidence print usage pattern derived from a button print completion event.",
                        source=_source(),
                        status="active",
                        confidence=0.61,
                        slot_key="pattern:print:button:afternoon:2026-03-14",
                        value_key="printed_output",
                        attributes={"request_source": "button", "daypart": "afternoon", "support_count": 20},
                    ),
                ),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            )
            store.apply_consolidation(result)

            relevant = store.select_relevant_objects(query_text="calculate 27 times 14", limit=3)
            more = store.select_relevant_objects(query_text="convert 20 celsius to fahrenheit", limit=3)

        self.assertEqual(relevant, ())
        self.assertEqual(more, ())

    def test_review_objects_can_filter_and_rank_durable_items(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            store = LongTermStructuredStore.from_config(config)
            extractor = make_test_extractor()
            consolidator = LongTermMemoryConsolidator(truth_maintainer=LongTermTruthMaintainer())
            extraction = extractor.extract_conversation_turn(
                transcript=(
                    "Today is a beautiful Sunday, it is really warm. "
                    "My wife Janina is at the eye doctor and is getting eye laser treatment."
                ),
                response="I hope Janina's appointment goes smoothly.",
                occurred_at=datetime(2026, 3, 14, 10, 30, tzinfo=ZoneInfo("Europe/Berlin")),
                turn_id="turn:test",
            )
            store.apply_consolidation(consolidator.consolidate(extraction=extraction))

            review = store.review_objects(
                query_text="What happened to Janina today?",
                status="active",
                include_episodes=False,
                limit=4,
            )

        self.assertGreaterEqual(review.total_count, 2)
        self.assertTrue(all(item.kind != "episode" for item in review.items))
        self.assertTrue(any("Janina is the user's wife." == item.summary for item in review.items))
        self.assertTrue(any("eye laser treatment" in item.summary for item in review.items))

    def test_invalidate_object_marks_memory_invalid_and_drops_related_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermStructuredStore.from_config(_config(temp_dir))
            result = LongTermConsolidationResultV1(
                turn_id="turn:test",
                occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                episodic_objects=(),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:corinna_phone_old",
                        kind="contact_method_fact",
                        summary="Corinna Maier can be reached at +491761234.",
                        source=_source(),
                        status="active",
                        confidence=0.95,
                        slot_key="contact:person:corinna_maier:phone",
                        value_key="+491761234",
                    ),
                ),
                deferred_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:corinna_phone_new",
                        kind="contact_method_fact",
                        summary="Corinna Maier can be reached at +4940998877.",
                        source=_source(),
                        status="uncertain",
                        confidence=0.92,
                        slot_key="contact:person:corinna_maier:phone",
                        value_key="+4940998877",
                    ),
                ),
                conflicts=(
                    LongTermMemoryConflictV1(
                        slot_key="contact:person:corinna_maier:phone",
                        candidate_memory_id="fact:corinna_phone_new",
                        existing_memory_ids=("fact:corinna_phone_old",),
                        question="Which phone number should I use for Corinna Maier?",
                        reason="Conflicting phone numbers exist.",
                    ),
                ),
                graph_edges=(),
            )
            store.apply_consolidation(result)

            mutation = store.invalidate_object("fact:corinna_phone_new", reason="User said this is outdated.")
            store.apply_memory_mutation(mutation)
            objects = {item.memory_id: item for item in store.load_objects()}
            conflicts = store.load_conflicts()

        self.assertEqual(mutation.action, "invalidate")
        self.assertEqual(objects["fact:corinna_phone_new"].status, "invalid")
        self.assertEqual(objects["fact:corinna_phone_new"].attributes["invalidation_reason"], "User said this is outdated.")
        self.assertEqual(conflicts, ())

    def test_delete_object_removes_memory_and_cleans_reference_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermStructuredStore.from_config(_config(temp_dir))
            result = LongTermConsolidationResultV1(
                turn_id="turn:test",
                occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                episodic_objects=(),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_wife",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        slot_key="relationship:user:main:wife",
                        value_key="person:janina",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="summary:janina_thread",
                        kind="thread_summary",
                        summary="Ongoing thread about Janina and her appointments.",
                        source=_source(),
                        status="active",
                        confidence=0.88,
                        conflicts_with=("fact:janina_wife",),
                        supersedes=("fact:janina_wife",),
                    ),
                ),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            )
            store.apply_consolidation(result)

            mutation = store.delete_object("fact:janina_wife")
            store.apply_memory_mutation(mutation)
            objects = {item.memory_id: item for item in store.load_objects()}

        self.assertEqual(mutation.deleted_memory_ids, ("fact:janina_wife",))
        self.assertNotIn("fact:janina_wife", objects)
        self.assertEqual(objects["summary:janina_thread"].conflicts_with, ())
        self.assertEqual(objects["summary:janina_thread"].supersedes, ())


if __name__ == "__main__":
    unittest.main()
