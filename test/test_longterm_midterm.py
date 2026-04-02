from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.memory.longterm import (
    LongTermConversationTurn,
    LongTermMemoryObjectV1,
    LongTermMemoryReflector,
    LongTermMidtermStore,
    LongTermReflectionResultV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.reasoning.midterm import (
    _memory_object_to_prompt_payload,
    _midterm_reflection_schema,
    _normalize_reflection_result,
)
from twinr.memory.longterm.reasoning.turn_continuity import LongTermTurnContinuityCompiler
from twinr.memory.longterm.retrieval.retriever import LongTermRetriever
from twinr.memory.longterm.storage.store import LongTermStructuredStore
from twinr.memory.context_store import PromptContextStore
from twinr.memory.query_normalization import LongTermQueryProfile
from twinr.memory.longterm.reasoning.conflicts import LongTermConflictResolver
from twinr.memory.longterm.retrieval.subtext import LongTermSubtextBuilder
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore


def _config(root: str) -> TwinrConfig:
    return TwinrConfig(
        project_root=root,
        personality_dir="personality",
        memory_markdown_path=str(Path(root) / "state" / "MEMORY.md"),
        long_term_memory_enabled=True,
        long_term_memory_path=str(Path(root) / "state" / "chonkydb"),
        long_term_memory_recall_limit=4,
        long_term_memory_midterm_enabled=True,
        long_term_memory_midterm_limit=3,
        user_display_name="Erika",
    )


def _source(event_id: str) -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=(event_id,),
        speaker="user",
        modality="voice",
    )


class StubReflectionProgram:
    def compile_reflection(
        self,
        *,
        objects,
        timezone_name: str,
        packet_limit: int,
    ):
        del timezone_name
        del packet_limit
        object_ids = {item.memory_id for item in objects}
        if {"fact:janina_wife", "event:janina_eye_doctor_today"}.issubset(object_ids):
            return {
                "midterm_packets": [
                    {
                        "packet_id": "midterm:janina_today",
                        "kind": "recent_life_bundle",
                        "summary": "Janina is the user's wife and has an eye doctor appointment today.",
                        "details": "Recent context suggests that questions about Janina today may relate to that appointment.",
                        "source_memory_ids": [
                            "fact:janina_wife",
                            "event:janina_eye_doctor_today",
                        ],
                        "query_hints": [
                            "janina",
                            "wife",
                            "eye doctor",
                            "today",
                        ],
                        "sensitivity": "sensitive",
                        "valid_from": "2026-03-15",
                        "valid_to": "2026-03-15",
                        "attributes": {
                            "scope": "recent_window",
                            "focus_refs": ["person:janina"],
                        },
                    }
                ]
            }
        return {"midterm_packets": []}


class _EnglishPacketFromGermanSourceProgram:
    def compile_reflection(
        self,
        *,
        objects,
        timezone_name: str,
        packet_limit: int,
    ):
        del timezone_name
        del packet_limit
        object_ids = {item.memory_id for item in objects}
        if "event:lea_soup_dropoff" not in object_ids:
            return {"midterm_packets": []}
        return {
            "midterm_packets": [
                {
                    "packet_id": "midterm:lea_soup_dropoff",
                    "kind": "upcoming_event",
                    "summary": "Lea will bring a thermos of lentil soup this evening.",
                    "details": "At 7 PM Lea will drop off homemade lentil soup in a thermos.",
                    "source_memory_ids": ["event:lea_soup_dropoff"],
                    "query_hints": ["lea", "thermos", "lentil soup", "7 pm"],
                    "sensitivity": "normal",
                    "valid_from": None,
                    "valid_to": None,
                }
            ]
        }


class _FakeRemoteState:
    def __init__(self) -> None:
        self.enabled = True
        self.required = False
        self.namespace = "test-namespace"
        self.client = _FakeMidtermChonkyClient()
        self.read_client = self.client
        self.write_client = self.client
        self.config = SimpleNamespace(
            long_term_memory_migration_enabled=False,
            long_term_memory_migration_batch_size=64,
            long_term_memory_remote_read_timeout_s=8.0,
            long_term_memory_remote_write_timeout_s=15.0,
            long_term_memory_remote_flush_timeout_s=60.0,
            long_term_memory_remote_bulk_request_max_bytes=512 * 1024,
            long_term_memory_remote_shard_max_content_chars=1000,
            long_term_memory_remote_max_content_chars=2_000_000,
            long_term_memory_remote_read_cache_ttl_s=0.0,
        )
        self.snapshots: dict[str, dict[str, object]] = {}
        self.load_calls: list[dict[str, object]] = []

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        self.load_calls.append({"snapshot_kind": snapshot_kind, "local_path": local_path})
        del local_path
        payload = self.snapshots.get(snapshot_kind)
        return dict(payload) if isinstance(payload, dict) else None

    def save_snapshot(self, *, snapshot_kind: str, payload):
        self.snapshots[snapshot_kind] = dict(payload)


class _FakeMidtermChonkyClient:
    def __init__(self) -> None:
        self._next_document_id = 1
        self.records_by_document_id: dict[str, dict[str, object]] = {}
        self.records_by_uri: dict[str, dict[str, object]] = {}

    def store_records_bulk(self, request):
        items = tuple(getattr(request, "items", ()))
        response_items: list[dict[str, object]] = []
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
        del include_content, max_content_chars
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
        allowed = {str(value) for value in payload.get("allowed_doc_ids", ()) if str(value)}
        ranked: list[dict[str, object]] = []
        for document_id, record in self.records_by_document_id.items():
            if allowed and document_id not in allowed:
                continue
            content = str(record.get("content") or "").lower()
            if query_text == "__allowed_doc_ids__" and allowed:
                pass
            elif query_text and query_text not in content:
                continue
            ranked.append(
                {
                    "payload_id": document_id,
                    "document_id": document_id,
                    "relevance_score": 1.0,
                    "metadata": dict(record.get("metadata") or {}),
                    "payload": dict(record.get("payload") or {}),
                    "content": record.get("content"),
                    "source_index": "fulltext",
                    "candidate_origin": "fulltext",
                }
            )
        return SimpleNamespace(
            success=True,
            mode="advanced",
            results=tuple(SimpleNamespace(**item) for item in ranked),
            indexes_used=("fulltext",),
            raw={"results": [dict(item) for item in ranked]},
        )


class LongTermMidtermTests(unittest.TestCase):
    def test_midterm_schema_is_openai_strict_compatible(self) -> None:
        schema = _midterm_reflection_schema(max_packets=3)
        packet_schema = schema["properties"]["midterm_packets"]["items"]
        properties = packet_schema["properties"]
        required = packet_schema["required"]

        self.assertEqual(set(required), set(properties))
        self.assertEqual(properties["details"]["anyOf"][1]["type"], "null")
        self.assertEqual(properties["valid_from"]["anyOf"][1]["type"], "null")
        self.assertEqual(properties["valid_to"]["anyOf"][1]["type"], "null")

    def test_reflector_can_compile_midterm_packets(self) -> None:
        reflector = LongTermMemoryReflector(program=StubReflectionProgram(), midterm_packet_limit=3)
        wife = LongTermMemoryObjectV1(
            memory_id="fact:janina_wife",
            kind="fact",
            summary="Janina is the user's wife.",
            source=_source("turn:1"),
            status="active",
            confidence=0.98,
            slot_key="relationship:user:main:wife",
            value_key="person:janina",
            sensitivity="private",
            attributes={
                "person_ref": "person:janina",
                "person_name": "Janina",
                "relation": "wife",
                "fact_type": "relationship",
                "support_count": 2,
            },
        )
        appointment = LongTermMemoryObjectV1(
            memory_id="event:janina_eye_doctor_today",
            kind="event",
            summary="Janina has an eye doctor appointment today.",
            source=_source("turn:2"),
            status="active",
            confidence=0.93,
            slot_key="event:person:janina:eye_doctor:2026-03-15",
            value_key="appointment:janina:eye_doctor:2026-03-15",
            sensitivity="sensitive",
            valid_from="2026-03-15",
            valid_to="2026-03-15",
            attributes={
                "person_ref": "person:janina",
                "person_name": "Janina",
                "action": "eye doctor appointment",
                "memory_domain": "appointment",
            },
        )

        result = reflector.reflect(objects=(wife, appointment))

        self.assertEqual(len(result.midterm_packets), 1)
        packet = result.midterm_packets[0]
        self.assertEqual(packet.packet_id, "midterm:janina_today")
        self.assertEqual(packet.kind, "recent_life_bundle")
        self.assertIn("Janina", packet.summary)
        self.assertIn("eye doctor", packet.summary)
        self.assertEqual(packet.source_memory_ids, ("fact:janina_wife", "event:janina_eye_doctor_today"))

    def test_reflector_preserves_source_language_hints_for_canonical_midterm_packets(self) -> None:
        dropoff = LongTermMemoryObjectV1(
            memory_id="event:lea_soup_dropoff",
            kind="event",
            summary="Lea bringt heute Abend um 19 Uhr eine Thermoskanne mit Linsensuppe vorbei.",
            details="Heute Abend kommt Lea vorbei und bringt dir eine Thermoskanne mit selbstgemachter Linsensuppe.",
            source=_source("turn:lea"),
            status="active",
            confidence=0.95,
            slot_key="event:lea:soup_dropoff:2026-03-21T19:00:00+01:00",
            value_key="event:lea:soup_dropoff",
            sensitivity="normal",
            valid_from="2026-03-21T19:00:00+01:00",
            valid_to="2026-03-21T20:00:00+01:00",
            attributes={
                "person_name": "Lea",
                "item": "Thermoskanne mit Linsensuppe",
                "memory_domain": "delivery",
            },
        )

        prompt_payload = _memory_object_to_prompt_payload(dropoff)
        normalized = _normalize_reflection_result(
            _EnglishPacketFromGermanSourceProgram().compile_reflection(
                objects=(dropoff,),
                timezone_name="Europe/Berlin",
                packet_limit=3,
            ),
            valid_memory_ids={"event:lea_soup_dropoff"},
            packet_limit=3,
            source_payload_by_memory_id={"event:lea_soup_dropoff": prompt_payload},
        )

        self.assertEqual(len(normalized["midterm_packets"]), 1)
        packet = LongTermMidtermStore.packet_type.from_payload(normalized["midterm_packets"][0])
        self.assertIn("thermos", packet.query_hints)
        self.assertTrue(
            any("Thermoskanne" in hint or "Linsensuppe" in hint for hint in packet.query_hints),
            packet.query_hints,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermMidtermStore.from_config(_config(temp_dir))
            store.save_packets(packets=(packet,))
            matches = store.select_relevant_packets("Was bringt Lea heute Abend vorbei?", limit=2)

        self.assertEqual([item.packet_id for item in matches], ["midterm:lea_soup_dropoff"])

    def test_turn_continuity_compiler_preserves_source_language_turn_recall(self) -> None:
        packet = LongTermTurnContinuityCompiler().compile_packet(
            turn=LongTermConversationTurn(
                transcript="Meine Tochter Lea bringt mir heute Abend eine Thermoskanne mit Linsensuppe vorbei.",
                response="Ich merke mir, dass Lea dir heute Abend die Thermoskanne mit Linsensuppe bringt.",
                source="conversation",
            )
        )

        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.kind, "recent_turn_continuity")
        self.assertIn("latest user-assistant turn", packet.summary)
        self.assertIn("Thermoskanne", packet.details or "")
        self.assertIn("Linsensuppe", packet.details or "")
        self.assertIn("lea", packet.query_hints)
        self.assertTrue(
            any("thermoskanne" in hint.lower() or "linsensuppe" in hint.lower() for hint in packet.query_hints),
            packet.query_hints,
        )
        self.assertIn("recent conversation", packet.query_hints)
        self.assertIn("worueber gesprochen", packet.query_hints)
        self.assertEqual(packet.attributes["persistence_scope"], "turn_continuity")
        self.assertIn("Thermoskanne", packet.attributes["transcript_excerpt"])
        self.assertIn("Lea", packet.attributes["response_excerpt"])

    def test_turn_continuity_packets_match_generic_recap_queries_in_both_languages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            store = LongTermMidtermStore.from_config(config)
            store.save_packets(
                packets=(
                    store.packet_type(
                        packet_id="midterm:turn:recap",
                        kind="recent_turn_continuity",
                        summary="Recent conversation continuity from the latest user-assistant turn.",
                        details="User said: Wir haben ueber Medikamente und den Arzttermin gesprochen.",
                        query_hints=("medikamente", "arzttermin"),
                        attributes={"persistence_scope": "turn_continuity"},
                    ),
                )
            )

            german_packets = store.select_relevant_packets("Worüber haben wir heute gesprochen?", limit=2)
            english_packets = store.select_relevant_packets("What did we talk about today?", limit=2)
            control_packets = store.select_relevant_packets("Was ist ein Regenbogen?", limit=2)

        self.assertEqual([item.packet_id for item in german_packets], ["midterm:turn:recap"])
        self.assertEqual([item.packet_id for item in english_packets], ["midterm:turn:recap"])
        self.assertEqual(control_packets, ())

    def test_midterm_store_selects_query_relevant_packets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            store = LongTermMidtermStore.from_config(config)
            store.save_packets(
                packets=(
                    store.packet_type(
                        packet_id="midterm:janina_today",
                        kind="recent_life_bundle",
                        summary="Janina has an eye doctor appointment today.",
                        details="Useful when the user asks about Janina today.",
                        source_memory_ids=("event:janina_eye_doctor_today",),
                        query_hints=("janina", "eye doctor", "today"),
                        sensitivity="sensitive",
                    ),
                    store.packet_type(
                        packet_id="midterm:tea_shopping",
                        kind="shopping_bundle",
                        summary="The user usually buys tea at Laden Seidel.",
                        details="Useful for nearby tea suggestions.",
                        source_memory_ids=("fact:tea_store",),
                        query_hints=("tea", "laden seidel"),
                        sensitivity="normal",
                    ),
                )
            )

            janina_packets = store.select_relevant_packets("How is Janina doing today?", limit=2)
            math_packets = store.select_relevant_packets("What is 27 times 14?", limit=2)

        self.assertEqual([item.packet_id for item in janina_packets], ["midterm:janina_today"])
        self.assertEqual(math_packets, ())

    def test_midterm_store_matches_compound_terms_without_off_topic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            store = LongTermMidtermStore.from_config(config)
            store.save_packets(
                packets=(
                    store.packet_type(
                        packet_id="midterm:jam_restart",
                        kind="adaptive_restart_recall_policy",
                        summary="Persistent restart recall for this stable durable memory: Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Use this packet as direct grounding after fresh runtime restarts when the current turn overlaps the same topic.",
                        source_memory_ids=("fact:jam_preference_new",),
                        query_hints=("preference:breakfast:jam", "bestaetigt"),
                        attributes={"persistence_scope": "restart_recall"},
                    ),
                )
            )

            jam_packets = store.select_relevant_packets(
                "Welche Marmelade ist jetzt als bestaetigt gespeichert?",
                limit=2,
            )
            control_packets = store.select_relevant_packets("Was ist ein Regenbogen?", limit=2)

        self.assertEqual([item.packet_id for item in jam_packets], ["midterm:jam_restart"])
        self.assertEqual(control_packets, ())

    def test_ensure_remote_snapshot_seeds_empty_midterm_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermMidtermStore(
                base_path=Path(temp_dir) / "state" / "chonkydb",
                remote_state=remote_state,
            )

            with self.assertNoLogs("twinr.memory.longterm.storage.midterm_store", level=logging.WARNING):
                created = store.ensure_remote_snapshot()
                remote_head = store.load_remote_current_head()

        self.assertTrue(created)
        self.assertIsNotNone(remote_head)
        assert remote_head is not None
        self.assertEqual(remote_head["schema"], "twinr_memory_midterm_catalog_v3")
        self.assertEqual(remote_head["items_count"], 0)
        self.assertEqual(store.remote_current_packet_ids(), ())
        self.assertIn("midterm", remote_state.snapshots)
        self.assertEqual(remote_state.snapshots["midterm"]["schema"], "twinr_memory_midterm_catalog_v3")
        self.assertTrue(
            any(uri.endswith("/midterm/catalog/current") for uri in remote_state.client.records_by_uri),
            remote_state.client.records_by_uri,
        )

    def test_midterm_readiness_bootstrap_keeps_fresh_required_namespace_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            store = LongTermMidtermStore(
                base_path=Path(temp_dir) / "state" / "chonkydb",
                remote_state=remote_state,
            )

            created = store.ensure_remote_snapshot_for_readiness()
            remote_head = store.probe_remote_current_head_for_readiness()

        self.assertFalse(created)
        self.assertIsNotNone(remote_head)
        assert remote_head is not None
        self.assertEqual(remote_head["schema"], "twinr_memory_midterm_store")
        self.assertEqual(remote_head["packets"], [])
        self.assertEqual(remote_head["written_at"], "1970-01-01T00:00:00+00:00")
        self.assertNotIn("midterm", remote_state.snapshots)
        self.assertFalse(
            any(uri.endswith("/midterm/catalog/current") for uri in remote_state.client.records_by_uri),
            remote_state.client.records_by_uri,
        )

    def test_ensure_remote_snapshot_uses_compatible_current_head_when_direct_head_lags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            primary_store = LongTermMidtermStore(
                base_path=Path(temp_dir) / "state" / "chonkydb",
                remote_state=remote_state,
            )
            self.assertTrue(primary_store.ensure_remote_snapshot())
            lagging_uris = tuple(
                uri for uri in remote_state.client.records_by_uri if uri.endswith("/midterm/catalog/current")
            )
            for uri in lagging_uris:
                remote_state.client.records_by_uri.pop(uri, None)

            fresh_store = LongTermMidtermStore(
                base_path=Path(temp_dir) / "fresh" / "state" / "chonkydb",
                remote_state=remote_state,
            )

            created = fresh_store.ensure_remote_snapshot()
            remote_head = fresh_store.load_remote_current_head()

        self.assertFalse(created)
        self.assertIsNotNone(remote_head)
        assert remote_head is not None
        self.assertEqual(remote_head["schema"], "twinr_memory_midterm_catalog_v3")
        self.assertEqual(remote_head["items_count"], 0)
        self.assertTrue(remote_state.load_calls)
        self.assertTrue(all(call["snapshot_kind"] == "midterm" for call in remote_state.load_calls))
        self.assertIn("midterm", remote_state.snapshots)

    def test_probe_remote_current_head_uses_compatible_current_head_when_direct_head_lags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            primary_store = LongTermMidtermStore(
                base_path=Path(temp_dir) / "state" / "chonkydb",
                remote_state=remote_state,
            )
            self.assertTrue(primary_store.ensure_remote_snapshot())
            lagging_uris = tuple(
                uri for uri in remote_state.client.records_by_uri if uri.endswith("/midterm/catalog/current")
            )
            for uri in lagging_uris:
                remote_state.client.records_by_uri.pop(uri, None)

            fresh_store = LongTermMidtermStore(
                base_path=Path(temp_dir) / "fresh" / "state" / "chonkydb",
                remote_state=remote_state,
            )

            remote_head = fresh_store.probe_remote_current_head()

        self.assertIsNotNone(remote_head)
        assert remote_head is not None
        self.assertEqual(remote_head["schema"], "twinr_memory_midterm_catalog_v3")
        self.assertTrue(remote_state.load_calls)
        self.assertTrue(all(call["snapshot_kind"] == "midterm" for call in remote_state.load_calls))

    def test_remote_turn_continuity_packets_match_generic_recap_queries_after_remote_hydration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            primary_store = LongTermMidtermStore(
                base_path=Path(temp_dir) / "state" / "chonkydb",
                remote_state=remote_state,
            )
            primary_store.save_packets(
                packets=(
                    primary_store.packet_type(
                        packet_id="midterm:turn:remote_recap",
                        kind="recent_turn_continuity",
                        summary="Recent conversation continuity from the latest user-assistant turn.",
                        details="User said: Wir haben ueber Medikamente und den Arzttermin gesprochen.",
                        query_hints=("medikamente", "arzttermin"),
                        attributes={"persistence_scope": "turn_continuity"},
                    ),
                )
            )
            fresh_store = LongTermMidtermStore(
                base_path=Path(temp_dir) / "fresh" / "state" / "chonkydb",
                remote_state=remote_state,
            )

            german_packets = fresh_store.select_relevant_packets("Worüber haben wir heute gesprochen?", limit=2)
            english_packets = fresh_store.select_relevant_packets("What did we talk about today?", limit=2)
            control_packets = fresh_store.select_relevant_packets("Was ist ein Regenbogen?", limit=2)

        self.assertEqual([item.packet_id for item in german_packets], ["midterm:turn:remote_recap"])
        self.assertEqual([item.packet_id for item in english_packets], ["midterm:turn:remote_recap"])
        self.assertEqual(control_packets, ())

    def test_load_packets_uses_catalog_entry_projection_when_midterm_item_documents_lag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            primary_store = LongTermMidtermStore(
                base_path=Path(temp_dir) / "state" / "chonkydb",
                remote_state=remote_state,
            )
            packet = primary_store.packet_type(
                packet_id="midterm:corinna_today",
                kind="recent_contact_bundle",
                summary="Corinna Maier is a recent practical contact.",
                details="Useful when the user asks for the Corinna number.",
                source_memory_ids=("fact:corinna_phone_old",),
                query_hints=("corinna", "phone", "number"),
                sensitivity="normal",
                attributes={"person_ref": "person:corinna_maier"},
            )
            primary_store.save_packets(packets=(packet,))
            item_uri = primary_store._remote_midterm._catalog.item_uri(
                snapshot_kind="midterm",
                item_id="midterm:corinna_today",
            )
            item_record = remote_state.client.records_by_uri.pop(item_uri, None)
            if isinstance(item_record, dict):
                remote_state.client.records_by_document_id.pop(str(item_record.get("document_id") or ""), None)

            fresh_store = LongTermMidtermStore(
                base_path=Path(temp_dir) / "fresh" / "state" / "chonkydb",
                remote_state=remote_state,
            )

            packets = fresh_store.load_packets()
            selected = fresh_store.select_relevant_packets("Corinna phone number", limit=2)

        self.assertEqual([item.packet_id for item in packets], ["midterm:corinna_today"])
        self.assertEqual([item.packet_id for item in selected], ["midterm:corinna_today"])

    def test_save_packets_writes_world_readable_midterm_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermMidtermStore(base_path=Path(temp_dir) / "state" / "chonkydb")

            store.save_packets(
                packets=(
                    store.packet_type(
                        packet_id="midterm:tea_preference",
                        kind="preference_bundle",
                        summary="The user prefers Oolong tea in the afternoon.",
                        source_memory_ids=("fact:drink_preference",),
                        query_hints=("oolong", "tea", "afternoon"),
                        sensitivity="normal",
                    ),
                )
            )

            mode = stat.S_IMODE(store.packets_path.stat().st_mode)

        self.assertEqual(mode, 0o644)

    def test_load_packets_ignores_missing_local_snapshot_without_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermMidtermStore(base_path=Path(temp_dir) / "state" / "chonkydb")

            with self.assertNoLogs("twinr.memory.longterm.storage.midterm_store", level=logging.WARNING):
                packets = store.load_packets()

        self.assertEqual(packets, ())

    def test_load_packets_uses_remote_current_head_without_warning_when_local_cache_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermMidtermStore(
                base_path=Path(temp_dir) / "state" / "chonkydb",
                remote_state=remote_state,
            )
            store.save_packets(
                packets=(
                    store.packet_type(
                        packet_id="midterm:tea_preference",
                        kind="preference_bundle",
                        summary="The user prefers Oolong tea in the afternoon.",
                        details="Useful for drink recommendations and recall questions.",
                        source_memory_ids=("fact:drink_preference",),
                        query_hints=("oolong", "tea", "afternoon"),
                        sensitivity="normal",
                    ),
                )
            )
            store.packets_path.unlink()

            with self.assertNoLogs("twinr.memory.longterm.storage.midterm_store", level=logging.WARNING):
                packets = store.load_packets()

        self.assertEqual([item.packet_id for item in packets], ["midterm:tea_preference"])

    def test_replace_packets_with_attribute_preserves_other_packet_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            store = LongTermMidtermStore.from_config(config)
            store.save_packets(
                packets=(
                    store.packet_type(
                        packet_id="midterm:reflection",
                        kind="recent_life_bundle",
                        summary="Reflection packet",
                        source_memory_ids=("fact:reflection",),
                        query_hints=("reflection",),
                        sensitivity="normal",
                    ),
                    store.packet_type(
                        packet_id="adaptive:restart:old",
                        kind="adaptive_restart_recall_policy",
                        summary="Old restart packet",
                        source_memory_ids=("fact:old",),
                        query_hints=("old",),
                        sensitivity="normal",
                        attributes={"persistence_scope": "restart_recall"},
                    ),
                )
            )

            store.replace_packets_with_attribute(
                packets=(
                    store.packet_type(
                        packet_id="adaptive:restart:new",
                        kind="adaptive_restart_recall_policy",
                        summary="New restart packet",
                        source_memory_ids=("fact:new",),
                        query_hints=("new",),
                        sensitivity="normal",
                        attributes={"persistence_scope": "restart_recall"},
                    ),
                ),
                attribute_key="persistence_scope",
                attribute_value="restart_recall",
            )
            loaded = store.load_packets()

        self.assertEqual(
            [item.packet_id for item in loaded],
            ["adaptive:restart:new", "midterm:reflection"],
        )

    def test_apply_reflection_preserves_restart_recall_and_recent_turn_continuity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            store = LongTermMidtermStore.from_config(config)
            store.save_packets(
                packets=(
                    store.packet_type(
                        packet_id="midterm:reflection:old",
                        kind="conversation_context",
                        summary="Old reflection packet",
                        source_memory_ids=("fact:old_reflection",),
                        query_hints=("reflection",),
                        sensitivity="normal",
                    ),
                    store.packet_type(
                        packet_id="midterm:restart",
                        kind="adaptive_restart_recall_policy",
                        summary="Stable restart recall packet",
                        source_memory_ids=("fact:stable",),
                        query_hints=("stable",),
                        sensitivity="normal",
                        attributes={"persistence_scope": "restart_recall"},
                    ),
                    store.packet_type(
                        packet_id="midterm:turn:latest",
                        kind="recent_turn_continuity",
                        summary="The last turn was about the doctor visit.",
                        source_memory_ids=("turn:latest",),
                        query_hints=("doctor visit",),
                        sensitivity="normal",
                        updated_at=datetime(2026, 3, 22, 8, 30, tzinfo=ZoneInfo("UTC")),
                        attributes={"persistence_scope": "turn_continuity"},
                    ),
                )
            )

            store.apply_reflection(
                LongTermReflectionResultV1(
                    reflected_objects=(),
                    created_summaries=(),
                    midterm_packets=(
                        store.packet_type(
                            packet_id="midterm:reflection:new",
                            kind="conversation_context",
                            summary="Fresh reflection packet",
                            source_memory_ids=("fact:new_reflection",),
                            query_hints=("fresh reflection",),
                            sensitivity="normal",
                        ),
                    ),
                )
            )
            loaded = store.load_packets()

        self.assertEqual(
            [item.packet_id for item in loaded],
            ["midterm:reflection:new", "midterm:restart", "midterm:turn:latest"],
        )

    def test_retriever_includes_midterm_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            prompt_context_store = PromptContextStore.from_config(config)
            graph_store = TwinrPersonalGraphStore.from_config(config)
            object_store = LongTermStructuredStore.from_config(config)
            midterm_store = LongTermMidtermStore.from_config(config)
            midterm_store.save_packets(
                packets=(
                    midterm_store.packet_type(
                        packet_id="midterm:janina_today",
                        kind="recent_life_bundle",
                        summary="Janina has an eye doctor appointment today.",
                        details="Useful when the user asks about Janina today.",
                        source_memory_ids=("event:janina_eye_doctor_today",),
                        query_hints=("janina", "eye doctor", "today"),
                        sensitivity="sensitive",
                    ),
                )
            )
            retriever = LongTermRetriever(
                config=config,
                prompt_context_store=prompt_context_store,
                graph_store=graph_store,
                object_store=object_store,
                midterm_store=midterm_store,
                conflict_resolver=LongTermConflictResolver(),
                subtext_builder=LongTermSubtextBuilder(config=config, graph_store=graph_store),
            )

            context = retriever.build_context(
                query=LongTermQueryProfile.from_text(
                    "Wie geht es Janina heute?",
                    canonical_english_text="How is Janina doing today?",
                ),
                original_query_text="Wie geht es Janina heute?",
            )

        self.assertIsNotNone(context.midterm_context)
        self.assertIn("twinr_long_term_midterm_context_v1", context.midterm_context or "")
        self.assertIn("Janina has an eye doctor appointment today.", context.midterm_context or "")


if __name__ == "__main__":
    unittest.main()
