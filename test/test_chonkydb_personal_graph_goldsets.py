from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.memory.chonkydb import TwinrPersonalGraphStore


_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "chonkydb_personal_graph_goldset.json"
_TEST_CORINNA_PHONE = "5551234"
_TEST_CORINNA_NEIGHBOR_PHONE = "5559988"
_TEST_CORINNA_ALT_PHONE = "5557777"
_TEST_ANNA_PHONE = "5552345"


def _fixture_payload() -> dict[str, object]:
    today = datetime.now(ZoneInfo("Europe/Berlin")).date()
    tomorrow = today + timedelta(days=1)
    text = _FIXTURE_PATH.read_text(encoding="utf-8")
    text = text.replace("__TODAY__", today.isoformat())
    text = text.replace("__TOMORROW__", tomorrow.isoformat())
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise AssertionError("fixture payload must be a JSON object")
    return payload


def _memory_payload(context: str) -> dict[str, object]:
    _header, _sep, payload = context.partition("\n")
    return json.loads(payload)


class TwinrPersonalGraphGoldsetTests(unittest.TestCase):
    def _make_store(self, temp_dir: str) -> TwinrPersonalGraphStore:
        config = TwinrConfig(
            project_root=temp_dir,
            long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
            user_display_name="Erika",
            openai_web_search_timezone="Europe/Berlin",
        )
        return TwinrPersonalGraphStore.from_config(config)

    def _load_fixture(self, store: TwinrPersonalGraphStore) -> dict[str, object]:
        payload = _fixture_payload()
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return payload

    def test_goldset_fixture_loads_and_preserves_core_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._make_store(temp_dir)
            expected = self._load_fixture(store)

            document = store.load_document()

        self.assertEqual(document.schema_name, "twinr_graph")
        self.assertEqual(document.schema_version, 2)
        self.assertEqual(document.subject_node_id, "user:main")
        self.assertEqual(len(document.nodes), len(expected["nodes"]))
        self.assertEqual(len(document.edges), len(expected["edges"]))
        self.assertEqual(len([node for node in document.nodes if node.node_type == "person"]), 3)
        self.assertEqual(len([edge for edge in document.edges if edge.edge_type == "general_has_contact_method"]), 4)

    def test_goldset_lookup_queries_cover_exact_role_family_and_miss_cases(self) -> None:
        cases = (
            ({"name": "Corinna"}, "needs_clarification", None, None, 2),
            ({"name": "Corinna", "family_name": "Maier"}, "found", "Corinna Maier", (_TEST_CORINNA_PHONE,), 0),
            ({"name": "Corinna", "role": "Physiotherapist"}, "found", "Corinna Maier", (_TEST_CORINNA_PHONE,), 0),
            ({"name": "Corinna", "role": "Neighbor"}, "found", "Corinna Schmidt", (_TEST_CORINNA_NEIGHBOR_PHONE,), 0),
            ({"name": "Anna"}, "found", "Anna Becker", (_TEST_ANNA_PHONE,), 0),
            ({"name": "Holger"}, "not_found", None, None, 0),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._make_store(temp_dir)
            self._load_fixture(store)

            for lookup_args, expected_status, expected_label, expected_phones, expected_options in cases:
                result = store.lookup_contact(**lookup_args)
                self.assertEqual(result.status, expected_status, lookup_args)
                if expected_label is not None:
                    self.assertIsNotNone(result.match, lookup_args)
                    self.assertEqual(result.match.label, expected_label, lookup_args)
                    self.assertEqual(result.match.phones, expected_phones, lookup_args)
                else:
                    self.assertIsNone(result.match, lookup_args)
                self.assertEqual(len(result.options), expected_options, lookup_args)

    def test_goldset_contact_prompt_context_only_surfaces_relevant_people(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._make_store(temp_dir)
            self._load_fixture(store)

            corinna_context = _memory_payload(store.build_prompt_context("Wie ist die Nummer von Corinna?") or "")
            physio_context = _memory_payload(store.build_prompt_context("Wie ist die Nummer meiner Physiotherapeutin?") or "")
            daughter_context = _memory_payload(store.build_prompt_context("Wie ist die Mail von meiner Tochter?") or "")

        corinna_names = {item["name"] for item in corinna_context["contacts"]}
        self.assertIn("Corinna Maier", corinna_names)
        self.assertIn("Corinna Schmidt", corinna_names)
        self.assertNotIn("Anna Becker", corinna_names)

        physio_roles = {item["name"]: item.get("role") for item in physio_context["contacts"]}
        self.assertEqual(physio_roles["Corinna Maier"], "Physiotherapist")
        self.assertEqual(physio_roles["Corinna Schmidt"], "Neighbor")
        self.assertEqual(physio_roles["Anna Becker"], "Daughter")

        daughter_lookup = next(item for item in daughter_context["contacts"] if item["name"] == "Anna Becker")
        self.assertEqual(daughter_lookup["role"], "Daughter")
        self.assertEqual(daughter_lookup["emails"], ["anna@example.com"])

    def test_goldset_prompt_context_is_canonical_english_even_for_german_queries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._make_store(temp_dir)
            self._load_fixture(store)

            shopping_context = store.build_prompt_context("Wo kann ich heute Kaffee kaufen?")
            weather_context = store.build_prompt_context("Wie wird das Wetter heute?")
            joke_context = store.build_prompt_context("Erzaehl mir einen Witz.")

        self.assertIsNotNone(shopping_context)
        self.assertIn("Structured long-term memory graph for this turn.", shopping_context or "")
        self.assertIn('"value": "Melitta"', shopping_context or "")
        self.assertIn('"value": "Store Z"', shopping_context or "")
        self.assertIn('"for_product": "coffee"', shopping_context or "")
        self.assertNotIn("Kaffee", shopping_context or "")
        self.assertNotIn("Aktueller Plan", shopping_context or "")

        self.assertIsNotNone(weather_context)
        self.assertIn("go for a walk", weather_context or "")
        self.assertIn('"when": "today"', weather_context or "")
        self.assertNotIn("spazieren gehen", weather_context or "")
        self.assertIsNotNone(joke_context)
        self.assertIn("Structured long-term memory graph for this turn.", joke_context or "")

    def test_goldset_structured_context_contains_multihop_store_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._make_store(temp_dir)
            self._load_fixture(store)

            payload = _memory_payload(store.build_prompt_context("Wo bekomme ich heute meinen Lieblingskaffee in der Naehe?") or "")

        store_entry = next(item for item in payload["preferences"] if item["value"] == "Store Z")
        self.assertEqual(store_entry["value"], "Store Z")
        self.assertEqual(store_entry["for_product"], "coffee")
        self.assertEqual(store_entry["type"], "preference")
        self.assertEqual(store_entry["associations"], [{"relation": "carries", "label": "Melitta", "type": "brand"}])
        self.assertTrue(store_entry["nearby"])

    def test_goldset_structured_context_contains_temporal_plan_person_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._make_store(temp_dir)
            self._load_fixture(store)

            payload = _memory_payload(store.build_prompt_context("Wer macht morgen meine Physiotherapie?") or "")

        physio_plan = next(item for item in payload["plans"] if item["summary"] == "physiotherapy")
        self.assertEqual(physio_plan["when"], "tomorrow")
        self.assertIn("Corinna Maier", physio_plan["related_people"])

    def test_goldset_context_keeps_contact_disambiguation_facts_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._make_store(temp_dir)
            self._load_fixture(store)

            payload = _memory_payload(
                store.build_prompt_context("Welche Nummer hat die Corinna von meiner Physiotherapie morgen?") or ""
            )

        contacts = {item["name"]: item for item in payload["contacts"]}
        self.assertEqual(contacts["Corinna Maier"]["role"], "Physiotherapist")
        self.assertEqual(contacts["Corinna Maier"]["phones"], [_TEST_CORINNA_PHONE])
        self.assertEqual(contacts["Corinna Schmidt"]["role"], "Neighbor")

    def test_goldset_persistence_roundtrip_keeps_new_contact_details_and_new_memories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._make_store(temp_dir)
            self._load_fixture(store)

            updated_contact = store.remember_contact(
                given_name="Corinna",
                family_name="Maier",
                phone=_TEST_CORINNA_PHONE,
                email="corinna.maier@example.com",
                role="Physiotherapist",
            )
            new_preference = store.remember_preference(
                category="drink",
                value="Chamomile tea",
                sentiment="like",
                details="liked in the evening",
            )
            new_plan = store.remember_plan(
                summary="water the flowers",
                when_text="today",
            )

            reloaded = self._make_store(temp_dir)
            lookup = reloaded.lookup_contact(name="Corinna", role="Physiotherapist")
            plan_context = reloaded.build_prompt_context("Was wollte ich heute noch machen?")

        self.assertEqual(updated_contact.status, "updated")
        self.assertEqual(new_preference.status, "updated")
        self.assertIn(new_plan.status, {"created", "updated"})
        self.assertEqual(lookup.status, "found")
        self.assertIn("corinna.maier@example.com", lookup.match.emails)
        self.assertIsNotNone(plan_context)
        self.assertIn("water the flowers", plan_context or "")

    def test_goldset_contact_updates_refuse_silent_merge_when_number_is_new(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._make_store(temp_dir)
            self._load_fixture(store)

            result = store.remember_contact(
                given_name="Corinna",
                phone=_TEST_CORINNA_ALT_PHONE,
            )

        self.assertEqual(result.status, "needs_clarification")
        self.assertIn("Corinna", result.question or "")
        self.assertEqual(len(result.options), 2)

    def test_goldset_first_name_contact_can_be_enriched_later_when_number_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._make_store(temp_dir)

            created = store.remember_contact(
                given_name="Corinna",
                phone=_TEST_CORINNA_PHONE,
            )
            updated = store.remember_contact(
                given_name="Corinna",
                family_name="Maier",
                phone=_TEST_CORINNA_PHONE,
                role="Physiotherapist",
            )
            lookup = store.lookup_contact(name="Corinna", family_name="Maier")

        self.assertEqual(created.status, "created")
        self.assertEqual(updated.status, "updated")
        self.assertEqual(lookup.status, "found")
        self.assertEqual(lookup.match.label, "Corinna Maier")
        self.assertEqual(lookup.match.role, "Physiotherapist")
        self.assertEqual(lookup.match.phones, (_TEST_CORINNA_PHONE,))

    def test_goldset_context_has_stable_json_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._make_store(temp_dir)
            self._load_fixture(store)
            store.remember_preference(category="brand", value="Dallmayr", for_product="coffee")
            store.remember_preference(category="store", value="Bio Shop West", for_product="coffee")
            store.remember_plan(summary="buy coffee", when_text="today")

            payload = _memory_payload(store.build_prompt_context("Wo kann ich heute Kaffee kaufen?") or "")

        self.assertEqual(payload["schema"], "twinr_graph_memory_context_v1")
        self.assertGreaterEqual(len(payload["contacts"]), 2)
        self.assertGreaterEqual(len(payload["preferences"]), 3)
        self.assertGreaterEqual(len(payload["plans"]), 2)

    def test_goldset_rejects_unsupported_schema_version_on_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self._make_store(temp_dir)
            payload = self._load_fixture(store)
            payload["schema"]["version"] = 99
            store.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported Twinr graph schema version"):
                store.load_document()


if __name__ == "__main__":
    unittest.main()
