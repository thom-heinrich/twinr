"""Regression tests for long-term reflection behavior."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermSourceRefV1
from twinr.memory.longterm.reasoning.reflect import LongTermMemoryReflector


def _source() -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=("turn:test",),
        speaker="user",
        modality="voice",
    )


class LongTermMemoryReflectorTests(unittest.TestCase):
    def test_reflector_can_promote_repeated_uncertain_fact(self) -> None:
        reflector = LongTermMemoryReflector()
        item = LongTermMemoryObjectV1(
            memory_id="fact:corinna_role",
            kind="relationship_fact",
            summary="Corinna Maier is the user's physiotherapist.",
            source=_source(),
            status="uncertain",
            confidence=0.62,
            slot_key="relationship:user:main:physiotherapist",
            value_key="person:corinna_maier",
            attributes={
                "person_ref": "person:corinna_maier",
                "person_name": "Corinna Maier",
                "relation": "physiotherapist",
                "support_count": 2,
            },
        )

        result = reflector.reflect(objects=(item,))

        self.assertEqual(len(result.reflected_objects), 1)
        self.assertEqual(result.reflected_objects[0].status, "active")
        self.assertGreaterEqual(result.reflected_objects[0].confidence, 0.79)

    def test_reflector_can_create_thread_summary_for_person(self) -> None:
        reflector = LongTermMemoryReflector()
        relationship = LongTermMemoryObjectV1(
            memory_id="fact:janina_wife",
            kind="fact",
            summary="Janina is the user's wife.",
            source=_source(),
            status="active",
            confidence=0.98,
            slot_key="relationship:user:main:wife",
            value_key="person:janina",
            attributes={
                "person_ref": "person:janina",
                "person_name": "Janina",
                "relation": "wife",
                "fact_type": "relationship",
                "support_count": 2,
            },
        )
        appointment_event = LongTermMemoryObjectV1(
            memory_id="event:janina_eye_laser",
            kind="event",
            summary="Janina has eye laser treatment at the eye doctor on 2026-03-14.",
            source=_source(),
            status="active",
            confidence=0.92,
            slot_key="event:person:janina:eye_laser_treatment:2026-03-14",
            value_key="event:janina_eye_laser_2026_03_14",
            valid_from="2026-03-14",
            valid_to="2026-03-14",
            sensitivity="sensitive",
            attributes={
                "person_ref": "person:janina",
                "person_name": "Janina",
                "memory_domain": "appointment",
                "event_domain": "appointment",
                "action": "eye laser treatment",
                "place": "the eye doctor",
                "support_count": 1,
            },
        )

        result = reflector.reflect(objects=(relationship, appointment_event))

        self.assertEqual(len(result.created_summaries), 1)
        summary = result.created_summaries[0]
        self.assertEqual(summary.kind, "summary")
        self.assertEqual((summary.attributes or {}).get("summary_type"), "thread")
        self.assertIn("Ongoing thread about Janina", summary.summary)
        self.assertIn("eye laser treatment", summary.summary)
        self.assertEqual(summary.sensitivity, "sensitive")

    def test_reflector_can_create_thread_summary_from_canonical_proposition_fields(self) -> None:
        reflector = LongTermMemoryReflector()
        relationship = LongTermMemoryObjectV1(
            memory_id="fact:janina_wife_canonical",
            kind="fact",
            summary="Janina is the user's wife.",
            source=_source(),
            status="active",
            confidence=0.97,
            slot_key="fact:person:janina:is_wife_of",
            value_key="user:main",
            attributes={
                "subject_ref": "person:janina",
                "object_ref": "user:main",
                "predicate": "is wife of",
                "support_count": 2,
            },
        )
        appointment_event = LongTermMemoryObjectV1(
            memory_id="event:janina_eye_laser_canonical",
            kind="event",
            summary="Janina has eye laser treatment at the eye doctor on 2026-03-14.",
            source=_source(),
            status="active",
            confidence=0.93,
            slot_key="event:person:janina:has_eye_laser_treatment:2026-03-14",
            value_key="place:eye_doctor",
            valid_from="2026-03-14",
            valid_to="2026-03-14",
            sensitivity="sensitive",
            attributes={
                "subject_ref": "person:janina",
                "object_ref": "place:eye_doctor",
                "predicate": "has eye laser treatment",
                "support_count": 1,
            },
        )

        result = reflector.reflect(objects=(relationship, appointment_event))

        self.assertEqual(len(result.created_summaries), 1)
        summary = result.created_summaries[0]
        self.assertEqual((summary.attributes or {}).get("person_ref"), "person:janina")
        self.assertEqual((summary.attributes or {}).get("summary_type"), "thread")
        self.assertIn("Ongoing thread about Janina", summary.summary)
        self.assertIn("wife", summary.summary)
        self.assertIn("eye laser treatment", summary.summary)
        self.assertEqual(summary.sensitivity, "sensitive")

    def test_reflector_builds_environment_reflection_summary_and_midterm_packet(self) -> None:
        reflector = LongTermMemoryReflector()
        profile = LongTermMemoryObjectV1(
            memory_id="environment_profile:home_main:day:2026-03-16",
            kind="summary",
            summary="Room-agnostic smart-home environment profile compiled for one day.",
            details="Daily motion-derived smart-home markers for longitudinal routine learning and deviation detection.",
            source=_source(),
            status="active",
            confidence=0.78,
            slot_key="environment_profile:home:main:2026-03-16",
            value_key="environment_day_profile",
            valid_from="2026-03-16",
            valid_to="2026-03-16",
            attributes={
                "memory_domain": "smart_home_environment",
                "summary_type": "environment_day_profile",
                "environment_id": "home:main",
                "date": "2026-03-16",
                "weekday_class": "weekday",
                "markers": {
                    "active_epoch_count_day": 2,
                    "first_activity_minute_local": 1080,
                    "last_activity_minute_local": 1140,
                    "unique_active_node_count_day": 1,
                    "sensor_coverage_ratio_day": 0.5,
                },
                "quality_flags": ["device_offline_present"],
            },
        )
        baseline = LongTermMemoryObjectV1(
            memory_id="environment_baseline:home_main:all_days:rolling_7d",
            kind="pattern",
            summary="Rolling smart-home environment baseline for all days.",
            details="Robust baseline built from prior daily room-agnostic motion markers.",
            source=_source(),
            status="active",
            confidence=0.84,
            slot_key="environment_baseline:home:main:all_days",
            value_key="environment_baseline",
            valid_from="2026-03-10",
            valid_to="2026-03-16",
            attributes={
                "memory_domain": "smart_home_environment",
                "pattern_type": "environment_baseline",
                "environment_id": "home:main",
                "weekday_class": "all_days",
                "window_days": 7,
                "sample_count": 6,
                "marker_stats": {
                    "active_epoch_count_day": {"median": 6.0, "iqr": 1.0, "ewma": 5.4},
                    "unique_active_node_count_day": {"median": 2.0, "iqr": 0.5, "ewma": 2.0},
                },
            },
        )
        deviation = LongTermMemoryObjectV1(
            memory_id="environment_deviation:home_main:daily_activity_drop:2026-03-16",
            kind="summary",
            summary="Smart-home environment deviation detected: less activity than usual.",
            details="Observed less activity than usual compared with the rolling room-agnostic smart-home baseline.",
            source=_source(),
            status="candidate",
            confidence=0.78,
            slot_key="environment_deviation:home:main:daily_activity_drop:2026-03-16",
            value_key="daily_activity_drop",
            valid_from="2026-03-16",
            valid_to="2026-03-16",
            attributes={
                "memory_domain": "smart_home_environment",
                "summary_type": "environment_deviation",
                "environment_id": "home:main",
                "observed_at": "2026-03-16T12:00:00+00:00",
                "deviation_type": "daily_activity_drop",
                "severity": "high",
                "time_scale": "day",
                "markers": [
                    {
                        "name": "active_epoch_count_day",
                        "observed": 2.0,
                        "baseline_median": 6.0,
                        "delta_ratio": -0.6667,
                    }
                ],
                "quality_flags": ["device_offline_present"],
                "blocked_by": ["sensor_quality_limited"],
                "explanation": {
                    "short_label": "less activity than usual",
                    "human_readable": "Observed less activity than usual compared with the rolling baseline.",
                },
            },
        )
        node = LongTermMemoryObjectV1(
            memory_id="environment_node:home_main:motion_node_1",
            kind="summary",
            summary="Environment node motion_node_1 was active in the smart-home profile window.",
            details="Room-agnostic smart-home node summary compiled from motion and health history.",
            source=_source(),
            status="active",
            confidence=0.72,
            slot_key="environment_node:home:main:motion-node-1",
            value_key="environment_node_summary",
            valid_from="2026-03-10",
            valid_to="2026-03-16",
            attributes={
                "memory_domain": "smart_home_environment",
                "summary_type": "environment_node",
                "environment_id": "home:main",
                "node_id": "motion-node-1",
                "provider_label": "Esszimmer 1",
                "provider_area_label": "Erdgeschoss",
                "motion_event_count": 12,
                "active_day_count": 6,
                "last_health_state": "offline",
            },
        )

        result = reflector.reflect(objects=(profile, baseline, deviation, node))

        environment_summaries = [
            item
            for item in result.created_summaries
            if (item.attributes or {}).get("summary_type") == "environment_reflection"
        ]
        self.assertEqual(len(environment_summaries), 1)
        summary = environment_summaries[0]
        self.assertIn("less activity than usual", summary.summary)
        self.assertEqual((summary.attributes or {}).get("memory_domain"), "smart_home_environment")
        self.assertEqual((summary.attributes or {}).get("profile_day"), "2026-03-16")
        self.assertIn("sensor_quality_limited", (summary.attributes or {}).get("blocked_by", ()))

        self.assertEqual(len(result.midterm_packets), 1)
        packet = result.midterm_packets[0]
        self.assertEqual(packet.kind, "recent_environment_pattern")
        self.assertIn("less activity than usual", packet.summary)
        self.assertIn("home activity", packet.query_hints)
        self.assertEqual(packet.valid_from, "2026-03-16")


if __name__ == "__main__":
    unittest.main()
