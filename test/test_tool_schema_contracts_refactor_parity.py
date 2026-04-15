from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Callable, cast
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.tools.schemas.contracts import (
    build_agent_tool_schemas,
    build_compact_agent_tool_schemas,
    build_realtime_tool_schemas,
)

SchemaDict = dict[str, Any]
SchemaBuilder = Callable[[Any], list[SchemaDict]]

build_agent_tool_schemas_impl: SchemaBuilder | None
build_compact_agent_tool_schemas_impl: SchemaBuilder | None
build_realtime_tool_schemas_impl: SchemaBuilder | None

try:
    from twinr.agent.tools.schemas.contracts_impl.main import (
        build_agent_tool_schemas as _build_agent_tool_schemas_impl,
        build_compact_agent_tool_schemas as _build_compact_agent_tool_schemas_impl,
        build_realtime_tool_schemas as _build_realtime_tool_schemas_impl,
    )
except ImportError:  # pragma: no cover - pre-refactor baseline
    build_agent_tool_schemas_impl = None
    build_compact_agent_tool_schemas_impl = None
    build_realtime_tool_schemas_impl = None
else:
    build_agent_tool_schemas_impl = _build_agent_tool_schemas_impl
    build_compact_agent_tool_schemas_impl = _build_compact_agent_tool_schemas_impl
    build_realtime_tool_schemas_impl = _build_realtime_tool_schemas_impl


ALL_TOOL_NAMES = (
    "print_receipt",
    "search_live_info",
    "browser_automation",
    "connect_service_integration",
    "schedule_reminder",
    "list_automations",
    "create_time_automation",
    "create_sensor_automation",
    "update_time_automation",
    "update_sensor_automation",
    "delete_automation",
    "list_smart_home_entities",
    "read_smart_home_state",
    "control_smart_home_entities",
    "read_smart_home_sensor_stream",
    "propose_skill_learning",
    "answer_skill_question",
    "confirm_skill_activation",
    "rollback_skill_activation",
    "pause_skill_activation",
    "reactivate_skill_activation",
    "remember_memory",
    "review_saved_memories",
    "remember_contact",
    "lookup_contact",
    "send_whatsapp_message",
    "get_memory_conflicts",
    "resolve_memory_conflict",
    "remember_preference",
    "remember_plan",
    "update_user_profile",
    "update_personality",
    "manage_user_discovery",
    "configure_world_intelligence",
    "update_simple_setting",
    "manage_voice_quiet_mode",
    "enroll_voice_profile",
    "get_voice_profile_status",
    "reset_voice_profile",
    "enroll_portrait_identity",
    "get_portrait_identity_status",
    "reset_portrait_identity",
    "manage_household_identity",
    "end_conversation",
    "inspect_camera",
)

_EXPECTED_GOLDEN_DIGESTS = {
    "canonical": "5b72418747e9685c47d0c483c0071cac9efd8c80e89efbcbd74b27256ba5a3a5",
    "compact": "27497a6e7963befbefebf80ee7c7174e7b36c6828d73a8091ac8bb141a685f34",
    "realtime": "dd458c9a248a38ee4821d4e5e477cc1d8e53eb837e86c866ff5a86979efa25d5",
}

_EXPECTED_TIME_RULES = [
    {
        "if": {"properties": {"schedule": {"const": "once"}}, "required": ["schedule"]},
        "then": {"required": ["due_at"]},
    },
    {
        "if": {"properties": {"schedule": {"const": "daily"}}, "required": ["schedule"]},
        "then": {"required": ["time_of_day"]},
    },
    {
        "if": {"properties": {"schedule": {"const": "weekly"}}, "required": ["schedule"]},
        "then": {"required": ["time_of_day", "weekdays"]},
    },
]


def _payload_digest(payload: object) -> str:
    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(rendered.encode("utf-8")).hexdigest()


class ToolSchemaContractsRefactorParityTests(unittest.TestCase):
    def _schema_by_name(self, *tool_names: str) -> dict[str, SchemaDict]:
        return {
            str(item["name"]): item
            for item in build_agent_tool_schemas(tool_names)
        }

    def test_public_wrapper_keeps_module_path(self) -> None:
        self.assertEqual(build_agent_tool_schemas.__module__, "twinr.agent.tools.schemas.contracts")
        self.assertEqual(build_compact_agent_tool_schemas.__module__, "twinr.agent.tools.schemas.contracts")
        self.assertEqual(build_realtime_tool_schemas.__module__, "twinr.agent.tools.schemas.contracts")

    def test_tool_name_normalization_and_order_are_stable(self) -> None:
        canonical = build_agent_tool_schemas(ALL_TOOL_NAMES)
        self.assertEqual(len(canonical), len(ALL_TOOL_NAMES))
        self.assertEqual([item["name"] for item in canonical], list(ALL_TOOL_NAMES))
        self.assertEqual(build_agent_tool_schemas(None), [])
        self.assertEqual(
            [item["name"] for item in build_agent_tool_schemas("browser_automation")],
            ["browser_automation"],
        )
        self.assertEqual(
            [item["name"] for item in build_agent_tool_schemas(b"browser_automation")],
            ["browser_automation"],
        )
        self.assertEqual(
            [
                item["name"]
                for item in build_agent_tool_schemas(
                    cast(Any, ("browser_automation", "browser_automation", "", None))
                )
            ],
            ["browser_automation"],
        )

    def test_full_schema_golden_hashes_remain_stable(self) -> None:
        payloads = {
            "canonical": build_agent_tool_schemas(ALL_TOOL_NAMES),
            "compact": build_compact_agent_tool_schemas(ALL_TOOL_NAMES),
            "realtime": build_realtime_tool_schemas(ALL_TOOL_NAMES),
        }
        for name, payload in payloads.items():
            with self.subTest(kind=name):
                self.assertEqual(_payload_digest(payload), _EXPECTED_GOLDEN_DIGESTS[name])

    def test_representative_tool_shapes_remain_stable(self) -> None:
        schemas = self._schema_by_name(
            "browser_automation",
            "create_time_automation",
            "list_smart_home_entities",
            "manage_user_discovery",
            "update_simple_setting",
            "manage_household_identity",
        )

        browser = schemas["browser_automation"]
        self.assertEqual(browser["parameters"]["required"], ["goal", "allowed_domains"])
        self.assertEqual(
            browser["parameters"]["properties"]["allowed_domains"]["items"],
            {
                "description": "One allowed host name for this browser run.",
                "minLength": 1,
                "type": "string",
            },
        )
        self.assertEqual(browser["parameters"]["properties"]["max_steps"]["maximum"], 32)
        self.assertIn(
            "A short follow-up assent to an already proposed deeper site check counts as explicit approval",
            browser["description"],
        )

        time_automation = schemas["create_time_automation"]
        self.assertEqual(time_automation["parameters"]["allOf"], _EXPECTED_TIME_RULES)
        self.assertEqual(
            time_automation["parameters"]["properties"]["time_of_day"]["pattern"],
            "^(?:[01]\\d|2[0-3]):[0-5]\\d$",
        )
        self.assertEqual(
            time_automation["parameters"]["properties"]["delivery"]["enum"],
            ["spoken", "printed"],
        )

        smart_home = schemas["list_smart_home_entities"]
        self.assertEqual(
            smart_home["parameters"]["properties"]["entity_class"]["enum"],
            [
                "alarm",
                "battery_sensor",
                "button",
                "device_health",
                "light",
                "light_group",
                "light_sensor",
                "motion_sensor",
                "scene",
                "switch",
                "temperature_sensor",
                "unknown",
            ],
        )

        self.assertEqual(
            smart_home["parameters"]["properties"]["aggregate_by"]["items"]["enum"],
            ["area", "controllable", "entity_class", "online", "provider", "readable"],
        )
        self.assertEqual(
            smart_home["parameters"]["properties"]["state_filters"]["items"]["properties"]["value"],
            {
                "anyOf": [
                    {"minLength": 1, "type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                ],
                "description": "Exact scalar value that the selected state key must equal.",
            },
        )

        discovery = schemas["manage_user_discovery"]
        self.assertEqual(
            discovery["parameters"]["properties"]["action"]["enum"],
            [
                "start_or_resume",
                "answer",
                "skip_topic",
                "pause_session",
                "snooze",
                "status",
                "review_profile",
                "replace_fact",
                "delete_fact",
            ],
        )
        self.assertEqual(
            discovery["parameters"]["properties"]["memory_routes"]["items"]["properties"]["route_kind"]["enum"],
            ["user_profile", "personality", "contact", "preference", "plan", "durable_memory"],
        )
        self.assertEqual(discovery["parameters"]["required"], ["action"])

        simple_setting = schemas["update_simple_setting"]
        self.assertEqual(len(simple_setting["parameters"]["allOf"]), 6)
        spoken_voice_rule = simple_setting["parameters"]["allOf"][-1]
        self.assertEqual(
            spoken_voice_rule,
            {
                "if": {"properties": {"setting": {"const": "spoken_voice"}}, "required": ["setting"]},
                "then": {
                    "properties": {
                        "action": {"const": "set"},
                        "value": {
                            "enum": ["alloy", "cedar", "coral", "echo", "marin", "sage"],
                            "minLength": 1,
                            "type": "string",
                        },
                    },
                    "required": ["value"],
                },
            },
        )
        self.assertEqual(
            simple_setting["parameters"]["properties"]["setting"]["enum"],
            [
                "follow_up_timeout_s",
                "memory_capacity",
                "speech_pause_ms",
                "speech_speed",
                "spoken_voice",
            ],
        )

        household_identity = schemas["manage_household_identity"]
        self.assertEqual(
            household_identity["parameters"]["properties"]["action"]["enum"],
            ["status", "enroll_face", "enroll_voice", "confirm_identity", "deny_identity"],
        )
        self.assertEqual(household_identity["parameters"]["required"], ["action"])

    def test_self_coding_scope_fields_use_json_string_transport(self) -> None:
        schemas = self._schema_by_name("propose_skill_learning", "answer_skill_question")

        propose_skill_learning = schemas["propose_skill_learning"]
        self.assertEqual(
            propose_skill_learning["parameters"]["properties"]["scope"]["type"],
            "string",
        )
        self.assertIn(
            "JSON object string",
            propose_skill_learning["parameters"]["properties"]["scope"]["description"],
        )

        answer_skill_question = schemas["answer_skill_question"]
        self.assertEqual(
            answer_skill_question["parameters"]["properties"]["scope"]["type"],
            "string",
        )
        self.assertIn(
            "JSON object string",
            answer_skill_question["parameters"]["properties"]["scope"]["description"],
        )

    def test_public_wrapper_matches_internal_implementation(self) -> None:
        if (
            build_agent_tool_schemas_impl is None
            or build_compact_agent_tool_schemas_impl is None
            or build_realtime_tool_schemas_impl is None
        ):
            self.skipTest("contracts_impl package is not available before the refactor lands")

        canonical_impl = cast(SchemaBuilder, build_agent_tool_schemas_impl)
        compact_impl = cast(SchemaBuilder, build_compact_agent_tool_schemas_impl)
        realtime_impl = cast(SchemaBuilder, build_realtime_tool_schemas_impl)

        self.assertEqual(
            build_agent_tool_schemas(ALL_TOOL_NAMES),
            canonical_impl(ALL_TOOL_NAMES),
        )
        self.assertEqual(
            build_compact_agent_tool_schemas(ALL_TOOL_NAMES),
            compact_impl(ALL_TOOL_NAMES),
        )
        self.assertEqual(
            build_realtime_tool_schemas(ALL_TOOL_NAMES),
            realtime_impl(ALL_TOOL_NAMES),
        )
