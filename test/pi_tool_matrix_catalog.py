"""Define slice groups and result merging for the live Pi tool matrix.

The Pi tool matrix already exercises the real spoken/runtime path with live
provider calls. Running the full matrix in one SSH invocation is expensive,
though, so this module provides two clean seams:

- stable named groups for bounded slice runs on the real Pi
- deterministic merging of multiple slice artifacts into one combined matrix

Keeping this metadata outside ``run_pi_tool_matrix.py`` lets the harness stay
focused on executing live turns instead of owning slice policy and result
aggregation as well.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from typing import cast

_DIMENSIONS = ("single_turn", "multi_turn", "persistence")
_STATUS_RANK = {"n/a": 0, "pass": 1, "fail": 2}

MATRIX_GROUP_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "core": (
        "print_receipt",
        "search_live_info",
        "inspect_camera",
        "end_conversation",
        "manage_voice_quiet_mode",
    ),
    "browser_channels": (
        "browser_automation",
        "connect_service_integration",
        "send_whatsapp_message",
    ),
    "reminder_automation": (
        "schedule_reminder",
        "create_time_automation",
        "list_automations",
        "update_time_automation",
        "delete_automation",
        "create_sensor_automation",
        "update_sensor_automation",
    ),
    "memory_profile": (
        "remember_memory",
        "remember_contact",
        "lookup_contact",
        "get_memory_conflicts",
        "resolve_memory_conflict",
        "remember_preference",
        "remember_plan",
        "update_user_profile",
        "update_personality",
        "update_simple_setting",
    ),
    "voice_profile": (
        "enroll_voice_profile",
        "get_voice_profile_status",
        "reset_voice_profile",
    ),
    "discovery_world": (
        "manage_user_discovery",
        "configure_world_intelligence",
    ),
    "local_identity": (
        "enroll_portrait_identity",
        "get_portrait_identity_status",
        "reset_portrait_identity",
        "manage_household_identity",
    ),
    "smart_home": (
        "list_smart_home_entities",
        "read_smart_home_state",
        "control_smart_home_entities",
        "read_smart_home_sensor_stream",
    ),
    "self_coding": (
        "propose_skill_learning",
        "answer_skill_question",
        "confirm_skill_activation",
        "pause_skill_activation",
        "reactivate_skill_activation",
        "rollback_skill_activation",
    ),
}


def available_matrix_groups() -> tuple[str, ...]:
    """Return the stable ordered list of supported matrix slice groups."""

    return tuple(MATRIX_GROUP_TOOL_NAMES)


def normalize_matrix_groups(groups: Iterable[str] | None) -> tuple[str, ...]:
    """Return ordered unique matrix groups or all groups when omitted.

    Args:
        groups: Optional iterable of group names from CLI or tests.

    Returns:
        Ordered, duplicate-free group names.

    Raises:
        ValueError: If one requested group name is unknown.
    """

    if groups is None:
        return available_matrix_groups()
    selected: list[str] = []
    seen: set[str] = set()
    for raw_group in groups:
        group = str(raw_group).strip()
        if not group:
            continue
        if group not in MATRIX_GROUP_TOOL_NAMES:
            raise ValueError(f"unknown matrix group: {group}")
        if group in seen:
            continue
        seen.add(group)
        selected.append(group)
    return tuple(selected)


def merge_tool_matrix_results(results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Merge multiple slice artifacts into one combined Pi matrix result.

    Args:
        results: Parsed JSON payloads from ``run_pi_tool_matrix.py`` slice runs.

    Returns:
        One deterministic combined result with merged per-tool dimensions.

    Raises:
        ValueError: If no results were supplied.
    """

    if not results:
        raise ValueError("at least one tool-matrix result is required")

    first = results[0]
    tool_names = _coerce_str_sequence(first.get("tool_names", ()))
    merged_tools: dict[str, dict[str, object]] = {}
    for tool_name in tool_names:
        merged_entry: dict[str, object] = {}
        for dimension in _DIMENSIONS:
            payloads = [
                _coerce_dimension_payload(result.get("tools", {}), tool_name, dimension)
                for result in results
            ]
            merged_entry[dimension] = _merge_dimension_payloads(payloads)
        merged_entry["overall"] = _overall_status(merged_entry)
        merged_tools[tool_name] = merged_entry

    passed_tools = sorted(name for name, entry in merged_tools.items() if entry["overall"] == "pass")
    failed_tools = sorted(name for name, entry in merged_tools.items() if entry["overall"] == "fail")
    return {
        "merged": True,
        "merged_result_count": len(results),
        "merged_selected_groups": sorted(
            {
                group
                for result in results
                for group in _coerce_str_sequence(result.get("selected_groups", ()))
            }
        ),
        "base_env_paths": sorted(
            {
                str(result.get("base_env_path", "")).strip()
                for result in results
                if str(result.get("base_env_path", "")).strip()
            }
        ),
        "planned_stack": deepcopy(first.get("planned_stack", {})),
        "tool_names": list(tool_names),
        "tool_count": len(tool_names),
        "tools": merged_tools,
        "summary": {
            "passed_tool_count": len(passed_tools),
            "failed_tool_count": len(failed_tools),
            "passed_tools": passed_tools,
            "failed_tools": failed_tools,
        },
        "source_summaries": [
            {
                "selected_groups": list(_coerce_str_sequence(result.get("selected_groups", ()))),
                "summary": deepcopy(result.get("summary", {})),
                "artifacts": deepcopy(result.get("artifacts", {})),
            }
            for result in results
        ],
        "scenarios": [
            deepcopy(scenario)
            for result in results
            for scenario in _coerce_object_sequence(result.get("scenarios", ()))
        ],
        "printer_outputs": [
            output
            for result in results
            for output in _coerce_str_sequence(result.get("printer_outputs", ()))
        ],
    }


def _coerce_dimension_payload(
    tools_payload: object,
    tool_name: str,
    dimension: str,
) -> Mapping[str, object]:
    """Return one dimension payload or a synthetic ``n/a`` payload."""

    if not isinstance(tools_payload, Mapping):
        return {"status": "n/a", "detail": ""}
    tool_payload = tools_payload.get(tool_name)
    if not isinstance(tool_payload, Mapping):
        return {"status": "n/a", "detail": ""}
    dimension_payload = tool_payload.get(dimension)
    if not isinstance(dimension_payload, Mapping):
        return {"status": "n/a", "detail": ""}
    return dimension_payload


def _merge_dimension_payloads(payloads: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Pick the strongest status and accumulate unique detail fragments."""

    winning: dict[str, object] = {"status": "n/a", "detail": ""}
    fragments: list[str] = []
    for payload in payloads:
        status = str(payload.get("status", "n/a")).strip().lower() or "n/a"
        detail = str(payload.get("detail", "")).strip()
        if detail and detail not in fragments:
            fragments.append(detail)
        if _STATUS_RANK.get(status, 0) > _STATUS_RANK.get(str(winning["status"]), 0):
            winning["status"] = status
    winning["detail"] = " | ".join(fragments)
    return winning


def _coerce_str_sequence(value: object) -> tuple[str, ...]:
    """Return a tuple of strings from one loose JSON value."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(str(item) for item in value)


def _coerce_object_sequence(value: object) -> tuple[object, ...]:
    """Return one generic object tuple from a loose JSON value."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(value)


def _overall_status(entry: Mapping[str, object]) -> str:
    """Return the merged overall status for one tool entry."""

    statuses = [str(cast(Mapping[str, object], entry[dimension])["status"]) for dimension in _DIMENSIONS]
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "pass" for status in statuses):
        return "pass"
    return "n/a"
