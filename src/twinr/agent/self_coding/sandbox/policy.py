"""Define explicit broker-policy manifests for sandboxed self-coding skills.

The manifest makes the allowed ``ctx.*`` surface explicit per capability so the
compiled artifact, runtime, and operator UI all agree on the same bounded
execution contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping  # AUDIT-FIX(#4): Scalar-aware validation prevents string char-splitting and bad container coercion.
from dataclasses import dataclass
from types import MappingProxyType  # AUDIT-FIX(#1): Freeze validated capability mappings so the manifest stays immutable after construction.
from typing import Any

_MANIFEST_SCHEMA = "twinr_self_coding_broker_policy_manifest_v1"
_ALLOWED_IDENTIFIER_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-")

_CAPABILITY_METHODS: dict[str, tuple[str, ...]] = {
    "calendar": ("list_calendar_events",),
    "camera": ("current_presence_session_id", "current_sensor_facts"),
    "email": ("list_recent_emails",),
    "llm_call": ("summarize_text",),
    "memory": ("delete_json", "list_json_keys", "load_json", "merge_json", "store_json"),
    "pir": ("current_presence_session_id", "current_sensor_facts"),
    "rules": ("log_event",),
    "safety": ("current_presence_session_id", "current_sensor_facts", "is_night_mode", "is_private_for_speech", "log_event"),
    "scheduler": ("now_iso", "today_local_date"),
    "speaker": ("say",),
    "web_search": ("search_web",),
}


def _normalize_identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):  # AUDIT-FIX(#5): Reject non-string identifiers instead of silently stringifying malformed inputs.
        raise TypeError(f"{field_name} must be a string")
    text = value.strip().lower()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    if any(char not in _ALLOWED_IDENTIFIER_CHARS for char in text):
        raise ValueError(f"{field_name} must be a stable identifier")
    return text


def _coerce_string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):  # AUDIT-FIX(#4): Treat scalar strings as single items instead of iterating character-by-character.
        return (value,)
    if isinstance(value, (bytes, bytearray)):
        raise TypeError(f"{field_name} must be text, not bytes")
    if isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a string or an iterable of strings")
    if not isinstance(value, Iterable):
        raise TypeError(f"{field_name} must be a string or an iterable of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):  # AUDIT-FIX(#5): Fail fast on mixed-type payload arrays to keep diagnostics precise.
            raise TypeError(f"{field_name} items must be strings")
        items.append(item)
    return tuple(items)


def _normalize_capability_name(value: object, *, field_name: str) -> str:
    normalized_name = _normalize_identifier(value, field_name=field_name)
    if normalized_name not in _CAPABILITY_METHODS:  # AUDIT-FIX(#2): Reject unknown capabilities instead of silently dropping or accepting them.
        raise ValueError(f"{field_name} contains unknown capability: {normalized_name}")
    return normalized_name


def _normalize_method_items(capability_name: str, raw_methods: object) -> tuple[str, ...]:
    method_items = tuple(
        item.strip()
        for item in _coerce_string_tuple(
            raw_methods,
            field_name=f"capability_methods[{capability_name}]",
        )
    )
    normalized_methods = tuple(sorted(item for item in _dedupe(method_items) if item))
    unsupported_methods = tuple(
        method_name
        for method_name in normalized_methods
        if method_name not in _CAPABILITY_METHODS[capability_name]
    )
    if unsupported_methods:  # AUDIT-FIX(#2): Enforce the broker whitelist so payloads cannot widen the ctx.* surface.
        raise ValueError(
            f"capability_methods[{capability_name}] contains unsupported methods: "
            f"{', '.join(unsupported_methods)}"
        )
    return normalized_methods


def _dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _normalize_capability_mapping(raw_mapping: Mapping[object, object]) -> dict[str, tuple[str, ...]]:
    normalized_mapping: dict[str, tuple[str, ...]] = {}
    for capability_name, raw_methods in raw_mapping.items():
        normalized_name = _normalize_capability_name(capability_name, field_name="capability_name")
        normalized_methods = _normalize_method_items(normalized_name, raw_methods)
        if normalized_name in normalized_mapping:  # AUDIT-FIX(#6): Merge colliding normalized keys instead of silently overwriting earlier entries.
            merged_methods = _dedupe(normalized_mapping[normalized_name] + normalized_methods)
            normalized_mapping[normalized_name] = tuple(sorted(item for item in merged_methods if item))
            continue
        normalized_mapping[normalized_name] = normalized_methods
    return normalized_mapping


@dataclass(frozen=True, slots=True)
class CapabilityBrokerManifest:
    """Describe the broker methods explicitly allowed for one compiled skill."""

    schema: str = _MANIFEST_SCHEMA
    required_capabilities: tuple[str, ...] = ()
    capability_methods: Mapping[str, tuple[str, ...]] | None = None  # AUDIT-FIX(#1): Store read-only mappings after validation.

    def __post_init__(self) -> None:
        if not isinstance(self.schema, str):
            raise TypeError("policy manifest schema must be a string")
        normalized_schema = self.schema.strip()
        if normalized_schema != _MANIFEST_SCHEMA:  # AUDIT-FIX(#3): Reject blank or foreign schema values instead of coercing them to the current schema.
            raise ValueError("policy manifest schema mismatch")
        object.__setattr__(self, "schema", normalized_schema)

        normalized_capabilities = tuple(
            _normalize_capability_name(value, field_name="required_capabilities")
            for value in _coerce_string_tuple(
                self.required_capabilities,
                field_name="required_capabilities",
            )
        )
        required_capabilities = _dedupe(normalized_capabilities)

        raw_mapping = self.capability_methods
        if raw_mapping is None:
            normalized_mapping = {  # AUDIT-FIX(#2): Derive the explicit broker surface from the declared capability set when no mapping was supplied.
                capability_name: _CAPABILITY_METHODS[capability_name]
                for capability_name in required_capabilities
            }
        else:
            if not isinstance(raw_mapping, Mapping):
                raise TypeError("policy manifest capability_methods must be an object")
            normalized_mapping = _normalize_capability_mapping(raw_mapping)
            if required_capabilities:
                missing_capabilities = tuple(
                    capability_name
                    for capability_name in required_capabilities
                    if capability_name not in normalized_mapping
                )
                unexpected_capabilities = tuple(
                    capability_name
                    for capability_name in normalized_mapping
                    if capability_name not in required_capabilities
                )
                if missing_capabilities or unexpected_capabilities:  # AUDIT-FIX(#2): Keep required_capabilities and capability_methods in lockstep.
                    details: list[str] = []
                    if missing_capabilities:
                        details.append(f"missing mappings for: {', '.join(missing_capabilities)}")
                    if unexpected_capabilities:
                        details.append(f"unexpected mappings for: {', '.join(unexpected_capabilities)}")
                    raise ValueError(f"policy manifest capability set mismatch ({'; '.join(details)})")
            else:
                required_capabilities = tuple(normalized_mapping)

        object.__setattr__(self, "required_capabilities", required_capabilities)
        ordered_mapping = {
            capability_name: normalized_mapping[capability_name]
            for capability_name in required_capabilities
        }
        object.__setattr__(
            self,
            "capability_methods",
            MappingProxyType(ordered_mapping),  # AUDIT-FIX(#1): Expose a read-only mapping so validated policies cannot be mutated later.
        )

    @property
    def allowed_methods(self) -> tuple[str, ...]:
        """Return the sorted union of all methods permitted by this manifest."""

        ordered: set[str] = set()
        for methods in self.capability_methods.values():
            ordered.update(methods)
        return tuple(sorted(ordered))

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload for persistence in skill artifacts."""

        return {
            "schema": self.schema,
            "required_capabilities": list(self.required_capabilities),
            "allowed_methods": list(self.allowed_methods),
            "capability_methods": {
                capability_name: list(self.capability_methods[capability_name])  # AUDIT-FIX(#1): Serialize from the frozen canonical mapping.
                for capability_name in self.required_capabilities
            },
        }

    @classmethod
    def from_payload(cls, payload: object) -> "CapabilityBrokerManifest":
        """Build one manifest from a persisted JSON payload."""

        if not isinstance(payload, dict):
            raise TypeError("policy manifest must be a JSON object")

        if "schema" in payload:
            schema = payload["schema"]
            if not isinstance(schema, str):
                raise TypeError("policy manifest schema must be a string")
        else:
            schema = _MANIFEST_SCHEMA  # AUDIT-FIX(#3): Only apply the default when the schema key is truly absent.

        raw_capability_methods = payload.get("capability_methods")
        if raw_capability_methods is not None and not isinstance(raw_capability_methods, Mapping):
            raise TypeError("policy manifest capability_methods must be an object")

        normalized_payload_mapping: dict[str, tuple[str, ...]] | None = None
        if raw_capability_methods is not None:
            normalized_payload_mapping = {}
            for key, value in raw_capability_methods.items():
                if not isinstance(key, str):  # AUDIT-FIX(#5): Reject non-string object keys from malformed payloads.
                    raise TypeError("policy manifest capability_methods keys must be strings")
                normalized_payload_mapping[key] = _coerce_string_tuple(
                    value,
                    field_name=f"capability_methods[{key}]",
                )

        return cls(
            schema=schema,
            required_capabilities=_coerce_string_tuple(  # AUDIT-FIX(#4): Preserve scalar strings as single capabilities during payload loading.
                payload.get("required_capabilities", ()),
                field_name="required_capabilities",
            ),
            capability_methods=normalized_payload_mapping,
        )


def build_capability_broker_manifest(capabilities: tuple[str, ...] | list[str] | str | None) -> CapabilityBrokerManifest:
    """Create the explicit broker manifest for one capability set."""

    normalized_capabilities = tuple(
        _normalize_capability_name(value, field_name="required_capabilities")
        for value in _coerce_string_tuple(  # AUDIT-FIX(#2): Fail on unknown capabilities instead of silently dropping them from the broker map.
            capabilities,
            field_name="required_capabilities",
        )
    )
    return CapabilityBrokerManifest(
        required_capabilities=_dedupe(normalized_capabilities),
    )


__all__ = [
    "CapabilityBrokerManifest",
    "build_capability_broker_manifest",
]