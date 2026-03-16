from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from functools import lru_cache
from hashlib import sha256
import json
import math
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.memory.longterm.core.models import (
    LongTermMemoryObjectV1,
    LongTermMultimodalEvidence,
    LongTermSourceRefV1,
    LongTermTurnExtractionV1,
)
from twinr.text_utils import collapse_whitespace, slugify_identifier


_DEFAULT_TIMEZONE_NAME = "Europe/Berlin"
_MAX_DETAIL_STRING_LENGTH = 512
_MAX_DETAIL_ITEMS = 32
_MAX_KEY_COMPONENT_LENGTH = 64
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]+")
_TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_VALUES = frozenset({"0", "false", "f", "no", "n", "off", ""})
_SENSITIVE_KEY_PARTS = (
    "token",
    "secret",
    "password",
    "passwd",
    "authorization",
    "cookie",
    "credential",
    "session",
    "api_key",
    "apikey",
    "private_key",
    "access_token",
    "refresh_token",
)


def _to_text(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _normalize_text(value: object | None, *, limit: int | None = None) -> str:
    raw_text = _CONTROL_CHAR_RE.sub(" ", _to_text(value))
    text = collapse_whitespace(raw_text)
    if limit is None or len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _slugify(value: str, *, fallback: str) -> str:
    return slugify_identifier(value, fallback=fallback)


# AUDIT-FIX(#5): Separate human-readable labels from bounded, storage-safe key components.
def _safe_key_component(
    value: object | None,
    *,
    fallback: str,
    limit: int = _MAX_KEY_COMPONENT_LENGTH,
) -> str:
    text = _normalize_text(value, limit=limit).lower()
    if not text:
        return fallback
    slug = _slugify(text, fallback=fallback).replace("-", "_")
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        return fallback
    return slug[:limit].strip("_") or fallback


# AUDIT-FIX(#3): Coerce heterogeneous payloads into mappings without throwing on malformed evidence.
def _mapping_or_empty(value: object | None) -> dict[str, object]:
    if value is None:
        return {}

    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            dumped = None
        if isinstance(dumped, Mapping):
            return {str(key): item for key, item in dumped.items()}

    legacy_dict = getattr(value, "dict", None)
    if callable(legacy_dict):
        try:
            dumped = legacy_dict()
        except Exception:
            dumped = None
        if isinstance(dumped, Mapping):
            return {str(key): item for key, item in dumped.items()}

    try:
        coerced = dict(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}
    return {str(key): item for key, item in coerced.items()}


# AUDIT-FIX(#7): Treat scalar strings as a single event name instead of exploding them into characters.
def _string_tuple(
    value: object | None,
    *,
    item_limit: int = 8,
    text_limit: int = 64,
) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, (str, bytes, bytearray)):
        items = [value]
    elif isinstance(value, (set, frozenset)):
        items = list(value)
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        items = [value]

    normalized: list[str] = []
    for item in items[:item_limit]:
        text = _normalize_text(item, limit=text_limit)
        if text:
            normalized.append(text)
    return tuple(normalized)


# AUDIT-FIX(#4): Parse sensor booleans explicitly so values like "false" or "0" stay false.
def _coerce_bool(value: object | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return False
        return value != 0

    text = _normalize_text(value, limit=16).lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return False


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


# AUDIT-FIX(#3): Convert arbitrary nested payloads into JSON-safe values with cycle protection.
# AUDIT-FIX(#8): Redact obvious secrets and cap nested payload size before persisting to long-term memory.
def _json_safe(
    value: object,
    *,
    max_depth: int = 4,
    max_items: int = _MAX_DETAIL_ITEMS,
    max_string_length: int = _MAX_DETAIL_STRING_LENGTH,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> object:
    if _seen is None:
        _seen = set()

    if value is None or isinstance(value, (bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else _normalize_text(str(value), limit=32)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, tzinfo):
        return str(value)

    if isinstance(value, (str, bytes, bytearray)):
        return _normalize_text(value, limit=max_string_length)

    if _depth >= max_depth:
        return "<truncated>"

    object_id = id(value)
    if isinstance(value, Mapping):
        if object_id in _seen:
            return "<circular>"
        _seen.add(object_id)
        items: dict[str, object] = {}
        try:
            for index, (key, item) in enumerate(value.items()):
                if index >= max_items:
                    items["__truncated_items__"] = f"{len(value) - max_items} more"
                    break
                key_text = _normalize_text(key, limit=80) or f"key_{index}"
                if _is_sensitive_key(key_text):
                    items[key_text] = "<redacted>"
                else:
                    items[key_text] = _json_safe(
                        item,
                        max_depth=max_depth,
                        max_items=max_items,
                        max_string_length=max_string_length,
                        _depth=_depth + 1,
                        _seen=_seen,
                    )
        finally:
            _seen.remove(object_id)
        return items

    if isinstance(value, (list, tuple, set, frozenset)):
        if object_id in _seen:
            return "<circular>"
        _seen.add(object_id)
        try:
            items_list: list[object] = []
            for index, item in enumerate(value):
                if index >= max_items:
                    items_list.append("<truncated>")
                    break
                items_list.append(
                    _json_safe(
                        item,
                        max_depth=max_depth,
                        max_items=max_items,
                        max_string_length=max_string_length,
                        _depth=_depth + 1,
                        _seen=_seen,
                    )
                )
        finally:
            _seen.remove(object_id)
        return items_list

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            dumped = None
        if dumped is not None:
            return _json_safe(
                dumped,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
                _depth=_depth + 1,
                _seen=_seen,
            )

    legacy_dict = getattr(value, "dict", None)
    if callable(legacy_dict):
        try:
            dumped = legacy_dict()
        except Exception:
            dumped = None
        if dumped is not None:
            return _json_safe(
                dumped,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
                _depth=_depth + 1,
                _seen=_seen,
            )

    return _normalize_text(value, limit=max_string_length)


def _safe_json(value: object) -> str:
    try:
        return json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        fallback_payload = {"_serialization_error": "unserializable_payload"}
        return json.dumps(fallback_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _payload_for_details(value: object | None) -> object:
    payload = _mapping_or_empty(value)
    if payload:
        return payload
    if value is None:
        return {}
    return {"_raw": value}


# AUDIT-FIX(#1): Resolve configured time zones defensively and fall back deterministically on bad config or missing tzdata.
@lru_cache(maxsize=32)
def _resolve_timezone(name: str) -> tzinfo:
    zone_name = _normalize_text(name, limit=128) or _DEFAULT_TIMEZONE_NAME
    try:
        return ZoneInfo(zone_name)
    except (ZoneInfoNotFoundError, ValueError):
        if zone_name != _DEFAULT_TIMEZONE_NAME:
            try:
                return ZoneInfo(_DEFAULT_TIMEZONE_NAME)
            except (ZoneInfoNotFoundError, ValueError):
                return timezone.utc
        return timezone.utc


# AUDIT-FIX(#1): Naive device timestamps are treated as local wall-clock times in the configured timezone, not as host-local time.
def _localize_datetime(value: datetime, *, target_tz: tzinfo) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=target_tz)
    return value.astimezone(target_tz)


# AUDIT-FIX(#2): Add a stable payload fingerprint so same-second events do not collide in file-backed stores.
def _event_fingerprint(
    evidence: LongTermMultimodalEvidence,
    *,
    occurred_at: datetime,
) -> str:
    fingerprint_payload = {
        "created_at": occurred_at.isoformat(timespec="microseconds"),
        "event_name": _normalize_text(evidence.event_name, limit=128),
        "source": _normalize_text(evidence.source, limit=64),
        "modality": _normalize_text(evidence.modality, limit=64),
        "message": _normalize_text(evidence.message, limit=256),
        "data": _payload_for_details(evidence.data),
    }
    return sha256(_safe_json(fingerprint_payload).encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class LongTermMultimodalExtractor:
    timezone_name: str = _DEFAULT_TIMEZONE_NAME

    def extract_evidence(
        self,
        evidence: LongTermMultimodalEvidence,
    ) -> LongTermTurnExtractionV1:
        # AUDIT-FIX(#1): Resolve and apply timezone explicitly to avoid host-local drift and ZoneInfo crashes.
        target_tz = _resolve_timezone(self.timezone_name)
        occurred_at = _localize_datetime(evidence.created_at, target_tz=target_tz)
        date_key = occurred_at.date().isoformat()
        daypart = self._daypart(occurred_at)
        event_name_display = _normalize_text(evidence.event_name, limit=128) or "event"
        event_name_key = _safe_key_component(event_name_display, fallback="event")
        # AUDIT-FIX(#2): Use microseconds plus a stable fingerprint to make event IDs collision-resistant.
        event_id = (
            f"multimodal:{occurred_at.strftime('%Y%m%dT%H%M%S%f%z')}:"
            f"{event_name_key}:{_event_fingerprint(evidence, occurred_at=occurred_at)}"
        )
        source_ref = LongTermSourceRefV1(
            source_type=evidence.source,
            event_ids=(event_id,),
            modality=evidence.modality,
        )

        candidates: list[LongTermMemoryObjectV1] = []
        candidate_extraction_degraded = False
        try:
            if event_name_key == "sensor_observation":
                candidates.extend(
                    self._extract_sensor_observation(
                        evidence=evidence,
                        source_ref=source_ref,
                        date_key=date_key,
                        daypart=daypart,
                    )
                )
            elif event_name_key == "button_interaction":
                item = self._extract_button_interaction(
                    evidence=evidence,
                    source_ref=source_ref,
                    date_key=date_key,
                    daypart=daypart,
                )
                if item is not None:
                    candidates.append(item)
            elif event_name_key == "print_completed":
                item = self._extract_print_completed(
                    evidence=evidence,
                    source_ref=source_ref,
                    date_key=date_key,
                    daypart=daypart,
                )
                if item is not None:
                    candidates.append(item)
            elif event_name_key == "camera_capture":
                item = self._extract_camera_capture(
                    evidence=evidence,
                    source_ref=source_ref,
                    date_key=date_key,
                    daypart=daypart,
                )
                if item is not None:
                    candidates.append(item)
        except Exception:
            # AUDIT-FIX(#6): Preserve the episode even when auxiliary candidate extraction fails on malformed payloads.
            candidates = []
            candidate_extraction_degraded = True

        episode_attributes = {
            "event_name": event_name_display,
            "event_name_key": event_name_key,
            "modality": evidence.modality,
            "daypart": daypart,
        }
        if candidate_extraction_degraded:
            # AUDIT-FIX(#6): Mark degraded extraction explicitly instead of failing the entire turn.
            episode_attributes["candidate_extraction"] = "degraded"

        episode = LongTermMemoryObjectV1(
            memory_id=f"episode:{_slugify(event_id, fallback='multimodal')}",
            kind="episode",
            summary=f"Multimodal device event recorded: {event_name_display}.",
            details=self._episode_details(evidence),
            source=source_ref,
            status="candidate",
            confidence=0.92,
            sensitivity="normal",
            slot_key=f"episode:{event_id}",
            value_key=event_name_key,
            valid_from=date_key,
            valid_to=date_key,
            attributes=episode_attributes,
        )

        return LongTermTurnExtractionV1(
            turn_id=event_id,
            occurred_at=occurred_at,
            episode=episode,
            candidate_objects=tuple(candidates),
            graph_edges=(),
        )

    def _episode_details(self, evidence: LongTermMultimodalEvidence) -> str:
        # AUDIT-FIX(#8): Persist a bounded/redacted payload snapshot rather than raw multimodal internals.
        payload = {
            "event_name": _normalize_text(evidence.event_name, limit=128),
            "modality": _normalize_text(evidence.modality, limit=64),
            "message": _normalize_text(evidence.message, limit=_MAX_DETAIL_STRING_LENGTH),
            "data": _payload_for_details(evidence.data),
        }
        return f"Structured multimodal evidence: {_safe_json(payload)}"

    def _extract_sensor_observation(
        self,
        *,
        evidence: LongTermMultimodalEvidence,
        source_ref: LongTermSourceRefV1,
        date_key: str,
        daypart: str,
    ) -> list[LongTermMemoryObjectV1]:
        # AUDIT-FIX(#3): Parse payloads through safe coercion so malformed evidence does not raise here.
        payload = _mapping_or_empty(evidence.data)
        facts = _mapping_or_empty(payload.get("facts"))
        fact_map = facts or payload
        pir = _mapping_or_empty(fact_map.get("pir"))
        camera = _mapping_or_empty(fact_map.get("camera"))
        event_names_value = payload.get("event_names")
        if event_names_value is None:
            event_names_value = fact_map.get("event_names")
        event_names = _string_tuple(event_names_value)
        objects: list[LongTermMemoryObjectV1] = []
        person_visible = _coerce_bool(camera.get("person_visible"))
        motion_detected = _coerce_bool(pir.get("motion_detected"))
        object_near_camera = _coerce_bool(camera.get("hand_or_object_near_camera"))
        # AUDIT-FIX(#9): Event-derived observations/patterns are day-scoped and should expire on the observation date.
        if motion_detected:
            objects.append(
                LongTermMemoryObjectV1(
                    memory_id=f"observation:{date_key}:pir_motion",
                    kind="observation",
                    summary="Motion was detected near the device.",
                    details="Derived from PIR sensor activity.",
                    source=source_ref,
                    confidence=0.62,
                    sensitivity="low",
                    slot_key=f"observation:pir_motion:{date_key}",
                    value_key="motion_detected",
                    valid_from=date_key,
                    valid_to=date_key,
                    attributes={"topic": "pir_motion", "event_names": event_names, "observation_type": "situational"},
                )
            )
        if person_visible:
            objects.append(
                LongTermMemoryObjectV1(
                    memory_id=f"observation:{date_key}:camera_person_visible",
                    kind="observation",
                    summary="A person was visible near the device.",
                    details="Derived from camera-based presence observation.",
                    source=source_ref,
                    confidence=0.64,
                    sensitivity="low",
                    slot_key=f"observation:camera_person_visible:{date_key}",
                    value_key="person_visible",
                    valid_from=date_key,
                    valid_to=date_key,
                    attributes={"topic": "camera_presence", "event_names": event_names, "observation_type": "situational"},
                )
            )
        if object_near_camera:
            objects.append(
                LongTermMemoryObjectV1(
                    memory_id=f"observation:{date_key}:camera_object_near",
                    kind="observation",
                    summary="An object or hand was near the camera.",
                    details="Derived from camera interaction observation.",
                    source=source_ref,
                    confidence=0.64,
                    sensitivity="low",
                    slot_key=f"observation:camera_object_near:{date_key}",
                    value_key="object_near_camera",
                    valid_from=date_key,
                    valid_to=date_key,
                    attributes={"topic": "camera_interaction", "event_names": event_names, "observation_type": "situational"},
                )
            )
        if motion_detected or person_visible:
            objects.append(
                LongTermMemoryObjectV1(
                    memory_id=f"pattern:presence:{daypart}:near_device",
                    kind="pattern",
                    summary=f"Presence near the device was observed in the {daypart}.",
                    details="Low-confidence multimodal pattern derived from PIR/camera evidence.",
                    source=source_ref,
                    confidence=0.58,
                    sensitivity="low",
                    slot_key=f"pattern:presence:{daypart}:near_device",
                    value_key="presence_observed",
                    valid_from=date_key,
                    valid_to=date_key,
                    attributes={
                        "daypart": daypart,
                        "uses_pir": motion_detected,
                        "uses_camera": person_visible,
                        "pattern_type": "presence",
                    },
                )
            )
        if object_near_camera:
            objects.append(
                LongTermMemoryObjectV1(
                    memory_id=f"pattern:camera_interaction:{daypart}",
                    kind="pattern",
                    summary=f"Camera-side interaction was observed in the {daypart}.",
                    details="Low-confidence multimodal pattern derived from repeated camera-side interaction signals.",
                    source=source_ref,
                    confidence=0.57,
                    sensitivity="low",
                    slot_key=f"pattern:camera_interaction:{daypart}",
                    value_key="camera_interaction_observed",
                    valid_from=date_key,
                    valid_to=date_key,
                    attributes={"daypart": daypart, "event_names": event_names, "pattern_type": "interaction"},
                )
            )
        return objects

    def _extract_button_interaction(
        self,
        *,
        evidence: LongTermMultimodalEvidence,
        source_ref: LongTermSourceRefV1,
        date_key: str,
        daypart: str,
    ) -> LongTermMemoryObjectV1 | None:
        # AUDIT-FIX(#3): Parse payloads through safe coercion so malformed evidence does not raise here.
        payload = _mapping_or_empty(evidence.data)
        button_display = _normalize_text(payload.get("button"), limit=48).lower()
        action_display = _normalize_text(payload.get("action"), limit=48).lower()
        if not button_display or not action_display:
            return None
        button = _safe_key_component(button_display, fallback="unknown_button")
        action = _safe_key_component(action_display, fallback="unknown_action")
        action_label = {
            "start_listening": "start a conversation",
            "print_request": "request a printed answer",
        }.get(action, action_display.replace("_", " "))
        # AUDIT-FIX(#9): Event-derived interaction patterns are day-scoped and should expire on the observation date.
        return LongTermMemoryObjectV1(
            memory_id=f"pattern:button:{button}:{action}:{daypart}",
            kind="pattern",
            summary=f"The {button_display} button was used to {action_label} in the {daypart}.",
            details="Low-confidence button usage pattern derived from a physical interaction event.",
            source=source_ref,
            confidence=0.6,
            sensitivity="low",
            slot_key=f"pattern:button:{button}:{action}:{daypart}",
            value_key="button_used",
            valid_from=date_key,
            valid_to=date_key,
            attributes={
                "button": button,
                "button_label": button_display,
                "action": action,
                "action_label": action_display,
                "daypart": daypart,
                "pattern_type": "interaction",
            },
        )

    def _extract_print_completed(
        self,
        *,
        evidence: LongTermMultimodalEvidence,
        source_ref: LongTermSourceRefV1,
        date_key: str,
        daypart: str,
    ) -> LongTermMemoryObjectV1 | None:
        # AUDIT-FIX(#3): Parse payloads through safe coercion so malformed evidence does not raise here.
        payload = _mapping_or_empty(evidence.data)
        request_source_display = _normalize_text(payload.get("request_source"), limit=48).lower() or "unknown"
        request_source = _safe_key_component(request_source_display, fallback="unknown")
        # AUDIT-FIX(#9): Event-derived interaction patterns are day-scoped and should expire on the observation date.
        return LongTermMemoryObjectV1(
            memory_id=f"pattern:print:{request_source}:{daypart}",
            kind="pattern",
            summary=f"Printed Twinr output was used in the {daypart}.",
            details=(
                "Low-confidence print usage pattern derived from a "
                f"{request_source_display} print completion event."
            ),
            source=source_ref,
            confidence=0.61,
            sensitivity="low",
            slot_key=f"pattern:print:{request_source}:{daypart}",
            value_key="printed_output",
            valid_from=date_key,
            valid_to=date_key,
            attributes={
                "request_source": request_source,
                "request_source_label": request_source_display,
                "daypart": daypart,
                "pattern_type": "interaction",
            },
        )

    def _extract_camera_capture(
        self,
        *,
        evidence: LongTermMultimodalEvidence,
        source_ref: LongTermSourceRefV1,
        date_key: str,
        daypart: str,
    ) -> LongTermMemoryObjectV1 | None:
        # AUDIT-FIX(#3): Parse payloads through safe coercion so malformed evidence does not raise here.
        payload = _mapping_or_empty(evidence.data)
        purpose_display = _normalize_text(payload.get("purpose"), limit=48).lower() or "camera use"
        purpose = _safe_key_component(purpose_display, fallback="camera_use")
        # AUDIT-FIX(#9): Event-derived interaction patterns are day-scoped and should expire on the observation date.
        return LongTermMemoryObjectV1(
            memory_id=f"pattern:camera_use:{purpose}:{daypart}",
            kind="pattern",
            summary=f"The device camera was used in the {daypart}.",
            details=(
                "Low-confidence camera usage pattern derived from a "
                f"{purpose_display.replace('_', ' ')} event."
            ),
            source=source_ref,
            confidence=0.59,
            sensitivity="low",
            slot_key=f"pattern:camera_use:{purpose}:{daypart}",
            value_key="camera_used",
            valid_from=date_key,
            valid_to=date_key,
            attributes={
                "purpose": purpose,
                "purpose_label": purpose_display,
                "daypart": daypart,
                "pattern_type": "interaction",
            },
        )

    def _daypart(self, occurred_at: datetime) -> str:
        hour = occurred_at.hour
        if 5 <= hour < 11:
            return "morning"
        if 11 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 22:
            return "evening"
        return "night"


__all__ = ["LongTermMultimodalExtractor"]