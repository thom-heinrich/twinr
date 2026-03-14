from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from zoneinfo import ZoneInfo

from twinr.memory.longterm.models import (
    LongTermMemoryObjectV1,
    LongTermMultimodalEvidence,
    LongTermSourceRefV1,
    LongTermTurnExtractionV1,
)
from twinr.text_utils import collapse_whitespace, slugify_identifier


def _normalize_text(value: str | None, *, limit: int | None = None) -> str:
    text = collapse_whitespace(value)
    if limit is None or len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _slugify(value: str, *, fallback: str) -> str:
    return slugify_identifier(value, fallback=fallback)


def _safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True, slots=True)
class LongTermMultimodalExtractor:
    timezone_name: str = "Europe/Berlin"

    def extract_evidence(
        self,
        evidence: LongTermMultimodalEvidence,
    ) -> LongTermTurnExtractionV1:
        occurred_at = evidence.created_at.astimezone(ZoneInfo(self.timezone_name))
        date_key = occurred_at.date().isoformat()
        daypart = self._daypart(occurred_at)
        event_slug = _slugify(evidence.event_name, fallback="event")
        event_id = f"multimodal:{occurred_at.strftime('%Y%m%dT%H%M%S%z')}:{event_slug}"
        source_ref = LongTermSourceRefV1(
            source_type=evidence.source,
            event_ids=(event_id,),
            modality=evidence.modality,
        )
        episode = LongTermMemoryObjectV1(
            memory_id=f"episode:{_slugify(event_id, fallback='multimodal')}",
            kind="episode",
            summary=f"Multimodal device event recorded: {evidence.event_name}.",
            details=self._episode_details(evidence),
            source=source_ref,
            status="candidate",
            confidence=0.92,
            sensitivity="normal",
            slot_key=f"episode:{event_id}",
            value_key=event_slug,
            valid_from=date_key,
            valid_to=date_key,
            attributes={
                "event_name": evidence.event_name,
                "modality": evidence.modality,
                "daypart": daypart,
            },
        )

        candidates: list[LongTermMemoryObjectV1] = []
        if evidence.event_name == "sensor_observation":
            candidates.extend(
                self._extract_sensor_observation(
                    evidence=evidence,
                    source_ref=source_ref,
                    date_key=date_key,
                    daypart=daypart,
                )
            )
        elif evidence.event_name == "button_interaction":
            item = self._extract_button_interaction(
                evidence=evidence,
                source_ref=source_ref,
                date_key=date_key,
                daypart=daypart,
            )
            if item is not None:
                candidates.append(item)
        elif evidence.event_name == "print_completed":
            item = self._extract_print_completed(
                evidence=evidence,
                source_ref=source_ref,
                date_key=date_key,
                daypart=daypart,
            )
            if item is not None:
                candidates.append(item)
        elif evidence.event_name == "camera_capture":
            item = self._extract_camera_capture(
                evidence=evidence,
                source_ref=source_ref,
                date_key=date_key,
                daypart=daypart,
            )
            if item is not None:
                candidates.append(item)

        return LongTermTurnExtractionV1(
            turn_id=event_id,
            occurred_at=occurred_at,
            episode=episode,
            candidate_objects=tuple(candidates),
            graph_edges=(),
        )

    def _episode_details(self, evidence: LongTermMultimodalEvidence) -> str:
        payload = {
            "event_name": evidence.event_name,
            "modality": evidence.modality,
            "message": evidence.message,
            "data": dict(evidence.data or {}),
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
        payload = dict(evidence.data or {})
        facts = payload.get("facts")
        fact_map = dict(facts) if isinstance(facts, dict) else payload
        pir = dict(fact_map.get("pir") or {}) if isinstance(fact_map.get("pir"), dict) else {}
        camera = dict(fact_map.get("camera") or {}) if isinstance(fact_map.get("camera"), dict) else {}
        event_names = tuple(str(item) for item in (payload.get("event_names") or ()) if str(item).strip())
        objects: list[LongTermMemoryObjectV1] = []
        person_visible = bool(camera.get("person_visible"))
        motion_detected = bool(pir.get("motion_detected"))
        object_near_camera = bool(camera.get("hand_or_object_near_camera"))
        if motion_detected:
            objects.append(
                LongTermMemoryObjectV1(
                    memory_id=f"observation:{date_key}:pir_motion",
                    kind="situational_observation",
                    summary="Motion was detected near the device.",
                    details="Derived from PIR sensor activity.",
                    source=source_ref,
                    confidence=0.62,
                    sensitivity="low",
                    slot_key=f"observation:pir_motion:{date_key}",
                    value_key="motion_detected",
                    valid_from=date_key,
                    valid_to=date_key,
                    attributes={"topic": "pir_motion", "event_names": event_names},
                )
            )
        if person_visible:
            objects.append(
                LongTermMemoryObjectV1(
                    memory_id=f"observation:{date_key}:camera_person_visible",
                    kind="situational_observation",
                    summary="A person was visible near the device.",
                    details="Derived from camera-based presence observation.",
                    source=source_ref,
                    confidence=0.64,
                    sensitivity="low",
                    slot_key=f"observation:camera_person_visible:{date_key}",
                    value_key="person_visible",
                    valid_from=date_key,
                    valid_to=date_key,
                    attributes={"topic": "camera_presence", "event_names": event_names},
                )
            )
        if object_near_camera:
            objects.append(
                LongTermMemoryObjectV1(
                    memory_id=f"observation:{date_key}:camera_object_near",
                    kind="situational_observation",
                    summary="An object or hand was near the camera.",
                    details="Derived from camera interaction observation.",
                    source=source_ref,
                    confidence=0.64,
                    sensitivity="low",
                    slot_key=f"observation:camera_object_near:{date_key}",
                    value_key="object_near_camera",
                    valid_from=date_key,
                    valid_to=date_key,
                    attributes={"topic": "camera_interaction", "event_names": event_names},
                )
            )
        if motion_detected or person_visible:
            objects.append(
                LongTermMemoryObjectV1(
                    memory_id=f"pattern:presence:{daypart}:near_device",
                    kind="presence_pattern_fact",
                    summary=f"Presence near the device was observed in the {daypart}.",
                    details="Low-confidence multimodal pattern derived from PIR/camera evidence.",
                    source=source_ref,
                    confidence=0.58,
                    sensitivity="low",
                    slot_key=f"pattern:presence:{daypart}:near_device",
                    value_key="presence_observed",
                    attributes={
                        "daypart": daypart,
                        "uses_pir": motion_detected,
                        "uses_camera": person_visible,
                    },
                )
            )
        if object_near_camera:
            objects.append(
                LongTermMemoryObjectV1(
                    memory_id=f"pattern:camera_interaction:{daypart}",
                    kind="interaction_pattern_fact",
                    summary=f"Camera-side interaction was observed in the {daypart}.",
                    details="Low-confidence multimodal pattern derived from repeated camera-side interaction signals.",
                    source=source_ref,
                    confidence=0.57,
                    sensitivity="low",
                    slot_key=f"pattern:camera_interaction:{daypart}",
                    value_key="camera_interaction_observed",
                    attributes={"daypart": daypart, "event_names": event_names},
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
        payload = dict(evidence.data or {})
        button = _normalize_text(str(payload.get("button", ""))).lower()
        action = _normalize_text(str(payload.get("action", ""))).lower()
        if not button or not action:
            return None
        action_label = {
            "start_listening": "start a conversation",
            "print_request": "request a printed answer",
        }.get(action, action.replace("_", " "))
        return LongTermMemoryObjectV1(
            memory_id=f"pattern:button:{button}:{action}:{daypart}",
            kind="interaction_pattern_fact",
            summary=f"The {button} button was used to {action_label} in the {daypart}.",
            details="Low-confidence button usage pattern derived from a physical interaction event.",
            source=source_ref,
            confidence=0.6,
            sensitivity="low",
            slot_key=f"pattern:button:{button}:{action}:{daypart}",
            value_key="button_used",
            valid_from=date_key,
            attributes={"button": button, "action": action, "daypart": daypart},
        )

    def _extract_print_completed(
        self,
        *,
        evidence: LongTermMultimodalEvidence,
        source_ref: LongTermSourceRefV1,
        date_key: str,
        daypart: str,
    ) -> LongTermMemoryObjectV1 | None:
        payload = dict(evidence.data or {})
        request_source = _normalize_text(str(payload.get("request_source", ""))).lower() or "unknown"
        return LongTermMemoryObjectV1(
            memory_id=f"pattern:print:{request_source}:{daypart}",
            kind="interaction_pattern_fact",
            summary=f"Printed Twinr output was used in the {daypart}.",
            details=f"Low-confidence print usage pattern derived from a {request_source} print completion event.",
            source=source_ref,
            confidence=0.61,
            sensitivity="low",
            slot_key=f"pattern:print:{request_source}:{daypart}",
            value_key="printed_output",
            valid_from=date_key,
            attributes={"request_source": request_source, "daypart": daypart},
        )

    def _extract_camera_capture(
        self,
        *,
        evidence: LongTermMultimodalEvidence,
        source_ref: LongTermSourceRefV1,
        date_key: str,
        daypart: str,
    ) -> LongTermMemoryObjectV1 | None:
        payload = dict(evidence.data or {})
        purpose = _normalize_text(str(payload.get("purpose", ""))).lower() or "camera_use"
        return LongTermMemoryObjectV1(
            memory_id=f"pattern:camera_use:{purpose}:{daypart}",
            kind="interaction_pattern_fact",
            summary=f"The device camera was used in the {daypart}.",
            details=f"Low-confidence camera usage pattern derived from a {purpose.replace('_', ' ')} event.",
            source=source_ref,
            confidence=0.59,
            sensitivity="low",
            slot_key=f"pattern:camera_use:{purpose}:{daypart}",
            value_key="camera_used",
            valid_from=date_key,
            attributes={"purpose": purpose, "daypart": daypart},
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
