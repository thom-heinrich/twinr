from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from twinr.memory.longterm.models import LongTermMultimodalEvidence


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _parse_created_at(value: object) -> datetime | None:
    text = _normalize_text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class LongTermOpsBackfillBuildResult:
    evidence: tuple[LongTermMultimodalEvidence, ...]
    scanned_events: int
    generated_evidence: int
    sensor_observations: int
    button_interactions: int
    print_completions: int


@dataclass(frozen=True, slots=True)
class LongTermOpsBackfillRunResult:
    scanned_events: int
    generated_evidence: int
    applied_evidence: int
    skipped_existing: int
    sensor_observations: int
    button_interactions: int
    print_completions: int
    reflected_objects: int
    created_summaries: int
    reflection_error: str | None = None


@dataclass(slots=True)
class _ReplayState:
    last_sensor_flags: dict[str, bool] = field(default_factory=dict)
    pending_print_request_source: str | None = None
    pending_print_started_at: datetime | None = None
    pending_print_payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LongTermOpsEventBackfiller:
    def load_entries(self, path: str | Path) -> tuple[dict[str, object], ...]:
        source = Path(path)
        if not source.exists():
            return ()
        entries: list[dict[str, object]] = []
        for raw_line in source.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                entries.append(parsed)
        return tuple(entries)

    def build_evidence(
        self,
        entries: Iterable[Mapping[str, object]],
    ) -> LongTermOpsBackfillBuildResult:
        state = _ReplayState()
        generated: list[LongTermMultimodalEvidence] = []
        scanned_events = 0
        sensor_count = 0
        button_count = 0
        print_count = 0
        for entry in entries:
            scanned_events += 1
            occurred_at = _parse_created_at(entry.get("created_at"))
            if occurred_at is None:
                continue
            event = _normalize_text(entry.get("event"))
            data = entry.get("data")
            payload = dict(data) if isinstance(data, Mapping) else {}
            if event == "proactive_observation":
                evidence = self._sensor_observation_evidence(
                    occurred_at=occurred_at,
                    payload=payload,
                    state=state,
                )
                generated.append(evidence)
                sensor_count += 1
                continue
            if event == "turn_started" and _normalize_text(payload.get("request_source")).lower() == "button":
                generated.append(
                    LongTermMultimodalEvidence(
                        event_name="button_interaction",
                        modality="button",
                        source="ops_backfill",
                        message="Backfilled green-button conversation start from ops history.",
                        data={"button": "green", "action": "start_listening", "backfill_source_event": event},
                        created_at=occurred_at,
                    )
                )
                button_count += 1
                continue
            if event == "print_started":
                button = _normalize_text(payload.get("button")).lower()
                request_source = _normalize_text(payload.get("request_source")).lower()
                if button == "yellow":
                    generated.append(
                        LongTermMultimodalEvidence(
                            event_name="button_interaction",
                            modality="button",
                            source="ops_backfill",
                            message="Backfilled yellow-button print request from ops history.",
                            data={"button": "yellow", "action": "print_request", "backfill_source_event": event},
                            created_at=occurred_at,
                        )
                    )
                    button_count += 1
                    state.pending_print_request_source = "button"
                elif request_source:
                    state.pending_print_request_source = request_source
                else:
                    state.pending_print_request_source = "unknown"
                state.pending_print_started_at = occurred_at
                state.pending_print_payload = payload
                continue
            if event == "print_job_sent":
                generated.append(
                    self._print_completed_evidence(
                        occurred_at=occurred_at,
                        payload=payload,
                        request_source=state.pending_print_request_source,
                        inferred=False,
                    )
                )
                print_count += 1
                state.pending_print_request_source = None
                state.pending_print_started_at = None
                state.pending_print_payload = {}
                continue
            if event == "print_finished" and state.pending_print_started_at is not None:
                generated.append(
                    self._print_completed_evidence(
                        occurred_at=state.pending_print_started_at,
                        payload=state.pending_print_payload,
                        request_source=state.pending_print_request_source,
                        inferred=True,
                    )
                )
                print_count += 1
                state.pending_print_request_source = None
                state.pending_print_started_at = None
                state.pending_print_payload = {}
                continue
        if state.pending_print_started_at is not None:
            generated.append(
                self._print_completed_evidence(
                    occurred_at=state.pending_print_started_at,
                    payload=state.pending_print_payload,
                    request_source=state.pending_print_request_source,
                    inferred=True,
                )
            )
            print_count += 1
        generated.sort(key=lambda item: item.created_at)
        return LongTermOpsBackfillBuildResult(
            evidence=tuple(generated),
            scanned_events=scanned_events,
            generated_evidence=len(generated),
            sensor_observations=sensor_count,
            button_interactions=button_count,
            print_completions=print_count,
        )

    def _sensor_observation_evidence(
        self,
        *,
        occurred_at: datetime,
        payload: Mapping[str, object],
        state: _ReplayState,
    ) -> LongTermMultimodalEvidence:
        facts = {
            "sensor": {
                "inspected": bool(payload.get("inspected")),
                "observed_at": occurred_at.timestamp(),
            },
            "pir": {
                "motion_detected": bool(payload.get("pir_motion_detected")),
                "low_motion": bool(payload.get("low_motion")),
            },
            "camera": {
                "person_visible": bool(payload.get("person_visible")),
                "looking_toward_device": bool(payload.get("looking_toward_device")),
                "body_pose": _normalize_text(payload.get("body_pose")).lower() or "unknown",
                "smiling": bool(payload.get("smiling")),
                "hand_or_object_near_camera": bool(payload.get("hand_or_object_near_camera")),
            },
            "vad": {
                "speech_detected": bool(payload.get("speech_detected")),
                "distress_detected": bool(payload.get("distress_detected")),
                "quiet": not bool(payload.get("speech_detected")),
            },
        }
        flags = {
            "pir.motion_detected": bool(facts["pir"]["motion_detected"]),
            "camera.person_visible": bool(facts["camera"]["person_visible"]),
            "camera.hand_or_object_near_camera": bool(facts["camera"]["hand_or_object_near_camera"]),
            "vad.speech_detected": bool(facts["vad"]["speech_detected"]),
        }
        event_names = [
            key
            for key, active in flags.items()
            if active and state.last_sensor_flags.get(key) is not True
        ]
        state.last_sensor_flags = flags
        return LongTermMultimodalEvidence(
            event_name="sensor_observation",
            modality="sensor",
            source="ops_backfill",
            message="Backfilled proactive sensor observation from ops history.",
            data={
                "facts": facts,
                "event_names": event_names,
                "backfill_source_event": "proactive_observation",
            },
            created_at=occurred_at,
        )

    def _print_completed_evidence(
        self,
        *,
        occurred_at: datetime,
        payload: Mapping[str, object],
        request_source: str | None,
        inferred: bool,
    ) -> LongTermMultimodalEvidence:
        queue = _normalize_text(payload.get("queue"))
        job = _normalize_text(payload.get("job"))
        source = _normalize_text(request_source).lower() or "unknown"
        return LongTermMultimodalEvidence(
            event_name="print_completed",
            modality="printer",
            source="ops_backfill",
            message="Backfilled printed-output usage from ops history.",
            data={
                "request_source": source,
                "queue": queue,
                "job": job,
                "inferred_from_finish_only": inferred,
                "backfill_source_event": "print_finished" if inferred else "print_job_sent",
            },
            created_at=occurred_at,
        )
