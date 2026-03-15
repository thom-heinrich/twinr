from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone  # AUDIT-FIX(#2): Normalize all parsed timestamps to UTC-aware datetimes.
import json
import logging  # AUDIT-FIX(#6): Add bounded diagnostics for malformed input and dropped incomplete events.
import os  # AUDIT-FIX(#3): Use low-level file open flags to reduce TOCTOU/symlink exposure on local file reads.
from pathlib import Path
import stat  # AUDIT-FIX(#3): Verify the opened path is a regular file before reading from it.

from twinr.memory.longterm.models import LongTermMultimodalEvidence

_LOGGER = logging.getLogger(__name__)  # AUDIT-FIX(#6): Emit operator-visible warnings instead of silently discarding important failures.

_TRUE_TEXT_VALUES = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_TEXT_VALUES = frozenset({"0", "false", "f", "no", "n", "off", "", "none", "null"})


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _coerce_bool(value: object) -> bool:  # AUDIT-FIX(#5): Parse common serialized boolean spellings instead of relying on bool("false")==True.
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, float):
        return value != 0.0
    text = _normalize_text(value).lower()
    if text in _TRUE_TEXT_VALUES:
        return True
    if text in _FALSE_TEXT_VALUES:
        return False
    return bool(text)


def _parse_created_at(value: object) -> datetime | None:
    text = _normalize_text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00").replace("z", "+00:00"))  # AUDIT-FIX(#2): Accept UTC "Z"/"z" forms consistently before normalization.
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)  # AUDIT-FIX(#2): Normalize naive timestamps to UTC to avoid mixed naive/aware sort crashes.
    return parsed.astimezone(timezone.utc)  # AUDIT-FIX(#2): Normalize aware timestamps to a single timezone for deterministic ordering and epoch conversion.


def _merge_payloads(*payloads: Mapping[str, object]) -> dict[str, object]:  # AUDIT-FIX(#1): Preserve the best available print metadata by merging start and completion payloads.
    merged: dict[str, object] = {}
    for payload in payloads:
        merged.update(dict(payload))
    return merged


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


def _reset_pending_print(state: _ReplayState) -> None:  # AUDIT-FIX(#1): Centralize pending-print cleanup so stale metadata cannot bleed into the next job.
    state.pending_print_request_source = None
    state.pending_print_started_at = None
    state.pending_print_payload = {}


@dataclass(frozen=True, slots=True)
class LongTermOpsEventBackfiller:
    def load_entries(self, path: str | Path) -> tuple[dict[str, object], ...]:
        source = Path(path)
        open_flags = os.O_RDONLY  # AUDIT-FIX(#3): Avoid Path.exists()/read_text() TOCTOU and open the file exactly once.
        if hasattr(os, "O_NOFOLLOW"):
            open_flags |= os.O_NOFOLLOW  # AUDIT-FIX(#3): Refuse symlink traversal where the platform supports it.

        fd: int | None = None
        entries: list[dict[str, object]] = []
        skipped_lines = 0

        try:
            fd = os.open(source, open_flags)
            source_stat = os.fstat(fd)
            if not stat.S_ISREG(source_stat.st_mode):
                _LOGGER.warning("Refusing to read non-regular ops history file %s.", source)  # AUDIT-FIX(#3): Reject directories, devices, and other special files.
                return ()

            with os.fdopen(fd, "r", encoding="utf-8") as handle:  # AUDIT-FIX(#4): Stream line-by-line to avoid whole-file peak memory spikes on RPi-class hardware.
                fd = None
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        skipped_lines += 1  # AUDIT-FIX(#6): Count malformed lines for a single bounded warning after the read completes.
                        continue
                    if isinstance(parsed, dict):
                        entries.append(parsed)
                    else:
                        skipped_lines += 1  # AUDIT-FIX(#6): Non-object JSON lines are invalid for this replay format; report them once, not silently.

        except FileNotFoundError:
            return ()
        except (OSError, UnicodeDecodeError) as exc:
            _LOGGER.warning("Unable to read ops history file %s: %s", source, exc)  # AUDIT-FIX(#3): Degrade gracefully instead of crashing the caller on local file errors.
            return ()
        finally:
            if fd is not None:
                os.close(fd)

        if skipped_lines:
            _LOGGER.warning("Skipped %d malformed ops history line(s) while reading %s.", skipped_lines, source)  # AUDIT-FIX(#6): Surface data-quality issues for operators and remote support.

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
                        payload=_merge_payloads(state.pending_print_payload, payload),  # AUDIT-FIX(#1): Preserve start metadata while letting the completion event override stale fields.
                        request_source=state.pending_print_request_source,
                        inferred=False,
                        source_event="print_job_sent",
                    )
                )
                print_count += 1
                _reset_pending_print(state)  # AUDIT-FIX(#1): Clear pending print state after the completion signal to prevent cross-job contamination.
                continue
            if event == "print_finished" and state.pending_print_started_at is not None:
                generated.append(
                    self._print_completed_evidence(
                        occurred_at=occurred_at,  # AUDIT-FIX(#1): Use the actual finish timestamp instead of backdating completion to print start.
                        payload=_merge_payloads(state.pending_print_payload, payload),  # AUDIT-FIX(#1): Retain queued job metadata when finish events carry only partial details.
                        request_source=state.pending_print_request_source,
                        inferred=False,
                        source_event="print_finished",
                    )
                )
                print_count += 1
                _reset_pending_print(state)  # AUDIT-FIX(#1): Clear pending state after a physical finish event.
                continue
        if state.pending_print_started_at is not None:
            _LOGGER.warning(
                "Dropping incomplete print_started event from %s without a completion signal.",
                state.pending_print_started_at.isoformat(),
            )  # AUDIT-FIX(#1): Do not fabricate a completed print from an unconfirmed start event; surface the gap instead.
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
        speech_detected = _coerce_bool(payload.get("speech_detected"))  # AUDIT-FIX(#5): Parse serialized booleans once so quiet/distress logic stays internally consistent.
        facts = {
            "sensor": {
                "inspected": _coerce_bool(payload.get("inspected")),
                "observed_at": occurred_at.timestamp(),
            },
            "pir": {
                "motion_detected": _coerce_bool(payload.get("pir_motion_detected")),
                "low_motion": _coerce_bool(payload.get("low_motion")),
            },
            "camera": {
                "person_visible": _coerce_bool(payload.get("person_visible")),
                "looking_toward_device": _coerce_bool(payload.get("looking_toward_device")),
                "body_pose": _normalize_text(payload.get("body_pose")).lower() or "unknown",
                "smiling": _coerce_bool(payload.get("smiling")),
                "hand_or_object_near_camera": _coerce_bool(payload.get("hand_or_object_near_camera")),
            },
            "vad": {
                "speech_detected": speech_detected,
                "distress_detected": _coerce_bool(payload.get("distress_detected")),
                "quiet": not speech_detected,
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
        source_event: str,
    ) -> LongTermMultimodalEvidence:
        queue = _normalize_text(payload.get("queue"))
        job = _normalize_text(payload.get("job"))
        source = _normalize_text(request_source).lower()
        if not source:
            source = _normalize_text(payload.get("request_source")).lower()  # AUDIT-FIX(#1): Recover request provenance when completion arrives without a pending start state.
        source = source or "unknown"
        return LongTermMultimodalEvidence(
            event_name="print_completed",
            modality="printer",
            source="ops_backfill",
            message="Backfilled printed-output usage from ops history.",
            data={
                "request_source": source,
                "queue": queue,
                "job": job,
                "inferred_from_print_start": inferred,
                "backfill_source_event": source_event,  # AUDIT-FIX(#1): Record the real completion signal for downstream provenance checks.
            },
            created_at=occurred_at,
        )


__all__ = [
    "LongTermOpsBackfillBuildResult",
    "LongTermOpsBackfillRunResult",
    "LongTermOpsEventBackfiller",
]