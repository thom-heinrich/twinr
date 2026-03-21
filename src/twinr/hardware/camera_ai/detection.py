"""Parse IMX500 SSD outputs into Twinr's bounded detection contract.

Real "no person detected" frames must stay distinct from camera/runtime
failures. When the IMX500 session cannot start, tensors are missing, or parse
fails, callers need an explicit failure instead of an empty detection result so
they can keep the last stable user-facing target and expose camera health
correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any

from .config import AICameraAdapterConfig, _coerce_float
from .geometry import box_from_detection, zone_from_center
from .models import AICameraBox, AICameraObjectDetection, AICameraVisiblePerson, AICameraZone


LOGGER = logging.getLogger(__name__)  # AUDIT-FIX(#1): Emit capture/parse failures without taking down the caller.
_MAX_DETECTIONS = 100
_MAX_LABEL_LENGTH = 64  # AUDIT-FIX(#5): Keep normalized labels bounded and safe for downstream UI/logging.

_DEFAULT_UNDEFINED_LABELS = frozenset({"-", "", "unknown"})
_PERSON_LABELS = frozenset({"person"})


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Describe one parsed detection frame."""

    person_count: int
    primary_person_box: AICameraBox | None
    primary_person_zone: AICameraZone
    visible_persons: tuple[AICameraVisiblePerson, ...]
    person_near_device: bool | None
    hand_or_object_near_camera: bool
    objects: tuple[AICameraObjectDetection, ...]


def capture_detection(
    *,
    runtime_manager: Any,
    runtime: dict[str, Any],
    config: AICameraAdapterConfig,
    observed_at: float,
) -> DetectionResult:
    """Capture and parse one detection frame from the configured SSD network."""

    try:
        session = runtime_manager.ensure_session(
            runtime,
            network_path=config.detection_network_path,
            task_name="detection",
        )
        metadata = runtime_manager.capture_metadata(session, observed_at=observed_at)
        outputs = session.imx500.get_outputs(metadata, add_batch=True)
    except Exception as exc:  # AUDIT-FIX(#1): Surface capture/runtime faults so callers do not mistake them for "no person".
        LOGGER.warning("ai_camera_detection_capture_failed", exc_info=True)
        raise RuntimeError("detection_capture_failed") from exc

    if outputs is None:  # AUDIT-FIX(#2): Avoid ambiguous truthiness on ndarray-like outputs.
        LOGGER.warning("ai_camera_detection_outputs_missing")
        raise RuntimeError("detection_outputs_missing")

    try:
        output_count = len(outputs)
    except TypeError:  # AUDIT-FIX(#2): Reject non-container output payloads deterministically.
        LOGGER.warning("ai_camera_detection_outputs_invalid_container")
        raise RuntimeError("detection_outputs_invalid_container")

    if output_count < 4:
        LOGGER.warning("ai_camera_detection_outputs_incomplete")
        raise RuntimeError("detection_outputs_incomplete")

    try:
        labels = tuple(getattr(getattr(session.imx500, "network_intrinsics", None), "labels", ()) or ())
        box_entries = _unwrap_batch_entries(outputs[0], expected_unbatched_depth=2)  # AUDIT-FIX(#3): Read batched box tensors defensively.
        score_entries = _unwrap_batch_entries(outputs[1], expected_unbatched_depth=1)  # AUDIT-FIX(#3): Read batched score tensors defensively.
        class_entries = _unwrap_batch_entries(outputs[2], expected_unbatched_depth=1)  # AUDIT-FIX(#3): Read batched class tensors defensively.
        raw_count = _extract_detection_count(outputs[3])  # AUDIT-FIX(#4): Coerce counts safely and reject NaN/Inf.
        max_entries = min(
            _MAX_DETECTIONS,
            _safe_len(box_entries),
            _safe_len(score_entries),
            _safe_len(class_entries),
        )
        count = min(raw_count, max_entries)  # AUDIT-FIX(#3): Never trust the reported detection count over tensor bounds.

        if count <= 0:
            return _empty_detection_result()

        person_boxes: list[tuple[float, float, AICameraBox]] = []
        object_detections: list[AICameraObjectDetection] = []

        for index in range(count):
            score = _coerce_probability(score_entries[index], default=0.0)  # AUDIT-FIX(#4): Clamp confidence into the bounded contract.
            class_index = _coerce_non_negative_int(class_entries[index], default=0)  # AUDIT-FIX(#4): Avoid round-induced class drift.
            label = labels[class_index] if class_index < len(labels) else f"class_{class_index}"
            normalized_label = _normalize_label(label)
            if normalized_label in _DEFAULT_UNDEFINED_LABELS:
                continue

            box = _safe_box_from_detection(box_entries[index])  # AUDIT-FIX(#6): Skip only malformed boxes instead of failing the whole frame.
            if box is None:
                continue

            if normalized_label in _PERSON_LABELS:
                if score >= config.person_confidence_threshold:
                    person_boxes.append((score, box.area, box))  # AUDIT-FIX(#7): Deterministic tie-breaker reduces primary-target flapping.
                continue

            if score < config.object_confidence_threshold:
                continue

            object_detections.append(
                AICameraObjectDetection(
                    label=normalized_label,
                    confidence=score,
                    zone=_safe_zone_from_center(box.center_x),  # AUDIT-FIX(#6): Invalid center values degrade to UNKNOWN instead of crashing.
                    box=box,
                )
            )

        person_boxes.sort(key=lambda item: (item[0], item[1]), reverse=True)  # AUDIT-FIX(#7): Prefer higher confidence, then larger box.
        primary_person_box = person_boxes[0][2] if person_boxes else None
        person_count = len(person_boxes)
        visible_persons = tuple(
            AICameraVisiblePerson(
                box=box,
                zone=_safe_zone_from_center(box.center_x),
                confidence=score,
            )
            for score, _area, box in person_boxes
        )
        person_near_device = None
        primary_zone = AICameraZone.UNKNOWN
        if primary_person_box is not None:
            primary_zone = _safe_zone_from_center(primary_person_box.center_x)  # AUDIT-FIX(#6): Invalid center values degrade safely.
            person_near_device = (
                primary_person_box.area >= config.person_near_area_threshold
                or primary_person_box.height >= config.person_near_height_threshold
            )

        hand_or_object_near_camera = any(
            detection.box.area >= config.object_near_area_threshold
            for detection in object_detections
        )
    except Exception as exc:  # AUDIT-FIX(#1): Parsing faults are camera/runtime failures, not authoritative empty frames.
        LOGGER.warning("ai_camera_detection_parse_failed", exc_info=True)
        raise RuntimeError("detection_parse_failed") from exc

    return DetectionResult(
        person_count=person_count,
        primary_person_box=primary_person_box,
        primary_person_zone=primary_zone,
        visible_persons=visible_persons,
        person_near_device=person_near_device,
        hand_or_object_near_camera=hand_or_object_near_camera,
        objects=tuple(object_detections),
    )


def _empty_detection_result() -> DetectionResult:  # AUDIT-FIX(#1): Standardize the safe fallback frame for capture/parse failures.
    """Return the safest empty detection frame."""

    return DetectionResult(
        person_count=0,
        primary_person_box=None,
        primary_person_zone=AICameraZone.UNKNOWN,
        visible_persons=(),
        person_near_device=None,
        hand_or_object_near_camera=False,
        objects=(),
    )


def _unwrap_batch_entries(value: Any, *, expected_unbatched_depth: int) -> Any:  # AUDIT-FIX(#3): Unwrap batch dimensions without corrupting unbatched tensors.
    """Return the first batch slice when the tensor includes a batch dimension."""

    if _safe_len(value) <= 0:
        return ()
    if _safe_len(value) == 1 and _nesting_depth(value, max_depth=expected_unbatched_depth + 1) >= expected_unbatched_depth + 1:
        try:
            return value[0]
        except (IndexError, KeyError, TypeError):
            return ()
    return value


def _nesting_depth(value: Any, *, max_depth: int) -> int:  # AUDIT-FIX(#3): Infer tensor depth cheaply for list/ndarray-like outputs.
    """Estimate how many indexable levels a tensor-like object exposes."""

    depth = 0
    current = value
    while depth < max_depth and _safe_len(current) > 0:
        try:
            current = current[0]
        except (IndexError, KeyError, TypeError):
            break
        depth += 1
    return depth


def _extract_detection_count(value: Any) -> int:  # AUDIT-FIX(#4): Sanitize reported detection count before indexing tensors.
    """Read one model-reported detection count and clamp it to sane bounds."""

    current = value
    for _ in range(2):
        if _safe_len(current) <= 0:
            break
        try:
            current = current[0]
        except (IndexError, KeyError, TypeError):
            break
    return min(_MAX_DETECTIONS, _coerce_non_negative_int(current, default=0))


def _coerce_non_negative_int(value: object, *, default: int = 0) -> int:  # AUDIT-FIX(#4): Prevent round-up and NaN/Inf crashes on model integers.
    """Convert one model value to a non-negative integer without rounding up."""

    number = _coerce_finite_float(value, default=float(default))
    if number <= 0.0:
        return 0
    return int(number)


def _coerce_probability(value: object, *, default: float = 0.0) -> float:  # AUDIT-FIX(#4): Keep confidences inside the bounded contract.
    """Convert one model confidence value into the inclusive range [0.0, 1.0]."""

    number = _coerce_finite_float(value, default=default)
    if number <= 0.0:
        return 0.0
    if number >= 1.0:
        return 1.0
    return number


def _coerce_finite_float(value: object, *, default: float) -> float:  # AUDIT-FIX(#4): Reject non-finite numeric payloads from camera tensors.
    """Convert one value to a finite float or fall back to a default."""

    try:
        number = _coerce_float(value, default=default)
    except Exception:
        return default
    if not math.isfinite(number):
        return default
    return number


def _safe_box_from_detection(value: Any) -> AICameraBox | None:  # AUDIT-FIX(#6): Isolate malformed boxes to a single skipped detection.
    """Parse one raw detection box and reject unusable numeric values."""

    try:
        box = box_from_detection(value)
    except Exception:
        return None

    if not _is_finite_number(getattr(box, "center_x", None)):
        return None
    if not _is_finite_number(getattr(box, "area", None)):
        return None
    if not _is_finite_number(getattr(box, "height", None)):
        return None
    if box.area < 0.0 or box.height < 0.0:
        return None
    return box


def _safe_zone_from_center(value: object) -> AICameraZone:  # AUDIT-FIX(#6): Convert invalid center values into UNKNOWN instead of exceptions.
    """Map one horizontal center value to a zone without raising."""

    if not _is_finite_number(value):
        return AICameraZone.UNKNOWN
    try:
        return zone_from_center(float(value))
    except Exception:
        return AICameraZone.UNKNOWN


def _safe_len(value: Any) -> int:  # AUDIT-FIX(#3): Treat scalar tensor leaves as zero-length for safe bounds logic.
    """Return len(value) or zero for scalar / invalid containers."""

    try:
        return len(value)
    except TypeError:
        return 0


def _is_finite_number(value: object) -> bool:  # AUDIT-FIX(#6): Validate derived box metrics before downstream use.
    """Return True when the value can be represented as a finite float."""

    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _normalize_label(value: object) -> str:  # AUDIT-FIX(#5): Decode/sanitize labels before they enter the detection contract.
    """Normalize one object label to an inspectable token."""

    if isinstance(value, (bytes, bytearray)):  # AUDIT-FIX(#5): IMX labels may arrive as bytes from camera metadata.
        text = value.decode("utf-8", errors="ignore")
    else:
        text = str(value or "")
    collapsed = "_".join(text.strip().lower().split())
    sanitized_chars: list[str] = []
    last_was_separator = False
    for char in collapsed:
        if char.isalnum():
            sanitized_chars.append(char)
            last_was_separator = False
            continue
        if char in {"_", "-"}:
            if not last_was_separator:
                sanitized_chars.append(char)
            last_was_separator = True
            continue
        if not last_was_separator:
            sanitized_chars.append("_")
        last_was_separator = True
    normalized = "".join(sanitized_chars).strip("_-")
    return normalized[:_MAX_LABEL_LENGTH]


__all__ = ["DetectionResult", "capture_detection"]
