"""Parse IMX500 object-detection outputs into Twinr's bounded detection contract.

Real "no person detected" frames must stay distinct from camera/runtime
failures. When the IMX500 session cannot start, tensors are missing, or parse
fails, callers need an explicit failure instead of an empty detection result so
they can keep the last stable user-facing target and expose camera health
correctly.
"""

# CHANGELOG: 2026-03-28
# BUG-1: Fixed misparsed boxes for IMX500 models that rely on network_intrinsics
#        (bbox_normalization / bbox_order). The old parser could return wrong
#        zones, areas, and "near device" decisions for current YOLO/EfficientDet
#        style IMX500 models.
# BUG-2: Fixed outright parse failure for Nanodet-style IMX500 models and for
#        valid 3-tensor detection outputs. The old code hard-required 4 outputs.
# BUG-3: Fixed false positives caused by malformed class tensors defaulting to
#        class 0, which can silently hallucinate a person on corrupted frames.
# SEC-1: Added rate-limited warning emission so repeated camera/runtime faults
#        cannot spam logs indefinitely and exhaust SD-card-backed storage.
# IMP-1: Upgraded the parser to a multi-backend IMX500 detection parser that
#        auto-consumes network intrinsics and supports current Raspberry Pi AI
#        Camera reference detector families instead of only SSD-style outputs.
# IMP-2: Added lightweight temporal primary-person stabilization using the
#        existing runtime dict, reducing target flapping without introducing a
#        heavyweight tracker dependency.
# IMP-3: Added axis-aware bbox normalization and optional config-driven label
#        aliases / thresholds so custom non-square detectors remain correct.

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Any, Callable

from .config import AICameraAdapterConfig, _coerce_float
from .geometry import box_from_detection, zone_from_center
from .models import AICameraBox, AICameraObjectDetection, AICameraVisiblePerson, AICameraZone


LOGGER = logging.getLogger(__name__)

_MAX_DETECTIONS = 100
_MAX_LABEL_LENGTH = 64

_REQUIRED_STANDARD_OUTPUTS = 3
_OPTIONAL_COUNT_OUTPUT_INDEX = 3

_ALWAYS_UNDEFINED_LABELS = frozenset({"-", ""})
_OPTIONAL_UNDEFINED_LABELS = frozenset({"unknown", "background", "__background__", "bg"})
_DEFAULT_PERSON_LABELS = frozenset({"person"})

_DEFAULT_NANODET_IOU_THRESHOLD = 0.65
_DEFAULT_PRIMARY_STABILITY_WINDOW = 0.05
_DEFAULT_PRIMARY_CENTER_BIAS = 0.08
_DEFAULT_PRIMARY_AREA_BIAS = 0.04

_RUNTIME_PRIMARY_HINT_KEY = "_twinr_detection_primary_hint"
_LOG_RATE_LIMIT_SECONDS = 30.0

_LOG_STATE_LOCK = threading.Lock()
_LOG_STATE: dict[str, tuple[float, int]] = {}


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


@dataclass(frozen=True, slots=True)
class _DetectionIntrinsics:
    """Normalized parsing hints carried by the current IMX500 network."""

    postprocess: str
    bbox_normalization: bool
    bbox_order: str
    ignore_undefined_labels: bool
    input_width: int | None
    input_height: int | None
    labels: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _PreparedDetection:
    """One detector output after backend-specific parsing."""

    box: tuple[float, float, float, float]
    score: float
    class_index: int


def capture_detection(
    *,
    runtime_manager: Any,
    runtime: dict[str, Any],
    config: AICameraAdapterConfig,
    observed_at: float,
) -> DetectionResult:
    """Capture and parse one detection frame from the configured detection network."""

    try:
        session = runtime_manager.ensure_session(
            runtime,
            network_path=config.detection_network_path,
            task_name="detection",
        )
        metadata = runtime_manager.capture_metadata(session, observed_at=observed_at)
        outputs = session.imx500.get_outputs(metadata, add_batch=True)
    except Exception as exc:
        _log_warning_rate_limited(
            "ai_camera_detection_capture_failed",
            exc_info=True,
        )
        raise RuntimeError("detection_capture_failed") from exc

    if outputs is None:
        _log_warning_rate_limited("ai_camera_detection_outputs_missing")
        raise RuntimeError("detection_outputs_missing")

    try:
        output_count = len(outputs)
    except TypeError as exc:
        _log_warning_rate_limited("ai_camera_detection_outputs_invalid_container")
        raise RuntimeError("detection_outputs_invalid_container") from exc

    intrinsics = _read_detection_intrinsics(session=session, config=config)

    try:
        prepared_detections = _prepare_detections(
            session=session,
            outputs=outputs,
            output_count=output_count,
            intrinsics=intrinsics,
            config=config,
        )

        if not prepared_detections:
            runtime.pop(_RUNTIME_PRIMARY_HINT_KEY, None)
            return _empty_detection_result()

        person_labels = _read_person_labels(config)
        person_entries: list[tuple[float, float, AICameraBox]] = []
        object_detections: list[AICameraObjectDetection] = []

        for prepared in prepared_detections:
            label = _label_for_class_index(intrinsics.labels, prepared.class_index)
            normalized_label = _normalize_label(label)
            if _should_skip_label(
                normalized_label,
                ignore_undefined_labels=intrinsics.ignore_undefined_labels,
            ):
                continue

            box = _safe_box_from_detection(prepared.box)
            if box is None:
                continue

            if normalized_label in person_labels:
                if prepared.score >= _read_person_confidence_threshold(config):
                    person_entries.append((prepared.score, box.area, box))
                continue

            if prepared.score < _read_object_confidence_threshold(config):
                continue

            object_detections.append(
                AICameraObjectDetection(
                    label=normalized_label,
                    confidence=prepared.score,
                    zone=_safe_zone_from_center(box.center_x),
                    box=box,
                )
            )

        primary_person_box: AICameraBox | None = None
        primary_zone = AICameraZone.UNKNOWN
        person_near_device: bool | None = None
        visible_persons: tuple[AICameraVisiblePerson, ...] = ()

        if person_entries:
            ordered_person_entries = _order_person_entries(
                person_entries,
                runtime=runtime,
                config=config,
            )
            primary_person_box = ordered_person_entries[0][2]
            primary_zone = _safe_zone_from_center(primary_person_box.center_x)
            person_near_device = (
                primary_person_box.area >= _read_person_near_area_threshold(config)
                or primary_person_box.height >= _read_person_near_height_threshold(config)
            )
            visible_persons = tuple(
                AICameraVisiblePerson(
                    box=box,
                    zone=_safe_zone_from_center(box.center_x),
                    confidence=score,
                )
                for score, _area, box in ordered_person_entries
            )
            runtime[_RUNTIME_PRIMARY_HINT_KEY] = (
                _coerce_finite_float(primary_person_box.center_x, default=0.0),
                _coerce_finite_float(primary_person_box.area, default=0.0),
            )
        else:
            runtime.pop(_RUNTIME_PRIMARY_HINT_KEY, None)

        hand_or_object_near_camera = any(
            detection.box.area >= _read_object_near_area_threshold(config)
            for detection in object_detections
        )
    except Exception as exc:
        _log_warning_rate_limited(
            "ai_camera_detection_parse_failed",
            exc_info=True,
        )
        raise RuntimeError("detection_parse_failed") from exc

    return DetectionResult(
        person_count=len(person_entries),
        primary_person_box=primary_person_box,
        primary_person_zone=primary_zone,
        visible_persons=visible_persons,
        person_near_device=person_near_device,
        hand_or_object_near_camera=hand_or_object_near_camera,
        objects=tuple(object_detections),
    )


def _empty_detection_result() -> DetectionResult:
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


def _prepare_detections(
    *,
    session: Any,
    outputs: Any,
    output_count: int,
    intrinsics: _DetectionIntrinsics,
    config: AICameraAdapterConfig,
) -> tuple[_PreparedDetection, ...]:
    """Parse detector outputs into a backend-agnostic detection list."""

    if intrinsics.postprocess == "nanodet":
        return _prepare_nanodet_detections(
            outputs=outputs,
            output_count=output_count,
            intrinsics=intrinsics,
            config=config,
        )

    if output_count < _REQUIRED_STANDARD_OUTPUTS:
        _log_warning_rate_limited("ai_camera_detection_outputs_incomplete")
        raise RuntimeError("detection_outputs_incomplete")

    box_entries = _unwrap_batch_entries(outputs[0], expected_unbatched_depth=2)
    score_entries = _unwrap_batch_entries(outputs[1], expected_unbatched_depth=1)
    class_entries = _unwrap_batch_entries(outputs[2], expected_unbatched_depth=1)

    max_entries = min(
        _read_max_detections(config),
        _safe_len(box_entries),
        _safe_len(score_entries),
        _safe_len(class_entries),
    )

    if max_entries <= 0:
        return ()

    raw_count: int | None = None
    if output_count > _OPTIONAL_COUNT_OUTPUT_INDEX:
        raw_count = _extract_detection_count(outputs[_OPTIONAL_COUNT_OUTPUT_INDEX])

    count = max_entries if raw_count is None else min(max_entries, raw_count)
    if count <= 0:
        return ()

    detections: list[_PreparedDetection] = []
    for index in range(count):
        score = _coerce_probability(score_entries[index], default=0.0)
        if score <= 0.0:
            continue

        class_index = _coerce_class_index(class_entries[index])
        if class_index is None:
            continue

        prepared_box = _prepare_standard_box_entry(
            box_entries[index],
            intrinsics=intrinsics,
        )
        if prepared_box is None:
            continue

        detections.append(
            _PreparedDetection(
                box=prepared_box,
                score=score,
                class_index=class_index,
            )
        )
    return tuple(detections)


def _prepare_nanodet_detections(
    *,
    outputs: Any,
    output_count: int,
    intrinsics: _DetectionIntrinsics,
    config: AICameraAdapterConfig,
) -> tuple[_PreparedDetection, ...]:
    """Run Picamera2's NanoDet post-processing when required by network intrinsics."""

    if output_count <= 0:
        _log_warning_rate_limited("ai_camera_detection_outputs_incomplete")
        raise RuntimeError("detection_outputs_incomplete")

    postprocess_nanodet_detection, scale_boxes = _load_nanodet_postprocess()
    input_width = intrinsics.input_width
    input_height = intrinsics.input_height
    if input_width is None or input_height is None or input_width <= 0 or input_height <= 0:
        _log_warning_rate_limited("ai_camera_detection_input_size_missing")
        raise RuntimeError("detection_input_size_missing")

    raw_boxes, raw_scores, raw_classes = postprocess_nanodet_detection(
        outputs=outputs[0],
        conf=_read_min_detection_confidence(config),
        iou_thres=_read_nanodet_iou_threshold(config),
        max_out_dets=_read_max_detections(config),
    )[0]

    scaled_boxes = scale_boxes(raw_boxes, 1, 1, input_height, input_width, False, False)

    detections: list[_PreparedDetection] = []
    max_entries = min(
        _read_max_detections(config),
        _safe_len(scaled_boxes),
        _safe_len(raw_scores),
        _safe_len(raw_classes),
    )
    for index in range(max_entries):
        score = _coerce_probability(raw_scores[index], default=0.0)
        if score <= 0.0:
            continue

        class_index = _coerce_class_index(raw_classes[index])
        if class_index is None:
            continue

        prepared_box = _prepare_nanodet_box_entry(scaled_boxes[index])
        if prepared_box is None:
            continue

        detections.append(
            _PreparedDetection(
                box=prepared_box,
                score=score,
                class_index=class_index,
            )
        )
    return tuple(detections)


def _load_nanodet_postprocess() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Import Picamera2 NanoDet helpers lazily to keep import-time coupling low."""

    try:
        from picamera2.devices.imx500 import postprocess_nanodet_detection
        from picamera2.devices.imx500.postprocess import scale_boxes
    except Exception as exc:
        raise RuntimeError("nanodet_postprocess_unavailable") from exc
    return postprocess_nanodet_detection, scale_boxes


def _read_detection_intrinsics(
    *,
    session: Any,
    config: AICameraAdapterConfig,
) -> _DetectionIntrinsics:
    """Read parsing hints from the current IMX500 session with config fallbacks."""

    network_intrinsics = getattr(getattr(session, "imx500", None), "network_intrinsics", None)

    labels_value = getattr(network_intrinsics, "labels", None)
    if labels_value is None:
        labels_value = (
            getattr(config, "detection_labels", None)
            or getattr(config, "object_detection_labels", None)
            or ()
        )
    labels = tuple(labels_value or ())

    bbox_order = _read_override(
        network_intrinsics,
        config,
        names=("detection_bbox_order", "bbox_order"),
        attr_name="bbox_order",
        default="yx",
    )
    if bbox_order not in {"xy", "yx"}:
        bbox_order = "yx"

    input_size = _read_input_size(session, network_intrinsics)

    return _DetectionIntrinsics(
        postprocess=_read_override(
            network_intrinsics,
            config,
            names=("detection_postprocess", "postprocess"),
            attr_name="postprocess",
            default="",
        ),
        bbox_normalization=bool(
            _read_override(
                network_intrinsics,
                config,
                names=("detection_bbox_normalization", "bbox_normalization"),
                attr_name="bbox_normalization",
                default=False,
            )
        ),
        bbox_order=bbox_order,
        ignore_undefined_labels=bool(
            _read_override(
                network_intrinsics,
                config,
                names=("detection_ignore_undefined_labels", "ignore_dash_labels"),
                attr_name="ignore_dash_labels",
                default=False,
            )
        ),
        input_width=input_size[0],
        input_height=input_size[1],
        labels=labels,
    )


def _read_input_size(session: Any, network_intrinsics: Any) -> tuple[int | None, int | None]:
    """Read input tensor size from the session when available."""

    get_input_size = getattr(getattr(session, "imx500", None), "get_input_size", None)
    if callable(get_input_size):
        try:
            width, height = get_input_size()
            return _coerce_positive_int(width), _coerce_positive_int(height)
        except Exception:
            return None, None
    return None, None


def _prepare_standard_box_entry(
    value: Any,
    *,
    intrinsics: _DetectionIntrinsics,
) -> tuple[float, float, float, float] | None:
    """Convert one detector box into normalized (y0, x0, y1, x1) coordinates."""

    coords = _extract_box_coordinates(value)
    if coords is None:
        return None

    y0, x0, y1, x1 = coords
    if intrinsics.bbox_order == "xy":
        x0, y0, x1, y1 = y0, x0, y1, x1

    if intrinsics.bbox_normalization:
        y0, x0, y1, x1 = _normalize_box_by_input_size(
            y0=y0,
            x0=x0,
            y1=y1,
            x1=x1,
            input_width=intrinsics.input_width,
            input_height=intrinsics.input_height,
        )

    return _sanitize_box_coords((y0, x0, y1, x1))


def _prepare_nanodet_box_entry(value: Any) -> tuple[float, float, float, float] | None:
    """Convert one NanoDet post-processed box into normalized (y0, x0, y1, x1) coordinates."""

    coords = _extract_box_coordinates(value)
    if coords is None:
        return None
    return _sanitize_box_coords(coords)


def _extract_box_coordinates(value: Any) -> tuple[float, float, float, float] | None:
    """Read the first four finite coordinates from one tensor entry."""

    try:
        c0 = _coerce_finite_float(value[0], default=math.nan)
        c1 = _coerce_finite_float(value[1], default=math.nan)
        c2 = _coerce_finite_float(value[2], default=math.nan)
        c3 = _coerce_finite_float(value[3], default=math.nan)
    except Exception:
        return None

    coords = (c0, c1, c2, c3)
    if not all(math.isfinite(coord) for coord in coords):
        return None
    return coords


def _normalize_box_by_input_size(
    *,
    y0: float,
    x0: float,
    y1: float,
    x1: float,
    input_width: int | None,
    input_height: int | None,
) -> tuple[float, float, float, float]:
    """Normalize pixel-space boxes against the model input tensor dimensions."""

    if not input_width or not input_height:
        raise RuntimeError("detection_input_size_missing")
    return (
        y0 / float(input_height),
        x0 / float(input_width),
        y1 / float(input_height),
        x1 / float(input_width),
    )


def _sanitize_box_coords(coords: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    """Clamp and validate one normalized box."""

    y0, x0, y1, x1 = coords
    if not all(math.isfinite(value) for value in coords):
        return None

    y0 = max(0.0, min(1.0, y0))
    x0 = max(0.0, min(1.0, x0))
    y1 = max(0.0, min(1.0, y1))
    x1 = max(0.0, min(1.0, x1))

    if y1 <= y0 or x1 <= x0:
        return None
    return (y0, x0, y1, x1)


def _label_for_class_index(labels: tuple[object, ...], class_index: int) -> object:
    """Resolve one class index against the active label list."""

    if 0 <= class_index < len(labels):
        return labels[class_index]
    return f"class_{class_index}"


def _should_skip_label(normalized_label: str, *, ignore_undefined_labels: bool) -> bool:
    """Return True when the label should be excluded from downstream contract output."""

    if normalized_label in _ALWAYS_UNDEFINED_LABELS:
        return True
    if ignore_undefined_labels and normalized_label in _OPTIONAL_UNDEFINED_LABELS:
        return True
    return False


def _order_person_entries(
    person_entries: list[tuple[float, float, AICameraBox]],
    *,
    runtime: dict[str, Any],
    config: AICameraAdapterConfig,
) -> list[tuple[float, float, AICameraBox]]:
    """Order persons for output while reducing primary-target flapping."""

    if len(person_entries) <= 1:
        return sorted(person_entries, key=lambda item: (item[0], item[1]), reverse=True)

    sorted_entries = sorted(person_entries, key=lambda item: (item[0], item[1]), reverse=True)
    lead_score = sorted_entries[0][0]
    runner_up_score = sorted_entries[1][0]
    if (lead_score - runner_up_score) > _read_primary_stability_window(config):
        return sorted_entries

    hint = runtime.get(_RUNTIME_PRIMARY_HINT_KEY)
    if (
        not isinstance(hint, tuple)
        or len(hint) != 2
        or not _is_finite_number(hint[0])
        or not _is_finite_number(hint[1])
    ):
        return sorted_entries

    previous_center_x = float(hint[0])
    previous_area = float(hint[1])

    def rank_key(item: tuple[float, float, AICameraBox]) -> tuple[float, float, float]:
        score, area, box = item
        center_x = _coerce_finite_float(getattr(box, "center_x", None), default=previous_center_x)
        center_distance = abs(center_x - previous_center_x)
        area_delta = abs(area - previous_area) / max(previous_area, 1e-6)
        stability_bonus = (
            max(0.0, 1.0 - center_distance) * _read_primary_center_bias(config)
            + max(0.0, 1.0 - min(1.0, area_delta)) * _read_primary_area_bias(config)
        )
        return (score + stability_bonus, score, area)

    return sorted(person_entries, key=rank_key, reverse=True)


def _unwrap_batch_entries(value: Any, *, expected_unbatched_depth: int) -> Any:
    """Return the first batch slice when the tensor includes a batch dimension."""

    if _safe_len(value) <= 0:
        return ()
    if (
        _safe_len(value) == 1
        and _nesting_depth(value, max_depth=expected_unbatched_depth + 1) >= expected_unbatched_depth + 1
    ):
        try:
            return value[0]
        except (IndexError, KeyError, TypeError):
            return ()
    return value


def _nesting_depth(value: Any, *, max_depth: int) -> int:
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


def _extract_detection_count(value: Any) -> int | None:
    """Read one model-reported detection count when available."""

    current = value
    for _ in range(2):
        if _safe_len(current) <= 0:
            break
        try:
            current = current[0]
        except (IndexError, KeyError, TypeError):
            break
    count = _coerce_class_index(current)
    if count is None:
        return None
    return min(_MAX_DETECTIONS, count)


def _coerce_class_index(value: object) -> int | None:
    """Convert one model class identifier into a non-negative integer or None."""

    number = _coerce_finite_float(value, default=math.nan)
    if not math.isfinite(number) or number < 0.0:
        return None
    rounded = round(number)
    if abs(number - rounded) > 1e-3:
        return None
    return int(rounded)


def _coerce_positive_int(value: object) -> int | None:
    """Convert one value to a positive integer or None."""

    number = _coerce_finite_float(value, default=math.nan)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return int(number)


def _coerce_probability(value: object, *, default: float = 0.0) -> float:
    """Convert one model confidence value into the inclusive range [0.0, 1.0]."""

    number = _coerce_finite_float(value, default=default)
    if number <= 0.0:
        return 0.0
    if number >= 1.0:
        return 1.0
    return number


def _coerce_finite_float(value: object, *, default: float) -> float:
    """Convert one value to a finite float or fall back to a default."""

    try:
        number = _coerce_float(value, default=default)
    except Exception:
        return default
    if not math.isfinite(number):
        return default
    return number


def _safe_box_from_detection(value: Any) -> AICameraBox | None:
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


def _safe_zone_from_center(value: object) -> AICameraZone:
    """Map one horizontal center value to a zone without raising."""

    if not _is_finite_number(value):
        return AICameraZone.UNKNOWN
    try:
        return zone_from_center(float(value))
    except Exception:
        return AICameraZone.UNKNOWN


def _safe_len(value: Any) -> int:
    """Return len(value) or zero for scalar / invalid containers."""

    try:
        return len(value)
    except TypeError:
        return 0


def _is_finite_number(value: object) -> bool:
    """Return True when the value can be represented as a finite float."""

    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _normalize_label(value: object) -> str:
    """Normalize one object label to an inspectable token."""

    if isinstance(value, (bytes, bytearray)):
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


def _read_override(
    network_intrinsics: Any,
    config: AICameraAdapterConfig,
    *,
    names: tuple[str, ...],
    attr_name: str,
    default: Any,
) -> Any:
    """Read one setting from intrinsics first, then config aliases, then default."""

    intrinsics_value = getattr(network_intrinsics, attr_name, None)
    if intrinsics_value is not None:
        return intrinsics_value
    for name in names:
        config_value = getattr(config, name, None)
        if config_value is not None:
            return config_value
    return default


def _read_person_labels(config: AICameraAdapterConfig) -> frozenset[str]:
    """Read configured person-class aliases with a safe default."""

    configured = (
        getattr(config, "person_labels", None)
        or getattr(config, "person_detection_labels", None)
        or _DEFAULT_PERSON_LABELS
    )
    labels = {
        _normalize_label(label)
        for label in configured
        if _normalize_label(label)
    }
    return frozenset(labels or _DEFAULT_PERSON_LABELS)


def _read_person_confidence_threshold(config: AICameraAdapterConfig) -> float:
    return _coerce_finite_float(
        getattr(config, "person_confidence_threshold", 0.0),
        default=0.0,
    )


def _read_object_confidence_threshold(config: AICameraAdapterConfig) -> float:
    return _coerce_finite_float(
        getattr(config, "object_confidence_threshold", 0.0),
        default=0.0,
    )


def _read_min_detection_confidence(config: AICameraAdapterConfig) -> float:
    """Read the broadest confidence threshold for detector-side post-processing."""

    return min(
        _read_person_confidence_threshold(config),
        _read_object_confidence_threshold(config),
    )


def _read_person_near_area_threshold(config: AICameraAdapterConfig) -> float:
    return _coerce_finite_float(
        getattr(config, "person_near_area_threshold", math.inf),
        default=math.inf,
    )


def _read_person_near_height_threshold(config: AICameraAdapterConfig) -> float:
    return _coerce_finite_float(
        getattr(config, "person_near_height_threshold", math.inf),
        default=math.inf,
    )


def _read_object_near_area_threshold(config: AICameraAdapterConfig) -> float:
    return _coerce_finite_float(
        getattr(config, "object_near_area_threshold", math.inf),
        default=math.inf,
    )


def _read_max_detections(config: AICameraAdapterConfig) -> int:
    configured = (
        getattr(config, "max_detections", None)
        or getattr(config, "detection_max_detections", None)
        or getattr(config, "object_max_detections", None)
    )
    value = _coerce_positive_int(configured)
    if value is None:
        return _MAX_DETECTIONS
    return min(_MAX_DETECTIONS, value)


def _read_nanodet_iou_threshold(config: AICameraAdapterConfig) -> float:
    configured = (
        getattr(config, "detection_iou_threshold", None)
        or getattr(config, "object_detection_iou_threshold", None)
        or getattr(config, "detection_iou", None)
        or getattr(config, "iou_threshold", None)
        or _DEFAULT_NANODET_IOU_THRESHOLD
    )
    value = _coerce_finite_float(
        configured,
        default=_DEFAULT_NANODET_IOU_THRESHOLD,
    )
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _read_primary_stability_window(config: AICameraAdapterConfig) -> float:
    value = _coerce_finite_float(
        getattr(config, "primary_person_stability_window", _DEFAULT_PRIMARY_STABILITY_WINDOW),
        default=_DEFAULT_PRIMARY_STABILITY_WINDOW,
    )
    return max(0.0, value)


def _read_primary_center_bias(config: AICameraAdapterConfig) -> float:
    value = _coerce_finite_float(
        getattr(config, "primary_person_center_bias", _DEFAULT_PRIMARY_CENTER_BIAS),
        default=_DEFAULT_PRIMARY_CENTER_BIAS,
    )
    return max(0.0, value)


def _read_primary_area_bias(config: AICameraAdapterConfig) -> float:
    value = _coerce_finite_float(
        getattr(config, "primary_person_area_bias", _DEFAULT_PRIMARY_AREA_BIAS),
        default=_DEFAULT_PRIMARY_AREA_BIAS,
    )
    return max(0.0, value)


def _log_warning_rate_limited(event_name: str, *, exc_info: bool = False) -> None:
    """Emit one warning at most once per time window and summarize suppressed repeats."""

    now = time.monotonic()
    suppressed_count = 0
    should_emit = False

    with _LOG_STATE_LOCK:
        last_emitted_at, suppressed = _LOG_STATE.get(event_name, (0.0, 0))
        if (now - last_emitted_at) >= _LOG_RATE_LIMIT_SECONDS:
            should_emit = True
            suppressed_count = suppressed
            _LOG_STATE[event_name] = (now, 0)
        else:
            _LOG_STATE[event_name] = (last_emitted_at, suppressed + 1)

    if not should_emit:
        return

    if suppressed_count > 0:
        LOGGER.warning(
            "%s suppressed_repeats=%d window_seconds=%.1f",
            event_name,
            suppressed_count,
            _LOG_RATE_LIMIT_SECONDS,
            exc_info=exc_info,
        )
        return
    LOGGER.warning("%s", event_name, exc_info=exc_info)


__all__ = ["DetectionResult", "capture_detection"]