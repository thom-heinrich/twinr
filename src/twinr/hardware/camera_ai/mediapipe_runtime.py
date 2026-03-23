"""Own the lazy MediaPipe task instances and timestamp discipline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import errno
import importlib
import math
import os
import stat
import threading

from .config import MediaPipeVisionConfig
from .fine_hand_gestures import builtin_gesture_category_denylist

_MEDIAPIPE_IMAGE_DTYPES = {"uint8", "uint16", "float32"}


def normalize_image_data(frame_rgb: Any) -> Any:
    """Return one MediaPipe-compatible image buffer without forcing numpy imports.

    The local Pi camera path can hand MediaPipe either full-frame RGB arrays or
    ROI crop views. Full frames are typically already C-contiguous, while ROI
    slices inherit the parent strides and can fail native MediaPipe image
    construction. This helper preserves non-array test doubles untouched,
    coerces array-like inputs to a supported dtype when the object exposes
    ``astype()``, and forces a C-order copy only when the incoming buffer
    explicitly reports a non-contiguous layout.
    """

    if getattr(frame_rgb, "shape", None) is None or getattr(frame_rgb, "dtype", None) is None:
        return frame_rgb

    image_data = frame_rgb
    dtype_name = str(getattr(image_data, "dtype", "")).strip().lower()
    if dtype_name not in _MEDIAPIPE_IMAGE_DTYPES:
        astype = getattr(image_data, "astype", None)
        if callable(astype):
            try:
                image_data = astype("uint8", copy=False)
            except TypeError:
                image_data = astype("uint8")

    contiguous = getattr(getattr(image_data, "flags", None), "c_contiguous", None)
    if contiguous is False:
        copy_fn = getattr(image_data, "copy", None)
        if callable(copy_fn):
            try:
                image_data = copy_fn(order="C")
            except TypeError:
                image_data = copy_fn()

    return image_data


class MediaPipeTaskRuntime:
    """Manage lazy MediaPipe imports and reusable task instances."""

    def __init__(self, *, config: MediaPipeVisionConfig) -> None:
        """Initialize one lazy MediaPipe task runtime."""

        self.config = config
        self._pose_landmarker: Any | None = None
        self._gesture_recognizer: Any | None = None
        self._roi_gesture_recognizer: Any | None = None
        self._custom_gesture_recognizer: Any | None = None
        self._custom_roi_gesture_recognizer: Any | None = None
        self._live_gesture_recognizer: Any | None = None
        self._live_custom_gesture_recognizer: Any | None = None
        self._last_timestamp_ms = 0
        self._lock = threading.RLock()  # AUDIT-FIX(#6): Serialize shared mutable state for init/close/timestamp paths.

    def close(self) -> None:
        """Close active MediaPipe task instances when supported."""

        first_error: Exception | None = None
        with self._lock:
            try:
                for instance in (
                    self._pose_landmarker,
                    self._gesture_recognizer,
                    self._roi_gesture_recognizer,
                    self._custom_gesture_recognizer,
                    self._custom_roi_gesture_recognizer,
                    self._live_gesture_recognizer,
                    self._live_custom_gesture_recognizer,
                ):
                    if instance is None:
                        continue
                    close_fn = getattr(instance, "close", None)
                    if callable(close_fn):
                        try:
                            close_fn()
                        except Exception as exc:
                            if first_error is None:
                                first_error = exc  # AUDIT-FIX(#3): Keep closing remaining instances after the first close() failure.
            finally:
                self._pose_landmarker = None  # AUDIT-FIX(#3): Always clear stale instances, even when close() raises.
                self._gesture_recognizer = None
                self._roi_gesture_recognizer = None
                self._custom_gesture_recognizer = None
                self._custom_roi_gesture_recognizer = None
                self._live_gesture_recognizer = None
                self._live_custom_gesture_recognizer = None
                self._last_timestamp_ms = 0

        if first_error is not None:
            raise first_error

    def load_runtime(self) -> dict[str, Any]:
        """Import the minimum MediaPipe runtime objects lazily."""

        try:
            mp = importlib.import_module("mediapipe")
            tasks_python = importlib.import_module("mediapipe.tasks.python")
            classifier_options = importlib.import_module(
                "mediapipe.tasks.python.components.processors.classifier_options"
            )
            vision = importlib.import_module("mediapipe.tasks.python.vision")
            base_options = getattr(tasks_python, "BaseOptions")  # AUDIT-FIX(#4): Validate required runtime symbols during import.
            getattr(mp, "Image")
            image_format = getattr(mp, "ImageFormat")
            getattr(image_format, "SRGB")
            getattr(vision, "RunningMode")
            getattr(classifier_options, "ClassifierOptions")
        except Exception as exc:  # pragma: no cover - depends on local environment.
            raise RuntimeError("mediapipe_unavailable") from exc
        return {
            "mp": mp,
            "BaseOptions": base_options,
            "ClassifierOptions": classifier_options.ClassifierOptions,
            "vision": vision,
        }

    def build_image(self, runtime: dict[str, Any], *, frame_rgb: Any) -> Any:
        """Wrap one RGB array-like frame in a MediaPipe image container."""

        if frame_rgb is None:
            raise ValueError("mediapipe_frame_missing")  # AUDIT-FIX(#7): Fail fast on missing camera frame payloads.
        mp = self._require_runtime_entry(runtime, "mp")
        image_format = self._require_attribute(mp, "ImageFormat", error_code="mediapipe_runtime_incomplete")
        image_ctor = self._require_attribute(mp, "Image", error_code="mediapipe_runtime_incomplete")
        srgb_format = self._require_attribute(image_format, "SRGB", error_code="mediapipe_runtime_incomplete")
        try:
            return image_ctor(
                image_format=srgb_format,
                data=normalize_image_data(frame_rgb),
            )
        except Exception as exc:
            raise RuntimeError("mediapipe_image_build_failed") from exc  # AUDIT-FIX(#7): Surface stable image-build errors.

    def timestamp_ms(self, observed_at: float) -> int:
        """Return a monotonically increasing task timestamp in milliseconds."""

        timestamp_ms = self._coerce_observed_at_to_ms(observed_at)
        with self._lock:
            if timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = timestamp_ms
            return timestamp_ms

    def reserve_timestamp(self, timestamp_ms: int) -> None:
        """Advance the internal clock after multi-candidate gesture inference."""

        reserved_timestamp_ms = self._coerce_reserved_timestamp_ms(timestamp_ms)
        with self._lock:
            self._last_timestamp_ms = max(self._last_timestamp_ms, reserved_timestamp_ms)

    def ensure_pose_landmarker(self, runtime: dict[str, Any]) -> Any:
        """Reuse or create the configured pose landmarker."""

        with self._lock:
            if self._pose_landmarker is not None:
                return self._pose_landmarker

            model_path, model_bytes = self._load_model_asset(
                self.config.pose_model_path,
                missing_code="mediapipe_pose_model_missing",
            )
            vision = self._require_runtime_entry(runtime, "vision")
            running_mode = self._require_attribute(vision, "RunningMode", error_code="mediapipe_runtime_incomplete")
            pose_landmarker_options = self._require_attribute(
                vision,
                "PoseLandmarkerOptions",
                error_code="mediapipe_runtime_incomplete",
            )
            pose_landmarker = self._require_attribute(
                vision,
                "PoseLandmarker",
                error_code="mediapipe_runtime_incomplete",
            )

            options = pose_landmarker_options(
                base_options=self._build_base_options(
                    runtime,
                    model_path=model_path,
                    model_bytes=model_bytes,
                ),
                running_mode=running_mode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self._validate_probability(
                    self.config.min_pose_detection_confidence,
                    name="min_pose_detection_confidence",
                ),
                min_pose_presence_confidence=self._validate_probability(
                    self.config.min_pose_presence_confidence,
                    name="min_pose_presence_confidence",
                ),
                min_tracking_confidence=self._validate_probability(
                    self.config.min_pose_tracking_confidence,
                    name="min_pose_tracking_confidence",
                ),
                output_segmentation_masks=False,
            )
            self._pose_landmarker = pose_landmarker.create_from_options(options)
            return self._pose_landmarker

    def ensure_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Reuse or create the configured built-in gesture recognizer."""

        with self._lock:
            if self._gesture_recognizer is not None:
                return self._gesture_recognizer
            self._gesture_recognizer = self._create_gesture_recognizer(
                runtime=runtime,
                model_path=self.config.gesture_model_path,
                missing_code="mediapipe_gesture_model_missing",
                running_mode_name="VIDEO",
                model_family="builtin",
            )
            return self._gesture_recognizer

    def ensure_roi_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Reuse or create the built-in image-mode recognizer for ROI fallback."""

        with self._lock:
            if self._roi_gesture_recognizer is not None:
                return self._roi_gesture_recognizer
            self._roi_gesture_recognizer = self._create_gesture_recognizer(
                runtime=runtime,
                model_path=self.config.gesture_model_path,
                missing_code="mediapipe_gesture_model_missing",
                running_mode_name="IMAGE",
                model_family="builtin",
            )
            return self._roi_gesture_recognizer

    def ensure_custom_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Reuse or create the configured custom gesture recognizer."""

        with self._lock:
            if self._custom_gesture_recognizer is not None:
                return self._custom_gesture_recognizer
            self._custom_gesture_recognizer = self._create_gesture_recognizer(
                runtime=runtime,
                model_path=self.config.custom_gesture_model_path,
                missing_code="mediapipe_custom_gesture_model_missing",
                running_mode_name="VIDEO",
                model_family="custom",
            )
            return self._custom_gesture_recognizer

    def ensure_custom_roi_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Reuse or create the custom image-mode recognizer for ROI fallback."""

        with self._lock:
            if self._custom_roi_gesture_recognizer is not None:
                return self._custom_roi_gesture_recognizer
            self._custom_roi_gesture_recognizer = self._create_gesture_recognizer(
                runtime=runtime,
                model_path=self.config.custom_gesture_model_path,
                missing_code="mediapipe_custom_gesture_model_missing",
                running_mode_name="IMAGE",
                model_family="custom",
            )
            return self._custom_roi_gesture_recognizer

    def ensure_live_gesture_recognizer(
        self,
        runtime: dict[str, Any],
        *,
        result_callback: Any,
        num_hands_override: int | None = None,
    ) -> Any:
        """Reuse or create the built-in live-stream recognizer for fast gesture ack."""

        with self._lock:
            if self._live_gesture_recognizer is not None:
                return self._live_gesture_recognizer
            self._live_gesture_recognizer = self._create_gesture_recognizer(
                runtime=runtime,
                model_path=self.config.gesture_model_path,
                missing_code="mediapipe_gesture_model_missing",
                running_mode_name="LIVE_STREAM",
                model_family="builtin",
                result_callback=result_callback,
                num_hands_override=num_hands_override,
            )
            return self._live_gesture_recognizer

    def ensure_live_custom_gesture_recognizer(
        self,
        runtime: dict[str, Any],
        *,
        result_callback: Any,
    ) -> Any:
        """Reuse or create the custom live-stream recognizer for fast gesture ack."""

        with self._lock:
            if self._live_custom_gesture_recognizer is not None:
                return self._live_custom_gesture_recognizer
            self._live_custom_gesture_recognizer = self._create_gesture_recognizer(
                runtime=runtime,
                model_path=self.config.custom_gesture_model_path,
                missing_code="mediapipe_custom_gesture_model_missing",
                running_mode_name="LIVE_STREAM",
                model_family="custom",
                result_callback=result_callback,
            )
            return self._live_custom_gesture_recognizer

    def _create_gesture_recognizer(
        self,
        *,
        runtime: dict[str, Any],
        model_path: str | None,
        missing_code: str,
        running_mode_name: str,
        model_family: str,
        result_callback: Any | None = None,
        num_hands_override: int | None = None,
    ) -> Any:
        """Create one gesture recognizer with the requested MediaPipe running mode."""

        resolved_model_path, model_bytes = self._load_model_asset(
            model_path,
            missing_code=missing_code,
        )
        vision = self._require_runtime_entry(runtime, "vision")
        running_mode = self._require_attribute(vision, "RunningMode", error_code="mediapipe_runtime_incomplete")
        gesture_recognizer_options = self._require_attribute(
            vision,
            "GestureRecognizerOptions",
            error_code="mediapipe_runtime_incomplete",
        )
        gesture_recognizer = self._require_attribute(
            vision,
            "GestureRecognizer",
            error_code="mediapipe_runtime_incomplete",
        )
        classifier_options = self._require_runtime_entry(runtime, "ClassifierOptions")
        selected_running_mode = self._require_attribute(
            running_mode,
            running_mode_name,
            error_code="mediapipe_runtime_incomplete",
        )
        options_kwargs: dict[str, Any] = {
            "base_options": self._build_base_options(
                runtime,
                model_path=resolved_model_path,
                model_bytes=model_bytes,
            ),
            "running_mode": selected_running_mode,
            "num_hands": self._validate_num_hands(
                self.config.num_hands if num_hands_override is None else num_hands_override
            ),
            "min_hand_detection_confidence": self._validate_probability(
                self.config.min_hand_detection_confidence,
                name="min_hand_detection_confidence",
            ),
            "min_hand_presence_confidence": self._validate_probability(
                self.config.min_hand_presence_confidence,
                name="min_hand_presence_confidence",
            ),
            "min_tracking_confidence": self._validate_probability(
                self.config.min_hand_tracking_confidence,
                name="min_hand_tracking_confidence",
            ),
            "result_callback": result_callback,
        }
        if model_family == "builtin":
            options_kwargs["canned_gesture_classifier_options"] = classifier_options(
                score_threshold=self._validate_probability(
                    self.config.builtin_gesture_min_score,
                    name="builtin_gesture_min_score",
                ),
                category_denylist=list(builtin_gesture_category_denylist()),
            )
        options = gesture_recognizer_options(**options_kwargs)
        return gesture_recognizer.create_from_options(options)

    def _require_runtime_entry(self, runtime: dict[str, Any], key: str) -> Any:
        """Return one required runtime entry or raise a stable runtime error."""

        try:
            value = runtime[key]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"mediapipe_runtime_incomplete:{key}") from exc  # AUDIT-FIX(#4): Avoid raw KeyError/TypeError from malformed runtime data.
        if value is None:
            raise RuntimeError(f"mediapipe_runtime_incomplete:{key}")  # AUDIT-FIX(#4): Reject None-valued runtime slots.
        return value

    def _require_attribute(self, obj: Any, attribute_name: str, *, error_code: str) -> Any:
        """Return one required attribute or raise a stable runtime error."""

        try:
            return getattr(obj, attribute_name)
        except AttributeError as exc:
            raise RuntimeError(f"{error_code}:{attribute_name}") from exc  # AUDIT-FIX(#4): Avoid raw AttributeError on runtime/API drift.

    def _coerce_finite_float(self, value: Any, *, error_code: str) -> float:
        """Parse one finite float value or raise a stable validation error."""

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(error_code) from exc  # AUDIT-FIX(#5): Reject non-numeric timestamp/config values.
        if not math.isfinite(numeric_value):
            raise ValueError(error_code)  # AUDIT-FIX(#5): Reject NaN/inf timestamp/config values.
        return numeric_value

    def _coerce_observed_at_to_ms(self, observed_at: Any) -> int:
        """Convert one observed-at second value to MediaPipe milliseconds."""

        observed_at_seconds = self._coerce_finite_float(
            observed_at,
            error_code="mediapipe_invalid_observed_at",
        )
        return max(1, int(round(observed_at_seconds * 1000.0)))

    def _coerce_reserved_timestamp_ms(self, timestamp_ms: Any) -> int:
        """Convert one reserved timestamp to a positive millisecond integer."""

        reserved_timestamp = self._coerce_finite_float(
            timestamp_ms,
            error_code="mediapipe_invalid_reserved_timestamp_ms",
        )
        return max(1, int(reserved_timestamp))

    def _validate_probability(self, value: Any, *, name: str) -> float:
        """Validate one probability-like config value."""

        probability = self._coerce_finite_float(
            value,
            error_code=f"mediapipe_invalid_config:{name}",
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"mediapipe_invalid_config:{name}")  # AUDIT-FIX(#8): Enforce valid confidence ranges before MediaPipe init.
        return probability

    def _validate_num_hands(self, value: Any) -> int:
        """Validate the configured hand count."""

        try:
            num_hands = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("mediapipe_invalid_config:num_hands") from exc  # AUDIT-FIX(#8): Reject non-integer hand-count config.
        if num_hands < 1:
            raise ValueError("mediapipe_invalid_config:num_hands")  # AUDIT-FIX(#8): Reject invalid hand counts before MediaPipe init.
        return num_hands

    def _load_model_asset(self, raw_path: str | None, *, missing_code: str) -> tuple[Path, bytes]:
        """Open one model asset securely and return the resolved path plus bytes."""

        normalized_path = "" if raw_path is None else str(raw_path).strip()
        if not normalized_path:
            raise FileNotFoundError(f"{missing_code}:<unset>")  # AUDIT-FIX(#2): Reject empty/unset model paths instead of resolving them to ".".

        candidate_path = Path(normalized_path).expanduser()
        open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)

        try:
            file_descriptor = os.open(str(candidate_path), open_flags)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{missing_code}:{candidate_path}") from exc
        except IsADirectoryError as exc:
            raise FileNotFoundError(f"{missing_code}:{candidate_path}") from exc
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise PermissionError(f"{missing_code}_symlink_forbidden:{candidate_path}") from exc  # AUDIT-FIX(#1): Reject symlink model targets.
            raise RuntimeError(f"{missing_code}_unreadable:{candidate_path}") from exc

        with os.fdopen(file_descriptor, "rb") as model_file:
            file_stat = os.fstat(model_file.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise FileNotFoundError(f"{missing_code}:{candidate_path}")  # AUDIT-FIX(#1): Reject non-regular files such as dirs/devices/FIFOs.
            model_bytes = model_file.read()

        resolved_path = Path(os.path.realpath(candidate_path))
        if not model_bytes:
            raise RuntimeError(f"{missing_code}_empty:{resolved_path}")  # AUDIT-FIX(#1): Fail fast on empty model payloads.

        return resolved_path, model_bytes

    def _build_base_options(self, runtime: dict[str, Any], *, model_path: Path, model_bytes: bytes) -> Any:
        """Create BaseOptions from validated model bytes, with a path fallback for compatibility."""

        BaseOptions = self._require_runtime_entry(runtime, "BaseOptions")
        try:
            return BaseOptions(model_asset_buffer=model_bytes)  # AUDIT-FIX(#1): Prefer validated in-memory bytes to avoid path re-open races.
        except TypeError:
            return BaseOptions(model_asset_path=str(model_path))


__all__ = ["MediaPipeTaskRuntime", "normalize_image_data"]
