# CHANGELOG: 2026-03-28
# BUG-1: Live recognizer caches now rebuild when the callback or num_hands override changes, preventing silent dispatch to stale callbacks or wrong multi-hand settings.
# BUG-2: Cached task instances now rebuild when model files or effective config values change, preventing stale models/thresholds in long-lived processes.
# BUG-3: Stable live callback cache keys now reuse recognizers across observe() ticks while still hot-swapping the active Python callback target.
# SEC-1: Model assets are size-capped before read and, on old BaseOptions fallbacks, are exposed through verified in-memory compatibility files instead of reopening the original path.
# IMP-1: Live-stream callbacks now default to a latest-wins background dispatcher so slow Python callbacks do not stall MediaPipe live streams.
# IMP-2: Model assets are cached and reused across IMAGE/VIDEO/LIVE_STREAM task instances, avoiding repeated SD-card reads and duplicate memory churn on Raspberry Pi.
# IMP-3: Optional 2026-era classifier/delegate hooks are supported via config or env without breaking existing config classes.
"""Own the lazy MediaPipe task instances and timestamp discipline."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import errno
import importlib
import math
import os
import stat
import tempfile
import threading
import traceback

from .config import MediaPipeVisionConfig
from .fine_hand_gestures import builtin_gesture_category_denylist

_MEDIAPIPE_IMAGE_DTYPES = {"uint8", "uint16", "float32"}
_DEFAULT_MAX_MODEL_ASSET_BYTES = 128 * 1024 * 1024
_DEFAULT_LIVE_CALLBACK_QUEUE_SIZE = 1
_DEFAULT_LIVE_CALLBACK_SHUTDOWN_TIMEOUT_S = 1.0
MEDIAPIPE_LIVE_CALLBACK_CACHE_KEY_ATTR = "__twinr_mediapipe_live_callback_cache_key__"
_INSTANCE_SLOTS = (
    "_pose_landmarker",
    "_gesture_recognizer",
    "_roi_gesture_recognizer",
    "_custom_gesture_recognizer",
    "_custom_roi_gesture_recognizer",
    "_live_gesture_recognizer",
    "_live_custom_gesture_recognizer",
)


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


@dataclass
class _ModelAssetRecord:
    """One verified model asset plus compatibility handles for old runtimes."""

    requested_path: Path
    resolved_path: Path
    file_key: tuple[int, int, int, int]
    model_bytes: bytes
    compat_fd: int | None = None
    compat_path: str | None = None
    compat_temp_path: str | None = None

    def close(self) -> None:
        """Release compatibility resources, if any."""

        temp_path = self.compat_temp_path
        fd = self.compat_fd
        self.compat_temp_path = None
        self.compat_path = None
        self.compat_fd = None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            except OSError:
                pass


class _LatestWinsCallbackDispatcher:
    """Run live-stream callbacks off the MediaPipe thread with bounded backlog."""

    def __init__(self, callback_proxy: "_MutableLiveCallbackProxy", *, queue_size: int, name: str) -> None:
        self._callback_proxy = callback_proxy
        self._queue_size = max(1, int(queue_size))
        self._queue: deque[tuple[Any, Any, int]] = deque(maxlen=self._queue_size)
        self._condition = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    @property
    def spec(self) -> tuple[Any, int]:
        """Return the dispatcher identity relevant to cache invalidation."""

        return ((*self._callback_proxy.spec,), self._queue_size)

    def enqueue(self, result: Any, output_image: Any, timestamp_ms: int) -> None:
        """Queue the newest callback payload without blocking the caller."""

        with self._condition:
            if self._closed:
                return
            self._queue.append((result, output_image, timestamp_ms))
            self._condition.notify()

    def close(self, *, timeout_s: float) -> None:
        """Stop the dispatcher without waiting indefinitely on user callbacks."""

        with self._condition:
            self._closed = True
            self._queue.clear()
            self._condition.notify_all()
        if timeout_s > 0.0:
            self._thread.join(timeout_s)

    def _run(self) -> None:
        """Invoke the user callback on a dedicated worker thread."""

        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if not self._queue and self._closed:
                    return
                result, output_image, timestamp_ms = self._queue.popleft()

            try:
                self._callback_proxy(result, output_image, timestamp_ms)
            except Exception as exc:
                # BREAKING: callback failures are isolated from MediaPipe's live thread.
                # Keep the stream alive and surface the original callback traceback.
                traceback.print_exception(type(exc), exc, exc.__traceback__)


class _MutableLiveCallbackProxy:
    """Keep one stable callable identity while swapping the target callback."""

    def __init__(self, callback: Any, *, cache_key: Any) -> None:
        self._callback: Any | None = None
        self._cache_key = cache_key
        self._closed = False
        self._lock = threading.Lock()
        self.update_callback(callback)

    @property
    def spec(self) -> tuple[Any]:
        """Return the stable cache identity for one live callback target."""

        return (self._cache_key,)

    def update_callback(self, callback: Any) -> None:
        """Swap in the current Python callback without changing callable identity."""

        if callback is None:
            raise ValueError("mediapipe_live_result_callback_missing")
        if not callable(callback):
            raise ValueError("mediapipe_live_result_callback_invalid")
        with self._lock:
            self._callback = callback
            self._closed = False

    def close(self) -> None:
        """Fail closed so stale direct callbacks after teardown are ignored."""

        with self._lock:
            self._closed = True
            self._callback = None

    def __call__(self, result: Any, output_image: Any, timestamp_ms: int) -> None:
        """Invoke the latest callback target if this proxy is still active."""

        with self._lock:
            if self._closed or self._callback is None:
                return
            callback = self._callback
        callback(result, output_image, timestamp_ms)


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
        self._lock = threading.RLock()  # Serialize shared mutable state for init/close/timestamp/cache paths.
        self._runtime: dict[str, Any] | None = None
        self._instance_specs: dict[str, tuple[Any, ...]] = {}
        self._model_asset_cache: dict[str, _ModelAssetRecord] = {}
        self._live_callback_proxies: dict[str, _MutableLiveCallbackProxy] = {}
        self._live_callback_dispatchers: dict[str, _LatestWinsCallbackDispatcher] = {}

    def close(self) -> None:
        """Close active MediaPipe task instances when supported."""

        first_error: Exception | None = None
        with self._lock:
            try:
                for slot_name in _INSTANCE_SLOTS:
                    try:
                        self._close_instance_slot(slot_name)
                    except Exception as exc:
                        if first_error is None:
                            first_error = exc
                for cache_key, record in tuple(self._model_asset_cache.items()):
                    record.close()
                    self._model_asset_cache.pop(cache_key, None)
            finally:
                self._last_timestamp_ms = 0

        if first_error is not None:
            raise first_error

    def load_runtime(self) -> dict[str, Any]:
        """Import the minimum MediaPipe runtime objects lazily."""

        with self._lock:
            if self._runtime is not None:
                return self._runtime

        try:
            mp = importlib.import_module("mediapipe")
            tasks = getattr(mp, "tasks")
            base_options = getattr(tasks, "BaseOptions")
            vision = getattr(tasks, "vision")
            components = getattr(tasks, "components")
            processors = getattr(components, "processors")
            classifier_options = getattr(processors, "ClassifierOptions")
            getattr(mp, "Image")
            image_format = getattr(mp, "ImageFormat")
            getattr(image_format, "SRGB")
            getattr(vision, "RunningMode")
        except Exception as exc:  # pragma: no cover - depends on local environment.
            raise RuntimeError("mediapipe_unavailable") from exc

        runtime = {
            "mp": mp,
            "BaseOptions": base_options,
            "ClassifierOptions": classifier_options,
            "vision": vision,
        }
        with self._lock:
            self._runtime = runtime
            return runtime

    def build_image(self, runtime: dict[str, Any], *, frame_rgb: Any) -> Any:
        """Wrap one RGB array-like frame in a MediaPipe image container."""

        if frame_rgb is None:
            raise ValueError("mediapipe_frame_missing")
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
            raise RuntimeError("mediapipe_image_build_failed") from exc

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

        min_detection = self._validate_probability(
            self.config.min_pose_detection_confidence,
            name="min_pose_detection_confidence",
        )
        min_presence = self._validate_probability(
            self.config.min_pose_presence_confidence,
            name="min_pose_presence_confidence",
        )
        min_tracking = self._validate_probability(
            self.config.min_pose_tracking_confidence,
            name="min_pose_tracking_confidence",
        )
        asset_spec = self._model_asset_spec(
            self.config.pose_model_path,
            missing_code="mediapipe_pose_model_missing",
        )
        spec = (
            "pose",
            asset_spec,
            self._base_options_spec(),
            min_detection,
            min_presence,
            min_tracking,
        )

        with self._lock:
            if self._matches_cached_instance("_pose_landmarker", spec):
                return self._pose_landmarker

            self._close_instance_slot("_pose_landmarker")

            asset = self._load_model_asset(
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
                base_options=self._build_base_options(runtime, asset=asset),
                running_mode=running_mode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=min_detection,
                min_pose_presence_confidence=min_presence,
                min_tracking_confidence=min_tracking,
                output_segmentation_masks=False,
            )
            instance = pose_landmarker.create_from_options(options)
            self._pose_landmarker = instance
            self._instance_specs["_pose_landmarker"] = spec
            return instance

    def ensure_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Reuse or create the configured built-in gesture recognizer."""

        spec = self._gesture_instance_spec(
            model_path=self.config.gesture_model_path,
            missing_code="mediapipe_gesture_model_missing",
            running_mode_name="VIDEO",
            model_family="builtin",
            num_hands_override=None,
            result_callback=None,
        )
        with self._lock:
            if self._matches_cached_instance("_gesture_recognizer", spec):
                return self._gesture_recognizer
            self._close_instance_slot("_gesture_recognizer")
            self._gesture_recognizer = self._create_gesture_recognizer(
                runtime=runtime,
                model_path=self.config.gesture_model_path,
                missing_code="mediapipe_gesture_model_missing",
                running_mode_name="VIDEO",
                model_family="builtin",
            )
            self._instance_specs["_gesture_recognizer"] = spec
            return self._gesture_recognizer

    def ensure_roi_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Reuse or create the built-in image-mode recognizer for ROI fallback."""

        spec = self._gesture_instance_spec(
            model_path=self.config.gesture_model_path,
            missing_code="mediapipe_gesture_model_missing",
            running_mode_name="IMAGE",
            model_family="builtin",
            num_hands_override=None,
            result_callback=None,
        )
        with self._lock:
            if self._matches_cached_instance("_roi_gesture_recognizer", spec):
                return self._roi_gesture_recognizer
            self._close_instance_slot("_roi_gesture_recognizer")
            self._roi_gesture_recognizer = self._create_gesture_recognizer(
                runtime=runtime,
                model_path=self.config.gesture_model_path,
                missing_code="mediapipe_gesture_model_missing",
                running_mode_name="IMAGE",
                model_family="builtin",
            )
            self._instance_specs["_roi_gesture_recognizer"] = spec
            return self._roi_gesture_recognizer

    def ensure_custom_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Reuse or create the configured custom gesture recognizer."""

        spec = self._gesture_instance_spec(
            model_path=self.config.custom_gesture_model_path,
            missing_code="mediapipe_custom_gesture_model_missing",
            running_mode_name="VIDEO",
            model_family="custom",
            num_hands_override=None,
            result_callback=None,
        )
        with self._lock:
            if self._matches_cached_instance("_custom_gesture_recognizer", spec):
                return self._custom_gesture_recognizer
            self._close_instance_slot("_custom_gesture_recognizer")
            self._custom_gesture_recognizer = self._create_gesture_recognizer(
                runtime=runtime,
                model_path=self.config.custom_gesture_model_path,
                missing_code="mediapipe_custom_gesture_model_missing",
                running_mode_name="VIDEO",
                model_family="custom",
            )
            self._instance_specs["_custom_gesture_recognizer"] = spec
            return self._custom_gesture_recognizer

    def ensure_custom_roi_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Reuse or create the custom image-mode recognizer for ROI fallback."""

        spec = self._gesture_instance_spec(
            model_path=self.config.custom_gesture_model_path,
            missing_code="mediapipe_custom_gesture_model_missing",
            running_mode_name="IMAGE",
            model_family="custom",
            num_hands_override=None,
            result_callback=None,
        )
        with self._lock:
            if self._matches_cached_instance("_custom_roi_gesture_recognizer", spec):
                return self._custom_roi_gesture_recognizer
            self._close_instance_slot("_custom_roi_gesture_recognizer")
            self._custom_roi_gesture_recognizer = self._create_gesture_recognizer(
                runtime=runtime,
                model_path=self.config.custom_gesture_model_path,
                missing_code="mediapipe_custom_gesture_model_missing",
                running_mode_name="IMAGE",
                model_family="custom",
            )
            self._instance_specs["_custom_roi_gesture_recognizer"] = spec
            return self._custom_roi_gesture_recognizer

    def ensure_live_gesture_recognizer(
        self,
        runtime: dict[str, Any],
        *,
        result_callback: Any,
        num_hands_override: int | None = None,
    ) -> Any:
        """Reuse or create the built-in live-stream recognizer for fast gesture ack."""

        callback_spec = self._live_callback_spec("_live_gesture_recognizer", result_callback)
        spec = self._gesture_instance_spec(
            model_path=self.config.gesture_model_path,
            missing_code="mediapipe_gesture_model_missing",
            running_mode_name="LIVE_STREAM",
            model_family="builtin",
            num_hands_override=num_hands_override,
            result_callback=callback_spec,
        )

        with self._lock:
            if self._matches_cached_instance("_live_gesture_recognizer", spec):
                self._prepare_live_callback("_live_gesture_recognizer", result_callback)
                return self._live_gesture_recognizer
            self._close_instance_slot("_live_gesture_recognizer")
            wrapped_callback = self._prepare_live_callback("_live_gesture_recognizer", result_callback)
            self._live_gesture_recognizer = self._create_gesture_recognizer(
                runtime=runtime,
                model_path=self.config.gesture_model_path,
                missing_code="mediapipe_gesture_model_missing",
                running_mode_name="LIVE_STREAM",
                model_family="builtin",
                result_callback=wrapped_callback,
                num_hands_override=num_hands_override,
            )
            self._instance_specs["_live_gesture_recognizer"] = spec
            return self._live_gesture_recognizer

    def ensure_live_custom_gesture_recognizer(
        self,
        runtime: dict[str, Any],
        *,
        result_callback: Any,
        num_hands_override: int | None = None,
    ) -> Any:
        """Reuse or create the custom live-stream recognizer for fast gesture ack."""

        callback_spec = self._live_callback_spec("_live_custom_gesture_recognizer", result_callback)
        spec = self._gesture_instance_spec(
            model_path=self.config.custom_gesture_model_path,
            missing_code="mediapipe_custom_gesture_model_missing",
            running_mode_name="LIVE_STREAM",
            model_family="custom",
            num_hands_override=num_hands_override,
            result_callback=callback_spec,
        )

        with self._lock:
            if self._matches_cached_instance("_live_custom_gesture_recognizer", spec):
                self._prepare_live_callback("_live_custom_gesture_recognizer", result_callback)
                return self._live_custom_gesture_recognizer
            self._close_instance_slot("_live_custom_gesture_recognizer")
            wrapped_callback = self._prepare_live_callback("_live_custom_gesture_recognizer", result_callback)
            self._live_custom_gesture_recognizer = self._create_gesture_recognizer(
                runtime=runtime,
                model_path=self.config.custom_gesture_model_path,
                missing_code="mediapipe_custom_gesture_model_missing",
                running_mode_name="LIVE_STREAM",
                model_family="custom",
                result_callback=wrapped_callback,
                num_hands_override=num_hands_override,
            )
            self._instance_specs["_live_custom_gesture_recognizer"] = spec
            return self._live_custom_gesture_recognizer

    def reset_live_gesture_recognizers(self) -> None:
        """Drop the live-stream recognizers so the next observe() rebuilds them.

        Live-stream Tasks can accumulate internal queue pressure when callers
        continue pushing frames after the current callback missed its latency
        budget. Expose one focused reset seam so higher layers can fail closed
        on stale in-flight work without tearing down unrelated IMAGE/VIDEO mode
        helpers that remain healthy.
        """

        with self._lock:
            self._close_instance_slot("_live_gesture_recognizer")
            self._close_instance_slot("_live_custom_gesture_recognizer")

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

        asset = self._load_model_asset(
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
        selected_running_mode = self._require_attribute(
            running_mode,
            running_mode_name,
            error_code="mediapipe_runtime_incomplete",
        )
        options_kwargs: dict[str, Any] = {
            "base_options": self._build_base_options(
                runtime,
                asset=asset,
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
            options_kwargs["canned_gesture_classifier_options"] = self._build_builtin_classifier_options(runtime)
        elif model_family == "custom":
            # Custom gesture thresholds are enforced by Twinr's post-recognition
            # arbitration so the same contract stays stable across MediaPipe
            # runtimes that differ in custom classifier-option support.
            pass
        else:
            raise ValueError("mediapipe_invalid_config:model_family")
        options = gesture_recognizer_options(**options_kwargs)
        return gesture_recognizer.create_from_options(options)

    def _matches_cached_instance(self, slot_name: str, spec: tuple[Any, ...]) -> bool:
        """Return whether the cached instance matches the requested effective config."""

        return getattr(self, slot_name) is not None and self._instance_specs.get(slot_name) == spec

    def _close_instance_slot(self, slot_name: str) -> None:
        """Close one cached task instance and its callback dispatcher, if any."""

        dispatcher = self._live_callback_dispatchers.pop(slot_name, None)
        if dispatcher is not None:
            dispatcher.close(timeout_s=self._live_callback_shutdown_timeout_s())
        callback_proxy = self._live_callback_proxies.pop(slot_name, None)
        if callback_proxy is not None:
            callback_proxy.close()

        instance = getattr(self, slot_name)
        setattr(self, slot_name, None)
        self._instance_specs.pop(slot_name, None)
        if instance is None:
            return

        close_fn = getattr(instance, "close", None)
        if callable(close_fn):
            close_fn()

    def _live_callback_shutdown_timeout_s(self) -> float:
        """Return the bounded wait used when stopping callback workers."""

        raw_value = getattr(
            self.config,
            "mediapipe_live_callback_shutdown_timeout_s",
            os.getenv("TWINR_MEDIAPIPE_LIVE_CALLBACK_SHUTDOWN_TIMEOUT_S", _DEFAULT_LIVE_CALLBACK_SHUTDOWN_TIMEOUT_S),
        )
        timeout_s = self._coerce_finite_float(
            raw_value,
            error_code="mediapipe_invalid_config:mediapipe_live_callback_shutdown_timeout_s",
        )
        if timeout_s < 0.0:
            raise ValueError("mediapipe_invalid_config:mediapipe_live_callback_shutdown_timeout_s")
        return timeout_s

    def _live_callback_async_enabled(self) -> bool:
        """Return whether live callbacks should run off the MediaPipe thread."""

        raw_value = getattr(
            self.config,
            "mediapipe_live_callback_async",
            os.getenv("TWINR_MEDIAPIPE_LIVE_CALLBACK_ASYNC", ""),
        )
        if raw_value in ("", None):
            return True
        if isinstance(raw_value, bool):
            return raw_value
        normalized = str(raw_value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError("mediapipe_invalid_config:mediapipe_live_callback_async")

    def _live_callback_queue_size(self) -> int:
        """Return the bounded latest-wins callback queue size."""

        raw_value = getattr(
            self.config,
            "mediapipe_live_callback_queue_size",
            os.getenv("TWINR_MEDIAPIPE_LIVE_CALLBACK_QUEUE_SIZE", _DEFAULT_LIVE_CALLBACK_QUEUE_SIZE),
        )
        try:
            queue_size = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("mediapipe_invalid_config:mediapipe_live_callback_queue_size") from exc
        if queue_size < 1:
            raise ValueError("mediapipe_invalid_config:mediapipe_live_callback_queue_size")
        return queue_size

    def _live_callback_spec(self, slot_name: str, result_callback: Any) -> tuple[Any, ...]:
        """Return the effective live callback spec used for cache invalidation."""

        if result_callback is None:
            raise ValueError("mediapipe_live_result_callback_missing")
        if not callable(result_callback):
            raise ValueError("mediapipe_live_result_callback_invalid")
        callback_cache_key = self._live_callback_cache_key(slot_name, result_callback)
        if not self._live_callback_async_enabled():
            return ("direct", callback_cache_key)
        return ("async", callback_cache_key, self._live_callback_queue_size())

    def _prepare_live_callback(self, slot_name: str, result_callback: Any) -> Any:
        """Wrap one live callback with a bounded background dispatcher when enabled."""

        if result_callback is None:
            raise ValueError("mediapipe_live_result_callback_missing")
        if not callable(result_callback):
            raise ValueError("mediapipe_live_result_callback_invalid")
        callback_cache_key = self._live_callback_cache_key(slot_name, result_callback)
        callback_proxy = self._live_callback_proxies.get(slot_name)
        expected_proxy_spec = (callback_cache_key,)
        if callback_proxy is None or callback_proxy.spec != expected_proxy_spec:
            if callback_proxy is not None:
                callback_proxy.close()
            callback_proxy = _MutableLiveCallbackProxy(
                result_callback,
                cache_key=callback_cache_key,
            )
            self._live_callback_proxies[slot_name] = callback_proxy
        else:
            callback_proxy.update_callback(result_callback)

        if not self._live_callback_async_enabled():
            existing = self._live_callback_dispatchers.pop(slot_name, None)
            if existing is not None:
                existing.close(timeout_s=self._live_callback_shutdown_timeout_s())
            return callback_proxy

        queue_size = self._live_callback_queue_size()
        existing = self._live_callback_dispatchers.get(slot_name)
        expected_spec = ((*callback_proxy.spec,), queue_size)
        if existing is None or existing.spec != expected_spec:
            if existing is not None:
                existing.close(timeout_s=self._live_callback_shutdown_timeout_s())
            # BREAKING: live-stream callbacks now run on a dedicated worker thread by default.
            # The dispatcher uses a bounded latest-wins queue to avoid callback-induced stalls.
            existing = _LatestWinsCallbackDispatcher(
                callback_proxy,
                queue_size=queue_size,
                name=f"MediaPipeCallback:{slot_name}",
            )
            self._live_callback_dispatchers[slot_name] = existing
        return existing.enqueue

    def _live_callback_cache_key(self, slot_name: str, result_callback: Any) -> tuple[str, Any]:
        """Return the stable cache identity for one live callback slot."""

        callback_cache_key = getattr(
            result_callback,
            MEDIAPIPE_LIVE_CALLBACK_CACHE_KEY_ATTR,
            result_callback,
        )
        return (slot_name, callback_cache_key)

    def _gesture_instance_spec(
        self,
        *,
        model_path: str | None,
        missing_code: str,
        running_mode_name: str,
        model_family: str,
        num_hands_override: int | None,
        result_callback: Any,
    ) -> tuple[Any, ...]:
        """Return the effective recognizer spec used for cache invalidation."""

        num_hands = self._validate_num_hands(self.config.num_hands if num_hands_override is None else num_hands_override)
        return (
            model_family,
            running_mode_name,
            self._model_asset_spec(model_path, missing_code=missing_code),
            self._base_options_spec(),
            num_hands,
            self._validate_probability(
                self.config.min_hand_detection_confidence,
                name="min_hand_detection_confidence",
            ),
            self._validate_probability(
                self.config.min_hand_presence_confidence,
                name="min_hand_presence_confidence",
            ),
            self._validate_probability(
                self.config.min_hand_tracking_confidence,
                name="min_hand_tracking_confidence",
            ),
            self._gesture_classifier_spec(model_family),
            result_callback,
        )

    def _gesture_classifier_spec(self, model_family: str) -> tuple[Any, ...]:
        """Return classifier-related config that influences gesture outputs."""

        if model_family == "builtin":
            allowlist = self._normalize_category_list(
                getattr(self.config, "builtin_gesture_category_allowlist", None),
                name="builtin_gesture_category_allowlist",
            )
            raw_denylist = self._normalize_category_list(
                getattr(self.config, "builtin_gesture_category_denylist", None),
                name="builtin_gesture_category_denylist",
            )
            self._validated_category_filters(
                allowlist=allowlist,
                denylist=raw_denylist,
                allowlist_name="builtin_gesture_category_allowlist",
                denylist_name="builtin_gesture_category_denylist",
            )
            if allowlist:
                denylist: list[str] = []
            else:
                denylist = self._merged_builtin_category_denylist()
            return (
                "builtin",
                self._validate_probability(
                    self.config.builtin_gesture_min_score,
                    name="builtin_gesture_min_score",
                ),
                tuple(allowlist),
                tuple(denylist),
                self._optional_positive_int(
                    getattr(self.config, "builtin_gesture_max_results", None),
                    name="builtin_gesture_max_results",
                ),
            )

        if model_family == "custom":
            allowlist = self._normalize_category_list(
                getattr(self.config, "custom_gesture_category_allowlist", None),
                name="custom_gesture_category_allowlist",
            )
            raw_denylist = self._normalize_category_list(
                getattr(self.config, "custom_gesture_category_denylist", None),
                name="custom_gesture_category_denylist",
            )
            self._validated_category_filters(
                allowlist=allowlist,
                denylist=raw_denylist,
                allowlist_name="custom_gesture_category_allowlist",
                denylist_name="custom_gesture_category_denylist",
            )
            denylist = [] if allowlist else raw_denylist
            return (
                "custom",
                self._optional_probability(
                    getattr(self.config, "custom_gesture_min_score", None),
                    name="custom_gesture_min_score",
                ),
                tuple(allowlist),
                tuple(denylist),
                self._optional_positive_int(
                    getattr(self.config, "custom_gesture_max_results", None),
                    name="custom_gesture_max_results",
                ),
            )

        raise ValueError("mediapipe_invalid_config:model_family")

    def _build_builtin_classifier_options(self, runtime: dict[str, Any]) -> Any:
        """Build classifier options for the built-in gesture head."""

        allowlist = self._normalize_category_list(
            getattr(self.config, "builtin_gesture_category_allowlist", None),
            name="builtin_gesture_category_allowlist",
        )
        raw_denylist = self._normalize_category_list(
            getattr(self.config, "builtin_gesture_category_denylist", None),
            name="builtin_gesture_category_denylist",
        )
        self._validated_category_filters(
            allowlist=allowlist,
            denylist=raw_denylist,
            allowlist_name="builtin_gesture_category_allowlist",
            denylist_name="builtin_gesture_category_denylist",
        )
        denylist = [] if allowlist else self._merged_builtin_category_denylist()
        return self._build_classifier_options(
            runtime,
            score_threshold=self._validate_probability(
                self.config.builtin_gesture_min_score,
                name="builtin_gesture_min_score",
            ),
            allowlist=allowlist,
            denylist=denylist,
            max_results=self._optional_positive_int(
                getattr(self.config, "builtin_gesture_max_results", None),
                name="builtin_gesture_max_results",
            ),
        )

    def _build_custom_classifier_options(self, runtime: dict[str, Any]) -> Any | None:
        """Build classifier options for a custom gesture head when configured."""

        score_threshold = self._optional_probability(
            getattr(self.config, "custom_gesture_min_score", None),
            name="custom_gesture_min_score",
        )
        allowlist = self._normalize_category_list(
            getattr(self.config, "custom_gesture_category_allowlist", None),
            name="custom_gesture_category_allowlist",
        )
        raw_denylist = self._normalize_category_list(
            getattr(self.config, "custom_gesture_category_denylist", None),
            name="custom_gesture_category_denylist",
        )
        self._validated_category_filters(
            allowlist=allowlist,
            denylist=raw_denylist,
            allowlist_name="custom_gesture_category_allowlist",
            denylist_name="custom_gesture_category_denylist",
        )
        denylist = [] if allowlist else raw_denylist
        max_results = self._optional_positive_int(
            getattr(self.config, "custom_gesture_max_results", None),
            name="custom_gesture_max_results",
        )
        if score_threshold is None and not allowlist and not denylist and max_results is None:
            return None
        return self._build_classifier_options(
            runtime,
            score_threshold=score_threshold,
            allowlist=allowlist,
            denylist=denylist,
            max_results=max_results,
        )

    def _build_classifier_options(
        self,
        runtime: dict[str, Any],
        *,
        score_threshold: float | None,
        allowlist: list[str],
        denylist: list[str],
        max_results: int | None,
    ) -> Any:
        """Create one MediaPipe classifier options object from validated inputs."""

        if allowlist and denylist:
            raise ValueError("mediapipe_invalid_config:classifier_category_filters")

        classifier_options = self._require_runtime_entry(runtime, "ClassifierOptions")
        options_kwargs: dict[str, Any] = {}
        if score_threshold is not None:
            options_kwargs["score_threshold"] = score_threshold
        if allowlist:
            options_kwargs["category_allowlist"] = allowlist
        elif denylist:
            options_kwargs["category_denylist"] = denylist
        if max_results is not None:
            options_kwargs["max_results"] = max_results
        return classifier_options(**options_kwargs)

    def _validated_category_filters(
        self,
        *,
        allowlist: list[str],
        denylist: list[str],
        allowlist_name: str,
        denylist_name: str,
    ) -> tuple[list[str], list[str]]:
        """Reject mutually-exclusive category filter configs."""

        if allowlist and denylist:
            raise ValueError(f"mediapipe_invalid_config:{allowlist_name}|{denylist_name}")
        return allowlist, denylist

    def _merged_builtin_category_denylist(self) -> list[str]:
        """Return the default built-in denylist merged with any configured extras."""

        denylist = list(builtin_gesture_category_denylist())
        extra = self._normalize_category_list(
            getattr(self.config, "builtin_gesture_category_denylist", None),
            name="builtin_gesture_category_denylist",
        )
        for category_name in extra:
            if category_name not in denylist:
                denylist.append(category_name)
        return denylist

    def _normalize_category_list(self, value: Any, *, name: str) -> list[str]:
        """Validate an optional category list config entry."""

        if value in (None, ""):
            return []
        if isinstance(value, str):
            items = [value]
        else:
            try:
                items = list(value)
            except TypeError as exc:
                raise ValueError(f"mediapipe_invalid_config:{name}") from exc

        normalized: list[str] = []
        seen: set[str] = set()
        for item in items:
            category_name = str(item).strip()
            if not category_name or category_name in seen:
                continue
            normalized.append(category_name)
            seen.add(category_name)
        return normalized

    def _optional_probability(self, value: Any, *, name: str) -> float | None:
        """Validate one optional probability-like config value."""

        if value in (None, ""):
            return None
        return self._validate_probability(value, name=name)

    def _optional_positive_int(self, value: Any, *, name: str) -> int | None:
        """Validate one optional positive integer config value."""

        if value in (None, ""):
            return None
        try:
            integer_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"mediapipe_invalid_config:{name}") from exc
        if integer_value < 1:
            raise ValueError(f"mediapipe_invalid_config:{name}")
        return integer_value

    def _base_options_spec(self) -> tuple[Any, ...]:
        """Return the effective BaseOptions-related config state."""

        return (
            self._resolved_delegate_name(),
        )

    def _resolved_delegate_name(self) -> str | None:
        """Resolve the optional MediaPipe delegate name from config or env."""

        raw_value = getattr(
            self.config,
            "mediapipe_delegate",
            os.getenv("TWINR_MEDIAPIPE_DELEGATE", ""),
        )
        if raw_value in ("", None):
            return None
        delegate_name = str(raw_value).strip().upper()
        if delegate_name in {"DEFAULT", "AUTO", "CPU"}:
            return None
        if delegate_name == "GPU":
            return delegate_name
        raise ValueError("mediapipe_invalid_config:mediapipe_delegate")

    def _model_asset_spec(self, raw_path: str | None, *, missing_code: str) -> tuple[Any, ...]:
        """Return one file-backed model spec used for cache invalidation."""

        candidate_path = self._candidate_model_path(raw_path, missing_code=missing_code)
        try:
            file_stat = os.lstat(candidate_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{missing_code}:{candidate_path}") from exc
        except OSError as exc:
            raise RuntimeError(f"{missing_code}_unreadable:{candidate_path}") from exc
        if stat.S_ISLNK(file_stat.st_mode):
            raise PermissionError(f"{missing_code}_symlink_forbidden:{candidate_path}")
        if not stat.S_ISREG(file_stat.st_mode):
            raise FileNotFoundError(f"{missing_code}:{candidate_path}")
        if file_stat.st_size <= 0:
            raise RuntimeError(f"{missing_code}_empty:{candidate_path}")
        if file_stat.st_size > self._max_model_asset_bytes():
            raise RuntimeError(f"{missing_code}_too_large:{candidate_path}")
        return (
            str(candidate_path),
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_mtime_ns,
            file_stat.st_size,
        )

    def _max_model_asset_bytes(self) -> int:
        """Return the maximum allowed verified model asset size."""

        raw_value = getattr(
            self.config,
            "mediapipe_max_model_asset_bytes",
            os.getenv("TWINR_MEDIAPIPE_MAX_MODEL_ASSET_BYTES", _DEFAULT_MAX_MODEL_ASSET_BYTES),
        )
        try:
            max_bytes = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("mediapipe_invalid_config:mediapipe_max_model_asset_bytes") from exc
        if max_bytes < 1:
            raise ValueError("mediapipe_invalid_config:mediapipe_max_model_asset_bytes")
        return max_bytes

    def _candidate_model_path(self, raw_path: str | None, *, missing_code: str) -> Path:
        """Normalize a configured model path without silently accepting empty values."""

        normalized_path = "" if raw_path is None else str(raw_path).strip()
        if not normalized_path:
            raise FileNotFoundError(f"{missing_code}:<unset>")
        return Path(normalized_path).expanduser()

    def _load_model_asset(self, raw_path: str | None, *, missing_code: str) -> _ModelAssetRecord:
        """Open one model asset securely and return the verified cached record."""

        candidate_path = self._candidate_model_path(raw_path, missing_code=missing_code)
        cache_key = str(candidate_path)
        spec = self._model_asset_spec(raw_path, missing_code=missing_code)
        file_key = (spec[1], spec[2], spec[3], spec[4])

        cached_record = self._model_asset_cache.get(cache_key)
        if cached_record is not None and cached_record.file_key == file_key:
            return cached_record
        if cached_record is not None:
            cached_record.close()

        open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            file_descriptor = os.open(str(candidate_path), open_flags)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{missing_code}:{candidate_path}") from exc
        except IsADirectoryError as exc:
            raise FileNotFoundError(f"{missing_code}:{candidate_path}") from exc
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise PermissionError(f"{missing_code}_symlink_forbidden:{candidate_path}") from exc
            raise RuntimeError(f"{missing_code}_unreadable:{candidate_path}") from exc

        with os.fdopen(file_descriptor, "rb") as model_file:
            file_stat = os.fstat(model_file.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise FileNotFoundError(f"{missing_code}:{candidate_path}")
            if file_stat.st_size <= 0:
                raise RuntimeError(f"{missing_code}_empty:{candidate_path}")
            if file_stat.st_size > self._max_model_asset_bytes():
                raise RuntimeError(f"{missing_code}_too_large:{candidate_path}")
            model_bytes = model_file.read()

        resolved_path = Path(os.path.realpath(candidate_path))
        if not model_bytes:
            raise RuntimeError(f"{missing_code}_empty:{resolved_path}")

        record = _ModelAssetRecord(
            requested_path=candidate_path,
            resolved_path=resolved_path,
            file_key=(file_stat.st_dev, file_stat.st_ino, file_stat.st_mtime_ns, file_stat.st_size),
            model_bytes=model_bytes,
        )
        self._model_asset_cache[cache_key] = record
        return record

    def _build_base_options(self, runtime: dict[str, Any], *, asset: _ModelAssetRecord) -> Any:
        """Create BaseOptions from validated bytes, with a verified-path fallback for compatibility."""

        base_options = self._require_runtime_entry(runtime, "BaseOptions")
        delegate = self._resolve_delegate(runtime)

        buffer_kwargs: dict[str, Any] = {"model_asset_buffer": asset.model_bytes}
        if delegate is not None:
            buffer_kwargs["delegate"] = delegate
        try:
            return base_options(**buffer_kwargs)
        except TypeError:
            compat_path = self._ensure_compat_model_path(asset)
            path_kwargs: dict[str, Any] = {"model_asset_path": compat_path}
            if delegate is not None:
                path_kwargs["delegate"] = delegate
            # BREAKING: older runtimes now load from a verified compatibility file rather than
            # reopening the original model path. This preserves drop-in behavior while removing
            # the stale-path race on legacy BaseOptions implementations.
            return base_options(**path_kwargs)

    def _ensure_compat_model_path(self, asset: _ModelAssetRecord) -> str:
        """Expose verified model bytes through a path for old BaseOptions implementations."""

        if asset.compat_path:
            return asset.compat_path

        memfd_create = getattr(os, "memfd_create", None)
        if callable(memfd_create):
            try:
                compat_fd = memfd_create(
                    f"twinr-mediapipe-{abs(hash(asset.resolved_path)) & 0xFFFFFFFF:x}",
                    getattr(os, "MFD_CLOEXEC", 0),
                )
                self._write_all(compat_fd, asset.model_bytes)
                os.lseek(compat_fd, 0, os.SEEK_SET)
                compat_path = f"/proc/self/fd/{compat_fd}"
                asset.compat_fd = compat_fd
                asset.compat_path = compat_path
                return compat_path
            except OSError:
                if asset.compat_fd is not None:
                    try:
                        os.close(asset.compat_fd)
                    except OSError:
                        pass
                    asset.compat_fd = None

        compat_fd, compat_path = tempfile.mkstemp(prefix="twinr-mediapipe-", suffix=".task")
        try:
            self._write_all(compat_fd, asset.model_bytes)
            os.lseek(compat_fd, 0, os.SEEK_SET)
            os.chmod(compat_path, 0o600)
        except Exception:
            try:
                os.close(compat_fd)
            except OSError:
                pass
            try:
                os.unlink(compat_path)
            except OSError:
                pass
            raise
        asset.compat_fd = compat_fd
        asset.compat_path = compat_path
        asset.compat_temp_path = compat_path
        return compat_path

    def _write_all(self, file_descriptor: int, payload: bytes) -> None:
        """Write the full payload to one file descriptor."""

        view = memoryview(payload)
        total_written = 0
        while total_written < len(view):
            written = os.write(file_descriptor, view[total_written:])
            if written <= 0:
                raise OSError("mediapipe_model_compat_write_failed")
            total_written += written

    def _resolve_delegate(self, runtime: dict[str, Any]) -> Any | None:
        """Resolve the optional BaseOptions delegate enum."""

        delegate_name = self._resolved_delegate_name()
        if delegate_name is None:
            return None
        base_options = self._require_runtime_entry(runtime, "BaseOptions")
        delegate_enum = self._require_attribute(base_options, "Delegate", error_code="mediapipe_runtime_incomplete")
        return self._require_attribute(delegate_enum, delegate_name, error_code="mediapipe_runtime_incomplete")

    def _require_runtime_entry(self, runtime: dict[str, Any], key: str) -> Any:
        """Return one required runtime entry or raise a stable runtime error."""

        try:
            value = runtime[key]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"mediapipe_runtime_incomplete:{key}") from exc
        if value is None:
            raise RuntimeError(f"mediapipe_runtime_incomplete:{key}")
        return value

    def _require_attribute(self, obj: Any, attribute_name: str, *, error_code: str) -> Any:
        """Return one required attribute or raise a stable runtime error."""

        try:
            return getattr(obj, attribute_name)
        except AttributeError as exc:
            raise RuntimeError(f"{error_code}:{attribute_name}") from exc

    def _coerce_finite_float(self, value: Any, *, error_code: str) -> float:
        """Parse one finite float value or raise a stable validation error."""

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(error_code) from exc
        if not math.isfinite(numeric_value):
            raise ValueError(error_code)
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
            raise ValueError(f"mediapipe_invalid_config:{name}")
        return probability

    def _validate_num_hands(self, value: Any) -> int:
        """Validate the configured hand count."""

        try:
            num_hands = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("mediapipe_invalid_config:num_hands") from exc
        if num_hands < 1:
            raise ValueError("mediapipe_invalid_config:num_hands")
        return num_hands


__all__ = ["MediaPipeTaskRuntime", "normalize_image_data"]
