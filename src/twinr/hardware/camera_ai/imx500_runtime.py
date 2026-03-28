# CHANGELOG: 2026-03-28
# BUG-1: Capture RGB and metadata from the same Picamera2 request so tensors, metadata, and preview pixels stay aligned.
# BUG-2: Fix SensorTimestamp unit normalization so freshness checks work correctly from early boot, not only after long uptimes.
# SEC-1: Reject non-.rpk network files and keep all model loads inside the trusted IMX500 model root.
# IMP-1: Apply IMX500 network_intrinsics (task, inference_rate, aspect-ratio preservation) to match current Raspberry Pi reference flows.
# IMP-2: Prefer flush-aware captured_request/capture_request, add IMX500 KPI telemetry, and use a larger preview buffer budget for AI-camera workloads.

"""Own the bounded Picamera2 and IMX500 session lifecycle for Twinr."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import importlib
import logging
import math
import os
import threading
import time

from twinr.runtime_paths import prime_raspberry_pi_system_site_packages

from .config import AICameraAdapterConfig


logger = logging.getLogger(__name__)

_METADATA_POLL_INTERVAL_S = 0.05
_CLOCK_DOMAIN_TOLERANCE_S = 86_400.0
_OBSERVATION_REUSE_TOLERANCE_S = 0.001
_DEFAULT_IMX500_PREVIEW_BUFFER_COUNT = 12
_MIN_IMX500_FRAME_RATE = 1.0


@dataclass(slots=True)
class NetworkSession:
    """Store one active Picamera2 session and its loaded IMX500 network."""

    network_path: str
    task_name: str
    picam2: Any
    imx500: Any
    input_size: tuple[int, int]
    configured_frame_rate: float
    manual_low_light_mode: bool = False
    generation: int = 0
    base_frame_rate: float = 0.0
    model_task: str | None = None
    recommended_frame_rate: float | None = None
    preserve_aspect_ratio: bool = False


@dataclass(slots=True)
class CameraRuntimeMetrics:
    """Describe the latest bounded IMX500 camera telemetry for operators/debug."""

    lux: float | None = None
    exposure_time_us: int | None = None
    analogue_gain: float | None = None
    frame_duration_us: int | None = None
    configured_frame_rate: float | None = None
    base_frame_rate: float | None = None
    recommended_frame_rate: float | None = None
    model_task: str | None = None
    low_light_mode: bool = False
    manual_low_light_mode: bool = False
    exposure_saturated: bool = False
    auto_exposure_capped: bool = False
    dnn_runtime_ms: float | None = None
    dsp_runtime_ms: float | None = None


@dataclass(slots=True)
class ObservationCapture:
    """Cache one request-aligned observation for metadata/rgb consumers."""

    session_generation: int
    observed_at: float
    captured_at_monotonic: float
    metadata: dict[str, Any]
    rgb_frame: Any | None = None


class _RequestCaptureContext:
    """Context-manager adapter for capture_request() when captured_request() is unavailable."""

    def __init__(self, picam2: Any, *, flush: int) -> None:
        self._picam2 = picam2
        self._flush = flush
        self._request: Any = None

    def __enter__(self) -> Any:
        capture_request = getattr(self._picam2, "capture_request")
        try:
            self._request = capture_request(flush=self._flush)
        except TypeError:
            try:
                self._request = capture_request(flush=True)
            except TypeError:
                self._request = capture_request()
        return self._request

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        request = self._request
        self._request = None
        if request is not None and hasattr(request, "release"):
            try:
                request.release()
            except Exception:
                logger.debug("Ignoring CompletedRequest.release failure during cleanup.", exc_info=True)
        return False


class IMX500RuntimeSessionManager:
    """Manage the single live IMX500 session used by the local camera adapter."""

    def __init__(self, *, config: AICameraAdapterConfig, sleep_fn: Any = time.sleep) -> None:
        """Initialize one bounded runtime/session manager."""

        self.config = config
        self._sleep = sleep_fn
        self._session: NetworkSession | None = None
        self._lock = threading.RLock()
        self._last_camera_metrics: CameraRuntimeMetrics | None = None
        self._last_observation: ObservationCapture | None = None
        self._session_generation = 0

    def _close_component(self, label: str, component: Any, method_name: str) -> None:
        """Best-effort close helper for camera-related resources."""

        if component is None or not hasattr(component, method_name):
            return
        try:
            getattr(component, method_name)()
        except Exception:
            logger.debug("Ignoring %s.%s failure during AI camera cleanup.", label, method_name, exc_info=True)

    def _clear_observation_cache(self) -> None:
        """Forget any cached request-aligned observation."""

        self._last_observation = None

    def _safe_close_session(self, session: NetworkSession | None) -> None:
        """Best-effort cleanup for one runtime session."""

        if session is None:
            return
        self._close_component("Picamera2", session.picam2, "stop")
        self._close_component("Picamera2", session.picam2, "close")
        self._close_component("IMX500", session.imx500, "close")

    def _invalidate_session(self, session: NetworkSession | None) -> None:
        """Drop the active session and clear cached request state."""

        if session is self._session:
            self._session = None
        self._clear_observation_cache()
        self._safe_close_session(session)

    def _require_runtime_component(self, runtime: dict[str, Any], name: str) -> Any:
        """Return one required runtime component or raise a normalized runtime error."""

        try:
            component = runtime[name]
        except Exception as exc:
            raise RuntimeError("picamera2_unavailable") from exc
        if component is None:
            raise RuntimeError("picamera2_unavailable")
        return component

    def _allowed_model_root(self) -> Path:
        """Resolve the only directory tree from which IMX500 models may be loaded."""

        configured_root = os.getenv("TWINR_IMX500_MODEL_ROOT", "").strip()
        if configured_root:
            return Path(configured_root).expanduser().resolve(strict=False)
        return Path(self.config.detection_network_path).expanduser().resolve(strict=False).parent

    def _validate_network_path(self, network_path: str) -> str:
        """Resolve and validate a network path before handing it to the camera runtime."""

        raw_path = Path(network_path).expanduser()
        try:
            resolved_path = raw_path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"model_missing:{network_path}") from exc
        except OSError as exc:
            raise RuntimeError("model_path_invalid") from exc

        if not resolved_path.is_file():
            raise FileNotFoundError(f"model_missing:{network_path}")

        if resolved_path.suffix.lower() != ".rpk":
            raise RuntimeError(f"model_file_type_unsupported:{network_path}")

        allowed_root = self._allowed_model_root()
        try:
            resolved_path.relative_to(allowed_root)
        except ValueError as exc:
            raise PermissionError(f"model_path_outside_root:{network_path}") from exc

        return str(resolved_path)

    def _normalize_input_size(self, imx500: Any) -> tuple[int, int]:
        """Validate and normalize the IMX500 network input size."""

        try:
            raw_input_size = tuple(imx500.get_input_size())
        except Exception as exc:
            raise RuntimeError("invalid_input_size") from exc

        if len(raw_input_size) != 2:
            raise RuntimeError("invalid_input_size")

        try:
            width = int(raw_input_size[0])
            height = int(raw_input_size[1])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("invalid_input_size") from exc

        if width <= 0 or height <= 0:
            raise RuntimeError("invalid_input_size")

        return (width, height)

    def _extract_camera_model(self, item: Any) -> str:
        """Extract a normalized camera model string from one camera-info entry."""

        getter = getattr(item, "get", None)
        if callable(getter):
            try:
                return str(getter("Model", "")).strip().lower()
            except Exception:
                return ""
        return str(getattr(item, "Model", "")).strip().lower()

    def _assert_active_session(self, session: NetworkSession) -> None:
        """Ensure callers do not use stale/closed session objects."""

        if self._session is not session:
            raise RuntimeError("session_inactive")

    def _metadata_is_fresh(self, metadata: dict[str, Any], observed_at: float) -> bool:
        """Best-effort freshness check using SensorTimestamp when the clock domain matches."""

        sensor_seconds = _normalize_runtime_timestamp_s(metadata.get("SensorTimestamp"))
        if sensor_seconds is None:
            return True

        monotonic_now = time.monotonic()
        if (
            abs(sensor_seconds - monotonic_now) <= _CLOCK_DOMAIN_TOLERANCE_S
            and abs(observed_at - monotonic_now) <= _CLOCK_DOMAIN_TOLERANCE_S
        ):
            return sensor_seconds >= observed_at

        wall_now = time.time()
        if (
            abs(sensor_seconds - wall_now) <= _CLOCK_DOMAIN_TOLERANCE_S
            and abs(observed_at - wall_now) <= _CLOCK_DOMAIN_TOLERANCE_S
        ):
            return sensor_seconds >= observed_at

        return True

    def _observation_matches(
        self,
        observation: ObservationCapture | None,
        *,
        session: NetworkSession,
        observed_at: float,
        require_tensor: bool,
        require_frame: bool,
    ) -> bool:
        """Check whether a cached observation is still valid for this caller."""

        if observation is None:
            return False
        if observation.session_generation != session.generation:
            return False
        if abs(float(observation.observed_at) - float(observed_at)) > _OBSERVATION_REUSE_TOLERANCE_S:
            return False
        if require_tensor and not bool(observation.metadata.get("CnnOutputTensor")):
            return False
        if require_frame and observation.rgb_frame is None:
            return False
        return True

    def _capture_request_context(self, picam2: Any, *, flush_ns: int) -> Any:
        """Return a context manager that yields one CompletedRequest."""

        captured_request = getattr(picam2, "captured_request", None)
        if callable(captured_request):
            try:
                return captured_request(flush=flush_ns)
            except TypeError:
                try:
                    return captured_request(flush=True)
                except TypeError:
                    return captured_request()
        capture_request = getattr(picam2, "capture_request", None)
        if callable(capture_request):
            return _RequestCaptureContext(picam2, flush=flush_ns)
        raise RuntimeError("request_capture_unavailable")

    def _request_flush_ns(self, observed_at: float) -> int:
        """Translate one observation timestamp into Picamera2's monotonic-ns flush domain."""

        if not isinstance(observed_at, (int, float)):
            return time.monotonic_ns()

        observed = float(observed_at)
        if not math.isfinite(observed):
            return time.monotonic_ns()

        monotonic_now = time.monotonic()
        if abs(observed - monotonic_now) <= _CLOCK_DOMAIN_TOLERANCE_S:
            return max(0, int(round(observed * 1_000_000_000)))

        wall_now = time.time()
        if abs(observed - wall_now) <= _CLOCK_DOMAIN_TOLERANCE_S:
            delta_ns = int(round((observed - wall_now) * 1_000_000_000))
            return max(0, time.monotonic_ns() + delta_ns)

        return time.monotonic_ns()

    def _extract_request_metadata(self, request: Any) -> dict[str, Any]:
        """Read metadata from one completed request."""

        metadata = request.get_metadata()
        if isinstance(metadata, dict):
            return dict(metadata)
        try:
            return dict(metadata)
        except Exception as exc:
            raise RuntimeError("metadata_capture_failed") from exc

    def _extract_request_frame(self, request: Any) -> Any:
        """Read the main RGB preview stream from one completed request."""

        make_array = getattr(request, "make_array", None)
        if not callable(make_array):
            raise RuntimeError("rgb_capture_failed")

        errors: list[BaseException] = []
        for args in (("main",), tuple()):
            try:
                return make_array(*args)
            except TypeError as exc:
                errors.append(exc)
            except Exception as exc:
                errors.append(exc)

        if errors:
            raise RuntimeError("rgb_capture_failed") from errors[-1]
        raise RuntimeError("rgb_capture_failed")

    def _legacy_capture_observation(
        self,
        session: NetworkSession,
        *,
        observed_at: float,
        require_tensor: bool,
        include_frame: bool,
        failure_code: str,
        timeout_code: str | None,
    ) -> ObservationCapture:
        """Compatibility path for older runtimes without request capture."""

        metadata_wait_s = max(0.0, float(self.config.metadata_wait_s))
        deadline = time.monotonic() + metadata_wait_s
        while time.monotonic() < deadline:
            try:
                metadata = session.picam2.capture_metadata()
            except Exception as exc:
                self._invalidate_session(session)
                raise RuntimeError(failure_code) from exc

            if not isinstance(metadata, dict):
                self._sleep(_METADATA_POLL_INTERVAL_S)
                continue

            if require_tensor and not bool(metadata.get("CnnOutputTensor")):
                self._sleep(_METADATA_POLL_INTERVAL_S)
                continue

            if not self._metadata_is_fresh(metadata, observed_at):
                self._sleep(_METADATA_POLL_INTERVAL_S)
                continue

            rgb_frame = None
            if include_frame:
                try:
                    try:
                        rgb_frame = session.picam2.capture_array("main")
                    except TypeError:
                        rgb_frame = session.picam2.capture_array()
                except Exception as exc:
                    self._invalidate_session(session)
                    raise RuntimeError("rgb_capture_failed") from exc
                if rgb_frame is None:
                    self._invalidate_session(session)
                    raise RuntimeError("rgb_frame_missing")

            self._maybe_update_low_light_camera_controls(session, metadata=metadata)
            self._update_last_camera_metrics(session, metadata=metadata)

            observation = ObservationCapture(
                session_generation=session.generation,
                observed_at=float(observed_at),
                captured_at_monotonic=time.monotonic(),
                metadata=dict(metadata),
                rgb_frame=rgb_frame,
            )
            self._last_observation = observation
            return observation

        logger.warning(
            "Timed out waiting for IMX500 observation on %s after %.2fs.",
            session.task_name,
            metadata_wait_s,
        )
        self._invalidate_session(session)
        raise RuntimeError(timeout_code or failure_code)

    def _capture_observation(
        self,
        session: NetworkSession,
        *,
        observed_at: float,
        require_tensor: bool,
        include_frame: bool,
        failure_code: str,
        timeout_code: str | None = None,
    ) -> ObservationCapture:
        """Capture one request-aligned observation and cache it for paired consumers."""

        self._assert_active_session(session)

        cached = self._last_observation
        if self._observation_matches(
            cached,
            session=session,
            observed_at=observed_at,
            require_tensor=require_tensor,
            require_frame=include_frame,
        ):
            return cached  # type: ignore[return-value]

        metadata_wait_s = max(0.0, float(self.config.metadata_wait_s))
        deadline = time.monotonic() + metadata_wait_s
        flush_ns = self._request_flush_ns(float(observed_at))
        request_capture_supported = callable(getattr(session.picam2, "captured_request", None)) or callable(
            getattr(session.picam2, "capture_request", None)
        )

        if not request_capture_supported:
            return self._legacy_capture_observation(
                session,
                observed_at=observed_at,
                require_tensor=require_tensor,
                include_frame=include_frame,
                failure_code=failure_code,
                timeout_code=timeout_code,
            )

        while time.monotonic() < deadline:
            try:
                with self._capture_request_context(session.picam2, flush_ns=flush_ns) as request:
                    metadata = self._extract_request_metadata(request)

                    if require_tensor and not bool(metadata.get("CnnOutputTensor")):
                        self._sleep(_METADATA_POLL_INTERVAL_S)
                        continue

                    if not self._metadata_is_fresh(metadata, observed_at):
                        self._sleep(_METADATA_POLL_INTERVAL_S)
                        continue

                    rgb_frame = self._extract_request_frame(request) if include_frame else None

                    self._maybe_update_low_light_camera_controls(session, metadata=metadata)
                    self._update_last_camera_metrics(session, metadata=metadata)

                    observation = ObservationCapture(
                        session_generation=session.generation,
                        observed_at=float(observed_at),
                        captured_at_monotonic=time.monotonic(),
                        metadata=metadata,
                        rgb_frame=rgb_frame,
                    )
                    self._last_observation = observation
                    return observation
            except Exception as exc:
                self._invalidate_session(session)
                raise RuntimeError(failure_code) from exc

        logger.warning(
            "Timed out waiting for IMX500 observation on %s after %.2fs.",
            session.task_name,
            metadata_wait_s,
        )
        self._invalidate_session(session)
        raise RuntimeError(timeout_code or failure_code)

    def _preview_buffer_count(self) -> int:
        """Return the preview buffer budget for IMX500 request capture."""

        raw_value = os.getenv("TWINR_IMX500_BUFFER_COUNT", "").strip()
        if not raw_value:
            return _DEFAULT_IMX500_PREVIEW_BUFFER_COUNT
        try:
            return max(1, int(raw_value))
        except ValueError:
            logger.warning(
                "Ignoring invalid TWINR_IMX500_BUFFER_COUNT=%r; using %d.",
                raw_value,
                _DEFAULT_IMX500_PREVIEW_BUFFER_COUNT,
            )
            return _DEFAULT_IMX500_PREVIEW_BUFFER_COUNT

    def _read_network_intrinsics(self, imx500: Any) -> tuple[str | None, float | None, bool]:
        """Extract task/framerate/aspect-ratio hints from IMX500 model intrinsics."""

        intrinsics = getattr(imx500, "network_intrinsics", None)
        if intrinsics is None:
            return (None, None, False)

        updater = getattr(intrinsics, "update_with_defaults", None)
        if callable(updater):
            try:
                updater()
            except Exception:
                logger.debug("Failed to apply IMX500 network_intrinsics defaults.", exc_info=True)

        task_name = _coerce_optional_str(_safe_attribute(intrinsics, "task"))
        recommended_frame_rate = _coerce_positive_float(_safe_attribute(intrinsics, "inference_rate"))
        preserve_aspect_ratio_value = _safe_attribute(intrinsics, "preserve_aspect_ratio")
        preserve_aspect_ratio = bool(preserve_aspect_ratio_value) if preserve_aspect_ratio_value is not None else False
        return (task_name, recommended_frame_rate, preserve_aspect_ratio)

    def _pick_base_frame_rate(self, recommended_frame_rate: float | None) -> float:
        """Choose the nominal camera/inference rate for this network session."""

        configured_frame_rate = max(_MIN_IMX500_FRAME_RATE, float(self.config.frame_rate))
        if recommended_frame_rate is None:
            return configured_frame_rate
        return max(_MIN_IMX500_FRAME_RATE, min(configured_frame_rate, float(recommended_frame_rate)))

    def close(self) -> None:
        """Stop and discard the current Picamera2 session."""

        with self._lock:
            session = self._session
            self._session = None
            self._clear_observation_cache()
            self._safe_close_session(session)

    def last_camera_metrics(self) -> dict[str, object] | None:
        """Return the newest bounded camera telemetry snapshot for debug surfaces."""

        with self._lock:
            if self._last_camera_metrics is None:
                return None
            metrics = asdict(self._last_camera_metrics)

        return {
            "camera_lux": metrics["lux"],
            "camera_exposure_time_us": metrics["exposure_time_us"],
            "camera_analogue_gain": metrics["analogue_gain"],
            "camera_frame_duration_us": metrics["frame_duration_us"],
            "camera_configured_frame_rate": metrics["configured_frame_rate"],
            "camera_base_frame_rate": metrics["base_frame_rate"],
            "camera_network_recommended_frame_rate": metrics["recommended_frame_rate"],
            "camera_model_task": metrics["model_task"],
            "camera_low_light_mode": metrics["low_light_mode"],
            "camera_manual_low_light_mode": metrics["manual_low_light_mode"],
            "camera_exposure_saturated": metrics["exposure_saturated"],
            "camera_auto_exposure_capped": metrics["auto_exposure_capped"],
            "camera_dnn_runtime_ms": metrics["dnn_runtime_ms"],
            "camera_dsp_runtime_ms": metrics["dsp_runtime_ms"],
        }

    def load_detection_runtime(self) -> dict[str, Any]:
        """Import the minimum Picamera2 runtime needed for IMX500 inference."""

        prime_raspberry_pi_system_site_packages()
        try:
            picamera2_module = importlib.import_module("picamera2")
            imx500_module = importlib.import_module("picamera2.devices.imx500")
            picamera2_class = getattr(picamera2_module, "Picamera2")
            imx500_class = getattr(imx500_module, "IMX500")
        except Exception as exc:
            raise RuntimeError("picamera2_unavailable") from exc
        return {
            "Picamera2": picamera2_class,
            "IMX500": imx500_class,
        }

    def load_pose_postprocess(self) -> Any:
        """Import the HigherHRNet postprocess helper when available."""

        prime_raspberry_pi_system_site_packages()
        try:
            module = importlib.import_module("picamera2.devices.imx500.postprocess_highernet")
            return getattr(module, "postprocess_higherhrnet")
        except Exception as exc:
            raise RuntimeError("pose_postprocess_unavailable") from exc

    def probe_online(self, runtime: dict[str, Any]) -> str | None:
        """Return ``None`` when an IMX500 camera is enumerated, else one error code."""

        Picamera2 = self._require_runtime_component(runtime, "Picamera2")
        try:
            camera_info = Picamera2.global_camera_info()
        except Exception as exc:
            logger.warning("IMX500 enumeration failed: %s", exc)
            return "camera_probe_failed"
        for item in camera_info or ():
            if "imx500" in self._extract_camera_model(item):
                return None
        return "imx500_not_enumerated"

    def ensure_session(
        self,
        runtime: dict[str, Any],
        *,
        network_path: str,
        task_name: str,
    ) -> NetworkSession:
        """Reuse or create one Picamera2 session for the requested network."""

        with self._lock:
            resolved_network_path = self._validate_network_path(network_path)
            if self._session is not None and self._session.network_path == resolved_network_path:
                self._session.task_name = task_name
                return self._session

            previous_session = self._session
            self._session = None
            self._clear_observation_cache()
            self._safe_close_session(previous_session)

            IMX500 = self._require_runtime_component(runtime, "IMX500")
            Picamera2 = self._require_runtime_component(runtime, "Picamera2")

            imx500 = None
            picam2 = None
            try:
                imx500 = IMX500(resolved_network_path)
                model_task, recommended_frame_rate, preserve_aspect_ratio = self._read_network_intrinsics(imx500)
                base_frame_rate = self._pick_base_frame_rate(recommended_frame_rate)

                camera_num = getattr(imx500, "camera_num", None)
                if camera_num is None:
                    raise RuntimeError("imx500_camera_num_missing")

                picam2 = Picamera2(camera_num)
                configuration = picam2.create_preview_configuration(
                    main={"size": self.config.main_size, "format": "RGB888"},
                    controls={"FrameRate": base_frame_rate},
                    buffer_count=self._preview_buffer_count(),
                )
                picam2.configure(configuration)

                show_progress = getattr(imx500, "show_network_fw_progress_bar", None)
                if callable(show_progress):
                    try:
                        show_progress()
                    except Exception:
                        logger.debug("Failed to show IMX500 firmware upload progress.", exc_info=True)

                picam2.start()

                if preserve_aspect_ratio:
                    set_auto_aspect_ratio = getattr(imx500, "set_auto_aspect_ratio", None)
                    if callable(set_auto_aspect_ratio):
                        try:
                            set_auto_aspect_ratio()
                        except Exception:
                            logger.warning(
                                "Failed to apply IMX500 automatic aspect-ratio preservation.",
                                exc_info=True,
                            )

                if self.config.startup_warmup_s > 0:
                    self._sleep(float(self.config.startup_warmup_s))

                input_size = self._normalize_input_size(imx500)
            except Exception as exc:
                self._close_component("Picamera2", picam2, "stop")
                self._close_component("Picamera2", picam2, "close")
                self._close_component("IMX500", imx500, "close")
                raise RuntimeError("session_start_failed") from exc

            self._session_generation += 1
            self._session = NetworkSession(
                network_path=resolved_network_path,
                task_name=task_name,
                picam2=picam2,
                imx500=imx500,
                input_size=input_size,
                configured_frame_rate=float(base_frame_rate),
                manual_low_light_mode=False,
                generation=self._session_generation,
                base_frame_rate=float(base_frame_rate),
                model_task=model_task,
                recommended_frame_rate=recommended_frame_rate,
                preserve_aspect_ratio=preserve_aspect_ratio,
            )
            return self._session

    def capture_metadata(self, session: NetworkSession, *, observed_at: float) -> dict[str, Any]:
        """Wait for the next metadata payload that contains CNN tensors."""

        with self._lock:
            self._assert_active_session(session)
            observation = self._capture_observation(
                session,
                observed_at=observed_at,
                require_tensor=True,
                include_frame=False,
                failure_code="metadata_capture_failed",
                timeout_code="metadata_timeout",
            )
            return observation.metadata

    def capture_rgb_frame(self, runtime: dict[str, Any], *, observed_at: float) -> Any:
        """Capture one RGB preview frame from the live detection session."""

        with self._lock:
            session = self.ensure_session(
                runtime,
                network_path=self.config.detection_network_path,
                task_name="detection",
            )

            cached = self._last_observation
            if self._observation_matches(
                cached,
                session=session,
                observed_at=observed_at,
                require_tensor=False,
                require_frame=True,
            ):
                return cached.rgb_frame

            observation = self._capture_observation(
                session,
                observed_at=observed_at,
                require_tensor=False,
                include_frame=True,
                failure_code="rgb_capture_failed",
            )
            if observation.rgb_frame is None:
                self._invalidate_session(session)
                raise RuntimeError("rgb_frame_missing")
            return observation.rgb_frame

    def _maybe_update_low_light_camera_controls(
        self,
        session: NetworkSession,
        *,
        metadata: dict[str, Any],
    ) -> None:
        """Lower frame-rate first, then force manual long exposure only when AE remains capped."""

        configured_frame_rate = self._effective_base_frame_rate(session)
        low_light_frame_rate = float(
            max(
                _MIN_IMX500_FRAME_RATE,
                min(configured_frame_rate, float(self.config.low_light_frame_rate)),
            )
        )
        lux = _coerce_optional_float(metadata.get("Lux"))
        if lux is None:
            return
        if low_light_frame_rate >= configured_frame_rate:
            if session.manual_low_light_mode and lux >= float(self.config.low_light_recover_lux_threshold):
                self._restore_auto_low_light_controls(session, frame_rate=configured_frame_rate)
            return

        should_recover = lux >= float(self.config.low_light_recover_lux_threshold)
        if should_recover:
            if session.manual_low_light_mode:
                self._restore_auto_low_light_controls(session, frame_rate=configured_frame_rate)
            elif abs(session.configured_frame_rate - configured_frame_rate) > 1e-6:
                self._set_frame_rate_controls(session, frame_rate=configured_frame_rate)
            return

        should_lower_frame_rate = lux <= float(self.config.low_light_lux_threshold)
        if should_lower_frame_rate and abs(session.configured_frame_rate - low_light_frame_rate) > 1e-6:
            self._set_frame_rate_controls(session, frame_rate=low_light_frame_rate)

        if session.manual_low_light_mode:
            return
        if not should_lower_frame_rate:
            return
        if abs(session.configured_frame_rate - low_light_frame_rate) > 1e-6:
            return
        if not self._auto_exposure_is_capped(metadata):
            return
        self._enable_manual_low_light_exposure(session, frame_rate=low_light_frame_rate)

    def _set_frame_rate_controls(self, session: NetworkSession, *, frame_rate: float) -> None:
        """Apply one deterministic frame-rate plus matching frame-duration window."""

        target_frame_duration_us = _frame_duration_us_for_rate(frame_rate)
        try:
            session.picam2.set_controls(
                {
                    "FrameRate": float(frame_rate),
                    "FrameDurationLimits": (target_frame_duration_us, target_frame_duration_us),
                }
            )
        except Exception:
            logger.warning(
                "Failed to change IMX500 FrameRate to %.2f for low-light adaptation.",
                frame_rate,
                exc_info=True,
            )
            return
        session.configured_frame_rate = float(frame_rate)

    def _enable_manual_low_light_exposure(self, session: NetworkSession, *, frame_rate: float) -> None:
        """Force a longer fixed exposure only after AE proved it will not use the added frame budget."""

        frame_duration_us = _frame_duration_us_for_rate(frame_rate)
        target_exposure_us = max(
            1,
            min(
                frame_duration_us,
                int(round(frame_duration_us * float(self.config.low_light_manual_exposure_ratio))),
            ),
        )
        target_gain = float(max(1.0, self.config.low_light_manual_analogue_gain))
        try:
            session.picam2.set_controls(
                {
                    "AeEnable": False,
                    "FrameRate": float(frame_rate),
                    "FrameDurationLimits": (frame_duration_us, frame_duration_us),
                    "ExposureTime": target_exposure_us,
                    "AnalogueGain": target_gain,
                }
            )
        except Exception:
            logger.warning(
                "Failed to enable manual IMX500 low-light exposure at %.2f fps.",
                frame_rate,
                exc_info=True,
            )
            return
        session.configured_frame_rate = float(frame_rate)
        session.manual_low_light_mode = True

    def _restore_auto_low_light_controls(self, session: NetworkSession, *, frame_rate: float) -> None:
        """Return the camera to auto-exposure once the room is bright enough again."""

        frame_duration_us = _frame_duration_us_for_rate(frame_rate)
        try:
            session.picam2.set_controls(
                {
                    "AeEnable": True,
                    "FrameRate": float(frame_rate),
                    "FrameDurationLimits": (frame_duration_us, frame_duration_us),
                }
            )
        except Exception:
            logger.warning(
                "Failed to restore IMX500 auto exposure at %.2f fps.",
                frame_rate,
                exc_info=True,
            )
            return
        session.configured_frame_rate = float(frame_rate)
        session.manual_low_light_mode = False

    def _auto_exposure_is_capped(self, metadata: dict[str, Any]) -> bool:
        """Detect the IMX500 preview case where AE stays short despite a slower frame budget."""

        exposure_time_us = _coerce_optional_int(metadata.get("ExposureTime"))
        frame_duration_us = _coerce_optional_int(metadata.get("FrameDuration"))
        analogue_gain = _coerce_optional_float(metadata.get("AnalogueGain"))
        if exposure_time_us is None or frame_duration_us is None or analogue_gain is None:
            return False
        if frame_duration_us <= 0:
            return False
        required_gain = float(self.config.low_light_manual_analogue_gain) * 0.95
        cap_ratio = float(self.config.low_light_auto_exposure_cap_ratio)
        return analogue_gain >= required_gain and exposure_time_us <= int(frame_duration_us * cap_ratio)

    def _read_kpi_metrics(self, session: NetworkSession, metadata: dict[str, Any]) -> tuple[float | None, float | None]:
        """Extract DNN/DSP runtime telemetry when the runtime exposes it."""

        get_kpi_info = getattr(session.imx500, "get_kpi_info", None)
        if not callable(get_kpi_info):
            return (None, None)
        try:
            values = get_kpi_info(metadata)
        except Exception:
            logger.debug("Failed to read IMX500 KPI telemetry.", exc_info=True)
            return (None, None)

        if isinstance(values, tuple) and len(values) >= 2:
            return (_coerce_optional_float(values[0]), _coerce_optional_float(values[1]))
        return (None, None)

    def _update_last_camera_metrics(
        self,
        session: NetworkSession,
        *,
        metadata: dict[str, Any],
    ) -> None:
        """Persist one bounded camera telemetry snapshot from fresh metadata."""

        exposure_time_us = _coerce_optional_int(metadata.get("ExposureTime"))
        frame_duration_us = _coerce_optional_int(metadata.get("FrameDuration"))
        exposure_saturated = False
        if exposure_time_us is not None and frame_duration_us is not None and frame_duration_us > 0:
            exposure_saturated = exposure_time_us >= int(frame_duration_us * 0.95)
        auto_exposure_capped = self._auto_exposure_is_capped(metadata)
        dnn_runtime_ms, dsp_runtime_ms = self._read_kpi_metrics(session, metadata)

        self._last_camera_metrics = CameraRuntimeMetrics(
            lux=_coerce_optional_float(metadata.get("Lux")),
            exposure_time_us=exposure_time_us,
            analogue_gain=_coerce_optional_float(metadata.get("AnalogueGain")),
            frame_duration_us=frame_duration_us,
            configured_frame_rate=round(float(session.configured_frame_rate), 3),
            base_frame_rate=round(self._effective_base_frame_rate(session), 3),
            recommended_frame_rate=_coerce_positive_float(session.recommended_frame_rate),
            model_task=session.model_task,
            low_light_mode=session.configured_frame_rate < (self._effective_base_frame_rate(session) - 1e-6),
            manual_low_light_mode=session.manual_low_light_mode,
            exposure_saturated=exposure_saturated,
            auto_exposure_capped=auto_exposure_capped,
            dnn_runtime_ms=dnn_runtime_ms,
            dsp_runtime_ms=dsp_runtime_ms,
        )

    def _effective_base_frame_rate(self, session: NetworkSession) -> float:
        """Return the intended session base frame rate even for legacy/test sessions."""

        configured = _coerce_positive_float(session.base_frame_rate)
        if configured is not None:
            return float(max(_MIN_IMX500_FRAME_RATE, configured))
        recommended = _coerce_positive_float(session.recommended_frame_rate)
        if recommended is not None:
            return float(max(_MIN_IMX500_FRAME_RATE, recommended))
        return float(max(_MIN_IMX500_FRAME_RATE, self.config.frame_rate))


def _safe_attribute(obj: Any, name: str) -> Any:
    """Read one attribute defensively and return None on runtime errors."""

    try:
        return getattr(obj, name)
    except Exception:
        return None


def _coerce_optional_str(value: object) -> str | None:
    """Convert one optional runtime value into a non-empty string."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _coerce_optional_float(value: object) -> float | None:
    """Convert one optional runtime-metadata value into a bounded float."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, 3)


def _coerce_positive_float(value: object) -> float | None:
    """Convert one value into a positive finite float."""

    number = _coerce_optional_float(value)
    if number is None or number <= 0:
        return None
    return number


def _coerce_optional_int(value: object) -> int | None:
    """Convert one optional runtime-metadata value into an integer."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_runtime_timestamp_s(value: object) -> float | None:
    """Normalize wall/monotonic timestamps that may arrive in seconds or nanoseconds."""

    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(timestamp) or timestamp <= 0:
        return None

    scale_reference = max(time.monotonic(), time.time(), 1.0)
    if timestamp > (scale_reference * 1_000.0):
        timestamp /= 1_000_000_000.0
    return timestamp


def _frame_duration_us_for_rate(frame_rate: float) -> int:
    """Convert one positive frame-rate into a deterministic frame duration."""

    return max(1, int(round(1_000_000.0 / max(_MIN_IMX500_FRAME_RATE, float(frame_rate)))))


__all__ = ["IMX500RuntimeSessionManager", "NetworkSession"]
