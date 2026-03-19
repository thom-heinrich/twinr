"""Own the bounded Picamera2 and IMX500 session lifecycle for Twinr."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import importlib
import logging
import os
import threading
import time

from .config import AICameraAdapterConfig


logger = logging.getLogger(__name__)

_METADATA_POLL_INTERVAL_S = 0.05
_CLOCK_DOMAIN_TOLERANCE_S = 86_400.0


@dataclass(slots=True)
class NetworkSession:
    """Store one active Picamera2 session and its loaded IMX500 network."""

    network_path: str
    task_name: str
    picam2: Any
    imx500: Any
    input_size: tuple[int, int]


class IMX500RuntimeSessionManager:
    """Manage the single live IMX500 session used by the local camera adapter."""

    def __init__(self, *, config: AICameraAdapterConfig, sleep_fn: Any = time.sleep) -> None:
        """Initialize one bounded runtime/session manager."""

        self.config = config
        self._sleep = sleep_fn
        self._session: NetworkSession | None = None
        self._lock = threading.RLock()  # AUDIT-FIX(#1): Serialize session lifecycle changes to prevent concurrent open/close races.

    def _close_component(self, label: str, component: Any, method_name: str) -> None:
        """Best-effort close helper for camera-related resources."""

        if component is None or not hasattr(component, method_name):
            return
        try:
            getattr(component, method_name)()
        except Exception:  # pragma: no cover - depends on camera/runtime state.
            logger.debug("Ignoring %s.%s failure during AI camera cleanup.", label, method_name, exc_info=True)

    def _safe_close_session(self, session: NetworkSession | None) -> None:
        """Best-effort cleanup for one runtime session."""

        if session is None:
            return
        self._close_component("Picamera2", session.picam2, "stop")
        self._close_component("Picamera2", session.picam2, "close")
        self._close_component("IMX500", session.imx500, "close")  # AUDIT-FIX(#2): Close IMX500 too so partial/session teardown does not leak device resources.

    def _require_runtime_component(self, runtime: dict[str, Any], name: str) -> Any:
        """Return one required runtime component or raise a normalized runtime error."""

        try:
            component = runtime[name]
        except Exception as exc:
            raise RuntimeError("picamera2_unavailable") from exc  # AUDIT-FIX(#6): Normalize malformed runtime dictionaries into stable error codes.
        if component is None:
            raise RuntimeError("picamera2_unavailable")  # AUDIT-FIX(#6): Reject missing runtime entries early.
        return component

    def _allowed_model_root(self) -> Path:
        """Resolve the only directory tree from which IMX500 models may be loaded."""

        configured_root = os.getenv("TWINR_IMX500_MODEL_ROOT", "").strip()
        if configured_root:
            return Path(configured_root).expanduser().resolve(strict=False)  # AUDIT-FIX(#5): Allow an explicit model root override without breaking the existing config schema.
        return Path(self.config.detection_network_path).expanduser().resolve(strict=False).parent  # AUDIT-FIX(#5): Default to the configured model directory as the trust boundary.

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
            raise FileNotFoundError(f"model_missing:{network_path}")  # AUDIT-FIX(#5): Reject directories and non-files.

        allowed_root = self._allowed_model_root()
        try:
            resolved_path.relative_to(allowed_root)
        except ValueError as exc:
            raise PermissionError(f"model_path_outside_root:{network_path}") from exc  # AUDIT-FIX(#5): Block traversal/symlink escapes outside the trusted model directory.

        return str(resolved_path)

    def _normalize_input_size(self, imx500: Any) -> tuple[int, int]:
        """Validate and normalize the IMX500 network input size."""

        try:
            raw_input_size = tuple(imx500.get_input_size())
        except Exception as exc:
            raise RuntimeError("invalid_input_size") from exc

        if len(raw_input_size) != 2:
            raise RuntimeError("invalid_input_size")  # AUDIT-FIX(#9): Fail fast on malformed input sizes instead of storing an invalid tuple.

        try:
            width = int(raw_input_size[0])
            height = int(raw_input_size[1])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("invalid_input_size") from exc

        if width <= 0 or height <= 0:
            raise RuntimeError("invalid_input_size")  # AUDIT-FIX(#9): Reject nonsensical zero/negative dimensions.

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
            raise RuntimeError("session_inactive")  # AUDIT-FIX(#4): Reject stale sessions before they hit closed camera objects.

    def _metadata_is_fresh(self, metadata: dict[str, Any], observed_at: float) -> bool:
        """Best-effort freshness check using SensorTimestamp when the clock domain matches."""

        sensor_timestamp = metadata.get("SensorTimestamp")
        if not isinstance(sensor_timestamp, (int, float)):
            return True

        sensor_seconds = float(sensor_timestamp)
        if sensor_seconds > 1_000_000_000_000:
            sensor_seconds /= 1_000_000_000

        monotonic_now = time.monotonic()
        if (
            abs(sensor_seconds - monotonic_now) <= _CLOCK_DOMAIN_TOLERANCE_S
            and abs(observed_at - monotonic_now) <= _CLOCK_DOMAIN_TOLERANCE_S
        ):
            return sensor_seconds >= observed_at  # AUDIT-FIX(#8): Use observed_at when both timestamps look monotonic-based.

        wall_now = time.time()
        if (
            abs(sensor_seconds - wall_now) <= _CLOCK_DOMAIN_TOLERANCE_S
            and abs(observed_at - wall_now) <= _CLOCK_DOMAIN_TOLERANCE_S
        ):
            return sensor_seconds >= observed_at  # AUDIT-FIX(#8): Also support wall-clock callers when the metadata clock matches.

        return True

    def close(self) -> None:
        """Stop and discard the current Picamera2 session."""

        with self._lock:
            session = self._session
            self._session = None
            self._safe_close_session(session)  # AUDIT-FIX(#1): Keep teardown inside the lock so a new session cannot start while the old one is still shutting down.

    def load_detection_runtime(self) -> dict[str, Any]:
        """Import the minimum Picamera2 runtime needed for SSD detection."""

        try:
            picamera2_module = importlib.import_module("picamera2")
            imx500_module = importlib.import_module("picamera2.devices.imx500")
            picamera2_class = getattr(picamera2_module, "Picamera2")
            imx500_class = getattr(imx500_module, "IMX500")
        except Exception as exc:  # pragma: no cover - depends on local environment.
            raise RuntimeError("picamera2_unavailable") from exc  # AUDIT-FIX(#6): Normalize missing modules and missing attributes the same way.
        return {
            "Picamera2": picamera2_class,
            "IMX500": imx500_class,
        }

    def load_pose_postprocess(self) -> Any:
        """Import the HigherHRNet postprocess helper when available."""

        try:
            module = importlib.import_module("picamera2.devices.imx500.postprocess_highernet")
            return getattr(module, "postprocess_higherhrnet")
        except Exception as exc:  # pragma: no cover - depends on local environment.
            raise RuntimeError("pose_postprocess_unavailable") from exc  # AUDIT-FIX(#6): Keep optional postprocess loading failure predictable for callers.

    def probe_online(self, runtime: dict[str, Any]) -> str | None:
        """Return ``None`` when an IMX500 camera is enumerated, else one error code."""

        Picamera2 = self._require_runtime_component(runtime, "Picamera2")
        try:
            camera_info = Picamera2.global_camera_info()
        except Exception as exc:  # pragma: no cover - depends on libcamera runtime.
            logger.warning("IMX500 enumeration failed: %s", exc)
            return "camera_probe_failed"
        for item in camera_info or ():
            if "imx500" in self._extract_camera_model(item):  # AUDIT-FIX(#7): Parse camera-info entries safely and match model names more robustly.
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
                self._session.task_name = task_name  # AUDIT-FIX(#10): Keep the task label aligned when reusing the same model session.
                return self._session

            previous_session = self._session
            self._session = None
            self._safe_close_session(previous_session)  # AUDIT-FIX(#1): Replace the session atomically and fully tear down the old one before another caller can reuse the camera.

            IMX500 = self._require_runtime_component(runtime, "IMX500")
            Picamera2 = self._require_runtime_component(runtime, "Picamera2")

            imx500 = None
            picam2 = None
            try:
                imx500 = IMX500(resolved_network_path)
                camera_num = getattr(imx500, "camera_num", None)
                if camera_num is None:
                    raise RuntimeError("imx500_camera_num_missing")
                picam2 = Picamera2(camera_num)
                configuration = picam2.create_preview_configuration(
                    main={"size": self.config.main_size, "format": "RGB888"},
                    controls={"FrameRate": self.config.frame_rate},
                )
                picam2.configure(configuration)
                picam2.start()
                if self.config.startup_warmup_s > 0:
                    self._sleep(float(self.config.startup_warmup_s))
                input_size = self._normalize_input_size(imx500)
            except Exception as exc:
                self._close_component("Picamera2", picam2, "stop")
                self._close_component("Picamera2", picam2, "close")
                self._close_component("IMX500", imx500, "close")
                raise RuntimeError("session_start_failed") from exc  # AUDIT-FIX(#2): Cleanup partially-created resources on every startup failure path.

            self._session = NetworkSession(
                network_path=resolved_network_path,
                task_name=task_name,
                picam2=picam2,
                imx500=imx500,
                input_size=input_size,
            )
            return self._session

    def capture_metadata(self, session: NetworkSession, *, observed_at: float) -> dict[str, Any]:
        """Wait for the next metadata payload that contains CNN tensors."""

        with self._lock:
            self._assert_active_session(session)
            metadata_wait_s = max(0.0, float(self.config.metadata_wait_s))  # AUDIT-FIX(#3): Normalize timeout values once so timeout math and logging stay consistent.
            deadline = time.monotonic() + metadata_wait_s
            while time.monotonic() < deadline:
                try:
                    metadata = session.picam2.capture_metadata()
                except Exception as exc:
                    self._safe_close_session(self._session)
                    self._session = None
                    raise RuntimeError("metadata_capture_failed") from exc  # AUDIT-FIX(#3): Normalize camera read failures and force a clean restart next call.

                if isinstance(metadata, dict):
                    has_tensor = bool(metadata.get("CnnOutputTensor"))
                    if has_tensor and self._metadata_is_fresh(metadata, observed_at):
                        return metadata  # AUDIT-FIX(#8): Do not accept stale metadata when a comparable timestamp is available.

                self._sleep(_METADATA_POLL_INTERVAL_S)

            logger.warning(
                "Timed out waiting for IMX500 metadata on %s after %.2fs.",
                session.task_name,
                metadata_wait_s,
            )
            self._safe_close_session(self._session)
            self._session = None
            raise RuntimeError("metadata_timeout")  # AUDIT-FIX(#3): Preserve the method contract instead of returning metadata without tensors.

    def capture_rgb_frame(self, runtime: dict[str, Any], *, observed_at: float) -> Any:
        """Capture one RGB preview frame from the live detection session."""

        del observed_at  # AUDIT-FIX(#8): capture_array exposes no portable timestamp here; freshness is enforced on metadata only.
        with self._lock:
            session = self.ensure_session(
                runtime,
                network_path=self.config.detection_network_path,
                task_name="detection",
            )
            try:
                try:
                    frame = session.picam2.capture_array("main")
                except TypeError:
                    frame = session.picam2.capture_array()
            except Exception as exc:
                self._safe_close_session(self._session)
                self._session = None
                raise RuntimeError("rgb_capture_failed") from exc  # AUDIT-FIX(#3): Normalize RGB capture errors and reset the broken session.
            if frame is None:
                self._safe_close_session(self._session)
                self._session = None
                raise RuntimeError("rgb_frame_missing")
            return frame


__all__ = ["IMX500RuntimeSessionManager", "NetworkSession"]