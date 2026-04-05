# CHANGELOG: 2026-03-28
# BUG-1: Serialize lazy pipeline init/teardown to prevent duplicate MediaPipe/gesture pipeline construction, leaked Pi-side resources, and inconsistent state under concurrent callers.
# BUG-2: Make pose/motion cache cleanup best-effort too, so secondary cleanup errors never mask the primary capture/runtime failure.
# BUG-3: Runtime reset now separates generic pipeline teardown from the dedicated live-gesture pipeline so non-gesture recovery does not force gesture cold-start churn.
# SEC-1: Close a practical availability hole where concurrent requests could force repeated heavy pipeline creation on a Raspberry Pi 4 and exhaust memory/CPU.
# IMP-1: Add lifecycle telemetry and generation counters for self-healing, stale-result fencing, and field debugging.
# IMP-2: Add explicit backend factory hooks plus a centralized config builder so callers can swap in 2026-era accelerated backends without rewriting lifecycle ownership code.
# IMP-3: On MediaPipe temporal-reset failure, also invalidate derived caches and bump the runtime generation to avoid stale pose/motion bleed-through.
"""Runtime/pipeline lifecycle helpers for the local AI-camera adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import threading
import time
from typing import Any

from ..config import MediaPipeVisionConfig
from ..detection import DetectionResult, capture_detection
from ..live_gesture_pipeline import LiveGesturePipeline
from ..mediapipe_pipeline import MediaPipeVisionPipeline
from .common import LOGGER

logger = LOGGER

_LOCK_INIT_GUARD = threading.Lock()


@dataclass(slots=True)
class _RuntimeLifecycleTelemetry:
    """Low-cost lifecycle counters for field diagnostics and stale-result fencing."""

    runtime_generation: int = 0
    mediapipe_generation: int = 0
    live_gesture_generation: int = 0

    runtime_close_count: int = 0
    mediapipe_create_count: int = 0
    mediapipe_close_count: int = 0
    live_gesture_create_count: int = 0
    live_gesture_close_count: int = 0

    runtime_close_failures: int = 0
    mediapipe_create_failures: int = 0
    mediapipe_close_failures: int = 0
    live_gesture_create_failures: int = 0
    live_gesture_close_failures: int = 0
    mediapipe_temporal_reset_failures: int = 0

    pose_cache_clear_count: int = 0
    motion_state_clear_count: int = 0
    cleanup_failures: int = 0

    last_error_component: str | None = None
    last_error_type: str | None = None
    last_error_message: str | None = None
    last_error_monotonic_ns: int | None = None


class AICameraAdapterRuntimeMixin:
    """Own lazy runtime initialization and best-effort teardown."""

    def _run_cleanup_callback(self, cleanup: Callable[[], object]) -> None:
        """Execute one already-validated cleanup hook."""

        cleanup()

    def _get_runtime_lifecycle_lock(self) -> threading.RLock:
        """Return the per-instance lifecycle lock, creating it lazily if needed."""

        lock = getattr(self, "_runtime_lifecycle_lock", None)
        if lock is not None:
            return lock

        with _LOCK_INIT_GUARD:
            lock = getattr(self, "_runtime_lifecycle_lock", None)
            if lock is None:
                lock = threading.RLock()
                setattr(self, "_runtime_lifecycle_lock", lock)
            return lock

    def _get_runtime_lifecycle_telemetry(self) -> _RuntimeLifecycleTelemetry:
        """Return the per-instance lifecycle telemetry, creating it lazily if needed."""

        telemetry = getattr(self, "_runtime_lifecycle_telemetry", None)
        if telemetry is not None:
            return telemetry

        with _LOCK_INIT_GUARD:
            telemetry = getattr(self, "_runtime_lifecycle_telemetry", None)
            if telemetry is None:
                telemetry = _RuntimeLifecycleTelemetry()
                setattr(self, "_runtime_lifecycle_telemetry", telemetry)
            return telemetry

    def _record_lifecycle_error_locked(self, component: str, exc: BaseException) -> None:
        """Store the most recent lifecycle failure for debugging and health checks."""

        telemetry = self._get_runtime_lifecycle_telemetry()
        telemetry.last_error_component = component
        telemetry.last_error_type = type(exc).__name__
        telemetry.last_error_message = str(exc)
        telemetry.last_error_monotonic_ns = time.monotonic_ns()

    def _bump_runtime_generation_locked(self) -> None:
        """Invalidate runtime-bound work after a disruptive lifecycle transition."""

        self._get_runtime_lifecycle_telemetry().runtime_generation += 1

    def _bump_mediapipe_generation_locked(self) -> None:
        """Invalidate MediaPipe-bound work after a disruptive lifecycle transition."""

        self._get_runtime_lifecycle_telemetry().mediapipe_generation += 1

    def _bump_live_gesture_generation_locked(self) -> None:
        """Invalidate live-gesture-bound work after a disruptive lifecycle transition."""

        self._get_runtime_lifecycle_telemetry().live_gesture_generation += 1

    def _get_runtime_generation(self) -> int:
        """Expose the current runtime generation for stale-result fencing."""

        with self._get_runtime_lifecycle_lock():
            return self._get_runtime_lifecycle_telemetry().runtime_generation

    def _get_mediapipe_pipeline_generation(self) -> int:
        """Expose the current MediaPipe generation for stale-result fencing."""

        with self._get_runtime_lifecycle_lock():
            return self._get_runtime_lifecycle_telemetry().mediapipe_generation

    def _get_live_gesture_pipeline_generation(self) -> int:
        """Expose the current live-gesture generation for stale-result fencing."""

        with self._get_runtime_lifecycle_lock():
            return self._get_runtime_lifecycle_telemetry().live_gesture_generation

    def _snapshot_runtime_lifecycle(self) -> dict[str, Any]:
        """Return lightweight lifecycle state for observability and field debugging."""

        with self._get_runtime_lifecycle_lock():
            telemetry = asdict(self._get_runtime_lifecycle_telemetry())
            last_error_age_s: float | None = None
            last_error_monotonic_ns = telemetry["last_error_monotonic_ns"]
            if last_error_monotonic_ns is not None:
                last_error_age_s = max(
                    0.0,
                    (time.monotonic_ns() - last_error_monotonic_ns) / 1_000_000_000,
                )
            telemetry["last_error_age_s"] = last_error_age_s
            telemetry["mediapipe_pipeline_open"] = getattr(self, "_mediapipe_pipeline", None) is not None
            telemetry["live_gesture_pipeline_open"] = getattr(self, "_live_gesture_pipeline", None) is not None
            runtime_manager = getattr(self, "_runtime_manager", None)
            telemetry["runtime_manager_type"] = (
                type(runtime_manager).__name__ if runtime_manager is not None else None
            )
            return telemetry

    def _build_vision_config(self) -> MediaPipeVisionConfig:
        """Centralize adapter->pipeline config translation for easy backend swaps."""

        return MediaPipeVisionConfig.from_ai_camera_config(self.config)

    def _create_mediapipe_pipeline(self, config: MediaPipeVisionConfig) -> MediaPipeVisionPipeline:
        """Build the local MediaPipe pipeline. Subclasses may override for new backends."""

        return MediaPipeVisionPipeline(config=config)

    def _create_live_gesture_pipeline(self, config: MediaPipeVisionConfig) -> LiveGesturePipeline:
        """Build the live-stream gesture pipeline. Subclasses may override for new backends."""

        return LiveGesturePipeline(config=config)

    def _ensure_mediapipe_pipeline(self) -> MediaPipeVisionPipeline:
        """Reuse or create the Pi-side MediaPipe inference pipeline lazily."""

        with self._get_runtime_lifecycle_lock():
            pipeline = getattr(self, "_mediapipe_pipeline", None)
            if pipeline is not None:
                return pipeline

            try:
                pipeline = self._create_mediapipe_pipeline(self._build_vision_config())
                if pipeline is None:
                    raise RuntimeError("_create_mediapipe_pipeline() returned None.")
            except Exception as exc:
                telemetry = self._get_runtime_lifecycle_telemetry()
                telemetry.mediapipe_create_failures += 1
                self._record_lifecycle_error_locked("mediapipe_create", exc)
                logger.debug("Local AI camera MediaPipe pipeline creation failed.", exc_info=True)
                raise

            setattr(self, "_mediapipe_pipeline", pipeline)
            telemetry = self._get_runtime_lifecycle_telemetry()
            telemetry.mediapipe_create_count += 1
            self._bump_mediapipe_generation_locked()
            return pipeline

    def _ensure_live_gesture_pipeline(self) -> LiveGesturePipeline:
        """Reuse or create the dedicated live-stream gesture pipeline lazily."""

        with self._get_runtime_lifecycle_lock():
            pipeline = getattr(self, "_live_gesture_pipeline", None)
            if pipeline is not None:
                return pipeline

            try:
                pipeline = self._create_live_gesture_pipeline(self._build_vision_config())
                if pipeline is None:
                    raise RuntimeError("_create_live_gesture_pipeline() returned None.")
            except Exception as exc:
                telemetry = self._get_runtime_lifecycle_telemetry()
                telemetry.live_gesture_create_failures += 1
                self._record_lifecycle_error_locked("live_gesture_create", exc)
                logger.debug("Local AI camera live gesture pipeline creation failed.", exc_info=True)
                raise

            setattr(self, "_live_gesture_pipeline", pipeline)
            telemetry = self._get_runtime_lifecycle_telemetry()
            telemetry.live_gesture_create_count += 1
            self._bump_live_gesture_generation_locked()
            return pipeline

    def _load_detection_runtime(self) -> dict[str, Any]:
        """Preserve the historic detection-runtime override point for tests."""

        with self._get_runtime_lifecycle_lock():
            return self._runtime_manager.load_detection_runtime()

    def _probe_online(self, runtime: dict[str, Any]) -> str | None:
        """Preserve the historic online-probe override point for tests."""

        with self._get_runtime_lifecycle_lock():
            return self._runtime_manager.probe_online(runtime)

    def _capture_detection(self, runtime: dict[str, Any], *, observed_at: float) -> DetectionResult:
        """Preserve the historic detection-capture override point for tests."""

        with self._get_runtime_lifecycle_lock():
            return capture_detection(
                runtime_manager=self._runtime_manager,
                runtime=runtime,
                config=self.config,
                observed_at=observed_at,
            )

    def _capture_rgb_frame(self, runtime: dict[str, Any], *, observed_at: float) -> Any:
        """Preserve the historic RGB-capture override point for tests."""

        with self._get_runtime_lifecycle_lock():
            return self._runtime_manager.capture_rgb_frame(runtime, observed_at=observed_at)

    def _safe_clear_pose_cache_locked(self) -> None:
        """Clear pose cache without allowing cleanup to hide the primary failure."""

        with self._get_runtime_lifecycle_lock():
            clear_pose_cache = getattr(self, "_clear_pose_cache", None)
            if clear_pose_cache is None or not callable(clear_pose_cache):
                return

            try:
                self._run_cleanup_callback(clear_pose_cache)
                self._get_runtime_lifecycle_telemetry().pose_cache_clear_count += 1
            except Exception:  # pragma: no cover - depends on cache/runtime state.
                telemetry = self._get_runtime_lifecycle_telemetry()
                telemetry.cleanup_failures += 1
                self._record_lifecycle_error_locked("pose_cache_clear", Exception("pose cache clear failed"))
                logger.debug("Ignoring pose-cache clear failure during AI camera cleanup.", exc_info=True)

    def _safe_clear_motion_state_locked(self) -> None:
        """Clear motion state without allowing cleanup to hide the primary failure."""

        with self._get_runtime_lifecycle_lock():
            clear_motion_state = getattr(self, "_clear_motion_state", None)
            if clear_motion_state is None or not callable(clear_motion_state):
                return

            try:
                self._run_cleanup_callback(clear_motion_state)
                self._get_runtime_lifecycle_telemetry().motion_state_clear_count += 1
            except Exception:  # pragma: no cover - depends on cache/runtime state.
                telemetry = self._get_runtime_lifecycle_telemetry()
                telemetry.cleanup_failures += 1
                self._record_lifecycle_error_locked("motion_state_clear", Exception("motion state clear failed"))
                logger.debug("Ignoring motion-state clear failure during AI camera cleanup.", exc_info=True)

    def _reset_runtime_state_locked(
        self,
        *,
        close_pipeline: bool,
        close_live_gesture_pipeline: bool | None = None,
        clear_pose: bool,
        clear_motion: bool,
    ) -> None:
        """Best-effort cleanup for runtime failures while the adapter lock is held."""

        close_live_pipeline = close_pipeline if close_live_gesture_pipeline is None else close_live_gesture_pipeline
        with self._get_runtime_lifecycle_lock():
            self._bump_runtime_generation_locked()
            setattr(self, "_attention_stream_state_by_lane", {})
            self._safe_close_runtime_locked()  # Cleanup must never raise over the primary capture failure.
            if close_pipeline:
                self._safe_close_mediapipe_pipeline_locked()  # Drop potentially corrupted MediaPipe state before the next attempt.
            if close_live_pipeline:
                self._safe_close_live_gesture_pipeline_locked()
            if clear_pose:
                self._safe_clear_pose_cache_locked()  # Reset stale pose cache across runtime failures and close().
            if clear_motion:
                self._safe_clear_motion_state_locked()  # Reset stale motion history across runtime failures and close().

    def _safe_close_runtime_locked(self) -> None:
        """Close the IMX500 runtime manager without raising."""

        with self._get_runtime_lifecycle_lock():
            runtime_manager = getattr(self, "_runtime_manager", None)
            if runtime_manager is None:
                return

            try:
                runtime_manager.close()
                self._get_runtime_lifecycle_telemetry().runtime_close_count += 1
            except Exception as exc:  # pragma: no cover - depends on runtime state.
                telemetry = self._get_runtime_lifecycle_telemetry()
                telemetry.runtime_close_failures += 1
                telemetry.cleanup_failures += 1
                self._record_lifecycle_error_locked("runtime_close", exc)
                logger.debug("Ignoring runtime close failure during AI camera cleanup.", exc_info=True)

    def _safe_close_mediapipe_pipeline_locked(self) -> None:
        """Close the MediaPipe pipeline without raising."""

        with self._get_runtime_lifecycle_lock():
            pipeline = getattr(self, "_mediapipe_pipeline", None)
            setattr(self, "_mediapipe_pipeline", None)
            if pipeline is None:
                return

            self._bump_mediapipe_generation_locked()
            try:
                pipeline.close()
                self._get_runtime_lifecycle_telemetry().mediapipe_close_count += 1
            except Exception as exc:  # pragma: no cover - depends on MediaPipe runtime state.
                telemetry = self._get_runtime_lifecycle_telemetry()
                telemetry.mediapipe_close_failures += 1
                telemetry.cleanup_failures += 1
                self._record_lifecycle_error_locked("mediapipe_close", exc)
                logger.debug("Ignoring MediaPipe close failure during AI camera cleanup.", exc_info=True)

    def _safe_close_live_gesture_pipeline_locked(self) -> None:
        """Close the live-stream gesture pipeline without raising."""

        with self._get_runtime_lifecycle_lock():
            pipeline = getattr(self, "_live_gesture_pipeline", None)
            setattr(self, "_live_gesture_pipeline", None)
            if pipeline is None:
                return

            self._bump_live_gesture_generation_locked()
            try:
                pipeline.close()
                self._get_runtime_lifecycle_telemetry().live_gesture_close_count += 1
            except Exception as exc:  # pragma: no cover - depends on MediaPipe runtime state.
                telemetry = self._get_runtime_lifecycle_telemetry()
                telemetry.live_gesture_close_failures += 1
                telemetry.cleanup_failures += 1
                self._record_lifecycle_error_locked("live_gesture_close", exc)
                logger.debug("Ignoring live gesture pipeline close failure during AI camera cleanup.", exc_info=True)

    def _safe_reset_mediapipe_temporal_state_locked(self) -> None:
        """Reset MediaPipe temporal state without failing observation capture."""

        with self._get_runtime_lifecycle_lock():
            pipeline = getattr(self, "_mediapipe_pipeline", None)
            if pipeline is None:
                return

            try:
                pipeline.reset_temporal_state()
            except Exception as exc:  # pragma: no cover - depends on MediaPipe runtime state.
                telemetry = self._get_runtime_lifecycle_telemetry()
                telemetry.mediapipe_temporal_reset_failures += 1
                self._record_lifecycle_error_locked("mediapipe_temporal_reset", exc)
                logger.warning("Local AI camera MediaPipe temporal reset failed; recreating pipeline.")
                logger.debug("Local AI camera MediaPipe temporal reset exception details.", exc_info=True)
                self._bump_runtime_generation_locked()
                self._safe_close_mediapipe_pipeline_locked()
                self._safe_clear_pose_cache_locked()
                self._safe_clear_motion_state_locked()
