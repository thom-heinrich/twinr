"""Coordinate proactive sensing, wakeword arming, and monitor lifecycle.

This module assembles the runtime-facing proactive monitor used by Twinr
workflows. It wires sensor observers, wakeword policy, presence-session
tracking, buffered vision review, automation observation export, and the
background worker lifecycle without owning the lower-level scoring or hardware
adapters themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from typing import TYPE_CHECKING, Any, Callable
import logging
import os
import tempfile
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware import GpioPirMonitor, V4L2StillCamera, configured_pir_monitor
from twinr.hardware.audio import AmbientAudioCaptureWindow, AmbientAudioSampler, pcm16_to_wav_bytes
from twinr.ops.paths import resolve_ops_paths_for_config
from twinr.providers.openai import OpenAIBackend

from ..social.camera_surface import ProactiveCameraSnapshot, ProactiveCameraSurface, ProactiveCameraSurfaceUpdate
from ..social.engine import SocialAudioObservation, SocialObservation, SocialTriggerDecision, SocialTriggerEngine, SocialVisionObservation
from ..social.observers import AmbientAudioObservationProvider, NullAudioObservationProvider, OpenAIVisionObservationProvider
from ..social.vision_review import (
    OpenAIProactiveVisionReviewer,
    ProactiveVisionFrameBuffer,
    ProactiveVisionReview,
    is_reviewable_image_trigger,
)
from ..wakeword.calibration import WakewordCalibrationStore, apply_wakeword_calibration
from ..wakeword.matching import WakewordMatch, WakewordPhraseSpotter
from ..wakeword.policy import SttWakewordVerifier, WakewordDecision, WakewordDecisionPolicy, normalize_wakeword_backend
from ..wakeword.spotter import WakewordOpenWakeWordSpotter
from ..wakeword.stream import OpenWakeWordStreamingMonitor, WakewordStreamDetection
from .presence import PresenceSessionController, PresenceSessionSnapshot

if TYPE_CHECKING:
    from twinr.agent.base_agent.runtime import TwinrRuntime

_WAKEWORD_SUPPRESSION_EXEMPT_TRIGGERS = frozenset(
    {"possible_fall", "floor_stillness", "distress_possible"}
)
_VISION_REVIEW_FAIL_OPEN_TRIGGERS = _WAKEWORD_SUPPRESSION_EXEMPT_TRIGGERS
_MAX_WAKEWORD_STREAM_EVENTS_PER_CYCLE = 8
_DEFAULT_CLOSE_JOIN_TIMEOUT_S = 5.0
_MAX_CAPTURE_PHRASE_TOKEN_LEN = 64

_LOGGER = logging.getLogger(__name__)


# AUDIT-FIX(#1): Isolate telemetry and log formatting so ops-event persistence or emit sinks cannot kill safety-critical monitoring.
def _safe_emit(emit: Callable[[str], None] | None, line: str) -> None:
    """Emit one telemetry line while suppressing sink failures."""

    if emit is None:
        return
    try:
        emit(line)
    except Exception:
        _LOGGER.warning("Proactive emit sink failed.", exc_info=True)
        return


# AUDIT-FIX(#1): Sanitize exception text before logging to avoid multiline/log-injection issues and overlong payloads.
def _exception_text(error: BaseException | object, *, limit: int = 240) -> str:
    """Normalize one exception payload into bounded log-safe text."""

    raw = str(error) if not isinstance(error, BaseException) else (str(error) or error.__class__.__name__)
    text = " ".join(raw.split())
    if not text:
        text = "unknown_error"
    if len(text) > limit:
        return f"{text[: limit - 3]}..."
    return text


# AUDIT-FIX(#1): Degrade gracefully if file-backed ops-event persistence is unavailable.
def _append_ops_event(
    runtime: TwinrRuntime,
    *,
    event: str,
    message: str,
    data: dict[str, Any],
    emit: Callable[[str], None] | None = None,
    level: str | None = None,
) -> None:
    """Append one ops event without letting persistence failures escape."""

    kwargs: dict[str, Any] = {
        "event": event,
        "message": message,
        "data": data,
    }
    if level is not None:
        kwargs["level"] = level
    try:
        runtime.ops_events.append(**kwargs)
    except Exception:
        _safe_emit(emit, f"ops_event_append_failed={event}")


# AUDIT-FIX(#8): Normalize optional config strings so None/whitespace values do not crash subsystem initialization.
def _normalize_optional_text(*values: Any) -> str:
    """Return the first non-blank text value from a config-like list."""

    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


# AUDIT-FIX(#8): Normalize sequence-like config inputs from older env schemas before using them.
def _normalize_text_tuple(values: Any) -> tuple[str, ...]:
    """Normalize sequence-like config input to one tuple of non-blank strings."""

    if values is None:
        return ()
    if isinstance(values, (str, Path)):
        values = (values,)
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


# AUDIT-FIX(#7): Refuse obviously unsafe capture directories and require the final path to stay under artifacts_root.
def _ensure_safe_capture_directory(artifacts_root: Path) -> Path:
    """Create and validate the wakeword capture directory under artifacts root."""

    captures_dir = artifacts_root / "ops" / "wakeword_captures"
    captures_dir.mkdir(parents=True, exist_ok=True)
    for candidate in (artifacts_root, artifacts_root / "ops", captures_dir):
        if candidate.exists() and candidate.is_symlink():
            raise RuntimeError(f"refusing symlinked capture directory component: {candidate}")
    resolved_root = artifacts_root.resolve(strict=False)
    resolved_dir = captures_dir.resolve(strict=False)
    try:
        resolved_dir.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError("wakeword capture directory escapes artifacts root") from exc
    return captures_dir


# AUDIT-FIX(#7): Write captures atomically so partial files or power loss do not leave corrupt artifacts behind.
def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Persist bytes atomically to one existing directory."""

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            _LOGGER.warning("Failed to remove proactive capture temp file after write error.", exc_info=True)
        raise


# AUDIT-FIX(#4): Log degraded-mode component startup instead of failing the whole monitor for optional hardware or provider issues.
def _record_component_warning(
    *,
    runtime: TwinrRuntime,
    emit: Callable[[str], None] | None,
    reason: str,
    detail: str,
) -> None:
    """Record one degraded-mode warning during proactive monitor setup."""

    _safe_emit(emit, f"proactive_component_warning={reason}")
    _append_ops_event(
        runtime,
        event="proactive_component_warning",
        level="warning",
        message="Proactive monitor is running in degraded mode.",
        data={
            "reason": reason,
            "detail": detail,
        },
        emit=emit,
    )


@dataclass(frozen=True, slots=True)
class ProactiveTickResult:
    """Describe the externally relevant outcome of one monitor tick."""

    decision: SocialTriggerDecision | None = None
    wakeword_match: WakewordMatch | None = None
    inspected: bool = False
    person_visible: bool = False


# AUDIT-FIX(#4): Provide a no-op trigger engine so wakeword/audio monitoring can still run when proactive triggers are disabled or engine setup fails.
class _NullSocialTriggerEngine:
    """Provide a no-op trigger engine when proactive triggers are disabled."""

    best_evaluation = None

    def observe(self, observation: SocialObservation) -> SocialTriggerDecision | None:
        """Return no proactive decision for the given observation."""

        return None


class ProactiveCoordinator:
    """Coordinate one proactive monitor tick across sensors and policies.

    The coordinator owns the runtime-facing orchestration for PIR, vision,
    ambient audio, wakeword detection, presence sessions, trigger review, and
    automation observation export. Lower-level trigger scoring and wakeword
    matching remain in sibling packages.
    """

    def __init__(
        self,
        *,
        config: TwinrConfig,
        runtime: TwinrRuntime,
        engine: SocialTriggerEngine,
        trigger_handler: Callable[[SocialTriggerDecision], bool],
        vision_observer,
        pir_monitor: GpioPirMonitor | None = None,
        audio_observer=None,
        presence_session: PresenceSessionController | None = None,
        wakeword_spotter: WakewordPhraseSpotter | None = None,
        wakeword_stream: OpenWakeWordStreamingMonitor | None = None,
        wakeword_policy: WakewordDecisionPolicy | None = None,
        vision_reviewer: OpenAIProactiveVisionReviewer | None = None,
        wakeword_handler: Callable[[WakewordMatch], bool] | None = None,
        idle_predicate: Callable[[], bool] | None = None,
        observation_handler: Callable[[dict[str, Any], tuple[str, ...]], None] | None = None,
        emit: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        audio_observer_fallback_factory: Callable[[], Any] | None = None,
    ) -> None:
        """Initialize one coordinator from already-built dependencies."""

        self.config = config
        self.runtime = runtime
        self.engine = engine
        self.trigger_handler = trigger_handler
        self.vision_observer = vision_observer
        self.pir_monitor = pir_monitor
        self._null_audio_observer = NullAudioObservationProvider()  # AUDIT-FIX(#6): keep a guaranteed local fallback snapshot source for audio-path failures.
        self.audio_observer = audio_observer or self._null_audio_observer
        self._audio_observer_fallback_factory = audio_observer_fallback_factory
        self.presence_session = presence_session
        self.wakeword_spotter = wakeword_spotter
        self.wakeword_stream = wakeword_stream
        self.wakeword_policy = wakeword_policy
        self.vision_reviewer = vision_reviewer
        self.wakeword_handler = wakeword_handler
        self.idle_predicate = idle_predicate
        self.observation_handler = observation_handler
        self.emit = emit or (lambda _line: None)
        self.clock = clock
        self._normalized_wakeword_backend = normalize_wakeword_backend(
            getattr(config, "wakeword_primary_backend", config.wakeword_backend),
            default="openwakeword",
        )
        self._last_motion_at: float | None = None
        self._last_capture_at: float | None = None
        self._last_observation_key: tuple[object, ...] | None = None
        self._camera_surface = ProactiveCameraSurface.from_config(config)
        self._last_sensor_flags: dict[str, bool] = {}
        self._speech_detected_since: float | None = None
        self._quiet_since: float | None = None
        self._last_presence_key: tuple[bool, str | None] | None = None
        self._last_wakeword_attempt_at: float | None = None
        self._last_possible_fall_prompted_session_id: int | None = None
        self.latest_presence_snapshot: PresenceSessionSnapshot | None = None

    # AUDIT-FIX(#1): Route all module-local emits through a non-throwing wrapper.
    def _emit(self, line: str) -> None:
        """Emit one coordinator-local telemetry line safely."""

        _safe_emit(self.emit, line)

    # AUDIT-FIX(#1): Route all module-local ops events through a non-throwing wrapper.
    def _append_ops_event(
        self,
        *,
        event: str,
        message: str,
        data: dict[str, Any],
        level: str | None = None,
    ) -> None:
        """Append one coordinator-local ops event safely."""

        _append_ops_event(
            self.runtime,
            event=event,
            message=message,
            data=data,
            level=level,
            emit=self.emit,
        )

    # AUDIT-FIX(#1): Standardize internal fault reporting so recovery code never re-throws while handling errors.
    def _record_fault(
        self,
        *,
        event: str,
        message: str,
        error: BaseException | object,
        data: dict[str, Any] | None = None,
        level: str = "error",
    ) -> None:
        """Record one recoverable coordinator fault in telemetry and ops events."""

        error_text = _exception_text(error)
        payload = dict(data or {})
        payload["error"] = error_text
        self._emit(f"{event}={error_text}")
        self._append_ops_event(
            event=event,
            level=level,
            message=message,
            data=payload,
        )

    # AUDIT-FIX(#6): Audio observation faults should degrade to a null snapshot instead of aborting the full monitoring cycle.
    def _observe_audio_safe(self):
        """Collect one ambient-audio snapshot with null fallback on failure."""

        try:
            return self.audio_observer.observe()
        except Exception as exc:
            self._record_fault(
                event="proactive_audio_observe_failed",
                message="Ambient audio observation failed; using a null audio snapshot for this tick.",
                error=exc,
            )
            try:
                return self._null_audio_observer.observe()
            except Exception as fallback_exc:
                self._record_fault(
                    event="proactive_null_audio_observe_failed",
                    message="Null audio observation fallback failed.",
                    error=fallback_exc,
                )
                raise

    # AUDIT-FIX(#6): Vision observation faults should fall back to an uninspected tick so wakeword/presence logic can continue.
    def _observe_vision_safe(self):
        """Collect one vision snapshot or return None on failure/unavailability."""

        if self.vision_observer is None:
            return None
        try:
            return self.vision_observer.observe()
        except Exception as exc:
            self._record_fault(
                event="proactive_vision_observe_failed",
                message="Vision observation failed; continuing without a camera inspection this tick.",
                error=exc,
            )
            return None

    # AUDIT-FIX(#6): Snapshot buffering is optional; a buffer write failure must not suppress trigger evaluation.
    def _record_vision_snapshot_safe(self, snapshot) -> None:
        """Buffer one vision snapshot for later review when available."""

        if self.vision_reviewer is None:
            return
        if getattr(snapshot, "image", None) is None:
            return
        try:
            self.vision_reviewer.record_snapshot(snapshot)
        except Exception as exc:
            self._record_fault(
                event="proactive_vision_snapshot_buffer_failed",
                message="Failed to buffer a proactive vision snapshot for later review.",
                error=exc,
            )

    # AUDIT-FIX(#5): Honor proactive_enabled as a hard privacy gate for camera-triggered proactive behavior.
    def _proactive_triggers_enabled(self) -> bool:
        """Return whether camera-driven proactive triggers may run."""

        return bool(self.config.proactive_enabled)

    # AUDIT-FIX(#2): Safety-critical triggers must fail open when review is unavailable, otherwise the device can miss a fall/distress prompt.
    def _should_fail_open_without_vision_review(self, decision: SocialTriggerDecision) -> bool:
        """Return whether missing review must not block this safety trigger."""

        return decision.trigger_id in _VISION_REVIEW_FAIL_OPEN_TRIGGERS

    # AUDIT-FIX(#6): Keep decision suppression logic centralized after wakeword/presence checks.
    def _process_decision(
        self,
        *,
        now: float,
        decision: SocialTriggerDecision | None,
        observation: SocialObservation,
        inspected: bool,
        presence_snapshot: PresenceSessionSnapshot,
    ) -> ProactiveTickResult:
        """Apply final suppression, review, and dispatch to one trigger decision."""

        if decision is None:
            return ProactiveTickResult(
                inspected=inspected,
                person_visible=observation.vision.person_visible,
            )
        suppressed_age_s = self._recent_wakeword_attempt_age_s(now, decision)
        if suppressed_age_s is not None:
            self._record_trigger_skipped_recent_wakeword_attempt(
                decision,
                wakeword_attempt_age_s=suppressed_age_s,
            )
            return ProactiveTickResult(
                inspected=inspected,
                person_visible=observation.vision.person_visible,
            )
        blocked_reason = self._presence_session_block_reason(
            decision,
            presence_snapshot=presence_snapshot,
        )
        if blocked_reason is not None:
            self._record_trigger_skipped_presence_session(
                decision,
                presence_snapshot=presence_snapshot,
                reason=blocked_reason,
            )
            return ProactiveTickResult(
                inspected=inspected,
                person_visible=observation.vision.person_visible,
            )
        reviewed_decision, review = self._review_trigger(
            decision,
            observation=observation,
        )
        if reviewed_decision is None:
            return ProactiveTickResult(
                inspected=inspected,
                person_visible=observation.vision.person_visible,
            )
        return self._handle_decision(
            reviewed_decision,
            observation=observation,
            inspected=inspected,
            presence_snapshot=presence_snapshot,
            review=review,
        )

    # AUDIT-FIX(#6): Reuse one absence-path implementation for normal no-inspection ticks and degraded vision failures.
    def _run_without_inspection(
        self,
        *,
        now: float,
        motion_active: bool,
        audio_snapshot,
        proactive_enabled: bool,
    ) -> ProactiveTickResult:
        """Run one proactive cycle when no fresh camera inspection is available."""

        observation, decision = self._feed_absence(
            now=now,
            motion_active=motion_active,
            audio_observation=audio_snapshot.observation,
            evaluate_trigger=proactive_enabled,
        )
        presence_snapshot = self._observe_presence(
            now=now,
            person_visible=None,
            motion_active=motion_active,
            speech_detected=audio_snapshot.observation.speech_detected is True,
        )
        self._dispatch_automation_observation(observation, inspected=False)
        self._record_observation_if_changed(
            observation,
            inspected=False,
            audio_snapshot=audio_snapshot,
            presence_snapshot=presence_snapshot,
        )
        wakeword_match = self._maybe_detect_wakeword(
            now=now,
            audio_snapshot=audio_snapshot,
            presence_snapshot=presence_snapshot,
        )
        if wakeword_match is not None:
            return ProactiveTickResult(
                wakeword_match=wakeword_match,
                inspected=False,
                person_visible=False,
            )
        return self._process_decision(
            now=now,
            decision=decision,
            observation=observation,
            inspected=False,
            presence_snapshot=presence_snapshot,
        )

    def tick(self) -> ProactiveTickResult:
        """Run one proactive monitor tick across idle, sensor, and trigger paths."""

        now = self.clock()
        if self.idle_predicate is not None:
            try:
                if not self.idle_predicate():
                    return ProactiveTickResult()
            except Exception as exc:
                self._record_fault(
                    event="proactive_idle_predicate_failed",
                    message="Idle predicate failed; skipping this proactive tick.",
                    error=exc,
                )
                return ProactiveTickResult()
        motion_active = self._update_motion(now)
        try:
            if self.runtime.status.value != "waiting":
                return ProactiveTickResult()
        except Exception as exc:
            self._record_fault(
                event="proactive_runtime_status_failed",
                message="Failed to read runtime status; skipping this proactive tick.",
                error=exc,
            )
            return ProactiveTickResult()

        audio_snapshot = self._observe_audio_safe()
        proactive_enabled = self._proactive_triggers_enabled()
        if not self._should_inspect(now, motion_active=motion_active):
            return self._run_without_inspection(
                now=now,
                motion_active=motion_active,
                audio_snapshot=audio_snapshot,
                proactive_enabled=proactive_enabled,
            )

        snapshot = self._observe_vision_safe()
        if snapshot is None:
            return self._run_without_inspection(
                now=now,
                motion_active=motion_active,
                audio_snapshot=audio_snapshot,
                proactive_enabled=proactive_enabled,
            )

        self._last_capture_at = now
        self._record_vision_snapshot_safe(snapshot)
        low_motion = self._is_low_motion(now, motion_active=motion_active)
        observation = SocialObservation(
            observed_at=now,
            inspected=True,
            pir_motion_detected=motion_active,
            low_motion=low_motion,
            vision=snapshot.observation,
            audio=audio_snapshot.observation,
        )
        decision = self.engine.observe(observation) if proactive_enabled else None  # AUDIT-FIX(#5): never evaluate proactive triggers when proactive mode is disabled.
        presence_snapshot = self._observe_presence(
            now=now,
            person_visible=snapshot.observation.person_visible,
            motion_active=motion_active,
            speech_detected=audio_snapshot.observation.speech_detected is True,
        )
        self._dispatch_automation_observation(observation, inspected=True)
        self._record_observation_if_changed(
            observation,
            inspected=True,
            vision_snapshot=snapshot,
            audio_snapshot=audio_snapshot,
            presence_snapshot=presence_snapshot,
        )
        self._emit(f"proactive_person_visible={str(snapshot.observation.person_visible).lower()}")  # AUDIT-FIX(#1): safe emit wrapper prevents telemetry failures from aborting monitor execution.
        self._emit(
            "proactive_speech_detected="
            f"{str(audio_snapshot.observation.speech_detected).lower()}"
        )
        if self.presence_session is not None:
            self._emit(f"wakeword_armed={str(presence_snapshot.armed).lower()}")
        if audio_snapshot.observation.distress_detected is not None:
            self._emit(
                "proactive_distress_detected="
                f"{str(audio_snapshot.observation.distress_detected).lower()}"
            )
        if audio_snapshot.sample is not None:
            self._emit(f"proactive_audio_peak_rms={audio_snapshot.sample.peak_rms}")
        wakeword_match = self._maybe_detect_wakeword(
            now=now,
            audio_snapshot=audio_snapshot,
            presence_snapshot=presence_snapshot,
        )
        if wakeword_match is not None:
            return ProactiveTickResult(
                wakeword_match=wakeword_match,
                inspected=True,
                person_visible=snapshot.observation.person_visible,
            )
        return self._process_decision(
            now=now,
            decision=decision,
            observation=observation,
            inspected=True,
            presence_snapshot=presence_snapshot,
        )

    def _feed_absence(
        self,
        *,
        now: float,
        motion_active: bool,
        audio_observation: SocialAudioObservation,
        evaluate_trigger: bool = True,
    ) -> tuple[SocialObservation, SocialTriggerDecision | None]:
        """Build one observation for the no-camera path and optionally score it."""

        observation = SocialObservation(
            observed_at=now,
            inspected=False,
            pir_motion_detected=motion_active,
            low_motion=self._is_low_motion(now, motion_active=motion_active),
            vision=SocialVisionObservation(person_visible=False),
            audio=audio_observation,
        )
        return observation, (self.engine.observe(observation) if evaluate_trigger else None)  # AUDIT-FIX(#5): bypass trigger evaluation in wakeword-only/privacy-disabled proactive mode.

    def _observe_presence(
        self,
        *,
        now: float,
        person_visible: bool | None,
        motion_active: bool,
        speech_detected: bool,
    ) -> PresenceSessionSnapshot:
        """Update and publish the current wakeword presence-session snapshot."""

        if self.presence_session is None:
            snapshot = PresenceSessionSnapshot(
                armed=False,
                reason="disabled",
                person_visible=bool(person_visible),
            )
            self.latest_presence_snapshot = snapshot
            return snapshot
        snapshot = self.presence_session.observe(
            now=now,
            person_visible=person_visible,
            motion_active=motion_active,
            speech_detected=speech_detected,
        )
        self.latest_presence_snapshot = snapshot
        self._record_presence_if_changed(snapshot)
        if self.wakeword_stream is not None:
            self.wakeword_stream.update_presence(snapshot)
        return snapshot

    def poll_wakeword_stream(self) -> WakewordMatch | None:
        """Drain one wakeword stream detection when streaming is enabled."""

        if self.wakeword_stream is None or self.wakeword_handler is None:
            return None
        try:
            error = self.wakeword_stream.poll_error()
        except Exception as exc:
            self._handle_wakeword_stream_failure(exc)  # AUDIT-FIX(#6): stream poll failures should downgrade the stream path instead of bubbling out every loop.
            return None
        if error:
            self._handle_wakeword_stream_failure(error)
            return None
        try:
            detection = self.wakeword_stream.poll_detection()
        except Exception as exc:
            self._handle_wakeword_stream_failure(exc)
            return None
        if detection is None:
            return None
        return self._handle_stream_detection(detection)

    # AUDIT-FIX(#6): Replace a failed streaming wakeword path with a secondary audio observer when available so the monitor degrades instead of repeatedly faulting.
    def _handle_wakeword_stream_failure(self, error: BaseException | object) -> None:
        """Disable the broken wakeword stream and try to enable audio fallback."""

        failed_stream = self.wakeword_stream
        if failed_stream is None:
            return
        self.wakeword_stream = None
        self._record_fault(
            event="wakeword_stream_failed",
            message="openWakeWord streaming failed; switching to a degraded audio path when available.",
            error=error,
        )
        try:
            failed_stream.close()
        except Exception as close_exc:
            self._record_fault(
                event="wakeword_stream_close_failed",
                message="Failed to close the broken wakeword stream cleanly.",
                error=close_exc,
            )
        fallback_factory = self._audio_observer_fallback_factory
        if fallback_factory is None:
            self.audio_observer = self._null_audio_observer
            self._append_ops_event(
                event="proactive_audio_fallback_unavailable",
                level="warning",
                message="Wakeword streaming failed and no secondary audio observer was configured.",
                data={},
            )
            return
        try:
            fallback_observer = fallback_factory()
            self.audio_observer = fallback_observer or self._null_audio_observer
        except Exception as exc:
            self.audio_observer = self._null_audio_observer
            self._record_fault(
                event="proactive_audio_fallback_init_failed",
                message="Wakeword streaming failed and the secondary audio observer could not be started.",
                error=exc,
            )
            return
        self._append_ops_event(
            event="proactive_audio_fallback_enabled",
            message="Wakeword streaming failed and a secondary audio observer was started.",
            data={},
        )

    # AUDIT-FIX(#6): Callback isolation for wakeword handlers prevents a bad downstream action from aborting wakeword detection.
    def _dispatch_wakeword_match(self, match: WakewordMatch, *, source: str) -> bool:
        """Send one accepted wakeword match to the downstream handler safely."""

        if self.wakeword_handler is None:
            return False
        try:
            handled = bool(self.wakeword_handler(match))
        except Exception as exc:
            self._record_fault(
                event="wakeword_handler_failed",
                message="Wakeword handler failed.",
                error=exc,
                data={
                    "source": source,
                    "matched_phrase": match.matched_phrase,
                    "detector_label": match.detector_label,
                },
            )
            return False
        if handled:
            self._emit("wakeword_detected=true")
        return handled

    def _maybe_detect_wakeword(
        self,
        *,
        now: float,
        audio_snapshot,
        presence_snapshot: PresenceSessionSnapshot,
    ) -> WakewordMatch | None:
        """Run one ambient-audio wakeword attempt when the session is armed."""

        if self.wakeword_stream is not None:
            return None
        if self.wakeword_spotter is None or self.wakeword_handler is None:
            return None
        if not presence_snapshot.armed:
            return None
        if not self._wakeword_audio_candidate(audio_snapshot):
            return None
        if self._last_wakeword_attempt_at is not None:
            if (now - self._last_wakeword_attempt_at) < self.config.wakeword_attempt_cooldown_s:
                return None
        self._last_wakeword_attempt_at = now
        capture_window = AmbientAudioCaptureWindow(
            sample=audio_snapshot.sample,
            pcm_bytes=audio_snapshot.pcm_bytes,
            sample_rate=audio_snapshot.sample_rate,
            channels=audio_snapshot.channels,
        )
        try:  # AUDIT-FIX(#6): wakeword detector faults must degrade cleanly instead of aborting the entire cycle.
            match = self.wakeword_spotter.detect(capture_window)
        except Exception as exc:
            self._record_fault(
                event="wakeword_detect_failed",
                message="Wakeword phrase detection failed.",
                error=exc,
            )
            return None
        return self._handle_wakeword_decision(
            match=match,
            capture_window=capture_window,
            presence_snapshot=presence_snapshot,
            source="ambient_spotter",
        )

    def _handle_stream_detection(self, detection: WakewordStreamDetection) -> WakewordMatch | None:
        """Finalize one streaming wakeword detection through the normal policy path."""

        self._last_wakeword_attempt_at = self.clock()
        return self._handle_wakeword_decision(
            match=detection.match,
            capture_window=detection.capture_window,
            presence_snapshot=detection.presence_snapshot,
            source="streaming_spotter",
        )

    def _handle_wakeword_decision(
        self,
        *,
        match: WakewordMatch,
        capture_window: AmbientAudioCaptureWindow | None,
        presence_snapshot: PresenceSessionSnapshot,
        source: str,
    ) -> WakewordMatch | None:
        """Persist and dispatch one wakeword policy decision."""

        decision = self._decide_wakeword(match=match, capture_window=capture_window, source=source)
        capture_path = self._persist_wakeword_capture(
            decision.match,
            capture_window=capture_window,
            outcome=decision.outcome,
        )
        if capture_path is not None:
            decision = replace(decision, capture_path=capture_path)
        self._record_wakeword_attempt(
            decision.match,
            presence_snapshot=presence_snapshot,
            decision=decision,
        )
        self._record_wakeword_decision(
            decision=decision,
            presence_snapshot=presence_snapshot,
        )
        if not decision.detected:
            return None
        if self._dispatch_wakeword_match(decision.match, source=source):
            return decision.match
        return None

    def _decide_wakeword(
        self,
        *,
        match: WakewordMatch,
        capture_window: AmbientAudioCaptureWindow | None,
        source: str,
    ) -> WakewordDecision:
        """Run the configured wakeword policy or a trivial pass-through fallback."""

        if self.wakeword_policy is None:
            return WakewordDecision(
                detected=match.detected,
                outcome="detected" if match.detected else "rejected",
                match=match,
                source=source,
                backend_used=match.backend,
                primary_backend=self._normalized_wakeword_backend,
                fallback_backend=None,
                verifier_mode="disabled",
                verifier_used=False,
                verifier_status="not_needed",
            )
        return self.wakeword_policy.decide(
            match=match,
            capture=capture_window,
            source=source,
        )

    def _persist_wakeword_capture(
        self,
        match: WakewordMatch,
        *,
        capture_window: AmbientAudioCaptureWindow,
        outcome: str,
    ) -> str | None:
        """Persist one evaluated wakeword audio window and return its path."""

        if capture_window is None or not capture_window.pcm_bytes:
            return None
        try:
            captures_dir = _ensure_safe_capture_directory(
                Path(resolve_ops_paths_for_config(self.config).artifacts_root)
            )  # AUDIT-FIX(#7): validate the capture directory before writing local artifacts.
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            unique_token = f"{time.time_ns():019d}"  # AUDIT-FIX(#7): add a monotonic-unique suffix so same-second detections cannot overwrite each other.
            try:
                score_value = 0.0 if match.score is None else max(0.0, float(match.score))
            except (TypeError, ValueError):
                score_value = 0.0
            score_token = f"{score_value:.4f}".replace(".", "_")
            phrase_token = (match.matched_phrase or match.detector_label or "unknown").strip().replace(" ", "_")
            safe_phrase = (
                "".join(char for char in phrase_token if char.isalnum() or char in {"_", "-"})
                or "unknown"
            )[:_MAX_CAPTURE_PHRASE_TOKEN_LEN]
            safe_outcome = "".join(char for char in outcome.strip() if char.isalnum() or char in {"_", "-"}) or "unknown"
            path = captures_dir / f"{timestamp}__{unique_token}__{safe_outcome}__score-{score_token}__{safe_phrase}.wav"
            wav_bytes = pcm16_to_wav_bytes(
                capture_window.pcm_bytes,
                sample_rate=capture_window.sample_rate,
                channels=capture_window.channels,
            )
            _write_bytes_atomic(path, wav_bytes)
        except Exception as exc:
            self._record_fault(
                event="wakeword_capture_persist_failed",
                message="Failed to persist the wakeword capture window.",
                error=exc,
            )
            return None

        sample = getattr(capture_window, "sample", None)
        data: dict[str, Any] = {
            "path": str(path),
            "outcome": outcome,
            "matched_phrase": match.matched_phrase,
            "detector_label": match.detector_label,
            "score": match.score,
        }
        if sample is not None:
            data.update(
                {
                    "duration_ms": getattr(sample, "duration_ms", None),
                    "peak_rms": getattr(sample, "peak_rms", None),
                }
            )
        self._append_ops_event(
            event="wakeword_capture_saved",
            message="Saved the local audio window that was evaluated by the wakeword policy.",
            data=data,
        )
        return str(path)

    def _wakeword_audio_candidate(self, audio_snapshot) -> bool:
        """Return whether one audio snapshot is worth wakeword evaluation."""

        sample = getattr(audio_snapshot, "sample", None)
        if sample is None:
            return False
        if not getattr(audio_snapshot, "pcm_bytes", None):
            return False
        if getattr(audio_snapshot, "sample_rate", None) is None:
            return False
        if getattr(audio_snapshot, "channels", None) is None:
            return False
        if self._normalized_wakeword_backend == "openwakeword":  # AUDIT-FIX(#9): normalized comparison avoids config case/whitespace mismatches.
            if sample.active_chunk_count >= max(1, self.config.wakeword_min_active_chunks):
                return True
            if audio_snapshot.observation.speech_detected is True:
                return True
            peak_threshold = max(1400, int(self.config.audio_speech_threshold * 2))
            return sample.peak_rms >= peak_threshold
        if sample.active_chunk_count < max(1, self.config.wakeword_min_active_chunks):
            return False
        if sample.active_ratio >= self.config.wakeword_min_active_ratio:
            return True
        peak_threshold = max(1400, int(self.config.audio_speech_threshold * 2))
        return sample.peak_rms >= peak_threshold

    def _handle_decision(
        self,
        decision: SocialTriggerDecision,
        *,
        observation: SocialObservation,
        inspected: bool,
        presence_snapshot: PresenceSessionSnapshot,
        review: ProactiveVisionReview | None = None,
    ) -> ProactiveTickResult:
        """Dispatch one reviewed proactive trigger and update session-side state."""

        self._record_trigger_detected(decision, observation=observation, review=review)
        try:  # AUDIT-FIX(#6): downstream trigger actions are external code and must not crash the monitor loop.
            handled = bool(self.trigger_handler(decision))
        except Exception as exc:
            self._record_fault(
                event="proactive_trigger_handler_failed",
                message="Proactive trigger handler failed.",
                error=exc,
                data={"trigger": decision.trigger_id},
            )
            handled = False
        presence_session_id = getattr(presence_snapshot, "session_id", None)
        if handled:
            self._emit(f"proactive_trigger={decision.trigger_id}")  # AUDIT-FIX(#1): safe emit wrapper keeps trigger dispatch resilient even when emit sinks break.
            if (
                decision.trigger_id == "possible_fall"
                and self.config.proactive_possible_fall_once_per_presence_session
                and presence_snapshot.armed
                and presence_session_id is not None
            ):
                self._last_possible_fall_prompted_session_id = presence_session_id
        return ProactiveTickResult(
            decision=decision if handled else None,
            inspected=inspected,
            person_visible=observation.vision.person_visible,
        )

    def _review_trigger(
        self,
        decision: SocialTriggerDecision,
        *,
        observation: SocialObservation,
    ) -> tuple[SocialTriggerDecision | None, ProactiveVisionReview | None]:
        """Run buffered vision review for one trigger when configured."""

        if self.vision_reviewer is None:
            return decision, None
        if not is_reviewable_image_trigger(decision.trigger_id):
            return decision, None
        try:
            review = self.vision_reviewer.review(decision, observation=observation)
        except Exception as exc:
            self._record_fault(
                event="proactive_vision_review_failed",
                message="Buffered proactive vision review failed.",
                error=exc,
                data={"trigger": decision.trigger_id},
            )
            if self._should_fail_open_without_vision_review(decision):
                self._append_ops_event(
                    event="proactive_vision_review_fail_open",
                    level="warning",
                    message="A safety-critical proactive trigger proceeded because buffered vision review failed.",
                    data={"trigger": decision.trigger_id},
                )
                return decision, None
            self._record_trigger_skipped_vision_review_unavailable(decision)
            return None, None
        if review is None:
            self._append_ops_event(
                event="proactive_vision_review_unavailable",
                message="Buffered proactive vision review was enabled but no usable result was available.",
                data={"trigger": decision.trigger_id},
            )
            if self._should_fail_open_without_vision_review(decision):
                self._append_ops_event(
                    event="proactive_vision_review_fail_open",
                    level="warning",
                    message="A safety-critical proactive trigger proceeded because buffered vision review was unavailable.",
                    data={"trigger": decision.trigger_id},
                )
                return decision, None
            self._record_trigger_skipped_vision_review_unavailable(decision)
            return None, None
        self._record_vision_review(decision, review=review)
        if not review.approved:
            self._record_trigger_skipped_vision_review(decision, review=review)
            return None, review
        return decision, review

    def _recent_wakeword_attempt_age_s(
        self,
        now: float,
        decision: SocialTriggerDecision,
    ) -> float | None:
        """Return the age of a recent wakeword attempt that should suppress prompts."""

        if decision.trigger_id in _WAKEWORD_SUPPRESSION_EXEMPT_TRIGGERS:
            return None
        if self._last_wakeword_attempt_at is None:
            return None
        age_s = now - self._last_wakeword_attempt_at
        if age_s < 0:
            return None
        if age_s > self.config.wakeword_block_proactive_after_attempt_s:
            return None
        return age_s

    def _update_motion(self, now: float) -> bool:
        """Poll PIR state and return whether motion is currently active."""

        if self.pir_monitor is None:
            return False
        motion_active = False
        try:  # AUDIT-FIX(#6): sensor backend faults must not abort the entire proactive tick.
            while True:
                event = self.pir_monitor.poll(timeout=0.0)
                if event is None:
                    break
                if event.motion_detected:
                    self._last_motion_at = now
                    motion_active = True
        except Exception as exc:
            self._record_fault(
                event="pir_poll_failed",
                message="PIR event polling failed; continuing with the last known motion state.",
                error=exc,
            )
        try:
            if self.pir_monitor.motion_detected():
                self._last_motion_at = now
                motion_active = True
        except Exception as exc:
            self._record_fault(
                event="pir_read_failed",
                message="Direct PIR state read failed; continuing with queued motion state only.",
                error=exc,
            )
        return motion_active

    def _should_inspect(self, now: float, *, motion_active: bool) -> bool:
        """Return whether the next tick should trigger a camera inspection."""

        if not self._proactive_triggers_enabled():
            return False
        if self.vision_observer is None:
            return False
        if self._last_motion_at is None:
            return False
        if not motion_active and (now - self._last_motion_at) > self.config.proactive_motion_window_s:
            return False
        if self._last_capture_at is not None and (now - self._last_capture_at) < self.config.proactive_capture_interval_s:
            return False
        return True

    def _is_low_motion(self, now: float, *, motion_active: bool) -> bool:
        """Return whether recent PIR history qualifies as low motion."""

        if motion_active:
            return False
        if self._last_motion_at is None:
            return False  # AUDIT-FIX(#4): do not infer "low motion" before any real motion history exists, especially when PIR hardware is missing or has not fired yet.
        return (now - self._last_motion_at) >= self.config.proactive_low_motion_after_s

    def _record_observation_if_changed(
        self,
        observation: SocialObservation,
        *,
        inspected: bool,
        vision_snapshot=None,
        audio_snapshot=None,
        presence_snapshot: PresenceSessionSnapshot | None = None,
    ) -> None:
        """Append one observation event only when the visible state changed."""

        presence_session_id = None if presence_snapshot is None else getattr(presence_snapshot, "session_id", None)
        observation_key = (
            inspected,
            observation.pir_motion_detected,
            observation.low_motion,
            observation.vision.person_visible,
            observation.vision.looking_toward_device,
            observation.vision.body_pose.value,
            observation.vision.smiling,
            observation.vision.hand_or_object_near_camera,
            observation.audio.speech_detected,
            observation.audio.distress_detected,
            None if presence_snapshot is None else presence_snapshot.armed,
            None if presence_snapshot is None else presence_snapshot.reason,
            presence_session_id,
        )
        if observation_key == self._last_observation_key:
            return
        if not inspected and self._last_observation_key is None:
            self._last_observation_key = observation_key
            return
        self._last_observation_key = observation_key
        data = {
            "inspected": inspected,
            "pir_motion_detected": observation.pir_motion_detected,
            "low_motion": observation.low_motion,
            "person_visible": observation.vision.person_visible,
            "looking_toward_device": observation.vision.looking_toward_device,
            "body_pose": observation.vision.body_pose.value,
            "smiling": observation.vision.smiling,
            "hand_or_object_near_camera": observation.vision.hand_or_object_near_camera,
            "speech_detected": observation.audio.speech_detected,
            "distress_detected": observation.audio.distress_detected,
        }
        if presence_snapshot is not None:
            data.update(
                {
                    "wakeword_armed": presence_snapshot.armed,
                    "wakeword_presence_reason": presence_snapshot.reason,
                    "wakeword_presence_session_id": presence_session_id,
                }
            )
        if vision_snapshot is not None:
            data.update(
                {
                    "vision_model": vision_snapshot.model,
                    "vision_request_id": vision_snapshot.request_id,
                    "vision_response_id": vision_snapshot.response_id,
                }
            )
        if audio_snapshot is not None and audio_snapshot.sample is not None:
            data.update(
                {
                    "audio_average_rms": audio_snapshot.sample.average_rms,
                    "audio_peak_rms": audio_snapshot.sample.peak_rms,
                    "audio_active_ratio": audio_snapshot.sample.active_ratio,
                    "audio_active_chunks": audio_snapshot.sample.active_chunk_count,
                    "audio_chunk_count": audio_snapshot.sample.chunk_count,
                }
            )
        top_evaluation = self.engine.best_evaluation
        if top_evaluation is not None and top_evaluation.score > 0.0:
            data.update(
                {
                    "top_trigger": top_evaluation.trigger_id,
                    "top_score": top_evaluation.score,
                    "top_threshold": top_evaluation.threshold,
                    "top_trigger_passed": top_evaluation.passed,
                }
            )
            if top_evaluation.blocked_reason is not None:
                data["top_blocked_reason"] = top_evaluation.blocked_reason
        self._append_ops_event(
            event="proactive_observation",
            message="Proactive monitor recorded a changed observation.",
            data=data,
        )

    def _record_presence_if_changed(self, snapshot: PresenceSessionSnapshot) -> None:
        """Append one ops event when the presence-session state changes."""

        session_id = getattr(snapshot, "session_id", None)
        key = (snapshot.armed, snapshot.reason, session_id)
        if key == self._last_presence_key:
            return
        self._last_presence_key = key
        self._append_ops_event(
            event="wakeword_presence_changed",
            message="Wakeword presence session changed.",
            data={
                "armed": snapshot.armed,
                "reason": snapshot.reason,
                "session_id": session_id,
                "person_visible": snapshot.person_visible,
                "last_person_seen_age_s": snapshot.last_person_seen_age_s,
                "last_motion_age_s": snapshot.last_motion_age_s,
                "last_speech_age_s": snapshot.last_speech_age_s,
            },
        )

    def _record_wakeword_attempt(
        self,
        match: WakewordMatch,
        *,
        presence_snapshot: PresenceSessionSnapshot,
        decision: WakewordDecision | None = None,
    ) -> None:
        """Record one wakeword attempt and optional policy outcome."""

        data = {
            "detected": match.detected,
            "matched_phrase": match.matched_phrase,
            "remaining_text": match.remaining_text,
            "presence_reason": presence_snapshot.reason,
            "wakeword_armed": presence_snapshot.armed,
            "backend": match.backend,
            "detector_label": match.detector_label,
            "score": match.score,
        }
        if decision is not None:
            data.update(
                {
                    "outcome": decision.outcome,
                    "source": decision.source,
                    "primary_backend": decision.primary_backend,
                    "backend_used": decision.backend_used,
                    "fallback_backend": decision.fallback_backend,
                    "verifier_mode": decision.verifier_mode,
                    "verifier_used": decision.verifier_used,
                    "verifier_status": decision.verifier_status,
                    "capture_path": decision.capture_path,
                }
            )
            if decision.verifier_reason:
                data["verifier_reason"] = decision.verifier_reason
        if match.transcript:
            data["transcript_preview"] = match.transcript[:160]
        if match.normalized_transcript:
            data["normalized_transcript_preview"] = match.normalized_transcript[:160]
        self._append_ops_event(
            event="wakeword_attempted",
            message="Wakeword phrase spotting inspected recent ambient speech.",
            data=data,
        )

    def _record_wakeword_decision(
        self,
        *,
        decision: WakewordDecision,
        presence_snapshot: PresenceSessionSnapshot,
    ) -> None:
        """Record the final wakeword policy decision for one attempt."""

        self._append_ops_event(
            event="wakeword_decision",
            message="Wakeword policy decided whether Twinr should open a turn.",
            data={
                "detected": decision.detected,
                "outcome": decision.outcome,
                "source": decision.source,
                "primary_backend": decision.primary_backend,
                "backend_used": decision.backend_used,
                "fallback_backend": decision.fallback_backend,
                "verifier_mode": decision.verifier_mode,
                "verifier_used": decision.verifier_used,
                "verifier_status": decision.verifier_status,
                "verifier_reason": decision.verifier_reason,
                "matched_phrase": decision.match.matched_phrase,
                "remaining_text": decision.match.remaining_text,
                "detector_label": decision.match.detector_label,
                "score": decision.match.score,
                "capture_path": decision.capture_path,
                "presence_reason": presence_snapshot.reason,
                "wakeword_armed": presence_snapshot.armed,
            },
        )

    def _record_trigger_detected(
        self,
        decision: SocialTriggerDecision,
        *,
        observation: SocialObservation,
        review: ProactiveVisionReview | None = None,
    ) -> None:
        """Record one proactive trigger that reached dispatch evaluation."""

        data = {
            "trigger": decision.trigger_id,
            "reason": decision.reason,
            "priority": int(decision.priority),
            "prompt": decision.prompt,
            "score": decision.score,
            "threshold": decision.threshold,
            "evidence": [item.to_dict() for item in decision.evidence],
            "person_visible": observation.vision.person_visible,
            "body_pose": observation.vision.body_pose.value,
            "speech_detected": observation.audio.speech_detected,
            "distress_detected": observation.audio.distress_detected,
            "low_motion": observation.low_motion,
        }
        if review is not None:
            data.update(
                {
                    "vision_review_decision": review.decision,
                    "vision_review_confidence": review.confidence,
                    "vision_review_reason": review.reason,
                    "vision_review_scene": review.scene,
                    "vision_review_frame_count": review.frame_count,
                    "vision_review_response_id": review.response_id,
                    "vision_review_request_id": review.request_id,
                    "vision_review_model": review.model,
                }
            )
        self._append_ops_event(
            event="proactive_trigger_detected",
            message="Proactive trigger conditions were met.",
            data=data,
        )

    def _record_vision_review(
        self,
        decision: SocialTriggerDecision,
        *,
        review: ProactiveVisionReview,
    ) -> None:
        """Record one buffered vision review result."""

        self._append_ops_event(
            event="proactive_vision_reviewed",
            message="Buffered proactive camera frames were reviewed before speaking.",
            data={
                "trigger": decision.trigger_id,
                "approved": review.approved,
                "decision": review.decision,
                "confidence": review.confidence,
                "reason": review.reason,
                "scene": review.scene,
                "frame_count": review.frame_count,
                "response_id": review.response_id,
                "request_id": review.request_id,
                "model": review.model,
            },
        )

    def _record_trigger_skipped_vision_review(
        self,
        decision: SocialTriggerDecision,
        *,
        review: ProactiveVisionReview,
    ) -> None:
        """Record that buffered vision review rejected one trigger."""

        self._emit("social_trigger_skipped=vision_review_rejected")
        self._append_ops_event(
            event="social_trigger_skipped",
            message="Social trigger prompt was skipped because buffered frame review rejected it.",
            data={
                "trigger": decision.trigger_id,
                "reason": "vision_review_rejected",
                "priority": int(decision.priority),
                "score": decision.score,
                "threshold": decision.threshold,
                "vision_review_decision": review.decision,
                "vision_review_confidence": review.confidence,
                "vision_review_reason": review.reason,
                "vision_review_scene": review.scene,
                "vision_review_frame_count": review.frame_count,
                "vision_review_response_id": review.response_id,
                "vision_review_request_id": review.request_id,
                "vision_review_model": review.model,
            },
        )

    def _record_trigger_skipped_vision_review_unavailable(
        self,
        decision: SocialTriggerDecision,
    ) -> None:
        """Record that buffered vision review was unavailable for one trigger."""

        self._emit("social_trigger_skipped=vision_review_unavailable")
        self._append_ops_event(
            event="social_trigger_skipped",
            message="Social trigger prompt was skipped because buffered frame review was unavailable.",
            data={
                "trigger": decision.trigger_id,
                "reason": "vision_review_unavailable",
                "priority": int(decision.priority),
                "score": decision.score,
                "threshold": decision.threshold,
            },
        )

    def _record_trigger_skipped_recent_wakeword_attempt(
        self,
        decision: SocialTriggerDecision,
        *,
        wakeword_attempt_age_s: float,
    ) -> None:
        """Record that a recent wakeword attempt suppressed one trigger."""

        self._emit("social_trigger_skipped=recent_wakeword_attempt")
        self._append_ops_event(
            event="social_trigger_skipped",
            message="Social trigger prompt was skipped because a wakeword attempt was seen recently.",
            data={
                "trigger": decision.trigger_id,
                "reason": "recent_wakeword_attempt",
                "wakeword_attempt_age_s": round(wakeword_attempt_age_s, 3),
                "priority": int(decision.priority),
                "score": decision.score,
                "threshold": decision.threshold,
            },
        )

    def _presence_session_block_reason(
        self,
        decision: SocialTriggerDecision,
        *,
        presence_snapshot: PresenceSessionSnapshot,
    ) -> str | None:
        """Return a presence-session block reason for one trigger if any."""

        session_id = getattr(presence_snapshot, "session_id", None)
        if decision.trigger_id != "possible_fall":
            return None
        if not self.config.proactive_possible_fall_once_per_presence_session:
            return None
        if not presence_snapshot.armed or session_id is None:
            return None
        if self._last_possible_fall_prompted_session_id != session_id:
            return None
        return "already_prompted_this_presence_session"

    def _record_trigger_skipped_presence_session(
        self,
        decision: SocialTriggerDecision,
        *,
        presence_snapshot: PresenceSessionSnapshot,
        reason: str,
    ) -> None:
        """Record that a trigger was skipped by per-session suppression."""

        session_id = getattr(presence_snapshot, "session_id", None)
        self._emit(f"social_trigger_skipped={reason}")
        self._append_ops_event(
            event="social_trigger_skipped",
            message="Social trigger prompt was skipped because it already fired in the current presence session.",
            data={
                "trigger": decision.trigger_id,
                "reason": reason,
                "presence_session_id": session_id,
                "presence_reason": presence_snapshot.reason,
                "priority": int(decision.priority),
                "score": decision.score,
                "threshold": decision.threshold,
            },
        )

    def _dispatch_automation_observation(self, observation: SocialObservation, *, inspected: bool) -> None:
        """Publish one normalized observation to the automation observer hook."""

        if self.observation_handler is None:
            return
        try:
            camera_update = self._observe_camera_surface(observation, inspected=inspected)
            facts = self._build_automation_facts(
                observation,
                inspected=inspected,
                camera_snapshot=camera_update.snapshot,
            )
            event_names = self._derive_sensor_events(facts, camera_event_names=camera_update.event_names)
            self.observation_handler(facts, event_names)
        except Exception as exc:
            self._record_fault(
                event="proactive_observation_handler_failed",
                message="Automation observation dispatch failed.",
                error=exc,
            )

    def _observe_camera_surface(
        self,
        observation: SocialObservation,
        *,
        inspected: bool,
    ) -> ProactiveCameraSurfaceUpdate:
        """Project one raw vision observation onto the stabilized camera surface."""

        return self._camera_surface.observe(
            inspected=inspected,
            observed_at=observation.observed_at,
            observation=observation.vision,
        )

    def _build_automation_facts(
        self,
        observation: SocialObservation,
        *,
        inspected: bool,
        camera_snapshot: ProactiveCameraSnapshot,
    ) -> dict[str, Any]:
        """Build the automation-facing fact payload for one observation."""

        now = observation.observed_at
        audio = observation.audio

        speech_detected = audio.speech_detected is True
        quiet = audio.speech_detected is False
        self._speech_detected_since = self._next_since(speech_detected, self._speech_detected_since, now)
        self._quiet_since = self._next_since(quiet, self._quiet_since, now)

        no_motion_for_s = 0.0
        if not observation.pir_motion_detected and self._last_motion_at is not None:
            no_motion_for_s = max(0.0, now - self._last_motion_at)

        return {
            "sensor": {
                "inspected": inspected,
                "observed_at": now,
            },
            "pir": {
                "motion_detected": observation.pir_motion_detected,
                "low_motion": observation.low_motion,
                "no_motion_for_s": round(no_motion_for_s, 3),
            },
            "camera": camera_snapshot.to_automation_facts(),
            "vad": {
                "speech_detected": speech_detected,
                "speech_detected_for_s": round(self._duration_since(self._speech_detected_since, now), 3),
                "quiet": quiet,
                "quiet_for_s": round(self._duration_since(self._quiet_since, now), 3),
                "distress_detected": audio.distress_detected is True,
            },
        }

    def _derive_sensor_events(
        self,
        facts: dict[str, Any],
        *,
        camera_event_names: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Return rising-edge event names derived from the latest fact payload."""

        current_flags = {
            "pir.motion_detected": bool(facts["pir"]["motion_detected"]),
            "vad.speech_detected": bool(facts["vad"]["speech_detected"]),
        }
        event_names: list[str] = list(camera_event_names)
        for key, value in current_flags.items():
            previous = self._last_sensor_flags.get(key)
            if value and previous is not True:
                event_names.append(key)
        self._last_sensor_flags = current_flags
        return tuple(event_names)

    def _next_since(self, active: bool, since: float | None, now: float) -> float | None:
        """Advance or clear one duration anchor depending on activity."""

        if active:
            return now if since is None else since
        return None

    def _duration_since(self, since: float | None, now: float) -> float:
        """Return the elapsed duration for one optional activity anchor."""

        if since is None:
            return 0.0
        return max(0.0, now - since)


class ProactiveMonitorService:
    """Run the proactive coordinator in a bounded background worker."""

    def __init__(
        self,
        coordinator: ProactiveCoordinator,
        *,
        poll_interval_s: float,
        wakeword_stream: OpenWakeWordStreamingMonitor | None = None,
        emit: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize one monitor service around a configured coordinator."""

        self.coordinator = coordinator
        self.poll_interval_s = max(0.2, poll_interval_s)
        self.wakeword_stream = wakeword_stream
        self.emit = emit or (lambda _line: None)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._lifecycle_lock = Lock()  # AUDIT-FIX(#3): serialize open/close so concurrent lifecycle calls cannot double-start or half-close the worker.
        self._resources_open = False
        self._close_join_timeout_s = _DEFAULT_CLOSE_JOIN_TIMEOUT_S

    # AUDIT-FIX(#1): Safe telemetry wrappers for service lifecycle and worker faults.
    def _emit(self, line: str) -> None:
        """Emit one service-local telemetry line safely."""

        _safe_emit(self.emit, line)

    # AUDIT-FIX(#1): Safe telemetry wrappers for service lifecycle and worker faults.
    def _append_ops_event(
        self,
        *,
        event: str,
        message: str,
        data: dict[str, Any],
        level: str | None = None,
    ) -> None:
        """Append one service-local ops event safely."""

        _append_ops_event(
            self.coordinator.runtime,
            event=event,
            message=message,
            data=data,
            level=level,
            emit=self.emit,
        )

    # AUDIT-FIX(#3): Close optional resources defensively because shutdown cleanup must not raise while handling a stop request.
    def _safe_close_resource(self, resource: Any, *, name: str) -> None:
        """Close one optional resource while suppressing cleanup failures."""

        if resource is None:
            return
        close = getattr(resource, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception as exc:
            self._append_ops_event(
                event="proactive_resource_close_failed",
                level="error",
                message="Failed to close a proactive monitor resource cleanly.",
                data={
                    "resource": name,
                    "error": _exception_text(exc),
                },
            )

    # AUDIT-FIX(#3): Open hardware resources under the lifecycle lock so partial startups are unwound immediately.
    def _open_resources_locked(self) -> None:
        """Open hardware resources under the lifecycle lock."""

        if self._resources_open:
            return
        opened_pir = False
        opened_stream = False
        try:
            if self.coordinator.pir_monitor is not None:
                self.coordinator.pir_monitor.open()
                opened_pir = True
            if self.wakeword_stream is not None:
                self.wakeword_stream.open()
                opened_stream = True
            self._resources_open = True
        except Exception as exc:
            if opened_stream:
                self._safe_close_resource(self.wakeword_stream, name="wakeword_stream")
            if opened_pir:
                self._safe_close_resource(self.coordinator.pir_monitor, name="pir_monitor")
            self._resources_open = False
            self._append_ops_event(
                event="proactive_monitor_start_failed",
                level="error",
                message="Failed to open proactive monitor resources.",
                data={"error": _exception_text(exc)},
            )
            raise

    # AUDIT-FIX(#3): Closing resources before join helps unblock hardware/audio backends during shutdown.
    def _close_resources_locked(self) -> None:
        """Close any opened hardware resources under the lifecycle lock."""

        if not self._resources_open:
            return
        self._safe_close_resource(self.coordinator.audio_observer, name="audio_observer")
        self._safe_close_resource(self.wakeword_stream, name="wakeword_stream")
        self._safe_close_resource(self.coordinator.pir_monitor, name="pir_monitor")
        self._resources_open = False

    def open(self) -> "ProactiveMonitorService":
        """Open resources and start the background proactive worker."""

        with self._lifecycle_lock:
            if self._thread is not None and not self._thread.is_alive():
                self._thread = None
            if self._thread is not None:
                return self
            self._open_resources_locked()
            self._stop_event.clear()
            thread = Thread(target=self._run, daemon=True, name="twinr-proactive")
            self._thread = thread
            try:
                thread.start()
            except Exception as exc:
                self._thread = None
                self._close_resources_locked()
                self._append_ops_event(
                    event="proactive_monitor_start_failed",
                    level="error",
                    message="Failed to start the proactive monitor worker thread.",
                    data={"error": _exception_text(exc)},
                )
                raise
            self._append_ops_event(
                event="proactive_monitor_started",
                message="Proactive monitor started.",
                data={
                    "poll_interval_s": self.poll_interval_s,
                    "wakeword_streaming": self.wakeword_stream is not None,
                },
            )
            self._emit("proactive_monitor=started")
            return self

    def close(self) -> None:
        """Request worker shutdown and close monitor resources."""

        thread_to_join: Thread | None = None
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None and not self._resources_open:
                return
            self._stop_event.set()
            self._close_resources_locked()
            if thread is current_thread():
                self._append_ops_event(
                    event="proactive_monitor_stop_requested",
                    message="Proactive monitor stop was requested from the worker thread.",
                    data={},
                )
                self._emit("proactive_monitor=stopping")
                return
            thread_to_join = thread
        if thread_to_join is not None:
            thread_to_join.join(timeout=self._close_join_timeout_s)
            if thread_to_join.is_alive():
                self._append_ops_event(
                    event="proactive_monitor_stop_timeout",
                    level="error",
                    message="Proactive monitor worker did not stop within the shutdown budget.",
                    data={"join_timeout_s": self._close_join_timeout_s},
                )
                self._emit("proactive_monitor=stop_timeout")

    def __enter__(self) -> "ProactiveMonitorService":
        """Enter the monitor context by starting the service."""

        return self.open()  # AUDIT-FIX(#3): centralize all startup/open logic in one path so resource handling stays consistent.

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit the monitor context by stopping the service."""

        self.close()

    def _run(self) -> None:
        """Run the wakeword drain and proactive tick loop until stopped."""

        next_tick_at = 0.0
        try:
            while not self._stop_event.is_set():
                did_work = False
                try:
                    for _ in range(_MAX_WAKEWORD_STREAM_EVENTS_PER_CYCLE):  # AUDIT-FIX(#10): bound per-cycle stream draining so a noisy stream cannot starve normal proactive ticks.
                        if self.coordinator.poll_wakeword_stream() is None:
                            break
                        did_work = True
                except Exception as exc:
                    error_text = _exception_text(exc)
                    self._emit(f"proactive_error={error_text}")
                    self._append_ops_event(
                        event="proactive_error",
                        level="error",
                        message="Proactive wakeword stream handling failed.",
                        data={"error": error_text},
                    )
                now = time.monotonic()
                if now >= next_tick_at:
                    try:
                        self.coordinator.tick()
                        did_work = True
                    except Exception as exc:
                        error_text = _exception_text(exc)
                        self._emit(f"proactive_error={error_text}")
                        self._append_ops_event(
                            event="proactive_error",
                            level="error",
                            message="Proactive monitor tick failed.",
                            data={"error": error_text},
                        )
                    next_tick_at = time.monotonic() + self.poll_interval_s
                wait_s = max(0.02, min(0.05, max(0.0, next_tick_at - time.monotonic())))
                if did_work:
                    wait_s = 0.02
                if self._stop_event.wait(wait_s):
                    return
        finally:
            with self._lifecycle_lock:
                if self._thread is current_thread():
                    self._thread = None
                self._close_resources_locked()
                self._append_ops_event(
                    event="proactive_monitor_stopped",
                    message="Proactive monitor stopped.",
                    data={},
                )
                self._emit("proactive_monitor=stopped")


def build_default_proactive_monitor(
    *,
    config: TwinrConfig,
    runtime: TwinrRuntime,
    backend: OpenAIBackend,
    camera: V4L2StillCamera,
    camera_lock: Lock | None,
    audio_lock: Lock | None,
    trigger_handler: Callable[[SocialTriggerDecision], bool],
    wakeword_handler: Callable[[WakewordMatch], bool] | None = None,
    idle_predicate: Callable[[], bool] | None = None,
    observation_handler: Callable[[dict[str, Any], tuple[str, ...]], None] | None = None,
    emit: Callable[[str], None] | None = None,
) -> ProactiveMonitorService | None:
    """Build the default proactive monitor stack from Twinr runtime services."""

    config = _load_wakeword_runtime_config(config=config, runtime=runtime, emit=emit)
    if not config.proactive_enabled and not config.wakeword_enabled:
        return None

    try:
        engine: SocialTriggerEngine | _NullSocialTriggerEngine = (
            SocialTriggerEngine.from_config(config)
            if config.proactive_enabled
            else _NullSocialTriggerEngine()
        )
    except Exception as exc:
        _record_component_warning(
            runtime=runtime,
            emit=emit,
            reason="social_trigger_engine_init_failed",
            detail=(
                "The social trigger engine could not be initialized. "
                f"Proactive triggers were disabled: {_exception_text(exc)}"
            ),
        )
        engine = _NullSocialTriggerEngine()  # AUDIT-FIX(#4): keep wakeword/monitoring alive even when proactive trigger engine initialization fails.

    primary_backend = normalize_wakeword_backend(
        getattr(config, "wakeword_primary_backend", config.wakeword_backend),
        default="openwakeword",
    )
    fallback_backend = normalize_wakeword_backend(
        getattr(config, "wakeword_fallback_backend", "stt"),
        default="stt",
    )
    use_openwakeword_stream = config.wakeword_enabled and primary_backend == "openwakeword"

    pir_monitor = None
    if config.pir_enabled:
        try:
            pir_monitor = configured_pir_monitor(config)
        except Exception as exc:
            _record_component_warning(
                runtime=runtime,
                emit=emit,
                reason="pir_init_failed",
                detail=f"PIR monitoring could not be initialized: {_exception_text(exc)}",
            )
    else:
        _record_component_warning(
            runtime=runtime,
            emit=emit,
            reason="pir_unconfigured",
            detail=(
                "PIR is not configured. Motion-gated camera inspection is disabled, "
                "but wakeword and audio-only proactive logic can still run."
            ),
        )

    presence_session = None
    wakeword_spotter = None
    wakeword_stream = None
    wakeword_policy = None
    if config.wakeword_enabled:
        presence_session = PresenceSessionController(
            presence_grace_s=config.wakeword_presence_grace_s,
            motion_grace_s=config.wakeword_motion_grace_s,
            speech_grace_s=config.wakeword_speech_grace_s,
        )
        wakeword_policy = _build_wakeword_policy(
            config=config,
            backend=backend,
        )
        if use_openwakeword_stream:
            wakeword_stream = _build_openwakeword_stream(
                config=config,
                runtime=runtime,
                emit=emit,
            )
        wakeword_spotter = _build_wakeword_spotter(
            config=config,
            runtime=runtime,
            backend=backend,
            emit=emit,
            selected_backend=primary_backend,
            fallback_backend=fallback_backend,
        )

    audio_observer = NullAudioObservationProvider()
    audio_observer_fallback_factory: Callable[[], Any] | None = None
    distress_enabled = bool(
        config.proactive_enabled and config.proactive_audio_distress_enabled
    )  # AUDIT-FIX(#5): do not run proactive distress classification when proactive mode is disabled.
    if config.proactive_audio_enabled or config.wakeword_enabled:
        if wakeword_stream is not None:
            try:
                audio_observer = AmbientAudioObservationProvider(
                    sampler=wakeword_stream,
                    audio_lock=None,
                    sample_ms=config.proactive_audio_sample_ms,
                    distress_enabled=distress_enabled,
                )
            except Exception as exc:
                _record_component_warning(
                    runtime=runtime,
                    emit=emit,
                    reason="audio_observer_init_failed",
                    detail=f"Streaming audio observer initialization failed: {_exception_text(exc)}",
                )
                audio_observer = NullAudioObservationProvider()

            # AUDIT-FIX(#6): prepare a secondary non-streaming observer so runtime stream failures can recover to a working audio path.
            def _fallback_audio_observer_factory() -> Any:
                sampler = AmbientAudioSampler.from_config(config)
                return AmbientAudioObservationProvider(
                    sampler=sampler,
                    audio_lock=audio_lock,
                    sample_ms=config.proactive_audio_sample_ms,
                    distress_enabled=distress_enabled,
                )

            audio_observer_fallback_factory = _fallback_audio_observer_factory
        else:
            try:
                sampler = AmbientAudioSampler.from_config(config)
                audio_observer = AmbientAudioObservationProvider(
                    sampler=sampler,
                    audio_lock=audio_lock,
                    sample_ms=config.proactive_audio_sample_ms,
                    distress_enabled=distress_enabled,
                )
            except Exception as exc:
                _record_component_warning(
                    runtime=runtime,
                    emit=emit,
                    reason="audio_sampler_init_failed",
                    detail=f"Ambient audio sampling could not be initialized: {_exception_text(exc)}",
                )
                audio_observer = NullAudioObservationProvider()

    vision_observer = None
    if config.proactive_enabled:
        try:
            vision_observer = OpenAIVisionObservationProvider(
                backend=backend,
                camera=camera,
                camera_lock=camera_lock,
            )
        except Exception as exc:
            _record_component_warning(
                runtime=runtime,
                emit=emit,
                reason="vision_observer_init_failed",
                detail=f"Vision observation provider initialization failed: {_exception_text(exc)}",
            )

    vision_reviewer = None
    if config.proactive_enabled and config.proactive_vision_review_enabled and vision_observer is not None:
        try:
            vision_reviewer = OpenAIProactiveVisionReviewer(
                backend=backend,
                frame_buffer=ProactiveVisionFrameBuffer(
                    max_items=config.proactive_vision_review_buffer_frames,
                ),
                max_frames=config.proactive_vision_review_max_frames,
                max_age_s=config.proactive_vision_review_max_age_s,
                min_spacing_s=config.proactive_vision_review_min_spacing_s,
            )
        except Exception as exc:
            _record_component_warning(
                runtime=runtime,
                emit=emit,
                reason="vision_reviewer_init_failed",
                detail=f"Buffered vision reviewer initialization failed: {_exception_text(exc)}",
            )

    if (
        pir_monitor is None
        and wakeword_stream is None
        and isinstance(audio_observer, NullAudioObservationProvider)
    ):
        _record_component_warning(
            runtime=runtime,
            emit=emit,
            reason="no_operational_sensor_path",
            detail=(
                "No operational PIR, streaming wakeword, or ambient audio path could be initialized. "
                "The proactive monitor was not started."
            ),
        )
        return None  # AUDIT-FIX(#4): avoid starting an inert monitor that gives a false sense of protection.

    coordinator = ProactiveCoordinator(
        config=config,
        runtime=runtime,
        engine=engine,
        trigger_handler=trigger_handler,
        vision_observer=vision_observer,
        audio_observer=audio_observer,
        audio_observer_fallback_factory=audio_observer_fallback_factory,
        presence_session=presence_session,
        wakeword_spotter=wakeword_spotter,
        wakeword_stream=wakeword_stream,
        wakeword_policy=wakeword_policy,
        vision_reviewer=vision_reviewer,
        wakeword_handler=wakeword_handler,
        pir_monitor=pir_monitor,
        idle_predicate=idle_predicate,
        observation_handler=observation_handler,
        emit=emit,
    )
    return ProactiveMonitorService(
        coordinator,
        poll_interval_s=config.proactive_poll_interval_s,
        wakeword_stream=wakeword_stream,
        emit=emit,
    )


def _load_wakeword_runtime_config(
    *,
    config: TwinrConfig,
    runtime: TwinrRuntime,
    emit: Callable[[str], None] | None,
) -> TwinrConfig:
    """Apply stored wakeword calibration to the runtime config when enabled."""

    if not config.wakeword_enabled:
        return config
    try:
        profile = WakewordCalibrationStore.from_config(config).load()
    except Exception as exc:
        _record_component_warning(
            runtime=runtime,
            emit=emit,
            reason="wakeword_calibration_load_failed",
            detail=f"Wakeword calibration profile could not be loaded: {_exception_text(exc)}",
        )
        return config
    if profile is None:
        return config
    return apply_wakeword_calibration(config, profile)


def _build_wakeword_policy(
    *,
    config: TwinrConfig,
    backend: OpenAIBackend,
) -> WakewordDecisionPolicy:
    """Build the runtime wakeword policy and optional STT verifier."""

    verifier = None
    if config.wakeword_verifier_mode != "disabled":
        verifier = SttWakewordVerifier(
            backend=backend,
            phrases=config.wakeword_phrases,
            language=config.openai_realtime_language,
        )
    return WakewordDecisionPolicy(
        primary_backend=config.wakeword_primary_backend,
        fallback_backend=config.wakeword_fallback_backend,
        verifier_mode=config.wakeword_verifier_mode,
        verifier_margin=config.wakeword_verifier_margin,
        primary_threshold=config.wakeword_openwakeword_threshold,
        verifier=verifier,
    )


def _build_openwakeword_stream(
    *,
    config: TwinrConfig,
    runtime: TwinrRuntime,
    emit: Callable[[str], None] | None,
) -> OpenWakeWordStreamingMonitor | None:
    """Build the streaming openWakeWord monitor when runtime constraints match."""

    models = _normalize_text_tuple(config.wakeword_openwakeword_models)
    if config.audio_sample_rate != 16000 or config.audio_channels != 1:
        _record_wakeword_backend_warning(
            runtime=runtime,
            emit=emit,
            reason="openwakeword_requires_16khz_mono",
            detail=(
                "openWakeWord needs 16 kHz mono audio. Configure "
                "TWINR_AUDIO_SAMPLE_RATE=16000 and TWINR_AUDIO_CHANNELS=1."
            ),
            fallback_backend=config.wakeword_fallback_backend,
        )
        return None
    if not models:
        _record_wakeword_backend_warning(
            runtime=runtime,
            emit=emit,
            reason="openwakeword_models_missing",
            detail=(
                "openWakeWord backend selected, but no models were configured. "
                "Set TWINR_WAKEWORD_OPENWAKEWORD_MODELS to model names or file paths."
            ),
            fallback_backend=config.wakeword_fallback_backend,
        )
        return None
    try:
        from twinr.proactive.wakeword.spotter import WakewordOpenWakeWordFrameSpotter

        return OpenWakeWordStreamingMonitor(
            device=_normalize_optional_text(
                config.proactive_audio_input_device,
                config.audio_input_device,
            ),  # AUDIT-FIX(#8): avoid None.strip() when no explicit audio input device is configured.
            sample_rate=config.audio_sample_rate,
            channels=config.audio_channels,
            spotter=WakewordOpenWakeWordFrameSpotter(
                wakeword_models=models,
                phrases=config.wakeword_phrases,
                threshold=config.wakeword_openwakeword_threshold,
                vad_threshold=config.wakeword_openwakeword_vad_threshold,
                patience_frames=config.wakeword_openwakeword_patience_frames,
                activation_samples=config.wakeword_openwakeword_activation_samples,
                deactivation_threshold=config.wakeword_openwakeword_deactivation_threshold,
                enable_speex_noise_suppression=config.wakeword_openwakeword_enable_speex,
                inference_framework=config.wakeword_openwakeword_inference_framework,
            ),
            attempt_cooldown_s=config.wakeword_attempt_cooldown_s,
            speech_threshold=config.audio_speech_threshold,
            emit=emit,
        )
    except Exception as exc:
        _record_wakeword_backend_warning(
            runtime=runtime,
            emit=emit,
            reason="openwakeword_init_failed",
            detail=f"openWakeWord streaming initialization failed: {_exception_text(exc)}",
            fallback_backend=config.wakeword_fallback_backend,
        )
        return None


def _build_wakeword_spotter(
    *,
    config: TwinrConfig,
    runtime: TwinrRuntime,
    backend: OpenAIBackend,
    emit: Callable[[str], None] | None,
    selected_backend: str | None = None,
    fallback_backend: str | None = None,
):
    """Build the non-streaming wakeword spotter for the selected backend."""

    selected_backend = normalize_wakeword_backend(
        selected_backend or config.wakeword_primary_backend or config.wakeword_backend,
        default="openwakeword",
    )
    fallback_backend = normalize_wakeword_backend(
        fallback_backend or config.wakeword_fallback_backend,
        default="stt",
    )
    if selected_backend == "openwakeword":
        models = _normalize_text_tuple(config.wakeword_openwakeword_models)
        if not models:
            _record_wakeword_backend_warning(
                runtime=runtime,
                emit=emit,
                reason="openwakeword_models_missing",
                detail=(
                    "openWakeWord backend selected, but no models were configured. "
                    "Set TWINR_WAKEWORD_OPENWAKEWORD_MODELS to model names or file paths."
                ),
                fallback_backend=fallback_backend,
            )
        else:
            try:
                return WakewordOpenWakeWordSpotter(
                    wakeword_models=models,
                    phrases=config.wakeword_phrases,
                    threshold=config.wakeword_openwakeword_threshold,
                    vad_threshold=config.wakeword_openwakeword_vad_threshold,
                    patience_frames=config.wakeword_openwakeword_patience_frames,
                    activation_samples=config.wakeword_openwakeword_activation_samples,
                    deactivation_threshold=config.wakeword_openwakeword_deactivation_threshold,
                    enable_speex_noise_suppression=config.wakeword_openwakeword_enable_speex,
                    inference_framework=config.wakeword_openwakeword_inference_framework,
                    backend=None,
                    language=config.openai_realtime_language,
                    transcribe_on_detect=False,
                )
            except Exception as exc:
                _record_wakeword_backend_warning(
                    runtime=runtime,
                    emit=emit,
                    reason="openwakeword_init_failed",
                    detail=f"openWakeWord initialization failed: {_exception_text(exc)}",
                    fallback_backend=fallback_backend,
                )
        if fallback_backend != "stt":
            return None
    if selected_backend == "disabled":
        return None
    return WakewordPhraseSpotter(
        backend=backend,
        phrases=config.wakeword_phrases,
        language=config.openai_realtime_language,
    )


def _record_wakeword_backend_warning(
    *,
    runtime: TwinrRuntime,
    emit: Callable[[str], None] | None,
    reason: str,
    detail: str,
    fallback_backend: str,
) -> None:
    """Record one warning about falling back from the preferred wakeword backend."""

    _safe_emit(emit, f"wakeword_backend_warning={reason}")  # AUDIT-FIX(#1): backend fallback warnings must not fail open/close paths when emit sinks are broken.
    _append_ops_event(
        runtime,
        event="wakeword_backend_warning",
        level="warning",
        message="Wakeword backend configuration could not use the preferred local detector.",
        data={
            "reason": reason,
            "detail": detail,
            "fallback_backend": fallback_backend,
        },
        emit=emit,
    )


__all__ = [
    "ProactiveCoordinator",
    "ProactiveMonitorService",
    "ProactiveTickResult",
    "build_default_proactive_monitor",
]
