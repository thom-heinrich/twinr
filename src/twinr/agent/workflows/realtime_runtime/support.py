"""Provide shared support helpers for realtime-style workflow loops."""

from __future__ import annotations

from contextlib import nullcontext
import time
import uuid
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, RLock, Thread, current_thread
from typing import Callable

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.conversation.closure import (
    ConversationClosureEvaluator,
    StructuredConversationClosureEvaluator,
    ToolCallingConversationClosureEvaluator,
)
from twinr.agent.base_agent.conversation.turn_controller import ToolCallingTurnDecisionEvaluator
from twinr.hardware.household_identity import HouseholdIdentityManager
from twinr.memory.longterm.storage.remote_read_diagnostics import extract_remote_write_context
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.agent.workflows.forensics import WorkflowForensics
from twinr.agent.workflows.playback_coordinator import PlaybackCoordinator, PlaybackPriority
from twinr.agent.workflows.realtime_runtime import required_remote_support, vision_support
from twinr.agent.workflows.required_remote_snapshot import (
    assess_required_remote_watchdog_snapshot,
    ensure_required_remote_watchdog_snapshot_ready,
)
from twinr.agent.workflows.working_feedback import WorkingFeedbackKind, start_working_feedback_loop
from twinr.proactive.runtime.runtime_contract import ReSpeakerRuntimeContractError
from twinr.providers.openai import (
    OpenAIConversationClosureDecisionProvider,
    OpenAIImageInput,
    OpenAIToolCallingAgentProvider,
)

_SEARCH_FEEDBACK_TONE_PATTERNS: tuple[tuple[tuple[int, int], ...], ...] = (
    (
        (698, 55),
        (932, 45),
        (784, 60),
    ),
    (
        (988, 40),
        (740, 50),
        (880, 55),
    ),
)
_ALLOWED_REFERENCE_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
)
_DEFAULT_REFERENCE_IMAGE_MAX_BYTES = 10 * 1024 * 1024
_DEFAULT_TTS_QUEUE_MAX_CHUNKS = 16
_DEFAULT_TTS_FIRST_CHUNK_TIMEOUT_SECONDS = 20.0
_DEFAULT_TTS_STREAM_CHUNK_TIMEOUT_SECONDS = 15.0
_DEFAULT_STOP_JOIN_TIMEOUT_SECONDS = 2.0
_DEFAULT_REQUIRED_REMOTE_HEALTHCHECK_INTERVAL_SECONDS = 5.0
_NO_SPEECH_TIMEOUT_MARKERS: tuple[str, ...] = (
    "no speech detected before timeout",
    "no speech detected",
    "speech timeout",
    "timeout waiting for speech",
    "timeout waiting for user speech",
    "no input audio received",
)

def _default_emit(line: str) -> None:
    """Print one workflow telemetry line to stdout."""
    print(line, flush=True)


class TwinrRealtimeSupportMixin:
    """Share guarded emit, media, config, and feedback helpers.

    These helpers are reused by the realtime and streaming workflow loops so
    the session classes can stay focused on orchestration.
    """

    # AUDIT-FIX(#4,#5,#10): Lazily create missing locks so mixin methods stay safe under concurrent use.
    def _get_lock(self, name: str) -> RLock:
        lock = getattr(self, name, None)
        if lock is None:
            lock = RLock()
            setattr(self, name, lock)
        return lock

    def _get_playback_coordinator(self) -> PlaybackCoordinator:
        """Return the workflow-wide speaker coordinator, creating it lazily."""

        coordinator = getattr(self, "playback_coordinator", None)
        if coordinator is None:
            coordinator = PlaybackCoordinator(
                self.player,
                emit=getattr(self, "emit", None),
                io_lock=getattr(self, "_audio_lock", None),
            )
            setattr(self, "playback_coordinator", coordinator)
        return coordinator

    def _new_workflow_trace_id(self) -> str:
        """Return one stable trace id for a workflow session."""

        return uuid.uuid4().hex

    def _workflow_trace_active_id(self) -> str | None:
        return getattr(self, "_workflow_active_trace_id", None)

    def _workflow_trace_set_active(self, trace_id: str | None) -> None:
        self._workflow_active_trace_id = trace_id

    def _trace_event(
        self,
        msg: str,
        *,
        kind: str = "workflow",
        details: dict[str, object] | None = None,
        reason: dict[str, object] | None = None,
        kpi: dict[str, object] | None = None,
        level: str = "INFO",
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        tracer = getattr(self, "workflow_forensics", None)
        if not isinstance(tracer, WorkflowForensics):
            return
        tracer.event(
            kind=kind,
            msg=msg,
            details=details,
            reason=reason,
            kpi=kpi,
            level=level,
            trace_id=trace_id or self._workflow_trace_active_id(),
            span_id=span_id,
            loc_skip=3,
        )

    def _trace_decision(
        self,
        msg: str,
        *,
        question: str,
        selected: dict[str, object],
        options: list[dict[str, object]],
        context: dict[str, object] | None = None,
        confidence: object | None = None,
        guardrails: list[str] | None = None,
        kpi_impact_estimate: dict[str, object] | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        tracer = getattr(self, "workflow_forensics", None)
        if not isinstance(tracer, WorkflowForensics):
            return
        tracer.decision(
            msg=msg,
            question=question,
            selected=selected,
            options=options,
            context=context,
            confidence=confidence,
            guardrails=guardrails,
            kpi_impact_estimate=kpi_impact_estimate,
            trace_id=trace_id or self._workflow_trace_active_id(),
            span_id=span_id,
        )

    def _trace_span(
        self,
        *,
        name: str,
        kind: str = "span",
        details: dict[str, object] | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ):
        tracer = getattr(self, "workflow_forensics", None)
        if not isinstance(tracer, WorkflowForensics):
            return nullcontext()
        return tracer.span(
            name=name,
            kind=kind,
            details=details,
            trace_id=trace_id or self._workflow_trace_active_id(),
            parent_span_id=parent_span_id,
        )

    # AUDIT-FIX(#4): Sanitize exception text before emitting it outside the process boundary.
    def _safe_error_text(self, exc: BaseException) -> str:
        message = " ".join(str(exc).split()).strip()
        if not message:
            message = exc.__class__.__name__
        lower_message = message.casefold()
        if any(
            marker in lower_message
            for marker in ("api_key", "authorization", "bearer ", "token=", "password=", "secret=")
        ):
            message = "internal error"
        if len(message) > 240:
            message = f"{message[:237]}..."
        if message == exc.__class__.__name__:
            return message
        return f"{exc.__class__.__name__}: {message}"

    def _required_remote_dependency_interval_seconds(self) -> float:
        return required_remote_support.required_remote_dependency_interval_seconds(
            self,
            default_interval_s=_DEFAULT_REQUIRED_REMOTE_HEALTHCHECK_INTERVAL_SECONDS,
        )

    def _required_remote_dependency_recovery_hold_seconds(self) -> float:
        return required_remote_support.required_remote_dependency_recovery_hold_seconds(
            self,
            default_interval_s=_DEFAULT_REQUIRED_REMOTE_HEALTHCHECK_INTERVAL_SECONDS,
        )

    def _remote_dependency_is_required(self) -> bool:
        return required_remote_support.remote_dependency_is_required(self)

    def _required_remote_dependency_uses_watchdog_artifact(self) -> bool:
        return required_remote_support.required_remote_dependency_uses_watchdog_artifact(self)

    def _best_effort_stop_player(self) -> None:
        required_remote_support.best_effort_stop_player(self)

    def _enter_required_remote_error(self, exc: BaseException | str) -> bool:
        return required_remote_support.enter_required_remote_error(
            self,
            exc,
            extract_remote_write_context=extract_remote_write_context,
            default_interval_s=_DEFAULT_REQUIRED_REMOTE_HEALTHCHECK_INTERVAL_SECONDS,
        )

    def _required_remote_dependency_current_ready(self) -> bool:
        return required_remote_support.required_remote_dependency_current_ready(self)

    def _request_required_remote_dependency_refresh(self) -> None:
        required_remote_support.request_required_remote_dependency_refresh(self)

    def _attest_watchdog_artifact_remote_ready(self) -> None:
        required_remote_support.attest_watchdog_artifact_remote_ready(self)

    def _refresh_required_remote_dependency(self, *, force: bool = False, force_sync: bool = False) -> bool:
        return required_remote_support.refresh_required_remote_dependency(
            self,
            force=force,
            force_sync=force_sync,
            ensure_watchdog_ready=ensure_required_remote_watchdog_snapshot_ready,
            assess_watchdog_snapshot=assess_required_remote_watchdog_snapshot,
            extract_remote_write_context=extract_remote_write_context,
            default_interval_s=_DEFAULT_REQUIRED_REMOTE_HEALTHCHECK_INTERVAL_SECONDS,
        )

    # AUDIT-FIX(#4): Error reporting must not throw while handling another failure.
    def _try_emit(self, line: str) -> None:
        try:
            self.emit(line)
        except Exception:
            _default_emit(line)

    # AUDIT-FIX(#2): Resolve the parent directory strictly while deferring the final component to O_NOFOLLOW open().
    def _normalize_reference_image_path(self, raw_path: str) -> Path:
        return vision_support.normalize_reference_image_path(raw_path)

    # AUDIT-FIX(#2): Optional base-dir enforcement keeps reference media inside an explicit safe area when configured.
    def _validate_reference_image_base_dir(self, path: Path) -> bool:
        return vision_support.validate_reference_image_base_dir(self, path)

    # AUDIT-FIX(#2): Open reference images without following symlinks and reject oversized/non-regular files.
    def _safe_read_reference_image_bytes(self, path: Path, *, max_bytes: int) -> bytes:
        return vision_support.safe_read_reference_image_bytes(path, max_bytes=max_bytes)

    # AUDIT-FIX(#2,#7): Guess a safe content type and keep only a basename when passing files downstream.
    def _build_image_input(self, data: bytes, *, path: Path, label: str) -> OpenAIImageInput:
        return vision_support.build_image_input(data, path=path, label=label)

    # AUDIT-FIX(#1): Support both legacy and renamed cooldown config fields with a safe default.
    def _print_button_cooldown_seconds(self) -> float:
        raw_value = getattr(
            self.config,
            "print_button_cooldown_seconds",
            getattr(self.config, "print_button_cooldown_s", 0.0),
        )
        try:
            return max(0.0, float(raw_value))
        except (TypeError, ValueError):
            return 0.0

    # AUDIT-FIX(#6): Bound stream buffering on the Raspberry Pi and validate provider output before playback.
    def _coerce_audio_chunk(self, chunk: object) -> bytes:
        if isinstance(chunk, bytes):
            return chunk
        if isinstance(chunk, bytearray):
            return bytes(chunk)
        if isinstance(chunk, memoryview):
            return chunk.tobytes()
        raise TypeError(f"TTS stream yielded unsupported chunk type: {type(chunk).__name__}")

    # AUDIT-FIX(#3): Centralize the provider list so config updates and rollbacks stay consistent.
    def _iter_config_targets(self) -> tuple[object, ...]:
        seen: set[int] = set()
        targets: list[object] = []
        for name in (
            "stt_provider",
            "agent_provider",
            "tts_provider",
            "print_backend",
            "tool_agent_provider",
            "turn_stt_provider",
            "turn_tool_agent_provider",
        ):
            provider = getattr(self, name, None)
            if provider is None:
                continue
            provider_id = id(provider)
            if provider_id in seen:
                continue
            seen.add(provider_id)
            targets.append(provider)
        return tuple(targets)

    # AUDIT-FIX(#3): Apply config only to targets that actually accept a config attribute.
    def _apply_config_to_targets(self, config: TwinrConfig) -> None:
        for provider in self._iter_config_targets():
            if hasattr(provider, "config"):
                provider.config = config
        session = getattr(self, "realtime_session", None)
        if session is not None and hasattr(session, "config"):
            session.config = config

    # AUDIT-FIX(#3): Build the turn evaluator from the new config without leaving stale state behind.
    def _build_turn_decision_evaluator(self, config: TwinrConfig) -> ToolCallingTurnDecisionEvaluator | None:
        turn_tool_agent_provider = getattr(self, "turn_tool_agent_provider", None)
        if turn_tool_agent_provider is None or not config.turn_controller_enabled:
            return None
        return ToolCallingTurnDecisionEvaluator(
            config=config,
            provider=turn_tool_agent_provider,
        )

    def _build_conversation_closure_evaluator(self, config: TwinrConfig) -> ConversationClosureEvaluator | None:
        turn_tool_agent_provider = getattr(self, "turn_tool_agent_provider", None)
        if turn_tool_agent_provider is None or not config.conversation_closure_guard_enabled:
            return None
        if isinstance(turn_tool_agent_provider, OpenAIToolCallingAgentProvider):
            return StructuredConversationClosureEvaluator(
                config=config,
                provider=OpenAIConversationClosureDecisionProvider(
                    turn_tool_agent_provider.backend,
                    model_override=config.conversation_closure_model,
                    reasoning_effort_override=config.conversation_closure_reasoning_effort,
                ),
            )
        return ToolCallingConversationClosureEvaluator(
            config=config,
            provider=turn_tool_agent_provider,
        )

    def _handle_error(self, exc: Exception) -> None:
        if isinstance(exc, LongTermRemoteUnavailableError) and self._enter_required_remote_error(exc):
            return
        safe_error = self._safe_error_text(exc)  # AUDIT-FIX(#4): Never emit unsanitized exception text.
        if getattr(self, "_required_remote_dependency_error_active", False):
            # A new non-remote blocker must not inherit stale remote-recovery state,
            # otherwise the next healthy remote probe can incorrectly reset an
            # unrelated runtime error back to waiting.
            self._required_remote_dependency_error_active = False
            self._required_remote_dependency_error_message = None
            self._required_remote_dependency_recovery_started_at = None
            self._trace_event(
                "required_remote_error_cleared_for_non_remote_failure",
                kind="invariant",
                details={"error_type": type(exc).__name__},
            )
        self._trace_event(
            "workflow_error_handler_entered",
            kind="exception",
            level="ERROR",
            details={"error": safe_error, "error_type": type(exc).__name__},
        )
        try:
            self.runtime.fail(safe_error)
        except Exception as runtime_exc:  # AUDIT-FIX(#4): Preserve the original failure even if error-state persistence fails.
            _default_emit(f"runtime_fail_error={self._safe_error_text(runtime_exc)}")
        self._emit_status(force=True)
        self._try_emit(f"error={safe_error}")  # AUDIT-FIX(#4): Emission is best-effort only.
        if isinstance(exc, ReSpeakerRuntimeContractError):
            self._trace_event(
                "workflow_error_handler_hold",
                kind="exception",
                details={
                    "error": safe_error,
                    "error_type": type(exc).__name__,
                    "reason": "sticky_runtime_contract_blocker",
                },
            )
            return
        sleep_seconds = max(0.0, float(getattr(self, "error_reset_seconds", 0.0) or 0.0))
        if sleep_seconds > 0:
            try:
                self.sleep(sleep_seconds)
            except Exception as sleep_exc:  # AUDIT-FIX(#4): A broken sleep implementation must not trap the device in error handling.
                _default_emit(f"error_reset_sleep_error={self._safe_error_text(sleep_exc)}")
        try:
            self.runtime.reset_error()
        except Exception as runtime_exc:  # AUDIT-FIX(#4): Failing to clear the error state should not raise a second exception here.
            _default_emit(f"runtime_reset_error={self._safe_error_text(runtime_exc)}")
        self._emit_status(force=True)
        self._trace_event(
            "workflow_error_handler_completed",
            kind="exception",
            details={"error": safe_error},
        )

    def _current_runtime_error_matches_required_remote(self) -> bool:
        return required_remote_support.current_runtime_error_matches_required_remote(self)

    def _emit_status(self, *, force: bool = False) -> None:
        status = getattr(getattr(self.runtime, "status", None), "value", "unknown")  # AUDIT-FIX(#9): Guard the first emit when _last_status is unset or runtime is partially initialised.
        if force or status != getattr(self, "_last_status", None):
            self._try_emit(f"status={status}")  # AUDIT-FIX(#4): Status reporting must not crash the main flow.
            self._trace_event(
                "runtime_status_emitted",
                kind="metric",
                details={"status": status, "force": force},
            )
            self._last_status = status

    def _reload_live_config_from_env(self, env_path: Path) -> None:
        config_lock = self._get_lock("_config_lock")  # AUDIT-FIX(#3): Serialise live config reloads and rollbacks.
        with config_lock:
            previous_config = getattr(self, "config", None)
            previous_evaluator = getattr(self, "turn_decision_evaluator", None)
            previous_closure_evaluator = getattr(self, "conversation_closure_evaluator", None)
            try:
                updated_config = TwinrConfig.from_env(env_path)
                self.runtime.apply_live_config(updated_config)
                self.config = updated_config
                if hasattr(self, "_current_turn_audio_sample_rate"):
                    self._current_turn_audio_sample_rate = updated_config.openai_realtime_input_sample_rate
                self._apply_config_to_targets(updated_config)
                if hasattr(self, "turn_decision_evaluator"):
                    self.turn_decision_evaluator = self._build_turn_decision_evaluator(updated_config)
                if hasattr(self, "conversation_closure_evaluator"):
                    self.conversation_closure_evaluator = self._build_conversation_closure_evaluator(updated_config)
            except Exception as exc:
                if previous_config is not None:
                    try:
                        self.runtime.apply_live_config(previous_config)
                        self.config = previous_config
                        if hasattr(self, "_current_turn_audio_sample_rate"):
                            self._current_turn_audio_sample_rate = previous_config.openai_realtime_input_sample_rate
                        self._apply_config_to_targets(previous_config)
                        if hasattr(self, "turn_decision_evaluator"):
                            self.turn_decision_evaluator = previous_evaluator
                        if hasattr(self, "conversation_closure_evaluator"):
                            self.conversation_closure_evaluator = previous_closure_evaluator
                    except Exception as rollback_exc:
                        _default_emit(f"config_reload_rollback_error={self._safe_error_text(rollback_exc)}")
                self._try_emit(f"config_reload_error={self._safe_error_text(exc)}")
                return

    def _record_event(self, event: str, message: str, *, level: str = "info", **data: object) -> None:
        try:
            self.runtime.ops_events.append(event=event, message=message, level=level, data=data)
        except Exception as exc:  # AUDIT-FIX(#8): Ops-event persistence failures must not break the active interaction.
            self._try_emit(f"ops_event_error={self._safe_error_text(exc)}")

    def _record_usage(
        self,
        *,
        request_kind: str,
        source: str,
        model: str | None,
        response_id: str | None,
        request_id: str | None,
        used_web_search: bool | None,
        token_usage,
        **metadata: object,
    ) -> None:
        try:
            self.usage_store.append(
                source=source,
                request_kind=request_kind,
                model=model,
                response_id=response_id,
                request_id=request_id,
                used_web_search=used_web_search,
                token_usage=token_usage,
                metadata=metadata,
            )
        except Exception as exc:  # AUDIT-FIX(#8): Usage-accounting write failures must degrade gracefully on flaky storage.
            self._try_emit(f"usage_store_error={self._safe_error_text(exc)}")

    def _update_voice_assessment_from_pcm(self, audio_pcm: bytes) -> None:
        config = self.config  # AUDIT-FIX(#3): Snapshot config during assessment to avoid mixed live-reload reads.
        household_manager = getattr(self, "household_identity_manager", None)
        if household_manager is None:
            try:
                household_manager = HouseholdIdentityManager.from_config(
                    config,
                    camera=self.camera,
                    camera_lock=getattr(self, "_camera_lock", None),
                )
            except Exception:
                household_manager = None
            else:
                setattr(self, "household_identity_manager", household_manager)

        if household_manager is not None:
            try:
                household_assessment = household_manager.assess_voice(
                    audio_pcm,
                    sample_rate=config.openai_realtime_input_sample_rate,
                    channels=config.audio_channels,
                )
            except Exception as exc:
                self._try_emit(f"household_voice_profile_error={self._safe_error_text(exc)}")
            else:
                if household_assessment.status != "not_enrolled":
                    if not household_assessment.should_persist:
                        return
                    try:
                        self.runtime.update_user_voice_assessment(
                            status=household_assessment.status,
                            confidence=household_assessment.confidence,
                            checked_at=household_assessment.checked_at,
                            user_id=household_assessment.matched_user_id,
                            user_display_name=household_assessment.matched_user_display_name,
                            match_source="household_voice_identity",
                        )
                    except Exception as exc:
                        self._try_emit(f"voice_profile_persist_error={self._safe_error_text(exc)}")
                        return
                    self._try_emit(f"voice_profile_status={household_assessment.status}")
                    if household_assessment.confidence is not None:
                        self._try_emit(f"voice_profile_confidence={household_assessment.confidence:.2f}")
                    if household_assessment.matched_user_id:
                        self._try_emit(f"voice_profile_user_id={household_assessment.matched_user_id}")
                    return

        try:
            assessment = self.voice_profile_monitor.assess_pcm16(
                audio_pcm,
                sample_rate=config.openai_realtime_input_sample_rate,
                channels=config.audio_channels,
            )
        except Exception as exc:
            self._try_emit(f"voice_profile_error={self._safe_error_text(exc)}")  # AUDIT-FIX(#4): Sanitise provider errors.
            return
        if not assessment.should_persist:
            return
        try:
            self.runtime.update_user_voice_assessment(
                status=assessment.status,
                confidence=assessment.confidence,
                checked_at=assessment.checked_at,
                user_id=None,
                user_display_name=None,
                match_source="legacy_voice_profile",
            )
        except Exception as exc:  # AUDIT-FIX(#8): State persistence is best-effort here.
            self._try_emit(f"voice_profile_persist_error={self._safe_error_text(exc)}")
            return
        self._try_emit(f"voice_profile_status={assessment.status}")
        if assessment.confidence is not None:
            self._try_emit(f"voice_profile_confidence={assessment.confidence:.2f}")

    def _play_listen_beep(self) -> None:
        config = self.config  # AUDIT-FIX(#3): Use one config snapshot per tone.
        started = time.monotonic()
        self._trace_event(
            "listen_beep_started",
            kind="io",
            details={
                "frequency_hz": config.audio_beep_frequency_hz,
                "duration_ms": config.audio_beep_duration_ms,
                "volume": config.audio_beep_volume,
            },
        )
        try:
            self._get_playback_coordinator().play_tone(
                owner="listen_beep",
                priority=PlaybackPriority.BEEP,
                frequency_hz=config.audio_beep_frequency_hz,
                duration_ms=config.audio_beep_duration_ms,
                volume=config.audio_beep_volume,
                sample_rate=config.openai_realtime_input_sample_rate,
            )
        except Exception as exc:
            self._try_emit(f"beep_error={self._safe_error_text(exc)}")
            self._trace_event(
                "listen_beep_failed",
                kind="exception",
                level="ERROR",
                details={"error": self._safe_error_text(exc)},
                kpi={"duration_ms": round((time.monotonic() - started) * 1000.0, 3)},
            )
            return
        self._trace_event(
            "listen_beep_played",
            kind="io",
            kpi={"duration_ms": round((time.monotonic() - started) * 1000.0, 3)},
        )
        if config.audio_beep_settle_ms > 0:
            try:
                self.sleep(config.audio_beep_settle_ms / 1000.0)
            except Exception as exc:  # AUDIT-FIX(#4): Post-tone settling must not crash the interaction loop.
                self._try_emit(f"beep_settle_error={self._safe_error_text(exc)}")
                self._trace_event(
                    "listen_beep_settle_failed",
                    kind="exception",
                    level="WARN",
                    details={"error": self._safe_error_text(exc)},
                )
                return
        self._trace_event(
            "listen_beep_completed",
            kind="io",
            details={"settle_ms": config.audio_beep_settle_ms},
        )

    def _start_working_feedback_loop(self, kind: WorkingFeedbackKind) -> Callable[[], None]:
        feedback_lock = self._get_lock("_feedback_lock")  # AUDIT-FIX(#10): Coordinate stop/start mutations across threads.
        with feedback_lock:
            previous_stop = getattr(self, "_working_feedback_stop", None)
            previous_generation = int(getattr(self, "_working_feedback_generation", 0) or 0)
        if callable(previous_stop):
            try:
                previous_stop()
            except Exception as exc:  # AUDIT-FIX(#10): A broken previous stop callback must not block the new loop.
                self._try_emit(f"working_feedback_stop_error={self._safe_error_text(exc)}")
                self._trace_event(
                    "working_feedback_stop_failed",
                    kind="exception",
                    level="WARN",
                    details={"kind": kind, "error": self._safe_error_text(exc)},
                )
        config = self.config  # AUDIT-FIX(#3): Snapshot delay-related config for the new loop.
        try:
            stop = start_working_feedback_loop(
                self.player,
                kind=kind,
                sample_rate=config.openai_realtime_input_sample_rate,
                config=config,
                emit=self._try_emit,
                delay_override_ms=(
                    config.processing_feedback_delay_ms
                    if kind == "processing"
                    else None
                ),
                playback_coordinator=self._get_playback_coordinator(),
            )
        except Exception as exc:  # AUDIT-FIX(#10): Feedback audio is optional and must not take down the turn.
            self._try_emit(f"working_feedback_error={self._safe_error_text(exc)}")
            self._trace_event(
                "working_feedback_start_failed",
                kind="exception",
                level="WARN",
                details={"kind": kind, "error": self._safe_error_text(exc)},
            )
            return lambda: None
        generation = previous_generation + 1
        with feedback_lock:
            self._working_feedback_generation = generation
            self._working_feedback_stop = stop
        self._trace_event(
            "working_feedback_started",
            kind="io",
            details={"kind": kind, "generation": generation},
        )

        def stop_current() -> None:
            with feedback_lock:
                if getattr(self, "_working_feedback_generation", None) != generation:
                    self._trace_event(
                        "working_feedback_stop_skipped_stale",
                        kind="branch",
                        details={"kind": kind, "generation": generation},
                    )
                    return
                active_stop = getattr(self, "_working_feedback_stop", None)
                self._working_feedback_stop = None
            if callable(active_stop):
                try:
                    active_stop()
                except Exception as exc:  # AUDIT-FIX(#10): Feedback stop callbacks are best-effort.
                    self._try_emit(f"working_feedback_stop_error={self._safe_error_text(exc)}")
                    self._trace_event(
                        "working_feedback_stop_failed",
                        kind="exception",
                        level="WARN",
                        details={"kind": kind, "generation": generation, "error": self._safe_error_text(exc)},
                    )
                    return
            self._trace_event(
                "working_feedback_stopped",
                kind="io",
                details={"kind": kind, "generation": generation},
            )

        return stop_current

    def _stop_working_feedback(self) -> None:
        feedback_lock = self._get_lock("_feedback_lock")  # AUDIT-FIX(#10): Prevent concurrent stop/start races.
        with feedback_lock:
            active_stop = getattr(self, "_working_feedback_stop", None)
            self._working_feedback_stop = None
        if callable(active_stop):
            try:
                active_stop()
            except Exception as exc:  # AUDIT-FIX(#10): Feedback stop callbacks are best-effort.
                self._try_emit(f"working_feedback_stop_error={self._safe_error_text(exc)}")

    def _start_search_feedback_loop(self) -> Callable[[], None]:
        config = self.config  # AUDIT-FIX(#3): Freeze timing/volume values for the lifetime of this loop.
        if not config.search_feedback_tones_enabled:
            return lambda: None

        feedback_lock = self._get_lock("_feedback_lock")  # AUDIT-FIX(#10): Ensure only one search-tone loop is active at once.
        with feedback_lock:
            previous_stop = getattr(self, "_search_feedback_stop", None)
            previous_generation = int(getattr(self, "_search_feedback_generation", 0) or 0)
        if callable(previous_stop):
            previous_stop_call = getattr(previous_stop, "__call__", None)
            try:
                if callable(previous_stop_call):
                    previous_stop_call()  # pylint: disable=not-callable
            except Exception as exc:
                self._try_emit(f"search_feedback_stop_error={self._safe_error_text(exc)}")

        generation = previous_generation + 1
        stop_event = Event()
        delay_seconds = max(0.0, config.search_feedback_delay_ms / 1000.0)
        pause_seconds = max(0.12, config.search_feedback_pause_ms / 1000.0)
        join_timeout_seconds = max(
            0.1,
            float(getattr(config, "search_feedback_stop_join_timeout_seconds", _DEFAULT_STOP_JOIN_TIMEOUT_SECONDS)),
        )
        owner = "search_feedback"

        def worker() -> None:
            if stop_event.wait(delay_seconds):
                return
            pattern_index = 0
            while not stop_event.is_set():
                try:
                    pattern = _SEARCH_FEEDBACK_TONE_PATTERNS[pattern_index % len(_SEARCH_FEEDBACK_TONE_PATTERNS)]
                    pattern_index += 1
                    for frequency_hz, duration_ms in pattern:
                        if stop_event.is_set():
                            return
                        self._get_playback_coordinator().play_tone(
                            owner="search_feedback",
                            priority=PlaybackPriority.FEEDBACK,
                            frequency_hz=frequency_hz,
                            duration_ms=duration_ms,
                            volume=config.search_feedback_volume,
                            sample_rate=config.openai_realtime_input_sample_rate,
                            should_stop=stop_event.is_set,
                        )
                        if stop_event.wait(0.05):
                            return
                except Exception as exc:
                    self._try_emit(f"search_feedback_error={self._safe_error_text(exc)}")
                    return
                if stop_event.wait(pause_seconds):
                    return

        thread = Thread(target=worker, daemon=True)
        thread.start()

        def stop() -> None:
            stop_event.set()
            self._get_playback_coordinator().stop_owner(owner)
            if thread is current_thread():
                return
            thread.join(timeout=join_timeout_seconds)  # AUDIT-FIX(#5): Avoid indefinite hangs during stop/shutdown.
            if thread.is_alive():
                self._try_emit("search_feedback_warning=stop_timeout")
            with feedback_lock:
                if getattr(self, "_search_feedback_generation", None) == generation:
                    self._search_feedback_stop = None

        with feedback_lock:
            self._search_feedback_generation = generation
            self._search_feedback_stop = stop

        return stop

    def _stop_search_feedback(self) -> None:
        """Stop the active search-progress tones, even if the producer is still blocked."""

        feedback_lock = self._get_lock("_feedback_lock")
        with feedback_lock:
            active_stop = getattr(self, "_search_feedback_stop", None)
            self._search_feedback_stop = None
        if callable(active_stop):
            active_stop_call = getattr(active_stop, "__call__", None)
            try:
                if callable(active_stop_call):
                    active_stop_call()  # pylint: disable=not-callable
            except Exception as exc:
                self._try_emit(f"search_feedback_stop_error={self._safe_error_text(exc)}")

    def _play_streaming_tts_with_feedback(
        self,
        text: str,
        *,
        turn_started: float,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[int, int | None]:
        config = self.config  # AUDIT-FIX(#3): Keep timing and queue limits consistent for this TTS turn.
        tts_started = time.monotonic()
        first_audio_at: list[float | None] = [None]
        queue_max_chunks = max(
            4,
            int(getattr(config, "tts_stream_queue_max_chunks", _DEFAULT_TTS_QUEUE_MAX_CHUNKS)),
        )
        first_chunk_timeout_seconds = max(
            1.0,
            float(
                getattr(
                    config,
                    "tts_first_chunk_timeout_seconds",
                    _DEFAULT_TTS_FIRST_CHUNK_TIMEOUT_SECONDS,
                )
            ),
        )
        chunk_timeout_seconds = max(
            1.0,
            float(
                getattr(
                    config,
                    "tts_stream_chunk_timeout_seconds",
                    _DEFAULT_TTS_STREAM_CHUNK_TIMEOUT_SECONDS,
                )
            ),
        )
        chunk_queue: Queue[bytes | Exception | object] = Queue(maxsize=queue_max_chunks)  # AUDIT-FIX(#6): Bound memory growth under slow playback.
        sentinel = object()
        producer_stop = Event()
        feedback_started = False

        def _noop_stop_answering_feedback() -> None:
            return None

        stop_answering_feedback: Callable[[], None] = _noop_stop_answering_feedback

        def queue_put(item: bytes | Exception | object) -> bool:
            while not producer_stop.is_set():
                try:
                    chunk_queue.put(item, timeout=0.1)
                    return True
                except Full:
                    continue
            return False

        def synth_worker() -> None:
            try:
                for chunk in self.tts_provider.synthesize_stream(text):
                    if producer_stop.is_set():
                        return
                    chunk_bytes = self._coerce_audio_chunk(chunk)
                    if not chunk_bytes:
                        continue
                    if not queue_put(chunk_bytes):
                        return
            except Exception as exc:
                if not producer_stop.is_set():
                    queue_put(exc)
            finally:
                if not producer_stop.is_set():
                    queue_put(sentinel)

        synth_thread = Thread(target=synth_worker, daemon=True)
        synth_thread.start()
        first_chunk: bytes | None = None
        first_chunk_deadline = tts_started + first_chunk_timeout_seconds
        try:
            while first_chunk is None:
                if should_stop is not None and should_stop():
                    return int((time.monotonic() - tts_started) * 1000), None
                timeout_remaining = first_chunk_deadline - time.monotonic()
                if timeout_remaining <= 0:
                    raise TimeoutError("TTS stream timed out before first audio chunk")
                try:
                    item = chunk_queue.get(timeout=min(0.05, timeout_remaining))
                except Empty:
                    if not feedback_started:
                        feedback_started = True
                        stop_answering_feedback = self._start_working_feedback_loop("answering")
                    continue
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                first_chunk = item
            stop_answering_feedback()
            if first_chunk is None:
                raise RuntimeError("TTS stream ended without audio")
            self._try_emit("realtime_tts_first_chunk_received=true")

            def playback_chunks():
                if first_audio_at[0] is None:
                    first_audio_at[0] = time.monotonic()
                yield first_chunk
                while True:
                    if should_stop is not None and should_stop():
                        return
                    try:
                        item = chunk_queue.get(timeout=chunk_timeout_seconds)
                    except Empty:
                        raise TimeoutError("TTS stream stalled while waiting for audio chunk")
                    if item is sentinel:
                        return
                    if isinstance(item, Exception):
                        raise item
                    yield item

            self._try_emit("realtime_tts_playback_started=true")
            self._get_playback_coordinator().play_wav_chunks(
                owner="realtime_tts",
                priority=PlaybackPriority.SPEECH,
                chunks=playback_chunks(),
                should_stop=should_stop,
            )
            self._try_emit("realtime_tts_playback_completed=true")
        finally:
            producer_stop.set()  # AUDIT-FIX(#6): Let the worker thread exit if the consumer aborts early.
            stop_answering_feedback()
            synth_thread.join(timeout=_DEFAULT_STOP_JOIN_TIMEOUT_SECONDS)
            if synth_thread.is_alive():
                self._try_emit("tts_warning=synth_thread_still_running")

        tts_ms = int((time.monotonic() - tts_started) * 1000)
        if first_audio_at[0] is None:
            return tts_ms, None
        return tts_ms, int((first_audio_at[0] - turn_started) * 1000)

    def _build_vision_images(self) -> list[OpenAIImageInput]:
        return vision_support.build_vision_images(
            self,
            allowed_suffixes=_ALLOWED_REFERENCE_IMAGE_SUFFIXES,
            default_max_bytes=_DEFAULT_REFERENCE_IMAGE_MAX_BYTES,
        )

    def _load_reference_image(self) -> OpenAIImageInput | None:
        return vision_support.load_reference_image(
            self,
            allowed_suffixes=_ALLOWED_REFERENCE_IMAGE_SUFFIXES,
            default_max_bytes=_DEFAULT_REFERENCE_IMAGE_MAX_BYTES,
        )

    def _build_vision_prompt(self, question: str, *, include_reference: bool) -> str:
        return vision_support.build_vision_prompt(question, include_reference=include_reference)

    def _is_no_speech_timeout(self, exc: Exception) -> bool:
        message = str(exc).casefold()  # AUDIT-FIX(#11): Provider errors drift over time; avoid brittle exact-match classification.
        return any(marker in message for marker in _NO_SPEECH_TIMEOUT_MARKERS)

    def _is_print_cooldown_active(self) -> bool:
        last_print_request_at = getattr(self, "_last_print_request_at", None)
        if last_print_request_at is None:
            return False
        try:
            elapsed_seconds = time.monotonic() - float(last_print_request_at)
        except (TypeError, ValueError):  # AUDIT-FIX(#1): Corrupt in-memory state should fail open, not crash the print path.
            return False
        return elapsed_seconds < self._print_button_cooldown_seconds()
