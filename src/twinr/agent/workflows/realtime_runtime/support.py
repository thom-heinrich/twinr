# CHANGELOG: 2026-03-27
# BUG-1: Fix lazy lock creation race that could return different lock instances to concurrent callers.
# BUG-2: Fix feedback-loop generation races that could start overlapping working/search tones.
# BUG-3: Fix playback coordinator lazy init race and ensure a shared audio lock is always attached.
# BUG-4: Make workflow forensics best-effort so telemetry failures cannot abort active interactions.
# SEC-1: Harden emitted error redaction to avoid leaking API keys, bearer tokens, signed URLs, or passwords.
# IMP-1: Add cooperative TTS stream cancellation hooks and Python 3.13+ Queue.shutdown support for deterministic teardown.
# IMP-2: Add dedicated playback sample-rate selection and an optional structured logger bridge for OTel-style log/trace correlation.
# BUG-5: Prewarm the default processing media clip outside live turns so the first THINKING cue can reach the
#        dragon MP3 path without blocking transcript submission.
# BUG-6: Bound web-search feedback to one calm cue so long lookups do not keep
#        alternating through repeated chirps during thinking.
"""Provide shared support helpers for realtime-style workflow loops."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
import inspect
import json
import logging
import re
import time
import uuid
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, RLock, Thread, current_thread
from typing import TYPE_CHECKING, Any, Callable, cast

try:
    from queue import ShutDown as QueueShutDown
except ImportError:  # pragma: no cover - Python < 3.13 fallback.
    class QueueShutDown(Exception):
        """Compatibility shim for Queue.shutdown() support on older Pythons."""


from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import TextToSpeechProvider
from twinr.agent.base_agent.conversation.closure import (
    ConversationClosureEvaluator,
    StructuredConversationClosureEvaluator,
    ToolCallingConversationClosureEvaluator,
)
from twinr.agent.base_agent.conversation.turn_controller import ToolCallingTurnDecisionEvaluator
from twinr.memory.longterm.storage.remote_read_diagnostics import extract_remote_write_context
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.agent.workflows.forensics import WorkflowForensics
from twinr.agent.workflows.playback_coordinator import PlaybackCoordinator, PlaybackPriority
from twinr.agent.workflows.realtime_runtime import required_remote_support, vision_support
from twinr.agent.workflows.required_remote_snapshot import (
    assess_required_remote_watchdog_snapshot,
    ensure_required_remote_watchdog_snapshot_ready,
)
from twinr.agent.workflows.voice_identity_runtime import (
    update_household_voice_assessment_from_pcm,
)
from twinr.agent.workflows.working_feedback import (
    WorkingFeedbackKind,
    prewarm_working_feedback_media,
    start_working_feedback_loop,
)
from twinr.proactive.runtime.runtime_contract import ReSpeakerRuntimeContractError
from twinr.providers.openai import (
    OpenAIConversationClosureDecisionProvider,
    OpenAIImageInput,
    OpenAIToolCallingAgentProvider,
)

if TYPE_CHECKING:
    from twinr.agent.base_agent.runtime.runtime import TwinrRuntime
    from twinr.hardware.voice_profile import VoiceProfileMonitor
    from twinr.ops.usage import TwinrUsageStore

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
_DEFAULT_PLAYBACK_SAMPLE_RATE_HZ = 24000
_DEFAULT_WORKFLOW_TRACE_MIN_SESSION_EVENT_BUDGET = 512
_NO_SPEECH_TIMEOUT_MARKERS: tuple[str, ...] = (
    "no speech detected before timeout",
    "no speech detected",
    "speech timeout",
    "timeout waiting for speech",
    "timeout waiting for user speech",
    "no input audio received",
)
_TTS_STOP_CALLBACK_PARAM_NAMES: tuple[str, ...] = (
    "should_stop",
    "is_cancelled",
    "cancelled",
    "should_abort",
)
_TTS_STOP_EVENT_PARAM_NAMES: tuple[str, ...] = (
    "stop_event",
    "cancel_event",
    "abort_event",
)
_LOCK_CREATION_GUARD = Lock()
_WHITESPACE_RE = re.compile(r"\s+")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?P<key>x-?api-?key|api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret|signature|sig)\b\s*[:=]\s*(?P<value>[^\s,;]+)"
)


def _call_finalize_speculative_clear(
    callback: Callable[..., object],
    *,
    status: str,
) -> None:
    """Invoke one optional status hook with a bounded runtime status payload."""

    callback(status=status)
_AUTHORIZATION_RE = re.compile(
    r"(?i)\bauthorization\b\s*[:=]\s*(?P<scheme>bearer\s+)?(?P<value>[^\s,;]+)"
)
_SECRET_QUERY_PARAM_RE = re.compile(
    r"(?i)(?P<prefix>[?&](?:x-?api-?key|api[_-]?key|access_token|refresh_token|token|password|secret|sig|signature)=)(?P<value>[^&\s]+)"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+(?P<value>[A-Za-z0-9._~+/=-]{8,})\b")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b")


def _as_void_callback(value: object) -> Callable[[], None] | None:
    """Return one zero-arg callback when the runtime attribute is callable."""

    if callable(value):
        return value
    return None
_UNSAFE_SECRET_RE = re.compile(
    r"(?i)\b(?:authorization|x-?api-?key|api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret|signature|sig)\b\s*[:=]\s*\S{6,}"
)


def _default_emit(line: str) -> None:
    """Print one workflow telemetry line to stdout."""
    print(line, flush=True)


def _is_lock_like(candidate: object) -> bool:
    return all(hasattr(candidate, attr) for attr in ("acquire", "release", "__enter__", "__exit__"))


def _redact_secret_text(message: str) -> str:
    def _redact_authorization(match: re.Match[str]) -> str:
        scheme = (match.group("scheme") or "").strip()
        if scheme:
            return f"authorization: {scheme} <redacted>"
        return "authorization=<redacted>"

    redacted = _AUTHORIZATION_RE.sub(_redact_authorization, message)
    redacted = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group('key')}=<redacted>",
        redacted,
    )
    redacted = _SECRET_QUERY_PARAM_RE.sub(
        lambda match: f"{match.group('prefix')}<redacted>",
        redacted,
    )
    redacted = _BEARER_TOKEN_RE.sub("Bearer <redacted>", redacted)
    redacted = _OPENAI_KEY_RE.sub("sk-<redacted>", redacted)
    redacted = _JWT_RE.sub("<jwt-redacted>", redacted)
    if _UNSAFE_SECRET_RE.search(redacted) or _OPENAI_KEY_RE.search(redacted) or _JWT_RE.search(redacted):
        return "internal error"
    return redacted


class _BestEffortSpanContext:
    """Keep tracing failures from breaking the active workflow."""

    def __init__(
        self,
        context: object,
        *,
        on_error: Callable[[str, BaseException], None],
    ) -> None:
        self._context = context
        self._on_error = on_error

    def __enter__(self) -> object | None:
        if self._context is None:
            return None
        enter = getattr(self._context, "__enter__", None)
        if not callable(enter):
            return self._context
        try:
            return enter()
        except Exception as exc:  # pragma: no cover - depends on tracer implementation.
            self._on_error("enter", exc)
            self._context = None
            return None

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        if self._context is None:
            return False
        exit_method = getattr(self._context, "__exit__", None)
        if not callable(exit_method):
            return False
        try:
            return bool(exit_method(exc_type, exc, exc_tb))
        except Exception as span_exc:  # pragma: no cover - depends on tracer implementation.
            self._on_error("exit", span_exc)
            return False


class TwinrRealtimeSupportMixin:
    """Share guarded emit, media, config, and feedback helpers.

    These helpers are reused by the realtime and streaming workflow loops so
    the session classes can stay focused on orchestration.
    """

    config: TwinrConfig
    emit: Callable[[str], None]
    player: Any
    runtime: TwinrRuntime
    sleep: Callable[[float], None]
    usage_store: TwinrUsageStore
    voice_profile_monitor: VoiceProfileMonitor
    tts_provider: TextToSpeechProvider
    _working_feedback_stop: Callable[[], None] | None = None
    _search_feedback_stop: Callable[[], None] | None = None

    def _get_lock(self, name: str) -> RLock:
        lock = getattr(self, name, None)
        if _is_lock_like(lock):
            return lock
        with _LOCK_CREATION_GUARD:
            lock = getattr(self, name, None)
            if _is_lock_like(lock):
                return lock
            lock = RLock()
            setattr(self, name, lock)
            return lock

    def _get_playback_coordinator(self) -> PlaybackCoordinator:
        """Return the workflow-wide speaker coordinator, creating it lazily."""

        coordinator = getattr(self, "playback_coordinator", None)
        if isinstance(coordinator, PlaybackCoordinator):
            return coordinator
        coordinator_lock = self._get_lock("_playback_coordinator_lock")
        with coordinator_lock:
            coordinator = getattr(self, "playback_coordinator", None)
            if isinstance(coordinator, PlaybackCoordinator):
                return coordinator
            coordinator = PlaybackCoordinator(
                self.player,
                emit=getattr(self, "emit", None),
                io_lock=self._get_lock("_audio_lock"),
            )
            setattr(self, "playback_coordinator", coordinator)
            return coordinator

    def _playback_sample_rate_hz(self, config: TwinrConfig | None = None) -> int:
        config = self.config if config is None else config
        for attr_name in (
            "audio_output_sample_rate_hz",
            "audio_output_sample_rate",
            "audio_playback_sample_rate_hz",
            "audio_playback_sample_rate",
            "playback_sample_rate_hz",
            "tts_output_sample_rate_hz",
            "tts_output_sample_rate",
        ):
            raw_value = getattr(config, attr_name, None)
            try:
                sample_rate = int(raw_value)
            except (TypeError, ValueError):
                continue
            if sample_rate >= 8000:
                return sample_rate
        try:
            fallback = int(getattr(config, "openai_realtime_input_sample_rate", _DEFAULT_PLAYBACK_SAMPLE_RATE_HZ))
        except (TypeError, ValueError):
            fallback = _DEFAULT_PLAYBACK_SAMPLE_RATE_HZ
        return max(8000, fallback)

    def _prewarm_working_feedback_media(self, kind: WorkingFeedbackKind = "processing") -> None:
        """Best-effort cache warmup for feedback media outside an active turn."""

        try:
            prewarm_working_feedback_media(
                kind=kind,
                config=self.config,
                player=self.player,
                playback_coordinator=self._get_playback_coordinator(),
                emit=self._try_emit,
            )
        except Exception as exc:
            self._try_emit(f"working_feedback_prewarm_error={self._safe_error_text(exc)}")
            self._trace_event(
                "working_feedback_prewarm_failed",
                kind="exception",
                level="WARN",
                details={"kind": kind, "error": self._safe_error_text(exc)},
            )

    def _resolve_workflow_logger(self) -> logging.Logger | None:
        for candidate in (
            getattr(self, "workflow_logger", None),
            getattr(getattr(self, "runtime", None), "workflow_logger", None),
        ):
            if isinstance(candidate, logging.Logger):
                return candidate
        return None

    def _structured_workflow_log(
        self,
        *,
        record_type: str,
        msg: str,
        level: str = "INFO",
        trace_id: str | None = None,
        span_id: str | None = None,
        **payload: object,
    ) -> None:
        logger = self._resolve_workflow_logger()
        if logger is None:
            return
        body = {
            "record_type": record_type,
            "msg": msg,
            "trace_id": trace_id,
            "span_id": span_id,
            **payload,
        }
        level_name = str(level).upper()
        level_no = getattr(logging, level_name, logging.INFO)
        try:
            logger.log(
                level_no,
                json.dumps(body, default=str, ensure_ascii=False, separators=(",", ":")),
                extra={
                    "twinr_trace_id": trace_id or "",
                    "twinr_span_id": span_id or "",
                },
            )
        except Exception:
            return

    def _new_workflow_trace_id(self) -> str:
        """Return one stable trace id for a workflow session."""

        return uuid.uuid4().hex

    def _workflow_trace_active_id(self) -> str | None:
        return getattr(self, "_workflow_active_trace_id", None)

    def _workflow_trace_set_active(self, trace_id: str | None) -> None:
        self._workflow_active_trace_id = trace_id

    def _workflow_trace_min_session_event_budget(self) -> int:
        """Return the minimum remaining event budget required to start a fresh turn."""

        return _DEFAULT_WORKFLOW_TRACE_MIN_SESSION_EVENT_BUDGET

    def _ensure_workflow_trace_capacity_for_session(
        self,
        *,
        initial_source: str,
        proactive_trigger: str | None,
        seed_present: bool,
    ) -> None:
        """Rotate the workflow run pack when the current one is exhausted."""

        current = getattr(self, "workflow_forensics", None)
        if not isinstance(current, WorkflowForensics) or not current.enabled:
            return
        remaining_budget = current.remaining_event_budget()
        if current.can_accept_events() and remaining_budget >= self._workflow_trace_min_session_event_budget():
            return

        try:
            replacement = WorkflowForensics.from_env(
                project_root=Path(getattr(self, "_project_root", Path.cwd())),
                service=getattr(current, "service", self.__class__.__name__),
            )
        except Exception as exc:
            self._try_emit(
                f"workflow_forensics_rotate_failed={self._safe_error_text(exc)}"
            )
            return
        if not isinstance(replacement, WorkflowForensics) or not replacement.enabled:
            try:
                replacement.close()
            except Exception:
                pass
            self._try_emit("workflow_forensics_rotate_failed=disabled")
            return

        self.workflow_forensics = replacement
        voice_orchestrator = getattr(self, "voice_orchestrator", None)
        if voice_orchestrator is not None:
            try:
                voice_orchestrator._forensics = replacement
            except Exception:
                pass

        self._trace_event(
            "workflow_trace_rotated_for_conversation_session",
            kind="run_start",
            details={
                "initial_source": initial_source,
                "proactive_trigger": proactive_trigger,
                "seed_present": seed_present,
                "previous_run_id": getattr(current, "run_id", ""),
                "previous_remaining_event_budget": remaining_budget,
                "previous_can_accept_events": current.can_accept_events(),
                "replacement_run_id": getattr(replacement, "run_id", ""),
            },
        )

        Thread(
            target=self._close_workflow_trace_replacement,
            args=(current,),
            daemon=True,
            name="twinr-workflow-trace-rotate-close",
        ).start()

    def _close_workflow_trace_replacement(self, tracer: WorkflowForensics) -> None:
        """Close one replaced workflow tracer off the active turn path."""

        try:
            tracer.close()
        except Exception as exc:
            self._try_emit(
                f"workflow_forensics_close_failed={self._safe_error_text(exc)}"
            )

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
        active_trace_id = trace_id or self._workflow_trace_active_id()
        self._structured_workflow_log(
            record_type="workflow_event",
            msg=msg,
            level=level,
            trace_id=active_trace_id,
            span_id=span_id,
            kind=kind,
            details=details,
            reason=reason,
            kpi=kpi,
        )
        tracer = getattr(self, "workflow_forensics", None)
        if not isinstance(tracer, WorkflowForensics):
            return
        try:
            tracer.event(
                kind=kind,
                msg=msg,
                details=details,
                reason=reason,
                kpi=kpi,
                level=level,
                trace_id=active_trace_id,
                span_id=span_id,
                loc_skip=3,
            )
        except Exception as exc:
            self._try_emit(f"workflow_forensics_event_error={self._safe_error_text(exc)}")

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
        active_trace_id = trace_id or self._workflow_trace_active_id()
        self._structured_workflow_log(
            record_type="workflow_decision",
            msg=msg,
            level="INFO",
            trace_id=active_trace_id,
            span_id=span_id,
            question=question,
            selected=selected,
            options=options,
            context=context,
            confidence=confidence,
            guardrails=guardrails,
            kpi_impact_estimate=kpi_impact_estimate,
        )
        tracer = getattr(self, "workflow_forensics", None)
        if not isinstance(tracer, WorkflowForensics):
            return
        try:
            tracer.decision(
                msg=msg,
                question=question,
                selected=selected,
                options=options,
                context=context,
                confidence=confidence,
                guardrails=guardrails,
                kpi_impact_estimate=kpi_impact_estimate,
                trace_id=active_trace_id,
                span_id=span_id,
            )
        except Exception as exc:
            self._try_emit(f"workflow_forensics_decision_error={self._safe_error_text(exc)}")

    def _trace_span(
        self,
        *,
        name: str,
        kind: str = "span",
        details: dict[str, object] | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ):
        active_trace_id = trace_id or self._workflow_trace_active_id()
        self._structured_workflow_log(
            record_type="workflow_span",
            msg=name,
            level="DEBUG",
            trace_id=active_trace_id,
            span_id=parent_span_id,
            kind=kind,
            details=details,
        )
        tracer = getattr(self, "workflow_forensics", None)
        if not isinstance(tracer, WorkflowForensics):
            return nullcontext()
        try:
            span_context = tracer.span(
                name=name,
                kind=kind,
                details=details,
                trace_id=active_trace_id,
                parent_span_id=parent_span_id,
            )
        except Exception as exc:
            self._try_emit(f"workflow_forensics_span_error={self._safe_error_text(exc)}")
            return nullcontext()
        return _BestEffortSpanContext(
            span_context,
            on_error=lambda phase, exc: self._try_emit(
                f"workflow_forensics_span_{phase}_error={self._safe_error_text(exc)}"
            ),
        )

    def _safe_error_text(self, exc: BaseException) -> str:
        message = _WHITESPACE_RE.sub(" ", str(exc)).strip()
        if not message:
            message = exc.__class__.__name__
        message = _redact_secret_text(message)
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
            assess_watchdog_snapshot=assess_required_remote_watchdog_snapshot,
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

    def _try_emit(self, line: str) -> None:
        try:
            self.emit(line)
            return
        except Exception:
            pass
        try:
            _default_emit(line)
        except Exception:
            return

    def _normalize_reference_image_path(self, raw_path: str) -> Path:
        return vision_support.normalize_reference_image_path(raw_path)

    def _validate_reference_image_base_dir(self, path: Path) -> bool:
        return vision_support.validate_reference_image_base_dir(self, path)

    def _safe_read_reference_image_bytes(self, path: Path, *, max_bytes: int) -> bytes:
        return vision_support.safe_read_reference_image_bytes(path, max_bytes=max_bytes)

    def _build_image_input(self, data: bytes, *, path: Path, label: str) -> OpenAIImageInput:
        return vision_support.build_image_input(data, path=path, label=label)

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

    def _coerce_audio_chunk(self, chunk: object) -> bytes:
        if isinstance(chunk, bytes):
            return chunk
        if isinstance(chunk, bytearray):
            return bytes(chunk)
        if isinstance(chunk, memoryview):
            return chunk.tobytes()
        raise TypeError(f"TTS stream yielded unsupported chunk type: {type(chunk).__name__}")

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

    def _apply_config_to_targets(self, config: TwinrConfig) -> None:
        for provider in self._iter_config_targets():
            if hasattr(provider, "config"):
                provider.config = config
        session = getattr(self, "realtime_session", None)
        if session is not None and hasattr(session, "config"):
            session.config = config

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
        safe_error = self._safe_error_text(exc)
        if getattr(self, "_required_remote_dependency_error_active", False):
            self._required_remote_dependency_error_active = False
            self._required_remote_dependency_error_message = None
            self._required_remote_dependency_recovery_started_at = None
            self._required_remote_dependency_last_failure_sample_fingerprint = None
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
        except Exception as runtime_exc:
            _default_emit(f"runtime_fail_error={self._safe_error_text(runtime_exc)}")
        self._emit_status(force=True)
        self._try_emit(f"error={safe_error}")
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
            except Exception as sleep_exc:
                _default_emit(f"error_reset_sleep_error={self._safe_error_text(sleep_exc)}")
        try:
            self.runtime.reset_error()
        except Exception as runtime_exc:
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
        runtime = getattr(self, "runtime", None)
        status = getattr(getattr(runtime, "status", None), "value", "unknown")
        if force or status != getattr(self, "_last_status", None):
            self._try_emit(f"status={status}")
            self._trace_event(
                "runtime_status_emitted",
                kind="metric",
                details={"status": status, "force": force},
            )
            self._last_status = status
        active_statuses = tuple(
            str(getattr(item, "value", item) or "").strip().lower()
            for item in getattr(getattr(runtime, "state_machine", None), "active_statuses", ())
            if str(getattr(item, "value", item) or "").strip()
        )
        active_statuses_label = ",".join(active_statuses) if active_statuses else status
        if force or active_statuses_label != getattr(self, "_last_active_statuses", None):
            self._try_emit(f"active_statuses={active_statuses_label}")
            self._trace_event(
                "runtime_active_statuses_emitted",
                kind="metric",
                details={"active_statuses": active_statuses_label, "force": force},
            )
            self._last_active_statuses = active_statuses_label
        finalize_speculative_clear = getattr(
            self,
            "_finalize_pending_speculative_wake_display_clear",
            None,
        )
        if callable(finalize_speculative_clear):
            finalize_speculative_clear_fn = cast(
                Callable[..., object],
                finalize_speculative_clear,
            )
            _call_finalize_speculative_clear(
                finalize_speculative_clear_fn,
                status=status,
            )

    def _reload_live_config_from_env(self, env_path: Path) -> None:
        config_lock = self._get_lock("_config_lock")
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
                self._prewarm_working_feedback_media("processing")
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
        except Exception as exc:
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
        except Exception as exc:
            self._try_emit(f"usage_store_error={self._safe_error_text(exc)}")

    def _update_voice_assessment_from_pcm(self, audio_pcm: bytes) -> None:
        config = self.config
        household_assessment = update_household_voice_assessment_from_pcm(
            self,
            audio_pcm,
            source="local_capture",
        )
        if household_assessment is not None and household_assessment.status != "not_enrolled":
            if not household_assessment.should_persist:
                return
            return

        try:
            assessment = self.voice_profile_monitor.assess_pcm16(
                audio_pcm,
                sample_rate=config.openai_realtime_input_sample_rate,
                channels=config.audio_channels,
            )
        except Exception as exc:
            self._try_emit(f"voice_profile_error={self._safe_error_text(exc)}")
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
        except Exception as exc:
            self._try_emit(f"voice_profile_persist_error={self._safe_error_text(exc)}")
            return
        self._try_emit(f"voice_profile_status={assessment.status}")
        if assessment.confidence is not None:
            self._try_emit(f"voice_profile_confidence={assessment.confidence:.2f}")

    def _play_listen_beep(self) -> None:
        config = self.config
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
                sample_rate=self._playback_sample_rate_hz(config),
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
            except Exception as exc:
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

    def _acknowledge_follow_up_open(self, *, request_source: str) -> None:
        """Confirm a reopened follow-up listening window with the shared earcon."""

        self._play_listen_beep()
        self.emit("conversation_follow_up_ack=earcon")
        self._record_event(
            "conversation_follow_up_acknowledged",
            "Twinr confirmed a reopened follow-up listening window with an earcon.",
            prompt="earcon",
            request_source=request_source,
        )

    def _start_working_feedback_loop(self, kind: WorkingFeedbackKind) -> Callable[[], None]:
        feedback_lock = self._get_lock("_feedback_lock")
        with feedback_lock:
            previous_stop = _as_void_callback(getattr(self, "_working_feedback_stop", None))
            generation = int(getattr(self, "_working_feedback_generation", 0) or 0) + 1
            self._working_feedback_generation = generation
            self._working_feedback_stop = None
        if previous_stop is not None:
            try:
                previous_stop()
            except Exception as exc:
                self._try_emit(f"working_feedback_stop_error={self._safe_error_text(exc)}")
                self._trace_event(
                    "working_feedback_stop_failed",
                    kind="exception",
                    level="WARN",
                    details={"kind": kind, "error": self._safe_error_text(exc)},
                )
        config = self.config
        try:
            stop = start_working_feedback_loop(
                self.player,
                kind=kind,
                sample_rate=self._playback_sample_rate_hz(config),
                config=config,
                emit=self._try_emit,
                delay_override_ms=(
                    config.processing_feedback_delay_ms
                    if kind == "processing"
                    else None
                ),
                playback_coordinator=self._get_playback_coordinator(),
            )
        except Exception as exc:
            self._try_emit(f"working_feedback_error={self._safe_error_text(exc)}")
            self._trace_event(
                "working_feedback_start_failed",
                kind="exception",
                level="WARN",
                details={"kind": kind, "error": self._safe_error_text(exc)},
            )
            return lambda: None

        stale_generation = False
        with feedback_lock:
            if getattr(self, "_working_feedback_generation", None) != generation:
                stale_generation = True
            else:
                self._working_feedback_stop = stop
        if stale_generation:
            try:
                stop()
            except Exception as exc:
                self._try_emit(f"working_feedback_stop_error={self._safe_error_text(exc)}")
            self._trace_event(
                "working_feedback_discarded_stale_start",
                kind="branch",
                details={"kind": kind, "generation": generation},
            )
            return lambda: None

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
                active_stop = _as_void_callback(getattr(self, "_working_feedback_stop", None))
                self._working_feedback_stop = None
            if active_stop is not None:
                try:
                    active_stop()
                except Exception as exc:
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
        feedback_lock = self._get_lock("_feedback_lock")
        with feedback_lock:
            active_stop = _as_void_callback(getattr(self, "_working_feedback_stop", None))
            self._working_feedback_stop = None
        if active_stop is not None:
            try:
                active_stop()
            except Exception as exc:
                self._try_emit(f"working_feedback_stop_error={self._safe_error_text(exc)}")

    def _start_search_feedback_loop(self) -> Callable[[], None]:
        config = self.config
        if not config.search_feedback_tones_enabled:
            return lambda: None

        feedback_lock = self._get_lock("_feedback_lock")
        with feedback_lock:
            previous_stop = _as_void_callback(getattr(self, "_search_feedback_stop", None))
            generation = int(getattr(self, "_search_feedback_generation", 0) or 0) + 1
            self._search_feedback_generation = generation
            self._search_feedback_stop = None
        if previous_stop is not None:
            try:
                previous_stop()
            except Exception as exc:
                self._try_emit(f"search_feedback_stop_error={self._safe_error_text(exc)}")

        stop_event = Event()
        delay_seconds = max(0.0, config.search_feedback_delay_ms / 1000.0)
        pause_seconds = max(0.12, config.search_feedback_pause_ms / 1000.0)
        join_timeout_seconds = max(
            0.1,
            float(getattr(config, "search_feedback_stop_join_timeout_seconds", _DEFAULT_STOP_JOIN_TIMEOUT_SECONDS)),
        )
        owner = "search_feedback"
        playback_sample_rate = self._playback_sample_rate_hz(config)

        def worker() -> None:
            if stop_event.wait(delay_seconds):
                return
            try:
                for frequency_hz, duration_ms in _SEARCH_FEEDBACK_TONE_PATTERNS[0]:
                    if stop_event.is_set():
                        return
                    self._get_playback_coordinator().play_tone(
                        owner=owner,
                        priority=PlaybackPriority.FEEDBACK,
                        frequency_hz=frequency_hz,
                        duration_ms=duration_ms,
                        volume=config.search_feedback_volume,
                        sample_rate=playback_sample_rate,
                        should_stop=stop_event.is_set,
                    )
                    if stop_event.wait(0.05):
                        return
            except Exception as exc:
                self._try_emit(f"search_feedback_error={self._safe_error_text(exc)}")
                return
            if stop_event.wait(pause_seconds):
                return

        thread = Thread(target=worker, name="twinr-search-feedback", daemon=True)
        thread.start()

        def stop() -> None:
            stop_event.set()
            self._get_playback_coordinator().stop_owner(owner)
            if thread is current_thread():
                return
            thread.join(timeout=join_timeout_seconds)
            if thread.is_alive():
                self._try_emit("search_feedback_warning=stop_timeout")
            with feedback_lock:
                if getattr(self, "_search_feedback_generation", None) == generation:
                    self._search_feedback_stop = None

        stale_generation = False
        with feedback_lock:
            if getattr(self, "_search_feedback_generation", None) != generation:
                stale_generation = True
            else:
                self._search_feedback_stop = stop
        if stale_generation:
            try:
                stop()
            except Exception as exc:
                self._try_emit(f"search_feedback_stop_error={self._safe_error_text(exc)}")
            self._trace_event(
                "search_feedback_discarded_stale_start",
                kind="branch",
                details={"generation": generation},
            )
            return lambda: None

        return stop

    def _stop_search_feedback(self) -> None:
        """Stop the active search-progress tones, even if the producer is still blocked."""

        feedback_lock = self._get_lock("_feedback_lock")
        with feedback_lock:
            active_stop = _as_void_callback(getattr(self, "_search_feedback_stop", None))
            self._search_feedback_stop = None
        if active_stop is not None:
            try:
                active_stop()
            except Exception as exc:
                self._try_emit(f"search_feedback_stop_error={self._safe_error_text(exc)}")

    def _build_tts_stream_iterator(
        self,
        text: str,
        *,
        producer_stop: Event,
        should_stop: Callable[[], bool] | None,
    ) -> Iterator[object]:
        synthesize_stream = self.tts_provider.synthesize_stream

        def combined_should_stop() -> bool:
            return producer_stop.is_set() or (should_stop is not None and should_stop())

        kwargs: dict[str, object] = {}
        try:
            signature = inspect.signature(synthesize_stream)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            parameter_names = {
                parameter.name
                for parameter in signature.parameters.values()
                if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            if accepts_kwargs or any(name in parameter_names for name in _TTS_STOP_CALLBACK_PARAM_NAMES):
                for name in _TTS_STOP_CALLBACK_PARAM_NAMES:
                    if accepts_kwargs or name in parameter_names:
                        kwargs[name] = combined_should_stop
                        break
            if accepts_kwargs or any(name in parameter_names for name in _TTS_STOP_EVENT_PARAM_NAMES):
                for name in _TTS_STOP_EVENT_PARAM_NAMES:
                    if accepts_kwargs or name in parameter_names:
                        kwargs[name] = producer_stop
                        break
        return synthesize_stream(text, **kwargs) if kwargs else synthesize_stream(text)

    def _cancel_tts_provider_stream(self, producer_stop: Event) -> None:
        for method_name in ("cancel_current_stream", "cancel_stream"):
            cancel = getattr(self.tts_provider, method_name, None)
            if not callable(cancel):
                continue
            try:
                cancel()
            except TypeError:
                try:
                    cancel(producer_stop)
                except Exception as exc:
                    self._try_emit(f"tts_cancel_error={self._safe_error_text(exc)}")
            except Exception as exc:
                self._try_emit(f"tts_cancel_error={self._safe_error_text(exc)}")
            break

    def _shutdown_tts_queue(self, chunk_queue: Queue[object]) -> None:
        shutdown = getattr(chunk_queue, "shutdown", None)
        if not callable(shutdown):
            return
        try:
            shutdown(immediate=True)
        except Exception as exc:
            self._try_emit(f"tts_queue_shutdown_error={self._safe_error_text(exc)}")

    def _play_streaming_tts_with_feedback(
        self,
        text: str,
        *,
        turn_started: float,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[int, int | None]:
        config = self.config
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
        chunk_queue: Queue[bytes | Exception | object] = Queue(maxsize=queue_max_chunks)
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
                except QueueShutDown:
                    return False
            return False

        def synth_worker() -> None:
            try:
                for chunk in self._build_tts_stream_iterator(
                    text,
                    producer_stop=producer_stop,
                    should_stop=should_stop,
                ):
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

        synth_thread = Thread(target=synth_worker, name="twinr-tts-synth", daemon=True)
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
                except QueueShutDown:
                    break
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                first_chunk = self._coerce_audio_chunk(item)
            stop_answering_feedback()
            if first_chunk is None:
                raise RuntimeError("TTS stream ended without audio")
            self._try_emit("realtime_tts_first_chunk_received=true")

            def playback_chunks() -> Iterator[bytes]:
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
                    except QueueShutDown:
                        return
                    if item is sentinel:
                        return
                    if isinstance(item, Exception):
                        raise item
                    yield self._coerce_audio_chunk(item)

            self._try_emit("realtime_tts_playback_started=true")
            self._get_playback_coordinator().play_wav_chunks(
                owner="realtime_tts",
                priority=PlaybackPriority.SPEECH,
                chunks=playback_chunks(),
                should_stop=should_stop,
            )
            self._try_emit("realtime_tts_playback_completed=true")
        finally:
            producer_stop.set()
            stop_answering_feedback()
            self._cancel_tts_provider_stream(producer_stop)
            self._shutdown_tts_queue(chunk_queue)
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
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            message = str(current).casefold()
            if any(marker in message for marker in _NO_SPEECH_TIMEOUT_MARKERS):
                return True
            current = current.__cause__ or current.__context__
        return False

    def _is_print_cooldown_active(self) -> bool:
        last_print_request_at = getattr(self, "_last_print_request_at", None)
        if last_print_request_at is None:
            return False
        try:
            elapsed_seconds = time.monotonic() - float(last_print_request_at)
        except (TypeError, ValueError):
            return False
        return elapsed_seconds < self._print_button_cooldown_seconds()
