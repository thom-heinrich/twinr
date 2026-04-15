"""Run the text-only orchestrator probe without full hardware-loop bootstrap.

The operator-facing ``--orchestrator-probe-turn`` path needs runtime context
and the local realtime tool surface, but it does not need GPIO polling, audio
capture, speaker playback, the live voice gateway, or proactive sensor startup.
This module keeps that lighter probe bootstrap separate from ``__main__`` so
the CLI can validate websocket turns without inheriting unrelated Pi hardware
startup stalls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from time import monotonic
from typing import Any, Callable, Mapping, cast

from twinr.agent.base_agent.contracts import (
    SupervisorDecision,
    normalize_supervisor_decision_context_scope,
    supervisor_decision_requires_full_context,
)
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.prompting.personality import load_supervisor_loop_instructions
from twinr.agent.tools import build_supervisor_decision_instructions
from twinr.agent.tools.runtime.runtime_local_handoff import has_executable_runtime_local_tool_call
from twinr.agent.workflows.forensics import bind_workflow_forensics
from twinr.display.render_state import assess_visible_display_state
from twinr.agent.workflows.realtime_runner import TwinrRealtimeHardwareLoop
from twinr.orchestrator.client import OrchestratorWebSocketClient
from twinr.orchestrator.contracts import OrchestratorClientTurnResult, OrchestratorTurnRequest
from twinr.orchestrator.local_bridge_target import resolve_local_orchestrator_probe_target
from twinr.orchestrator.remote_tool_timeout import read_remote_tool_timeout_seconds
from twinr.providers import build_streaming_provider_bundle
from twinr.providers.openai import OpenAIBackend, OpenAISupervisorDecisionProvider


_PROBE_SKIPPED_COMPONENTS: tuple[str, ...] = (
    "button_monitor",
    "recorder",
    "player",
    "voice_orchestrator",
    "ambient_audio_sampler",
    "default_proactive_monitor",
)
# Keep probe cleanup aligned with the runtime's own long-term writer shutdown
# budget so successful multimodal turns can flush truthfully instead of
# emitting artificial 0.2s cleanup stalls.
_PROBE_RUNTIME_CLOSE_TIMEOUT_S = 2.0
# Operator probes are explicit acceptance runs, not latency-critical voice
# turns. Give accepted long-term writes enough budget to drain normally before
# shutdown so queue/task_done invariants stay truthful for multimodal evidence.
_PROBE_LONG_TERM_FLUSH_TIMEOUT_S = 10.0


@dataclass(frozen=True, slots=True)
class OrchestratorProbeTurnOutcome:
    """Capture the completed text probe turn and its streamed deltas."""

    result: OrchestratorClientTurnResult
    deltas: tuple[str, ...]
    tool_handler_count: int
    stage_results: tuple["OrchestratorProbeStageResult", ...] = ()


@dataclass(frozen=True, slots=True)
class OrchestratorProbeStageResult:
    """Describe one bounded probe stage or milestone."""

    stage: str
    status: str
    elapsed_ms: int
    details: dict[str, object] = field(default_factory=dict)


class _ProbeRealtimeSession:
    """Minimal realtime-session placeholder for tool-surface sync only."""

    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.tool_handlers: dict[str, Callable[..., Any]] = {}

    def set_tool_handlers(self, tool_handlers: Mapping[str, Callable[..., Any]]) -> None:
        self.tool_handlers = dict(tool_handlers)

    def update_tool_handlers(self, tool_handlers: Mapping[str, Callable[..., Any]]) -> None:
        self.tool_handlers = dict(tool_handlers)

    def replace_tool_handlers(self, tool_handlers: Mapping[str, Callable[..., Any]]) -> None:
        self.tool_handlers = dict(tool_handlers)

    def close(self) -> None:
        return None


class _ProbeButtonMonitor:
    """Placeholder button monitor so text probes never wait on GPIO setup."""

    def close(self) -> None:
        return None


class _ProbeRecorder:
    """Fail-closed recorder placeholder for the text-only probe runtime."""

    def record_to_wav_bytes(self, *args, **kwargs) -> bytes:
        del args, kwargs
        raise RuntimeError("Audio capture is unavailable in the text-only orchestrator probe runtime.")


class _ProbePlayer:
    """No-op player used so the probe runtime never touches playback hardware."""

    def play_wav_bytes(self, *args, **kwargs) -> None:
        del args, kwargs
        return None

    def play_pcm16_chunks(self, *args, **kwargs) -> None:
        del args, kwargs
        return None

    def stop_playback(self) -> None:
        return None


class _ProbeAmbientAudioSampler:
    """Placeholder sampler so probe bootstrap never opens ambient capture."""

    def close(self) -> None:
        return None


class _ProbeProactiveMonitor:
    """Placeholder proactive monitor for text-only websocket probes."""

    def close(self) -> None:
        return None


def _safe_probe_text(value: object, *, max_len: int = 240) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _safe_record_event(
    owner: object | None,
    event: str,
    message: str,
    *,
    level: str = "info",
    **payload: object,
) -> None:
    record_event = getattr(owner, "_record_event", None)
    if not callable(record_event):
        return
    record_event(event, message, level=level, **payload)


def _emit_display_visible_state(
    config: TwinrConfig,
    *,
    emit_line: Callable[[str], None],
    owner: object | None,
) -> None:
    assessment = assess_visible_display_state(config)
    visible_state = _safe_probe_text(assessment.visible_runtime_status or "unknown", max_len=64)
    visible_operator_status = _safe_probe_text(assessment.visible_operator_status or "unknown", max_len=64)
    rendered_at = _safe_probe_text(assessment.rendered_at or "", max_len=64)
    reason = _safe_probe_text(assessment.reason, max_len=96)
    source = _safe_probe_text(assessment.source, max_len=64)
    emit_parts = [
        f"display_visible_state={visible_state}",
        f"display_visible_operator_status={visible_operator_status}",
        f"display_visible_state_verdict={assessment.verdict}",
        f"display_visible_state_source={source}",
        f"display_visible_state_reason={reason}",
    ]
    event_payload: dict[str, object] = {
        "display_visible_state": visible_state,
        "display_visible_operator_status": visible_operator_status,
        "display_visible_state_verdict": assessment.verdict,
        "display_visible_state_source": source,
        "display_visible_state_reason": reason,
    }
    if rendered_at:
        emit_parts.append(f"display_visible_rendered_at={rendered_at}")
        event_payload["display_visible_rendered_at"] = rendered_at
    emit_line(" ".join(emit_parts))
    _safe_record_event(
        owner,
        "orchestrator_probe_display_visible_state",
        "Resolved visible display state for the probe surface.",
        display_visible_state=event_payload["display_visible_state"],
        display_visible_operator_status=event_payload["display_visible_operator_status"],
        display_visible_state_verdict=event_payload["display_visible_state_verdict"],
        display_visible_state_source=event_payload["display_visible_state_source"],
        display_visible_state_reason=event_payload["display_visible_state_reason"],
        display_visible_rendered_at=event_payload.get("display_visible_rendered_at", ""),
    )


def _emit_probe_stage(
    emit_line: Callable[[str], None],
    *,
    owner: object | None,
    stage: str,
    status: str,
    started_at: float,
    error: BaseException | None = None,
    details: Mapping[str, object] | None = None,
) -> OrchestratorProbeStageResult:
    elapsed_ms = max(0, int(round((monotonic() - started_at) * 1000.0)))
    stage_details: dict[str, object] = {
        "stage": stage,
        "status": status,
        "elapsed_ms": elapsed_ms,
    }
    parts = [
        f"probe_stage={stage}",
        f"status={status}",
        f"elapsed_ms={elapsed_ms}",
    ]
    level = "info"
    if error is not None:
        level = "error"
        stage_details["error_type"] = type(error).__name__
        parts.append(f"error_type={type(error).__name__}")
        safe_error = _safe_probe_text(error)
        if safe_error:
            stage_details["error"] = safe_error
            parts.append(f"error={safe_error}")
    if details:
        for key, value in details.items():
            safe_key = _safe_probe_text(key, max_len=64).replace(" ", "_")
            if not safe_key:
                continue
            if isinstance(value, bool):
                safe_value = str(value).lower()
            else:
                safe_value = _safe_probe_text(value)
            if not safe_value:
                continue
            stage_details[safe_key] = safe_value if not isinstance(value, (int, float)) else value
            parts.append(f"{safe_key}={safe_value}")
    _safe_record_event(
        owner,
        "orchestrator_probe_stage",
        "Orchestrator probe stage completed." if error is None else "Orchestrator probe stage failed.",
        level=level,
        **stage_details,
    )
    emit_line(" ".join(parts))
    persisted_details = {
        key: value
        for key, value in stage_details.items()
        if key not in {"stage", "status", "elapsed_ms"}
    }
    return OrchestratorProbeStageResult(
        stage=stage,
        status=status,
        elapsed_ms=elapsed_ms,
        details=persisted_details,
    )


def _build_probe_runtime_config(config: TwinrConfig) -> TwinrConfig:
    """Disable subsystems that are irrelevant to a text websocket probe."""

    return replace(
        config,
        voice_orchestrator_enabled=False,
    )


def _extract_tool_handlers(loop: object) -> dict[str, Callable[..., Any]]:
    handlers = getattr(loop, "_tool_handlers", None)
    if not isinstance(handlers, Mapping):
        raise RuntimeError("Probe tool runtime did not expose a realtime tool handler map.")
    extracted = {str(name): handler for name, handler in handlers.items() if callable(handler)}
    return extracted


def _authorize_probe_tool_surface(loop: object) -> tuple[str, ...]:
    """Explicitly unlock the full operator probe tool surface."""

    authorize = getattr(loop, "authorize_realtime_sensitive_tools", None)
    if not callable(authorize):
        raise RuntimeError(
            "Probe tool runtime did not expose sensitive-tool authorization."
        )
    authorized_names = authorize("orchestrator_probe_turn")
    if not isinstance(authorized_names, tuple):
        authorized_names = tuple(str(name) for name in authorized_names)
    return tuple(str(name) for name in authorized_names)


def _seed_probe_runtime_transcript(runtime: object, *, prompt: str) -> None:
    """Prime the runtime with the probe prompt before context assembly."""

    begin_listening = getattr(runtime, "begin_listening", None)
    if not callable(begin_listening):
        raise RuntimeError("Probe runtime did not expose begin_listening().")
    submit_transcript = getattr(runtime, "submit_transcript", None)
    if not callable(submit_transcript):
        raise RuntimeError("Probe runtime did not expose submit_transcript().")
    begin_listening(request_source="orchestrator_probe_turn")
    submit_transcript(prompt)


def _attach_probe_trace_hooks(
    runtime: object,
    loop: object,
    *,
    trace_id: str | None,
) -> Callable[[], None]:
    """Bridge loop-owned workflow tracing into runtime context builders."""

    restorers: list[tuple[str, bool, object | None]] = []

    def _set_hook(attr_name: str, hook: object) -> None:
        had_attr = hasattr(runtime, attr_name)
        previous = getattr(runtime, attr_name, None)
        setattr(runtime, attr_name, hook)
        restorers.append((attr_name, had_attr, previous))

    trace_event = getattr(loop, "_trace_event", None)
    if callable(trace_event):
        def _runtime_trace_event(message: str, **kwargs) -> None:
            payload = dict(kwargs)
            payload.setdefault("trace_id", trace_id)
            trace_event(message, **payload)

        _set_hook("_trace_event", _runtime_trace_event)

    trace_decision = getattr(loop, "_trace_decision", None)
    if callable(trace_decision):
        def _runtime_trace_decision(message: str, **kwargs) -> None:
            payload = dict(kwargs)
            payload.setdefault("trace_id", trace_id)
            trace_decision(message, **payload)

        _set_hook("_trace_decision", _runtime_trace_decision)

    trace_span = getattr(loop, "_trace_span", None)
    if callable(trace_span):
        def _runtime_trace_span(*, name: str, kind: str = "span", details: dict[str, object] | None = None):
            return trace_span(
                name=name,
                kind=kind,
                details=details,
                trace_id=trace_id,
            )

        _set_hook("_trace_span", _runtime_trace_span)

    def _restore() -> None:
        for attr_name, had_attr, previous in reversed(restorers):
            if had_attr:
                setattr(runtime, attr_name, previous)
            else:
                delattr(runtime, attr_name)

    return _restore


def _probe_prefetched_token_usage_payload(value: object | None) -> dict[str, object] | None:
    """Normalize provider token-usage metadata for websocket transport."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            return None
        if isinstance(payload, Mapping):
            return {str(key): item for key, item in payload.items()}
    return None


def _prefetched_supervisor_decision_payload(
    decision: SupervisorDecision | None,
) -> dict[str, object] | None:
    """Serialize one validated supervisor decision into transport-safe JSON."""

    if decision is None:
        return None
    payload = cast(dict[str, object], asdict(decision))
    payload["token_usage"] = _probe_prefetched_token_usage_payload(payload.get("token_usage"))
    return payload


def _probe_prefetched_supervisor_decision(
    *,
    config: TwinrConfig,
    backend: object,
    prompt: str,
    supervisor_conversation: tuple[tuple[str, str], ...],
) -> SupervisorDecision:
    """Resolve the structured supervisor decision before the websocket turn starts."""

    provider = OpenAISupervisorDecisionProvider(
        cast(OpenAIBackend, backend),
        model_override=config.streaming_supervisor_model,
        reasoning_effort_override=config.streaming_supervisor_reasoning_effort,
        base_instructions_override=load_supervisor_loop_instructions(config),
        replace_base_instructions=True,
    )
    return provider.decide(
        prompt,
        conversation=supervisor_conversation,
        instructions=build_supervisor_decision_instructions(
            config,
            extra_instructions=config.openai_realtime_instructions,
        ),
    )


def _probe_tool_context_reader(
    runtime: object,
    *,
    decision: SupervisorDecision | None,
) -> tuple[Callable[[], tuple[tuple[str, str], ...]], str]:
    """Choose the lightest safe tool-context scope for one probe turn."""

    tool_context_reader = getattr(runtime, "tool_provider_conversation_context", None)
    if not callable(tool_context_reader):
        raise RuntimeError("Probe runtime did not expose tool_provider_conversation_context().")

    tiny_recent_reader = getattr(runtime, "tool_provider_tiny_recent_conversation_context", None)
    action = str(getattr(decision, "action", "") or "").strip().lower()
    context_scope = normalize_supervisor_decision_context_scope(
        getattr(decision, "context_scope", None)
    )
    prefer_tiny_recent = (
        decision is not None
        and not supervisor_decision_requires_full_context(decision)
        and (
            context_scope == "tiny_recent"
            or action in {"direct", "end_conversation"}
            or has_executable_runtime_local_tool_call(decision)
        )
    )
    if prefer_tiny_recent and callable(tiny_recent_reader):
        return tiny_recent_reader, "tiny_recent"
    return tool_context_reader, "full_context"


def run_orchestrator_probe_turn(
    *,
    config: TwinrConfig,
    runtime: object,
    backend: object,
    prompt: str,
    emit_line: Callable[[str], None] = print,
) -> OrchestratorProbeTurnOutcome:
    """Run one operator websocket probe through a lightweight local tool runtime."""

    if backend is None:
        raise RuntimeError("Orchestrator probe requires a configured backend.")

    probe_config = _build_probe_runtime_config(config)
    skipped = ",".join(_PROBE_SKIPPED_COMPONENTS)
    emit_line(f"probe_decision=lightweight_realtime_runtime skipped_components={skipped}")
    _emit_display_visible_state(config, emit_line=emit_line, owner=runtime)
    _safe_record_event(
        runtime,
        "orchestrator_probe_decision",
        "Using lightweight realtime runtime for text-only probe.",
        strategy="lightweight_realtime_runtime",
        skipped_components=skipped,
    )
    stage_results: list[OrchestratorProbeStageResult] = []
    outcome: OrchestratorProbeTurnOutcome | None = None

    bundle_started = monotonic()
    try:
        provider_bundle = build_streaming_provider_bundle(probe_config, support_backend=backend)
    except Exception as exc:
        stage_results.append(_emit_probe_stage(
            emit_line,
            owner=runtime,
            stage="provider_bundle",
            status="error",
            started_at=bundle_started,
            error=exc,
        ))
        raise
    stage_results.append(_emit_probe_stage(
        emit_line,
        owner=runtime,
        stage="provider_bundle",
        status="ok",
        started_at=bundle_started,
    ))

    loop_started = monotonic()
    try:
        loop = TwinrRealtimeHardwareLoop(
            config=probe_config,
            runtime=runtime,
            realtime_session=_ProbeRealtimeSession(probe_config),
            print_backend=provider_bundle.print_backend,
            stt_provider=provider_bundle.stt,
            verification_stt_provider=getattr(provider_bundle, "verification_stt", None),
            agent_provider=provider_bundle.agent,
            tts_provider=provider_bundle.tts,
            turn_stt_provider=getattr(provider_bundle, "stt", None),
            turn_tool_agent_provider=provider_bundle.tool_agent,
            button_monitor=_ProbeButtonMonitor(),
            recorder=_ProbeRecorder(),
            player=_ProbePlayer(),
            ambient_audio_sampler=_ProbeAmbientAudioSampler(),
            proactive_monitor=_ProbeProactiveMonitor(),
            emit=emit_line,
        )
    except Exception as exc:
        stage_results.append(_emit_probe_stage(
            emit_line,
            owner=runtime,
            stage="tool_runtime_bootstrap",
            status="error",
            started_at=loop_started,
            error=exc,
        ))
        raise

    stage_results.append(_emit_probe_stage(
        emit_line,
        owner=loop,
        stage="tool_runtime_bootstrap",
        status="ok",
        started_at=loop_started,
    ))
    # Operator probes validate live tool behavior against the Pi runtime, but
    # they must not pollute durable multimodal memory with acceptance-only
    # camera/print artifacts.
    setattr(loop, "_persist_multimodal_evidence", False)

    try:
        workflow_binding = bind_workflow_forensics(getattr(loop, "workflow_forensics", None))
        with workflow_binding as bound_trace_id:
            set_active_trace = getattr(loop, "_workflow_trace_set_active", None)
            if callable(set_active_trace):
                set_active_trace(bound_trace_id)
            restore_trace_hooks = _attach_probe_trace_hooks(
                runtime,
                loop,
                trace_id=bound_trace_id,
            )
            try:
                if bound_trace_id:
                    emit_line(f"probe_workflow_trace_id={bound_trace_id}")
                workflow_run_dir = getattr(getattr(loop, "workflow_forensics", None), "_run_dir", None)
                if workflow_run_dir:
                    emit_line(f"probe_workflow_run_dir={workflow_run_dir}")

                prompt_seed_started = monotonic()
                _seed_probe_runtime_transcript(runtime, prompt=prompt)
                stage_results.append(_emit_probe_stage(
                    emit_line,
                    owner=runtime,
                    stage="prompt_seed",
                    status="ok",
                    started_at=prompt_seed_started,
                ))
                authorized_tool_names = _authorize_probe_tool_surface(loop)
                emit_line("probe_sensitive_tools_authorized=true")
                tool_handlers = _extract_tool_handlers(loop)
                emit_line(f"probe_tool_handler_count={len(tool_handlers)}")
                _safe_record_event(
                    loop,
                    "orchestrator_probe_tool_surface",
                    "Probe tool runtime initialized.",
                    tool_handler_count=len(tool_handlers),
                    sensitive_tools_authorized=True,
                    runtime_tool_count=len(authorized_tool_names),
                )
                resolved_target = resolve_local_orchestrator_probe_target(probe_config)
                emit_line(f"probe_orchestrator_target={resolved_target.url}")
                if resolved_target.reason:
                    emit_line(f"probe_orchestrator_target_reason={resolved_target.reason}")
                _safe_record_event(
                    loop,
                    "orchestrator_probe_transport_target",
                    "Resolved websocket target for the orchestrator probe.",
                    orchestrator_target=resolved_target.url,
                    orchestrator_target_rewritten=resolved_target.rewritten,
                    orchestrator_target_reason=resolved_target.reason or "",
                )

                client = OrchestratorWebSocketClient(
                    resolved_target.url,
                    shared_secret=probe_config.orchestrator_shared_secret,
                    tool_timeout_seconds=read_remote_tool_timeout_seconds(),
                    require_tls=not bool(getattr(probe_config, "orchestrator_allow_insecure_ws", False)),
                )
                deltas: list[str] = []
                turn_started = monotonic()
                transport_times: dict[str, float] = {"turn_started": turn_started}
                tool_started: dict[str, float] = {}

                def _on_transport_event(event: str, payload: dict[str, object]) -> None:
                    now = monotonic()
                    if event == "request_prepare_tool_handlers":
                        transport_times["request_prepare_tool_handlers"] = now
                        stage_results.append(
                            _emit_probe_stage(
                                emit_line,
                                owner=loop,
                                stage="request_prepare_tool_handlers",
                                status="ok",
                                started_at=transport_times.get("request_prepare_request_object_ready", transport_times["turn_started"]),
                                details={"handler_count": payload.get("handler_count", "")},
                            )
                        )
                        return
                    if event == "request_prepare_headers":
                        transport_times["request_prepare_headers"] = now
                        stage_results.append(
                            _emit_probe_stage(
                                emit_line,
                                owner=loop,
                                stage="request_prepare_headers",
                                status="ok",
                                started_at=transport_times.get("request_prepare_tool_handlers", transport_times["turn_started"]),
                                details={"has_headers": bool(payload.get("has_headers", False))},
                            )
                        )
                        return
                    if event == "request_prepare_connector_kwargs":
                        transport_times["request_prepare_connector_kwargs"] = now
                        stage_results.append(
                            _emit_probe_stage(
                                emit_line,
                                owner=loop,
                                stage="request_prepare_connector_kwargs",
                                status="ok",
                                started_at=transport_times.get("request_prepare_headers", transport_times["turn_started"]),
                                details={"kwarg_count": payload.get("kwarg_count", "")},
                            )
                        )
                        return
                    if event == "request_prepare_deadline":
                        transport_times["request_prepare_deadline"] = now
                        stage_results.append(
                            _emit_probe_stage(
                                emit_line,
                                owner=loop,
                                stage="request_prepare_deadline",
                                status="ok",
                                started_at=transport_times.get("request_prepare_connector_kwargs", transport_times["turn_started"]),
                                details={"bounded": bool(payload.get("bounded", False))},
                            )
                        )
                        return
                    if event == "request_prepare_payload":
                        transport_times["request_prepare_payload"] = now
                        stage_results.append(
                            _emit_probe_stage(
                                emit_line,
                                owner=loop,
                                stage="request_prepare_payload",
                                status="ok",
                                started_at=transport_times.get("request_prepare_deadline", transport_times["turn_started"]),
                                details={
                                    "conversation_messages": payload.get("conversation_messages", ""),
                                    "supervisor_messages": payload.get("supervisor_messages", ""),
                                },
                            )
                        )
                        return
                    if event == "request_prepared":
                        transport_times["request_prepared"] = now
                        stage_results.append(
                            _emit_probe_stage(
                                emit_line,
                                owner=loop,
                                stage="request_prepare",
                                status="ok",
                                started_at=transport_times["turn_started"],
                            )
                        )
                        return
                    if event == "ws_connected":
                        transport_times["ws_connected"] = now
                        stage_results.append(
                            _emit_probe_stage(
                                emit_line,
                                owner=loop,
                                stage="ws_connect",
                                status="ok",
                                started_at=transport_times.get("request_prepared", transport_times["turn_started"]),
                            )
                        )
                        return
                    if event == "turn_request_sent":
                        transport_times["turn_request_sent"] = now
                        stage_results.append(
                            _emit_probe_stage(
                                emit_line,
                                owner=loop,
                                stage="turn_submit",
                                status="ok",
                                started_at=transport_times.get("ws_connected", transport_times["turn_started"]),
                            )
                        )
                        return
                    if event == "first_server_event":
                        transport_times["first_server_event"] = now
                        stage_results.append(
                            _emit_probe_stage(
                                emit_line,
                                owner=loop,
                                stage="first_server_event",
                                status="ok",
                                started_at=transport_times.get("turn_request_sent", transport_times["turn_started"]),
                                details={"message_type": payload.get("message_type", "")},
                            )
                        )
                        return
                    if event == "tool_request_received":
                        call_id = _safe_probe_text(payload.get("call_id", ""))
                        if call_id:
                            tool_started[call_id] = now
                        return
                    if event == "tool_response_sent":
                        call_id = _safe_probe_text(payload.get("call_id", ""))
                        stage_results.append(
                            _emit_probe_stage(
                                emit_line,
                                owner=loop,
                                stage="tool_call",
                                status="ok" if bool(payload.get("ok", False)) else "error",
                                started_at=tool_started.pop(call_id, transport_times.get("first_server_event", transport_times["turn_started"])),
                                details={
                                    "tool_name": payload.get("tool_name", ""),
                                    "call_id": call_id,
                                    "ok": bool(payload.get("ok", False)),
                                    "error": payload.get("error", ""),
                                },
                            )
                        )
                        return
                    if event == "turn_complete_received":
                        stage_results.append(
                            _emit_probe_stage(
                                emit_line,
                                owner=loop,
                                stage="turn_complete",
                                status="ok",
                                started_at=transport_times.get("first_server_event", transport_times["turn_started"]),
                                details={
                                    "rounds": payload.get("rounds", ""),
                                    "used_web_search": bool(payload.get("used_web_search", False)),
                                    "model": payload.get("model", ""),
                                },
                            )
                        )
                        return
                    if event == "turn_error_received":
                        stage_results.append(
                            _emit_probe_stage(
                                emit_line,
                                owner=loop,
                                stage="turn_complete",
                                status="error",
                                started_at=transport_times.get("first_server_event", transport_times["turn_started"]),
                                details={"error": payload.get("error", "")},
                            )
                        )

                supervisor_context_started = monotonic()
                try:
                    supervisor_conversation = runtime.supervisor_provider_conversation_context()
                except Exception as exc:
                    stage_results.append(
                        _emit_probe_stage(
                            emit_line,
                            owner=loop,
                            stage="request_prepare_supervisor_context",
                            status="error",
                            started_at=supervisor_context_started,
                            error=exc,
                        )
                    )
                    raise
                transport_times["request_prepare_supervisor_context_ready"] = monotonic()
                stage_results.append(
                    _emit_probe_stage(
                        emit_line,
                        owner=loop,
                        stage="request_prepare_supervisor_context",
                        status="ok",
                        started_at=supervisor_context_started,
                        details={"messages": len(supervisor_conversation)},
                    )
                )

                supervisor_decision_started = monotonic()
                try:
                    prefetched_decision = _probe_prefetched_supervisor_decision(
                        config=probe_config,
                        backend=backend,
                        prompt=prompt,
                        supervisor_conversation=supervisor_conversation,
                    )
                except Exception as exc:
                    stage_results.append(
                        _emit_probe_stage(
                            emit_line,
                            owner=loop,
                            stage="request_prepare_supervisor_decision",
                            status="error",
                            started_at=supervisor_decision_started,
                            error=exc,
                        )
                    )
                    raise
                transport_times["request_prepare_supervisor_decision_ready"] = monotonic()
                stage_results.append(
                    _emit_probe_stage(
                        emit_line,
                        owner=loop,
                        stage="request_prepare_supervisor_decision",
                        status="ok",
                        started_at=supervisor_decision_started,
                        details={
                            "action": prefetched_decision.action,
                            "context_scope": prefetched_decision.context_scope or "",
                            "runtime_tool_name": prefetched_decision.runtime_tool_name or "",
                        },
                    )
                )

                tool_context_started = monotonic()
                try:
                    tool_context_reader, tool_context_scope = _probe_tool_context_reader(
                        runtime,
                        decision=prefetched_decision,
                    )
                    tool_conversation = tool_context_reader()
                except Exception as exc:
                    stage_results.append(
                        _emit_probe_stage(
                            emit_line,
                            owner=loop,
                            stage="request_prepare_tool_context",
                            status="error",
                            started_at=tool_context_started,
                            error=exc,
                        )
                    )
                    raise
                transport_times["request_prepare_tool_context_ready"] = monotonic()
                stage_results.append(
                    _emit_probe_stage(
                        emit_line,
                        owner=loop,
                        stage="request_prepare_tool_context",
                        status="ok",
                        started_at=tool_context_started,
                        details={
                            "messages": len(tool_conversation),
                            "context_scope": tool_context_scope,
                        },
                    )
                )

                request_object_started = monotonic()
                try:
                    turn_request = OrchestratorTurnRequest(
                        prompt=prompt,
                        conversation=tool_conversation,
                        supervisor_conversation=supervisor_conversation,
                        prefetched_supervisor_decision=_prefetched_supervisor_decision_payload(
                            prefetched_decision
                        ),
                    )
                except Exception as exc:
                    stage_results.append(
                        _emit_probe_stage(
                            emit_line,
                            owner=loop,
                            stage="request_prepare_request_object",
                            status="error",
                            started_at=request_object_started,
                            error=exc,
                        )
                    )
                    raise
                transport_times["request_prepare_request_object_ready"] = monotonic()
                stage_results.append(
                    _emit_probe_stage(
                        emit_line,
                        owner=loop,
                        stage="request_prepare_request_object",
                        status="ok",
                        started_at=request_object_started,
                        details={
                            "conversation_messages": len(turn_request.conversation),
                            "supervisor_messages": len(turn_request.supervisor_conversation),
                            "context_scope": tool_context_scope,
                            "prefetched_supervisor_decision": bool(
                                turn_request.prefetched_supervisor_decision
                            ),
                        },
                    )
                )

                try:
                    result = client.run_turn(
                        turn_request,
                        tool_handlers=tool_handlers,
                        on_ack=lambda event: emit_line(f"ack={event.ack_id}:{event.text}"),
                        on_text_delta=lambda delta: deltas.append(delta),
                        on_transport_event=_on_transport_event,
                    )
                except Exception as exc:
                    stage_results.append(_emit_probe_stage(
                        emit_line,
                        owner=loop,
                        stage="websocket_turn",
                        status="error",
                        started_at=turn_started,
                        error=exc,
                    ))
                    raise

                stage_results.append(_emit_probe_stage(
                    emit_line,
                    owner=loop,
                    stage="websocket_turn",
                    status="ok",
                    started_at=turn_started,
                ))
                flush_long_term_memory = getattr(runtime, "flush_long_term_memory", None)
                if callable(flush_long_term_memory):
                    flush_started = monotonic()
                    try:
                        flushed = bool(flush_long_term_memory(timeout_s=_PROBE_LONG_TERM_FLUSH_TIMEOUT_S))
                    except Exception as exc:
                        stage_results.append(_emit_probe_stage(
                            emit_line,
                            owner=loop,
                            stage="long_term_flush",
                            status="error",
                            started_at=flush_started,
                            error=exc,
                        ))
                        raise
                    if not flushed:
                        stage_results.append(_emit_probe_stage(
                            emit_line,
                            owner=loop,
                            stage="long_term_flush",
                            status="error",
                            started_at=flush_started,
                            details={"timeout_s": _PROBE_LONG_TERM_FLUSH_TIMEOUT_S},
                        ))
                        raise RuntimeError(
                            "Probe long-term memory flush did not finish within "
                            f"{_PROBE_LONG_TERM_FLUSH_TIMEOUT_S:.2f}s."
                        )
                    stage_results.append(_emit_probe_stage(
                        emit_line,
                        owner=loop,
                        stage="long_term_flush",
                        status="ok",
                        started_at=flush_started,
                    ))
                outcome = OrchestratorProbeTurnOutcome(
                    result=result,
                    deltas=tuple(deltas),
                    tool_handler_count=len(tool_handlers),
                )
            finally:
                restore_trace_hooks()
                if callable(set_active_trace):
                    set_active_trace(None)
    finally:
        cleanup_started = monotonic()
        loop.close(timeout_s=_PROBE_RUNTIME_CLOSE_TIMEOUT_S)
        stage_results.append(_emit_probe_stage(
            emit_line,
            owner=loop,
            stage="cleanup_complete",
            status="ok",
            started_at=cleanup_started,
        ))
    if outcome is None:
        raise RuntimeError("Orchestrator probe exited without a completed outcome.")
    return OrchestratorProbeTurnOutcome(
        result=outcome.result,
        deltas=outcome.deltas,
        tool_handler_count=outcome.tool_handler_count,
        stage_results=tuple(stage_results),
    )
