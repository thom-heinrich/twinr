"""Run the text-only orchestrator probe without full hardware-loop bootstrap.

The operator-facing ``--orchestrator-probe-turn`` path needs runtime context
and the local realtime tool surface, but it does not need GPIO polling, audio
capture, speaker playback, the live voice gateway, or proactive sensor startup.
This module keeps that lighter probe bootstrap separate from ``__main__`` so
the CLI can validate websocket turns without inheriting unrelated Pi hardware
startup stalls.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import monotonic
from typing import Any, Callable, Mapping

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.realtime_runner import TwinrRealtimeHardwareLoop
from twinr.orchestrator.client import OrchestratorWebSocketClient
from twinr.orchestrator.contracts import OrchestratorClientTurnResult, OrchestratorTurnRequest
from twinr.providers import build_streaming_provider_bundle


_PROBE_SKIPPED_COMPONENTS: tuple[str, ...] = (
    "button_monitor",
    "recorder",
    "player",
    "voice_orchestrator",
    "ambient_audio_sampler",
    "default_proactive_monitor",
)


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
    try:
        record_event(event, message, level=level, **payload)
    except Exception:
        return


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

    try:
        tool_handlers = _extract_tool_handlers(loop)
        emit_line(f"probe_tool_handler_count={len(tool_handlers)}")
        _safe_record_event(
            loop,
            "orchestrator_probe_tool_surface",
            "Probe tool runtime initialized.",
            tool_handler_count=len(tool_handlers),
        )

        client = OrchestratorWebSocketClient(
            probe_config.orchestrator_ws_url,
            shared_secret=probe_config.orchestrator_shared_secret,
            require_tls=not bool(getattr(probe_config, "orchestrator_allow_insecure_ws", False)),
        )
        deltas: list[str] = []
        turn_started = monotonic()
        transport_times: dict[str, float] = {"turn_started": turn_started}
        tool_started: dict[str, float] = {}

        def _on_transport_event(event: str, payload: dict[str, object]) -> None:
            now = monotonic()
            if event == "ws_connected":
                transport_times["ws_connected"] = now
                stage_results.append(
                    _emit_probe_stage(
                        emit_line,
                        owner=loop,
                        stage="ws_connect",
                        status="ok",
                        started_at=transport_times["turn_started"],
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

        try:
            result = client.run_turn(
                OrchestratorTurnRequest(
                    prompt=prompt,
                    conversation=runtime.tool_provider_conversation_context(),
                    supervisor_conversation=runtime.supervisor_provider_conversation_context(),
                ),
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
        outcome = OrchestratorProbeTurnOutcome(
            result=result,
            deltas=tuple(deltas),
            tool_handler_count=len(tool_handlers),
        )
    finally:
        cleanup_started = monotonic()
        loop.close(timeout_s=0.2)
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
