"""Expose the colocated remote-ASR HTTP surface for the voice gateway.

The voice websocket session talks to a bounded HTTP transcription contract via
``RemoteAsrBackendAdapter``. This module hosts that contract inside the same
FastAPI service when the configured remote-ASR URL points back to the local
orchestrator server. The route stays transport-focused: auth, multipart/raw
audio parsing, provider dispatch, and compact JSON responses only.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as email_policy_default
import hmac
import hashlib
import io
import logging
from typing import Protocol
from urllib.parse import urlsplit
import wave

from fastapi import APIRouter, Header, HTTPException, Request

from twinr.agent.workflows.forensics import WorkflowForensics
from twinr.agent.base_agent.config import TwinrConfig
from twinr.providers.deepgram import DeepgramSpeechToTextProvider
from twinr.providers.openai import OpenAIBackend, OpenAISpeechToTextProvider


logger = logging.getLogger(__name__)


class _SpeechToTextProvider(Protocol):
    """Describe the minimal provider surface the remote-ASR route needs."""

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """Transcribe one bounded audio payload into plain text."""


@dataclass(frozen=True, slots=True)
class _ParsedAudioRequest:
    """Store one normalized remote-ASR HTTP request."""

    audio_bytes: bytes
    filename: str
    content_type: str
    language: str | None = None
    mode: str | None = None
    prompt: str | None = None


def remote_asr_url_targets_local_orchestrator(config: TwinrConfig) -> bool:
    """Return whether the configured remote-ASR URL points at this server."""

    base_url = str(getattr(config, "voice_orchestrator_remote_asr_url", "") or "").strip()
    if not base_url:
        return False
    split = urlsplit(base_url.rstrip("/"))
    hostname = (split.hostname or "").strip().lower()
    if split.scheme not in {"http", "https"} or not hostname:
        return False
    target_port = split.port or (443 if split.scheme == "https" else 80)
    if target_port != int(getattr(config, "orchestrator_port", 0) or 0):
        return False
    orchestrator_host = str(getattr(config, "orchestrator_host", "") or "").strip().lower()
    if hostname in {"127.0.0.1", "localhost", "::1"}:
        return orchestrator_host in {"", "127.0.0.1", "0.0.0.0", "localhost", "::1", "::"}
    return hostname == orchestrator_host


class RemoteAsrHttpService:
    """Serve the bounded `/v1/transcribe` contract on the orchestrator app."""

    def __init__(
        self,
        config: TwinrConfig,
        *,
        provider: _SpeechToTextProvider | None = None,
        forensics: WorkflowForensics | None = None,
    ) -> None:
        self.config = config
        self._provider = provider or _build_remote_asr_provider(config)
        self._bearer_token = (
            str(getattr(config, "voice_orchestrator_remote_asr_bearer_token", "") or "").strip() or None
        )
        self._forensics = forensics if isinstance(forensics, WorkflowForensics) and forensics.enabled else None

    def build_router(self) -> APIRouter:
        """Return the FastAPI router for health and transcription calls."""

        router = APIRouter()
        service = self

        @router.get("/health")
        async def health() -> dict[str, object]:
            return {
                "ok": True,
                "service": "remote_asr",
                "provider": _provider_label(service.config),
            }

        @router.post("/v1/transcribe")
        async def transcribe(
            request: Request,
            authorization: str | None = Header(default=None),
        ) -> dict[str, object]:
            service._assert_authorized(authorization)
            parsed = _parse_audio_request(
                content_type_header=request.headers.get("content-type"),
                body=await request.body(),
            )
            request_id = str(request.headers.get("x-twinr-request-id") or "").strip()
            trace_id = (
                str(request.headers.get("x-twinr-voice-trace-id") or "").strip()
                or request_id
                or None
            )
            request_details = {
                "request_id": request_id or None,
                "session_id": request.headers.get("x-twinr-voice-session-id"),
                "stage": request.headers.get("x-twinr-voice-stage"),
                "state": request.headers.get("x-twinr-voice-state"),
                "origin_state": request.headers.get("x-twinr-voice-origin-state"),
                "capture_duration_ms": request.headers.get("x-twinr-voice-capture-duration-ms"),
                "capture_average_rms": request.headers.get("x-twinr-voice-capture-average-rms"),
                "capture_peak_rms": request.headers.get("x-twinr-voice-capture-peak-rms"),
                "capture_active_ratio": request.headers.get("x-twinr-voice-capture-active-ratio"),
                "capture_signal_nonzero_ratio": request.headers.get("x-twinr-voice-nonzero-ratio"),
                "capture_signal_clipped_ratio": request.headers.get("x-twinr-voice-clipped-ratio"),
                "capture_signal_zero_crossing_ratio": request.headers.get("x-twinr-voice-zero-crossing-ratio"),
                "capture_signal_sha256": request.headers.get("x-twinr-voice-capture-sha256"),
                "audio_bytes": len(parsed.audio_bytes),
                "audio_sha256": hashlib.sha256(parsed.audio_bytes).hexdigest()[:12],
                "filename": parsed.filename,
                "content_type": parsed.content_type,
                "language": parsed.language,
                "mode": parsed.mode,
                "prompt_chars": len(str(parsed.prompt or "").strip()),
                "provider": _provider_label(service.config),
            }
            started_at = asyncio.get_running_loop().time()
            service._trace_event(
                "remote_asr_service_request",
                kind="http",
                trace_id=trace_id,
                details=request_details,
            )
            try:
                transcript = await asyncio.to_thread(
                    service._provider.transcribe,
                    parsed.audio_bytes,
                    filename=parsed.filename,
                    content_type=parsed.content_type,
                    language=parsed.language,
                    prompt=parsed.prompt,
                )
            except Exception as exc:
                service._trace_event(
                    "remote_asr_service_failure",
                    kind="exception",
                    trace_id=trace_id,
                    level="ERROR",
                    details={
                        **request_details,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:240],
                    },
                    kpi={"latency_ms": round((asyncio.get_running_loop().time() - started_at) * 1000.0, 3)},
                )
                logger.exception("Remote ASR transcription failed")
                raise HTTPException(status_code=503, detail=str(exc)[:240] or "transcription_failed") from exc
            response_text = str(transcript or "").strip()
            service._trace_event(
                "remote_asr_service_response",
                kind="http",
                trace_id=trace_id,
                details={
                    **request_details,
                    "response_text_chars": len(response_text),
                },
                kpi={"latency_ms": round((asyncio.get_running_loop().time() - started_at) * 1000.0, 3)},
            )
            return {
                "text": response_text,
                "language": parsed.language or getattr(service.config, "voice_orchestrator_remote_asr_language", None),
                "segments": [],
                "duration_sec": _infer_duration_sec(parsed.audio_bytes, content_type=parsed.content_type),
                "mode": parsed.mode,
                "provider": _provider_label(service.config),
            }

        return router

    def close(self) -> None:
        """Release provider-owned resources when the server shuts down."""

        close = getattr(self._provider, "close", None)
        if callable(close):
            close()

    def _assert_authorized(self, authorization: str | None) -> None:
        """Reject unauthorized HTTP requests with a stable 401 response."""

        expected = self._bearer_token
        if not expected:
            return
        actual = str(authorization or "").strip()
        if not actual.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="missing_bearer_token")
        token = actual[7:].strip()
        if not hmac.compare_digest(token, expected):
            raise HTTPException(status_code=401, detail="invalid_bearer_token")

    def _trace_event(
        self,
        msg: str,
        *,
        kind: str,
        details: dict[str, object],
        trace_id: str | None,
        level: str = "INFO",
        kpi: dict[str, object] | None = None,
    ) -> None:
        tracer = self._forensics
        if not isinstance(tracer, WorkflowForensics):
            return
        tracer.event(
            kind=kind,
            msg=msg,
            details=details,
            trace_id=trace_id,
            level=level,
            kpi=kpi,
            loc_skip=2,
        )


def _build_remote_asr_provider(config: TwinrConfig) -> _SpeechToTextProvider:
    """Build the configured remote-ASR transcription provider."""

    provider_name = str(getattr(config, "stt_provider", "openai") or "openai").strip().lower()
    if provider_name == "deepgram":
        return DeepgramSpeechToTextProvider(config=config)
    if provider_name == "openai":
        return OpenAISpeechToTextProvider(backend=OpenAIBackend(config=config))
    raise RuntimeError(f"Unsupported remote ASR provider for voice gateway: {provider_name}")


def _provider_label(config: TwinrConfig) -> str:
    """Return a short stable provider label for health/debug responses."""

    return str(getattr(config, "stt_provider", "openai") or "openai").strip().lower() or "openai"


def _parse_audio_request(*, content_type_header: str | None, body: bytes) -> _ParsedAudioRequest:
    """Parse either raw audio bytes or Twinr's multipart upload contract."""

    content_type = str(content_type_header or "").strip()
    if not body:
        raise HTTPException(status_code=400, detail="empty_audio_body")
    if content_type.lower().startswith("multipart/form-data"):
        return _parse_multipart_audio_request(content_type_header=content_type, body=body)
    if content_type.lower().startswith("audio/"):
        return _ParsedAudioRequest(
            audio_bytes=bytes(body),
            filename="audio.wav",
            content_type=content_type,
        )
    raise HTTPException(status_code=415, detail="unsupported_content_type")


def _parse_multipart_audio_request(*, content_type_header: str, body: bytes) -> _ParsedAudioRequest:
    """Parse the bounded multipart upload emitted by `RemoteAsrBackendAdapter`."""

    message = BytesParser(policy=email_policy_default).parsebytes(
        (
            "MIME-Version: 1.0\r\n"
            f"Content-Type: {content_type_header}\r\n\r\n"
        ).encode("utf-8")
        + body
    )
    if not message.is_multipart():
        raise HTTPException(status_code=400, detail="invalid_multipart_payload")
    audio_bytes: bytes | None = None
    filename = "audio.wav"
    content_type = "application/octet-stream"
    language: str | None = None
    mode: str | None = None
    prompt: str | None = None
    for part in message.iter_parts():
        field_name = str(part.get_param("name", header="content-disposition") or "").strip()
        payload = bytes(part.get_payload(decode=True) or b"")
        if field_name == "audio":
            audio_bytes = payload
            filename = str(part.get_filename() or filename).strip() or filename
            content_type = str(part.get_content_type() or content_type).strip() or content_type
            continue
        value = payload.decode(part.get_content_charset() or "utf-8", errors="replace").strip()
        if field_name == "language":
            language = value or None
        elif field_name == "mode":
            mode = value or None
        elif field_name == "prompt":
            prompt = value or None
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="missing_audio_part")
    return _ParsedAudioRequest(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
        language=language,
        mode=mode,
        prompt=prompt,
    )


def _infer_duration_sec(audio_bytes: bytes, *, content_type: str) -> float | None:
    """Infer WAV duration when the request already carries WAV bytes."""

    normalized_type = str(content_type or "").strip().lower()
    if normalized_type not in {"audio/wav", "audio/x-wav", "audio/wave"}:
        return None
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                return None
            return wav_file.getnframes() / frame_rate
    except (wave.Error, EOFError):
        return None


__all__ = [
    "RemoteAsrHttpService",
    "remote_asr_url_targets_local_orchestrator",
]
