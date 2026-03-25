"""Call the remote thh1986 ASR service from Twinr's server voice gateway.

This module owns only the bounded HTTP client contract and WAV upload shaping
for the remote transcript-first path. It does not contain activation logic or
session policy.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from uuid import uuid4

from twinr.agent.workflows.forensics import WorkflowForensics
from twinr.hardware.audio import AmbientAudioCaptureWindow, pcm16_to_wav_bytes


class RemoteAsrServiceError(RuntimeError):
    """Raise when the remote ASR service rejects or malforms one transcription."""


@dataclass(frozen=True, slots=True)
class RemoteAsrTranscript:
    """Return one normalized remote ASR response."""

    text: str
    language: str | None = None
    segments: tuple[dict[str, Any], ...] = ()
    duration_sec: float | None = None


class RemoteAsrBackendAdapter:
    """Provide an OpenAI-like ``transcribe`` method backed by one remote ASR API."""

    def __init__(
        self,
        *,
        base_url: str,
        bearer_token: str | None = None,
        language: str | None = None,
        mode: str = "active_listening",
        timeout_s: float = 3.0,
        retry_attempts: int = 0,
        retry_backoff_s: float = 0.35,
    ) -> None:
        normalized_base_url = str(base_url or "").strip().rstrip("/")
        if not normalized_base_url:
            raise ValueError("Remote ASR base_url must not be empty.")
        self.base_url = normalized_base_url
        self.bearer_token = str(bearer_token or "").strip() or None
        self.language = str(language or "").strip() or None
        self.mode = str(mode or "active_listening").strip() or "active_listening"
        self.timeout_s = max(0.25, float(timeout_s))
        self.retry_attempts = max(0, int(retry_attempts))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self._forensics: WorkflowForensics | None = None

    def set_forensics(self, tracer: WorkflowForensics | None) -> None:
        """Bind one shared tracer so client-side ASR calls hit the run pack."""

        if isinstance(tracer, WorkflowForensics) and tracer.enabled:
            self._forensics = tracer
            return
        self._forensics = None

    @contextmanager
    def bind_request_context(self, details: dict[str, object] | None):
        """Attach bounded context to the next remote-ASR request on this thread."""

        token = _ACTIVE_REMOTE_ASR_REQUEST_CONTEXT.set(dict(details or {}))
        try:
            yield
        finally:
            _ACTIVE_REMOTE_ASR_REQUEST_CONTEXT.reset(token)

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """Transcribe one in-memory audio payload and return only the text field."""

        result = self.transcribe_bytes(
            audio_bytes,
            filename=filename,
            content_type=content_type,
            language=language,
            prompt=prompt,
        )
        return result.text

    def transcribe_capture(self, capture: AmbientAudioCaptureWindow) -> RemoteAsrTranscript:
        """Convert one PCM capture window to WAV and transcribe it through the service."""

        audio_bytes = pcm16_to_wav_bytes(
            capture.pcm_bytes,
            sample_rate=capture.sample_rate,
            channels=capture.channels,
        )
        return self.transcribe_bytes(audio_bytes, filename="voice-window.wav", content_type="audio/wav")

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        language: str | None = None,
        prompt: str | None = None,
    ) -> RemoteAsrTranscript:
        """Upload one audio payload to ``/v1/transcribe`` and normalize the JSON response."""

        if not audio_bytes:
            raise ValueError("audio_bytes must not be empty")
        normalized_prompt = str(prompt or "").strip()
        request_body, request_content_type = _encode_multipart_form(
            file_field="audio",
            filename=filename,
            file_content_type=content_type,
            file_bytes=audio_bytes,
            text_fields={
                "language": str(language or self.language or "").strip(),
                "mode": self.mode,
                "prompt": normalized_prompt,
            },
        )
        headers = {
            "Accept": "application/json",
            "Content-Type": request_content_type,
        }
        request_id = uuid4().hex[:12]
        request_context = dict(_ACTIVE_REMOTE_ASR_REQUEST_CONTEXT.get() or {})
        trace_id = str(request_context.get("trace_id") or "").strip() or None
        headers["X-Twinr-Request-Id"] = request_id
        headers.update(_build_trace_headers(request_context))
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = urllib_request.Request(
            f"{self.base_url}/v1/transcribe",
            data=request_body,
            headers=headers,
            method="POST",
        )
        attempt = 0
        request_started_at = time.monotonic()
        self._trace_event(
            "remote_asr_client_request",
            kind="http",
            trace_id=trace_id,
            details={
                "request_id": request_id,
                "base_url": self.base_url,
                "filename": filename,
                "content_type": content_type,
                "audio_bytes": len(audio_bytes),
                "mode": self.mode,
                "prompt_chars": len(normalized_prompt),
                "timeout_s": self.timeout_s,
                **request_context,
            },
        )
        while True:
            try:
                with urllib_request.urlopen(request, timeout=self.timeout_s) as response:
                    payload_bytes = response.read()
                    status = int(getattr(response, "status", response.getcode()))
                break
            except urllib_error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace").strip()
                self._trace_event(
                    "remote_asr_client_http_error",
                    kind="warning",
                    trace_id=trace_id,
                    level="WARN",
                    details={
                        "request_id": request_id,
                        "http_status": int(exc.code),
                        "attempt": attempt,
                        "detail": detail[:240],
                        **request_context,
                    },
                )
                if exc.code == 429 and attempt < self.retry_attempts:
                    attempt += 1
                    time.sleep(self._retry_delay_s(attempt))
                    continue
                raise RemoteAsrServiceError(
                    f"Remote ASR service returned HTTP {exc.code}: {detail[:240]}"
                ) from exc
            except urllib_error.URLError as exc:
                self._trace_event(
                    "remote_asr_client_unavailable",
                    kind="warning",
                    trace_id=trace_id,
                    level="WARN",
                    details={
                        "request_id": request_id,
                        "attempt": attempt,
                        "reason": str(exc.reason),
                        **request_context,
                    },
                )
                raise RemoteAsrServiceError(f"Remote ASR service unavailable: {exc.reason}") from exc
        payload_text = payload_bytes.decode("utf-8", errors="replace").strip()
        if status >= 400:
            raise RemoteAsrServiceError(f"Remote ASR service returned HTTP {status}: {payload_text[:240]}")
        try:
            payload = json.loads(payload_text or "{}")
        except json.JSONDecodeError as exc:
            raise RemoteAsrServiceError("Remote ASR service returned invalid JSON.") from exc
        text = str(payload.get("text") or "").strip()
        language_value = str(payload.get("language") or "").strip() or None
        raw_segments = payload.get("segments")
        segments: tuple[dict[str, Any], ...]
        if isinstance(raw_segments, list):
            segments = tuple(segment for segment in raw_segments if isinstance(segment, dict))
        else:
            segments = ()
        duration_value = payload.get("duration_sec")
        try:
            duration_sec = float(duration_value) if duration_value is not None else None
        except (TypeError, ValueError):
            duration_sec = None
        self._trace_event(
            "remote_asr_client_response",
            kind="http",
            trace_id=trace_id,
            details={
                "request_id": request_id,
                "http_status": status,
                "attempt_count": attempt + 1,
                "response_text_chars": len(text),
                "response_language": language_value,
                "response_duration_sec": duration_sec,
                **request_context,
            },
            kpi={"latency_ms": round((time.monotonic() - request_started_at) * 1000.0, 3)},
        )
        return RemoteAsrTranscript(
            text=text,
            language=language_value,
            segments=segments,
            duration_sec=duration_sec,
        )

    def _retry_delay_s(self, attempt: int) -> float:
        """Return one bounded backoff delay for retryable service contention."""

        if self.retry_backoff_s <= 0.0:
            return 0.0
        return self.retry_backoff_s * attempt

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


_ACTIVE_REMOTE_ASR_REQUEST_CONTEXT: ContextVar[dict[str, object] | None] = ContextVar(
    "twinr_remote_asr_request_context",
    default=None,
)


def _bounded_header_value(value: object, *, limit: int = 128) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:limit]


def _build_trace_headers(context: dict[str, object]) -> dict[str, str]:
    """Project bounded request metadata into transport headers."""

    mapping = {
        "session_id": "X-Twinr-Voice-Session-Id",
        "trace_id": "X-Twinr-Voice-Trace-Id",
        "stage": "X-Twinr-Voice-Stage",
        "state": "X-Twinr-Voice-State",
        "origin_state": "X-Twinr-Voice-Origin-State",
        "capture_duration_ms": "X-Twinr-Voice-Capture-Duration-Ms",
        "capture_average_rms": "X-Twinr-Voice-Capture-Average-Rms",
        "capture_peak_rms": "X-Twinr-Voice-Capture-Peak-Rms",
        "capture_active_ratio": "X-Twinr-Voice-Capture-Active-Ratio",
        "capture_signal_nonzero_ratio": "X-Twinr-Voice-Nonzero-Ratio",
        "capture_signal_clipped_ratio": "X-Twinr-Voice-Clipped-Ratio",
        "capture_signal_zero_crossing_ratio": "X-Twinr-Voice-Zero-Crossing-Ratio",
        "capture_signal_sha256": "X-Twinr-Voice-Capture-Sha256",
    }
    headers: dict[str, str] = {}
    for key, header_name in mapping.items():
        value = _bounded_header_value(context.get(key))
        if value is not None:
            headers[header_name] = value
    return headers


def _encode_multipart_form(
    *,
    file_field: str,
    filename: str,
    file_content_type: str,
    file_bytes: bytes,
    text_fields: dict[str, str],
) -> tuple[bytes, str]:
    """Encode one small multipart body without third-party HTTP helpers."""

    boundary = f"twinr-remote-asr-{uuid4().hex}"
    chunks: list[bytes] = []
    for key, value in text_fields.items():
        normalized_value = str(value or "").strip()
        if not normalized_value:
            continue
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                normalized_value.encode("utf-8"),
                b"\r\n",
            ]
        )
    safe_filename = Path(str(filename or "audio.wav")).name or "audio.wav"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; filename="{safe_filename}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {file_content_type or 'application/octet-stream'}\r\n\r\n".encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


__all__ = [
    "RemoteAsrBackendAdapter",
    "RemoteAsrServiceError",
    "RemoteAsrTranscript",
]
