# CHANGELOG: 2026-03-29
# BUG-1: Fixed raw-audio filename mislabeling (`audio.wav` for every MIME type), which could break codec detection for MP3/WebM/MP4/M4A uploads.
# BUG-2: Fixed false negatives in local-orchestrator detection when the server binds `0.0.0.0`/`::` but the remote-ASR URL uses a concrete local IP or hostname.
# SEC-1: Replaced unbounded full-body reads and ad-hoc multipart parsing with streamed, size-bounded parsing, bounded provider concurrency, request admission backpressure, and safer default auth behavior.
# IMP-1: Upgraded the response path to preserve structured provider metadata (language, segments, duration, provider) whenever the backing provider exposes it.
# IMP-2: Added provider-aware audio format normalization/validation, request-size enforcement, and Pi-class deployment defaults for timeouts and payload bounds.

"""Expose the colocated remote-ASR HTTP surface for the voice gateway.

The voice websocket session talks to a bounded HTTP transcription contract via
``RemoteAsrBackendAdapter``. This module hosts that contract inside the same
FastAPI service when the configured remote-ASR URL points back to the local
orchestrator server. The route stays transport-focused: auth, bounded
multipart/raw audio parsing, provider dispatch, and compact JSON responses.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
import functools
import hashlib
import hmac
import io
import ipaddress
import logging
import os
import socket
from typing import Any, Protocol
from urllib.parse import urlsplit
import wave

from fastapi import APIRouter, Header, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.forensics import WorkflowForensics
from twinr.providers.deepgram import DeepgramSpeechToTextProvider
from twinr.providers.openai import OpenAIBackend, OpenAISpeechToTextProvider

try:
    from starlette.formparsers import MultiPartException, MultiPartParser
except Exception:  # pragma: no cover - starlette is a hard dependency via FastAPI
    MultiPartException = RuntimeError  # type: ignore[assignment]
    MultiPartParser = None  # type: ignore[assignment]

MultiPartParserBase = MultiPartParser if MultiPartParser is not None else object


logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_BIND_ALL_HOSTS = frozenset({"0.0.0.0", "::", ""})

# The colocated voice gateway should only need short-turn uploads. Keep the
# default tighter than vendor API maxima so a Pi 4 remains stable under abuse.
# BREAKING: deployments relying on larger uploads must raise this via config.
_DEFAULT_MAX_AUDIO_BYTES = 8 * 1024 * 1024  # 8 MiB
_DEFAULT_MAX_MULTIPART_OVERHEAD_BYTES = 128 * 1024  # 128 KiB
_DEFAULT_MAX_CONCURRENCY = max(1, min(2, os.cpu_count() or 1))
_DEFAULT_ADMISSION_TIMEOUT_SEC = 0.75
_DEFAULT_PROVIDER_TIMEOUT_SEC = 20.0
_DEFAULT_MAX_PROMPT_CHARS = 2048
_DEFAULT_MAX_LANGUAGE_CHARS = 32
_DEFAULT_MAX_MODE_CHARS = 32
_DEFAULT_MAX_MULTIPART_FIELDS = 8
_DEFAULT_MAX_MULTIPART_FILES = 1

# Common containerized formats supported by modern transcription APIs.
# OpenAI currently supports flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, and webm.
_SUPPORTED_AUDIO_CONTENT_TYPES = frozenset(
    {
        "audio/flac",
        "audio/m4a",
        "audio/mp3",
        "audio/mp4",
        "audio/mpeg",
        "audio/mpga",
        "audio/ogg",
        "audio/wav",
        "audio/wave",
        "audio/webm",
        "audio/x-flac",
        "audio/x-m4a",
        "audio/x-wav",
        "application/octet-stream",  # accepted only when bytes/filename sniffing resolves to a supported format
        "application/ogg",
        "video/mp4",
        "video/webm",
    }
)

_CANONICAL_CONTENT_TYPE_ALIASES: dict[str, str] = {
    "audio/flac": "audio/flac",
    "audio/m4a": "audio/mp4",
    "audio/mp3": "audio/mpeg",
    "audio/mp4": "audio/mp4",
    "audio/mpeg": "audio/mpeg",
    "audio/mpga": "audio/mpeg",
    "audio/ogg": "audio/ogg",
    "audio/wav": "audio/wav",
    "audio/wave": "audio/wav",
    "audio/webm": "audio/webm",
    "audio/x-flac": "audio/flac",
    "audio/x-m4a": "audio/mp4",
    "audio/x-wav": "audio/wav",
    "application/ogg": "audio/ogg",
    "video/mp4": "audio/mp4",
    "video/webm": "audio/webm",
}

_CONTENT_TYPE_TO_EXTENSION: dict[str, str] = {
    "audio/flac": "flac",
    "audio/mpeg": "mp3",
    "audio/mp4": "mp4",
    "audio/ogg": "ogg",
    "audio/wav": "wav",
    "audio/webm": "webm",
}


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
    ) -> object:
        """Transcribe one bounded audio payload into plain text or a structured result."""


@dataclass(frozen=True, slots=True)
class _ParsedAudioRequest:
    """Store one normalized remote-ASR HTTP request."""

    audio_bytes: bytes
    filename: str
    content_type: str
    language: str | None = None
    mode: str | None = None
    prompt: str | None = None


@dataclass(frozen=True, slots=True)
class _NormalizedTranscriptionResult:
    """Store one normalized provider result for the HTTP response surface."""

    text: str
    language: str | None
    segments: list[dict[str, object]]
    duration_sec: float | None
    provider: str | None


class _PayloadTooLarge(RuntimeError):
    """Signal request-body overrun while streaming/parsing."""


class _BoundedMultiPartParser(MultiPartParserBase):  # type: ignore[misc]
    """Starlette multipart parser with explicit per-file byte limits."""

    def __init__(
        self,
        headers: Any,
        stream: AsyncGenerator[bytes, None],
        *,
        max_files: int,
        max_fields: int,
        max_part_size: int,
        max_file_size: int,
    ) -> None:
        if MultiPartParser is None:
            raise RuntimeError("multipart_parser_unavailable")
        super().__init__(
            headers=headers,
            stream=stream,
            max_files=max_files,
            max_fields=max_fields,
            max_part_size=max_part_size,
        )
        self._max_file_size = max_file_size
        self._current_file_bytes = 0

    def on_part_begin(self) -> None:
        self._current_file_bytes = 0
        parent_handler = getattr(MultiPartParserBase, "on_part_begin", None)
        if callable(parent_handler):
            parent_handler(self)

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        message_bytes = data[start:end]
        if getattr(self, "_current_part", None) is not None and getattr(self._current_part, "file", None) is not None:
            self._current_file_bytes += len(message_bytes)
            if self._current_file_bytes > self._max_file_size:
                raise MultiPartException(
                    f"File exceeded maximum size of {int(self._max_file_size / 1024)}KB."
                )
        parent_handler = getattr(MultiPartParserBase, "on_part_data", None)
        if callable(parent_handler):
            parent_handler(self, data, start, end)


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
    if hostname == orchestrator_host:
        return True
    if _is_bind_all_host(orchestrator_host):
        return _host_points_to_this_machine(hostname)
    if _is_loopback_host(hostname):
        return _is_loopback_host(orchestrator_host) or _is_bind_all_host(orchestrator_host)
    if _is_loopback_host(orchestrator_host):
        return False
    return bool(_resolve_host_addresses(hostname) & _resolve_host_addresses(orchestrator_host))


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
        self._provider_name = _provider_label(config)
        self._bearer_token = (
            str(getattr(config, "voice_orchestrator_remote_asr_bearer_token", "") or "").strip() or None
        )
        self._forensics = forensics if isinstance(forensics, WorkflowForensics) and forensics.enabled else None

        # BREAKING: if no bearer token is configured, only loopback callers are
        # accepted by default. Disable explicitly if you truly need LAN callers.
        self._allow_unauthenticated_loopback_only = _config_bool(
            config,
            "voice_orchestrator_remote_asr_allow_unauthenticated_loopback_only",
            default=True,
        )
        self._max_audio_bytes = _config_int(
            config,
            "voice_orchestrator_remote_asr_max_audio_bytes",
            default=_DEFAULT_MAX_AUDIO_BYTES,
            minimum=64 * 1024,
        )
        self._max_request_bytes = self._max_audio_bytes + _config_int(
            config,
            "voice_orchestrator_remote_asr_max_multipart_overhead_bytes",
            default=_DEFAULT_MAX_MULTIPART_OVERHEAD_BYTES,
            minimum=8 * 1024,
        )
        self._max_prompt_chars = _config_int(
            config,
            "voice_orchestrator_remote_asr_max_prompt_chars",
            default=_DEFAULT_MAX_PROMPT_CHARS,
            minimum=0,
        )
        self._max_language_chars = _config_int(
            config,
            "voice_orchestrator_remote_asr_max_language_chars",
            default=_DEFAULT_MAX_LANGUAGE_CHARS,
            minimum=0,
        )
        self._max_mode_chars = _config_int(
            config,
            "voice_orchestrator_remote_asr_max_mode_chars",
            default=_DEFAULT_MAX_MODE_CHARS,
            minimum=0,
        )
        self._max_multipart_fields = _config_int(
            config,
            "voice_orchestrator_remote_asr_max_multipart_fields",
            default=_DEFAULT_MAX_MULTIPART_FIELDS,
            minimum=1,
        )
        self._max_multipart_files = _config_int(
            config,
            "voice_orchestrator_remote_asr_max_multipart_files",
            default=_DEFAULT_MAX_MULTIPART_FILES,
            minimum=1,
        )
        self._max_concurrency = _config_int(
            config,
            "voice_orchestrator_remote_asr_max_concurrency",
            default=_DEFAULT_MAX_CONCURRENCY,
            minimum=1,
        )
        self._admission_timeout_sec = _config_float(
            config,
            "voice_orchestrator_remote_asr_admission_timeout_sec",
            default=_DEFAULT_ADMISSION_TIMEOUT_SEC,
            minimum=0.0,
        )
        self._provider_timeout_sec = _config_float(
            config,
            "voice_orchestrator_remote_asr_timeout_sec",
            default=_DEFAULT_PROVIDER_TIMEOUT_SEC,
            minimum=1.0,
        )
        self._default_language = (
            str(getattr(config, "voice_orchestrator_remote_asr_language", "") or "").strip() or None
        )

        self._transcription_slots = asyncio.Semaphore(self._max_concurrency)
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_concurrency,
            thread_name_prefix="twinr-remote-asr",
        )

    def build_router(self) -> APIRouter:
        """Return the FastAPI router for health and transcription calls."""

        router = APIRouter()
        service = self

        @router.get("/health")
        async def health() -> dict[str, object]:
            return {
                "ok": True,
                "service": "remote_asr",
                "provider": service._provider_name,
            }

        @router.post("/v1/transcribe")
        async def transcribe(
            request: Request,
            authorization: str | None = Header(default=None),
        ) -> dict[str, object]:
            service._assert_authorized(request=request, authorization=authorization)
            request_id = str(request.headers.get("x-twinr-request-id") or "").strip()
            trace_id = (
                str(request.headers.get("x-twinr-voice-trace-id") or "").strip()
                or request_id
                or None
            )

            started_at = asyncio.get_running_loop().time()
            try:
                parsed = await _parse_audio_request_from_http_request(
                    request=request,
                    provider_name=service._provider_name,
                    max_audio_bytes=service._max_audio_bytes,
                    max_request_bytes=service._max_request_bytes,
                    max_prompt_chars=service._max_prompt_chars,
                    max_language_chars=service._max_language_chars,
                    max_mode_chars=service._max_mode_chars,
                    max_multipart_fields=service._max_multipart_fields,
                    max_multipart_files=service._max_multipart_files,
                )
            except HTTPException as exc:
                service._trace_event(
                    "remote_asr_service_rejected",
                    kind="http",
                    trace_id=trace_id,
                    level="WARNING" if exc.status_code < 500 else "ERROR",
                    details={
                        "request_id": request_id or None,
                        "client_host": _request_client_host(request),
                        "reason": str(exc.detail),
                        "status_code": exc.status_code,
                        "provider": service._provider_name,
                    },
                    kpi={"latency_ms": round((asyncio.get_running_loop().time() - started_at) * 1000.0, 3)},
                )
                raise

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
                "client_host": _request_client_host(request),
                "content_length": request.headers.get("content-length"),
                "audio_bytes": len(parsed.audio_bytes),
                "audio_sha256": hashlib.sha256(parsed.audio_bytes).hexdigest()[:12],
                "filename": parsed.filename,
                "content_type": parsed.content_type,
                "language": parsed.language,
                "mode": parsed.mode,
                "prompt_chars": len(str(parsed.prompt or "").strip()),
                "provider": service._provider_name,
            }
            service._trace_event(
                "remote_asr_service_request",
                kind="http",
                trace_id=trace_id,
                details=request_details,
            )

            try:
                async with service._transcription_slot():
                    provider_result = await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            service._executor,
                            functools.partial(
                                service._provider.transcribe,
                                parsed.audio_bytes,
                                filename=parsed.filename,
                                content_type=parsed.content_type,
                                language=parsed.language or service._default_language,
                                prompt=parsed.prompt,
                            ),
                        ),
                        timeout=service._provider_timeout_sec,
                    )
            except HTTPException as exc:
                service._trace_event(
                    "remote_asr_service_busy",
                    kind="http",
                    trace_id=trace_id,
                    level="WARNING",
                    details={**request_details, "reason": str(exc.detail), "status_code": exc.status_code},
                    kpi={"latency_ms": round((asyncio.get_running_loop().time() - started_at) * 1000.0, 3)},
                )
                raise
            except TimeoutError as exc:
                service._trace_event(
                    "remote_asr_service_timeout",
                    kind="exception",
                    trace_id=trace_id,
                    level="ERROR",
                    details={**request_details, "timeout_sec": service._provider_timeout_sec},
                    kpi={"latency_ms": round((asyncio.get_running_loop().time() - started_at) * 1000.0, 3)},
                )
                raise HTTPException(status_code=504, detail="transcription_timeout") from exc
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
                raise HTTPException(status_code=503, detail="transcription_failed") from exc

            normalized = _normalize_transcription_result(
                provider_result,
                fallback_language=parsed.language or service._default_language,
                fallback_duration_sec=_infer_duration_sec(
                    parsed.audio_bytes,
                    content_type=parsed.content_type,
                    filename=parsed.filename,
                ),
                fallback_provider=service._provider_name,
            )
            response_text = normalized.text.strip()
            service._trace_event(
                "remote_asr_service_response",
                kind="http",
                trace_id=trace_id,
                details={
                    **request_details,
                    "response_text_chars": len(response_text),
                    "response_segments": len(normalized.segments),
                    "response_language": normalized.language,
                },
                kpi={"latency_ms": round((asyncio.get_running_loop().time() - started_at) * 1000.0, 3)},
            )
            return {
                "text": response_text,
                "language": normalized.language,
                "segments": normalized.segments,
                "duration_sec": normalized.duration_sec,
                "mode": parsed.mode,
                "provider": normalized.provider or service._provider_name,
            }

        return router

    def close(self) -> None:
        """Release provider-owned resources when the server shuts down."""

        close = getattr(self._provider, "close", None)
        if callable(close):
            close()
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _assert_authorized(self, *, request: Request, authorization: str | None) -> None:
        """Reject unauthorized HTTP requests with a stable response surface."""

        expected = self._bearer_token
        if expected:
            actual = str(authorization or "").strip()
            if not actual.lower().startswith("bearer "):
                raise HTTPException(status_code=401, detail="missing_bearer_token")
            token = actual[7:].strip()
            if not hmac.compare_digest(token, expected):
                raise HTTPException(status_code=401, detail="invalid_bearer_token")
            return

        if self._allow_unauthenticated_loopback_only and not _is_loopback_host(_request_client_host(request)):
            raise HTTPException(status_code=401, detail="bearer_token_required")

    @asynccontextmanager
    async def _transcription_slot(self) -> AsyncGenerator[None, None]:
        """Bound concurrent provider calls and fail fast when the route is overloaded."""

        try:
            await asyncio.wait_for(
                self._transcription_slots.acquire(),
                timeout=self._admission_timeout_sec,
            )
        except TimeoutError as exc:
            raise HTTPException(status_code=429, detail="remote_asr_busy") from exc
        try:
            yield
        finally:
            self._transcription_slots.release()

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

    provider_name = _provider_label(config)
    if provider_name == "deepgram":
        return DeepgramSpeechToTextProvider(config=config)
    if provider_name == "openai":
        return OpenAISpeechToTextProvider(backend=OpenAIBackend(config=config))
    raise RuntimeError(f"Unsupported remote ASR provider for voice gateway: {provider_name}")


def _provider_label(config: TwinrConfig) -> str:
    """Return a short stable provider label for health/debug responses."""

    return str(getattr(config, "stt_provider", "openai") or "openai").strip().lower() or "openai"


def _config_int(config: TwinrConfig, name: str, *, default: int, minimum: int = 0) -> int:
    """Read one integer config value defensively."""

    raw = getattr(config, name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _config_float(config: TwinrConfig, name: str, *, default: float, minimum: float = 0.0) -> float:
    """Read one float config value defensively."""

    raw = getattr(config, name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _config_bool(config: TwinrConfig, name: str, *, default: bool) -> bool:
    """Read one boolean config value defensively."""

    raw = getattr(config, name, default)
    if isinstance(raw, bool):
        return raw
    text = str(raw or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


async def _parse_audio_request_from_http_request(
    *,
    request: Request,
    provider_name: str,
    max_audio_bytes: int,
    max_request_bytes: int,
    max_prompt_chars: int,
    max_language_chars: int,
    max_mode_chars: int,
    max_multipart_fields: int,
    max_multipart_files: int,
) -> _ParsedAudioRequest:
    """Parse either raw audio bytes or the bounded multipart upload contract."""

    content_type_header = str(request.headers.get("content-type") or "").strip()
    normalized_content_type = _normalize_content_type(content_type_header)
    if not normalized_content_type:
        raise HTTPException(status_code=415, detail="missing_content_type")

    if _is_multipart_content_type(normalized_content_type):
        _validate_content_length_header(request=request, max_bytes=max_request_bytes)
        return await _parse_multipart_audio_request(
            request=request,
            provider_name=provider_name,
            max_audio_bytes=max_audio_bytes,
            max_request_bytes=max_request_bytes,
            max_prompt_chars=max_prompt_chars,
            max_language_chars=max_language_chars,
            max_mode_chars=max_mode_chars,
            max_multipart_fields=max_multipart_fields,
            max_multipart_files=max_multipart_files,
        )

    if not _is_supported_raw_audio_content_type(normalized_content_type):
        raise HTTPException(status_code=415, detail="unsupported_content_type")

    _validate_content_length_header(request=request, max_bytes=max_audio_bytes)
    audio_bytes = await _read_stream_bounded(
        request.stream(),
        max_bytes=max_audio_bytes,
        empty_detail="empty_audio_body",
    )
    filename, canonical_content_type = _normalize_audio_identity(
        provider_name=provider_name,
        audio_bytes=audio_bytes,
        filename=None,
        content_type=normalized_content_type,
    )
    return _ParsedAudioRequest(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=canonical_content_type,
    )


async def _parse_multipart_audio_request(
    *,
    request: Request,
    provider_name: str,
    max_audio_bytes: int,
    max_request_bytes: int,
    max_prompt_chars: int,
    max_language_chars: int,
    max_mode_chars: int,
    max_multipart_fields: int,
    max_multipart_files: int,
) -> _ParsedAudioRequest:
    """Parse the bounded multipart upload emitted by `RemoteAsrBackendAdapter`."""

    if MultiPartParser is None:
        raise HTTPException(status_code=503, detail="multipart_parser_unavailable")

    stream = _iter_request_stream_bounded(request.stream(), max_bytes=max_request_bytes)
    parser = _BoundedMultiPartParser(
        headers=request.headers,
        stream=stream,
        max_files=max_multipart_files,
        max_fields=max_multipart_fields,
        max_part_size=max_audio_bytes,
        max_file_size=max_audio_bytes,
    )
    try:
        form = await parser.parse()
    except _PayloadTooLarge as exc:
        _close_parser_tempfiles(parser)
        raise HTTPException(status_code=413, detail="audio_too_large") from exc
    except MultiPartException as exc:
        _close_parser_tempfiles(parser)
        detail = str(exc).lower()
        if "maximum size" in detail or "too many files" in detail or "too many fields" in detail:
            raise HTTPException(status_code=413, detail="audio_too_large") from exc
        raise HTTPException(status_code=400, detail="invalid_multipart_payload") from exc
    except HTTPException:
        _close_parser_tempfiles(parser)
        raise
    except Exception as exc:
        _close_parser_tempfiles(parser)
        logger.exception("Failed to parse multipart remote ASR request")
        raise HTTPException(status_code=400, detail="invalid_multipart_payload") from exc

    try:
        audio_values = form.getlist("audio") if hasattr(form, "getlist") else [form.get("audio")]
    except Exception:
        audio_values = [form.get("audio")]
    if len(audio_values) != 1 or audio_values[0] is None:
        await _close_form_uploads(form)
        raise HTTPException(status_code=400, detail="missing_audio_part")
    audio_part = audio_values[0]

    language = _normalize_optional_field(
        "language",
        form.get("language"),
        max_chars=max_language_chars,
    )
    mode = _normalize_optional_field(
        "mode",
        form.get("mode"),
        max_chars=max_mode_chars,
    )
    prompt = _normalize_optional_field(
        "prompt",
        form.get("prompt"),
        max_chars=max_prompt_chars,
    )

    try:
        if isinstance(audio_part, UploadFile) or callable(getattr(audio_part, "read", None)):
            audio_bytes = await audio_part.read()
            filename = str(getattr(audio_part, "filename", "") or "").strip() or None
            content_type = str(getattr(audio_part, "content_type", "") or "").strip() or None
        elif isinstance(audio_part, str):
            try:
                audio_bytes = audio_part.encode("latin-1")
            except UnicodeEncodeError:
                try:
                    audio_bytes = audio_part.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise HTTPException(status_code=415, detail="invalid_audio_part") from exc
            filename = None
            content_type = None
        else:
            raise HTTPException(status_code=400, detail="invalid_audio_part")
    finally:
        await _close_form_uploads(form)

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty_audio_body")
    if len(audio_bytes) > max_audio_bytes:
        raise HTTPException(status_code=413, detail="audio_too_large")

    normalized_filename, normalized_content_type = _normalize_audio_identity(
        provider_name=provider_name,
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
    )
    return _ParsedAudioRequest(
        audio_bytes=audio_bytes,
        filename=normalized_filename,
        content_type=normalized_content_type,
        language=language,
        mode=mode,
        prompt=prompt,
    )


def _close_parser_tempfiles(parser: Any) -> None:
    """Best-effort cleanup for multipart temp files on parser failure paths."""

    files = getattr(parser, "_files_to_close_on_error", None)
    if not isinstance(files, list):
        return
    for file_obj in files:
        try:
            file_obj.close()
        except Exception:
            logger.debug("Failed to close multipart temp file during parser cleanup", exc_info=True)


async def _close_form_uploads(form: Any) -> None:
    """Best-effort close for UploadFile items stored in the parsed multipart form."""

    if not isinstance(form, Mapping):
        return
    for value in form.values():
        if isinstance(value, UploadFile):
            try:
                await value.close()
            except Exception:
                logger.debug("Failed to close uploaded file", exc_info=True)


def _normalize_optional_field(name: str, value: object, *, max_chars: int) -> str | None:
    """Normalize one optional multipart text field with strict size bounds."""

    if value is None:
        return None
    if isinstance(value, UploadFile):
        raise HTTPException(status_code=400, detail=f"invalid_{name}_field")
    normalized = str(value).strip()
    if not normalized:
        return None
    if max_chars >= 0 and len(normalized) > max_chars:
        raise HTTPException(status_code=413, detail=f"{name}_too_large")
    return normalized


def _validate_content_length_header(*, request: Request, max_bytes: int) -> None:
    """Reject obviously oversized bodies before any parser work starts."""

    raw = str(request.headers.get("content-length") or "").strip()
    if not raw:
        return
    try:
        value = int(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid_content_length") from exc
    if value < 0:
        raise HTTPException(status_code=400, detail="invalid_content_length")
    if value > max_bytes:
        raise HTTPException(status_code=413, detail="audio_too_large")


async def _read_stream_bounded(
    stream: AsyncGenerator[bytes, None],
    *,
    max_bytes: int,
    empty_detail: str,
) -> bytes:
    """Read one async byte stream with a hard upper bound."""

    buffer = bytearray()
    total = 0
    async for chunk in stream:
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="audio_too_large")
        buffer.extend(chunk)
    if not buffer:
        raise HTTPException(status_code=400, detail=empty_detail)
    return bytes(buffer)


async def _iter_request_stream_bounded(
    stream: AsyncGenerator[bytes, None],
    *,
    max_bytes: int,
) -> AsyncGenerator[bytes, None]:
    """Yield one ASGI request body stream while enforcing a total byte ceiling."""

    total = 0
    async for chunk in stream:
        if chunk:
            total += len(chunk)
            if total > max_bytes:
                raise _PayloadTooLarge()
        yield chunk


def _normalize_audio_identity(
    *,
    provider_name: str,
    audio_bytes: bytes,
    filename: str | None,
    content_type: str | None,
) -> tuple[str, str]:
    """Validate and normalize filename/content-type pairs for provider uploads."""

    provided_content_type = _canonicalize_audio_content_type(content_type)
    sniffed_content_type = _sniff_audio_content_type(audio_bytes)
    suffix_from_filename = _extract_supported_extension(filename)

    canonical_content_type = provided_content_type
    if canonical_content_type == "application/octet-stream":
        canonical_content_type = None

    if canonical_content_type is None and suffix_from_filename is not None:
        canonical_content_type = _content_type_from_extension(suffix_from_filename)
    if canonical_content_type is None and sniffed_content_type is not None:
        canonical_content_type = sniffed_content_type
    if canonical_content_type is None:
        raise HTTPException(status_code=415, detail="unsupported_audio_format")

    if canonical_content_type not in _SUPPORTED_AUDIO_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="unsupported_audio_format")

    canonical_content_type = _canonicalize_audio_content_type(canonical_content_type)
    if canonical_content_type is None:
        raise HTTPException(status_code=415, detail="unsupported_audio_format")

    extension = suffix_from_filename or _CONTENT_TYPE_TO_EXTENSION.get(canonical_content_type)
    if extension is None and sniffed_content_type is not None:
        extension = _CONTENT_TYPE_TO_EXTENSION.get(_canonicalize_audio_content_type(sniffed_content_type) or "")
    if extension is None:
        raise HTTPException(status_code=415, detail="unsupported_audio_format")

    normalized_filename = _normalized_filename(filename, extension=extension)
    if provider_name not in {"openai", "deepgram"}:
        raise HTTPException(status_code=415, detail="unsupported_audio_format")
    return normalized_filename, canonical_content_type


def _is_supported_raw_audio_content_type(content_type: str) -> bool:
    """Return whether the raw request media type is a supported audio container."""

    normalized = _normalize_content_type(content_type)
    return bool(normalized and normalized in _SUPPORTED_AUDIO_CONTENT_TYPES and not _is_multipart_content_type(normalized))


def _is_multipart_content_type(content_type: str | None) -> bool:
    """Return whether the media type is multipart/form-data."""

    return _normalize_content_type(content_type) == "multipart/form-data"


def _normalize_content_type(content_type: str | None) -> str:
    """Drop media-type parameters and lowercase for stable comparisons."""

    return str(content_type or "").split(";", 1)[0].strip().lower()


def _canonicalize_audio_content_type(content_type: str | None) -> str | None:
    """Map content-type aliases onto one stable canonical audio media type."""

    normalized = _normalize_content_type(content_type)
    if not normalized:
        return None
    if normalized not in _SUPPORTED_AUDIO_CONTENT_TYPES:
        return None
    return _CANONICAL_CONTENT_TYPE_ALIASES.get(normalized, normalized)


def _sniff_audio_content_type(audio_bytes: bytes) -> str | None:
    """Best-effort magic-byte sniffing for common audio containers."""

    if len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return "audio/wav"
    if audio_bytes.startswith(b"fLaC"):
        return "audio/flac"
    if audio_bytes.startswith(b"OggS"):
        return "audio/ogg"
    if len(audio_bytes) >= 12 and audio_bytes[4:8] == b"ftyp":
        brand = audio_bytes[8:12]
        if brand in {b"M4A ", b"M4B ", b"M4P ", b"m4a ", b"isom", b"mp41", b"mp42"}:
            return "audio/mp4"
    if audio_bytes.startswith(b"ID3"):
        return "audio/mpeg"
    if len(audio_bytes) >= 2 and audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0:
        return "audio/mpeg"
    if audio_bytes.startswith(b"\x1A\x45\xDF\xA3") and b"webm" in audio_bytes[:4096].lower():
        return "audio/webm"
    return None


def _extract_supported_extension(filename: str | None) -> str | None:
    """Extract one supported extension from a user-supplied filename."""

    cleaned = str(filename or "").strip().lower()
    if "." not in cleaned:
        return None
    suffix = cleaned.rsplit(".", 1)[-1]
    if suffix in {"flac", "m4a", "mp3", "mp4", "ogg", "wav", "webm"}:
        return suffix
    if suffix in {"mpeg", "mpga"}:
        return "mp3"
    return None


def _content_type_from_extension(extension: str) -> str | None:
    """Map a known filename extension to its canonical upload content type."""

    normalized = str(extension or "").strip().lower()
    if normalized == "flac":
        return "audio/flac"
    if normalized in {"m4a", "mp4"}:
        return "audio/mp4"
    if normalized == "mp3":
        return "audio/mpeg"
    if normalized == "ogg":
        return "audio/ogg"
    if normalized == "wav":
        return "audio/wav"
    if normalized == "webm":
        return "audio/webm"
    return None


def _normalized_filename(filename: str | None, *, extension: str) -> str:
    """Return a provider-safe synthetic filename with the correct suffix."""

    stem = str(filename or "").strip()
    if stem and "." in stem:
        stem = stem.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        base = stem.rsplit(".", 1)[0].strip() or "audio"
    else:
        base = "audio"
    safe_base = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in base).strip("._") or "audio"
    safe_extension = "".join(ch for ch in extension if ch.isalnum()).lower() or "bin"
    return f"{safe_base}.{safe_extension}"


def _normalize_transcription_result(
    result: object,
    *,
    fallback_language: str | None,
    fallback_duration_sec: float | None,
    fallback_provider: str,
) -> _NormalizedTranscriptionResult:
    """Collapse provider-specific result shapes onto Twinr's compact JSON contract."""

    if isinstance(result, str):
        payload: dict[str, Any] = {"text": result}
    else:
        encoded = jsonable_encoder(result)
        payload = encoded if isinstance(encoded, dict) else {"text": str(result or "")}

    text = str(
        payload.get("text")
        or payload.get("transcript")
        or payload.get("utterance")
        or ""
    ).strip()
    language = _first_present_str(payload, "language", "lang") or fallback_language
    duration_sec = _coerce_optional_float(payload.get("duration_sec"))
    if duration_sec is None:
        duration_sec = _coerce_optional_float(payload.get("duration"))
    if duration_sec is None:
        duration_sec = fallback_duration_sec

    raw_segments = payload.get("segments") or payload.get("speaker_segments") or []
    encoded_segments = jsonable_encoder(raw_segments)
    segments: list[dict[str, object]]
    if isinstance(encoded_segments, list):
        segments = [segment for segment in encoded_segments if isinstance(segment, dict)]
    else:
        segments = []

    provider = _first_present_str(payload, "provider") or fallback_provider
    return _NormalizedTranscriptionResult(
        text=text,
        language=language,
        segments=segments,
        duration_sec=duration_sec,
        provider=provider,
    )


def _first_present_str(payload: Mapping[str, object], *keys: str) -> str | None:
    """Return the first non-empty stringish value among several candidate keys."""

    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _coerce_optional_float(value: object) -> float | None:
    """Coerce one optional provider field into a float without throwing."""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_duration_sec(audio_bytes: bytes, *, content_type: str, filename: str | None = None) -> float | None:
    """Infer WAV duration when the request already carries WAV bytes."""

    normalized_type = _canonicalize_audio_content_type(content_type)
    if normalized_type != "audio/wav" and _extract_supported_extension(filename) != "wav":
        if _sniff_audio_content_type(audio_bytes) != "audio/wav":
            return None
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                return None
            return wav_file.getnframes() / frame_rate
    except (wave.Error, EOFError):
        return None


def _request_client_host(request: Request) -> str:
    """Extract one stable client-host label for auth/tracing."""

    client = getattr(request, "client", None)
    host = str(getattr(client, "host", "") or "").strip().lower()
    return host or ""


def _is_loopback_host(host: str | None) -> bool:
    """Return whether one host label resolves to loopback semantics."""

    normalized = str(host or "").strip().lower()
    if normalized in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _is_bind_all_host(host: str | None) -> bool:
    """Return whether one host label means 'listen on all interfaces'."""

    normalized = str(host or "").strip().lower()
    if normalized in _BIND_ALL_HOSTS:
        return True
    try:
        return ipaddress.ip_address(normalized).is_unspecified
    except ValueError:
        return False


def _host_points_to_this_machine(host: str) -> bool:
    """Return whether one hostname/IP resolves to the current machine."""

    normalized = str(host or "").strip().lower()
    if _is_loopback_host(normalized):
        return True
    if normalized in _local_hostnames():
        return True
    resolved = _resolve_host_addresses(normalized)
    if not resolved:
        return False
    return bool(resolved & _local_interface_addresses())


@functools.lru_cache(maxsize=1)
def _local_hostnames() -> frozenset[str]:
    """Return common local hostname aliases for this process."""

    values = {
        socket.gethostname().strip().lower(),
        socket.getfqdn().strip().lower(),
        "localhost",
    }
    return frozenset(value for value in values if value)


@functools.lru_cache(maxsize=64)
def _resolve_host_addresses(host: str) -> frozenset[str]:
    """Resolve one host label into IP-address strings."""

    normalized = str(host or "").strip().lower()
    if not normalized:
        return frozenset()
    if _is_loopback_host(normalized):
        return frozenset({"127.0.0.1", "::1"})
    try:
        return frozenset({ipaddress.ip_address(normalized).compressed})
    except ValueError:
        pass

    resolved: set[str] = set()
    try:
        infos = socket.getaddrinfo(normalized, None, type=socket.SOCK_STREAM)
    except OSError:
        return frozenset()
    for _, _, _, _, sockaddr in infos:
        try:
            resolved.add(ipaddress.ip_address(sockaddr[0]).compressed)
        except (TypeError, ValueError):
            continue
    return frozenset(resolved)


@functools.lru_cache(maxsize=1)
def _local_interface_addresses() -> frozenset[str]:
    """Collect loopback and hostname-resolved interface addresses for this machine."""

    resolved: set[str] = {"127.0.0.1", "::1"}
    for host in _local_hostnames():
        resolved.update(_resolve_host_addresses(host))
    return frozenset(resolved)


__all__ = [
    "RemoteAsrHttpService",
    "remote_asr_url_targets_local_orchestrator",
]
