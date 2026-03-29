# CHANGELOG: 2026-03-29
# BUG-1: Add bounded retries for transient network failures and retryable 5xx/429/408 responses; the old client failed fast on common Pi/Wi-Fi outages.
# BUG-2: Cap outbound audio size and inbound response size to prevent runaway memory use and malformed/oversized gateway payloads from destabilizing the process.
# SEC-1: Reject plain HTTP for non-loopback endpoints unless explicitly opted in; bearer tokens, audio, and trace metadata must not traverse the network in cleartext.
# SEC-2: Disable trust in proxy-related environment variables by default so HTTP(S)_PROXY / system proxy settings cannot silently exfiltrate audio or auth headers.
# SEC-3: Remove handcrafted multipart header construction and sanitize header-projected metadata to eliminate CR/LF and multipart header injection paths from caller-controlled fields.
# IMP-1: Replace urllib.request (which sends Connection: close and gives no pooled client lifecycle) with a long-lived httpx.Client for keep-alive pooling, strict timeout classes, optional HTTP/2, and explicit close semantics.
# IMP-2: Add extensible multipart form fields so modern ASR features such as diarization, timestamp granularities, chunking, hotwords/keyterms, and provider-specific knobs can be passed through without rewriting the adapter.

"""Call the remote thh1986 ASR service from Twinr's server voice gateway.

This module owns only the bounded HTTP client contract and WAV upload shaping
for the remote transcript-first path. It does not contain activation logic or
session policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
from pathlib import Path
import random
import re
import ssl
import time
from typing import Any
from urllib.parse import SplitResult, urlsplit
from uuid import uuid4

import httpx  # BREAKING: requires httpx for pooled, bounded transport instead of urllib.request.

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

    _RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

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
        retry_backoff_max_s: float = 4.0,
        retry_jitter_s: float = 0.15,
        connect_timeout_s: float | None = None,
        read_timeout_s: float | None = None,
        write_timeout_s: float | None = None,
        pool_timeout_s: float | None = None,
        max_audio_bytes: int = 25 * 1024 * 1024,
        max_response_bytes: int = 256 * 1024,
        http2: bool = False,
        trust_env: bool = False,
        allow_insecure_http: bool = False,
        ssl_context: ssl.SSLContext | None = None,
        ca_bundle_path: str | None = None,
        client_cert_path: str | None = None,
        client_key_path: str | None = None,
        client_cert_password: str | None = None,
        max_keepalive_connections: int = 4,
        max_connections: int = 8,
        keepalive_expiry_s: float = 20.0,
        extra_form_fields: Mapping[str, object] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed_base_url = _normalize_base_url(base_url, allow_insecure_http=allow_insecure_http)
        normalized_timeout_s = max(0.25, float(timeout_s))
        self.base_url = parsed_base_url.geturl().rstrip("/")
        self.bearer_token = _normalize_optional_text(bearer_token)
        self.language = _normalize_optional_text(language)
        self.mode = _normalize_optional_text(mode) or "active_listening"
        self.timeout_s = normalized_timeout_s
        self.retry_attempts = max(0, int(retry_attempts))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.retry_backoff_max_s = max(0.0, float(retry_backoff_max_s))
        self.retry_jitter_s = max(0.0, float(retry_jitter_s))
        self.max_audio_bytes = max(1, int(max_audio_bytes))
        self.max_response_bytes = max(1024, int(max_response_bytes))
        self.http2 = bool(http2)
        # BREAKING: proxy/system environment variables are ignored unless trust_env=True.
        self.trust_env = bool(trust_env)
        self.allow_insecure_http = bool(allow_insecure_http)
        self.extra_form_fields = dict(extra_form_fields or {})
        self._forensics: WorkflowForensics | None = None

        if ssl_context is not None and any(
            value is not None for value in (ca_bundle_path, client_cert_path, client_key_path, client_cert_password)
        ):
            raise ValueError(
                "Pass either ssl_context or certificate path parameters, not both."
            )
        if client_key_path and not client_cert_path:
            raise ValueError("client_key_path requires client_cert_path.")
        if client_cert_password and not client_cert_path:
            raise ValueError("client_cert_password requires client_cert_path.")
        self._ssl_context = None
        if parsed_base_url.scheme == "http" and any(
            value is not None for value in (ssl_context, ca_bundle_path, client_cert_path, client_key_path, client_cert_password)
        ):
            raise ValueError("TLS configuration cannot be used with an http base_url.")
        if parsed_base_url.scheme == "https":
            self._ssl_context = ssl_context or _build_ssl_context(
                ca_bundle_path=ca_bundle_path,
                client_cert_path=client_cert_path,
                client_key_path=client_key_path,
                client_cert_password=client_cert_password,
            )

        self._timeout = httpx.Timeout(
            timeout=normalized_timeout_s,
            connect=_pick_timeout(connect_timeout_s, normalized_timeout_s),
            read=_pick_timeout(read_timeout_s, normalized_timeout_s),
            write=_pick_timeout(write_timeout_s, normalized_timeout_s),
            pool=_pick_timeout(pool_timeout_s, normalized_timeout_s),
        )
        normalized_max_keepalive_connections = max(0, int(max_keepalive_connections))
        normalized_max_connections = max(1, int(max_connections), normalized_max_keepalive_connections)
        self._limits = httpx.Limits(
            max_keepalive_connections=normalized_max_keepalive_connections,
            max_connections=normalized_max_connections,
            keepalive_expiry=max(0.0, float(keepalive_expiry_s)),
        )
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self._timeout,
            limits=self._limits,
            verify=self._ssl_context if self._ssl_context is not None else True,
            follow_redirects=False,
            trust_env=self.trust_env,
            http2=self.http2,
            transport=transport,
            headers={"Accept": "application/json"},
        )

    def __enter__(self) -> "RemoteAsrBackendAdapter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""

        self._client.close()

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
        extra_form_fields: Mapping[str, object] | None = None,
    ) -> str:
        """Transcribe one in-memory audio payload and return only the text field."""

        result = self.transcribe_bytes(
            audio_bytes,
            filename=filename,
            content_type=content_type,
            language=language,
            prompt=prompt,
            extra_form_fields=extra_form_fields,
        )
        return result.text

    def transcribe_capture(
        self,
        capture: AmbientAudioCaptureWindow,
        *,
        language: str | None = None,
        prompt: str | None = None,
        extra_form_fields: Mapping[str, object] | None = None,
    ) -> RemoteAsrTranscript:
        """Convert one PCM capture window to WAV and transcribe it through the service."""

        audio_bytes = pcm16_to_wav_bytes(
            capture.pcm_bytes,
            sample_rate=capture.sample_rate,
            channels=capture.channels,
        )
        return self.transcribe_bytes(
            audio_bytes,
            filename="voice-window.wav",
            content_type="audio/wav",
            language=language,
            prompt=prompt,
            extra_form_fields=extra_form_fields,
        )

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        *,
        filename: str,
        content_type: str,
        language: str | None = None,
        prompt: str | None = None,
        extra_form_fields: Mapping[str, object] | None = None,
    ) -> RemoteAsrTranscript:
        """Upload one audio payload to ``/v1/transcribe`` and normalize the JSON response."""

        if not audio_bytes:
            raise ValueError("audio_bytes must not be empty")
        if len(audio_bytes) > self.max_audio_bytes:
            raise ValueError(
                f"audio_bytes exceeds max_audio_bytes ({len(audio_bytes)} > {self.max_audio_bytes})"
            )

        normalized_prompt = _normalize_optional_text(prompt) or ""
        safe_filename = _sanitize_filename(filename)
        safe_content_type = _sanitize_content_type(content_type)
        form_fields = _build_multipart_text_fields(
            base_fields={
                "language": _normalize_optional_text(language) or self.language,
                "mode": self.mode,
                "prompt": normalized_prompt,
            },
            default_extra_fields=self.extra_form_fields,
            request_extra_fields=extra_form_fields,
        )
        files = {
            "audio": (
                safe_filename,
                audio_bytes,
                safe_content_type,
            )
        }

        request_id = uuid4().hex[:12]
        request_context = dict(_ACTIVE_REMOTE_ASR_REQUEST_CONTEXT.get() or {})
        trace_id = str(request_context.get("trace_id") or "").strip() or None
        headers = {
            "X-Twinr-Request-Id": request_id,
            **_build_trace_headers(request_context),
        }
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        request_started_at = time.monotonic()
        self._trace_event(
            "remote_asr_client_request",
            kind="http",
            trace_id=trace_id,
            details={
                "request_id": request_id,
                "base_url": self.base_url,
                "filename": safe_filename,
                "content_type": safe_content_type,
                "audio_bytes": len(audio_bytes),
                "mode": self.mode,
                "prompt_chars": len(normalized_prompt),
                "timeout_s": self.timeout_s,
                "retry_attempts": self.retry_attempts,
                **request_context,
            },
        )

        attempt = 0
        while True:
            attempt += 1
            try:
                with self._client.stream(
                    "POST",
                    "/v1/transcribe",
                    headers=headers,
                    data=form_fields,
                    files=files,
                ) as response:
                    payload_bytes = _read_bounded_response_bytes(
                        response,
                        limit_bytes=self.max_response_bytes,
                    )
                    status = int(response.status_code)
                    retry_after_s = _retry_after_delay_s(response.headers.get("Retry-After"))
                if status in self._RETRYABLE_STATUS_CODES and attempt <= self.retry_attempts:
                    self._trace_event(
                        "remote_asr_client_retryable_status",
                        kind="warning",
                        trace_id=trace_id,
                        level="WARN",
                        details={
                            "request_id": request_id,
                            "http_status": status,
                            "attempt": attempt,
                            "retry_after_s": retry_after_s,
                            **request_context,
                        },
                    )
                    self._sleep_before_retry(attempt=attempt, retry_after_s=retry_after_s)
                    continue
                break
            except httpx.HTTPStatusError as exc:
                detail = _truncate_text(str(exc), limit=240)
                self._trace_event(
                    "remote_asr_client_http_error",
                    kind="warning",
                    trace_id=trace_id,
                    level="WARN",
                    details={
                        "request_id": request_id,
                        "attempt": attempt,
                        "detail": detail,
                        **request_context,
                    },
                )
                raise RemoteAsrServiceError(f"Remote ASR service error: {detail}") from exc
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if attempt <= self.retry_attempts:
                    self._trace_event(
                        "remote_asr_client_retryable_connect_error",
                        kind="warning",
                        trace_id=trace_id,
                        level="WARN",
                        details={
                            "request_id": request_id,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "reason": _truncate_text(str(exc), limit=240),
                            **request_context,
                        },
                    )
                    self._sleep_before_retry(attempt=attempt, retry_after_s=None)
                    continue
                self._trace_event(
                    "remote_asr_client_unavailable",
                    kind="warning",
                    trace_id=trace_id,
                    level="WARN",
                    details={
                        "request_id": request_id,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "reason": _truncate_text(str(exc), limit=240),
                        **request_context,
                    },
                )
                raise RemoteAsrServiceError(f"Remote ASR service unavailable: {exc}") from exc
            except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ReadError, httpx.WriteError, httpx.PoolTimeout) as exc:
                if attempt <= self.retry_attempts:
                    self._trace_event(
                        "remote_asr_client_retryable_io_error",
                        kind="warning",
                        trace_id=trace_id,
                        level="WARN",
                        details={
                            "request_id": request_id,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "reason": _truncate_text(str(exc), limit=240),
                            **request_context,
                        },
                    )
                    self._sleep_before_retry(attempt=attempt, retry_after_s=None)
                    continue
                self._trace_event(
                    "remote_asr_client_transport_error",
                    kind="warning",
                    trace_id=trace_id,
                    level="WARN",
                    details={
                        "request_id": request_id,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "reason": _truncate_text(str(exc), limit=240),
                        **request_context,
                    },
                )
                raise RemoteAsrServiceError(f"Remote ASR transport error: {exc}") from exc
            except httpx.HTTPError as exc:
                self._trace_event(
                    "remote_asr_client_httpx_error",
                    kind="warning",
                    trace_id=trace_id,
                    level="WARN",
                    details={
                        "request_id": request_id,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "reason": _truncate_text(str(exc), limit=240),
                        **request_context,
                    },
                )
                raise RemoteAsrServiceError(f"Remote ASR service unavailable: {exc}") from exc

        payload_text = payload_bytes.decode("utf-8", errors="replace").strip()
        if status >= 400:
            self._trace_event(
                "remote_asr_client_http_error",
                kind="warning",
                trace_id=trace_id,
                level="WARN",
                details={
                    "request_id": request_id,
                    "http_status": status,
                    "attempt": attempt,
                    "detail": _truncate_text(payload_text, limit=240),
                    **request_context,
                },
            )
            raise RemoteAsrServiceError(
                f"Remote ASR service returned HTTP {status}: {_truncate_text(payload_text, limit=240)}"
            )
        try:
            payload = json.loads(payload_text or "{}")
        except json.JSONDecodeError as exc:
            raise RemoteAsrServiceError(
                f"Remote ASR service returned invalid JSON: {_truncate_text(payload_text, limit=160)}"
            ) from exc

        if not isinstance(payload, dict):
            raise RemoteAsrServiceError("Remote ASR service returned a non-object JSON payload.")

        text = _normalize_optional_text(payload.get("text")) or _normalize_optional_text(payload.get("transcript")) or ""
        language_value = (
            _normalize_optional_text(payload.get("language"))
            or _normalize_optional_text(payload.get("detected_language"))
        )
        raw_segments = payload.get("segments")
        segments: tuple[dict[str, Any], ...]
        if isinstance(raw_segments, list):
            segments = tuple(dict(segment) for segment in raw_segments if isinstance(segment, Mapping))
        else:
            segments = ()
        duration_sec = _parse_duration_sec(payload.get("duration_sec"))
        if duration_sec is None:
            duration_sec = _parse_duration_sec(payload.get("duration"))

        self._trace_event(
            "remote_asr_client_response",
            kind="http",
            trace_id=trace_id,
            details={
                "request_id": request_id,
                "http_status": status,
                "attempt_count": attempt,
                "response_text_chars": len(text),
                "response_language": language_value,
                "response_duration_sec": duration_sec,
                "response_segments": len(segments),
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

    def _sleep_before_retry(self, *, attempt: int, retry_after_s: float | None) -> None:
        delay_s = retry_after_s if retry_after_s is not None else self._retry_delay_s(attempt)
        if delay_s > 0.0:
            time.sleep(delay_s)

    def _retry_delay_s(self, attempt: int) -> float:
        """Return one bounded backoff delay for retryable service contention."""

        if self.retry_backoff_s <= 0.0:
            return 0.0
        delay_s = self.retry_backoff_s * (2 ** max(0, attempt - 1))
        if self.retry_jitter_s > 0.0:
            delay_s += random.uniform(0.0, self.retry_jitter_s)
        if self.retry_backoff_max_s > 0.0:
            delay_s = min(delay_s, self.retry_backoff_max_s)
        return delay_s

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


_ALLOWED_FORM_FIELD_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-\[\]]{1,64}$")
_ALLOWED_CONTENT_TYPE_RE = re.compile(r"^[A-Za-z0-9!#$&^_.+\-]+/[A-Za-z0-9!#$&^_.+\-]+$")


def _pick_timeout(value: float | None, default: float) -> float:
    return max(0.05, float(value if value is not None else default))


def _normalize_optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _parse_duration_sec(value: object) -> float | None:
    try:
        duration_sec = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    if duration_sec is None or not math.isfinite(duration_sec) or duration_sec < 0.0:
        return None
    return duration_sec


def _normalize_base_url(base_url: str, *, allow_insecure_http: bool) -> SplitResult:
    normalized_base_url = str(base_url or "").strip().rstrip("/")
    if not normalized_base_url:
        raise ValueError("Remote ASR base_url must not be empty.")
    parsed = urlsplit(normalized_base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Remote ASR base_url must use http or https.")
    if not parsed.hostname:
        raise ValueError("Remote ASR base_url must include a hostname.")
    if parsed.query or parsed.fragment:
        raise ValueError("Remote ASR base_url must not include query parameters or fragments.")
    if parsed.username or parsed.password:
        raise ValueError("Remote ASR base_url must not include embedded credentials.")
    if parsed.scheme == "http" and not allow_insecure_http and not _is_loopback_hostname(parsed.hostname):
        # BREAKING: non-loopback cleartext HTTP is blocked by default.
        raise ValueError(
            "Remote ASR base_url must use https for non-loopback endpoints."
        )
    return parsed


def _is_loopback_hostname(hostname: str) -> bool:
    candidate = str(hostname or "").strip().lower()
    return candidate in {"localhost", "127.0.0.1", "::1", "[::1]"}


def _build_ssl_context(
    *,
    ca_bundle_path: str | None,
    client_cert_path: str | None,
    client_key_path: str | None,
    client_cert_password: str | None,
) -> ssl.SSLContext:
    context = ssl.create_default_context()
    if ca_bundle_path:
        context.load_verify_locations(cafile=str(ca_bundle_path))
    if client_cert_path:
        context.load_cert_chain(
            certfile=str(client_cert_path),
            keyfile=str(client_key_path) if client_key_path else None,
            password=client_cert_password,
        )
    return context


def _build_multipart_text_fields(
    *,
    base_fields: Mapping[str, object],
    default_extra_fields: Mapping[str, object] | None,
    request_extra_fields: Mapping[str, object] | None,
) -> dict[str, str | list[str]]:
    merged_extra_fields = dict(default_extra_fields or {})
    merged_extra_fields.update(dict(request_extra_fields or {}))
    fields: dict[str, str | list[str]] = {}
    for key, value in base_fields.items():
        _append_form_field(fields, key, value)
    for key, value in merged_extra_fields.items():
        _append_form_field(fields, key, value)
    return fields


def _append_form_field(fields: dict[str, str | list[str]], key: object, value: object) -> None:
    field_name = str(key or "").strip()
    if not field_name:
        return
    if not _ALLOWED_FORM_FIELD_NAME_RE.fullmatch(field_name):
        raise ValueError(f"Invalid multipart field name: {field_name!r}")
    if value is None:
        return
    if isinstance(value, Mapping):
        raise ValueError(f"Multipart field {field_name!r} must not be a nested mapping.")
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _append_form_field(fields, field_name, item)
        return
    if isinstance(value, bool):
        normalized_value = "true" if value else "false"
    elif isinstance(value, bytes):
        normalized_value = value.decode("utf-8", errors="replace").strip()
    else:
        normalized_value = str(value).strip()
    if not normalized_value:
        return
    existing = fields.get(field_name)
    if existing is None:
        fields[field_name] = normalized_value
        return
    if isinstance(existing, list):
        existing.append(normalized_value)
        return
    fields[field_name] = [existing, normalized_value]


def _encode_multipart_form(
    *,
    file_field: str,
    filename: str,
    file_content_type: str,
    file_bytes: bytes,
    text_fields: Mapping[str, object],
) -> tuple[bytes, str]:
    """Build one compact multipart/form-data body for tests and local probes."""

    boundary = f"twinr-boundary-{uuid4().hex}"
    body = bytearray()

    def append_line(line: str) -> None:
        body.extend(line.encode("utf-8"))
        body.extend(b"\r\n")

    normalized_fields = _build_multipart_text_fields(
        base_fields=text_fields,
        default_extra_fields=None,
        request_extra_fields=None,
    )
    safe_file_field = str(file_field or "").strip() or "audio"
    safe_filename = _sanitize_filename(filename)
    safe_content_type = _sanitize_content_type(file_content_type)

    for key, value in normalized_fields.items():
        values = value if isinstance(value, list) else [value]
        for item in values:
            append_line(f"--{boundary}")
            append_line(f'Content-Disposition: form-data; name="{key}"')
            append_line("")
            append_line(str(item))

    append_line(f"--{boundary}")
    append_line(
        f'Content-Disposition: form-data; name="{safe_file_field}"; filename="{safe_filename}"'
    )
    append_line(f"Content-Type: {safe_content_type}")
    append_line("")
    body.extend(bytes(file_bytes))
    body.extend(b"\r\n")
    append_line(f"--{boundary}--")
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _read_bounded_response_bytes(response: httpx.Response, *, limit_bytes: int) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > limit_bytes:
                raise RemoteAsrServiceError(
                    f"Remote ASR response exceeded max_response_bytes ({content_length} > {limit_bytes})."
                )
        except ValueError:
            pass
    chunks: list[bytes] = []
    total_bytes = 0
    for chunk in response.iter_bytes():
        total_bytes += len(chunk)
        if total_bytes > limit_bytes:
            raise RemoteAsrServiceError(
                f"Remote ASR response exceeded max_response_bytes ({total_bytes} > {limit_bytes})."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _retry_after_delay_s(header_value: str | None, *, max_delay_s: float = 30.0) -> float | None:
    text = str(header_value or "").strip()
    if not text:
        return None
    try:
        delay_s = float(text)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        delay_s = (parsed - datetime.now(timezone.utc)).total_seconds()
    if not math.isfinite(delay_s):
        return None
    return max(0.0, min(delay_s, max_delay_s))


def _bounded_header_value(value: object, *, limit: int = 128) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    sanitized = []
    for char in text:
        codepoint = ord(char)
        if codepoint in (9, 32) or 33 <= codepoint <= 126:
            sanitized.append(char)
        elif codepoint < 128:
            sanitized.append(" ")
        else:
            sanitized.append("?")
    normalized = re.sub(r"\s+", " ", "".join(sanitized)).strip()
    if not normalized:
        return None
    return normalized[:limit]


def _truncate_text(text: object, *, limit: int) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit]


def _sanitize_filename(filename: str) -> str:
    candidate = Path(str(filename or "audio.wav")).name or "audio.wav"
    sanitized = []
    for char in candidate:
        codepoint = ord(char)
        if char in {'"', "\\", "/", ";"} or codepoint < 32 or codepoint == 127:
            sanitized.append("_")
        else:
            sanitized.append(char)
    normalized = "".join(sanitized).strip(" .") or "audio.wav"
    return normalized[:128]


def _sanitize_content_type(content_type: str) -> str:
    normalized = str(content_type or "").strip().lower()
    if _ALLOWED_CONTENT_TYPE_RE.fullmatch(normalized):
        return normalized
    return "application/octet-stream"


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


__all__ = [
    "RemoteAsrBackendAdapter",
    "RemoteAsrServiceError",
    "RemoteAsrTranscript",
]
