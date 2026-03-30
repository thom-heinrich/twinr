"""Provide Deepgram-backed speech-to-text adapters for Twinr.

The module exposes a synchronous batch transcription helper and a bounded
streaming session wrapper around Deepgram's websocket API. Runtime workflows
use this provider through the higher-level contracts in
``twinr.agent.base_agent.contracts``.
"""

# CHANGELOG: 2026-03-30
# BUG-1: Streaming prompt bias was silently ignored. Live STT now applies
#        Deepgram keyterms/keywords on both Nova and Flux streams.
# BUG-2: Flux used the wrong transport contract (`/v1/listen`, Nova events,
#        `Finalize`, `language=`). Streaming now switches automatically to the
#        2026 Flux `/v2/listen` protocol and parses TurnInfo events.
# BUG-3: Nova `UtteranceEnd` / `SpeechStarted` handling was misconfigured
#        because `vad_events=true` was never sent. Endpoint callbacks now fire
#        reliably when those features are enabled.
# BUG-4: `transcribe_path()` loaded whole files into RAM and accepted arbitrary
#        regular files. It now streams bounded, probable-audio files only.
# BUG-5: Batch and connect paths lacked transient retry/backoff, which caused
#        avoidable turn failures on flaky Raspberry Pi / Wi-Fi deployments.
# SEC-1: Insecure `http://` / `ws://` Deepgram URLs could leak senior audio and
#        credentials. Insecure transport is now refused by default outside
#        explicit local / self-hosted deployments.
# SEC-2: The provider now prefers short-lived Deepgram access tokens when
#        available and opts requests out of Deepgram's Model Improvement
#        Program by default unless config disables it.
# IMP-1: Added 2026-first streaming support for Flux end-of-turn detection,
#        eager EOT, TurnResumed handling, and configurable thresholds.
# IMP-2: Added PCM micro-chunking (80 ms default for Flux) and bounded retries /
#        request-id-rich errors for lower latency and better observability.

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit
import ipaddress
import json
import logging
import mimetypes
import os
import stat
import time

import httpx
from websockets.sync.client import connect as websocket_connect

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import (
    StreamingSpeechEndpointEvent,
    StreamingSpeechToTextSession,
    StreamingTranscriptionResult,
)

logger = logging.getLogger(__name__)
_DEEPGRAM_KEYWORD_INTENSIFIER = 2.0
_DEEPGRAM_TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_DEEPGRAM_ACCESS_TOKEN_ENV = "DEEPGRAM_TOKEN"
_DEEPGRAM_API_KEY_ENV = "DEEPGRAM_API_KEY"

_AUDIO_EXTENSIONS = frozenset(
    {
        ".aac",
        ".aif",
        ".aiff",
        ".amr",
        ".au",
        ".caf",
        ".flac",
        ".m4a",
        ".m4b",
        ".mka",
        ".mkv",
        ".mov",
        ".mp2",
        ".mp3",
        ".mp4",
        ".oga",
        ".ogg",
        ".opus",
        ".pcm",
        ".raw",
        ".wav",
        ".wave",
        ".webm",
        ".wma",
    }
)
_AUDIO_MIME_PREFIXES = ("audio/",)
_AUDIO_MIME_ALLOWLIST = frozenset(
    {
        "application/ogg",
        "application/octet-stream",
        "video/mp4",
        "video/quicktime",
        "video/webm",
    }
)
_FLUX_SUPPORTED_SAMPLE_RATES = frozenset({8000, 16000, 24000, 44100, 48000})


def _normalize_model_name(model: str | None) -> str:
    """Normalize provider model names while preserving backwards compatibility."""
    normalized = str(model or "").strip()
    if not normalized:
        return "nova-3"
    if normalized.casefold() == "flux":
        logger.warning(
            "Deepgram model 'flux' is ambiguous in 2026; using 'flux-general-en'."
        )
        return "flux-general-en"
    return normalized


def _is_flux_model(model: str | None) -> bool:
    """Return whether the configured model uses Deepgram's Flux protocol."""
    return _normalize_model_name(model).casefold().startswith("flux")


def _is_nova3_model(model: str | None) -> bool:
    """Return whether the configured model is from the Nova-3 family."""
    return _normalize_model_name(model).casefold().startswith("nova-3")


def _extract_transcript(payload: dict[str, object]) -> str:
    """Extract the best transcript string from a batch Deepgram response."""
    if not isinstance(payload, dict):
        raise RuntimeError("Deepgram response payload must be a JSON object")

    results = payload.get("results")
    if not isinstance(results, dict):
        raise RuntimeError("Deepgram response missing 'results' object")

    channels = results.get("channels")
    if not isinstance(channels, list) or not channels:
        return ""

    first_channel = channels[0]
    if not isinstance(first_channel, dict):
        raise RuntimeError("Deepgram response channel entry was not an object")

    alternatives = first_channel.get("alternatives")
    if not isinstance(alternatives, list) or not alternatives:
        return ""

    first_alternative = alternatives[0]
    if not isinstance(first_alternative, dict):
        raise RuntimeError("Deepgram response alternative entry was not an object")

    transcript = first_alternative.get("transcript", "")
    if not isinstance(transcript, str):
        raise RuntimeError("Deepgram response did not contain a string transcript")
    return transcript.strip()


def _extract_request_id_from_headers(headers: Any) -> str | None:
    """Extract a Deepgram request identifier from an HTTP or websocket header map."""
    if headers is None:
        return None
    for key in ("x-dg-request-id", "dg-request-id", "DG-Request-Id", "x-request-id"):
        try:
            value = headers.get(key)
        except Exception:
            value = None
        if value:
            return str(value).strip()
    return None


def _extract_request_id_from_payload(payload: dict[str, object]) -> str | None:
    """Extract a Deepgram request identifier from a JSON payload."""
    request_id = payload.get("request_id")
    if isinstance(request_id, str) and request_id.strip():
        return request_id.strip()

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        nested_request_id = metadata.get("request_id")
        if isinstance(nested_request_id, str) and nested_request_id.strip():
            return nested_request_id.strip()

    return None


def _extract_transient_retry_delay_s(response: httpx.Response) -> float | None:
    """Parse a simple Retry-After header into seconds when present."""
    retry_after = response.headers.get("Retry-After")
    if retry_after is None:
        return None
    try:
        delay_s = float(retry_after.strip())
    except (TypeError, ValueError):
        return None
    return max(0.0, delay_s)


def _redacted_error_details(response: httpx.Response) -> str:
    """Build a compact, request-id-rich error string without leaking payload bytes."""
    request_id = _extract_request_id_from_headers(response.headers)
    response_text = ""
    try:
        response_text = response.text
    except Exception:
        response_text = ""
    response_text = " ".join(response_text.split())
    if len(response_text) > 240:
        response_text = f"{response_text[:237]}..."

    detail_parts = [f"status={response.status_code}"]
    if request_id:
        detail_parts.append(f"request_id={request_id}")
    if response_text:
        detail_parts.append(f"body={response_text}")
    return ", ".join(detail_parts)


def _best_response_request_id(
    *,
    response: httpx.Response | None = None,
    payload: dict[str, object] | None = None,
) -> str | None:
    """Return the most useful request id available from response or payload."""
    payload_request_id = _extract_request_id_from_payload(payload or {})
    if payload_request_id:
        return payload_request_id
    if response is not None:
        return _extract_request_id_from_headers(response.headers)
    return None


def _prompt_bias_terms(
    prompt: str | None,
    *,
    max_terms: int = 100,
    max_token_budget: int = 500,
) -> tuple[str, ...]:
    """Extract a bounded list of Deepgram-compatible bias terms from a prompt string."""
    normalized_prompt = str(prompt or "").strip()
    if not normalized_prompt:
        return ()

    translated = normalized_prompt
    for separator in ("\n", "\r", ";", "|"):
        translated = translated.replace(separator, ",")

    terms: list[str] = []
    seen: set[str] = set()
    consumed_tokens = 0

    for raw_part in translated.split(","):
        term = raw_part.strip(" \t.,!?:")
        normalized_term = term.casefold()
        if not term or normalized_term in seen:
            continue

        approximate_tokens = max(1, len(term.split()))
        if len(terms) >= max_terms or consumed_tokens + approximate_tokens > max_token_budget:
            logger.warning(
                "Deepgram prompt bias truncated to %s terms / ~%s tokens for API limits.",
                len(terms),
                consumed_tokens,
            )
            break

        seen.add(normalized_term)
        terms.append(term)
        consumed_tokens += approximate_tokens

    return tuple(terms)


def _deepgram_prompt_params(*, model: str, prompt: str | None) -> list[tuple[str, str]]:
    """Map one provider-agnostic prompt string onto Deepgram bias parameters."""
    terms = _prompt_bias_terms(prompt)
    if not terms:
        return []

    normalized_model = _normalize_model_name(model).lower()
    if normalized_model.startswith("flux") or normalized_model.startswith("nova-3"):
        return [("keyterm", term) for term in terms]
    return [("keywords", f"{term}:{_DEEPGRAM_KEYWORD_INTENSIFIER:g}") for term in terms]


def _extract_streaming_transcript(payload: dict[str, object]) -> str:
    """Extract the latest transcript fragment from a Nova streaming payload."""
    channel = payload.get("channel", {})
    if not isinstance(channel, dict):
        return ""
    alternatives = channel.get("alternatives", ())
    if not isinstance(alternatives, list) or not alternatives:
        return ""
    first = alternatives[0]
    if not isinstance(first, dict):
        return ""
    transcript = first.get("transcript", "")
    return transcript.strip() if isinstance(transcript, str) else ""


def _extract_streaming_confidence(payload: dict[str, object]) -> float | None:
    """Extract a confidence estimate from a Nova streaming payload."""
    channel = payload.get("channel", {})
    if not isinstance(channel, dict):
        return None
    alternatives = channel.get("alternatives", ())
    if not isinstance(alternatives, list) or not alternatives:
        return None
    first = alternatives[0]
    if not isinstance(first, dict):
        return None
    raw_confidence = first.get("confidence")
    if isinstance(raw_confidence, (int, float)):
        confidence = float(raw_confidence)
        if 0.0 <= confidence <= 1.0:
            return confidence
    words = first.get("words")
    if not isinstance(words, list) or not words:
        return None
    return _average_word_confidence(words)


def _average_word_confidence(words: object) -> float | None:
    """Average a Deepgram words array into one confidence score."""
    if not isinstance(words, list) or not words:
        return None
    confidences: list[float] = []
    for word in words:
        if not isinstance(word, dict):
            continue
        raw_word_confidence = word.get("confidence")
        if not isinstance(raw_word_confidence, (int, float)):
            continue
        confidence = float(raw_word_confidence)
        if 0.0 <= confidence <= 1.0:
            confidences.append(confidence)
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def _extract_flux_confidence(payload: dict[str, object]) -> float | None:
    """Extract a confidence estimate from a Flux TurnInfo payload."""
    raw_confidence = payload.get("end_of_turn_confidence")
    if isinstance(raw_confidence, (int, float)):
        confidence = float(raw_confidence)
        if 0.0 <= confidence <= 1.0:
            return confidence
    return _average_word_confidence(payload.get("words"))


def _require_positive_int(name: str, value: int) -> int:
    """Return a validated positive integer for audio transport settings."""
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _require_non_negative_float(name: str, value: Any) -> float:
    """Validate that a configuration value is a non-negative float."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative float") from exc
    if parsed < 0.0:
        raise ValueError(f"{name} must be a non-negative float")
    return parsed


def _require_flux_sample_rate(sample_rate: int) -> int:
    """Validate Flux raw-audio sample rates according to the 2026 API contract."""
    if sample_rate not in _FLUX_SUPPORTED_SAMPLE_RATES:
        raise ValueError(
            "Flux requires sample_rate to be one of "
            f"{sorted(_FLUX_SUPPORTED_SAMPLE_RATES)} for raw audio"
        )
    return sample_rate


def _parse_bool(value: Any, default: bool) -> bool:
    """Parse permissive config booleans without surprising truthiness."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _host_is_local_or_private(host: str | None) -> bool:
    """Return whether a URL host is clearly local / RFC1918 / loopback."""
    normalized = (host or "").strip().strip("[]").lower()
    if not normalized:
        return False
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True
    if normalized.endswith(".local"):
        return True
    if "." not in normalized and ":" not in normalized:
        return True
    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local)


def _validate_transport_security(base_url: str, *, allow_insecure: bool) -> Any:
    """Validate the Deepgram base URL and refuse unsafe remote cleartext transports."""
    normalized_base_url = base_url.strip()
    if not normalized_base_url:
        raise ValueError("Deepgram base URL must not be empty")

    split = urlsplit(normalized_base_url.rstrip("/"))
    if split.scheme not in {"http", "https", "ws", "wss"} or not split.netloc:
        raise ValueError(f"Invalid Deepgram base URL: {base_url!r}")

    hostname = split.hostname
    if split.scheme in {"http", "ws"} and not allow_insecure and not _host_is_local_or_private(hostname):
        # BREAKING: Public cleartext Deepgram transport is now rejected by default.
        raise RuntimeError(
            "Refusing insecure Deepgram transport to a non-local host; use HTTPS/WSS "
            "or set deepgram_allow_insecure_transport=True for an explicit self-hosted deployment"
        )
    return split


def _join_listen_path(base_path: str, version: str) -> str:
    """Replace any trailing Deepgram version/listen suffix with the requested version."""
    parts = [part for part in base_path.split("/") if part]
    if parts and parts[-1] == "listen":
        parts.pop()
    if parts and parts[-1] in {"v1", "v2"}:
        parts.pop()
    parts.extend([version, "listen"])
    return "/" + "/".join(parts)


def _build_listen_url(
    *,
    base_url: str,
    version: str,
    websocket: bool,
    params: Iterable[tuple[str, str]],
    allow_insecure: bool,
) -> str:
    """Build an HTTP(S) or WS(S) Deepgram listen URL with the requested API version."""
    split = _validate_transport_security(base_url, allow_insecure=allow_insecure)
    scheme_map = {
        False: {"http": "http", "https": "https", "ws": "http", "wss": "https"},
        True: {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"},
    }
    scheme = scheme_map[websocket][split.scheme]
    path = _join_listen_path(split.path.rstrip("/"), version)
    return urlunsplit((scheme, split.netloc, path, urlencode(list(params), doseq=True), ""))


def _iter_file_chunks(file_obj, *, chunk_size: int) -> Iterator[bytes]:
    """Yield a regular file as bounded binary chunks."""
    while True:
        chunk = file_obj.read(chunk_size)
        if not chunk:
            return
        yield chunk


def _looks_like_audio_bytes(header: bytes) -> bool:
    """Cheap magic-byte sniffing for common audio containers."""
    if not header:
        return False
    if header.startswith(b"RIFF") and b"WAVE" in header[:16]:
        return True
    if header.startswith(b"fLaC"):
        return True
    if header.startswith(b"OggS"):
        return True
    if header.startswith(b"ID3"):
        return True
    if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return True
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return True
    if header.startswith(b"\x1A\x45\xDF\xA3"):
        return True
    if header.startswith(b"FORM") and b"AIFF" in header[:16]:
        return True
    if header.startswith(b"#!AMR"):
        return True
    return False


def _is_probable_audio_path(
    *,
    path: Path,
    content_type: str,
    header: bytes,
) -> bool:
    """Return whether a local file is very likely to be an audio file."""
    suffix = path.suffix.lower()
    if suffix in _AUDIO_EXTENSIONS:
        return True
    normalized_content_type = str(content_type or "").strip().lower()
    if any(normalized_content_type.startswith(prefix) for prefix in _AUDIO_MIME_PREFIXES):
        return True
    if normalized_content_type in _AUDIO_MIME_ALLOWLIST:
        return True
    return _looks_like_audio_bytes(header)


def _compute_pcm_chunk_bytes(
    *,
    sample_rate: int,
    channels: int,
    chunk_ms: float,
    sample_width_bytes: int = 2,
) -> int:
    """Convert a target PCM chunk duration to a byte count."""
    if chunk_ms <= 0.0:
        return 0
    frames = max(1, int(round(sample_rate * (chunk_ms / 1000.0))))
    return frames * channels * sample_width_bytes


class _DeepgramStreamingSession(StreamingSpeechToTextSession):
    """Manage a bounded Deepgram websocket session for live transcription."""

    def __init__(
        self,
        *,
        connection,
        protocol: str,
        finalize_timeout_s: float,
        keepalive_interval_s: float,
        send_timeout_s: float,
        max_pending_messages: int,
        outgoing_audio_chunk_bytes: int,
        finalize_control_message: str | None,
        close_control_message: str | None,
        keepalive_control_message: str | None,
        on_interim: Callable[[str], None] | None = None,
        on_endpoint: Callable[[StreamingSpeechEndpointEvent], None] | None = None,
    ) -> None:
        """Start sender, reader, and keepalive workers for one websocket stream."""
        self._connection = connection
        self._protocol = str(protocol).strip().lower() or "nova"
        self._finalize_timeout_s = max(0.5, float(finalize_timeout_s))
        self._keepalive_interval_s = max(0.0, float(keepalive_interval_s))
        self._send_timeout_s = max(0.5, float(send_timeout_s))
        self._outgoing_audio_chunk_bytes = max(0, int(outgoing_audio_chunk_bytes))
        self._finalize_control_message = finalize_control_message
        self._close_control_message = close_control_message
        self._keepalive_control_message = keepalive_control_message
        self._on_interim = on_interim
        self._on_endpoint = on_endpoint

        self._state_lock = Lock()
        self._send_lock = Lock()
        self._outgoing: Queue[tuple[bytes | str, Event | None]] = Queue(
            maxsize=max(1, int(max_pending_messages))
        )
        self._done = Event()
        self._closed = Event()
        self._finalize_requested = Event()
        self._stream_error: Exception | None = None
        self._final_segments: list[str] = []
        self._latest_interim: str = ""
        self._saw_interim = False
        self._saw_speech_final = False
        self._saw_utterance_end = False
        self._latest_confidence: float | None = None
        self._last_send_monotonic = time.monotonic()
        self._request_id = self._read_request_id()
        self._last_interim_callback_value = ""
        self._sender = Thread(
            target=self._sender_loop, daemon=True, name="deepgram-stt-sender"
        )
        self._sender.start()
        self._reader = Thread(
            target=self._reader_loop, daemon=True, name="deepgram-stt-reader"
        )
        self._reader.start()
        self._keepalive = Thread(
            target=self._keepalive_loop, daemon=True, name="deepgram-stt-keepalive"
        )
        self._keepalive.start()

    def send_pcm(self, pcm_bytes: bytes) -> None:
        """Queue raw PCM bytes for delivery to the Deepgram websocket."""
        if not pcm_bytes:
            return
        self._raise_if_unusable()
        payload = bytes(pcm_bytes)
        if self._outgoing_audio_chunk_bytes > 0 and len(payload) > self._outgoing_audio_chunk_bytes:
            for start in range(0, len(payload), self._outgoing_audio_chunk_bytes):
                self._enqueue(payload[start : start + self._outgoing_audio_chunk_bytes])
            return
        self._enqueue(payload)

    def finalize(self) -> StreamingTranscriptionResult:
        """Request finalization and return the last stable transcription snapshot."""
        self._raise_if_unusable(allow_closed=False)
        self._finalize_requested.set()

        finalize_ack = Event()
        if self._sender.is_alive() and self._finalize_control_message:
            self._enqueue(
                self._finalize_control_message,
                ack=finalize_ack,
                allow_error=True,
            )
            finalize_ack.wait(timeout=self._send_timeout_s)

        self._done.wait(timeout=self._finalize_timeout_s)
        transcript = self._current_transcript()
        error = self._get_stream_error()
        try:
            if error is not None and not transcript:
                raise error
            return self._result_snapshot(transcript=transcript)
        finally:
            self.close()

    def snapshot(self) -> StreamingTranscriptionResult:
        """Return the current best-effort streaming transcription state."""
        return self._result_snapshot(transcript=self._current_transcript())

    def close(self) -> None:
        """Shut down the websocket session and join background workers briefly."""
        if self._closed.is_set():
            return

        self._closed.set()
        close_ack = Event()
        if self._sender.is_alive() and self._close_control_message:
            try:
                self._enqueue(
                    self._close_control_message,
                    ack=close_ack,
                    allow_error=True,
                    allow_closed=True,
                )
                close_ack.wait(timeout=self._send_timeout_s)
            except Exception:
                logger.warning(
                    "Deepgram CloseStream enqueue failed during session shutdown.",
                    exc_info=True,
                )

        try:
            self._connection.close()
        except Exception:
            logger.warning(
                "Deepgram websocket close failed during session shutdown.",
                exc_info=True,
            )

        self._done.set()

        if self._keepalive.is_alive():
            self._keepalive.join(timeout=1.0)
        if self._sender.is_alive():
            self._sender.join(timeout=1.0)
        if self._reader.is_alive():
            self._reader.join(timeout=1.0)

    def _read_request_id(self) -> str | None:
        """Read the request identifier exposed by the websocket handshake."""
        response = getattr(self._connection, "response", None)
        headers = getattr(response, "headers", None)
        return _extract_request_id_from_headers(headers)

    def _reader_loop(self) -> None:
        """Consume Deepgram events and fold them into the session state."""
        try:
            for message in self._connection:
                if isinstance(message, bytes):
                    continue
                payload = json.loads(message)
                if not isinstance(payload, dict):
                    continue

                payload_request_id = _extract_request_id_from_payload(payload)
                if payload_request_id:
                    with self._state_lock:
                        if self._request_id is None:
                            self._request_id = payload_request_id

                event_type = str(payload.get("type", "")).strip()

                if self._protocol == "flux":
                    self._handle_flux_message(event_type, payload)
                else:
                    self._handle_nova_message(event_type, payload)

        except Exception as exc:  # pragma: no cover - network close races
            if not self._closed.is_set():
                self._set_stream_error(exc)
        finally:
            self._done.set()

    def _handle_nova_message(self, event_type: str, payload: dict[str, object]) -> None:
        """Fold a Nova streaming event into session state."""
        if event_type == "Results":
            transcript = _extract_streaming_transcript(payload)
            confidence = _extract_streaming_confidence(payload)
            is_final = bool(payload.get("is_final"))
            speech_final = bool(payload.get("speech_final"))
            from_finalize = bool(payload.get("from_finalize"))

            with self._state_lock:
                if transcript:
                    if is_final:
                        if not self._final_segments or self._final_segments[-1] != transcript:
                            self._final_segments.append(transcript)
                        self._latest_interim = transcript
                    else:
                        self._latest_interim = transcript
                        self._saw_interim = True
                if confidence is not None:
                    self._latest_confidence = confidence
                if speech_final:
                    self._saw_speech_final = True

            if transcript and not is_final:
                self._emit_interim(transcript)

            if speech_final and transcript and not from_finalize:
                self._safe_invoke_callback(
                    self._on_endpoint,
                    StreamingSpeechEndpointEvent(
                        transcript=transcript,
                        event_type="speech_final",
                        request_id=self._request_id,
                        is_final=is_final,
                        speech_final=speech_final,
                        from_finalize=from_finalize,
                    ),
                )

            if is_final:
                self._done.set()
            return

        if event_type == "UtteranceEnd":
            with self._state_lock:
                self._saw_utterance_end = True
            transcript = self._current_transcript()
            if transcript:
                self._safe_invoke_callback(
                    self._on_endpoint,
                    StreamingSpeechEndpointEvent(
                        transcript=transcript,
                        event_type="utterance_end",
                        request_id=self._request_id,
                    ),
                )
            self._done.set()
            return

        if event_type == "Metadata":
            return

        if event_type in {"CloseStream", "SpeechStarted"}:
            return

        if event_type == "Error":
            message_text = str(
                payload.get("description")
                or payload.get("message")
                or "Deepgram streaming error"
            ).strip()
            self._set_stream_error(RuntimeError(message_text))
            return

    def _handle_flux_message(self, event_type: str, payload: dict[str, object]) -> None:
        """Fold a Flux streaming event into session state."""
        if event_type == "Connected":
            return

        if event_type == "TurnInfo":
            turn_event = str(payload.get("event", "")).strip()
            transcript = str(payload.get("transcript") or "").strip()
            confidence = _extract_flux_confidence(payload)

            with self._state_lock:
                if transcript:
                    if turn_event == "EndOfTurn":
                        if not self._final_segments or self._final_segments[-1] != transcript:
                            self._final_segments.append(transcript)
                        self._latest_interim = transcript
                    else:
                        self._latest_interim = transcript
                        self._saw_interim = True
                if confidence is not None:
                    self._latest_confidence = confidence
                if turn_event == "EndOfTurn":
                    self._saw_speech_final = True

            if turn_event == "Update" and transcript:
                self._emit_interim(transcript)
                return

            if turn_event == "EagerEndOfTurn":
                if transcript:
                    self._safe_invoke_callback(
                        self._on_endpoint,
                        StreamingSpeechEndpointEvent(
                            transcript=transcript,
                            event_type="eager_end_of_turn",
                            request_id=self._request_id,
                            is_final=False,
                            speech_final=False,
                            from_finalize=False,
                        ),
                    )
                return

            if turn_event == "TurnResumed":
                if transcript:
                    self._emit_interim(transcript)
                self._safe_invoke_callback(
                    self._on_endpoint,
                    StreamingSpeechEndpointEvent(
                        transcript=transcript,
                        event_type="turn_resumed",
                        request_id=self._request_id,
                        is_final=False,
                        speech_final=False,
                        from_finalize=False,
                    ),
                )
                return

            if turn_event == "EndOfTurn":
                if transcript:
                    self._safe_invoke_callback(
                        self._on_endpoint,
                        StreamingSpeechEndpointEvent(
                            transcript=transcript,
                            event_type="end_of_turn",
                            request_id=self._request_id,
                            is_final=True,
                            speech_final=True,
                            from_finalize=False,
                        ),
                    )
                self._done.set()
                return

            return

        if event_type == "ConfigureSuccess":
            return

        if event_type == "ConfigureFailure":
            description = str(payload.get("description") or "Flux configure rejected").strip()
            logger.warning("Deepgram Flux ConfigureFailure: %s", description)
            return

        if event_type in {"CloseStream"}:
            return

        if event_type == "Error":
            message_text = str(
                payload.get("description")
                or payload.get("message")
                or "Deepgram Flux streaming error"
            ).strip()
            self._set_stream_error(RuntimeError(message_text))
            return

    def _result_snapshot(self, *, transcript: str) -> StreamingTranscriptionResult:
        """Build the contract object exposed to Twinr runtime callers."""
        with self._state_lock:
            request_id = self._request_id
            saw_interim = self._saw_interim
            saw_speech_final = self._saw_speech_final
            saw_utterance_end = self._saw_utterance_end
            confidence = self._latest_confidence
        return StreamingTranscriptionResult(
            transcript=transcript,
            request_id=request_id,
            saw_interim=saw_interim,
            saw_speech_final=saw_speech_final,
            saw_utterance_end=saw_utterance_end,
            confidence=confidence,
        )

    def _sender_loop(self) -> None:
        """Drain queued audio and control frames to the websocket connection."""
        while True:
            try:
                payload, ack = self._outgoing.get(timeout=0.5)
            except Empty:
                if self._closed.is_set():
                    return
                continue

            try:
                if self._get_stream_error() is not None and payload not in {"CloseStream", "Finalize"}:
                    continue

                with self._send_lock:
                    if isinstance(payload, bytes):
                        self._connection.send(payload)
                    else:
                        self._connection.send(json.dumps({"type": payload}))

                with self._state_lock:
                    self._last_send_monotonic = time.monotonic()

                if isinstance(payload, str) and payload == "CloseStream":
                    return

            except Exception as exc:  # pragma: no cover - network close races
                if not self._closed.is_set():
                    self._set_stream_error(exc)
                return

            finally:
                if ack is not None:
                    ack.set()
                self._outgoing.task_done()

    def _keepalive_loop(self) -> None:
        """Send Deepgram keepalives during long idle periods."""
        if self._keepalive_interval_s <= 0 or not self._keepalive_control_message:
            return

        while not self._closed.wait(timeout=self._keepalive_interval_s):
            if self._finalize_requested.is_set() or self._get_stream_error() is not None:
                continue
            if not self._sender.is_alive():
                return

            with self._state_lock:
                idle_for_s = time.monotonic() - self._last_send_monotonic

            if idle_for_s < self._keepalive_interval_s:
                continue

            try:
                self._enqueue(self._keepalive_control_message, allow_error=True)
            except Exception:
                logger.warning(
                    "Deepgram keepalive enqueue failed; continuing without keepalive",
                    exc_info=True,
                )
                return

    def _emit_interim(self, transcript: str) -> None:
        """Emit an interim callback only when the text actually changed."""
        should_emit = False
        with self._state_lock:
            if transcript and transcript != self._last_interim_callback_value:
                self._last_interim_callback_value = transcript
                should_emit = True
        if should_emit:
            self._safe_invoke_callback(self._on_interim, transcript)

    def _enqueue(
        self,
        payload: bytes | str,
        *,
        ack: Event | None = None,
        allow_error: bool = False,
        allow_closed: bool = False,
    ) -> None:
        """Queue outbound payloads while enforcing close and backpressure rules."""
        if not allow_closed and self._closed.is_set():
            raise RuntimeError("Streaming session is closed")
        if not allow_error:
            self._raise_if_unusable()
        elif not self._sender.is_alive():
            raise RuntimeError("Streaming session sender thread is not running")
        try:
            if ack is None and isinstance(payload, bytes):
                self._outgoing.put_nowait((payload, ack))
            else:
                self._outgoing.put((payload, ack), timeout=self._send_timeout_s)
        except Full as exc:
            raise RuntimeError("Streaming session outbound queue is full") from exc

    def _current_transcript(self) -> str:
        """Return the best transcript assembled so far for the session."""
        with self._state_lock:
            transcript = " ".join(segment for segment in self._final_segments if segment).strip()
            if not transcript:
                transcript = self._latest_interim.strip()
            return transcript

    def _safe_invoke_callback(
        self,
        callback: Callable[[Any], None] | None,
        value: Any,
    ) -> None:
        """Invoke a caller callback without letting callback failures break STT."""
        if callback is None:
            return
        try:
            callback(value)
        except Exception:  # pragma: no cover - callback behavior belongs to caller tests
            logger.exception("Deepgram streaming callback raised and was ignored")

    def _set_stream_error(self, exc: Exception) -> None:
        """Store the first terminal stream error and wake blocked waiters."""
        with self._state_lock:
            if self._stream_error is None:
                self._stream_error = exc
        self._done.set()

    def _get_stream_error(self) -> Exception | None:
        """Return the stored terminal stream error, if one exists."""
        with self._state_lock:
            return self._stream_error

    def _raise_if_unusable(self, *, allow_closed: bool = False) -> None:
        """Raise if the session has failed or is already closed."""
        error = self._get_stream_error()
        if error is not None:
            raise error
        if not allow_closed and self._closed.is_set():
            raise RuntimeError("Streaming session is closed")
        if not self._sender.is_alive():
            raise RuntimeError("Streaming session sender thread is not running")


class DeepgramSpeechToTextProvider:
    """Transcribe files or live PCM streams with the configured Deepgram account."""

    def __init__(
        self,
        config: TwinrConfig,
        *,
        client: httpx.Client | None = None,
        websocket_connector: Callable[..., Any] | None = None,
    ) -> None:
        """Create the provider with optional HTTP and websocket test doubles."""
        self.config = config
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(float(self.config.deepgram_timeout_s)),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
        self._websocket_connector = websocket_connector or websocket_connect

    def close(self) -> None:
        """Close the provider-owned HTTP client if this instance created it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> DeepgramSpeechToTextProvider:
        """Support ``with`` blocks around provider-owned HTTP resources."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close provider-owned resources when leaving a context manager."""
        self.close()

    def _get_config(self, name: str, default: Any) -> Any:
        """Read optional config fields without requiring a TwinrConfig schema bump."""
        return getattr(self.config, name, default)

    def _allow_insecure_transport(self) -> bool:
        """Whether cleartext HTTP/WS should be allowed for Deepgram transport."""
        return _parse_bool(self._get_config("deepgram_allow_insecure_transport", False), False)

    def _mip_opt_out_enabled(self) -> bool:
        """Whether to opt out of Deepgram's model-improvement retention path."""
        # BREAKING: Privacy-first default is now True unless config explicitly disables it.
        return _parse_bool(self._get_config("deepgram_mip_opt_out", True), True)

    def _resolve_credentials(self) -> tuple[str, str]:
        """Prefer short-lived access tokens, then fall back to API keys."""
        access_token = str(
            self._get_config("deepgram_access_token", None)
            or self._get_config("deepgram_token", None)
            or os.getenv(_DEEPGRAM_ACCESS_TOKEN_ENV, "")
        ).strip()
        if access_token:
            return ("Bearer", access_token)

        api_key = str(
            self._get_config("deepgram_api_key", None)
            or os.getenv(_DEEPGRAM_API_KEY_ENV, "")
        ).strip()
        if api_key:
            return ("Token", api_key)

        raise RuntimeError(
            "DEEPGRAM_TOKEN or DEEPGRAM_API_KEY is required to use the Deepgram speech provider"
        )

    def _authorization_header_value(self) -> str:
        """Build the Authorization header for Deepgram requests."""
        scheme, credential = self._resolve_credentials()
        return f"{scheme} {credential}"

    def _resolved_streaming_model(self) -> str:
        """Return the model used for live streaming STT."""
        return _normalize_model_name(self.config.deepgram_stt_model)

    def _resolved_batch_model(self) -> str:
        """Return the model used for batch transcription requests."""
        configured_batch_model = self._get_config("deepgram_batch_stt_model", None)
        if configured_batch_model:
            return _normalize_model_name(configured_batch_model)

        streaming_model = self._resolved_streaming_model()
        if _is_flux_model(streaming_model):
            # BREAKING: Flux is streaming-only; batch now falls back to Nova-3 unless
            #           a dedicated deepgram_batch_stt_model is configured.
            return "nova-3"
        return streaming_model

    def _max_retries(self) -> int:
        """Return bounded retry count for transient HTTP / connect failures."""
        return max(0, int(self._get_config("deepgram_max_retries", 2)))

    def _retry_backoff_s(self) -> float:
        """Base backoff between transient retry attempts."""
        return max(0.0, float(self._get_config("deepgram_retry_backoff_s", 0.5)))

    def _max_retry_backoff_s(self) -> float:
        """Upper bound for retry delays."""
        return max(0.1, float(self._get_config("deepgram_retry_max_backoff_s", 4.0)))

    def _sleep_before_retry(self, *, attempt: int, response: httpx.Response | None = None) -> None:
        """Sleep using Retry-After or capped exponential backoff."""
        delay_s = None
        if response is not None:
            delay_s = _extract_transient_retry_delay_s(response)
        if delay_s is None:
            delay_s = min(
                self._max_retry_backoff_s(),
                self._retry_backoff_s() * (2 ** attempt),
            )
        if delay_s > 0.0:
            time.sleep(delay_s)

    def _run_with_retries(
        self,
        *,
        operation_name: str,
        request_fn: Callable[[], httpx.Response],
    ) -> httpx.Response:
        """Run an idempotent Deepgram request with bounded retry/backoff."""
        last_error: Exception | None = None
        for attempt in range(self._max_retries() + 1):
            try:
                response = request_fn()
                if (
                    response.status_code in _DEEPGRAM_TRANSIENT_STATUS_CODES
                    and attempt < self._max_retries()
                ):
                    logger.warning(
                        "Transient Deepgram %s response (%s); retrying attempt %s/%s",
                        operation_name,
                        response.status_code,
                        attempt + 1,
                        self._max_retries(),
                    )
                    response.close()
                    self._sleep_before_retry(attempt=attempt, response=response)
                    continue
                response.raise_for_status()
                return response

            except httpx.HTTPStatusError as exc:
                last_error = exc
                response = exc.response
                if (
                    response is not None
                    and response.status_code in _DEEPGRAM_TRANSIENT_STATUS_CODES
                    and attempt < self._max_retries()
                ):
                    logger.warning(
                        "Transient Deepgram %s error (%s); retrying attempt %s/%s",
                        operation_name,
                        response.status_code,
                        attempt + 1,
                        self._max_retries(),
                    )
                    response.close()
                    self._sleep_before_retry(attempt=attempt, response=response)
                    continue

                if response is not None:
                    details = _redacted_error_details(response)
                    response.close()
                    raise RuntimeError(
                        f"Deepgram {operation_name} failed: {details}"
                    ) from exc
                raise RuntimeError(f"Deepgram {operation_name} failed: {exc}") from exc

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt < self._max_retries():
                    logger.warning(
                        "Transient Deepgram %s transport error (%s); retrying attempt %s/%s",
                        operation_name,
                        exc.__class__.__name__,
                        attempt + 1,
                        self._max_retries(),
                    )
                    self._sleep_before_retry(attempt=attempt)
                    continue
                raise RuntimeError(
                    f"Deepgram {operation_name} failed after retries: {exc}"
                ) from exc

        if last_error is None:
            raise RuntimeError(f"Deepgram {operation_name} failed with an unknown error")
        raise RuntimeError(f"Deepgram {operation_name} failed after retries: {last_error}") from last_error

    def _common_query_params(self) -> list[tuple[str, str]]:
        """Return query params that should be attached to all Deepgram listen requests."""
        params: list[tuple[str, str]] = []
        if self._mip_opt_out_enabled():
            params.append(("mip_opt_out", "true"))
        request_tag = str(self._get_config("deepgram_request_tag", "") or "").strip()
        if request_tag:
            params.append(("tag", request_tag))
        return params

    def _batch_query_params(
        self,
        *,
        model: str,
        language: str | None,
        prompt: str | None,
    ) -> list[tuple[str, str]]:
        """Build the query parameter list for batch transcription."""
        params: list[tuple[str, str]] = [("model", model)]
        resolved_language = (language or self.config.deepgram_stt_language or "").strip()
        if resolved_language and not _is_flux_model(model):
            params.append(("language", resolved_language))
        if self.config.deepgram_stt_smart_format:
            params.append(("smart_format", "true"))
        params.extend(_deepgram_prompt_params(model=model, prompt=prompt))
        params.extend(self._common_query_params())
        return params

    def _streaming_query_params(
        self,
        *,
        model: str,
        language: str | None,
        prompt: str | None,
        sample_rate: int,
        channels: int,
        on_interim: Callable[[str], None] | None,
    ) -> tuple[str, list[tuple[str, str]], bool]:
        """Build the query parameter list for a streaming websocket session."""
        if _is_flux_model(model):
            params: list[tuple[str, str]] = [
                ("model", model),
                ("encoding", "linear16"),
                ("sample_rate", str(sample_rate)),
                ("channels", str(channels)),
            ]
            params.extend(_deepgram_prompt_params(model=model, prompt=prompt))

            eot_threshold = self._get_config("deepgram_flux_eot_threshold", None)
            if eot_threshold is not None:
                threshold = _require_non_negative_float("deepgram_flux_eot_threshold", eot_threshold)
                if not 0.5 <= threshold <= 0.9:
                    raise ValueError("deepgram_flux_eot_threshold must be between 0.5 and 0.9")
                params.append(("eot_threshold", f"{threshold:g}"))

            eager_eot_threshold = self._get_config("deepgram_flux_eager_eot_threshold", None)
            if eager_eot_threshold is not None:
                threshold = _require_non_negative_float(
                    "deepgram_flux_eager_eot_threshold",
                    eager_eot_threshold,
                )
                if not 0.3 <= threshold <= 0.9:
                    raise ValueError(
                        "deepgram_flux_eager_eot_threshold must be between 0.3 and 0.9"
                    )
                params.append(("eager_eot_threshold", f"{threshold:g}"))

            eot_timeout_ms = self._get_config("deepgram_flux_eot_timeout_ms", None)
            if eot_timeout_ms is not None:
                timeout_ms = int(eot_timeout_ms)
                if not 500 <= timeout_ms <= 10000:
                    raise ValueError("deepgram_flux_eot_timeout_ms must be between 500 and 10000")
                params.append(("eot_timeout_ms", str(timeout_ms)))

            if language:
                logger.warning(
                    "Ignoring Deepgram language=%r for Flux; language is encoded in the model name.",
                    language,
                )

            params.extend(self._common_query_params())
            return ("flux", params, False)

        interim_results_enabled = bool(
            self.config.deepgram_streaming_interim_results
            or on_interim is not None
            or int(self.config.deepgram_streaming_utterance_end_ms) > 0
        )
        vad_events_enabled = _parse_bool(
            self._get_config("deepgram_streaming_vad_events", True),
            True,
        )

        params = [
            ("model", model),
            ("encoding", "linear16"),
            ("sample_rate", str(sample_rate)),
            ("channels", str(channels)),
        ]
        resolved_language = (language or self.config.deepgram_stt_language or "").strip()
        if resolved_language:
            params.append(("language", resolved_language))
        if self.config.deepgram_stt_smart_format:
            params.append(("smart_format", "true"))
        if interim_results_enabled:
            params.append(("interim_results", "true"))
        if vad_events_enabled:
            params.append(("vad_events", "true"))
        if self.config.deepgram_streaming_endpointing_ms > 0:
            params.append(("endpointing", str(self.config.deepgram_streaming_endpointing_ms)))
        if self.config.deepgram_streaming_utterance_end_ms > 0:
            params.append(("utterance_end_ms", str(self.config.deepgram_streaming_utterance_end_ms)))
        params.extend(_deepgram_prompt_params(model=model, prompt=prompt))
        params.extend(self._common_query_params())
        return ("nova", params, True)

    def _batch_listen_url(self, *, params: Iterable[tuple[str, str]]) -> str:
        """Build the hosted or self-hosted URL for batch `/v1/listen` requests."""
        return _build_listen_url(
            base_url=self.config.deepgram_base_url,
            version="v1",
            websocket=False,
            params=params,
            allow_insecure=self._allow_insecure_transport(),
        )

    def _streaming_listen_url(self, *, protocol: str, params: Iterable[tuple[str, str]]) -> str:
        """Build the hosted or self-hosted websocket URL for streaming listen requests."""
        version = "v2" if protocol == "flux" else "v1"
        return _build_listen_url(
            base_url=self.config.deepgram_base_url,
            version=version,
            websocket=True,
            params=params,
            allow_insecure=self._allow_insecure_transport(),
        )

    def _transcribe_bytes_request(
        self,
        *,
        audio_bytes: bytes,
        content_type: str,
        params: list[tuple[str, str]],
    ) -> dict[str, object]:
        """POST in-memory audio bytes to Deepgram with retries and structured errors."""
        max_request_bytes = max(
            64 * 1024,
            int(self._get_config("deepgram_transcribe_max_bytes", 32 * 1024 * 1024)),
        )
        if len(audio_bytes) > max_request_bytes:
            raise RuntimeError(
                f"Audio payload too large for Deepgram batch transcription: {len(audio_bytes)} > {max_request_bytes} bytes"
            )

        def request_fn() -> httpx.Response:
            return self._client.post(
                self._batch_listen_url(params=params),
                headers={
                    "Authorization": self._authorization_header_value(),
                    "Content-Type": content_type or "application/octet-stream",
                },
                content=audio_bytes,
            )

        response = self._run_with_retries(
            operation_name="transcription request",
            request_fn=request_fn,
        )
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Deepgram transcription response was not valid JSON: {_redacted_error_details(response)}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Deepgram transcription response JSON was not an object")
        return payload

    def _validate_audio_path(self, path: Path) -> tuple[object, int, str, bytes]:
        """Open and validate a local path before sending bytes off-device."""
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

        open_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            open_flags |= os.O_NOFOLLOW

        try:
            fd = os.open(path, open_flags)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Audio path does not exist: {path}") from exc
        except OSError as exc:
            if path.is_symlink():
                raise RuntimeError(f"Refusing to transcribe symlinked path: {path}") from exc
            raise RuntimeError(f"Unable to open audio path safely: {path}") from exc

        file_obj = os.fdopen(fd, "rb")
        try:
            file_stat = os.fstat(file_obj.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise RuntimeError(f"Audio path is not a regular file: {path}")

            max_request_bytes = max(
                64 * 1024,
                int(self._get_config("deepgram_transcribe_max_bytes", 32 * 1024 * 1024)),
            )
            if file_stat.st_size > max_request_bytes:
                raise RuntimeError(
                    f"Audio path exceeds configured transcription size limit: {file_stat.st_size} > {max_request_bytes} bytes"
                )

            header = file_obj.read(64)
            file_obj.seek(0)

            allow_non_audio = _parse_bool(
                self._get_config("deepgram_allow_non_audio_files", False),
                False,
            )
            if not allow_non_audio and not _is_probable_audio_path(
                path=path,
                content_type=content_type,
                header=header,
            ):
                raise RuntimeError(
                    f"Refusing to upload a non-audio-looking file for transcription: {path}"
                )

            return file_obj, file_stat.st_size, content_type, header
        except Exception:
            file_obj.close()
            raise

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """Transcribe an in-memory audio payload with Deepgram's REST API."""
        del filename
        if not audio_bytes:
            return ""

        model = self._resolved_batch_model()
        payload = self._transcribe_bytes_request(
            audio_bytes=audio_bytes,
            content_type=content_type,
            params=self._batch_query_params(
                model=model,
                language=language,
                prompt=prompt,
            ),
        )
        return _extract_transcript(payload)

    def transcribe_path(
        self,
        path: str | Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """Safely read a regular audio file from disk and transcribe it."""
        audio_path = Path(path)
        model = self._resolved_batch_model()
        file_chunk_bytes = max(
            8 * 1024,
            int(self._get_config("deepgram_transcribe_file_chunk_bytes", 128 * 1024)),
        )
        params = self._batch_query_params(model=model, language=language, prompt=prompt)

        audio_file, _, content_type, _ = self._validate_audio_path(audio_path)
        try:
            def request_fn() -> httpx.Response:
                audio_file.seek(0)
                return self._client.post(
                    self._batch_listen_url(params=params),
                    headers={
                        "Authorization": self._authorization_header_value(),
                        "Content-Type": content_type,
                    },
                    content=_iter_file_chunks(audio_file, chunk_size=file_chunk_bytes),
                )

            response = self._run_with_retries(
                operation_name="path transcription request",
                request_fn=request_fn,
            )
            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Deepgram transcription response was not valid JSON: {_redacted_error_details(response)}"
                ) from exc
            if not isinstance(payload, dict):
                raise RuntimeError("Deepgram transcription response JSON was not an object")
            return _extract_transcript(payload)
        finally:
            audio_file.close()

    def start_streaming_session(
        self,
        *,
        sample_rate: int,
        channels: int,
        language: str | None = None,
        prompt: str | None = None,
        on_interim: Callable[[str], None] | None = None,
        on_endpoint: Callable[[StreamingSpeechEndpointEvent], None] | None = None,
    ) -> StreamingSpeechToTextSession:
        """Open a bounded Deepgram websocket session for live PCM input."""
        sample_rate = _require_positive_int("sample_rate", sample_rate)
        channels = _require_positive_int("channels", channels)

        model = self._resolved_streaming_model()
        if _is_flux_model(model):
            _require_flux_sample_rate(sample_rate)

        protocol, params, supports_keepalive = self._streaming_query_params(
            model=model,
            language=language,
            prompt=prompt,
            sample_rate=sample_rate,
            channels=channels,
            on_interim=on_interim,
        )

        max_message_bytes = max(
            64 * 1024,
            int(self._get_config("deepgram_streaming_max_message_bytes", 4 * 1024 * 1024)),
        )
        keepalive_interval_s = max(
            0.0,
            float(self._get_config("deepgram_streaming_keepalive_interval_s", 4.0)),
        )
        send_timeout_s = max(
            0.5,
            float(
                self._get_config(
                    "deepgram_streaming_send_timeout_s",
                    self.config.deepgram_timeout_s,
                )
            ),
        )
        max_pending_messages = max(
            8,
            int(self._get_config("deepgram_streaming_max_pending_messages", 256)),
        )

        default_chunk_ms = 80.0 if protocol == "flux" else 100.0
        chunk_ms = max(
            0.0,
            float(self._get_config("deepgram_streaming_audio_chunk_ms", default_chunk_ms)),
        )
        outgoing_audio_chunk_bytes = _compute_pcm_chunk_bytes(
            sample_rate=sample_rate,
            channels=channels,
            chunk_ms=chunk_ms,
        )

        websocket_url = self._streaming_listen_url(protocol=protocol, params=params)

        def connect_once():
            return self._websocket_connector(
                websocket_url,
                additional_headers={
                    "Authorization": self._authorization_header_value(),
                },
                open_timeout=self.config.deepgram_timeout_s,
                close_timeout=self.config.deepgram_timeout_s,
                max_size=max_message_bytes,
            )

        last_error: Exception | None = None
        connection = None
        for attempt in range(self._max_retries() + 1):
            try:
                connection = connect_once()
                break
            except Exception as exc:
                last_error = exc
                if attempt >= self._max_retries():
                    break
                logger.warning(
                    "Transient Deepgram websocket connect failure (%s); retrying attempt %s/%s",
                    exc.__class__.__name__,
                    attempt + 1,
                    self._max_retries(),
                )
                self._sleep_before_retry(attempt=attempt)
        if connection is None:
            raise RuntimeError(f"Deepgram websocket connect failed: {last_error}") from last_error

        return _DeepgramStreamingSession(
            connection=connection,
            protocol=protocol,
            finalize_timeout_s=self.config.deepgram_streaming_finalize_timeout_s,
            keepalive_interval_s=keepalive_interval_s if supports_keepalive else 0.0,
            send_timeout_s=send_timeout_s,
            max_pending_messages=max_pending_messages,
            outgoing_audio_chunk_bytes=outgoing_audio_chunk_bytes,
            finalize_control_message="CloseStream" if protocol == "flux" else "Finalize",
            close_control_message="CloseStream",
            keepalive_control_message="KeepAlive" if supports_keepalive else None,
            on_interim=on_interim,
            on_endpoint=on_endpoint,
        )
