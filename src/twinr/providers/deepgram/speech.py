"""Provide Deepgram-backed speech-to-text adapters for Twinr.

The module exposes a synchronous batch transcription helper and a bounded
streaming session wrapper around Deepgram's websocket API. Runtime workflows
use this provider through the higher-level contracts in
``twinr.agent.base_agent.contracts``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit
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


def _extract_transcript(payload: dict[str, object]) -> str:
    """Extract the best transcript string from a batch Deepgram response."""
    # AUDIT-FIX(#5): Validate Deepgram payload structure defensively so malformed 2xx responses do not explode with AttributeError/IndexError.
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


def _extract_streaming_transcript(payload: dict[str, object]) -> str:
    """Extract the latest transcript fragment from a streaming event payload."""
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
    """Extract a confidence estimate from a streaming event payload."""
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


def _build_websocket_url(
    *,
    base_url: str,
    model: str,
    language: str | None,
    smart_format: bool,
    sample_rate: int,
    channels: int,
    interim_results: bool,
    endpointing_ms: int,
    utterance_end_ms: int,
) -> str:
    """Build the Deepgram websocket URL for a streaming transcription session."""
    # AUDIT-FIX(#1): Preserve secure ws/wss schemes and reject invalid base URLs instead of silently downgrading TLS or producing malformed URLs.
    normalized_base_url = base_url.strip()
    if not normalized_base_url:
        raise ValueError("Deepgram base URL must not be empty")

    split = urlsplit(normalized_base_url.rstrip("/"))
    if split.scheme not in {"http", "https", "ws", "wss"} or not split.netloc:
        raise ValueError(f"Invalid Deepgram base URL: {base_url!r}")

    scheme_map = {
        "http": "ws",
        "https": "wss",
        "ws": "ws",
        "wss": "wss",
    }
    scheme = scheme_map[split.scheme]
    params: dict[str, str] = {
        "model": model,
        "encoding": "linear16",
        "sample_rate": str(sample_rate),
        "channels": str(channels),
    }
    if language:
        params["language"] = language
    if smart_format:
        params["smart_format"] = "true"
    if interim_results:
        params["interim_results"] = "true"
    if endpointing_ms > 0:
        params["endpointing"] = str(endpointing_ms)
    if utterance_end_ms > 0:
        params["utterance_end_ms"] = str(utterance_end_ms)
    path = split.path.rstrip("/") + "/listen"
    return urlunsplit((scheme, split.netloc, path, urlencode(params), ""))


def _require_positive_int(name: str, value: int) -> int:
    """Return a validated positive integer for audio transport settings."""
    # AUDIT-FIX(#9): Fail fast on invalid audio transport settings locally instead of shipping broken parameters to Deepgram.
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class _DeepgramStreamingSession(StreamingSpeechToTextSession):
    """Manage a bounded Deepgram websocket session for live transcription."""

    def __init__(
        self,
        *,
        connection,
        finalize_timeout_s: float,
        keepalive_interval_s: float,
        send_timeout_s: float,
        max_pending_messages: int,
        on_interim: Callable[[str], None] | None = None,
        on_endpoint: Callable[[StreamingSpeechEndpointEvent], None] | None = None,
    ) -> None:
        """Start sender, reader, and keepalive workers for one websocket stream."""
        self._connection = connection
        self._finalize_timeout_s = max(0.5, float(finalize_timeout_s))
        self._keepalive_interval_s = max(0.0, float(keepalive_interval_s))
        self._send_timeout_s = max(0.5, float(send_timeout_s))
        self._on_interim = on_interim
        self._on_endpoint = on_endpoint
        self._state_lock = Lock()
        self._send_lock = Lock()
        # AUDIT-FIX(#2): Carry optional per-message acknowledgements so finalize/close no longer depend on unbounded Queue.join() waits.
        # AUDIT-FIX(#6): Bound the outbound queue to prevent memory blowups if the network stalls while audio continues arriving.
        self._outgoing: Queue[tuple[bytes | str, Event | None]] = Queue(maxsize=max(1, int(max_pending_messages)))
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
        self._sender = Thread(target=self._sender_loop, daemon=True, name="deepgram-stt-sender")
        self._sender.start()
        self._reader = Thread(target=self._reader_loop, daemon=True, name="deepgram-stt-reader")
        self._reader.start()
        # AUDIT-FIX(#4): Keep the Deepgram stream alive across long senior pauses to avoid NET-0001 disconnects during silence.
        self._keepalive = Thread(target=self._keepalive_loop, daemon=True, name="deepgram-stt-keepalive")
        self._keepalive.start()

    def send_pcm(self, pcm_bytes: bytes) -> None:
        """Queue raw PCM bytes for delivery to the Deepgram websocket."""
        if not pcm_bytes:
            return
        # AUDIT-FIX(#6): Reject writes after shutdown or sender failure instead of silently queueing audio that can never be delivered.
        self._raise_if_unusable()
        self._enqueue(bytes(pcm_bytes))

    def finalize(self) -> StreamingTranscriptionResult:
        """Request finalization and return the last stable transcription snapshot."""
        self._raise_if_unusable(allow_closed=False)
        self._finalize_requested.set()

        # AUDIT-FIX(#2): Bound control-message delivery waits so finalize cannot deadlock the voice turn forever if the sender thread dies.
        finalize_ack = Event()
        if self._sender.is_alive():
            self._enqueue("Finalize", ack=finalize_ack, allow_error=True)
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
        # AUDIT-FIX(#2): Attempt a graceful Deepgram CloseStream, but never block indefinitely on a dead sender or stuck queue.
        close_ack = Event()
        if self._sender.is_alive():
            try:
                self._enqueue("CloseStream", ack=close_ack, allow_error=True, allow_closed=True)
                close_ack.wait(timeout=self._send_timeout_s)
            except Exception:
                logger.warning("Deepgram CloseStream enqueue failed during session shutdown.", exc_info=True)

        try:
            self._connection.close()
        except Exception:
            logger.warning("Deepgram websocket close failed during session shutdown.", exc_info=True)

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
        if headers is None:
            return None
        for key in ("dg-request-id", "DG-Request-Id", "x-request-id"):
            value = headers.get(key)
            if value:
                return str(value)
        return None

    def _reader_loop(self) -> None:
        """Consume Deepgram events and fold them into the session state."""
        try:
            for message in self._connection:
                if isinstance(message, bytes):
                    continue
                payload = json.loads(message)
                event_type = str(payload.get("type", "")).strip()

                if event_type == "Results":
                    transcript = _extract_streaming_transcript(payload)
                    confidence = _extract_streaming_confidence(payload)
                    is_final = bool(payload.get("is_final"))
                    speech_final = bool(payload.get("speech_final"))
                    from_finalize = bool(payload.get("from_finalize"))
                    metadata = payload.get("metadata", {})

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

                        if isinstance(metadata, dict) and self._request_id is None:
                            request_id = metadata.get("request_id")
                            if isinstance(request_id, str) and request_id.strip():
                                self._request_id = request_id.strip()

                        if speech_final:
                            self._saw_speech_final = True

                    if transcript and not is_final:
                        # AUDIT-FIX(#3): Isolate callback failures so UI/observer bugs do not tear down the transcription transport.
                        self._safe_invoke_callback(self._on_interim, transcript)

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

                    # AUDIT-FIX(#11): A bare speech_final callback is only an
                    # endpoint hint for the turn controller, not a stable final
                    # transcript. finalize() must keep waiting until Deepgram
                    # delivers an actual final segment, otherwise Twinr can
                    # return and close on a truncated fragment.
                    if is_final:
                        self._done.set()

                elif event_type == "UtteranceEnd":
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

                elif event_type == "Metadata":
                    # AUDIT-FIX(#7): Capture request_id from Metadata frames because Deepgram may only emit it there after close/finalize.
                    request_id = payload.get("request_id")
                    if isinstance(request_id, str) and request_id.strip():
                        with self._state_lock:
                            if self._request_id is None:
                                self._request_id = request_id.strip()
                    continue

                elif event_type == "CloseStream":
                    continue

                elif event_type == "SpeechStarted":
                    continue

                elif event_type == "Error":
                    message_text = str(payload.get("description") or payload.get("message") or "Deepgram streaming error").strip()
                    self._set_stream_error(RuntimeError(message_text))
                    return

        except Exception as exc:  # pragma: no cover - network close races
            if not self._closed.is_set():
                self._set_stream_error(exc)
        finally:
            self._done.set()

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
        try:
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

                    if payload == "CloseStream":
                        return

                except Exception as exc:  # pragma: no cover - network close races
                    if not self._closed.is_set():
                        self._set_stream_error(exc)
                    return

                finally:
                    if ack is not None:
                        ack.set()
                    self._outgoing.task_done()
        finally:
            self._done.set()

    def _keepalive_loop(self) -> None:
        """Send Deepgram keepalives during long idle periods."""
        if self._keepalive_interval_s <= 0:
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
                self._enqueue("KeepAlive", allow_error=True)
            except Exception:
                logger.warning("Deepgram keepalive enqueue failed; continuing without keepalive", exc_info=True)
                return

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

    def _safe_invoke_callback(self, callback: Callable[[Any], None] | None, value: Any) -> None:
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
        # AUDIT-FIX(#8): Track ownership and expose close() so provider-created HTTP clients can release pooled sockets cleanly at shutdown.
        self._client = client or httpx.Client(timeout=self.config.deepgram_timeout_s)
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

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """Transcribe an in-memory audio payload with Deepgram's REST API.

        Args:
            audio_bytes: Raw audio bytes to send to Deepgram.
            filename: Unused compatibility parameter kept for provider parity.
            content_type: MIME type describing ``audio_bytes``.
            language: Optional override for the configured transcription language.
            prompt: Unused compatibility parameter kept for provider parity.

        Returns:
            The best transcript string returned by Deepgram, or an empty string
            when the provider produced no transcript text.

        Raises:
            RuntimeError: If configuration is incomplete or the request fails.
        """
        del filename, prompt
        if not audio_bytes:
            return ""

        api_key = (self.config.deepgram_api_key or "").strip()
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is required to use the Deepgram speech provider")

        params: dict[str, str] = {
            "model": self.config.deepgram_stt_model,
        }
        resolved_language = (language or self.config.deepgram_stt_language or "").strip()
        if resolved_language:
            params["language"] = resolved_language
        if self.config.deepgram_stt_smart_format:
            params["smart_format"] = "true"

        try:
            response = self._client.post(
                f"{self.config.deepgram_base_url.rstrip('/')}/listen",
                params=params,
                headers={
                    "Authorization": f"Token {api_key}",
                    "Content-Type": content_type or "application/octet-stream",
                },
                content=audio_bytes,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            # AUDIT-FIX(#11): Re-raise transport and decode failures as controlled runtime errors for cleaner upstream recovery paths.
            raise RuntimeError(f"Deepgram transcription request failed: {exc}") from exc

        return _extract_transcript(payload)

    def transcribe_path(
        self,
        path: str | Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """Safely read a regular audio file from disk and transcribe it.

        Args:
            path: Filesystem path to a regular audio file.
            language: Optional override for the configured transcription language.
            prompt: Unused compatibility parameter kept for provider parity.

        Returns:
            The transcript produced by :meth:`transcribe`.

        Raises:
            RuntimeError: If the file cannot be opened safely or transcription fails.
        """
        audio_path = Path(path)
        content_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"

        open_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            # AUDIT-FIX(#12): Refuse symlink traversal on platforms that support O_NOFOLLOW.
            open_flags |= os.O_NOFOLLOW

        try:
            fd = os.open(audio_path, open_flags)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Audio path does not exist: {audio_path}") from exc
        except OSError as exc:
            if audio_path.is_symlink():
                raise RuntimeError(f"Refusing to transcribe symlinked path: {audio_path}") from exc
            raise RuntimeError(f"Unable to open audio path safely: {audio_path}") from exc

        try:
            file_stat = os.fstat(fd)
            # AUDIT-FIX(#12): Only regular files are accepted; devices, pipes, and directories are rejected before any bytes leave the device.
            if not stat.S_ISREG(file_stat.st_mode):
                raise RuntimeError(f"Audio path is not a regular file: {audio_path}")

            # AUDIT-FIX(#13): Use an explicit file descriptor wrapper so the file handle is always closed deterministically.
            with os.fdopen(fd, "rb") as audio_file:
                fd = -1
                audio_bytes = audio_file.read()
        finally:
            if fd >= 0:
                os.close(fd)

        return self.transcribe(
            audio_bytes,
            filename=audio_path.name,
            content_type=content_type,
            language=language,
            prompt=prompt,
        )

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
        """Open a bounded Deepgram websocket session for live PCM input.

        Args:
            sample_rate: PCM sample rate in Hz.
            channels: Number of PCM channels sent through the stream.
            language: Optional override for the configured transcription language.
            prompt: Unused compatibility parameter kept for provider parity.
            on_interim: Optional callback for interim transcript fragments.
            on_endpoint: Optional callback for endpoint notifications.

        Returns:
            A live :class:`StreamingSpeechToTextSession` ready to receive PCM bytes.

        Raises:
            RuntimeError: If configuration is incomplete or the websocket cannot open.
            ValueError: If ``sample_rate`` or ``channels`` are not positive integers.
        """
        del prompt
        api_key = (self.config.deepgram_api_key or "").strip()
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is required to use the Deepgram speech provider")

        sample_rate = _require_positive_int("sample_rate", sample_rate)
        channels = _require_positive_int("channels", channels)
        resolved_language = (language or self.config.deepgram_stt_language or "").strip() or None

        # AUDIT-FIX(#14): Bound incoming message size instead of disabling limits completely; STT results are tiny, unbounded frames are unnecessary risk.
        max_message_bytes = max(
            64 * 1024,
            int(getattr(self.config, "deepgram_streaming_max_message_bytes", 4 * 1024 * 1024)),
        )
        keepalive_interval_s = max(
            0.0,
            float(getattr(self.config, "deepgram_streaming_keepalive_interval_s", 4.0)),
        )
        send_timeout_s = max(
            0.5,
            float(getattr(self.config, "deepgram_streaming_send_timeout_s", self.config.deepgram_timeout_s)),
        )

        max_pending_messages = max(
            8,
            int(getattr(self.config, "deepgram_streaming_max_pending_messages", 256)),
        )

        connection = self._websocket_connector(
            _build_websocket_url(
                base_url=self.config.deepgram_base_url,
                model=self.config.deepgram_stt_model,
                language=resolved_language,
                smart_format=self.config.deepgram_stt_smart_format,
                sample_rate=sample_rate,
                channels=channels,
                interim_results=self.config.deepgram_streaming_interim_results,
                endpointing_ms=self.config.deepgram_streaming_endpointing_ms,
                utterance_end_ms=self.config.deepgram_streaming_utterance_end_ms,
            ),
            additional_headers={
                "Authorization": f"Token {api_key}",
            },
            open_timeout=self.config.deepgram_timeout_s,
            close_timeout=self.config.deepgram_timeout_s,
            max_size=max_message_bytes,
        )
        return _DeepgramStreamingSession(
            connection=connection,
            finalize_timeout_s=self.config.deepgram_streaming_finalize_timeout_s,
            keepalive_interval_s=keepalive_interval_s,
            send_timeout_s=send_timeout_s,
            max_pending_messages=max_pending_messages,
            on_interim=on_interim,
            on_endpoint=on_endpoint,
        )
