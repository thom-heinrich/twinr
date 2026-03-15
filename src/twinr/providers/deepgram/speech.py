from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit
import json
import mimetypes

import httpx
from websockets.sync.client import connect as websocket_connect

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import (
    StreamingSpeechEndpointEvent,
    StreamingSpeechToTextSession,
    StreamingTranscriptionResult,
)


def _extract_transcript(payload: dict[str, object]) -> str:
    transcript = (
        payload.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [{}])[0]
        .get("transcript", "")
    )
    if not isinstance(transcript, str):
        raise RuntimeError("Deepgram response did not contain a string transcript")
    return transcript.strip()


def _extract_streaming_transcript(payload: dict[str, object]) -> str:
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
    split = urlsplit(base_url.rstrip("/"))
    scheme = "wss" if split.scheme == "https" else "ws"
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


class _DeepgramStreamingSession(StreamingSpeechToTextSession):
    def __init__(
        self,
        *,
        connection,
        finalize_timeout_s: float,
        on_interim: Callable[[str], None] | None = None,
        on_endpoint: Callable[[StreamingSpeechEndpointEvent], None] | None = None,
    ) -> None:
        self._connection = connection
        self._finalize_timeout_s = max(0.5, float(finalize_timeout_s))
        self._on_interim = on_interim
        self._on_endpoint = on_endpoint
        self._send_lock = Lock()
        self._outgoing: Queue[bytes | str] = Queue()
        self._done = Event()
        self._closed = Event()
        self._reader_error: Exception | None = None
        self._final_segments: list[str] = []
        self._latest_interim: str = ""
        self._saw_interim = False
        self._saw_speech_final = False
        self._saw_utterance_end = False
        self._request_id = self._read_request_id()
        self._sender = Thread(target=self._sender_loop, daemon=True)
        self._sender.start()
        self._reader = Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    def send_pcm(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        if self._reader_error is not None:
            raise self._reader_error
        self._outgoing.put(bytes(pcm_bytes))

    def finalize(self) -> StreamingTranscriptionResult:
        if self._reader_error is not None:
            raise self._reader_error
        if self._sender.is_alive():
            self._outgoing.join()
        try:
            self._outgoing.put("Finalize")
            if self._sender.is_alive():
                self._outgoing.join()
        except Exception:
            pass
        self._done.wait(timeout=self._finalize_timeout_s)
        if self._reader_error is not None and not self._final_segments and not self._latest_interim:
            raise self._reader_error
        transcript = " ".join(segment for segment in self._final_segments if segment).strip()
        if not transcript:
            transcript = self._latest_interim.strip()
        self.close()
        return self._result_snapshot(transcript=transcript)

    def snapshot(self) -> StreamingTranscriptionResult:
        transcript = " ".join(segment for segment in self._final_segments if segment).strip()
        if not transcript:
            transcript = self._latest_interim.strip()
        return self._result_snapshot(transcript=transcript)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._outgoing.put("CloseStream")
        if self._sender.is_alive():
            self._outgoing.join()
        try:
            self._connection.close()
        except Exception:
            pass
        self._done.set()
        if self._sender.is_alive():
            self._sender.join(timeout=1.0)
        if self._reader.is_alive():
            self._reader.join(timeout=1.0)

    def _read_request_id(self) -> str | None:
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
        try:
            for message in self._connection:
                if isinstance(message, bytes):
                    continue
                payload = json.loads(message)
                event_type = str(payload.get("type", "")).strip()
                if event_type == "Results":
                    transcript = _extract_streaming_transcript(payload)
                    is_final = bool(payload.get("is_final"))
                    speech_final = bool(payload.get("speech_final"))
                    from_finalize = bool(payload.get("from_finalize"))
                    if transcript:
                        if is_final:
                            if not self._final_segments or self._final_segments[-1] != transcript:
                                self._final_segments.append(transcript)
                            self._latest_interim = transcript
                        else:
                            self._latest_interim = transcript
                            self._saw_interim = True
                            if self._on_interim is not None:
                                self._on_interim(transcript)
                    metadata = payload.get("metadata", {})
                    if self._request_id is None and isinstance(metadata, dict):
                        request_id = metadata.get("request_id")
                        if isinstance(request_id, str) and request_id.strip():
                            self._request_id = request_id.strip()
                    if speech_final and transcript and not from_finalize and self._on_endpoint is not None:
                        self._on_endpoint(
                            StreamingSpeechEndpointEvent(
                                transcript=transcript,
                                event_type="speech_final",
                                request_id=self._request_id,
                                is_final=is_final,
                                speech_final=speech_final,
                                from_finalize=from_finalize,
                            )
                        )
                    if speech_final or (is_final and from_finalize):
                        self._saw_speech_final = True
                        self._done.set()
                elif event_type == "UtteranceEnd":
                    self._saw_utterance_end = True
                    transcript = " ".join(segment for segment in self._final_segments if segment).strip()
                    if not transcript:
                        transcript = self._latest_interim.strip()
                    if transcript and self._on_endpoint is not None:
                        self._on_endpoint(
                            StreamingSpeechEndpointEvent(
                                transcript=transcript,
                                event_type="utterance_end",
                                request_id=self._request_id,
                            )
                        )
                    self._done.set()
                elif event_type in {"CloseStream", "Metadata"}:
                    continue
        except Exception as exc:  # pragma: no cover - network close races
            if not self._closed.is_set():
                self._reader_error = exc
        finally:
            self._done.set()

    def _result_snapshot(self, *, transcript: str) -> StreamingTranscriptionResult:
        return StreamingTranscriptionResult(
            transcript=transcript,
            request_id=self._request_id,
            saw_interim=self._saw_interim,
            saw_speech_final=self._saw_speech_final,
            saw_utterance_end=self._saw_utterance_end,
        )

    def _sender_loop(self) -> None:
        try:
            while True:
                payload = self._outgoing.get()
                try:
                    if self._reader_error is not None and payload not in {"CloseStream", "Finalize"}:
                        continue
                    with self._send_lock:
                        if isinstance(payload, bytes):
                            self._connection.send(payload)
                        else:
                            self._connection.send(json.dumps({"type": payload}))
                    if payload == "CloseStream":
                        return
                except Exception as exc:  # pragma: no cover - network close races
                    if not self._closed.is_set():
                        self._reader_error = exc
                        self._done.set()
                    return
                finally:
                    self._outgoing.task_done()
        finally:
            self._done.set()


class DeepgramSpeechToTextProvider:
    def __init__(
        self,
        config: TwinrConfig,
        *,
        client: httpx.Client | None = None,
        websocket_connector: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._client = client or httpx.Client(timeout=self.config.deepgram_timeout_s)
        self._websocket_connector = websocket_connector or websocket_connect

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        del filename, prompt
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
        return _extract_transcript(payload)

    def transcribe_path(
        self,
        path: str | Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        audio_path = Path(path)
        content_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
        return self.transcribe(
            audio_path.read_bytes(),
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
        del prompt
        api_key = (self.config.deepgram_api_key or "").strip()
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is required to use the Deepgram speech provider")
        resolved_language = (language or self.config.deepgram_stt_language or "").strip() or None
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
            max_size=None,
        )
        return _DeepgramStreamingSession(
            connection=connection,
            finalize_timeout_s=self.config.deepgram_streaming_finalize_timeout_s,
            on_interim=on_interim,
            on_endpoint=on_endpoint,
        )
