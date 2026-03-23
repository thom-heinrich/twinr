"""Provide OpenAI speech-to-text and text-to-speech helpers for Twinr.

This module isolates audio request shaping, file-safety checks, model fallback,
and binary-stream cleanup for the OpenAI backend package.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from threading import Event, Lock
from typing import Any
import logging
import mimetypes
import os
import stat

from ..core.instructions import (
    STT_MODEL_FALLBACKS,
    TTS_MODEL_FALLBACKS,
    _LEGACY_TTS_FALLBACK_VOICE,
    _LEGACY_TTS_VOICES,
)

_DEFAULT_OPENAI_AUDIO_TIMEOUT_SECONDS = 30.0
_DEFAULT_TTS_SPEED = 1.0
_MIN_TTS_SPEED = 0.25
_MAX_TTS_SPEED = 4.0
_MAX_TTS_INPUT_CHARS = 4096
_JSON_ONLY_TRANSCRIPTION_MODELS = frozenset(
    {
        "gpt-4o-transcribe",
        "gpt-4o-mini-transcribe",
        "gpt-4o-mini-transcribe-2025-12-15",
    }
)
_PROMPT_UNSUPPORTED_TRANSCRIPTION_MODELS = frozenset({"gpt-4o-transcribe-diarize"})
_LEGACY_TTS_MODELS = frozenset({"tts-1", "tts-1-hd"})
_SUPPORTED_AUDIO_SUFFIXES = frozenset(
    {
        ".flac",
        ".m4a",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mpga",
        ".ogg",
        ".wav",
        ".webm",
    }
)

logger = logging.getLogger(__name__)


class _ClosableIterator(Iterator[bytes]):
    """Wrap an iterator so callers can close the underlying stream explicitly."""

    # AUDIT-FIX(#6): Wrap the generator so consumers can close the stream explicitly and abandoned iterators do not keep sockets open until arbitrary GC.
    def __init__(
        self,
        iterator: Iterator[bytes],
        *,
        close_callback: callable | None = None,
    ) -> None:
        """Store the wrapped byte iterator."""

        self._iterator = iterator
        self._close_callback = close_callback

    def __iter__(self) -> _ClosableIterator:
        """Return the iterator itself."""

        return self

    def __next__(self) -> bytes:
        """Yield the next chunk from the wrapped iterator."""

        return next(self._iterator)

    def close(self) -> None:
        """Close the wrapped iterator when it exposes ``close()``."""

        if callable(self._close_callback):
            self._close_callback()
            return
        close = getattr(self._iterator, "close", None)
        if callable(close):
            close()

    def __del__(self) -> None:
        """Attempt best-effort cleanup when the wrapper is garbage collected."""

        try:
            self.close()
        except Exception:
            logger.warning("OpenAI speech iterator cleanup failed during garbage collection.", exc_info=True)


class OpenAISpeechMixin:
    """Provide STT and TTS helpers for OpenAI-backed Twinr runtimes.

    The mixin keeps audio requests bounded, validates local file access, and
    applies model-aware request shaping for current OpenAI speech endpoints.
    """

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """Transcribe in-memory audio bytes into text.

        Args:
            audio_bytes: Audio payload to transcribe.
            filename: Filename hint sent to the provider.
            content_type: MIME type for the upload payload.
            language: Optional language hint.
            prompt: Optional transcription prompt when the model supports it.

        Returns:
            The normalized transcript text.

        Raises:
            ValueError: If ``audio_bytes`` is empty.
            RuntimeError: If the provider returns an unsupported response shape.
        """

        if not audio_bytes:
            # AUDIT-FIX(#5): Fail fast on empty payloads instead of turning a local misuse into an opaque provider error.
            raise ValueError("audio_bytes must not be empty")

        # AUDIT-FIX(#4): Bind STT/TTS calls to explicit per-request client options so voice requests do not inherit the SDK's long default timeout implicitly.
        request_client = self._get_audio_client()
        safe_filename = self._sanitize_upload_filename(filename)
        safe_content_type = self._normalize_content_type(content_type, safe_filename)

        response, _model_used = self._call_with_model_fallback(
            self.config.openai_stt_model,
            STT_MODEL_FALLBACKS,
            lambda model: request_client.audio.transcriptions.create(
                # AUDIT-FIX(#2): Build the transcription request per resolved fallback model because response_format/prompt support differs across current STT models.
                **self._build_transcription_request(
                    model=model,
                    file=(safe_filename, audio_bytes, safe_content_type),
                    language=language,
                    prompt=prompt,
                )
            ),
        )
        # AUDIT-FIX(#5): Reject unknown response objects explicitly so SDK/API contract changes cannot masquerade as an empty transcript.
        return self._extract_transcription_text(response)

    def transcribe_path(
        self,
        path: str | Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """Transcribe a local audio file after path and file-safety checks.

        Args:
            path: Local path to the audio file.
            language: Optional language hint.
            prompt: Optional transcription prompt when the model supports it.

        Returns:
            The normalized transcript text.
        """

        # AUDIT-FIX(#1): Resolve and confine local file access before reading/transmitting audio so path traversal, symlink swaps, and non-audio file exfiltration are harder to exploit.
        file_path = self._resolve_transcription_path(path)
        # AUDIT-FIX(#4): Bind STT/TTS calls to explicit per-request client options so voice requests do not inherit the SDK's long default timeout implicitly.
        request_client = self._get_audio_client()

        with self._open_audio_file(file_path) as audio_file:
            def call(model: str) -> Any:
                audio_file.seek(0)
                return request_client.audio.transcriptions.create(
                    # AUDIT-FIX(#1): Stream the local file handle directly instead of read_bytes() to avoid avoidable memory spikes on the RPi and keep the nofollow-open file descriptor.
                    **self._build_transcription_request(
                        model=model,
                        file=audio_file,
                        language=language,
                        prompt=prompt,
                    )
                )

            response, _model_used = self._call_with_model_fallback(
                self.config.openai_stt_model,
                STT_MODEL_FALLBACKS,
                call,
            )
        # AUDIT-FIX(#5): Reject unknown response objects explicitly so SDK/API contract changes cannot masquerade as an empty transcript.
        return self._extract_transcription_text(response)

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        """Synthesize speech into one in-memory audio payload.

        Args:
            text: Text to synthesize.
            voice: Optional voice override.
            response_format: Optional audio format override.
            instructions: Optional TTS instructions override.

        Returns:
            The synthesized audio payload as bytes.
        """

        tts_instructions = (
            instructions if instructions is not None else self.config.openai_tts_instructions
        )
        # AUDIT-FIX(#4): Bind STT/TTS calls to explicit per-request client options so voice requests do not inherit the SDK's long default timeout implicitly.
        request_client = self._get_audio_client()

        response, _model_used = self._call_with_model_fallback(
            self.config.openai_tts_model,
            TTS_MODEL_FALLBACKS,
            lambda model: request_client.audio.speech.create(
                **self._build_tts_request(
                    text,
                    model=model,
                    voice=voice,
                    response_format=response_format,
                    instructions=tts_instructions,
                )
            ),
        )
        # AUDIT-FIX(#6): Read and close binary responses deterministically so the mixin does not rely on GC for socket cleanup.
        return self._extract_binary_response(response)

    def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
        instructions: str | None = None,
        chunk_size: int = 4096,
    ) -> Iterator[bytes]:
        """Stream synthesized speech in chunks.

        Args:
            text: Text to synthesize.
            voice: Optional voice override.
            response_format: Optional audio format override.
            instructions: Optional TTS instructions override.
            chunk_size: Byte size for each streamed chunk.

        Returns:
            An iterator yielding synthesized audio chunks.

        Raises:
            ValueError: If ``chunk_size`` is not positive.
            RuntimeError: If no accessible TTS model can satisfy the request.
        """

        if chunk_size <= 0:
            # AUDIT-FIX(#8): Reject invalid chunk sizes locally instead of handing undefined values to iter_bytes().
            raise ValueError("chunk_size must be greater than 0")

        tts_instructions = (
            instructions if instructions is not None else self.config.openai_tts_instructions
        )
        # AUDIT-FIX(#4): Bind STT/TTS calls to explicit per-request client options so voice requests do not inherit the SDK's long default timeout implicitly.
        request_client = self._get_audio_client()
        stop_requested = Event()
        response_lock = Lock()
        response_holder: dict[str, Any | None] = {"response": None}

        def request_close() -> None:
            stop_requested.set()
            with response_lock:
                response = response_holder["response"]
            close = getattr(response, "close", None)
            if callable(close):
                close()

        def iterator() -> Iterator[bytes]:
            attempted_models: list[str] = []
            last_error: Exception | None = None
            for model in (self.config.openai_tts_model, *TTS_MODEL_FALLBACKS):
                if not model or model in attempted_models:
                    continue
                attempted_models.append(model)
                response = None
                try:
                    with request_client.audio.speech.with_streaming_response.create(
                        **self._build_tts_request(
                            text,
                            model=model,
                            voice=voice,
                            response_format=response_format,
                            instructions=tts_instructions,
                        )
                    ) as response:
                        with response_lock:
                            response_holder["response"] = response
                        for chunk in response.iter_bytes(chunk_size):
                            if stop_requested.is_set():
                                return
                            if chunk:
                                yield bytes(chunk)
                    return
                except Exception as exc:
                    if stop_requested.is_set():
                        return
                    if not self._is_model_access_error(exc):
                        raise
                    last_error = exc
                finally:
                    with response_lock:
                        if response_holder["response"] is response:
                            response_holder["response"] = None
            if last_error is not None:
                candidate_list = ", ".join(attempted_models)
                raise RuntimeError(
                    f"OpenAI project does not have access to any candidate models for this request: {candidate_list}"
                ) from last_error
            raise RuntimeError("No model candidates were available for the OpenAI request")

        # AUDIT-FIX(#6): Return a closable wrapper so callers can abort mid-stream without leaking the underlying response context.
        return _ClosableIterator(iterator(), close_callback=request_close)

    def _build_transcription_request(
        self,
        *,
        model: str,
        file: Any,
        language: str | None,
        prompt: str | None,
    ) -> dict[str, Any]:
        """Build a model-aware transcription request payload."""

        normalized_model = self._normalize_model_name(model)
        request: dict[str, Any] = {
            "model": model,
            "file": file,
            # AUDIT-FIX(#2): Use model-aware response_format selection because current transcription models no longer share a single compatible format.
            "response_format": self._transcription_response_format(normalized_model),
        }

        normalized_language = self._normalize_optional_text(language)
        if normalized_language is not None:
            request["language"] = normalized_language

        normalized_prompt = self._normalize_optional_text(prompt)
        if (
            normalized_prompt is not None
            and normalized_model not in _PROMPT_UNSUPPORTED_TRANSCRIPTION_MODELS
        ):
            # AUDIT-FIX(#2): Omit prompt on diarization models where the API does not accept it.
            request["prompt"] = normalized_prompt

        return request

    def _build_tts_request(
        self,
        text: str,
        *,
        model: str,
        voice: str | None,
        response_format: str | None,
        instructions: str | None,
    ) -> dict[str, Any]:
        """Build a validated text-to-speech request payload."""

        resolved_voice = voice if voice is not None else self.config.openai_tts_voice
        resolved_response_format = (
            response_format if response_format is not None else self.config.openai_tts_format
        )
        request: dict[str, Any] = {
            "model": model,
            "voice": self._resolve_tts_voice(model, resolved_voice),
            # AUDIT-FIX(#7): Coerce and range-check speed locally so a malformed .env value does not crash or produce an invalid provider request.
            "speed": self._coerce_tts_speed(self.config.openai_tts_speed),
            # AUDIT-FIX(#7): Normalize and bound TTS input locally so blank/oversized payloads fail deterministically before the provider call.
            "input": self._normalize_tts_input_text(text),
            "response_format": self._normalize_tts_response_format(resolved_response_format),
        }

        normalized_instructions = self._normalize_optional_text(instructions)
        if normalized_instructions is not None and not self._is_legacy_tts_model(model):
            # AUDIT-FIX(#3): Exclude instructions when the selected model is tts-1/tts-1-hd because those legacy models reject that parameter.
            request["instructions"] = normalized_instructions
        return request

    # AUDIT-FIX(#5): Normalize accepted STT response shapes centrally and fail closed for everything else.
    def _extract_transcription_text(self, response: Any) -> str:
        """Extract transcript text from supported response shapes."""

        if isinstance(response, str):
            return response.strip()

        if hasattr(response, "text"):
            return str(getattr(response, "text") or "").strip()

        raise RuntimeError("Unexpected transcription response type")

    # AUDIT-FIX(#6): Normalize accepted binary response shapes centrally and always attempt deterministic cleanup.
    def _extract_binary_response(self, response: Any) -> bytes:
        """Extract bytes from supported speech responses and close them."""

        close = getattr(response, "close", None)
        try:
            reader = getattr(response, "read", None)
            if reader is not None:
                payload = reader() if callable(reader) else reader
                if isinstance(payload, (bytes, bytearray)):
                    return bytes(payload)
                raise RuntimeError("Unexpected speech synthesis payload type from read()")

            content = getattr(response, "content", None)
            if content is not None:
                payload = content() if callable(content) else content
                if isinstance(payload, (bytes, bytearray)):
                    return bytes(payload)
                raise RuntimeError("Unexpected speech synthesis payload type from content")

            if isinstance(response, (bytes, bytearray)):
                return bytes(response)

            raise RuntimeError("Unexpected speech synthesis response type")
        finally:
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.warning("OpenAI speech response close failed after synthesis.", exc_info=True)

    # AUDIT-FIX(#2): Current STT models no longer share one universal response_format contract.
    def _transcription_response_format(self, normalized_model: str) -> str:
        """Return the response format supported by the transcription model."""

        if normalized_model in _JSON_ONLY_TRANSCRIPTION_MODELS:
            return "json"
        return "text"

    def _resolve_tts_voice(self, model: str, requested_voice: str) -> str:
        """Resolve the final TTS voice for the chosen model."""

        normalized_model = self._normalize_model_name(model)
        normalized_voice = str(requested_voice or "").strip()
        if self._is_legacy_tts_model(normalized_model) and normalized_voice not in _LEGACY_TTS_VOICES:
            return _LEGACY_TTS_FALLBACK_VOICE
        if not normalized_voice:
            # AUDIT-FIX(#7): Reject empty voices explicitly for non-legacy models instead of sending a malformed request downstream.
            raise ValueError("TTS voice must not be empty")
        return normalized_voice

    def _is_legacy_tts_model(self, model: str) -> bool:
        """Return whether the model uses the legacy TTS API contract."""

        return self._normalize_model_name(model) in _LEGACY_TTS_MODELS

    def _normalize_model_name(self, model: str) -> str:
        """Normalize a model identifier for comparisons."""

        return str(model or "").strip().lower()

    def _normalize_optional_text(self, value: str | None) -> str | None:
        """Strip optional text and collapse empty strings to ``None``."""

        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    def _normalize_tts_input_text(self, text: str) -> str:
        """Validate and normalize TTS input text."""

        normalized = text.strip()
        if not normalized:
            raise ValueError("TTS input text must not be empty")
        if len(normalized) > _MAX_TTS_INPUT_CHARS:
            raise ValueError(
                f"TTS input exceeds the supported limit of {_MAX_TTS_INPUT_CHARS} characters"
            )
        return normalized

    def _normalize_tts_response_format(self, response_format: str | None) -> str:
        """Validate and normalize the TTS response format."""

        normalized = self._normalize_optional_text(response_format)
        if normalized is None:
            raise ValueError("TTS response format must not be empty")
        return normalized

    def _coerce_tts_speed(self, raw_speed: Any) -> float:
        """Convert configured TTS speed into a supported float value."""

        try:
            speed = float(raw_speed)
        except (TypeError, ValueError):
            return _DEFAULT_TTS_SPEED

        if not (_MIN_TTS_SPEED <= speed <= _MAX_TTS_SPEED):
            return _DEFAULT_TTS_SPEED
        return speed

    def _sanitize_upload_filename(self, filename: str) -> str:
        """Return a printable filename safe to send in audio uploads."""

        raw_name = Path(str(filename or "audio.wav")).name.strip()
        candidate = "".join(ch for ch in raw_name if ch.isprintable() and ch not in {"/", "\\"})
        return candidate or "audio.wav"

    def _normalize_content_type(self, content_type: str | None, filename: str) -> str:
        """Return a valid content type for an uploaded audio payload."""

        if content_type is not None:
            normalized_content_type = content_type.strip()
            if (
                normalized_content_type
                and "/" in normalized_content_type
                and normalized_content_type.isprintable()
            ):
                return normalized_content_type
        return mimetypes.guess_type(filename)[0] or "application/octet-stream"

    # AUDIT-FIX(#4): Derive a request-scoped client view so timeout/retry policy is enforced consistently across STT, TTS, and streaming TTS.
    def _get_audio_client(self) -> Any:
        """Return a client view with audio-specific timeout and retry options."""

        options: dict[str, Any] = {}
        timeout = self._get_audio_request_timeout()
        if timeout is not None:
            options["timeout"] = timeout

        max_retries = self._get_audio_max_retries()
        if max_retries is not None:
            options["max_retries"] = max_retries

        if options and hasattr(self._client, "with_options"):
            return self._client.with_options(**options)
        return self._client

    def _get_audio_request_timeout(self) -> float | None:
        """Return the timeout applied to STT and TTS requests."""

        for attr_name in (
            "openai_audio_timeout_seconds",
            "openai_request_timeout",
            "openai_timeout_seconds",
        ):
            candidate = getattr(self.config, attr_name, None)
            if candidate is None:
                continue
            candidate_text = str(candidate).strip()
            if not candidate_text:
                continue
            try:
                timeout = float(candidate_text)
            except (TypeError, ValueError):
                return _DEFAULT_OPENAI_AUDIO_TIMEOUT_SECONDS
            return timeout if timeout > 0 else _DEFAULT_OPENAI_AUDIO_TIMEOUT_SECONDS

        # AUDIT-FIX(#4): Default to a bounded voice-request timeout instead of inheriting the SDK's much longer generic timeout.
        return _DEFAULT_OPENAI_AUDIO_TIMEOUT_SECONDS

    def _get_audio_max_retries(self) -> int | None:
        """Return the retry limit applied to audio requests, if configured."""

        for attr_name in ("openai_audio_max_retries", "openai_max_retries"):
            candidate = getattr(self.config, attr_name, None)
            if candidate is None:
                continue
            candidate_text = str(candidate).strip()
            if not candidate_text:
                continue
            try:
                return max(int(candidate_text), 0)
            except (TypeError, ValueError):
                return None
        return None

    # AUDIT-FIX(#1): Allow deployments to confine path-based transcription to an explicit root without breaking the existing config schema.
    def _get_audio_input_root(self) -> Path | None:
        """Return the optional filesystem root allowed for path transcription."""

        for attr_name in ("openai_audio_input_root", "audio_input_root"):
            candidate = getattr(self.config, attr_name, None)
            if candidate is None:
                continue
            candidate_text = str(candidate).strip()
            if candidate_text:
                return Path(candidate_text)
        return None

    # AUDIT-FIX(#1): Resolve the caller path once, reject obvious non-audio inputs, and optionally enforce a configured root before opening the file descriptor.
    def _resolve_transcription_path(self, path: str | Path) -> Path:
        """Resolve and validate a local audio path for transcription."""

        candidate_path = Path(path).expanduser()
        resolved_path = candidate_path.resolve(strict=True)

        if resolved_path.suffix.lower() not in _SUPPORTED_AUDIO_SUFFIXES:
            raise ValueError(
                "Unsupported audio file extension for transcription: "
                f"{resolved_path.suffix or '<none>'}"
            )

        if candidate_path.is_symlink():
            raise PermissionError("Symlinked audio inputs are not allowed")

        allowed_root = self._get_audio_input_root()
        if allowed_root is not None:
            resolved_root = allowed_root.expanduser().resolve(strict=True)
            try:
                resolved_path.relative_to(resolved_root)
            except ValueError as exc:
                raise PermissionError(
                    f"Audio input path escapes configured root: {resolved_root}"
                ) from exc

        return resolved_path

    # AUDIT-FIX(#1): Open the file descriptor with nofollow/close-on-exec flags and verify a regular file to reduce symlink and special-file abuse.
    def _open_audio_file(self, path: Path) -> Any:
        """Open a regular audio file with nofollow-style safety checks."""

        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW

        fd = os.open(path, flags)
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError("Audio input path must reference a regular file")
            return os.fdopen(fd, "rb")
        except Exception:
            os.close(fd)
            raise
