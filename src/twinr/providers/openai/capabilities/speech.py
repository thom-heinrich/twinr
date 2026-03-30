# CHANGELOG: 2026-03-30
# BUG-1: Fixed TTS response_format handling; the helper no longer requires callers/config to set an explicit format and now defaults to the API defaults (`mp3` for buffered synthesis, `wav` for low-latency streaming).
# BUG-2: Fixed transcribe_path() confinement when an audio input root is configured; relative paths are now resolved against that root instead of the process CWD.
# BUG-3: Fixed late failure on oversized STT uploads by preflighting the 25 MB API limit and auto-chunking long plain-transcription inputs with PyDub when available.
# SEC-1: Hardened path- and bytes-based transcription against practical file exfiltration by validating container signatures in addition to suffixes, bounding upload sizes, and preserving nofollow/regular-file checks on opened descriptors.
# IMP-1: Added 2026 OpenAI speech features exposed by the current API: streamed transcription events, diarized transcription helpers, custom voice IDs, stream_format control, and current built-in voice validation.
# IMP-2: Added Pi-friendly low-latency defaults and continuity-aware long-audio stitching for plain transcription, including chunk overlap and prompt carry-forward across chunks.

"""Provide OpenAI speech-to-text and text-to-speech helpers for Twinr.

This module isolates audio request shaping, file-safety checks, model fallback,
and binary-stream cleanup for the OpenAI backend package.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Event, Lock
from typing import Any
import base64
import logging
import mimetypes
import os
import re
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
_MAX_TTS_INSTRUCTIONS_CHARS = 4096
_MAX_TRANSCRIPTION_UPLOAD_BYTES = 25 * 1024 * 1024
_DEFAULT_TTS_RESPONSE_FORMAT = "mp3"
_DEFAULT_TTS_STREAM_RESPONSE_FORMAT = "wav"
_DEFAULT_TTS_STREAM_FORMAT = "audio"
_DEFAULT_TRANSCRIPTION_CHUNK_OVERLAP_MS = 750
_DEFAULT_MAX_TRANSCRIPTION_CHUNK_DURATION_MS = 300_000
_MIN_TRANSCRIPTION_CHUNK_DURATION_MS = 10_000
_AUDIO_SIGNATURE_READ_BYTES = 64

_JSON_DEFAULT_TRANSCRIPTION_MODELS = frozenset(
    {
        "gpt-4o-transcribe",
        "gpt-4o-transcribe-latest",
        "gpt-4o-mini-transcribe",
        "gpt-4o-mini-transcribe-2025-03-20",
        "gpt-4o-mini-transcribe-2025-12-15",
    }
)
_PROMPT_UNSUPPORTED_TRANSCRIPTION_MODELS = frozenset({"gpt-4o-transcribe-diarize"})
_STREAM_UNSUPPORTED_TRANSCRIPTION_MODELS = frozenset({"whisper-1"})
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
        ".oga",
        ".wav",
        ".webm",
    }
)
_SUPPORTED_TTS_RESPONSE_FORMATS = frozenset({"mp3", "opus", "aac", "flac", "wav", "pcm"})
_SUPPORTED_TTS_STREAM_FORMATS = frozenset({"sse", "audio"})
_BUILTIN_TTS_VOICES = frozenset(
    {
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "nova",
        "onyx",
        "sage",
        "shimmer",
        "verse",
        "marin",
        "cedar",
    }
)
_NON_LEGACY_DEFAULT_TTS_VOICE = "marin"
_CUSTOM_VOICE_PREFIX = "voice_"
_AUDIO_FAMILY_TO_CONTENT_TYPE = {
    "flac": "audio/flac",
    "mpeg": "audio/mpeg",
    "mp4": "audio/mp4",
    "ogg": "audio/ogg",
    "wav": "audio/wav",
    "webm": "audio/webm",
}
_AUDIO_SUFFIX_FAMILIES = {
    ".flac": "flac",
    ".m4a": "mp4",
    ".mp3": "mpeg",
    ".mp4": "mp4",
    ".mpeg": "mpeg",
    ".mpga": "mpeg",
    ".ogg": "ogg",
    ".oga": "ogg",
    ".wav": "wav",
    ".webm": "webm",
}

logger = logging.getLogger(__name__)

VoiceLike = str | Mapping[str, Any] | None


class _ClosableIterator(Iterator[Any]):
    """Wrap an iterator so callers can close the underlying stream explicitly."""

    def __init__(
        self,
        iterator: Iterator[Any],
        *,
        close_callback: Callable[[], None] | None = None,
    ) -> None:
        self._iterator = iterator
        self._close_callback = close_callback

    def __iter__(self) -> _ClosableIterator:
        return self

    def __next__(self) -> Any:
        return next(self._iterator)

    def close(self) -> None:
        if callable(self._close_callback):
            self._close_callback()
            return
        close = getattr(self._iterator, "close", None)
        if callable(close):
            close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            logger.warning(
                "OpenAI speech iterator cleanup failed during garbage collection.",
                exc_info=True,
            )


class OpenAISpeechMixin:
    """Provide STT and TTS helpers for OpenAI-backed Twinr runtimes."""

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        if not audio_bytes:
            raise ValueError("audio_bytes must not be empty")

        safe_filename = self._sanitize_upload_filename(filename)
        sniffed_family = self._sniff_audio_family(audio_bytes[:_AUDIO_SIGNATURE_READ_BYTES])
        self._validate_audio_signature(
            filename=safe_filename,
            content_type=content_type,
            prefix=audio_bytes[:_AUDIO_SIGNATURE_READ_BYTES],
            sniffed_family=sniffed_family,
        )
        safe_content_type = self._normalize_content_type(
            content_type,
            safe_filename,
            sniffed_family=sniffed_family,
        )

        if len(audio_bytes) > self._get_transcription_max_upload_bytes():
            return self._transcribe_large_audio_bytes(
                audio_bytes,
                filename=safe_filename,
                sniffed_family=sniffed_family,
                language=language,
                prompt=prompt,
            )

        response = self._transcribe_upload(
            file=(safe_filename, audio_bytes, safe_content_type),
            language=language,
            prompt=prompt,
        )
        return self._extract_transcription_text(response)

    def transcribe_path(
        self,
        path: str | Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        file_path = self._resolve_transcription_path(path)
        with self._open_audio_file(file_path) as audio_file:
            file_size = self._get_file_size(audio_file)
            prefix = self._read_file_prefix(audio_file)
            sniffed_family = self._sniff_audio_family(prefix)
            self._validate_audio_signature(
                filename=file_path.name,
                content_type=mimetypes.guess_type(file_path.name)[0],
                prefix=prefix,
                sniffed_family=sniffed_family,
            )
            if file_size > self._get_transcription_max_upload_bytes():
                return self._transcribe_large_audio_path(
                    file_path,
                    language=language,
                    prompt=prompt,
                )

            response = self._transcribe_upload(
                file=audio_file,
                language=language,
                prompt=prompt,
            )

        return self._extract_transcription_text(response)

    def transcribe_stream(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> Iterator[Any]:
        if not audio_bytes:
            raise ValueError("audio_bytes must not be empty")

        safe_filename = self._sanitize_upload_filename(filename)
        sniffed_family = self._sniff_audio_family(audio_bytes[:_AUDIO_SIGNATURE_READ_BYTES])
        self._validate_audio_signature(
            filename=safe_filename,
            content_type=content_type,
            prefix=audio_bytes[:_AUDIO_SIGNATURE_READ_BYTES],
            sniffed_family=sniffed_family,
        )
        safe_content_type = self._normalize_content_type(
            content_type,
            safe_filename,
            sniffed_family=sniffed_family,
        )
        self._validate_transcription_stream_size(len(audio_bytes))

        response, _model_used = self._call_with_model_fallback(
            self._primary_streaming_stt_model(),
            self._streaming_stt_fallbacks(),
            lambda model: self._get_audio_client().audio.transcriptions.create(
                **self._build_transcription_request(
                    model=model,
                    file=(safe_filename, audio_bytes, safe_content_type),
                    language=language,
                    prompt=prompt,
                    response_format="text",
                    stream=True,
                )
            ),
        )
        return self._wrap_streaming_transcription_response(response)

    def transcribe_path_stream(
        self,
        path: str | Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> Iterator[Any]:
        file_path = self._resolve_transcription_path(path)
        with self._open_audio_file(file_path) as audio_file:
            file_size = self._get_file_size(audio_file)
            prefix = self._read_file_prefix(audio_file)
            sniffed_family = self._sniff_audio_family(prefix)
            self._validate_audio_signature(
                filename=file_path.name,
                content_type=mimetypes.guess_type(file_path.name)[0],
                prefix=prefix,
                sniffed_family=sniffed_family,
            )
            self._validate_transcription_stream_size(file_size)

            response, _model_used = self._call_with_model_fallback(
                self._primary_streaming_stt_model(),
                self._streaming_stt_fallbacks(),
                lambda model: self._stream_transcription_file_handle(
                    audio_file,
                    model=model,
                    language=language,
                    prompt=prompt,
                ),
            )

        return self._wrap_streaming_transcription_response(response)

    def transcribe_diarized(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        known_speaker_names: Sequence[str] | None = None,
        known_speaker_references: Sequence[bytes | str | Path] | None = None,
        chunking_strategy: str | Mapping[str, Any] = "auto",
    ) -> dict[str, Any]:
        if not audio_bytes:
            raise ValueError("audio_bytes must not be empty")
        if len(audio_bytes) > self._get_transcription_max_upload_bytes():
            raise ValueError(
                "Diarized transcription currently requires an upload smaller than the configured "
                "transcription size limit"
            )

        safe_filename = self._sanitize_upload_filename(filename)
        sniffed_family = self._sniff_audio_family(audio_bytes[:_AUDIO_SIGNATURE_READ_BYTES])
        self._validate_audio_signature(
            filename=safe_filename,
            content_type=content_type,
            prefix=audio_bytes[:_AUDIO_SIGNATURE_READ_BYTES],
            sniffed_family=sniffed_family,
        )
        safe_content_type = self._normalize_content_type(
            content_type,
            safe_filename,
            sniffed_family=sniffed_family,
        )

        response, _model_used = self._call_with_model_fallback(
            "gpt-4o-transcribe-diarize",
            (),
            lambda model: self._get_audio_client().audio.transcriptions.create(
                **self._build_transcription_request(
                    model=model,
                    file=(safe_filename, audio_bytes, safe_content_type),
                    language=language,
                    prompt=None,
                    response_format="diarized_json",
                    chunking_strategy=chunking_strategy,
                    extra_body=self._build_known_speaker_extra_body(
                        known_speaker_names,
                        known_speaker_references,
                    ),
                )
            ),
        )
        return self._normalize_structured_response(response)

    def transcribe_diarized_path(
        self,
        path: str | Path,
        *,
        language: str | None = None,
        known_speaker_names: Sequence[str] | None = None,
        known_speaker_references: Sequence[bytes | str | Path] | None = None,
        chunking_strategy: str | Mapping[str, Any] = "auto",
    ) -> dict[str, Any]:
        file_path = self._resolve_transcription_path(path)
        with self._open_audio_file(file_path) as audio_file:
            file_size = self._get_file_size(audio_file)
            prefix = self._read_file_prefix(audio_file)
            sniffed_family = self._sniff_audio_family(prefix)
            self._validate_audio_signature(
                filename=file_path.name,
                content_type=mimetypes.guess_type(file_path.name)[0],
                prefix=prefix,
                sniffed_family=sniffed_family,
            )
            if file_size > self._get_transcription_max_upload_bytes():
                raise ValueError(
                    "Diarized transcription currently requires an upload smaller than the configured "
                    "transcription size limit"
                )

            response, _model_used = self._call_with_model_fallback(
                "gpt-4o-transcribe-diarize",
                (),
                lambda model: self._transcribe_file_handle(
                    audio_file,
                    model=model,
                    language=language,
                    prompt=None,
                    response_format="diarized_json",
                    chunking_strategy=chunking_strategy,
                    extra_body=self._build_known_speaker_extra_body(
                        known_speaker_names,
                        known_speaker_references,
                    ),
                ),
            )

        return self._normalize_structured_response(response)

    def synthesize(
        self,
        text: str,
        *,
        voice: VoiceLike = None,
        response_format: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        tts_instructions = (
            instructions if instructions is not None else self.config.openai_tts_instructions
        )

        response, _model_used = self._call_with_model_fallback(
            self.config.openai_tts_model,
            TTS_MODEL_FALLBACKS,
            lambda model: self._get_audio_client().audio.speech.create(
                **self._build_tts_request(
                    text,
                    model=model,
                    voice=voice,
                    response_format=response_format,
                    instructions=tts_instructions,
                    for_stream=False,
                )
            ),
        )
        return self._extract_binary_response(response)

    def synthesize_stream(
        self,
        text: str,
        *,
        voice: VoiceLike = None,
        response_format: str | None = None,
        instructions: str | None = None,
        chunk_size: int = 4096,
        stream_format: str | None = None,
    ) -> Iterator[bytes]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        tts_instructions = (
            instructions if instructions is not None else self.config.openai_tts_instructions
        )
        request_client = self._get_audio_client()
        stop_requested = Event()
        response_lock = Lock()
        response_holder: dict[str, Any | None] = {"response": None}

        def request_close() -> None:
            stop_requested.set()
            with response_lock:
                response = response_holder["response"]
            close_method = getattr(response, "close", None)
            if callable(close_method):
                close_method()  # pylint: disable=not-callable

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
                            for_stream=True,
                            stream_format=stream_format,
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
                    "OpenAI project does not have access to any candidate models for this request: "
                    f"{candidate_list}"
                ) from last_error
            raise RuntimeError("No model candidates were available for the OpenAI request")

        return _ClosableIterator(iterator(), close_callback=request_close)

    def _transcribe_upload(
        self,
        *,
        file: Any,
        language: str | None,
        prompt: str | None,
    ) -> Any:
        response, _model_used = self._call_with_model_fallback(
            self.config.openai_stt_model,
            STT_MODEL_FALLBACKS,
            lambda model: self._transcribe_file_handle(
                file,
                model=model,
                language=language,
                prompt=prompt,
            ),
        )
        return response

    def _transcribe_file_handle(
        self,
        file: Any,
        *,
        model: str,
        language: str | None,
        prompt: str | None,
        response_format: str | None = None,
        stream: bool = False,
        include: Sequence[str] | None = None,
        chunking_strategy: str | Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> Any:
        request_client = self._get_audio_client()
        seek = getattr(file, "seek", None)
        if callable(seek):
            seek(0)
        return request_client.audio.transcriptions.create(
            **self._build_transcription_request(
                model=model,
                file=file,
                language=language,
                prompt=prompt,
                response_format=response_format,
                stream=stream,
                include=include,
                chunking_strategy=chunking_strategy,
                extra_body=extra_body,
            )
        )

    def _stream_transcription_file_handle(
        self,
        file: Any,
        *,
        model: str,
        language: str | None,
        prompt: str | None,
    ) -> Any:
        return self._transcribe_file_handle(
            file,
            model=model,
            language=language,
            prompt=prompt,
            response_format="text",
            stream=True,
        )

    def _wrap_streaming_transcription_response(self, response: Any) -> Iterator[Any]:
        iterator = iter(response)
        close = getattr(response, "close", None)
        return _ClosableIterator(iterator, close_callback=close if callable(close) else None)

    def _build_transcription_request(
        self,
        *,
        model: str,
        file: Any,
        language: str | None,
        prompt: str | None,
        response_format: str | None = None,
        stream: bool = False,
        include: Sequence[str] | None = None,
        chunking_strategy: str | Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_model = self._normalize_model_name(model)
        request: dict[str, Any] = {
            "model": model,
            "file": file,
            "response_format": self._transcription_response_format(
                normalized_model,
                requested=response_format,
            ),
        }

        normalized_language = self._normalize_optional_text(language)
        if normalized_language is not None:
            request["language"] = normalized_language

        normalized_prompt = self._normalize_optional_text(prompt)
        if (
            normalized_prompt is not None
            and normalized_model not in _PROMPT_UNSUPPORTED_TRANSCRIPTION_MODELS
        ):
            request["prompt"] = normalized_prompt

        if stream:
            request["stream"] = True

        if include:
            request["include"] = list(include)

        if chunking_strategy is not None:
            request["chunking_strategy"] = chunking_strategy
        elif normalized_model == "gpt-4o-transcribe-diarize":
            request["chunking_strategy"] = "auto"

        if extra_body:
            request["extra_body"] = dict(extra_body)

        return request

    def _build_tts_request(
        self,
        text: str,
        *,
        model: str,
        voice: VoiceLike,
        response_format: str | None,
        instructions: str | None,
        for_stream: bool,
        stream_format: str | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model,
            "voice": self._resolve_tts_voice(model, voice),
            "speed": self._coerce_tts_speed(self.config.openai_tts_speed),
            "input": self._normalize_tts_input_text(text),
            "response_format": self._resolve_tts_response_format(
                response_format,
                for_stream=for_stream,
            ),
        }

        normalized_instructions = self._normalize_tts_instructions(instructions)
        if normalized_instructions is not None and not self._is_legacy_tts_model(model):
            request["instructions"] = normalized_instructions

        if for_stream:
            normalized_stream_format = self._resolve_tts_stream_format(model, stream_format)
            if normalized_stream_format is not None:
                request["stream_format"] = normalized_stream_format

        return request

    def _extract_transcription_text(self, response: Any) -> str:
        if isinstance(response, str):
            return response.strip()

        if isinstance(response, Mapping):
            return str(response.get("text") or "").strip()

        if hasattr(response, "text"):
            return str(getattr(response, "text") or "").strip()

        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            payload = model_dump()
            if isinstance(payload, Mapping):
                return str(payload.get("text") or "").strip()

        raise RuntimeError("Unexpected transcription response type")

    def _normalize_structured_response(self, response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            return dict(response)

        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            payload = model_dump()
            if isinstance(payload, dict):
                return payload

        to_dict = getattr(response, "dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, dict):
                return payload

        raise RuntimeError("Unexpected structured transcription response type")

    def _extract_binary_response(self, response: Any) -> bytes:
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
                    logger.warning(
                        "OpenAI speech response close failed after synthesis.",
                        exc_info=True,
                    )

    def _transcription_response_format(
        self,
        normalized_model: str,
        *,
        requested: str | None,
    ) -> str:
        normalized_requested = self._normalize_optional_text(requested)
        if normalized_requested is None:
            if normalized_model in _JSON_DEFAULT_TRANSCRIPTION_MODELS:
                return "json"
            if normalized_model == "gpt-4o-transcribe-diarize":
                return "text"
            return "text"

        supported_formats = self._supported_transcription_response_formats(normalized_model)
        if normalized_requested not in supported_formats:
            supported_text = ", ".join(sorted(supported_formats))
            raise ValueError(
                f"Transcription response format {normalized_requested!r} is not supported by "
                f"model {normalized_model!r}; supported formats: {supported_text}"
            )
        return normalized_requested

    def _supported_transcription_response_formats(self, normalized_model: str) -> frozenset[str]:
        if normalized_model == "gpt-4o-transcribe-diarize":
            return frozenset({"json", "text", "diarized_json"})
        if normalized_model in _JSON_DEFAULT_TRANSCRIPTION_MODELS:
            # Current OpenAI docs are inconsistent here. We keep `json` as the safe default
            # but also allow explicit `text`, matching the official streaming example.
            return frozenset({"json", "text"})
        return frozenset({"json", "text", "srt", "verbose_json", "vtt"})

    def _resolve_tts_voice(self, model: str, requested_voice: VoiceLike) -> str | dict[str, str]:
        normalized_model = self._normalize_model_name(model)
        resolved_voice = self._default_tts_voice(requested_voice)

        if isinstance(resolved_voice, Mapping):
            voice_id = self._normalize_optional_text(str(resolved_voice.get("id") or ""))
            if voice_id is None:
                raise ValueError("Custom TTS voice objects must provide a non-empty 'id'")
            if self._is_legacy_tts_model(normalized_model):
                raise ValueError("Legacy TTS models do not support custom voice IDs")
            return {"id": voice_id}

        normalized_voice = self._normalize_optional_text(str(resolved_voice))
        if normalized_voice is None:
            normalized_voice = self._default_builtin_tts_voice(normalized_model)

        if self._is_legacy_tts_model(normalized_model):
            if normalized_voice not in _LEGACY_TTS_VOICES:
                return _LEGACY_TTS_FALLBACK_VOICE
            return normalized_voice

        if normalized_voice in _BUILTIN_TTS_VOICES:
            return normalized_voice
        if normalized_voice.startswith(_CUSTOM_VOICE_PREFIX):
            return {"id": normalized_voice}

        supported = ", ".join(sorted(_BUILTIN_TTS_VOICES))
        raise ValueError(
            f"Unsupported TTS voice {normalized_voice!r} for model {normalized_model!r}; "
            f"use one of: {supported}, or pass a custom voice id"
        )

    def _default_tts_voice(self, requested_voice: VoiceLike) -> VoiceLike:
        if requested_voice is not None:
            return requested_voice

        configured_voice_id = getattr(self.config, "openai_tts_voice_id", None)
        normalized_voice_id = self._normalize_optional_text(
            str(configured_voice_id) if configured_voice_id is not None else None
        )
        if normalized_voice_id is not None:
            return {"id": normalized_voice_id}

        configured_voice = getattr(self.config, "openai_tts_voice", None)
        if isinstance(configured_voice, Mapping):
            return configured_voice

        normalized_voice = self._normalize_optional_text(
            str(configured_voice) if configured_voice is not None else None
        )
        return normalized_voice

    def _default_builtin_tts_voice(self, normalized_model: str) -> str:
        if self._is_legacy_tts_model(normalized_model):
            return _LEGACY_TTS_FALLBACK_VOICE
        return _NON_LEGACY_DEFAULT_TTS_VOICE

    def _is_legacy_tts_model(self, model: str) -> bool:
        return self._normalize_model_name(model) in _LEGACY_TTS_MODELS

    def _normalize_model_name(self, model: str) -> str:
        return str(model or "").strip().lower()

    def _normalize_optional_text(self, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    def _normalize_tts_input_text(self, text: str) -> str:
        normalized = text.strip()
        if not normalized:
            raise ValueError("TTS input text must not be empty")
        if len(normalized) > _MAX_TTS_INPUT_CHARS:
            raise ValueError(
                f"TTS input exceeds the supported limit of {_MAX_TTS_INPUT_CHARS} characters"
            )
        return normalized

    def _normalize_tts_instructions(self, instructions: str | None) -> str | None:
        normalized = self._normalize_optional_text(instructions)
        if normalized is None:
            return None
        if len(normalized) > _MAX_TTS_INSTRUCTIONS_CHARS:
            raise ValueError(
                "TTS instructions exceed the supported limit of "
                f"{_MAX_TTS_INSTRUCTIONS_CHARS} characters"
            )
        return normalized

    def _resolve_tts_response_format(self, response_format: str | None, *, for_stream: bool) -> str:
        normalized = self._normalize_optional_text(response_format)
        if normalized is None:
            candidate_attrs = (
                ("openai_tts_stream_response_format", "openai_tts_format")
                if for_stream
                else ("openai_tts_format",)
            )
            for attr_name in candidate_attrs:
                candidate = getattr(self.config, attr_name, None)
                normalized_candidate = self._normalize_optional_text(
                    str(candidate) if candidate is not None else None
                )
                if normalized_candidate is not None:
                    normalized = normalized_candidate
                    break

        if normalized is None:
            return (
                _DEFAULT_TTS_STREAM_RESPONSE_FORMAT
                if for_stream
                else _DEFAULT_TTS_RESPONSE_FORMAT
            )

        if normalized not in _SUPPORTED_TTS_RESPONSE_FORMATS:
            supported = ", ".join(sorted(_SUPPORTED_TTS_RESPONSE_FORMATS))
            raise ValueError(
                f"Unsupported TTS response format {normalized!r}; supported formats: {supported}"
            )
        return normalized

    def _resolve_tts_stream_format(self, model: str, stream_format: str | None) -> str | None:
        normalized = self._normalize_optional_text(stream_format)
        if normalized is None:
            candidate = getattr(self.config, "openai_tts_stream_format", None)
            normalized = self._normalize_optional_text(
                str(candidate) if candidate is not None else None
            )

        if normalized is None:
            normalized = _DEFAULT_TTS_STREAM_FORMAT

        if normalized not in _SUPPORTED_TTS_STREAM_FORMATS:
            supported = ", ".join(sorted(_SUPPORTED_TTS_STREAM_FORMATS))
            raise ValueError(
                f"Unsupported TTS stream_format {normalized!r}; supported formats: {supported}"
            )

        if normalized == "sse" and self._is_legacy_tts_model(model):
            raise ValueError("stream_format='sse' is not supported for legacy TTS models")

        return normalized

    def _coerce_tts_speed(self, raw_speed: Any) -> float:
        try:
            speed = float(raw_speed)
        except (TypeError, ValueError):
            return _DEFAULT_TTS_SPEED

        if not (_MIN_TTS_SPEED <= speed <= _MAX_TTS_SPEED):
            return _DEFAULT_TTS_SPEED
        return speed

    def _sanitize_upload_filename(self, filename: str) -> str:
        raw_name = Path(str(filename or "audio.wav")).name.strip()
        candidate = "".join(ch for ch in raw_name if ch.isprintable() and ch not in {"/", "\\"})
        return candidate or "audio.wav"

    def _normalize_content_type(
        self,
        content_type: str | None,
        filename: str,
        *,
        sniffed_family: str | None = None,
    ) -> str:
        normalized_content_type = self._normalize_optional_text(content_type)
        if (
            normalized_content_type is not None
            and "/" in normalized_content_type
            and normalized_content_type.isprintable()
            and normalized_content_type.split("/", 1)[0] in {"audio", "video", "application"}
        ):
            if normalized_content_type != "application/octet-stream":
                return normalized_content_type

        if sniffed_family is not None:
            return _AUDIO_FAMILY_TO_CONTENT_TYPE.get(sniffed_family, "application/octet-stream")

        return mimetypes.guess_type(filename)[0] or "application/octet-stream"

    def _get_audio_client(self) -> Any:
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

        return _DEFAULT_OPENAI_AUDIO_TIMEOUT_SECONDS

    def _get_audio_max_retries(self) -> int | None:
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

    def _get_audio_input_root(self) -> Path | None:
        for attr_name in ("openai_audio_input_root", "audio_input_root"):
            candidate = getattr(self.config, attr_name, None)
            if candidate is None:
                continue
            candidate_text = str(candidate).strip()
            if candidate_text:
                return Path(candidate_text)
        return None

    def _resolve_transcription_path(self, path: str | Path) -> Path:
        candidate_path = Path(path).expanduser()
        allowed_root = self._get_audio_input_root()

        if allowed_root is not None and not candidate_path.is_absolute():
            candidate_path = allowed_root.expanduser() / candidate_path

        resolved_path = candidate_path.resolve(strict=True)

        if resolved_path.suffix.lower() not in _SUPPORTED_AUDIO_SUFFIXES:
            raise ValueError(
                "Unsupported audio file extension for transcription: "
                f"{resolved_path.suffix or '<none>'}"
            )

        if Path(path).expanduser().is_symlink():
            raise PermissionError("Symlinked audio inputs are not allowed")

        if allowed_root is not None:
            resolved_root = allowed_root.expanduser().resolve(strict=True)
            try:
                resolved_path.relative_to(resolved_root)
            except ValueError as exc:
                raise PermissionError(
                    f"Audio input path escapes configured root: {resolved_root}"
                ) from exc

        return resolved_path

    def _open_audio_file(self, path: Path) -> Any:
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

    def _get_file_size(self, file_obj: Any) -> int:
        fileno = getattr(file_obj, "fileno", None)
        if callable(fileno):
            return os.fstat(fileno()).st_size

        position = file_obj.tell()
        file_obj.seek(0, os.SEEK_END)
        size = file_obj.tell()
        file_obj.seek(position)
        return size

    def _read_file_prefix(self, file_obj: Any, size: int = _AUDIO_SIGNATURE_READ_BYTES) -> bytes:
        position = file_obj.tell()
        file_obj.seek(0)
        prefix = file_obj.read(size)
        file_obj.seek(position)
        return prefix

    def _sniff_audio_family(self, prefix: bytes) -> str | None:
        if len(prefix) >= 12 and prefix[:4] == b"RIFF" and prefix[8:12] == b"WAVE":
            return "wav"
        if prefix.startswith(b"fLaC"):
            return "flac"
        if prefix.startswith(b"OggS"):
            return "ogg"
        if len(prefix) >= 4 and prefix[:4] == b"\x1a\x45\xdf\xa3":
            return "webm"
        if len(prefix) >= 8 and prefix[4:8] == b"ftyp":
            return "mp4"
        if prefix.startswith(b"ID3"):
            return "mpeg"
        if len(prefix) >= 2 and prefix[0] == 0xFF and (prefix[1] & 0xE0) == 0xE0:
            return "mpeg"
        return None

    def _validate_audio_signature(
        self,
        *,
        filename: str,
        content_type: str | None,
        prefix: bytes,
        sniffed_family: str | None = None,
    ) -> None:
        if not prefix:
            raise ValueError("Audio payload must not be empty")

        suffix = Path(filename).suffix.lower()
        expected_family = _AUDIO_SUFFIX_FAMILIES.get(suffix)
        normalized_content_type = self._normalize_optional_text(content_type)
        normalized_content_type_lower = normalized_content_type.lower() if normalized_content_type else None
        family = sniffed_family or self._sniff_audio_family(prefix)
        if (
            family is None
            and prefix.startswith(b"RIFF")
            and (
                expected_family == "wav"
                or normalized_content_type_lower
                in {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}
            )
        ):
            family = "wav"
        if family is None:
            raise ValueError("Audio payload is not a recognized supported container")

        if expected_family is not None and family != expected_family:
            raise ValueError(
                f"Audio payload signature does not match filename extension {suffix!r}"
            )

        if normalized_content_type is None:
            return
        if normalized_content_type_lower == "application/octet-stream":
            return
        if "/" not in normalized_content_type_lower:
            raise ValueError("content_type must contain a media type")
        major_type = normalized_content_type_lower.split("/", 1)[0]
        if major_type not in {"audio", "video"}:
            raise ValueError("content_type must describe audio or video media")

    def _get_transcription_max_upload_bytes(self) -> int:
        for attr_name in (
            "openai_transcription_max_upload_bytes",
            "openai_audio_max_upload_bytes",
            "audio_max_upload_bytes",
        ):
            candidate = getattr(self.config, attr_name, None)
            if candidate is None:
                continue
            candidate_text = str(candidate).strip()
            if not candidate_text:
                continue
            try:
                parsed = int(candidate_text)
            except (TypeError, ValueError):
                return _MAX_TRANSCRIPTION_UPLOAD_BYTES
            return max(parsed, 1)
        return _MAX_TRANSCRIPTION_UPLOAD_BYTES

    def _validate_transcription_stream_size(self, payload_size: int) -> None:
        if payload_size > self._get_transcription_max_upload_bytes():
            raise ValueError(
                "Streaming transcription of completed audio recordings currently requires an "
                "upload smaller than the configured transcription size limit"
            )

    def _should_auto_chunk_large_transcriptions(self) -> bool:
        for attr_name in ("openai_audio_auto_chunk", "audio_auto_chunk"):
            candidate = getattr(self.config, attr_name, None)
            if candidate is None:
                continue
            if isinstance(candidate, bool):
                return candidate
            candidate_text = str(candidate).strip().lower()
            if candidate_text in {"1", "true", "yes", "on"}:
                return True
            if candidate_text in {"0", "false", "no", "off"}:
                return False
        return True

    def _get_transcription_chunk_overlap_ms(self) -> int:
        for attr_name in ("openai_audio_chunk_overlap_ms", "audio_chunk_overlap_ms"):
            candidate = getattr(self.config, attr_name, None)
            if candidate is None:
                continue
            try:
                return max(int(str(candidate).strip()), 0)
            except (TypeError, ValueError):
                return _DEFAULT_TRANSCRIPTION_CHUNK_OVERLAP_MS
        return _DEFAULT_TRANSCRIPTION_CHUNK_OVERLAP_MS

    def _get_transcription_max_chunk_duration_ms(self) -> int:
        for attr_name in (
            "openai_audio_max_chunk_duration_ms",
            "audio_max_chunk_duration_ms",
        ):
            candidate = getattr(self.config, attr_name, None)
            if candidate is None:
                continue
            try:
                return max(int(str(candidate).strip()), _MIN_TRANSCRIPTION_CHUNK_DURATION_MS)
            except (TypeError, ValueError):
                return _DEFAULT_MAX_TRANSCRIPTION_CHUNK_DURATION_MS
        return _DEFAULT_MAX_TRANSCRIPTION_CHUNK_DURATION_MS

    def _transcribe_large_audio_bytes(
        self,
        audio_bytes: bytes,
        *,
        filename: str,
        sniffed_family: str | None,
        language: str | None,
        prompt: str | None,
    ) -> str:
        if not self._should_auto_chunk_large_transcriptions():
            raise ValueError(
                "Audio upload exceeds the configured transcription size limit; enable auto-chunking "
                "or split the audio before calling transcribe()"
            )

        suffix = Path(filename).suffix or self._preferred_suffix_for_audio_family(sniffed_family)
        with NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            return self._transcribe_large_audio_path(
                Path(temp_file.name),
                language=language,
                prompt=prompt,
            )

    def _transcribe_large_audio_path(
        self,
        path: Path,
        *,
        language: str | None,
        prompt: str | None,
    ) -> str:
        if not self._should_auto_chunk_large_transcriptions():
            raise ValueError(
                "Audio upload exceeds the configured transcription size limit; enable auto-chunking "
                "or split the audio before calling transcribe_path()"
            )

        try:
            from pydub import AudioSegment  # type: ignore
        except ImportError as exc:
            raise ValueError(
                "Audio upload exceeds the configured transcription size limit and auto-chunking "
                "requires the optional 'pydub' dependency plus ffmpeg"
            ) from exc

        logger.info("Auto-chunking oversized audio file for transcription: %s", path)

        audio = AudioSegment.from_file(str(path))
        if len(audio) <= 0:
            raise ValueError("Audio file appears to be empty")

        chunk_duration_ms = self._derive_transcription_chunk_duration_ms(audio)
        overlap_ms = min(
            self._get_transcription_chunk_overlap_ms(),
            max(chunk_duration_ms // 10, 0),
        )
        step_ms = max(chunk_duration_ms - overlap_ms, _MIN_TRANSCRIPTION_CHUNK_DURATION_MS)

        chunk_transcripts: list[str] = []
        previous_transcript: str | None = None

        for start_ms in range(0, len(audio), step_ms):
            end_ms = min(start_ms + chunk_duration_ms, len(audio))
            if end_ms <= start_ms:
                break

            chunk = audio[start_ms:end_ms]
            if len(chunk) <= 0:
                continue

            chunk_prompt = self._build_chunk_prompt(prompt, previous_transcript)
            with NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                chunk.export(temp_file.name, format="wav")
                with open(temp_file.name, "rb") as exported_chunk:
                    chunk_text = self._extract_transcription_text(
                        self._transcribe_upload(
                            file=exported_chunk,
                            language=language,
                            prompt=chunk_prompt,
                        )
                    )

            normalized_chunk_text = chunk_text.strip()
            if normalized_chunk_text:
                chunk_transcripts.append(normalized_chunk_text)
                previous_transcript = normalized_chunk_text

            if end_ms >= len(audio):
                break

        if not chunk_transcripts:
            return ""

        return self._merge_transcript_chunks(chunk_transcripts)

    def _derive_transcription_chunk_duration_ms(self, audio_segment: Any) -> int:
        bytes_per_second = max(
            int(audio_segment.frame_rate)
            * int(audio_segment.channels)
            * int(audio_segment.sample_width),
            1,
        )
        usable_bytes = max(self._get_transcription_max_upload_bytes() - 4096, 1)
        derived_ms = int((usable_bytes / bytes_per_second) * 1000 * 0.98)
        bounded_ms = max(derived_ms, _MIN_TRANSCRIPTION_CHUNK_DURATION_MS)
        return min(bounded_ms, self._get_transcription_max_chunk_duration_ms())

    def _build_chunk_prompt(
        self,
        base_prompt: str | None,
        previous_transcript: str | None,
    ) -> str | None:
        parts: list[str] = []

        normalized_base = self._normalize_optional_text(base_prompt)
        if normalized_base is not None:
            parts.append(normalized_base)

        normalized_previous = self._normalize_optional_text(previous_transcript)
        if normalized_previous is not None:
            parts.append(normalized_previous[-800:])

        if not parts:
            return None
        return "\n\n".join(parts)

    def _merge_transcript_chunks(self, chunks: Sequence[str]) -> str:
        merged = ""
        for chunk in chunks:
            normalized_chunk = chunk.strip()
            if not normalized_chunk:
                continue
            if not merged:
                merged = normalized_chunk
                continue
            merged = self._merge_transcript_pair(merged, normalized_chunk)
        return merged.strip()

    def _merge_transcript_pair(self, left: str, right: str) -> str:
        left_words = left.split()
        right_words = right.split()
        max_overlap = min(len(left_words), len(right_words), 48)
        overlap = 0

        for candidate in range(max_overlap, 0, -1):
            left_slice = [self._normalize_overlap_token(token) for token in left_words[-candidate:]]
            right_slice = [self._normalize_overlap_token(token) for token in right_words[:candidate]]
            if left_slice == right_slice:
                overlap = candidate
                break

        if overlap:
            right_words = right_words[overlap:]

        if not right_words:
            return left
        separator = "" if left.endswith((" ", "\n")) else " "
        return left + separator + " ".join(right_words)

    def _normalize_overlap_token(self, token: str) -> str:
        return re.sub(r"^\W+|\W+$", "", token).lower()

    def _primary_streaming_stt_model(self) -> str:
        candidates = [self.config.openai_stt_model, *STT_MODEL_FALLBACKS]
        for candidate in candidates:
            normalized = self._normalize_model_name(candidate)
            if normalized and normalized not in _STREAM_UNSUPPORTED_TRANSCRIPTION_MODELS:
                return candidate
        raise RuntimeError("No streaming-capable STT model is configured")

    def _streaming_stt_fallbacks(self) -> tuple[str, ...]:
        fallbacks: list[str] = []
        primary = self._normalize_model_name(self._primary_streaming_stt_model())
        for candidate in STT_MODEL_FALLBACKS:
            normalized = self._normalize_model_name(candidate)
            if (
                normalized
                and normalized != primary
                and normalized not in _STREAM_UNSUPPORTED_TRANSCRIPTION_MODELS
            ):
                fallbacks.append(candidate)
        return tuple(fallbacks)

    def _build_known_speaker_extra_body(
        self,
        names: Sequence[str] | None,
        references: Sequence[bytes | str | Path] | None,
    ) -> dict[str, Any] | None:
        if names is None and references is None:
            return None
        if not names or not references:
            raise ValueError("known_speaker_names and known_speaker_references must both be set")
        if len(names) != len(references):
            raise ValueError("known_speaker_names and known_speaker_references must have equal length")
        if len(names) > 4:
            raise ValueError("OpenAI diarization supports at most 4 known speakers")

        normalized_names = [self._normalize_known_speaker_name(name) for name in names]
        data_urls = [self._coerce_audio_reference_to_data_url(reference) for reference in references]
        return {
            "known_speaker_names": normalized_names,
            "known_speaker_references": data_urls,
        }

    def _normalize_known_speaker_name(self, name: str) -> str:
        normalized = self._normalize_optional_text(name)
        if normalized is None:
            raise ValueError("Known speaker names must not be empty")
        return normalized

    def _preferred_suffix_for_audio_family(self, family: str | None) -> str:
        for suffix, suffix_family in _AUDIO_SUFFIX_FAMILIES.items():
            if suffix_family == family:
                return suffix
        return ".wav"

    def _coerce_audio_reference_to_data_url(self, reference: bytes | str | Path) -> str:
        if isinstance(reference, bytes):
            prefix = reference[:_AUDIO_SIGNATURE_READ_BYTES]
            family = self._sniff_audio_family(prefix)
            self._validate_audio_signature(
                filename="reference.wav" if family == "wav" else "reference.bin",
                content_type=None,
                prefix=prefix,
                sniffed_family=family,
            )
            content_type = _AUDIO_FAMILY_TO_CONTENT_TYPE.get(
                family or "",
                "application/octet-stream",
            )
            encoded = base64.b64encode(reference).decode("ascii")
            return f"data:{content_type};base64,{encoded}"

        reference_text = str(reference)
        if reference_text.startswith("data:"):
            return reference_text

        file_path = self._resolve_transcription_path(reference_text)
        with self._open_audio_file(file_path) as audio_file:
            prefix = self._read_file_prefix(audio_file)
            family = self._sniff_audio_family(prefix)
            self._validate_audio_signature(
                filename=file_path.name,
                content_type=mimetypes.guess_type(file_path.name)[0],
                prefix=prefix,
                sniffed_family=family,
            )
            payload = audio_file.read()

        content_type = _AUDIO_FAMILY_TO_CONTENT_TYPE.get(family or "", "application/octet-stream")
        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:{content_type};base64,{encoded}"
