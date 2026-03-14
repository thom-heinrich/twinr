from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
import mimetypes

from .instructions import (
    STT_MODEL_FALLBACKS,
    TTS_MODEL_FALLBACKS,
    _LEGACY_TTS_FALLBACK_VOICE,
    _LEGACY_TTS_VOICES,
)


class OpenAISpeechMixin:
    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.config.openai_stt_model,
            "file": (filename, audio_bytes, content_type),
            "response_format": "text",
        }
        if language:
            request["language"] = language
        if prompt:
            request["prompt"] = prompt
        response, _model_used = self._call_with_model_fallback(
            self.config.openai_stt_model,
            STT_MODEL_FALLBACKS,
            lambda model: self._client.audio.transcriptions.create(**{**request, "model": model}),
        )
        if isinstance(response, str):
            return response.strip()
        return str(getattr(response, "text", "")).strip()

    def transcribe_path(
        self,
        path: str | Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        file_path = Path(path)
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        return self.transcribe(
            file_path.read_bytes(),
            filename=file_path.name,
            content_type=content_type,
            language=language,
            prompt=prompt,
        )

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        tts_instructions = instructions or self.config.openai_tts_instructions
        response, _model_used = self._call_with_model_fallback(
            self.config.openai_tts_model,
            TTS_MODEL_FALLBACKS,
            lambda model: self._client.audio.speech.create(
                **self._build_tts_request(
                    text,
                    model=model,
                    voice=voice,
                    response_format=response_format,
                    instructions=tts_instructions,
                )
            ),
        )
        if hasattr(response, "read"):
            return bytes(response.read())
        if hasattr(response, "content"):
            return bytes(response.content)
        if isinstance(response, (bytes, bytearray)):
            return bytes(response)
        raise RuntimeError("Unexpected speech synthesis response type")

    def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
        instructions: str | None = None,
        chunk_size: int = 4096,
    ) -> Iterator[bytes]:
        tts_instructions = instructions or self.config.openai_tts_instructions

        def iterator() -> Iterator[bytes]:
            attempted_models: list[str] = []
            last_error: Exception | None = None
            for model in (self.config.openai_tts_model, *TTS_MODEL_FALLBACKS):
                if not model or model in attempted_models:
                    continue
                attempted_models.append(model)
                try:
                    with self._client.audio.speech.with_streaming_response.create(
                        **self._build_tts_request(
                            text,
                            model=model,
                            voice=voice,
                            response_format=response_format,
                            instructions=tts_instructions,
                        )
                    ) as response:
                        for chunk in response.iter_bytes(chunk_size):
                            if chunk:
                                yield bytes(chunk)
                    return
                except Exception as exc:
                    if not self._is_model_access_error(exc):
                        raise
                    last_error = exc
            if last_error is not None:
                candidate_list = ", ".join(attempted_models)
                raise RuntimeError(
                    f"OpenAI project does not have access to any candidate models for this request: {candidate_list}"
                ) from last_error
            raise RuntimeError("No model candidates were available for the OpenAI request")

        return iterator()

    def _build_tts_request(
        self,
        text: str,
        *,
        model: str,
        voice: str | None,
        response_format: str | None,
        instructions: str | None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model,
            "voice": self._resolve_tts_voice(model, voice or self.config.openai_tts_voice),
            "speed": float(self.config.openai_tts_speed),
            "input": text.strip(),
            "response_format": response_format or self.config.openai_tts_format,
        }
        if instructions:
            request["instructions"] = instructions
        return request

    def _resolve_tts_voice(self, model: str, requested_voice: str) -> str:
        normalized_model = model.strip().lower()
        if normalized_model in {"tts-1", "tts-1-hd"} and requested_voice not in _LEGACY_TTS_VOICES:
            return _LEGACY_TTS_FALLBACK_VOICE
        return requested_voice
