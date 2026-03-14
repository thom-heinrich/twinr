from __future__ import annotations

from pathlib import Path
import mimetypes

import httpx

from twinr.agent.base_agent.config import TwinrConfig


class DeepgramSpeechToTextProvider:
    def __init__(
        self,
        config: TwinrConfig,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self._client = client or httpx.Client(timeout=self.config.deepgram_timeout_s)

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
        transcript = (
            payload.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
        )
        if not isinstance(transcript, str):
            raise RuntimeError("Deepgram response did not contain a string transcript")
        return transcript.strip()

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
