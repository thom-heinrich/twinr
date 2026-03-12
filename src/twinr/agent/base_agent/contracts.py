from __future__ import annotations

from typing import Protocol


class SpeechToTextProvider(Protocol):
    def transcribe(self, audio_bytes: bytes) -> str:
        ...


class FoundationModelProvider(Protocol):
    def respond(self, prompt: str) -> str:
        ...


class TextToSpeechProvider(Protocol):
    def synthesize(self, text: str) -> bytes:
        ...


class PrintFormatter(Protocol):
    def format_for_print(self, text: str) -> str:
        ...
