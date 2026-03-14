from __future__ import annotations

from dataclasses import dataclass

from twinr.ops.usage import TokenUsage


@dataclass(frozen=True, slots=True)
class GroqTextResponse:
    text: str
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: TokenUsage | None = None
    used_web_search: bool = False
