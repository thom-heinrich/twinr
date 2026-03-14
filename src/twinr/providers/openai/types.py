from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import mimetypes

from twinr.ops.usage import TokenUsage


@dataclass(frozen=True, slots=True)
class OpenAITextResponse:
    text: str
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: TokenUsage | None = None
    used_web_search: bool = False


@dataclass(frozen=True, slots=True)
class OpenAISearchResult:
    answer: str
    sources: tuple[str, ...] = ()
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: TokenUsage | None = None
    used_web_search: bool = False


@dataclass(frozen=True, slots=True)
class OpenAIImageInput:
    data: bytes
    content_type: str
    filename: str = "image"
    detail: str | None = None
    label: str | None = None

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        detail: str | None = None,
        label: str | None = None,
    ) -> "OpenAIImageInput":
        file_path = Path(path)
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        return cls(
            data=file_path.read_bytes(),
            content_type=content_type,
            filename=file_path.name,
            detail=detail,
            label=label,
        )


ConversationLike = Sequence[tuple[str, str]] | Sequence[object]
