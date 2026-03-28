from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from typing import TypedDict


_QR_DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aW7cAAAAASUVORK5CYII="


@dataclass(frozen=True, slots=True)
class _FakeFont:
    size: int
    bold: bool


class _RoundedRectangleCall(TypedDict):
    box: object
    radius: int
    fill: object
    outline: object | None
    width: int


class _TextCall(TypedDict):
    position: tuple[int, int]
    text: str
    fill: object
    font: _FakeFont


class _FakeSceneTools:
    def _font(self, size: int, *, bold: bool) -> _FakeFont:
        return _FakeFont(size=size, bold=bold)

    def _text_width(self, draw: object, text: str, *, font: object | None = None) -> int:
        del draw
        size = getattr(font, "size", 16)
        return len(text) * max(8, size // 2)

    def _text_height(self, draw: object, *, font: object | None = None) -> int:
        del draw
        return int(getattr(font, "size", 16))

    def _truncate_text(self, draw: object, text: str, *, max_width: int, font: object | None = None) -> str:
        del draw, max_width, font
        return text

    def _wrapped_lines(
        self,
        draw: object,
        lines: tuple[str, ...],
        *,
        max_width: int,
        font: object,
        max_lines: int,
    ) -> tuple[str, ...]:
        del draw, max_width, font
        wrapped: list[str] = []
        for line in lines:
            if not line:
                continue
            parts = [part.strip() for part in str(line).split("|") if part.strip()]
            if not parts:
                parts = [str(line)]
            for part in parts:
                wrapped.append(part)
                if len(wrapped) >= max_lines:
                    return tuple(wrapped[:max_lines])
        return tuple(wrapped[:max_lines])

    def _normalise_text(self, value: object, *, fallback: str) -> str:
        compact = " ".join(str(value or "").split()).strip()
        return compact or fallback

    def _render_emoji_glyph(self, emoji: str, *, target_size: int) -> object | None:
        del emoji, target_size
        return None


class _RecordingDraw:
    def __init__(self) -> None:
        self.rounded_rectangles: list[_RoundedRectangleCall] = []
        self.text_calls: list[_TextCall] = []

    def rounded_rectangle(
        self,
        box: object,
        *,
        radius: int,
        fill: object | None = None,
        outline: object | None = None,
        width: int = 1,
    ) -> None:
        self.rounded_rectangles.append(
            {
                "box": box,
                "radius": radius,
                "fill": fill,
                "outline": outline,
                "width": width,
            }
        )

    def text(
        self,
        position: object,
        text: str,
        *,
        fill: object,
        font: object,
    ) -> None:
        recorded_position = cast(tuple[int, int], position)
        recorded_font = cast(_FakeFont, font)
        self.text_calls.append(
            {
                "position": recorded_position,
                "text": text,
                "fill": fill,
                "font": recorded_font,
            }
        )
