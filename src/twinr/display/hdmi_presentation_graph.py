# CHANGELOG: 2026-03-28
# BUG-1: Restore the stable HDMI scene-box contract; Twinr's layout boxes are LTRB, so the presentation graph must not treat them as XYWH by default.
# BUG-2: Harden emotion parsing and numeric coercion so malformed cue values do not crash the HDMI presentation loop.
# BUG-3: Bound and sanitize title/subtitle/body payloads to prevent LLM-sized cue text from stalling the Pi 4 render path.
# SEC-1: Sanitize image_path; reject missing, non-regular, symlinked, non-image, or oversized files before the renderer touches them.
# IMP-1: Add 2026 reduced-motion support with fade-first presentation states aligned with current accessibility guidance.
# IMP-2: Add policy-driven motion, box semantics, accent contrast normalization, and pluggable emotion resolution for context-aware HRI.

"""Resolve Twinr's HDMI presentation cues into a small scene graph.

The framebuffer/Wayland adapters should not decide card priority, morph stage,
or face-sync behavior inline. This module owns the capability-driven graph
model that selects one active presentation node, computes calm transition
states, and derives an optional face reaction while keeping those concerns out
of the transport and status-loop layers.
"""

from __future__ import annotations

import math
import os
import stat as stat_module
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from twinr.display.face_cues import DisplayFaceCue
from twinr.display.face_expressions import DisplayFaceEmotion, DisplayFaceExpression, DisplayFaceGazeDirection
from twinr.display.presentation_cues import DisplayPresentationCardCue, DisplayPresentationCue


Box = tuple[int, int, int, int]
RgbColor = tuple[int, int, int]
BoxMode = Literal["xywh", "ltrb"]

_DEFAULT_ALLOWED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
_DEFAULT_ACCENT_BACKGROUND: RgbColor = (0, 0, 0)
_REDUCE_MOTION_ENV_VARS = ("TWINR_REDUCE_MOTION", "PREFERS_REDUCED_MOTION")
_FOCUSED_STAGES = frozenset({"focused", "steady", "settled", "visible", "idle"})
_CONTROL_CHAR_TABLE = {
    ordinal: " "
    for ordinal in range(32)
    if chr(ordinal) not in ("\n", "\r", "\t")
}

_PRESENTATION_ACCENTS: dict[str, RgbColor] = {
    "alert": (215, 108, 92),
    "info": (84, 146, 255),
    "success": (64, 168, 118),
    "warm": (228, 152, 34),
}
_ACCENT_EMOTIONS = {
    "alert": DisplayFaceEmotion.FOCUSED,
    "info": DisplayFaceEmotion.CURIOUS,
    "success": DisplayFaceEmotion.HAPPY,
    "warm": DisplayFaceEmotion.HAPPY,
}


@dataclass(frozen=True, slots=True)
class HdmiPresentationMotionProfile:
    """Progress segmentation used to derive calm card transitions."""

    chrome_start: float
    chrome_end: float
    content_start: float
    content_end: float
    body_start: float
    body_end: float
    box_morph: bool = True
    use_easing: bool = True
    style: str = "morph"


_GENTLE_MOTION_PROFILE = HdmiPresentationMotionProfile(
    chrome_start=0.00,
    chrome_end=0.42,
    content_start=0.14,
    content_end=0.76,
    body_start=0.30,
    body_end=0.92,
    box_morph=True,
    use_easing=True,
    style="morph",
)
_REDUCED_MOTION_PROFILE = HdmiPresentationMotionProfile(
    chrome_start=0.00,
    chrome_end=0.18,
    content_start=0.00,
    content_end=0.30,
    body_start=0.04,
    body_end=0.42,
    box_morph=False,
    use_easing=True,
    style="fade",
)


@dataclass(frozen=True, slots=True)
class HdmiPresentationNode:
    """Describe one active HDMI presentation node ready for drawing."""

    key: str
    kind: str
    priority: int
    stage: str
    progress: float
    eased_progress: float
    chrome_progress: float
    content_progress: float
    body_progress: float
    box: Box
    title: str
    subtitle: str
    body_lines: tuple[str, ...]
    image_path: str | None
    accent: RgbColor
    motion_style: str = "morph"


@dataclass(frozen=True, slots=True)
class HdmiPresentationSceneGraph:
    """Prepared HDMI presentation graph with one active node and queue info."""

    active_node: HdmiPresentationNode | None = None
    queued_cards: tuple[DisplayPresentationCardCue, ...] = ()
    face_cue: DisplayFaceCue | None = None
    reduce_motion: bool = False

    def telemetry_signature(self) -> tuple[object, ...] | None:
        """Return a semantic signature suitable for low-noise display logs."""

        node = self.active_node
        if node is None:
            return None
        return (node.key, node.kind, node.priority, node.stage, len(self.queued_cards))


@dataclass(slots=True)
class HdmiPresentationSceneGraphBuilder:
    """Resolve one optional presentation cue into a calm HDMI scene graph."""

    reduce_motion: bool | None = None
    box_mode: BoxMode = "ltrb"
    max_title_chars: int = 96
    max_subtitle_chars: int = 160
    max_body_lines: int = 6
    max_body_line_chars: int = 180
    max_image_bytes: int = 16 * 1024 * 1024
    allow_symlink_images: bool = False
    allowed_image_suffixes: tuple[str, ...] = _DEFAULT_ALLOWED_IMAGE_SUFFIXES
    allowed_image_roots: tuple[str | os.PathLike[str], ...] = ()
    accent_palette: Mapping[str, RgbColor] = field(default_factory=lambda: dict(_PRESENTATION_ACCENTS))
    accent_background: RgbColor = _DEFAULT_ACCENT_BACKGROUND
    min_accent_contrast: float = 3.0
    emotion_resolver: Callable[[DisplayPresentationCardCue], DisplayFaceEmotion | str | None] | None = None
    face_sync_enabled: bool = True

    def __post_init__(self) -> None:
        if self.reduce_motion is None:
            self.reduce_motion = _read_reduce_motion_env()
        self.box_mode = self._normalize_box_mode(self.box_mode)
        self.allowed_image_suffixes = tuple(
            sorted(
                {
                    suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
                    for suffix in self.allowed_image_suffixes
                }
            )
        )
        self.allowed_image_roots = tuple(
            Path(root).expanduser().resolve(strict=False) for root in self.allowed_image_roots
        )
        self.accent_palette = {
            str(name).strip().lower(): _coerce_rgb_triplet(
                color,
                fallback=_PRESENTATION_ACCENTS["info"],
            )
            for name, color in self.accent_palette.items()
        }
        self.accent_background = _coerce_rgb_triplet(
            self.accent_background,
            fallback=_DEFAULT_ACCENT_BACKGROUND,
        )
        self.max_title_chars = max(8, int(self.max_title_chars))
        self.max_subtitle_chars = max(8, int(self.max_subtitle_chars))
        self.max_body_lines = max(1, int(self.max_body_lines))
        self.max_body_line_chars = max(8, int(self.max_body_line_chars))
        self.max_image_bytes = max(1, int(self.max_image_bytes))
        self.min_accent_contrast = max(1.0, float(self.min_accent_contrast))

    def build(
        self,
        *,
        cue: DisplayPresentationCue | None,
        face_box: Box,
        panel_box: Box,
        now: datetime | None = None,
    ) -> HdmiPresentationSceneGraph | None:
        """Return one bounded graph for the active HDMI presentation cue."""

        if cue is None:
            return None
        active_card = cue.active_card()
        if active_card is None:
            return None

        stage = self._normalize_stage(cue.transition_stage(now=now))
        progress = self._clamp01(cue.transition_progress(now=now))
        motion_profile = self._motion_profile()
        eased_progress = self._ease_progress(progress) if motion_profile.use_easing else progress

        normalized_face_box = self._normalize_box(face_box)
        normalized_panel_box = self._normalize_box(panel_box)
        fullscreen_box = self._union_box(normalized_face_box, normalized_panel_box)
        node_box = (
            fullscreen_box
            if not motion_profile.box_morph
            else self._interpolate_box(normalized_panel_box, fullscreen_box, eased_progress)
        )

        kind = self._sanitize_inline_text(getattr(active_card, "kind", ""), max_chars=32)
        title = self._sanitize_inline_text(getattr(active_card, "title", ""), max_chars=self.max_title_chars)
        subtitle = self._sanitize_inline_text(
            getattr(active_card, "subtitle", ""),
            max_chars=self.max_subtitle_chars,
        )
        node_key = self._sanitize_inline_text(getattr(active_card, "key", ""), max_chars=96)
        if not node_key:
            node_key = _truncate(f"{kind}:{title}".strip(":"), max_chars=96) or "presentation"

        active_node = HdmiPresentationNode(
            key=node_key,
            kind=kind,
            priority=self._coerce_int(getattr(active_card, "priority", 0)),
            stage=stage,
            progress=progress,
            eased_progress=eased_progress,
            chrome_progress=self._segment_progress(
                eased_progress,
                start=motion_profile.chrome_start,
                end=motion_profile.chrome_end,
            ),
            content_progress=self._segment_progress(
                eased_progress,
                start=motion_profile.content_start,
                end=motion_profile.content_end,
            ),
            body_progress=self._segment_progress(
                eased_progress,
                start=motion_profile.body_start,
                end=motion_profile.body_end,
            ),
            box=node_box,
            title=title,
            subtitle=subtitle,
            body_lines=self._sanitize_body_lines(getattr(active_card, "body_lines", ())),
            image_path=self._sanitize_image_path(getattr(active_card, "image_path", None)),
            accent=self._accent_for_card(active_card),
            motion_style=motion_profile.style,
        )
        queued_cards = tuple(cue.queued_cards())
        return HdmiPresentationSceneGraph(
            active_node=active_node,
            queued_cards=queued_cards,
            face_cue=self._face_sync_for_card(active_card, stage=stage, progress=eased_progress),
            reduce_motion=bool(self.reduce_motion),
        )

    def _motion_profile(self) -> HdmiPresentationMotionProfile:
        return _REDUCED_MOTION_PROFILE if self.reduce_motion else _GENTLE_MOTION_PROFILE

    def _normalize_box_mode(self, box_mode: str) -> BoxMode:
        normalized = str(box_mode or "ltrb").strip().lower()
        if normalized not in {"xywh", "ltrb"}:
            return "ltrb"
        return normalized  # type: ignore[return-value]

    def _normalize_box(self, box: Sequence[object]) -> Box:
        values = tuple(self._coerce_int(value) for value in tuple(box)[:4])
        if len(values) != 4:
            return (0, 0, 0, 0)
        if self.box_mode == "ltrb":
            left, top, right, bottom = values
            if right < left:
                left, right = right, left
            if bottom < top:
                top, bottom = bottom, top
            return (left, top, right, bottom)
        x, y, width, height = values
        if width < 0:
            x += width
            width = abs(width)
        if height < 0:
            y += height
            height = abs(height)
        return (x, y, width, height)

    def _union_box(self, first: Box, second: Box) -> Box:
        if self.box_mode == "ltrb":
            return (
                min(first[0], second[0]),
                min(first[1], second[1]),
                max(first[2], second[2]),
                max(first[3], second[3]),
            )
        x0 = min(first[0], second[0])
        y0 = min(first[1], second[1])
        x1 = max(first[0] + first[2], second[0] + second[2])
        y1 = max(first[1] + first[3], second[1] + second[3])
        return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))

    def _interpolate_box(
        self,
        start: Box,
        end: Box,
        progress: float,
    ) -> Box:
        """Interpolate one rectangle between start and end geometry."""

        safe_progress = self._clamp01(progress)
        return tuple(
            int(round(start_value + ((end_value - start_value) * safe_progress)))
            for start_value, end_value in zip(start, end)
        )  # type: ignore[return-value]

    def _ease_progress(self, progress: float) -> float:
        """Apply a smoothstep easing curve for calmer card morphs."""

        safe_progress = self._clamp01(progress)
        return safe_progress * safe_progress * (3.0 - (2.0 * safe_progress))

    def _segment_progress(self, progress: float, *, start: float, end: float) -> float:
        """Map one eased progress value into a later content-reveal segment."""

        if progress <= start:
            return 0.0
        if progress >= end:
            return 1.0
        return self._clamp01((progress - start) / max(0.01, end - start))

    def _face_sync_for_card(
        self,
        card: DisplayPresentationCardCue,
        *,
        stage: str,
        progress: float,
    ) -> DisplayFaceCue | None:
        """Return one optional face reaction synchronized to the active card."""

        if not self.face_sync_enabled:
            return None
        if stage in _FOCUSED_STAGES and progress >= 0.94:
            return None

        emotion = self._emotion_for_card(card)
        gaze = DisplayFaceGazeDirection.RIGHT
        if emotion == DisplayFaceEmotion.THOUGHTFUL:
            gaze = DisplayFaceGazeDirection.UP_RIGHT
        elif emotion == DisplayFaceEmotion.SAD:
            gaze = DisplayFaceGazeDirection.DOWN_RIGHT

        head_dx = 0 if self.reduce_motion or stage in _FOCUSED_STAGES or progress >= 0.80 else 1
        expression = DisplayFaceExpression.from_emotion(
            emotion,
            intensity=0.8,
            gaze=gaze,
            head_dx=head_dx,
            head_dy=0,
        )
        return expression.to_cue(source="presentation_graph")

    def _emotion_for_card(self, card: DisplayPresentationCardCue) -> DisplayFaceEmotion:
        """Derive one conservative face emotion for the active card."""

        if self.emotion_resolver is not None:
            try:
                resolved_value = self.emotion_resolver(card)
            except Exception:
                resolved_value = None
            resolved = self._coerce_emotion(resolved_value)
            if resolved is not None:
                return resolved

        face_emotion = self._coerce_emotion(getattr(card, "face_emotion", None))
        if face_emotion is not None:
            return face_emotion

        if getattr(card, "kind", "") == "image":
            return DisplayFaceEmotion.CURIOUS
        return _ACCENT_EMOTIONS.get(getattr(card, "accent", ""), DisplayFaceEmotion.FOCUSED)

    def _coerce_emotion(self, value: object) -> DisplayFaceEmotion | None:
        if value in (None, ""):
            return None
        if isinstance(value, DisplayFaceEmotion):
            return value
        try:
            return DisplayFaceEmotion(value)
        except Exception:
            if isinstance(value, str):
                normalized = value.strip().upper()
                for emotion in DisplayFaceEmotion:
                    if emotion.name == normalized:
                        return emotion
            return None

    def _sanitize_inline_text(self, value: object, *, max_chars: int) -> str:
        text = _strip_controls(str(value or ""))
        text = " ".join(text.split())
        return _truncate(text, max_chars=max_chars)

    def _sanitize_body_lines(self, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            raw_lines = value.splitlines() or [value]
        elif isinstance(value, Sequence):
            raw_lines = [str(item) for item in value]
        else:
            raw_lines = [str(value)]

        sanitized: list[str] = []
        for raw_line in raw_lines[: self.max_body_lines]:
            line = _strip_controls(raw_line).replace("\n", " ").replace("\r", " ")
            line = " ".join(line.split())
            sanitized.append(_truncate(line, max_chars=self.max_body_line_chars))
        return tuple(sanitized)

    def _sanitize_image_path(self, raw_path: object) -> str | None:
        if raw_path in (None, ""):
            return None

        # BREAKING: invalid, non-image, symlinked, special, or oversized paths are now dropped to None.
        candidate = Path(str(raw_path)).expanduser()
        try:
            if candidate.is_symlink() and not self.allow_symlink_images:
                return None
        except OSError:
            return None

        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            return None
        if self.allowed_image_roots and not _is_within_any_root(resolved, self.allowed_image_roots):
            return None
        if resolved.suffix.lower() not in self.allowed_image_suffixes:
            return None

        try:
            stat_result = resolved.stat()
        except OSError:
            return None

        if not stat_module.S_ISREG(stat_result.st_mode):
            return None
        if stat_result.st_size > self.max_image_bytes:
            return None
        return str(resolved)

    def _accent_for_card(self, card: DisplayPresentationCardCue) -> RgbColor:
        accent_key = self._sanitize_inline_text(getattr(card, "accent", ""), max_chars=24).lower()
        base_accent = self.accent_palette.get(accent_key, self.accent_palette.get("info", _PRESENTATION_ACCENTS["info"]))
        return _ensure_minimum_contrast(
            base_accent,
            background=self.accent_background,
            min_ratio=self.min_accent_contrast,
        )

    def _normalize_stage(self, stage: object) -> str:
        normalized = self._sanitize_inline_text(stage, max_chars=32).lower()
        return normalized or "focused"

    def _coerce_int(self, value: object) -> int:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0
        if not math.isfinite(numeric):
            return 0
        return int(round(numeric))

    def _clamp01(self, value: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(numeric):
            return 0.0
        return max(0.0, min(1.0, numeric))


def _read_reduce_motion_env() -> bool:
    for key in _REDUCE_MOTION_ENV_VARS:
        raw_value = os.getenv(key)
        if raw_value is None:
            continue
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return False


def _truncate(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 1:
        return value[:max_chars]
    return value[: max_chars - 1].rstrip() + "…"


def _strip_controls(value: str) -> str:
    return value.translate(_CONTROL_CHAR_TABLE)


def _coerce_rgb_triplet(value: object, *, fallback: RgbColor) -> RgbColor:
    if not isinstance(value, Sequence) or len(value) != 3:
        return fallback
    try:
        red, green, blue = (int(float(channel)) for channel in value)
    except (TypeError, ValueError):
        return fallback
    return (_clamp_byte(red), _clamp_byte(green), _clamp_byte(blue))


def _clamp_byte(value: int) -> int:
    return max(0, min(255, int(value)))


def _relative_luminance(color: RgbColor) -> float:
    def _channel(channel: int) -> float:
        normalized = channel / 255.0
        if normalized <= 0.03928:
            return normalized / 12.92
        return ((normalized + 0.055) / 1.055) ** 2.4

    red, green, blue = color
    return (
        0.2126 * _channel(red)
        + 0.7152 * _channel(green)
        + 0.0722 * _channel(blue)
    )


def _contrast_ratio(first: RgbColor, second: RgbColor) -> float:
    lighter = max(_relative_luminance(first), _relative_luminance(second))
    darker = min(_relative_luminance(first), _relative_luminance(second))
    return (lighter + 0.05) / (darker + 0.05)


def _mix_colors(start: RgbColor, end: RgbColor, weight: float) -> RgbColor:
    safe_weight = max(0.0, min(1.0, float(weight)))
    return tuple(
        int(round(start_channel + ((end_channel - start_channel) * safe_weight)))
        for start_channel, end_channel in zip(start, end)
    )  # type: ignore[return-value]


def _ensure_minimum_contrast(
    foreground: RgbColor,
    *,
    background: RgbColor,
    min_ratio: float,
) -> RgbColor:
    if _contrast_ratio(foreground, background) >= min_ratio:
        return foreground

    brighten_target = (255, 255, 255)
    darken_target = (0, 0, 0)
    preferred_target = brighten_target if _relative_luminance(background) < 0.5 else darken_target

    best = foreground
    low = 0.0
    high = 1.0
    for _ in range(12):
        mid = (low + high) / 2.0
        candidate = _mix_colors(foreground, preferred_target, mid)
        if _contrast_ratio(candidate, background) >= min_ratio:
            best = candidate
            high = mid
        else:
            low = mid
    return best


def _is_within_any_root(path: Path, roots: Sequence[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False
