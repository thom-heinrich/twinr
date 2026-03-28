# CHANGELOG: 2026-03-28
# BUG-1: Deduplicate duplicate state fields with last-write-wins semantics so stale values and duplicate HDMI cards are no longer rendered.
# BUG-2: Skip blank synthesized detail rows and normalize empty/raw time and status values so the UI no longer emits empty cards or blank clock text.
# BUG-3: Canonicalize status/value aliases with Unicode-normalized caseless matching so equivalent runtime states no longer fall through to wrong headlines, helper text, or colors.
# SEC-1: Strip dangerous Unicode bidi/control characters from externally supplied HDMI text to reduce practical UI-spoofing risks on operator-facing displays.
# SEC-2: Cap externally supplied text lengths before downstream wrapping/measurement to bound render cost and reduce practical text-based UI DoS on Raspberry Pi 4.
# IMP-1: Introduce typed canonical status/semantic models plus centralized copy tables for consistent, frontier-grade edge UI behavior.
# IMP-2: Cache normalization and hot-path semantic lookups to reduce per-frame overhead on resource-constrained devices.

"""Shared models and helper functions for Twinr's default HDMI scene."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Protocol
import unicodedata

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11 fallback
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport-compatible fallback for Python versions without enum.StrEnum."""

        def __str__(self) -> str:
            return str(self.value)


from .typing_contracts import (
    HdmiAmbientMomentLike,
    HdmiEmojiCueLike,
    HdmiFaceCueLike,
    HdmiHeaderSignalLike,
    HdmiPanelDrawSurface,
    HdmiPresentationGraphLike,
    HdmiReserveBusLike,
)

_STATE_CARD_ORDER: Final[tuple[str, ...]] = ("Status", "Internet", "AI", "System", "Zeit", "Hinweis")
_STATE_CARD_ORDER_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(_STATE_CARD_ORDER)
}
_DETAIL_MAX_LINES: Final[int] = 3
_DEFAULT_HEADER_SIGNAL_ROWS: Final[int] = 2
_DEFAULT_SCENE_SHOW_TICKER: Final[bool] = False
_PROMPT_MODE_LINE_GAP: Final[int] = 6
_PROMPT_MODE_SECTION_GAP: Final[int] = 8
_PROMPT_MODE_WRAP_MAX_LINES: Final[int] = 16

# Bounded text budgets keep render cost predictable on a Raspberry Pi 4 even when
# upstream state comes from noisy or untrusted sources.
_MAX_LOOKUP_TEXT_CHARS: Final[int] = 128
_MAX_FIELD_LABEL_CHARS: Final[int] = 24
_MAX_FIELD_VALUE_CHARS: Final[int] = 72
_MAX_DETAIL_VALUE_CHARS: Final[int] = 160
_MAX_HEADLINE_CHARS: Final[int] = 48
_MAX_HELPER_CHARS: Final[int] = 120
_MAX_TIME_CHARS: Final[int] = 16

# Unicode bidi formatting controls can make HDMI text visually misleading.
_BIDI_FORMATTING_CHARS: Final[frozenset[str]] = frozenset(
    {
        "\u061c",  # ARABIC LETTER MARK
        "\u200e",  # LEFT-TO-RIGHT MARK
        "\u200f",  # RIGHT-TO-LEFT MARK
        "\u202a",  # LEFT-TO-RIGHT EMBEDDING
        "\u202b",  # RIGHT-TO-LEFT EMBEDDING
        "\u202c",  # POP DIRECTIONAL FORMATTING
        "\u202d",  # LEFT-TO-RIGHT OVERRIDE
        "\u202e",  # RIGHT-TO-LEFT OVERRIDE
        "\u2066",  # LEFT-TO-RIGHT ISOLATE
        "\u2067",  # RIGHT-TO-LEFT ISOLATE
        "\u2068",  # FIRST STRONG ISOLATE
        "\u2069",  # POP DIRECTIONAL ISOLATE
    }
)

DisplayStateFields = tuple[tuple[str, str], ...]


class HdmiRuntimeStatus(StrEnum):
    """Canonical runtime statuses for the senior-facing Twinr HDMI scene."""

    WAITING = "waiting"
    LISTENING = "listening"
    PROCESSING = "processing"
    ANSWERING = "answering"
    PRINTING = "printing"
    ERROR = "error"


class HdmiValueSemantic(StrEnum):
    """Canonical semantic meaning for short HDMI state values."""

    OK = "ok"
    READY = "ready"
    ONLINE = "online"
    OFFLINE = "offline"
    MISSING = "missing"
    WAITING = "waiting"
    UNKNOWN = "unknown"
    WARNING = "warning"
    ERROR = "error"
    WARM = "warm"
    UPDATING = "updating"
    UNAVAILABLE = "unavailable"
    CONNECTING = "connecting"


_FIELD_NAME_ALIASES: Final[Mapping[str, str]] = {
    "status": "Status",
    "zustand": "Status",
    "internet": "Internet",
    "network": "Internet",
    "netzwerk": "Internet",
    "netz": "Internet",
    "wifi": "Internet",
    "wi-fi": "Internet",
    "wlan": "Internet",
    "ai": "AI",
    "assistant": "AI",
    "model": "AI",
    "system": "System",
    "sys": "System",
    "zeit": "Zeit",
    "time": "Zeit",
    "clock": "Zeit",
    "uhr": "Zeit",
    "hinweis": "Hinweis",
    "notice": "Hinweis",
    "note": "Hinweis",
}

_STATUS_ALIASES: Final[Mapping[str, HdmiRuntimeStatus]] = {
    "waiting": HdmiRuntimeStatus.WAITING,
    "idle": HdmiRuntimeStatus.WAITING,
    "standby": HdmiRuntimeStatus.WAITING,
    "ready": HdmiRuntimeStatus.WAITING,
    "listening": HdmiRuntimeStatus.LISTENING,
    "recording": HdmiRuntimeStatus.LISTENING,
    "capturing": HdmiRuntimeStatus.LISTENING,
    "processing": HdmiRuntimeStatus.PROCESSING,
    "thinking": HdmiRuntimeStatus.PROCESSING,
    "working": HdmiRuntimeStatus.PROCESSING,
    "loading": HdmiRuntimeStatus.PROCESSING,
    "answering": HdmiRuntimeStatus.ANSWERING,
    "speaking": HdmiRuntimeStatus.ANSWERING,
    "talking": HdmiRuntimeStatus.ANSWERING,
    "responding": HdmiRuntimeStatus.ANSWERING,
    "printing": HdmiRuntimeStatus.PRINTING,
    "printer": HdmiRuntimeStatus.PRINTING,
    "error": HdmiRuntimeStatus.ERROR,
    "failed": HdmiRuntimeStatus.ERROR,
    "failure": HdmiRuntimeStatus.ERROR,
    "fault": HdmiRuntimeStatus.ERROR,
}

_VALUE_ALIASES: Final[Mapping[str, HdmiValueSemantic]] = {
    "?": HdmiValueSemantic.UNKNOWN,
    "unknown": HdmiValueSemantic.UNKNOWN,
    "unbekannt": HdmiValueSemantic.UNKNOWN,
    "ok": HdmiValueSemantic.OK,
    "okay": HdmiValueSemantic.OK,
    "healthy": HdmiValueSemantic.OK,
    "bereit": HdmiValueSemantic.READY,
    "ready": HdmiValueSemantic.READY,
    "online": HdmiValueSemantic.ONLINE,
    "verbunden": HdmiValueSemantic.ONLINE,
    "connected": HdmiValueSemantic.ONLINE,
    "offline": HdmiValueSemantic.OFFLINE,
    "disconnected": HdmiValueSemantic.OFFLINE,
    "fehlt": HdmiValueSemantic.MISSING,
    "missing": HdmiValueSemantic.MISSING,
    "not available": HdmiValueSemantic.MISSING,
    "not_available": HdmiValueSemantic.MISSING,
    "wartet": HdmiValueSemantic.WAITING,
    "waiting": HdmiValueSemantic.WAITING,
    "pending": HdmiValueSemantic.WAITING,
    "achtung": HdmiValueSemantic.WARNING,
    "warnung": HdmiValueSemantic.WARNING,
    "warning": HdmiValueSemantic.WARNING,
    "degraded": HdmiValueSemantic.WARNING,
    "fehler": HdmiValueSemantic.ERROR,
    "error": HdmiValueSemantic.ERROR,
    "failed": HdmiValueSemantic.ERROR,
    "warm": HdmiValueSemantic.WARM,
    "hot": HdmiValueSemantic.WARM,
    "status wird aktualisiert": HdmiValueSemantic.UPDATING,
    "status wird aktualisiert…": HdmiValueSemantic.UPDATING,
    "status not available": HdmiValueSemantic.UNAVAILABLE,
    "status unavailable": HdmiValueSemantic.UNAVAILABLE,
    "status nicht verfügbar": HdmiValueSemantic.UNAVAILABLE,
    "unavailable": HdmiValueSemantic.UNAVAILABLE,
    "updating": HdmiValueSemantic.UPDATING,
    "aktualisiert": HdmiValueSemantic.UPDATING,
    "aktualisierung": HdmiValueSemantic.UPDATING,
    "connecting": HdmiValueSemantic.CONNECTING,
    "verbinde": HdmiValueSemantic.CONNECTING,
    "verbinden": HdmiValueSemantic.CONNECTING,
}

_FIELD_SPECIFIC_DISPLAY: Final[Mapping[str, Mapping[HdmiValueSemantic, str]]] = {
    "Internet": {
        HdmiValueSemantic.OK: "Online",
        HdmiValueSemantic.READY: "Online",
        HdmiValueSemantic.ONLINE: "Online",
        HdmiValueSemantic.OFFLINE: "Offline",
        HdmiValueSemantic.MISSING: "Offline",
        HdmiValueSemantic.WAITING: "Waiting",
        HdmiValueSemantic.CONNECTING: "Connecting",
        HdmiValueSemantic.UPDATING: "Updating",
        HdmiValueSemantic.UNAVAILABLE: "Status unavailable",
        HdmiValueSemantic.UNKNOWN: "Unknown",
        HdmiValueSemantic.WARNING: "Warning",
        HdmiValueSemantic.ERROR: "Error",
    },
    "AI": {
        HdmiValueSemantic.OK: "Ready",
        HdmiValueSemantic.READY: "Ready",
        HdmiValueSemantic.ONLINE: "Ready",
        HdmiValueSemantic.OFFLINE: "Offline",
        HdmiValueSemantic.MISSING: "Missing",
        HdmiValueSemantic.WAITING: "Waiting",
        HdmiValueSemantic.CONNECTING: "Connecting",
        HdmiValueSemantic.UPDATING: "Updating",
        HdmiValueSemantic.UNAVAILABLE: "Status unavailable",
        HdmiValueSemantic.UNKNOWN: "Unknown",
        HdmiValueSemantic.WARNING: "Warning",
        HdmiValueSemantic.ERROR: "Error",
    },
    "System": {
        HdmiValueSemantic.OK: "OK",
        HdmiValueSemantic.READY: "OK",
        HdmiValueSemantic.ONLINE: "OK",
        HdmiValueSemantic.OFFLINE: "Offline",
        HdmiValueSemantic.MISSING: "Missing",
        HdmiValueSemantic.WAITING: "Waiting",
        HdmiValueSemantic.UPDATING: "Updating",
        HdmiValueSemantic.UNAVAILABLE: "Status unavailable",
        HdmiValueSemantic.UNKNOWN: "Unknown",
        HdmiValueSemantic.WARNING: "Warning",
        HdmiValueSemantic.ERROR: "Error",
        HdmiValueSemantic.WARM: "Warm",
    },
}

_GENERIC_DISPLAY: Final[Mapping[HdmiValueSemantic, str]] = {
    HdmiValueSemantic.OK: "Ready",
    HdmiValueSemantic.READY: "Ready",
    HdmiValueSemantic.ONLINE: "Online",
    HdmiValueSemantic.OFFLINE: "Offline",
    HdmiValueSemantic.MISSING: "Missing",
    HdmiValueSemantic.WAITING: "Waiting",
    HdmiValueSemantic.CONNECTING: "Connecting",
    HdmiValueSemantic.UPDATING: "Updating status",
    HdmiValueSemantic.UNAVAILABLE: "Status unavailable",
    HdmiValueSemantic.UNKNOWN: "Unknown",
    HdmiValueSemantic.WARNING: "Warning",
    HdmiValueSemantic.ERROR: "Error",
    HdmiValueSemantic.WARM: "Warm",
}

_STATUS_HEADLINES: Final[Mapping[HdmiRuntimeStatus, str]] = {
    HdmiRuntimeStatus.WAITING: "Waiting",
    HdmiRuntimeStatus.LISTENING: "Listening",
    HdmiRuntimeStatus.PROCESSING: "Thinking",
    HdmiRuntimeStatus.ANSWERING: "Speaking",
    HdmiRuntimeStatus.PRINTING: "Printing",
    HdmiRuntimeStatus.ERROR: "Check system",
}

_STATUS_HELPERS: Final[Mapping[HdmiRuntimeStatus, str]] = {
    HdmiRuntimeStatus.WAITING: "Press the green button and speak naturally.",
    HdmiRuntimeStatus.LISTENING: "Listening now. Speak at your own pace.",
    HdmiRuntimeStatus.PROCESSING: "Thinking for a moment.",
    HdmiRuntimeStatus.ANSWERING: "Speaking now.",
    HdmiRuntimeStatus.PRINTING: "Preparing the print.",
    HdmiRuntimeStatus.ERROR: "Please check the system in debug view.",
}

_STATUS_ACCENTS: Final[Mapping[HdmiRuntimeStatus, tuple[int, int, int]]] = {
    HdmiRuntimeStatus.WAITING: (90, 132, 196),
    HdmiRuntimeStatus.LISTENING: (36, 163, 130),
    HdmiRuntimeStatus.PROCESSING: (226, 164, 51),
    HdmiRuntimeStatus.ANSWERING: (82, 114, 222),
    HdmiRuntimeStatus.PRINTING: (197, 128, 35),
    HdmiRuntimeStatus.ERROR: (205, 89, 74),
}

_SEMANTIC_COLORS: Final[Mapping[HdmiValueSemantic, tuple[int, int, int]]] = {
    HdmiValueSemantic.OK: (40, 167, 117),
    HdmiValueSemantic.READY: (40, 167, 117),
    HdmiValueSemantic.ONLINE: (40, 167, 117),
    HdmiValueSemantic.OFFLINE: (205, 89, 74),
    HdmiValueSemantic.MISSING: (205, 89, 74),
    HdmiValueSemantic.ERROR: (205, 89, 74),
    HdmiValueSemantic.WARNING: (228, 152, 34),
    HdmiValueSemantic.WARM: (228, 152, 34),
    HdmiValueSemantic.WAITING: (130, 145, 166),
    HdmiValueSemantic.UNKNOWN: (130, 145, 166),
    HdmiValueSemantic.UPDATING: (130, 145, 166),
    HdmiValueSemantic.UNAVAILABLE: (130, 145, 166),
    HdmiValueSemantic.CONNECTING: (130, 145, 166),
}

_DEFAULT_ACCENT_COLOR: Final[tuple[int, int, int]] = (102, 126, 150)
_DEFAULT_VALUE_COLOR: Final[tuple[int, int, int]] = (90, 132, 196)


class _HdmiSceneTools(Protocol):
    """Text and sanitization helpers supplied by the active HDMI backend.

    Backend note: modern Pillow backends should implement measurement via
    textbbox()/textlength() rather than legacy size APIs.
    """

    def _font(self, size: int, *, bold: bool) -> object: ...

    def _text_width(self, draw: HdmiPanelDrawSurface, text: str, *, font: object | None = None) -> int: ...

    def _text_height(self, draw: HdmiPanelDrawSurface, *, font: object | None = None) -> int: ...

    def _truncate_text(
        self,
        draw: HdmiPanelDrawSurface,
        text: str,
        *,
        max_width: int,
        font: object | None = None,
    ) -> str: ...

    def _wrapped_lines(
        self,
        draw: HdmiPanelDrawSurface,
        lines: tuple[str, ...],
        *,
        max_width: int,
        font: object,
        max_lines: int,
    ) -> tuple[str, ...]: ...

    def _normalise_text(self, value: object, *, fallback: str) -> str: ...

    def _render_emoji_glyph(self, emoji: str, *, target_size: int) -> object | None: ...


@dataclass(frozen=True, slots=True)
class HdmiSceneCard:
    """Describe one status card in the right-hand HDMI status panel."""

    key: str
    label: str
    value: str
    accent: tuple[int, int, int]
    detail_lines: tuple[str, ...] = ()
    emphasis: float = 1.0
    column_span: int = 1
    row_span: int = 1


@dataclass(frozen=True, slots=True)
class HdmiHeaderModel:
    """Prepared content for the extended top HDMI status header."""

    brand: str
    state: str
    time_value: str
    system_value: str
    system_accent: tuple[int, int, int]
    debug_signals: tuple[HdmiHeaderSignalLike, ...] = ()


@dataclass(frozen=True, slots=True)
class HdmiStatusPanelModel:
    """Prepared future content for the right-hand HDMI reserve area."""

    eyebrow: str
    headline: str
    helper_text: str
    cards: tuple[HdmiSceneCard, ...]
    symbol: str | None = None
    accent: str = "neutral"
    prompt_mode: bool = False
    image_data_url: str | None = None


@dataclass(frozen=True, slots=True)
class HdmiNewsTickerModel:
    """Prepared content for the bottom HDMI news ticker."""

    label: str
    text: str


@dataclass(frozen=True, slots=True)
class _HdmiPromptModeLayout:
    """Describe one fitted prompt-mode text layout inside the reserve card."""

    headline_font: object
    body_font: object
    headline_lines: tuple[str, ...]
    body_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HdmiDefaultSceneLayout:
    """Geometry contract for the default Twinr HDMI scene."""

    header_box: tuple[int, int, int, int]
    face_box: tuple[int, int, int, int]
    panel_box: tuple[int, int, int, int]
    ticker_box: tuple[int, int, int, int]
    ticker_reserved: bool
    compact_panel: bool


@dataclass(frozen=True, slots=True)
class HdmiDefaultScene:
    """Full scene state for one rendered HDMI frame."""

    status: str
    animation_frame: int
    layout: HdmiDefaultSceneLayout
    header: HdmiHeaderModel
    panel: HdmiStatusPanelModel
    ticker: HdmiNewsTickerModel | None = None
    face_cue: HdmiFaceCueLike | None = None
    emoji_cue: HdmiEmojiCueLike | None = None
    reserve_bus: HdmiReserveBusLike | None = None
    ambient_moment: HdmiAmbientMomentLike | None = None
    presentation_graph: HdmiPresentationGraphLike | None = None


@lru_cache(maxsize=4096)
def _sanitize_text(raw: str, fallback: str, max_chars: int) -> str:
    """Normalize, de-control, and clip text for deterministic HDMI rendering."""

    text = unicodedata.normalize("NFKC", raw)
    filtered: list[str] = []
    for char in text:
        if char in _BIDI_FORMATTING_CHARS:
            continue

        category = unicodedata.category(char)
        if category.startswith("C"):
            if char in {"\t", "\n", "\r"}:
                filtered.append(" ")
            continue

        filtered.append(char)

    compact = " ".join("".join(filtered).split())
    if not compact:
        compact = fallback

    if max_chars > 0 and len(compact) > max_chars:
        if max_chars == 1:
            return "…"
        compact = compact[: max_chars - 1].rstrip() + "…"

    return compact


def _safe_display_text(value: object, *, fallback: str, max_chars: int) -> str:
    """Return sanitized HDMI-safe text from arbitrary runtime input."""

    raw = "" if value is None else str(value)
    return _sanitize_text(raw, str(fallback), max_chars)


def _normalise_for_display(
    normalise_text: Callable[..., str] | None,
    value: object,
    *,
    fallback: str,
    max_chars: int,
) -> str:
    """Run backend normalization if available, then apply HDMI-safe normalization."""

    raw_fallback = str(fallback)
    candidate: object = value
    if normalise_text is not None:
        try:
            candidate = normalise_text(value, fallback=raw_fallback)
        except Exception:
            candidate = value if value is not None else raw_fallback
    return _safe_display_text(candidate, fallback=raw_fallback, max_chars=max_chars)


@lru_cache(maxsize=2048)
def _lookup_key_from_text(text: str) -> str:
    """Return a Unicode-normalized, caseless key for alias lookup tables."""

    return _sanitize_text(text, "", _MAX_LOOKUP_TEXT_CHARS).casefold()


def _lookup_key(value: object) -> str:
    return _lookup_key_from_text("" if value is None else str(value))


def _canonical_field_name(
    field_name: object,
    normalise_text: Callable[..., str] | None = None,
) -> str:
    """Map field aliases to stable HDMI labels while preserving unknown labels."""

    cleaned = _normalise_for_display(
        normalise_text,
        field_name,
        fallback="Info",
        max_chars=_MAX_FIELD_LABEL_CHARS,
    )
    return _FIELD_NAME_ALIASES.get(_lookup_key(cleaned), cleaned)


def _canonical_status(status: object) -> HdmiRuntimeStatus | None:
    """Return the canonical runtime status for a raw status string, if known."""

    return _STATUS_ALIASES.get(_lookup_key(status))


def _value_semantic(value: object) -> HdmiValueSemantic | None:
    """Return the canonical semantic class for a raw state value, if known."""

    return _VALUE_ALIASES.get(_lookup_key(value))


def _prepare_state_fields(
    normalise_text: Callable[..., str],
    state_fields: DisplayStateFields,
) -> DisplayStateFields:
    """Normalize, deduplicate, and order HDMI state fields for stable rendering."""

    deduped: dict[str, str] = {}
    discovery_order: dict[str, int] = {}

    for item in state_fields:
        if not isinstance(item, tuple) or len(item) != 2:
            continue

        raw_name, raw_value = item
        name = _canonical_field_name(raw_name, normalise_text)
        value = _normalise_for_display(
            normalise_text,
            raw_value,
            fallback="--",
            max_chars=_MAX_FIELD_VALUE_CHARS,
        )

        if name not in discovery_order:
            discovery_order[name] = len(discovery_order)
        deduped[name] = value

    ordered_items = list(deduped.items())
    ordered_items.sort(
        key=lambda item: (
            _STATE_CARD_ORDER_INDEX.get(item[0], len(_STATE_CARD_ORDER_INDEX)),
            discovery_order[item[0]],
        )
    )
    return tuple(ordered_items)


def _humanize_status_fallback(status: object) -> str:
    """Turn an unknown machine status into calm title-cased HDMI copy."""

    candidate = _safe_display_text(status, fallback="Unknown", max_chars=_MAX_HEADLINE_CHARS)
    candidate = candidate.replace("_", " ").replace("-", " ")
    candidate = " ".join(candidate.split())
    return candidate.title() if candidate else "Unknown"


def order_state_fields(
    normalise_text: Callable[..., str],
    state_fields: DisplayStateFields,
    details: tuple[str, ...],
) -> DisplayStateFields:
    """Keep state fields stable for rendering or synthesize short detail rows."""

    prepared_fields = _prepare_state_fields(normalise_text, state_fields)
    if prepared_fields:
        return prepared_fields

    synthesized: list[tuple[str, str]] = []
    for detail in details:
        cleaned_detail = _normalise_for_display(
            normalise_text,
            detail,
            fallback="",
            max_chars=_MAX_DETAIL_VALUE_CHARS,
        )
        if not cleaned_detail:
            continue

        synthesized.append((f"Info {len(synthesized) + 1}", cleaned_detail))
        if len(synthesized) >= _DETAIL_MAX_LINES:
            break

    return tuple(synthesized)


def state_field_value(
    normalise_text: Callable[..., str],
    state_fields: DisplayStateFields,
    name: str | tuple[str, ...],
    *,
    fallback: str = "--",
) -> str:
    """Return one display state-field value with stable fallback behavior."""

    fields_by_name = dict(_prepare_state_fields(normalise_text, state_fields))
    names = (name,) if isinstance(name, str) else name
    for field_name in names:
        canonical_name = _canonical_field_name(field_name, normalise_text)
        if canonical_name in fields_by_name:
            return fields_by_name[canonical_name]

    return _safe_display_text(fallback, fallback=fallback, max_chars=_MAX_FIELD_VALUE_CHARS)


def display_state_value(normalise_text: Callable[..., str], field_name: str, value: str) -> str:
    """Translate mixed-language runtime values into stable HDMI copy."""

    canonical_field = _canonical_field_name(field_name, normalise_text)
    compact = _normalise_for_display(
        normalise_text,
        value,
        fallback="--",
        max_chars=_MAX_FIELD_VALUE_CHARS,
    )

    if canonical_field == "Zeit":
        if compact == "--":
            return "--:--"
        return _safe_display_text(compact, fallback="--:--", max_chars=_MAX_TIME_CHARS)

    semantic = _value_semantic(compact)
    if semantic is None:
        return compact

    field_mapping = _FIELD_SPECIFIC_DISPLAY.get(canonical_field)
    if field_mapping is not None:
        mapped = field_mapping.get(semantic)
        if mapped is not None:
            return mapped

    return _GENERIC_DISPLAY.get(semantic, compact)


def status_headline(normalise_text: Callable[..., str], status: str, *, fallback: str | None) -> str:
    """Return the senior-facing main headline for one runtime status."""

    canonical_status = _canonical_status(status)
    if canonical_status is not None:
        return _STATUS_HEADLINES[canonical_status]

    fallback_text = _normalise_for_display(
        normalise_text,
        fallback,
        fallback="",
        max_chars=_MAX_HEADLINE_CHARS,
    )
    if fallback_text:
        return fallback_text

    return _humanize_status_fallback(status)


def status_helper_text(status: str) -> str:
    """Return a short, calm helper sentence for the active status."""

    canonical_status = _canonical_status(status)
    if canonical_status is not None:
        return _STATUS_HELPERS[canonical_status]
    return "Twinr is updating the current status."


def status_accent_color(status: str) -> tuple[int, int, int]:
    """Return the accent color associated with one runtime status."""

    canonical_status = _canonical_status(status)
    if canonical_status is None:
        return _DEFAULT_ACCENT_COLOR
    return _STATUS_ACCENTS[canonical_status]


def state_value_color(normalise_text: Callable[..., str], value: str) -> tuple[int, int, int]:
    """Return the accent color for a specific panel-card value."""

    normalized_value = _normalise_for_display(
        normalise_text,
        value,
        fallback="",
        max_chars=_MAX_FIELD_VALUE_CHARS,
    )
    semantic = _value_semantic(normalized_value)
    if semantic is None:
        return _DEFAULT_VALUE_COLOR
    return _SEMANTIC_COLORS.get(semantic, _DEFAULT_VALUE_COLOR)


def time_value(state_fields: DisplayStateFields) -> str:
    """Return the visible clock value for the header."""

    for item in reversed(state_fields):
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        name, value = item
        if _canonical_field_name(name) == "Zeit":
            return _safe_display_text(value, fallback="--:--", max_chars=_MAX_TIME_CHARS)
    return "--:--"