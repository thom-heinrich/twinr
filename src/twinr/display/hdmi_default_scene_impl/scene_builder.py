# mypy: disable-error-code="attr-defined"
# CHANGELOG: 2026-03-28
# BUG-1: build_scene accepted helper_text but discarded it entirely. The default panel now
#        renders helper_text/headline fallbacks whenever no reserve cue owns the panel.
# BUG-2: blank or missing "System" telemetry no longer forces a false red ERROR banner.
#        The header badge now resolves OK/WARN/ERROR and falls back to status when telemetry is absent.
# BUG-3: non-positive framebuffer dimensions could generate invalid/inverted boxes during
#        startup or mode switches. Dimensions are now sanitized before layout math runs.
# SEC-1: service_connect_cue.qr_image_data_url is now raster-only, strictly base64-validated,
#        signature-checked, and size-limited to block SVG/script payloads and memory-amplification attacks.
# IMP-1: layout geometry and lazy imports are cached to reduce per-frame work on Raspberry Pi 4.
# IMP-2: scene text is length-bounded and canonicalized so pathological payloads do not tank readability
#        or blow up downstream rendering cost on large displays.
"""Scene-building helpers for the default HDMI scene renderer."""

from __future__ import annotations

import base64
import binascii
from datetime import datetime
from functools import lru_cache
from importlib import import_module
from typing import Any, cast

from .models import (
    _DEFAULT_SCENE_SHOW_TICKER,
    HdmiDefaultScene,
    HdmiDefaultSceneLayout,
    HdmiHeaderModel,
    HdmiNewsTickerModel,
    HdmiStatusPanelModel,
    state_field_value,
    status_headline,
    time_value,
)
from .typing_contracts import (
    HdmiAmbientImpulseCueLike,
    HdmiAmbientMomentLike,
    HdmiEmojiCueLike,
    HdmiFaceCueLike,
    HdmiPresentationGraphLike,
    HdmiReserveBusLike,
    HdmiServiceConnectCueLike,
)

_SCENE_HEADLINE_MAX_CHARS = 96
_SCENE_HELPER_MAX_CHARS = 240
_SCENE_EYEBROW_MAX_CHARS = 40
_SCENE_TICKER_MAX_CHARS = 320
_SCENE_SYSTEM_BADGE_MAX_CHARS = 24

_QR_DATA_URL_MAX_CHARS = 700_000
_QR_IMAGE_MAX_BYTES = 512_000
_ALLOWED_QR_IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})

_HEALTHY_SYSTEM_STATES = frozenset(
    {
        "ok",
        "healthy",
        "ready",
        "online",
        "connected",
        "available",
        "up",
        "nominal",
    }
)
_WARNING_SYSTEM_STATES = frozenset(
    {
        "achtung",
        "warn",
        "warnung",
        "warning",
        "degraded",
        "starting",
        "booting",
        "initializing",
        "initialising",
        "connecting",
        "syncing",
        "busy",
    }
)
_ERROR_SYSTEM_STATES = frozenset(
    {
        "error",
        "err",
        "fehler",
        "failed",
        "fail",
        "fault",
        "offline",
        "down",
        "critical",
        "unavailable",
    }
)
_OK_ACCENT = (116, 242, 170)
_WARN_ACCENT = (255, 208, 102)
_ERROR_ACCENT = (255, 134, 110)


@lru_cache(maxsize=1)
def _display_runtime_symbols() -> tuple[Any, Any, Any, Any, Any]:
    ambient_module = import_module("twinr.display.ambient_impulse_cues")
    emoji_module = import_module("twinr.display.emoji_cues")
    reserve_bus_module = import_module("twinr.display.reserve_bus")
    service_module = import_module("twinr.display.service_connect_cues")
    return (
        getattr(ambient_module, "DisplayAmbientImpulseCue"),
        getattr(emoji_module, "DisplayEmojiCue"),
        getattr(reserve_bus_module, "DisplayReserveBusState"),
        getattr(reserve_bus_module, "resolve_display_reserve_bus"),
        getattr(service_module, "DisplayServiceConnectCue"),
    )


@lru_cache(maxsize=1)
def _presentation_scene_graph_builder_cls() -> Any:
    presentation_module = import_module("twinr.display.hdmi_presentation_graph")
    return getattr(presentation_module, "HdmiPresentationSceneGraphBuilder")


class HdmiSceneBuilderMixin:
    """Build normalized default-scene models independent of transport backends."""

    def build_scene(
        self,
        *,
        width: int,
        height: int,
        status: str,
        headline: str,
        helper_text: str,
        state_fields,
        debug_signals=(),
        animation_frame: int = 0,
        ticker_text: str | None = None,
        face_cue: HdmiFaceCueLike | None = None,
        emoji_cue: HdmiEmojiCueLike | None = None,
        ambient_impulse_cue: HdmiAmbientImpulseCueLike | None = None,
        service_connect_cue: HdmiServiceConnectCueLike | None = None,
        presentation_cue: object | None = None,
        presentation_now: datetime | None = None,
        ambient_now: datetime | None = None,
    ) -> HdmiDefaultScene:
        """Build the scene model so layout and content can evolve independently."""

        width, height = self._sanitize_dimensions(width=width, height=height)
        debug_signals_snapshot = self._snapshot_signals(debug_signals)
        helper_text_compact = self._scene_text(
            helper_text,
            fallback="",
            max_chars=_SCENE_HELPER_MAX_CHARS,
        )
        ticker = self._build_ticker_model(ticker_text) if _DEFAULT_SCENE_SHOW_TICKER else None
        layout = self._layout_for_size(
            width=width,
            height=height,
            reserve_ticker=ticker is not None,
            reserve_card_active=(
                ambient_impulse_cue is not None
                or service_connect_cue is not None
            )
            and presentation_cue is None,
            header_debug_signal_count=len(debug_signals_snapshot),
        )
        presentation_graph = self._presentation_graph(
            cue=presentation_cue,
            layout=layout,
            now=presentation_now,
        )
        ambient_moment = self._ambient_moment(
            status=status,
            face_cue=face_cue,
            presentation_graph=presentation_graph,
            ambient_now=ambient_now,
        )
        resolved_face_cue = face_cue
        if resolved_face_cue is None and presentation_graph is not None:
            resolved_face_cue = presentation_graph.face_cue
        if resolved_face_cue is None and ambient_moment is not None:
            resolved_face_cue = ambient_moment.face_cue
        resolved_face_cue = self._effective_face_cue(status=status, face_cue=resolved_face_cue)

        (
            DisplayAmbientImpulseCue,
            DisplayEmojiCue,
            DisplayReserveBusState,
            resolve_display_reserve_bus,
            DisplayServiceConnectCue,
        ) = _display_runtime_symbols()

        reserve_bus: HdmiReserveBusLike = DisplayReserveBusState.empty(reason="presentation_surface_owned")
        if presentation_graph is None:
            reserve_bus = resolve_display_reserve_bus(
                service_connect_cue=cast(DisplayServiceConnectCue | None, service_connect_cue),
                emoji_cue=cast(DisplayEmojiCue | None, emoji_cue),
                ambient_impulse_cue=cast(DisplayAmbientImpulseCue | None, ambient_impulse_cue),
            )
        return HdmiDefaultScene(
            status=status,
            animation_frame=animation_frame,
            layout=layout,
            header=self._build_header_model(
                status=status,
                headline=headline,
                state_fields=state_fields,
                debug_signals=debug_signals_snapshot,
            ),
            panel=self._build_panel_model(
                reserve_bus=reserve_bus,
                status=status,
                headline="" if presentation_graph is not None else headline,
                helper_text="" if presentation_graph is not None else helper_text_compact,
            ),
            ticker=None if presentation_graph is not None else ticker,
            face_cue=resolved_face_cue,
            emoji_cue=reserve_bus.emoji_cue,
            reserve_bus=reserve_bus,
            ambient_moment=ambient_moment,
            presentation_graph=presentation_graph,
        )

    def _effective_face_cue(
        self,
        *,
        status: str,
        face_cue: HdmiFaceCueLike | None,
    ) -> HdmiFaceCueLike | None:
        """Keep status-owned expressions authoritative while allowing local gaze-follow."""

        if face_cue is None:
            return None
        if status != "error":
            return face_cue
        from twinr.display.face_cues import DisplayFaceCue

        return DisplayFaceCue(
            source=face_cue.source,
            updated_at=face_cue.updated_at,
            expires_at=face_cue.expires_at,
            gaze_x=face_cue.gaze_x,
            gaze_y=face_cue.gaze_y,
            head_dx=face_cue.head_dx,
            head_dy=face_cue.head_dy,
            blink=face_cue.blink,
        )

    def _layout_for_size(
        self,
        *,
        width: int,
        height: int,
        reserve_ticker: bool,
        reserve_card_active: bool = False,
        header_debug_signal_count: int = 0,
    ) -> HdmiDefaultSceneLayout:
        del header_debug_signal_count
        header_box, face_box, panel_box, ticker_box, compact_panel = self._layout_geometry(
            width=width,
            height=height,
            reserve_ticker=reserve_ticker,
            reserve_card_active=reserve_card_active,
        )
        return HdmiDefaultSceneLayout(
            header_box=header_box,
            face_box=face_box,
            panel_box=panel_box,
            ticker_box=ticker_box,
            ticker_reserved=reserve_ticker,
            compact_panel=compact_panel,
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def _layout_geometry(
        *,
        width: int,
        height: int,
        reserve_ticker: bool,
        reserve_card_active: bool,
    ) -> tuple[
        tuple[int, int, int, int],
        tuple[int, int, int, int],
        tuple[int, int, int, int],
        tuple[int, int, int, int],
        bool,
    ]:
        if width < 560 or height < 320:
            header_bottom = 102
            header_box = (12, 10, width - 12, header_bottom)
            if reserve_ticker:
                ticker_box = (12, height - 54, width - 12, height - 12)
                content_bottom = ticker_box[1] - 10
            else:
                ticker_box = (12, height - 12, width - 12, height - 12)
                content_bottom = height - 12
            face_box_width = max(94, min(126, int(width * 0.33)))
            content_top = 110
            face_box = (18, content_top, 18 + face_box_width, content_bottom)
            panel_box = (face_box[2] + 14, content_top - 2, width - 18, content_bottom)
        else:
            content_left, content_right = HdmiSceneBuilderMixin._content_frame_edges(width)
            content_width = content_right - content_left
            widescreen = width >= 1280
            compact_hdmi = not widescreen and width >= 720
            header_bottom = 112 if widescreen else 108 if compact_hdmi else 118
            content_top = 130 if widescreen else 126 if compact_hdmi else 136
            panel_gap = 34 if widescreen else 30 if compact_hdmi else 24
            if widescreen:
                panel_width = (
                    min(700, max(468, int(content_width * 0.45)))
                    if reserve_card_active
                    else min(620, max(420, int(content_width * 0.40)))
                )
            elif compact_hdmi:
                panel_width = (
                    min(430, max(388, int(content_width * 0.52)))
                    if reserve_card_active
                    else min(336, max(316, int(content_width * 0.43)))
                )
            else:
                panel_width = (
                    min(428, max(376, int(content_width * 0.50)))
                    if reserve_card_active
                    else min(392, max(348, int(content_width * 0.46)))
                )
            header_box = (content_left, 18, content_right, header_bottom)
            if reserve_ticker:
                ticker_box = (content_left, height - 74, content_right, height - 24)
                content_bottom = ticker_box[1] - 16
            else:
                ticker_box = (content_left, height - 24, content_right, height - 24)
                content_bottom = height - 24
            panel_box = (content_right - panel_width, content_top, content_right, content_bottom)
            face_box = (content_left + 8, content_top, panel_box[0] - panel_gap, content_bottom)
        compact_panel = HdmiSceneBuilderMixin._should_use_compact_panel(
            panel_box=panel_box,
            reserve_ticker=reserve_ticker,
        )
        return (header_box, face_box, panel_box, ticker_box, compact_panel)

    @staticmethod
    @lru_cache(maxsize=64)
    def _content_frame_edges(width: int) -> tuple[int, int]:
        """Return centered content bounds so widescreen HDMI does not stretch infinitely."""

        if width >= 1280:
            outer_padding = max(48, min(140, width // 20))
            content_width = min(width - (outer_padding * 2), 1440)
        else:
            outer_padding = 24
            content_width = width - (outer_padding * 2)
        left = max(outer_padding, (width - content_width) // 2)
        return (left, width - left)

    @staticmethod
    def _should_use_compact_panel(
        *,
        panel_box: tuple[int, int, int, int],
        reserve_ticker: bool,
    ) -> bool:
        """Return whether the status panel must collapse to its compact card mode."""

        panel_width = panel_box[2] - panel_box[0]
        panel_height = panel_box[3] - panel_box[1]
        if panel_width < 320 or panel_height < 240:
            return True
        if reserve_ticker and panel_height < 320:
            return True
        return False

    def _build_header_model(
        self,
        *,
        status: str,
        headline: str,
        state_fields,
        debug_signals=(),
    ) -> HdmiHeaderModel:
        normalise_text = self.tools._normalise_text
        system_raw = state_field_value(normalise_text, state_fields, "System")
        system_value, system_accent = self._resolve_system_badge(
            status=status,
            system_raw=system_raw,
        )
        return HdmiHeaderModel(
            brand="TWINR",
            state=self._scene_text(
                headline,
                fallback=status_headline(normalise_text, status, fallback=headline),
                max_chars=_SCENE_HEADLINE_MAX_CHARS,
            ).upper(),
            time_value=time_value(state_fields),
            system_value=system_value,
            system_accent=system_accent,
            debug_signals=debug_signals,
        )

    def _build_panel_model(
        self,
        *,
        reserve_bus: HdmiReserveBusLike,
        status: str = "waiting",
        headline: str = "",
        helper_text: str = "",
    ) -> HdmiStatusPanelModel:
        """Build the optional right-hand reserve card from the active ambient cue."""

        service_connect_cue = reserve_bus.service_connect_cue
        if service_connect_cue is not None:
            return HdmiStatusPanelModel(
                eyebrow=self._scene_text(
                    service_connect_cue.service_label,
                    fallback="",
                    max_chars=_SCENE_EYEBROW_MAX_CHARS,
                ).upper(),
                headline=self._scene_text(
                    service_connect_cue.summary,
                    fallback="",
                    max_chars=_SCENE_HEADLINE_MAX_CHARS,
                ),
                helper_text=self._scene_text(
                    service_connect_cue.detail,
                    fallback="",
                    max_chars=_SCENE_HELPER_MAX_CHARS,
                ),
                cards=(),
                accent=service_connect_cue.accent,
                image_data_url=self._sanitize_qr_image_data_url(service_connect_cue.qr_image_data_url),
            )
        ambient_impulse_cue = reserve_bus.ambient_impulse_cue
        if ambient_impulse_cue is not None:
            eyebrow = self._scene_text(
                ambient_impulse_cue.eyebrow,
                fallback="",
                max_chars=_SCENE_EYEBROW_MAX_CHARS,
            ).upper()
            return HdmiStatusPanelModel(
                eyebrow=eyebrow,
                headline=self._scene_text(
                    ambient_impulse_cue.headline,
                    fallback="",
                    max_chars=_SCENE_HEADLINE_MAX_CHARS,
                ),
                helper_text=self._scene_text(
                    ambient_impulse_cue.body,
                    fallback="",
                    max_chars=_SCENE_HELPER_MAX_CHARS,
                ),
                cards=(),
                symbol=ambient_impulse_cue.glyph(),
                accent=ambient_impulse_cue.accent,
                prompt_mode=not eyebrow,
            )
        return HdmiStatusPanelModel(
            eyebrow="",
            headline="",
            helper_text="",
            cards=(),
        )

    def _build_ticker_model(self, ticker_text: str | None) -> HdmiNewsTickerModel | None:
        """Return the optional bottom-bar ticker model."""

        compact = self._scene_text(
            ticker_text,
            fallback="",
            max_chars=_SCENE_TICKER_MAX_CHARS,
        )
        if not compact:
            return None
        return HdmiNewsTickerModel(label="NEWS", text=compact)

    def _presentation_graph(
        self,
        *,
        cue: object | None,
        layout: HdmiDefaultSceneLayout,
        now: datetime | None,
    ) -> HdmiPresentationGraphLike | None:
        """Resolve the optional presentation cue into a dedicated scene graph."""

        from twinr.display.presentation_cues import DisplayPresentationCue

        builder_cls = _presentation_scene_graph_builder_cls()
        builder = builder_cls()
        return builder.build(
            cue=cast(DisplayPresentationCue | None, cue),
            face_box=layout.face_box,
            panel_box=layout.panel_box,
            now=now,
        )

    def _ambient_moment(
        self,
        *,
        status: str,
        face_cue: HdmiFaceCueLike | None,
        presentation_graph: HdmiPresentationGraphLike | None,
        ambient_now: datetime | None,
    ) -> HdmiAmbientMomentLike | None:
        """Resolve the optional idle-only ambient moment for the waiting face."""

        return self.ambient_director.resolve(
            status=status,
            now=ambient_now,
            face_cue_active=face_cue is not None,
            presentation_active=presentation_graph is not None,
        )

    def _scene_text(
        self,
        value: object | None,
        *,
        fallback: str,
        max_chars: int,
    ) -> str:
        compact = self.tools._normalise_text(value, fallback=fallback)
        return self._truncate_text(compact, max_chars=max_chars)

    @staticmethod
    def _truncate_text(value: str, *, max_chars: int) -> str:
        if max_chars < 1:
            return ""
        if len(value) <= max_chars:
            return value
        if max_chars == 1:
            return "…"
        return value[: max_chars - 1].rstrip() + "…"

    @staticmethod
    def _snapshot_signals(debug_signals) -> tuple[Any, ...]:
        if debug_signals is None:
            return ()
        if isinstance(debug_signals, tuple):
            return debug_signals
        try:
            return tuple(debug_signals)
        except TypeError:
            return (debug_signals,)

    @staticmethod
    def _sanitize_dimensions(*, width: object, height: object) -> tuple[int, int]:
        width_value = HdmiSceneBuilderMixin._positive_int_or_default(width, default=640)
        height_value = HdmiSceneBuilderMixin._positive_int_or_default(height, default=360)
        return (width_value, height_value)

    @staticmethod
    def _positive_int_or_default(value: object, *, default: int) -> int:
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return default
        return coerced if coerced > 0 else default

    def _resolve_system_badge(
        self,
        *,
        status: str,
        system_raw: object | None,
    ) -> tuple[str, tuple[int, int, int]]:
        system_value = self._scene_text(
            system_raw,
            fallback="",
            max_chars=_SCENE_SYSTEM_BADGE_MAX_CHARS,
        ).lower()
        if system_value in _HEALTHY_SYSTEM_STATES:
            return ("OK", _OK_ACCENT)
        # BREAKING: system_value may now be "WARN" for degraded or transitional telemetry
        # instead of coercing every non-"ok" value into a false red "ERROR" badge.
        if system_value in _WARNING_SYSTEM_STATES:
            return ("WARN", _WARN_ACCENT)
        if system_value in _ERROR_SYSTEM_STATES:
            return ("ERROR", _ERROR_ACCENT)

        status_value = self._scene_text(
            status,
            fallback="",
            max_chars=_SCENE_SYSTEM_BADGE_MAX_CHARS,
        ).lower()
        if not system_value:
            if status_value in {"error", "failed", "fault", "offline", "down"}:
                return ("ERROR", _ERROR_ACCENT)
            if status_value in {"warn", "warning", "degraded", "booting", "starting", "connecting"}:
                return ("WARN", _WARN_ACCENT)
            return ("OK", _OK_ACCENT)
        return ("ERROR", _ERROR_ACCENT)

    def _sanitize_qr_image_data_url(self, data_url: str | None) -> str | None:
        if not data_url:
            return None
        candidate = data_url.strip()
        if not candidate or len(candidate) > _QR_DATA_URL_MAX_CHARS:
            return None
        prefix, separator, payload = candidate.partition(",")
        if not separator or not prefix.lower().startswith("data:"):
            return None

        meta = prefix[5:]
        parts = [part.strip().lower() for part in meta.split(";") if part.strip()]
        if not parts:
            return None

        mime_type = parts[0]
        # BREAKING: only raster image data URLs are accepted here. SVG and other active/image-capable
        # payloads are rejected because this field can cross trust boundaries into HTML/SVG-capable renderers.
        if mime_type not in _ALLOWED_QR_IMAGE_MIME_TYPES:
            return None
        if "base64" not in parts[1:]:
            return None

        try:
            decoded = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            return None
        if not decoded or len(decoded) > _QR_IMAGE_MAX_BYTES:
            return None
        if not self._matches_image_signature(mime_type=mime_type, data=decoded):
            return None

        canonical_payload = base64.b64encode(decoded).decode("ascii")
        return f"data:{mime_type};base64,{canonical_payload}"

    @staticmethod
    def _matches_image_signature(*, mime_type: str, data: bytes) -> bool:
        if mime_type == "image/png":
            return data.startswith(b"\x89PNG\r\n\x1a\n")
        if mime_type == "image/jpeg":
            return data.startswith(b"\xff\xd8\xff")
        if mime_type == "image/webp":
            return len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP"
        return False
