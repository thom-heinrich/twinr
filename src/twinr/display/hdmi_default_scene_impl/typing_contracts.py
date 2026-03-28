"""Local typing contracts for the decomposed HDMI default scene.

These protocols intentionally mirror the typed subset of Pillow's drawing and
image APIs used by the HDMI default scene. The goal is to keep structural
compatibility with modern typed Pillow surfaces while preserving lightweight
local contracts for the renderer graph.
"""

# CHANGELOG: 2026-03-28
# BUG-1: Leading draw/image parameters used Pillow-incompatible keyword names
#        ("box", "position", "image"), so type-checked keyword calls could still
#        crash at runtime on real Pillow objects. Those parameters are now
#        positional-only across the surface protocols.
# BUG-2: Surface methods were typed mostly with "object", which made the
#        contracts too imprecise for 2026 typed Pillow and too weak to catch
#        malformed geometry/color/font payloads during static analysis.
# SEC-1: No practically exploitable security issue was identified in this
#        declarative typing-only module; no security fix was required here.
# IMP-1: Added Pillow-aligned local type aliases for geometry, colors, fonts,
#        image sources, and paste boxes.
# IMP-2: Made the public typing surface explicit and more tool-friendly via
#        __all__, wider read-only sequence returns where appropriate, and
#        PathLike support for presentation image paths.

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

if TYPE_CHECKING:
    from PIL import Image as _PILImageModule
    from PIL import ImageFont as _PILImageFontModule
else:  # pragma: no cover - runtime fallback for typing-only consumers
    try:
        from PIL import Image as _PILImageModule
        from PIL import ImageFont as _PILImageFontModule
    except ImportError:  # pragma: no cover
        _PILImageModule = None  # type: ignore[assignment]
        _PILImageFontModule = None  # type: ignore[assignment]

InkLike: TypeAlias = float | tuple[int, ...] | str
CoordsLike: TypeAlias = Sequence[float] | Sequence[Sequence[float]]
PointLike: TypeAlias = tuple[float, float]
RoundedCornersLike: TypeAlias = tuple[bool, bool, bool, bool]
BoxQuadLike: TypeAlias = tuple[int, int, int, int]
RgbTripletLike: TypeAlias = tuple[int, int, int]
ImagePathLike: TypeAlias = str | PathLike[str]

if _PILImageModule is not None and _PILImageFontModule is not None:
    ImageLike: TypeAlias = _PILImageModule.Image
    FontLike: TypeAlias = (
        _PILImageFontModule.ImageFont
        | _PILImageFontModule.FreeTypeFont
        | _PILImageFontModule.TransposedFont
    )
else:
    ImageLike: TypeAlias = object
    FontLike: TypeAlias = object

PasteSourceLike: TypeAlias = ImageLike | str | float | tuple[float, ...]
PasteBoxLike: TypeAlias = ImageLike | tuple[int, int] | BoxQuadLike | None

__all__: Final[tuple[str, ...]] = (
    "InkLike",
    "CoordsLike",
    "PointLike",
    "RoundedCornersLike",
    "BoxQuadLike",
    "RgbTripletLike",
    "ImagePathLike",
    "ImageLike",
    "FontLike",
    "PasteSourceLike",
    "PasteBoxLike",
    "HdmiPanelDrawSurface",
    "HdmiCanvasDrawSurface",
    "HdmiImageSurface",
    "HdmiHeaderSignalLike",
    "HdmiEmojiCueLike",
    "HdmiFaceCueLike",
    "HdmiAmbientImpulseCueLike",
    "HdmiServiceConnectCueLike",
    "HdmiAmbientMomentLike",
    "HdmiPresentationNodeLike",
    "HdmiPresentationGraphLike",
    "HdmiReserveBusLike",
)


# BREAKING: The leading geometry/source parameters in the surface protocols are
# now positional-only. This prevents the contracts from falsely blessing
# keyword calls like box=..., position=..., or image=..., which do not match
# Pillow's public API names and can raise TypeError at runtime on real surfaces.
class HdmiPanelDrawSurface(Protocol):
    """Describe the text-and-card drawing methods used by HDMI panel renderers."""

    def rounded_rectangle(
        self,
        box: CoordsLike,
        /,
        radius: float,
        fill: InkLike | None = None,
        outline: InkLike | None = None,
        width: int = 1,
        *,
        corners: RoundedCornersLike | None = None,
    ) -> None: ...

    def text(
        self,
        position: PointLike,
        text: str,
        /,
        fill: InkLike | None = None,
        font: FontLike | None = None,
    ) -> None: ...


class HdmiCanvasDrawSurface(HdmiPanelDrawSurface, Protocol):
    """Describe the broader shape-drawing surface used by the full HDMI canvas."""

    def rectangle(
        self,
        box: CoordsLike,
        /,
        fill: InkLike | None = None,
        outline: InkLike | None = None,
        width: int = 1,
    ) -> None: ...

    def line(
        self,
        xy: CoordsLike,
        /,
        fill: InkLike | None = None,
        width: int = 0,
    ) -> None: ...

    def ellipse(
        self,
        box: CoordsLike,
        /,
        fill: InkLike | None = None,
        outline: InkLike | None = None,
        width: int = 1,
    ) -> None: ...

    def arc(
        self,
        box: CoordsLike,
        /,
        start: float,
        end: float,
        fill: InkLike | None = None,
        width: int = 1,
    ) -> None: ...

    def polygon(
        self,
        xy: CoordsLike,
        /,
        fill: InkLike | None = None,
        outline: InkLike | None = None,
        width: int = 1,
    ) -> None: ...


class HdmiImageSurface(Protocol):
    """Describe the bounded image API used by QR and presentation overlays."""

    def paste(
        self,
        image: PasteSourceLike,
        /,
        box: PasteBoxLike = None,
        mask: ImageLike | None = None,
    ) -> None: ...


class HdmiHeaderSignalLike(Protocol):
    """Describe the normalized signal-pill fields used by the HDMI header."""

    @property
    def key(self) -> str: ...

    @property
    def label(self) -> str: ...

    @property
    def accent(self) -> str: ...

    @property
    def priority(self) -> int: ...


class HdmiEmojiCueLike(Protocol):
    """Describe the minimal emoji cue API needed by the reserve panel."""

    @property
    def accent(self) -> str: ...

    def glyph(self) -> str: ...


class HdmiFaceCueLike(Protocol):
    """Describe the minimal face-cue fields used by HDMI face rendering."""

    @property
    def source(self) -> str: ...

    @property
    def updated_at(self) -> str | None: ...

    @property
    def expires_at(self) -> str | None: ...

    @property
    def gaze_x(self) -> int: ...

    @property
    def gaze_y(self) -> int: ...

    @property
    def head_dx(self) -> int: ...

    @property
    def head_dy(self) -> int: ...

    @property
    def blink(self) -> bool | None: ...

    @property
    def mouth(self) -> str | None: ...

    @property
    def brows(self) -> str | None: ...


class HdmiAmbientImpulseCueLike(Protocol):
    """Describe the minimal ambient reserve-card cue surface."""

    @property
    def eyebrow(self) -> str: ...

    @property
    def headline(self) -> str: ...

    @property
    def body(self) -> str: ...

    @property
    def accent(self) -> str: ...

    def glyph(self) -> str: ...


class HdmiServiceConnectCueLike(Protocol):
    """Describe the minimal service-connect cue surface."""

    @property
    def service_label(self) -> str: ...

    @property
    def summary(self) -> str: ...

    @property
    def detail(self) -> str: ...

    @property
    def accent(self) -> str: ...

    @property
    def qr_image_data_url(self) -> str | None: ...


class HdmiAmbientMomentLike(Protocol):
    """Describe one resolved idle-only HDMI ambient moment."""

    @property
    def ornament(self) -> str: ...

    @property
    def progress(self) -> float: ...

    @property
    def face_cue(self) -> HdmiFaceCueLike | None: ...


class HdmiPresentationNodeLike(Protocol):
    """Describe the bounded presentation node rendered over the default scene."""

    @property
    def box(self) -> BoxQuadLike: ...

    @property
    def kind(self) -> str: ...

    @property
    def accent(self) -> RgbTripletLike: ...

    @property
    def title(self) -> str: ...

    @property
    def subtitle(self) -> str: ...

    @property
    def body_lines(self) -> Sequence[str]: ...

    @property
    def chrome_progress(self) -> float: ...

    @property
    def content_progress(self) -> float: ...

    @property
    def body_progress(self) -> float: ...

    @property
    def image_path(self) -> ImagePathLike | None: ...


class HdmiPresentationGraphLike(Protocol):
    """Describe the active presentation graph resolved for one HDMI frame."""

    @property
    def active_node(self) -> HdmiPresentationNodeLike | None: ...

    @property
    def queued_cards(self) -> Sequence[object]: ...

    @property
    def face_cue(self) -> HdmiFaceCueLike | None: ...


class HdmiReserveBusLike(Protocol):
    """Describe the active owner of the right-hand HDMI reserve area."""

    @property
    def owner(self) -> str: ...

    @property
    def service_connect_cue(self) -> HdmiServiceConnectCueLike | None: ...

    @property
    def emoji_cue(self) -> HdmiEmojiCueLike | None: ...

    @property
    def ambient_impulse_cue(self) -> HdmiAmbientImpulseCueLike | None: ...

    @property
    def reason(self) -> str: ...