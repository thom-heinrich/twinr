"""Expose Twinr's stable display package exports lazily.

Lazy resolution keeps internal helpers such as ``display.heartbeat`` import-safe
for ops modules that should not have to import the full rendering stack.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DisplayFaceBrowStyle",
    "DisplayFaceCue",
    "DisplayFaceCueStore",
    "DisplayEmojiController",
    "DisplayEmojiCue",
    "DisplayEmojiCueStore",
    "DisplayEmojiSymbol",
    "DisplayFaceEmotion",
    "DisplayFaceExpression",
    "DisplayFaceExpressionController",
    "DisplayFaceGazeDirection",
    "DisplayFaceMouthStyle",
    "DisplayPresentationController",
    "DisplayPresentationCardCue",
    "DisplayPresentationCue",
    "DisplayPresentationStore",
    "HdmiFramebufferDisplay",
    "HdmiWaylandDisplay",
    "TwinrStatusDisplayLoop",
    "WaveshareEPD4In2V2",
    "create_display_adapter",
]


def __getattr__(name: str) -> Any:
    """Resolve public display exports on demand."""

    if name == "DisplayFaceCue":
        from twinr.display.face_cues import DisplayFaceCue

        return DisplayFaceCue
    if name == "DisplayFaceCueStore":
        from twinr.display.face_cues import DisplayFaceCueStore

        return DisplayFaceCueStore
    if name == "DisplayEmojiCue":
        from twinr.display.emoji_cues import DisplayEmojiCue

        return DisplayEmojiCue
    if name == "DisplayEmojiCueStore":
        from twinr.display.emoji_cues import DisplayEmojiCueStore

        return DisplayEmojiCueStore
    if name == "DisplayEmojiSymbol":
        from twinr.display.emoji_cues import DisplayEmojiSymbol

        return DisplayEmojiSymbol
    if name == "DisplayEmojiController":
        from twinr.display.emoji_cues import DisplayEmojiController

        return DisplayEmojiController
    if name == "DisplayFaceBrowStyle":
        from twinr.display.face_expressions import DisplayFaceBrowStyle

        return DisplayFaceBrowStyle
    if name == "DisplayFaceMouthStyle":
        from twinr.display.face_expressions import DisplayFaceMouthStyle

        return DisplayFaceMouthStyle
    if name == "DisplayFaceGazeDirection":
        from twinr.display.face_expressions import DisplayFaceGazeDirection

        return DisplayFaceGazeDirection
    if name == "DisplayFaceEmotion":
        from twinr.display.face_expressions import DisplayFaceEmotion

        return DisplayFaceEmotion
    if name == "DisplayFaceExpression":
        from twinr.display.face_expressions import DisplayFaceExpression

        return DisplayFaceExpression
    if name == "DisplayFaceExpressionController":
        from twinr.display.face_expressions import DisplayFaceExpressionController

        return DisplayFaceExpressionController
    if name == "DisplayPresentationCardCue":
        from twinr.display.presentation_cues import DisplayPresentationCardCue

        return DisplayPresentationCardCue
    if name == "DisplayPresentationCue":
        from twinr.display.presentation_cues import DisplayPresentationCue

        return DisplayPresentationCue
    if name == "DisplayPresentationStore":
        from twinr.display.presentation_cues import DisplayPresentationStore

        return DisplayPresentationStore
    if name == "DisplayPresentationController":
        from twinr.display.presentation_cues import DisplayPresentationController

        return DisplayPresentationController
    if name == "TwinrStatusDisplayLoop":
        from twinr.display.service import TwinrStatusDisplayLoop

        return TwinrStatusDisplayLoop
    if name == "HdmiFramebufferDisplay":
        from twinr.display.hdmi_fbdev import HdmiFramebufferDisplay

        return HdmiFramebufferDisplay
    if name == "HdmiWaylandDisplay":
        from twinr.display.hdmi_wayland import HdmiWaylandDisplay

        return HdmiWaylandDisplay
    if name == "WaveshareEPD4In2V2":
        from twinr.display.waveshare_v2 import WaveshareEPD4In2V2

        return WaveshareEPD4In2V2
    if name == "create_display_adapter":
        from twinr.display.factory import create_display_adapter

        return create_display_adapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
