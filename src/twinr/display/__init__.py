"""Expose Twinr's stable display package exports lazily.

Lazy resolution keeps internal helpers such as ``display.heartbeat`` import-safe
for ops modules that should not have to import the full rendering stack.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "DisplayAmbientImpulseController",
    "DisplayAmbientImpulseCue",
    "DisplayAmbientImpulseCueStore",
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


_EXPORTS = {
    "DisplayAmbientImpulseCue": "twinr.display.ambient_impulse_cues",
    "DisplayAmbientImpulseCueStore": "twinr.display.ambient_impulse_cues",
    "DisplayAmbientImpulseController": "twinr.display.ambient_impulse_cues",
    "DisplayFaceCue": "twinr.display.face_cues",
    "DisplayFaceCueStore": "twinr.display.face_cues",
    "DisplayEmojiCue": "twinr.display.emoji_cues",
    "DisplayEmojiCueStore": "twinr.display.emoji_cues",
    "DisplayEmojiSymbol": "twinr.display.emoji_cues",
    "DisplayEmojiController": "twinr.display.emoji_cues",
    "DisplayFaceBrowStyle": "twinr.display.face_expressions",
    "DisplayFaceMouthStyle": "twinr.display.face_expressions",
    "DisplayFaceGazeDirection": "twinr.display.face_expressions",
    "DisplayFaceEmotion": "twinr.display.face_expressions",
    "DisplayFaceExpression": "twinr.display.face_expressions",
    "DisplayFaceExpressionController": "twinr.display.face_expressions",
    "DisplayPresentationCardCue": "twinr.display.presentation_cues",
    "DisplayPresentationCue": "twinr.display.presentation_cues",
    "DisplayPresentationStore": "twinr.display.presentation_cues",
    "DisplayPresentationController": "twinr.display.presentation_cues",
    "TwinrStatusDisplayLoop": "twinr.display.service",
    "HdmiFramebufferDisplay": "twinr.display.hdmi_fbdev",
    "HdmiWaylandDisplay": "twinr.display.hdmi_wayland",
    "WaveshareEPD4In2V2": "twinr.display.waveshare_v2",
    "create_display_adapter": "twinr.display.factory",
}


def __getattr__(name: str) -> object:
    """Resolve public display exports on demand."""

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
