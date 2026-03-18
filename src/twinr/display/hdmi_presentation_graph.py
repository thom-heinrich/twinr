"""Resolve Twinr's HDMI presentation cues into a small scene graph.

The framebuffer/Wayland adapters should not decide card priority, morph stage,
or face-sync behavior inline. This module owns the capability-driven graph
model that selects one active presentation node, computes calm transition
states, and derives an optional face reaction while keeping those concerns out
of the transport and status-loop layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from twinr.display.face_cues import DisplayFaceCue
from twinr.display.face_expressions import DisplayFaceEmotion, DisplayFaceExpression, DisplayFaceGazeDirection
from twinr.display.presentation_cues import DisplayPresentationCardCue, DisplayPresentationCue


_PRESENTATION_ACCENTS = {
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
    box: tuple[int, int, int, int]
    title: str
    subtitle: str
    body_lines: tuple[str, ...]
    image_path: str | None
    accent: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class HdmiPresentationSceneGraph:
    """Prepared HDMI presentation graph with one active node and queue info."""

    active_node: HdmiPresentationNode | None = None
    queued_cards: tuple[DisplayPresentationCardCue, ...] = ()
    face_cue: DisplayFaceCue | None = None

    def telemetry_signature(self) -> tuple[object, ...] | None:
        """Return a semantic signature suitable for low-noise display logs."""

        node = self.active_node
        if node is None:
            return None
        return (node.key, node.kind, node.priority, node.stage, len(self.queued_cards))


@dataclass(slots=True)
class HdmiPresentationSceneGraphBuilder:
    """Resolve one optional presentation cue into a calm HDMI scene graph."""

    def build(
        self,
        *,
        cue: DisplayPresentationCue | None,
        face_box: tuple[int, int, int, int],
        panel_box: tuple[int, int, int, int],
        now: datetime | None = None,
    ) -> HdmiPresentationSceneGraph | None:
        """Return one bounded graph for the active HDMI presentation cue."""

        if cue is None:
            return None
        active_card = cue.active_card()
        if active_card is None:
            return None
        progress = cue.transition_progress(now=now)
        eased_progress = self._ease_progress(progress)
        stage = cue.transition_stage(now=now)
        fullscreen_box = (
            face_box[0],
            min(face_box[1], panel_box[1]),
            panel_box[2],
            max(face_box[3], panel_box[3]),
        )
        active_node = HdmiPresentationNode(
            key=active_card.key,
            kind=active_card.kind,
            priority=active_card.priority,
            stage=stage,
            progress=progress,
            eased_progress=eased_progress,
            chrome_progress=max(0.1, eased_progress),
            content_progress=self._segment_progress(eased_progress, start=0.18, end=0.82),
            body_progress=self._segment_progress(eased_progress, start=0.34, end=0.96),
            box=self._interpolate_box(panel_box, fullscreen_box, eased_progress),
            title=active_card.title,
            subtitle=active_card.subtitle,
            body_lines=active_card.body_lines,
            image_path=active_card.image_path,
            accent=_PRESENTATION_ACCENTS.get(active_card.accent, _PRESENTATION_ACCENTS["info"]),
        )
        return HdmiPresentationSceneGraph(
            active_node=active_node,
            queued_cards=cue.queued_cards(),
            face_cue=self._face_sync_for_card(active_card, stage=stage, progress=eased_progress),
        )

    def _interpolate_box(
        self,
        start: tuple[int, int, int, int],
        end: tuple[int, int, int, int],
        progress: float,
    ) -> tuple[int, int, int, int]:
        """Interpolate one rectangle between start and end geometry."""

        safe_progress = max(0.0, min(1.0, float(progress)))
        return tuple(int(round(s + ((e - s) * safe_progress))) for s, e in zip(start, end))

    def _ease_progress(self, progress: float) -> float:
        """Apply a smoothstep easing curve for calmer card morphs."""

        safe_progress = max(0.0, min(1.0, float(progress)))
        return safe_progress * safe_progress * (3.0 - (2.0 * safe_progress))

    def _segment_progress(self, progress: float, *, start: float, end: float) -> float:
        """Map one eased progress value into a later content-reveal segment."""

        if progress <= start:
            return 0.0
        if progress >= end:
            return 1.0
        return max(0.0, min(1.0, (progress - start) / max(0.01, end - start)))

    def _face_sync_for_card(
        self,
        card: DisplayPresentationCardCue,
        *,
        stage: str,
        progress: float,
    ) -> DisplayFaceCue | None:
        """Return one optional face reaction synchronized to the active card."""

        if progress >= 0.98 and stage == "focused":
            return None
        emotion = self._emotion_for_card(card)
        gaze = DisplayFaceGazeDirection.RIGHT
        if emotion == DisplayFaceEmotion.THOUGHTFUL:
            gaze = DisplayFaceGazeDirection.UP_RIGHT
        if emotion == DisplayFaceEmotion.SAD:
            gaze = DisplayFaceGazeDirection.DOWN_RIGHT
        expression = DisplayFaceExpression.from_emotion(
            emotion,
            gaze=gaze,
            head_dx=1 if progress < 0.92 else 0,
            head_dy=0,
        )
        return expression.to_cue(source="presentation_graph")

    def _emotion_for_card(self, card: DisplayPresentationCardCue) -> DisplayFaceEmotion:
        """Derive one conservative face emotion for the active card."""

        if card.face_emotion:
            return DisplayFaceEmotion(card.face_emotion)
        if card.kind == "image":
            return DisplayFaceEmotion.CURIOUS
        return _ACCENT_EMOTIONS.get(card.accent, DisplayFaceEmotion.FOCUSED)
