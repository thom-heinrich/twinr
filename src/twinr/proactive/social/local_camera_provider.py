"""Map the local IMX500 hardware adapter onto Twinr's social vision contract."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.ai_camera import (
    AICameraBodyPose,
    AICameraGestureEvent,
    AICameraObservation,
    AICameraZone,
    LocalAICameraAdapter,
)

from .engine import (
    SocialBodyPose,
    SocialDetectedObject,
    SocialGestureEvent,
    SocialPersonZone,
    SocialSpatialBox,
    SocialVisionObservation,
)
from .observers import ProactiveVisionSnapshot


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LocalAICameraProviderConfig:
    """Store provider-level wiring settings for the local camera path."""

    source_device: str = "imx500"
    input_format: str | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LocalAICameraProviderConfig":
        """Build one provider config from ``TwinrConfig``."""

        return cls(
            source_device=str(getattr(config, "proactive_local_camera_source_device", "imx500") or "imx500"),
            input_format=getattr(config, "proactive_local_camera_input_format", None),
        )


class LocalAICameraObservationProvider:
    """Return bounded local-first camera observations without cloud dependency."""

    def __init__(
        self,
        *,
        adapter: LocalAICameraAdapter,
        config: LocalAICameraProviderConfig | None = None,
    ) -> None:
        """Initialize one provider from the low-level IMX500 adapter."""

        self.adapter = adapter
        self.config = config or LocalAICameraProviderConfig()

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LocalAICameraObservationProvider":
        """Build one provider directly from ``TwinrConfig``."""

        return cls(
            adapter=LocalAICameraAdapter.from_config(config),
            config=LocalAICameraProviderConfig.from_config(config),
        )

    def observe(self) -> ProactiveVisionSnapshot:
        """Capture one local camera snapshot and map it to the social contract."""

        try:
            observation = self.adapter.observe()
        except Exception:  # pragma: no cover - adapter already degrades conservatively.
            logger.exception("Local AI camera provider failed unexpectedly; returning a conservative health snapshot.")
            observation = AICameraObservation(
                observed_at=0.0,
                camera_online=False,
                camera_ready=False,
                camera_ai_ready=False,
                camera_error="local_ai_camera_provider_failed",
            )

        social = self._to_social_observation(observation)
        return ProactiveVisionSnapshot(
            observation=social,
            response_text=self._response_text(observation),
            captured_at=observation.last_camera_frame_at or observation.observed_at,
            image=None,
            source_device=self.config.source_device,
            input_format=self.config.input_format,
            response_id=None,
            request_id=None,
            model=observation.model,
        )

    def _to_social_observation(self, observation: AICameraObservation) -> SocialVisionObservation:
        """Map one adapter observation onto the stable social vision contract."""

        primary_person_box = None
        if observation.primary_person_box is not None:
            primary_person_box = SocialSpatialBox(
                top=observation.primary_person_box.top,
                left=observation.primary_person_box.left,
                bottom=observation.primary_person_box.bottom,
                right=observation.primary_person_box.right,
            )

        return SocialVisionObservation(
            person_visible=observation.person_count > 0,
            person_count=observation.person_count,
            primary_person_zone=_map_zone(observation.primary_person_zone),
            primary_person_box=primary_person_box,
            primary_person_center_x=observation.primary_person_center_x,
            primary_person_center_y=observation.primary_person_center_y,
            looking_toward_device=bool(observation.looking_toward_device),
            person_near_device=observation.person_near_device,
            engaged_with_device=observation.engaged_with_device,
            visual_attention_score=observation.visual_attention_score,
            body_pose=_map_body_pose(observation.body_pose),
            pose_confidence=observation.pose_confidence,
            smiling=False,
            hand_or_object_near_camera=observation.hand_or_object_near_camera,
            showing_intent_likely=observation.showing_intent_likely,
            gesture_event=_map_gesture(observation.gesture_event),
            gesture_confidence=observation.gesture_confidence,
            objects=tuple(
                SocialDetectedObject(
                    label=item.label,
                    confidence=item.confidence,
                    zone=_map_zone(item.zone),
                    stable=False,
                    box=(
                        None
                        if item.box is None
                        else SocialSpatialBox(
                            top=item.box.top,
                            left=item.box.left,
                            bottom=item.box.bottom,
                            right=item.box.right,
                        )
                    ),
                )
                for item in observation.objects
            ),
            camera_online=observation.camera_online,
            camera_ready=observation.camera_ready,
            camera_ai_ready=observation.camera_ai_ready,
            camera_error=observation.camera_error,
            last_camera_frame_at=observation.last_camera_frame_at,
            last_camera_health_change_at=observation.last_camera_health_change_at,
        )

    def _response_text(self, observation: AICameraObservation) -> str:
        """Render one short inspectable provider summary."""

        return (
            "provider=local_ai_camera "
            f"ready={'yes' if observation.camera_ready else 'no'} "
            f"ai_ready={'yes' if observation.camera_ai_ready else 'no'} "
            f"person_count={observation.person_count} "
            f"body_pose={_map_body_pose(observation.body_pose).value} "
            f"gesture={_map_gesture(observation.gesture_event).value} "
            f"error={observation.camera_error or 'none'}"
        )


def _map_zone(value: AICameraZone) -> SocialPersonZone:
    """Map one hardware zone enum to the social zone enum."""

    mapping = {
        AICameraZone.LEFT: SocialPersonZone.LEFT,
        AICameraZone.CENTER: SocialPersonZone.CENTER,
        AICameraZone.RIGHT: SocialPersonZone.RIGHT,
        AICameraZone.UNKNOWN: SocialPersonZone.UNKNOWN,
    }
    return mapping.get(value, SocialPersonZone.UNKNOWN)


def _map_body_pose(value: AICameraBodyPose) -> SocialBodyPose:
    """Map one hardware body pose enum to the social body pose enum."""

    mapping = {
        AICameraBodyPose.UPRIGHT: SocialBodyPose.UPRIGHT,
        AICameraBodyPose.SEATED: SocialBodyPose.SEATED,
        AICameraBodyPose.SLUMPED: SocialBodyPose.SLUMPED,
        AICameraBodyPose.LYING_LOW: SocialBodyPose.LYING_LOW,
        AICameraBodyPose.FLOOR: SocialBodyPose.FLOOR,
        AICameraBodyPose.UNKNOWN: SocialBodyPose.UNKNOWN,
    }
    return mapping.get(value, SocialBodyPose.UNKNOWN)


def _map_gesture(value: AICameraGestureEvent) -> SocialGestureEvent:
    """Map one hardware gesture enum to the social gesture enum."""

    mapping = {
        AICameraGestureEvent.NONE: SocialGestureEvent.NONE,
        AICameraGestureEvent.STOP: SocialGestureEvent.STOP,
        AICameraGestureEvent.DISMISS: SocialGestureEvent.DISMISS,
        AICameraGestureEvent.CONFIRM: SocialGestureEvent.CONFIRM,
        AICameraGestureEvent.UNKNOWN: SocialGestureEvent.UNKNOWN,
    }
    return mapping.get(value, SocialGestureEvent.UNKNOWN)


__all__ = [
    "LocalAICameraObservationProvider",
    "LocalAICameraProviderConfig",
]
