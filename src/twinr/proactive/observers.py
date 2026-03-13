from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from twinr.hardware.audio import AmbientAudioLevelSample, AmbientAudioSampler
from twinr.hardware.camera import V4L2StillCamera
from twinr.proactive.engine import SocialAudioObservation, SocialBodyPose, SocialVisionObservation
from twinr.providers.openai.backend import OpenAIBackend, OpenAIImageInput


@dataclass(frozen=True, slots=True)
class ProactiveVisionSnapshot:
    observation: SocialVisionObservation
    response_text: str
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None


@dataclass(frozen=True, slots=True)
class ProactiveAudioSnapshot:
    observation: SocialAudioObservation
    sample: AmbientAudioLevelSample | None = None


class NullAudioObservationProvider:
    def observe(self) -> ProactiveAudioSnapshot:
        return ProactiveAudioSnapshot(observation=SocialAudioObservation())


class AmbientAudioObservationProvider:
    def __init__(
        self,
        *,
        sampler: AmbientAudioSampler,
        audio_lock: Lock | None = None,
        sample_ms: int = 1000,
        distress_enabled: bool = False,
        speech_min_active_ratio: float = 0.2,
        distress_min_active_ratio: float = 0.45,
        distress_peak_threshold: int | None = None,
    ) -> None:
        self.sampler = sampler
        self.audio_lock = audio_lock or Lock()
        self.sample_ms = max(getattr(sampler, "chunk_ms", 20), sample_ms)
        self.distress_enabled = distress_enabled
        self.speech_min_active_ratio = max(0.0, speech_min_active_ratio)
        self.distress_min_active_ratio = max(0.0, distress_min_active_ratio)
        self.distress_peak_threshold = distress_peak_threshold

    def observe(self) -> ProactiveAudioSnapshot:
        with self.audio_lock:
            sample = self.sampler.sample_levels(duration_ms=self.sample_ms)
        speech_detected = (
            sample.active_chunk_count > 0 and sample.active_ratio >= self.speech_min_active_ratio
        )
        distress_detected: bool | None = None
        if self.distress_enabled:
            speech_threshold = getattr(self.sampler, "speech_threshold", 700)
            peak_threshold = self.distress_peak_threshold or max(1800, int(speech_threshold * 2.5))
            distress_detected = (
                speech_detected
                and sample.active_ratio >= self.distress_min_active_ratio
                and sample.peak_rms >= peak_threshold
            )
        return ProactiveAudioSnapshot(
            observation=SocialAudioObservation(
                speech_detected=speech_detected,
                distress_detected=distress_detected,
            ),
            sample=sample,
        )


class OpenAIVisionObservationProvider:
    def __init__(
        self,
        *,
        backend: OpenAIBackend,
        camera: V4L2StillCamera,
        camera_lock: Lock | None = None,
    ) -> None:
        self.backend = backend
        self.camera = camera
        self.camera_lock = camera_lock or Lock()

    def observe(self) -> ProactiveVisionSnapshot:
        with self.camera_lock:
            capture = self.camera.capture_photo(filename="proactive-camera-capture.png")
        response = self.backend.respond_to_images_with_metadata(
            _VISION_CLASSIFIER_PROMPT,
            images=(
                OpenAIImageInput(
                    data=capture.data,
                    content_type=capture.content_type,
                    filename=capture.filename,
                    label="Live proactive camera frame from the device.",
                ),
            ),
            allow_web_search=False,
        )
        return ProactiveVisionSnapshot(
            observation=parse_vision_observation_text(response.text),
            response_text=response.text,
            response_id=response.response_id,
            request_id=response.request_id,
            model=response.model,
        )


def parse_vision_observation_text(text: str) -> SocialVisionObservation:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip().lower()

    return SocialVisionObservation(
        person_visible=_parse_bool(values.get("person_visible")),
        looking_toward_device=_parse_bool(values.get("looking_toward_device")),
        body_pose=_parse_pose(values.get("body_pose")),
        smiling=_parse_bool(values.get("smiling")),
        hand_or_object_near_camera=_parse_bool(values.get("hand_or_object_near_camera")),
    )


def _parse_bool(value: str | None) -> bool:
    if value in {"yes", "true", "1"}:
        return True
    return False


def _parse_pose(value: str | None) -> SocialBodyPose:
    if value == "upright":
        return SocialBodyPose.UPRIGHT
    if value == "slumped":
        return SocialBodyPose.SLUMPED
    if value == "floor":
        return SocialBodyPose.FLOOR
    return SocialBodyPose.UNKNOWN


_VISION_CLASSIFIER_PROMPT = (
    "You classify a single live camera frame for Twinr's proactive trigger engine. "
    "Return only ASCII lines in this exact key=value format:\n"
    "person_visible=yes|no\n"
    "looking_toward_device=yes|no\n"
    "body_pose=upright|slumped|floor|unknown\n"
    "smiling=yes|no\n"
    "hand_or_object_near_camera=yes|no\n\n"
    "Rules:\n"
    "- Report only what is visually obvious in the image.\n"
    "- If uncertain, use no or unknown.\n"
    "- Do not infer identity, emotions, diagnosis, age, or intent beyond the listed keys.\n"
    "- `person_visible=yes` if any meaningful part of a person is visible, even when the body is partly cropped.\n"
    "- `floor` means the visible person is low to the floor, clearly lying down, collapsed at ground level, or only partially visible near the floor after a drop.\n"
    "- Prefer `floor` over `unknown` when a person is obviously very low in the scene.\n"
    "- `slumped` means seated or standing but visibly collapsed forward or drooping.\n"
)


__all__ = [
    "AmbientAudioObservationProvider",
    "NullAudioObservationProvider",
    "OpenAIVisionObservationProvider",
    "ProactiveAudioSnapshot",
    "ProactiveVisionSnapshot",
    "parse_vision_observation_text",
]
