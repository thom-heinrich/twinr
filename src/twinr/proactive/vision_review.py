from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

from twinr.proactive.engine import SocialObservation, SocialTriggerDecision
from twinr.proactive.observers import ProactiveVisionSnapshot
from twinr.providers.openai.backend import OpenAIBackend, OpenAIImageInput

_REVIEWABLE_TRIGGER_IDS = frozenset(
    {
        "person_returned",
        "attention_window",
        "slumped_quiet",
        "possible_fall",
        "floor_stillness",
        "showing_intent",
        "distress_possible",
        "positive_contact",
    }
)

_TRIGGER_GUIDANCE = {
    "person_returned": (
        "Approve only if a person is clearly visible in the newest frames. "
        "If the room looks empty or the person signal is inconsistent, skip."
    ),
    "attention_window": (
        "Approve only if a person is clearly visible and appears to be attending to the device now. "
        "If the person is absent, cropped too heavily, or not clearly engaged, skip."
    ),
    "slumped_quiet": (
        "Approve only if a person is clearly visible and visibly slumped or unwell across the recent frames. "
        "If posture is unclear, seated normally, or absent, skip."
    ),
    "possible_fall": (
        "Approve only if the sequence clearly suggests a concerning drop, a person low to the ground, "
        "or a concerning disappearance immediately after visible presence. "
        "If it could just be leaving the frame, sitting down, or a false person sighting, skip."
    ),
    "floor_stillness": (
        "Approve only if a person appears low to the floor or collapsed and the sequence supports concern. "
        "If the floor is not visible, the room is empty, or the pattern is ambiguous, skip."
    ),
    "showing_intent": (
        "Approve only if a person or a hand or object is clearly being shown to the device in the newest frames. "
        "If the scene is empty or the object signal is weak, skip."
    ),
    "distress_possible": (
        "Approve only if the visible posture or appearance adds real concern. "
        "If the concern seems to come only from audio or the visual evidence is weak, skip."
    ),
    "positive_contact": (
        "Approve only if a person is clearly visible and looks positively engaged in the newest frames. "
        "If visibility is weak or uncertain, skip."
    ),
}

_VISION_REVIEW_INSTRUCTIONS = (
    "You are Twinr's conservative visual second-opinion checker for proactive prompts. "
    "Review the recent camera frames in time order from oldest to newest. "
    "Approve speaking only when the visual evidence clearly supports the proposed proactive trigger right now. "
    "If the frames are inconsistent, empty, badly cropped, or ambiguous, choose skip. "
    "Do not infer identity, diagnosis, or hidden emotion beyond what is visually obvious. "
    "Return only ASCII lines in exactly this format:\n"
    "decision=speak|skip\n"
    "confidence=high|medium|low\n"
    "reason=<short ascii reason>\n"
    "scene=<short ascii scene summary>\n"
)


@dataclass(frozen=True, slots=True)
class ProactiveVisionReview:
    approved: bool
    decision: str
    confidence: str
    reason: str
    scene: str = ""
    frame_count: int = 0
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    raw_text: str = ""


def is_reviewable_image_trigger(trigger_id: str) -> bool:
    return trigger_id.strip().lower() in _REVIEWABLE_TRIGGER_IDS


class ProactiveVisionFrameBuffer:
    def __init__(self, *, max_items: int = 12) -> None:
        self.max_items = max(1, max_items)
        self._frames: deque[ProactiveVisionSnapshot] = deque(maxlen=self.max_items)

    def record(self, snapshot: ProactiveVisionSnapshot) -> None:
        if snapshot.image is None or snapshot.captured_at is None:
            return
        self._frames.append(snapshot)

    def sample(
        self,
        *,
        now: float,
        max_frames: int,
        max_age_s: float,
        min_spacing_s: float,
    ) -> tuple[ProactiveVisionSnapshot, ...]:
        recent = [
            item
            for item in self._frames
            if item.captured_at is not None and (now - item.captured_at) <= max_age_s
        ]
        if not recent:
            return ()
        sampled: list[ProactiveVisionSnapshot] = []
        for item in recent:
            if not sampled:
                sampled.append(item)
                continue
            previous_at = sampled[-1].captured_at
            if previous_at is None:
                previous_at = item.captured_at if item.captured_at is not None else now
            current_at = item.captured_at if item.captured_at is not None else now
            if (current_at - previous_at) >= min_spacing_s:
                sampled.append(item)
        if sampled[-1] is not recent[-1]:
            sampled.append(recent[-1])
        if len(sampled) <= max_frames:
            return tuple(sampled)
        return tuple(self._downsample(sampled, target=max_frames))

    def _downsample(
        self,
        items: list[ProactiveVisionSnapshot],
        *,
        target: int,
    ) -> list[ProactiveVisionSnapshot]:
        target = max(1, target)
        if len(items) <= target:
            return list(items)
        if target == 1:
            return [items[-1]]
        chosen_indices: list[int] = []
        last_index = len(items) - 1
        for position in range(target):
            raw_index = round(position * last_index / (target - 1))
            if chosen_indices and raw_index <= chosen_indices[-1]:
                raw_index = chosen_indices[-1] + 1
            remaining_slots = target - position - 1
            max_allowed = last_index - remaining_slots
            raw_index = min(raw_index, max_allowed)
            chosen_indices.append(raw_index)
        return [items[index] for index in chosen_indices]


class OpenAIProactiveVisionReviewer:
    def __init__(
        self,
        *,
        backend: OpenAIBackend,
        frame_buffer: ProactiveVisionFrameBuffer | None = None,
        max_frames: int = 4,
        max_age_s: float = 12.0,
        min_spacing_s: float = 1.2,
    ) -> None:
        self.backend = backend
        self.frame_buffer = frame_buffer or ProactiveVisionFrameBuffer()
        self.max_frames = max(1, max_frames)
        self.max_age_s = max(1.0, max_age_s)
        self.min_spacing_s = max(0.0, min_spacing_s)

    def record_snapshot(self, snapshot: ProactiveVisionSnapshot) -> None:
        self.frame_buffer.record(snapshot)

    def review(
        self,
        trigger: SocialTriggerDecision,
        *,
        observation: SocialObservation,
    ) -> ProactiveVisionReview | None:
        if not is_reviewable_image_trigger(trigger.trigger_id):
            return None
        frames = self.frame_buffer.sample(
            now=observation.observed_at,
            max_frames=self.max_frames,
            max_age_s=self.max_age_s,
            min_spacing_s=self.min_spacing_s,
        )
        if not frames:
            return None
        response = self.backend.respond_to_images_with_metadata(
            self._build_prompt(trigger, observation=observation, frames=frames),
            images=self._build_images(frames, now=observation.observed_at),
            instructions=_VISION_REVIEW_INSTRUCTIONS,
            allow_web_search=False,
        )
        return parse_proactive_vision_review_text(
            response.text,
            frame_count=len(frames),
            response_id=response.response_id,
            request_id=response.request_id,
            model=response.model,
        )

    def _build_prompt(
        self,
        trigger: SocialTriggerDecision,
        *,
        observation: SocialObservation,
        frames: tuple[ProactiveVisionSnapshot, ...],
    ) -> str:
        vision = observation.vision
        audio = observation.audio
        lines = [
            "Proposed proactive trigger review.",
            f"trigger_id={trigger.trigger_id}",
            f"default_prompt={trigger.prompt}",
            f"trigger_reason={trigger.reason}",
            f"trigger_guidance={_TRIGGER_GUIDANCE.get(trigger.trigger_id, 'Be conservative and skip if unclear.')}",
            "Current sensor state:",
            (
                f"inspected={'yes' if observation.inspected else 'no'} "
                f"person_visible={'yes' if vision.person_visible else 'no'} "
                f"looking_toward_device={'yes' if vision.looking_toward_device else 'no'} "
                f"body_pose={vision.body_pose.value} "
                f"smiling={'yes' if vision.smiling else 'no'} "
                f"hand_or_object_near_camera={'yes' if vision.hand_or_object_near_camera else 'no'} "
                f"speech_detected={audio.speech_detected} "
                f"distress_detected={audio.distress_detected} "
                f"low_motion={'yes' if observation.low_motion else 'no'}"
            ),
        ]
        evidence_lines = [item.detail.strip() for item in trigger.evidence if item.detail.strip()]
        if evidence_lines:
            lines.append("Trigger evidence:")
            lines.extend(f"- {item}" for item in evidence_lines[:4])
        if not observation.inspected:
            lines.append(
                "The newest tick is sensor-only with no fresh camera frame. "
                "Use the buffered frames plus the current sensor state."
            )
        lines.append(f"frame_count={len(frames)}")
        lines.append("Make the decision now.")
        return "\n".join(lines)

    def _build_images(
        self,
        frames: tuple[ProactiveVisionSnapshot, ...],
        *,
        now: float,
    ) -> list[OpenAIImageInput]:
        images: list[OpenAIImageInput] = []
        total = len(frames)
        for index, frame in enumerate(frames, start=1):
            if frame.image is None:
                continue
            age_s = 0.0 if frame.captured_at is None else max(0.0, now - frame.captured_at)
            label = (
                f"Frame {index} of {total}, about {age_s:.1f}s ago. "
                f"classifier: person_visible={'yes' if frame.observation.person_visible else 'no'}, "
                f"looking_toward_device={'yes' if frame.observation.looking_toward_device else 'no'}, "
                f"body_pose={frame.observation.body_pose.value}, "
                f"smiling={'yes' if frame.observation.smiling else 'no'}, "
                f"hand_or_object_near_camera={'yes' if frame.observation.hand_or_object_near_camera else 'no'}."
            )
            images.append(replace(frame.image, label=label))
        return images


def parse_proactive_vision_review_text(
    text: str,
    *,
    frame_count: int,
    response_id: str | None = None,
    request_id: str | None = None,
    model: str | None = None,
) -> ProactiveVisionReview | None:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip()
    decision = values.get("decision", "").strip().lower()
    if decision not in {"speak", "skip"}:
        return None
    confidence = values.get("confidence", "medium").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    reason = values.get("reason", "").strip() or "no reason provided"
    scene = values.get("scene", "").strip()
    return ProactiveVisionReview(
        approved=decision == "speak",
        decision=decision,
        confidence=confidence,
        reason=reason,
        scene=scene,
        frame_count=frame_count,
        response_id=response_id,
        request_id=request_id,
        model=model,
        raw_text=text.strip(),
    )


__all__ = [
    "OpenAIProactiveVisionReviewer",
    "ProactiveVisionFrameBuffer",
    "ProactiveVisionReview",
    "is_reviewable_image_trigger",
    "parse_proactive_vision_review_text",
]
