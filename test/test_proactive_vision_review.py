from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.proactive import (
    SocialAudioObservation,
    SocialBodyPose,
    SocialObservation,
    SocialTriggerDecision,
    SocialTriggerPriority,
    SocialVisionObservation,
)
from twinr.proactive.social.observers import ProactiveVisionSnapshot
from twinr.proactive.social.vision_review import (
    OpenAIProactiveVisionReviewer,
    ProactiveVisionFrameBuffer,
    parse_proactive_vision_review_text,
)
from twinr.providers.openai.types import OpenAIImageInput


class FakeBackend:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = []

    def respond_to_images_with_metadata(
        self,
        prompt,
        *,
        images,
        conversation=None,
        instructions=None,
        allow_web_search=None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "images": list(images),
                "conversation": conversation,
                "instructions": instructions,
                "allow_web_search": allow_web_search,
            }
        )
        return SimpleNamespace(
            text=self.text,
            response_id="resp_review",
            request_id="req_review",
            model="gpt-5.2",
        )


def _snapshot(
    *,
    captured_at: float,
    filename: str,
    person_visible: bool = True,
    looking_toward_device: bool = False,
    body_pose: SocialBodyPose = SocialBodyPose.UPRIGHT,
    hand_or_object_near_camera: bool = False,
) -> ProactiveVisionSnapshot:
    return ProactiveVisionSnapshot(
        observation=SocialVisionObservation(
            person_visible=person_visible,
            looking_toward_device=looking_toward_device,
            body_pose=body_pose,
            hand_or_object_near_camera=hand_or_object_near_camera,
        ),
        response_text="ok",
        captured_at=captured_at,
        image=OpenAIImageInput(
            data=f"frame:{filename}".encode("utf-8"),
            content_type="image/png",
            filename=filename,
            label=f"raw {filename}",
        ),
        source_device="/dev/video0",
        input_format="bayer_grbg8",
        response_id="resp_classify",
        request_id="req_classify",
        model="gpt-5.2",
    )


class ProactiveVisionReviewTests(unittest.TestCase):
    def test_parse_proactive_vision_review_text(self) -> None:
        review = parse_proactive_vision_review_text(
            "\n".join(
                [
                    "decision=speak",
                    "confidence=high",
                    "reason=person clearly visible",
                    "scene=person near device",
                ]
            ),
            frame_count=3,
            response_id="resp",
            request_id="req",
            model="gpt-5.2",
        )

        self.assertIsNotNone(review)
        assert review is not None
        self.assertTrue(review.approved)
        self.assertEqual(review.frame_count, 3)
        self.assertEqual(review.reason, "person clearly visible")
        self.assertEqual(review.scene, "person near device")

    def test_frame_buffer_samples_oldest_to_newest_with_spacing(self) -> None:
        buffer = ProactiveVisionFrameBuffer(max_items=8)
        buffer.record(_snapshot(captured_at=0.0, filename="frame-1.png"))
        buffer.record(_snapshot(captured_at=0.3, filename="frame-2.png"))
        buffer.record(_snapshot(captured_at=1.4, filename="frame-3.png"))
        buffer.record(_snapshot(captured_at=2.7, filename="frame-4.png"))

        sampled = buffer.sample(
            now=3.0,
            max_frames=3,
            max_age_s=10.0,
            min_spacing_s=1.0,
        )

        self.assertEqual([item.image.filename for item in sampled if item.image is not None], [
            "frame-1.png",
            "frame-3.png",
            "frame-4.png",
        ])

    def test_reviewer_sends_multiple_images_in_time_order(self) -> None:
        backend = FakeBackend(
            "\n".join(
                [
                    "decision=skip",
                    "confidence=medium",
                    "reason=person leaves frame normally",
                    "scene=person exits left side of image",
                ]
            )
        )
        reviewer = OpenAIProactiveVisionReviewer(
            backend=backend,
            frame_buffer=ProactiveVisionFrameBuffer(max_items=8),
            max_frames=3,
            max_age_s=10.0,
            min_spacing_s=1.0,
        )
        reviewer.record_snapshot(
            _snapshot(
                captured_at=0.0,
                filename="frame-1.png",
                person_visible=True,
                body_pose=SocialBodyPose.UPRIGHT,
            )
        )
        reviewer.record_snapshot(
            _snapshot(
                captured_at=1.6,
                filename="frame-2.png",
                person_visible=True,
                body_pose=SocialBodyPose.SLUMPED,
            )
        )
        reviewer.record_snapshot(
            _snapshot(
                captured_at=3.1,
                filename="frame-3.png",
                person_visible=False,
                body_pose=SocialBodyPose.UNKNOWN,
            )
        )

        trigger = SocialTriggerDecision(
            trigger_id="possible_fall",
            prompt="Brauchst du Hilfe?",
            reason="visibility loss after slumped posture",
            observed_at=3.2,
            priority=SocialTriggerPriority.POSSIBLE_FALL,
            score=0.81,
            threshold=0.65,
        )
        observation = SocialObservation(
            observed_at=3.2,
            inspected=False,
            low_motion=True,
            vision=SocialVisionObservation(person_visible=False, body_pose=SocialBodyPose.UNKNOWN),
            audio=SocialAudioObservation(speech_detected=False),
        )

        review = reviewer.review(trigger, observation=observation)

        self.assertIsNotNone(review)
        assert review is not None
        self.assertFalse(review.approved)
        self.assertEqual(review.frame_count, 3)
        self.assertEqual(len(backend.calls), 1)
        request = backend.calls[0]
        self.assertFalse(request["allow_web_search"])
        self.assertIn("trigger_id=possible_fall", request["prompt"])
        self.assertIn("sensor-only with no fresh camera frame", request["prompt"])
        self.assertEqual(
            [image.filename for image in request["images"]],
            ["frame-1.png", "frame-2.png", "frame-3.png"],
        )
        self.assertIn("Frame 1 of 3", request["images"][0].label or "")
        self.assertIn("Frame 3 of 3", request["images"][-1].label or "")


if __name__ == "__main__":
    unittest.main()
