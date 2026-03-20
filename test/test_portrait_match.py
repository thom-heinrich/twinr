from pathlib import Path
from types import SimpleNamespace
import math
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.portrait_match import (
    PortraitEmbeddingResult,
    PortraitMatchConfig,
    PortraitMatchProvider,
)


class FakeClock:
    def __init__(self, *values: float) -> None:
        self.values = list(values)
        self.fallback = values[-1] if values else 0.0

    def __call__(self) -> float:
        if self.values:
            self.fallback = self.values.pop(0)
        return self.fallback


class FakeCamera:
    def __init__(self, payloads: list[bytes]) -> None:
        self.calls = 0
        self.payloads = list(payloads)

    def capture_photo(self, *, filename: str = "x.png"):
        self.calls += 1
        payload = self.payloads.pop(0) if self.payloads else b"live-primary"
        return SimpleNamespace(
            data=payload,
            content_type="image/png",
            filename=filename,
            source_device="/dev/video0",
            input_format="bayer_grbg8",
        )


class FakeBackend:
    def __init__(self) -> None:
        self.name = "fake_portrait_backend"
        self.extract_calls: list[tuple[str, bytes]] = []
        self.embeddings = {
            b"legacy-reference": (1.0, 0.0, 0.0),
            b"ref-a": (1.0, 0.0, 0.0),
            b"ref-b": (0.96, 0.04, 0.0),
            b"guest-ref": (0.0, 1.0, 0.0),
            b"live-primary": (0.98, 0.02, 0.0),
            b"live-primary-2": (0.97, 0.03, 0.0),
            b"live-guest": (0.02, 0.98, 0.0),
        }

    def extract_embedding(self, image_bytes: bytes, *, image_label: str) -> PortraitEmbeddingResult:
        self.extract_calls.append((image_label, image_bytes))
        embedding = self.embeddings.get(image_bytes)
        if embedding is None:
            return PortraitEmbeddingResult(state="decode_failed", detail=f"{image_label}:missing_fixture")
        norm = math.sqrt(sum(value * value for value in embedding))
        normalized = tuple(round(value / norm, 8) for value in embedding)
        return PortraitEmbeddingResult(
            state="ok",
            embedding=normalized,
            face_count=1,
            detector_confidence=0.95,
        )

    def similarity_score(
        self,
        *,
        reference_embedding: tuple[float, ...],
        live_embedding: tuple[float, ...],
    ) -> float | None:
        return round(sum(left * right for left, right in zip(reference_embedding, live_embedding)), 4)


def _make_config(
    temp_dir: str,
    *,
    reference_image_path: Path | None = None,
    max_age_s: float = 60.0,
) -> PortraitMatchConfig:
    return PortraitMatchConfig(
        reference_image_path=reference_image_path,
        detector_model_path=Path(temp_dir) / "detector.onnx",
        recognizer_model_path=Path(temp_dir) / "recognizer.onnx",
        identity_store_path=Path(temp_dir) / "portrait_identities.json",
        identity_reference_image_dir=Path(temp_dir) / "portrait_identities",
        primary_user_id="main_user",
        max_reference_images_per_user=6,
        identity_margin=0.05,
        temporal_window_s=300.0,
        temporal_min_observations=2,
        temporal_max_observations=12,
        max_age_s=max_age_s,
        likely_threshold=0.45,
        uncertain_threshold=0.34,
        reference_max_bytes=4096,
        capture_lock_timeout_s=0.1,
    )


class PortraitMatchProviderTests(unittest.TestCase):
    def test_provider_bootstraps_legacy_reference_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference.jpg"
            reference_path.write_bytes(b"legacy-reference")
            provider = PortraitMatchProvider(
                camera=FakeCamera([b"live-primary"]),
                config=_make_config(temp_dir, reference_image_path=reference_path, max_age_s=60.0),
                backend=FakeBackend(),
                clock=FakeClock(100.0, 100.0),
            )

            summary = provider.summary()
            first = provider.observe()
            second = provider.observe()

        self.assertTrue(summary.enrolled)
        self.assertEqual(summary.reference_image_count, 1)
        self.assertEqual(first.state, "likely_reference_user")
        self.assertTrue(first.matches_reference_user)
        self.assertEqual(first.backend_name, "fake_portrait_backend")
        self.assertEqual(first.capture_source_device, "/dev/video0")
        self.assertEqual(first.reference_image_count, 1)
        self.assertEqual(first.temporal_state, "insufficient_history")
        self.assertEqual(second, first)
        self.assertEqual(provider.camera.calls, 1)

    def test_enrollment_flow_supports_multiple_reference_images_and_temporal_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = PortraitMatchProvider(
                camera=FakeCamera([b"live-primary", b"live-primary-2"]),
                config=_make_config(temp_dir, max_age_s=0.0),
                backend=FakeBackend(),
                clock=FakeClock(10.0, 11.0),
            )

            first_enroll = provider.enroll_reference_image_bytes(
                b"ref-a",
                user_id="main_user",
                display_name="Thom",
                filename_hint="thom-a.jpg",
                source="manual_import",
            )
            second_enroll = provider.enroll_reference_image_bytes(
                b"ref-b",
                user_id="main_user",
                display_name="Thom",
                filename_hint="thom-b.jpg",
                source="manual_import",
            )
            first = provider.observe()
            second = provider.observe()

        self.assertEqual(first_enroll.status, "enrolled")
        self.assertEqual(second_enroll.status, "enrolled")
        self.assertEqual(second_enroll.reference_image_count, 2)
        self.assertEqual(first.state, "likely_reference_user")
        self.assertEqual(first.temporal_state, "insufficient_history")
        self.assertEqual(first.temporal_observation_count, 1)
        self.assertEqual(first.reference_image_count, 2)
        self.assertEqual(second.state, "likely_reference_user")
        self.assertEqual(second.temporal_state, "stable_match")
        self.assertEqual(second.temporal_observation_count, 2)
        self.assertGreaterEqual(second.fused_confidence or 0.0, first.confidence or 0.0)

    def test_provider_detects_other_enrolled_user(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = PortraitMatchProvider(
                camera=FakeCamera([b"live-guest"]),
                config=_make_config(temp_dir, max_age_s=0.0),
                backend=FakeBackend(),
                clock=FakeClock(20.0),
            )

            provider.enroll_reference_image_bytes(
                b"ref-a",
                user_id="main_user",
                display_name="Thom",
                filename_hint="thom.jpg",
                source="manual_import",
            )
            provider.enroll_reference_image_bytes(
                b"guest-ref",
                user_id="guest_user",
                display_name="Guest",
                filename_hint="guest.jpg",
                source="manual_import",
            )
            observation = provider.observe()

        self.assertEqual(observation.state, "known_other_user")
        self.assertFalse(observation.matches_reference_user)
        self.assertEqual(observation.matched_user_id, "guest_user")
        self.assertEqual(observation.reference_image_count, 1)

    def test_remove_reference_and_clear_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = PortraitMatchProvider(
                camera=FakeCamera([b"live-primary"]),
                config=_make_config(temp_dir),
                backend=FakeBackend(),
                clock=FakeClock(30.0),
            )

            enrollment = provider.enroll_reference_image_bytes(
                b"ref-a",
                user_id="main_user",
                display_name="Thom",
                filename_hint="thom.jpg",
                source="manual_import",
            )
            removed = provider.remove_reference_image(reference_id=enrollment.reference_id or "", user_id="main_user")
            cleared = provider.clear_identity_profile(user_id="main_user")

        self.assertEqual(removed.status, "removed")
        self.assertEqual(removed.reference_image_count, 0)
        self.assertEqual(cleared.status, "profile_unavailable")

    def test_provider_without_reference_image_or_store_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = PortraitMatchProvider(
                camera=FakeCamera([b"live-primary"]),
                config=_make_config(temp_dir),
                backend=FakeBackend(),
                clock=FakeClock(40.0),
            )

            observation = provider.observe()

        self.assertEqual(observation.state, "reference_image_unavailable")
        self.assertFalse(observation.matches_reference_user)
        self.assertEqual(observation.backend_name, "fake_portrait_backend")


if __name__ == "__main__":
    unittest.main()
