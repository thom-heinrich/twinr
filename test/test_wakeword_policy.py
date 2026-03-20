from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.audio import AmbientAudioCaptureWindow, AmbientAudioLevelSample
from twinr.proactive.wakeword import WakewordMatch
from twinr.proactive.wakeword.policy import (
    SttWakewordVerifier,
    WakewordDecisionPolicy,
    WakewordVerification,
    normalize_wakeword_backend,
)


class FakeBackend:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.calls = 0

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        del audio_bytes, filename, content_type, language, prompt
        self.calls += 1
        return self.transcript


class FakeCaptureVerifier:
    def __init__(self, status: str, *, reason: str | None = None) -> None:
        self.status = status
        self.reason = reason
        self.calls = 0

    def verify(self, capture, *, detector_match):
        del capture, detector_match
        self.calls += 1
        return WakewordVerification(
            status=self.status,
            backend="local_sequence",
            reason=self.reason,
        )


def _capture() -> AmbientAudioCaptureWindow:
    return AmbientAudioCaptureWindow(
        sample=AmbientAudioLevelSample(
            duration_ms=1500,
            chunk_count=8,
            active_chunk_count=4,
            average_rms=760,
            peak_rms=1540,
            active_ratio=0.5,
        ),
        pcm_bytes=b"\x01\x02" * 800,
        sample_rate=16000,
        channels=1,
    )


class WakewordDecisionPolicyTests(unittest.TestCase):
    def test_normalize_wakeword_backend_accepts_kws(self) -> None:
        self.assertEqual(normalize_wakeword_backend("kws", default="openwakeword"), "kws")

    def test_verifier_accepts_calibrated_twin_alias_for_twinr(self) -> None:
        verifier = SttWakewordVerifier(
            backend=FakeBackend("Hallo Twin wie gehts"),
            phrases=("hallo twinr", "hallo twin", "twinr", "twin"),
            language="de",
        )

        verification = verifier.verify(
            _capture(),
            detector_match=WakewordMatch(
                detected=True,
                transcript="",
                matched_phrase="twinr",
                backend="openwakeword",
                detector_label="twinr",
                score=0.58,
            ),
        )

        self.assertEqual(verification.status, "accepted")
        self.assertEqual(verification.matched_phrase, "hallo twin")
        self.assertEqual(verification.remaining_text, "wie gehts")

    def test_ambiguity_only_verifies_borderline_openwakeword_hit(self) -> None:
        backend = FakeBackend("hey twinr wie gehts")
        verifier = SttWakewordVerifier(
            backend=backend,
            phrases=("hey twinr", "twinr"),
            language="de",
        )
        policy = WakewordDecisionPolicy(
            primary_backend="openwakeword",
            fallback_backend="stt",
            verifier_mode="ambiguity_only",
            verifier_margin=0.08,
            primary_threshold=0.5,
            verifier=verifier,
        )

        decision = policy.decide(
            match=WakewordMatch(
                detected=True,
                transcript="",
                matched_phrase="hey twinr",
                backend="openwakeword",
                detector_label="hey_twinr",
                score=0.53,
            ),
            capture=_capture(),
            source="streaming_spotter",
        )

        self.assertTrue(decision.detected)
        self.assertEqual(decision.outcome, "verified")
        self.assertTrue(decision.verifier_used)
        self.assertEqual(decision.verifier_status, "accepted")
        self.assertEqual(decision.match.remaining_text, "wie gehts")
        self.assertEqual(backend.calls, 1)

    def test_verifier_can_reject_borderline_hit(self) -> None:
        verifier = SttWakewordVerifier(
            backend=FakeBackend("guten morgen tina"),
            phrases=("hey twinr", "twinr"),
            language="de",
        )
        policy = WakewordDecisionPolicy(
            primary_backend="openwakeword",
            fallback_backend="stt",
            verifier_mode="ambiguity_only",
            verifier_margin=0.08,
            primary_threshold=0.5,
            verifier=verifier,
        )

        decision = policy.decide(
            match=WakewordMatch(
                detected=True,
                transcript="",
                matched_phrase="twinr",
                backend="openwakeword",
                detector_label="twinr",
                score=0.51,
            ),
            capture=_capture(),
            source="ambient_spotter",
        )

        self.assertFalse(decision.detected)
        self.assertEqual(decision.outcome, "rejected_by_verifier")
        self.assertTrue(decision.verifier_used)
        self.assertEqual(decision.verifier_status, "rejected")

    def test_local_sequence_verifier_rejects_before_stt_verification(self) -> None:
        backend = FakeBackend("hey twinr wie gehts")
        verifier = SttWakewordVerifier(
            backend=backend,
            phrases=("hey twinr", "twinr"),
            language="de",
        )
        local_verifier = FakeCaptureVerifier("rejected", reason="score:0.12")
        policy = WakewordDecisionPolicy(
            primary_backend="openwakeword",
            fallback_backend="stt",
            verifier_mode="always",
            primary_threshold=0.5,
            verifier=verifier,
            local_verifier=local_verifier,
        )

        decision = policy.decide(
            match=WakewordMatch(
                detected=True,
                transcript="",
                matched_phrase="twinr",
                backend="openwakeword",
                detector_label="twinr_v2",
                score=0.91,
            ),
            capture=_capture(),
            source="streaming_spotter",
        )

        self.assertFalse(decision.detected)
        self.assertEqual(decision.outcome, "rejected_by_local_verifier")
        self.assertEqual(decision.local_verifier_status, "rejected")
        self.assertEqual(decision.local_verifier_reason, "score:0.12")
        self.assertEqual(local_verifier.calls, 1)
        self.assertEqual(backend.calls, 0)

    def test_local_sequence_verifier_can_accept_before_stt_verifier_runs(self) -> None:
        backend = FakeBackend("hey twinr wie gehts")
        verifier = SttWakewordVerifier(
            backend=backend,
            phrases=("hey twinr", "twinr"),
            language="de",
        )
        local_verifier = FakeCaptureVerifier("accepted", reason="score:0.91")
        policy = WakewordDecisionPolicy(
            primary_backend="openwakeword",
            fallback_backend="stt",
            verifier_mode="ambiguity_only",
            verifier_margin=0.08,
            primary_threshold=0.5,
            verifier=verifier,
            local_verifier=local_verifier,
        )

        decision = policy.decide(
            match=WakewordMatch(
                detected=True,
                transcript="",
                matched_phrase="hey twinr",
                backend="openwakeword",
                detector_label="twinr_v2",
                score=0.54,
            ),
            capture=_capture(),
            source="streaming_spotter",
        )

        self.assertTrue(decision.detected)
        self.assertEqual(decision.outcome, "verified")
        self.assertTrue(decision.local_verifier_used)
        self.assertEqual(decision.local_verifier_status, "accepted")
        self.assertEqual(local_verifier.calls, 1)
        self.assertEqual(backend.calls, 1)

    def test_missing_verifier_or_capture_keeps_borderline_hit_unverified(self) -> None:
        policy = WakewordDecisionPolicy(
            primary_backend="openwakeword",
            fallback_backend="stt",
            verifier_mode="always",
            primary_threshold=0.5,
            verifier=None,
        )

        decision = policy.decide(
            match=WakewordMatch(
                detected=True,
                transcript="",
                matched_phrase="hey twinr",
                backend="openwakeword",
                detector_label="hey_twinr",
                score=0.92,
            ),
            capture=None,
            source="streaming_spotter",
        )

        self.assertTrue(decision.detected)
        self.assertEqual(decision.outcome, "detected_unverified")
        self.assertFalse(decision.verifier_used)
        self.assertEqual(decision.verifier_status, "unavailable")

    def test_fallback_backend_detection_is_tagged_without_verifier(self) -> None:
        policy = WakewordDecisionPolicy(
            primary_backend="openwakeword",
            fallback_backend="stt",
            verifier_mode="ambiguity_only",
            primary_threshold=0.5,
            verifier=None,
        )

        decision = policy.decide(
            match=WakewordMatch(
                detected=True,
                transcript="hey twinr",
                matched_phrase="hey twinr",
                backend="stt",
                score=1.0,
            ),
            capture=_capture(),
            source="fallback_spotter",
        )

        self.assertTrue(decision.detected)
        self.assertEqual(decision.outcome, "fallback_detected")
        self.assertFalse(decision.verifier_used)
        self.assertEqual(decision.backend_used, "stt")


if __name__ == "__main__":
    unittest.main()
