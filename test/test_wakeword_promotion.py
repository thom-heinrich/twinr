"""Regression coverage for runtime-faithful wakeword promotion replay."""

from pathlib import Path
from struct import pack
import json
import os
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.wakeword.evaluation import WakewordEvalMetrics, WakewordScoreSanityReport
from twinr.proactive.wakeword.promotion import (
    _build_stream_policy,
    WakewordStreamEvalReport,
    load_wakeword_promotion_spec,
    run_wakeword_promotion_eval,
    run_wakeword_stream_eval,
)


def _write_wav(path: Path, *, amplitude: int, sample_count: int = 16000) -> None:
    frames = b"".join(pack("<h", amplitude) for _ in range(sample_count))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(frames)


class FakeAdaptiveModel:
    """Return one simple score based on the peak amplitude of a frame."""

    def __init__(self, *, wakeword_models) -> None:
        self.models = {str(name): object() for name in wakeword_models}

    def reset(self) -> None:
        return None

    def predict(self, samples):
        peak = max(abs(int(sample)) for sample in samples) if len(samples) else 0
        score = 0.12 if peak >= 900 else 0.03
        return {"twinr": score}


def _model_factory(**kwargs):
    return FakeAdaptiveModel(wakeword_models=kwargs.get("wakeword_models", ("twinr",)))


def _touch_kws_bundle(root: Path) -> dict[str, Path]:
    bundle = {
        "tokens": root / "tokens.txt",
        "encoder": root / "encoder.onnx",
        "decoder": root / "decoder.onnx",
        "joiner": root / "joiner.onnx",
        "keywords": root / "keywords.txt",
    }
    for path in bundle.values():
        path.write_text("x\n", encoding="utf-8")
    return bundle


class FakeKeywordStream:
    def __init__(self) -> None:
        self.pending_results: list[str] = []

    def accept_waveform(self, sample_rate, samples) -> None:
        del sample_rate
        peak = max((abs(float(sample)) for sample in samples), default=0.0)
        if peak >= 0.2:
            self.pending_results.append("twinna")

    def input_finished(self) -> None:
        return None


class FakeKeywordSpotter:
    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)

    def create_stream(self):
        return FakeKeywordStream()

    def is_ready(self, stream) -> bool:
        return bool(stream.pending_results)

    def decode_stream(self, stream) -> None:
        return None

    def get_result(self, stream) -> str:
        if not stream.pending_results:
            return ""
        return stream.pending_results.pop(0)

    def reset_stream(self, stream) -> None:
        stream.pending_results.clear()


class FakeWekwsFrameSpotter:
    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)

    def frame_bytes_for_channels(self, channels: int) -> int:
        return 1600 * max(1, int(channels)) * 2

    def reset(self) -> None:
        return None

    def process_pcm_bytes(self, pcm_bytes: bytes, *, channels: int):
        del channels
        peak = 0
        for offset in range(0, len(pcm_bytes) - (len(pcm_bytes) % 2), 2):
            sample = int.from_bytes(pcm_bytes[offset:offset + 2], "little", signed=True)
            peak = max(peak, abs(sample))
        if peak < 900:
            return None
        from twinr.proactive.wakeword.matching import WakewordMatch

        return WakewordMatch(
            detected=True,
            transcript="",
            matched_phrase="twinr",
            backend="wekws",
            detector_label="<TWINR_FAMILY>",
            score=0.72,
        )


class WakewordPromotionTests(unittest.TestCase):
    def test_build_stream_policy_attaches_sequence_verifier_for_wekws(self) -> None:
        config = TwinrConfig(
            wakeword_enabled=True,
            wakeword_primary_backend="wekws",
            wakeword_fallback_backend="stt",
            wakeword_verifier_mode="disabled",
            wakeword_openwakeword_sequence_verifier_models=(
                ("twinr_family", "/tmp/twinr_family.sequence_verifier.pkl"),
            ),
            wakeword_openwakeword_sequence_verifier_threshold=0.19,
        )

        with patch(
            "twinr.proactive.wakeword.promotion.build_configured_sequence_capture_verifier"
        ) as builder:
            sentinel = object()
            builder.return_value = sentinel
            policy = _build_stream_policy(config=config, backend=None)

        self.assertIs(policy.local_verifier, sentinel)
        builder.assert_called_once_with(config)

    def test_run_wakeword_stream_eval_replays_runtime_stream_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            positive = root / "wakeword_positive.wav"
            negative = root / "wakeword_negative.wav"
            manifest = root / "manifest.json"
            _write_wav(positive, amplitude=1200)
            _write_wav(negative, amplitude=100)
            manifest.write_text(
                json.dumps(
                    [
                        {"captured_audio_path": str(positive), "label": "correct"},
                        {"captured_audio_path": str(negative), "label": "false_positive"},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig(
                project_root=temp_dir,
                wakeword_enabled=True,
                wakeword_primary_backend="openwakeword",
                wakeword_fallback_backend="stt",
                wakeword_verifier_mode="disabled",
                wakeword_phrases=("twinr",),
                wakeword_openwakeword_models=("twinr",),
                wakeword_openwakeword_threshold=0.08,
                wakeword_openwakeword_patience_frames=1,
                wakeword_openwakeword_activation_samples=1,
                audio_sample_rate=16000,
                audio_channels=1,
                audio_speech_threshold=400,
                wakeword_attempt_cooldown_s=10.0,
            )

            report = run_wakeword_stream_eval(
                config=config,
                manifest_path=manifest,
                backend=None,
                model_factory=_model_factory,
            )
            report_path_exists = bool(report.report_path and report.report_path.exists())

        self.assertEqual(report.evaluated_entries, 2)
        self.assertEqual(report.metrics.true_positive, 1)
        self.assertEqual(report.metrics.true_negative, 1)
        self.assertEqual(report.metrics.false_positive, 0)
        self.assertEqual(report.metrics.false_negative, 0)
        self.assertEqual(report.accepted_detection_count, 1)
        self.assertGreater(report.total_audio_seconds, 1.9)
        self.assertIsNotNone(report.report_path)
        self.assertTrue(report_path_exists)

    def test_run_wakeword_stream_eval_supports_kws_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            positive = root / "wakeword_positive.wav"
            negative = root / "wakeword_negative.wav"
            manifest = root / "manifest.json"
            bundle = _touch_kws_bundle(root)
            _write_wav(positive, amplitude=16000)
            _write_wav(negative, amplitude=100)
            manifest.write_text(
                json.dumps(
                    [
                        {"captured_audio_path": str(positive), "label": "correct"},
                        {"captured_audio_path": str(negative), "label": "false_positive"},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig(
                project_root=temp_dir,
                wakeword_enabled=True,
                wakeword_primary_backend="kws",
                wakeword_fallback_backend="stt",
                wakeword_verifier_mode="disabled",
                wakeword_phrases=("twinna",),
                wakeword_kws_tokens_path=str(bundle["tokens"]),
                wakeword_kws_encoder_path=str(bundle["encoder"]),
                wakeword_kws_decoder_path=str(bundle["decoder"]),
                wakeword_kws_joiner_path=str(bundle["joiner"]),
                wakeword_kws_keywords_file_path=str(bundle["keywords"]),
                wakeword_kws_sample_rate=16000,
                wakeword_kws_keywords_threshold=0.25,
                audio_sample_rate=16000,
                audio_channels=1,
                audio_speech_threshold=400,
                wakeword_attempt_cooldown_s=10.0,
            )

            report = run_wakeword_stream_eval(
                config=config,
                manifest_path=manifest,
                backend=None,
                model_factory=lambda **_kwargs: FakeKeywordSpotter(),
            )

        self.assertEqual(report.evaluated_entries, 2)
        self.assertEqual(report.metrics.true_positive, 1)
        self.assertEqual(report.metrics.true_negative, 1)
        self.assertEqual(report.metrics.false_positive, 0)
        self.assertEqual(report.metrics.false_negative, 0)
        self.assertEqual(report.accepted_detection_count, 1)

    def test_run_wakeword_stream_eval_supports_wekws_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            positive = root / "wakeword_positive.wav"
            negative = root / "wakeword_negative.wav"
            manifest = root / "manifest.json"
            _write_wav(positive, amplitude=16000)
            _write_wav(negative, amplitude=100)
            manifest.write_text(
                json.dumps(
                    [
                        {"captured_audio_path": str(positive), "label": "correct"},
                        {"captured_audio_path": str(negative), "label": "false_positive"},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig(
                project_root=temp_dir,
                wakeword_enabled=True,
                wakeword_primary_backend="wekws",
                wakeword_fallback_backend="stt",
                wakeword_verifier_mode="disabled",
                wakeword_phrases=("twinna", "twinr"),
                wakeword_wekws_model_path=str(root / "model.onnx"),
                wakeword_wekws_config_path=str(root / "config.yaml"),
                wakeword_wekws_words_path=str(root / "words.txt"),
                wakeword_wekws_cmvn_path=str(root / "global_cmvn"),
                wakeword_wekws_threshold=0.25,
                audio_sample_rate=16000,
                audio_channels=1,
                audio_speech_threshold=400,
                wakeword_attempt_cooldown_s=10.0,
            )

            with patch(
                "twinr.proactive.wakeword.promotion.WakewordWekwsFrameSpotter",
                return_value=FakeWekwsFrameSpotter(),
            ) as factory:
                report = run_wakeword_stream_eval(
                    config=config,
                    manifest_path=manifest,
                    backend=None,
                    model_factory=None,
                )

        self.assertEqual(report.evaluated_entries, 2)
        self.assertEqual(report.metrics.true_positive, 1)
        self.assertEqual(report.metrics.true_negative, 1)
        self.assertEqual(report.metrics.false_positive, 0)
        self.assertEqual(report.metrics.false_negative, 0)
        self.assertEqual(report.accepted_detection_count, 1)
        factory.assert_called_once()

    def test_load_wakeword_promotion_spec_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite_manifest = root / "suite.json"
            ambient_manifest = root / "ambient.json"
            suite_manifest.write_text("[]\n", encoding="utf-8")
            ambient_manifest.write_text("[]\n", encoding="utf-8")
            spec_path = root / "promotion_spec.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "suites": [
                            {
                                "name": "critical",
                                "manifest_path": suite_manifest.name,
                                "max_false_negatives": 0,
                            }
                        ],
                        "ambient_guards": [
                            {
                                "name": "ambient",
                                "manifest_path": ambient_manifest.name,
                                "max_false_accepts_per_hour": 0.2,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            spec = load_wakeword_promotion_spec(spec_path)

        self.assertEqual(spec.suites[0].manifest_path, suite_manifest)
        self.assertEqual(spec.ambient_guards[0].manifest_path, ambient_manifest)

    def test_run_wakeword_promotion_eval_blocks_failed_wekws_bundle_sanity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite_manifest = root / "suite.json"
            suite_manifest.write_text(
                json.dumps(
                    [
                        {"captured_audio_path": str(root / "positive.wav"), "label": "correct"},
                        {"captured_audio_path": str(root / "negative.wav"), "label": "false_positive"},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            spec_path = root / "promotion_spec.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "suites": [
                            {
                                "name": "critical",
                                "manifest_path": str(suite_manifest),
                                "max_false_negatives": 0,
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig(
                project_root=temp_dir,
                wakeword_enabled=True,
                wakeword_primary_backend="wekws",
                wakeword_fallback_backend="stt",
                wakeword_verifier_mode="disabled",
                wakeword_phrases=("twinna", "twinr"),
                wakeword_wekws_model_path=str(root / "model.onnx"),
                wakeword_wekws_config_path=str(root / "config.yaml"),
                wakeword_wekws_words_path=str(root / "words.txt"),
                wakeword_wekws_cmvn_path=str(root / "global_cmvn"),
            )
            passing_stream_report = WakewordStreamEvalReport(
                metrics=WakewordEvalMetrics(
                    total=2,
                    true_positive=1,
                    false_positive=0,
                    true_negative=1,
                    false_negative=0,
                    precision=1.0,
                    recall=1.0,
                    false_positive_rate=0.0,
                    false_negative_rate=0.0,
                ),
                evaluated_entries=2,
                accepted_detection_count=1,
                total_audio_seconds=2.0,
            )
            failing_sanity = WakewordScoreSanityReport(
                backend="wekws",
                passed=False,
                positive_examples=8,
                negative_examples=8,
                positive_score_floor=0.21,
                negative_score_ceiling=0.24,
                min_positive_score=0.20,
                max_negative_score=0.25,
                separation_margin=-0.03,
                reason=None,
            )

            with patch(
                "twinr.proactive.wakeword.promotion.WakewordWekwsSpotter",
                return_value=object(),
            ), patch(
                "twinr.proactive.wakeword.promotion.evaluate_wekws_score_sanity_entries",
                return_value=failing_sanity,
            ), patch(
                "twinr.proactive.wakeword.promotion.evaluate_wakeword_stream_entries",
                return_value=passing_stream_report,
            ):
                report = run_wakeword_promotion_eval(
                    config=config,
                    spec_path=spec_path,
                    backend=None,
                    model_factory=None,
                )

        self.assertFalse(report.passed)
        self.assertIsNotNone(report.backend_sanity)
        self.assertIn("wekws_bundle_sanity", report.blockers[0])

    def test_run_wakeword_promotion_eval_blocks_suite_and_ambient_regressions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite_positive = root / "suite_positive.wav"
            suite_ambient = root / "ambient_false_accept.wav"
            _write_wav(suite_positive, amplitude=100)
            _write_wav(suite_ambient, amplitude=1200, sample_count=16000 * 10)
            suite_manifest = root / "suite_manifest.json"
            ambient_manifest = root / "ambient_manifest.json"
            suite_manifest.write_text(
                json.dumps([{"captured_audio_path": str(suite_positive), "label": "correct"}]) + "\n",
                encoding="utf-8",
            )
            ambient_manifest.write_text(
                json.dumps([{"captured_audio_path": str(suite_ambient), "label": "false_positive"}]) + "\n",
                encoding="utf-8",
            )
            spec_path = root / "promotion_spec.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "suites": [
                            {
                                "name": "critical16",
                                "manifest_path": str(suite_manifest),
                                "max_false_negatives": 0,
                            }
                        ],
                        "ambient_guards": [
                            {
                                "name": "ambient_longform",
                                "manifest_path": str(ambient_manifest),
                                "max_false_accepts_per_hour": 0.2,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig(
                project_root=temp_dir,
                wakeword_enabled=True,
                wakeword_primary_backend="openwakeword",
                wakeword_fallback_backend="stt",
                wakeword_verifier_mode="disabled",
                wakeword_phrases=("twinr",),
                wakeword_openwakeword_models=("twinr",),
                wakeword_openwakeword_threshold=0.08,
                wakeword_openwakeword_patience_frames=1,
                wakeword_openwakeword_activation_samples=1,
                audio_sample_rate=16000,
                audio_channels=1,
                audio_speech_threshold=400,
                wakeword_attempt_cooldown_s=10.0,
            )

            report = run_wakeword_promotion_eval(
                config=config,
                spec_path=spec_path,
                backend=None,
                model_factory=_model_factory,
            )
            report_path_exists = bool(report.report_path and report.report_path.exists())

        self.assertFalse(report.passed)
        self.assertTrue(any("critical16" in blocker for blocker in report.blockers))
        self.assertTrue(any("ambient_longform" in blocker for blocker in report.blockers))
        self.assertIsNotNone(report.report_path)
        self.assertTrue(report_path_exists)

    def test_run_wakeword_promotion_eval_replaces_read_only_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite_positive = root / "suite_positive.wav"
            _write_wav(suite_positive, amplitude=1200)
            suite_manifest = root / "suite_manifest.json"
            suite_manifest.write_text(
                json.dumps([{"captured_audio_path": str(suite_positive), "label": "correct"}]) + "\n",
                encoding="utf-8",
            )
            spec_path = root / "promotion_spec.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "suites": [
                            {
                                "name": "critical16",
                                "manifest_path": str(suite_manifest),
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            existing_report_path = root / "artifacts" / "ops" / "wakeword_eval" / "latest_promotion.json"
            existing_report_path.parent.mkdir(parents=True, exist_ok=True)
            existing_report_path.write_text('{"stale": true}\n', encoding="utf-8")
            os.chmod(existing_report_path, 0o444)
            config = TwinrConfig(
                project_root=temp_dir,
                wakeword_enabled=True,
                wakeword_primary_backend="openwakeword",
                wakeword_fallback_backend="stt",
                wakeword_verifier_mode="disabled",
                wakeword_phrases=("twinr",),
                wakeword_openwakeword_models=("twinr",),
                wakeword_openwakeword_threshold=0.08,
                wakeword_openwakeword_patience_frames=1,
                wakeword_openwakeword_activation_samples=1,
                audio_sample_rate=16000,
                audio_channels=1,
                audio_speech_threshold=400,
                wakeword_attempt_cooldown_s=10.0,
            )

            report = run_wakeword_promotion_eval(
                config=config,
                spec_path=spec_path,
                backend=None,
                model_factory=_model_factory,
            )

            payload = json.loads(existing_report_path.read_text(encoding="utf-8"))

        self.assertTrue(report.passed)
        self.assertEqual(report.report_path, existing_report_path)
        self.assertNotIn("stale", payload)
        self.assertEqual(payload["suites"][0]["metrics"]["true_positive"], 1)
