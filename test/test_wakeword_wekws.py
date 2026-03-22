"""Tests for the Twinr WeKws ONNX wakeword backend."""

from __future__ import annotations

import json
from pathlib import Path
from struct import pack
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.audio import AmbientAudioCaptureWindow, AmbientAudioLevelSample
from twinr.proactive.wakeword.wekws import (
    WakewordWekwsFrameSpotter,
    WakewordWekwsSpotter,
)


def _pcm_frame(*, amplitude: int, sample_count: int = 1600) -> bytes:
    return b"".join(pack("<h", amplitude) for _ in range(sample_count))


def _capture(*, amplitude: int, sample_rate: int = 16000) -> AmbientAudioCaptureWindow:
    pcm_bytes = _pcm_frame(amplitude=amplitude, sample_count=sample_rate)
    return AmbientAudioCaptureWindow(
        sample=AmbientAudioLevelSample(
            duration_ms=1000,
            chunk_count=10,
            active_chunk_count=8 if amplitude else 0,
            average_rms=abs(amplitude),
            peak_rms=abs(amplitude),
            active_ratio=0.8 if amplitude else 0.0,
        ),
        pcm_bytes=pcm_bytes,
        sample_rate=sample_rate,
        channels=1,
    )


def _touch_wekws_bundle(
    root: Path,
    *,
    keyword_token: str = "twinna",
    cmvn_mean_value: float = 0.0,
    cmvn_var_value: float = 1.0,
) -> dict[str, Path]:
    model_path = root / "avg_30.onnx"
    config_path = root / "config.yaml"
    words_path = root / "words.txt"
    cmvn_path = root / "global_cmvn.json"
    model_path.write_text("onnx\n", encoding="utf-8")
    config_path.write_text(
        "\n".join(
            [
                "dataset_conf:",
                "  feats_type: fbank",
                "  resample_conf:",
                "    resample_rate: 16000",
                "  fbank_conf:",
                "    num_mel_bins: 80",
                "    frame_shift: 10",
                "    frame_length: 25",
                "model:",
                "  cmvn:",
                f"    cmvn_file: {cmvn_path}",
                "    norm_var: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    words_path.write_text(f"<FILLER>\n{keyword_token}\n", encoding="utf-8")
    cmvn_path.write_text(
        json.dumps(
            {
                "mean_stat": [cmvn_mean_value] * 80,
                "var_stat": [cmvn_var_value] * 80,
                "frame_num": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "model": model_path,
        "config": config_path,
        "words": words_path,
        "cmvn": cmvn_path,
    }


class _FakeModelMeta:
    def __init__(
        self,
        *,
        cache_dim: int = 4,
        cache_len: int = 3,
        cmvn_mode: str | None = None,
    ) -> None:
        self.custom_metadata_map = {
            "cache_dim": str(cache_dim),
            "cache_len": str(cache_len),
        }
        if cmvn_mode is not None:
            self.custom_metadata_map["cmvn_mode"] = str(cmvn_mode)


class _FakeSession:
    def __init__(self, outputs: list[np.ndarray], *, meta: _FakeModelMeta | None = None) -> None:
        self._outputs = list(outputs)
        self._meta = meta or _FakeModelMeta()
        self.feeds: list[dict[str, np.ndarray]] = []

    def get_modelmeta(self):
        return self._meta

    def run(self, _output_names, feeds):
        self.feeds.append({name: np.array(value, copy=True) for name, value in feeds.items()})
        score = self._outputs.pop(0) if self._outputs else np.zeros((1, 1, 1), dtype=np.float32)
        cache = np.array(feeds["cache"], copy=True)
        return [score, cache]


class WakewordWekwsTests(unittest.TestCase):
    def test_frame_spotter_detects_streaming_hit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = _touch_wekws_bundle(Path(temp_dir), keyword_token="twinna")
            fake_session = _FakeSession(
                [
                    np.array([[[0.10]]], dtype=np.float32),
                    np.array([[[0.72]]], dtype=np.float32),
                ]
            )
            extractor_calls: list[int] = []

            def _feature_extractor(samples, **kwargs):
                extractor_calls.append(int(np.max(np.abs(samples))) if samples.size else 0)
                return np.ones((1, kwargs["feature_dim"]), dtype=np.float32)

            spotter = WakewordWekwsFrameSpotter(
                model_path=str(bundle["model"]),
                config_path=str(bundle["config"]),
                words_path=str(bundle["words"]),
                cmvn_path=str(bundle["cmvn"]),
                phrases=("twinna", "twinr"),
                session_factory=lambda **_kwargs: fake_session,
                feature_extractor=_feature_extractor,
            )

            quiet = spotter.process_pcm_bytes(_pcm_frame(amplitude=400), channels=1)
            loud = spotter.process_pcm_bytes(_pcm_frame(amplitude=16000), channels=1)

        self.assertIsNone(quiet)
        self.assertIsNotNone(loud)
        self.assertTrue(loud.detected)
        self.assertEqual(loud.backend, "wekws")
        self.assertEqual(loud.detector_label, "twinna")
        self.assertEqual(loud.matched_phrase, "twinna")
        self.assertGreater(float(loud.score or 0.0), 0.7)
        self.assertEqual(extractor_calls, [400, 16000])

    def test_clip_spotter_detects_capture_hit_and_passes_source_sample_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = _touch_wekws_bundle(Path(temp_dir), keyword_token="<TWINR_FAMILY>")
            fake_session = _FakeSession([np.array([[[0.81]]], dtype=np.float32)])
            extractor_sample_rates: list[int] = []

            def _feature_extractor(samples, **kwargs):
                extractor_sample_rates.append(int(kwargs["source_sample_rate"]))
                return np.ones((2, kwargs["feature_dim"]), dtype=np.float32)

            spotter = WakewordWekwsSpotter(
                model_path=str(bundle["model"]),
                config_path=str(bundle["config"]),
                words_path=str(bundle["words"]),
                cmvn_path=str(bundle["cmvn"]),
                phrases=("twinna", "twinr"),
                session_factory=lambda **_kwargs: fake_session,
                feature_extractor=_feature_extractor,
            )

            match = spotter.detect(_capture(amplitude=16000, sample_rate=8000))

        self.assertTrue(match.detected)
        self.assertEqual(match.backend, "wekws")
        self.assertEqual(match.detector_label, "<TWINR_FAMILY>")
        self.assertEqual(match.matched_phrase, "twinr")
        self.assertEqual(extractor_sample_rates, [8000])

    def test_default_feature_stack_runs_without_torch_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = _touch_wekws_bundle(Path(temp_dir), keyword_token="twinna")
            fake_session = _FakeSession([np.array([[[0.83]]], dtype=np.float32)])
            spotter = WakewordWekwsSpotter(
                model_path=str(bundle["model"]),
                config_path=str(bundle["config"]),
                words_path=str(bundle["words"]),
                cmvn_path=str(bundle["cmvn"]),
                phrases=("twinna", "twinr"),
                session_factory=lambda **_kwargs: fake_session,
            )

            match = spotter.detect(_capture(amplitude=16000))

        self.assertTrue(match.detected)
        self.assertEqual(match.backend, "wekws")
        self.assertEqual(match.matched_phrase, "twinna")

    def test_wekws_spotter_requires_existing_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = _touch_wekws_bundle(root)
            missing_words = root / "missing_words.txt"

            with self.assertRaises(FileNotFoundError):
                WakewordWekwsSpotter(
                    model_path=str(bundle["model"]),
                    config_path=str(bundle["config"]),
                    words_path=str(missing_words),
                    cmvn_path=str(bundle["cmvn"]),
                    phrases=("twinna",),
                    session_factory=lambda **_kwargs: _FakeSession([]),
                    feature_extractor=lambda samples, **kwargs: np.zeros((0, kwargs["feature_dim"]), dtype=np.float32),
                )

    def test_auto_cmvn_mode_skips_legacy_sidecar_stats_for_embedded_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = _touch_wekws_bundle(
                Path(temp_dir),
                cmvn_mean_value=1.0,
                cmvn_var_value=2.0,
            )
            fake_session = _FakeSession(
                [np.array([[[0.81]]], dtype=np.float32)],
                meta=_FakeModelMeta(),
            )
            spotter = WakewordWekwsSpotter(
                model_path=str(bundle["model"]),
                config_path=str(bundle["config"]),
                words_path=str(bundle["words"]),
                cmvn_path=str(bundle["cmvn"]),
                phrases=("twinna",),
                session_factory=lambda **_kwargs: fake_session,
                feature_extractor=lambda samples, **kwargs: np.ones((2, kwargs["feature_dim"]), dtype=np.float32),
            )

            score, detector_label = spotter.score_capture(_capture(amplitude=16000))

        self.assertEqual(detector_label, "twinna")
        self.assertGreater(score, 0.8)
        observed = fake_session.feeds[0]["input"][0]
        self.assertTrue(np.allclose(observed, 1.0))
        self.assertEqual(spotter.model_config.cmvn_mode, "embedded")

    def test_explicit_external_cmvn_mode_applies_sidecar_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = _touch_wekws_bundle(
                Path(temp_dir),
                cmvn_mean_value=1.0,
                cmvn_var_value=2.0,
            )
            fake_session = _FakeSession([np.array([[[0.81]]], dtype=np.float32)])
            spotter = WakewordWekwsSpotter(
                model_path=str(bundle["model"]),
                config_path=str(bundle["config"]),
                words_path=str(bundle["words"]),
                cmvn_path=str(bundle["cmvn"]),
                cmvn_mode="external",
                phrases=("twinna",),
                session_factory=lambda **_kwargs: fake_session,
                feature_extractor=lambda samples, **kwargs: np.ones((2, kwargs["feature_dim"]), dtype=np.float32),
            )

            score, detector_label = spotter.score_capture(_capture(amplitude=16000))

        self.assertEqual(detector_label, "twinna")
        self.assertGreater(score, 0.8)
        observed = fake_session.feeds[0]["input"][0]
        self.assertTrue(np.allclose(observed, 0.0))
        self.assertEqual(spotter.model_config.cmvn_mode, "external")


if __name__ == "__main__":
    unittest.main()
