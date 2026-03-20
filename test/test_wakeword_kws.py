from pathlib import Path
from struct import pack
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.audio import AmbientAudioCaptureWindow, AmbientAudioLevelSample
from twinr.proactive.wakeword.kws import (
    WakewordSherpaOnnxFrameSpotter,
    WakewordSherpaOnnxSpotter,
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
        self.accepted_sample_rates: list[int] = []
        self.finished = False

    def accept_waveform(self, sample_rate, samples) -> None:
        self.accepted_sample_rates.append(int(sample_rate))
        peak = max((abs(float(sample)) for sample in samples), default=0.0)
        if peak >= 0.2:
            self.pending_results.append("twinna")

    def input_finished(self) -> None:
        self.finished = True


class FakeKeywordSpotter:
    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)
        self.reset_calls = 0
        self.streams: list[FakeKeywordStream] = []

    def create_stream(self):
        stream = FakeKeywordStream()
        self.streams.append(stream)
        return stream

    def is_ready(self, stream) -> bool:
        return bool(stream.pending_results)

    def decode_stream(self, stream) -> None:
        return None

    def get_result(self, stream) -> str:
        if not stream.pending_results:
            return ""
        return stream.pending_results.pop(0)

    def reset_stream(self, stream) -> None:
        self.reset_calls += 1
        stream.pending_results.clear()


class WakewordKwsTests(unittest.TestCase):
    def test_frame_spotter_detects_streaming_hit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = _touch_kws_bundle(Path(temp_dir))
            fake_keyword_spotter = FakeKeywordSpotter()
            spotter = WakewordSherpaOnnxFrameSpotter(
                tokens_path=str(bundle["tokens"]),
                encoder_path=str(bundle["encoder"]),
                decoder_path=str(bundle["decoder"]),
                joiner_path=str(bundle["joiner"]),
                keywords_file_path=str(bundle["keywords"]),
                phrases=("twinna", "twinr"),
                keyword_spotter_factory=lambda **_kwargs: fake_keyword_spotter,
            )

            quiet = spotter.process_pcm_bytes(_pcm_frame(amplitude=400), channels=1)
            loud = spotter.process_pcm_bytes(_pcm_frame(amplitude=16000), channels=1)

        self.assertIsNone(quiet)
        self.assertIsNotNone(loud)
        self.assertTrue(loud.detected)
        self.assertEqual(loud.backend, "kws")
        self.assertEqual(loud.detector_label, "twinna")
        self.assertEqual(loud.matched_phrase, "twinna")
        self.assertEqual(fake_keyword_spotter.reset_calls, 1)

    def test_clip_spotter_detects_capture_hit_and_forwards_sample_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = _touch_kws_bundle(Path(temp_dir))
            fake_keyword_spotter = FakeKeywordSpotter()
            spotter = WakewordSherpaOnnxSpotter(
                tokens_path=str(bundle["tokens"]),
                encoder_path=str(bundle["encoder"]),
                decoder_path=str(bundle["decoder"]),
                joiner_path=str(bundle["joiner"]),
                keywords_file_path=str(bundle["keywords"]),
                phrases=("twinna", "twinr"),
                keyword_spotter_factory=lambda **_kwargs: fake_keyword_spotter,
            )

            match = spotter.detect(_capture(amplitude=16000, sample_rate=8000))

        self.assertTrue(match.detected)
        self.assertEqual(match.backend, "kws")
        self.assertEqual(match.detector_label, "twinna")
        self.assertEqual(match.matched_phrase, "twinna")
        self.assertEqual(fake_keyword_spotter.streams[-1].accepted_sample_rates[0], 8000)
        self.assertTrue(fake_keyword_spotter.streams[-1].finished)

    def test_kws_spotter_requires_existing_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = _touch_kws_bundle(root)
            missing_keywords = root / "missing_keywords.txt"

            with self.assertRaises(FileNotFoundError):
                WakewordSherpaOnnxSpotter(
                    tokens_path=str(bundle["tokens"]),
                    encoder_path=str(bundle["encoder"]),
                    decoder_path=str(bundle["decoder"]),
                    joiner_path=str(bundle["joiner"]),
                    keywords_file_path=str(missing_keywords),
                    phrases=("twinna",),
                    keyword_spotter_factory=lambda **_kwargs: FakeKeywordSpotter(),
                )


if __name__ == "__main__":
    unittest.main()
