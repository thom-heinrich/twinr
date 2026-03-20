"""Run Twinr wakeword detection through a streaming KWS engine.

This module wraps ``sherpa-onnx`` keyword spotting for Twinr's wakeword
boundary. It keeps KWS-specific asset validation and streaming state here so
runtime/service, promotion, and eval paths can stay backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from threading import RLock
from typing import Any, Callable

import numpy as np

from twinr.hardware.audio import AmbientAudioCaptureWindow

from .matching import WakewordMatch, phrase_from_detector_label

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WakewordKwsAssetBundle:
    """Describe one resolved sherpa-onnx keyword-spotter asset bundle."""

    tokens_path: str
    encoder_path: str
    decoder_path: str
    joiner_path: str
    keywords_file_path: str


def _normalize_nonempty_text(name: str, value: object | None) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string.")
    return text


def _normalize_nonempty_strings(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values or ():
        text = str(value).strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _resolve_existing_file(
    path_value: object | None,
    *,
    name: str,
    project_root: str | Path | None,
) -> str:
    raw_path = _normalize_nonempty_text(name, path_value)
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute() and project_root is not None:
        candidate = Path(project_root).expanduser().resolve(strict=False) / candidate
    resolved = candidate.resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return str(resolved)


def _coerce_positive_int(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc
    if normalized < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return normalized


def _coerce_nonnegative_float(name: str, value: object, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a float >= {minimum}.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float >= {minimum}.") from exc
    if normalized < minimum:
        raise ValueError(f"{name} must be a float >= {minimum}.")
    return normalized


def _pcm16_to_float32_mono_samples(pcm_bytes: bytes, *, channels: int) -> np.ndarray:
    validated_channels = _coerce_positive_int("channels", channels)
    usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    samples = np.frombuffer(pcm_bytes[:usable], dtype=np.int16)
    usable_samples = len(samples) - (len(samples) % validated_channels)
    if usable_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    samples = samples[:usable_samples]
    if validated_channels > 1:
        samples = (
            samples.reshape(-1, validated_channels)
            .astype(np.int32)
            .mean(axis=1)
            .round()
            .clip(-32768, 32767)
            .astype(np.int16)
        )
    return samples.astype(np.float32) / 32768.0


def _default_keyword_spotter_factory(**kwargs):
    import sherpa_onnx

    return sherpa_onnx.KeywordSpotter(**kwargs)


def _empty_match(*, backend: str, detector_label: str | None = None) -> WakewordMatch:
    return WakewordMatch(
        detected=False,
        transcript="",
        normalized_transcript="",
        backend=backend,
        detector_label=detector_label,
        score=None,
    )


class WakewordSherpaOnnxFrameSpotter:
    """Detect wakewords from fixed PCM frames via a streaming sherpa KWS stream."""

    _FRAME_MS = 100

    def __init__(
        self,
        *,
        tokens_path: str,
        encoder_path: str,
        decoder_path: str,
        joiner_path: str,
        keywords_file_path: str,
        phrases: tuple[str, ...] | list[str],
        project_root: str | Path | None = None,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        max_active_paths: int = 4,
        keywords_score: float = 1.0,
        keywords_threshold: float = 0.25,
        num_trailing_blanks: int = 1,
        num_threads: int = 2,
        provider: str = "cpu",
        keyword_spotter_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.assets = WakewordKwsAssetBundle(
            tokens_path=_resolve_existing_file(tokens_path, name="tokens_path", project_root=project_root),
            encoder_path=_resolve_existing_file(encoder_path, name="encoder_path", project_root=project_root),
            decoder_path=_resolve_existing_file(decoder_path, name="decoder_path", project_root=project_root),
            joiner_path=_resolve_existing_file(joiner_path, name="joiner_path", project_root=project_root),
            keywords_file_path=_resolve_existing_file(
                keywords_file_path,
                name="keywords_file_path",
                project_root=project_root,
            ),
        )
        self.phrases = _normalize_nonempty_strings(phrases)
        if not self.phrases:
            raise ValueError("phrases must contain at least one wakeword phrase.")
        self.sample_rate = _coerce_positive_int("sample_rate", sample_rate)
        self.feature_dim = _coerce_positive_int("feature_dim", feature_dim)
        self.max_active_paths = _coerce_positive_int("max_active_paths", max_active_paths)
        self.keywords_score = _coerce_nonnegative_float("keywords_score", keywords_score)
        self.keywords_threshold = _coerce_nonnegative_float("keywords_threshold", keywords_threshold)
        self.num_trailing_blanks = _coerce_positive_int(
            "num_trailing_blanks",
            num_trailing_blanks,
        )
        self.num_threads = _coerce_positive_int("num_threads", num_threads)
        self.provider = _normalize_nonempty_text("provider", provider).lower()
        self._samples_per_frame = max(1, int(round(self.sample_rate * (self._FRAME_MS / 1000.0))))
        self._frame_buffer = bytearray()
        self._buffer_channels: int | None = None
        self._lock = RLock()
        factory = keyword_spotter_factory or _default_keyword_spotter_factory
        try:
            self._keyword_spotter = factory(
                tokens=self.assets.tokens_path,
                encoder=self.assets.encoder_path,
                decoder=self.assets.decoder_path,
                joiner=self.assets.joiner_path,
                keywords_file=self.assets.keywords_file_path,
                num_threads=self.num_threads,
                sample_rate=self.sample_rate,
                feature_dim=self.feature_dim,
                max_active_paths=self.max_active_paths,
                keywords_score=self.keywords_score,
                keywords_threshold=self.keywords_threshold,
                num_trailing_blanks=self.num_trailing_blanks,
                provider=self.provider,
            )
        except Exception as exc:  # pragma: no cover - depends on external package/runtime
            raise RuntimeError(
                "Failed to initialize sherpa-onnx keyword spotter."
            ) from exc
        self._stream = self._keyword_spotter.create_stream()

    @property
    def frame_bytes(self) -> int:
        return self.frame_bytes_for_channels(self._buffer_channels or 1)

    def frame_bytes_for_channels(self, channels: int) -> int:
        return self._samples_per_frame * 2 * _coerce_positive_int("channels", channels)

    def reset(self) -> None:
        with self._lock:
            self._frame_buffer.clear()
            self._buffer_channels = None
            try:
                self._stream = self._keyword_spotter.create_stream()
            except Exception:  # pragma: no cover - depends on external package/runtime
                LOGGER.exception("Failed to reset sherpa-onnx stream; keeping existing KWS stream.")

    def process_pcm_bytes(self, pcm_bytes: bytes, *, channels: int = 1) -> WakewordMatch | None:
        if not pcm_bytes:
            return None
        validated_channels = _coerce_positive_int("channels", channels)
        with self._lock:
            if self._buffer_channels is None:
                self._buffer_channels = validated_channels
            elif self._buffer_channels != validated_channels:
                LOGGER.warning(
                    "PCM channel count changed from %s to %s; resetting KWS stream state.",
                    self._buffer_channels,
                    validated_channels,
                )
                self.reset()
                self._buffer_channels = validated_channels
            self._frame_buffer.extend(pcm_bytes)
            detected_match: WakewordMatch | None = None
            frame_bytes = self.frame_bytes_for_channels(validated_channels)
            while len(self._frame_buffer) >= frame_bytes:
                frame = bytes(self._frame_buffer[:frame_bytes])
                del self._frame_buffer[:frame_bytes]
                detected_match = self.process_pcm_frame(frame, channels=validated_channels) or detected_match
            return detected_match

    def process_pcm_frame(self, pcm_frame: bytes, *, channels: int = 1) -> WakewordMatch | None:
        if not pcm_frame:
            return None
        validated_channels = _coerce_positive_int("channels", channels)
        required_bytes = self.frame_bytes_for_channels(validated_channels)
        if len(pcm_frame) < required_bytes:
            return None
        if len(pcm_frame) > required_bytes:
            pcm_frame = pcm_frame[:required_bytes]
        samples = _pcm16_to_float32_mono_samples(pcm_frame, channels=validated_channels)
        if samples.size == 0:
            return None
        with self._lock:
            try:
                self._stream.accept_waveform(self.sample_rate, samples)
                while self._keyword_spotter.is_ready(self._stream):
                    self._keyword_spotter.decode_stream(self._stream)
                    result = str(self._keyword_spotter.get_result(self._stream) or "").strip()
                    if not result:
                        continue
                    self._keyword_spotter.reset_stream(self._stream)
                    matched_phrase = phrase_from_detector_label(result, phrases=self.phrases)
                    return WakewordMatch(
                        detected=True,
                        transcript="",
                        matched_phrase=matched_phrase,
                        remaining_text="",
                        normalized_transcript="",
                        backend="kws",
                        detector_label=result,
                        score=None,
                    )
            except Exception:  # pragma: no cover - depends on external package/runtime
                LOGGER.exception("sherpa-onnx frame spotting failed.")
                self.reset()
                return None
        return None


class WakewordSherpaOnnxSpotter:
    """Detect wakewords from one buffered capture via sherpa-onnx KWS."""

    def __init__(
        self,
        *,
        tokens_path: str,
        encoder_path: str,
        decoder_path: str,
        joiner_path: str,
        keywords_file_path: str,
        phrases: tuple[str, ...] | list[str],
        project_root: str | Path | None = None,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        max_active_paths: int = 4,
        keywords_score: float = 1.0,
        keywords_threshold: float = 0.25,
        num_trailing_blanks: int = 1,
        num_threads: int = 2,
        provider: str = "cpu",
        keyword_spotter_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.assets = WakewordKwsAssetBundle(
            tokens_path=_resolve_existing_file(tokens_path, name="tokens_path", project_root=project_root),
            encoder_path=_resolve_existing_file(encoder_path, name="encoder_path", project_root=project_root),
            decoder_path=_resolve_existing_file(decoder_path, name="decoder_path", project_root=project_root),
            joiner_path=_resolve_existing_file(joiner_path, name="joiner_path", project_root=project_root),
            keywords_file_path=_resolve_existing_file(
                keywords_file_path,
                name="keywords_file_path",
                project_root=project_root,
            ),
        )
        self.phrases = _normalize_nonempty_strings(phrases)
        if not self.phrases:
            raise ValueError("phrases must contain at least one wakeword phrase.")
        self.sample_rate = _coerce_positive_int("sample_rate", sample_rate)
        self.feature_dim = _coerce_positive_int("feature_dim", feature_dim)
        self.max_active_paths = _coerce_positive_int("max_active_paths", max_active_paths)
        self.keywords_score = _coerce_nonnegative_float("keywords_score", keywords_score)
        self.keywords_threshold = _coerce_nonnegative_float("keywords_threshold", keywords_threshold)
        self.num_trailing_blanks = _coerce_positive_int(
            "num_trailing_blanks",
            num_trailing_blanks,
        )
        self.num_threads = _coerce_positive_int("num_threads", num_threads)
        self.provider = _normalize_nonempty_text("provider", provider).lower()
        factory = keyword_spotter_factory or _default_keyword_spotter_factory
        try:
            self._keyword_spotter = factory(
                tokens=self.assets.tokens_path,
                encoder=self.assets.encoder_path,
                decoder=self.assets.decoder_path,
                joiner=self.assets.joiner_path,
                keywords_file=self.assets.keywords_file_path,
                num_threads=self.num_threads,
                sample_rate=self.sample_rate,
                feature_dim=self.feature_dim,
                max_active_paths=self.max_active_paths,
                keywords_score=self.keywords_score,
                keywords_threshold=self.keywords_threshold,
                num_trailing_blanks=self.num_trailing_blanks,
                provider=self.provider,
            )
        except Exception as exc:  # pragma: no cover - depends on external package/runtime
            raise RuntimeError(
                "Failed to initialize sherpa-onnx keyword spotter."
            ) from exc

    def detect(self, capture: AmbientAudioCaptureWindow) -> WakewordMatch:
        pcm_bytes = capture.pcm_bytes or b""
        if not pcm_bytes:
            return _empty_match(backend="kws")
        try:
            channels = _coerce_positive_int("channels", capture.channels)
            sample_rate = _coerce_positive_int("sample_rate", capture.sample_rate)
        except ValueError:
            LOGGER.warning(
                "Ignoring KWS capture with invalid audio metadata: sample_rate=%r channels=%r",
                capture.sample_rate,
                capture.channels,
            )
            return _empty_match(backend="kws")
        samples = _pcm16_to_float32_mono_samples(pcm_bytes, channels=channels)
        if samples.size == 0:
            return _empty_match(backend="kws")
        try:
            stream = self._keyword_spotter.create_stream()
            stream.accept_waveform(sample_rate, samples)
            tail_padding = np.zeros(int(round(0.66 * sample_rate)), dtype=np.float32)
            if tail_padding.size:
                stream.accept_waveform(sample_rate, tail_padding)
            if hasattr(stream, "input_finished"):
                stream.input_finished()
            while self._keyword_spotter.is_ready(stream):
                self._keyword_spotter.decode_stream(stream)
                result = str(self._keyword_spotter.get_result(stream) or "").strip()
                if not result:
                    continue
                matched_phrase = phrase_from_detector_label(result, phrases=self.phrases)
                return WakewordMatch(
                    detected=True,
                    transcript="",
                    matched_phrase=matched_phrase,
                    remaining_text="",
                    normalized_transcript="",
                    backend="kws",
                    detector_label=result,
                    score=None,
                )
        except Exception:  # pragma: no cover - depends on external package/runtime
            LOGGER.exception("sherpa-onnx clip spotting failed.")
        return _empty_match(backend="kws")


__all__ = [
    "WakewordKwsAssetBundle",
    "WakewordSherpaOnnxFrameSpotter",
    "WakewordSherpaOnnxSpotter",
]
