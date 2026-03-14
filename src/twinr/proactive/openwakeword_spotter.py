from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable

from twinr.hardware.audio import AmbientAudioCaptureWindow, pcm16_to_wav_bytes
from twinr.proactive.wakeword import (
    WakewordMatch,
    match_wakeword_transcript,
    phrase_from_detector_label,
    wakeword_primary_prompt,
)
from twinr.providers.openai.backend import OpenAIBackend


@dataclass(frozen=True, slots=True)
class OpenWakeWordPrediction:
    label: str | None
    score: float


class WakewordOpenWakeWordFrameSpotter:
    _FRAME_SAMPLES = 1280

    def __init__(
        self,
        *,
        wakeword_models: tuple[str, ...],
        phrases: tuple[str, ...],
        threshold: float = 0.5,
        vad_threshold: float = 0.0,
        patience_frames: int = 1,
        activation_samples: int = 1,
        deactivation_threshold: float = 0.0,
        enable_speex_noise_suppression: bool = False,
        inference_framework: str = "tflite",
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        normalized_models = tuple(model for model in wakeword_models if str(model).strip())
        if not normalized_models:
            raise ValueError("openWakeWord requires at least one configured model name or path.")
        self.wakeword_models = normalized_models
        self.phrases = tuple(phrase for phrase in phrases if str(phrase).strip())
        self.threshold = max(0.0, min(1.0, float(threshold)))
        self.vad_threshold = max(0.0, min(1.0, float(vad_threshold)))
        self.patience_frames = max(1, int(patience_frames))
        self.activation_samples = max(1, int(activation_samples))
        self.deactivation_threshold = max(0.0, min(1.0, float(deactivation_threshold)))
        self.inference_framework = (inference_framework or "tflite").strip().lower() or "tflite"
        factory = model_factory or _default_model_factory
        self._model = factory(
            wakeword_models=list(normalized_models),
            vad_threshold=self.vad_threshold,
            enable_speex_noise_suppression=enable_speex_noise_suppression,
            inference_framework=self.inference_framework,
        )
        self._frame_buffer = bytearray()
        self._consecutive_frames: dict[str, int] = defaultdict(int)
        self._score_history: dict[str, deque[float]] = {}
        self._latched_labels: set[str] = set()

    @property
    def frame_bytes(self) -> int:
        return self._FRAME_SAMPLES * 2

    def reset(self) -> None:
        self._frame_buffer.clear()
        self._consecutive_frames.clear()
        self._score_history.clear()
        self._latched_labels.clear()
        if hasattr(self._model, "reset"):
            self._model.reset()

    def process_pcm_bytes(self, pcm_bytes: bytes, *, channels: int = 1) -> WakewordMatch | None:
        if not pcm_bytes:
            return None
        self._frame_buffer.extend(pcm_bytes)
        detected_match: WakewordMatch | None = None
        while len(self._frame_buffer) >= self.frame_bytes:
            frame = bytes(self._frame_buffer[: self.frame_bytes])
            del self._frame_buffer[: self.frame_bytes]
            detected_match = self.process_pcm_frame(frame, channels=channels) or detected_match
        return detected_match

    def process_pcm_frame(self, pcm_frame: bytes, *, channels: int = 1) -> WakewordMatch | None:
        if not pcm_frame:
            return None
        samples = _pcm16_to_int16_samples(pcm_frame, channels=channels)
        if len(samples) == 0:
            return None
        predictions = self._model.predict(samples)
        best_prediction = self._thresholded_stream_prediction(predictions)
        if best_prediction.score <= 0.0:
            return None
        matched_phrase = phrase_from_detector_label(best_prediction.label, phrases=self.phrases)
        return WakewordMatch(
            detected=True,
            transcript="",
            matched_phrase=matched_phrase,
            remaining_text="",
            normalized_transcript="",
            backend="openwakeword",
            detector_label=best_prediction.label,
            score=best_prediction.score,
        )

    def _thresholded_stream_prediction(self, predictions: dict[str, float] | None) -> OpenWakeWordPrediction:
        if not predictions:
            return OpenWakeWordPrediction(label=None, score=0.0)
        thresholded_best = OpenWakeWordPrediction(label=None, score=0.0)
        current_labels = set(self._consecutive_frames.keys()) | {str(label) for label in predictions.keys()}
        for label in current_labels:
            score = float(predictions.get(label, 0.0) or 0.0)
            history = self._score_history.setdefault(
                label,
                deque(maxlen=self.activation_samples),
            )
            history.append(score)
            if len(history) < self.activation_samples:
                if score >= self.threshold:
                    self._consecutive_frames[label] = self._consecutive_frames.get(label, 0) + 1
                else:
                    self._consecutive_frames[label] = 0
                continue
            average_score = sum(history) / len(history)
            if label in self._latched_labels:
                if average_score <= self.deactivation_threshold:
                    self._latched_labels.discard(label)
                continue
            if score >= self.threshold:
                self._consecutive_frames[label] = self._consecutive_frames.get(label, 0) + 1
            else:
                self._consecutive_frames[label] = 0
            if (
                average_score >= self.threshold
                and self._consecutive_frames[label] >= self.patience_frames
                and average_score > thresholded_best.score
            ):
                thresholded_best = OpenWakeWordPrediction(label=label, score=average_score)
        if thresholded_best.label is not None:
            self._latched_labels.add(thresholded_best.label)
        return thresholded_best


class WakewordOpenWakeWordSpotter:
    _CLIP_PADDING_S = 1
    _CHUNK_SIZE_SAMPLES = 1280

    def __init__(
        self,
        *,
        wakeword_models: tuple[str, ...],
        phrases: tuple[str, ...],
        threshold: float = 0.5,
        vad_threshold: float = 0.0,
        patience_frames: int = 1,
        activation_samples: int = 1,
        deactivation_threshold: float = 0.0,
        enable_speex_noise_suppression: bool = False,
        inference_framework: str = "tflite",
        backend: OpenAIBackend | None = None,
        language: str | None = None,
        transcribe_on_detect: bool = True,
        min_prefix_ratio: float = 0.9,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        normalized_models = tuple(model for model in wakeword_models if str(model).strip())
        if not normalized_models:
            raise ValueError("openWakeWord requires at least one configured model name or path.")
        self.wakeword_models = normalized_models
        self.phrases = tuple(phrase for phrase in phrases if str(phrase).strip())
        self.threshold = max(0.0, min(1.0, float(threshold)))
        self.vad_threshold = max(0.0, min(1.0, float(vad_threshold)))
        self.patience_frames = max(1, int(patience_frames))
        self.activation_samples = max(1, int(activation_samples))
        self.deactivation_threshold = max(0.0, min(1.0, float(deactivation_threshold)))
        self.inference_framework = (inference_framework or "tflite").strip().lower() or "tflite"
        self.backend = backend
        self.language = (language or "").strip() or None
        self.transcribe_on_detect = bool(transcribe_on_detect)
        self.min_prefix_ratio = max(0.5, min(1.0, float(min_prefix_ratio)))
        factory = model_factory or _default_model_factory
        self._model = factory(
            wakeword_models=list(normalized_models),
            vad_threshold=self.vad_threshold,
            enable_speex_noise_suppression=enable_speex_noise_suppression,
            inference_framework=self.inference_framework,
        )
        model_names = getattr(self._model, "models", {})
        names = tuple(str(name) for name in model_names.keys()) if hasattr(model_names, "keys") else normalized_models
        self._threshold_by_model = {name: self.threshold for name in names}
        self._patience_by_model = {name: self.patience_frames for name in names}

    def detect(self, capture: AmbientAudioCaptureWindow) -> WakewordMatch:
        pcm_bytes = capture.pcm_bytes or b""
        if not pcm_bytes:
            return WakewordMatch(detected=False, transcript="", backend="openwakeword")
        if capture.sample_rate != 16000:
            raise ValueError("openWakeWord requires 16 kHz input audio. Set TWINR_AUDIO_SAMPLE_RATE=16000.")
        samples = _pcm16_to_int16_samples(pcm_bytes, channels=capture.channels)
        if len(samples) == 0:
            return WakewordMatch(detected=False, transcript="", backend="openwakeword")
        if hasattr(self._model, "reset"):
            self._model.reset()
        best_prediction = self._detect_peak_prediction(samples)
        matched_phrase = phrase_from_detector_label(best_prediction.label, phrases=self.phrases)
        if best_prediction.score < self.threshold:
            return WakewordMatch(
                detected=False,
                transcript="",
                matched_phrase=matched_phrase,
                backend="openwakeword",
                detector_label=best_prediction.label,
                score=best_prediction.score,
            )

        transcript = ""
        normalized_transcript = ""
        remaining_text = ""
        if self.backend is not None and self.transcribe_on_detect:
            transcript = self._transcribe(capture)
            transcript_match = match_wakeword_transcript(
                transcript,
                phrases=self.phrases,
                min_prefix_ratio=self.min_prefix_ratio,
                backend="openwakeword",
                detector_label=best_prediction.label,
                score=best_prediction.score,
            )
            if transcript_match.detected:
                return transcript_match
            normalized_transcript = transcript_match.normalized_transcript
            return WakewordMatch(
                detected=False,
                transcript=transcript,
                matched_phrase=matched_phrase,
                remaining_text="",
                normalized_transcript=normalized_transcript,
                backend="openwakeword",
                detector_label=best_prediction.label,
                score=best_prediction.score,
            )

        return WakewordMatch(
            detected=True,
            transcript=transcript,
            matched_phrase=matched_phrase,
            remaining_text=remaining_text,
            normalized_transcript=normalized_transcript,
            backend="openwakeword",
            detector_label=best_prediction.label,
            score=best_prediction.score,
        )

    def _detect_peak_prediction(self, samples) -> OpenWakeWordPrediction:
        if hasattr(self._model, "predict_clip"):
            clip_predictions = self._model.predict_clip(
                samples,
                padding=self._CLIP_PADDING_S,
                chunk_size=self._CHUNK_SIZE_SAMPLES,
            )
            return _best_prediction_sequence(
                clip_predictions,
                threshold=self.threshold,
                patience_frames=self.patience_frames,
                activation_samples=self.activation_samples,
                deactivation_threshold=self.deactivation_threshold,
            )

        predictions = self._model.predict(samples)
        return _best_prediction(predictions)

    def _transcribe(self, capture: AmbientAudioCaptureWindow) -> str:
        if self.backend is None:
            return ""
        audio_bytes = pcm16_to_wav_bytes(
            capture.pcm_bytes,
            sample_rate=capture.sample_rate,
            channels=capture.channels,
        )
        primary_prompt = wakeword_primary_prompt(self.phrases)
        transcript = self.backend.transcribe(
            audio_bytes,
            filename="wakeword-openwakeword.wav",
            content_type="audio/wav",
            language=self.language,
            prompt=primary_prompt,
        )
        return transcript


def _default_model_factory(**kwargs):
    import os

    import openwakeword.utils
    from openwakeword.model import Model

    model_names = [str(item) for item in kwargs.get("wakeword_models", []) if str(item).strip()]
    named_models = [item for item in model_names if not os.path.exists(item)]
    if named_models:
        openwakeword.utils.download_models(model_names=named_models)
    return Model(**kwargs)


def _pcm16_to_int16_samples(pcm_bytes: bytes, *, channels: int):
    import array
    import sys

    pcm_samples = array.array("h")
    pcm_samples.frombytes(pcm_bytes)
    if sys.byteorder != "little":
        pcm_samples.byteswap()
    try:
        import numpy as np
    except ModuleNotFoundError:
        if channels <= 1:
            return pcm_samples.tolist()
        usable = len(pcm_samples) - (len(pcm_samples) % channels)
        if usable <= 0:
            return []
        return [pcm_samples[index] for index in range(0, usable, channels)]
    if channels <= 1:
        return np.asarray(pcm_samples, dtype=np.int16)
    usable = len(pcm_samples) - (len(pcm_samples) % channels)
    if usable <= 0:
        return np.empty(0, dtype=np.int16)
    return np.asarray([pcm_samples[index] for index in range(0, usable, channels)], dtype=np.int16)


def _best_prediction(predictions: dict[str, float] | None) -> OpenWakeWordPrediction:
    if not predictions:
        return OpenWakeWordPrediction(label=None, score=0.0)
    best_label = None
    best_score = 0.0
    for label, raw_score in predictions.items():
        score = float(raw_score or 0.0)
        if score > best_score:
            best_label = str(label)
            best_score = score
    return OpenWakeWordPrediction(label=best_label, score=best_score)


def _best_prediction_sequence(
    predictions: list[dict[str, float]] | tuple[dict[str, float], ...] | None,
    *,
    threshold: float = 0.0,
    patience_frames: int = 1,
    activation_samples: int = 1,
    deactivation_threshold: float = 0.0,
) -> OpenWakeWordPrediction:
    if not predictions:
        return OpenWakeWordPrediction(label=None, score=0.0)
    best = OpenWakeWordPrediction(label=None, score=0.0)
    thresholded_best = OpenWakeWordPrediction(label=None, score=0.0)
    consecutive_frames: dict[str, int] = defaultdict(int)
    score_history: dict[str, deque[float]] = {}
    latched_labels: set[str] = set()
    for prediction in predictions:
        candidate = _best_prediction(prediction)
        if candidate.score > best.score:
            best = candidate
        current_labels = set(consecutive_frames.keys()) | {str(label) for label in prediction.keys()}
        for label in current_labels:
            score = float(prediction.get(label, 0.0) or 0.0)
            history = score_history.setdefault(
                label,
                deque(maxlen=max(1, int(activation_samples))),
            )
            history.append(score)
            if len(history) < max(1, int(activation_samples)):
                if score >= threshold:
                    consecutive_frames[label] += 1
                else:
                    consecutive_frames[label] = 0
                continue
            average_score = sum(history) / len(history)
            if label in latched_labels:
                if average_score <= deactivation_threshold:
                    latched_labels.discard(label)
                continue
            if score >= threshold:
                consecutive_frames[label] += 1
            else:
                consecutive_frames[label] = 0
            if (
                average_score >= threshold
                and consecutive_frames[label] >= patience_frames
                and average_score > thresholded_best.score
            ):
                thresholded_best = OpenWakeWordPrediction(label=label, score=average_score)
                latched_labels.add(label)
    if thresholded_best.score > 0.0:
        return thresholded_best
    return best


__all__ = [
    "OpenWakeWordPrediction",
    "WakewordOpenWakeWordFrameSpotter",
    "WakewordOpenWakeWordSpotter",
]
