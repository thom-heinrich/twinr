"""Run openWakeWord clip and stream spotting for Twinr wakeword detection.

This module wraps openWakeWord model initialization, clip-level and frame-level
inference, optional STT confirmation, and resilient audio preprocessing so the
proactive runtime can fail closed instead of crashing on model or audio errors.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import logging
import math
from threading import RLock
from typing import Any, Callable

from twinr.hardware.audio import AmbientAudioCaptureWindow, pcm16_to_wav_bytes
from twinr.providers.openai import OpenAIBackend

from .matching import WakewordMatch, match_wakeword_transcript, phrase_from_detector_label, wakeword_primary_prompt

# AUDIT-FIX(#1): Zentrales Logging fuer sichere Degradation statt unhandled Exceptions im Hot Path.
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OpenWakeWordPrediction:
    """Describe one openWakeWord label and score pair."""

    label: str | None
    score: float


class WakewordOpenWakeWordFrameSpotter:
    """Detect wakeword hits from exact openWakeWord-sized PCM frames.

    This stateful spotter smooths frame scores across a live stream, latches
    active labels until deactivation, and returns ``WakewordMatch`` objects
    without transcribing the audio.
    """

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
        # AUDIT-FIX(#6): Konfig-Strings werden kanonisiert, damit Whitespace keine stillen Fehlkonfigurationen ausloest.
        normalized_models = _normalize_nonempty_strings(wakeword_models)
        if not normalized_models:
            raise ValueError("openWakeWord requires at least one configured model name or path.")
        self.wakeword_models = normalized_models
        # AUDIT-FIX(#6): Wakeword-Phrasen werden ebenfalls sauber normalisiert.
        self.phrases = _normalize_nonempty_strings(phrases)
        # AUDIT-FIX(#6): Probability-Konfig wird mit klarem Fehlerkontext validiert statt still auf unsichere Defaults zu fallen.
        self.threshold = _coerce_probability_config("threshold", threshold)
        self.vad_threshold = _coerce_probability_config("vad_threshold", vad_threshold)
        self.patience_frames = max(1, int(patience_frames))
        self.activation_samples = max(1, int(activation_samples))
        self.deactivation_threshold = _coerce_probability_config(
            "deactivation_threshold",
            deactivation_threshold,
        )
        self.inference_framework = (inference_framework or "tflite").strip().lower() or "tflite"
        factory = model_factory or _default_model_factory
        try:
            self._model = factory(
                wakeword_models=list(normalized_models),
                vad_threshold=self.vad_threshold,
                enable_speex_noise_suppression=enable_speex_noise_suppression,
                inference_framework=self.inference_framework,
            )
        except Exception as exc:  # pragma: no cover - depends on external package/runtime
            # AUDIT-FIX(#6): Modellinitialisierung liefert jetzt einen klaren operator-tauglichen Fehlerkontext.
            raise RuntimeError(
                f"Failed to initialize openWakeWord model(s): {', '.join(normalized_models)}"
            ) from exc
        self._frame_buffer = bytearray()
        self._consecutive_frames: dict[str, int] = defaultdict(int)
        self._score_history: dict[str, deque[float]] = {}
        self._latched_labels: set[str] = set()
        # AUDIT-FIX(#7): Stream-State und Modellzugriffe werden gegen threaded Reuse abgesichert.
        self._lock = RLock()
        # AUDIT-FIX(#2): Buffer merkt sich den Kanalmodus, damit Chunking bei Kanalwechseln nicht korrupt wird.
        self._buffer_channels: int | None = None

    @property
    def frame_bytes(self) -> int:
        """Return the frame size in bytes for the active channel mode."""

        # AUDIT-FIX(#2): Rueckgabe orientiert sich am zuletzt gesehenen Kanalmodus statt immer Mono anzunehmen.
        return self._frame_bytes_for_channels(self._buffer_channels or 1)

    # AUDIT-FIX(#2): Oeffentliche Hilfe fuer call sites, die die korrekte Frame-Groesse pro Kanal brauchen.
    def frame_bytes_for_channels(self, channels: int) -> int:
        """Return the frame size in bytes for a specific channel count."""

        validated_channels = _validate_positive_int(channels)
        if validated_channels is None:
            raise ValueError(f"channels must be a positive integer, got {channels!r}")
        return self._frame_bytes_for_channels(validated_channels)

    def _frame_bytes_for_channels(self, channels: int) -> int:
        return self._FRAME_SAMPLES * 2 * channels

    def reset(self) -> None:
        """Clear buffered state and reset the underlying model if supported."""

        with self._lock:
            self._frame_buffer.clear()
            self._consecutive_frames.clear()
            self._score_history.clear()
            self._latched_labels.clear()
            self._buffer_channels = None
            self._safe_reset_model()

    def _safe_reset_model(self) -> None:
        if hasattr(self._model, "reset"):
            try:
                self._model.reset()
            except Exception:  # pragma: no cover - depends on external package/runtime
                # AUDIT-FIX(#1): Modell-Reset darf den Wakeword-Loop nicht abwerfen.
                LOGGER.exception("openWakeWord frame model reset failed.")

    def process_pcm_bytes(self, pcm_bytes: bytes, *, channels: int = 1) -> WakewordMatch | None:
        """Feed arbitrary PCM bytes and emit the latest detected wakeword match."""

        if not pcm_bytes:
            return None
        validated_channels = _validate_positive_int(channels)
        if validated_channels is None:
            # AUDIT-FIX(#1): Ungueltige Kanalangaben werden abgefangen statt spaeter in Division/Array-Fehler zu laufen.
            LOGGER.warning("Ignoring PCM bytes with invalid channel count: %r", channels)
            return None
        with self._lock:
            if self._buffer_channels is None:
                self._buffer_channels = validated_channels
            elif self._buffer_channels != validated_channels:
                # AUDIT-FIX(#2): Kanalwechsel invalidiert den Byte-Buffer; wir resetten sauber statt Frames falsch zu zerschneiden.
                LOGGER.warning(
                    "PCM channel count changed from %s to %s; resetting wakeword stream state.",
                    self._buffer_channels,
                    validated_channels,
                )
                self.reset()
                self._buffer_channels = validated_channels
            self._frame_buffer.extend(pcm_bytes)
            detected_match: WakewordMatch | None = None
            frame_bytes = self._frame_bytes_for_channels(validated_channels)
            while len(self._frame_buffer) >= frame_bytes:
                frame = bytes(self._frame_buffer[:frame_bytes])
                del self._frame_buffer[:frame_bytes]
                detected_match = self.process_pcm_frame(frame, channels=validated_channels) or detected_match
            return detected_match

    def process_pcm_frame(self, pcm_frame: bytes, *, channels: int = 1) -> WakewordMatch | None:
        """Run wakeword inference on one exact PCM frame."""

        if not pcm_frame:
            return None
        validated_channels = _validate_positive_int(channels)
        if validated_channels is None:
            LOGGER.warning("Ignoring PCM frame with invalid channel count: %r", channels)
            return None
        required_frame_bytes = self._frame_bytes_for_channels(validated_channels)
        if len(pcm_frame) < required_frame_bytes:
            # AUDIT-FIX(#1): Unvollstaendige Frames werden verworfen statt eine Exception in der Modell-API zu riskieren.
            return None
        if len(pcm_frame) > required_frame_bytes:
            # AUDIT-FIX(#2): Direktaufrufe mit Oversize-Frames werden auf genau 1 openWakeWord-Frame begrenzt.
            pcm_frame = pcm_frame[:required_frame_bytes]
        with self._lock:
            samples = _pcm16_to_int16_samples(pcm_frame, channels=validated_channels)
            if len(samples) == 0:
                return None
            try:
                predictions = self._model.predict(samples)
            except Exception:  # pragma: no cover - depends on external package/runtime
                # AUDIT-FIX(#1): Vorhersagefehler werden isoliert, geloggt und der Modellzustand wird fuer Recovery resettet.
                LOGGER.exception("openWakeWord frame prediction failed.")
                self._safe_reset_model()
                return None
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
        normalized_predictions = _normalize_prediction_map(predictions)
        if not normalized_predictions:
            return OpenWakeWordPrediction(label=None, score=0.0)
        thresholded_best = OpenWakeWordPrediction(label=None, score=0.0)
        current_labels = set(self._consecutive_frames.keys()) | set(normalized_predictions.keys())
        for label in current_labels:
            score = normalized_predictions.get(label, 0.0)
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
    """Detect wakewords from buffered captures with optional confirmation.

    This spotter resamples audio to 16 kHz for openWakeWord inference and can
    optionally transcribe successful detector hits to confirm that one of the
    configured wakeword phrases was actually spoken.
    """

    _TARGET_SAMPLE_RATE = 16000
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
        # AUDIT-FIX(#6): Konfig-Strings werden kanonisiert, damit Whitespace keine stillen Fehlkonfigurationen ausloest.
        normalized_models = _normalize_nonempty_strings(wakeword_models)
        if not normalized_models:
            raise ValueError("openWakeWord requires at least one configured model name or path.")
        self.wakeword_models = normalized_models
        # AUDIT-FIX(#6): Wakeword-Phrasen werden ebenfalls sauber normalisiert.
        self.phrases = _normalize_nonempty_strings(phrases)
        # AUDIT-FIX(#6): Probability-Konfig wird mit klarem Fehlerkontext validiert statt still auf unsichere Defaults zu fallen.
        self.threshold = _coerce_probability_config("threshold", threshold)
        self.vad_threshold = _coerce_probability_config("vad_threshold", vad_threshold)
        self.patience_frames = max(1, int(patience_frames))
        self.activation_samples = max(1, int(activation_samples))
        self.deactivation_threshold = _coerce_probability_config(
            "deactivation_threshold",
            deactivation_threshold,
        )
        self.inference_framework = (inference_framework or "tflite").strip().lower() or "tflite"
        self.backend = backend
        self.language = (language or "").strip() or None
        self.transcribe_on_detect = bool(transcribe_on_detect)
        self.min_prefix_ratio = _coerce_probability_config(
            "min_prefix_ratio",
            min_prefix_ratio,
            minimum=0.5,
            maximum=1.0,
        )
        factory = model_factory or _default_model_factory
        try:
            self._model = factory(
                wakeword_models=list(normalized_models),
                vad_threshold=self.vad_threshold,
                enable_speex_noise_suppression=enable_speex_noise_suppression,
                inference_framework=self.inference_framework,
            )
        except Exception as exc:  # pragma: no cover - depends on external package/runtime
            # AUDIT-FIX(#6): Modellinitialisierung liefert jetzt einen klaren operator-tauglichen Fehlerkontext.
            raise RuntimeError(
                f"Failed to initialize openWakeWord model(s): {', '.join(normalized_models)}"
            ) from exc
        model_names = getattr(self._model, "models", {})
        names = tuple(str(name) for name in model_names.keys()) if hasattr(model_names, "keys") else normalized_models
        self._threshold_by_model = {name: self.threshold for name in names}
        self._patience_by_model = {name: self.patience_frames for name in names}
        # AUDIT-FIX(#7): Modellzugriffe werden gegen threaded Reuse abgesichert.
        self._lock = RLock()

    def detect(self, capture: AmbientAudioCaptureWindow) -> WakewordMatch:
        """Detect a wakeword in one captured audio window."""

        pcm_bytes = capture.pcm_bytes or b""
        if not pcm_bytes:
            return WakewordMatch(detected=False, transcript="", backend="openwakeword")
        validated_channels = _validate_positive_int(capture.channels)
        validated_sample_rate = _validate_positive_int(capture.sample_rate)
        if validated_channels is None or validated_sample_rate is None:
            # AUDIT-FIX(#1): Defektes Audio-Metadaten-Input wird sicher abgefangen statt spaeter zu crashen.
            LOGGER.warning(
                "Ignoring wakeword capture with invalid audio metadata: sample_rate=%r channels=%r",
                capture.sample_rate,
                capture.channels,
            )
            return WakewordMatch(detected=False, transcript="", backend="openwakeword")
        # AUDIT-FIX(#3): Nicht-16-kHz-Input wird auf 16 kHz vorbereitet statt die komplette Detection hart zu crashen.
        samples = _prepare_detection_samples(
            pcm_bytes,
            sample_rate=validated_sample_rate,
            channels=validated_channels,
            target_sample_rate=self._TARGET_SAMPLE_RATE,
        )
        if len(samples) == 0:
            return WakewordMatch(detected=False, transcript="", backend="openwakeword")

        with self._lock:
            # AUDIT-FIX(#7): Nur der stateful Modellteil wird serialisiert; langsame STT-Aufrufe blockieren dadurch keine Parallel-Nutzung.
            self._safe_reset_model()
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
            try:
                transcript_match = match_wakeword_transcript(
                    transcript,
                    phrases=self.phrases,
                    min_prefix_ratio=self.min_prefix_ratio,
                    backend="openwakeword",
                    detector_label=best_prediction.label,
                    score=best_prediction.score,
                )
            except Exception:  # pragma: no cover - depends on external package/runtime
                # AUDIT-FIX(#1): Transcript-Matching-Fehler deaktivieren nur diese Aktivierung statt den Service abzuschiessen.
                LOGGER.exception("Wakeword transcript matching failed; rejecting activation.")
                return WakewordMatch(
                    detected=False,
                    transcript=transcript,
                    matched_phrase=matched_phrase,
                    remaining_text="",
                    normalized_transcript="",
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

    def _safe_reset_model(self) -> None:
        if hasattr(self._model, "reset"):
            try:
                self._model.reset()
            except Exception:  # pragma: no cover - depends on external package/runtime
                # AUDIT-FIX(#1): Modell-Reset darf die Wakeword-Erkennung nicht umwerfen.
                LOGGER.exception("openWakeWord spotter model reset failed.")

    def _detect_peak_prediction(self, samples) -> OpenWakeWordPrediction:
        try:
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
        except Exception:  # pragma: no cover - depends on external package/runtime
            # AUDIT-FIX(#1): Predict-Fehler werden isoliert und fuehren zu einer sicheren No-Match-Antwort.
            LOGGER.exception("openWakeWord clip prediction failed.")
            self._safe_reset_model()
            return OpenWakeWordPrediction(label=None, score=0.0)

    def _transcribe(self, capture: AmbientAudioCaptureWindow) -> str:
        if self.backend is None:
            return ""
        try:
            audio_bytes = pcm16_to_wav_bytes(
                capture.pcm_bytes,
                sample_rate=capture.sample_rate,
                channels=capture.channels,
            )
        except Exception:  # pragma: no cover - depends on external package/runtime
            # AUDIT-FIX(#1): WAV-Konvertierungsfehler duerfen keine Aktivierungsschleife crashen.
            LOGGER.exception("Failed to convert wakeword audio to WAV for transcription.")
            return ""
        primary_prompt = wakeword_primary_prompt(self.phrases)
        try:
            transcript = self.backend.transcribe(
                audio_bytes,
                filename="wakeword-openwakeword.wav",
                content_type="audio/wav",
                language=self.language,
                prompt=primary_prompt,
            )
        except Exception:  # pragma: no cover - depends on external backend/runtime
            # AUDIT-FIX(#1): STT-Ausfaelle werden fail-closed behandelt statt als Exception nach oben zu schiessen.
            LOGGER.exception("Wakeword transcription failed; rejecting activation.")
            return ""
        return str(transcript or "")


def _default_model_factory(**kwargs):
    """Build one openWakeWord model, downloading named models when needed."""

    import os

    import openwakeword.utils
    from openwakeword.model import Model

    model_names = _normalize_nonempty_strings(tuple(kwargs.get("wakeword_models", [])))
    named_models: list[str] = []
    resolved_model_paths: list[str] = []
    for item in model_names:
        expanded_item = os.path.abspath(os.path.expanduser(item))
        if _looks_like_local_model_path(item):
            if not os.path.isfile(expanded_item):
                # AUDIT-FIX(#6): Pfadartige Modellnamen werden nicht mehr versehentlich als Download-IDs behandelt.
                raise FileNotFoundError(f"Configured openWakeWord model path does not exist: {item}")
            resolved_model_paths.append(expanded_item)
            continue
        resolved_model_paths.append(item)
        named_models.append(item)
    if named_models:
        try:
            openwakeword.utils.download_models(model_names=named_models)
        except Exception as exc:  # pragma: no cover - depends on external network/runtime
            # AUDIT-FIX(#6): Modell-Downloadfehler liefern klaren Kontext statt eines diffusen Startup-Crashs.
            raise RuntimeError(
                f"Failed to download openWakeWord model(s): {', '.join(named_models)}"
            ) from exc
    kwargs = dict(kwargs)
    kwargs["wakeword_models"] = resolved_model_paths
    return Model(**kwargs)


def _prepare_detection_samples(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int,
    target_sample_rate: int,
):
    samples = _pcm16_to_int16_samples(pcm_bytes, channels=channels)
    if len(samples) == 0:
        return samples
    if sample_rate == target_sample_rate:
        return samples
    return _resample_int16_samples(
        samples,
        source_sample_rate=sample_rate,
        target_sample_rate=target_sample_rate,
    )


def _pcm16_to_int16_samples(pcm_bytes: bytes, *, channels: int):
    import array
    import sys

    validated_channels = _validate_positive_int(channels)
    if validated_channels is None:
        return []
    if not pcm_bytes:
        return []
    if len(pcm_bytes) % 2 != 0:
        # AUDIT-FIX(#4): Ungerade PCM-Byte-Laenge wird defensiv gekuerzt statt array.frombytes crashen zu lassen.
        pcm_bytes = pcm_bytes[: len(pcm_bytes) - 1]
    if not pcm_bytes:
        return []

    pcm_samples = array.array("h")
    try:
        pcm_samples.frombytes(pcm_bytes)
    except ValueError:
        # AUDIT-FIX(#1): Defekte PCM-Fragmente liefern eine sichere Leerausgabe statt den Audio-Pfad zu crashen.
        LOGGER.exception("Failed to decode PCM16 bytes.")
        return []
    if sys.byteorder != "little":
        pcm_samples.byteswap()

    usable = len(pcm_samples) - (len(pcm_samples) % validated_channels)
    if usable <= 0:
        return []
    trimmed_pcm_samples = pcm_samples[:usable]

    try:
        import numpy as np
    except ModuleNotFoundError:
        if validated_channels <= 1:
            return trimmed_pcm_samples.tolist()
        # AUDIT-FIX(#4): Mehrkanal-Audio wird sauber zu Mono gemittelt statt nur Kanal 0 zu nehmen.
        return [
            _clamp_int16(
                round(
                    sum(int(trimmed_pcm_samples[index + offset]) for offset in range(validated_channels))
                    / validated_channels
                )
            )
            for index in range(0, usable, validated_channels)
        ]
    if validated_channels <= 1:
        return np.asarray(trimmed_pcm_samples, dtype=np.int16)
    reshaped = np.asarray(trimmed_pcm_samples, dtype=np.int16).reshape(-1, validated_channels).astype(np.int32)
    # AUDIT-FIX(#4): Mehrkanal-Audio wird sauber zu Mono gemittelt statt nur Kanal 0 zu nehmen.
    averaged = np.rint(np.mean(reshaped, axis=1))
    return np.clip(averaged, -32768, 32767).astype(np.int16)


def _resample_int16_samples(
    samples,
    *,
    source_sample_rate: int,
    target_sample_rate: int,
):
    validated_source = _validate_positive_int(source_sample_rate)
    validated_target = _validate_positive_int(target_sample_rate)
    if validated_source is None or validated_target is None:
        return []
    if validated_source == validated_target:
        return samples
    sample_count = len(samples)
    if sample_count == 0:
        return samples[:0] if hasattr(samples, "__getitem__") else []
    output_count = max(1, int(round(sample_count * (validated_target / validated_source))))
    if output_count == sample_count:
        return samples
    try:
        import numpy as np
    except ModuleNotFoundError:
        if sample_count == 1:
            return [_clamp_int16(int(samples[0])) for _ in range(output_count)]
        scale = (sample_count - 1) / (output_count - 1) if output_count > 1 else 0.0
        resampled: list[int] = []
        for output_index in range(output_count):
            source_position = output_index * scale
            left_index = int(source_position)
            right_index = min(left_index + 1, sample_count - 1)
            interpolation_weight = source_position - left_index
            left_value = int(samples[left_index])
            right_value = int(samples[right_index])
            value = ((1.0 - interpolation_weight) * left_value) + (interpolation_weight * right_value)
            resampled.append(_clamp_int16(round(value)))
        return resampled
    if sample_count == 1:
        return np.full(output_count, int(samples[0]), dtype=np.int16)
    source = np.asarray(samples, dtype=np.float32)
    old_positions = np.arange(sample_count, dtype=np.float32)
    new_positions = np.linspace(0.0, sample_count - 1, num=output_count, dtype=np.float32)
    # AUDIT-FIX(#3): Nicht-16-kHz-Audio wird leichtgewichtig resampled statt hart abgelehnt.
    resampled = np.interp(new_positions, old_positions, source)
    return np.clip(np.rint(resampled), -32768, 32767).astype(np.int16)


def _best_prediction(predictions: dict[str, float] | None) -> OpenWakeWordPrediction:
    normalized_predictions = _normalize_prediction_map(predictions)
    if not normalized_predictions:
        return OpenWakeWordPrediction(label=None, score=0.0)
    best_label = None
    best_score = 0.0
    for label, score in normalized_predictions.items():
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
    threshold = _clamp_probability(threshold)
    patience_frames = max(1, int(patience_frames))
    activation_samples = max(1, int(activation_samples))
    deactivation_threshold = _clamp_probability(deactivation_threshold)

    best = OpenWakeWordPrediction(label=None, score=0.0)
    thresholded_best = OpenWakeWordPrediction(label=None, score=0.0)
    consecutive_frames: dict[str, int] = defaultdict(int)
    score_history: dict[str, deque[float]] = {}
    latched_labels: set[str] = set()
    for prediction in predictions:
        normalized_prediction = _normalize_prediction_map(prediction)
        candidate = _best_prediction(normalized_prediction)
        if candidate.score > best.score:
            best = candidate
        current_labels = set(consecutive_frames.keys()) | set(normalized_prediction.keys())
        for label in current_labels:
            score = normalized_prediction.get(label, 0.0)
            history = score_history.setdefault(
                label,
                deque(maxlen=activation_samples),
            )
            history.append(score)
            if len(history) < activation_samples:
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
    # AUDIT-FIX(#5): Ohne bestandenes Sequenz-Gate bleibt der Score bei 0, damit patience/activation nicht umgangen werden.
    return OpenWakeWordPrediction(label=best.label, score=0.0)


def _normalize_nonempty_strings(values) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values or ():
        text = str(value).strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _normalize_prediction_map(predictions: Any) -> dict[str, float]:
    if not predictions or not hasattr(predictions, "items"):
        return {}
    normalized: dict[str, float] = {}
    for label, raw_score in predictions.items():
        normalized[str(label)] = _clamp_probability(raw_score)
    return normalized


def _looks_like_local_model_path(value: str) -> bool:
    separators = ["/"]
    try:
        import os

        separators.append(os.sep)
        if os.altsep:
            separators.append(os.altsep)
    except Exception:  # pragma: no cover - defensive only
        LOGGER.warning("Failed to inspect local path separators for wakeword model path detection.", exc_info=True)
    return (
        value.startswith(("~", ".", "/"))
        or any(separator and separator in value for separator in separators)
        or value.lower().endswith((".onnx", ".tflite", ".pb"))
    )


# AUDIT-FIX(#6): Konfig-Werte bekommen eine harte, operator-lesbare Validierung statt diffuser Folgebugs.
def _coerce_probability_config(
    name: str,
    value: Any,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float between {minimum} and {maximum}, got {value!r}") from exc
    if not math.isfinite(score):
        raise ValueError(f"{name} must be a finite float between {minimum} and {maximum}, got {value!r}")
    return max(minimum, min(maximum, score))


def _clamp_probability(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(1.0, score))


def _validate_positive_int(value: Any) -> int | None:
    try:
        integer_value = int(value)
    except (TypeError, ValueError):
        return None
    return integer_value if integer_value > 0 else None


def _clamp_int16(value: int) -> int:
    return max(-32768, min(32767, int(value)))


__all__ = [
    "OpenWakeWordPrediction",
    "WakewordOpenWakeWordFrameSpotter",
    "WakewordOpenWakeWordSpotter",
]
