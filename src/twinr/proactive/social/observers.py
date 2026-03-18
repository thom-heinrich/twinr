"""Collect bounded audio and vision observations for social triggers.

This module wraps Twinr camera and ambient-audio adapters, normalizes their
outputs, and returns conservative snapshots when capture or model calls fail.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import logging
import math
import os
import tempfile
from threading import Lock
from types import SimpleNamespace
import time

from twinr.hardware.audio import AmbientAudioLevelSample, AmbientAudioSampler
from twinr.hardware.camera import V4L2StillCamera
from twinr.providers.openai import OpenAIBackend, OpenAIImageInput

from .engine import SocialAudioObservation, SocialBodyPose, SocialPersonZone, SocialVisionObservation


logger = logging.getLogger(__name__)  # AUDIT-FIX(#2): emit provider-level failure telemetry instead of failing silently.

_DEFAULT_AUDIO_LOCK_TIMEOUT_S = 2.0
_DEFAULT_CAMERA_LOCK_TIMEOUT_S = 5.0
_DEFAULT_VISION_API_TIMEOUT_S = 20.0
_DEFAULT_VISION_API_RETRIES = 1
_DEFAULT_VISION_API_RETRY_BACKOFF_S = 0.75
_DEFAULT_VISION_CIRCUIT_BREAKER_THRESHOLD = 3
_DEFAULT_VISION_CIRCUIT_BREAKER_COOLDOWN_S = 30.0
_SANITIZED_CAPTURE_FILENAME = "proactive-camera-capture.png"  # AUDIT-FIX(#1): stable metadata name without exposing local filesystem paths.
_VISION_OBSERVATION_UNAVAILABLE_TEXT = "vision_observation_unavailable"
_VISION_OBSERVATION_SKIPPED_TEXT = "vision_observation_skipped_circuit_open"
_TRUE_BOOL_TOKENS = frozenset({"yes", "true", "1"})
_FALSE_BOOL_TOKENS = frozenset({"no", "false", "0"})
_REQUIRED_VISION_KEYS = frozenset(
    {
        "person_visible",
        "looking_toward_device",
        "body_pose",
        "smiling",
        "hand_or_object_near_camera",
    }
)
_OPTIONAL_VISION_KEYS = frozenset({"person_count", "primary_person_zone"})


@dataclass(frozen=True, slots=True)
class ProactiveVisionSnapshot:
    """Capture one normalized vision observation and its response metadata."""

    observation: SocialVisionObservation
    response_text: str
    captured_at: float | None = None
    image: OpenAIImageInput | None = None
    source_device: str | None = None
    input_format: str | None = None
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None


@dataclass(frozen=True, slots=True)
class ProactiveAudioSnapshot:
    """Capture one normalized audio observation and optional sample data."""

    observation: SocialAudioObservation
    sample: AmbientAudioLevelSample | None = None
    pcm_bytes: bytes | None = None
    sample_rate: int | None = None
    channels: int | None = None


class NullAudioObservationProvider:
    """Return a conservative silent audio observation."""

    def observe(self) -> ProactiveAudioSnapshot:
        """Return one silent audio snapshot when audio sensing is disabled."""

        return ProactiveAudioSnapshot(
            observation=SocialAudioObservation(  # AUDIT-FIX(#3): return an explicit conservative null observation.
                speech_detected=False,
                distress_detected=None,
            )
        )


class AmbientAudioObservationProvider:
    """Sample ambient audio and derive bounded speech and distress observations."""

    def __init__(
        self,
        *,
        sampler: AmbientAudioSampler,
        audio_lock: Lock | None = None,
        sample_ms: int = 1000,
        distress_enabled: bool = False,
        speech_min_active_ratio: float = 0.2,
        distress_min_active_ratio: float = 0.45,
        distress_peak_threshold: int | None = None,
        lock_timeout_s: float = _DEFAULT_AUDIO_LOCK_TIMEOUT_S,
    ) -> None:
        """Initialize one audio observer from a sampler and detection thresholds."""

        self.sampler = sampler
        self.audio_lock = audio_lock or Lock()
        chunk_ms = _coerce_positive_int(getattr(sampler, "chunk_ms", 20), default=20)
        requested_sample_ms = _coerce_positive_int(sample_ms, default=1000)
        self.sample_ms = max(chunk_ms, requested_sample_ms)  # AUDIT-FIX(#5): sanitize chunk/sample durations before using them.
        self.distress_enabled = bool(distress_enabled)
        self.speech_min_active_ratio = _clamp_ratio(speech_min_active_ratio)  # AUDIT-FIX(#5): clamp detection ratios into the valid [0, 1] range.
        self.distress_min_active_ratio = _clamp_ratio(distress_min_active_ratio)  # AUDIT-FIX(#5): clamp detection ratios into the valid [0, 1] range.
        self.distress_peak_threshold = _coerce_optional_non_negative_int(distress_peak_threshold)  # AUDIT-FIX(#5): preserve explicit 0/None semantics and reject invalid thresholds.
        self.lock_timeout_s = _coerce_non_negative_float(lock_timeout_s, default=_DEFAULT_AUDIO_LOCK_TIMEOUT_S, minimum=0.05)

    def observe(self) -> ProactiveAudioSnapshot:
        """Capture one audio window and convert it into one observation snapshot."""

        if not self.audio_lock.acquire(timeout=self.lock_timeout_s):  # AUDIT-FIX(#3): bound lock wait so one stalled reader does not wedge the device indefinitely.
            logger.warning(
                "Audio observation skipped because the audio lock could not be acquired within %.2fs.",
                self.lock_timeout_s,
            )
            return _unavailable_audio_snapshot()
        try:
            if hasattr(self.sampler, "sample_window"):
                capture = self.sampler.sample_window(duration_ms=self.sample_ms)
            else:
                sample = self.sampler.sample_levels(duration_ms=self.sample_ms)
                capture = SimpleNamespace(
                    sample=sample,
                    pcm_bytes=None,
                    sample_rate=None,
                    channels=None,
                )

            sample = getattr(capture, "sample", None)
            if sample is None:
                raise ValueError("Audio sampler returned no sample object.")

            speech_threshold = _coerce_non_negative_int(getattr(self.sampler, "speech_threshold", 700), default=700)
            speech_peak_threshold = max(1800, int(speech_threshold * 2.5))
            active_chunk_count = _coerce_non_negative_int(getattr(sample, "active_chunk_count", 0), default=0)
            active_ratio = _clamp_ratio(getattr(sample, "active_ratio", 0.0))
            peak_rms = _coerce_non_negative_int(getattr(sample, "peak_rms", 0), default=0)

            speech_detected = False
            if active_chunk_count > 0 and active_ratio >= self.speech_min_active_ratio:
                speech_detected = True
            elif active_chunk_count >= 2 and peak_rms >= speech_peak_threshold:
                # Short utterances near the device can be brief enough to miss the
                # active-ratio gate while still being clear speech in practice.
                speech_detected = True

            distress_detected: bool | None = None
            if self.distress_enabled:
                peak_threshold = (
                    self.distress_peak_threshold
                    if self.distress_peak_threshold is not None
                    else max(1800, int(speech_threshold * 2.5))
                )
                distress_detected = (
                    speech_detected
                    and active_ratio >= self.distress_min_active_ratio
                    and peak_rms >= peak_threshold
                )

            return ProactiveAudioSnapshot(
                observation=SocialAudioObservation(
                    speech_detected=speech_detected,
                    distress_detected=distress_detected,
                ),
                sample=sample,
                pcm_bytes=getattr(capture, "pcm_bytes", None),
                sample_rate=_coerce_optional_positive_int(getattr(capture, "sample_rate", None)),
                channels=_coerce_optional_positive_int(getattr(capture, "channels", None)),
            )
        except Exception:
            logger.exception(
                "Audio observation failed; returning a conservative silent snapshot."
            )  # AUDIT-FIX(#3): degrade safely instead of bubbling hardware errors into the event loop.
            return _unavailable_audio_snapshot()
        finally:
            self.audio_lock.release()


class OpenAIVisionObservationProvider:
    """Capture one still frame and classify it for social-trigger use."""

    def __init__(
        self,
        *,
        backend: OpenAIBackend,
        camera: V4L2StillCamera,
        camera_lock: Lock | None = None,
        lock_timeout_s: float = _DEFAULT_CAMERA_LOCK_TIMEOUT_S,
        api_timeout_s: float | None = _DEFAULT_VISION_API_TIMEOUT_S,
        api_retries: int = _DEFAULT_VISION_API_RETRIES,
        api_retry_backoff_s: float = _DEFAULT_VISION_API_RETRY_BACKOFF_S,
        circuit_breaker_threshold: int = _DEFAULT_VISION_CIRCUIT_BREAKER_THRESHOLD,
        circuit_breaker_cooldown_s: float = _DEFAULT_VISION_CIRCUIT_BREAKER_COOLDOWN_S,
    ) -> None:
        """Initialize one vision observer with bounded lock, timeout, and retry policy."""

        self.backend = backend
        self.camera = camera
        self.camera_lock = camera_lock or Lock()
        self.lock_timeout_s = _coerce_non_negative_float(
            lock_timeout_s,
            default=_DEFAULT_CAMERA_LOCK_TIMEOUT_S,
            minimum=0.05,
        )
        self.api_timeout_s = (
            None
            if api_timeout_s is None
            else _coerce_non_negative_float(
                api_timeout_s,
                default=_DEFAULT_VISION_API_TIMEOUT_S,
                minimum=0.1,
            )
        )
        self.api_retries = _coerce_non_negative_int(api_retries, default=_DEFAULT_VISION_API_RETRIES)
        self.api_retry_backoff_s = _coerce_non_negative_float(
            api_retry_backoff_s,
            default=_DEFAULT_VISION_API_RETRY_BACKOFF_S,
            minimum=0.0,
        )
        self.circuit_breaker_threshold = max(
            1,
            _coerce_non_negative_int(
                circuit_breaker_threshold,
                default=_DEFAULT_VISION_CIRCUIT_BREAKER_THRESHOLD,
            ),
        )
        self.circuit_breaker_cooldown_s = _coerce_non_negative_float(
            circuit_breaker_cooldown_s,
            default=_DEFAULT_VISION_CIRCUIT_BREAKER_COOLDOWN_S,
            minimum=0.0,
        )
        self._failure_state_lock = Lock()
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def observe(self) -> ProactiveVisionSnapshot:
        """Capture and classify one frame, falling back to unknown on failure."""

        if self._is_circuit_open():  # AUDIT-FIX(#2): short-circuit repeated backend failures instead of hammering a dead dependency.
            logger.warning("Vision observation skipped because the circuit breaker is open.")
            return self._fallback_snapshot(response_text=_VISION_OBSERVATION_SKIPPED_TEXT)

        image: OpenAIImageInput | None = None
        source_device: str | None = None
        input_format: str | None = None
        response_text = ""
        response_id: str | None = None
        request_id: str | None = None
        model: str | None = None
        captured_at: float | None = None

        try:
            image, source_device, input_format, captured_at = self._capture_image()
            response = self._call_backend_with_retries(image)
            response_id = getattr(response, "response_id", None)
            request_id = getattr(response, "request_id", None)
            model = getattr(response, "model", None)
            response_text = getattr(response, "text", "")
            if not isinstance(response_text, str) or not response_text.strip():
                raise ValueError("Vision backend returned an empty classifier response.")

            observation = parse_vision_observation_text(response_text)
        except Exception:
            self._record_failure()  # AUDIT-FIX(#2): convert camera/backend/parser failures into conservative snapshots and provider health state.
            logger.exception("Vision observation failed; returning a conservative unknown snapshot.")
            return self._fallback_snapshot(
                response_text=response_text or _VISION_OBSERVATION_UNAVAILABLE_TEXT,
                captured_at=captured_at,
                image=image,
                source_device=source_device,
                input_format=input_format,
                response_id=response_id,
                request_id=request_id,
                model=model,
            )

        self._record_success()
        return ProactiveVisionSnapshot(
            observation=observation,
            response_text=response_text,
            captured_at=captured_at,  # AUDIT-FIX(#4): use a wall-clock epoch timestamp that survives restarts and log correlation.
            image=image,
            source_device=source_device,
            input_format=input_format,
            response_id=response_id,
            request_id=request_id,
            model=model,
        )

    def _capture_image(self) -> tuple[OpenAIImageInput, str | None, str | None, float]:
        """Capture one still frame and package it for the backend."""

        with tempfile.TemporaryDirectory(prefix="twinr-proactive-") as temp_dir:  # AUDIT-FIX(#1): capture into a private temporary directory to avoid filename collisions and symlink clobbering.
            capture_path = os.path.join(temp_dir, _SANITIZED_CAPTURE_FILENAME)
            if not self.camera_lock.acquire(timeout=self.lock_timeout_s):
                raise TimeoutError(f"Timed out waiting {self.lock_timeout_s:.2f}s for the camera lock.")
            try:
                capture = self.camera.capture_photo(filename=capture_path)
                captured_at = time.time()  # AUDIT-FIX(#4): record wall-clock capture time instead of a monotonic counter.
            finally:
                self.camera_lock.release()

            image_data = getattr(capture, "data", None)
            if not image_data:
                raise ValueError("Camera capture returned empty image data.")

            image = OpenAIImageInput(
                data=image_data,
                content_type=getattr(capture, "content_type", None) or "image/png",
                filename=_SANITIZED_CAPTURE_FILENAME,  # AUDIT-FIX(#1): do not expose the temporary on-disk path outside this provider.
                label="Live proactive camera frame from the device.",
            )
            return (
                image,
                getattr(capture, "source_device", None),
                getattr(capture, "input_format", None),
                captured_at,
            )

    def _call_backend_with_retries(self, image: OpenAIImageInput):
        """Call the vision backend with bounded retries."""

        backend_method = self.backend.respond_to_images_with_metadata
        backend_kwargs = _supported_backend_kwargs(
            backend_method,
            timeout_s=self.api_timeout_s,
            disable_inner_retries=True,
        )
        last_error: Exception | None = None
        for attempt in range(self.api_retries + 1):  # AUDIT-FIX(#2): add bounded retry logic for transient network/model failures.
            try:
                return backend_method(
                    _VISION_CLASSIFIER_PROMPT,
                    images=(image,),
                    allow_web_search=False,
                    **backend_kwargs,
                )
            except Exception as exc:  # pragma: no cover - depends on backend behavior.
                last_error = exc
                is_last_attempt = attempt >= self.api_retries
                if is_last_attempt:
                    break
                backoff_s = self.api_retry_backoff_s * (2**attempt)
                logger.warning(
                    "Vision backend call failed on attempt %s/%s with %s; retrying in %.2fs.",
                    attempt + 1,
                    self.api_retries + 1,
                    exc.__class__.__name__,
                    backoff_s,
                )
                if backoff_s > 0:
                    time.sleep(backoff_s)

        assert last_error is not None
        raise last_error

    def _is_circuit_open(self) -> bool:
        """Return whether the vision circuit breaker is currently open."""

        now = time.monotonic()
        with self._failure_state_lock:
            return now < self._circuit_open_until

    def _record_success(self) -> None:
        """Reset failure state after one successful vision call."""

        with self._failure_state_lock:
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0

    def _record_failure(self) -> None:
        """Record one failed vision attempt and open the circuit if needed."""

        now = time.monotonic()
        with self._failure_state_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.circuit_breaker_threshold:
                self._circuit_open_until = max(
                    self._circuit_open_until,
                    now + self.circuit_breaker_cooldown_s,
                )

    def _fallback_snapshot(
        self,
        *,
        response_text: str,
        captured_at: float | None = None,
        image: OpenAIImageInput | None = None,
        source_device: str | None = None,
        input_format: str | None = None,
        response_id: str | None = None,
        request_id: str | None = None,
        model: str | None = None,
    ) -> ProactiveVisionSnapshot:
        """Build one conservative unknown-vision snapshot."""

        return ProactiveVisionSnapshot(
            observation=_unknown_vision_observation(),
            response_text=response_text,
            captured_at=captured_at,
            image=image,
            source_device=source_device,
            input_format=input_format,
            response_id=response_id,
            request_id=request_id,
            model=model,
        )


def parse_vision_observation_text(text: str) -> SocialVisionObservation:
    """Parse one structured vision-classifier response into an observation."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Vision classifier response must be a non-empty string.")

    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip().lower()
        if normalized_key not in _REQUIRED_VISION_KEYS and normalized_key not in _OPTIONAL_VISION_KEYS:
            continue
        if normalized_key in values:
            raise ValueError(
                f"Duplicate vision classifier key: {normalized_key}"
            )  # AUDIT-FIX(#4): reject ambiguous model output instead of silently overriding it.
        values[normalized_key] = _normalize_token(value)

    missing_keys = sorted(_REQUIRED_VISION_KEYS.difference(values))
    if missing_keys:
        raise ValueError(
            f"Missing required vision classifier keys: {', '.join(missing_keys)}"
        )  # AUDIT-FIX(#4): fail closed on malformed classifier output.

    person_visible = _parse_bool(values.get("person_visible"))
    person_count = _parse_non_negative_int(values.get("person_count"), default=0)
    primary_person_zone = _parse_zone(values.get("primary_person_zone"))
    if person_visible:
        person_count = max(1, person_count)
    else:
        person_count = 0
        primary_person_zone = SocialPersonZone.UNKNOWN

    return SocialVisionObservation(
        person_visible=person_visible,
        person_count=person_count,
        primary_person_zone=primary_person_zone,
        looking_toward_device=_parse_bool(values.get("looking_toward_device")),
        body_pose=_parse_pose(values.get("body_pose")),
        smiling=_parse_bool(values.get("smiling")),
        hand_or_object_near_camera=_parse_bool(values.get("hand_or_object_near_camera")),
    )


def _parse_bool(value: str | None) -> bool:
    """Parse one ``yes`` or ``no`` style token from classifier output."""

    normalized_value = _normalize_token(value)
    if normalized_value in _TRUE_BOOL_TOKENS:
        return True
    if normalized_value in _FALSE_BOOL_TOKENS:
        return False
    raise ValueError(
        f"Invalid boolean token in vision classifier output: {value!r}"
    )  # AUDIT-FIX(#4): stop silently coercing malformed values to False.


def _parse_pose(value: str | None) -> SocialBodyPose:
    """Parse one classifier pose token into ``SocialBodyPose``."""

    normalized_value = _normalize_token(value)
    if normalized_value == "upright":
        return SocialBodyPose.UPRIGHT
    if normalized_value == "slumped":
        return SocialBodyPose.SLUMPED
    if normalized_value == "floor":
        return SocialBodyPose.FLOOR
    if normalized_value == "unknown":
        return SocialBodyPose.UNKNOWN
    raise ValueError(
        f"Invalid body_pose token in vision classifier output: {value!r}"
    )  # AUDIT-FIX(#4): stop silently coercing malformed poses to UNKNOWN.


def _parse_zone(value: str | None) -> SocialPersonZone:
    """Parse one optional primary-person zone token."""

    if value is None:
        return SocialPersonZone.UNKNOWN
    normalized_value = _normalize_token(value)
    if not normalized_value:
        return SocialPersonZone.UNKNOWN
    try:
        return SocialPersonZone(normalized_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid primary_person_zone token in vision classifier output: {value!r}"
        ) from exc


def _parse_non_negative_int(value: str | None, *, default: int) -> int:
    """Parse one optional non-negative integer token."""

    if value is None:
        return default
    normalized_value = _normalize_token(value)
    if not normalized_value:
        return default
    try:
        number = int(normalized_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid integer token in vision classifier output: {value!r}"
        ) from exc
    if number < 0:
        raise ValueError(
            f"Invalid non-negative integer token in vision classifier output: {value!r}"
        )
    return number


# AUDIT-FIX(#5): shared coercion helpers keep provider thresholds predictable under bad config or driver metadata.
def _coerce_positive_int(value: object, *, default: int) -> int:
    """Coerce one value to a positive integer with fallback."""

    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, coerced)


def _coerce_non_negative_int(value: object, *, default: int) -> int:
    """Coerce one value to a non-negative integer with fallback."""

    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, coerced)


def _coerce_optional_non_negative_int(value: object | None) -> int | None:
    """Coerce one optional value to a non-negative integer."""

    if value is None:
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, coerced)


def _coerce_optional_positive_int(value: object | None) -> int | None:
    """Coerce one optional value to a positive integer."""

    if value is None:
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


def _coerce_non_negative_float(value: object, *, default: float, minimum: float = 0.0) -> float:
    """Coerce one value to a finite non-negative float."""

    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(coerced):
        return default
    return max(minimum, coerced)


def _clamp_ratio(value: object, *, default: float = 0.0) -> float:
    """Clamp one ratio-like value into ``[0.0, 1.0]``."""

    coerced = _coerce_non_negative_float(value, default=default, minimum=0.0)
    return min(1.0, coerced)


def _normalize_token(value: str | None) -> str:
    """Normalize one classifier token for parsing."""

    if value is None:
        return ""
    normalized = value.strip().strip("`").strip().strip('"').strip("'").strip().lower()
    return normalized


def _supported_backend_kwargs(
    method: object,
    *,
    timeout_s: float | None,
    disable_inner_retries: bool,
) -> dict[str, float | int]:
    """Detect timeout and retry kwargs supported by one backend method."""

    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return {}

    kwargs: dict[str, float | int] = {}
    if timeout_s is not None:
        for timeout_name in ("timeout_s", "timeout", "request_timeout"):
            if timeout_name in parameters:
                kwargs[timeout_name] = timeout_s
                break
    if disable_inner_retries:
        for retries_name in ("max_retries", "retries", "retry_count"):
            if retries_name in parameters:
                kwargs[retries_name] = 0
                break
    return kwargs


def _unknown_vision_observation() -> SocialVisionObservation:
    """Return the conservative unknown vision observation."""

    return SocialVisionObservation(
        person_visible=False,
        person_count=0,
        primary_person_zone=SocialPersonZone.UNKNOWN,
        looking_toward_device=False,
        body_pose=SocialBodyPose.UNKNOWN,
        smiling=False,
        hand_or_object_near_camera=False,
    )


def _unavailable_audio_snapshot() -> ProactiveAudioSnapshot:
    """Return the conservative unavailable-audio snapshot."""

    return ProactiveAudioSnapshot(
        observation=SocialAudioObservation(
            speech_detected=False,
            distress_detected=None,
        )
    )


_VISION_CLASSIFIER_PROMPT = (
    "You classify a single live camera frame for Twinr's proactive trigger engine. "
    "Return only ASCII lines in this exact key=value format:\n"
    "person_visible=yes|no\n"
    "person_count=0|1|2|...\n"
    "primary_person_zone=left|center|right|unknown\n"
    "looking_toward_device=yes|no\n"
    "body_pose=upright|slumped|floor|unknown\n"
    "smiling=yes|no\n"
    "hand_or_object_near_camera=yes|no\n\n"
    "Rules:\n"
    "- Report only what is visually obvious in the image.\n"
    "- If uncertain, use no or unknown.\n"
    "- Do not infer identity, emotions, diagnosis, age, or intent beyond the listed keys.\n"
    "- `person_visible=yes` if any meaningful part of a person is visible, even when the body is partly cropped.\n"
    "- `person_count` is the non-negative number of plausibly visible people in the frame; use 0 when none are visible.\n"
    "- `primary_person_zone` refers to the most prominent visible person. Use `left`, `center`, or `right` based on horizontal thirds of the image, or `unknown` if no person is visible or the zone is unclear.\n"
    "- `floor` means the visible person is low to the floor, clearly lying down, collapsed at ground level, or only partially visible near the floor after a drop.\n"
    "- Prefer `floor` over `unknown` when a person is obviously very low in the scene.\n"
    "- `slumped` means seated or standing but visibly collapsed forward or drooping.\n"
)


__all__ = [
    "AmbientAudioObservationProvider",
    "NullAudioObservationProvider",
    "OpenAIVisionObservationProvider",
    "ProactiveAudioSnapshot",
    "ProactiveVisionSnapshot",
    "parse_vision_observation_text",
]
