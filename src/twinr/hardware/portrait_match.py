# CHANGELOG: 2026-03-28
# BUG-1: Fixed non-match identity leakage. The original code exposed matched_user_id and
#        matched_user_display_name even for ambiguous_identity and unknown_face states,
#        which could mislead downstream runtime policy.
# BUG-2: Fixed real race conditions around enrollment, observation, cache invalidation,
#        and backend model use. Public state-changing methods now execute under one
#        provider lock, and the OpenCV backend serializes mutable detector/recognizer use.
# BUG-3: Fixed impossible temporal stabilization when temporal_max_observations was
#        configured below temporal_min_observations.
# BUG-4: Fixed permanent legacy-reference bootstrap failure after a transient startup
#        problem. Bootstrap now retries with backoff instead of failing forever.
# BUG-5: Fixed stale positive observations being reused for too long. Cache TTL is now
#        state-aware and the default max_age is reduced for live scenes.
# SEC-1: Added input byte-budget and decoded-pixel budget enforcement to prevent memory/
#        CPU denial-of-service via oversized or highly compressed images on Raspberry Pi.
# SEC-2: Hardened biometric storage permissions for the identity store and reference-image
#        directory to reduce practical local exfiltration risk on shared deployments.
# IMP-1: Added burst capture with cross-frame consensus and better temporal evidence fusion
#        instead of relying on one still frame.
# IMP-2: Added low-cost face quality gating (blur/size/exposure/confidence proxy) so low-
#        quality enrollment/live frames are rejected before matching.
# IMP-3: Replaced ad-hoc best/top3-only aggregation with prototype-aware profile scoring.
# IMP-4: Downscaled oversized inputs before detection to bound latency on Pi 4 while
#        preserving recognition utility.
# BREAKING: Default thresholds changed to values aligned with OpenCV SFace cosine guidance
#           and edge deployment behavior. Default max_age_s is reduced from 45s to 3s.

"""Perform bounded on-device portrait matching against local identity profiles.

This module keeps local face detection, embedding extraction, still-camera
capture, persistent identity enrollment, and temporal evidence fusion behind
one hardware-facing adapter. Runtime policy consumes its structured outputs
elsewhere; this module only owns the local capture and identity-match path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock, RLock
import math
import os
import stat
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.camera import CameraError, V4L2StillCamera
from twinr.hardware.portrait_identity import (
    PortraitIdentityProfile,
    PortraitIdentityStore,
    PortraitIdentitySummary,
)

_ALLOWED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_DEFAULT_REFERENCE_MAX_BYTES = 5 * 1024 * 1024
_DEFAULT_INPUT_IMAGE_MAX_BYTES = 8 * 1024 * 1024
_DEFAULT_DECODED_IMAGE_MAX_PIXELS = 1280 * 960
_DEFAULT_WORKING_IMAGE_MAX_SIDE = 1280
_DEFAULT_CAPTURE_LOCK_TIMEOUT_S = 5.0
# BREAKING: Reduced the default cache horizon to avoid stale identity reuse in live scenes.
_DEFAULT_MAX_AGE_S = 3.0
# BREAKING: Updated default cosine thresholds to the documented SFace operating region.
_DEFAULT_LIKELY_THRESHOLD = 0.36
_DEFAULT_UNCERTAIN_THRESHOLD = 0.30
_DEFAULT_IDENTITY_MARGIN = 0.05
_DEFAULT_MAX_REFERENCE_IMAGES_PER_USER = 6
_DEFAULT_TEMPORAL_WINDOW_S = 300.0
_DEFAULT_TEMPORAL_MIN_OBSERVATIONS = 2
_DEFAULT_TEMPORAL_MAX_OBSERVATIONS = 12
_DEFAULT_DETECTOR_SCORE_THRESHOLD = 0.90
_DEFAULT_DETECTOR_NMS_THRESHOLD = 0.30
_DEFAULT_DETECTOR_TOP_K = 20
_DEFAULT_MIN_ENROLLMENT_QUALITY = 0.42
_DEFAULT_MIN_LIVE_QUALITY = 0.30
_DEFAULT_LIVE_CAPTURE_BURST_COUNT = 3
_DEFAULT_LIVE_CAPTURE_BURST_INTERVAL_S = 0.18
_DEFAULT_LIVE_CAPTURE_MIN_CONSENSUS = 2
_DEFAULT_IDENTITY_STORE_DIR_MODE = 0o700
_DEFAULT_IDENTITY_STORE_FILE_MODE = 0o600
_DEFAULT_LEGACY_BOOTSTRAP_RETRY_S = 15.0
_DEFAULT_SOURCE_IMAGE_PIXELS_GUARD = 16_000_000


@dataclass(frozen=True, slots=True)
class PortraitMatchConfig:
    """Store normalized local portrait-match configuration."""

    reference_image_path: Path | None
    detector_model_path: Path
    recognizer_model_path: Path
    identity_store_path: Path
    identity_reference_image_dir: Path
    primary_user_id: str = "main_user"
    max_reference_images_per_user: int = _DEFAULT_MAX_REFERENCE_IMAGES_PER_USER
    identity_margin: float = _DEFAULT_IDENTITY_MARGIN
    temporal_window_s: float = _DEFAULT_TEMPORAL_WINDOW_S
    temporal_min_observations: int = _DEFAULT_TEMPORAL_MIN_OBSERVATIONS
    temporal_max_observations: int = _DEFAULT_TEMPORAL_MAX_OBSERVATIONS
    max_age_s: float = _DEFAULT_MAX_AGE_S
    likely_threshold: float = _DEFAULT_LIKELY_THRESHOLD
    uncertain_threshold: float = _DEFAULT_UNCERTAIN_THRESHOLD
    reference_max_bytes: int = _DEFAULT_REFERENCE_MAX_BYTES
    input_image_max_bytes: int = _DEFAULT_INPUT_IMAGE_MAX_BYTES
    decoded_image_max_pixels: int = _DEFAULT_DECODED_IMAGE_MAX_PIXELS
    working_image_max_side: int = _DEFAULT_WORKING_IMAGE_MAX_SIDE
    detector_score_threshold: float = _DEFAULT_DETECTOR_SCORE_THRESHOLD
    detector_nms_threshold: float = _DEFAULT_DETECTOR_NMS_THRESHOLD
    detector_top_k: int = _DEFAULT_DETECTOR_TOP_K
    min_enrollment_quality: float = _DEFAULT_MIN_ENROLLMENT_QUALITY
    min_live_quality: float = _DEFAULT_MIN_LIVE_QUALITY
    live_capture_burst_count: int = _DEFAULT_LIVE_CAPTURE_BURST_COUNT
    live_capture_burst_interval_s: float = _DEFAULT_LIVE_CAPTURE_BURST_INTERVAL_S
    live_capture_min_consensus: int = _DEFAULT_LIVE_CAPTURE_MIN_CONSENSUS
    capture_lock_timeout_s: float = _DEFAULT_CAPTURE_LOCK_TIMEOUT_S
    secure_dir_mode: int = _DEFAULT_IDENTITY_STORE_DIR_MODE
    secure_file_mode: int = _DEFAULT_IDENTITY_STORE_FILE_MODE
    legacy_bootstrap_retry_s: float = _DEFAULT_LEGACY_BOOTSTRAP_RETRY_S

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "PortraitMatchConfig":
        """Build one portrait-match config from ``TwinrConfig``."""

        project_root = Path(config.project_root)
        reference_path = _resolve_optional_path(
            project_root=project_root,
            raw_value=getattr(config, "vision_reference_image_path", None),
        )
        detector_model_path = _resolve_required_path(
            project_root=project_root,
            raw_value=getattr(
                config,
                "portrait_match_detector_model_path",
                "state/opencv/models/face_detection_yunet_2023mar.onnx",
            ),
        )
        recognizer_model_path = _resolve_required_path(
            project_root=project_root,
            raw_value=getattr(
                config,
                "portrait_match_recognizer_model_path",
                "state/opencv/models/face_recognition_sface_2021dec.onnx",
            ),
        )
        likely_threshold, uncertain_threshold = _normalize_thresholds(
            getattr(config, "portrait_match_likely_threshold", _DEFAULT_LIKELY_THRESHOLD),
            getattr(config, "portrait_match_uncertain_threshold", _DEFAULT_UNCERTAIN_THRESHOLD),
        )
        temporal_min_observations = max(
            1,
            _coerce_positive_int(
                getattr(
                    config,
                    "portrait_match_temporal_min_observations",
                    _DEFAULT_TEMPORAL_MIN_OBSERVATIONS,
                ),
                default=_DEFAULT_TEMPORAL_MIN_OBSERVATIONS,
            ),
        )
        temporal_max_observations = max(
            temporal_min_observations,
            _coerce_positive_int(
                getattr(
                    config,
                    "portrait_match_temporal_max_observations",
                    _DEFAULT_TEMPORAL_MAX_OBSERVATIONS,
                ),
                default=_DEFAULT_TEMPORAL_MAX_OBSERVATIONS,
            ),
        )
        live_capture_burst_count = max(
            1,
            _coerce_positive_int(
                getattr(
                    config,
                    "portrait_match_live_capture_burst_count",
                    _DEFAULT_LIVE_CAPTURE_BURST_COUNT,
                ),
                default=_DEFAULT_LIVE_CAPTURE_BURST_COUNT,
            ),
        )
        live_capture_min_consensus = max(
            1,
            min(
                live_capture_burst_count,
                _coerce_positive_int(
                    getattr(
                        config,
                        "portrait_match_live_capture_min_consensus",
                        _DEFAULT_LIVE_CAPTURE_MIN_CONSENSUS,
                    ),
                    default=_DEFAULT_LIVE_CAPTURE_MIN_CONSENSUS,
                ),
            ),
        )
        reference_max_bytes = max(
            1024,
            _coerce_positive_int(
                getattr(config, "vision_reference_image_max_bytes", _DEFAULT_REFERENCE_MAX_BYTES),
                default=_DEFAULT_REFERENCE_MAX_BYTES,
            ),
        )
        input_image_max_bytes = max(
            reference_max_bytes,
            _coerce_positive_int(
                getattr(config, "portrait_match_input_image_max_bytes", _DEFAULT_INPUT_IMAGE_MAX_BYTES),
                default=_DEFAULT_INPUT_IMAGE_MAX_BYTES,
            ),
        )
        return cls(
            reference_image_path=reference_path,
            detector_model_path=detector_model_path,
            recognizer_model_path=recognizer_model_path,
            identity_store_path=_resolve_path(
                project_root=project_root,
                raw_value=getattr(config, "portrait_match_store_path", "state/portrait_identities.json"),
            ),
            identity_reference_image_dir=_resolve_path(
                project_root=project_root,
                raw_value=getattr(config, "portrait_match_reference_image_dir", "state/portrait_identities"),
            ),
            primary_user_id=_normalize_user_id(
                getattr(config, "portrait_match_primary_user_id", "main_user") or "main_user"
            ),
            max_reference_images_per_user=max(
                1,
                _coerce_positive_int(
                    getattr(
                        config,
                        "portrait_match_max_reference_images_per_user",
                        _DEFAULT_MAX_REFERENCE_IMAGES_PER_USER,
                    ),
                    default=_DEFAULT_MAX_REFERENCE_IMAGES_PER_USER,
                ),
            ),
            identity_margin=_coerce_ratio(
                getattr(config, "portrait_match_identity_margin", _DEFAULT_IDENTITY_MARGIN),
                default=_DEFAULT_IDENTITY_MARGIN,
            ),
            temporal_window_s=_coerce_non_negative_float(
                getattr(config, "portrait_match_temporal_window_s", _DEFAULT_TEMPORAL_WINDOW_S),
                default=_DEFAULT_TEMPORAL_WINDOW_S,
            ),
            temporal_min_observations=temporal_min_observations,
            temporal_max_observations=temporal_max_observations,
            max_age_s=_coerce_non_negative_float(
                getattr(config, "portrait_match_max_age_s", _DEFAULT_MAX_AGE_S),
                default=_DEFAULT_MAX_AGE_S,
            ),
            likely_threshold=likely_threshold,
            uncertain_threshold=uncertain_threshold,
            reference_max_bytes=reference_max_bytes,
            input_image_max_bytes=input_image_max_bytes,
            decoded_image_max_pixels=max(
                256 * 256,
                _coerce_positive_int(
                    getattr(
                        config,
                        "portrait_match_decoded_image_max_pixels",
                        _DEFAULT_DECODED_IMAGE_MAX_PIXELS,
                    ),
                    default=_DEFAULT_DECODED_IMAGE_MAX_PIXELS,
                ),
            ),
            working_image_max_side=max(
                256,
                _coerce_positive_int(
                    getattr(
                        config,
                        "portrait_match_working_image_max_side",
                        _DEFAULT_WORKING_IMAGE_MAX_SIDE,
                    ),
                    default=_DEFAULT_WORKING_IMAGE_MAX_SIDE,
                ),
            ),
            detector_score_threshold=_coerce_ratio(
                getattr(
                    config,
                    "portrait_match_detector_score_threshold",
                    _DEFAULT_DETECTOR_SCORE_THRESHOLD,
                ),
                default=_DEFAULT_DETECTOR_SCORE_THRESHOLD,
            ),
            detector_nms_threshold=_coerce_ratio(
                getattr(
                    config,
                    "portrait_match_detector_nms_threshold",
                    _DEFAULT_DETECTOR_NMS_THRESHOLD,
                ),
                default=_DEFAULT_DETECTOR_NMS_THRESHOLD,
            ),
            detector_top_k=max(
                1,
                _coerce_positive_int(
                    getattr(config, "portrait_match_detector_top_k", _DEFAULT_DETECTOR_TOP_K),
                    default=_DEFAULT_DETECTOR_TOP_K,
                ),
            ),
            min_enrollment_quality=_coerce_ratio(
                getattr(
                    config,
                    "portrait_match_min_enrollment_quality",
                    _DEFAULT_MIN_ENROLLMENT_QUALITY,
                ),
                default=_DEFAULT_MIN_ENROLLMENT_QUALITY,
            ),
            min_live_quality=_coerce_ratio(
                getattr(config, "portrait_match_min_live_quality", _DEFAULT_MIN_LIVE_QUALITY),
                default=_DEFAULT_MIN_LIVE_QUALITY,
            ),
            live_capture_burst_count=live_capture_burst_count,
            live_capture_burst_interval_s=_coerce_non_negative_float(
                getattr(
                    config,
                    "portrait_match_live_capture_burst_interval_s",
                    _DEFAULT_LIVE_CAPTURE_BURST_INTERVAL_S,
                ),
                default=_DEFAULT_LIVE_CAPTURE_BURST_INTERVAL_S,
            ),
            live_capture_min_consensus=live_capture_min_consensus,
            capture_lock_timeout_s=_coerce_non_negative_float(
                getattr(config, "portrait_match_capture_lock_timeout_s", _DEFAULT_CAPTURE_LOCK_TIMEOUT_S),
                default=_DEFAULT_CAPTURE_LOCK_TIMEOUT_S,
            ),
            secure_dir_mode=_coerce_mode(
                getattr(config, "portrait_match_secure_dir_mode", _DEFAULT_IDENTITY_STORE_DIR_MODE),
                default=_DEFAULT_IDENTITY_STORE_DIR_MODE,
            ),
            secure_file_mode=_coerce_mode(
                getattr(config, "portrait_match_secure_file_mode", _DEFAULT_IDENTITY_STORE_FILE_MODE),
                default=_DEFAULT_IDENTITY_STORE_FILE_MODE,
            ),
            legacy_bootstrap_retry_s=_coerce_non_negative_float(
                getattr(
                    config,
                    "portrait_match_legacy_bootstrap_retry_s",
                    _DEFAULT_LEGACY_BOOTSTRAP_RETRY_S,
                ),
                default=_DEFAULT_LEGACY_BOOTSTRAP_RETRY_S,
            ),
        )


@dataclass(frozen=True, slots=True)
class PortraitEmbeddingResult:
    """Describe one embedding extraction attempt."""

    state: str
    embedding: tuple[float, ...] | None = None
    face_count: int = 0
    detector_confidence: float | None = None
    quality_score: float | None = None
    face_box: tuple[int, int, int, int] | None = None
    image_size: tuple[int, int] | None = None
    sharpness_score: float | None = None
    exposure_score: float | None = None
    embedding_norm: float | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class PortraitMatchObservation:
    """Describe one local portrait-match result for runtime consumers."""

    checked_at: float | None = None
    state: str = "provider_unavailable"
    matches_reference_user: bool = False
    confidence: float | None = None
    similarity_score: float | None = None
    live_face_count: int | None = None
    reference_face_count: int | None = None
    reference_image_count: int | None = None
    matched_user_id: str | None = None
    matched_user_display_name: str | None = None
    candidate_user_count: int | None = None
    temporal_state: str | None = None
    temporal_observation_count: int | None = None
    fused_confidence: float | None = None
    capture_source_device: str | None = None
    live_quality_score: float | None = None
    burst_sample_count: int | None = None
    burst_success_count: int | None = None
    backend_name: str = "local_portrait_match"
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class PortraitEnrollmentResult:
    """Describe one local portrait enrollment or management attempt."""

    status: str
    user_id: str
    display_name: str | None = None
    reference_id: str | None = None
    reference_image_count: int = 0
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class _ProfileMatchCandidate:
    user_id: str
    display_name: str | None
    primary_user: bool
    similarity_score: float
    best_reference_score: float
    prototype_similarity_score: float | None
    reference_image_count: int


@dataclass(frozen=True, slots=True)
class _TemporalPortraitObservation:
    checked_at: float
    user_id: str
    state: str
    confidence: float


@dataclass(frozen=True, slots=True)
class _FrameMatchEvaluation:
    state: str
    confidence: float
    similarity_score: float | None
    live_result: PortraitEmbeddingResult
    capture_source_device: str | None
    candidate: _ProfileMatchCandidate | None = None
    second_candidate: _ProfileMatchCandidate | None = None
    candidate_user_count: int = 0
    matches_reference_user: bool = False
    detail: str | None = None


class OpenCVPortraitEmbeddingBackend:
    """Use OpenCV YuNet + SFace for local face detection and embedding."""

    def __init__(
        self,
        *,
        detector_model_path: Path,
        recognizer_model_path: Path,
        detector_score_threshold: float = _DEFAULT_DETECTOR_SCORE_THRESHOLD,
        detector_nms_threshold: float = _DEFAULT_DETECTOR_NMS_THRESHOLD,
        detector_top_k: int = _DEFAULT_DETECTOR_TOP_K,
        max_input_image_bytes: int = _DEFAULT_INPUT_IMAGE_MAX_BYTES,
        decoded_image_max_pixels: int = _DEFAULT_DECODED_IMAGE_MAX_PIXELS,
        working_image_max_side: int = _DEFAULT_WORKING_IMAGE_MAX_SIDE,
    ) -> None:
        self.detector_model_path = detector_model_path
        self.recognizer_model_path = recognizer_model_path
        self.detector_score_threshold = detector_score_threshold
        self.detector_nms_threshold = detector_nms_threshold
        self.detector_top_k = detector_top_k
        self.max_input_image_bytes = max(1024, int(max_input_image_bytes))
        self.decoded_image_max_pixels = max(256 * 256, int(decoded_image_max_pixels))
        self.working_image_max_side = max(256, int(working_image_max_side))
        self.source_image_pixels_guard = max(
            _DEFAULT_SOURCE_IMAGE_PIXELS_GUARD,
            self.decoded_image_max_pixels * 8,
        )
        self.name = "opencv_yunet_sface"
        self._cv2 = None
        self._np = None
        self._detector = None
        self._recognizer = None
        self._runtime_lock = RLock()

    def extract_embedding(
        self,
        image_bytes: bytes,
        *,
        image_label: str,
    ) -> PortraitEmbeddingResult:
        """Extract one face embedding from image bytes."""

        if not image_bytes:
            return PortraitEmbeddingResult(state="decode_failed", detail=f"{image_label}:empty_bytes")
        if len(image_bytes) > self.max_input_image_bytes:
            return PortraitEmbeddingResult(
                state="input_too_large",
                detail=f"{image_label}:input_bytes_exceeded",
            )
        inspected_size = _inspect_image_size(image_bytes)
        if inspected_size is not None:
            inspected_width, inspected_height = inspected_size
            if inspected_width * inspected_height > self.source_image_pixels_guard:
                return PortraitEmbeddingResult(
                    state="input_too_large",
                    detail=f"{image_label}:source_pixels_exceeded",
                    image_size=inspected_size,
                )

        try:
            cv2, np = self._ensure_dependencies()
            detector, recognizer = self._ensure_models()
        except ModuleNotFoundError:
            return PortraitEmbeddingResult(
                state="backend_unavailable",
                detail="opencv_python_unavailable",
            )
        except FileNotFoundError as exc:
            return PortraitEmbeddingResult(
                state="model_unavailable",
                detail=str(exc),
            )

        with self._runtime_lock:
            try:
                buffer = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            except Exception as exc:
                return PortraitEmbeddingResult(
                    state="decode_failed",
                    detail=f"{image_label}:{type(exc).__name__}",
                )
            if image is None:
                return PortraitEmbeddingResult(state="decode_failed", detail=f"{image_label}:imdecode_none")

            try:
                image = _resize_cv_image_to_budget(
                    image,
                    cv2=cv2,
                    max_pixels=self.decoded_image_max_pixels,
                    max_side=self.working_image_max_side,
                )
                height, width = image.shape[:2]
            except Exception as exc:
                return PortraitEmbeddingResult(
                    state="decode_failed",
                    detail=f"{image_label}:resize_failed:{type(exc).__name__}",
                )
            if width < 16 or height < 16:
                return PortraitEmbeddingResult(state="decode_failed", detail=f"{image_label}:image_too_small")

            try:
                detector.setInputSize((int(width), int(height)))
                _retval, faces = detector.detect(image)
            except Exception as exc:
                return PortraitEmbeddingResult(state="backend_unavailable", detail=f"detect_failed:{type(exc).__name__}")

            if faces is None:
                return PortraitEmbeddingResult(
                    state="no_face_detected",
                    face_count=0,
                    image_size=(width, height),
                )

            try:
                face_count = int(faces.shape[0])
            except Exception:
                try:
                    face_count = len(faces)
                except Exception:
                    face_count = 0
            if face_count <= 0:
                return PortraitEmbeddingResult(
                    state="no_face_detected",
                    face_count=0,
                    image_size=(width, height),
                )
            if face_count != 1:
                return PortraitEmbeddingResult(
                    state="ambiguous_face_count",
                    face_count=face_count,
                    image_size=(width, height),
                )

            face_row = faces[0]
            detector_confidence = _coerce_optional_float(face_row[-1] if len(face_row) else None)
            face_box = _extract_face_box(face_row, width=width, height=height)
            quality_score, sharpness_score, exposure_score = _estimate_face_quality(
                image=image,
                face_box=face_box,
                detector_confidence=detector_confidence,
                cv2=cv2,
            )
            try:
                aligned = recognizer.alignCrop(image, face_row)
                feature = recognizer.feature(aligned)
            except Exception as exc:
                return PortraitEmbeddingResult(
                    state="embedding_failed",
                    face_count=face_count,
                    detector_confidence=detector_confidence,
                    quality_score=quality_score,
                    face_box=face_box,
                    image_size=(width, height),
                    sharpness_score=sharpness_score,
                    exposure_score=exposure_score,
                    detail=type(exc).__name__,
                )
            if feature is None:
                return PortraitEmbeddingResult(
                    state="embedding_failed",
                    face_count=face_count,
                    detector_confidence=detector_confidence,
                    quality_score=quality_score,
                    face_box=face_box,
                    image_size=(width, height),
                    sharpness_score=sharpness_score,
                    exposure_score=exposure_score,
                )

            try:
                raw_vector = tuple(float(value) for value in feature.flatten().tolist())
            except Exception as exc:
                return PortraitEmbeddingResult(
                    state="embedding_failed",
                    face_count=face_count,
                    detector_confidence=detector_confidence,
                    quality_score=quality_score,
                    face_box=face_box,
                    image_size=(width, height),
                    sharpness_score=sharpness_score,
                    exposure_score=exposure_score,
                    detail=type(exc).__name__,
                )
            if not raw_vector or not all(math.isfinite(value) for value in raw_vector):
                return PortraitEmbeddingResult(
                    state="embedding_failed",
                    face_count=face_count,
                    detector_confidence=detector_confidence,
                    quality_score=quality_score,
                    face_box=face_box,
                    image_size=(width, height),
                    sharpness_score=sharpness_score,
                    exposure_score=exposure_score,
                    detail="non_finite_embedding",
                )

            embedding_norm = math.sqrt(sum(value * value for value in raw_vector))
            return PortraitEmbeddingResult(
                state="ok",
                embedding=_normalize_embedding(raw_vector),
                face_count=face_count,
                detector_confidence=detector_confidence,
                quality_score=quality_score,
                face_box=face_box,
                image_size=(width, height),
                sharpness_score=sharpness_score,
                exposure_score=exposure_score,
                embedding_norm=embedding_norm if math.isfinite(embedding_norm) else None,
            )

    def similarity_score(
        self,
        *,
        reference_embedding: tuple[float, ...],
        live_embedding: tuple[float, ...],
    ) -> float | None:
        """Return one finite similarity score between two embeddings."""

        try:
            cv2, np = self._ensure_dependencies()
            _detector, recognizer = self._ensure_models()
        except (ModuleNotFoundError, FileNotFoundError):
            return None
        with self._runtime_lock:
            try:
                reference = np.asarray(reference_embedding, dtype=np.float32).reshape(1, -1)
                live = np.asarray(live_embedding, dtype=np.float32).reshape(1, -1)
                metric = getattr(cv2, "FaceRecognizerSF_FR_COSINE", 0)
                score = float(recognizer.match(reference, live, metric))
            except Exception:
                return None
        if not math.isfinite(score):
            return None
        return score

    def _ensure_dependencies(self):
        if self._cv2 is None or self._np is None:
            import cv2  # type: ignore[import-not-found]
            import numpy  # type: ignore[import-not-found]

            self._cv2 = cv2
            self._np = numpy
        return self._cv2, self._np

    def _ensure_models(self):
        cv2, _np = self._ensure_dependencies()
        if self._detector is None:
            if not self.detector_model_path.is_file():
                raise FileNotFoundError(f"detector model missing: {self.detector_model_path}")
            self._detector = cv2.FaceDetectorYN.create(
                str(self.detector_model_path),
                "",
                (320, 320),
                self.detector_score_threshold,
                self.detector_nms_threshold,
                self.detector_top_k,
            )
        if self._recognizer is None:
            if not self.recognizer_model_path.is_file():
                raise FileNotFoundError(f"recognizer model missing: {self.recognizer_model_path}")
            self._recognizer = cv2.FaceRecognizerSF.create(
                str(self.recognizer_model_path),
                "",
            )
        return self._detector, self._recognizer


class PortraitMatchProvider:
    """Capture and cache bounded local portrait-match observations."""

    def __init__(
        self,
        *,
        camera: V4L2StillCamera,
        config: PortraitMatchConfig,
        camera_lock: Lock | None = None,
        backend: object | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.camera = camera
        self.config = config
        self.camera_lock = camera_lock
        self.backend = backend or OpenCVPortraitEmbeddingBackend(
            detector_model_path=config.detector_model_path,
            recognizer_model_path=config.recognizer_model_path,
            detector_score_threshold=config.detector_score_threshold,
            detector_nms_threshold=config.detector_nms_threshold,
            detector_top_k=config.detector_top_k,
            max_input_image_bytes=config.input_image_max_bytes,
            decoded_image_max_pixels=config.decoded_image_max_pixels,
            working_image_max_side=config.working_image_max_side,
        )
        _ensure_directory_mode(config.identity_store_path.parent, mode=config.secure_dir_mode)
        _ensure_directory_mode(config.identity_reference_image_dir, mode=config.secure_dir_mode)
        self.identity_store = PortraitIdentityStore(
            store_path=config.identity_store_path,
            image_dir=config.identity_reference_image_dir,
        )
        self.clock = clock
        self._lock = RLock()
        self._last_observation: PortraitMatchObservation | None = None
        self._temporal_history: list[_TemporalPortraitObservation] = []
        self._legacy_reference_bootstrap_successful = False
        self._legacy_reference_bootstrap_checked_at: float | None = None
        self._harden_identity_storage_paths()

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        camera: V4L2StillCamera,
        camera_lock: Lock | None = None,
        backend: object | None = None,
    ) -> "PortraitMatchProvider":
        """Build one portrait-match provider from ``TwinrConfig``."""

        return cls(
            camera=camera,
            config=PortraitMatchConfig.from_config(config),
            camera_lock=camera_lock,
            backend=backend,
        )

    def summary(self, *, user_id: str | None = None) -> PortraitIdentitySummary:
        """Return the current enrolled identity summary for one user."""

        with self._lock:
            self._bootstrap_legacy_reference_profile()
            return self.identity_store.summary(self._resolved_user_id(user_id))

    def list_profiles(self) -> tuple[PortraitIdentityProfile, ...]:
        """Return all enrolled portrait identity profiles."""

        with self._lock:
            self._bootstrap_legacy_reference_profile()
            return self.identity_store.list_profiles()

    def enroll_reference_image_bytes(
        self,
        image_bytes: bytes,
        *,
        user_id: str | None = None,
        display_name: str | None = None,
        filename_hint: str = "reference.jpg",
        source: str = "manual_import",
    ) -> PortraitEnrollmentResult:
        """Enroll one reference image into the local portrait identity store."""

        resolved_user_id = self._resolved_user_id(user_id)
        with self._lock:
            self._bootstrap_legacy_reference_profile()
            extraction = self.backend.extract_embedding(image_bytes, image_label="enrollment_reference")
            if extraction.state != "ok" or extraction.embedding is None:
                summary = self.identity_store.summary(resolved_user_id)
                return PortraitEnrollmentResult(
                    status=extraction.state,
                    user_id=summary.user_id,
                    display_name=summary.display_name,
                    reference_image_count=summary.reference_image_count,
                    detail=extraction.detail,
                )
            # BREAKING: low_quality_reference is now emitted explicitly instead of silently
            # accepting weak enrollment material that degrades later matching.
            if (
                extraction.quality_score is not None
                and extraction.quality_score < self.config.min_enrollment_quality
            ):
                summary = self.identity_store.summary(resolved_user_id)
                return PortraitEnrollmentResult(
                    status="low_quality_reference",
                    user_id=summary.user_id,
                    display_name=summary.display_name,
                    reference_image_count=summary.reference_image_count,
                    detail="quality_below_enrollment_threshold",
                )

            profile, reference, duplicate = self.identity_store.upsert_reference(
                user_id=resolved_user_id,
                display_name=display_name,
                primary_user=resolved_user_id == self.config.primary_user_id,
                image_bytes=image_bytes,
                filename_hint=filename_hint,
                embedding=extraction.embedding,
                detector_confidence=extraction.detector_confidence,
                source=source,
                max_reference_images=self.config.max_reference_images_per_user,
            )
            self._invalidate_cache(clear_temporal=True)
            return PortraitEnrollmentResult(
                status="duplicate_reference" if duplicate else "enrolled",
                user_id=profile.user_id,
                display_name=profile.display_name,
                reference_id=reference.reference_id,
                reference_image_count=len(profile.reference_images),
                detail=None,
            )

    def enroll_reference_image_path(
        self,
        path: str | Path,
        *,
        user_id: str | None = None,
        display_name: str | None = None,
        source: str = "manual_import",
    ) -> PortraitEnrollmentResult:
        """Enroll one reference image from disk into the local identity store."""

        resolved_path = Path(path).expanduser()
        try:
            image_bytes = _safe_read_regular_file_bytes(
                resolved_path,
                max_bytes=min(self.config.reference_max_bytes, self.config.input_image_max_bytes),
            )
        except FileNotFoundError:
            summary = self.summary(user_id=user_id)
            return PortraitEnrollmentResult(
                status="reference_image_unavailable",
                user_id=summary.user_id,
                display_name=summary.display_name,
                reference_image_count=summary.reference_image_count,
                detail="missing_reference_image",
            )
        except OSError as exc:
            summary = self.summary(user_id=user_id)
            return PortraitEnrollmentResult(
                status="reference_image_unavailable",
                user_id=summary.user_id,
                display_name=summary.display_name,
                reference_image_count=summary.reference_image_count,
                detail=str(exc),
            )
        return self.enroll_reference_image_bytes(
            image_bytes,
            user_id=user_id,
            display_name=display_name,
            filename_hint=resolved_path.name,
            source=source,
        )

    def capture_and_enroll_reference(
        self,
        *,
        user_id: str | None = None,
        display_name: str | None = None,
        source: str = "camera_capture",
    ) -> PortraitEnrollmentResult:
        """Capture one live photo and enroll it as a local portrait reference."""

        try:
            captures = self._capture_live_images(count=1)
        except CameraError as exc:
            summary = self.summary(user_id=user_id)
            return PortraitEnrollmentResult(
                status="capture_unavailable",
                user_id=summary.user_id,
                display_name=summary.display_name,
                reference_image_count=summary.reference_image_count,
                detail=str(exc),
            )
        capture = captures[0]
        return self.enroll_reference_image_bytes(
            capture.data,
            user_id=user_id,
            display_name=display_name,
            filename_hint=capture.filename or "portrait-enrollment.png",
            source=source,
        )

    def remove_reference_image(
        self,
        *,
        reference_id: str,
        user_id: str | None = None,
    ) -> PortraitEnrollmentResult:
        """Remove one enrolled reference image from the local identity store."""

        resolved_user_id = self._resolved_user_id(user_id)
        with self._lock:
            self._bootstrap_legacy_reference_profile()
            profile = self.identity_store.load_profile(resolved_user_id)
            if profile is None:
                return PortraitEnrollmentResult(status="profile_unavailable", user_id=resolved_user_id)
            if not any(image.reference_id == str(reference_id).strip() for image in profile.reference_images):
                return PortraitEnrollmentResult(
                    status="reference_not_found",
                    user_id=profile.user_id,
                    display_name=profile.display_name,
                    reference_image_count=len(profile.reference_images),
                )
            updated_profile = self.identity_store.remove_reference(
                user_id=resolved_user_id,
                reference_id=reference_id,
            )
            self._invalidate_cache(clear_temporal=True)
            return PortraitEnrollmentResult(
                status="removed",
                user_id=resolved_user_id,
                display_name=None if updated_profile is None else updated_profile.display_name,
                reference_id=str(reference_id).strip() or None,
                reference_image_count=0 if updated_profile is None else len(updated_profile.reference_images),
            )

    def clear_identity_profile(self, *, user_id: str | None = None) -> PortraitEnrollmentResult:
        """Delete all enrolled reference images for one local identity profile."""

        resolved_user_id = self._resolved_user_id(user_id)
        with self._lock:
            self._bootstrap_legacy_reference_profile()
            existing = self.identity_store.summary(resolved_user_id)
            cleared = self.identity_store.clear_profile(user_id=resolved_user_id)
            if cleared:
                self._invalidate_cache(clear_temporal=True)
            return PortraitEnrollmentResult(
                status="cleared" if cleared else "profile_unavailable",
                user_id=resolved_user_id,
                display_name=existing.display_name,
                reference_image_count=0 if cleared else existing.reference_image_count,
            )

    def observe(self, *, force_refresh: bool = False) -> PortraitMatchObservation:
        """Return one cached or freshly captured portrait-match observation."""

        with self._lock:
            self._bootstrap_legacy_reference_profile()
            now = self.clock()
            if (
                not force_refresh
                and self._last_observation is not None
                and self._last_observation.checked_at is not None
                and now - self._last_observation.checked_at <= self._cache_ttl_for(self._last_observation)
            ):
                return self._last_observation

            observation = self._observe_uncached(checked_at=now)
            self._last_observation = observation
            return observation

    def _observe_uncached(self, *, checked_at: float) -> PortraitMatchObservation:
        profiles = tuple(
            profile for profile in self.identity_store.list_profiles() if profile.reference_images
        )
        if not profiles:
            return PortraitMatchObservation(
                checked_at=checked_at,
                state="reference_image_unavailable",
                backend_name=_backend_name(self.backend),
            )

        try:
            captures = self._capture_live_images(count=self.config.live_capture_burst_count)
        except CameraError as exc:
            return PortraitMatchObservation(
                checked_at=checked_at,
                state="capture_unavailable",
                confidence=0.0,
                reference_image_count=sum(len(profile.reference_images) for profile in profiles),
                capture_source_device=getattr(self.camera, "device", None),
                backend_name=_backend_name(self.backend),
                detail=str(exc),
            )

        frame_results = [
            self._evaluate_live_frame(profiles=profiles, capture=capture)
            for capture in captures
        ]
        return self._fuse_live_frame_matches(
            checked_at=checked_at,
            profiles=profiles,
            frame_results=frame_results,
        )

    def _evaluate_live_frame(
        self,
        *,
        profiles: tuple[PortraitIdentityProfile, ...],
        capture: object,
    ) -> _FrameMatchEvaluation:
        live_result = self.backend.extract_embedding(capture.data, image_label="live_capture")
        if live_result.state != "ok" or live_result.embedding is None:
            return _FrameMatchEvaluation(
                state=live_result.state,
                confidence=0.0,
                similarity_score=None,
                live_result=live_result,
                capture_source_device=getattr(capture, "source_device", None),
                detail=live_result.detail,
            )
        # BREAKING: low_quality_face is now emitted explicitly for weak live frames.
        if (
            live_result.quality_score is not None
            and live_result.quality_score < self.config.min_live_quality
        ):
            return _FrameMatchEvaluation(
                state="low_quality_face",
                confidence=0.0,
                similarity_score=None,
                live_result=live_result,
                capture_source_device=getattr(capture, "source_device", None),
                detail="quality_below_live_threshold",
            )

        candidates = self._profile_candidates(profiles=profiles, live_embedding=live_result.embedding)
        if not candidates:
            return _FrameMatchEvaluation(
                state="backend_unavailable",
                confidence=0.0,
                similarity_score=None,
                live_result=live_result,
                capture_source_device=getattr(capture, "source_device", None),
                detail="similarity_unavailable",
            )

        primary_candidate = candidates[0]
        second_candidate = None if len(candidates) < 2 else candidates[1]
        confidence = _match_confidence(
            primary_candidate.similarity_score,
            likely_threshold=self.config.likely_threshold,
            uncertain_threshold=self.config.uncertain_threshold,
        )
        confidence = _quality_adjusted_confidence(confidence, live_result.quality_score)
        ambiguous_identity = (
            second_candidate is not None
            and primary_candidate.similarity_score >= self.config.uncertain_threshold
            and second_candidate.similarity_score >= self.config.uncertain_threshold
            and (primary_candidate.similarity_score - second_candidate.similarity_score) < self.config.identity_margin
        )
        if ambiguous_identity:
            state = "ambiguous_identity"
            matches_reference_user = False
        elif primary_candidate.similarity_score >= self.config.likely_threshold:
            if primary_candidate.primary_user:
                state = "likely_reference_user"
                matches_reference_user = True
            else:
                state = "known_other_user"
                matches_reference_user = False
        elif primary_candidate.similarity_score >= self.config.uncertain_threshold:
            state = "uncertain_match"
            matches_reference_user = False
        else:
            state = "unknown_face"
            matches_reference_user = False

        return _FrameMatchEvaluation(
            state=state,
            confidence=confidence,
            similarity_score=primary_candidate.similarity_score,
            live_result=live_result,
            capture_source_device=getattr(capture, "source_device", None),
            candidate=primary_candidate,
            second_candidate=second_candidate,
            candidate_user_count=len(candidates),
            matches_reference_user=matches_reference_user,
            detail="identity_candidates_too_close" if ambiguous_identity else None,
        )

    def _fuse_live_frame_matches(
        self,
        *,
        checked_at: float,
        profiles: tuple[PortraitIdentityProfile, ...],
        frame_results: list[_FrameMatchEvaluation],
    ) -> PortraitMatchObservation:
        total_reference_image_count = sum(len(profile.reference_images) for profile in profiles)
        candidate_frames = [frame for frame in frame_results if frame.candidate is not None]
        if not candidate_frames:
            failure = self._best_failure_frame(frame_results)
            return PortraitMatchObservation(
                checked_at=checked_at,
                state=failure.state,
                confidence=0.0,
                live_face_count=failure.live_result.face_count,
                reference_image_count=total_reference_image_count,
                capture_source_device=failure.capture_source_device,
                live_quality_score=failure.live_result.quality_score,
                burst_sample_count=len(frame_results),
                burst_success_count=0,
                backend_name=_backend_name(self.backend),
                detail=failure.detail or failure.live_result.detail,
            )

        best_frame = max(
            candidate_frames,
            key=lambda item: (
                item.confidence,
                item.similarity_score if item.similarity_score is not None else -1.0,
                item.live_result.quality_score if item.live_result.quality_score is not None else -1.0,
            ),
        )
        strong_frames = [
            frame
            for frame in candidate_frames
            if frame.state in {"likely_reference_user", "known_other_user", "uncertain_match"}
            and frame.candidate is not None
        ]
        consensus_user_id: str | None = None
        consensus_frames: list[_FrameMatchEvaluation] = []
        conflicting_frames: list[_FrameMatchEvaluation] = []
        if strong_frames:
            grouped: dict[str, list[_FrameMatchEvaluation]] = {}
            for frame in strong_frames:
                assert frame.candidate is not None
                grouped.setdefault(frame.candidate.user_id, []).append(frame)
            ordered_groups = sorted(
                grouped.items(),
                key=lambda item: (
                    len(item[1]),
                    max(candidate.confidence for candidate in item[1]),
                    max(
                        candidate.similarity_score if candidate.similarity_score is not None else -1.0
                        for candidate in item[1]
                    ),
                ),
                reverse=True,
            )
            consensus_user_id, consensus_frames = ordered_groups[0]
            for _other_user_id, other_frames in ordered_groups[1:]:
                conflicting_frames.extend(other_frames)
        elif best_frame.state == "ambiguous_identity":
            conflicting_frames = [best_frame]

        representative = best_frame
        if consensus_frames:
            representative = max(
                consensus_frames,
                key=lambda item: (
                    item.confidence,
                    item.similarity_score if item.similarity_score is not None else -1.0,
                    item.live_result.quality_score if item.live_result.quality_score is not None else -1.0,
                ),
            )

        best_conflict_frame = None if not conflicting_frames else max(
            conflicting_frames,
            key=lambda item: (
                item.confidence,
                item.similarity_score if item.similarity_score is not None else -1.0,
            ),
        )
        ambiguous_identity = representative.state == "ambiguous_identity"
        if (
            not ambiguous_identity
            and representative.candidate is not None
            and best_conflict_frame is not None
            and best_conflict_frame.candidate is not None
            and representative.similarity_score is not None
            and best_conflict_frame.similarity_score is not None
        ):
            score_gap = representative.similarity_score - best_conflict_frame.similarity_score
            same_vote_count = len(consensus_frames)
            conflict_vote_count = max(
                0,
                sum(1 for frame in conflicting_frames if frame.candidate is not None)
            )
            ambiguous_identity = (
                score_gap < self.config.identity_margin
                or (
                    conflict_vote_count > 0
                    and same_vote_count <= conflict_vote_count
                    and best_conflict_frame.confidence >= 0.58
                )
            )

        similarity_frames = consensus_frames or [representative]
        similarity_score = _fuse_similarity_scores(
            [frame.similarity_score for frame in similarity_frames if frame.similarity_score is not None]
        )
        confidence = _fuse_confidence_scores(
            [frame.confidence for frame in similarity_frames],
            consensus_count=len(consensus_frames),
            consensus_target=self.config.live_capture_min_consensus,
        )
        live_quality_score = _fuse_confidence_scores(
            [
                frame.live_result.quality_score
                for frame in similarity_frames
                if frame.live_result.quality_score is not None
            ],
            consensus_count=0,
            consensus_target=0,
        )

        if ambiguous_identity:
            state = "ambiguous_identity"
            matches_reference_user = False
            matched_user_id = None
            matched_user_display_name = None
            detail = "identity_candidates_too_close_across_burst"
            reference_image_count = representative.candidate.reference_image_count if representative.candidate else total_reference_image_count
        else:
            state = representative.state
            matches_reference_user = representative.matches_reference_user
            detail = representative.detail
            reference_image_count = (
                representative.candidate.reference_image_count
                if representative.candidate is not None
                else total_reference_image_count
            )
            if state in {"likely_reference_user", "known_other_user", "uncertain_match"}:
                matched_user_id = None if representative.candidate is None else representative.candidate.user_id
                matched_user_display_name = None if representative.candidate is None else representative.candidate.display_name
            else:
                matched_user_id = None
                matched_user_display_name = None
        if (
            detail is None
            and state in {"likely_reference_user", "known_other_user"}
            and consensus_frames
            and len(consensus_frames) < self.config.live_capture_min_consensus
        ):
            detail = "single_frame_consensus"

        temporal_state, temporal_count, fused_confidence = self._build_temporal_match(
            checked_at=checked_at,
            state=state,
            matched_user_id=matched_user_id if state in {"likely_reference_user", "known_other_user"} else None,
            confidence=confidence,
        )
        return PortraitMatchObservation(
            checked_at=checked_at,
            state=state,
            matches_reference_user=matches_reference_user,
            confidence=confidence,
            similarity_score=similarity_score,
            live_face_count=representative.live_result.face_count,
            reference_face_count=1 if representative.candidate is not None else None,
            reference_image_count=reference_image_count,
            matched_user_id=matched_user_id,
            matched_user_display_name=matched_user_display_name,
            candidate_user_count=representative.candidate_user_count,
            temporal_state=temporal_state,
            temporal_observation_count=temporal_count,
            fused_confidence=fused_confidence,
            capture_source_device=representative.capture_source_device,
            live_quality_score=live_quality_score,
            burst_sample_count=len(frame_results),
            burst_success_count=len(candidate_frames),
            backend_name=_backend_name(self.backend),
            detail=detail,
        )

    def _profile_candidates(
        self,
        *,
        profiles: tuple[PortraitIdentityProfile, ...],
        live_embedding: tuple[float, ...],
    ) -> tuple[_ProfileMatchCandidate, ...]:
        candidates: list[_ProfileMatchCandidate] = []
        for profile in profiles:
            similarities: list[float] = []
            reference_embeddings: list[tuple[float, ...]] = []
            for reference in profile.reference_images:
                embedding = tuple(reference.embedding)
                reference_embeddings.append(embedding)
                score = self.backend.similarity_score(
                    reference_embedding=embedding,
                    live_embedding=live_embedding,
                )
                if score is None:
                    continue
                similarities.append(score)
            if not similarities:
                continue
            prototype_similarity_score = None
            prototype_embedding = _mean_embedding(reference_embeddings)
            if prototype_embedding is not None:
                prototype_similarity_score = self.backend.similarity_score(
                    reference_embedding=prototype_embedding,
                    live_embedding=live_embedding,
                )
            candidate_score = _aggregate_profile_similarity(
                similarities,
                prototype_score=prototype_similarity_score,
            )
            candidates.append(
                _ProfileMatchCandidate(
                    user_id=profile.user_id,
                    display_name=profile.display_name,
                    primary_user=profile.primary_user or profile.user_id == self.config.primary_user_id,
                    similarity_score=candidate_score,
                    best_reference_score=max(score for score in similarities if math.isfinite(score)),
                    prototype_similarity_score=prototype_similarity_score,
                    reference_image_count=len(profile.reference_images),
                )
            )
        candidates.sort(
            key=lambda item: (
                item.similarity_score,
                item.best_reference_score,
                item.prototype_similarity_score if item.prototype_similarity_score is not None else -1.0,
            ),
            reverse=True,
        )
        return tuple(candidates)

    def _build_temporal_match(
        self,
        *,
        checked_at: float,
        state: str,
        matched_user_id: str | None,
        confidence: float,
    ) -> tuple[str | None, int | None, float | None]:
        self._prune_temporal_history(checked_at)
        if state in {"likely_reference_user", "known_other_user"} and matched_user_id:
            self._append_temporal_history(
                checked_at=checked_at,
                user_id=matched_user_id,
                state=state,
                confidence=confidence,
            )
            same_user = [
                observation
                for observation in self._temporal_history
                if observation.user_id == matched_user_id
            ]
            conflicting = [
                observation
                for observation in self._temporal_history
                if observation.user_id != matched_user_id and observation.confidence >= 0.78
            ]
            fused_confidence = round(
                max(
                    confidence,
                    sum(item.confidence for item in same_user) / max(1, len(same_user)),
                ),
                4,
            )
            if conflicting:
                return "recent_conflict", len(same_user), fused_confidence
            if len(same_user) >= self.config.temporal_min_observations:
                return "stable_match", len(same_user), fused_confidence
            return "insufficient_history", len(same_user), fused_confidence
        return None, None, None

    def _append_temporal_history(
        self,
        *,
        checked_at: float,
        user_id: str,
        state: str,
        confidence: float,
    ) -> None:
        if (
            self._temporal_history
            and self._temporal_history[-1].checked_at == checked_at
            and self._temporal_history[-1].user_id == user_id
            and self._temporal_history[-1].state == state
        ):
            return
        self._temporal_history.append(
            _TemporalPortraitObservation(
                checked_at=checked_at,
                user_id=user_id,
                state=state,
                confidence=confidence,
            )
        )
        self._prune_temporal_history(checked_at)

    def _prune_temporal_history(self, checked_at: float) -> None:
        min_checked_at = checked_at - self.config.temporal_window_s
        self._temporal_history = [
            observation
            for observation in self._temporal_history
            if observation.checked_at >= min_checked_at
        ]
        if len(self._temporal_history) > self.config.temporal_max_observations:
            self._temporal_history = self._temporal_history[-self.config.temporal_max_observations :]

    def _bootstrap_legacy_reference_profile(self) -> None:
        if self._legacy_reference_bootstrap_successful:
            return
        reference_path = self.config.reference_image_path
        if reference_path is None:
            self._legacy_reference_bootstrap_successful = True
            return
        now = self.clock()
        if (
            self._legacy_reference_bootstrap_checked_at is not None
            and now - self._legacy_reference_bootstrap_checked_at < self.config.legacy_bootstrap_retry_s
        ):
            return
        self._legacy_reference_bootstrap_checked_at = now
        existing = self.identity_store.load_profile(self.config.primary_user_id)
        if existing is not None and existing.reference_images:
            self._legacy_reference_bootstrap_successful = True
            return
        try:
            image_bytes = _safe_read_regular_file_bytes(
                reference_path,
                max_bytes=min(self.config.reference_max_bytes, self.config.input_image_max_bytes),
            )
        except (FileNotFoundError, OSError):
            return
        extraction = self.backend.extract_embedding(image_bytes, image_label="legacy_reference_image")
        if extraction.state != "ok" or extraction.embedding is None:
            return
        if (
            extraction.quality_score is not None
            and extraction.quality_score < self.config.min_enrollment_quality
        ):
            return
        self.identity_store.upsert_reference(
            user_id=self.config.primary_user_id,
            display_name=None,
            primary_user=True,
            image_bytes=image_bytes,
            filename_hint=reference_path.name,
            embedding=extraction.embedding,
            detector_confidence=extraction.detector_confidence,
            source="legacy_reference_image",
            max_reference_images=self.config.max_reference_images_per_user,
        )
        self._legacy_reference_bootstrap_successful = True
        self._invalidate_cache(clear_temporal=True)

    def _resolved_user_id(self, user_id: str | None) -> str:
        return _normalize_user_id(user_id or self.config.primary_user_id)

    def _invalidate_cache(self, *, clear_temporal: bool) -> None:
        self._last_observation = None
        if clear_temporal:
            self._temporal_history = []
        self._harden_identity_storage_paths()

    def _cache_ttl_for(self, observation: PortraitMatchObservation) -> float:
        max_age = max(0.0, self.config.max_age_s)
        if max_age <= 0.0:
            return 0.0
        if observation.state in {"likely_reference_user", "known_other_user"}:
            if observation.temporal_state == "stable_match":
                return max_age
            return min(max_age, 1.5)
        if observation.state in {"capture_unavailable", "backend_unavailable", "model_unavailable"}:
            return min(max_age, 2.0)
        return min(max_age, 0.75)

    def _best_failure_frame(self, frames: list[_FrameMatchEvaluation]) -> _FrameMatchEvaluation:
        priority = {
            "capture_unavailable": 9,
            "backend_unavailable": 8,
            "model_unavailable": 7,
            "low_quality_face": 6,
            "ambiguous_face_count": 5,
            "no_face_detected": 4,
            "input_too_large": 3,
            "decode_failed": 2,
            "embedding_failed": 1,
        }
        return max(
            frames,
            key=lambda item: (
                priority.get(item.state, 0),
                item.live_result.quality_score if item.live_result.quality_score is not None else -1.0,
                item.live_result.face_count,
            ),
        )

    def _capture_live_images(self, *, count: int):
        resolved_count = max(1, int(count))
        captures: list[object] = []
        interval = max(0.0, self.config.live_capture_burst_interval_s)
        if self.camera_lock is None:
            for index in range(resolved_count):
                try:
                    captures.append(self.camera.capture_photo(filename=f"portrait-match-capture-{index + 1}.png"))
                except CameraError:
                    if captures:
                        break
                    raise
                if index + 1 < resolved_count and interval > 0.0:
                    time.sleep(interval)
            return tuple(captures)
        if not self.camera_lock.acquire(timeout=self.config.capture_lock_timeout_s):
            raise CameraError("portrait_match_camera_lock_timeout")
        try:
            for index in range(resolved_count):
                try:
                    captures.append(self.camera.capture_photo(filename=f"portrait-match-capture-{index + 1}.png"))
                except CameraError:
                    if captures:
                        break
                    raise
                if index + 1 < resolved_count and interval > 0.0:
                    time.sleep(interval)
            return tuple(captures)
        finally:
            self.camera_lock.release()

    def _harden_identity_storage_paths(self) -> None:
        _ensure_directory_mode(self.config.identity_store_path.parent, mode=self.config.secure_dir_mode)
        _ensure_directory_mode(self.config.identity_reference_image_dir, mode=self.config.secure_dir_mode)
        _chmod_if_exists(self.config.identity_store_path, self.config.secure_file_mode)
        try:
            for child in self.config.identity_reference_image_dir.iterdir():
                try:
                    if child.is_file():
                        os.chmod(child, self.config.secure_file_mode)
                    elif child.is_dir():
                        os.chmod(child, self.config.secure_dir_mode)
                except OSError:
                    continue
        except OSError:
            pass


def _backend_name(backend: object) -> str:
    name = getattr(backend, "name", None)
    text = str(name or "").strip()
    return text or "local_portrait_match"


def _resolve_optional_path(*, project_root: Path, raw_value: object | None) -> Path | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    return _resolve_path(project_root=project_root, raw_value=text)


def _resolve_required_path(*, project_root: Path, raw_value: object | None) -> Path:
    text = str(raw_value or "").strip()
    if not text:
        raise ValueError("portrait-match path must not be blank")
    return _resolve_path(project_root=project_root, raw_value=text)


def _resolve_path(*, project_root: Path, raw_value: str) -> Path:
    raw_path = Path(raw_value).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve(strict=False)
    return (project_root / raw_path).resolve(strict=False)


def _normalize_thresholds(likely_threshold: object, uncertain_threshold: object) -> tuple[float, float]:
    likely = _coerce_ratio(likely_threshold, default=_DEFAULT_LIKELY_THRESHOLD)
    uncertain = _coerce_ratio(uncertain_threshold, default=_DEFAULT_UNCERTAIN_THRESHOLD)
    if likely < uncertain:
        likely, uncertain = uncertain, likely
    return likely, uncertain


def _coerce_ratio(value: object, *, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if not math.isfinite(numeric):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _coerce_positive_int(value: object, *, default: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, numeric)


def _coerce_non_negative_float(value: object, *, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric) or numeric < 0.0:
        return default
    return numeric


def _coerce_optional_float(value: object | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _coerce_mode(value: object, *, default: int) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return default
        if text.startswith("0o"):
            base = 8
        elif all(char in "01234567" for char in text):
            base = 8
        else:
            base = 10
        try:
            numeric = int(text, base)
        except ValueError:
            return default
    else:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return default
    if numeric < 0:
        return default
    return min(0o777, numeric)


def _safe_read_regular_file_bytes(path: Path, *, max_bytes: int) -> bytes:
    if path.suffix.casefold() not in _ALLOWED_IMAGE_SUFFIXES:
        raise OSError("unsupported_reference_image_type")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("reference_image_not_regular")
        if metadata.st_size > max_bytes:
            raise OSError("reference_image_too_large")
        with os.fdopen(fd, "rb") as handle:
            data = handle.read(max_bytes + 1)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    if len(data) > max_bytes:
        raise OSError("reference_image_too_large")
    return data


def _normalize_embedding(values: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0 or not math.isfinite(norm):
        return values
    return tuple(round(value / norm, 8) for value in values)


def _normalize_user_id(value: object | None, *, default: str = "main_user") -> str:
    raw = str(value or default).strip().lower()
    normalized: list[str] = []
    previous_separator = False
    for char in raw:
        if char.isalnum():
            normalized.append(char)
            previous_separator = False
            continue
        if char in {"-", "_", " "}:
            if not previous_separator:
                normalized.append("_")
                previous_separator = True
    text = "".join(normalized).strip("_")
    return text or default


def _aggregate_profile_similarity(scores: list[float], *, prototype_score: float | None = None) -> float:
    ordered = sorted(
        (score for score in scores if math.isfinite(score)),
        reverse=True,
    )
    if not ordered:
        return 0.0
    top_scores = ordered[: min(3, len(ordered))]
    best_score = top_scores[0]
    average_score = sum(top_scores) / len(top_scores)
    if prototype_score is not None and math.isfinite(prototype_score):
        return round((best_score * 0.55) + (average_score * 0.25) + (prototype_score * 0.20), 4)
    return round((best_score * 0.70) + (average_score * 0.30), 4)


def _match_confidence(
    score: float,
    *,
    likely_threshold: float,
    uncertain_threshold: float,
) -> float:
    if score >= likely_threshold:
        denominator = max(1e-6, 1.0 - likely_threshold)
        progress = min(1.0, max(0.0, (score - likely_threshold) / denominator))
        return round(0.78 + progress * 0.17, 4)
    if score >= uncertain_threshold:
        denominator = max(1e-6, likely_threshold - uncertain_threshold)
        progress = min(1.0, max(0.0, (score - uncertain_threshold) / denominator))
        return round(0.58 + progress * 0.16, 4)
    base = max(0.0, min(1.0, score))
    return round(0.18 + base * 0.24, 4)


def _quality_adjusted_confidence(confidence: float, quality_score: float | None) -> float:
    if quality_score is None:
        return confidence
    quality = max(0.0, min(1.0, quality_score))
    return round(confidence * (0.75 + 0.25 * quality), 4)


def _mean_embedding(embeddings: list[tuple[float, ...]]) -> tuple[float, ...] | None:
    if not embeddings:
        return None
    dimension = len(embeddings[0])
    if dimension <= 0:
        return None
    accumulator = [0.0] * dimension
    count = 0
    for embedding in embeddings:
        if len(embedding) != dimension:
            continue
        for index, value in enumerate(embedding):
            if not math.isfinite(value):
                break
            accumulator[index] += float(value)
        else:
            count += 1
    if count <= 0:
        return None
    return _normalize_embedding(tuple(value / count for value in accumulator))


def _fuse_similarity_scores(scores: list[float]) -> float | None:
    finite_scores = [score for score in scores if math.isfinite(score)]
    if not finite_scores:
        return None
    finite_scores.sort(reverse=True)
    top = finite_scores[: min(3, len(finite_scores))]
    return round(sum(top) / len(top), 4)


def _fuse_confidence_scores(
    scores: list[float | None],
    *,
    consensus_count: int,
    consensus_target: int,
) -> float | None:
    finite_scores = [float(score) for score in scores if score is not None and math.isfinite(score)]
    if not finite_scores:
        return None
    base = sum(finite_scores) / len(finite_scores)
    if consensus_target > 0 and consensus_count > 0:
        bonus = min(0.04, 0.02 * max(0, consensus_count - consensus_target + 1))
        return round(min(0.99, base + bonus), 4)
    return round(base, 4)


def _extract_face_box(face_row: object, *, width: int, height: int) -> tuple[int, int, int, int]:
    try:
        x = int(float(face_row[0]))
        y = int(float(face_row[1]))
        w = int(float(face_row[2]))
        h = int(float(face_row[3]))
    except Exception:
        return (0, 0, max(1, width), max(1, height))
    x = max(0, min(x, max(0, width - 1)))
    y = max(0, min(y, max(0, height - 1)))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return (x, y, w, h)


def _estimate_face_quality(*, image, face_box: tuple[int, int, int, int], detector_confidence: float | None, cv2):
    x, y, w, h = face_box
    roi = image[y : y + h, x : x + w]
    if roi is None or getattr(roi, "size", 0) <= 0:
        roi = image
        h, w = image.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if getattr(gray, "size", 0) > 0 else 0.0
    brightness = float(gray.mean()) if getattr(gray, "size", 0) > 0 else 0.0
    image_height, image_width = image.shape[:2]
    face_ratio = (max(1, w) * max(1, h)) / max(1.0, float(image_height * image_width))
    face_dimension_score = min(1.0, min(w, h) / 160.0)
    occupancy_score = min(1.0, math.sqrt(max(0.0, face_ratio) / 0.08)) if face_ratio > 0.0 else 0.0
    sharpness_score = min(1.0, math.log1p(max(0.0, blur_variance)) / math.log1p(180.0))
    exposure_score = max(0.0, 1.0 - min(1.0, abs(brightness - 120.0) / 120.0))
    detector_score = max(0.0, min(1.0, detector_confidence if detector_confidence is not None else 0.0))
    quality_score = round(
        (detector_score * 0.35)
        + (sharpness_score * 0.30)
        + (face_dimension_score * 0.20)
        + (occupancy_score * 0.10)
        + (exposure_score * 0.05),
        4,
    )
    return quality_score, round(sharpness_score, 4), round(exposure_score, 4)


def _resize_cv_image_to_budget(image, *, cv2, max_pixels: int, max_side: int):
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image
    scale = 1.0
    longest_side = max(height, width)
    if longest_side > max_side:
        scale = min(scale, float(max_side) / float(longest_side))
    pixel_count = height * width
    if pixel_count > max_pixels:
        scale = min(scale, math.sqrt(float(max_pixels) / float(pixel_count)))
    if scale >= 0.999:
        return image
    resized_width = max(16, int(round(width * scale)))
    resized_height = max(16, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)


def _inspect_image_size(image_bytes: bytes) -> tuple[int, int] | None:
    try:
        from PIL import Image  # type: ignore[import-not-found]

        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            return None
        return int(width), int(height)
    except Exception:
        return None


def _ensure_directory_mode(path: Path, *, mode: int) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=mode)
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _chmod_if_exists(path: Path, mode: int) -> None:
    try:
        if path.exists():
            os.chmod(path, mode)
    except OSError:
        pass


__all__ = [
    "OpenCVPortraitEmbeddingBackend",
    "PortraitEnrollmentResult",
    "PortraitEmbeddingResult",
    "PortraitMatchConfig",
    "PortraitMatchObservation",
    "PortraitMatchProvider",
]