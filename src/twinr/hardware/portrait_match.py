"""Perform bounded on-device portrait matching against local identity profiles.

This module keeps local face detection, embedding extraction, still-camera
capture, persistent identity enrollment, and temporal evidence fusion behind
one hardware-facing adapter. Runtime policy consumes its structured outputs
elsewhere; this module only owns the local capture and identity-match path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
_DEFAULT_CAPTURE_LOCK_TIMEOUT_S = 5.0
_DEFAULT_MAX_AGE_S = 45.0
_DEFAULT_LIKELY_THRESHOLD = 0.45
_DEFAULT_UNCERTAIN_THRESHOLD = 0.34
_DEFAULT_IDENTITY_MARGIN = 0.05
_DEFAULT_MAX_REFERENCE_IMAGES_PER_USER = 6
_DEFAULT_TEMPORAL_WINDOW_S = 300.0
_DEFAULT_TEMPORAL_MIN_OBSERVATIONS = 2
_DEFAULT_TEMPORAL_MAX_OBSERVATIONS = 12
_DEFAULT_DETECTOR_SCORE_THRESHOLD = 0.9
_DEFAULT_DETECTOR_NMS_THRESHOLD = 0.3
_DEFAULT_DETECTOR_TOP_K = 20


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
    capture_lock_timeout_s: float = _DEFAULT_CAPTURE_LOCK_TIMEOUT_S

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "PortraitMatchConfig":
        """Build one portrait-match config from ``TwinrConfig``."""

        reference_path = _resolve_optional_path(
            project_root=Path(config.project_root),
            raw_value=getattr(config, "vision_reference_image_path", None),
        )
        detector_model_path = _resolve_required_path(
            project_root=Path(config.project_root),
            raw_value=getattr(
                config,
                "portrait_match_detector_model_path",
                "state/opencv/models/face_detection_yunet_2023mar.onnx",
            ),
        )
        recognizer_model_path = _resolve_required_path(
            project_root=Path(config.project_root),
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
        return cls(
            reference_image_path=reference_path,
            detector_model_path=detector_model_path,
            recognizer_model_path=recognizer_model_path,
            identity_store_path=_resolve_path(
                project_root=Path(config.project_root),
                raw_value=getattr(config, "portrait_match_store_path", "state/portrait_identities.json"),
            ),
            identity_reference_image_dir=_resolve_path(
                project_root=Path(config.project_root),
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
            temporal_min_observations=max(
                1,
                _coerce_positive_int(
                    getattr(
                        config,
                        "portrait_match_temporal_min_observations",
                        _DEFAULT_TEMPORAL_MIN_OBSERVATIONS,
                    ),
                    default=_DEFAULT_TEMPORAL_MIN_OBSERVATIONS,
                ),
            ),
            temporal_max_observations=max(
                1,
                _coerce_positive_int(
                    getattr(
                        config,
                        "portrait_match_temporal_max_observations",
                        _DEFAULT_TEMPORAL_MAX_OBSERVATIONS,
                    ),
                    default=_DEFAULT_TEMPORAL_MAX_OBSERVATIONS,
                ),
            ),
            max_age_s=_coerce_non_negative_float(
                getattr(config, "portrait_match_max_age_s", _DEFAULT_MAX_AGE_S),
                default=_DEFAULT_MAX_AGE_S,
            ),
            likely_threshold=likely_threshold,
            uncertain_threshold=uncertain_threshold,
            reference_max_bytes=max(
                1024,
                _coerce_positive_int(
                    getattr(config, "vision_reference_image_max_bytes", _DEFAULT_REFERENCE_MAX_BYTES),
                    default=_DEFAULT_REFERENCE_MAX_BYTES,
                ),
            ),
            capture_lock_timeout_s=_coerce_non_negative_float(
                getattr(config, "portrait_match_capture_lock_timeout_s", _DEFAULT_CAPTURE_LOCK_TIMEOUT_S),
                default=_DEFAULT_CAPTURE_LOCK_TIMEOUT_S,
            ),
        )


@dataclass(frozen=True, slots=True)
class PortraitEmbeddingResult:
    """Describe one embedding extraction attempt."""

    state: str
    embedding: tuple[float, ...] | None = None
    face_count: int = 0
    detector_confidence: float | None = None
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
    reference_image_count: int


@dataclass(frozen=True, slots=True)
class _TemporalPortraitObservation:
    checked_at: float
    user_id: str
    state: str
    confidence: float


class OpenCVPortraitEmbeddingBackend:
    """Use OpenCV YuNet + SFace for local face detection and embedding."""

    def __init__(
        self,
        *,
        detector_model_path: Path,
        recognizer_model_path: Path,
    ) -> None:
        self.detector_model_path = detector_model_path
        self.recognizer_model_path = recognizer_model_path
        self.name = "opencv_yunet_sface"
        self._cv2 = None
        self._np = None
        self._detector = None
        self._recognizer = None

    def extract_embedding(
        self,
        image_bytes: bytes,
        *,
        image_label: str,
    ) -> PortraitEmbeddingResult:
        """Extract one face embedding from image bytes."""

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

        try:
            buffer = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except Exception as exc:
            return PortraitEmbeddingResult(state="decode_failed", detail=f"{image_label}:{type(exc).__name__}")
        if image is None:
            return PortraitEmbeddingResult(state="decode_failed", detail=f"{image_label}:imdecode_none")

        try:
            height, width = image.shape[:2]
        except Exception:
            return PortraitEmbeddingResult(state="decode_failed", detail=f"{image_label}:invalid_shape")
        if width < 16 or height < 16:
            return PortraitEmbeddingResult(state="decode_failed", detail=f"{image_label}:image_too_small")

        try:
            detector.setInputSize((int(width), int(height)))
            _retval, faces = detector.detect(image)
        except Exception as exc:
            return PortraitEmbeddingResult(state="backend_unavailable", detail=f"detect_failed:{type(exc).__name__}")

        if faces is None:
            return PortraitEmbeddingResult(state="no_face_detected", face_count=0)

        try:
            face_count = int(faces.shape[0])
        except Exception:
            try:
                face_count = len(faces)
            except Exception:
                face_count = 0
        if face_count <= 0:
            return PortraitEmbeddingResult(state="no_face_detected", face_count=0)
        if face_count != 1:
            return PortraitEmbeddingResult(state="ambiguous_face_count", face_count=face_count)

        face_row = faces[0]
        try:
            aligned = recognizer.alignCrop(image, face_row)
            feature = recognizer.feature(aligned)
        except Exception as exc:
            return PortraitEmbeddingResult(state="embedding_failed", face_count=face_count, detail=type(exc).__name__)
        if feature is None:
            return PortraitEmbeddingResult(state="embedding_failed", face_count=face_count)

        try:
            vector = tuple(float(value) for value in feature.flatten().tolist())
        except Exception as exc:
            return PortraitEmbeddingResult(state="embedding_failed", face_count=face_count, detail=type(exc).__name__)
        if not vector or not all(math.isfinite(value) for value in vector):
            return PortraitEmbeddingResult(state="embedding_failed", face_count=face_count, detail="non_finite_embedding")
        return PortraitEmbeddingResult(
            state="ok",
            embedding=_normalize_embedding(vector),
            face_count=face_count,
            detector_confidence=_coerce_optional_float(face_row[-1] if len(face_row) else None),
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
                _DEFAULT_DETECTOR_SCORE_THRESHOLD,
                _DEFAULT_DETECTOR_NMS_THRESHOLD,
                _DEFAULT_DETECTOR_TOP_K,
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
        )
        self.identity_store = PortraitIdentityStore(
            store_path=config.identity_store_path,
            image_dir=config.identity_reference_image_dir,
        )
        self.clock = clock
        self._lock = RLock()
        self._last_observation: PortraitMatchObservation | None = None
        self._temporal_history: list[_TemporalPortraitObservation] = []
        self._legacy_reference_bootstrapped = False

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
                max_bytes=self.config.reference_max_bytes,
            )
        except FileNotFoundError:
            summary = self.identity_store.summary(self._resolved_user_id(user_id))
            return PortraitEnrollmentResult(
                status="reference_image_unavailable",
                user_id=summary.user_id,
                display_name=summary.display_name,
                reference_image_count=summary.reference_image_count,
                detail="missing_reference_image",
            )
        except OSError as exc:
            summary = self.identity_store.summary(self._resolved_user_id(user_id))
            return PortraitEnrollmentResult(
                status="reference_image_unavailable",
                user_id=summary.user_id,
                display_name=summary.display_name,
                reference_image_count=summary.reference_image_count,
                detail=type(exc).__name__,
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
            capture = self._capture_live_image()
        except CameraError as exc:
            summary = self.identity_store.summary(self._resolved_user_id(user_id))
            return PortraitEnrollmentResult(
                status="capture_unavailable",
                user_id=summary.user_id,
                display_name=summary.display_name,
                reference_image_count=summary.reference_image_count,
                detail=str(exc),
            )
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

    def observe(self) -> PortraitMatchObservation:
        """Return one cached or freshly captured portrait-match observation."""

        with self._lock:
            self._bootstrap_legacy_reference_profile()
            now = self.clock()
            if (
                self._last_observation is not None
                and self._last_observation.checked_at is not None
                and now - self._last_observation.checked_at <= self.config.max_age_s
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
            capture = self._capture_live_image()
        except CameraError as exc:
            return PortraitMatchObservation(
                checked_at=checked_at,
                state="capture_unavailable",
                confidence=0.0,
                reference_image_count=sum(len(profile.reference_images) for profile in profiles),
                capture_source_device=self.camera.device,
                backend_name=_backend_name(self.backend),
                detail=str(exc),
            )

        live_result = self.backend.extract_embedding(capture.data, image_label="live_capture")
        if live_result.state != "ok" or live_result.embedding is None:
            return PortraitMatchObservation(
                checked_at=checked_at,
                state=live_result.state,
                confidence=0.0,
                live_face_count=live_result.face_count,
                reference_image_count=sum(len(profile.reference_images) for profile in profiles),
                capture_source_device=capture.source_device,
                backend_name=_backend_name(self.backend),
                detail=live_result.detail,
            )

        candidates = self._profile_candidates(profiles=profiles, live_embedding=live_result.embedding)
        if not candidates:
            return PortraitMatchObservation(
                checked_at=checked_at,
                state="backend_unavailable",
                confidence=0.0,
                live_face_count=live_result.face_count,
                reference_image_count=sum(len(profile.reference_images) for profile in profiles),
                capture_source_device=capture.source_device,
                backend_name=_backend_name(self.backend),
                detail="similarity_unavailable",
            )

        primary_candidate = candidates[0]
        second_candidate = None if len(candidates) < 2 else candidates[1]
        confidence = _match_confidence(
            primary_candidate.similarity_score,
            likely_threshold=self.config.likely_threshold,
            uncertain_threshold=self.config.uncertain_threshold,
        )
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

        temporal_state, temporal_count, fused_confidence = self._build_temporal_match(
            checked_at=checked_at,
            state=state,
            matched_user_id=None if ambiguous_identity else primary_candidate.user_id,
            confidence=confidence,
        )
        return PortraitMatchObservation(
            checked_at=checked_at,
            state=state,
            matches_reference_user=matches_reference_user,
            confidence=confidence,
            similarity_score=primary_candidate.similarity_score,
            live_face_count=live_result.face_count,
            reference_face_count=1,
            reference_image_count=primary_candidate.reference_image_count,
            matched_user_id=primary_candidate.user_id,
            matched_user_display_name=primary_candidate.display_name,
            candidate_user_count=len(candidates),
            temporal_state=temporal_state,
            temporal_observation_count=temporal_count,
            fused_confidence=fused_confidence,
            capture_source_device=capture.source_device,
            backend_name=_backend_name(self.backend),
            detail=(
                "identity_candidates_too_close"
                if ambiguous_identity
                else None
            ),
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
            for reference in profile.reference_images:
                score = self.backend.similarity_score(
                    reference_embedding=reference.embedding,
                    live_embedding=live_embedding,
                )
                if score is None:
                    continue
                similarities.append(score)
            if not similarities:
                continue
            candidates.append(
                _ProfileMatchCandidate(
                    user_id=profile.user_id,
                    display_name=profile.display_name,
                    primary_user=profile.primary_user or profile.user_id == self.config.primary_user_id,
                    similarity_score=_aggregate_profile_similarity(similarities),
                    reference_image_count=len(profile.reference_images),
                )
            )
        candidates.sort(key=lambda item: item.similarity_score, reverse=True)
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
        if self._legacy_reference_bootstrapped:
            return
        self._legacy_reference_bootstrapped = True
        reference_path = self.config.reference_image_path
        if reference_path is None:
            return
        existing = self.identity_store.load_profile(self.config.primary_user_id)
        if existing is not None and existing.reference_images:
            return
        try:
            image_bytes = _safe_read_regular_file_bytes(
                reference_path,
                max_bytes=self.config.reference_max_bytes,
            )
        except (FileNotFoundError, OSError):
            return
        extraction = self.backend.extract_embedding(image_bytes, image_label="legacy_reference_image")
        if extraction.state != "ok" or extraction.embedding is None:
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
        self._invalidate_cache(clear_temporal=True)

    def _resolved_user_id(self, user_id: str | None) -> str:
        return _normalize_user_id(user_id or self.config.primary_user_id)

    def _invalidate_cache(self, *, clear_temporal: bool) -> None:
        with self._lock:
            self._last_observation = None
            if clear_temporal:
                self._temporal_history = []

    def _capture_live_image(self):
        if self.camera_lock is None:
            return self.camera.capture_photo(filename="portrait-match-capture.png")
        if not self.camera_lock.acquire(timeout=self.config.capture_lock_timeout_s):
            raise CameraError("portrait_match_camera_lock_timeout")
        try:
            return self.camera.capture_photo(filename="portrait-match-capture.png")
        finally:
            self.camera_lock.release()


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


def _aggregate_profile_similarity(scores: list[float]) -> float:
    if not scores:
        return 0.0
    ordered = sorted(
        (score for score in scores if math.isfinite(score)),
        reverse=True,
    )
    if not ordered:
        return 0.0
    top_scores = ordered[: min(3, len(ordered))]
    best_score = top_scores[0]
    average_score = sum(top_scores) / len(top_scores)
    return round((best_score * 0.65) + (average_score * 0.35), 4)


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


__all__ = [
    "OpenCVPortraitEmbeddingBackend",
    "PortraitEnrollmentResult",
    "PortraitEmbeddingResult",
    "PortraitMatchConfig",
    "PortraitMatchObservation",
    "PortraitMatchProvider",
]
