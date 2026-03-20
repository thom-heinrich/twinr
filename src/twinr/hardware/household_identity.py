"""Manage shared local household identity across face, voice, and feedback.

This module owns the bounded household-level identity manager used by agent
tools and runtime voice assessment. It combines portrait enrollment and live
matching, multi-user voice enrollment and matching, explicit confirm/deny
feedback, and short in-memory session history into one conservative local
identity surface.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import threading
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.portrait_match import PortraitEnrollmentResult, PortraitMatchObservation, PortraitMatchProvider

from .camera import V4L2StillCamera
from .household_voice_identity import (
    HouseholdVoiceAssessment,
    HouseholdVoiceIdentityMonitor,
    HouseholdVoiceSummary,
)


_STORE_VERSION = 1
_DEFAULT_FEEDBACK_STORE_PATH = "state/household_identity_feedback.json"
_DEFAULT_SESSION_WINDOW_S = 300.0
_DEFAULT_SESSION_MAX_EVENTS = 24
_DEFAULT_FEEDBACK_MAX_EVENTS = 128


def _utc_iso(value: datetime | None = None) -> str:
    moment = datetime.now(UTC) if value is None else value
    if moment.tzinfo is None or moment.utcoffset() is None:
        moment = moment.replace(tzinfo=UTC)
    else:
        moment = moment.astimezone(UTC)
    return moment.isoformat()


def _normalize_text(value: object | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _normalize_user_id(value: object | None, *, default: str = "main_user") -> str:
    raw = _normalize_text(value or default).strip().lower()
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


def _normalize_display_name(value: object | None) -> str | None:
    text = _normalize_text(value).strip()
    return text or None


def _resolve_feedback_store_path(config: TwinrConfig) -> Path:
    project_root = Path(config.project_root).expanduser().resolve(strict=False)
    raw_value = getattr(config, "household_identity_feedback_store_path", None)
    text = str(raw_value or "").strip() or _DEFAULT_FEEDBACK_STORE_PATH
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    return (project_root / candidate).resolve(strict=False)


def _coerce_positive_int(value: object | None, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _coerce_ratio(value: object | None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return max(0.0, min(1.0, parsed))


@dataclass(frozen=True, slots=True)
class HouseholdIdentityFeedbackEvent:
    """Persist explicit user confirmation or rejection of a household match."""

    user_id: str
    display_name: str | None
    outcome: str
    modalities: tuple[str, ...]
    created_at: str
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "outcome": self.outcome,
            "modalities": list(self.modalities),
            "created_at": self.created_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "HouseholdIdentityFeedbackEvent | None":
        outcome = _normalize_text(payload.get("outcome")).strip().lower()
        if outcome not in {"confirm", "deny"}:
            return None
        modalities_raw = payload.get("modalities", ())
        if not isinstance(modalities_raw, list):
            return None
        modalities: list[str] = []
        for item in modalities_raw:
            normalized = _normalize_text(item).strip().lower()
            if normalized in {"face", "voice", "session"} and normalized not in modalities:
                modalities.append(normalized)
        return cls(
            user_id=_normalize_user_id(payload.get("user_id")),
            display_name=_normalize_display_name(payload.get("display_name")),
            outcome=outcome,
            modalities=tuple(modalities),
            created_at=_normalize_text(payload.get("created_at")).strip() or _utc_iso(),
            source=_normalize_text(payload.get("source")).strip() or "agent_tool",
        )


@dataclass(frozen=True, slots=True)
class HouseholdIdentityQuality:
    """Describe how complete and trustworthy one enrolled member profile is."""

    score: float
    state: str
    face_reference_count: int
    voice_sample_count: int
    confirm_count: int
    deny_count: int
    recommended_next_step: str
    guidance_hints: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HouseholdIdentityMemberStatus:
    """Summarize one local household identity member."""

    user_id: str
    display_name: str | None
    primary_user: bool
    portrait_reference_count: int
    voice_sample_count: int
    confirm_count: int
    deny_count: int
    quality: HouseholdIdentityQuality


@dataclass(frozen=True, slots=True)
class HouseholdIdentityObservation:
    """Describe the current conservative live household identity signal."""

    state: str
    matched_user_id: str | None
    matched_user_display_name: str | None
    confidence: float | None
    modalities: tuple[str, ...]
    temporal_state: str | None
    session_support_ratio: float | None
    session_observation_count: int
    policy_recommendation: str
    block_reason: str | None
    voice_assessment: HouseholdVoiceAssessment | None
    portrait_observation: PortraitMatchObservation | None


@dataclass(frozen=True, slots=True)
class HouseholdIdentityStatus:
    """Return the full shared household identity status for one tool call."""

    primary_user_id: str
    members: tuple[HouseholdIdentityMemberStatus, ...]
    current_observation: HouseholdIdentityObservation | None


@dataclass(frozen=True, slots=True)
class _SessionObservation:
    observed_at: float
    user_id: str | None
    confidence: float | None
    modalities: tuple[str, ...]
    conflict: bool


class HouseholdIdentityFeedbackStore:
    """Persist bounded explicit identity feedback events on device."""

    def __init__(self, path: str | Path, *, max_events: int = _DEFAULT_FEEDBACK_MAX_EVENTS) -> None:
        self.path = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
        self.max_events = max(1, int(max_events))
        self._process_lock = threading.RLock()

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "HouseholdIdentityFeedbackStore":
        return cls(
            _resolve_feedback_store_path(config),
            max_events=_coerce_positive_int(
                getattr(config, "household_identity_feedback_max_events", _DEFAULT_FEEDBACK_MAX_EVENTS),
                default=_DEFAULT_FEEDBACK_MAX_EVENTS,
            ),
        )

    def list_events(self) -> tuple[HouseholdIdentityFeedbackEvent, ...]:
        parent = self.path.parent
        if not parent.exists():
            return ()
        try:
            with self._exclusive_lock(create_parent=False):
                return self._load_unlocked()
        except OSError:
            return ()

    def append(self, event: HouseholdIdentityFeedbackEvent) -> HouseholdIdentityFeedbackEvent:
        with self._exclusive_lock(create_parent=True):
            events = list(self._load_unlocked())
            events.append(event)
            if len(events) > self.max_events:
                events = events[-self.max_events :]
            self._write_unlocked(events)
        return event

    @contextmanager
    def _exclusive_lock(self, *, create_parent: bool) -> Iterator[None]:
        parent = self.path.parent
        if create_parent:
            parent.mkdir(parents=True, exist_ok=True)
        elif not parent.exists():
            raise FileNotFoundError(parent)
        lock_path = parent / f".{self.path.name}.lock"
        with self._process_lock:
            with open(lock_path, "a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load_unlocked(self) -> tuple[HouseholdIdentityFeedbackEvent, ...]:
        if not self.path.exists():
            return ()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.path, flags)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("household_identity_feedback_store_not_regular")
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        if not isinstance(payload, dict):
            return ()
        version = int(payload.get("store_version", _STORE_VERSION) or _STORE_VERSION)
        if version != _STORE_VERSION:
            return ()
        events_raw = payload.get("events", ())
        if not isinstance(events_raw, list):
            return ()
        events: list[HouseholdIdentityFeedbackEvent] = []
        for item in events_raw:
            if not isinstance(item, dict):
                return ()
            parsed = HouseholdIdentityFeedbackEvent.from_dict(item)
            if parsed is None:
                return ()
            events.append(parsed)
        return tuple(events)

    def _write_unlocked(self, events: list[HouseholdIdentityFeedbackEvent]) -> None:
        payload = {
            "store_version": _STORE_VERSION,
            "events": [event.to_dict() for event in events],
        }
        encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass


class HouseholdIdentityManager:
    """Coordinate local household identity enrollment, matching, and feedback."""

    def __init__(
        self,
        *,
        config: TwinrConfig,
        portrait_provider: PortraitMatchProvider,
        voice_monitor: HouseholdVoiceIdentityMonitor,
        feedback_store: HouseholdIdentityFeedbackStore,
        clock: callable = time.monotonic,
    ) -> None:
        self.config = config
        self.primary_user_id = _normalize_user_id(getattr(config, "portrait_match_primary_user_id", "main_user") or "main_user")
        self.portrait_provider = portrait_provider
        self.voice_monitor = voice_monitor
        self.feedback_store = feedback_store
        self.clock = clock
        self._lock = threading.RLock()
        self._session_history: list[_SessionObservation] = []
        self._session_window_s = float(
            max(
                30.0,
                getattr(config, "household_identity_session_window_s", _DEFAULT_SESSION_WINDOW_S) or _DEFAULT_SESSION_WINDOW_S,
            )
        )
        self._session_max_events = _coerce_positive_int(
            getattr(config, "household_identity_session_max_events", _DEFAULT_SESSION_MAX_EVENTS),
            default=_DEFAULT_SESSION_MAX_EVENTS,
        )
        self._last_observation: HouseholdIdentityObservation | None = None

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        camera: V4L2StillCamera,
        camera_lock=None,
    ) -> "HouseholdIdentityManager":
        return cls(
            config=config,
            portrait_provider=PortraitMatchProvider.from_config(
                config,
                camera=camera,
                camera_lock=camera_lock,
            ),
            voice_monitor=HouseholdVoiceIdentityMonitor.from_config(config),
            feedback_store=HouseholdIdentityFeedbackStore.from_config(config),
        )

    def list_members(self) -> tuple[HouseholdIdentityMemberStatus, ...]:
        portrait_profiles = {
            profile.user_id: profile
            for profile in self.portrait_provider.list_profiles()
        }
        voice_profiles = {
            summary.user_id: summary
            for summary in self.voice_monitor.list_summaries()
        }
        feedback_by_user = self._feedback_counts()
        user_ids = sorted(set(portrait_profiles) | set(voice_profiles), key=str.casefold)
        members: list[HouseholdIdentityMemberStatus] = []
        for user_id in user_ids:
            portrait_profile = portrait_profiles.get(user_id)
            voice_summary = voice_profiles.get(user_id)
            display_name = None
            if portrait_profile is not None and portrait_profile.display_name:
                display_name = portrait_profile.display_name
            elif voice_summary is not None and voice_summary.display_name:
                display_name = voice_summary.display_name
            confirm_count, deny_count = feedback_by_user.get(user_id, (0, 0))
            quality = _member_quality(
                portrait_reference_count=0 if portrait_profile is None else len(portrait_profile.reference_images),
                voice_sample_count=0 if voice_summary is None else voice_summary.sample_count,
                confirm_count=confirm_count,
                deny_count=deny_count,
            )
            members.append(
                HouseholdIdentityMemberStatus(
                    user_id=user_id,
                    display_name=display_name,
                    primary_user=bool(
                        (portrait_profile is not None and portrait_profile.primary_user)
                        or (voice_summary is not None and voice_summary.primary_user)
                        or user_id == self.primary_user_id
                    ),
                    portrait_reference_count=0 if portrait_profile is None else len(portrait_profile.reference_images),
                    voice_sample_count=0 if voice_summary is None else voice_summary.sample_count,
                    confirm_count=confirm_count,
                    deny_count=deny_count,
                    quality=quality,
                )
            )
        return tuple(members)

    def enroll_face(
        self,
        *,
        user_id: str | None = None,
        display_name: str | None = None,
        source: str = "tool_household_identity_face",
    ) -> tuple[PortraitEnrollmentResult, HouseholdIdentityMemberStatus | None]:
        resolved_user_id = _normalize_user_id(user_id, default=self.primary_user_id)
        result = self.portrait_provider.capture_and_enroll_reference(
            user_id=resolved_user_id,
            display_name=display_name,
            source=source,
        )
        return result, self._member_status_or_none(resolved_user_id)

    def enroll_voice(
        self,
        audio_pcm: bytes,
        *,
        sample_rate: int,
        channels: int,
        user_id: str | None = None,
        display_name: str | None = None,
    ) -> tuple[HouseholdVoiceSummary, HouseholdIdentityMemberStatus | None]:
        resolved_user_id = _normalize_user_id(user_id, default=self.primary_user_id)
        summary = self.voice_monitor.enroll_pcm16(
            audio_pcm,
            sample_rate=sample_rate,
            channels=channels,
            user_id=resolved_user_id,
            display_name=display_name,
        )
        return summary, self._member_status_or_none(resolved_user_id)

    def assess_voice(
        self,
        audio_pcm: bytes,
        *,
        sample_rate: int,
        channels: int,
    ) -> HouseholdVoiceAssessment:
        return self.voice_monitor.assess_pcm16(
            audio_pcm,
            sample_rate=sample_rate,
            channels=channels,
        )

    def observe(
        self,
        *,
        audio_pcm: bytes | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
    ) -> HouseholdIdentityObservation:
        with self._lock:
            portrait_observation = self._safe_portrait_observation()
            voice_assessment = self._safe_voice_assessment(
                audio_pcm=audio_pcm,
                sample_rate=sample_rate,
                channels=channels,
            )
            observation = self._build_observation(
                portrait_observation=portrait_observation,
                voice_assessment=voice_assessment,
            )
            self._last_observation = observation
            return observation

    def status(
        self,
        *,
        audio_pcm: bytes | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
    ) -> HouseholdIdentityStatus:
        return HouseholdIdentityStatus(
            primary_user_id=self.primary_user_id,
            members=self.list_members(),
            current_observation=self.observe(
                audio_pcm=audio_pcm,
                sample_rate=sample_rate,
                channels=channels,
            ),
        )

    def record_feedback(
        self,
        *,
        outcome: str,
        user_id: str | None = None,
        display_name: str | None = None,
    ) -> tuple[HouseholdIdentityFeedbackEvent, HouseholdIdentityMemberStatus | None]:
        normalized_outcome = _normalize_text(outcome).strip().lower()
        if normalized_outcome not in {"confirm", "deny"}:
            raise ValueError("feedback outcome must be confirm or deny")
        with self._lock:
            resolved_user_id = _normalize_user_id(
                user_id or (None if self._last_observation is None else self._last_observation.matched_user_id),
                default=self.primary_user_id,
            )
            if not resolved_user_id:
                raise ValueError("No household identity candidate is available to confirm or deny.")
            observation = self._last_observation
            modalities = ()
            if observation is not None and observation.matched_user_id == resolved_user_id:
                modalities = observation.modalities
            event = HouseholdIdentityFeedbackEvent(
                user_id=resolved_user_id,
                display_name=_normalize_display_name(display_name)
                or (None if observation is None else observation.matched_user_display_name),
                outcome=normalized_outcome,
                modalities=tuple(modalities),
                created_at=_utc_iso(),
                source="agent_tool",
            )
            self.feedback_store.append(event)
            return event, self._member_status_or_none(resolved_user_id)

    def _safe_portrait_observation(self) -> PortraitMatchObservation | None:
        try:
            return self.portrait_provider.observe()
        except Exception:
            return None

    def _safe_voice_assessment(
        self,
        *,
        audio_pcm: bytes | None,
        sample_rate: int | None,
        channels: int | None,
    ) -> HouseholdVoiceAssessment | None:
        if not audio_pcm:
            return None
        if not sample_rate or not channels:
            return None
        try:
            return self.voice_monitor.assess_pcm16(
                audio_pcm,
                sample_rate=sample_rate,
                channels=channels,
            )
        except Exception:
            return None

    def _build_observation(
        self,
        *,
        portrait_observation: PortraitMatchObservation | None,
        voice_assessment: HouseholdVoiceAssessment | None,
    ) -> HouseholdIdentityObservation:
        portrait_user_id = None if portrait_observation is None else _normalize_user_id(
            portrait_observation.matched_user_id,
            default="",
        ) or None
        portrait_display_name = None if portrait_observation is None else portrait_observation.matched_user_display_name
        portrait_confidence = None if portrait_observation is None else (
            portrait_observation.fused_confidence
            if portrait_observation.fused_confidence is not None
            else portrait_observation.confidence
        )
        voice_user_id = None if voice_assessment is None else _normalize_user_id(
            voice_assessment.matched_user_id,
            default="",
        ) or None
        voice_display_name = None if voice_assessment is None else voice_assessment.matched_user_display_name
        voice_confidence = None if voice_assessment is None else voice_assessment.confidence

        state = "no_identity_signal"
        matched_user_id: str | None = None
        matched_display_name: str | None = None
        confidence: float | None = None
        modalities: tuple[str, ...] = ()
        policy_recommendation = "confirm_first"
        block_reason: str | None = "identity_signal_unavailable"
        conflict = False

        if portrait_user_id and voice_user_id:
            if portrait_user_id == voice_user_id:
                matched_user_id = portrait_user_id
                matched_display_name = portrait_display_name or voice_display_name
                confidence = _mean_confidence(portrait_confidence, voice_confidence)
                modalities = ("face", "voice")
                state = "multimodal_match"
                policy_recommendation = (
                    "calm_personalization_only"
                    if matched_user_id == self.primary_user_id and confidence is not None and confidence >= 0.72
                    else "confirm_first"
                )
                block_reason = None if policy_recommendation == "calm_personalization_only" else "requires_confirmation"
            else:
                matched_user_id = portrait_user_id
                matched_display_name = portrait_display_name
                confidence = _mean_confidence(portrait_confidence, voice_confidence)
                modalities = ("face", "voice")
                state = "modality_conflict"
                policy_recommendation = "confirm_first"
                block_reason = "face_voice_identity_conflict"
                conflict = True
        elif portrait_user_id:
            matched_user_id = portrait_user_id
            matched_display_name = portrait_display_name
            confidence = portrait_confidence
            modalities = ("face",)
            state = "portrait_only"
            policy_recommendation = "confirm_first"
            block_reason = "voice_signal_unavailable"
        elif voice_user_id:
            matched_user_id = voice_user_id
            matched_display_name = voice_display_name
            confidence = voice_confidence
            modalities = ("voice",)
            state = "voice_only"
            policy_recommendation = (
                "calm_personalization_only"
                if matched_user_id == self.primary_user_id and voice_assessment is not None and voice_assessment.status == "likely_user"
                else "confirm_first"
            )
            block_reason = None if policy_recommendation == "calm_personalization_only" else (
                None if voice_assessment is None else voice_assessment.status
            )
        elif voice_assessment is not None and voice_assessment.status == "ambiguous_match":
            confidence = voice_confidence
            modalities = ("voice",)
            state = "ambiguous_voice_match"
            block_reason = "ambiguous_voice_match"
        elif portrait_observation is not None and portrait_observation.state == "ambiguous_identity":
            confidence = portrait_confidence
            modalities = ("face",)
            state = "ambiguous_face_match"
            block_reason = "ambiguous_face_match"

        temporal_state = None
        support_ratio = None
        observation_count = 0
        now = float(self.clock())
        self._prune_session_history(now)
        if matched_user_id or conflict:
            self._session_history.append(
                _SessionObservation(
                    observed_at=now,
                    user_id=matched_user_id,
                    confidence=confidence,
                    modalities=modalities,
                    conflict=conflict,
                )
            )
            if len(self._session_history) > self._session_max_events:
                self._session_history = self._session_history[-self._session_max_events :]
        if matched_user_id:
            support = [item for item in self._session_history if item.user_id == matched_user_id]
            observation_count = len(support)
            support_ratio = round(observation_count / max(1, len(self._session_history)), 4)
            if any(item.conflict for item in self._session_history[-3:]):
                temporal_state = "recent_conflict"
            elif observation_count >= 3 and support_ratio >= 0.67:
                temporal_state = "stable_session"
            else:
                temporal_state = "insufficient_history"

        return HouseholdIdentityObservation(
            state=state,
            matched_user_id=matched_user_id,
            matched_user_display_name=matched_display_name,
            confidence=confidence,
            modalities=modalities,
            temporal_state=temporal_state,
            session_support_ratio=support_ratio,
            session_observation_count=observation_count,
            policy_recommendation=policy_recommendation,
            block_reason=block_reason,
            voice_assessment=voice_assessment,
            portrait_observation=portrait_observation,
        )

    def _prune_session_history(self, now: float) -> None:
        minimum_time = now - self._session_window_s
        self._session_history = [item for item in self._session_history if item.observed_at >= minimum_time]

    def _feedback_counts(self) -> dict[str, tuple[int, int]]:
        counts: dict[str, tuple[int, int]] = {}
        for event in self.feedback_store.list_events():
            confirm_count, deny_count = counts.get(event.user_id, (0, 0))
            if event.outcome == "confirm":
                confirm_count += 1
            else:
                deny_count += 1
            counts[event.user_id] = (confirm_count, deny_count)
        return counts

    def _member_status_or_none(self, user_id: str) -> HouseholdIdentityMemberStatus | None:
        for member in self.list_members():
            if member.user_id == _normalize_user_id(user_id):
                return member
        return None


def _mean_confidence(*values: float | None) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return round(sum(usable) / len(usable), 4)


def _member_quality(
    *,
    portrait_reference_count: int,
    voice_sample_count: int,
    confirm_count: int,
    deny_count: int,
) -> HouseholdIdentityQuality:
    face_score = 0.0
    if portrait_reference_count >= 3:
        face_score = 0.92
    elif portrait_reference_count == 2:
        face_score = 0.8
    elif portrait_reference_count == 1:
        face_score = 0.48

    voice_score = 0.0
    if voice_sample_count >= 4:
        voice_score = 0.9
    elif voice_sample_count >= 2:
        voice_score = 0.76
    elif voice_sample_count == 1:
        voice_score = 0.44

    total_feedback = confirm_count + deny_count
    feedback_score = 0.5
    if total_feedback > 0:
        feedback_score = max(
            0.0,
            min(1.0, 0.5 + (((confirm_count - deny_count) / total_feedback) * 0.25)),
        )

    weighted_parts: list[tuple[float, float]] = []
    if face_score > 0.0:
        weighted_parts.append((face_score, 0.45))
    if voice_score > 0.0:
        weighted_parts.append((voice_score, 0.35))
    weighted_parts.append((feedback_score, 0.20))
    total_weight = sum(weight for _score, weight in weighted_parts)
    score = 0.0 if total_weight <= 0 else round(
        sum(score * weight for score, weight in weighted_parts) / total_weight,
        4,
    )
    if face_score >= 0.8 and voice_score >= 0.76:
        score = min(1.0, round(score + 0.05, 4))

    guidance_hints: list[str] = []
    if portrait_reference_count <= 0:
        recommended_next_step = "enroll_face"
        guidance_hints.extend(("single_face_in_frame", "face_camera", "steady_pose"))
    elif portrait_reference_count < 2:
        recommended_next_step = "capture_more_face_angles"
        guidance_hints.extend(("slight_left_profile", "slight_right_profile", "steady_pose"))
    elif voice_sample_count <= 0:
        recommended_next_step = "enroll_voice"
        guidance_hints.extend(("speak_clear_sentence", "quiet_room", "one_speaker_only"))
    elif voice_sample_count < 2:
        recommended_next_step = "capture_more_voice_samples"
        guidance_hints.extend(("speak_clear_sentence", "quiet_room", "one_speaker_only"))
    elif deny_count > confirm_count:
        recommended_next_step = "reconfirm_identity"
        guidance_hints.append("ask_to_confirm_identity")
    else:
        recommended_next_step = "done"

    if score >= 0.82:
        state = "strong"
    elif score >= 0.62:
        state = "usable"
    elif score > 0.0:
        state = "partial"
    else:
        state = "empty"

    return HouseholdIdentityQuality(
        score=score,
        state=state,
        face_reference_count=portrait_reference_count,
        voice_sample_count=voice_sample_count,
        confirm_count=confirm_count,
        deny_count=deny_count,
        recommended_next_step=recommended_next_step,
        guidance_hints=tuple(guidance_hints),
    )


__all__ = [
    "HouseholdIdentityFeedbackEvent",
    "HouseholdIdentityFeedbackStore",
    "HouseholdIdentityManager",
    "HouseholdIdentityMemberStatus",
    "HouseholdIdentityObservation",
    "HouseholdIdentityQuality",
    "HouseholdIdentityStatus",
]
