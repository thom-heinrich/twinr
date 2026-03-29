# CHANGELOG: 2026-03-28
# BUG-1: Prevent feedback misattribution when no prior observation exists or when the last observation is ambiguous or conflicting.
# BUG-2: Harden feedback-store parsing so malformed JSON, invalid versions, and partial bad records no longer crash status calls; corrupt stores are quarantined and valid events are salvaged.
# BUG-3: Coerce float config values robustly so string-based env/config inputs cannot crash initialization.
# SEC-1: Secure feedback and lock file handling with no-follow opens, private permissions, atomic replace, and parent-directory fsync for better integrity on Linux edge devices.
# SEC-2: Add conservative anti-spoof, liveness, and quality gating plus stricter auto-personalization rules to reduce practical replay, photo, and cloned-voice abuse.
# IMP-1: Replace naive confidence averaging with reliability-aware multimodal fusion that consumes optional quality, spoof, liveness, and duration hints exposed by providers.
# IMP-2: Add decay-weighted session stabilization and negative-feedback gating so identity decisions adapt better to missing or noisy modalities.
# IMP-3: Include the primary user and feedback-only users in member listings, and cache feedback-store reads to reduce Raspberry Pi 4 I/O overhead.

"""Manage shared local household identity across face, voice, and feedback.

This module owns the bounded household-level identity manager used by agent
tools and runtime voice assessment. It combines portrait enrollment and live
matching, multi-user voice enrollment and matching, explicit confirm/deny
feedback, and short in-memory session history into one conservative local
identity surface.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import fcntl
import json
from json import JSONDecodeError
import math
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
from typing import Any

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

_DEFAULT_PRIVATE_FILE_MODE = 0o600

_DEFAULT_SESSION_HALF_LIFE_S = 90.0
_DEFAULT_STABLE_SESSION_SUPPORT_RATIO = 0.67
_DEFAULT_STABLE_SESSION_MIN_OBSERVATIONS = 3
_DEFAULT_RECENT_CONFLICT_WINDOW_EVENTS = 3

_DEFAULT_FACE_MIN_EFFECTIVE_CONFIDENCE = 0.58
_DEFAULT_VOICE_MIN_EFFECTIVE_CONFIDENCE = 0.60
_DEFAULT_FUSED_CONFIRM_CONFIDENCE = 0.64
_DEFAULT_FUSED_AUTO_CONFIDENCE = 0.82

_DEFAULT_FACE_QUALITY_MIN = 0.40
_DEFAULT_VOICE_QUALITY_MIN = 0.35
_DEFAULT_LIVENESS_SCORE_MIN = 0.45
_DEFAULT_SPOOF_SCORE_BLOCK_THRESHOLD = 0.65
_DEFAULT_MIN_VOICE_DURATION_S = 1.20
_DEFAULT_FEEDBACK_RECENT_HORIZON_DAYS = 30


def _utc_iso(value: datetime | None = None) -> str:
    moment = datetime.now(UTC) if value is None else value
    if moment.tzinfo is None or moment.utcoffset() is None:
        moment = moment.replace(tzinfo=UTC)
    else:
        moment = moment.astimezone(UTC)
    return moment.isoformat()


def _parse_utc_iso(value: object | None) -> datetime | None:
    text = _normalize_text(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return parsed


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


def _coerce_positive_float(value: object | None, *, default: float, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(minimum, parsed)


def _coerce_ratio(value: object | None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return max(0.0, min(1.0, parsed))


def _round_ratio(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return round(max(0.0, min(1.0, value)), 4)


def _safe_lower_text(value: object | None) -> str | None:
    text = _normalize_text(value).strip().lower()
    return text or None


def _ratio_from_bool(value: object | None) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = _safe_lower_text(value)
    if text is None:
        return None
    if text in {"true", "yes", "y", "1", "live", "bonafide", "genuine", "real"}:
        return 1.0
    if text in {"false", "no", "n", "0", "spoof", "attack", "fake", "replay"}:
        return 0.0
    return None


def _attr_ratio(obj: object | None, *names: str) -> float | None:
    for name in names:
        if obj is None:
            break
        try:
            value = getattr(obj, name)
        except AttributeError:
            continue
        ratio = _coerce_ratio(value)
        if ratio is not None:
            return ratio
        ratio = _ratio_from_bool(value)
        if ratio is not None:
            return ratio
    return None


def _attr_float(
    obj: object | None,
    *names: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    for name in names:
        if obj is None:
            break
        try:
            value = getattr(obj, name)
        except AttributeError:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(parsed):
            continue
        if minimum is not None and parsed < minimum:
            continue
        if maximum is not None and parsed > maximum:
            continue
        return parsed
    return None


def _attr_text(obj: object | None, *names: str) -> str | None:
    for name in names:
        if obj is None:
            break
        try:
            value = getattr(obj, name)
        except AttributeError:
            continue
        text = _normalize_text(value).strip()
        if text:
            return text
    return None


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non_finite_json_constant:{value}")


def _blend_floor(ratio: float | None, *, floor: float) -> float:
    bounded = _coerce_ratio(ratio)
    if bounded is None:
        return 1.0
    return max(0.0, min(1.0, floor + ((1.0 - floor) * bounded)))


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
        raw_user_id = _normalize_user_id(payload.get("user_id"), default="")
        if not raw_user_id:
            return None
        outcome = _normalize_text(payload.get("outcome")).strip().lower()
        if outcome not in {"confirm", "deny"}:
            return None
        modalities_raw = payload.get("modalities", ())
        modalities: list[str] = []
        if isinstance(modalities_raw, (list, tuple)):
            for item in modalities_raw:
                normalized = _normalize_text(item).strip().lower()
                if normalized in {"face", "voice", "session"} and normalized not in modalities:
                    modalities.append(normalized)
        created_at = _parse_utc_iso(payload.get("created_at"))
        return cls(
            user_id=raw_user_id,
            display_name=_normalize_display_name(payload.get("display_name")),
            outcome=outcome,
            modalities=tuple(modalities),
            created_at=_utc_iso(created_at),
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


@dataclass(frozen=True, slots=True)
class _ModalityEvidence:
    modality: str
    user_id: str | None
    display_name: str | None
    raw_confidence: float | None
    effective_confidence: float | None
    reliability: float | None
    quality_score: float | None
    liveness_score: float | None
    spoof_score: float | None
    duration_s: float | None
    status: str | None
    blocked: bool
    block_reason: str | None


@dataclass(frozen=True, slots=True)
class _FeedbackSignal:
    confirm_count: int
    deny_count: int
    recent_confirm_count: int
    recent_deny_count: int
    recent_negative_streak: int
    last_outcome: str | None


class HouseholdIdentityFeedbackStore:
    """Persist bounded explicit identity feedback events on device."""

    def __init__(self, path: str | Path, *, max_events: int = _DEFAULT_FEEDBACK_MAX_EVENTS) -> None:
        self.path = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
        self.max_events = max(1, int(max_events))
        self._process_lock = threading.RLock()
        self._cache_token: tuple[int, int, int] | None = None
        self._cache_events: tuple[HouseholdIdentityFeedbackEvent, ...] = ()

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
                token = self._token_unlocked()
                if token is not None and token == self._cache_token:
                    return self._cache_events
                events = self._load_unlocked(recover=True)
                token = self._token_unlocked()
                self._cache_events = events
                self._cache_token = token
                return events
        except OSError:
            return self._cache_events

    def append(self, event: HouseholdIdentityFeedbackEvent) -> HouseholdIdentityFeedbackEvent:
        with self._exclusive_lock(create_parent=True):
            events = list(self._load_unlocked(recover=True))
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
            with self._open_private_file(lock_path, read_only=False) as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _open_private_file(self, path: Path, *, read_only: bool) -> Any:
        flags = getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        if read_only:
            flags |= os.O_RDONLY
            fd = os.open(path, flags)
            mode = "r"
        else:
            flags |= os.O_RDWR | os.O_CREAT
            fd = os.open(path, flags, _DEFAULT_PRIVATE_FILE_MODE)
            mode = "a+"
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("household_identity_feedback_store_not_regular")
            with suppress(OSError):
                os.fchmod(fd, _DEFAULT_PRIVATE_FILE_MODE)
            return os.fdopen(fd, mode, encoding="utf-8")
        except Exception:
            with suppress(OSError):
                os.close(fd)
            raise

    def _token_unlocked(self) -> tuple[int, int, int] | None:
        try:
            metadata = os.stat(self.path, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("household_identity_feedback_store_not_regular")
        return (int(metadata.st_ino), int(metadata.st_size), int(metadata.st_mtime_ns))

    def _load_unlocked(self, *, recover: bool) -> tuple[HouseholdIdentityFeedbackEvent, ...]:
        try:
            payload = self._read_payload_unlocked()
            events = self._parse_payload(payload)
            return tuple(events[-self.max_events :])
        except (OSError, JSONDecodeError, UnicodeDecodeError, ValueError, TypeError):
            if recover:
                self._quarantine_corrupt_store_unlocked()
                self._cache_events = ()
                self._cache_token = None
                return ()
            raise

    def _read_payload_unlocked(self) -> object:
        if not self.path.exists():
            return {"store_version": _STORE_VERSION, "events": []}
        fd = -1
        try:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(self.path, flags)
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("household_identity_feedback_store_not_regular")
            with suppress(OSError):
                os.fchmod(fd, _DEFAULT_PRIVATE_FILE_MODE)
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                fd = -1
                return json.load(handle, parse_constant=_reject_nonfinite_json_constant)
        finally:
            if fd >= 0:
                with suppress(OSError):
                    os.close(fd)

    def _parse_payload(self, payload: object) -> list[HouseholdIdentityFeedbackEvent]:
        if not isinstance(payload, dict):
            raise ValueError("household_identity_feedback_store_invalid_payload")
        try:
            version = int(payload.get("store_version", _STORE_VERSION) or _STORE_VERSION)
        except (TypeError, ValueError):
            raise ValueError("household_identity_feedback_store_invalid_version") from None
        if version != _STORE_VERSION:
            raise ValueError("household_identity_feedback_store_unsupported_version")
        events_raw = payload.get("events", ())
        if not isinstance(events_raw, list):
            raise ValueError("household_identity_feedback_store_invalid_events")
        events: list[HouseholdIdentityFeedbackEvent] = []
        for item in events_raw:
            if not isinstance(item, dict):
                continue
            parsed = HouseholdIdentityFeedbackEvent.from_dict(item)
            if parsed is not None:
                events.append(parsed)
        return events

    def _quarantine_corrupt_store_unlocked(self) -> None:
        if not self.path.exists():
            return
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        base_name = f"{self.path.name}.corrupt.{timestamp}"
        for index in range(100):
            suffix = "" if index == 0 else f".{index}"
            candidate = self.path.with_name(base_name + suffix)
            if candidate.exists():
                continue
            try:
                os.replace(self.path, candidate)
            except FileNotFoundError:
                return
            except OSError:
                return
            self._sync_parent_directory_unlocked()
            return

    def _write_unlocked(self, events: list[HouseholdIdentityFeedbackEvent]) -> None:
        payload = {
            "store_version": _STORE_VERSION,
            "events": [event.to_dict() for event in events[-self.max_events :]],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        fd, tmp_path = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                with suppress(OSError):
                    os.fchmod(handle.fileno(), _DEFAULT_PRIVATE_FILE_MODE)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
            with suppress(OSError):
                os.chmod(self.path, _DEFAULT_PRIVATE_FILE_MODE)
            self._sync_parent_directory_unlocked()
            token = self._token_unlocked()
            self._cache_events = tuple(events[-self.max_events :])
            self._cache_token = token
        finally:
            with suppress(OSError):
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    def _sync_parent_directory_unlocked(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
        try:
            fd = os.open(self.path.parent, flags)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            with suppress(OSError):
                os.close(fd)


class HouseholdIdentityManager:
    """Coordinate local household identity enrollment, matching, and feedback."""

    def __init__(
        self,
        *,
        config: TwinrConfig,
        portrait_provider: PortraitMatchProvider,
        voice_monitor: HouseholdVoiceIdentityMonitor,
        feedback_store: HouseholdIdentityFeedbackStore,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.primary_user_id = _normalize_user_id(getattr(config, "portrait_match_primary_user_id", "main_user") or "main_user")
        self.portrait_provider = portrait_provider
        self.voice_monitor = voice_monitor
        self.feedback_store = feedback_store
        self.clock = clock
        self._lock = threading.RLock()
        self._session_history: list[_SessionObservation] = []
        self._session_window_s = _coerce_positive_float(
            getattr(config, "household_identity_session_window_s", _DEFAULT_SESSION_WINDOW_S),
            default=_DEFAULT_SESSION_WINDOW_S,
            minimum=30.0,
        )
        self._session_max_events = _coerce_positive_int(
            getattr(config, "household_identity_session_max_events", _DEFAULT_SESSION_MAX_EVENTS),
            default=_DEFAULT_SESSION_MAX_EVENTS,
        )
        self._session_half_life_s = _coerce_positive_float(
            getattr(config, "household_identity_session_half_life_s", min(_DEFAULT_SESSION_HALF_LIFE_S, self._session_window_s)),
            default=min(_DEFAULT_SESSION_HALF_LIFE_S, self._session_window_s),
            minimum=1.0,
        )
        self._stable_session_support_ratio = _coerce_ratio(
            getattr(config, "household_identity_stable_session_support_ratio", _DEFAULT_STABLE_SESSION_SUPPORT_RATIO)
        ) or _DEFAULT_STABLE_SESSION_SUPPORT_RATIO
        self._stable_session_min_observations = _coerce_positive_int(
            getattr(config, "household_identity_stable_session_min_observations", _DEFAULT_STABLE_SESSION_MIN_OBSERVATIONS),
            default=_DEFAULT_STABLE_SESSION_MIN_OBSERVATIONS,
        )
        self._recent_conflict_window_events = _coerce_positive_int(
            getattr(config, "household_identity_recent_conflict_window_events", _DEFAULT_RECENT_CONFLICT_WINDOW_EVENTS),
            default=_DEFAULT_RECENT_CONFLICT_WINDOW_EVENTS,
        )
        self._face_min_effective_confidence = _coerce_ratio(
            getattr(config, "household_identity_face_min_effective_confidence", _DEFAULT_FACE_MIN_EFFECTIVE_CONFIDENCE)
        ) or _DEFAULT_FACE_MIN_EFFECTIVE_CONFIDENCE
        self._voice_min_effective_confidence = _coerce_ratio(
            getattr(config, "household_identity_voice_min_effective_confidence", _DEFAULT_VOICE_MIN_EFFECTIVE_CONFIDENCE)
        ) or _DEFAULT_VOICE_MIN_EFFECTIVE_CONFIDENCE
        self._fused_confirm_confidence = _coerce_ratio(
            getattr(config, "household_identity_fused_confirm_confidence", _DEFAULT_FUSED_CONFIRM_CONFIDENCE)
        ) or _DEFAULT_FUSED_CONFIRM_CONFIDENCE
        self._fused_auto_confidence = _coerce_ratio(
            getattr(config, "household_identity_fused_auto_confidence", _DEFAULT_FUSED_AUTO_CONFIDENCE)
        ) or _DEFAULT_FUSED_AUTO_CONFIDENCE
        self._face_quality_min = _coerce_ratio(
            getattr(config, "household_identity_face_quality_min", _DEFAULT_FACE_QUALITY_MIN)
        ) or _DEFAULT_FACE_QUALITY_MIN
        self._voice_quality_min = _coerce_ratio(
            getattr(config, "household_identity_voice_quality_min", _DEFAULT_VOICE_QUALITY_MIN)
        ) or _DEFAULT_VOICE_QUALITY_MIN
        self._liveness_score_min = _coerce_ratio(
            getattr(config, "household_identity_liveness_score_min", _DEFAULT_LIVENESS_SCORE_MIN)
        ) or _DEFAULT_LIVENESS_SCORE_MIN
        self._spoof_score_block_threshold = _coerce_ratio(
            getattr(config, "household_identity_spoof_score_block_threshold", _DEFAULT_SPOOF_SCORE_BLOCK_THRESHOLD)
        ) or _DEFAULT_SPOOF_SCORE_BLOCK_THRESHOLD
        self._min_voice_duration_s = _coerce_positive_float(
            getattr(config, "household_identity_min_voice_duration_s", _DEFAULT_MIN_VOICE_DURATION_S),
            default=_DEFAULT_MIN_VOICE_DURATION_S,
            minimum=0.10,
        )
        self._feedback_recent_horizon = timedelta(
            days=_coerce_positive_int(
                getattr(config, "household_identity_feedback_recent_horizon_days", _DEFAULT_FEEDBACK_RECENT_HORIZON_DAYS),
                default=_DEFAULT_FEEDBACK_RECENT_HORIZON_DAYS,
            )
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
        with self._lock:
            portrait_profiles = {
                profile.user_id: profile
                for profile in self.portrait_provider.list_profiles()
            }
            voice_profiles = {
                summary.user_id: summary
                for summary in self.voice_monitor.list_summaries()
            }
            feedback_events = self.feedback_store.list_events()
            feedback_by_user = self._feedback_counts_from_events(feedback_events)
            feedback_names = self._feedback_display_names_from_events(feedback_events)
            user_ids = sorted(
                set(portrait_profiles) | set(voice_profiles) | set(feedback_by_user) | {self.primary_user_id},
                key=str.casefold,
            )
            members: list[HouseholdIdentityMemberStatus] = []
            for user_id in user_ids:
                portrait_profile = portrait_profiles.get(user_id)
                voice_summary = voice_profiles.get(user_id)
                display_name = None
                if portrait_profile is not None and portrait_profile.display_name:
                    display_name = portrait_profile.display_name
                elif voice_summary is not None and voice_summary.display_name:
                    display_name = voice_summary.display_name
                else:
                    display_name = feedback_names.get(user_id)
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
            observation = self._last_observation
            explicit_user_id = _normalize_user_id(user_id, default="") if user_id is not None else ""
            if explicit_user_id:
                resolved_user_id = explicit_user_id
            else:
                if observation is None or not observation.matched_user_id:
                    # BREAKING: feedback without an explicit user_id is now rejected unless the last observation nominated one unambiguous candidate.
                    raise ValueError("No unambiguous household identity candidate is available to confirm or deny.")
                if observation.state in {"modality_conflict", "ambiguous_voice_match", "ambiguous_face_match", "no_identity_signal"}:
                    # BREAKING: conflicting or ambiguous observations must be disambiguated by passing user_id explicitly.
                    raise ValueError("Explicit user_id is required because the last observation is ambiguous or conflicting.")
                resolved_user_id = observation.matched_user_id
            event = HouseholdIdentityFeedbackEvent(
                user_id=resolved_user_id,
                display_name=_normalize_display_name(display_name)
                or (
                    None
                    if observation is None or observation.matched_user_id != resolved_user_id
                    else observation.matched_user_display_name
                ),
                outcome=normalized_outcome,
                modalities=tuple(
                    ()
                    if observation is None or observation.matched_user_id != resolved_user_id
                    else observation.modalities
                ),
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
        face = self._face_evidence(portrait_observation)
        voice = self._voice_evidence(voice_assessment)

        face_candidate = self._trusted_evidence(face, minimum=self._face_min_effective_confidence)
        voice_candidate = self._trusted_evidence(voice, minimum=self._voice_min_effective_confidence)

        state = "no_identity_signal"
        matched_user_id: str | None = None
        matched_display_name: str | None = None
        base_confidence: float | None = None
        modalities: tuple[str, ...] = ()
        block_reason: str | None = None
        conflict = False

        if face_candidate and voice_candidate and face_candidate.user_id == voice_candidate.user_id:
            matched_user_id = face_candidate.user_id
            matched_display_name = face_candidate.display_name or voice_candidate.display_name
            base_confidence = self._fuse_effective_confidence(face_candidate, voice_candidate)
            modalities = ("face", "voice")
            state = "multimodal_match"
            if face_candidate.blocked:
                block_reason = face_candidate.block_reason
            elif voice_candidate.blocked:
                block_reason = voice_candidate.block_reason
        elif face_candidate and voice_candidate and face_candidate.user_id != voice_candidate.user_id:
            # BREAKING: conflicting modality observations no longer nominate an arbitrary user_id.
            matched_user_id = None
            matched_display_name = None
            base_confidence = self._fuse_effective_confidence(face_candidate, voice_candidate)
            modalities = ("face", "voice")
            state = "modality_conflict"
            block_reason = "face_voice_identity_conflict"
            conflict = True
        elif face_candidate:
            matched_user_id = face_candidate.user_id
            matched_display_name = face_candidate.display_name
            base_confidence = face_candidate.effective_confidence
            modalities = ("face",)
            state = "portrait_only"
            block_reason = face_candidate.block_reason or voice.block_reason
        elif voice_candidate:
            matched_user_id = voice_candidate.user_id
            matched_display_name = voice_candidate.display_name
            base_confidence = voice_candidate.effective_confidence
            modalities = ("voice",)
            state = "voice_only"
            block_reason = voice_candidate.block_reason or face.block_reason
        elif face.blocked and face.block_reason:
            state = "ambiguous_face_match"
            modalities = ("face",)
            block_reason = face.block_reason
        elif voice.blocked and voice.block_reason:
            state = "ambiguous_voice_match"
            modalities = ("voice",)
            block_reason = voice.block_reason
        elif voice.user_id or (voice.status is not None and "ambiguous" in voice.status):
            state = "ambiguous_voice_match"
            modalities = ("voice",)
            base_confidence = voice.effective_confidence or voice.raw_confidence
            block_reason = voice.block_reason or "ambiguous_voice_match"
        elif face.user_id or (
            portrait_observation is not None
            and _safe_lower_text(getattr(portrait_observation, "state", None)) == "ambiguous_identity"
        ):
            state = "ambiguous_face_match"
            modalities = ("face",)
            base_confidence = face.effective_confidence or face.raw_confidence
            block_reason = face.block_reason or "ambiguous_face_match"
        else:
            block_reason = "identity_signal_unavailable"

        now = float(self.clock())
        self._prune_session_history(now)
        if matched_user_id or conflict:
            self._session_history.append(
                _SessionObservation(
                    observed_at=now,
                    user_id=matched_user_id,
                    confidence=base_confidence,
                    modalities=modalities,
                    conflict=conflict,
                )
            )
            if len(self._session_history) > self._session_max_events:
                self._session_history = self._session_history[-self._session_max_events :]

        temporal_state = None
        support_ratio = None
        observation_count = 0
        if matched_user_id:
            support_ratio, observation_count, temporal_state = self._session_support(now, matched_user_id)

        feedback = self._feedback_signal(matched_user_id)
        confidence = self._adjust_confidence(
            base_confidence=base_confidence,
            temporal_state=temporal_state,
            support_ratio=support_ratio,
            feedback=feedback,
        )

        policy_recommendation = "confirm_first"
        if conflict:
            policy_recommendation = "confirm_first"
            block_reason = "face_voice_identity_conflict"
        elif block_reason in {"face_spoof_suspected", "voice_spoof_suspected", "face_liveness_low", "voice_liveness_low"}:
            policy_recommendation = "block_sensitive_actions"
        elif matched_user_id and confidence is not None:
            if feedback.recent_negative_streak >= 2 or feedback.recent_deny_count > feedback.recent_confirm_count:
                block_reason = block_reason or "negative_identity_feedback"
                policy_recommendation = "confirm_first"
            elif self._can_auto_personalize(
                matched_user_id=matched_user_id,
                modalities=modalities,
                confidence=confidence,
                temporal_state=temporal_state,
                face=face,
                voice=voice,
            ):
                # BREAKING: auto-personalization now requires stronger temporal support and anti-spoof/liveness compatibility than the 2025 logic.
                policy_recommendation = "calm_personalization_only"
                block_reason = None
            else:
                policy_recommendation = "confirm_first"
                if block_reason is None:
                    if confidence < self._fused_confirm_confidence:
                        block_reason = "low_identity_confidence"
                    elif temporal_state != "stable_session" and modalities in {("voice",), ("face", "voice")}:
                        block_reason = "insufficient_temporal_support"
                    else:
                        block_reason = "requires_confirmation"

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

    def _face_evidence(self, portrait_observation: PortraitMatchObservation | None) -> _ModalityEvidence:
        if portrait_observation is None:
            return _ModalityEvidence(
                modality="face",
                user_id=None,
                display_name=None,
                raw_confidence=None,
                effective_confidence=None,
                reliability=None,
                quality_score=None,
                liveness_score=None,
                spoof_score=None,
                duration_s=None,
                status=None,
                blocked=False,
                block_reason=None,
            )
        user_id = _normalize_user_id(_attr_text(portrait_observation, "matched_user_id"), default="") or None
        raw_confidence = _round_ratio(
            _attr_ratio(portrait_observation, "fused_confidence", "confidence", "match_confidence")
        )
        quality_score = _round_ratio(
            _attr_ratio(
                portrait_observation,
                "quality_score",
                "image_quality",
                "face_quality",
                "utility_score",
                "recognizability_score",
                "capture_quality",
            )
        )
        if quality_score is None and raw_confidence is not None and user_id:
            quality_score = raw_confidence
        liveness_score = _round_ratio(
            _attr_ratio(
                portrait_observation,
                "liveness_score",
                "live_score",
                "is_live",
                "bonafide_score",
            )
        )
        spoof_score = _round_ratio(
            _attr_ratio(
                portrait_observation,
                "spoof_score",
                "anti_spoof_score",
                "presentation_attack_score",
                "pad_score",
                "deepfake_score",
            )
        )
        status = _safe_lower_text(_attr_text(portrait_observation, "state", "status"))
        blocked = False
        block_reason = None
        if spoof_score is not None and spoof_score >= self._spoof_score_block_threshold:
            blocked = True
            block_reason = "face_spoof_suspected"
        elif liveness_score is not None and liveness_score < self._liveness_score_min:
            blocked = True
            block_reason = "face_liveness_low"

        reliability = 1.0
        if quality_score is not None:
            reliability *= _blend_floor(quality_score, floor=0.35)
        if liveness_score is not None:
            reliability *= _blend_floor(liveness_score, floor=0.15)
        if spoof_score is not None:
            reliability *= max(0.0, 1.0 - spoof_score)
        if status is not None and any(token in status for token in ("ambiguous", "multiple", "occluded", "blur", "far", "partial")):
            reliability *= 0.65
        if quality_score is not None and quality_score < self._face_quality_min:
            reliability *= 0.70
        effective_confidence = _round_ratio(None if raw_confidence is None else raw_confidence * reliability)
        return _ModalityEvidence(
            modality="face",
            user_id=user_id,
            display_name=_normalize_display_name(_attr_text(portrait_observation, "matched_user_display_name")),
            raw_confidence=raw_confidence,
            effective_confidence=effective_confidence,
            reliability=_round_ratio(reliability),
            quality_score=quality_score,
            liveness_score=liveness_score,
            spoof_score=spoof_score,
            duration_s=None,
            status=status,
            blocked=blocked,
            block_reason=block_reason,
        )

    def _voice_evidence(self, voice_assessment: HouseholdVoiceAssessment | None) -> _ModalityEvidence:
        if voice_assessment is None:
            return _ModalityEvidence(
                modality="voice",
                user_id=None,
                display_name=None,
                raw_confidence=None,
                effective_confidence=None,
                reliability=None,
                quality_score=None,
                liveness_score=None,
                spoof_score=None,
                duration_s=None,
                status=None,
                blocked=False,
                block_reason=None,
            )
        user_id = _normalize_user_id(_attr_text(voice_assessment, "matched_user_id"), default="") or None
        raw_confidence = _round_ratio(
            _attr_ratio(voice_assessment, "confidence", "match_confidence", "speaker_confidence")
        )
        duration_s = _attr_float(
            voice_assessment,
            "duration_s",
            "speech_duration_s",
            "voiced_duration_s",
            "utterance_duration_s",
            minimum=0.0,
        )
        quality_score = _round_ratio(
            _attr_ratio(
                voice_assessment,
                "quality_score",
                "signal_quality",
                "audio_quality",
                "utterance_quality",
                "snr_quality",
            )
        )
        if quality_score is None and duration_s is not None and self._min_voice_duration_s > 0.0:
            quality_score = _round_ratio(min(1.0, duration_s / self._min_voice_duration_s))
        if quality_score is None and raw_confidence is not None and user_id:
            quality_score = raw_confidence
        liveness_score = _round_ratio(
            _attr_ratio(
                voice_assessment,
                "liveness_score",
                "is_live",
                "bonafide_score",
                "human_score",
            )
        )
        spoof_score = _round_ratio(
            _attr_ratio(
                voice_assessment,
                "spoof_score",
                "anti_spoof_score",
                "deepfake_score",
                "replay_score",
                "clone_score",
            )
        )
        status = _safe_lower_text(_attr_text(voice_assessment, "status"))
        blocked = False
        block_reason = None
        if spoof_score is not None and spoof_score >= self._spoof_score_block_threshold:
            blocked = True
            block_reason = "voice_spoof_suspected"
        elif liveness_score is not None and liveness_score < self._liveness_score_min:
            blocked = True
            block_reason = "voice_liveness_low"

        reliability = 1.0
        if quality_score is not None:
            reliability *= _blend_floor(quality_score, floor=0.30)
        if liveness_score is not None:
            reliability *= _blend_floor(liveness_score, floor=0.15)
        if spoof_score is not None:
            reliability *= max(0.0, 1.0 - spoof_score)
        if duration_s is not None and self._min_voice_duration_s > 0.0:
            reliability *= max(0.35, min(1.0, duration_s / self._min_voice_duration_s))
        if status is not None:
            if any(token in status for token in ("ambiguous", "uncertain", "multiple", "overlap")):
                reliability *= 0.60
            if any(token in status for token in ("short", "brief")):
                reliability *= 0.72
            if any(token in status for token in ("noise", "noisy", "low_snr", "quiet")):
                reliability *= 0.78
            if any(token in status for token in ("spoof", "fake", "replay", "clone")) and block_reason is None:
                blocked = True
                block_reason = "voice_spoof_suspected"
        if quality_score is not None and quality_score < self._voice_quality_min:
            reliability *= 0.72
        effective_confidence = _round_ratio(None if raw_confidence is None else raw_confidence * reliability)
        return _ModalityEvidence(
            modality="voice",
            user_id=user_id,
            display_name=_normalize_display_name(_attr_text(voice_assessment, "matched_user_display_name")),
            raw_confidence=raw_confidence,
            effective_confidence=effective_confidence,
            reliability=_round_ratio(reliability),
            quality_score=quality_score,
            liveness_score=liveness_score,
            spoof_score=spoof_score,
            duration_s=None if duration_s is None else round(duration_s, 4),
            status=status,
            blocked=blocked,
            block_reason=block_reason,
        )

    def _trusted_evidence(self, evidence: _ModalityEvidence, *, minimum: float) -> _ModalityEvidence | None:
        if evidence.blocked:
            return None
        if not evidence.user_id:
            return None
        if evidence.effective_confidence is None:
            return None
        if evidence.effective_confidence < minimum:
            return None
        return evidence

    def _fuse_effective_confidence(self, left: _ModalityEvidence, right: _ModalityEvidence) -> float | None:
        values: list[tuple[float, float]] = []
        if left.effective_confidence is not None:
            values.append((left.effective_confidence, max(0.1, left.reliability or 0.5)))
        if right.effective_confidence is not None:
            values.append((right.effective_confidence, max(0.1, right.reliability or 0.5)))
        if not values:
            return None
        total_weight = sum(weight for _score, weight in values)
        if total_weight <= 0.0:
            return None
        return round(sum(score * weight for score, weight in values) / total_weight, 4)

    def _session_support(self, now: float, user_id: str) -> tuple[float | None, int, str]:
        relevant = [item for item in self._session_history if not item.conflict and item.user_id]
        support = [item for item in relevant if item.user_id == user_id]
        observation_count = len(support)
        if not relevant:
            return None, observation_count, "insufficient_history"
        total_weight = 0.0
        support_weight = 0.0
        for item in relevant:
            decay = math.exp(-max(0.0, now - item.observed_at) / max(self._session_half_life_s, 1.0))
            total_weight += decay
            if item.user_id == user_id:
                support_weight += decay
        support_ratio = _round_ratio(None if total_weight <= 0.0 else support_weight / total_weight)
        recent_tail = self._session_history[-self._recent_conflict_window_events :]
        if any(item.conflict for item in recent_tail):
            temporal_state = "recent_conflict"
        elif observation_count >= self._stable_session_min_observations and (support_ratio or 0.0) >= self._stable_session_support_ratio:
            temporal_state = "stable_session"
        else:
            temporal_state = "insufficient_history"
        return support_ratio, observation_count, temporal_state

    def _adjust_confidence(
        self,
        *,
        base_confidence: float | None,
        temporal_state: str | None,
        support_ratio: float | None,
        feedback: _FeedbackSignal,
    ) -> float | None:
        if base_confidence is None:
            return None
        adjusted = base_confidence
        if temporal_state == "stable_session" and support_ratio is not None:
            adjusted = min(1.0, adjusted + (0.05 * support_ratio))
        elif temporal_state == "recent_conflict":
            adjusted = max(0.0, adjusted - 0.12)
        if feedback.confirm_count > feedback.deny_count:
            adjusted = min(1.0, adjusted + min(0.04, 0.01 * (feedback.confirm_count - feedback.deny_count)))
        elif feedback.deny_count > feedback.confirm_count:
            adjusted = max(0.0, adjusted - min(0.10, 0.02 * (feedback.deny_count - feedback.confirm_count)))
        return round(adjusted, 4)

    def _can_auto_personalize(
        self,
        *,
        matched_user_id: str,
        modalities: tuple[str, ...],
        confidence: float,
        temporal_state: str | None,
        face: _ModalityEvidence,
        voice: _ModalityEvidence,
    ) -> bool:
        if matched_user_id != self.primary_user_id:
            return False
        if confidence < self._fused_auto_confidence:
            return False
        if modalities == ("face", "voice"):
            if temporal_state == "stable_session":
                return True
            return self._strong_liveness(face) and (voice.liveness_score is None or self._strong_liveness(voice))
        if modalities == ("voice",):
            return temporal_state == "stable_session" and confidence >= min(1.0, self._fused_auto_confidence + 0.03)
        if modalities == ("face",):
            return temporal_state == "stable_session" and self._strong_liveness(face)
        return False

    def _strong_liveness(self, evidence: _ModalityEvidence) -> bool:
        if evidence.blocked:
            return False
        if evidence.liveness_score is not None and evidence.liveness_score >= max(0.75, self._liveness_score_min):
            return True
        if evidence.spoof_score is not None and evidence.spoof_score <= 0.10 and (evidence.quality_score or 0.0) >= 0.75:
            return True
        return False

    def _prune_session_history(self, now: float) -> None:
        minimum_time = now - self._session_window_s
        self._session_history = [item for item in self._session_history if item.observed_at >= minimum_time]

    def _feedback_signal(self, user_id: str | None) -> _FeedbackSignal:
        if not user_id:
            return _FeedbackSignal(0, 0, 0, 0, 0, None)
        events = [event for event in self.feedback_store.list_events() if event.user_id == user_id]
        if not events:
            return _FeedbackSignal(0, 0, 0, 0, 0, None)
        now = datetime.now(UTC)
        confirm_count = 0
        deny_count = 0
        recent_confirm_count = 0
        recent_deny_count = 0
        for event in events:
            if event.outcome == "confirm":
                confirm_count += 1
            else:
                deny_count += 1
            created_at = _parse_utc_iso(event.created_at)
            if created_at is not None and (now - created_at) <= self._feedback_recent_horizon:
                if event.outcome == "confirm":
                    recent_confirm_count += 1
                else:
                    recent_deny_count += 1
        recent_negative_streak = 0
        for event in reversed(events):
            if event.outcome != "deny":
                break
            recent_negative_streak += 1
        return _FeedbackSignal(
            confirm_count=confirm_count,
            deny_count=deny_count,
            recent_confirm_count=recent_confirm_count,
            recent_deny_count=recent_deny_count,
            recent_negative_streak=recent_negative_streak,
            last_outcome=events[-1].outcome,
        )

    def _feedback_counts(self) -> dict[str, tuple[int, int]]:
        return self._feedback_counts_from_events(self.feedback_store.list_events())

    def _feedback_counts_from_events(
        self,
        events: tuple[HouseholdIdentityFeedbackEvent, ...],
    ) -> dict[str, tuple[int, int]]:
        counts: dict[str, tuple[int, int]] = {}
        for event in events:
            confirm_count, deny_count = counts.get(event.user_id, (0, 0))
            if event.outcome == "confirm":
                confirm_count += 1
            else:
                deny_count += 1
            counts[event.user_id] = (confirm_count, deny_count)
        return counts

    def _feedback_display_names_from_events(
        self,
        events: tuple[HouseholdIdentityFeedbackEvent, ...],
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for event in events:
            if event.display_name:
                result[event.user_id] = event.display_name
        return result

    def _member_status_or_none(self, user_id: str) -> HouseholdIdentityMemberStatus | None:
        normalized_user_id = _normalize_user_id(user_id)
        for member in self.list_members():
            if member.user_id == normalized_user_id:
                return member
        return None


def _mean_confidence(*values: float | None) -> float | None:
    usable = [value for value in values if value is not None and math.isfinite(value)]
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