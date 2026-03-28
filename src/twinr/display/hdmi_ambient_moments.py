# CHANGELOG: 2026-03-28
# BUG-1: Naive datetime inputs are now treated as local civil time by default
#        instead of being silently reinterpreted as UTC.
# BUG-2: When the director owns the clock (`now=None`), backwards wall-clock
#        steps no longer replay already-seen moments within the same process.
# SEC-1: No practical, directly exploitable security issue was found in this
#        isolated pure-scheduling module.
# IMP-1: Scheduling is now deterministically salted per device/user via keyed
#        BLAKE2 instead of global fleet-wide lockstep hashing.
# IMP-2: Moment selection is now daypart-aware, reduced-motion aware,
#        quiet-hours aware, and repeat-suppressed with deterministic weighted
#        ranking.
# IMP-3: Invalid numeric config values are hardened at runtime so a bad config
#        degrades safely instead of crashing the idle surface.

"""Schedule calm ambient moments for Twinr's idle HDMI face.

The senior-facing waiting surface should stay calm most of the time, but it can
occasionally do something small and cute so the screen does not feel lifeless.
This module keeps that policy separate from the renderer: it deterministically
selects infrequent idle-only moments and describes their temporary face cue plus
tiny ornament.

2026 upgrades in this version:
- Deterministic per-device/user seeding instead of fleet-wide lockstep behavior.
- Context-aware ranking (daypart, quiet hours, reduced motion, repeat damping).
- Safer handling of naive datetimes and backwards wall-clock corrections.
- Same public output type: HdmiAmbientMoment | None.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import math
import os
import socket
from typing import Mapping

from twinr.display.face_cues import DisplayFaceCue


_DEFAULT_BUCKET_SECONDS = 5.0 * 60.0
_DEFAULT_ACTIVE_WINDOW_S = 7.5
_DEFAULT_TRIGGER_DIVISOR = 6
_DEFAULT_REPEAT_COOLDOWN_BUCKETS = 2

_MOMENT_KEYS = ("sparkle", "heart", "curious", "sleepy", "wave", "crown")
_PERSONALIZATION = b"TwAmbV2"

_MOTION_LEVEL = {
    "sparkle": 1,
    "heart": 1,
    "curious": 1,
    "sleepy": 1,
    "wave": 2,
    "crown": 0,
}

_BASE_WEIGHTS = {
    "sparkle": 1.00,
    "heart": 0.90,
    "curious": 0.95,
    "sleepy": 0.70,
    "wave": 0.80,
    "crown": 0.55,
}

_DAYPART_WEIGHTS = {
    "night": {
        "sparkle": 0.75,
        "heart": 0.35,
        "curious": 0.45,
        "sleepy": 2.10,
        "wave": 0.10,
        "crown": 0.20,
    },
    "morning": {
        "sparkle": 1.15,
        "heart": 0.85,
        "curious": 0.95,
        "sleepy": 0.20,
        "wave": 1.45,
        "crown": 0.70,
    },
    "day": {
        "sparkle": 1.00,
        "heart": 0.85,
        "curious": 1.30,
        "sleepy": 0.30,
        "wave": 0.95,
        "crown": 0.75,
    },
    "evening": {
        "sparkle": 0.95,
        "heart": 1.20,
        "curious": 0.80,
        "sleepy": 1.20,
        "wave": 0.50,
        "crown": 1.00,
    },
}


@dataclass(frozen=True, slots=True)
class HdmiAmbientMoment:
    """Describe one active idle-only HDMI ambient moment."""

    key: str
    ornament: str
    progress: float
    face_cue: DisplayFaceCue


@dataclass(frozen=True, slots=True)
class _WeightedCandidate:
    key: str
    weight: float


@dataclass(slots=True)
class HdmiAmbientMomentDirector:
    """Resolve rare ambient moments for the waiting HDMI scene.

    Deterministic selection remains the core policy, but selection is now
    context-aware: different moments are preferred at different times of day,
    a reduced-motion mode can suppress higher-motion cues, quiet hours can
    disable ambient moments entirely, and recently-seen moments are damped so
    the screen feels less repetitive.

    Public API compatibility:
    - resolve(...) still returns HdmiAmbientMoment | None.
    - bucket_seed(bucket_index) still returns a stable integer seed.

    New knobs:
    - seed_key: stable per-device/per-user salt. If omitted, a device-local
      default is derived from env vars or hostname.
    - repeat_cooldown_buckets: number of previous time buckets used for repeat
      damping (deterministic, no global RNG state).
    - quiet_hours_local: optional (start_hour, end_hour) local-time blackout.
    - reduced_motion: reduce frequency and remove higher-motion moments.
    - moment_weights: optional per-key weight multipliers.
    - naive_datetime_is_local: interpret naive datetimes as local civil time.
    """

    bucket_seconds: float = _DEFAULT_BUCKET_SECONDS
    active_window_s: float = _DEFAULT_ACTIVE_WINDOW_S
    trigger_divisor: int = _DEFAULT_TRIGGER_DIVISOR
    seed_key: str | bytes | None = None
    repeat_cooldown_buckets: int = _DEFAULT_REPEAT_COOLDOWN_BUCKETS
    quiet_hours_local: tuple[int, int] | None = None
    reduced_motion: bool = False
    moment_weights: Mapping[str, float] | None = None
    naive_datetime_is_local: bool = True

    _seed_material: bytes = field(init=False, repr=False)
    _last_runtime_epoch_s: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        raw_seed_key = self.seed_key
        if raw_seed_key is None:
            # BREAKING: a stable device-local seed is now the default so deployed
            # units do not all emote in lockstep. Set seed_key="global" to
            # recover the legacy fleet-synchronized schedule.
            raw_seed_key = (
                os.getenv("TWINR_AMBIENT_SEED")
                or os.getenv("TWINR_DEVICE_ID")
                or socket.gethostname()
                or "global"
            )
        self._seed_material = _normalize_seed_material(raw_seed_key)

    def resolve(
        self,
        *,
        status: str,
        now: datetime | None,
        face_cue_active: bool,
        presentation_active: bool,
    ) -> HdmiAmbientMoment | None:
        """Return the active ambient moment, if the idle surface should show one.

        `now` may be aware or naive. By default, naive datetimes are interpreted
        as system-local civil time before conversion to UTC.
        """

        if status != "waiting" or face_cue_active or presentation_active:
            return None

        safe_now = _normalize_now(now, naive_is_local=self.naive_datetime_is_local)
        bucket_seconds = _coerce_finite_float(
            self.bucket_seconds,
            default=_DEFAULT_BUCKET_SECONDS,
            minimum=60.0,
        )
        active_window_s = _coerce_finite_float(
            self.active_window_s,
            default=_DEFAULT_ACTIVE_WINDOW_S,
            minimum=1.0,
            maximum=max(1.0, bucket_seconds - 1.0),
        )
        trigger_divisor = _coerce_finite_int(
            self.trigger_divisor,
            default=_DEFAULT_TRIGGER_DIVISOR,
            minimum=1,
        )
        repeat_cooldown_buckets = _coerce_finite_int(
            self.repeat_cooldown_buckets,
            default=_DEFAULT_REPEAT_COOLDOWN_BUCKETS,
            minimum=0,
        )

        if self.reduced_motion:
            active_window_s = min(active_window_s, 4.0)

        local_now = safe_now.astimezone()
        if _in_quiet_hours(local_now.hour, self.quiet_hours_local):
            return None
        daypart = _daypart_for_hour(local_now.hour)

        epoch_s = safe_now.timestamp()
        if now is None and self._last_runtime_epoch_s is not None and epoch_s < self._last_runtime_epoch_s:
            epoch_s = self._last_runtime_epoch_s
        if now is None:
            self._last_runtime_epoch_s = epoch_s

        bucket_index = int(math.floor(epoch_s / bucket_seconds))
        bucket_offset_s = epoch_s - (bucket_index * bucket_seconds)
        if bucket_offset_s >= active_window_s:
            return None

        effective_divisor = _effective_trigger_divisor(
            trigger_divisor=trigger_divisor,
            daypart=daypart,
            reduced_motion=self.reduced_motion,
        )
        if self._hash64("activate", bucket_index, effective_divisor) % effective_divisor != 0:
            return None

        key = self._select_key_for_bucket(
            bucket_index=bucket_index,
            bucket_seconds=bucket_seconds,
            trigger_divisor=trigger_divisor,
            repeat_cooldown_buckets=repeat_cooldown_buckets,
        )
        if key is None:
            return None

        progress = min(1.0, max(0.0, bucket_offset_s / active_window_s))
        return _build_moment(key=key, progress=progress)

    def bucket_seed(self, bucket_index: int) -> int:
        """Return a stable integer seed for one ambient time bucket."""

        return self._hash64("bucket", int(bucket_index))

    def _select_key_for_bucket(
        self,
        *,
        bucket_index: int,
        bucket_seconds: float,
        trigger_divisor: int,
        repeat_cooldown_buckets: int,
    ) -> str | None:
        key = self._raw_key_for_bucket(
            bucket_index=bucket_index,
            bucket_seconds=bucket_seconds,
            trigger_divisor=trigger_divisor,
        )
        if key is None or repeat_cooldown_buckets <= 0:
            return key

        recent_keys = {
            recent_key
            for offset in range(1, repeat_cooldown_buckets + 1)
            for recent_key in (
                self._raw_key_for_bucket(
                    bucket_index=bucket_index - offset,
                    bucket_seconds=bucket_seconds,
                    trigger_divisor=trigger_divisor,
                ),
            )
            if recent_key is not None
        }
        if key not in recent_keys:
            return key

        local_dt = _bucket_local_datetime(bucket_index, bucket_seconds)
        daypart = _daypart_for_hour(local_dt.hour)

        candidates = _candidate_keys(
            daypart=daypart,
            reduced_motion=self.reduced_motion,
            moment_weights=self.moment_weights,
            recent_keys=recent_keys,
        )
        if not candidates:
            return None

        scored = [
            (
                _weighted_gumbel_score(
                    weight=candidate.weight,
                    uniform01=self._uniform01("choose", bucket_index, candidate.key),
                ),
                candidate.key,
            )
            for candidate in candidates
        ]
        scored.sort(reverse=True)
        return scored[0][1]

    def _raw_key_for_bucket(
        self,
        *,
        bucket_index: int,
        bucket_seconds: float,
        trigger_divisor: int,
    ) -> str | None:
        local_dt = _bucket_local_datetime(bucket_index, bucket_seconds)
        if _in_quiet_hours(local_dt.hour, self.quiet_hours_local):
            return None

        daypart = _daypart_for_hour(local_dt.hour)
        effective_divisor = _effective_trigger_divisor(
            trigger_divisor=trigger_divisor,
            daypart=daypart,
            reduced_motion=self.reduced_motion,
        )
        if self._hash64("activate", bucket_index, effective_divisor) % effective_divisor != 0:
            return None

        candidates = _candidate_keys(
            daypart=daypart,
            reduced_motion=self.reduced_motion,
            moment_weights=self.moment_weights,
            recent_keys=frozenset(),
        )
        if not candidates:
            return None

        scored = [
            (
                _weighted_gumbel_score(
                    weight=candidate.weight,
                    uniform01=self._uniform01("choose", bucket_index, candidate.key),
                ),
                candidate.key,
            )
            for candidate in candidates
        ]
        scored.sort(reverse=True)
        return scored[0][1]

    def _uniform01(self, namespace: str, *parts: object) -> float:
        raw = self._hash64(namespace, *parts)
        return ((raw & ((1 << 53) - 1)) + 0.5) / float(1 << 53)

    def _hash64(self, namespace: str, *parts: object) -> int:
        return _hash64(self._seed_material, namespace, *parts)


@lru_cache(maxsize=256)
def _normalize_seed_material(value: str | bytes) -> bytes:
    if isinstance(value, bytes):
        raw = value
    else:
        raw = str(value).encode("utf-8", "surrogatepass")
    if not raw:
        raw = b"global"
    return hashlib.blake2b(raw, digest_size=32, person=_PERSONALIZATION).digest()


@lru_cache(maxsize=16384)
def _hash64(seed_material: bytes, namespace: str, *parts: object) -> int:
    hasher = hashlib.blake2b(
        digest_size=16,
        key=seed_material,
        person=_PERSONALIZATION,
    )
    hasher.update(namespace.encode("utf-8", "strict"))
    for part in parts:
        hasher.update(b"\x1f")
        hasher.update(str(part).encode("utf-8", "surrogatepass"))
    return int.from_bytes(hasher.digest()[:8], "big", signed=False)


def _normalize_now(value: datetime | None, *, naive_is_local: bool) -> datetime:
    """Return one aware UTC datetime for ambient scheduling."""

    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        if naive_is_local:
            # BREAKING: naive datetimes are treated as system-local civil time now.
            # The legacy behavior silently treated them as UTC, which shifted the
            # schedule by the local UTC offset for callers that passed
            # datetime.now().
            return value.astimezone(timezone.utc)
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _build_moment(*, key: str, progress: float) -> HdmiAmbientMoment:
    """Build one active ambient moment from its symbolic key."""

    clipped_progress = min(1.0, max(0.0, float(progress)))
    if key == "sleepy":
        return HdmiAmbientMoment(
            key=key,
            ornament="crescent",
            progress=clipped_progress,
            face_cue=DisplayFaceCue(
                source="ambient_moment",
                gaze_x=-1,
                gaze_y=-1 if clipped_progress < 0.5 else 0,
                mouth="neutral",
                brows="soft",
                head_dx=-1 if clipped_progress < 0.35 else 0,
                head_dy=1,
            ),
        )
    if key == "wave":
        return HdmiAmbientMoment(
            key=key,
            ornament="wave_marks",
            progress=clipped_progress,
            face_cue=DisplayFaceCue(
                source="ambient_moment",
                gaze_x=1,
                gaze_y=0,
                mouth="smile",
                brows="outward_tilt",
                head_dx=1 if clipped_progress < 0.45 else 0,
                head_dy=0,
            ),
        )
    if key == "crown":
        return HdmiAmbientMoment(
            key=key,
            ornament="crown",
            progress=clipped_progress,
            face_cue=DisplayFaceCue(
                source="ambient_moment",
                gaze_x=0,
                gaze_y=-1,
                mouth="smile",
                brows="raised",
                head_dx=0,
                head_dy=-1,
            ),
        )
    if key == "heart":
        return HdmiAmbientMoment(
            key=key,
            ornament="heart",
            progress=clipped_progress,
            face_cue=DisplayFaceCue(
                source="ambient_moment",
                gaze_x=1,
                gaze_y=-1 if clipped_progress < 0.35 else 0,
                mouth="smile",
                brows="soft",
                head_dx=1 if clipped_progress < 0.45 else 0,
                head_dy=-1 if clipped_progress < 0.35 else 0,
            ),
        )
    if key == "curious":
        return HdmiAmbientMoment(
            key=key,
            ornament="dot_cluster",
            progress=clipped_progress,
            face_cue=DisplayFaceCue(
                source="ambient_moment",
                gaze_x=-1,
                gaze_y=-1,
                mouth="thinking",
                brows="roof",
                head_dx=-1 if clipped_progress < 0.5 else 0,
                head_dy=0,
            ),
        )
    return HdmiAmbientMoment(
        key="sparkle",
        ornament="sparkle",
        progress=clipped_progress,
        face_cue=DisplayFaceCue(
            source="ambient_moment",
            gaze_x=1,
            gaze_y=-1,
            mouth="smile",
            brows="raised",
            head_dx=1 if clipped_progress < 0.4 else 0,
            head_dy=-1 if clipped_progress < 0.25 else 0,
        ),
    )


def _bucket_local_datetime(bucket_index: int, bucket_seconds: float) -> datetime:
    bucket_midpoint_utc = datetime.fromtimestamp(
        (float(bucket_index) * bucket_seconds) + 0.5,
        tz=timezone.utc,
    )
    return bucket_midpoint_utc.astimezone()


def _candidate_keys(
    *,
    daypart: str,
    reduced_motion: bool,
    moment_weights: Mapping[str, float] | None,
    recent_keys: frozenset[str] | set[str],
) -> list[_WeightedCandidate]:
    candidates: list[_WeightedCandidate] = []
    daypart_weights = _DAYPART_WEIGHTS[daypart]
    for key in _MOMENT_KEYS:
        weight = _BASE_WEIGHTS[key] * daypart_weights[key]

        if reduced_motion:
            if _MOTION_LEVEL[key] > 1:
                weight = 0.0
            else:
                weight *= 0.55

        if moment_weights is not None and key in moment_weights:
            try:
                custom_weight = float(moment_weights[key])
            except (TypeError, ValueError):
                custom_weight = 1.0
            if math.isfinite(custom_weight) and custom_weight >= 0.0:
                weight *= custom_weight

        if key in recent_keys:
            weight *= 0.10

        if weight > 0.0:
            candidates.append(_WeightedCandidate(key=key, weight=weight))
    return candidates


def _weighted_gumbel_score(*, weight: float, uniform01: float) -> float:
    safe_u = min(1.0 - 1e-12, max(1e-12, uniform01))
    return math.log(weight) - math.log(-math.log(safe_u))


def _daypart_for_hour(hour: int) -> str:
    if 22 <= hour or hour < 7:
        return "night"
    if hour < 11:
        return "morning"
    if hour < 17:
        return "day"
    return "evening"


def _effective_trigger_divisor(
    *,
    trigger_divisor: int,
    daypart: str,
    reduced_motion: bool,
) -> int:
    multiplier = 1.0
    if daypart == "night":
        multiplier *= 2.0
    elif daypart == "evening":
        multiplier *= 1.25
    if reduced_motion:
        multiplier *= 2.0
    return max(1, int(math.ceil(trigger_divisor * multiplier)))


def _in_quiet_hours(hour: int, quiet_hours_local: tuple[int, int] | None) -> bool:
    if quiet_hours_local is None:
        return False

    start, end = quiet_hours_local
    start = int(start) % 24
    end = int(end) % 24

    if start == end:
        return True
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def _coerce_finite_float(
    value: object,
    *,
    default: float,
    minimum: float,
    maximum: float | None = None,
) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        candidate = default

    if not math.isfinite(candidate):
        candidate = default

    candidate = max(minimum, candidate)
    if maximum is not None:
        candidate = min(maximum, candidate)
    return candidate


def _coerce_finite_int(
    value: object,
    *,
    default: int,
    minimum: int,
) -> int:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        candidate = default
    return max(minimum, candidate)