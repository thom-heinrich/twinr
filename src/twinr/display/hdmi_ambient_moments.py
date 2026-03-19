"""Schedule rare ambient moments for Twinr's idle HDMI face.

The senior-facing waiting surface should stay calm most of the time, but it can
occasionally do something small and cute so the screen does not feel lifeless.
This module keeps that policy separate from the renderer: it deterministically
selects infrequent idle-only moments and describes their temporary face cue plus
tiny ornament.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib

from twinr.display.face_cues import DisplayFaceCue


_DEFAULT_BUCKET_SECONDS = 5.0 * 60.0
_DEFAULT_ACTIVE_WINDOW_S = 7.5
_DEFAULT_TRIGGER_DIVISOR = 6
_MOMENT_KEYS = ("sparkle", "heart", "curious", "sleepy", "wave", "crown")


@dataclass(frozen=True, slots=True)
class HdmiAmbientMoment:
    """Describe one active idle-only HDMI ambient moment."""

    key: str
    ornament: str
    progress: float
    face_cue: DisplayFaceCue


@dataclass(slots=True)
class HdmiAmbientMomentDirector:
    """Resolve rare ambient moments for the waiting HDMI scene.

    The schedule is deterministic instead of using process-global randomness, so
    tests can prove the behavior and the screen remains stable across restarts.
    Every time bucket has a small chance of becoming active for a short window.
    """

    bucket_seconds: float = _DEFAULT_BUCKET_SECONDS
    active_window_s: float = _DEFAULT_ACTIVE_WINDOW_S
    trigger_divisor: int = _DEFAULT_TRIGGER_DIVISOR

    def resolve(
        self,
        *,
        status: str,
        now: datetime | None,
        face_cue_active: bool,
        presentation_active: bool,
    ) -> HdmiAmbientMoment | None:
        """Return the active ambient moment, if the idle surface should show one."""

        if status != "waiting" or face_cue_active or presentation_active:
            return None

        safe_now = _normalize_now(now)
        bucket_seconds = max(60.0, float(self.bucket_seconds))
        active_window_s = min(max(1.0, float(self.active_window_s)), bucket_seconds - 1.0)
        trigger_divisor = max(1, int(self.trigger_divisor))
        epoch_s = safe_now.timestamp()
        bucket_index = int(epoch_s // bucket_seconds)
        bucket_offset_s = epoch_s - (bucket_index * bucket_seconds)
        if bucket_offset_s > active_window_s:
            return None

        seed = self.bucket_seed(bucket_index)
        if trigger_divisor > 1 and (seed % trigger_divisor) != 0:
            return None

        selection_seed = seed if trigger_divisor <= 1 else (seed // trigger_divisor)
        key = _MOMENT_KEYS[selection_seed % len(_MOMENT_KEYS)]
        progress = min(1.0, max(0.0, bucket_offset_s / active_window_s))
        return _build_moment(key=key, progress=progress)

    def bucket_seed(self, bucket_index: int) -> int:
        """Return a stable integer seed for one ambient time bucket."""

        digest = hashlib.sha256(f"twinr-hdmi-ambient:{int(bucket_index)}".encode("ascii")).digest()
        return int.from_bytes(digest[:8], "big", signed=False)


def _normalize_now(value: datetime | None) -> datetime:
    """Return one aware UTC datetime for ambient scheduling."""

    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
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
