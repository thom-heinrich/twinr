from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
import json
import os
import time

from twinr.agent.base_agent.config import TwinrConfig

AdaptiveWindowKind = Literal["button", "follow_up"]


def _clamp_float(value: float, *, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _clamp_int(value: int, *, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


@dataclass(frozen=True, slots=True)
class AdaptiveListeningWindow:
    start_timeout_s: float
    speech_pause_ms: int
    pause_grace_ms: int


@dataclass(frozen=True, slots=True)
class AdaptiveTimingProfile:
    button_start_timeout_s: float
    follow_up_start_timeout_s: float
    speech_pause_ms: int
    pause_grace_ms: int
    button_success_count: int = 0
    button_timeout_count: int = 0
    follow_up_success_count: int = 0
    follow_up_timeout_count: int = 0
    pause_resume_count: int = 0
    clean_pause_streak: int = 0
    button_fast_start_streak: int = 0
    follow_up_fast_start_streak: int = 0

    def to_payload(self) -> dict[str, object]:
        return {
            "button_start_timeout_s": round(self.button_start_timeout_s, 3),
            "follow_up_start_timeout_s": round(self.follow_up_start_timeout_s, 3),
            "speech_pause_ms": self.speech_pause_ms,
            "pause_grace_ms": self.pause_grace_ms,
            "button_success_count": self.button_success_count,
            "button_timeout_count": self.button_timeout_count,
            "follow_up_success_count": self.follow_up_success_count,
            "follow_up_timeout_count": self.follow_up_timeout_count,
            "pause_resume_count": self.pause_resume_count,
            "clean_pause_streak": self.clean_pause_streak,
            "button_fast_start_streak": self.button_fast_start_streak,
            "follow_up_fast_start_streak": self.follow_up_fast_start_streak,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveTimingBounds:
    button_start_timeout_min_s: float
    button_start_timeout_max_s: float
    follow_up_start_timeout_min_s: float
    follow_up_start_timeout_max_s: float
    speech_pause_min_ms: int
    speech_pause_max_ms: int
    pause_grace_min_ms: int
    pause_grace_max_ms: int

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "AdaptiveTimingBounds":
        button_min = max(4.0, float(config.audio_start_timeout_s))
        follow_up_min = max(2.0, float(config.conversation_follow_up_timeout_s))
        speech_pause_min = max(700, int(config.speech_pause_ms))
        pause_grace_min = max(300, int(config.adaptive_timing_pause_grace_ms))
        return cls(
            button_start_timeout_min_s=button_min,
            button_start_timeout_max_s=max(button_min + 6.0, 14.0),
            follow_up_start_timeout_min_s=follow_up_min,
            follow_up_start_timeout_max_s=max(follow_up_min + 4.0, 8.0),
            speech_pause_min_ms=speech_pause_min,
            speech_pause_max_ms=max(speech_pause_min + 1200, 2600),
            pause_grace_min_ms=pause_grace_min,
            pause_grace_max_ms=max(pause_grace_min + 600, 1500),
        )


class AdaptiveTimingStore:
    def __init__(self, path: str | Path, *, config: TwinrConfig) -> None:
        self.path = Path(path)
        self.config = config
        self.bounds = AdaptiveTimingBounds.from_config(config)

    def current(self) -> AdaptiveTimingProfile:
        loaded = self._load_raw()
        if loaded is None:
            return self.default_profile()
        return self._coerce_profile(loaded)

    def ensure_saved(self) -> AdaptiveTimingProfile:
        profile = self.current()
        self._write(profile)
        return profile

    def reset(self) -> AdaptiveTimingProfile:
        profile = self.default_profile()
        self._write(profile)
        return profile

    def default_profile(self) -> AdaptiveTimingProfile:
        return AdaptiveTimingProfile(
            button_start_timeout_s=self.bounds.button_start_timeout_min_s,
            follow_up_start_timeout_s=self.bounds.follow_up_start_timeout_min_s,
            speech_pause_ms=self.bounds.speech_pause_min_ms,
            pause_grace_ms=self.bounds.pause_grace_min_ms,
        )

    def listening_window(
        self,
        *,
        initial_source: str,
        follow_up: bool,
    ) -> AdaptiveListeningWindow:
        profile = self.current()
        kind = self.window_kind(initial_source=initial_source, follow_up=follow_up)
        start_timeout_s = (
            profile.button_start_timeout_s
            if kind == "button"
            else profile.follow_up_start_timeout_s
        )
        return AdaptiveListeningWindow(
            start_timeout_s=start_timeout_s,
            speech_pause_ms=profile.speech_pause_ms,
            pause_grace_ms=profile.pause_grace_ms,
        )

    def record_no_speech_timeout(
        self,
        *,
        initial_source: str,
        follow_up: bool,
    ) -> AdaptiveTimingProfile:
        profile = self.current()
        kind = self.window_kind(initial_source=initial_source, follow_up=follow_up)
        if kind == "button":
            updated = replace(
                profile,
                button_start_timeout_s=_clamp_float(
                    profile.button_start_timeout_s + 0.75,
                    lower=self.bounds.button_start_timeout_min_s,
                    upper=self.bounds.button_start_timeout_max_s,
                ),
                button_timeout_count=profile.button_timeout_count + 1,
                button_fast_start_streak=0,
                clean_pause_streak=0,
            )
        else:
            updated = replace(
                profile,
                follow_up_start_timeout_s=_clamp_float(
                    profile.follow_up_start_timeout_s + 0.5,
                    lower=self.bounds.follow_up_start_timeout_min_s,
                    upper=self.bounds.follow_up_start_timeout_max_s,
                ),
                follow_up_timeout_count=profile.follow_up_timeout_count + 1,
                follow_up_fast_start_streak=0,
                clean_pause_streak=0,
            )
        self._write(updated)
        return updated

    def record_capture(
        self,
        *,
        initial_source: str,
        follow_up: bool,
        speech_started_after_ms: int,
        resumed_after_pause_count: int,
    ) -> AdaptiveTimingProfile:
        profile = self.current()
        kind = self.window_kind(initial_source=initial_source, follow_up=follow_up)
        updated = self._adapt_start_timeout(
            profile,
            kind=kind,
            speech_started_after_ms=max(0, int(speech_started_after_ms)),
        )
        updated = self._adapt_pause_behavior(
            updated,
            resumed_after_pause_count=max(0, int(resumed_after_pause_count)),
        )
        self._write(updated)
        return updated

    @staticmethod
    def window_kind(*, initial_source: str, follow_up: bool) -> AdaptiveWindowKind:
        if initial_source == "button" and not follow_up:
            return "button"
        return "follow_up"

    def _adapt_start_timeout(
        self,
        profile: AdaptiveTimingProfile,
        *,
        kind: AdaptiveWindowKind,
        speech_started_after_ms: int,
    ) -> AdaptiveTimingProfile:
        if kind == "button":
            current = profile.button_start_timeout_s
            fast_streak = profile.button_fast_start_streak
            min_s = self.bounds.button_start_timeout_min_s
            max_s = self.bounds.button_start_timeout_max_s
            margin_ms = 1800
            step_down_s = 0.15
            success_count_field = "button_success_count"
            fast_streak_field = "button_fast_start_streak"
        else:
            current = profile.follow_up_start_timeout_s
            fast_streak = profile.follow_up_fast_start_streak
            min_s = self.bounds.follow_up_start_timeout_min_s
            max_s = self.bounds.follow_up_start_timeout_max_s
            margin_ms = 1000
            step_down_s = 0.1
            success_count_field = "follow_up_success_count"
            fast_streak_field = "follow_up_fast_start_streak"

        updates: dict[str, object] = {
            success_count_field: getattr(profile, success_count_field) + 1,
        }
        target_timeout_s = _clamp_float(
            (speech_started_after_ms + margin_ms) / 1000.0,
            lower=min_s,
            upper=max_s,
        )
        if target_timeout_s > current + 0.05:
            updates[fast_streak_field] = 0
            new_timeout_s = _clamp_float(target_timeout_s, lower=min_s, upper=max_s)
        else:
            fast_threshold_ms = max(900, int(current * 1000 * 0.6))
            if speech_started_after_ms <= fast_threshold_ms:
                fast_streak += 1
                if fast_streak >= 3:
                    new_timeout_s = _clamp_float(
                        current - step_down_s,
                        lower=min_s,
                        upper=max_s,
                    )
                    fast_streak = 0
                else:
                    new_timeout_s = current
                updates[fast_streak_field] = fast_streak
            else:
                updates[fast_streak_field] = 0
                new_timeout_s = current

        if kind == "button":
            updates["button_start_timeout_s"] = new_timeout_s
        else:
            updates["follow_up_start_timeout_s"] = new_timeout_s
        return replace(profile, **updates)

    def _adapt_pause_behavior(
        self,
        profile: AdaptiveTimingProfile,
        *,
        resumed_after_pause_count: int,
    ) -> AdaptiveTimingProfile:
        if resumed_after_pause_count > 0:
            pause_step = min(240, 140 * resumed_after_pause_count)
            grace_step = min(180, 90 * resumed_after_pause_count)
            return replace(
                profile,
                speech_pause_ms=_clamp_int(
                    profile.speech_pause_ms + pause_step,
                    lower=self.bounds.speech_pause_min_ms,
                    upper=self.bounds.speech_pause_max_ms,
                ),
                pause_grace_ms=_clamp_int(
                    profile.pause_grace_ms + grace_step,
                    lower=self.bounds.pause_grace_min_ms,
                    upper=self.bounds.pause_grace_max_ms,
                ),
                pause_resume_count=profile.pause_resume_count + resumed_after_pause_count,
                clean_pause_streak=0,
            )

        clean_pause_streak = profile.clean_pause_streak + 1
        if clean_pause_streak < 3:
            return replace(profile, clean_pause_streak=clean_pause_streak)
        return replace(
            profile,
            speech_pause_ms=_clamp_int(
                profile.speech_pause_ms - 40,
                lower=self.bounds.speech_pause_min_ms,
                upper=self.bounds.speech_pause_max_ms,
            ),
            pause_grace_ms=_clamp_int(
                profile.pause_grace_ms - 20,
                lower=self.bounds.pause_grace_min_ms,
                upper=self.bounds.pause_grace_max_ms,
            ),
            clean_pause_streak=0,
        )

    def _load_raw(self) -> dict[str, object] | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        profile = payload.get("profile")
        if isinstance(profile, dict):
            return profile
        return payload

    def _coerce_profile(self, payload: dict[str, object]) -> AdaptiveTimingProfile:
        default = self.default_profile()
        return AdaptiveTimingProfile(
            button_start_timeout_s=_clamp_float(
                float(payload.get("button_start_timeout_s", default.button_start_timeout_s)),
                lower=self.bounds.button_start_timeout_min_s,
                upper=self.bounds.button_start_timeout_max_s,
            ),
            follow_up_start_timeout_s=_clamp_float(
                float(payload.get("follow_up_start_timeout_s", default.follow_up_start_timeout_s)),
                lower=self.bounds.follow_up_start_timeout_min_s,
                upper=self.bounds.follow_up_start_timeout_max_s,
            ),
            speech_pause_ms=_clamp_int(
                int(payload.get("speech_pause_ms", default.speech_pause_ms)),
                lower=self.bounds.speech_pause_min_ms,
                upper=self.bounds.speech_pause_max_ms,
            ),
            pause_grace_ms=_clamp_int(
                int(payload.get("pause_grace_ms", default.pause_grace_ms)),
                lower=self.bounds.pause_grace_min_ms,
                upper=self.bounds.pause_grace_max_ms,
            ),
            button_success_count=max(0, int(payload.get("button_success_count", 0))),
            button_timeout_count=max(0, int(payload.get("button_timeout_count", 0))),
            follow_up_success_count=max(0, int(payload.get("follow_up_success_count", 0))),
            follow_up_timeout_count=max(0, int(payload.get("follow_up_timeout_count", 0))),
            pause_resume_count=max(0, int(payload.get("pause_resume_count", 0))),
            clean_pause_streak=max(0, int(payload.get("clean_pause_streak", 0))),
            button_fast_start_streak=max(0, int(payload.get("button_fast_start_streak", 0))),
            follow_up_fast_start_streak=max(0, int(payload.get("follow_up_fast_start_streak", 0))),
        )

    def _write(self, profile: AdaptiveTimingProfile) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        payload = {
            "version": 1,
            "profile": profile.to_payload(),
        }
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)
