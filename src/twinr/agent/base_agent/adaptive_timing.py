from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal
import contextlib
import fcntl
import json
import logging
import math
import os
import stat
import time

from twinr.agent.base_agent.config import TwinrConfig

LOG = logging.getLogger(__name__)  # AUDIT-FIX(#1): Persistenz- und Parsefehler protokollieren statt unkontrolliert Interaktionspfade abstürzen zu lassen.
_FILE_LOCK_POLL_S = 0.05  # AUDIT-FIX(#2): Kurzes Polling für nicht-blockierende Dateisperren auf dem shared file-backed state.
_FILE_LOCK_TIMEOUT_S = 0.25  # AUDIT-FIX(#2): Lock-Timeout begrenzen, damit Audio-Pfade bei hängender Sperre degradiert statt blockiert weiterlaufen.

AdaptiveWindowKind = Literal["button", "follow_up"]


def _clamp_float(value: float, *, lower: float, upper: float) -> float:
    candidate = float(value)
    if not math.isfinite(candidate):  # AUDIT-FIX(#3): Nicht-endliche Werte nicht in das Profil übernehmen.
        candidate = lower
    return max(lower, min(upper, candidate))


def _clamp_int(value: int, *, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _coerce_float(value: object, *, default: float) -> float:
    if isinstance(value, bool):  # AUDIT-FIX(#3): Bool-Werte aus JSON/.env nicht still als 0/1 akzeptieren.
        return default
    try:
        candidate = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(candidate):
        return default
    return candidate


def _coerce_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):  # AUDIT-FIX(#3): Bool-Werte sind für Timing-/Counter-Felder semantisch ungültig.
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        try:
            candidate = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        if not math.isfinite(candidate) or not candidate.is_integer():
            return default
        return int(candidate)


def _normalize_store_path(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    return Path(os.path.abspath(os.fspath(expanded)))  # AUDIT-FIX(#4): Relativpfade/.. vor Validierung deterministisch normalisieren.


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
        button_min = max(
            4.0,
            _coerce_float(getattr(config, "audio_start_timeout_s", 4.0), default=4.0),
        )  # AUDIT-FIX(#6): Fehlkonfigurierte .env/config-Werte dürfen die Store-Initialisierung nicht crashen.
        follow_up_min = max(
            2.0,
            _coerce_float(
                getattr(config, "conversation_follow_up_timeout_s", 2.0),
                default=2.0,
            ),
        )  # AUDIT-FIX(#6): Follow-up-Minimum robust gegen ungültige Config-Werte machen.
        speech_pause_min = max(
            700,
            _coerce_int(getattr(config, "speech_pause_ms", 700), default=700),
        )  # AUDIT-FIX(#6): Integer-Timings robust aus Config übernehmen.
        pause_grace_min = max(
            300,
            _coerce_int(
                getattr(config, "adaptive_timing_pause_grace_ms", 300),
                default=300,
            ),
        )  # AUDIT-FIX(#6): Grace-Window bei Konfigurationsfehlern sicher auf Mindestwert setzen.
        return cls(
            button_start_timeout_min_s=button_min,
            button_start_timeout_max_s=max(button_min + 6.0, 14.0),
            follow_up_start_timeout_min_s=follow_up_min,
            follow_up_start_timeout_max_s=max(follow_up_min + 4.0, 8.0),
            speech_pause_min_ms=speech_pause_min,
            speech_pause_max_ms=speech_pause_min + 400,
            pause_grace_min_ms=pause_grace_min,
            pause_grace_max_ms=pause_grace_min + 200,
        )


class AdaptiveTimingStore:
    def __init__(self, path: str | Path, *, config: TwinrConfig) -> None:
        self.path = _normalize_store_path(path)  # AUDIT-FIX(#4): Normalisierte Zielpfade reduzieren TOCTOU-/Traversal-Fehlkonfigurationen.
        self.config = config
        self.bounds = AdaptiveTimingBounds.from_config(config)
        self._cached_profile = self.default_profile()  # AUDIT-FIX(#1): Last-known-good Profil im Speicher halten, falls Persistenz temporär ausfällt.

    def current(self) -> AdaptiveTimingProfile:
        with self._storage_lock() as storage_path:  # AUDIT-FIX(#2): Reads mit derselben Dateisperre koordinieren wie Writes.
            return self._load_profile_locked(storage_path)

    def ensure_saved(self) -> AdaptiveTimingProfile:
        with self._storage_lock() as storage_path:  # AUDIT-FIX(#2): Laden und Speichern atomar unter derselben Sperre ausführen.
            profile = self._load_profile_locked(storage_path)
            self._write_locked(storage_path, profile)
            return profile

    def reset(self) -> AdaptiveTimingProfile:
        profile = self.default_profile()
        self._cached_profile = profile  # AUDIT-FIX(#1): Reset auch dann wirksam halten, wenn das Dateisystem gerade nicht schreibbar ist.
        with self._storage_lock() as storage_path:
            self._write_locked(storage_path, profile)
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
        kind = self.window_kind(initial_source=initial_source, follow_up=follow_up)

        def mutate(profile: AdaptiveTimingProfile) -> AdaptiveTimingProfile:
            if kind == "button":
                return replace(
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
            return replace(
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

        return self._mutate_profile(mutate)  # AUDIT-FIX(#2): Mutationen zentral als gelockten Read-Modify-Write ausführen.

    def record_capture(
        self,
        *,
        initial_source: str,
        follow_up: bool,
        speech_started_after_ms: int,
        resumed_after_pause_count: int,
    ) -> AdaptiveTimingProfile:
        kind = self.window_kind(initial_source=initial_source, follow_up=follow_up)
        speech_started_after_ms = max(
            0,
            _coerce_int(speech_started_after_ms, default=0),
        )  # AUDIT-FIX(#3): Ungültige Laufzeitparameter nicht per int() abstürzen lassen.
        resumed_after_pause_count = max(
            0,
            _coerce_int(resumed_after_pause_count, default=0),
        )  # AUDIT-FIX(#3): Counter-Eingaben defensiv normalisieren.

        def mutate(profile: AdaptiveTimingProfile) -> AdaptiveTimingProfile:
            updated = self._adapt_start_timeout(
                profile,
                kind=kind,
                speech_started_after_ms=speech_started_after_ms,
            )
            return self._adapt_pause_behavior(
                updated,
                resumed_after_pause_count=resumed_after_pause_count,
            )

        return self._mutate_profile(mutate)  # AUDIT-FIX(#2): Capture-Lernen atomar persistieren, um verlorene Streaks/Zähler zu verhindern.

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
            # Learn conservatively from resumed pauses so one noisy or hesitant turn
            # does not make future turn endings feel sluggish.
            pause_step = min(120, 30 * resumed_after_pause_count)
            grace_step = min(60, 20 * resumed_after_pause_count)
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
        if clean_pause_streak < 2:
            return replace(profile, clean_pause_streak=clean_pause_streak)
        return replace(
            profile,
            speech_pause_ms=_clamp_int(
                profile.speech_pause_ms - 60,
                lower=self.bounds.speech_pause_min_ms,
                upper=self.bounds.speech_pause_max_ms,
            ),
            pause_grace_ms=_clamp_int(
                profile.pause_grace_ms - 40,
                lower=self.bounds.pause_grace_min_ms,
                upper=self.bounds.pause_grace_max_ms,
            ),
            clean_pause_streak=0,
        )

    def _mutate_profile(
        self,
        mutator: Callable[[AdaptiveTimingProfile], AdaptiveTimingProfile],
    ) -> AdaptiveTimingProfile:
        with self._storage_lock() as storage_path:  # AUDIT-FIX(#2): Gemeinsame Lock-Hülle für jeden Profil-Mutationspfad.
            profile = self._load_profile_locked(storage_path)
            updated = mutator(profile)
            self._cached_profile = updated
            self._write_locked(storage_path, updated)
            return updated

    @contextlib.contextmanager
    def _storage_lock(self):
        storage_path = self._validated_storage_path()
        if storage_path is None:
            yield None
            return

        try:
            storage_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            LOG.warning(
                "adaptive timing storage unavailable while creating %s: %s",
                storage_path.parent,
                exc,
            )  # AUDIT-FIX(#1): Dateisystemfehler als Degradationsfall behandeln.
            yield None
            return

        storage_path = self._validated_storage_path()  # AUDIT-FIX(#4): Nach mkdir nochmals validieren, um Symlink-/Typ-Wechsel zwischen Prüfung und Nutzung zu erkennen.
        if storage_path is None:
            yield None
            return

        lock_path = storage_path.with_name(f".{storage_path.name}.lock")
        if lock_path.is_symlink():
            LOG.warning(
                "adaptive timing storage disabled because lock path is a symlink: %s",
                lock_path,
            )  # AUDIT-FIX(#4): Lock-Datei nicht über Symlink dereferenzieren.
            yield None
            return

        lock_flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            lock_fd = os.open(lock_path, lock_flags, 0o600)
        except OSError as exc:
            LOG.warning(
                "adaptive timing storage lock unavailable for %s: %s",
                storage_path,
                exc,
            )  # AUDIT-FIX(#1): Kann die Sperre nicht erstellt werden, wird nur in-memory weitergearbeitet.
            yield None
            return

        try:
            with os.fdopen(lock_fd, "r+", encoding="utf-8") as lock_file:
                deadline = time.monotonic() + _FILE_LOCK_TIMEOUT_S
                while True:
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            LOG.warning(
                                "adaptive timing storage lock timed out for %s",
                                storage_path,
                            )  # AUDIT-FIX(#2): Hängende Fremdsperren nicht unbegrenzt auf den Audiopfad durchschlagen lassen.
                            yield None
                            return
                        time.sleep(_FILE_LOCK_POLL_S)
                try:
                    yield storage_path
                finally:
                    with contextlib.suppress(OSError):
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            LOG.warning(
                "adaptive timing storage lock handling failed for %s: %s",
                storage_path,
                exc,
            )  # AUDIT-FIX(#1): Lock-bezogene OSErrors abfangen und degradiert weiterlaufen.

    def _validated_storage_path(self) -> Path | None:
        path = self.path
        for candidate in (path, *path.parents):
            if candidate.is_symlink():
                LOG.warning(
                    "adaptive timing storage disabled because path contains symlink component: %s",
                    candidate,
                )  # AUDIT-FIX(#4): Symlink-Komponenten in State-Pfaden als unsicher behandeln.
                return None

        if path.exists() and not path.is_file():
            LOG.warning(
                "adaptive timing storage disabled because target is not a regular file: %s",
                path,
            )  # AUDIT-FIX(#4): Verzeichnisse/devices/FIFOs nicht als State-Datei verwenden.
            return None

        ancestor = path.parent
        while not ancestor.exists():
            parent = ancestor.parent
            if parent == ancestor:
                break
            ancestor = parent
        if ancestor.exists() and not ancestor.is_dir():
            LOG.warning(
                "adaptive timing storage disabled because parent ancestor is not a directory: %s",
                ancestor,
            )  # AUDIT-FIX(#4): mkdir auf nicht-Verzeichnis-Vorfahren verhindern.
            return None
        return path

    def _load_profile_locked(self, storage_path: Path | None) -> AdaptiveTimingProfile:
        payload = self._load_raw_locked(storage_path)
        if payload is None:
            return self._cached_profile  # AUDIT-FIX(#1): Bei Lese-/Parsefehlern Last-known-good statt Reset auf volatile Defaults nutzen.
        profile = self._coerce_profile(payload)
        self._cached_profile = profile
        return profile

    def _load_raw_locked(self, storage_path: Path | None) -> dict[str, object] | None:
        if storage_path is None:
            return None

        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(storage_path, read_flags)
        except FileNotFoundError:
            return None
        except OSError as exc:
            LOG.warning(
                "adaptive timing storage read failed for %s: %s",
                storage_path,
                exc,
            )  # AUDIT-FIX(#1): Dateisystem-Lesefehler nicht eskalieren.
            return None

        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                LOG.warning(
                    "adaptive timing storage ignored non-regular file target: %s",
                    storage_path,
                )  # AUDIT-FIX(#4): Nur reguläre Dateien deserialisieren.
                return None
            with os.fdopen(fd, "r", encoding="utf-8") as file_obj:
                fd = -1
                try:
                    payload = json.load(file_obj)
                except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError) as exc:
                    LOG.warning(
                        "adaptive timing storage JSON decode failed for %s: %s",
                        storage_path,
                        exc,
                    )  # AUDIT-FIX(#1): Korrupten State sauber degradieren.
                    return None
        finally:
            if fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(fd)

        if not isinstance(payload, dict):
            LOG.warning(
                "adaptive timing storage ignored non-object JSON payload in %s",
                storage_path,
            )  # AUDIT-FIX(#3): Unerwartete JSON-Root-Typen nicht in _coerce_profile laufen lassen.
            return None

        profile = payload.get("profile")
        if isinstance(profile, dict):
            return profile
        return payload

    def _coerce_profile(self, payload: dict[str, object]) -> AdaptiveTimingProfile:
        default = self.default_profile()
        return AdaptiveTimingProfile(
            button_start_timeout_s=_clamp_float(
                _coerce_float(
                    payload.get("button_start_timeout_s", default.button_start_timeout_s),
                    default=default.button_start_timeout_s,
                ),
                lower=self.bounds.button_start_timeout_min_s,
                upper=self.bounds.button_start_timeout_max_s,
            ),  # AUDIT-FIX(#3): Kaputte Payload-Werte auf sicheren Default zurückführen statt ValueError/TypeError auszulösen.
            follow_up_start_timeout_s=_clamp_float(
                _coerce_float(
                    payload.get(
                        "follow_up_start_timeout_s",
                        default.follow_up_start_timeout_s,
                    ),
                    default=default.follow_up_start_timeout_s,
                ),
                lower=self.bounds.follow_up_start_timeout_min_s,
                upper=self.bounds.follow_up_start_timeout_max_s,
            ),  # AUDIT-FIX(#3): Nicht-finite/ungültige Timeouts defensiv behandeln.
            speech_pause_ms=_clamp_int(
                _coerce_int(
                    payload.get("speech_pause_ms", default.speech_pause_ms),
                    default=default.speech_pause_ms,
                ),
                lower=self.bounds.speech_pause_min_ms,
                upper=self.bounds.speech_pause_max_ms,
            ),  # AUDIT-FIX(#3): Pause-Felder robust aus beliebigem JSON parsen.
            pause_grace_ms=_clamp_int(
                _coerce_int(
                    payload.get("pause_grace_ms", default.pause_grace_ms),
                    default=default.pause_grace_ms,
                ),
                lower=self.bounds.pause_grace_min_ms,
                upper=self.bounds.pause_grace_max_ms,
            ),  # AUDIT-FIX(#3): Grace-Felder robust aus beliebigem JSON parsen.
            button_success_count=max(
                0,
                _coerce_int(payload.get("button_success_count", 0), default=0),
            ),  # AUDIT-FIX(#3): Counter nie aus kaputtem JSON crashen oder negativ übernehmen.
            button_timeout_count=max(
                0,
                _coerce_int(payload.get("button_timeout_count", 0), default=0),
            ),  # AUDIT-FIX(#3): Counter defensiv normalisieren.
            follow_up_success_count=max(
                0,
                _coerce_int(payload.get("follow_up_success_count", 0), default=0),
            ),  # AUDIT-FIX(#3): Counter defensiv normalisieren.
            follow_up_timeout_count=max(
                0,
                _coerce_int(payload.get("follow_up_timeout_count", 0), default=0),
            ),  # AUDIT-FIX(#3): Counter defensiv normalisieren.
            pause_resume_count=max(
                0,
                _coerce_int(payload.get("pause_resume_count", 0), default=0),
            ),  # AUDIT-FIX(#3): Counter defensiv normalisieren.
            clean_pause_streak=max(
                0,
                _coerce_int(payload.get("clean_pause_streak", 0), default=0),
            ),  # AUDIT-FIX(#3): Counter defensiv normalisieren.
            button_fast_start_streak=max(
                0,
                _coerce_int(payload.get("button_fast_start_streak", 0), default=0),
            ),  # AUDIT-FIX(#3): Counter defensiv normalisieren.
            follow_up_fast_start_streak=max(
                0,
                _coerce_int(payload.get("follow_up_fast_start_streak", 0), default=0),
            ),  # AUDIT-FIX(#3): Counter defensiv normalisieren.
        )

    def _write_locked(self, storage_path: Path | None, profile: AdaptiveTimingProfile) -> None:
        self._cached_profile = profile
        if storage_path is None:
            return

        tmp_path = storage_path.with_name(
            f".{storage_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        payload_bytes = json.dumps(
            {
                "version": 1,
                "profile": profile.to_payload(),
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8")

        tmp_fd = -1
        try:
            tmp_flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_TRUNC
                | getattr(os, "O_NOFOLLOW", 0)
            )
            tmp_fd = os.open(tmp_path, tmp_flags, 0o600)  # AUDIT-FIX(#5): Temp-Datei exklusiv und mit restriktiven Rechten anlegen.
            with os.fdopen(tmp_fd, "wb") as tmp_file:
                tmp_fd = -1
                tmp_file.write(payload_bytes)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())  # AUDIT-FIX(#5): Temp-Datei vor Rename dauerhaft auf Storage bringen.
            os.replace(tmp_path, storage_path)
            self._fsync_directory(storage_path.parent)  # AUDIT-FIX(#5): Directory fsync nach Rename reduziert Verlust bei Stromausfall.
        except OSError as exc:
            LOG.warning(
                "adaptive timing storage write failed for %s: %s",
                storage_path,
                exc,
            )  # AUDIT-FIX(#1): Schreibfehler nicht in den Runtime-Pfad eskalieren.
        finally:
            if tmp_fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(tmp_fd)
            with contextlib.suppress(FileNotFoundError, OSError):
                tmp_path.unlink()  # AUDIT-FIX(#5): Verwaiste Temp-Dateien nach Fehlern konsequent aufräumen.

    def _fsync_directory(self, directory: Path) -> None:
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        except OSError:
            return
        finally:
            with contextlib.suppress(OSError):
                os.close(dir_fd)