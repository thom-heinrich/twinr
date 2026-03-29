"""Prepare one safe subprocess environment for bounded audio helpers.

Twinr's productive Pi runtime can run as root while the HDMI Wayland surface is
borrowed from the logged-in desktop user. In that situation the process env may
contain ``XDG_RUNTIME_DIR=/run/user/1000`` and related session variables even
though the effective UID is 0. Blindly inheriting that env into ``arecord`` or
``aplay`` can make ALSA route through the non-root Pulse/PipeWire session and
produce misleading ``Device or resource busy`` or permission errors.

This helper keeps audio subprocesses independent from display/session transport
state: strip display-only variables unconditionally, and by default drop
user-session audio variables when their runtime dir is owned by a different
UID. Productive Pi services that intentionally borrow the logged-in desktop
user's audio session can opt back into those variables explicitly.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import os

_DISPLAY_ONLY_ENV_KEYS = (
    "WAYLAND_DISPLAY",
    "QT_QPA_PLATFORM",
    "SDL_VIDEODRIVER",
)
_SESSION_AUDIO_ENV_KEYS = (
    "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS",
    "PULSE_SERVER",
)


def build_audio_subprocess_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return one sanitized child env for ALSA-backed audio subprocesses."""

    return build_audio_subprocess_env_for_mode(base_env)


def build_audio_subprocess_env_for_mode(
    base_env: Mapping[str, str] | None = None,
    *,
    allow_root_borrowed_session_audio: bool = False,
) -> dict[str, str]:
    """Return one child env with optional root borrowed-session audio.

    When ``allow_root_borrowed_session_audio`` is enabled, root-owned helpers
    may keep the logged-in desktop user's audio-session variables instead of
    stripping them. This is required on productive Pi systems where the root
    supervisor intentionally borrows the desktop audio session to access the
    ReSpeaker capture path reliably.
    """

    env = dict(base_env or os.environ)
    for key in _DISPLAY_ONLY_ENV_KEYS:
        env.pop(key, None)

    runtime_dir = str(env.get("XDG_RUNTIME_DIR", "") or "").strip()
    uid_getter = getattr(os, "getuid", None)
    if not runtime_dir or not callable(uid_getter):
        return env

    current_uid = int(uid_getter())
    runtime_owner_uid = runtime_dir_owner_uid(runtime_dir)
    if runtime_owner_uid is None or runtime_owner_uid == current_uid:
        return env
    if _keep_foreign_session_audio(
        current_uid=current_uid,
        runtime_owner_uid=runtime_owner_uid,
        allow_root_borrowed_session_audio=allow_root_borrowed_session_audio,
    ):
        return env

    for key in _SESSION_AUDIO_ENV_KEYS:
        env.pop(key, None)
    return env


def _keep_foreign_session_audio(
    *,
    current_uid: int,
    runtime_owner_uid: int,
    allow_root_borrowed_session_audio: bool,
) -> bool:
    """Return whether a root-owned helper may keep another user's audio session."""

    return (
        bool(allow_root_borrowed_session_audio)
        and current_uid == 0
        and runtime_owner_uid > 0
    )


def runtime_dir_owner_uid(runtime_dir: str) -> int | None:
    """Return the owner uid for one runtime dir path when it exists."""

    try:
        return int(Path(runtime_dir).stat().st_uid)
    except OSError:
        return None


__all__ = [
    "build_audio_subprocess_env",
    "build_audio_subprocess_env_for_mode",
    "runtime_dir_owner_uid",
]
