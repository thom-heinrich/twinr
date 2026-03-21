"""Prepare one safe subprocess environment for bounded audio helpers.

Twinr's productive Pi runtime can run as root while the HDMI Wayland surface is
borrowed from the logged-in desktop user. In that situation the process env may
contain ``XDG_RUNTIME_DIR=/run/user/1000`` and related session variables even
though the effective UID is 0. Blindly inheriting that env into ``arecord`` or
``aplay`` can make ALSA route through the non-root Pulse/PipeWire session and
produce misleading ``Device or resource busy`` or permission errors.

This helper keeps audio subprocesses independent from display/session transport
state: strip display-only variables unconditionally, and drop user-session
audio variables when their runtime dir is owned by a different UID.
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

    for key in _SESSION_AUDIO_ENV_KEYS:
        env.pop(key, None)
    return env


def runtime_dir_owner_uid(runtime_dir: str) -> int | None:
    """Return the owner uid for one runtime dir path when it exists."""

    try:
        return int(Path(runtime_dir).stat().st_uid)
    except OSError:
        return None


__all__ = ["build_audio_subprocess_env", "runtime_dir_owner_uid"]
