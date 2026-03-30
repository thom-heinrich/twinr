"""Seed detached Twinr runtime processes with the user audio-session env.

Twinr's productive Pi loops are often launched outside an interactive shell or
desktop session. In that context ALSA's ``default`` device resolves through the
PulseAudio bridge only when session variables such as ``XDG_RUNTIME_DIR`` are
present. Without them, simple speaker actions like the initial listen beep fail
with ``audio open error: No such file or directory`` even though the hardware
itself is fine.

This module restores the minimum user-session environment that detached Twinr
processes need on the Pi, without overwriting explicit operator configuration.
"""

from __future__ import annotations

from pathlib import Path
import os


def _resolve_runtime_dir_candidate(
    *,
    configured_runtime_dir: str | os.PathLike[str] | None,
) -> Path | None:
    runtime_dir_text = str(os.environ.get("XDG_RUNTIME_DIR", "") or "").strip()
    if runtime_dir_text:
        runtime_dir = Path(runtime_dir_text)
        if runtime_dir.is_dir():
            return runtime_dir

    uid_getter = getattr(os, "getuid", None)
    if not callable(uid_getter):
        return None
    uid = int(uid_getter())

    configured_candidate: Path | None = None
    if configured_runtime_dir is not None:
        configured_text = str(configured_runtime_dir).strip()
        if configured_text:
            candidate = Path(configured_text).expanduser()
            if candidate.is_dir():
                configured_candidate = candidate

    # Productive Pi runtimes run as root but borrow the logged-in desktop
    # user's session for audio and Wayland. Prefer the configured desktop
    # runtime dir there over /run/user/0, which often does not carry the
    # needed DBus/Pulse sockets.
    if uid == 0 and configured_candidate is not None:
        return configured_candidate

    own_runtime_dir = Path(f"/run/user/{uid}")
    if own_runtime_dir.is_dir():
        return own_runtime_dir

    return configured_candidate


def prime_user_session_audio_env(
    *,
    configured_runtime_dir: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Fill missing user-session env vars needed for detached audio runtime.

    The helper only writes variables that are currently unset. It derives the
    canonical runtime directory from the effective UID or, on productive
    root-owned Pi runtimes, from the configured desktop runtime directory.
    It then populates the DBus and PulseAudio socket addresses when the
    corresponding sockets exist.
    """

    updates: dict[str, str] = {}
    if os.name != "posix":
        return updates

    runtime_dir = _resolve_runtime_dir_candidate(
        configured_runtime_dir=configured_runtime_dir,
    )
    if runtime_dir is None:
        return updates

    if not str(os.environ.get("XDG_RUNTIME_DIR", "")).strip():
        value = str(runtime_dir)
        os.environ["XDG_RUNTIME_DIR"] = value
        updates["XDG_RUNTIME_DIR"] = value

    bus_socket = runtime_dir / "bus"
    if bus_socket.exists() and not str(os.environ.get("DBUS_SESSION_BUS_ADDRESS", "")).strip():
        value = f"unix:path={bus_socket}"
        os.environ["DBUS_SESSION_BUS_ADDRESS"] = value
        updates["DBUS_SESSION_BUS_ADDRESS"] = value

    pulse_socket = runtime_dir / "pulse" / "native"
    if pulse_socket.exists() and not str(os.environ.get("PULSE_SERVER", "")).strip():
        value = f"unix:{pulse_socket}"
        os.environ["PULSE_SERVER"] = value
        updates["PULSE_SERVER"] = value

    return updates
