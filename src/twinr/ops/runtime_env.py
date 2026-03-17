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


def prime_user_session_audio_env() -> dict[str, str]:
    """Fill missing user-session env vars needed for detached audio runtime.

    The helper only writes variables that are currently unset. It derives the
    canonical runtime directory from the effective UID and then populates the
    DBus and PulseAudio socket addresses when the corresponding sockets exist.
    """

    updates: dict[str, str] = {}
    if os.name != "posix":
        return updates

    uid_getter = getattr(os, "getuid", None)
    if not callable(uid_getter):
        return updates
    uid = int(uid_getter())
    runtime_dir = Path(f"/run/user/{uid}")
    if not runtime_dir.is_dir():
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
