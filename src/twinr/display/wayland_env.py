"""Resolve and prepare Wayland runtime details for Twinr HDMI output.

The HDMI fullscreen backend needs a concrete Wayland socket even when Twinr is
started from SSH or a systemd service that does not inherit the interactive
desktop environment. Centralizing the lookup here keeps backend and ops checks
aligned.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_wayland_runtime_dir(
    display_name: str,
    *,
    configured_runtime_dir: str | None = None,
) -> Path | None:
    """Return the runtime directory that contains ``display_name``.

    Resolution order is:

    1. explicitly configured runtime dir
    2. current ``XDG_RUNTIME_DIR``
    3. any matching socket under ``/run/user/*``
    """

    normalized_display = _normalize_display_name(display_name)
    candidates: list[Path] = []
    if configured_runtime_dir:
        candidates.append(Path(configured_runtime_dir).expanduser())
    env_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if env_runtime_dir:
        candidates.append(Path(env_runtime_dir).expanduser())
    for candidate in candidates:
        if (candidate / normalized_display).is_socket():
            return candidate

    run_user_root = Path("/run/user")
    if not run_user_root.exists():
        return None
    for socket_path in sorted(run_user_root.glob(f"*/{normalized_display}")):
        if socket_path.is_socket():
            return socket_path.parent
    return None


def resolve_wayland_socket(
    display_name: str,
    *,
    configured_runtime_dir: str | None = None,
) -> Path | None:
    """Return the concrete Wayland socket path for ``display_name`` if present."""

    runtime_dir = resolve_wayland_runtime_dir(
        display_name,
        configured_runtime_dir=configured_runtime_dir,
    )
    if runtime_dir is None:
        return None
    return runtime_dir / _normalize_display_name(display_name)


def apply_wayland_environment(
    display_name: str,
    *,
    configured_runtime_dir: str | None = None,
) -> Path:
    """Export the resolved Wayland runtime env and return the socket path."""

    socket_path = resolve_wayland_socket(
        display_name,
        configured_runtime_dir=configured_runtime_dir,
    )
    if socket_path is None:
        raise RuntimeError(
            f"Wayland socket `{_normalize_display_name(display_name)}` was not found. "
            "Configure TWINR_DISPLAY_WAYLAND_RUNTIME_DIR or start Twinr inside the desktop session."
        )
    os.environ["XDG_RUNTIME_DIR"] = str(socket_path.parent)
    os.environ["WAYLAND_DISPLAY"] = socket_path.name
    os.environ["SDL_VIDEODRIVER"] = "wayland"
    os.environ["QT_QPA_PLATFORM"] = "wayland"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    return socket_path


def _normalize_display_name(value: str | None) -> str:
    normalized = str(value or "wayland-0").strip()
    return normalized or "wayland-0"
