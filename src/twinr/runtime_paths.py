"""Prime Pi runtime Python paths for OS-managed modules.

Twinr runs inside a project virtual environment on the Raspberry Pi, but some
hardware/runtime dependencies are intentionally provided by the OS image
instead of the repo venv. Examples include ``picamera2`` and, on some Pi
images, shared support packages such as ``python-dateutil``. When the venv was
created without ``--system-site-packages``, those modules disappear from the
runtime import path and Twinr fails before the productive loops can even start.

This module owns one small, explicit bootstrap step that appends the relevant
Pi system ``dist-packages`` directories to ``sys.path`` only on Raspberry Pi
hosts and only when those directories exist. Appending preserves the venv as
the primary source for repo-managed packages while restoring access to the
OS-managed runtime modules Twinr depends on in Pi acceptance.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys


_DEFAULT_DEVICE_MODEL_PATH = Path("/proc/device-tree/model")


def prime_raspberry_pi_system_site_packages(
    *,
    sys_path: list[str] | None = None,
    candidate_paths: tuple[str | Path, ...] | None = None,
    device_model_path: str | Path = _DEFAULT_DEVICE_MODEL_PATH,
) -> tuple[str, ...]:
    """Append missing Pi system ``dist-packages`` directories once.

    The helper is intentionally conservative:
    - no-op outside Raspberry Pi hosts
    - no-op for missing directories
    - keeps venv entries earlier on ``sys.path``
    - returns the exact paths it added for callers/tests
    """

    target_path = sys.path if sys_path is None else sys_path
    if not _is_raspberry_pi_host(device_model_path=device_model_path):
        return ()

    added: list[str] = []
    for candidate in candidate_paths or _default_system_site_package_paths():
        resolved = str(Path(candidate))
        if resolved in target_path:
            continue
        if not Path(resolved).exists():
            continue
        target_path.append(resolved)
        added.append(resolved)
    return tuple(added)


def _is_raspberry_pi_host(*, device_model_path: str | Path) -> bool:
    """Return whether the current process runs on Raspberry Pi hardware."""

    if os.name != "posix":
        return False
    try:
        model_text = Path(device_model_path).read_bytes()
    except OSError:
        return False
    return b"Raspberry Pi" in model_text


def _default_system_site_package_paths() -> tuple[Path, ...]:
    """Return the Pi OS ``dist-packages`` locations Twinr may rely on."""

    major = int(sys.version_info[0])
    minor = int(sys.version_info[1])
    return (
        Path(f"/usr/local/lib/python{major}.{minor}/dist-packages"),
        Path("/usr/lib/python3/dist-packages"),
        Path(f"/usr/lib/python{major}.{minor}/dist-packages"),
    )


__all__ = ["prime_raspberry_pi_system_site_packages"]
