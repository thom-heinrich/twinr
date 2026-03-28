"""Bridge Pi OS-managed dist-packages into the project virtualenv.

Twinr keeps ``/twinr/.venv`` isolated from the system interpreter, but some
Raspberry Pi hardware modules are intentionally installed by the OS image
instead of pip inside the repo venv. This helper writes one deterministic
``.pth`` file inside the venv so direct ``/twinr/.venv/bin/python`` imports
can still resolve those Pi-managed modules without recreating the whole
environment with ``--system-site-packages``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass(frozen=True, slots=True)
class PiVenvSystemSiteBridgeResult:
    """Describe one `.pth` bridge attestation/update for the Pi venv."""

    bridge_path: str
    active_paths: tuple[str, ...]
    changed: bool


def ensure_pi_system_site_packages_bridge(
    *,
    site_packages_dir: Path,
    candidate_paths: tuple[str | Path, ...] | None = None,
    bridge_filename: str = "twinr_pi_system_site.pth",
) -> PiVenvSystemSiteBridgeResult:
    """Ensure the venv exposes the existing Pi OS `dist-packages` paths."""

    site_packages_dir.mkdir(parents=True, exist_ok=True)
    bridge_path = site_packages_dir / bridge_filename
    active_paths = tuple(
        str(Path(raw_path))
        for raw_path in (candidate_paths or _default_pi_system_site_package_paths())
        if Path(raw_path).exists()
    )
    expected_content = "".join(f"{path}\n" for path in active_paths)
    previous_content = bridge_path.read_text(encoding="utf-8") if bridge_path.exists() else None
    changed = previous_content != expected_content

    if active_paths:
        if changed:
            bridge_path.write_text(expected_content, encoding="utf-8")
    elif bridge_path.exists():
        bridge_path.unlink()

    return PiVenvSystemSiteBridgeResult(
        bridge_path=str(bridge_path),
        active_paths=active_paths,
        changed=changed,
    )


def _default_pi_system_site_package_paths() -> tuple[Path, ...]:
    """Return the Pi OS `dist-packages` locations Twinr may rely on."""

    major = int(sys.version_info[0])
    minor = int(sys.version_info[1])
    return (
        Path(f"/usr/local/lib/python{major}.{minor}/dist-packages"),
        Path("/usr/lib/python3/dist-packages"),
        Path(f"/usr/lib/python{major}.{minor}/dist-packages"),
    )


__all__ = [
    "PiVenvSystemSiteBridgeResult",
    "ensure_pi_system_site_packages_bridge",
]
