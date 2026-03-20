"""Resolve and describe Twinr's local WhatsApp Node.js runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform

PINNED_WHATSAPP_NODE_VERSION = "20.20.1"

_SYSTEM_ALIASES = {
    "linux": "linux",
    "darwin": "darwin",
}

_ARCH_ALIASES = {
    "aarch64": "arm64",
    "arm64": "arm64",
    "amd64": "x64",
    "x86_64": "x64",
    "armv7l": "armv7l",
}


@dataclass(frozen=True, slots=True)
class WhatsAppNodeRuntimeSpec:
    """Describe one pinned local Node.js runtime staged under ``state/tools``."""

    project_root: Path
    version: str
    platform_name: str
    arch: str

    @property
    def runtime_slug(self) -> str:
        """Return the Node.js distribution slug used by the official archives."""

        return f"node-v{self.version}-{self.platform_name}-{self.arch}"

    @property
    def install_root(self) -> Path:
        """Return the canonical Twinr install directory for this runtime."""

        return self.project_root / "state" / "tools" / self.runtime_slug

    @property
    def binary_path(self) -> Path:
        """Return the expected absolute ``node`` binary path."""

        binary_name = "node.exe" if self.platform_name == "win" else "node"
        return self.install_root / "bin" / binary_name

    @property
    def archive_name(self) -> str:
        """Return the official archive filename for this runtime."""

        extension = "zip" if self.platform_name == "win" else "tar.xz"
        return f"{self.runtime_slug}.{extension}"

    @property
    def download_url(self) -> str:
        """Return the official Node.js download URL for this runtime."""

        return f"https://nodejs.org/dist/v{self.version}/{self.archive_name}"

    @property
    def shasums_url(self) -> str:
        """Return the official checksum manifest URL for this runtime release."""

        return f"https://nodejs.org/dist/v{self.version}/SHASUMS256.txt"


def detect_whatsapp_node_runtime_spec(
    project_root: str | Path,
    *,
    system_name: str | None = None,
    machine_name: str | None = None,
    version: str = PINNED_WHATSAPP_NODE_VERSION,
) -> WhatsAppNodeRuntimeSpec:
    """Return the pinned local Node.js runtime target for one host platform."""

    normalized_system = _normalize_system_name(system_name or platform.system())
    normalized_arch = _normalize_arch_name(machine_name or platform.machine())
    return WhatsAppNodeRuntimeSpec(
        project_root=Path(project_root).expanduser().resolve(strict=False),
        version=str(version).strip(),
        platform_name=normalized_system,
        arch=normalized_arch,
    )


def resolve_whatsapp_node_binary(
    project_root: str | Path,
    configured_value: str | None,
    *,
    system_name: str | None = None,
    machine_name: str | None = None,
) -> str:
    """Resolve the effective Node.js binary for the Baileys worker.

    If the config keeps the generic ``node`` default and a pinned Twinr-local
    runtime exists under ``state/tools``, prefer that local runtime so the Pi
    does not depend on a separately managed PATH installation.
    """

    normalized_value = str(configured_value or "").strip()
    if normalized_value in {"", "node"}:
        try:
            local_runtime = detect_whatsapp_node_runtime_spec(
                project_root,
                system_name=system_name,
                machine_name=machine_name,
            )
        except ValueError:
            return normalized_value or "node"
        if local_runtime.binary_path.is_file():
            return str(local_runtime.binary_path)
        return normalized_value or "node"

    candidate = Path(normalized_value).expanduser()
    if not candidate.is_absolute():
        candidate = Path(project_root).expanduser().resolve(strict=False) / candidate
    return str(candidate.resolve(strict=False))


def _normalize_system_name(raw_value: str) -> str:
    """Normalize one platform label into the Node.js archive naming scheme."""

    cleaned = str(raw_value or "").strip().lower()
    normalized = _SYSTEM_ALIASES.get(cleaned)
    if normalized is None:
        raise ValueError(f"Unsupported Node.js platform for the WhatsApp worker: {raw_value!r}")
    return normalized


def _normalize_arch_name(raw_value: str) -> str:
    """Normalize one machine architecture into the Node.js archive naming scheme."""

    cleaned = str(raw_value or "").strip().lower()
    normalized = _ARCH_ALIASES.get(cleaned)
    if normalized is None:
        raise ValueError(f"Unsupported Node.js architecture for the WhatsApp worker: {raw_value!r}")
    return normalized


__all__ = [
    "PINNED_WHATSAPP_NODE_VERSION",
    "WhatsAppNodeRuntimeSpec",
    "detect_whatsapp_node_runtime_spec",
    "resolve_whatsapp_node_binary",
]
