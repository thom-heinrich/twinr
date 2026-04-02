"""Resolve host-side text-probe websocket targets against the local bridge.

Host-side operator probes and non-voice acceptance should talk to the
authoritative local orchestrator bridge when the leading repo machine already
hosts that bridge on loopback. Repo-local ``.env`` files can retain a stale
LAN self-address across DHCP changes; this module rewrites that self-target to
``ws://127.0.0.1`` only for host-side probe contexts and only when the local
bridge port is actually reachable.
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
import socket
from typing import Final
from urllib.parse import SplitResult, urlsplit, urlunsplit

from twinr.agent.base_agent.config import TwinrConfig


_PI_RUNTIME_ROOT: Final[Path] = Path("/twinr").resolve()


@dataclass(frozen=True, slots=True)
class LocalOrchestratorBridgeTarget:
    """Describe the resolved websocket target for one host-side probe turn."""

    url: str
    rewritten: bool = False
    reason: str | None = None


def resolve_local_orchestrator_probe_target(
    config: TwinrConfig,
    *,
    connect_timeout_s: float = 0.25,
) -> LocalOrchestratorBridgeTarget:
    """Prefer the local loopback bridge for host-side text probes when present."""

    configured_url = str(getattr(config, "orchestrator_ws_url", "") or "").strip()
    if not configured_url:
        return LocalOrchestratorBridgeTarget(url=configured_url)

    parsed = urlsplit(configured_url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        return LocalOrchestratorBridgeTarget(url=configured_url)
    if _project_root_is_pi_runtime(getattr(config, "project_root", None)):
        return LocalOrchestratorBridgeTarget(url=configured_url)
    if _is_loopback_host(parsed.hostname):
        return LocalOrchestratorBridgeTarget(url=configured_url)

    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    if not _loopback_port_reachable(port, timeout_s=connect_timeout_s):
        return LocalOrchestratorBridgeTarget(url=configured_url)

    local_url = urlunsplit(_replace_target_host(parsed, host="127.0.0.1", scheme="ws", port=port))
    return LocalOrchestratorBridgeTarget(
        url=local_url,
        rewritten=True,
        reason="host_loopback_bridge_override",
    )


def _project_root_is_pi_runtime(project_root: str | Path | None) -> bool:
    """Return whether the configured project root targets the Pi acceptance tree."""

    try:
        resolved = Path(project_root or ".").expanduser().resolve(strict=False)
    except OSError:
        return False
    return resolved == _PI_RUNTIME_ROOT or _PI_RUNTIME_ROOT in resolved.parents


def _is_loopback_host(host: str) -> bool:
    """Return whether ``host`` already points to loopback."""

    normalized = str(host or "").strip().lower().strip("[]")
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _loopback_port_reachable(port: int, *, timeout_s: float) -> bool:
    """Return whether ``127.0.0.1:port`` accepts one TCP connection right now."""

    if port <= 0:
        return False
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=max(0.05, float(timeout_s))):
            return True
    except OSError:
        return False


def _replace_target_host(parsed: SplitResult, *, host: str, scheme: str, port: int) -> SplitResult:
    """Return ``parsed`` with a replaced scheme/netloc for the local bridge."""

    return parsed._replace(
        scheme=scheme,
        netloc=f"{host}:{port}",
    )


__all__ = [
    "LocalOrchestratorBridgeTarget",
    "resolve_local_orchestrator_probe_target",
]
