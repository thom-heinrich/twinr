"""Parse and validate the WhatsApp channel settings from ``TwinrConfig``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig

from .node_runtime import resolve_whatsapp_node_binary


def normalize_whatsapp_digits(value: str) -> str:
    """Extract the stable numeric user component from a phone number or JID."""

    if not isinstance(value, str):
        raise TypeError("WhatsApp identity must be a string")
    head = value.strip().split("@", 1)[0]
    digits = "".join(character for character in head if character.isdigit())
    if not digits:
        raise ValueError(f"WhatsApp identity has no digits: {value!r}")
    return digits


def normalize_whatsapp_jid(value: str) -> str:
    """Normalize a phone number or direct-message JID to ``digits@s.whatsapp.net``."""

    return f"{normalize_whatsapp_digits(value)}@s.whatsapp.net"


def _resolve_project_path(project_root: Path, configured_path: str) -> Path:
    """Resolve one project-relative WhatsApp path without requiring existence."""

    candidate = Path(str(configured_path or "")).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve(strict=False)


@dataclass(frozen=True, slots=True)
class WhatsAppChannelConfig:
    """Store the focused runtime settings needed by the WhatsApp transport."""

    allow_from: str
    allow_from_jid: str
    auth_dir: Path
    worker_root: Path
    node_binary: str
    groups_enabled: bool
    self_chat_mode: bool
    reconnect_base_delay_s: float
    reconnect_max_delay_s: float
    send_timeout_s: float
    sent_cache_ttl_s: float
    sent_cache_max_entries: int

    @classmethod
    def from_twinr_config(cls, config: TwinrConfig) -> "WhatsAppChannelConfig":
        """Build the focused WhatsApp settings snapshot from ``TwinrConfig``."""

        project_root = Path(str(getattr(config, "project_root", "") or ".")).expanduser().resolve(strict=False)
        raw_allow_from = str(getattr(config, "whatsapp_allow_from", "") or "").strip()
        if not raw_allow_from:
            raise ValueError("TWINR_WHATSAPP_ALLOW_FROM must be configured for the WhatsApp channel")

        auth_dir = _resolve_project_path(project_root, str(getattr(config, "whatsapp_auth_dir", "")))
        worker_root = _resolve_project_path(project_root, str(getattr(config, "whatsapp_worker_root", "")))
        node_binary = resolve_whatsapp_node_binary(
            project_root,
            str(getattr(config, "whatsapp_node_binary", "node") or "node"),
        )
        if not worker_root.exists() or not worker_root.is_dir():
            raise ValueError(f"WhatsApp worker root does not exist: {worker_root}")
        if not (worker_root / "package.json").exists():
            raise ValueError(f"WhatsApp worker package.json is missing: {worker_root / 'package.json'}")

        return cls(
            allow_from=raw_allow_from,
            allow_from_jid=normalize_whatsapp_jid(raw_allow_from),
            auth_dir=auth_dir,
            worker_root=worker_root,
            node_binary=str(node_binary).strip() or "node",
            groups_enabled=bool(getattr(config, "whatsapp_groups_enabled", False)),
            self_chat_mode=bool(getattr(config, "whatsapp_self_chat_mode", False)),
            reconnect_base_delay_s=max(0.1, float(getattr(config, "whatsapp_reconnect_base_delay_s", 2.0))),
            reconnect_max_delay_s=max(
                float(getattr(config, "whatsapp_reconnect_base_delay_s", 2.0)),
                float(getattr(config, "whatsapp_reconnect_max_delay_s", 30.0)),
            ),
            send_timeout_s=max(1.0, float(getattr(config, "whatsapp_send_timeout_s", 20.0))),
            sent_cache_ttl_s=max(1.0, float(getattr(config, "whatsapp_sent_cache_ttl_s", 180.0))),
            sent_cache_max_entries=max(16, int(getattr(config, "whatsapp_sent_cache_max_entries", 256))),
        )
