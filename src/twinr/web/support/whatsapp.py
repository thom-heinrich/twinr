"""Compatibility re-export for WhatsApp pairing helpers used by the web wizard."""

from twinr.channels.whatsapp import pairing as _pairing
from twinr.channels.whatsapp.pairing import (
    WhatsAppPairingCoordinator,
    WhatsAppRuntimeProbe,
    canonicalize_whatsapp_allow_from,
    normalize_project_relative_directory,
    probe_whatsapp_runtime,
)

subprocess = _pairing.subprocess

__all__ = [
    "WhatsAppPairingCoordinator",
    "WhatsAppRuntimeProbe",
    "canonicalize_whatsapp_allow_from",
    "normalize_project_relative_directory",
    "probe_whatsapp_runtime",
    "subprocess",
]
