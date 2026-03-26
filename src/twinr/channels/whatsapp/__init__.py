"""Expose the Baileys-backed WhatsApp channel loop for Twinr."""

from .config import WhatsAppChannelConfig
from .history_import import WhatsAppHistoryImportQueue, WhatsAppHistoryImportRequest, WhatsAppHistoryImportResult
from .loop import TwinrWhatsAppChannelLoop
from .outbound import WhatsAppOutboundQueue, WhatsAppOutboundRequest, WhatsAppOutboundResult
from .policy import WhatsAppMessagePolicy, WhatsAppPolicyDecision

__all__ = [
    "WhatsAppHistoryImportQueue",
    "WhatsAppHistoryImportRequest",
    "WhatsAppHistoryImportResult",
    "TwinrWhatsAppChannelLoop",
    "WhatsAppChannelConfig",
    "WhatsAppOutboundQueue",
    "WhatsAppOutboundRequest",
    "WhatsAppOutboundResult",
    "WhatsAppMessagePolicy",
    "WhatsAppPolicyDecision",
]
