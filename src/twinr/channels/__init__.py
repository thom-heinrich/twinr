"""Expose the generic Twinr text-channel contract and runtime helpers.

The ``channels`` package owns long-lived external messaging transports such as
WhatsApp and the generic text-turn service they share. Keep transport-specific
session, reconnect, and policy logic inside child packages.
"""

from .contracts import ChannelInboundMessage, ChannelOutboundMessage, ChannelTransportError
from .runtime import TwinrTextChannelTurnService

__all__ = [
    "ChannelInboundMessage",
    "ChannelOutboundMessage",
    "ChannelTransportError",
    "TwinrTextChannelTurnService",
]
