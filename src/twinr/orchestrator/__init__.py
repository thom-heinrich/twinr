"""Expose Twinr's edge-orchestrator package surface lazily.

Import commonly used text-turn and streaming-voice websocket contracts,
clients, server helpers, and local session bridges from this package root
without eagerly pulling runtime-heavy modules into unrelated imports.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ACK_ID_TO_TEXT",
    "EdgeOrchestratorServer",
    "EdgeOrchestratorSession",
    "EdgeOrchestratorVoiceSession",
    "OrchestratorAckEvent",
    "OrchestratorClientTurnResult",
    "OrchestratorToolRequest",
    "OrchestratorToolResponse",
    "OrchestratorTurnCompleteEvent",
    "OrchestratorTurnRequest",
    "OrchestratorVoiceAudioFrame",
    "OrchestratorVoiceBargeInInterruptEvent",
    "OrchestratorVoiceErrorEvent",
    "OrchestratorVoiceFollowUpClosedEvent",
    "OrchestratorVoiceHelloRequest",
    "OrchestratorVoiceReadyEvent",
    "OrchestratorVoiceRuntimeStateEvent",
    "OrchestratorVoiceTranscriptCommittedEvent",
    "OrchestratorVoiceWakeConfirmedEvent",
    "OrchestratorVoiceWebSocketClient",
    "OrchestratorWebSocketClient",
    "RemoteToolBridge",
    "ack_id_for_text",
    "ack_text_for_id",
    "create_app",
    "decode_voice_server_event",
]

_EXPORTS = {
    "ACK_ID_TO_TEXT": "twinr.orchestrator.acks",
    "EdgeOrchestratorServer": "twinr.orchestrator.server",
    "EdgeOrchestratorSession": "twinr.orchestrator.session",
    "EdgeOrchestratorVoiceSession": "twinr.orchestrator.voice_session",
    "OrchestratorAckEvent": "twinr.orchestrator.contracts",
    "OrchestratorClientTurnResult": "twinr.orchestrator.contracts",
    "OrchestratorToolRequest": "twinr.orchestrator.contracts",
    "OrchestratorToolResponse": "twinr.orchestrator.contracts",
    "OrchestratorTurnCompleteEvent": "twinr.orchestrator.contracts",
    "OrchestratorTurnRequest": "twinr.orchestrator.contracts",
    "OrchestratorVoiceAudioFrame": "twinr.orchestrator.voice_contracts",
    "OrchestratorVoiceBargeInInterruptEvent": "twinr.orchestrator.voice_contracts",
    "OrchestratorVoiceErrorEvent": "twinr.orchestrator.voice_contracts",
    "OrchestratorVoiceFollowUpClosedEvent": "twinr.orchestrator.voice_contracts",
    "OrchestratorVoiceHelloRequest": "twinr.orchestrator.voice_contracts",
    "OrchestratorVoiceReadyEvent": "twinr.orchestrator.voice_contracts",
    "OrchestratorVoiceRuntimeStateEvent": "twinr.orchestrator.voice_contracts",
    "OrchestratorVoiceTranscriptCommittedEvent": "twinr.orchestrator.voice_contracts",
    "OrchestratorVoiceWakeConfirmedEvent": "twinr.orchestrator.voice_contracts",
    "OrchestratorVoiceWebSocketClient": "twinr.orchestrator.voice_client",
    "OrchestratorWebSocketClient": "twinr.orchestrator.client",
    "RemoteToolBridge": "twinr.orchestrator.session",
    "ack_id_for_text": "twinr.orchestrator.acks",
    "ack_text_for_id": "twinr.orchestrator.acks",
    "create_app": "twinr.orchestrator.server",
    "decode_voice_server_event": "twinr.orchestrator.voice_contracts",
}


def __getattr__(name: str) -> object:
    """Resolve exported symbols lazily from their owning modules."""

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    """Return the combined static and lazy export names for introspection."""

    return sorted(set(globals()) | set(__all__))
