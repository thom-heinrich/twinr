from __future__ import annotations

from importlib import import_module

__all__ = [
    "AdaptiveListeningWindow",
    "AdaptiveTimingProfile",
    "AdaptiveTimingStore",
    "AgentToolCall",
    "AgentTextProvider",
    "AgentToolResult",
    "CombinedSpeechAgentProvider",
    "CompositeSpeechAgentProvider",
    "InvalidTransitionError",
    "LongTermMemoryService",
    "ManagedContextEntry",
    "ManagedContextFileStore",
    "PersistentMemoryEntry",
    "PersistentMemoryMarkdownStore",
    "PromptContextStore",
    "ProviderBundle",
    "ReminderEntry",
    "ReminderStore",
    "RuntimeSnapshot",
    "RuntimeSnapshotStore",
    "SpeechToTextProvider",
    "StreamingSpeechEndpointEvent",
    "StreamingSpeechToTextProvider",
    "StreamingSpeechToTextSession",
    "StreamingTranscriptionResult",
    "StreamingTurnController",
    "TextToSpeechProvider",
    "ToolCallingAgentProvider",
    "ToolCallingTurnDecisionEvaluator",
    "ToolCallingTurnResponse",
    "TurnDecision",
    "TurnEvaluationCandidate",
    "TwinrConfig",
    "TwinrEvent",
    "TwinrRuntime",
    "TwinrStateMachine",
    "TwinrStatus",
    "load_personality_instructions",
    "merge_instructions",
]

_EXPORTS = {
    "AdaptiveListeningWindow": "twinr.agent.base_agent.conversation.adaptive_timing",
    "AdaptiveTimingProfile": "twinr.agent.base_agent.conversation.adaptive_timing",
    "AdaptiveTimingStore": "twinr.agent.base_agent.conversation.adaptive_timing",
    "AgentToolCall": "twinr.agent.base_agent.contracts",
    "AgentTextProvider": "twinr.agent.base_agent.contracts",
    "AgentToolResult": "twinr.agent.base_agent.contracts",
    "CombinedSpeechAgentProvider": "twinr.agent.base_agent.contracts",
    "CompositeSpeechAgentProvider": "twinr.agent.base_agent.contracts",
    "InvalidTransitionError": "twinr.agent.base_agent.state.machine",
    "LongTermMemoryService": "twinr.memory.longterm",
    "ManagedContextEntry": "twinr.memory.context_store",
    "ManagedContextFileStore": "twinr.memory.context_store",
    "PersistentMemoryEntry": "twinr.memory.context_store",
    "PersistentMemoryMarkdownStore": "twinr.memory.context_store",
    "PromptContextStore": "twinr.memory.context_store",
    "ProviderBundle": "twinr.agent.base_agent.contracts",
    "ReminderEntry": "twinr.memory.reminders",
    "ReminderStore": "twinr.memory.reminders",
    "RuntimeSnapshot": "twinr.agent.base_agent.state.snapshot",
    "RuntimeSnapshotStore": "twinr.agent.base_agent.state.snapshot",
    "SpeechToTextProvider": "twinr.agent.base_agent.contracts",
    "StreamingSpeechEndpointEvent": "twinr.agent.base_agent.contracts",
    "StreamingSpeechToTextProvider": "twinr.agent.base_agent.contracts",
    "StreamingSpeechToTextSession": "twinr.agent.base_agent.contracts",
    "StreamingTranscriptionResult": "twinr.agent.base_agent.contracts",
    "StreamingTurnController": "twinr.agent.base_agent.conversation.turn_controller",
    "TextToSpeechProvider": "twinr.agent.base_agent.contracts",
    "ToolCallingAgentProvider": "twinr.agent.base_agent.contracts",
    "ToolCallingTurnDecisionEvaluator": "twinr.agent.base_agent.conversation.turn_controller",
    "ToolCallingTurnResponse": "twinr.agent.base_agent.contracts",
    "TurnDecision": "twinr.agent.base_agent.conversation.turn_controller",
    "TurnEvaluationCandidate": "twinr.agent.base_agent.conversation.turn_controller",
    "TwinrConfig": "twinr.agent.base_agent.config",
    "TwinrEvent": "twinr.agent.base_agent.state.machine",
    "TwinrRuntime": "twinr.agent.base_agent.runtime",
    "TwinrStateMachine": "twinr.agent.base_agent.state.machine",
    "TwinrStatus": "twinr.agent.base_agent.state.machine",
    "load_personality_instructions": "twinr.agent.base_agent.prompting.personality",
    "merge_instructions": "twinr.agent.base_agent.prompting.personality",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
