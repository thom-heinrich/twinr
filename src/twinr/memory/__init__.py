"""Expose Twinr's public memory package surface.

Import from this package when callers need the supported short-term, prompt-
context, reminder, graph-memory, or long-term memory types without binding to
the internal submodule layout.

The package resolves exports lazily so importing one narrow memory type does
not eagerly pull the full long-term/runtime stack into the process. That keeps
Pi deploy/runtime import attestation focused and avoids import-time coupling
between the prompt-context stores and the broader memory subsystem.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "ConversationTurn": ("twinr.memory.on_device", "ConversationTurn"),
    "ChonkyDBAuthInfo": ("twinr.memory.chonkydb", "ChonkyDBAuthInfo"),
    "ChonkyDBBulkRecordRequest": ("twinr.memory.chonkydb", "ChonkyDBBulkRecordRequest"),
    "ChonkyDBClient": ("twinr.memory.chonkydb", "ChonkyDBClient"),
    "ChonkyDBConnectionConfig": ("twinr.memory.chonkydb", "ChonkyDBConnectionConfig"),
    "ChonkyDBError": ("twinr.memory.chonkydb", "ChonkyDBError"),
    "ChonkyDBGraphAddEdgeRequest": ("twinr.memory.chonkydb", "ChonkyDBGraphAddEdgeRequest"),
    "ChonkyDBGraphAddEdgeSmartRequest": ("twinr.memory.chonkydb", "ChonkyDBGraphAddEdgeSmartRequest"),
    "ChonkyDBGraphNeighborsRequest": ("twinr.memory.chonkydb", "ChonkyDBGraphNeighborsRequest"),
    "ChonkyDBGraphPathRequest": ("twinr.memory.chonkydb", "ChonkyDBGraphPathRequest"),
    "ChonkyDBGraphPatternsRequest": ("twinr.memory.chonkydb", "ChonkyDBGraphPatternsRequest"),
    "ChonkyDBGraphStoreManyEdge": ("twinr.memory.chonkydb", "ChonkyDBGraphStoreManyEdge"),
    "ChonkyDBGraphStoreManyNode": ("twinr.memory.chonkydb", "ChonkyDBGraphStoreManyNode"),
    "ChonkyDBGraphStoreManyRequest": ("twinr.memory.chonkydb", "ChonkyDBGraphStoreManyRequest"),
    "ChonkyDBInstanceInfo": ("twinr.memory.chonkydb", "ChonkyDBInstanceInfo"),
    "ChonkyDBRecordItem": ("twinr.memory.chonkydb", "ChonkyDBRecordItem"),
    "ChonkyDBRecordListResponse": ("twinr.memory.chonkydb", "ChonkyDBRecordListResponse"),
    "ChonkyDBRecordRequest": ("twinr.memory.chonkydb", "ChonkyDBRecordRequest"),
    "ChonkyDBRecordSummary": ("twinr.memory.chonkydb", "ChonkyDBRecordSummary"),
    "ChonkyDBRetrieveHit": ("twinr.memory.chonkydb", "ChonkyDBRetrieveHit"),
    "ChonkyDBRetrieveRequest": ("twinr.memory.chonkydb", "ChonkyDBRetrieveRequest"),
    "ChonkyDBRetrieveResponse": ("twinr.memory.chonkydb", "ChonkyDBRetrieveResponse"),
    "ChonkyDBTopKRecordHit": ("twinr.memory.chonkydb", "ChonkyDBTopKRecordHit"),
    "ChonkyDBTopKRecordsRequest": ("twinr.memory.chonkydb", "ChonkyDBTopKRecordsRequest"),
    "ChonkyDBTopKRecordsResponse": ("twinr.memory.chonkydb", "ChonkyDBTopKRecordsResponse"),
    "ManagedContextEntry": ("twinr.memory.context_store", "ManagedContextEntry"),
    "ManagedContextFileStore": ("twinr.memory.context_store", "ManagedContextFileStore"),
    "LongTermConversationTurn": ("twinr.memory.longterm", "LongTermConversationTurn"),
    "LongTermEnqueueResult": ("twinr.memory.longterm", "LongTermEnqueueResult"),
    "LongTermMemoryContext": ("twinr.memory.longterm", "LongTermMemoryContext"),
    "LongTermMemoryService": ("twinr.memory.longterm", "LongTermMemoryService"),
    "LongTermMultimodalEvidence": ("twinr.memory.longterm", "LongTermMultimodalEvidence"),
    "LongTermMultimodalExtractor": ("twinr.memory.longterm", "LongTermMultimodalExtractor"),
    "LongTermSubtextBuilder": ("twinr.memory.longterm", "LongTermSubtextBuilder"),
    "AsyncLongTermMemoryWriter": ("twinr.memory.longterm", "AsyncLongTermMemoryWriter"),
    "AsyncLongTermMultimodalWriter": ("twinr.memory.longterm", "AsyncLongTermMultimodalWriter"),
    "MemoryLedgerItem": ("twinr.memory.on_device", "MemoryLedgerItem"),
    "PersistentMemoryEntry": ("twinr.memory.context_store", "PersistentMemoryEntry"),
    "PersistentMemoryMarkdownStore": ("twinr.memory.context_store", "PersistentMemoryMarkdownStore"),
    "MemoryState": ("twinr.memory.on_device", "MemoryState"),
    "OnDeviceMemory": ("twinr.memory.on_device", "OnDeviceMemory"),
    "PromptContextStore": ("twinr.memory.context_store", "PromptContextStore"),
    "ReminderEntry": ("twinr.memory.reminders", "ReminderEntry"),
    "ReminderStore": ("twinr.memory.reminders", "ReminderStore"),
    "SearchMemoryEntry": ("twinr.memory.on_device", "SearchMemoryEntry"),
    "TwinrGraphContactOption": ("twinr.memory.chonkydb", "TwinrGraphContactOption"),
    "TwinrGraphLookupResult": ("twinr.memory.chonkydb", "TwinrGraphLookupResult"),
    "TWINR_GRAPH_ALLOWED_EDGE_TYPES": ("twinr.memory.chonkydb", "TWINR_GRAPH_ALLOWED_EDGE_TYPES"),
    "TWINR_GRAPH_EDGE_TYPES_BY_NAMESPACE": ("twinr.memory.chonkydb", "TWINR_GRAPH_EDGE_TYPES_BY_NAMESPACE"),
    "TwinrGraphWriteResult": ("twinr.memory.chonkydb", "TwinrGraphWriteResult"),
    "TWINR_GRAPH_SCHEMA_NAME": ("twinr.memory.chonkydb", "TWINR_GRAPH_SCHEMA_NAME"),
    "TWINR_GRAPH_SCHEMA_VERSION": ("twinr.memory.chonkydb", "TWINR_GRAPH_SCHEMA_VERSION"),
    "TwinrGraphDocumentV1": ("twinr.memory.chonkydb", "TwinrGraphDocumentV1"),
    "TwinrGraphEdgeV1": ("twinr.memory.chonkydb", "TwinrGraphEdgeV1"),
    "TwinrGraphNodeV1": ("twinr.memory.chonkydb", "TwinrGraphNodeV1"),
    "TwinrPersonalGraphStore": ("twinr.memory.chonkydb", "TwinrPersonalGraphStore"),
    "chonkydb_data_path": ("twinr.memory.chonkydb", "chonkydb_data_path"),
    "format_due_label": ("twinr.memory.reminders", "format_due_label"),
    "graph_edge_namespace": ("twinr.memory.chonkydb", "graph_edge_namespace"),
    "is_allowed_graph_edge_type": ("twinr.memory.chonkydb", "is_allowed_graph_edge_type"),
    "now_in_timezone": ("twinr.memory.reminders", "now_in_timezone"),
    "parse_due_at": ("twinr.memory.reminders", "parse_due_at"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Resolve one public memory export on demand."""

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy exports in interactive and reflective contexts."""

    return sorted(set(globals()) | set(__all__))
