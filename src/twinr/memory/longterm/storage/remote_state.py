"""Compatibility wrapper for the split remote long-term snapshot store."""

from __future__ import annotations

from ._remote_state.base import LongTermRemoteStateStoreImpl
from ._remote_state.shared import (
    LongTermRemoteFetchAttempt,
    LongTermRemoteReadFailedError,
    LongTermRemoteSnapshotProbe,
    LongTermRemoteStatus,
    LongTermRemoteUnavailableError,
    remote_snapshot_document_hints_path,
)

for _export in (
    LongTermRemoteFetchAttempt,
    LongTermRemoteReadFailedError,
    LongTermRemoteSnapshotProbe,
    LongTermRemoteStatus,
    LongTermRemoteUnavailableError,
):
    _export.__module__ = __name__
remote_snapshot_document_hints_path.__module__ = __name__


class LongTermRemoteStateStore(LongTermRemoteStateStoreImpl):
    """Backward-compatible module-path wrapper for the split implementation."""


__all__ = [
    "LongTermRemoteFetchAttempt",
    "LongTermRemoteReadFailedError",
    "LongTermRemoteSnapshotProbe",
    "LongTermRemoteStateStore",
    "LongTermRemoteStatus",
    "LongTermRemoteUnavailableError",
    "remote_snapshot_document_hints_path",
]
