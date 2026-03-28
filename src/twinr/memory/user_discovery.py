"""Expose the stable public guided user-discovery API over the decomposed stack.

This module intentionally stays thin. The production implementation now lives
under ``twinr.memory.user_discovery_impl`` so the long-lived import path
remains stable while the discovery service is split by responsibility.
"""

from __future__ import annotations

##REFACTOR: 2026-03-27##

from . import user_discovery_impl as _impl

UserDiscoveryCommitCallbacks = _impl.UserDiscoveryCommitCallbacks
UserDiscoveryFact = _impl.UserDiscoveryFact
UserDiscoveryInvite = _impl.UserDiscoveryInvite
UserDiscoveryMemoryRoute = _impl.UserDiscoveryMemoryRoute
UserDiscoveryResult = _impl.UserDiscoveryResult
UserDiscoveryReviewItem = _impl.UserDiscoveryReviewItem
UserDiscoveryService = _impl.UserDiscoveryService
UserDiscoveryState = _impl.UserDiscoveryState
UserDiscoveryStateStore = _impl.UserDiscoveryStateStore
UserDiscoveryStoredFact = _impl.UserDiscoveryStoredFact
UserDiscoveryTopicDefinition = _impl.UserDiscoveryTopicDefinition
UserDiscoveryTopicState = _impl.UserDiscoveryTopicState

_PUBLIC_EXPORTS = {
    "UserDiscoveryCommitCallbacks",
    "UserDiscoveryFact",
    "UserDiscoveryInvite",
    "UserDiscoveryMemoryRoute",
    "UserDiscoveryResult",
    "UserDiscoveryReviewItem",
    "UserDiscoveryService",
    "UserDiscoveryState",
    "UserDiscoveryStateStore",
    "UserDiscoveryStoredFact",
    "UserDiscoveryTopicDefinition",
    "UserDiscoveryTopicState",
}
_COMPAT_EXPORTS = tuple(name for name in _impl.__all__ if name not in _PUBLIC_EXPORTS)

globals().update({name: getattr(_impl, name) for name in _COMPAT_EXPORTS})

__all__ = [
    "UserDiscoveryCommitCallbacks",
    "UserDiscoveryFact",
    "UserDiscoveryInvite",
    "UserDiscoveryMemoryRoute",
    "UserDiscoveryResult",
    "UserDiscoveryReviewItem",
    "UserDiscoveryService",
    "UserDiscoveryState",
    "UserDiscoveryStateStore",
    "UserDiscoveryStoredFact",
    "UserDiscoveryTopicState",
]

for _name in tuple(_PUBLIC_EXPORTS) + _COMPAT_EXPORTS:
    _exported = globals()[_name]
    try:
        _exported.__module__ = __name__
    except (AttributeError, TypeError):
        continue
