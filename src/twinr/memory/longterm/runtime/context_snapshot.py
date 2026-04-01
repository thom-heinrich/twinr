"""Capture one built long-term prompt context for operator-facing inspection.

The live runtime already assembles full long-term provider/tool context for the
actual turn. Operator surfaces such as Conversation Lab should be able to reuse
that exact built context instead of launching a second independent remote
search just to render a debug panel.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinr.memory.longterm.core.models import LongTermMemoryContext
from twinr.memory.longterm.runtime.prepared_context import PreparedContextProfile
from twinr.memory.query_normalization import LongTermQueryProfile


@dataclass(frozen=True, slots=True)
class LongTermContextSnapshot:
    """Describe one built provider/tool long-term context snapshot."""

    profile: PreparedContextProfile
    query_profile: LongTermQueryProfile
    context: LongTermMemoryContext
    source: str


__all__ = ["LongTermContextSnapshot"]
