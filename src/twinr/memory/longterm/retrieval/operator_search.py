"""Build a read-only operator search view over Twinr long-term memory.

This module mirrors the retrieval stack used for live recall, but keeps the
web/operator path explicitly read-only: no background writers, no mutation
helpers, and no runtime orchestration. It exists so the web portal can inspect
real long-term memory hits without constructing the full runtime service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from twinr.agent.base_agent import TwinrConfig
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.context_store import PersistentMemoryEntry, PromptContextStore
from twinr.memory.longterm.core.models import (
    LongTermConflictQueueItemV1,
    LongTermMemoryObjectV1,
    LongTermMidtermPacketV1,
)
from twinr.memory.longterm.proactive.state import LongTermProactiveStateStore
from twinr.memory.longterm.reasoning.conversation_recall import query_has_conversation_recap_semantics
from twinr.memory.longterm.reasoning.conflicts import LongTermConflictResolver
from twinr.memory.longterm.retrieval.adaptive_policy import LongTermAdaptivePolicyBuilder
from twinr.memory.longterm.retrieval.retriever import LongTermRetriever
from twinr.memory.longterm.retrieval.subtext import LongTermSubtextBuilder, LongTermSubtextCompiler
from twinr.memory.longterm.storage.midterm_store import LongTermMidtermStore
from twinr.memory.longterm.storage.store import LongTermStructuredStore
from twinr.memory.query_normalization import LongTermQueryProfile, LongTermQueryRewriter
from twinr.text_utils import collapse_whitespace

_DEFAULT_RESULT_LIMIT = 6
_MAX_QUERY_CHARS = 240


def _normalize_query_text(value: object | None) -> str:
    """Collapse an operator search query into one bounded line."""

    text = collapse_whitespace(str(value or "")).strip()
    if not text:
        return ""
    if len(text) <= _MAX_QUERY_CHARS:
        return text
    return text[: _MAX_QUERY_CHARS - 1].rstrip() + "…"


def _coerce_limit(limit: int) -> int:
    """Clamp result limits to a small operator-safe range."""

    try:
        normalized = int(limit)
    except (TypeError, ValueError):
        normalized = _DEFAULT_RESULT_LIMIT
    return max(1, min(normalized, 12))


@dataclass(frozen=True, slots=True)
class LongTermOperatorSearchResult:
    """Capture one operator-facing long-term memory search result."""

    query_text: str
    query_profile: LongTermQueryProfile
    durable_objects: tuple[LongTermMemoryObjectV1, ...] = ()
    episodic_entries: tuple[PersistentMemoryEntry, ...] = ()
    midterm_packets: tuple[LongTermMidtermPacketV1, ...] = ()
    conflict_queue: tuple[LongTermConflictQueueItemV1, ...] = ()
    graph_context: str | None = None

    @property
    def total_hits(self) -> int:
        """Return the total number of retrieved items across all sections."""

        return (
            len(self.durable_objects)
            + len(self.episodic_entries)
            + len(self.midterm_packets)
            + len(self.conflict_queue)
        )


@dataclass(slots=True)
class LongTermOperatorSearch:
    """Run read-only operator searches against the real long-term stores."""

    query_rewriter: LongTermQueryRewriter
    retriever: LongTermRetriever
    graph_store: TwinrPersonalGraphStore
    object_store: LongTermStructuredStore
    midterm_store: LongTermMidtermStore
    result_limit: int = _DEFAULT_RESULT_LIMIT

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        result_limit: int = _DEFAULT_RESULT_LIMIT,
    ) -> "LongTermOperatorSearch":
        """Build a read-only operator search stack from config."""

        prompt_context_store = PromptContextStore.from_config(config)
        graph_store = TwinrPersonalGraphStore.from_config(config)
        object_store = LongTermStructuredStore.from_config(config)
        midterm_store = LongTermMidtermStore.from_config(config)
        subtext_builder = LongTermSubtextBuilder(
            config=config,
            graph_store=graph_store,
            compiler=LongTermSubtextCompiler.from_config(config),
        )
        retriever = LongTermRetriever(
            config=config,
            prompt_context_store=prompt_context_store,
            graph_store=graph_store,
            object_store=object_store,
            midterm_store=midterm_store,
            conflict_resolver=LongTermConflictResolver(),
            subtext_builder=subtext_builder,
            adaptive_policy_builder=LongTermAdaptivePolicyBuilder(
                proactive_state_store=LongTermProactiveStateStore.from_config(config),
            ),
        )
        return cls(
            query_rewriter=LongTermQueryRewriter.from_config(config),
            retriever=retriever,
            graph_store=graph_store,
            object_store=object_store,
            midterm_store=midterm_store,
            result_limit=_coerce_limit(result_limit),
        )

    def search(self, query_text: str | None) -> LongTermOperatorSearchResult:
        """Search the real long-term retrieval stack for one operator query."""

        normalized_query = _normalize_query_text(query_text)
        is_conversation_recap = query_has_conversation_recap_semantics(normalized_query)
        query_profile = (
            LongTermQueryProfile.from_text(normalized_query)
            if is_conversation_recap
            else self.query_rewriter.profile(normalized_query)
        )
        retrieval_text = query_profile.retrieval_text
        if not retrieval_text:
            return LongTermOperatorSearchResult(
                query_text=normalized_query,
                query_profile=query_profile,
            )

        episodic_entries = tuple(
            entry
            for item in self.object_store.select_relevant_episodic_objects(
                query_text=retrieval_text,
                limit=self.result_limit,
                fallback_limit=0,
                require_query_match=False,
            )
            if (entry := self.retriever._episodic_entry_from_object(item)) is not None
        )
        if is_conversation_recap:
            return LongTermOperatorSearchResult(
                query_text=normalized_query,
                query_profile=query_profile,
                episodic_entries=episodic_entries,
            )
        durable_objects = tuple(
            self.object_store.select_relevant_objects(
                query_text=retrieval_text,
                limit=self.result_limit,
            )
        )
        midterm_packets = tuple(
            self.midterm_store.select_relevant_packets(
                retrieval_text,
                limit=self.result_limit,
            )
        )
        conflict_queue = tuple(
            self.retriever.select_conflict_queue(
                query=query_profile,
                limit=self.result_limit,
            )
        )
        graph_context = cast(str | None, self.graph_store.build_prompt_context(retrieval_text))
        return LongTermOperatorSearchResult(
            query_text=normalized_query,
            query_profile=query_profile,
            durable_objects=durable_objects,
            episodic_entries=episodic_entries,
            midterm_packets=midterm_packets,
            conflict_queue=conflict_queue,
            graph_context=graph_context,
        )


def run_long_term_operator_search(
    config: TwinrConfig,
    query_text: str | None,
    *,
    result_limit: int = _DEFAULT_RESULT_LIMIT,
) -> LongTermOperatorSearchResult:
    """Execute one read-only operator memory search from config."""

    search = LongTermOperatorSearch.from_config(config, result_limit=result_limit)
    return search.search(query_text)
