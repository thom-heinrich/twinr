from __future__ import annotations

from dataclasses import dataclass
import json

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.context_store import ManagedContextEntry, PersistentMemoryEntry, PromptContextStore
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.query_normalization import LongTermQueryProfile, LongTermQueryRewriter, tokenize_retrieval_text
from twinr.memory.longterm.consolidator import LongTermMemoryConsolidator
from twinr.memory.longterm.extract import LongTermTurnExtractor
from twinr.memory.longterm.models import (
    LongTermConsolidationResultV1,
    LongTermConversationTurn,
    LongTermEnqueueResult,
    LongTermMemoryContext,
)
from twinr.memory.longterm.subtext import LongTermSubtextBuilder
from twinr.memory.longterm.truth import LongTermTruthMaintainer
from twinr.memory.longterm.worker import AsyncLongTermMemoryWriter


def _normalize_text(value: str, *, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


@dataclass(slots=True)
class LongTermMemoryService:
    config: TwinrConfig
    prompt_context_store: PromptContextStore
    graph_store: TwinrPersonalGraphStore
    subtext_builder: LongTermSubtextBuilder
    query_rewriter: LongTermQueryRewriter
    extractor: LongTermTurnExtractor
    truth_maintainer: LongTermTruthMaintainer
    consolidator: LongTermMemoryConsolidator
    writer: AsyncLongTermMemoryWriter | None = None

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        graph_store: TwinrPersonalGraphStore | None = None,
        prompt_context_store: PromptContextStore | None = None,
    ) -> "LongTermMemoryService":
        store = prompt_context_store or PromptContextStore.from_config(config)
        graph = graph_store or TwinrPersonalGraphStore.from_config(config)
        truth_maintainer = LongTermTruthMaintainer()
        writer: AsyncLongTermMemoryWriter | None = None
        if config.long_term_memory_enabled and config.long_term_memory_background_store_turns:
            writer = AsyncLongTermMemoryWriter(
                write_callback=lambda item: cls._persist_episodic_turn(store=store, item=item),
                max_queue_size=config.long_term_memory_write_queue_size,
            )
        return cls(
            config=config,
            prompt_context_store=store,
            graph_store=graph,
            subtext_builder=LongTermSubtextBuilder(config=config, graph_store=graph),
            query_rewriter=LongTermQueryRewriter.from_config(config),
            extractor=LongTermTurnExtractor(timezone_name=config.local_timezone_name),
            truth_maintainer=truth_maintainer,
            consolidator=LongTermMemoryConsolidator(truth_maintainer=truth_maintainer),
            writer=writer,
        )

    def build_provider_context(self, query_text: str | None) -> LongTermMemoryContext:
        query = self.query_rewriter.profile(query_text)
        episodic_entries = self._select_episodic_entries(query)
        graph_context = self.graph_store.build_prompt_context(query.retrieval_text)
        episodic_context = self._render_episodic_context(episodic_entries)
        subtext_context = self.subtext_builder.build(
            query_text=query_text,
            retrieval_query_text=query.retrieval_text,
            episodic_entries=self._select_episodic_entries(
                query,
                fallback_limit=0,
                require_query_match=True,
            ),
        )
        return LongTermMemoryContext(
            subtext_context=subtext_context,
            episodic_context=episodic_context,
            graph_context=graph_context,
        )

    def enqueue_conversation_turn(
        self,
        *,
        transcript: str,
        response: str,
        source: str = "conversation",
    ) -> LongTermEnqueueResult | None:
        clean_transcript = _normalize_text(transcript, limit=320)
        clean_response = _normalize_text(response, limit=320)
        if not clean_transcript or not clean_response:
            return None
        if self.writer is None:
            return None
        return self.writer.enqueue(
            LongTermConversationTurn(
                transcript=clean_transcript,
                response=clean_response,
                source=source,
            )
        )

    def analyze_conversation_turn(
        self,
        *,
        transcript: str,
        response: str,
    ) -> LongTermConsolidationResultV1:
        extraction = self.extractor.extract_conversation_turn(
            transcript=transcript,
            response=response,
        )
        return self.consolidator.consolidate(extraction=extraction)

    def store_explicit_memory(
        self,
        *,
        kind: str,
        summary: str,
        details: str | None = None,
    ) -> PersistentMemoryEntry:
        return self.prompt_context_store.memory_store.remember(
            kind=kind,
            summary=summary,
            details=details,
        )

    def update_user_profile(
        self,
        *,
        category: str,
        instruction: str,
    ) -> ManagedContextEntry:
        return self.prompt_context_store.user_store.upsert(
            category=category,
            instruction=instruction,
        )

    def update_personality(
        self,
        *,
        category: str,
        instruction: str,
    ) -> ManagedContextEntry:
        return self.prompt_context_store.personality_store.upsert(
            category=category,
            instruction=instruction,
        )

    def flush(self, *, timeout_s: float = 2.0) -> bool:
        if self.writer is None:
            return True
        return self.writer.flush(timeout_s=timeout_s)

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        if self.writer is None:
            return
        self.writer.shutdown(timeout_s=timeout_s)

    @staticmethod
    def _persist_episodic_turn(*, store: PromptContextStore, item: LongTermConversationTurn) -> PersistentMemoryEntry:
        quoted_transcript = json.dumps(item.transcript, ensure_ascii=False)
        quoted_response = json.dumps(item.response, ensure_ascii=False)
        return store.memory_store.remember(
            kind="episodic_turn",
            summary=f"Conversation about {quoted_transcript}",
            details=f"User said: {quoted_transcript} Twinr answered: {quoted_response}",
        )

    def _select_episodic_entries(
        self,
        query: LongTermQueryProfile | str | None,
        *,
        fallback_limit: int = 2,
        require_query_match: bool = False,
    ) -> list[PersistentMemoryEntry]:
        entries = [
            entry
            for entry in self.prompt_context_store.memory_store.load_entries()
            if entry.kind == "episodic_turn"
        ]
        if not entries:
            return []
        limit = max(1, self.config.long_term_memory_recall_limit)
        query_profile = query if isinstance(query, LongTermQueryProfile) else LongTermQueryProfile.from_text(query)
        query_tokens = set(tokenize_retrieval_text(query_profile.retrieval_text))
        if query_tokens:
            ranked: list[tuple[int, PersistentMemoryEntry]] = []
            for entry in entries:
                score = 0
                score += len(query_tokens & set(tokenize_retrieval_text(entry.summary))) * 3
                score += len(query_tokens & set(tokenize_retrieval_text(entry.details))) * 2
                if score > 0:
                    ranked.append((score, entry))
            ranked.sort(key=lambda item: (-item[0], str(item[1].updated_at)), reverse=False)
            selected = [entry for _score, entry in ranked[:limit]]
            if not selected and not require_query_match and fallback_limit > 0:
                selected = list(entries[: min(limit, fallback_limit)])
        else:
            if require_query_match:
                return []
            selected = list(entries[:limit])
        return selected

    def _render_episodic_context(self, entries: list[PersistentMemoryEntry]) -> str | None:
        if not entries:
            return None
        payload = {
            "schema": "twinr_long_term_episodic_context_v1",
            "recent_episodes": [
                {
                    "summary": entry.summary,
                    "details": entry.details,
                    "created_at": entry.created_at.isoformat(),
                }
                for entry in entries
            ],
        }
        return (
            "Structured long-term episodic memory for this turn. Internal memory is canonical English. "
            "Use only when clearly relevant, and weave it in gently instead of reciting it. "
            "Do not announce prior memory with phrases like earlier, before, last time, or neulich unless the user directly asks to recall prior conversation. "
            "Do not invent personal details that are not grounded in the user's request or this memory context.\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
