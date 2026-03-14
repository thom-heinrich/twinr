from __future__ import annotations

from dataclasses import dataclass
import json
from json import JSONDecoder
from typing import Sequence

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.context_store import PersistentMemoryEntry


@dataclass(frozen=True, slots=True)
class LongTermSubtextBuilder:
    config: TwinrConfig
    graph_store: TwinrPersonalGraphStore

    def build(
        self,
        *,
        query_text: str | None,
        retrieval_query_text: str | None,
        episodic_entries: Sequence[PersistentMemoryEntry],
    ) -> str | None:
        graph_payload = self.graph_store.build_subtext_payload(retrieval_query_text)
        recent_threads = self._episodic_threads(episodic_entries)
        if not graph_payload and not recent_threads:
            return None
        payload: dict[str, object] = {
            "schema": "twinr_silent_personalization_context_v1",
            "query": " ".join(str(query_text or "").split()).strip(),
            "principles": [
                "Let relevant familiarity shape priorities, suggestions, and tone without announcing hidden memory.",
                "Use remembered preferences and ongoing situations only when they genuinely help the current reply.",
                "Prefer natural continuity over overt memory announcements unless direct recall is the point.",
                "Do not force a personal detail into every answer.",
                "Do not invent pets, routines, relationships, or other personal details that are not in the request or memory context.",
                "Do not say earlier, before, last time, neulich, or similar unless the user explicitly asks about past conversation.",
            ],
        }
        if graph_payload:
            payload["graph_cues"] = graph_payload
        if recent_threads:
            payload["recent_threads"] = recent_threads
        return (
            "Silent personalization background for this turn. Internal memory is canonical English. "
            "Use it as conversational subtext, not as a script or a fact dump. "
            "Keep it implicit unless explicit recall is necessary.\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def _episodic_threads(self, entries: Sequence[PersistentMemoryEntry]) -> list[dict[str, str]]:
        threads: list[dict[str, str]] = []
        for entry in entries[: max(1, self.config.long_term_memory_recall_limit)]:
            user_text, assistant_text = self._extract_turn(entry)
            if not user_text:
                continue
            thread = {
                "topic": user_text,
                "guidance": (
                    "If the current conversation naturally continues this thread, let it influence phrasing or "
                    "suggestions without explicitly citing hidden memory or saying earlier, before, last time, or neulich."
                ),
            }
            if assistant_text:
                thread["last_direction"] = assistant_text
            threads.append(thread)
        return threads

    def _extract_turn(self, entry: PersistentMemoryEntry) -> tuple[str | None, str | None]:
        summary_text = self._decode_embedded_json_string(entry.summary, prefix="Conversation about ")
        details = entry.details or ""
        user_text = self._decode_embedded_json_string(details, prefix="User said: ")
        assistant_text = self._decode_embedded_json_string(
            details,
            prefix="Twinr answered: ",
            start_at=details.find("Twinr answered: "),
        )
        return user_text or summary_text, assistant_text

    def _decode_embedded_json_string(
        self,
        value: str,
        *,
        prefix: str,
        start_at: int = 0,
    ) -> str | None:
        index = value.find(prefix, max(0, start_at))
        if index < 0:
            return None
        decoder = JSONDecoder()
        start = index + len(prefix)
        try:
            decoded, _end = decoder.raw_decode(value[start:])
        except json.JSONDecodeError:
            return None
        if not isinstance(decoded, str):
            return None
        text = " ".join(decoded.split()).strip()
        return text or None


__all__ = ["LongTermSubtextBuilder"]
