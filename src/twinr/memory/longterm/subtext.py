from __future__ import annotations

from dataclasses import dataclass, replace
import json
from json import JSONDecoder
from typing import Any, Sequence

from twinr.agent.base_agent.config import TwinrConfig
from twinr.llm_json import request_structured_json_object
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.context_store import PersistentMemoryEntry
from twinr.text_utils import collapse_whitespace, sanitize_text_fragment


def _normalize_query_text(value: str | None) -> str:
    return sanitize_text_fragment(value)


def _sanitize_structured_value(value: object) -> object:
    if isinstance(value, str):
        return sanitize_text_fragment(value)
    if isinstance(value, list):
        return [_sanitize_structured_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_structured_value(item) for item in value]
    if isinstance(value, dict):
        clean: dict[str, object] = {}
        for key, item in value.items():
            clean_key = sanitize_text_fragment(str(key))
            if not clean_key:
                continue
            clean[clean_key] = _sanitize_structured_value(item)
        return clean
    return value


@dataclass(slots=True)
class LongTermSubtextCompiler:
    config: TwinrConfig
    backend: Any | None = None
    _cache: dict[str, dict[str, object]] | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LongTermSubtextCompiler":
        backend: Any | None = None
        if config.long_term_memory_subtext_compiler_enabled and config.openai_api_key:
            from twinr.providers.openai import OpenAIBackend

            backend = OpenAIBackend(
                config=replace(
                    config,
                    openai_realtime_language="en",
                    openai_reasoning_effort="low",
                ),
                base_instructions="",
            )
        return cls(config=config, backend=backend, _cache={})

    def compile(
        self,
        *,
        query_text: str | None,
        retrieval_query_text: str | None,
        graph_payload: dict[str, object] | None,
        recent_threads: Sequence[dict[str, str]],
    ) -> dict[str, object] | None:
        if self.backend is None:
            return None
        payload: dict[str, object] = {}
        if graph_payload:
            payload["graph_cues"] = graph_payload
        if recent_threads:
            payload["recent_threads"] = list(recent_threads)
        if not payload:
            return None
        cache_key = json.dumps(
            {
                "query": _normalize_query_text(query_text),
                "retrieval_query": _normalize_query_text(retrieval_query_text),
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]
        try:
            compiled = request_structured_json_object(
                self.backend,
                prompt=(
                    "Compile a silent personalization program for a senior-friendly voice assistant.\n"
                    "The program is internal only and must stay in canonical English.\n"
                    "It should quietly shape the answer without explicit memory announcements.\n"
                    "Use recent_threads as latent continuity cues, even when they do not share literal words with the current query, if they would realistically affect the user's present decision or comfort.\n"
                    "Ignore recent_threads that are not genuinely relevant to the current query.\n"
                    "When relevant memory involves a known person, encode why that person matters for the current request and what practical framing that should create.\n"
                    "If the current query directly mentions a known person, make the response plan role-grounded and practical rather than generic.\n"
                    "If the memory cues provide a person role or relation, preserve that exact role or relation text in role_or_relation instead of downgrading it to a generic label.\n"
                    "For a known person, infer short practical topics that naturally follow from the role or relation and the current query.\n"
                    "Prefer role-grounded practical framing over generic social advice.\n"
                    "It is acceptable for the final answer to mention the practical domain implied by the role or relation, as long as it does not frame this as remembered hidden memory or biography.\n"
                    "Do not answer the user. Do not invent facts. Do not add contact details unless the current query requires exact lookup.\n"
                    f"Original user query: {_normalize_query_text(query_text)}\n"
                    f"Canonical retrieval query: {_normalize_query_text(retrieval_query_text)}\n"
                    "Relevant long-term memory cues JSON:\n"
                    f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
                ),
                instructions=(
                    "Return one strict JSON object only. "
                    "Keep every field concise and canonical English. "
                    "Keep conversation_goal under 20 words. "
                    "Keep each list item short and concrete. "
                    "If a known person's role or relation is present in the cues, preserve it verbatim in role_or_relation. "
                    "Do not emit markdown, code fences, or prose outside the JSON object."
                ),
                schema_name="twinr_silent_personalization_program_v1",
                schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "use_personalization": {"type": "boolean"},
                        "conversation_goal": {"type": "string", "maxLength": 160},
                        "helpful_biases": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 120},
                            "maxItems": 2,
                        },
                        "suggested_directions": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 140},
                            "maxItems": 2,
                        },
                        "follow_up_angles": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 120},
                            "maxItems": 2,
                        },
                        "known_people": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "person": {"type": "string", "maxLength": 80},
                                    "role_or_relation": {"type": "string", "maxLength": 100},
                                    "latent_relevance": {"type": "string", "maxLength": 120},
                                    "practical_topics": {
                                        "type": "array",
                                        "items": {"type": "string", "maxLength": 80},
                                        "maxItems": 3,
                                    },
                                },
                                "required": ["person", "role_or_relation", "latent_relevance", "practical_topics"],
                            },
                            "maxItems": 3,
                        },
                        "avoidances": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 120},
                            "maxItems": 3,
                        },
                    },
                    "required": [
                        "use_personalization",
                        "conversation_goal",
                        "helpful_biases",
                        "suggested_directions",
                        "follow_up_angles",
                        "known_people",
                        "avoidances",
                    ],
                },
                model=self.config.long_term_memory_subtext_compiler_model or self.backend.config.default_model,
                reasoning_effort="low",
                max_output_tokens=self.config.long_term_memory_subtext_compiler_max_output_tokens,
            )
        except Exception:
            return None
        compiled = _sanitize_structured_value(compiled)
        if not isinstance(compiled, dict):
            return None
        if self._cache is not None:
            self._cache[cache_key] = compiled
        return compiled


@dataclass(frozen=True, slots=True)
class LongTermSubtextBuilder:
    config: TwinrConfig
    graph_store: TwinrPersonalGraphStore
    compiler: LongTermSubtextCompiler | None = None

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
        compiled_program = None
        if self.compiler is not None:
            compiled_program = self.compiler.compile(
                query_text=query_text,
                retrieval_query_text=retrieval_query_text,
                graph_payload=graph_payload,
                recent_threads=recent_threads,
            )
        if compiled_program is not None:
            directive_lines = self._render_compiled_directives(compiled_program)
            payload: dict[str, object] = {
                "schema": "twinr_silent_personalization_program_v3",
                "query": _normalize_query_text(query_text),
                "canonical_retrieval_query": _normalize_query_text(retrieval_query_text),
                "program": compiled_program,
            }
            if directive_lines:
                payload["rendered_guidance"] = directive_lines
            rendered_directives = ""
            if directive_lines:
                rendered_directives = "\n".join(f"- {line}" for line in directive_lines) + "\n"
            return (
                "Silent personalization operating directives for this turn. Internal memory is canonical English. "
                "Apply the following directives when they improve the reply, and keep them implicit. "
                "If a concrete recommendation or decision is being made and one of these cues is clearly relevant, let at least one relevant cue materially shape the answer. "
                "Do not narrate that these cues came from memory.\n"
                + rendered_directives
                + "Structured program JSON for audit and grounding:\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)
            )
        payload: dict[str, object] = {
            "schema": "twinr_silent_personalization_context_v1",
            "query": _normalize_query_text(query_text),
            "principles": [
                "Let relevant familiarity shape priorities, suggestions, and tone without announcing hidden memory.",
                "Use remembered preferences and ongoing situations only when they genuinely help the current reply.",
                "Prefer natural continuity over overt memory announcements unless direct recall is the point.",
                "For known people, prefer implicit familiarity over explicitly naming remembered roles unless identity clarification or safety truly requires it.",
                "When a known person matters, prefer concrete reasons to contact them or concrete follow-up questions over restating the remembered role.",
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

    def _render_compiled_directives(self, compiled_program: dict[str, object]) -> list[str]:
        if not bool(compiled_program.get("use_personalization", False)):
            return []
        lines: list[str] = []
        conversation_goal = self._clean_compiled_text(compiled_program.get("conversation_goal"))
        if conversation_goal:
            lines.append(f"Conversation goal: {conversation_goal}")
        helpful_biases = self._clean_compiled_list(compiled_program.get("helpful_biases"))
        if helpful_biases:
            lines.append("Relevant helpful biases: " + "; ".join(helpful_biases[:2]))
        directions = self._clean_compiled_list(compiled_program.get("suggested_directions"))
        if directions:
            lines.append("Steer the reply toward: " + "; ".join(directions[:2]))
        follow_up_angles = self._clean_compiled_list(compiled_program.get("follow_up_angles"))
        if follow_up_angles:
            lines.append("If a follow-up is needed, prefer: " + "; ".join(follow_up_angles[:2]))
        known_people = compiled_program.get("known_people")
        if isinstance(known_people, list):
            for person_payload in known_people[:3]:
                if not isinstance(person_payload, dict):
                    continue
                person = self._clean_compiled_text(person_payload.get("person"))
                role = self._clean_compiled_text(person_payload.get("role_or_relation"))
                relevance = self._clean_compiled_text(person_payload.get("latent_relevance"))
                practical_topics = self._clean_compiled_list(person_payload.get("practical_topics"))
                if not person:
                    continue
                person_line = f"If {person} is directly relevant, treat them as {role or 'a known person'}."
                if relevance:
                    person_line += f" Practical frame: {relevance}."
                if practical_topics:
                    person_line += " Useful practical angles: " + "; ".join(practical_topics[:3]) + "."
                lines.append(person_line)
        avoidances = self._clean_compiled_list(compiled_program.get("avoidances"))
        if avoidances:
            lines.append("Avoid: " + "; ".join(avoidances[:3]))
        return lines

    def _clean_compiled_list(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            text = self._clean_compiled_text(item)
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned

    def _clean_compiled_text(self, value: object) -> str:
        text = sanitize_text_fragment("" if value is None else str(value))
        if not text:
            return ""
        return collapse_whitespace(text)

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
