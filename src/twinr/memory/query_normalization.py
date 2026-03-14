from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from twinr.agent.base_agent.config import TwinrConfig
from twinr.llm_json import request_structured_json_object
from twinr.text_utils import retrieval_terms, truncate_text

if TYPE_CHECKING:
    from twinr.providers.openai import OpenAIBackend


def _normalize_text(value: str | None, *, limit: int = 220) -> str:
    return truncate_text(value, limit=limit)


def tokenize_retrieval_text(value: str | None) -> tuple[str, ...]:
    return retrieval_terms(value)


@dataclass(frozen=True, slots=True)
class LongTermQueryProfile:
    original_text: str
    canonical_english_text: str | None
    retrieval_text: str

    @classmethod
    def from_text(
        cls,
        query_text: str | None,
        *,
        canonical_english_text: str | None = None,
    ) -> "LongTermQueryProfile":
        original = _normalize_text(query_text)
        canonical = _normalize_text(canonical_english_text)
        retrieval_text = canonical or original
        return cls(
            original_text=original,
            canonical_english_text=canonical or None,
            retrieval_text=retrieval_text or original,
        )


@dataclass(slots=True)
class LongTermQueryRewriter:
    config: TwinrConfig
    backend: "OpenAIBackend | None" = None
    _cache: dict[str, LongTermQueryProfile] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LongTermQueryRewriter":
        backend: OpenAIBackend | None = None
        language = (config.openai_realtime_language or "").strip().lower()
        if config.long_term_memory_query_rewrite_enabled and config.openai_api_key and language not in {"", "en"}:
            from twinr.providers.openai import OpenAIBackend

            backend = OpenAIBackend(
                replace(
                    config,
                    openai_realtime_language="en",
                    openai_reasoning_effort="low",
                ),
                base_instructions="",
            )
        return cls(config=config, backend=backend)

    def profile(self, query_text: str | None) -> LongTermQueryProfile:
        clean_query = _normalize_text(query_text)
        if not clean_query:
            return LongTermQueryProfile.from_text("")
        cached = self._cache.get(clean_query)
        if cached is not None:
            return cached
        canonical_query = self._canonicalize_query(clean_query)
        profile = LongTermQueryProfile.from_text(
            clean_query,
            canonical_english_text=canonical_query,
        )
        self._cache[clean_query] = profile
        return profile

    def _canonicalize_query(self, query_text: str) -> str | None:
        if self.backend is None:
            return None
        try:
            payload = request_structured_json_object(
                self.backend,
                prompt=(
                    "Rewrite the user's request into short canonical English for internal long-term-memory retrieval.\n"
                    "Do not answer the question.\n"
                    "Preserve names, brands, dates, places, phone numbers, and IDs verbatim.\n"
                    "Prefer concise semantic phrasing that improves retrieval.\n"
                    f"User request: {query_text}"
                ),
                instructions=(
                    "Return one strict JSON object only. "
                    "Do not emit markdown, code fences, or explanatory text."
                ),
                schema_name="twinr_long_term_query_rewrite_v1",
                schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "canonical_query": {"type": "string"},
                    },
                    "required": ["canonical_query"],
                },
                model=self.backend.config.default_model,
                reasoning_effort="low",
                max_output_tokens=120,
            )
        except Exception:
            return None
        return _normalize_text(str(payload.get("canonical_query", "")))


__all__ = [
    "LongTermQueryProfile",
    "LongTermQueryRewriter",
    "tokenize_retrieval_text",
]
