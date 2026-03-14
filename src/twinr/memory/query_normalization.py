from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import re
from typing import TYPE_CHECKING

from twinr.agent.base_agent.config import TwinrConfig

if TYPE_CHECKING:
    from twinr.providers.openai import OpenAIBackend

_TOKEN_RE = re.compile(r"[a-z0-9äöüß]+", re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "bei",
        "bin",
        "bist",
        "bitte",
        "da",
        "das",
        "de",
        "dem",
        "den",
        "der",
        "des",
        "die",
        "doch",
        "du",
        "ein",
        "eine",
        "einem",
        "einen",
        "einer",
        "er",
        "es",
        "etwas",
        "for",
        "gerade",
        "hat",
        "have",
        "hier",
        "i",
        "ich",
        "if",
        "im",
        "in",
        "is",
        "ist",
        "it",
        "ja",
        "kein",
        "keine",
        "mal",
        "mein",
        "meine",
        "mit",
        "my",
        "noch",
        "of",
        "oder",
        "on",
        "the",
        "to",
        "und",
        "was",
        "wenn",
        "wie",
        "wir",
        "wo",
        "would",
        "you",
    }
)


def _normalize_text(value: str | None, *, limit: int = 220) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def tokenize_retrieval_text(value: str | None) -> tuple[str, ...]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(str(value or "")):
        token = match.group(0).lower()
        if len(token) <= 1 and not token.isdigit():
            continue
        if token in _STOPWORDS and not token.isdigit():
            continue
        tokens.append(token)
    return tuple(tokens)


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
        retrieval_parts = [part for part in (original, canonical) if part]
        retrieval_text = "\n".join(dict.fromkeys(retrieval_parts))
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
        prompt = "\n".join(
            [
                "Rewrite the user's request into short canonical English for internal long-term-memory retrieval.",
                "Do not answer the question.",
                "Preserve names, brands, dates, places, phone numbers, and IDs verbatim.",
                "Prefer content words that help semantic retrieval.",
                f"User request: {query_text}",
                'Return JSON only: {"canonical_query": "..."}',
            ]
        )
        request = self.backend._build_response_request(
            prompt,
            instructions="Return one compact JSON object only.",
            allow_web_search=False,
            model=self.backend.config.default_model,
            reasoning_effort="low",
            max_output_tokens=120,
        )
        response = self.backend._client.responses.create(**request)
        text = self.backend._extract_output_text(response)
        try:
            payload = self._extract_json_object(text)
        except ValueError:
            return _normalize_text(text)
        return _normalize_text(str(payload.get("canonical_query", "")))

    def _extract_json_object(self, text: str) -> dict[str, object]:
        match = _JSON_OBJECT_RE.search(text)
        if match is None:
            raise ValueError("No JSON object found in canonical query response.")
        payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise ValueError("Canonical query response must be a JSON object.")
        return payload


__all__ = [
    "LongTermQueryProfile",
    "LongTermQueryRewriter",
    "tokenize_retrieval_text",
]
