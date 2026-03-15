from __future__ import annotations

from datetime import datetime
from typing import Any

from twinr.ops.usage import extract_model_name, extract_token_usage

from .instructions import SEARCH_AGENT_INSTRUCTIONS, SEARCH_MODEL_FALLBACKS
from .types import ConversationLike, OpenAISearchResult

_SEARCH_CONTEXT_MAX_TURNS = 3
_SEARCH_CONTEXT_CHAR_LIMIT = 160


def _collapse_whitespace(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


class OpenAISearchMixin:
    def search_live_info_with_metadata(
        self,
        question: str,
        *,
        conversation: ConversationLike | None = None,
        location_hint: str | None = None,
        date_context: str | None = None,
    ) -> OpenAISearchResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise RuntimeError("search_live_info requires a non-empty question")
        prompt = self._build_search_prompt(
            normalized_question,
            location_hint=location_hint,
            date_context=date_context,
        )
        instructions = SEARCH_AGENT_INSTRUCTIONS
        search_conversation = self._prepare_search_conversation(conversation)
        best_result: OpenAISearchResult | None = None

        for model in self._candidate_search_models():
            for max_output_tokens in (
                max(64, int(self.config.openai_search_max_output_tokens)),
                max(96, int(self.config.openai_search_retry_max_output_tokens)),
            ):
                request = self._build_response_request(
                    prompt,
                    conversation=search_conversation,
                    instructions=instructions,
                    allow_web_search=True,
                    model=model,
                    reasoning_effort="medium",
                    max_output_tokens=max_output_tokens,
                    prompt_cache_scope="search",
                )
                if request.get("tools"):
                    request["tools"][0]["search_context_size"] = self.config.openai_web_search_context_size
                request["include"] = ["web_search_call.action.sources"]
                response = self._client.responses.create(**request)
                candidate = OpenAISearchResult(
                    answer=self._sanitize_search_answer(self._extract_output_text(response)),
                    sources=self._extract_web_search_sources(response),
                    response_id=getattr(response, "id", None),
                    request_id=getattr(response, "_request_id", None),
                    model=extract_model_name(response, model),
                    token_usage=extract_token_usage(response),
                    used_web_search=self._used_web_search(response),
                )
                if candidate.answer and not self._response_has_incomplete_message(response):
                    return candidate
                if candidate.answer and (best_result is None or len(candidate.answer) > len(best_result.answer)):
                    best_result = candidate

        if best_result is not None:
            return best_result
        raise RuntimeError("OpenAI web search returned no usable answer text")

    def _build_search_prompt(
        self,
        question: str,
        *,
        location_hint: str | None,
        date_context: str | None,
    ) -> str:
        parts = [f"User question: {question}"]
        resolved_location = (location_hint or self.config.openai_web_search_city or "").strip()
        if resolved_location:
            parts.append(f"Location hint: {resolved_location}")
        resolved_date_context = (date_context or self._relative_date_context()).strip()
        if resolved_date_context:
            parts.append(f"Local date/time context: {resolved_date_context}")
            parts.append(
                "Treat that date/time context as the exact target reference for this search. "
                "If the user asked about a relative day like today, tomorrow, heute, morgen, or next Monday, "
                "answer only for the resolved absolute date that matches this reference. "
                "Do not answer for adjacent dates. If the live sources only mention another date or remain contradictory, "
                "say that the exact date could not be verified."
            )
        parts.append("Answer now with the best live information you can verify from web search.")
        return "\n".join(parts)

    def _extract_web_search_sources(self, response: Any) -> tuple[str, ...]:
        urls: list[str] = []
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) != "web_search_call":
                continue
            action = getattr(item, "action", None)
            sources = getattr(action, "sources", None) or []
            for source in sources:
                url = str(getattr(source, "url", "")).strip()
                if url:
                    urls.append(url)
        deduped: list[str] = []
        seen: set[str] = set()
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            deduped.append(url)
        return tuple(deduped)

    def _relative_date_context(self) -> str:
        timezone_name = self.config.openai_web_search_timezone or self.config.local_timezone_name
        try:
            now = datetime.now(datetime.now().astimezone().tzinfo if not timezone_name else __import__("zoneinfo").ZoneInfo(timezone_name))
        except Exception:
            now = datetime.now()
        return now.strftime(f"%A, %Y-%m-%d %H:%M ({timezone_name})")

    def _candidate_search_models(self) -> tuple[str, ...]:
        candidates: list[str] = []
        configured = (self.config.openai_search_model or "").strip()
        if configured:
            candidates.append(configured)
        default_model = (self.config.default_model or "").strip()
        if default_model and default_model not in candidates:
            candidates.append(default_model)
        for fallback in SEARCH_MODEL_FALLBACKS:
            if fallback not in candidates:
                candidates.append(fallback)
        return tuple(candidates)

    def _response_has_incomplete_message(self, response: Any) -> bool:
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) == "message" and getattr(item, "status", None) == "incomplete":
                return True
        return False

    def _sanitize_search_answer(self, text: str) -> str:
        return _collapse_whitespace(text.replace("\r", " ").replace("\n", " "))

    def _prepare_search_conversation(
        self,
        conversation: ConversationLike | None,
    ) -> tuple[tuple[str, str], ...] | None:
        if not conversation:
            return None
        filtered: list[tuple[str, str]] = []
        for item in conversation:
            role, content = self._coerce_message(item)
            normalized_role = role.strip().lower()
            if normalized_role not in {"user", "assistant"}:
                continue
            normalized_content = _collapse_whitespace(content)
            if not normalized_content:
                continue
            if len(normalized_content) > _SEARCH_CONTEXT_CHAR_LIMIT:
                normalized_content = normalized_content[: _SEARCH_CONTEXT_CHAR_LIMIT - 1].rstrip() + "…"
            filtered.append((normalized_role, normalized_content))
        if not filtered:
            return None
        trimmed = filtered[-_SEARCH_CONTEXT_MAX_TURNS :]
        if len(trimmed) > 1 and trimmed[0][0] == "assistant":
            trimmed = trimmed[1:]
        return tuple(trimmed) if trimmed else None
