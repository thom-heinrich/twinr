from __future__ import annotations

from datetime import datetime
from typing import Any
import re

from twinr.agent.base_agent.personality import merge_instructions
from twinr.ops.usage import extract_model_name, extract_token_usage

from .instructions import SEARCH_AGENT_INSTRUCTIONS, SEARCH_MODEL_FALLBACKS
from .types import ConversationLike, OpenAISearchResult


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
        instructions = merge_instructions(self._resolve_base_instructions(), SEARCH_AGENT_INSTRUCTIONS)
        best_result: OpenAISearchResult | None = None

        for model in self._candidate_search_models():
            for max_output_tokens in (320, 480):
                request = self._build_response_request(
                    prompt,
                    conversation=conversation,
                    instructions=instructions,
                    allow_web_search=True,
                    model=model,
                    reasoning_effort="medium",
                    max_output_tokens=max_output_tokens,
                )
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
        normalized = re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()
        return normalized
