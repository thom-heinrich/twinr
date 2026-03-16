from __future__ import annotations
##REFACTOR: 2026-03-16##

from datetime import date, datetime, timedelta, timezone
import re
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.ops.usage import TokenUsage, extract_model_name, extract_token_usage

from .instructions import SEARCH_AGENT_INSTRUCTIONS, SEARCH_MODEL_FALLBACKS
from .types import ConversationLike, OpenAISearchResult

_SEARCH_CONTEXT_MAX_TURNS = 3
_SEARCH_CONTEXT_CHAR_LIMIT = 160
_DATE_CONTEXT_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_TODAY_PATTERNS = (
    re.compile(r"\bheute\b", re.IGNORECASE),
    re.compile(r"\btoday\b", re.IGNORECASE),
)
_TOMORROW_PATTERNS = (
    re.compile(r"\bmorgen\b", re.IGNORECASE),
    re.compile(r"\btomorrow\b", re.IGNORECASE),
)
_DAY_AFTER_TOMORROW_PATTERNS = (
    re.compile(r"\bübermorgen\b", re.IGNORECASE),
    re.compile(r"\buebermorgen\b", re.IGNORECASE),
    re.compile(r"\bday after tomorrow\b", re.IGNORECASE),
)
_YESTERDAY_PATTERNS = (
    re.compile(r"\bgestern\b", re.IGNORECASE),
    re.compile(r"\byesterday\b", re.IGNORECASE),
)
_NEXT_WEEKDAY_EN_RE = re.compile(
    r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_NEXT_WEEKDAY_DE_RE = re.compile(
    r"\bn(?:ä|ae)chst(?:e|en|er|es)?\s+"
    r"(montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag)\b",
    re.IGNORECASE,
)
_ENGLISH_RELATIVE_PATTERNS = frozenset(
    {
        _TODAY_PATTERNS[1],
        _TOMORROW_PATTERNS[1],
        _DAY_AFTER_TOMORROW_PATTERNS[2],
        _YESTERDAY_PATTERNS[1],
    }
)
_WEEKDAY_INDEX_BY_NAME = {
    "monday": 0,
    "montag": 0,
    "tuesday": 1,
    "dienstag": 1,
    "wednesday": 2,
    "mittwoch": 2,
    "thursday": 3,
    "donnerstag": 3,
    "friday": 4,
    "freitag": 4,
    "saturday": 5,
    "samstag": 5,
    "sunday": 6,
    "sonntag": 6,
}
_VALID_SEARCH_CONTEXT_SIZES = frozenset({"low", "medium", "high"})
_DEFAULT_SEARCH_MAX_OUTPUT_TOKENS = 512
_DEFAULT_SEARCH_RETRY_MAX_OUTPUT_TOKENS = 768
_FALLBACK_SEARCH_MODEL = "gpt-5"


def _collapse_whitespace(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_positive_int(value: Any, *, default: int, minimum: int = 1) -> int:
    coerced = _coerce_int(value)
    if coerced is None:
        return default
    return max(minimum, coerced)


def _normalize_search_context_size(value: Any) -> str | None:
    normalized = _collapse_whitespace(None if value is None else str(value)).lower()
    if normalized in _VALID_SEARCH_CONTEXT_SIZES:
        return normalized
    return None


def _parse_context_reference_date(date_context: str | None) -> date | None:
    normalized_context = _collapse_whitespace(date_context)
    if not normalized_context:
        return None
    match = _DATE_CONTEXT_ISO_RE.search(normalized_context)
    if match is None:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def _contains_explicit_iso_date(value: str) -> bool:
    return _DATE_CONTEXT_ISO_RE.search(value) is not None


def _date_phrase_for_pattern(pattern: re.Pattern[str], resolved_date: str) -> str:
    if pattern in _ENGLISH_RELATIVE_PATTERNS:
        return f"on {resolved_date}"
    return f"am {resolved_date}"


def _replace_relative_day_reference(question: str, resolved_date: str) -> str | None:
    for patterns in (
        _DAY_AFTER_TOMORROW_PATTERNS,
        _TOMORROW_PATTERNS,
        _TODAY_PATTERNS,
        _YESTERDAY_PATTERNS,
    ):
        for pattern in patterns:
            if pattern.search(question):
                return pattern.sub(_date_phrase_for_pattern(pattern, resolved_date), question, count=1)
    return None


def _replace_next_weekday_reference(question: str, reference_date: date) -> str | None:
    for pattern, is_english in ((_NEXT_WEEKDAY_EN_RE, True), (_NEXT_WEEKDAY_DE_RE, False)):
        match = pattern.search(question)
        if match is None:
            continue
        weekday_name = match.group(1).strip().lower()
        target_weekday = _WEEKDAY_INDEX_BY_NAME.get(weekday_name)
        if target_weekday is None:
            continue
        delta_days = (target_weekday - reference_date.weekday()) % 7
        if delta_days == 0:
            delta_days = 7
        resolved_date = (reference_date + timedelta(days=delta_days)).isoformat()
        replacement = f"on {resolved_date}" if is_english else f"am {resolved_date}"
        return pattern.sub(replacement, question, count=1)
    return None


def _resolved_explicit_search_query(question: str, date_context: str | None) -> str | None:
    normalized_question = _collapse_whitespace(question)
    if not normalized_question:
        return None
    if _contains_explicit_iso_date(normalized_question):
        # AUDIT-FIX(#2): Do not append a second target date to a question that already names an ISO date.
        return normalized_question
    reference_date = _parse_context_reference_date(date_context)
    if reference_date is None:
        return None

    # AUDIT-FIX(#2): Resolve weekday-relative queries like "next Monday" into a concrete ISO date.
    replaced_weekday = _replace_next_weekday_reference(normalized_question, reference_date)
    if replaced_weekday is not None and replaced_weekday != normalized_question:
        return replaced_weekday

    # AUDIT-FIX(#2): Only synthesize an explicit-date retrieval query when the user actually used a relative date phrase.
    replaced_relative_day = _replace_relative_day_reference(normalized_question, reference_date.isoformat())
    if replaced_relative_day is not None and replaced_relative_day != normalized_question:
        return replaced_relative_day
    return None


class OpenAISearchMixin:
    def search_live_info_with_metadata(
        self,
        question: str,
        *,
        conversation: ConversationLike | None = None,
        location_hint: str | None = None,
        date_context: str | None = None,
    ) -> OpenAISearchResult:
        normalized_question = _collapse_whitespace(question)
        if not normalized_question:
            raise RuntimeError("search_live_info requires a non-empty question")
        prompt = self._build_search_prompt(
            normalized_question,
            location_hint=location_hint,
            date_context=date_context,
        )
        instructions = SEARCH_AGENT_INSTRUCTIONS
        search_conversation = self._prepare_search_conversation(conversation)
        last_error: Exception | None = None
        output_token_candidates = self._search_output_token_candidates()

        for model in self._candidate_search_models():
            if self._is_search_preview_model(model):
                if not self._search_preview_supported():
                    continue
                for max_output_tokens in output_token_candidates:
                    try:
                        # AUDIT-FIX(#3): Keep per-model fallback alive if a preview request fails transiently.
                        candidate, is_complete = self._search_with_preview_model(
                            model,
                            prompt,
                            conversation=search_conversation,
                            instructions=instructions,
                            location_hint=location_hint,
                            max_output_tokens=max_output_tokens,
                        )
                    except Exception as exc:  # pragma: no cover - defensive runtime fallback
                        last_error = exc
                        continue
                    if candidate.answer and is_complete:
                        return candidate
                continue

            for max_output_tokens in output_token_candidates:
                try:
                    request = self._build_response_request(
                        prompt,
                        conversation=search_conversation,
                        instructions=instructions,
                        allow_web_search=True,
                        model=model,
                        reasoning_effort="low",
                        max_output_tokens=max_output_tokens,
                        prompt_cache_scope="search",
                    )
                    # AUDIT-FIX(#6): Patch the actual web-search tool only, validate search_context_size, and preserve other include fields.
                    self._apply_web_search_request_overrides(request, location_hint=location_hint)
                    self._ensure_web_search_sources_included(request)
                    # AUDIT-FIX(#3): Keep trying remaining models/attempts on transient Responses API failures.
                    response = self._client.responses.create(**request)
                except Exception as exc:  # pragma: no cover - defensive runtime fallback
                    last_error = exc
                    continue

                candidate = OpenAISearchResult(
                    answer=self._sanitize_search_answer(self._extract_output_text(response)),
                    sources=self._extract_web_search_sources(response),
                    response_id=getattr(response, "id", None),
                    request_id=getattr(response, "_request_id", None),
                    model=extract_model_name(response, model),
                    token_usage=extract_token_usage(response),
                    used_web_search=self._used_web_search(response),
                )
                if candidate.answer and self._response_is_complete(response):
                    return candidate

        if last_error is not None:
            raise RuntimeError(
                f"OpenAI web search failed after exhausting configured models and retries: {last_error}"
            ) from last_error
        raise RuntimeError("OpenAI web search returned no usable answer text")

    def _search_with_preview_model(
        self,
        model: str,
        prompt: str,
        *,
        conversation: ConversationLike | None,
        instructions: str,
        location_hint: str | None,
        max_output_tokens: int,
    ) -> tuple[OpenAISearchResult, bool]:
        messages: list[dict[str, str]] = []
        if instructions.strip():
            messages.append({"role": "system", "content": instructions.strip()})
        if conversation:
            for item in conversation:
                role, content, _phase = self._coerce_message(item)
                normalized_role = role.strip().lower()
                if normalized_role not in {"user", "assistant"}:
                    continue
                normalized_content = _collapse_whitespace(content)
                if normalized_content:
                    messages.append({"role": normalized_role, "content": normalized_content})
        messages.append({"role": "user", "content": prompt})
        # AUDIT-FIX(#8): Provide structured geography/search-context hints to the Chat Completions web-search models.
        response = self._client.chat.completions.create(
            model=model,
            web_search_options=self._build_preview_web_search_options(location_hint=location_hint),
            max_completion_tokens=max_output_tokens,
            messages=messages,
        )
        message = response.choices[0].message if getattr(response, "choices", None) else None
        answer = self._sanitize_search_answer(getattr(message, "content", "") or "")
        result = OpenAISearchResult(
            answer=answer,
            sources=self._extract_preview_search_sources(message),
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, model),
            token_usage=self._extract_preview_usage(response),
            used_web_search=True,
        )
        # AUDIT-FIX(#7): Reject truncated/filtered preview answers instead of returning potentially partial output.
        return result, self._preview_response_is_complete(response)

    def _build_search_prompt(
        self,
        question: str,
        *,
        location_hint: str | None,
        date_context: str | None,
    ) -> str:
        parts = [f"User question: {question}"]
        resolved_location = (
            _collapse_whitespace(location_hint)
            or _collapse_whitespace(getattr(self.config, "openai_web_search_city", None))
        )
        if resolved_location:
            parts.append(f"Location hint: {resolved_location}")
        resolved_date_context = _collapse_whitespace(date_context) or self._relative_date_context()
        explicit_query = _resolved_explicit_search_query(question, resolved_date_context)
        resolved_date = None
        match = _DATE_CONTEXT_ISO_RE.search(resolved_date_context)
        if match is not None:
            resolved_date = match.group(1)
        if resolved_date_context:
            parts.append(f"Local date/time context: {resolved_date_context}")
            parts.append(
                "Treat that date/time context as the exact target reference for this search. "
                "If the user asked about a relative day like today, tomorrow, heute, morgen, or next Monday, "
                "answer only for the resolved absolute date that matches this reference. "
                "Do not answer for adjacent dates. If the live sources only mention another date or remain contradictory, "
                "say that the exact date could not be verified."
            )
        if resolved_date:
            parts.append(f"Target date (must match): {resolved_date}")
        if explicit_query:
            parts.append(f"Equivalent explicit-date query: {explicit_query}")
            parts.append("Use that explicit-date query internally for retrieval and answer only for that exact date.")
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
        timezone_name = self._configured_timezone_name()
        effective_label: str
        try:
            if timezone_name:
                # AUDIT-FIX(#1): Use a real ZoneInfo object and never emit a misleading configured label after fallback.
                now = datetime.now(ZoneInfo(timezone_name))
                effective_label = timezone_name
            else:
                now = datetime.now(timezone.utc).astimezone()
                effective_label = getattr(now.tzinfo, "key", None) or now.tzname() or "local"
        except ZoneInfoNotFoundError:
            now = datetime.now(timezone.utc).astimezone()
            effective_label = getattr(now.tzinfo, "key", None) or now.tzname() or "local"
        return now.strftime(f"%A, %Y-%m-%d %H:%M ({effective_label})")

    def _candidate_search_models(self) -> tuple[str, ...]:
        candidates: list[str] = []
        configured = _collapse_whitespace(getattr(self.config, "openai_search_model", None))
        if configured:
            candidates.append(configured)
        for fallback in SEARCH_MODEL_FALLBACKS:
            normalized_fallback = _collapse_whitespace(fallback)
            if normalized_fallback and normalized_fallback not in candidates:
                candidates.append(normalized_fallback)
        default_model = _collapse_whitespace(getattr(self.config, "default_model", None))
        if default_model and default_model not in candidates:
            candidates.append(default_model)
        if not candidates:
            # AUDIT-FIX(#5): Ensure a deterministic last-resort model candidate instead of an empty iteration set.
            candidates.append(_FALLBACK_SEARCH_MODEL)
        return tuple(candidates)

    def _is_search_preview_model(self, model: str) -> bool:
        normalized = _collapse_whitespace(model).lower()
        # AUDIT-FIX(#5): Search-chat model families are not limited to names containing "search-preview".
        return "search-preview" in normalized or normalized.endswith("-search-api")

    def _search_preview_supported(self) -> bool:
        chat = getattr(self._client, "chat", None)
        completions = getattr(chat, "completions", None)
        create = getattr(completions, "create", None)
        return callable(create)

    def _extract_preview_search_sources(self, message: Any) -> tuple[str, ...]:
        urls: list[str] = []
        for annotation in getattr(message, "annotations", None) or []:
            url_citation = getattr(annotation, "url_citation", None)
            url = str(getattr(url_citation, "url", "") or "").strip()
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

    def _extract_preview_usage(self, response: Any) -> TokenUsage | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        token_usage = TokenUsage(
            input_tokens=_coerce_int(getattr(usage, "prompt_tokens", None)),
            output_tokens=_coerce_int(getattr(usage, "completion_tokens", None)),
            total_tokens=_coerce_int(getattr(usage, "total_tokens", None)),
            cached_input_tokens=_coerce_int(
                getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
            ),
            reasoning_tokens=_coerce_int(
                getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
            ),
            audio_input_tokens=_coerce_int(
                getattr(getattr(usage, "prompt_tokens_details", None), "audio_tokens", None)
            ),
            audio_output_tokens=_coerce_int(
                getattr(getattr(usage, "completion_tokens_details", None), "audio_tokens", None)
            ),
        )
        if not token_usage.has_values:
            return None
        return token_usage

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
            role, content, _phase = self._coerce_message(item)
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

    def _configured_timezone_name(self) -> str | None:
        timezone_name = _collapse_whitespace(
            getattr(self.config, "openai_web_search_timezone", None)
            or getattr(self.config, "local_timezone_name", None)
        )
        return timezone_name or None

    def _search_output_token_candidates(self) -> tuple[int, ...]:
        # AUDIT-FIX(#4): Parse env-backed token limits defensively so invalid config cannot crash the search path.
        primary = _coerce_positive_int(
            getattr(self.config, "openai_search_max_output_tokens", None),
            default=_DEFAULT_SEARCH_MAX_OUTPUT_TOKENS,
            minimum=64,
        )
        retry = _coerce_positive_int(
            getattr(self.config, "openai_search_retry_max_output_tokens", None),
            default=_DEFAULT_SEARCH_RETRY_MAX_OUTPUT_TOKENS,
            minimum=96,
        )
        if primary == retry:
            return (primary,)
        return (primary, retry)

    def _response_is_complete(self, response: Any) -> bool:
        status = _collapse_whitespace(getattr(response, "status", None)).lower()
        if status and status != "completed":
            return False
        return not self._response_has_incomplete_message(response)

    def _preview_response_is_complete(self, response: Any) -> bool:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return False
        finish_reason = _collapse_whitespace(getattr(choices[0], "finish_reason", None)).lower()
        return finish_reason in {"", "stop"}

    def _apply_web_search_request_overrides(self, request: dict[str, Any], *, location_hint: str | None) -> None:
        tools = request.get("tools")
        if not isinstance(tools, list):
            return
        search_context_size = _normalize_search_context_size(
            getattr(self.config, "openai_web_search_context_size", None)
        )
        # AUDIT-FIX(#8): Use structured user_location on Responses web_search instead of relying on prompt prose alone.
        user_location = self._build_responses_web_search_user_location(location_hint=location_hint)
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_type = _collapse_whitespace(tool.get("type")).lower()
            if tool_type not in {"web_search", "web_search_preview"}:
                continue
            if search_context_size is not None:
                tool["search_context_size"] = search_context_size
            if user_location is not None and tool_type == "web_search":
                tool["user_location"] = user_location
            break

    def _ensure_web_search_sources_included(self, request: dict[str, Any]) -> None:
        include = request.get("include")
        if include is None:
            request["include"] = ["web_search_call.action.sources"]
            return
        if not isinstance(include, list):
            request["include"] = [include, "web_search_call.action.sources"]
            return
        if "web_search_call.action.sources" not in include:
            include.append("web_search_call.action.sources")

    def _build_responses_web_search_user_location(self, *, location_hint: str | None) -> dict[str, str] | None:
        city = _collapse_whitespace(location_hint) or _collapse_whitespace(
            getattr(self.config, "openai_web_search_city", None)
        )
        region = _collapse_whitespace(getattr(self.config, "openai_web_search_region", None))
        country = _collapse_whitespace(getattr(self.config, "openai_web_search_country", None)).upper()
        timezone_name = self._configured_timezone_name()

        payload: dict[str, str] = {"type": "approximate"}
        if city:
            payload["city"] = city
        if region:
            payload["region"] = region
        if len(country) == 2 and country.isalpha():
            payload["country"] = country
        if timezone_name:
            payload["timezone"] = timezone_name
        return payload if len(payload) > 1 else None

    def _build_preview_web_search_options(self, *, location_hint: str | None) -> dict[str, Any]:
        options: dict[str, Any] = {}
        search_context_size = _normalize_search_context_size(
            getattr(self.config, "openai_web_search_context_size", None)
        )
        if search_context_size is not None:
            options["search_context_size"] = search_context_size

        city = _collapse_whitespace(location_hint) or _collapse_whitespace(
            getattr(self.config, "openai_web_search_city", None)
        )
        region = _collapse_whitespace(getattr(self.config, "openai_web_search_region", None))
        country = _collapse_whitespace(getattr(self.config, "openai_web_search_country", None)).upper()
        timezone_name = self._configured_timezone_name()

        approximate: dict[str, str] = {}
        if city:
            approximate["city"] = city
        if region:
            approximate["region"] = region
        if len(country) == 2 and country.isalpha():
            approximate["country"] = country
        if timezone_name:
            approximate["timezone"] = timezone_name
        if approximate:
            options["user_location"] = {
                "type": "approximate",
                "approximate": approximate,
            }
        return options
