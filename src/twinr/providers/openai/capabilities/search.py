# CHANGELOG: 2026-03-30
# BUG-1: Fixed relative-day resolution; today/tomorrow/yesterday/übermorgen now resolve to the correct shifted ISO date.
# BUG-2: Live-search responses now require an actual web-search tool call; stale non-search answers are rejected.
# BUG-3: Replayed search conversation now preserves assistant phase to avoid GPT-5.4 commentary/final-answer misclassification.
# SEC-1: Disabled provider-side response storage by default for search/rewrite calls and sanitized model-proposed follow-up URLs to public http(s) targets only.
# IMP-1: Added bounded transient retry/backoff, optional domain filtering/external_web_access, and safer search-tool forcing.
# IMP-2: Upgraded model fallback and structured-rewrite selection for 2026 GPT-5.4 search latency/capability patterns.

"""Resolve live-information queries with OpenAI web-search capable models.

This module contains the search-specific prompt shaping, date disambiguation,
source extraction, and model fallback logic used by Twinr's OpenAI backend.
It keeps live-search behavior isolated from the generic response helpers.
"""

from __future__ import annotations
##REFACTOR: 2026-03-30##

from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
import ipaddress
import json
import random
import re
import time
from typing import Any, Iterable
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.agent.base_agent.config import DEFAULT_OPENAI_MAIN_MODEL
from twinr.ops.usage import TokenUsage, extract_model_name, extract_token_usage

from ..core.instructions import SEARCH_AGENT_INSTRUCTIONS, SEARCH_MODEL_FALLBACKS
from ..core.types import ConversationLike, OpenAISearchAttempt, OpenAISearchResult

_SEARCH_CONTEXT_MAX_TURNS = 3
_SEARCH_CONTEXT_CHAR_LIMIT = 160
_DATE_CONTEXT_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_GPT_VERSION_RE = re.compile(r"^gpt-(\d+)(?:\.(\d+))?(?:[-_.].*)?$")
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
_RELATIVE_DAY_PATTERN_GROUPS: tuple[tuple[tuple[re.Pattern[str], ...], int], ...] = (
    (_YESTERDAY_PATTERNS, -1),
    (_TODAY_PATTERNS, 0),
    (_TOMORROW_PATTERNS, 1),
    (_DAY_AFTER_TOMORROW_PATTERNS, 2),
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
_DEFAULT_SEARCH_MAX_OUTPUT_TOKENS = 1024
_DEFAULT_SEARCH_RETRY_MAX_OUTPUT_TOKENS = 1536
_SEARCH_OUTPUT_TOKEN_RETRY_LADDER = (512, 768, 1024, 1536, 2048, 3072)
_FRONTIER_DEFAULT_SEARCH_MODEL = "gpt-5.4-mini"
_FRONTIER_SEARCH_MODEL_PREFERENCES = (
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5-mini",
    "gpt-5.4-nano",
)
_SEARCH_VOICE_REWRITE_MAX_OUTPUT_TOKENS = 160
_MAX_SEARCH_ATTEMPT_DETAIL_CHARS = 240
_SEARCH_VERIFICATION_STATUSES = frozenset({"verified", "partial", "unverified"})
_SEARCH_TRANSIENT_RETRIES_DEFAULT = 2
_SEARCH_TRANSIENT_RETRY_BASE_DELAY_SECONDS = 0.35
_SEARCH_TRANSIENT_RETRY_MAX_DELAY_SECONDS = 2.5
_SEARCH_FOLLOW_UP_ALLOWED_SCHEMES = frozenset({"http", "https"})
_SEARCH_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
        "metadata",
    }
)
_SEARCH_BLOCKED_HOST_SUFFIXES = (
    ".local",
    ".internal",
    ".home",
    ".home.arpa",
    ".lan",
)
_SEARCH_SPOKEN_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "spoken_answer": {
            "type": "string",
            "maxLength": 320,
            "description": (
                "Short natural spoken answer for a voice assistant. "
                "Everything in this field is read aloud exactly as written. "
                "Use plain sentences only. Do not include markdown, URLs, citations, "
                "source names, brackets, bullets, headings, or meta commentary."
            ),
        },
        "verification_status": {
            "type": "string",
            "enum": sorted(_SEARCH_VERIFICATION_STATUSES),
            "description": (
                "Use verified only when the exact requested detail is currently confirmed. "
                "Use partial when the search found useful context but not the exact requested detail. "
                "Use unverified when current search results did not verify the requested detail."
            ),
        },
        "question_resolved": {
            "type": "boolean",
            "description": (
                "True only when this search result is sufficient as the final answer to the user's request."
            ),
        },
        "site_follow_up_recommended": {
            "type": "boolean",
            "description": (
                "True when checking a specific website could materially clarify the still-unverified detail."
            ),
        },
        "site_follow_up_reason": {
            "type": ["string", "null"],
            "maxLength": 200,
            "description": "Short reason why a website check may help when the result is unresolved.",
        },
        "site_follow_up_url": {
            "type": ["string", "null"],
            "maxLength": 1000,
            "description": "Best concrete website URL to inspect next when follow-up is recommended.",
        },
        "site_follow_up_domain": {
            "type": ["string", "null"],
            "maxLength": 253,
            "description": "Host name for the recommended follow-up website when available.",
        },
    },
    "required": [
        "spoken_answer",
        "verification_status",
        "question_resolved",
        "site_follow_up_recommended",
        "site_follow_up_reason",
        "site_follow_up_url",
        "site_follow_up_domain",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class _StructuredSearchVoiceResult:
    """Capture the structured spoken search result returned by the rewrite step."""

    spoken_answer: str
    verification_status: str
    question_resolved: bool
    site_follow_up_recommended: bool
    site_follow_up_reason: str | None = None
    site_follow_up_url: str | None = None
    site_follow_up_domain: str | None = None


def _collapse_whitespace(value: str | None) -> str:
    """Normalize text into a single-space string."""

    return " ".join(str(value or "").split()).strip()


def _coerce_int(value: Any) -> int | None:
    """Convert a loosely typed value to ``int`` when possible."""

    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_positive_int(value: Any, *, default: int, minimum: int = 1) -> int:
    """Return a positive integer or a bounded default."""

    coerced = _coerce_int(value)
    if coerced is None:
        return default
    return max(minimum, coerced)


def _coerce_bool(value: Any) -> bool | None:
    """Coerce a loosely typed config value into ``bool`` when possible."""

    if isinstance(value, bool):
        return value
    normalized = _collapse_whitespace(None if value is None else str(value)).lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _iter_string_values(value: Any) -> tuple[str, ...]:
    """Return one normalized flat sequence of candidate string values."""

    if value is None:
        return ()
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[\s,;]+", value) if part.strip()]
        return tuple(parts)
    if isinstance(value, Iterable):
        items: list[str] = []
        for item in value:
            normalized = _collapse_whitespace(None if item is None else str(item))
            if normalized:
                items.append(normalized)
        return tuple(items)
    normalized = _collapse_whitespace(str(value))
    return (normalized,) if normalized else ()


def _normalize_search_attempt_detail(value: object) -> str | None:
    """Return a bounded readable detail string for search-attempt journaling."""

    text = _collapse_whitespace(None if value is None else str(value))
    if not text:
        return None
    if len(text) <= _MAX_SEARCH_ATTEMPT_DETAIL_CHARS:
        return text
    return text[: _MAX_SEARCH_ATTEMPT_DETAIL_CHARS - 1].rstrip() + "…"


def _response_status_text(response: Any) -> str | None:
    """Return the normalized top-level response status when available."""

    status = _collapse_whitespace(getattr(response, "status", None)).lower()
    return status or None


def _build_search_attempt(
    *,
    model: str,
    api_path: str,
    max_output_tokens: int | None,
    outcome: str,
    status: str | None = None,
    detail: object = None,
) -> OpenAISearchAttempt:
    """Build one immutable search-attempt record for runtime journaling."""

    return OpenAISearchAttempt(
        model=model,
        api_path=api_path,
        max_output_tokens=max_output_tokens,
        outcome=outcome,
        status=status,
        detail=_normalize_search_attempt_detail(detail),
    )


def _same_search_model_identity(requested_model: str | None, actual_model: str | None) -> bool:
    """Return whether requested and actual model names refer to the same family."""

    requested = _collapse_whitespace(requested_model)
    actual = _collapse_whitespace(actual_model)
    if not requested or not actual:
        return False
    if requested == actual:
        return True
    if actual.startswith(f"{requested}-") or requested.startswith(f"{actual}-"):
        return True
    if requested.endswith("-latest"):
        requested_base = requested[: -len("-latest")].rstrip("-")
        if requested_base and (actual == requested_base or actual.startswith(f"{requested_base}-")):
            return True
    if actual.endswith("-latest"):
        actual_base = actual[: -len("-latest")].rstrip("-")
        if actual_base and (requested == actual_base or requested.startswith(f"{actual_base}-")):
            return True
    return False


def _parse_search_gpt_version(model: str) -> tuple[int, int] | None:
    """Parse a GPT-family model string into ``(major, minor)`` components."""

    match = _GPT_VERSION_RE.match(_collapse_whitespace(model).lower())
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return major, minor


def _normalize_search_context_size(value: Any) -> str | None:
    """Normalize configured search context size to an allowed enum value."""

    normalized = _collapse_whitespace(None if value is None else str(value)).lower()
    if normalized in _VALID_SEARCH_CONTEXT_SIZES:
        return normalized
    return None


def _extract_detail_message(detail: Any) -> str | None:
    """Extract a readable message from SDK error/detail payloads."""

    if detail is None:
        return None
    if isinstance(detail, dict):
        parts = [
            _collapse_whitespace(detail.get("code")),
            _collapse_whitespace(detail.get("reason")),
            _collapse_whitespace(detail.get("message")),
        ]
        text_parts = [part for part in parts if part]
        if text_parts:
            return ": ".join(text_parts)
        raw_text = _collapse_whitespace(str(detail))
        return raw_text or None
    parts = [
        _collapse_whitespace(getattr(detail, "code", None)),
        _collapse_whitespace(getattr(detail, "reason", None)),
        _collapse_whitespace(getattr(detail, "message", None)),
    ]
    text_parts = [part for part in parts if part]
    if text_parts:
        return ": ".join(text_parts)
    raw_text = _collapse_whitespace(str(detail))
    return raw_text or None


def _response_incomplete_detail(response: Any) -> str | None:
    """Return the best available incomplete-detail message from a response."""

    detail = _extract_detail_message(getattr(response, "incomplete_details", None))
    if detail:
        return detail
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message" or getattr(item, "status", None) != "incomplete":
            continue
        item_detail = _extract_detail_message(getattr(item, "incomplete_details", None))
        if item_detail:
            return item_detail
    return None


def _search_responses_text_format() -> dict[str, Any]:
    """Return the Responses API structured-output contract for spoken search."""

    return {
        "type": "json_schema",
        "name": "twinr_live_search_spoken_answer",
        "schema": _SEARCH_SPOKEN_ANSWER_SCHEMA,
        "strict": True,
    }


def _normalize_search_verification_status(value: object) -> str:
    """Normalize and validate the structured search verification status."""

    normalized = _collapse_whitespace(None if value is None else str(value)).lower()
    if normalized not in _SEARCH_VERIFICATION_STATUSES:
        raise ValueError("search structured output is missing a valid verification_status")
    return normalized


def _is_public_hostname(hostname: str | None) -> bool:
    """Return whether a host name is a public routable host candidate."""

    host = _collapse_whitespace(hostname).lower().rstrip(".")
    if not host:
        return False
    if host in _SEARCH_BLOCKED_HOSTNAMES:
        return False
    if any(host.endswith(suffix) for suffix in _SEARCH_BLOCKED_HOST_SUFFIXES):
        return False
    try:
        ip_value = ipaddress.ip_address(host)
    except ValueError:
        return "." in host and "@" not in host and " " not in host
    return not (
        ip_value.is_private
        or ip_value.is_loopback
        or ip_value.is_link_local
        or ip_value.is_multicast
        or ip_value.is_reserved
        or ip_value.is_unspecified
    )


def _normalize_public_hostname(value: str | None) -> str | None:
    """Normalize one public host name hint or return ``None``."""

    normalized = _collapse_whitespace(value).lower()
    if not normalized:
        return None
    if "://" in normalized:
        parsed = urlparse(normalized)
        return _normalize_public_hostname(parsed.hostname)
    candidate = normalized.lstrip("/").rstrip("/")
    if "/" in candidate:
        candidate = candidate.split("/", 1)[0]
    if "@" in candidate:
        return None
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    if ":" in candidate:
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            candidate = candidate.split(":", 1)[0]
    candidate = candidate.strip().rstrip(".")
    if not _is_public_hostname(candidate):
        return None
    return candidate


def _sanitize_public_url(url: str | None) -> str | None:
    """Return a sanitized public http(s) URL or ``None``."""

    normalized_url = _collapse_whitespace(url)
    if not normalized_url:
        return None
    parsed = urlparse(normalized_url)
    scheme = parsed.scheme.lower()
    if scheme not in _SEARCH_FOLLOW_UP_ALLOWED_SCHEMES:
        return None
    if parsed.username or parsed.password:
        return None
    hostname = _normalize_public_hostname(parsed.hostname)
    if hostname is None:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    netloc = hostname if port is None else f"{hostname}:{port}"
    return parsed._replace(netloc=netloc, fragment="").geturl()


def _extract_domain_hint(url: str | None) -> str | None:
    """Return a normalized host name for one candidate follow-up URL."""

    sanitized_url = _sanitize_public_url(url)
    if not sanitized_url:
        return None
    parsed = urlparse(sanitized_url)
    hostname = _collapse_whitespace(parsed.hostname)
    return hostname.lower() or None


def _extract_structured_search_answer(text: str) -> _StructuredSearchVoiceResult:
    """Parse and normalize the structured spoken search payload returned by search."""

    raw_text = str(text or "").strip()
    if not raw_text:
        raise ValueError("search returned empty structured output")
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("search structured output must be a JSON object")
    spoken_answer = _collapse_whitespace(payload.get("spoken_answer"))
    if not spoken_answer:
        raise ValueError("search structured output is missing spoken_answer")
    question_resolved = payload.get("question_resolved")
    if not isinstance(question_resolved, bool):
        raise ValueError("search structured output is missing question_resolved")
    site_follow_up_recommended = payload.get("site_follow_up_recommended")
    if not isinstance(site_follow_up_recommended, bool):
        raise ValueError("search structured output is missing site_follow_up_recommended")
    site_follow_up_reason = _collapse_whitespace(payload.get("site_follow_up_reason")) or None
    site_follow_up_url = _sanitize_public_url(payload.get("site_follow_up_url"))
    site_follow_up_domain = _normalize_public_hostname(payload.get("site_follow_up_domain"))
    normalized_follow_up_domain = _extract_domain_hint(site_follow_up_url) or site_follow_up_domain
    if not site_follow_up_recommended:
        site_follow_up_reason = None
        site_follow_up_url = None
        normalized_follow_up_domain = None
    return _StructuredSearchVoiceResult(
        spoken_answer=spoken_answer,
        verification_status=_normalize_search_verification_status(payload.get("verification_status")),
        question_resolved=question_resolved,
        site_follow_up_recommended=site_follow_up_recommended,
        site_follow_up_reason=site_follow_up_reason,
        site_follow_up_url=site_follow_up_url,
        site_follow_up_domain=normalized_follow_up_domain,
    )


def _parse_context_reference_date(date_context: str | None) -> date | None:
    """Extract an ISO reference date from a date-context string."""

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
    """Return whether text already contains an explicit ISO date."""

    return _DATE_CONTEXT_ISO_RE.search(value) is not None


def _date_phrase_for_pattern(pattern: re.Pattern[str], resolved_date: str) -> str:
    """Return a locale-appropriate replacement phrase for a resolved date."""

    if pattern in _ENGLISH_RELATIVE_PATTERNS:
        return f"on {resolved_date}"
    return f"am {resolved_date}"


def _should_skip_ambiguous_german_morgen(question: str, match: re.Match[str]) -> bool:
    """Avoid rewriting the time-of-day phrase ``heute morgen`` as tomorrow."""

    before = question[: match.start()].strip().lower()
    return before.endswith("heute")


def _replace_relative_day_reference(question: str, reference_date: date) -> str | None:
    """Replace one relative-day phrase with the authoritative explicit ISO date."""

    resolved_date = reference_date.isoformat()
    for patterns, _day_offset in _RELATIVE_DAY_PATTERN_GROUPS:
        for pattern in patterns:
            match = pattern.search(question)
            if match is None:
                continue
            if pattern is _TOMORROW_PATTERNS[0] and _should_skip_ambiguous_german_morgen(question, match):
                continue
            return pattern.sub(_date_phrase_for_pattern(pattern, resolved_date), question, count=1)
    return None


def _replace_next_weekday_reference(question: str, reference_date: date) -> str | None:
    """Replace one next-weekday phrase with the authoritative explicit ISO date."""

    resolved_date = reference_date.isoformat()
    for pattern, is_english in ((_NEXT_WEEKDAY_EN_RE, True), (_NEXT_WEEKDAY_DE_RE, False)):
        match = pattern.search(question)
        if match is None:
            continue
        weekday_name = match.group(1).strip().lower()
        target_weekday = _WEEKDAY_INDEX_BY_NAME.get(weekday_name)
        if target_weekday is None:
            continue
        replacement = f"on {resolved_date}" if is_english else f"am {resolved_date}"
        return pattern.sub(replacement, question, count=1)
    return None


def _resolved_explicit_search_query(question: str, date_context: str | None) -> str | None:
    """Return an explicit-date variant of a search question when applicable."""

    normalized_question = _collapse_whitespace(question)
    if not normalized_question:
        return None
    if _contains_explicit_iso_date(normalized_question):
        return normalized_question
    reference_date = _parse_context_reference_date(date_context)
    if reference_date is None:
        return None

    replaced_weekday = _replace_next_weekday_reference(normalized_question, reference_date)
    if replaced_weekday is not None and replaced_weekday != normalized_question:
        return replaced_weekday

    replaced_relative_day = _replace_relative_day_reference(normalized_question, reference_date)
    if replaced_relative_day is not None and replaced_relative_day != normalized_question:
        return replaced_relative_day
    return None


class OpenAISearchMixin:
    """Provide OpenAI-backed live-search helpers for Twinr backends.

    The mixin focuses on building date-stable search prompts, selecting a
    compatible search-capable model, and returning answer text with verified
    source metadata.
    """

    def search_live_info_with_metadata(
        self,
        question: str,
        *,
        conversation: ConversationLike | None = None,
        location_hint: str | None = None,
        date_context: str | None = None,
    ) -> OpenAISearchResult:
        """Answer a live-information question using OpenAI web search.

        Args:
            question: User question that requires fresh information.
            conversation: Optional recent conversation context.
            location_hint: Optional city or location context for search.
            date_context: Optional explicit local date/time context string.

        Returns:
            An ``OpenAISearchResult`` with answer text, sources, and metadata.

        Raises:
            RuntimeError: If no usable live answer can be produced.
        """

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
        candidate_models = self._candidate_search_models()
        requested_model = candidate_models[0] if candidate_models else None
        attempt_log: list[OpenAISearchAttempt] = []

        for model in candidate_models:
            if self._is_search_preview_model(model):
                if not self._search_preview_supported():
                    attempt_log.append(
                        _build_search_attempt(
                            model=model,
                            api_path="preview",
                            max_output_tokens=None,
                            outcome="skipped",
                            detail="preview_api_unavailable",
                        )
                    )
                    continue
                for max_output_tokens in output_token_candidates:
                    try:
                        candidate, is_complete, attempt = self._search_with_preview_model(
                            model,
                            prompt,
                            conversation=search_conversation,
                            instructions=instructions,
                            location_hint=location_hint,
                            max_output_tokens=max_output_tokens,
                        )
                        attempt_log.append(attempt)
                    except Exception as exc:  # pragma: no cover - defensive runtime fallback
                        last_error = exc
                        attempt_log.append(
                            _build_search_attempt(
                                model=model,
                                api_path="preview",
                                max_output_tokens=max_output_tokens,
                                outcome="error",
                                detail=f"{type(exc).__name__}: {exc}",
                            )
                        )
                        continue
                    if candidate.answer and is_complete:
                        return self._rewrite_search_result_for_voice(
                            question=normalized_question,
                            candidate=self._finalize_search_result(
                                candidate,
                                requested_model=requested_model,
                                attempt_log=attempt_log,
                            ),
                        )
                continue

            response_candidate, model_error, model_attempts = self._search_with_responses_model(
                model,
                prompt,
                conversation=search_conversation,
                instructions=instructions,
                location_hint=location_hint,
                output_token_candidates=output_token_candidates,
            )
            attempt_log.extend(model_attempts)
            if response_candidate is not None:
                return self._rewrite_search_result_for_voice(
                    question=normalized_question,
                    candidate=self._finalize_search_result(
                        response_candidate,
                        requested_model=requested_model,
                        attempt_log=attempt_log,
                    ),
                )
            if model_error is not None:
                last_error = model_error

        if last_error is not None:
            raise RuntimeError(
                f"OpenAI web search failed after exhausting configured models and retries: {last_error}"
            ) from last_error
        raise RuntimeError("OpenAI web search returned no usable answer text")

    def _finalize_search_result(
        self,
        candidate: OpenAISearchResult,
        *,
        requested_model: str | None,
        attempt_log: list[OpenAISearchAttempt] | tuple[OpenAISearchAttempt, ...],
    ) -> OpenAISearchResult:
        """Attach requested-model and fallback metadata to a successful result."""

        normalized_attempt_log = tuple(attempt_log)
        return replace(
            candidate,
            requested_model=requested_model,
            fallback_reason=self._derive_search_fallback_reason(
                requested_model=requested_model,
                actual_model=candidate.model,
                attempt_log=normalized_attempt_log,
            ),
            attempt_log=normalized_attempt_log,
        )

    def _derive_search_fallback_reason(
        self,
        *,
        requested_model: str | None,
        actual_model: str | None,
        attempt_log: tuple[OpenAISearchAttempt, ...],
    ) -> str | None:
        """Describe why a successful search result left its requested primary model."""

        normalized_requested = _collapse_whitespace(requested_model)
        normalized_actual = _collapse_whitespace(actual_model)
        if not normalized_requested or _same_search_model_identity(normalized_requested, normalized_actual):
            return None
        for attempt in attempt_log:
            if attempt.model != normalized_requested:
                continue
            if attempt.outcome == "success":
                continue
            detail = attempt.detail or attempt.status or attempt.outcome
            return _normalize_search_attempt_detail(
                f"{normalized_requested}->{normalized_actual}: {attempt.outcome}: {detail}"
            )
        return _normalize_search_attempt_detail(
            f"{normalized_requested}->{normalized_actual}: fallback_without_primary_success"
        )

    def _search_with_preview_model(
        self,
        model: str,
        prompt: str,
        *,
        conversation: ConversationLike | None,
        instructions: str,
        location_hint: str | None,
        max_output_tokens: int,
    ) -> tuple[OpenAISearchResult, bool, OpenAISearchAttempt]:
        """Run a search request through a preview chat-completions model."""

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

        response = self._call_with_transient_retries(
            lambda: self._client.chat.completions.create(
                model=model,
                web_search_options=self._build_preview_web_search_options(location_hint=location_hint),
                max_completion_tokens=max_output_tokens,
                messages=messages,
                store=self._search_store_enabled(),
            )
        )
        message = response.choices[0].message if getattr(response, "choices", None) else None
        answer = _collapse_whitespace((getattr(message, "content", "") or "").replace("\r", " ").replace("\n", " "))
        result = OpenAISearchResult(
            answer=answer,
            sources=self._extract_preview_search_sources(message),
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, model),
            token_usage=self._extract_preview_usage(response),
            used_web_search=True,
        )
        is_complete = self._preview_response_is_complete(response)
        attempt = _build_search_attempt(
            model=model,
            api_path="preview",
            max_output_tokens=max_output_tokens,
            outcome="success" if result.answer and is_complete else "unusable",
            status=_collapse_whitespace(getattr(response.choices[0], "finish_reason", None)).lower()
            if getattr(response, "choices", None)
            else None,
            detail=None if result.answer else "answer_text=blank",
        )
        return result, is_complete, attempt

    def _search_with_responses_model(
        self,
        model: str,
        prompt: str,
        *,
        conversation: ConversationLike | None,
        instructions: str,
        location_hint: str | None,
        output_token_candidates: tuple[int, ...],
    ) -> tuple[OpenAISearchResult | None, Exception | None, tuple[OpenAISearchAttempt, ...]]:
        """Run a search request through a Responses web-search model."""

        pending_output_tokens = list(output_token_candidates)
        attempted_output_tokens: set[int] = set()
        last_error: Exception | None = None
        attempt_log: list[OpenAISearchAttempt] = []

        while pending_output_tokens:
            max_output_tokens = pending_output_tokens.pop(0)
            if max_output_tokens in attempted_output_tokens:
                continue
            attempted_output_tokens.add(max_output_tokens)
            try:
                request = self._build_response_request(
                    prompt,
                    conversation=conversation,
                    instructions=instructions,
                    allow_web_search=True,
                    model=model,
                    reasoning_effort="low",
                    max_output_tokens=max_output_tokens,
                    prompt_cache_scope="search",
                )
                self._apply_web_search_request_overrides(request, location_hint=location_hint)
                self._apply_search_request_output_controls(request, model=model)
                self._ensure_web_search_sources_included(request)
                request["store"] = self._search_store_enabled()
                response = self._call_with_transient_retries(lambda: self._client.responses.create(**request))
                candidate = OpenAISearchResult(
                    answer=_collapse_whitespace(self._extract_output_text(response).replace("\r", " ").replace("\n", " ")),
                    sources=self._extract_web_search_sources(response),
                    response_id=getattr(response, "id", None),
                    request_id=getattr(response, "_request_id", None),
                    model=extract_model_name(response, model),
                    token_usage=extract_token_usage(response),
                    used_web_search=self._used_web_search(response),
                )
            except Exception as exc:  # pragma: no cover - defensive runtime fallback
                last_error = exc
                attempt_log.append(
                    _build_search_attempt(
                        model=model,
                        api_path="responses",
                        max_output_tokens=max_output_tokens,
                        outcome="error",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            if candidate.answer and candidate.used_web_search and self._response_is_complete(response):
                attempt_log.append(
                    _build_search_attempt(
                        model=model,
                        api_path="responses",
                        max_output_tokens=max_output_tokens,
                        outcome="success",
                        status=_response_status_text(response),
                    )
                )
                return candidate, None, tuple(attempt_log)

            retry_reason = self._search_budget_retry_reason(
                response,
                answer=candidate.answer,
                used_web_search=candidate.used_web_search,
                model=model,
            )
            retry_budget = self._next_search_retry_max_output_tokens(
                response,
                current_max_output_tokens=max_output_tokens,
                output_token_candidates=output_token_candidates,
                answer=candidate.answer,
                used_web_search=candidate.used_web_search,
                model=model,
            )
            if retry_budget is not None:
                attempt_log.append(
                    _build_search_attempt(
                        model=model,
                        api_path="responses",
                        max_output_tokens=max_output_tokens,
                        outcome="retry",
                        status=_response_status_text(response),
                        detail=retry_reason,
                    )
                )
                if retry_budget not in attempted_output_tokens and retry_budget not in pending_output_tokens:
                    pending_output_tokens.insert(0, retry_budget)
                continue

            if self._response_is_complete(response) and not candidate.used_web_search:
                last_error = RuntimeError(
                    f"OpenAI web search response unusable for model={model!r}; "
                    f"max_output_tokens={max_output_tokens}; status={_response_status_text(response)!r}; "
                    "used_web_search=false"
                )
                attempt_log.append(
                    _build_search_attempt(
                        model=model,
                        api_path="responses",
                        max_output_tokens=max_output_tokens,
                        outcome="unusable",
                        status=_response_status_text(response),
                        detail="used_web_search=false",
                    )
                )
                continue

            last_error = self._build_search_response_error(
                model=model,
                response=response,
                max_output_tokens=max_output_tokens,
                answer=candidate.answer,
                used_web_search=candidate.used_web_search,
            )
            attempt_log.append(
                _build_search_attempt(
                    model=model,
                    api_path="responses",
                    max_output_tokens=max_output_tokens,
                    outcome="unusable",
                    status=_response_status_text(response),
                    detail=last_error,
                )
            )

        return None, last_error, tuple(attempt_log)

    def _rewrite_search_result_for_voice(
        self,
        *,
        question: str,
        candidate: OpenAISearchResult,
    ) -> OpenAISearchResult:
        """Rewrite a search result into clean spoken output plus verification metadata."""

        rendered_sources = ", ".join(candidate.sources) if candidate.sources else "none"
        prompt = (
            f"Original user question: {question}\n"
            f"Live-search result: {candidate.answer}\n"
            f"Source URLs: {rendered_sources}\n"
            "Rewrite the search result as what Twinr should say aloud now."
        )
        instructions = (
            "You are Twinr preparing a spoken answer for a voice assistant. "
            "Everything you write will be read aloud exactly as written. "
            "Use only facts already present in the live-search result. "
            "Do not add facts, do not ask a follow-up question unless the live-search result itself lacks a key fact, "
            "and do not mention sources, URLs, markdown, citations, brackets, or reference phrases. "
            "Keep the reply short, natural, warm, spoken, senior-friendly, and usually to one or two short sentences. "
            "Avoid stiff, bureaucratic, or institutional wording. "
            "Prefer plain spoken wording over formal written-report phrasing or generic advisories unless the live-search result itself requires that wording. "
            "Return a JSON object with the spoken answer plus verification metadata. "
            "Mark verification_status as verified only when the exact requested detail is confirmed. "
            "A missing clear current hint, older menu, general page, or absent search-result evidence is not a verified negative answer. "
            "Mark those cases as partial or unverified instead of verified. "
            "Mark it as partial when the search found useful context but not the exact requested detail. "
            "Mark it as unverified when the current search did not verify the requested detail at all. "
            "Set question_resolved true only when Twinr can safely treat the result as a final resolution without a deeper website check. "
            "Set site_follow_up_recommended true when checking a specific website could materially clarify the unresolved detail, especially for a place, business, or organization where an official site exists but the current detail was not clearly verified from search results alone. "
            "When site_follow_up_recommended is true and one source clearly looks like the primary or official site, include that URL and host; otherwise keep them null. "
            "Keep site_follow_up_reason short and concrete."
        )
        last_error: Exception | None = None
        for model in self._structured_rewrite_models():
            try:
                request = self._build_response_request(
                    prompt,
                    conversation=None,
                    instructions=instructions,
                    allow_web_search=False,
                    model=model,
                    reasoning_effort="low",
                    max_output_tokens=_SEARCH_VOICE_REWRITE_MAX_OUTPUT_TOKENS,
                    prompt_cache_scope="search_voice",
                )
                request["store"] = self._search_store_enabled()
                request["text"] = {"format": _search_responses_text_format()}
                response = self._call_with_transient_retries(lambda: self._client.responses.create(**request))
                structured_result = _extract_structured_search_answer(self._extract_output_text(response))
                if structured_result.spoken_answer:
                    return OpenAISearchResult(
                        answer=structured_result.spoken_answer,
                        sources=candidate.sources,
                        response_id=candidate.response_id,
                        request_id=candidate.request_id,
                        model=candidate.model,
                        token_usage=candidate.token_usage,
                        used_web_search=candidate.used_web_search,
                        requested_model=candidate.requested_model,
                        fallback_reason=candidate.fallback_reason,
                        attempt_log=candidate.attempt_log,
                        verification_status=structured_result.verification_status,
                        question_resolved=structured_result.question_resolved,
                        site_follow_up_recommended=structured_result.site_follow_up_recommended,
                        site_follow_up_reason=structured_result.site_follow_up_reason,
                        site_follow_up_url=structured_result.site_follow_up_url,
                        site_follow_up_domain=structured_result.site_follow_up_domain,
                    )
            except Exception as exc:  # pragma: no cover - defensive runtime fallback
                last_error = exc
                continue
        if last_error is not None:
            return candidate
        return candidate

    def _build_search_prompt(
        self,
        question: str,
        *,
        location_hint: str | None,
        date_context: str | None,
    ) -> str:
        """Return a structured search prompt with explicit place/date context."""

        normalized_question = _collapse_whitespace(question)
        explicit_query = _resolved_explicit_search_query(normalized_question, date_context)
        normalized_location = _collapse_whitespace(location_hint)
        normalized_date_context = _collapse_whitespace(date_context)

        if not normalized_location and not normalized_date_context and explicit_query is None:
            return normalized_question

        lines = [f"User question: {normalized_question}"]
        if explicit_query is not None and explicit_query != normalized_question:
            lines.append(f"Resolved explicit-date variant: {explicit_query}")
        if normalized_location:
            lines.append(f"Explicit place context: {normalized_location}")
        if normalized_date_context:
            lines.append(f"Explicit local date/time context: {normalized_date_context}")
        lines.append(
            "Answer the user's actual request and use the explicit place/date context only to disambiguate partial wording, ASR noise, or relative dates."
        )
        return "\n".join(lines)

    def _extract_web_search_sources(self, response: Any) -> tuple[str, ...]:
        """Extract and deduplicate source URLs from a Responses search result."""

        urls: list[str] = []
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) != "web_search_call":
                continue
            action = getattr(item, "action", None)
            sources = getattr(action, "sources", None) or []
            for source in sources:
                url = _collapse_whitespace(getattr(source, "url", None))
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
        """Return the current local date/time context for relative-date searches."""

        timezone_name = self._configured_timezone_name()
        effective_label: str
        try:
            if timezone_name:
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
        """Return the ordered list of search model candidates to try."""

        candidates: list[str] = []
        for candidate in (
            getattr(self.config, "openai_search_model", None),
            getattr(self.config, "default_model", None),
            DEFAULT_OPENAI_MAIN_MODEL,
            *_FRONTIER_SEARCH_MODEL_PREFERENCES,
            *SEARCH_MODEL_FALLBACKS,
        ):
            normalized = _collapse_whitespace(candidate)
            if normalized and normalized not in candidates:
                candidates.append(normalized)
        if not candidates:
            candidates.append(_FRONTIER_DEFAULT_SEARCH_MODEL)
        return tuple(candidates)

    def _structured_rewrite_models(self) -> tuple[str, ...]:
        """Return ordered rewrite models that are likely to support structured output."""

        candidates: list[str] = []
        preferred_model = _collapse_whitespace(getattr(self.config, "default_model", None))
        for candidate in (
            preferred_model,
            *_FRONTIER_SEARCH_MODEL_PREFERENCES,
            DEFAULT_OPENAI_MAIN_MODEL,
            *SEARCH_MODEL_FALLBACKS,
        ):
            normalized = _collapse_whitespace(candidate)
            if not normalized or normalized in candidates:
                continue
            if not self._model_likely_supports_structured_outputs(normalized):
                continue
            candidates.append(normalized)
        if not candidates:
            candidates.append(_FRONTIER_DEFAULT_SEARCH_MODEL)
        return tuple(candidates)

    def _model_likely_supports_structured_outputs(self, model: str) -> bool:
        """Return a conservative structured-output capability heuristic."""

        normalized = _collapse_whitespace(model).lower()
        return "-pro" not in normalized

    def _is_search_preview_model(self, model: str) -> bool:
        """Return whether a model name should use the preview search path."""

        normalized = _collapse_whitespace(model).lower()
        return "search-preview" in normalized or normalized.endswith("-search-api")

    def _search_preview_supported(self) -> bool:
        """Return whether the client exposes the preview search API surface."""

        chat = getattr(self._client, "chat", None)
        completions = getattr(chat, "completions", None)
        create = getattr(completions, "create", None)
        return callable(create)

    def _extract_preview_search_sources(self, message: Any) -> tuple[str, ...]:
        """Extract and deduplicate URL citations from a preview response message."""

        urls: list[str] = []
        for annotation in getattr(message, "annotations", None) or []:
            url_citation = getattr(annotation, "url_citation", None)
            url = _collapse_whitespace(getattr(url_citation, "url", None))
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
        """Map preview-response usage data into Twinr's token-usage type."""

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
        """Return whether a Responses payload contains an incomplete message."""

        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) == "message" and getattr(item, "status", None) == "incomplete":
                return True
        return False

    # BREAKING: search conversation snippets now preserve assistant phase when available;
    # callers that assumed only ``(role, content)`` tuples must also accept ``(role, content, phase)``.
    def _prepare_search_conversation(
        self,
        conversation: ConversationLike | None,
    ) -> tuple[tuple[str, str, str | None], ...] | None:
        """Trim conversation history down to the search-specific context window."""

        if not conversation:
            return None
        filtered: list[tuple[str, str, str | None]] = []
        for item in conversation:
            role, content, phase = self._coerce_message(item)
            normalized_role = role.strip().lower()
            if normalized_role not in {"user", "assistant"}:
                continue
            normalized_content = _collapse_whitespace(content)
            if not normalized_content:
                continue
            if len(normalized_content) > _SEARCH_CONTEXT_CHAR_LIMIT:
                normalized_content = normalized_content[: _SEARCH_CONTEXT_CHAR_LIMIT - 1].rstrip() + "…"
            normalized_phase = _collapse_whitespace(phase).lower() or None
            filtered.append((normalized_role, normalized_content, normalized_phase))
        if not filtered:
            return None
        trimmed = filtered[-_SEARCH_CONTEXT_MAX_TURNS :]
        if len(trimmed) > 1 and trimmed[0][0] == "assistant":
            trimmed = trimmed[1:]
        return tuple(trimmed) if trimmed else None

    def _configured_timezone_name(self) -> str | None:
        """Return the configured timezone used for search date grounding."""

        timezone_name = _collapse_whitespace(
            getattr(self.config, "openai_web_search_timezone", None)
            or getattr(self.config, "local_timezone_name", None)
        )
        return timezone_name or None

    def _search_output_token_candidates(self) -> tuple[int, ...]:
        """Return the output-token limits used for primary and retry search attempts."""

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

    def _next_search_retry_max_output_tokens(
        self,
        response: Any,
        *,
        current_max_output_tokens: int,
        output_token_candidates: tuple[int, ...],
        answer: str,
        used_web_search: bool,
        model: str,
    ) -> int | None:
        """Return the next larger retry budget for token-exhausted search responses."""

        if self._search_budget_retry_reason(
            response,
            answer=answer,
            used_web_search=used_web_search,
            model=model,
        ) is None:
            return None
        larger_budgets = sorted({*output_token_candidates, *_SEARCH_OUTPUT_TOKEN_RETRY_LADDER})
        for budget in larger_budgets:
            if budget > current_max_output_tokens:
                return budget
        return None

    def _search_budget_retry_reason(
        self,
        response: Any,
        *,
        answer: str,
        used_web_search: bool,
        model: str,
    ) -> str | None:
        """Return why the same model deserves a larger search budget."""

        if self._response_hit_max_output_tokens_limit(response):
            return "max_output_tokens"
        if (
            used_web_search
            and not answer
            and self._response_is_complete(response)
            and self._model_supports_blank_answer_retry(model)
        ):
            return "answer_text=blank"
        return None

    def _model_supports_blank_answer_retry(self, model: str) -> bool:
        """Return whether blank completed search responses deserve the retry ladder."""

        version = _parse_search_gpt_version(model)
        if version is None:
            return False
        major, minor = version
        return major > 5 or (major == 5 and minor >= 4)

    def _response_hit_max_output_tokens_limit(self, response: Any) -> bool:
        """Return whether a search response stopped because max_output_tokens was exhausted."""

        status = _collapse_whitespace(getattr(response, "status", None)).lower()
        if status != "incomplete":
            return False
        incomplete_detail = (_response_incomplete_detail(response) or "").lower()
        return "max_output_tokens" in incomplete_detail

    def _build_search_response_error(
        self,
        *,
        model: str,
        response: Any,
        max_output_tokens: int,
        answer: str,
        used_web_search: bool,
    ) -> RuntimeError:
        """Return an informative error for an unusable search response."""

        parts = [
            f"OpenAI web search response unusable for model={model!r}",
            f"max_output_tokens={max_output_tokens}",
        ]
        status = _collapse_whitespace(getattr(response, "status", None)).lower()
        if status:
            parts.append(f"status={status!r}")
        incomplete_detail = _response_incomplete_detail(response)
        if incomplete_detail:
            parts.append(f"incomplete={incomplete_detail}")
        error_detail = _extract_detail_message(getattr(response, "error", None))
        if error_detail:
            parts.append(f"error={error_detail}")
        parts.append(f"used_web_search={str(bool(used_web_search)).lower()}")
        if answer:
            parts.append("answer_text=partial")
        else:
            parts.append("answer_text=blank")
        return RuntimeError("; ".join(parts))

    def _response_is_complete(self, response: Any) -> bool:
        """Return whether a Responses search result completed successfully."""

        status = _collapse_whitespace(getattr(response, "status", None)).lower()
        if status and status != "completed":
            return False
        return not self._response_has_incomplete_message(response)

    def _preview_response_is_complete(self, response: Any) -> bool:
        """Return whether a preview search response stopped cleanly."""

        choices = getattr(response, "choices", None) or []
        if not choices:
            return False
        finish_reason = _collapse_whitespace(getattr(choices[0], "finish_reason", None)).lower()
        return finish_reason in {"", "stop"}

    def _apply_web_search_request_overrides(self, request: dict[str, Any], *, location_hint: str | None) -> None:
        """Apply search-context, privacy, and location overrides to a request payload."""

        tools = request.get("tools")
        if not isinstance(tools, list):
            return
        request["tool_choice"] = "required"
        search_context_size = _normalize_search_context_size(
            getattr(self.config, "openai_web_search_context_size", None)
        )
        user_location = self._build_responses_web_search_user_location(location_hint=location_hint)
        allowed_domains = self._configured_search_allowed_domains()
        external_web_access = self._configured_search_external_web_access()
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_type = _collapse_whitespace(tool.get("type")).lower()
            if tool_type not in {"web_search", "web_search_preview"}:
                continue
            if search_context_size is not None:
                tool["search_context_size"] = search_context_size
            if tool_type == "web_search":
                if user_location is not None:
                    tool["user_location"] = user_location
                else:
                    tool.pop("user_location", None)
                if allowed_domains:
                    existing_filters = tool.get("filters")
                    if not isinstance(existing_filters, dict):
                        existing_filters = {}
                    existing_filters["allowed_domains"] = list(allowed_domains)
                    tool["filters"] = existing_filters
                if external_web_access is not None:
                    tool["external_web_access"] = external_web_access
            break

    def _apply_search_request_output_controls(self, request: dict[str, Any], *, model: str) -> None:
        """Apply lower-latency output controls to plain-text GPT-5 search requests."""

        normalized_model = _collapse_whitespace(model).lower()
        if not normalized_model.startswith("gpt-5"):
            return
        text_payload = request.get("text")
        if text_payload is None:
            request["text"] = {"format": {"type": "text"}, "verbosity": "low"}
            return
        if not isinstance(text_payload, dict):
            return
        format_payload = text_payload.get("format")
        if isinstance(format_payload, dict):
            format_type = _collapse_whitespace(format_payload.get("type")).lower()
            if format_type == "text":
                text_payload.setdefault("verbosity", "low")

    def _ensure_web_search_sources_included(self, request: dict[str, Any]) -> None:
        """Ensure source URLs are requested from the Responses API."""

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
        """Build a Responses API ``user_location`` payload when possible."""

        city = _collapse_whitespace(location_hint)
        if not city:
            return None
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
        """Build preview-model web-search options including approximate location."""

        options: dict[str, Any] = {}
        search_context_size = _normalize_search_context_size(
            getattr(self.config, "openai_web_search_context_size", None)
        )
        if search_context_size is not None:
            options["search_context_size"] = search_context_size

        city = _collapse_whitespace(location_hint)
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

    def _configured_search_allowed_domains(self) -> tuple[str, ...]:
        """Return a normalized allowlist of domains for Responses web search."""

        raw_value = getattr(self.config, "openai_web_search_allowed_domains", None)
        if raw_value is None:
            raw_value = getattr(self.config, "openai_search_allowed_domains", None)
        normalized_domains: list[str] = []
        for item in _iter_string_values(raw_value):
            normalized = _normalize_public_hostname(item)
            if normalized and normalized not in normalized_domains:
                normalized_domains.append(normalized)
        return tuple(normalized_domains[:100])

    def _configured_search_external_web_access(self) -> bool | None:
        """Return the configured external_web_access flag for Responses web search."""

        configured = _coerce_bool(getattr(self.config, "openai_web_search_external_web_access", None))
        if configured is not None:
            return configured
        return _coerce_bool(getattr(self.config, "openai_search_external_web_access", None))

    def _search_store_enabled(self) -> bool:
        """Return whether provider-side request/response storage is allowed for search."""

        configured = _coerce_bool(getattr(self.config, "openai_search_store", None))
        if configured is None:
            configured = _coerce_bool(getattr(self.config, "openai_store_search_responses", None))
        return bool(configured) if configured is not None else False

    def _search_transient_retry_attempts(self) -> int:
        """Return how many transient retries are allowed per API request."""

        return _coerce_positive_int(
            getattr(self.config, "openai_search_transient_retries", None),
            default=_SEARCH_TRANSIENT_RETRIES_DEFAULT,
            minimum=0,
        )

    def _is_transient_search_exception(self, exc: Exception) -> bool:
        """Return whether an SDK exception looks transient and retryable."""

        status_code = _coerce_int(getattr(exc, "status_code", None))
        if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
        type_name = type(exc).__name__.lower()
        if any(
            token in type_name
            for token in (
                "timeout",
                "connection",
                "rate",
                "internalserver",
                "serviceunavailable",
                "apiconnection",
            )
        ):
            return True
        message = _collapse_whitespace(str(exc)).lower()
        return any(
            token in message
            for token in (
                "timeout",
                "timed out",
                "connection",
                "temporarily unavailable",
                "rate limit",
                "try again",
                "server error",
                "overloaded",
            )
        )

    def _transient_retry_delay_seconds(self, retry_index: int) -> float:
        """Return one bounded exponential-backoff delay with small jitter."""

        base = _SEARCH_TRANSIENT_RETRY_BASE_DELAY_SECONDS * (2**retry_index)
        jitter = random.uniform(0.0, 0.15)
        return min(_SEARCH_TRANSIENT_RETRY_MAX_DELAY_SECONDS, base + jitter)

    def _call_with_transient_retries(self, operation):
        """Execute one API operation with bounded transient retries."""

        max_retries = self._search_transient_retry_attempts()
        for retry_index in range(max_retries + 1):
            try:
                return operation()
            except Exception as exc:
                if retry_index >= max_retries or not self._is_transient_search_exception(exc):
                    raise
                time.sleep(self._transient_retry_delay_seconds(retry_index))
        raise RuntimeError("unreachable transient retry state")
