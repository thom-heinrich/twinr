"""Refine raw nightly signals into user-facing morning digest content.

The overnight pipeline already produces raw/internal material such as
awareness threads, accepted personality deltas, and optional live-web search
results. This module adds a second bounded shaping layer so the morning-facing
digest does not leak raw provider/search artifacts directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
from typing import Protocol
from urllib.parse import urlparse, urlunparse

from twinr.proactive.runtime.display_reserve_support import compact_text

_MAX_LINE = 160
_MAX_SUMMARY = 480
_MAX_URL = 1024
_DEFAULT_HEADLINE_LIMIT = 3
_DEFAULT_INSIGHT_LIMIT = 2
_DEFAULT_CONTINUITY_LIMIT = 2
_DEFAULT_SOURCE_LIMIT = 5


class _ComposeLike(Protocol):
    """Protocol-like callable type for backend-backed text composition."""

    def __call__(self, *, prompt: str, max_len: int) -> str:  # pragma: no cover - protocol shape only
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class NightlyDigestRefinement:
    """Carry the user-facing nightly digest content after refinement."""

    headline_lines: tuple[str, ...] = ()
    live_news_summary: str | None = None
    news_sources: tuple[str, ...] = ()
    new_insights: tuple[str, ...] = ()
    continuity_shifts: tuple[str, ...] = ()


def _bounded_text(value: object | None, *, max_len: int) -> str:
    """Collapse one arbitrary value into bounded single-line text."""

    return compact_text(value, max_len=max_len) if value is not None else ""


def _bounded_optional_text(value: object | None, *, max_len: int) -> str | None:
    """Return one bounded line or ``None`` when empty."""

    text = _bounded_text(value, max_len=max_len)
    return text or None


def _bounded_block_text(value: object | None, *, max_len: int) -> str:
    """Return bounded text while preserving line breaks for line-oriented parsing."""

    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len]


def _bounded_lines(
    values: Iterable[object],
    *,
    limit: int,
    max_len: int,
) -> tuple[str, ...]:
    """Normalize, deduplicate, and bound an ordered line sequence."""

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        line = _bounded_text(value, max_len=max_len)
        if not line:
            continue
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(line)
        if len(normalized) >= max(1, int(limit)):
            break
    return tuple(normalized)


def _canonical_http_url(value: object | None) -> str | None:
    """Normalize one candidate URL to a bounded public http(s) string."""

    raw = _bounded_text(value, max_len=_MAX_URL)
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    normalized = urlunparse(parsed._replace(fragment="")).rstrip("/")
    return normalized or None


def _select_fallback_sources(
    candidate_sources: Sequence[str],
    *,
    limit: int,
) -> tuple[str, ...]:
    """Return the first bounded set of normalized source URLs."""

    selected: list[str] = []
    seen: set[str] = set()
    for value in candidate_sources:
        url = _canonical_http_url(value)
        if url is None or url in seen:
            continue
        seen.add(url)
        selected.append(url)
        if len(selected) >= max(1, int(limit)):
            break
    return tuple(selected)


def _compose_lines(
    compose: _ComposeLike | None,
    *,
    prompt: str,
    limit: int,
    max_len: int,
) -> tuple[str, ...]:
    """Run one backend composition request and parse it as plain lines."""

    if compose is None:
        return ()
    rendered = _bounded_block_text(compose(prompt=prompt, max_len=max_len), max_len=max_len)
    if not rendered:
        return ()
    return _bounded_lines(rendered.splitlines(), limit=limit, max_len=_MAX_LINE)


def _compose_summary(
    compose: _ComposeLike | None,
    *,
    prompt: str,
    max_len: int,
) -> str | None:
    """Run one backend composition request and normalize it as a paragraph."""

    if compose is None:
        return None
    return _bounded_optional_text(compose(prompt=prompt, max_len=max_len), max_len=max_len)


def _compose_sources(
    compose: _ComposeLike | None,
    *,
    prompt: str,
    candidate_sources: Sequence[str],
    limit: int,
) -> tuple[str, ...]:
    """Select supporting URLs from the provided candidate list."""

    normalized_candidates = _select_fallback_sources(candidate_sources, limit=max(limit, len(candidate_sources)))
    if compose is None or not normalized_candidates:
        return _select_fallback_sources(normalized_candidates, limit=limit)
    rendered = _bounded_block_text(compose(prompt=prompt, max_len=limit * 220), max_len=limit * 220)
    if not rendered:
        return _select_fallback_sources(normalized_candidates, limit=limit)
    allowed = {candidate: candidate for candidate in normalized_candidates}
    selected: list[str] = []
    seen: set[str] = set()
    for raw_line in rendered.splitlines():
        candidate = _canonical_http_url(raw_line)
        if candidate is None:
            continue
        resolved = allowed.get(candidate)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        selected.append(resolved)
        if len(selected) >= max(1, int(limit)):
            break
    return tuple(selected) if selected else _select_fallback_sources(normalized_candidates, limit=limit)


def _fallback_headlines(
    raw_headline_lines: Sequence[str],
    *,
    raw_live_news_summary: str | None,
    limit: int,
) -> tuple[str, ...]:
    """Return deterministic headline lines when model refinement is unavailable."""

    if raw_headline_lines:
        return _bounded_lines(raw_headline_lines, limit=limit, max_len=_MAX_LINE)
    if raw_live_news_summary:
        return _bounded_lines((raw_live_news_summary,), limit=1, max_len=_MAX_LINE)
    return ()


def _fallback_news_summary(
    headline_lines: Sequence[str],
    *,
    raw_live_news_summary: str | None,
) -> str | None:
    """Return a deterministic calm summary when the backend is unavailable."""

    if headline_lines:
        if len(headline_lines) == 1:
            return _bounded_optional_text(f"Aktuell wichtig: {headline_lines[0]}.", max_len=_MAX_SUMMARY)
        return _bounded_optional_text(
            f"Aktuell wichtig: {headline_lines[0]}. Außerdem: {headline_lines[1]}.",
            max_len=_MAX_SUMMARY,
        )
    return _bounded_optional_text(raw_live_news_summary, max_len=_MAX_SUMMARY)


def _news_lines_prompt(
    *,
    language: str | None,
    target_day_text: str,
    location_hint: str | None,
    raw_headline_lines: Sequence[str],
    raw_live_news_summary: str | None,
    candidate_sources: Sequence[str],
    limit: int,
) -> str:
    """Build the prompt used to refine user-facing headline lines."""

    return (
        "Prepare Twinr's morning news lines.\n"
        f"Language: {language or 'de'}\n"
        f"Target local day: {target_day_text}\n"
        f"Place context: {location_hint or 'none'}\n"
        "Rules:\n"
        "- Use only facts already present in the inputs.\n"
        "- Prefer local civic, service, transport, health, community, or weather-relevant updates.\n"
        "- Avoid event calendars, ticket listings, generic archives, and generic index pages.\n"
        f"- Return at most {limit} lines.\n"
        "- Each line must stand alone, stay calm, and fit a senior-friendly morning briefing.\n"
        "- Return only plain text lines, one per line, without bullets or numbering.\n"
        f"Raw awareness lines: {json.dumps(list(raw_headline_lines), ensure_ascii=False)}\n"
        f"Raw fallback news summary: {raw_live_news_summary or 'none'}\n"
        f"Candidate source URLs: {json.dumps(list(candidate_sources), ensure_ascii=False)}"
    )


def _news_summary_prompt(
    *,
    language: str | None,
    target_day_text: str,
    location_hint: str | None,
    headline_lines: Sequence[str],
    raw_live_news_summary: str | None,
) -> str:
    """Build the prompt used to compose one calm news paragraph."""

    return (
        "Prepare Twinr's calm morning news summary.\n"
        f"Language: {language or 'de'}\n"
        f"Target local day: {target_day_text}\n"
        f"Place context: {location_hint or 'none'}\n"
        "Rules:\n"
        "- Use only facts already present in the inputs.\n"
        "- Prefer local and practically relevant context for the day.\n"
        "- Maximum 2 short sentences.\n"
        "- Do not mention sources, URLs, archives, or ticket/event listings.\n"
        "- Return only the final summary text.\n"
        f"Refined morning news lines: {json.dumps(list(headline_lines), ensure_ascii=False)}\n"
        f"Raw fallback news summary: {raw_live_news_summary or 'none'}"
    )


def _source_selection_prompt(
    *,
    language: str | None,
    target_day_text: str,
    location_hint: str | None,
    headline_lines: Sequence[str],
    candidate_sources: Sequence[str],
    limit: int,
) -> str:
    """Build the prompt used to choose digest-supporting source URLs."""

    return (
        "Select supporting URLs for Twinr's morning digest.\n"
        f"Language: {language or 'de'}\n"
        f"Target local day: {target_day_text}\n"
        f"Place context: {location_hint or 'none'}\n"
        "Rules:\n"
        "- Choose only from the provided candidate URLs.\n"
        "- Prefer direct current reports or official service/press pages.\n"
        "- Avoid ticket pages, event calendars, generic archives, and generic index pages.\n"
        f"- Return at most {limit} URLs.\n"
        "- Return only URLs, one per line.\n"
        f"Digest lines: {json.dumps(list(headline_lines), ensure_ascii=False)}\n"
        f"Candidate URLs: {json.dumps(list(candidate_sources), ensure_ascii=False)}"
    )


def _insight_lines_prompt(
    *,
    language: str | None,
    target_day_text: str,
    raw_new_insights: Sequence[str],
    limit: int,
) -> str:
    """Build the prompt used to render user-facing overnight insight lines."""

    return (
        "Prepare Twinr's morning-visible overnight insight lines.\n"
        f"Language: {language or 'de'}\n"
        f"Target local day: {target_day_text}\n"
        "Rules:\n"
        "- Use only the provided learning signals.\n"
        "- Write warm, plain German.\n"
        "- Do not mention policies, models, deltas, support counts, or that a user explicitly asked for something.\n"
        "- Turn internal learning into calm practical takeaways for today.\n"
        f"- Return at most {limit} lines.\n"
        "- Return only plain text lines, one per line, without bullets or numbering.\n"
        f"Raw overnight learning signals: {json.dumps(list(raw_new_insights), ensure_ascii=False)}"
    )


def _continuity_lines_prompt(
    *,
    language: str | None,
    target_day_text: str,
    location_hint: str | None,
    raw_continuity_shifts: Sequence[str],
    raw_headline_lines: Sequence[str],
    limit: int,
) -> str:
    """Build the prompt used to render user-facing continuity lines."""

    return (
        "Prepare Twinr's morning continuity lines.\n"
        f"Language: {language or 'de'}\n"
        f"Target local day: {target_day_text}\n"
        f"Place context: {location_hint or 'none'}\n"
        "Rules:\n"
        "- Use only the provided continuity context.\n"
        "- Prefer family, appointments, neighbourhood, health, and ongoing community threads.\n"
        "- Do not mention feed tracking, source counts, or internal system wording.\n"
        f"- Return at most {limit} lines.\n"
        "- Return only plain text lines, one per line, without bullets or numbering.\n"
        f"Raw continuity threads: {json.dumps(list(raw_continuity_shifts), ensure_ascii=False)}\n"
        f"Raw morning news lines: {json.dumps(list(raw_headline_lines), ensure_ascii=False)}"
    )


def build_nightly_digest_refinement(
    *,
    compose: _ComposeLike | None,
    language: str | None,
    target_day_text: str,
    location_hint: str | None,
    raw_headline_lines: Sequence[str],
    raw_live_news_summary: str | None,
    candidate_news_sources: Sequence[str],
    raw_new_insights: Sequence[str],
    raw_continuity_shifts: Sequence[str],
    headline_limit: int = _DEFAULT_HEADLINE_LIMIT,
    insight_limit: int = _DEFAULT_INSIGHT_LIMIT,
    continuity_limit: int = _DEFAULT_CONTINUITY_LIMIT,
    source_limit: int = _DEFAULT_SOURCE_LIMIT,
) -> NightlyDigestRefinement:
    """Refine raw nightly news and insight inputs into morning-facing content."""

    fallback_headlines = _fallback_headlines(
        raw_headline_lines,
        raw_live_news_summary=raw_live_news_summary,
        limit=headline_limit,
    )
    headline_lines = _compose_lines(
        compose,
        prompt=_news_lines_prompt(
            language=language,
            target_day_text=target_day_text,
            location_hint=location_hint,
            raw_headline_lines=raw_headline_lines,
            raw_live_news_summary=raw_live_news_summary,
            candidate_sources=candidate_news_sources,
            limit=headline_limit,
        ),
        limit=headline_limit,
        max_len=max(220, headline_limit * 160),
    ) or fallback_headlines
    news_summary = _compose_summary(
        compose,
        prompt=_news_summary_prompt(
            language=language,
            target_day_text=target_day_text,
            location_hint=location_hint,
            headline_lines=headline_lines,
            raw_live_news_summary=raw_live_news_summary,
        ),
        max_len=_MAX_SUMMARY,
    ) or _fallback_news_summary(headline_lines, raw_live_news_summary=raw_live_news_summary)
    news_sources = _compose_sources(
        compose,
        prompt=_source_selection_prompt(
            language=language,
            target_day_text=target_day_text,
            location_hint=location_hint,
            headline_lines=headline_lines,
            candidate_sources=candidate_news_sources,
            limit=source_limit,
        ),
        candidate_sources=candidate_news_sources,
        limit=source_limit,
    )
    new_insights = _compose_lines(
        compose,
        prompt=_insight_lines_prompt(
            language=language,
            target_day_text=target_day_text,
            raw_new_insights=raw_new_insights,
            limit=insight_limit,
        ),
        limit=insight_limit,
        max_len=max(180, insight_limit * 140),
    ) or _bounded_lines(raw_new_insights, limit=insight_limit, max_len=_MAX_LINE)
    continuity_shifts = _compose_lines(
        compose,
        prompt=_continuity_lines_prompt(
            language=language,
            target_day_text=target_day_text,
            location_hint=location_hint,
            raw_continuity_shifts=raw_continuity_shifts,
            raw_headline_lines=headline_lines,
            limit=continuity_limit,
        ),
        limit=continuity_limit,
        max_len=max(200, continuity_limit * 150),
    ) or _bounded_lines(raw_continuity_shifts, limit=continuity_limit, max_len=_MAX_LINE)
    return NightlyDigestRefinement(
        headline_lines=_bounded_lines(headline_lines, limit=headline_limit, max_len=_MAX_LINE),
        live_news_summary=_bounded_optional_text(news_summary, max_len=_MAX_SUMMARY),
        news_sources=_select_fallback_sources(news_sources, limit=source_limit),
        new_insights=_bounded_lines(new_insights, limit=insight_limit, max_len=_MAX_LINE),
        continuity_shifts=_bounded_lines(continuity_shifts, limit=continuity_limit, max_len=_MAX_LINE),
    )


__all__ = [
    "NightlyDigestRefinement",
    "build_nightly_digest_refinement",
]
