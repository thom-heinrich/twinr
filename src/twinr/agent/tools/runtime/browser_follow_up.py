"""Carry pending browser follow-up hints from search into browser execution.

The search tool can return structured follow-up metadata that points to the
official site Twinr should inspect next. This module stores that hint on the
runtime between turns so a later authorized ``browser_automation`` call can
reuse the exact follow-up URL/domain instead of relying on a model-invented
host name.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import socket
from typing import Any
from urllib.parse import urlparse


_RUNTIME_HINT_ATTR = "_twinr_pending_browser_follow_up_hint"


@dataclass(frozen=True, slots=True)
class PendingBrowserFollowUpHint:
    """Describe one pending website follow-up proposed by live search."""

    question: str
    follow_up_url: str | None
    follow_up_domain: str | None
    site_follow_up_recommended: bool
    question_resolved: bool | None = None
    verification_status: str | None = None
    reason: str | None = None
    sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BrowserRequestRepair:
    """Describe one browser request repaired from pending search follow-up data."""

    original_start_url: str | None
    effective_start_url: str | None
    original_allowed_domains: tuple[str, ...]
    effective_allowed_domains: tuple[str, ...]
    reason: str
    hint_follow_up_url: str | None = None
    hint_follow_up_domain: str | None = None


def _normalize_text(value: object) -> str | None:
    """Return one stripped string or ``None`` when the value is blank."""

    text = str(value or "").strip()
    return text or None


def _normalize_host(value: object) -> str | None:
    """Normalize one host name into lowercase text without path fragments."""

    host = _normalize_text(value)
    if host is None:
        return None
    host = host.lower()
    if "/" in host or any(character.isspace() for character in host):
        return None
    return host


def _host_from_url(value: object) -> str | None:
    """Extract the lowercase host from one absolute URL when possible."""

    url = _normalize_text(value)
    if url is None:
        return None
    parsed = urlparse(url)
    host = parsed.hostname
    return host.lower() if isinstance(host, str) and host.strip() else None


def _dedupe_hosts(values: Sequence[str | None]) -> tuple[str, ...]:
    """Return the ordered unique host list without blanks."""

    deduped: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        host = _normalize_host(raw_value)
        if host is None or host in seen:
            continue
        deduped.append(host)
        seen.add(host)
    return tuple(deduped)


def _host_variants(host: str | None) -> tuple[str, ...]:
    """Return one host plus a simple ``www`` variant for same-site allowlists."""

    normalized = _normalize_host(host)
    if normalized is None:
        return ()
    if normalized.startswith("www."):
        return _dedupe_hosts((normalized, normalized[4:]))
    return _dedupe_hosts((normalized, f"www.{normalized}"))


def _hint_allowlist(hint: PendingBrowserFollowUpHint) -> tuple[str, ...]:
    """Return the narrow host allowlist implied by one follow-up hint."""

    hint_host = _normalize_host(hint.follow_up_domain) or _host_from_url(hint.follow_up_url)
    return _host_variants(hint_host)


def _host_resolves(host: str) -> bool:
    """Return whether DNS resolution succeeds for one host."""

    try:
        socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    except OSError:
        return False
    return True


def clear_pending_browser_follow_up_hint(runtime: Any) -> None:
    """Remove any pending browser follow-up hint from the runtime."""

    if runtime is None:
        return
    try:
        delattr(runtime, _RUNTIME_HINT_ATTR)
    except AttributeError:
        return


def remember_pending_browser_follow_up_hint(
    runtime: Any,
    *,
    question: str,
    follow_up_url: str | None,
    follow_up_domain: str | None,
    site_follow_up_recommended: bool,
    question_resolved: bool | None,
    verification_status: str | None,
    reason: str | None,
    sources: Sequence[str] = (),
) -> PendingBrowserFollowUpHint | None:
    """Store one pending browser follow-up hint on the runtime.

    Returns ``None`` and clears older state when the search result no longer
    recommends a concrete site follow-up.
    """

    if runtime is None:
        return None
    normalized_url = _normalize_text(follow_up_url)
    normalized_domain = _normalize_host(follow_up_domain) or _host_from_url(normalized_url)
    if not bool(site_follow_up_recommended) or (normalized_url is None and normalized_domain is None):
        clear_pending_browser_follow_up_hint(runtime)
        return None
    hint = PendingBrowserFollowUpHint(
        question=_normalize_text(question) or "",
        follow_up_url=normalized_url,
        follow_up_domain=normalized_domain,
        site_follow_up_recommended=True,
        question_resolved=question_resolved,
        verification_status=_normalize_text(verification_status),
        reason=_normalize_text(reason),
        sources=tuple(str(item).strip() for item in tuple(sources or ()) if str(item).strip()),
    )
    setattr(runtime, _RUNTIME_HINT_ATTR, hint)
    return hint


def peek_pending_browser_follow_up_hint(runtime: Any) -> PendingBrowserFollowUpHint | None:
    """Return the current pending browser follow-up hint without consuming it."""

    hint = getattr(runtime, _RUNTIME_HINT_ATTR, None)
    return hint if isinstance(hint, PendingBrowserFollowUpHint) else None


def repair_browser_request_from_pending_hint(
    runtime: Any,
    *,
    start_url: str | None,
    allowed_domains: Sequence[str],
) -> BrowserRequestRepair | None:
    """Repair one browser request from a pending follow-up hint when needed.

    The repair is conservative: Twinr only overrides the current host when the
    pending hint points to a different concrete site and the current host is
    missing or fails DNS resolution. This fixes model-invented or stale hosts
    without silently replacing a resolvable explicit website choice.
    """

    hint = peek_pending_browser_follow_up_hint(runtime)
    if hint is None:
        return None

    normalized_start_url = _normalize_text(start_url)
    original_allowed_domains = _dedupe_hosts(tuple(str(item or "").strip() for item in tuple(allowed_domains or ())))
    current_host = _host_from_url(normalized_start_url) or (original_allowed_domains[0] if original_allowed_domains else None)
    hint_host = _host_from_url(hint.follow_up_url) or _normalize_host(hint.follow_up_domain)
    hint_allowlist = _hint_allowlist(hint)
    if hint_host is None and hint.follow_up_url is None:
        clear_pending_browser_follow_up_hint(runtime)
        return None

    if current_host is not None and hint_host is not None and current_host == hint_host:
        clear_pending_browser_follow_up_hint(runtime)
        if original_allowed_domains == hint_allowlist or not hint_allowlist:
            return None
        return BrowserRequestRepair(
            original_start_url=normalized_start_url,
            effective_start_url=normalized_start_url or hint.follow_up_url,
            original_allowed_domains=original_allowed_domains,
            effective_allowed_domains=hint_allowlist or original_allowed_domains,
            reason="Pending browser follow-up hint narrowed the allowlist to the exact official site.",
            hint_follow_up_url=hint.follow_up_url,
            hint_follow_up_domain=hint.follow_up_domain,
        )

    if current_host is not None and _host_resolves(current_host):
        return None

    clear_pending_browser_follow_up_hint(runtime)
    effective_start_url = hint.follow_up_url or normalized_start_url
    effective_allowed_domains = hint_allowlist or original_allowed_domains
    if effective_start_url == normalized_start_url and effective_allowed_domains == original_allowed_domains:
        return None
    return BrowserRequestRepair(
        original_start_url=normalized_start_url,
        effective_start_url=effective_start_url,
        original_allowed_domains=original_allowed_domains,
        effective_allowed_domains=effective_allowed_domains,
        reason="Pending browser follow-up hint replaced an unresolved or missing start host with the concrete site suggested by live search.",
        hint_follow_up_url=hint.follow_up_url,
        hint_follow_up_domain=hint.follow_up_domain,
    )
