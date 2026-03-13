"""
Contract
- Purpose:
  - Normalize and validate healthstream event fields for stable downstream consumption.
- Inputs (types, units):
  - Raw strings from CLI callers and automated watchdogs.
- Outputs (types, units):
  - Normalized `kind`, `status`, `severity`, and validated `links` tokens.
- Invariants:
  - Allow a small controlled vocabulary, with explicit normalization of common aliases.
  - Never silently accept malformed link tokens (breaks Meta truth edges).
- Error semantics:
  - Raises `ValueError` for invalid values.
- External boundaries:
  - None.
- Telemetry:
  - None.
"""

##REFACTOR: 2026-01-16##

from __future__ import annotations  # NOTE: Deprecated in Python 3.14; kept for drop-in annotation behavior.

import re
from collections.abc import Iterable
from typing import Optional


# Allow RFC 3986-ish URL/URI characters in the value part (plus bounded length).
# Token format remains: <name>:<value>, where <name> is a small, strict identifier.
LINK_TOKEN_RE = re.compile(
    r"^[a-z][a-z0-9_]{0,32}:[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]{1,2048}$"
)

ALLOWED_KINDS: set[str] = {"health", "dq", "metric", "alert", "ops"}
ALLOWED_STATUSES: set[str] = {"ok", "degraded", "error", "unknown", "skipped"}
ALLOWED_SEVERITIES: set[str] = {"info", "warning", "critical"}

_STATUS_ALIASES: dict[str, str] = {
    "warn": "degraded",
    "warning": "degraded",
    "bad": "error",
    "critical": "error",
    "fail": "error",
    "failed": "error",
    "pass": "ok",
    "green": "ok",
    "yellow": "degraded",
    "red": "error",
}

_KIND_ALIASES: dict[str, str] = {
    "data_quality": "dq",
    "dq_watchdog": "dq",
}

_SEVERITY_ALIASES: dict[str, str] = {
    "warn": "warning",
    "warning": "warning",
    "err": "critical",
    "error": "critical",
}


def _coerce_str(value: object, *, field: str) -> str:
    """
    Best-effort coercion to text with stable ValueError semantics.

    - bytes are decoded as UTF-8
    - other objects are coerced via str()
    - failures are surfaced as ValueError (per contract)
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"{field} must be valid UTF-8 text") from e
    try:
        return str(value)
    except Exception as e:
        raise ValueError(f"{field} must be text") from e


def _coerce_str_or_empty(value: object, *, field: str) -> str:
    """
    Coerce to text but preserve legacy `(raw or "")` semantics:
    falsy values (e.g., 0/False/""/None) behave like empty string.
    """
    try:
        if not value:
            return ""
    except Exception:
        # If truthiness is not well-defined, treat as truthy and attempt coercion.
        pass
    return _coerce_str(value, field=field)


def _default_severity_for_status(status: str) -> str:
    if status == "ok":
        return "info"
    if status == "degraded":
        return "warning"
    if status in {"error", "unknown"}:
        return "critical"
    return "info"  # includes "skipped"


def normalize_kind(raw: str) -> str:
    v = _coerce_str_or_empty(raw, field="kind").strip().lower()
    if not v:
        raise ValueError("kind is required")
    v = _KIND_ALIASES.get(v, v)
    if v not in ALLOWED_KINDS:
        raise ValueError(f"invalid kind={v!r} (allowed={sorted(ALLOWED_KINDS)})")
    return v


def normalize_status(raw: str) -> str:
    v = _coerce_str_or_empty(raw, field="status").strip().lower()
    if not v:
        raise ValueError("status is required")
    v = _STATUS_ALIASES.get(v, v)
    if v not in ALLOWED_STATUSES:
        raise ValueError(f"invalid status={v!r} (allowed={sorted(ALLOWED_STATUSES)})")
    return v


def normalize_severity(raw: Optional[str], *, status: str) -> str:
    # Only normalize/validate status when it's needed for defaulting.
    if raw is None:
        return _default_severity_for_status(normalize_status(status))

    s = _coerce_str(raw, field="severity")
    if not s.strip():
        return _default_severity_for_status(normalize_status(status))

    v = s.strip().lower()
    v = _SEVERITY_ALIASES.get(v, v)
    if v not in ALLOWED_SEVERITIES:
        raise ValueError(f"invalid severity={v!r} (allowed={sorted(ALLOWED_SEVERITIES)})")
    return v


def normalize_links(links: Optional[Iterable[str]]) -> list[str]:
    out: list[str] = []
    if not links:
        return out

    # Protect against the common "single string passed as iterable" footgun.
    if isinstance(links, (str, bytes)):
        links_iter: Iterable[object] = (links,)
    else:
        links_iter = links

    for raw in links_iter:
        s = _coerce_str_or_empty(raw, field="link token").strip()
        if not s:
            continue
        if not LINK_TOKEN_RE.fullmatch(s):
            raise ValueError(f"invalid link token: {s!r}")
        out.append(s)
    return out
