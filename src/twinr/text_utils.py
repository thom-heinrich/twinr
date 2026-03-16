from __future__ import annotations

import json
import unicodedata
from typing import Any


_GERMAN_FOLDS = str.maketrans(
    {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
        "Ä": "Ae",
        "Ö": "Oe",
        "Ü": "Ue",
    }
)
_MAX_JSON_TEXT_CHARS = 1_000_000  # AUDIT-FIX(#2): Bound JSON scanning to avoid pathological payloads on RPi-class hardware.
_NAMESPACE_START_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz")  # AUDIT-FIX(#6): Restrict namespaces to ASCII to prevent homoglyph identifiers.
_NAMESPACE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")  # AUDIT-FIX(#6): Restrict namespaces to ASCII to keep identifiers stable across components.
_STABLE_START_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")  # AUDIT-FIX(#6): Restrict stable identifiers to ASCII leading characters.
_STABLE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._-:")  # AUDIT-FIX(#6): Restrict stable identifiers to ASCII allowed characters.
_DEFAULT_IDENTIFIER_FALLBACK = "item"  # AUDIT-FIX(#5): Guarantee a safe, non-empty identifier when both input and fallback collapse away.


def _coerce_optional_text(value: object | None) -> str:
    return "" if value is None else str(value)  # AUDIT-FIX(#3): Preserve valid falsey values like 0 instead of silently dropping them.


def collapse_whitespace(value: str | None) -> str:
    return " ".join(_coerce_optional_text(value).split()).strip()  # AUDIT-FIX(#3): Only None becomes empty; other falsey values remain representable.


def sanitize_text_fragment(value: str | None) -> str:
    normalized = collapse_whitespace(value)
    if not normalized:
        return ""
    clean_chars: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category.startswith("C"):
            continue
        clean_chars.append(char)
    return collapse_whitespace("".join(clean_chars))


def truncate_text(value: str | None, *, limit: int | None = None) -> str:
    text = sanitize_text_fragment(value)
    if limit is None:
        return text
    if limit <= 0:  # AUDIT-FIX(#4): Honor non-positive limits instead of returning a spurious ellipsis.
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:  # AUDIT-FIX(#4): A single-character budget can only carry the ellipsis marker.
        return "…"
    return text[: limit - 1].rstrip() + "…"


def folded_lookup_text(value: str | None) -> str:
    raw = sanitize_text_fragment(value).translate(_GERMAN_FOLDS)  # AUDIT-FIX(#1): Strip control characters before lookup folding so invisible bytes do not split tokens unpredictably.
    parts: list[str] = []
    current: list[str] = []
    for char in raw.lower():
        if char.isalnum():
            current.append(char)
            continue
        if current:
            parts.append("".join(current))
            current = []
    if current:
        parts.append("".join(current))
    return " ".join(part for part in parts if part)


def _ascii_identifier_tokens(value: str | None) -> tuple[str, ...]:
    raw = ascii_fold(value).lower()  # AUDIT-FIX(#5): Build identifier slugs from ASCII-only tokens to avoid unstable Unicode identifiers.
    parts: list[str] = []
    current: list[str] = []
    for char in raw:
        if char.isascii() and char.isalnum():
            current.append(char)
            continue
        if current:
            parts.append("".join(current))
            current = []
    if current:
        parts.append("".join(current))
    return tuple(part for part in parts if part)


def slugify_identifier(value: str | None, *, fallback: str) -> str:
    slug = "_".join(_ascii_identifier_tokens(value))  # AUDIT-FIX(#5): Ensure generated identifiers are ASCII-safe and filesystem/URL-stable.
    if slug:
        return slug
    folded_fallback = ascii_fold(fallback).lower()
    if is_valid_stable_identifier(folded_fallback):  # AUDIT-FIX(#5): Preserve already-valid fallback identifiers after ASCII folding.
        return folded_fallback
    fallback_slug = "_".join(_ascii_identifier_tokens(fallback))
    return fallback_slug or _DEFAULT_IDENTIFIER_FALLBACK  # AUDIT-FIX(#5): Never return an empty or unsafe fallback identifier.


def retrieval_terms(value: str | None) -> tuple[str, ...]:
    normalized = folded_lookup_text(value)
    if not normalized:
        return ()
    return tuple(term for term in normalized.split(" ") if term)


def fts_match_query(value: str | None) -> str:
    alpha_terms = tuple(
        dict.fromkeys(
            term
            for term in retrieval_terms(value)
            if len(term) >= 3 and any(char.isalpha() for char in term)
        )
    )
    numeric_terms = tuple(
        dict.fromkeys(
            term
            for term in retrieval_terms(value)
            if len(term) >= 2 and term.isdigit()
        )
    )
    if not alpha_terms and not numeric_terms:
        return ""
    if alpha_terms and numeric_terms:
        alpha_clause = " OR ".join(f'"{term}"' for term in alpha_terms)
        numeric_clause = " OR ".join(f'"{term}"' for term in numeric_terms)
        return f"({alpha_clause}) AND ({numeric_clause})"
    terms = alpha_terms or numeric_terms
    return " OR ".join(f'"{term}"' for term in terms)


def extract_json_object(text: str | None) -> dict[str, Any]:
    stripped = _coerce_optional_text(text).strip()  # AUDIT-FIX(#3): Treat None as empty input without erasing valid falsey payloads.
    if not stripped:
        raise ValueError("No JSON object found in empty text.")
    if len(stripped) > _MAX_JSON_TEXT_CHARS:  # AUDIT-FIX(#2): Refuse pathological payload sizes before attempting expensive JSON parsing.
        raise ValueError(f"JSON text exceeds maximum supported size of {_MAX_JSON_TEXT_CHARS} characters.")
    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, RecursionError):
        try:
            payload = json.loads(_balanced_json_slice(stripped))
        except (json.JSONDecodeError, RecursionError, ValueError) as exc:  # AUDIT-FIX(#2): Normalize decoder failures to a stable ValueError contract.
            raise ValueError("No valid JSON object found.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return payload


def _balanced_json_slice(text: str) -> str:
    decoder = json.JSONDecoder()  # AUDIT-FIX(#2): Scan every object start so prose braces before the real JSON do not break extraction.
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, end = decoder.raw_decode(text[index:])
        except (json.JSONDecodeError, RecursionError):
            continue
        if isinstance(payload, dict):
            return text[index : index + end]
    raise ValueError("No balanced JSON object found.")


def is_valid_identifier_namespace(value: str) -> bool:
    if not value:
        return False
    if value[0] not in _NAMESPACE_START_CHARS:  # AUDIT-FIX(#6): Enforce ASCII lowercase namespace starts to block homoglyph and non-portable identifiers.
        return False
    for char in value[1:]:
        if char not in _NAMESPACE_CHARS:
            return False
    return True


def is_valid_stable_identifier(value: str) -> bool:
    if not value:
        return False
    if value[0] not in _STABLE_START_CHARS:  # AUDIT-FIX(#6): Enforce ASCII starts so identifiers behave consistently across stores and transports.
        return False
    for char in value[1:]:
        if char not in _STABLE_CHARS:
            return False
    return True


def is_valid_namespaced_identifier(value: str) -> bool:
    namespace, separator, stable_id = str(value or "").partition(":")
    if separator != ":":
        return False
    return is_valid_identifier_namespace(namespace) and is_valid_stable_identifier(stable_id)


def ascii_fold(value: str | None) -> str:
    folded = sanitize_text_fragment(value).translate(_GERMAN_FOLDS)  # AUDIT-FIX(#1): Remove control characters before ASCII folding so escape bytes cannot survive into logs, terminals, or printers.
    normalized = unicodedata.normalize("NFKD", folded)
    return collapse_whitespace(normalized.encode("ascii", errors="ignore").decode("ascii"))  # AUDIT-FIX(#1): Re-collapse whitespace after folding for stable downstream comparisons.