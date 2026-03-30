# CHANGELOG: 2026-03-30
# BUG-1: Fixed canonical-equivalent lookup mismatches by normalizing lookup text with NFKC + casefold before tokenization.
# BUG-2: Fixed FTS query generation dropping short but meaningful alpha tokens like "TV", "AI", and "Pi".
# BUG-3: Fixed identifier collisions where non-empty non-Latin / symbol-heavy inputs collapsed to the same fallback slug.
# BUG-4: Fixed JSON extraction CPU spikes from repeated full decoder retries on brace-heavy text by switching to a linear balanced-slice scan.
# SEC-1: Rejected duplicate JSON keys and non-finite JSON numbers (NaN/Infinity) instead of silently accepting ambiguous / non-standard payloads.
# SEC-2: Added depth, container, and identifier length budgets to make the module actually enforce bounded inputs on Pi-class hardware.
# IMP-1: Upgraded ASCII folding to opportunistically use AnyAscii for multilingual transliteration while preserving a stdlib-only fallback.
# IMP-2: Added deterministic bounded identifier hashing so slugs remain stable and collision-resistant after normalization/truncation.

"""Normalize free-form text and identifiers shared across Twinr subsystems.

The helpers in this module keep user-facing text, lookup terms, JSON snippets,
and stable identifiers in a bounded format that works across memory, provider,
and operator-facing paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from typing import Any

try:  # IMP-1: Optional frontier transliteration path; pure-Python and safe to omit at runtime.
    from anyascii import anyascii as _anyascii
except Exception:  # pragma: no cover - optional dependency.
    _anyascii = None


__all__ = (
    "ascii_fold",
    "collapse_whitespace",
    "extract_json_object",
    "folded_lookup_text",
    "fts_match_query",
    "is_valid_identifier_namespace",
    "is_valid_namespaced_identifier",
    "is_valid_stable_identifier",
    "retrieval_terms",
    "sanitize_text_fragment",
    "slugify_identifier",
    "truncate_text",
)


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

_MAX_JSON_TEXT_CHARS = 1_000_000
_MAX_JSON_DEPTH = 64
_MAX_JSON_CONTAINER_ITEMS = 20_000
_MAX_JSON_OBJECT_KEYS = 4_096
_MAX_FTS_TERMS = 16
_MIN_FTS_ALPHA_TERM_CHARS = 2
_MIN_FTS_NUMERIC_TERM_CHARS = 2

_NAMESPACE_START_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz")
_NAMESPACE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")
_STABLE_START_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")
_STABLE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._-:")

# BREAKING: identifiers are now hard-bounded so this module really guarantees
# bounded storage keys across DB rows, logs, caches, and file paths.
_MAX_NAMESPACE_LENGTH = 64
_MAX_STABLE_IDENTIFIER_LENGTH = 128

_DEFAULT_IDENTIFIER_FALLBACK = "item"
_IDENTIFIER_HASH_HEX_CHARS = 12


def _coerce_optional_text(value: object | None) -> str:
    """Convert optional values to text without dropping valid falsey inputs."""

    return "" if value is None else str(value)


def _nfkc_casefold(value: str) -> str:
    """Return NFKC-normalized, caseless Unicode text."""

    normalized = unicodedata.normalize("NFKC", value)
    return unicodedata.normalize("NFKC", normalized.casefold())


def collapse_whitespace(value: str | None) -> str:
    """Collapse internal whitespace and trim leading/trailing gaps."""

    return " ".join(_coerce_optional_text(value).split()).strip()


def sanitize_text_fragment(value: str | None) -> str:
    """Strip control characters and normalize whitespace for display-safe text."""

    normalized = collapse_whitespace(value)
    if not normalized:
        return ""
    clean_chars: list[str] = []
    for char in normalized:
        if unicodedata.category(char).startswith("C"):
            continue
        clean_chars.append(char)
    return collapse_whitespace("".join(clean_chars))


def truncate_text(value: str | None, *, limit: int | None = None) -> str:
    """Truncate sanitized text to a bounded display or storage budget."""

    text = sanitize_text_fragment(value)
    if limit is None:
        return text
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    return text[: limit - 1].rstrip() + "…"


def _lookup_source_text(value: str | None) -> str:
    """Normalize lookup text so canonical-equivalent inputs tokenize identically."""

    clean = sanitize_text_fragment(value)
    if not clean:
        return ""
    return _nfkc_casefold(clean.translate(_GERMAN_FOLDS))


def _is_lookup_token_char(char: str) -> bool:
    """Return whether a Unicode character should stay attached to a lookup token."""

    return unicodedata.category(char)[:1] in {"L", "N", "M"}


def folded_lookup_text(value: str | None) -> str:
    """Normalize text into lookup tokens with Unicode-aware case folding."""

    raw = _lookup_source_text(value)
    if not raw:
        return ""
    parts: list[str] = []
    current: list[str] = []
    for char in raw:
        if _is_lookup_token_char(char):
            current.append(char)
            continue
        if current:
            parts.append("".join(current))
            current = []
    if current:
        parts.append("".join(current))
    return " ".join(part for part in parts if part)


def _ascii_transliterate(value: str) -> str:
    """Transliterate Unicode text to ASCII while preserving German folds."""

    if _anyascii is not None:
        return _anyascii(value)
    return unicodedata.normalize("NFKD", value).encode("ascii", errors="ignore").decode("ascii")


def ascii_fold(value: str | None) -> str:
    """Fold sanitized text to ASCII for stable storage and comparisons."""

    clean = sanitize_text_fragment(value)
    if not clean:
        return ""
    normalized = unicodedata.normalize("NFKC", clean.translate(_GERMAN_FOLDS))
    return collapse_whitespace(_ascii_transliterate(normalized))


def _ascii_identifier_tokens(value: str | None) -> tuple[str, ...]:
    """Return ASCII-only identifier tokens derived from free-form input text."""

    raw = ascii_fold(value).lower()
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


def _stable_hash_suffix(value: str) -> str:
    """Return a short deterministic hash suffix for collision-resistant IDs."""

    return hashlib.blake2s(
        value.encode("utf-8"),
        digest_size=_IDENTIFIER_HASH_HEX_CHARS // 2,
    ).hexdigest()


def _bound_stable_identifier(value: str) -> str:
    """Trim stable identifiers to the configured budget without losing uniqueness."""

    candidate = value.strip("._-:")
    if not candidate:
        return _DEFAULT_IDENTIFIER_FALLBACK
    if len(candidate) <= _MAX_STABLE_IDENTIFIER_LENGTH:
        return candidate

    digest = _stable_hash_suffix(candidate)
    keep = max(1, _MAX_STABLE_IDENTIFIER_LENGTH - len(digest) - 1)
    prefix = candidate[:keep].rstrip("._-:")
    if not prefix:
        prefix = _DEFAULT_IDENTIFIER_FALLBACK
    return f"{prefix}_{digest}"


def slugify_identifier(value: str | None, *, fallback: str) -> str:
    """Build a stable bounded ASCII identifier from free-form input."""

    slug = "_".join(_ascii_identifier_tokens(value))
    if slug:
        return _bound_stable_identifier(slug)

    folded_fallback = ascii_fold(fallback).lower()
    if is_valid_stable_identifier(folded_fallback):
        fallback_slug = folded_fallback
    else:
        fallback_slug = _bound_stable_identifier(
            "_".join(_ascii_identifier_tokens(fallback)) or _DEFAULT_IDENTIFIER_FALLBACK
        )

    source_text = sanitize_text_fragment(value)
    if not source_text:
        return fallback_slug

    # BREAKING: non-empty inputs that previously collapsed to the plain fallback
    # now receive a deterministic hash suffix, preventing cross-script collisions.
    return _bound_stable_identifier(
        f"{fallback_slug}_{_stable_hash_suffix(_nfkc_casefold(source_text))}"
    )


def retrieval_terms(value: str | None) -> tuple[str, ...]:
    """Split normalized lookup text into retrieval terms."""

    normalized = folded_lookup_text(value)
    if not normalized:
        return ()
    return tuple(term for term in normalized.split(" ") if term)


def _dedupe_terms_in_order(terms: tuple[str, ...]) -> tuple[str, ...]:
    """Return terms deduplicated while preserving first-seen order."""

    return tuple(dict.fromkeys(terms))


def fts_match_query(value: str | None) -> str:
    """Build a bounded FTS query string from normalized retrieval terms."""

    terms = _dedupe_terms_in_order(retrieval_terms(value))[:_MAX_FTS_TERMS]
    alpha_terms = tuple(
        term
        for term in terms
        if len(term) >= _MIN_FTS_ALPHA_TERM_CHARS and any(char.isalpha() for char in term)
    )
    numeric_terms = tuple(
        term
        for term in terms
        if len(term) >= _MIN_FTS_NUMERIC_TERM_CHARS and term.isdigit()
    )

    if not alpha_terms and not numeric_terms:
        return ""
    if alpha_terms and numeric_terms:
        alpha_clause = " OR ".join(f'"{term}"' for term in alpha_terms)
        numeric_clause = " OR ".join(f'"{term}"' for term in numeric_terms)
        return f"({alpha_clause}) AND ({numeric_clause})"

    terms_to_use = alpha_terms or numeric_terms
    return " OR ".join(f'"{term}"' for term in terms_to_use)


def _reject_non_finite_constant(token: str) -> Any:
    """Reject non-standard JSON numeric constants."""

    raise ValueError(f"Invalid JSON numeric constant: {token}")


def _reject_duplicate_keys_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        if len(result) >= _MAX_JSON_OBJECT_KEYS:
            raise ValueError(
                f"JSON object exceeds maximum key budget of {_MAX_JSON_OBJECT_KEYS}."
            )
        result[key] = value
    return result


# BREAKING: JSON extraction is now strict. Duplicate keys and NaN/Infinity are
# rejected instead of being silently accepted and normalized to a lossy dict.
_STRICT_JSON_DECODER = json.JSONDecoder(
    object_pairs_hook=_reject_duplicate_keys_object,
    parse_constant=_reject_non_finite_constant,
)


def _strict_json_loads(text: str) -> Any:
    """Decode a strict JSON document using Twinr's hardened policy."""

    return _STRICT_JSON_DECODER.decode(text)


def _validate_json_budget(payload: dict[str, Any]) -> None:
    """Reject JSON structures that are too deep or too wide for Pi-class hardware."""

    stack: list[tuple[dict[str, Any] | list[Any], int]] = [(payload, 1)]
    container_count = 0

    while stack:
        current, depth = stack.pop()
        container_count += 1
        if container_count > _MAX_JSON_CONTAINER_ITEMS:
            raise ValueError(
                f"JSON exceeds maximum container budget of {_MAX_JSON_CONTAINER_ITEMS}."
            )
        if depth > _MAX_JSON_DEPTH:
            raise ValueError(f"JSON exceeds maximum nesting depth of {_MAX_JSON_DEPTH}.")

        if isinstance(current, dict):
            if len(current) > _MAX_JSON_OBJECT_KEYS:
                raise ValueError(
                    f"JSON object exceeds maximum key budget of {_MAX_JSON_OBJECT_KEYS}."
                )
            for nested in current.values():
                if isinstance(nested, (dict, list)):
                    stack.append((nested, depth + 1))
                elif isinstance(nested, float) and not math.isfinite(nested):
                    raise ValueError("JSON contains a non-finite float.")
            continue

        if len(current) > _MAX_JSON_CONTAINER_ITEMS:
            raise ValueError(
                f"JSON array exceeds maximum item budget of {_MAX_JSON_CONTAINER_ITEMS}."
            )
        for nested in current:
            if isinstance(nested, (dict, list)):
                stack.append((nested, depth + 1))
            elif isinstance(nested, float) and not math.isfinite(nested):
                raise ValueError("JSON contains a non-finite float.")


def extract_json_object(text: str | None) -> dict[str, Any]:
    """Extract the first valid JSON object from free-form model output text.

    Args:
        text: Raw text that may contain a standalone JSON object or extra prose.

    Returns:
        The decoded JSON object.

    Raises:
        ValueError: If the input is empty, too large, or does not contain a
            valid JSON object.
    """

    stripped = _coerce_optional_text(text).strip()
    if not stripped:
        raise ValueError("No JSON object found in empty text.")
    if len(stripped) > _MAX_JSON_TEXT_CHARS:
        raise ValueError(
            f"JSON text exceeds maximum supported size of {_MAX_JSON_TEXT_CHARS} characters."
        )

    payload: Any | None = None
    if stripped.startswith("{"):
        try:
            payload = _strict_json_loads(stripped)
        except (json.JSONDecodeError, RecursionError, ValueError):
            payload = None

    if payload is None:
        try:
            payload = _strict_json_loads(_balanced_json_slice(stripped))
        except (json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise ValueError("No valid JSON object found.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")

    _validate_json_budget(payload)
    return payload


def _balanced_json_slice(text: str) -> str:
    """Return the first balanced strict-JSON-object slice found in the input."""

    start: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            if depth > _MAX_JSON_DEPTH:
                raise ValueError(f"JSON exceeds maximum nesting depth of {_MAX_JSON_DEPTH}.")
            continue

        if char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : index + 1]
                try:
                    payload = _strict_json_loads(candidate)
                except (json.JSONDecodeError, RecursionError, ValueError):
                    start = None
                    continue
                if isinstance(payload, dict):
                    return candidate
                start = None

    raise ValueError("No balanced JSON object found.")


def is_valid_identifier_namespace(value: str) -> bool:
    """Return whether a namespace token satisfies Twinr's ASCII rules."""

    if not value:
        return False
    if len(value) > _MAX_NAMESPACE_LENGTH:
        return False
    if value[0] not in _NAMESPACE_START_CHARS:
        return False
    for char in value[1:]:
        if char not in _NAMESPACE_CHARS:
            return False
    return True


def is_valid_stable_identifier(value: str) -> bool:
    """Return whether a stable identifier satisfies Twinr's ASCII rules."""

    if not value:
        return False
    if len(value) > _MAX_STABLE_IDENTIFIER_LENGTH:
        return False
    if value[0] not in _STABLE_START_CHARS:
        return False
    for char in value[1:]:
        if char not in _STABLE_CHARS:
            return False
    return True


def is_valid_namespaced_identifier(value: str) -> bool:
    """Return whether a value matches ``namespace:stable_id`` form."""

    namespace, separator, stable_id = str(value or "").partition(":")
    if separator != ":":
        return False
    return is_valid_identifier_namespace(namespace) and is_valid_stable_identifier(stable_id)