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


def collapse_whitespace(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


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
    if limit is None or len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def folded_lookup_text(value: str | None) -> str:
    raw = collapse_whitespace(value).translate(_GERMAN_FOLDS)
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


def slugify_identifier(value: str | None, *, fallback: str) -> str:
    folded = folded_lookup_text(value)
    if not folded:
        return fallback
    slug = "_".join(part for part in folded.split(" ") if part)
    return slug or fallback


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


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        raise ValueError("No JSON object found in empty text.")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = json.loads(_balanced_json_slice(stripped))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return payload


def _balanced_json_slice(text: str) -> str:
    start = -1
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if start < 0:
            if char == "{":
                start = index
                depth = 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("No balanced JSON object found.")


def is_valid_identifier_namespace(value: str) -> bool:
    if not value:
        return False
    if not value[0].isalpha() or not value[0].islower():
        return False
    for char in value[1:]:
        if not (char.islower() or char.isdigit() or char == "_"):
            return False
    return True


def is_valid_stable_identifier(value: str) -> bool:
    if not value:
        return False
    first = value[0]
    if not (first.islower() or first.isdigit()):
        return False
    for char in value[1:]:
        if char.islower() or char.isdigit():
            continue
        if char in {".", "_", "-", ":"}:
            continue
        return False
    return True


def is_valid_namespaced_identifier(value: str) -> bool:
    namespace, separator, stable_id = str(value or "").partition(":")
    if separator != ":":
        return False
    return is_valid_identifier_namespace(namespace) and is_valid_stable_identifier(stable_id)


def ascii_fold(value: str | None) -> str:
    folded = collapse_whitespace(value).translate(_GERMAN_FOLDS)
    normalized = unicodedata.normalize("NFKD", folded)
    return normalized.encode("ascii", errors="ignore").decode("ascii")
