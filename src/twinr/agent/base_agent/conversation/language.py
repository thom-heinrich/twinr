"""Build language instructions for conversation prompts and memory tools.

This module normalizes configured spoken-language tags before they are embedded
into provider instructions. User-facing replies stay in the configured spoken
language, while semantic memory payloads stay in canonical English.
"""

from __future__ import annotations

import re  # AUDIT-FIX(#2): Validate language codes before interpolating them into prompt text.
from collections.abc import Mapping  # AUDIT-FIX(#5): Type the immutable mapping precisely for 3.11 tooling.
from types import MappingProxyType  # AUDIT-FIX(#5): Prevent accidental runtime mutation of module-level state.
from typing import Final  # AUDIT-FIX(#5): Mark constants as immutable for static analysis.

_LANGUAGE_DISPLAY_NAMES: Final[Mapping[str, str]] = MappingProxyType({  # AUDIT-FIX(#5): Freeze supported language names.
    "de": "German",
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
})
_LANGUAGE_CODE_PATTERN: Final[re.Pattern[str]] = re.compile(  # AUDIT-FIX(#2): Only accept conservative BCP 47-like tags.
    r"^[A-Za-z]{2,8}(?:[-_][A-Za-z0-9]{1,8})*$"
)
_MAX_LANGUAGE_CODE_LENGTH: Final[int] = 64  # AUDIT-FIX(#2): Bound input size to stop oversized prompt pollution.


def _normalize_language_code(language_code: str | None) -> str | None:  # AUDIT-FIX(#1,#2,#3): Centralize locale normalization, validation, and type-guarding.
    if language_code is None:  # AUDIT-FIX(#3): Preserve None as a safe "unset" signal.
        return None
    if not isinstance(language_code, str):  # AUDIT-FIX(#3): Avoid AttributeError on unexpected runtime types.
        return None

    candidate = language_code.strip()  # AUDIT-FIX(#2): Normalize surrounding whitespace before validation.
    if not candidate:  # AUDIT-FIX(#2): Treat empty and whitespace-only values as unset.
        return None
    if len(candidate) > _MAX_LANGUAGE_CODE_LENGTH:  # AUDIT-FIX(#2): Reject invalid oversized values.
        return None
    if not _LANGUAGE_CODE_PATTERN.fullmatch(candidate):  # AUDIT-FIX(#2): Block malformed tags and prompt-injection text.
        return None

    primary_subtag = candidate.replace("_", "-").split("-", 1)[0].casefold()  # AUDIT-FIX(#1): Normalize common locale formats like de-DE and pt_BR.
    if not primary_subtag:  # AUDIT-FIX(#1): Fail closed if normalization produces no usable primary tag.
        return None
    return primary_subtag


def language_display_name(language_code: str | None) -> str:
    """Return the English display name for a configured language tag.

    Args:
        language_code: Configured spoken-language tag such as ``de`` or
            ``de-DE``.

    Returns:
        The English display name for the normalized primary subtag. Returns a
        generic fallback phrase when the tag is unset, malformed, or
        unsupported.
    """

    normalized = _normalize_language_code(language_code)  # AUDIT-FIX(#1): Reuse canonical normalization for consistent behavior.
    if not normalized:
        return "the configured spoken language"
    return _LANGUAGE_DISPLAY_NAMES.get(normalized, "the configured spoken language")  # AUDIT-FIX(#4): Fail closed on unsupported tags instead of echoing opaque codes into prompts.


def canonical_memory_instruction() -> str:
    """Return the fixed instruction for canonical memory language.

    Returns:
        A provider instruction that keeps semantic memory text in English while
        preserving verbatim quoted identifiers and exact user data.
    """

    return (
        "All internal memory context and all persistent memory or profile tool payloads must use canonical English "
        "for semantic text fields. Keep names, phone numbers, email addresses, IDs, codes, and exact quoted text verbatim."
    )


def user_response_language_instruction(language_code: str | None) -> str:
    """Return the reply-language instruction for the current turn.

    Args:
        language_code: Configured spoken-language tag for the active user.

    Returns:
        A provider instruction that constrains user-facing spoken and written
        replies to the normalized configured language or to a safe fallback
        phrase when the language is unknown.
    """

    language_name = language_display_name(language_code)
    return f"All user-facing spoken and written replies for this turn must be in {language_name}."


def memory_and_response_contract(language_code: str | None) -> str:
    """Return the combined language contract for replies and memory payloads.

    Args:
        language_code: Configured spoken-language tag for the active user.

    Returns:
        A single instruction string that combines the user reply language
        constraint with the canonical memory-language requirement.
    """

    return f"{user_response_language_instruction(language_code)} {canonical_memory_instruction()}"
