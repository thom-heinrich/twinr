from __future__ import annotations

_LANGUAGE_DISPLAY_NAMES = {
    "de": "German",
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
}


def language_display_name(language_code: str | None) -> str:
    normalized = (language_code or "").strip().lower()
    if not normalized:
        return "the configured spoken language"
    return _LANGUAGE_DISPLAY_NAMES.get(normalized, normalized)


def canonical_memory_instruction() -> str:
    return (
        "All internal memory context and all persistent memory or profile tool payloads must use canonical English "
        "for semantic text fields. Keep names, phone numbers, email addresses, IDs, codes, and exact quoted text verbatim."
    )


def user_response_language_instruction(language_code: str | None) -> str:
    language_name = language_display_name(language_code)
    return f"All user-facing spoken and written replies for this turn must be in {language_name}."


def memory_and_response_contract(language_code: str | None) -> str:
    return f"{user_response_language_instruction(language_code)} {canonical_memory_instruction()}"
