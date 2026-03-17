"""Expose bounded reasoning helpers for future self_coding skills.

The provider-backed implementation is not wired in this module yet. These
helpers validate their bounded contracts locally and then fail explicitly via
the shared runtime_unavailable hook.
"""

from __future__ import annotations

import math
import os
from typing import Any, NoReturn

from twinr.agent.self_coding.status import CapabilityRiskClass

from .base import SelfCodingModuleFunction, SelfCodingModuleSpec, runtime_unavailable


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    """Read an integer environment variable with safe clamping and fallback."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default

    if value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


# AUDIT-FIX(#1): Keep the advertised "bounded" contract enforceable and
# env-tunable without adding mandatory new configuration keys.
_MAX_TEXT_CHARS = _env_int("TWINR_SELF_CODING_LLM_CALL_MAX_TEXT_CHARS", 8_000, minimum=1, maximum=200_000)
_MAX_OPTIONS = _env_int("TWINR_SELF_CODING_LLM_CALL_MAX_OPTIONS", 32, minimum=1, maximum=256)
_MAX_OPTION_CHARS = _env_int("TWINR_SELF_CODING_LLM_CALL_MAX_OPTION_CHARS", 128, minimum=1, maximum=4_096)
_MAX_SCHEMA_DEPTH = _env_int("TWINR_SELF_CODING_LLM_CALL_MAX_SCHEMA_DEPTH", 8, minimum=1, maximum=64)
_MAX_SCHEMA_KEYS = _env_int("TWINR_SELF_CODING_LLM_CALL_MAX_SCHEMA_KEYS", 128, minimum=1, maximum=10_000)
_MAX_SCHEMA_SEQUENCE_ITEMS = _env_int(
    "TWINR_SELF_CODING_LLM_CALL_MAX_SCHEMA_SEQUENCE_ITEMS",
    128,
    minimum=1,
    maximum=10_000,
)
_MAX_SCHEMA_STRING_CHARS = _env_int(
    "TWINR_SELF_CODING_LLM_CALL_MAX_SCHEMA_STRING_CHARS",
    2_048,
    minimum=1,
    maximum=200_000,
)


# AUDIT-FIX(#1): Reject empty or oversized free text before any future provider
# dispatch can consume unbounded input on the single-process RPi runtime.
def _validate_text(text: str, *, field_name: str = "text") -> str:
    """Validate bounded free-text input."""

    if not isinstance(text, str):
        raise TypeError(f"{field_name} must be a str.")
    if not text.strip():
        raise ValueError(f"{field_name} must not be empty.")
    if len(text) > _MAX_TEXT_CHARS:
        raise ValueError(
            f"{field_name} exceeds the configured bound of {_MAX_TEXT_CHARS} characters."
        )
    return text


# AUDIT-FIX(#4): Reject ambiguous classification label sets before the helper is
# ever wired to a provider-backed classifier.
def _validate_options(options: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Validate the bounded option list for classification."""

    if isinstance(options, list):
        options = tuple(options)
    if not isinstance(options, tuple):
        raise TypeError("options must be a tuple[str, ...] or list[str].")
    if not options:
        raise ValueError("options must not be empty.")
    if len(options) > _MAX_OPTIONS:
        raise ValueError(f"options exceeds the configured bound of {_MAX_OPTIONS} entries.")

    seen: set[str] = set()
    for index, option in enumerate(options):
        if not isinstance(option, str):
            raise TypeError(f"options[{index}] must be a str.")
        if not option.strip():
            raise ValueError(f"options[{index}] must not be empty.")
        if len(option) > _MAX_OPTION_CHARS:
            raise ValueError(
                f"options[{index}] exceeds the configured bound of {_MAX_OPTION_CHARS} characters."
            )
        if option in seen:
            raise ValueError(f"options contains a duplicate label: {option!r}.")
        seen.add(option)

    return options


# AUDIT-FIX(#5): Keep extract() schemas JSON-safe, acyclic, and bounded so a
# later provider integration cannot recurse or serialize arbitrary objects.
def _validate_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Validate that the extraction schema is bounded and JSON-safe."""

    if not isinstance(schema, dict):
        raise TypeError("schema must be a dict[str, Any].")
    if not schema:
        raise ValueError("schema must not be empty.")

    seen_container_ids: set[int] = set()
    total_keys = 0

    def _walk(value: Any, *, depth: int, path: str) -> None:
        nonlocal total_keys

        if depth > _MAX_SCHEMA_DEPTH:
            raise ValueError(
                f"schema exceeds the configured nesting bound of {_MAX_SCHEMA_DEPTH} at {path}."
            )

        if isinstance(value, dict):
            value_id = id(value)
            if value_id in seen_container_ids:
                raise ValueError(f"schema contains a cycle at {path}.")
            seen_container_ids.add(value_id)
            try:
                for key, child in value.items():
                    if not isinstance(key, str):
                        raise TypeError(
                            f"schema keys must be strings; got {type(key).__name__} at {path}."
                        )
                    if not key:
                        raise ValueError(f"schema keys must not be empty at {path}.")
                    if len(key) > _MAX_SCHEMA_STRING_CHARS:
                        raise ValueError(
                            f"schema key {key!r} exceeds the configured bound of "
                            f"{_MAX_SCHEMA_STRING_CHARS} characters."
                        )
                    total_keys += 1
                    if total_keys > _MAX_SCHEMA_KEYS:
                        raise ValueError(
                            f"schema exceeds the configured bound of {_MAX_SCHEMA_KEYS} keys."
                        )
                    _walk(child, depth=depth + 1, path=f"{path}.{key}")
            finally:
                seen_container_ids.remove(value_id)
            return

        if isinstance(value, (list, tuple)):
            value_id = id(value)
            if value_id in seen_container_ids:
                raise ValueError(f"schema contains a cycle at {path}.")
            if len(value) > _MAX_SCHEMA_SEQUENCE_ITEMS:
                raise ValueError(
                    f"{path} exceeds the configured bound of "
                    f"{_MAX_SCHEMA_SEQUENCE_ITEMS} sequence items."
                )
            seen_container_ids.add(value_id)
            try:
                for index, item in enumerate(value):
                    _walk(item, depth=depth + 1, path=f"{path}[{index}]")
            finally:
                seen_container_ids.remove(value_id)
            return

        if isinstance(value, str):
            if len(value) > _MAX_SCHEMA_STRING_CHARS:
                raise ValueError(
                    f"{path} exceeds the configured bound of {_MAX_SCHEMA_STRING_CHARS} characters."
                )
            return

        if value is None or isinstance(value, bool) or isinstance(value, int):
            return

        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"{path} must not contain NaN or infinite floats.")
            return

        raise TypeError(
            f"{path} contains unsupported type {type(value).__name__}; "
            "only JSON-safe scalars, dict, list, and tuple are allowed."
        )

    _walk(schema, depth=1, path="schema")
    return schema


# AUDIT-FIX(#2): Decouple the public API contract from side effects inside
# runtime_unavailable() and guarantee an exception on every unavailable call.
def _raise_runtime_unavailable(capability_name: str) -> NoReturn:
    """Guarantee that unavailable helpers never fall through with a None return."""

    result = runtime_unavailable(capability_name)
    if isinstance(result, BaseException):
        raise result
    raise RuntimeError(
        f"{capability_name} is unavailable and runtime_unavailable() returned without raising."
    )


def classify(text: str, options: tuple[str, ...]) -> str:
    """Classify short text into one of the provided bounded options."""

    _validate_text(text)  # AUDIT-FIX(#1): Enforce bounded text input before runtime dispatch.
    _validate_options(options)  # AUDIT-FIX(#4): Reject empty or duplicate option sets early.
    _raise_runtime_unavailable(  # AUDIT-FIX(#2): Guarantee an exception instead of implicit None.
        "llm_call.classify"
    )


def extract(text: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Extract structured fields from short text into one bounded schema."""

    _validate_text(text)  # AUDIT-FIX(#1): Enforce bounded text input before runtime dispatch.
    _validate_schema(schema)  # AUDIT-FIX(#5): Reject cyclic and non-JSON-safe schema shapes.
    _raise_runtime_unavailable(  # AUDIT-FIX(#2): Guarantee an exception instead of implicit None.
        "llm_call.extract"
    )


def summarize(text: str) -> str:
    """Summarize bounded text into a short plain-language answer."""

    _validate_text(text)  # AUDIT-FIX(#1): Enforce bounded text input before runtime dispatch.
    _raise_runtime_unavailable(  # AUDIT-FIX(#2): Guarantee an exception instead of implicit None.
        "llm_call.summarize"
    )


MODULE_SPEC = SelfCodingModuleSpec(
    capability_id="llm_call",
    module_name="llm_call",
    # AUDIT-FIX(#3): Keep capability discovery truthful so planners do not treat
    # this placeholder as a live provider-backed capability.
    summary=(
        "Reserved bounded reasoning capability; provider-backed runtime dispatch "
        "is currently unavailable in this module."
    ),
    risk_class=CapabilityRiskClass.HIGH,
    public_api=(
        SelfCodingModuleFunction(
            name="classify",
            signature="classify(text: str, options: tuple[str, ...]) -> str",
            summary=(
                "Choose one label from a short bounded option list when provider "
                "runtime support is enabled."
            ),
            returns="one selected option label",
            tags=("bounded", "reasoning"),
        ),
        SelfCodingModuleFunction(
            name="extract",
            signature="extract(text: str, schema: dict[str, Any]) -> dict[str, Any]",
            summary=(
                "Extract structured fields that fit one bounded JSON-like schema "
                "when provider runtime support is enabled."
            ),
            returns="a JSON-safe mapping matching the requested schema",
            tags=("bounded", "reasoning"),
        ),
        SelfCodingModuleFunction(
            name="summarize",
            signature="summarize(text: str) -> str",
            summary=(
                "Summarize bounded text into a short plain-language answer when "
                "provider runtime support is enabled."
            ),
            returns="a short summary string",
            tags=("bounded", "reasoning"),
        ),
    ),
    tags=("reasoning", "builtin", "bounded"),
)

__all__ = ["MODULE_SPEC", "classify", "extract", "summarize"]