"""Expose bounded live-web research primitives for self_coding skills."""

from __future__ import annotations

# AUDIT-FIX(#3): NoReturn makes the stub's failure contract explicit for Python 3.11 callers and type checkers.
from typing import NoReturn

from twinr.agent.self_coding.status import CapabilityRiskClass

from .base import SelfCodingModuleFunction, SelfCodingModuleSpec, runtime_unavailable

# AUDIT-FIX(#2): Bound free-text inputs locally so invalid or pathological payloads fail fast
# before they can reach a future runtime backend or inflate logs on constrained hardware.
_MAX_QUESTION_LENGTH = 4096
_MAX_CONTEXT_LENGTH = 512


def _normalize_optional_text(value: str | None, *, field_name: str, max_length: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be text or None")
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} must be <= {max_length} characters")
    return normalized


def _normalize_required_text(value: str, *, field_name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be text")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} must be <= {max_length} characters")
    return normalized


def _raise_runtime_unavailable(capability_name: str) -> NoReturn:
    """Raise a deterministic runtime-unavailable error for this capability."""
    # AUDIT-FIX(#1): Keep the public stub as an explicit unavailable capability instead of a misleading implicit failure.
    # AUDIT-FIX(#3): Force an exception on every path even if runtime_unavailable() later logs or returns instead of raising.
    error = runtime_unavailable(capability_name)
    if isinstance(error, BaseException):
        raise error
    raise RuntimeError(f"{capability_name} is unavailable in this runtime.")


# AUDIT-FIX(#1): Make the runtime-gated behavior explicit at the public entry point.
# AUDIT-FIX(#2): Validate and bound caller inputs before any future backend handoff.
def search(
    question: str,
    *,
    location_hint: str | None = None,
    date_context: str | None = None,
) -> dict[str, object]:
    """Validate the request and raise if the managed search backend is unavailable."""
    _normalize_required_text(question, field_name="question", max_length=_MAX_QUESTION_LENGTH)
    _normalize_optional_text(location_hint, field_name="location_hint", max_length=_MAX_CONTEXT_LENGTH)
    _normalize_optional_text(date_context, field_name="date_context", max_length=_MAX_CONTEXT_LENGTH)
    _raise_runtime_unavailable("web_search.search")


MODULE_SPEC = SelfCodingModuleSpec(
    capability_id="web_search",
    module_name="web_search",
    # AUDIT-FIX(#1): Advertise the capability as runtime-gated so planners/operators are not told it always returns live results.
    summary="Run one bounded live web search through Twinr's managed search backend when available.",
    risk_class=CapabilityRiskClass.HIGH,
    public_api=(
        SelfCodingModuleFunction(
            name="search",
            signature=(
                "search(question: str, *, location_hint: str | None = None, "
                "date_context: str | None = None) -> dict[str, object]"
            ),
            # AUDIT-FIX(#1): Keep public metadata aligned with the implementation's runtime-gated behavior.
            summary="Validate one bounded live-web search request and return results when the backend is available.",
            returns="a JSON-safe result with answer and source URLs",
            effectful=True,
            tags=("effectful", "research", "web"),
        ),
    ),
    tags=("research", "builtin", "web"),
)

__all__ = ["MODULE_SPEC", "search"]