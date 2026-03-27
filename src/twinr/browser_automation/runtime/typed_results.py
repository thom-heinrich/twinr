"""Typed result helpers for Twinr browser automation.

These helpers keep evaluator-facing structured payloads out of the ignored
browser workspace and out of benchmark-specific code. The executor may still
return prose summaries, but any browser flow that can justify a structured
result should attach one canonical ``typed_result`` object for downstream
verifiers and benchmark adapters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import json

_ALLOWED_TYPED_RESULT_STATUSES = frozenset({"success", "not_found", "unsupported"})


def _jsonable(value: object, *, field_name: str) -> Any:
    """Return one JSON-safe value or fail clearly."""

    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be JSON serializable") from exc


def _normalize_points(values: Sequence[object] | None) -> tuple[str, ...]:
    """Return compact human-readable key points."""

    points: list[str] = []
    for item in list(values or ())[:8]:
        text = " ".join(str(item or "").strip().split())
        if text:
            points.append(text[:240])
    return tuple(points)


def results_schema_expects_null(results_schema: object | None) -> bool:
    """Return whether a results schema explicitly models absence as ``null``."""

    if not isinstance(results_schema, Mapping):
        return False
    return str(results_schema.get("type") or "").strip().lower() == "null"


@dataclass(frozen=True, slots=True)
class BrowserTypedResult:
    """Canonical typed result attached to browser runs when structured evidence exists."""

    status: str
    results: Any = None
    answer_text: str | None = None
    reason: str = ""
    key_points: tuple[str, ...] = ()
    used_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        status = str(self.status or "").strip().lower()
        if status not in _ALLOWED_TYPED_RESULT_STATUSES:
            raise ValueError(f"status must be one of {sorted(_ALLOWED_TYPED_RESULT_STATUSES)}")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "results", _jsonable(self.results, field_name="results"))
        answer_text = str(self.answer_text or "").strip() or None
        object.__setattr__(self, "answer_text", answer_text)
        object.__setattr__(self, "reason", " ".join(str(self.reason or "").strip().split())[:800])
        object.__setattr__(self, "key_points", _normalize_points(self.key_points))
        capabilities = tuple(
            " ".join(str(item or "").strip().split())[:120]
            for item in list(self.used_capabilities or ())[:8]
            if str(item or "").strip()
        )
        object.__setattr__(self, "used_capabilities", capabilities)

    def to_json(self) -> dict[str, Any]:
        """Return the JSON-safe representation stored in result data."""

        return {
            "status": self.status,
            "results": self.results,
            "answer_text": self.answer_text,
            "reason": self.reason,
            "key_points": list(self.key_points),
            "used_capabilities": list(self.used_capabilities),
        }


@dataclass(frozen=True, slots=True)
class BrowserAuthStateSummary:
    """Small typed snapshot of the authenticated browser state during a rescue."""

    current_url: str
    used_authenticated_context: bool
    visible_link_count: int
    content_block_count: int
    candidate_href_count: int
    visited_url_count: int

    def to_json(self) -> dict[str, Any]:
        """Return the JSON-safe auth-state payload stored in result data."""

        return {
            "current_url": str(self.current_url or "").strip(),
            "used_authenticated_context": bool(self.used_authenticated_context),
            "visible_link_count": int(max(0, self.visible_link_count)),
            "content_block_count": int(max(0, self.content_block_count)),
            "candidate_href_count": int(max(0, self.candidate_href_count)),
            "visited_url_count": int(max(0, self.visited_url_count)),
        }


def build_typed_result(
    *,
    status: str,
    results: object = None,
    answer_text: str | None = None,
    reason: str = "",
    key_points: Sequence[object] | None = None,
    used_capabilities: Sequence[object] | None = None,
) -> dict[str, Any]:
    """Build one canonical typed browser result."""

    return BrowserTypedResult(
        status=status,
        results=results,
        answer_text=answer_text,
        reason=reason,
        key_points=_normalize_points(key_points),
        used_capabilities=tuple(str(item or "") for item in list(used_capabilities or ())),
    ).to_json()


def normalize_typed_result(value: object) -> dict[str, Any] | None:
    """Validate one loosely-shaped typed result payload from browser data."""

    if not isinstance(value, Mapping):
        return None
    try:
        return BrowserTypedResult(
            status=str(value.get("status") or ""),
            results=value.get("results"),
            answer_text=str(value.get("answer_text") or "").strip() or None,
            reason=str(value.get("reason") or ""),
            key_points=tuple(value.get("key_points") or ()),
            used_capabilities=tuple(value.get("used_capabilities") or ()),
        ).to_json()
    except (TypeError, ValueError):
        return None


def build_auth_state_summary(
    *,
    current_url: str,
    used_authenticated_context: bool,
    visible_link_count: int,
    content_block_count: int,
    candidate_href_count: int,
    visited_url_count: int,
) -> dict[str, Any]:
    """Build one small auth-state summary payload."""

    return BrowserAuthStateSummary(
        current_url=str(current_url or "").strip(),
        used_authenticated_context=bool(used_authenticated_context),
        visible_link_count=int(visible_link_count),
        content_block_count=int(content_block_count),
        candidate_href_count=int(candidate_href_count),
        visited_url_count=int(visited_url_count),
    ).to_json()
