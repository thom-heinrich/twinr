"""Build small operator-facing summaries from self-coding runtime state."""

from __future__ import annotations

import logging  # AUDIT-FIX(#1): Add structured logging so degraded store reads do not fail silently.
from collections.abc import Callable  # AUDIT-FIX(#1): Type safe loader helpers for resilient store access.
from dataclasses import dataclass
from typing import Any  # AUDIT-FIX(#1): Allow safe attribute/mapping field extraction across malformed records.

from twinr.agent.self_coding.status import LearnedSkillStatus
from twinr.agent.self_coding.store import SelfCodingStore

logger = logging.getLogger(__name__)  # AUDIT-FIX(#1): Record degraded reads without crashing dashboard rendering.

_MAX_OPERATOR_TEXT_LENGTH = 120  # AUDIT-FIX(#3): Bound operator-card text so corrupted state cannot flood small displays.
_LATEST_STATUS_RECENCY_FIELDS: tuple[str, ...] = (
    "updated_at",
    "last_updated_at",
    "created_at",
    "timestamp",
)  # AUDIT-FIX(#2): Prefer explicit recency fields when store ordering is not guaranteed.


def _coerce_non_negative_int(value: object, *, default: int = 0) -> int:  # AUDIT-FIX(#3): Normalize malformed persisted counters to safe non-negative integers.
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return coerced if coerced >= 0 else default


def _sanitize_summary_text(value: object | None) -> str | None:  # AUDIT-FIX(#3): Collapse control characters/whitespace and cap text length for stable operator-facing rendering.
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    if len(text) <= _MAX_OPERATOR_TEXT_LENGTH:
        return text
    return f"{text[: _MAX_OPERATOR_TEXT_LENGTH - 1].rstrip()}…"


def _get_field(item: object, name: str, default: Any = None) -> Any:  # AUDIT-FIX(#1): Read both mapping- and attribute-based store records without raising on malformed entries.
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _safe_store_items(description: str, loader: Callable[[], object]) -> list[object]:  # AUDIT-FIX(#1): Degrade to an empty dataset if the file-backed store raises or returns an invalid shape.
    try:
        items = loader()
    except Exception:
        logger.exception("Failed to load %s for self-coding operator status.", description)
        return []

    if items is None:
        logger.warning(
            "Store returned None for %s while building self-coding operator status.",
            description,
        )
        return []

    if isinstance(items, (str, bytes)):
        logger.warning(
            "Store returned scalar text %r for %s while building self-coding operator status.",
            type(items).__name__,
            description,
        )
        return []

    try:
        return list(items)
    except TypeError:
        logger.warning(
            "Store returned non-iterable %r for %s while building self-coding operator status.",
            type(items).__name__,
            description,
        )
        return []


def _select_latest_compile_status(statuses: list[object]) -> object | None:  # AUDIT-FIX(#2): Pick the newest compile status by explicit recency markers instead of assuming index 0 is latest.
    if not statuses:
        return None

    for field_name in _LATEST_STATUS_RECENCY_FIELDS:
        field_values = [_get_field(item, field_name) for item in statuses]
        if any(value is None for value in field_values):
            continue
        try:
            return max(statuses, key=lambda item: _get_field(item, field_name))
        except TypeError:
            logger.warning(
                "Compile status field %s is not mutually comparable; falling back to store order.",
                field_name,
            )
            break

    return statuses[0]


def _load_latest_compile_status(store: SelfCodingStore) -> object | None:  # AUDIT-FIX(#1): Isolate store-read failures and keep rendering alive. AUDIT-FIX(#2): Prefer store-provided latest accessors when available.
    getter = getattr(store, "get_latest_compile_status", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            logger.exception(
                "Failed to load latest compile status directly for self-coding operator status.",
            )

    compile_statuses = _safe_store_items(
        "compile statuses",
        lambda: store.list_compile_statuses(),
    )
    return _select_latest_compile_status(compile_statuses)


def _count_activations(
    activations: list[object],
    expected_status: LearnedSkillStatus,
) -> int:  # AUDIT-FIX(#1): Ignore malformed activation records instead of crashing during status aggregation.
    return sum(1 for item in activations if _get_field(item, "status") == expected_status)


@dataclass(frozen=True, slots=True)
class SelfCodingOperatorStatus:
    """Compact dashboard summary for compile and activation state."""

    active_count: int = 0
    soft_launch_count: int = 0
    latest_phase: str | None = None
    latest_driver_name: str | None = None
    latest_event_kind: str | None = None
    latest_event_count: int = 0

    def __post_init__(self) -> None:  # AUDIT-FIX(#3): Validate and normalize persisted values at construction time so downstream properties stay total and side-effect free.
        object.__setattr__(self, "active_count", _coerce_non_negative_int(self.active_count))
        object.__setattr__(self, "soft_launch_count", _coerce_non_negative_int(self.soft_launch_count))
        object.__setattr__(self, "latest_event_count", _coerce_non_negative_int(self.latest_event_count))
        object.__setattr__(self, "latest_phase", _sanitize_summary_text(self.latest_phase))
        object.__setattr__(self, "latest_driver_name", _sanitize_summary_text(self.latest_driver_name))
        object.__setattr__(self, "latest_event_kind", _sanitize_summary_text(self.latest_event_kind))

    @property
    def has_activity(self) -> bool:
        return any(
            (
                self.active_count,
                self.soft_launch_count,
                self.latest_event_count,
                self.latest_phase,
                self.latest_driver_name,
                self.latest_event_kind,
            ),
        )  # AUDIT-FIX(#4): Treat compile metadata itself as activity so in-progress work is not hidden when event_count is zero.

    def card_value(self) -> str:
        return f"{self.active_count} active · {self.soft_launch_count} soft launch"

    def card_detail(self) -> str:
        parts: list[str] = []
        if self.latest_phase:
            parts.append(self.latest_phase.replace("_", " "))
        if self.latest_driver_name:
            parts.append(self.latest_driver_name)
        if self.latest_event_count:
            parts.append(f"{self.latest_event_count} events")
        if self.latest_event_kind:
            parts.append(self.latest_event_kind)
        if parts:
            return " · ".join(parts)

        activation_parts: list[str] = []
        if self.active_count:
            activation_parts.append(f"{self.active_count} active")
        if self.soft_launch_count:
            activation_parts.append(f"{self.soft_launch_count} soft launch ready")
        if activation_parts:
            return " · ".join(activation_parts)  # AUDIT-FIX(#5): Avoid falsely reporting "no activity" when activation counts exist but no compile-status detail is present.
        return "No recent self-coding activity."


def build_self_coding_operator_status(store: SelfCodingStore) -> SelfCodingOperatorStatus:
    """Summarize persisted self-coding state for dashboard rendering."""

    activations = _safe_store_items(
        "activations",
        lambda: store.list_activations(),
    )  # AUDIT-FIX(#1): Prevent file-backed store errors from taking down the dashboard summary path.
    latest_status = _load_latest_compile_status(store)
    return SelfCodingOperatorStatus(
        active_count=_count_activations(activations, LearnedSkillStatus.ACTIVE),
        soft_launch_count=_count_activations(
            activations,
            LearnedSkillStatus.SOFT_LAUNCH_READY,
        ),
        latest_phase=None if latest_status is None else _get_field(latest_status, "phase"),
        latest_driver_name=None if latest_status is None else _get_field(latest_status, "driver_name"),
        latest_event_kind=None if latest_status is None else _get_field(latest_status, "last_event_kind"),
        latest_event_count=0 if latest_status is None else _get_field(latest_status, "event_count", 0),
    )  # AUDIT-FIX(#1): Use safe field extraction so partially corrupted records produce a degraded summary instead of an exception. AUDIT-FIX(#3): Route values through the dataclass normalizer.