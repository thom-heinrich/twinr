"""Build small operator-facing summaries from self-coding runtime state."""

from __future__ import annotations

import logging
import math  # AUDIT-FIX(#4): Reject non-finite numeric payloads from corrupted persisted state.
from collections.abc import Callable, Mapping  # AUDIT-FIX(#1): Support safe access to generic mapping-backed records.
from dataclasses import dataclass
from datetime import datetime, timezone  # AUDIT-FIX(#3): Normalize timestamp comparisons to timezone-aware UTC.
from typing import Any

from twinr.agent.self_coding.status import LearnedSkillStatus
from twinr.agent.self_coding.store import SelfCodingStore

logger = logging.getLogger(__name__)

_MAX_OPERATOR_TEXT_LENGTH = 120
_MAX_OPERATOR_DETAIL_LENGTH = 160  # AUDIT-FIX(#5): Cap the fully rendered detail line for small operator displays.
_LATEST_STATUS_RECENCY_FIELDS: tuple[str, ...] = (
    "updated_at",
    "last_updated_at",
    "created_at",
    "timestamp",
)
_BIDI_CONTROL_CHARACTERS = frozenset(
    {
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    },
)  # AUDIT-FIX(#5): Strip invisible direction controls that can confuse display or printer rendering.
_BUILD_ISSUE_LABELS: dict[str, str] = {
    "activations": "activation data",
    "compile statuses": "compile status data",
    "live e2e statuses": "live test status",
}  # AUDIT-FIX(#6): Surface degraded reads in plain operator language instead of masking them as inactivity.


def _record_build_issue(issues: list[str] | None, description: str) -> None:
    if issues is None or description in issues:
        return
    issues.append(description)


def _coerce_non_negative_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):  # AUDIT-FIX(#4): Avoid silently turning malformed booleans into 0/1 counters.
        return default
    try:
        coerced = int(value)
    except (TypeError, ValueError, OverflowError):  # AUDIT-FIX(#4): Overflow is possible with corrupted numeric payloads.
        return default
    return coerced if coerced >= 0 else default


def _coerce_non_negative_float(value: object, *, default: float | None = None) -> float | None:
    if isinstance(value, bool):  # AUDIT-FIX(#4): Avoid silently turning malformed booleans into durations.
        return default
    try:
        coerced = float(value)
    except (TypeError, ValueError, OverflowError):  # AUDIT-FIX(#4): Reject overflow from malformed persisted values.
        return default
    if not math.isfinite(coerced) or coerced < 0.0:  # AUDIT-FIX(#4): Reject NaN/inf and negative durations from corrupted state.
        return default
    return coerced


def _sanitize_summary_text(
    value: object | None,
    *,
    max_length: int = _MAX_OPERATOR_TEXT_LENGTH,
) -> str | None:
    if value is None:
        return None
    try:
        raw_text = str(value)
    except Exception:  # AUDIT-FIX(#1): Malformed record objects can raise during __str__.
        logger.warning(
            "Failed to stringify %s while building self-coding operator status.",
            type(value).__name__,
        )
        return None

    filtered_text = "".join(
        char
        for char in raw_text
        if char not in _BIDI_CONTROL_CHARACTERS and (char.isprintable() or char.isspace())
    )  # AUDIT-FIX(#5): Remove non-printable/control characters before collapsing whitespace.
    text = " ".join(filtered_text.split())
    if not text:
        return None
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 1].rstrip()}…"


def _join_operator_summary_parts(parts: list[str], *, fallback: str) -> str:
    text = _sanitize_summary_text(
        " · ".join(part for part in parts if part),
        max_length=_MAX_OPERATOR_DETAIL_LENGTH,
    )  # AUDIT-FIX(#5): Cap the final joined operator string, not just individual fields.
    return text or fallback


def _get_field(item: object, name: str, default: Any = None) -> Any:
    if item is None:
        return default
    if isinstance(item, Mapping):
        try:
            return item.get(name, default)
        except Exception:  # AUDIT-FIX(#1): Custom mappings can raise during .get().
            logger.warning(
                "Failed to read mapping field %s from %s; using default.",
                name,
                type(item).__name__,
            )
            return default
    try:
        return getattr(item, name)
    except AttributeError:
        return default
    except Exception:  # AUDIT-FIX(#1): Descriptor/property access can raise on malformed records.
        logger.warning(
            "Failed to read attribute field %s from %s; using default.",
            name,
            type(item).__name__,
        )
        return default


def _safe_store_items(
    description: str,
    loader: Callable[[], object],
    *,
    issues: list[str] | None = None,
) -> list[object]:
    try:
        items = loader()
    except Exception:
        logger.exception("Failed to load %s for self-coding operator status.", description)
        _record_build_issue(issues, description)  # AUDIT-FIX(#6): Preserve visibility that the summary is degraded.
        return []

    if items is None:
        logger.warning(
            "Store returned None for %s while building self-coding operator status.",
            description,
        )
        _record_build_issue(issues, description)  # AUDIT-FIX(#6): None here is unavailable data, not real inactivity.
        return []

    if isinstance(items, (str, bytes)):
        logger.warning(
            "Store returned scalar text %r for %s while building self-coding operator status.",
            type(items).__name__,
            description,
        )
        _record_build_issue(issues, description)
        return []

    if isinstance(items, Mapping):  # AUDIT-FIX(#2): A mapping here is a malformed store shape; listing its keys would be wrong.
        logger.warning(
            "Store returned mapping %r for %s while building self-coding operator status.",
            type(items).__name__,
            description,
        )
        _record_build_issue(issues, description)
        return []

    try:
        return list(items)
    except Exception:  # AUDIT-FIX(#2): Iterators can fail mid-materialization with arbitrary exceptions, not just TypeError.
        logger.exception(
            "Failed to materialize %s while building self-coding operator status.",
            description,
        )
        _record_build_issue(issues, description)
        return []


def _coerce_recency_sort_key(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return numeric
    else:
        try:
            text = str(value).strip()
        except Exception:
            return None
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            if text.endswith("Z"):
                text = f"{text[:-1]}+00:00"
            try:
                timestamp = datetime.fromisoformat(text)
            except ValueError:
                return None
        else:
            if not math.isfinite(numeric):
                return None
            return numeric

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return None  # AUDIT-FIX(#3): Ignore naive datetimes to avoid DST/offset mis-ordering.
    return timestamp.astimezone(timezone.utc).timestamp()


def _select_latest_compile_status(
    statuses: list[object],
    *,
    issues: list[str] | None = None,
    description: str = "compile statuses",
) -> object | None:
    if not statuses:
        return None

    for field_name in _LATEST_STATUS_RECENCY_FIELDS:
        best_item: object | None = None
        best_key: tuple[float, int] | None = None
        unusable_count = 0

        for index, item in enumerate(statuses):
            recency_key = _coerce_recency_sort_key(_get_field(item, field_name))
            if recency_key is None:
                unusable_count += 1
                continue
            candidate_key = (recency_key, index)
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_item = item

        if best_item is not None:
            if unusable_count:
                logger.warning(
                    "Ignored %d %s entries with unusable %s while selecting latest.",
                    unusable_count,
                    description,
                    field_name,
                )
            return best_item

    if len(statuses) > 1:
        logger.warning(
            "No usable recency field found while selecting latest %s; falling back to store order.",
            description,
        )
        _record_build_issue(issues, description)  # AUDIT-FIX(#3): Signal ambiguous latest-selection instead of silently trusting store order.
    return statuses[0]


def _load_latest_compile_status(
    store: SelfCodingStore,
    *,
    issues: list[str] | None = None,
) -> object | None:
    getter = getattr(store, "get_latest_compile_status", None)
    if callable(getter):
        try:
            latest = getter()
        except Exception:
            logger.exception(
                "Failed to load latest compile status directly for self-coding operator status.",
            )
        else:
            if latest is not None:  # AUDIT-FIX(#3): Fall back to list-based selection when the direct accessor yields no record.
                return latest

    compile_statuses = _safe_store_items(
        "compile statuses",
        lambda: store.list_compile_statuses(),
        issues=issues,
    )
    return _select_latest_compile_status(
        compile_statuses,
        issues=issues,
        description="compile statuses",
    )


def _load_latest_live_e2e_status(
    store: SelfCodingStore,
    *,
    issues: list[str] | None = None,
) -> object | None:
    getter = getattr(store, "get_latest_live_e2e_status", None)  # AUDIT-FIX(#3): Prefer a store-provided latest accessor when present.
    if callable(getter):
        try:
            latest = getter()
        except Exception:
            logger.exception(
                "Failed to load latest live e2e status directly for self-coding operator status.",
            )
        else:
            if latest is not None:  # AUDIT-FIX(#3): Fall back to list-based selection when the direct accessor yields no record.
                return latest

    live_e2e_statuses = _safe_store_items(
        "live e2e statuses",
        lambda: store.list_live_e2e_statuses(),
        issues=issues,
    )
    return _select_latest_compile_status(
        live_e2e_statuses,
        issues=issues,
        description="live e2e statuses",
    )


def _normalize_status_token(value: str) -> str:
    return value.strip().replace("-", "_").replace(" ", "_").lower()


def _status_tokens(value: object) -> set[str]:
    tokens: set[str] = set()
    if value is None:
        return tokens

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return tokens
        tokens.add(_normalize_status_token(text))
        if "." in text:
            tokens.add(_normalize_status_token(text.rsplit(".", 1)[-1]))
        return tokens

    for candidate in (_get_field(value, "value"), _get_field(value, "name")):  # AUDIT-FIX(#7): Support enum-like persisted values and names.
        if isinstance(candidate, str) and candidate.strip():
            tokens.add(_normalize_status_token(candidate))

    try:
        text = str(value).strip()
    except Exception:
        return tokens
    if not text:
        return tokens

    tokens.add(_normalize_status_token(text))
    if "." in text:
        tokens.add(_normalize_status_token(text.rsplit(".", 1)[-1]))
    return tokens


def _status_matches(value: object, expected_status: LearnedSkillStatus) -> bool:
    if value == expected_status:
        return True
    expected_tokens = _status_tokens(expected_status)
    actual_tokens = _status_tokens(value)
    return bool(expected_tokens & actual_tokens)


def _count_activations(
    activations: list[object],
    expected_status: LearnedSkillStatus,
) -> int:
    return sum(
        1
        for item in activations
        if _status_matches(_get_field(item, "status"), expected_status)
    )  # AUDIT-FIX(#7): Count persisted enum/name/value/string forms consistently across JSON-backed stores.


def _summarize_build_issues(issues: list[str]) -> str | None:
    labels: list[str] = []
    for issue in issues:
        label = _BUILD_ISSUE_LABELS.get(issue, issue)
        if label not in labels:
            labels.append(label)

    if not labels:
        return None
    if len(labels) == 1:
        return _sanitize_summary_text(
            f"{labels[0]} temporarily unavailable",
            max_length=_MAX_OPERATOR_DETAIL_LENGTH,
        )
    return "Some self-coding status data is temporarily unavailable."  # AUDIT-FIX(#6): Make degraded-state visible in operator text.


@dataclass(frozen=True, slots=True)
class SelfCodingOperatorStatus:
    """Compact dashboard summary for compile and activation state."""

    active_count: int = 0
    soft_launch_count: int = 0
    paused_count: int = 0
    latest_phase: str | None = None
    latest_driver_name: str | None = None
    latest_event_kind: str | None = None
    latest_event_count: int = 0
    latest_model: str | None = None
    latest_reasoning_effort: str | None = None
    latest_duration_seconds: float | None = None
    latest_fallback_reason: str | None = None
    latest_live_e2e_status: str | None = None
    latest_live_e2e_suite: str | None = None
    latest_live_e2e_environment: str | None = None
    is_degraded: bool = False  # AUDIT-FIX(#6): Preserve whether the summary was built from partial/unavailable data.
    degraded_detail: str | None = None  # AUDIT-FIX(#6): Surface operator-readable unavailable-state text.

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_count", _coerce_non_negative_int(self.active_count))
        object.__setattr__(self, "soft_launch_count", _coerce_non_negative_int(self.soft_launch_count))
        object.__setattr__(self, "paused_count", _coerce_non_negative_int(self.paused_count))
        object.__setattr__(self, "latest_event_count", _coerce_non_negative_int(self.latest_event_count))
        object.__setattr__(self, "latest_phase", _sanitize_summary_text(self.latest_phase))
        object.__setattr__(self, "latest_driver_name", _sanitize_summary_text(self.latest_driver_name))
        object.__setattr__(self, "latest_event_kind", _sanitize_summary_text(self.latest_event_kind))
        object.__setattr__(self, "latest_model", _sanitize_summary_text(self.latest_model))
        object.__setattr__(self, "latest_reasoning_effort", _sanitize_summary_text(self.latest_reasoning_effort))
        object.__setattr__(self, "latest_duration_seconds", _coerce_non_negative_float(self.latest_duration_seconds))
        object.__setattr__(self, "latest_fallback_reason", _sanitize_summary_text(self.latest_fallback_reason))
        object.__setattr__(self, "latest_live_e2e_status", _sanitize_summary_text(self.latest_live_e2e_status))
        object.__setattr__(self, "latest_live_e2e_suite", _sanitize_summary_text(self.latest_live_e2e_suite))
        object.__setattr__(self, "latest_live_e2e_environment", _sanitize_summary_text(self.latest_live_e2e_environment))
        object.__setattr__(self, "is_degraded", bool(self.is_degraded))  # AUDIT-FIX(#6): Keep degraded state total and explicit.
        object.__setattr__(
            self,
            "degraded_detail",
            _sanitize_summary_text(
                self.degraded_detail,
                max_length=_MAX_OPERATOR_DETAIL_LENGTH,
            ),
        )

    @property
    def has_activity(self) -> bool:
        return any(
            (
                self.active_count,
                self.soft_launch_count,
                self.paused_count,
                self.latest_event_count,
                self.latest_phase,
                self.latest_driver_name,
                self.latest_event_kind,
                self.latest_model,
                self.latest_reasoning_effort,
                self.latest_duration_seconds is not None,
                self.latest_fallback_reason,
                self.latest_live_e2e_status,
                self.latest_live_e2e_suite,
                self.latest_live_e2e_environment,
                self.is_degraded,
                self.degraded_detail,
            ),
        )  # AUDIT-FIX(#6): Degraded/partial data should still keep the operator card visible.

    def card_value(self) -> str:
        if self.is_degraded and not any((self.active_count, self.soft_launch_count, self.paused_count)):
            return "Status degraded"  # AUDIT-FIX(#6): Avoid falsely showing zero activity when counts are unavailable.
        return f"{self.active_count} active · {self.soft_launch_count} soft launch"

    def card_detail(self) -> str:
        parts: list[str] = []
        if self.latest_phase:
            parts.append(self.latest_phase.replace("_", " "))
        if self.latest_driver_name:
            parts.append(self.latest_driver_name)
        if self.latest_model:
            parts.append(self.latest_model)
        if self.latest_event_count:
            parts.append(f"{self.latest_event_count} events")
        if self.latest_event_kind:
            parts.append(self.latest_event_kind)
        if self.latest_live_e2e_status:
            parts.append(f"live e2e {self.latest_live_e2e_status.replace('_', ' ')}")
        if self.is_degraded and self.degraded_detail:
            parts.append(self.degraded_detail)  # AUDIT-FIX(#6): Make partial-read conditions explicit to operators.
        if parts:
            return _join_operator_summary_parts(
                parts,
                fallback="Self-coding status available.",
            )

        activation_parts: list[str] = []
        if self.active_count:
            activation_parts.append(f"{self.active_count} active")
        if self.soft_launch_count:
            activation_parts.append(f"{self.soft_launch_count} soft launch ready")
        if self.paused_count:
            activation_parts.append(f"{self.paused_count} paused")
        if self.is_degraded and self.degraded_detail:
            activation_parts.append(self.degraded_detail)
        if activation_parts:
            return _join_operator_summary_parts(
                activation_parts,
                fallback="Self-coding status available.",
            )

        if self.is_degraded and self.degraded_detail:
            return self.degraded_detail
        return "No recent self-coding activity."


def build_self_coding_operator_status(store: SelfCodingStore) -> SelfCodingOperatorStatus:
    """Summarize persisted self-coding state for dashboard rendering."""

    issues: list[str] = []  # AUDIT-FIX(#6): Distinguish unavailable data from true inactivity in the rendered summary.
    activations = _safe_store_items(
        "activations",
        lambda: store.list_activations(),
        issues=issues,
    )
    latest_status = _load_latest_compile_status(store, issues=issues)
    latest_live_e2e = _load_latest_live_e2e_status(store, issues=issues)

    diagnostics_value = {} if latest_status is None else _get_field(latest_status, "diagnostics", {})
    diagnostics: Mapping[str, Any]
    if isinstance(diagnostics_value, Mapping):  # AUDIT-FIX(#1): Accept generic mapping-backed diagnostics payloads.
        diagnostics = diagnostics_value
    else:
        diagnostics = {}

    degraded_detail = _summarize_build_issues(issues)
    return SelfCodingOperatorStatus(
        active_count=_count_activations(activations, LearnedSkillStatus.ACTIVE),
        soft_launch_count=_count_activations(
            activations,
            LearnedSkillStatus.SOFT_LAUNCH_READY,
        ),
        paused_count=_count_activations(activations, LearnedSkillStatus.PAUSED),
        latest_phase=None if latest_status is None else _get_field(latest_status, "phase"),
        latest_driver_name=None if latest_status is None else _get_field(latest_status, "driver_name"),
        latest_event_kind=None if latest_status is None else _get_field(latest_status, "last_event_kind"),
        latest_event_count=0 if latest_status is None else _get_field(latest_status, "event_count", 0),
        latest_model=diagnostics.get("model"),
        latest_reasoning_effort=diagnostics.get("reasoning_effort"),
        latest_duration_seconds=diagnostics.get("duration_seconds"),
        latest_fallback_reason=diagnostics.get("fallback_reason"),
        latest_live_e2e_status=None if latest_live_e2e is None else _get_field(latest_live_e2e, "status"),
        latest_live_e2e_suite=None if latest_live_e2e is None else _get_field(latest_live_e2e, "suite_id"),
        latest_live_e2e_environment=None if latest_live_e2e is None else _get_field(latest_live_e2e, "environment"),
        is_degraded=degraded_detail is not None,
        degraded_detail=degraded_detail,
    )