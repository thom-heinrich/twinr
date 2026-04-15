"""Build operator-facing display status forensics and transition payloads.

The display loop already knows what it renders, but historically the causal
inputs behind operator-visible ``ok``/``warn``/``error`` states were scattered
across snapshot, health, and remote-memory logs. This module centralizes the
display-side decision record so the loop can persist one stable transition
event whenever the visible operator status or its cause changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinr.agent.base_agent.state.snapshot import RuntimeSnapshot
from twinr.ops.health import TwinrSystemHealth, assess_memory_pressure_status


_MAX_LABEL_CHARS = 32
_MAX_DETAIL_CHARS = 240
_MAX_REASON_CODE_CHARS = 48
_MAX_REASON_CODES = 8


def _compact_text(value: object | None, *, max_chars: int) -> str | None:
    if value is None:
        return None
    text = "".join(ch if ch.isprintable() else " " for ch in str(value))
    compact = " ".join(text.split()).strip()
    if max_chars > 0 and len(compact) > max_chars:
        compact = compact[:max_chars].rstrip()
    return compact or None


def _normalized_label(value: object | None, *, fallback: str = "") -> str:
    return (_compact_text(value, max_chars=_MAX_LABEL_CHARS) or fallback).strip()


def _normalized_optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return round(float(value), 1)
    if isinstance(value, str):
        try:
            return round(float(value.strip()), 1)
        except ValueError:
            return None
    try:
        return round(float(str(value).strip()), 1)
    except (TypeError, ValueError):
        return None


def _normalized_optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalize_reason_codes(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values[:_MAX_REASON_CODES]:
        compact = _compact_text(value, max_chars=_MAX_REASON_CODE_CHARS)
        if compact:
            normalized.append(compact)
    return tuple(normalized)


def _snapshot_status(snapshot: RuntimeSnapshot | None) -> str | None:
    if snapshot is None:
        return None
    return (_compact_text(getattr(snapshot, "status", None), max_chars=_MAX_LABEL_CHARS) or "").lower() or None


def _snapshot_error(snapshot: RuntimeSnapshot | None) -> str | None:
    if snapshot is None:
        return None
    return _compact_text(getattr(snapshot, "error_message", None), max_chars=_MAX_DETAIL_CHARS)


def _service_state(health: TwinrSystemHealth | None, key: str) -> tuple[bool | None, int | None, str | None]:
    if health is None:
        return None, None, None
    for service in getattr(health, "services", ()) or ():
        if getattr(service, "key", None) != key:
            continue
        detail = _compact_text(getattr(service, "detail", None), max_chars=_MAX_DETAIL_CHARS)
        count = _normalized_optional_int(getattr(service, "count", None))
        return bool(getattr(service, "running", False)), count, detail
    return None, None, None


def _system_label_for_status(status: str) -> str:
    if status == "error":
        return "Fehler"
    if status == "warn":
        return "Achtung"
    if status == "unknown":
        return "?"
    return "ok"


def _operator_status_from_label(label: str, *, fallback: str) -> str:
    normalized = _normalized_label(label).lower()
    if normalized in {"ok", "healthy", "ready"}:
        return "ok"
    if normalized in {"achtung", "warn", "warning", "degraded", "starting", "connecting", "booting"}:
        return "warn"
    if normalized in {"fehler", "error", "fail", "failed", "fault", "offline", "down"}:
        return "error"
    if normalized:
        return "error"
    return fallback


@dataclass(frozen=True, slots=True)
class DisplaySystemStatusDecision:
    """Capture the causal decision behind the visible operator system status."""

    status: str
    label: str
    reason_codes: tuple[str, ...] = ()
    reason_detail: str | None = None
    snapshot_status: str | None = None
    snapshot_error: str | None = None
    runtime_error: str | None = None
    health_status: str | None = None
    conversation_loop_running: bool | None = None
    conversation_loop_count: int | None = None
    conversation_loop_detail: str | None = None
    memory_pressure_status: str | None = None
    memory_owner_label: str | None = None
    memory_owner_detail: str | None = None
    memory_available_mb: int | None = None
    memory_used_percent: float | None = None
    disk_used_percent: float | None = None
    cpu_temperature_c: float | None = None

    def stability_signature(self) -> tuple[object, ...]:
        """Return the bounded qualitative decision used for recovery holds."""

        disk_status = "warn" if self.disk_used_percent is not None and self.disk_used_percent >= 85.0 else "ok"
        return (
            self.status,
            self.label,
            self.reason_codes,
            self.reason_detail,
            self.snapshot_status,
            self.snapshot_error,
            self.runtime_error,
            self.health_status,
            self.conversation_loop_running,
            self.conversation_loop_count,
            self.memory_pressure_status,
            disk_status,
        )


@dataclass(frozen=True, slots=True)
class DisplayStatusForensics:
    """Persistable display-side record for one visible operator status state."""

    operator_status: str
    operator_label: str
    runtime_status: str
    headline: str
    ai_label: str
    internet_label: str
    snapshot_status: str | None = None
    snapshot_stale: bool = False
    snapshot_error: str | None = None
    runtime_error: str | None = None
    health_status: str | None = None
    conversation_loop_running: bool | None = None
    conversation_loop_count: int | None = None
    conversation_loop_detail: str | None = None
    memory_pressure_status: str | None = None
    memory_owner_label: str | None = None
    memory_owner_detail: str | None = None
    memory_available_mb: int | None = None
    memory_used_percent: float | None = None
    disk_used_percent: float | None = None
    cpu_temperature_c: float | None = None
    reason_codes: tuple[str, ...] = ()
    reason_detail: str | None = None

    def transition_signature(self) -> tuple[object, ...]:
        """Return the bounded qualitative state used to suppress duplicate events."""

        disk_status = "warn" if self.disk_used_percent is not None and self.disk_used_percent >= 85.0 else "ok"
        return (
            self.operator_status,
            self.operator_label,
            self.ai_label,
            self.internet_label,
            self.snapshot_status,
            bool(self.snapshot_stale),
            self.snapshot_error,
            self.runtime_error,
            self.health_status,
            self.conversation_loop_running,
            self.conversation_loop_count,
            self.memory_pressure_status,
            disk_status,
            self.reason_codes,
            self.reason_detail,
        )

    def primary_reason(self) -> str | None:
        return self.reason_codes[0] if self.reason_codes else None

    def event_level(self) -> str:
        if self.operator_status == "error":
            return "error"
        if self.operator_status == "warn":
            return "warning"
        return "info"

    def event_message(self, previous: "DisplayStatusForensics | None") -> str:
        reason = self.primary_reason()
        if previous is None:
            base = f"Display operator status observed initial state {self.operator_status}."
        elif previous.operator_status != self.operator_status:
            base = (
                "Display operator status changed from "
                f"{previous.operator_status} to {self.operator_status}."
            )
        else:
            base = f"Display operator status cause changed while staying {self.operator_status}."
        if reason:
            return f"{base} cause={reason}."
        return base

    def telemetry_line(self, previous: "DisplayStatusForensics | None") -> str:
        parts = [
            f"display_operator_status={self.operator_status}",
            f"prev={(previous.operator_status if previous is not None else 'none')}",
        ]
        reason = self.primary_reason()
        if reason:
            parts.append(f"reason={reason}")
        if self.snapshot_status:
            parts.append(f"snapshot={self.snapshot_status}")
        if self.health_status:
            parts.append(f"health={self.health_status}")
        return " ".join(parts)

    def event_data(self, previous: "DisplayStatusForensics | None") -> dict[str, object]:
        return {
            "previous_status": None if previous is None else previous.operator_status,
            "status": self.operator_status,
            "previous_reason_codes": [] if previous is None else list(previous.reason_codes),
            "reason_codes": list(self.reason_codes),
            "previous_reason_detail": None if previous is None else previous.reason_detail,
            "reason_detail": self.reason_detail,
            "status_changed": previous is None or previous.operator_status != self.operator_status,
            "cause_changed": previous is None or previous.transition_signature() != self.transition_signature(),
            "operator_label": self.operator_label,
            "runtime_status": self.runtime_status,
            "headline": self.headline,
            "ai_label": self.ai_label,
            "internet_label": self.internet_label,
            "snapshot_status": self.snapshot_status,
            "snapshot_stale": self.snapshot_stale,
            "snapshot_error": self.snapshot_error,
            "runtime_error": self.runtime_error,
            "health_status": self.health_status,
            "conversation_loop_running": self.conversation_loop_running,
            "conversation_loop_count": self.conversation_loop_count,
            "conversation_loop_detail": self.conversation_loop_detail,
            "memory_pressure_status": self.memory_pressure_status,
            "memory_owner_label": self.memory_owner_label,
            "memory_owner_detail": self.memory_owner_detail,
            "memory_available_mb": self.memory_available_mb,
            "memory_used_percent": self.memory_used_percent,
            "disk_used_percent": self.disk_used_percent,
            "cpu_temperature_c": self.cpu_temperature_c,
        }


def build_system_status_decision(
    snapshot: RuntimeSnapshot | None,
    health: TwinrSystemHealth | None,
) -> DisplaySystemStatusDecision:
    """Return the exact system-label decision used by the operator display."""

    snapshot_status = _snapshot_status(snapshot)
    snapshot_error = _snapshot_error(snapshot)
    runtime_error = _compact_text(getattr(health, "runtime_error", None), max_chars=_MAX_DETAIL_CHARS)
    health_status = (_compact_text(getattr(health, "status", None), max_chars=_MAX_LABEL_CHARS) or "").lower() or None
    conversation_running, conversation_count, conversation_detail = _service_state(health, "conversation_loop")
    memory_available_mb = _normalized_optional_int(getattr(health, "memory_available_mb", None))
    memory_used_percent = _normalized_optional_float(getattr(health, "memory_used_percent", None))
    memory_owner_label = _compact_text(getattr(health, "memory_owner_label", None), max_chars=_MAX_LABEL_CHARS)
    memory_owner_detail = _compact_text(getattr(health, "memory_owner_detail", None), max_chars=_MAX_DETAIL_CHARS)
    memory_pressure_status = (_compact_text(getattr(health, "memory_pressure_status", None), max_chars=_MAX_LABEL_CHARS) or "").lower() or None
    if memory_pressure_status is None:
        memory_pressure_status = assess_memory_pressure_status(
            memory_available_mb=memory_available_mb,
            memory_used_percent=memory_used_percent,
            swap_total_mb=_normalized_optional_int(getattr(health, "swap_total_mb", None)),
            swap_used_percent=_normalized_optional_float(getattr(health, "swap_used_percent", None)),
        )
    disk_used_percent = _normalized_optional_float(getattr(health, "disk_used_percent", None))
    cpu_temperature_c = _normalized_optional_float(getattr(health, "cpu_temperature_c", None))
    temperature_warn = cpu_temperature_c is not None and cpu_temperature_c >= 72.0

    error_reasons: list[str] = []
    error_detail: str | None = None
    if snapshot is None:
        error_reasons.append("snapshot_missing")
        error_detail = error_detail or "Runtime snapshot is unavailable."
    if snapshot_status == "error":
        error_reasons.append("snapshot_status_error")
        error_detail = error_detail or snapshot_error or "Runtime snapshot reported error status."
    if snapshot_error:
        error_reasons.append("snapshot_error_present")
        error_detail = error_detail or snapshot_error
    if runtime_error:
        error_reasons.append("runtime_error_present")
        error_detail = error_detail or runtime_error
    if health_status == "fail":
        error_reasons.append("health_status_fail")
        error_detail = error_detail or "System health reported fail."
    if error_reasons:
        return DisplaySystemStatusDecision(
            status="error",
            label=_system_label_for_status("error"),
            reason_codes=_normalize_reason_codes(error_reasons),
            reason_detail=_compact_text(error_detail, max_chars=_MAX_DETAIL_CHARS),
            snapshot_status=snapshot_status,
            snapshot_error=snapshot_error,
            runtime_error=runtime_error,
            health_status=health_status,
            conversation_loop_running=conversation_running,
            conversation_loop_count=conversation_count,
            conversation_loop_detail=conversation_detail,
            memory_pressure_status=memory_pressure_status,
            memory_owner_label=memory_owner_label,
            memory_owner_detail=memory_owner_detail,
            memory_available_mb=memory_available_mb,
            memory_used_percent=memory_used_percent,
            disk_used_percent=disk_used_percent,
            cpu_temperature_c=cpu_temperature_c,
        )

    warn_reasons: list[str] = []
    warn_detail: str | None = None
    if health is None:
        return DisplaySystemStatusDecision(
            status="unknown",
            label=_system_label_for_status("unknown"),
            reason_codes=("health_unavailable",),
            reason_detail="System health is unavailable.",
            snapshot_status=snapshot_status,
        )
    if conversation_running is False:
        warn_reasons.append("conversation_loop_unhealthy")
        warn_detail = warn_detail or conversation_detail or "Conversation loop is not running."
    if memory_pressure_status in {"warn", "fail"}:
        warn_reasons.append(f"memory_pressure_{memory_pressure_status}")
        warn_detail = warn_detail or memory_owner_detail or (
            "Memory pressure is degraded."
            if memory_available_mb is None and memory_used_percent is None
            else (
                f"memory_available_mb={memory_available_mb} "
                f"memory_used_percent={memory_used_percent}"
            )
        )
    if disk_used_percent is not None and disk_used_percent >= 85.0:
        warn_reasons.append("disk_used_percent_warn")
        warn_detail = warn_detail or f"disk_used_percent={disk_used_percent}"
    if health_status == "warn" and not temperature_warn:
        warn_reasons.append("health_status_warn")
        warn_detail = warn_detail or "System health reported warn."
    if warn_reasons:
        return DisplaySystemStatusDecision(
            status="warn",
            label=_system_label_for_status("warn"),
            reason_codes=_normalize_reason_codes(warn_reasons),
            reason_detail=_compact_text(warn_detail, max_chars=_MAX_DETAIL_CHARS),
            snapshot_status=snapshot_status,
            snapshot_error=snapshot_error,
            runtime_error=runtime_error,
            health_status=health_status,
            conversation_loop_running=conversation_running,
            conversation_loop_count=conversation_count,
            conversation_loop_detail=conversation_detail,
            memory_pressure_status=memory_pressure_status,
            memory_owner_label=memory_owner_label,
            memory_owner_detail=memory_owner_detail,
            memory_available_mb=memory_available_mb,
            memory_used_percent=memory_used_percent,
            disk_used_percent=disk_used_percent,
            cpu_temperature_c=cpu_temperature_c,
        )

    ok_reasons = ("temperature_warn_ignored",) if health_status == "warn" and temperature_warn else ()
    ok_detail = "Temperature-only warn is intentionally ignored for the primary status card." if ok_reasons else None
    return DisplaySystemStatusDecision(
        status="ok",
        label=_system_label_for_status("ok"),
        reason_codes=ok_reasons,
        reason_detail=ok_detail,
        snapshot_status=snapshot_status,
        snapshot_error=snapshot_error,
        runtime_error=runtime_error,
        health_status=health_status,
        conversation_loop_running=conversation_running,
        conversation_loop_count=conversation_count,
        conversation_loop_detail=conversation_detail,
        memory_pressure_status=memory_pressure_status,
        memory_owner_label=memory_owner_label,
        memory_owner_detail=memory_owner_detail,
        memory_available_mb=memory_available_mb,
        memory_used_percent=memory_used_percent,
        disk_used_percent=disk_used_percent,
        cpu_temperature_c=cpu_temperature_c,
    )


def build_display_status_forensics(
    *,
    snapshot: RuntimeSnapshot | None,
    health: TwinrSystemHealth | None,
    runtime_status: str,
    headline: str,
    system_label: str,
    ai_label: str,
    internet_label: str,
    snapshot_stale: bool,
    system_decision: DisplaySystemStatusDecision | None = None,
) -> DisplayStatusForensics:
    """Build one persisted display decision record from the rendered labels."""

    system_decision = system_decision or build_system_status_decision(snapshot, health)
    normalized_label = _normalized_label(system_label, fallback=system_decision.label) or system_decision.label
    return DisplayStatusForensics(
        operator_status=_operator_status_from_label(normalized_label, fallback=system_decision.status),
        operator_label=normalized_label,
        runtime_status=_normalized_label(runtime_status, fallback="error").lower() or "error",
        headline=_normalized_label(headline),
        ai_label=_normalized_label(ai_label),
        internet_label=_normalized_label(internet_label),
        snapshot_status=system_decision.snapshot_status,
        snapshot_stale=bool(snapshot_stale),
        snapshot_error=system_decision.snapshot_error,
        runtime_error=system_decision.runtime_error,
        health_status=system_decision.health_status,
        conversation_loop_running=system_decision.conversation_loop_running,
        conversation_loop_count=system_decision.conversation_loop_count,
        conversation_loop_detail=system_decision.conversation_loop_detail,
        memory_pressure_status=system_decision.memory_pressure_status,
        memory_owner_label=system_decision.memory_owner_label,
        memory_owner_detail=system_decision.memory_owner_detail,
        memory_available_mb=system_decision.memory_available_mb,
        memory_used_percent=system_decision.memory_used_percent,
        disk_used_percent=system_decision.disk_used_percent,
        cpu_temperature_c=system_decision.cpu_temperature_c,
        reason_codes=system_decision.reason_codes,
        reason_detail=system_decision.reason_detail,
    )
