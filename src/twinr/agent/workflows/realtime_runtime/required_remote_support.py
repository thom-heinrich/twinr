"""Required-remote gating and recovery helpers for realtime workflow loops."""

from __future__ import annotations

import time
from typing import Any, Callable

from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError


def required_remote_dependency_uses_watchdog_artifact(loop: Any) -> bool:
    """Report whether live remote gating should use the external watchdog artifact."""

    config = getattr(loop, "config", None)
    mode = str(
        getattr(config, "long_term_memory_remote_runtime_check_mode", "direct") or "direct"
    ).strip().lower()
    return mode == "watchdog_artifact"


def required_remote_dependency_interval_seconds(loop: Any, *, default_interval_s: float) -> float:
    """Return the poll interval for required-remote readiness checks."""

    if required_remote_dependency_uses_watchdog_artifact(loop):
        raw_value = getattr(
            getattr(loop, "config", None),
            "long_term_memory_remote_watchdog_interval_s",
            1.0,
        )
        try:
            return max(0.1, float(raw_value))
        except (TypeError, ValueError):
            return 1.0
    raw_value = getattr(
        getattr(loop, "config", None),
        "long_term_memory_remote_keepalive_interval_s",
        default_interval_s,
    )
    try:
        return max(0.1, float(raw_value))
    except (TypeError, ValueError):
        return default_interval_s


def required_remote_dependency_recovery_hold_seconds(loop: Any, *, default_interval_s: float) -> float:
    """Return how long watchdog readiness must stay stable before recovery."""

    if not required_remote_dependency_uses_watchdog_artifact(loop):
        return 0.0
    interval_s = required_remote_dependency_interval_seconds(loop, default_interval_s=default_interval_s)
    raw_keepalive = getattr(
        getattr(loop, "config", None),
        "long_term_memory_remote_keepalive_interval_s",
        interval_s,
    )
    try:
        keepalive_s = max(0.0, float(raw_keepalive))
    except (TypeError, ValueError):
        keepalive_s = interval_s
    return max(interval_s * 3.0, keepalive_s)


def remote_dependency_is_required(loop: Any) -> bool:
    """Report whether the active runtime requires remote long-term memory."""

    runtime = getattr(loop, "runtime", None)
    checker = getattr(runtime, "remote_dependency_required", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception as exc:
            error_text = loop._safe_error_text(exc)
            loop._trace_event(
                "remote_dependency_required_check_failed",
                kind="error",
                details={"error": error_text},
                level="WARNING",
            )
            return False
    config = getattr(loop, "config", None)
    return bool(
        getattr(config, "long_term_memory_enabled", False)
        and str(getattr(config, "long_term_memory_mode", "") or "").strip().lower() == "remote_primary"
        and getattr(config, "long_term_memory_remote_required", False)
    )


def best_effort_stop_player(loop: Any) -> None:
    """Stop active playback during a required-remote fatal error."""

    coordinator = getattr(loop, "playback_coordinator", None)
    stop_from_coordinator = getattr(coordinator, "stop_playback", None)
    if callable(stop_from_coordinator):
        try:
            stop_from_coordinator()
            return
        except Exception as exc:
            loop._trace_event(
                "required_remote_stop_from_coordinator_failed",
                kind="error",
                level="ERROR",
                details={"error_type": type(exc).__name__, "error": loop._safe_error_text(exc)},
            )
    player = getattr(loop, "player", None)
    stop_fn = getattr(player, "stop_playback", None)
    if not callable(stop_fn):
        stop_fn = getattr(player, "stop", None)
    if not callable(stop_fn):
        return
    try:
        stop_fn()
    except Exception as exc:
        loop._trace_event(
            "required_remote_stop_player_failed",
            kind="error",
            level="ERROR",
            details={"error_type": type(exc).__name__, "error": loop._safe_error_text(exc)},
        )


def enter_required_remote_error(
    loop: Any,
    exc: BaseException | str,
    *,
    extract_remote_write_context: Callable[[BaseException], dict[str, object] | None],
    default_interval_s: float,
) -> bool:
    """Fail closed when required remote memory is unavailable."""

    if not remote_dependency_is_required(loop):
        return False
    message = loop._safe_error_text(exc) if isinstance(exc, BaseException) else str(exc or "").strip()
    if not message:
        message = "Required remote long-term memory is unavailable."
    remote_write_context = extract_remote_write_context(exc) if isinstance(exc, BaseException) else None
    loop._trace_event(
        "required_remote_error_entered",
        kind="invariant",
        level="ERROR",
        details={
            "message": message,
            "runtime_status": getattr(getattr(loop.runtime, "status", None), "value", "unknown"),
            "remote_write_context": remote_write_context,
        },
    )
    active = bool(getattr(loop, "_required_remote_dependency_error_active", False))
    loop._required_remote_dependency_cached_ready = False
    loop._required_remote_dependency_recovery_started_at = None
    loop._required_remote_dependency_next_check_at = (
        time.monotonic() + required_remote_dependency_interval_seconds(loop, default_interval_s=default_interval_s)
    )
    loop._required_remote_dependency_error_active = True
    loop._required_remote_dependency_error_message = message
    if active and getattr(getattr(loop.runtime, "status", None), "value", None) == "error":
        return True
    request_interrupt = getattr(loop, "_request_active_turn_interrupt", None)
    if callable(request_interrupt):
        try:
            request_interrupt("required_remote")
        except Exception as interrupt_exc:
            loop._trace_event(
                "required_remote_interrupt_request_failed",
                kind="error",
                level="ERROR",
                details={"error_type": type(interrupt_exc).__name__, "error": loop._safe_error_text(interrupt_exc)},
            )
    best_effort_stop_player(loop)
    loop.runtime.fail(message)
    loop._emit_status(force=True)
    loop._try_emit(f"error={message}")
    if isinstance(remote_write_context, dict):
        request_correlation_id = remote_write_context.get("request_correlation_id")
        if isinstance(request_correlation_id, str) and request_correlation_id:
            loop._try_emit(f"required_remote_correlation_id={request_correlation_id}")
    loop._try_emit("required_remote_dependency=false")
    return True


def required_remote_dependency_current_ready(loop: Any) -> bool:
    """Return the cached current readiness for required remote memory."""

    if not remote_dependency_is_required(loop):
        return True
    cached_ready = getattr(loop, "_required_remote_dependency_cached_ready", None)
    if cached_ready is not None:
        return bool(cached_ready)
    return getattr(getattr(loop.runtime, "status", None), "value", None) != "error"


def request_required_remote_dependency_refresh(loop: Any) -> None:
    """Ask the background watcher to refresh required-remote readiness."""

    watcher = getattr(loop, "_required_remote_dependency_watch", None)
    request_refresh = getattr(watcher, "request_refresh", None)
    if callable(request_refresh):
        try:
            request_refresh()
        except Exception as exc:
            loop._trace_event(
                "required_remote_refresh_request_failed",
                kind="error",
                level="ERROR",
                details={"error_type": type(exc).__name__, "error": loop._safe_error_text(exc)},
            )


def attest_watchdog_artifact_remote_ready(loop: Any) -> None:
    """Align runtime-local remote adapters with a successful watchdog proof."""

    long_term_memory = getattr(getattr(loop, "runtime", None), "long_term_memory", None)
    attest_external_remote_ready = getattr(long_term_memory, "attest_external_remote_ready", None)
    if not callable(attest_external_remote_ready):
        return
    attest_external_remote_ready()
    loop._trace_event(
        "required_remote_watchdog_artifact_runtime_attested",
        kind="invariant",
        details={"long_term_memory_type": type(long_term_memory).__name__},
    )


def current_runtime_error_matches_required_remote(loop: Any) -> bool:
    """Report whether the current runtime error belongs to required-remote gating."""

    expected = str(getattr(loop, "_required_remote_dependency_error_message", "") or "").strip()
    if not expected:
        return False
    snapshot_store = getattr(getattr(loop, "runtime", None), "snapshot_store", None)
    load = getattr(snapshot_store, "load", None)
    if not callable(load):
        return False
    try:
        snapshot = load()
    except Exception as exc:
        loop._trace_event(
            "required_remote_error_match_read_failed",
            kind="error",
            level="ERROR",
            details={"error_type": type(exc).__name__, "error": loop._safe_error_text(exc)},
        )
        return False
    if snapshot is None:
        return False
    current_error = " ".join(str(getattr(snapshot, "error_message", "") or "").split()).strip()
    return current_error == expected


def refresh_required_remote_dependency(
    loop: Any,
    *,
    force: bool,
    force_sync: bool,
    ensure_watchdog_ready: Callable[[Any], Any],
    assess_watchdog_snapshot: Callable[[Any], Any],
    extract_remote_write_context: Callable[[BaseException], dict[str, object] | None],
    default_interval_s: float,
) -> bool:
    """Refresh required-remote readiness and recover once the dependency is stable again."""

    with loop._get_lock("_required_remote_dependency_lock"):
        if not remote_dependency_is_required(loop):
            loop._required_remote_dependency_error_active = False
            loop._required_remote_dependency_error_message = None
            loop._required_remote_dependency_cached_ready = True
            loop._required_remote_dependency_recovery_started_at = None
            loop._required_remote_dependency_next_check_at = 0.0
            loop._trace_event(
                "required_remote_not_required",
                kind="invariant",
                details={"force": force, "force_sync": force_sync},
            )
            return True
        now = time.monotonic()
        next_check_at = float(getattr(loop, "_required_remote_dependency_next_check_at", 0.0) or 0.0)
        cached_ready = getattr(loop, "_required_remote_dependency_cached_ready", None)
        if not force and now < next_check_at and cached_ready is not None:
            loop._trace_event(
                "required_remote_cached_readiness_used",
                kind="cache",
                details={
                    "force": force,
                    "force_sync": force_sync,
                    "cached_ready": bool(cached_ready),
                    "next_check_in_ms": int(max(0.0, next_check_at - now) * 1000),
                },
            )
            return bool(cached_ready)
        checker = getattr(getattr(loop, "runtime", None), "check_required_remote_dependency", None)
        started = time.monotonic()
        try:
            if required_remote_dependency_uses_watchdog_artifact(loop):
                assessment = ensure_watchdog_ready(loop.config)
                loop._trace_event(
                    "required_remote_watchdog_artifact_ready",
                    kind="invariant",
                    details={
                        "artifact_path": assessment.artifact_path,
                        "sample_age_s": assessment.sample_age_s,
                        "max_sample_age_s": assessment.max_sample_age_s,
                        "pid_alive": assessment.pid_alive,
                    },
                )
                attest_watchdog_artifact_remote_ready(loop)
            else:
                if not callable(checker):
                    loop._trace_event(
                        "required_remote_checker_missing",
                        kind="warning",
                        level="WARN",
                        details={"force": force, "force_sync": force_sync},
                    )
                    return True
                checker(force_sync=force_sync)
        except LongTermRemoteUnavailableError as exc:
            if required_remote_dependency_uses_watchdog_artifact(loop):
                assessment = assess_watchdog_snapshot(loop.config)
                loop._trace_event(
                    "required_remote_watchdog_artifact_failed",
                    kind="exception",
                    level="ERROR",
                    details={
                        "force": force,
                        "force_sync": force_sync,
                        "artifact_path": assessment.artifact_path,
                        "detail": assessment.detail,
                        "sample_age_s": assessment.sample_age_s,
                        "max_sample_age_s": assessment.max_sample_age_s,
                        "pid_alive": assessment.pid_alive,
                        "sample_status": assessment.sample_status,
                        "sample_ready": assessment.sample_ready,
                    },
                )
            loop._trace_event(
                "required_remote_refresh_failed",
                kind="exception",
                level="ERROR",
                details={
                    "force": force,
                    "force_sync": force_sync,
                    "exception": loop._safe_error_text(exc),
                    "remote_write_context": extract_remote_write_context(exc),
                },
                kpi={"duration_ms": round((time.monotonic() - started) * 1000.0, 3)},
            )
            enter_required_remote_error(
                loop,
                exc,
                extract_remote_write_context=extract_remote_write_context,
                default_interval_s=default_interval_s,
            )
            return False
        except Exception as exc:
            loop._trace_event(
                "required_remote_refresh_failed",
                kind="exception",
                level="ERROR",
                details={"force": force, "force_sync": force_sync, "exception": loop._safe_error_text(exc)},
                kpi={"duration_ms": round((time.monotonic() - started) * 1000.0, 3)},
            )
            enter_required_remote_error(
                loop,
                exc,
                extract_remote_write_context=extract_remote_write_context,
                default_interval_s=default_interval_s,
            )
            return False

        runtime_status_value = getattr(getattr(loop.runtime, "status", None), "value", None)
        remote_error_owns_runtime = (
            getattr(loop, "_required_remote_dependency_error_active", False)
            and runtime_status_value == "error"
            and current_runtime_error_matches_required_remote(loop)
        )
        recovery_hold_s = required_remote_dependency_recovery_hold_seconds(
            loop,
            default_interval_s=default_interval_s,
        )
        if recovery_hold_s > 0.0 and remote_error_owns_runtime:
            recovery_started_at = getattr(loop, "_required_remote_dependency_recovery_started_at", None)
            if recovery_started_at is None:
                recovery_started_at = now
                loop._required_remote_dependency_recovery_started_at = now
            stable_ready_s = max(0.0, now - float(recovery_started_at))
            if stable_ready_s < recovery_hold_s:
                loop._required_remote_dependency_cached_ready = False
                loop._required_remote_dependency_next_check_at = (
                    now + required_remote_dependency_interval_seconds(loop, default_interval_s=default_interval_s)
                )
                loop._trace_event(
                    "required_remote_restore_pending",
                    kind="cache",
                    details={
                        "force": force,
                        "force_sync": force_sync,
                        "stable_ready_s": round(stable_ready_s, 3),
                        "required_stable_s": round(recovery_hold_s, 3),
                    },
                )
                return False

        loop._required_remote_dependency_cached_ready = True
        loop._required_remote_dependency_recovery_started_at = None
        loop._required_remote_dependency_next_check_at = (
            now + required_remote_dependency_interval_seconds(loop, default_interval_s=default_interval_s)
        )
        if (
            getattr(loop, "_required_remote_dependency_error_active", False)
            and runtime_status_value == "error"
            and not remote_error_owns_runtime
        ):
            loop._required_remote_dependency_error_active = False
            loop._required_remote_dependency_error_message = None
            loop._trace_event(
                "required_remote_restore_skipped_for_foreign_error",
                kind="invariant",
                details={"runtime_status": runtime_status_value},
            )
        loop._trace_event(
            "required_remote_refresh_succeeded",
            kind="invariant",
            details={"force": force, "force_sync": force_sync},
            kpi={"duration_ms": round((time.monotonic() - started) * 1000.0, 3)},
        )
        if remote_error_owns_runtime:
            loop.runtime.reset_error()
            loop._emit_status(force=True)
            loop._try_emit("required_remote_dependency_restored=true")
            loop._trace_event(
                "required_remote_restored",
                kind="invariant",
                details={"runtime_status": getattr(getattr(loop.runtime, "status", None), "value", "unknown")},
            )
        loop._required_remote_dependency_error_active = False
        loop._required_remote_dependency_error_message = None
        return True
