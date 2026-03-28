# CHANGELOG: 2026-03-27
# BUG-1: Fail-open bug fixed: if runtime.remote_dependency_required() raised, the helper previously disabled required-remote gating by returning False.
# BUG-2: Startup false-ready bug fixed: required_remote_dependency_current_ready() previously reported ready before any successful proof of remote readiness.
# BUG-3: Missing-checker bug fixed: direct mode previously treated a missing runtime checker as healthy, silently bypassing required-remote enforcement.
# BUG-4: Poll scheduling bug fixed: next_check_at was previously computed from check start time, causing near-immediate rechecks after long blocking checks/timeouts.
# BUG-5: Error-ownership bug fixed: recovery logic previously relied on exact snapshot error-text equality and could misclassify required-remote errors as foreign on snapshot read/format drift.
# SEC-1: Output/log injection fixed: untrusted exception text and correlation IDs are now control-character-neutralized before emission to line-oriented control-plane channels.
# SEC-2: Telemetry minimization added: raw remote exception details and remote_write_context are now redacted/canonicalized before tracing/emission to avoid leaking tokens/PII.
# IMP-1: Frontier readiness model: required-remote now follows startup/readiness semantics with fail-closed startup, recovery hysteresis, and conservative ownership handling.
# IMP-2: Frontier resilience model: required-remote polling now uses bounded backoff plus stable jitter to avoid synchronized retry storms during fleet-wide outages.
# IMP-3: Frontier concurrency model: slow remote checks no longer run while the module's state lock is held; a single in-flight refresh is coalesced for other callers.

"""Required-remote gating and recovery helpers for realtime workflow loops."""

from __future__ import annotations

import hashlib
import re
import socket
import time
from collections.abc import Mapping
from typing import Any, Callable, Literal

from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError

OwnershipState = Literal["owned", "foreign", "unknown"]

_MIN_INTERVAL_S = 0.1
_DEFAULT_BACKOFF_FACTOR = 2.0
_DEFAULT_BACKOFF_MAX_MULTIPLIER = 8.0
_DEFAULT_JITTER_RATIO = 0.10
_MAX_EMIT_TEXT_CHARS = 512
_MAX_TRACE_TEXT_CHARS = 1024
_REQUIRED_REMOTE_PUBLIC_ERROR_MESSAGE = "Required remote long-term memory is unavailable."

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]+")
_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(?:token|secret|password|authorization|cookie|credential|session|api[_-]?key|private[_-]?key)"
)
_SECRET_VALUE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)\b(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|cookie|session(?:[_-]?id)?)(\s*[:=]\s*)([^\s,;]+)"
        ),
        r"\1\2[REDACTED]",
    ),
    (re.compile(r"(?i)\b(Bearer)\s+[A-Za-z0-9._~+/=-]+"), r"\1 [REDACTED]"),
    (re.compile(r"://[^:/\s@]+:[^@/\s]+@"), "://[REDACTED]:[REDACTED]@"),
)


def _ensure_required_remote_state(loop: Any) -> None:
    defaults = {
        "_required_remote_dependency_error_active": False,
        "_required_remote_dependency_error_message": None,
        "_required_remote_dependency_last_error_detail": None,
        "_required_remote_dependency_cached_ready": None,
        "_required_remote_dependency_checked_once": False,
        "_required_remote_dependency_recovery_started_at": None,
        "_required_remote_dependency_next_check_at": 0.0,
        "_required_remote_dependency_failure_streak": 0,
        "_required_remote_dependency_last_required": None,
        "_required_remote_dependency_refresh_inflight": False,
        "_required_remote_dependency_last_success_at": None,
        "_required_remote_dependency_last_failure_at": None,
    }
    for name, value in defaults.items():
        if not hasattr(loop, name):
            setattr(loop, name, value)


def _config_float(
    loop: Any,
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw_value = getattr(getattr(loop, "config", None), name, default)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = float(default)
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _config_bool(loop: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(getattr(loop, "config", None), name, default))


def _normalize_text(text: object) -> str:
    return " ".join(str(text or "").split()).strip()


def _sanitize_telemetry_text(value: object, *, max_len: int) -> str:
    text = str(value or "")
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    text = _CONTROL_CHARS_RE.sub(" ", text)
    for pattern, replacement in _SECRET_VALUE_PATTERNS:
        text = pattern.sub(replacement, text)
    text = " ".join(text.split()).strip()
    if len(text) > max_len:
        text = f"{text[: max_len - 1]}…"
    return text


def _sanitize_telemetry_value(value: object, *, max_len: int) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return _sanitize_telemetry_mapping(value, max_len=max_len)
    if isinstance(value, (list, tuple, set, frozenset)):
        sanitized_items = [_sanitize_telemetry_value(item, max_len=max_len) for item in value]
        if isinstance(value, tuple):
            return tuple(sanitized_items)
        if isinstance(value, frozenset):
            return frozenset(sanitized_items)
        return sanitized_items
    return _sanitize_telemetry_text(value, max_len=max_len)


def _sanitize_telemetry_mapping(mapping: Mapping[str, object] | None, *, max_len: int) -> dict[str, object] | None:
    if not mapping:
        return None
    sanitized: dict[str, object] = {}
    for key, value in mapping.items():
        key_text = _sanitize_telemetry_text(key, max_len=128)
        if _SENSITIVE_KEY_RE.search(key_text):
            sanitized[key_text] = "[REDACTED]"
            continue
        sanitized[key_text] = _sanitize_telemetry_value(value, max_len=max_len)
    return sanitized


def _trace_required_remote_event(
    loop: Any,
    event_name: str,
    *,
    kind: str,
    details: Mapping[str, object] | None = None,
    level: str | None = None,
    kpi: Mapping[str, object] | None = None,
) -> None:
    sanitized_details = _sanitize_telemetry_mapping(details, max_len=_MAX_TRACE_TEXT_CHARS)
    sanitized_kpi = _sanitize_telemetry_mapping(kpi, max_len=128)
    loop._trace_event(
        event_name,
        kind=kind,
        details=sanitized_details,
        level=level,
        kpi=sanitized_kpi,
    )


def _try_emit_kv(loop: Any, key: str, value: object) -> None:
    emit = getattr(loop, "_try_emit", None)
    if not callable(emit):
        return
    safe_key = _sanitize_telemetry_text(key, max_len=128)
    safe_value = _sanitize_telemetry_text(value, max_len=_MAX_EMIT_TEXT_CHARS)
    emit(f"{safe_key}={safe_value}")


def _required_remote_public_message(loop: Any, detail_text: str) -> str:
    configured = str(
        getattr(
            getattr(loop, "config", None),
            "long_term_memory_remote_required_public_error_message",
            _REQUIRED_REMOTE_PUBLIC_ERROR_MESSAGE,
        )
        or _REQUIRED_REMOTE_PUBLIC_ERROR_MESSAGE
    ).strip()
    if not configured:
        configured = _REQUIRED_REMOTE_PUBLIC_ERROR_MESSAGE
    # BREAKING: public runtime/control-plane error text is now canonicalized and redacted by default.
    include_detail = _config_bool(loop, "long_term_memory_remote_required_include_error_detail_in_status", False)
    if include_detail and detail_text:
        suffix = _sanitize_telemetry_text(detail_text, max_len=160)
        if suffix:
            return f"{configured} ({suffix})"
    return configured


def _watchdog_assessment_details(assessment: Any) -> dict[str, object]:
    return {
        "artifact_path": getattr(assessment, "artifact_path", None),
        "detail": getattr(assessment, "detail", None),
        "sample_age_s": getattr(assessment, "sample_age_s", None),
        "max_sample_age_s": getattr(assessment, "max_sample_age_s", None),
        "pid_alive": getattr(assessment, "pid_alive", None),
        "sample_status": getattr(assessment, "sample_status", None),
        "sample_ready": getattr(assessment, "sample_ready", None),
    }


def _static_remote_dependency_required(loop: Any) -> bool:
    config = getattr(loop, "config", None)
    return bool(
        getattr(config, "long_term_memory_enabled", False)
        and str(getattr(config, "long_term_memory_mode", "") or "").strip().lower() == "remote_primary"
        and getattr(config, "long_term_memory_remote_required", False)
    )


def _stable_required_remote_jitter_unit(loop: Any) -> float:
    config = getattr(loop, "config", None)
    stable_id = (
        getattr(config, "device_id", None)
        or getattr(config, "assistant_id", None)
        or getattr(config, "instance_id", None)
        or socket.gethostname()
    )
    seed = f"{stable_id}:{type(loop).__module__}.{type(loop).__qualname__}:required_remote"
    digest = hashlib.blake2s(seed.encode("utf-8", "ignore"), digest_size=4).digest()
    return int.from_bytes(digest, "big") / 0xFFFFFFFF


def _apply_required_remote_jitter(loop: Any, delay_s: float) -> float:
    ratio = _config_float(
        loop,
        "long_term_memory_remote_required_jitter_ratio",
        _DEFAULT_JITTER_RATIO,
        minimum=0.0,
        maximum=0.5,
    )
    if ratio <= 0.0:
        return max(_MIN_INTERVAL_S, delay_s)
    unit = _stable_required_remote_jitter_unit(loop)
    multiplier = 1.0 - ratio + (2.0 * ratio * unit)
    return max(_MIN_INTERVAL_S, delay_s * multiplier)


def _required_remote_next_delay_seconds(
    loop: Any,
    *,
    default_interval_s: float,
    failure: bool,
) -> float:
    base_interval_s = required_remote_dependency_interval_seconds(loop, default_interval_s=default_interval_s)
    if not failure:
        return _apply_required_remote_jitter(loop, base_interval_s)
    failure_streak = max(0, int(getattr(loop, "_required_remote_dependency_failure_streak", 0) or 0))
    backoff_factor = _config_float(
        loop,
        "long_term_memory_remote_required_backoff_factor",
        _DEFAULT_BACKOFF_FACTOR,
        minimum=1.0,
        maximum=8.0,
    )
    max_interval_s = _config_float(
        loop,
        "long_term_memory_remote_required_max_backoff_interval_s",
        base_interval_s * _DEFAULT_BACKOFF_MAX_MULTIPLIER,
        minimum=base_interval_s,
    )
    delay_s = min(max_interval_s, base_interval_s * (backoff_factor ** max(0, failure_streak - 1)))
    return _apply_required_remote_jitter(loop, delay_s)


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
        return _config_float(
            loop,
            "long_term_memory_remote_watchdog_interval_s",
            1.0,
            minimum=_MIN_INTERVAL_S,
        )
    return _config_float(
        loop,
        "long_term_memory_remote_keepalive_interval_s",
        max(_MIN_INTERVAL_S, float(default_interval_s)),
        minimum=_MIN_INTERVAL_S,
    )


def required_remote_dependency_recovery_hold_seconds(loop: Any, *, default_interval_s: float) -> float:
    """Return how long watchdog readiness must stay stable before recovery."""

    explicit_hold_s = getattr(
        getattr(loop, "config", None),
        "long_term_memory_remote_required_recovery_hold_s",
        None,
    )
    if explicit_hold_s is not None:
        try:
            return max(0.0, float(explicit_hold_s))
        except (TypeError, ValueError):
            pass
    if not required_remote_dependency_uses_watchdog_artifact(loop):
        return 0.0
    interval_s = required_remote_dependency_interval_seconds(loop, default_interval_s=default_interval_s)
    keepalive_s = _config_float(
        loop,
        "long_term_memory_remote_keepalive_interval_s",
        interval_s,
        minimum=0.0,
    )
    return max(interval_s * 3.0, keepalive_s)


def remote_dependency_is_required(loop: Any) -> bool:
    """Report whether the active runtime requires remote long-term memory."""

    _ensure_required_remote_state(loop)
    config_required = _static_remote_dependency_required(loop)
    runtime = getattr(loop, "runtime", None)
    checker = getattr(runtime, "remote_dependency_required", None)
    if callable(checker):
        try:
            result = bool(checker())
        except Exception as exc:
            fallback = getattr(loop, "_required_remote_dependency_last_required", None)
            if fallback is None:
                fallback = config_required
            _trace_required_remote_event(
                loop,
                "remote_dependency_required_check_failed",
                kind="error",
                details={
                    "error": getattr(loop, "_safe_error_text", str)(exc),
                    "fallback_required": bool(fallback),
                    "fallback_source": "cached"
                    if getattr(loop, "_required_remote_dependency_last_required", None) is not None
                    else "config",
                },
                level="WARNING",
            )
            result = bool(fallback)
    else:
        result = config_required
    loop._required_remote_dependency_last_required = bool(result)
    return bool(result)


def best_effort_stop_player(loop: Any) -> None:
    """Stop active playback during a required-remote fatal error."""

    coordinator = getattr(loop, "playback_coordinator", None)
    stop_from_coordinator = getattr(coordinator, "stop_playback", None)
    if callable(stop_from_coordinator):
        try:
            stop_from_coordinator()
            return
        except Exception as exc:
            _trace_required_remote_event(
                loop,
                "required_remote_stop_from_coordinator_failed",
                kind="error",
                level="ERROR",
                details={"error_type": type(exc).__name__, "error": getattr(loop, "_safe_error_text", str)(exc)},
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
        _trace_required_remote_event(
            loop,
            "required_remote_stop_player_failed",
            kind="error",
            level="ERROR",
            details={"error_type": type(exc).__name__, "error": getattr(loop, "_safe_error_text", str)(exc)},
        )


def enter_required_remote_error(
    loop: Any,
    exc: BaseException | str,
    *,
    extract_remote_write_context: Callable[[BaseException], dict[str, object] | None],
    default_interval_s: float,
    observed_at_monotonic_s: float | None = None,
) -> bool:
    """Fail closed when required remote memory is unavailable."""

    _ensure_required_remote_state(loop)
    if not remote_dependency_is_required(loop):
        return False

    raw_message = getattr(loop, "_safe_error_text", str)(exc) if isinstance(exc, BaseException) else str(exc or "").strip()
    if not raw_message:
        raw_message = _REQUIRED_REMOTE_PUBLIC_ERROR_MESSAGE
    detail_text = _sanitize_telemetry_text(raw_message, max_len=_MAX_TRACE_TEXT_CHARS)
    public_message = _required_remote_public_message(loop, detail_text)

    remote_write_context = extract_remote_write_context(exc) if isinstance(exc, BaseException) else None
    sanitized_context = _sanitize_telemetry_mapping(remote_write_context, max_len=_MAX_TRACE_TEXT_CHARS)
    observed_at = float(observed_at_monotonic_s) if observed_at_monotonic_s is not None else time.monotonic()

    with loop._get_lock("_required_remote_dependency_lock"):
        active = bool(getattr(loop, "_required_remote_dependency_error_active", False))
        runtime_status_value = getattr(getattr(getattr(loop, "runtime", None), "status", None), "value", None)
        loop._required_remote_dependency_failure_streak = max(
            1,
            int(getattr(loop, "_required_remote_dependency_failure_streak", 0) or 0) + 1,
        )
        loop._required_remote_dependency_cached_ready = False
        loop._required_remote_dependency_checked_once = True
        loop._required_remote_dependency_recovery_started_at = None
        loop._required_remote_dependency_last_failure_at = observed_at
        loop._required_remote_dependency_next_check_at = (
            observed_at + _required_remote_next_delay_seconds(loop, default_interval_s=default_interval_s, failure=True)
        )
        loop._required_remote_dependency_error_active = True
        loop._required_remote_dependency_error_message = public_message
        loop._required_remote_dependency_last_error_detail = detail_text

    _trace_required_remote_event(
        loop,
        "required_remote_error_entered",
        kind="invariant",
        level="ERROR",
        details={
            "public_message": public_message,
            "error_detail": detail_text,
            "runtime_status": runtime_status_value or "unknown",
            "remote_write_context": sanitized_context,
            "failure_streak": getattr(loop, "_required_remote_dependency_failure_streak", 0),
        },
    )

    if active and runtime_status_value == "error":
        if isinstance(sanitized_context, dict):
            request_correlation_id = sanitized_context.get("request_correlation_id")
            if isinstance(request_correlation_id, str) and request_correlation_id:
                _try_emit_kv(loop, "required_remote_correlation_id", request_correlation_id)
        _try_emit_kv(loop, "required_remote_dependency", "false")
        return True

    request_interrupt = getattr(loop, "_request_active_turn_interrupt", None)
    if callable(request_interrupt):
        try:
            request_interrupt("required_remote")
        except Exception as interrupt_exc:
            _trace_required_remote_event(
                loop,
                "required_remote_interrupt_request_failed",
                kind="error",
                level="ERROR",
                details={
                    "error_type": type(interrupt_exc).__name__,
                    "error": getattr(loop, "_safe_error_text", str)(interrupt_exc),
                },
            )

    best_effort_stop_player(loop)

    if runtime_status_value == "error":
        _trace_required_remote_event(
            loop,
            "required_remote_error_overrode_runtime_error",
            kind="invariant",
            details={"previous_runtime_status": runtime_status_value or "unknown"},
        )

    loop.runtime.fail(public_message)
    loop._emit_status(force=True)
    _try_emit_kv(loop, "error", public_message)

    if isinstance(sanitized_context, dict):
        request_correlation_id = sanitized_context.get("request_correlation_id")
        if isinstance(request_correlation_id, str) and request_correlation_id:
            _try_emit_kv(loop, "required_remote_correlation_id", request_correlation_id)
    _try_emit_kv(loop, "required_remote_dependency", "false")
    return True


def required_remote_dependency_current_ready(loop: Any) -> bool:
    """Return the cached current readiness for required remote memory."""

    _ensure_required_remote_state(loop)
    if not remote_dependency_is_required(loop):
        return True
    if not bool(getattr(loop, "_required_remote_dependency_checked_once", False)):
        # BREAKING: required remote is now fail-closed until the first successful readiness proof.
        return False
    if bool(getattr(loop, "_required_remote_dependency_error_active", False)):
        return False
    cached_ready = getattr(loop, "_required_remote_dependency_cached_ready", None)
    if cached_ready is not None:
        return bool(cached_ready)
    return getattr(getattr(getattr(loop, "runtime", None), "status", None), "value", None) != "error"


def request_required_remote_dependency_refresh(loop: Any) -> None:
    """Ask the background watcher to refresh required-remote readiness."""

    _ensure_required_remote_state(loop)
    with loop._get_lock("_required_remote_dependency_lock"):
        loop._required_remote_dependency_next_check_at = 0.0
    watcher = getattr(loop, "_required_remote_dependency_watch", None)
    request_refresh = getattr(watcher, "request_refresh", None)
    if callable(request_refresh):
        try:
            request_refresh()
        except Exception as exc:
            _trace_required_remote_event(
                loop,
                "required_remote_refresh_request_failed",
                kind="error",
                level="ERROR",
                details={"error_type": type(exc).__name__, "error": getattr(loop, "_safe_error_text", str)(exc)},
            )


def attest_watchdog_artifact_remote_ready(loop: Any) -> None:
    """Align runtime-local remote adapters with a successful watchdog proof."""

    long_term_memory = getattr(getattr(loop, "runtime", None), "long_term_memory", None)
    attest_external_remote_ready = getattr(long_term_memory, "attest_external_remote_ready", None)
    if not callable(attest_external_remote_ready):
        return
    attest_external_remote_ready()
    _trace_required_remote_event(
        loop,
        "required_remote_watchdog_artifact_runtime_attested",
        kind="invariant",
        details={"long_term_memory_type": type(long_term_memory).__name__},
    )


def _current_runtime_error_ownership(loop: Any) -> OwnershipState:
    expected = _normalize_text(getattr(loop, "_required_remote_dependency_error_message", "") or "")
    if not expected:
        return "unknown"
    snapshot_store = getattr(getattr(loop, "runtime", None), "snapshot_store", None)
    load = getattr(snapshot_store, "load", None)
    if not callable(load):
        return "unknown"
    try:
        snapshot = load()
    except Exception:
        return "unknown"
    if snapshot is None:
        return "unknown"
    current_error = _normalize_text(getattr(snapshot, "error_message", "") or "")
    if not current_error:
        return "unknown"
    if current_error == expected or expected in current_error or current_error in expected:
        return "owned"
    return "foreign"


def current_runtime_error_matches_required_remote(loop: Any) -> bool:
    """Report whether the current runtime error belongs to required-remote gating."""

    return _current_runtime_error_ownership(loop) == "owned"


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

    _ensure_required_remote_state(loop)

    if not remote_dependency_is_required(loop):
        with loop._get_lock("_required_remote_dependency_lock"):
            loop._required_remote_dependency_error_active = False
            loop._required_remote_dependency_error_message = None
            loop._required_remote_dependency_last_error_detail = None
            loop._required_remote_dependency_cached_ready = True
            loop._required_remote_dependency_checked_once = True
            loop._required_remote_dependency_recovery_started_at = None
            loop._required_remote_dependency_next_check_at = 0.0
            loop._required_remote_dependency_failure_streak = 0
            loop._required_remote_dependency_refresh_inflight = False
        _trace_required_remote_event(
            loop,
            "required_remote_not_required",
            kind="invariant",
            details={"force": force, "force_sync": force_sync},
        )
        return True

    with loop._get_lock("_required_remote_dependency_lock"):
        now = time.monotonic()
        next_check_at = float(getattr(loop, "_required_remote_dependency_next_check_at", 0.0) or 0.0)
        cached_ready = getattr(loop, "_required_remote_dependency_cached_ready", None)
        checked_once = bool(getattr(loop, "_required_remote_dependency_checked_once", False))
        if not force and now < next_check_at and cached_ready is not None:
            _trace_required_remote_event(
                loop,
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
        if bool(getattr(loop, "_required_remote_dependency_refresh_inflight", False)):
            fallback_ready = bool(cached_ready) if cached_ready is not None else checked_once and not bool(
                getattr(loop, "_required_remote_dependency_error_active", False)
            )
            _trace_required_remote_event(
                loop,
                "required_remote_refresh_coalesced",
                kind="cache",
                details={
                    "force": force,
                    "force_sync": force_sync,
                    "fallback_ready": fallback_ready,
                    "has_cached_ready": cached_ready is not None,
                },
            )
            return fallback_ready
        loop._required_remote_dependency_refresh_inflight = True

    started = time.monotonic()
    check_error: BaseException | None = None
    watchdog_failure_assessment: Any = None
    success_assessment: Any = None

    try:
        checker = getattr(getattr(loop, "runtime", None), "check_required_remote_dependency", None)
        if required_remote_dependency_uses_watchdog_artifact(loop):
            success_assessment = ensure_watchdog_ready(loop.config)
            _trace_required_remote_event(
                loop,
                "required_remote_watchdog_artifact_ready",
                kind="invariant",
                details=_watchdog_assessment_details(success_assessment),
            )
            attest_watchdog_artifact_remote_ready(loop)
        else:
            if not callable(checker):
                # BREAKING: a missing checker in required direct mode is now fail-closed instead of silently healthy.
                raise RuntimeError("Required remote dependency checker is missing for direct readiness mode.")
            checker(force_sync=force_sync)
    except LongTermRemoteUnavailableError as exc:
        check_error = exc
        if required_remote_dependency_uses_watchdog_artifact(loop):
            try:
                watchdog_failure_assessment = assess_watchdog_snapshot(loop.config)
            except Exception as assessment_exc:
                _trace_required_remote_event(
                    loop,
                    "required_remote_watchdog_artifact_failure_assessment_failed",
                    kind="error",
                    level="ERROR",
                    details={
                        "error_type": type(assessment_exc).__name__,
                        "error": getattr(loop, "_safe_error_text", str)(assessment_exc),
                    },
                )
    except Exception as exc:
        check_error = exc

    completed_at = time.monotonic()
    duration_ms = round((completed_at - started) * 1000.0, 3)

    if check_error is not None:
        if watchdog_failure_assessment is not None:
            _trace_required_remote_event(
                loop,
                "required_remote_watchdog_artifact_failed",
                kind="exception",
                level="ERROR",
                details={
                    "force": force,
                    "force_sync": force_sync,
                    **_watchdog_assessment_details(watchdog_failure_assessment),
                },
            )
        _trace_required_remote_event(
            loop,
            "required_remote_refresh_failed",
            kind="exception",
            level="ERROR",
            details={
                "force": force,
                "force_sync": force_sync,
                "exception": getattr(loop, "_safe_error_text", str)(check_error),
                "remote_write_context": (
                    extract_remote_write_context(check_error) if isinstance(check_error, BaseException) else None
                ),
            },
            kpi={"duration_ms": duration_ms},
        )
        with loop._get_lock("_required_remote_dependency_lock"):
            loop._required_remote_dependency_refresh_inflight = False
        enter_required_remote_error(
            loop,
            check_error,
            extract_remote_write_context=extract_remote_write_context,
            default_interval_s=default_interval_s,
            observed_at_monotonic_s=completed_at,
        )
        return False

    should_reset_runtime_error = False
    pending_restore_details: dict[str, object] | None = None
    unresolved_ownership = False
    restored_runtime_status: str | None = None
    skipped_for_foreign_error = False

    with loop._get_lock("_required_remote_dependency_lock"):
        loop._required_remote_dependency_refresh_inflight = False
        loop._required_remote_dependency_checked_once = True

        runtime_status_value = getattr(getattr(getattr(loop, "runtime", None), "status", None), "value", None)
        error_active = bool(getattr(loop, "_required_remote_dependency_error_active", False))
        ownership: OwnershipState = "foreign"
        if error_active and runtime_status_value == "error":
            ownership = _current_runtime_error_ownership(loop)

        recovery_hold_s = required_remote_dependency_recovery_hold_seconds(
            loop,
            default_interval_s=default_interval_s,
        )
        restore_ready_now = False
        if recovery_hold_s > 0.0 and error_active and runtime_status_value == "error" and ownership == "owned":
            recovery_started_at = getattr(loop, "_required_remote_dependency_recovery_started_at", None)
            if recovery_started_at is None:
                recovery_started_at = completed_at
                loop._required_remote_dependency_recovery_started_at = completed_at
            stable_ready_s = max(0.0, completed_at - float(recovery_started_at))
            if stable_ready_s < recovery_hold_s:
                loop._required_remote_dependency_cached_ready = False
                loop._required_remote_dependency_next_check_at = (
                    completed_at + _required_remote_next_delay_seconds(loop, default_interval_s=default_interval_s, failure=False)
                )
                pending_restore_details = {
                    "force": force,
                    "force_sync": force_sync,
                    "stable_ready_s": round(stable_ready_s, 3),
                    "required_stable_s": round(recovery_hold_s, 3),
                }
            else:
                restore_ready_now = True
        elif error_active and runtime_status_value == "error" and ownership == "unknown":
            loop._required_remote_dependency_cached_ready = False
            loop._required_remote_dependency_next_check_at = (
                completed_at + _required_remote_next_delay_seconds(loop, default_interval_s=default_interval_s, failure=False)
            )
            unresolved_ownership = True
        else:
            restore_ready_now = True

        if restore_ready_now:
            loop._required_remote_dependency_cached_ready = True
            loop._required_remote_dependency_recovery_started_at = None
            loop._required_remote_dependency_next_check_at = (
                completed_at + _required_remote_next_delay_seconds(loop, default_interval_s=default_interval_s, failure=False)
            )
            loop._required_remote_dependency_last_success_at = completed_at
            loop._required_remote_dependency_failure_streak = 0
            if error_active and runtime_status_value == "error" and ownership == "owned":
                should_reset_runtime_error = True
            elif error_active and runtime_status_value == "error" and ownership == "foreign":
                loop._required_remote_dependency_error_active = False
                loop._required_remote_dependency_error_message = None
                loop._required_remote_dependency_last_error_detail = None
                skipped_for_foreign_error = True
            elif runtime_status_value != "error":
                loop._required_remote_dependency_error_active = False
                loop._required_remote_dependency_error_message = None
                loop._required_remote_dependency_last_error_detail = None
            restored_runtime_status = runtime_status_value or "unknown"

    if pending_restore_details is not None:
        _trace_required_remote_event(
            loop,
            "required_remote_restore_pending",
            kind="cache",
            details=pending_restore_details,
        )
        return False

    if unresolved_ownership:
        _trace_required_remote_event(
            loop,
            "required_remote_restore_deferred_ownership_unknown",
            kind="cache",
            details={"force": force, "force_sync": force_sync},
        )
        return False

    _trace_required_remote_event(
        loop,
        "required_remote_refresh_succeeded",
        kind="invariant",
        details={"force": force, "force_sync": force_sync},
        kpi={"duration_ms": duration_ms},
    )

    if skipped_for_foreign_error:
        _trace_required_remote_event(
            loop,
            "required_remote_restore_skipped_for_foreign_error",
            kind="invariant",
            details={"runtime_status": restored_runtime_status or "unknown"},
        )
        return True

    if should_reset_runtime_error:
        try:
            loop.runtime.reset_error()
            loop._emit_status(force=True)
            _try_emit_kv(loop, "required_remote_dependency_restored", "true")
        except Exception as exc:
            _trace_required_remote_event(
                loop,
                "required_remote_runtime_reset_failed",
                kind="error",
                level="ERROR",
                details={"error_type": type(exc).__name__, "error": getattr(loop, "_safe_error_text", str)(exc)},
            )
            with loop._get_lock("_required_remote_dependency_lock"):
                loop._required_remote_dependency_cached_ready = False
                loop._required_remote_dependency_next_check_at = (
                    time.monotonic()
                    + _required_remote_next_delay_seconds(loop, default_interval_s=default_interval_s, failure=False)
                )
            return False
        with loop._get_lock("_required_remote_dependency_lock"):
            loop._required_remote_dependency_error_active = False
            loop._required_remote_dependency_error_message = None
            loop._required_remote_dependency_last_error_detail = None
        _trace_required_remote_event(
            loop,
            "required_remote_restored",
            kind="invariant",
            details={"runtime_status": getattr(getattr(loop.runtime, "status", None), "value", "unknown")},
        )

    return True
