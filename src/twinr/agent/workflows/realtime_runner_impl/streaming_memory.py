"""Describe bounded streaming-memory checkpoints for the realtime workflow.

This module keeps ``session.py`` orchestration-focused by translating the live
run-loop state into stable memory-attribution labels and concise operator
details. The resulting checkpoints are written into
``streaming_memory_segments.json`` so Pi-side memory-pressure warnings can be
traced back to concrete background lanes instead of only showing a stale
startup phase.
"""

from __future__ import annotations

from collections.abc import Callable


def _callable_result(value: object, default: bool = False) -> bool:
    if not callable(value):
        return bool(default)
    try:
        return bool(value())
    except Exception:
        return bool(default)


def _thread_alive(thread: object | None) -> bool:
    return _callable_result(getattr(thread, "is_alive", None), default=False)


def _coerce_runtime_status(loop: object) -> str:
    runtime = getattr(loop, "runtime", None)
    status = getattr(runtime, "status", None)
    value = getattr(status, "value", None)
    text = str(value or "").strip()
    return text or "unknown"


def _required_remote_snapshot(loop: object) -> object | None:
    watch = getattr(loop, "_required_remote_dependency_watch", None)
    snapshot_fn = getattr(watch, "snapshot", None)
    if not callable(snapshot_fn):
        return None
    try:
        return snapshot_fn()
    except Exception:
        return None


def _voice_orchestrator_flags(loop: object) -> dict[str, object]:
    orchestrator = getattr(loop, "voice_orchestrator", None)
    if orchestrator is None:
        return {
            "present": False,
            "entered": False,
            "capture_thread_alive": False,
            "sender_thread_alive": False,
            "connected": False,
            "ready_backend": None,
            "queue_size": None,
            "queue_drop_count": None,
        }
    sender_queue = getattr(orchestrator, "_sender_queue", None)
    qsize_fn: Callable[[], object] | None = getattr(sender_queue, "qsize", None)
    queue_size: int | None = None
    if callable(qsize_fn):
        try:
            queue_size_value = qsize_fn()
            queue_size = int(str(queue_size_value).strip())
        except (TypeError, ValueError):
            queue_size = None
    ready_backend = getattr(orchestrator, "ready_backend", None)
    text_backend = str(ready_backend or "").strip() or None
    return {
        "present": True,
        "entered": bool(getattr(orchestrator, "entered", False)),
        "capture_thread_alive": _thread_alive(getattr(orchestrator, "_thread", None)),
        "sender_thread_alive": _thread_alive(getattr(orchestrator, "_sender_thread", None)),
        "connected": bool(getattr(orchestrator, "_connected", False)),
        "ready_backend": text_backend,
        "queue_size": queue_size,
        "queue_drop_count": getattr(orchestrator, "_queue_drop_count", None),
    }


def _proactive_monitor_flags(loop: object) -> dict[str, object]:
    monitor = getattr(loop, "proactive_monitor", None)
    if monitor is None:
        return {
            "present": False,
            "entered": False,
            "worker_alive": False,
            "resources_open": False,
            "background_lanes_open": False,
        }
    return {
        "present": True,
        "entered": bool(getattr(monitor, "entered", False)),
        "worker_alive": _thread_alive(getattr(monitor, "_thread", None)),
        "resources_open": bool(getattr(monitor, "_resources_open", False)),
        "background_lanes_open": bool(getattr(monitor, "_background_lanes_open", False)),
    }


def describe_required_remote_watch_probe(loop: object) -> tuple[str, str]:
    """Return the startup probe detail for the required-remote watch worker."""

    snapshot = _required_remote_snapshot(loop)
    if snapshot is None:
        return (
            "required_remote.watch",
            "Required-remote watch worker started, but no snapshot was readable yet.",
        )
    return (
        "required_remote.watch",
        " ".join(
            (
                "Required-remote watch worker started.",
                f"running={bool(getattr(snapshot, 'running', False))}",
                f"inflight={bool(getattr(snapshot, 'inflight', False))}",
                f"effective_ready={bool(getattr(snapshot, 'effective_ready', False))}",
                f"stale={bool(getattr(snapshot, 'stale', False))}",
                f"last_error_type={getattr(snapshot, 'last_error_type', None) or 'none'}",
            )
        ),
    )


def describe_voice_orchestrator_probe(loop: object) -> tuple[str, str]:
    """Return one voice-orchestrator checkpoint detail after context entry."""

    flags = _voice_orchestrator_flags(loop)
    return (
        "voice_orchestrator.capture_loop",
        " ".join(
            (
                "Voice orchestrator context entered from the realtime run loop.",
                f"present={bool(flags['present'])}",
                f"entered={bool(flags['entered'])}",
                f"capture_thread_alive={bool(flags['capture_thread_alive'])}",
                f"sender_thread_alive={bool(flags['sender_thread_alive'])}",
                f"connected={bool(flags['connected'])}",
                f"ready_backend={flags['ready_backend'] or 'none'}",
                f"queue_size={flags['queue_size'] if flags['queue_size'] is not None else 'unknown'}",
                f"queue_drop_count={flags['queue_drop_count'] if flags['queue_drop_count'] is not None else 'unknown'}",
            )
        ),
    )


def describe_proactive_monitor_probe(loop: object) -> tuple[str, str]:
    """Return one proactive-monitor checkpoint detail after context entry."""

    flags = _proactive_monitor_flags(loop)
    return (
        "proactive_monitor.background_worker",
        " ".join(
            (
                "Proactive monitor context entered from the realtime run loop.",
                f"present={bool(flags['present'])}",
                f"entered={bool(flags['entered'])}",
                f"worker_alive={bool(flags['worker_alive'])}",
                f"resources_open={bool(flags['resources_open'])}",
                f"background_lanes_open={bool(flags['background_lanes_open'])}",
            )
        ),
    )


def describe_runtime_heartbeat_probe(loop: object) -> tuple[str, str]:
    """Return the periodic realtime-loop heartbeat detail for live attribution."""

    runtime_status = _coerce_runtime_status(loop)
    snapshot = _required_remote_snapshot(loop)
    voice_flags = _voice_orchestrator_flags(loop)
    proactive_flags = _proactive_monitor_flags(loop)
    required_remote_bits = (
        "required_remote_snapshot=missing"
        if snapshot is None
        else " ".join(
            (
                f"required_remote_running={bool(getattr(snapshot, 'running', False))}",
                f"required_remote_inflight={bool(getattr(snapshot, 'inflight', False))}",
                f"required_remote_ready={bool(getattr(snapshot, 'effective_ready', False))}",
                f"required_remote_stale={bool(getattr(snapshot, 'stale', False))}",
                f"required_remote_last_error={getattr(snapshot, 'last_error_type', None) or 'none'}",
            )
        )
    )
    return (
        "realtime_runtime.loop",
        " ".join(
            (
                "Realtime run-loop heartbeat captured the current background-lane state.",
                f"runtime_status={runtime_status}",
                required_remote_bits,
                f"voice_present={bool(voice_flags['present'])}",
                f"voice_entered={bool(voice_flags['entered'])}",
                f"voice_capture_thread_alive={bool(voice_flags['capture_thread_alive'])}",
                f"voice_sender_thread_alive={bool(voice_flags['sender_thread_alive'])}",
                f"voice_connected={bool(voice_flags['connected'])}",
                f"voice_ready_backend={voice_flags['ready_backend'] or 'none'}",
                f"voice_queue_size={voice_flags['queue_size'] if voice_flags['queue_size'] is not None else 'unknown'}",
                f"voice_queue_drop_count={voice_flags['queue_drop_count'] if voice_flags['queue_drop_count'] is not None else 'unknown'}",
                f"proactive_present={bool(proactive_flags['present'])}",
                f"proactive_entered={bool(proactive_flags['entered'])}",
                f"proactive_worker_alive={bool(proactive_flags['worker_alive'])}",
                f"proactive_resources_open={bool(proactive_flags['resources_open'])}",
                f"proactive_background_lanes_open={bool(proactive_flags['background_lanes_open'])}",
            )
        ),
    )


__all__ = [
    "describe_proactive_monitor_probe",
    "describe_required_remote_watch_probe",
    "describe_runtime_heartbeat_probe",
    "describe_voice_orchestrator_probe",
]
