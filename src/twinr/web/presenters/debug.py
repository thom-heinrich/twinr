"""Build the tabbed operator debug page context for Twinr's web portal.

This module shapes runtime, ChonkyDB/watchdog, LLM usage, hardware, and raw
artifact state into one read-only view that operators can inspect without
switching across multiple pages.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from twinr.agent.base_agent import RuntimeSnapshot, TwinrConfig
from twinr.agent.workflows.required_remote_snapshot import RequiredRemoteWatchdogAssessment
from twinr.memory.longterm.evaluation.live_midterm_attest import (
    LiveMidtermAttestResult,
    default_live_midterm_attest_path,
    load_latest_live_midterm_attest,
)
from twinr.memory.longterm.retrieval.operator_search import LongTermOperatorSearchResult
from twinr.ops.checks import ConfigCheck
from twinr.ops.devices import DeviceOverview
from twinr.ops.health import ServiceHealth, TwinrSystemHealth
from twinr.ops.paths import TwinrOpsPaths
from twinr.ops.remote_memory_watchdog import RemoteMemoryWatchdogSnapshot
from twinr.ops.usage import UsageSummary
from twinr.web.presenters.conversation_lab import build_conversation_lab_panel_context
from twinr.web.presenters.memory_search import build_memory_search_panel_context
from twinr.web.presenters.ops import _format_log_rows, _format_usage_rows, _safe_pretty_json


_DEBUG_TABS: tuple[tuple[str, str, str], ...] = (
    ("overview", "Overview", "Start here for the current runtime, ChonkyDB, and artifact summary."),
    ("runtime", "Runtime", "Inspect the live snapshot, health state, and recent runtime errors."),
    ("chonkydb", "ChonkyDB", "Inspect the required-remote watchdog and recent remote-memory probes."),
    ("memory_search", "Memory Search", "Search the real long-term retrieval stack and inspect matching memories."),
    ("conversation_lab", "Conversation Lab", "Run one real text turn and inspect human-readable routing, tool, and memory traces."),
    ("llm", "LLM", "Inspect OpenAI usage summaries and the latest tracked response records."),
    ("events", "Events", "Inspect recent structured local ops events and their payloads."),
    ("hardware", "Hardware", "Inspect live host health plus the device overview used by Twinr ops."),
    ("raw", "Raw", "Inspect redacted env/config and the raw structured payloads behind this page."),
)
_VALID_DEBUG_TAB_KEYS = frozenset(key for key, _label, _description in _DEBUG_TABS)


def coerce_ops_debug_tab(raw_value: object | None) -> str:
    """Normalize the requested debug tab to one supported key."""

    text = str(raw_value or "").strip().lower()
    return text if text in _VALID_DEBUG_TAB_KEYS else "overview"


def build_ops_debug_page_context(
    *,
    active_tab: str,
    env_path: Path,
    config: TwinrConfig,
    ops_paths: TwinrOpsPaths,
    snapshot: RuntimeSnapshot,
    health: TwinrSystemHealth,
    remote_memory_watchdog: RemoteMemoryWatchdogSnapshot | None,
    remote_memory_watchdog_assessment: RequiredRemoteWatchdogAssessment | None,
    remote_memory_watchdog_error: str | None,
    recent_events: tuple[dict[str, object], ...],
    recent_usage: tuple[object, ...],
    summary_all: UsageSummary,
    summary_24h: UsageSummary,
    device_overview: DeviceOverview | None = None,
    redacted_env_values: dict[str, str] | None = None,
    config_checks: tuple[ConfigCheck, ...] = (),
    config_check_summary: dict[str, int] | None = None,
    memory_search_query: str = "",
    memory_search_result: LongTermOperatorSearchResult | None = None,
    memory_search_error: str | None = None,
    conversation_lab_state: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return template-ready context for the web operator debug page."""

    normalized_tab = coerce_ops_debug_tab(active_tab)
    watchdog_status = _watchdog_status(
        assessment=remote_memory_watchdog_assessment,
        snapshot=remote_memory_watchdog,
        error_message=remote_memory_watchdog_error,
    )
    event_rows = _format_log_rows(recent_events)
    usage_rows = _format_usage_rows(recent_usage)
    recent_error_rows = tuple(row for row in event_rows if row["level"] == "error")[-8:]
    memory_attest = _load_memory_attest(ops_paths.project_root)
    memory_attest_status = _memory_attest_status(memory_attest)
    artifact_rows = _artifact_rows(
        env_path=env_path,
        config=config,
        ops_paths=ops_paths,
        remote_memory_watchdog=remote_memory_watchdog,
        memory_attest=memory_attest,
    )
    config_summary = dict(config_check_summary or {"ok": 0, "warn": 0, "fail": 0})
    memory_search = build_memory_search_panel_context(
        query_text=memory_search_query,
        result=memory_search_result,
        error_message=memory_search_error,
    )
    conversation_lab = build_conversation_lab_panel_context(state=conversation_lab_state)

    return {
        "debug_tabs": tuple(
            {
                "key": key,
                "label": label,
                "description": description,
                "href": f"/ops/debug?tab={key}",
                "active": key == normalized_tab,
            }
            for key, label, description in _DEBUG_TABS
        ),
        "active_debug_tab": normalized_tab,
        "active_debug_description": next(
            description for key, _label, description in _DEBUG_TABS if key == normalized_tab
        ),
        "overview_cards": _overview_cards(
            snapshot=snapshot,
            health=health,
            watchdog_status=watchdog_status,
            memory_attest_status=memory_attest_status,
            recent_events=event_rows,
            summary_24h=summary_24h,
            artifact_rows=artifact_rows,
        ),
        "artifact_rows": artifact_rows,
        "runtime_rows": _runtime_rows(snapshot=snapshot, health=health),
        "service_rows": _service_rows(health.services),
        "recent_error_rows": recent_error_rows,
        "watchdog_status": watchdog_status,
        "watchdog_assessment_rows": _watchdog_assessment_rows(
            assessment=remote_memory_watchdog_assessment,
            error_message=remote_memory_watchdog_error,
        ),
        "memory_attest_status": memory_attest_status,
        "memory_attest_summary_rows": _memory_attest_summary_rows(memory_attest),
        "memory_attest_packet_rows": _memory_attest_packet_rows(memory_attest),
        "memory_search": memory_search,
        "conversation_lab": conversation_lab,
        "watchdog_current_rows": _watchdog_current_rows(
            snapshot=remote_memory_watchdog,
            error_message=remote_memory_watchdog_error,
        ),
        "watchdog_history_rows": _watchdog_history_rows(remote_memory_watchdog),
        "usage_summary_rows": _usage_summary_rows(summary_all=summary_all, summary_24h=summary_24h),
        "usage_rows": usage_rows,
        "event_rows": event_rows,
        "device_rows": _device_rows(device_overview),
        "raw_blocks": _raw_blocks(
            snapshot=snapshot,
            health=health,
            remote_memory_watchdog=remote_memory_watchdog,
            remote_memory_watchdog_assessment=remote_memory_watchdog_assessment,
            summary_all=summary_all,
            summary_24h=summary_24h,
            device_overview=device_overview,
            memory_attest=memory_attest,
            redacted_env_values=redacted_env_values,
            config_checks=config_checks,
            config_check_summary=config_summary,
        ),
        "config_summary_rows": _config_summary_rows(config_summary),
    }


def _overview_cards(
    *,
    snapshot: RuntimeSnapshot,
    health: TwinrSystemHealth,
    watchdog_status: dict[str, str],
    memory_attest_status: dict[str, str],
    recent_events: tuple[dict[str, object], ...],
    summary_24h: UsageSummary,
    artifact_rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    return (
        _detail_item(
            "Runtime",
            _title_case(snapshot.status or "unknown"),
            detail=snapshot.updated_at or "No runtime snapshot timestamp is available.",
            status=_status_class(snapshot.status),
        ),
        _detail_item(
            "System",
            _title_case(health.status or "unknown"),
            detail=f"{health.recent_error_count} recent errors",
            status=_status_class(health.status),
        ),
        _detail_item(
            "ChonkyDB",
            watchdog_status["label"],
            detail=watchdog_status["detail"],
            status=watchdog_status["status"],
        ),
        _detail_item(
            "Memory attest",
            memory_attest_status["label"],
            detail=memory_attest_status["detail"],
            status=memory_attest_status["status"],
        ),
        _detail_item(
            "LLM requests 24h",
            str(summary_24h.requests_total),
            detail=summary_24h.latest_model or "No tracked model yet",
            status="muted",
        ),
        _detail_item(
            "Recent events",
            str(len(recent_events)),
            detail=(recent_events[0]["event"] if recent_events else "No events yet"),
            status="muted",
        ),
        _detail_item(
            "Artifacts",
            str(len(artifact_rows)),
            detail="Runtime, ops, usage, watchdog, and support paths",
            status="muted",
        ),
    )


def _runtime_rows(*, snapshot: RuntimeSnapshot, health: TwinrSystemHealth) -> tuple[dict[str, object], ...]:
    memory_state = getattr(snapshot, "memory_state", None)
    open_loops = tuple(getattr(memory_state, "open_loops", ()) or ())
    rows = [
        _detail_item("Runtime status", _title_case(snapshot.status or "unknown"), status=_status_class(snapshot.status)),
        _detail_item("Snapshot updated", snapshot.updated_at or "—", copy=True),
        _detail_item("Runtime error", health.runtime_error or snapshot.error_message or "—", copy=True, wide=True),
        _detail_item("Memory owner", health.memory_owner_label or "—", copy=True),
        _detail_item("Memory owner detail", health.memory_owner_detail or "—", copy=True, wide=True),
        _detail_item("Last transcript", snapshot.last_transcript or "—", copy=True, wide=True),
        _detail_item("Last response", snapshot.last_response or "—", copy=True, wide=True),
        _detail_item("Memory turns", str(getattr(snapshot, "memory_count", 0))),
        _detail_item("Raw tail turns", str(len(getattr(snapshot, "memory_raw_tail", ()) or ()))),
        _detail_item("Ledger items", str(len(getattr(snapshot, "memory_ledger", ()) or ()))),
        _detail_item("Search memories", str(len(getattr(snapshot, "memory_search_results", ()) or ()))),
        _detail_item("Open loops", ", ".join(open_loops) if open_loops else "—", copy=True),
    ]
    return tuple(rows)


def _service_rows(services: tuple[ServiceHealth, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        _detail_item(
            service.label,
            str(service.count if service.running else 0),
            detail=service.detail or "No further service detail.",
            status="ok" if service.running else "warn",
            copy=True,
        )
        for service in services
    )


def _watchdog_status(
    *,
    assessment: RequiredRemoteWatchdogAssessment | None,
    snapshot: RemoteMemoryWatchdogSnapshot | None,
    error_message: str | None,
) -> dict[str, str]:
    if error_message:
        return {"label": "Error", "detail": error_message, "status": "fail"}
    if assessment is not None:
        if assessment.ready:
            return {"label": "Ok", "detail": assessment.detail or "Required remote is ready.", "status": "ok"}
        if assessment.snapshot_stale:
            return {"label": "Stale", "detail": assessment.detail, "status": "warn"}
        return {
            "label": _title_case(assessment.sample_status or "fail"),
            "detail": assessment.detail,
            "status": "fail",
        }
    if snapshot is not None:
        sample_status = _title_case(snapshot.current.status or "unknown")
        return {
            "label": sample_status,
            "detail": snapshot.current.detail or "Watchdog snapshot exists.",
            "status": _status_class(snapshot.current.status),
        }
    return {"label": "Missing", "detail": "No watchdog snapshot exists yet.", "status": "warn"}


def _watchdog_assessment_rows(
    *,
    assessment: RequiredRemoteWatchdogAssessment | None,
    error_message: str | None,
) -> tuple[dict[str, object], ...]:
    if error_message:
        return (_detail_item("Assessment error", error_message, status="fail", copy=True, wide=True),)
    if assessment is None:
        return (_detail_item("Assessment", "No assessment yet", status="warn"),)
    return (
        _detail_item("Ready", "yes" if assessment.ready else "no", status="ok" if assessment.ready else "fail"),
        _detail_item("Detail", assessment.detail or "—", copy=True, wide=True),
        _detail_item("Sample status", _title_case(assessment.sample_status or "unknown"), status=_status_class(assessment.sample_status)),
        _detail_item("Snapshot stale", "yes" if assessment.snapshot_stale else "no", status="warn" if assessment.snapshot_stale else "ok"),
        _detail_item("Sample age", _seconds_label(assessment.sample_age_s)),
        _detail_item("Max sample age", _seconds_label(assessment.max_sample_age_s)),
        _detail_item("Heartbeat age", _seconds_label(assessment.heartbeat_age_s)),
        _detail_item("Probe inflight", "yes" if assessment.probe_inflight else "no", status="warn" if assessment.probe_inflight else "muted"),
        _detail_item("Probe age", _seconds_label(assessment.probe_age_s)),
        _detail_item("PID alive", "yes" if assessment.pid_alive else "no", status="ok" if assessment.pid_alive else "fail"),
    )


def _load_memory_attest(project_root: Path) -> LiveMidtermAttestResult | None:
    try:
        return load_latest_live_midterm_attest(project_root)
    except Exception:
        return None


def _memory_attest_status(result: LiveMidtermAttestResult | None) -> dict[str, str]:
    if result is None:
        return {
            "label": "Missing",
            "detail": "No live memory attestation artifact exists yet.",
            "status": "warn",
        }
    if result.ready:
        detail = result.finished_at or "Latest live memory attestation passed."
        if result.writer_packet_ids:
            detail = f"{detail} · {len(result.writer_packet_ids)} packet(s)"
        return {"label": "Ok", "detail": detail, "status": "ok"}
    return {
        "label": "Fail",
        "detail": result.error_message or "Latest live memory attestation failed.",
        "status": "fail",
    }


def _memory_attest_summary_rows(result: LiveMidtermAttestResult | None) -> tuple[dict[str, object], ...]:
    if result is None:
        return (
            _detail_item(
                "Memory attest",
                "No artifact yet",
                detail="Run the live midterm acceptance script to populate this block.",
                status="warn",
            ),
        )
    rows = [
        _detail_item("Run status", _title_case(result.status or "unknown"), status=_status_class(result.status)),
        _detail_item("Probe id", result.probe_id or "—", copy=True),
        _detail_item("Finished", result.finished_at or "—", copy=True),
        _detail_item("Namespace", result.runtime_namespace or "—", copy=True, wide=True),
        _detail_item("Flush", "yes" if result.flush_ok else "no", status="ok" if result.flush_ok else "fail"),
        _detail_item(
            "Midterm context",
            "yes" if result.midterm_context_present else "no",
            status="ok" if result.midterm_context_present else "fail",
        ),
        _detail_item(
            "Path warning class",
            result.last_path_warning_class or "—",
            detail=result.last_path_warning_message or None,
            status="ok" if result.last_path_warning_class else "warn",
            copy=True,
            wide=bool(result.last_path_warning_message),
        ),
        _detail_item("Follow-up model", result.follow_up_model or "—", copy=True),
        _detail_item("Artifact", result.artifact_path or "—", copy=True, wide=True),
    ]
    if result.follow_up_query:
        rows.append(_detail_item("Follow-up query", result.follow_up_query, copy=True, wide=True))
    if result.follow_up_answer_text:
        rows.append(_detail_item("Follow-up answer", result.follow_up_answer_text, copy=True, wide=True))
    if result.error_message:
        rows.append(_detail_item("Error", result.error_message, status="fail", copy=True, wide=True))
    return tuple(rows)


def _memory_attest_packet_rows(result: LiveMidtermAttestResult | None) -> tuple[dict[str, object], ...]:
    if result is None:
        return ()
    matched_terms = ", ".join(result.matched_answer_terms) if result.matched_answer_terms else "—"
    expected_terms = ", ".join(result.expected_answer_terms) if result.expected_answer_terms else "—"
    return (
        _detail_item(
            "Writer packet ids",
            ", ".join(result.writer_packet_ids) if result.writer_packet_ids else "—",
            status="ok" if result.writer_packet_ids else "warn",
            copy=True,
            wide=True,
        ),
        _detail_item(
            "Remote packet ids",
            ", ".join(result.remote_packet_ids) if result.remote_packet_ids else "—",
            status="ok" if result.remote_packet_ids else "warn",
            copy=True,
            wide=True,
        ),
        _detail_item(
            "Fresh-reader packet ids",
            ", ".join(result.fresh_reader_packet_ids) if result.fresh_reader_packet_ids else "—",
            status="ok" if result.fresh_reader_packet_ids else "warn",
            copy=True,
            wide=True,
        ),
        _detail_item(
            "Answer terms",
            matched_terms,
            detail=f"expected {expected_terms}",
            status="ok" if result.expected_answer_terms and result.matched_answer_terms == result.expected_answer_terms else "warn",
            copy=True,
            wide=True,
        ),
    )


def _watchdog_current_rows(
    *,
    snapshot: RemoteMemoryWatchdogSnapshot | None,
    error_message: str | None,
) -> tuple[dict[str, object], ...]:
    if error_message:
        return (_detail_item("Watchdog error", error_message, status="fail", copy=True, wide=True),)
    if snapshot is None:
        return (_detail_item("Watchdog snapshot", "No snapshot yet", status="warn"),)
    current = snapshot.current
    return (
        _detail_item("Current status", _title_case(current.status or "unknown"), status=_status_class(current.status)),
        _detail_item("Last sample", current.captured_at or "—", copy=True),
        _detail_item("Mode", current.mode or "—"),
        _detail_item("Required", "yes" if current.required else "no"),
        _detail_item("Ready", "yes" if current.ready else "no", status="ok" if current.ready else "fail"),
        _detail_item("Latency", _latency_label(current.latency_ms)),
        _detail_item("Samples", str(snapshot.sample_count)),
        _detail_item("Failures", str(snapshot.failure_count)),
        _detail_item("Last ok", snapshot.last_ok_at or "—", copy=True),
        _detail_item("Last failure", snapshot.last_failure_at or "—", copy=True),
        _detail_item("Heartbeat", snapshot.heartbeat_at or "—", copy=True),
        _detail_item("Current detail", current.detail or "—", copy=True, wide=True),
    )


def _watchdog_history_rows(snapshot: RemoteMemoryWatchdogSnapshot | None) -> tuple[dict[str, object], ...]:
    if snapshot is None:
        return ()
    rows: list[dict[str, object]] = []
    for sample in reversed(snapshot.recent_samples[-8:]):
        rows.append(
            {
                "created_at": sample.captured_at or "—",
                "level": "info" if sample.ready else "error",
                "event": f"seq {sample.seq} · {sample.mode}",
                "message": (
                    f"status={sample.status} · ready={'yes' if sample.ready else 'no'} "
                    f"· latency={_latency_label(sample.latency_ms)} "
                    f"· ok_streak={sample.consecutive_ok} · fail_streak={sample.consecutive_fail}"
                ),
                "data_pretty": _safe_pretty_json(sample.to_dict()),
            }
        )
    return tuple(rows)


def _usage_summary_rows(*, summary_all: UsageSummary, summary_24h: UsageSummary) -> tuple[dict[str, object], ...]:
    return (
        _detail_item("Requests 24h", str(summary_24h.requests_total)),
        _detail_item("Tokens 24h", str(summary_24h.total_tokens)),
        _detail_item("Latest model", summary_24h.latest_model or "—", copy=True),
        _detail_item("Latest kind", summary_24h.latest_request_kind or "—", copy=True),
        _detail_item("All-time requests", str(summary_all.requests_total)),
        _detail_item("All-time tokens", str(summary_all.total_tokens)),
        _detail_item("Cached input", str(summary_all.cached_input_tokens)),
        _detail_item("Reasoning tokens", str(summary_all.reasoning_tokens)),
        _detail_item("Latest record", summary_all.latest_created_at or "—", copy=True),
        _detail_item("Models seen", str(len(summary_all.by_model or {}))),
    )


def _device_rows(device_overview: DeviceOverview | None) -> tuple[dict[str, object], ...]:
    if device_overview is None:
        return ()
    return tuple(
        {
            "label": device.label,
            "status": _status_class(device.status),
            "summary": device.summary,
            "facts": tuple({"label": fact.label, "value": fact.value} for fact in device.facts),
            "notes": tuple(device.notes),
        }
        for device in device_overview.devices
    )


def _raw_blocks(
    *,
    snapshot: RuntimeSnapshot,
    health: TwinrSystemHealth,
    remote_memory_watchdog: RemoteMemoryWatchdogSnapshot | None,
    remote_memory_watchdog_assessment: RequiredRemoteWatchdogAssessment | None,
    summary_all: UsageSummary,
    summary_24h: UsageSummary,
    device_overview: DeviceOverview | None,
    memory_attest: LiveMidtermAttestResult | None,
    redacted_env_values: dict[str, str] | None,
    config_checks: tuple[ConfigCheck, ...],
    config_check_summary: dict[str, int],
) -> tuple[dict[str, object], ...]:
    return (
        {"title": "Redacted env", "body": _safe_pretty_json(redacted_env_values or {})},
        {"title": "Config check summary", "body": _safe_pretty_json(config_check_summary)},
        {"title": "Config checks", "body": _safe_pretty_json([check.to_dict() for check in config_checks])},
        {"title": "Runtime snapshot", "body": _safe_pretty_json(_debug_payload(snapshot))},
        {"title": "System health", "body": _safe_pretty_json(_debug_payload(health))},
        {"title": "ChonkyDB watchdog", "body": _safe_pretty_json(_debug_payload(remote_memory_watchdog))},
        {"title": "ChonkyDB assessment", "body": _safe_pretty_json(_debug_payload(remote_memory_watchdog_assessment))},
        {
            "title": "LLM usage summary",
            "body": _safe_pretty_json(
                {
                    "summary_all": _debug_payload(summary_all),
                    "summary_24h": _debug_payload(summary_24h),
                }
            ),
        },
        {"title": "Hardware overview", "body": _safe_pretty_json(_debug_payload(device_overview))},
        {"title": "Memory attestation", "body": _safe_pretty_json(_debug_payload(memory_attest))},
    )


def _config_summary_rows(config_summary: dict[str, int]) -> tuple[dict[str, object], ...]:
    return (
        _detail_item("Config ok", str(config_summary.get("ok", 0)), status="ok"),
        _detail_item("Config warn", str(config_summary.get("warn", 0)), status="warn"),
        _detail_item("Config fail", str(config_summary.get("fail", 0)), status="fail"),
    )


def _artifact_rows(
    *,
    env_path: Path,
    config: TwinrConfig,
    ops_paths: TwinrOpsPaths,
    remote_memory_watchdog: RemoteMemoryWatchdogSnapshot | None,
    memory_attest: LiveMidtermAttestResult | None,
) -> tuple[dict[str, object], ...]:
    runtime_state_path = Path(config.runtime_state_path).expanduser()
    if not runtime_state_path.is_absolute():
        runtime_state_path = config.project_root / runtime_state_path
    remote_artifact_path = (
        Path(remote_memory_watchdog.artifact_path)
        if remote_memory_watchdog is not None and remote_memory_watchdog.artifact_path
        else ops_paths.ops_store_root / "remote_memory_watchdog.json"
    )
    memory_attest_path = (
        Path(memory_attest.artifact_path).expanduser()
        if memory_attest is not None and memory_attest.artifact_path
        else default_live_midterm_attest_path(ops_paths.project_root)
    )
    rows = (
        ("Env file", env_path),
        ("Project root", ops_paths.project_root),
        ("Runtime snapshot", runtime_state_path),
        ("Ops events", ops_paths.events_path),
        ("LLM usage", ops_paths.usage_path),
        ("Watchdog artifact", remote_artifact_path),
        ("Memory attest", memory_attest_path),
        ("Self-tests", ops_paths.self_tests_root),
        ("Support bundles", ops_paths.bundles_root),
    )
    return tuple(_path_row(label, path) for label, path in rows)


def _path_row(label: str, path: Path) -> dict[str, object]:
    exists = path.exists()
    is_dir = path.is_dir() if exists else False
    size_text = "—"
    if exists and not is_dir:
        try:
            size_text = _bytes_label(path.stat().st_size)
        except OSError:
            size_text = "unknown"
    updated_text = "—"
    if exists:
        try:
            updated_text = _timestamp_label(path.stat().st_mtime)
        except OSError:
            updated_text = "unknown"
    return _detail_item(
        label,
        str(path),
        detail=f"{'directory' if is_dir else 'file'} · {'present' if exists else 'missing'} · {size_text} · updated {updated_text}",
        status="ok" if exists else "warn",
        copy=True,
        wide=True,
    )


def _detail_item(
    label: str,
    value: str,
    *,
    detail: str | None = None,
    status: str = "muted",
    copy: bool = False,
    wide: bool = False,
) -> dict[str, object]:
    return {
        "label": label,
        "value": value,
        "detail": detail,
        "status": status,
        "copy": copy,
        "wide": wide,
    }


def _debug_payload(value: object | None) -> object:
    if value is None:
        return {}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            return {"value": str(value)}
    if is_dataclass(value):
        try:
            return asdict(value)
        except Exception:
            return {"value": str(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_debug_payload(item) for item in value]
    if isinstance(value, list):
        return [_debug_payload(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _debug_payload(item) for key, item in value.items()}
    return value


def _bytes_label(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _timestamp_label(timestamp: float | None) -> str:
    if timestamp is None:
        return "—"
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _latency_label(latency_ms: float | None) -> str:
    if latency_ms is None:
        return "—"
    return f"{float(latency_ms):.1f} ms"


def _seconds_label(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):.1f} s"


def _title_case(value: str) -> str:
    text = str(value or "").replace("_", " ").strip()
    return text.title() if text else "Unknown"


def _status_class(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text in {"ok", "healthy", "ready", "running", "waiting"}:
        return "ok"
    if text in {"warn", "warning", "blocked", "stale", "missing", "disabled"}:
        return "warn"
    if text in {"fail", "failed", "error", "down", "crash"}:
        return "fail"
    return "muted"
