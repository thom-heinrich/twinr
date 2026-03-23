"""Persist histogram and alert artifacts for remote long-term read operations.

This module is intentionally separate from failure classification so the
catalog/state adapters can emit concise operational telemetry without
embedding histogram bookkeeping or alert policy in their main read paths.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from twinr.memory.longterm.storage.remote_read_diagnostics import LongTermRemoteReadContext


_LOG = logging.getLogger(__name__)
_PROJECT_ROOT_OVERRIDE_ENV = "TWINR_REMOTE_READ_DIAGNOSTICS_PROJECT_ROOT"
_HISTOGRAM_SCHEMA = "twinr_longterm_remote_read_histograms_v1"
_ALERT_TIMEOUT_CLASS = "timeout"
_SLOW_ALERT_LATENCY_MS = 2_000.0
_HISTOGRAM_BUCKETS_MS: tuple[tuple[float, str], ...] = (
    (50.0, "lt_50_ms"),
    (100.0, "50_100_ms"),
    (250.0, "100_250_ms"),
    (500.0, "250_500_ms"),
    (1_000.0, "500_1000_ms"),
    (2_000.0, "1000_2000_ms"),
    (5_000.0, "2000_5000_ms"),
)
_SUPPORTED_OPERATIONS = frozenset(
    {
        "retrieve_search",
        "retrieve_batch",
        "topk_search",
        "topk_batch",
        "fast_topic_topk_search",
    }
)
_STATE_LOCK = threading.Lock()


def record_remote_read_observation(
    *,
    remote_state: object | None,
    context: "LongTermRemoteReadContext",
    latency_ms: float,
    outcome: str,
    classification: str,
) -> None:
    """Update persisted histograms and emit explicit ops alerts when needed."""

    operation = str(getattr(context, "operation", "") or "").strip()
    if operation not in _SUPPORTED_OPERATIONS:
        return
    store = _ops_event_store(remote_state)
    project_root = _project_root(remote_state)
    if store is None or project_root is None:
        return

    normalized_outcome = str(outcome or "ok").strip().lower() or "ok"
    normalized_classification = str(classification or "ok").strip().lower() or "ok"
    bounded_latency_ms = max(0.0, float(latency_ms))
    data = {
        "snapshot_kind": _normalize_text(getattr(context, "snapshot_kind", None)),
        "operation": operation,
        "outcome": normalized_outcome,
        "classification": normalized_classification,
        "latency_ms": round(bounded_latency_ms, 3),
        "latency_bucket": _latency_bucket(bounded_latency_ms),
        "allowed_doc_count": _normalize_int(getattr(context, "allowed_doc_count", None)),
        "result_limit": _normalize_int(getattr(context, "result_limit", None)),
        "batch_size": _normalize_int(getattr(context, "batch_size", None)),
        "catalog_entry_count": _normalize_int(getattr(context, "catalog_entry_count", None)),
    }

    histogram_path = project_root / "artifacts" / "stores" / "ops" / "longterm_remote_read_histograms.json"
    try:
        with _STATE_LOCK:
            payload = _load_histogram_payload(histogram_path)
            _update_histogram_payload(payload, data)
            _write_json_atomic(histogram_path, payload)
    except Exception:
        _LOG.warning("Failed to persist remote-read histogram artifact.", exc_info=True)
        return

    if normalized_classification == _ALERT_TIMEOUT_CLASS or bounded_latency_ms >= _SLOW_ALERT_LATENCY_MS:
        alert_level = "error" if normalized_outcome == "failed" else "warning"
        alert_kind = "timeout" if normalized_classification == _ALERT_TIMEOUT_CLASS else "slow_read"
        alert_data = dict(data)
        alert_data["alert_kind"] = alert_kind
        alert_data["histogram_path"] = str(histogram_path)
        message = (
            f"Remote long-term {data['snapshot_kind'] or 'unknown'} {operation} "
            f"hit {alert_kind} at {data['latency_ms']} ms."
        )
        try:
            store.append(
                event="longterm_remote_read_alert",
                level=alert_level,
                message=message,
                data=alert_data,
            )
        except Exception:
            _LOG.warning("Failed to append remote-read alert event.", exc_info=True)


def _project_root(remote_state: object | None) -> Path | None:
    config = getattr(remote_state, "config", None)
    project_root = os.environ.get(_PROJECT_ROOT_OVERRIDE_ENV) or getattr(config, "project_root", None)
    if not project_root:
        return None
    return Path(str(project_root)).expanduser().resolve(strict=False)


def _ops_event_store(remote_state: object | None) -> object | None:
    project_root = _project_root(remote_state)
    if project_root is None:
        return None
    try:
        from twinr.ops.events import TwinrOpsEventStore

        return TwinrOpsEventStore.from_project_root(project_root)
    except Exception:
        _LOG.warning("Failed to initialize remote-read alert store.", exc_info=True)
        return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: object | None) -> str | None:
    text = " ".join(str(value or "").split()).strip()
    return text or None


def _normalize_int(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _latency_bucket(latency_ms: float) -> str:
    for upper_bound, bucket_name in _HISTOGRAM_BUCKETS_MS:
        if latency_ms < upper_bound:
            return bucket_name
    return "gte_5000_ms"


def _load_histogram_payload(path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "schema": _HISTOGRAM_SCHEMA,
            "version": 1,
            "updated_at": _utc_now_iso(),
            "operations": {},
        }
    except Exception:
        return {
            "schema": _HISTOGRAM_SCHEMA,
            "version": 1,
            "updated_at": _utc_now_iso(),
            "operations": {},
        }
    if not isinstance(loaded, Mapping):
        return {
            "schema": _HISTOGRAM_SCHEMA,
            "version": 1,
            "updated_at": _utc_now_iso(),
            "operations": {},
        }
    operations = loaded.get("operations")
    return {
        "schema": _HISTOGRAM_SCHEMA,
        "version": 1,
        "updated_at": _utc_now_iso(),
        "operations": dict(operations) if isinstance(operations, Mapping) else {},
    }


def _update_histogram_payload(payload: dict[str, object], data: Mapping[str, object]) -> None:
    operations = payload.setdefault("operations", {})
    assert isinstance(operations, dict)
    operation = str(data.get("operation") or "unknown")
    snapshot_kind = str(data.get("snapshot_kind") or "unknown")
    key = f"{snapshot_kind}:{operation}"
    entry = operations.get(key)
    if not isinstance(entry, dict):
        entry = {
            "snapshot_kind": snapshot_kind,
            "operation": operation,
            "total_count": 0,
            "outcome_counts": {},
            "classification_counts": {},
            "latency_buckets_ms": {},
            "last_latency_ms": 0.0,
            "last_updated_at": _utc_now_iso(),
        }
        operations[key] = entry
    entry["total_count"] = int(entry.get("total_count", 0)) + 1
    outcome_counts = dict(entry.get("outcome_counts") or {})
    classification_counts = dict(entry.get("classification_counts") or {})
    latency_buckets = dict(entry.get("latency_buckets_ms") or {})
    outcome = str(data.get("outcome") or "ok")
    classification = str(data.get("classification") or "ok")
    latency_bucket = str(data.get("latency_bucket") or _latency_bucket(float(data.get("latency_ms") or 0.0)))
    outcome_counts[outcome] = int(outcome_counts.get(outcome, 0)) + 1
    classification_counts[classification] = int(classification_counts.get(classification, 0)) + 1
    latency_buckets[latency_bucket] = int(latency_buckets.get(latency_bucket, 0)) + 1
    entry["outcome_counts"] = outcome_counts
    entry["classification_counts"] = classification_counts
    entry["latency_buckets_ms"] = latency_buckets
    entry["last_latency_ms"] = float(data.get("latency_ms") or 0.0)
    entry["last_outcome"] = outcome
    entry["last_classification"] = classification
    entry["last_updated_at"] = _utc_now_iso()
    payload["updated_at"] = entry["last_updated_at"]


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


__all__ = ["record_remote_read_observation"]
