"""Run a live remote-memory retention canary against a fresh namespace.

This module provisions one isolated long-term-memory namespace, seeds a small
current object set that should trigger both archive and prune retention
outcomes, runs the real runtime ``run_retention()`` path while forbidding broad
snapshot-style object loaders, and then verifies the resulting state from a
fresh reader rooted at a different runtime directory. The goal is a sharp
deploy-time proof that online retention still uses the selective projection
path and that the resulting remote current/archive heads are readable after a
restart.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermSourceRefV1
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.memory.longterm.storage.remote_read_diagnostics import extract_remote_write_context
from twinr.memory.longterm.evaluation.live_midterm_acceptance import (
    _build_isolated_config,
    _normalize_base_project_root,
    _shutdown_service,
)
from twinr.ops.remote_memory_watchdog_state import RemoteMemoryWatchdogStore


_SCHEMA_VERSION = 1
_ACCEPT_KIND = "live_retention_canary"
_OPS_ARTIFACT_NAME = "retention_live_canary.json"
_REPORT_DIR_NAME = "retention_live_canary"
_MAX_EXCEPTION_TEXT_CHARS = 240


def _coerce_text(value: object | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _coerce_optional_text(value: object | None) -> str | None:
    text = _coerce_text(value)
    return text or None


def _coerce_str_tuple(values: object | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        text = _coerce_text(values)
        return (text,) if text else ()
    if not isinstance(values, (list, tuple)):
        return ()
    normalized: list[str] = []
    for item in values:
        text = _coerce_text(item)
        if text:
            normalized.append(text)
    return tuple(normalized)


def _json_safe(value: object, *, depth: int = 0) -> object:
    """Return a bounded JSON-safe copy for report artifacts."""

    if depth >= 5:
        return _coerce_optional_text(repr(value)) or "<truncated>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, depth=depth + 1)
            for key, item in list(value.items())[:32]
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item, depth=depth + 1) for item in list(value)[:32]]
    return _coerce_optional_text(repr(value)) or f"<{type(value).__name__}>"


def _coerce_object_dict(value: object | None) -> dict[str, object] | None:
    """Return one JSON-safe shallow object mapping when possible."""

    if not isinstance(value, dict):
        return None
    normalized = _json_safe(value)
    return normalized if isinstance(normalized, dict) else None


def _coerce_object_tuple(values: object | None) -> tuple[dict[str, object], ...]:
    """Return one tuple of JSON-safe object mappings."""

    if not isinstance(values, (list, tuple)):
        return ()
    normalized: list[dict[str, object]] = []
    for item in values:
        payload = _coerce_object_dict(item)
        if payload is not None:
            normalized.append(payload)
    return tuple(normalized)


def _clip_text(value: object | None, max_chars: int = _MAX_EXCEPTION_TEXT_CHARS) -> str | None:
    """Return one bounded single-line text fragment for diagnostics."""

    text = _coerce_optional_text(value)
    if text is None:
        return None
    normalized_text = str(text)
    if len(normalized_text) <= max_chars:
        return normalized_text
    return normalized_text[: max(1, max_chars - 1)].rstrip() + "…"


def _format_exception_text(exc: BaseException | None) -> str | None:
    """Return one clipped exception summary."""

    if exc is None:
        return None
    return _clip_text(f"{type(exc).__name__}: {exc}")


def _exception_chain(exc: BaseException | None) -> tuple[BaseException, ...]:
    """Return the causal chain for one exception without looping forever."""

    chain: list[BaseException] = []
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _root_cause_exception(exc: BaseException | None) -> BaseException | None:
    """Return the deepest available cause for one failure."""

    chain = _exception_chain(exc)
    if not chain:
        return None
    return chain[-1]


def _exception_remote_write_context(exc: BaseException | None) -> dict[str, object] | None:
    """Extract the first attached remote-write context from one failure chain."""

    for item in _exception_chain(exc):
        context = extract_remote_write_context(item)
        if isinstance(context, dict) and context:
            return dict(context)
    return None


def _exception_chain_payload(exc: BaseException | None) -> tuple[dict[str, object], ...]:
    """Return one bounded structured exception chain for report artifacts."""

    payloads: list[dict[str, object]] = []
    for item in _exception_chain(exc):
        payload: dict[str, object] = {"type": type(item).__name__}
        detail = _clip_text(item)
        if detail is not None:
            payload["detail"] = detail
        if isinstance(item, ChonkyDBError):
            if item.status_code is not None:
                payload["status_code"] = int(item.status_code)
            response_json = item.response_json if isinstance(item.response_json, Mapping) else None
            if response_json:
                payload["response_json"] = _json_safe(dict(response_json))
        payloads.append(payload)
    return tuple(payloads)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON payload atomically to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_namespace_suffix(value: str) -> str:
    safe_chars: list[str] = []
    for char in str(value or "").lower():
        safe_chars.append(char if char.isalnum() else "_")
    normalized = "".join(safe_chars).strip("_")
    return normalized or "retention_canary"


def _step_payload(
    *,
    name: str,
    started_monotonic: float,
    status: str,
    detail: str | None = None,
    evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one bounded per-stage canary evidence payload."""

    payload: dict[str, object] = {
        "name": _coerce_text(name),
        "status": _coerce_text(status).lower() or "unknown",
        "latency_ms": round(max(0.0, (time.monotonic() - started_monotonic) * 1000.0), 3),
    }
    normalized_detail = _coerce_optional_text(detail)
    if normalized_detail is not None:
        payload["detail"] = normalized_detail
    normalized_evidence = _coerce_object_dict(evidence)
    if normalized_evidence is not None:
        payload["evidence"] = normalized_evidence
    return payload


def _canary_proof_contract() -> dict[str, object]:
    """Return the exact contract the retention canary is intended to prove."""

    return {
        "contract_id": "isolated_namespace_retention_transaction",
        "contract_kind": "synthetic_transaction",
        "namespace_scope": "fresh_isolated_namespace",
        "mutation_scope": "write_retention_readback",
        "operations_proved": [
            "isolated namespace bootstrap on separate runtime roots",
            "fresh-namespace active-delta writes",
            "retention execution against seeded objects",
            "fresh-reader current-state fine-grained readback",
            "fresh-reader archive fine-grained readback",
            "post-retention item lookup for kept and archived memories",
        ],
        "operations_not_proved": [
            "configured production namespace warm-read health",
            "prompt/user/personality current-head readability on the live namespace",
        ],
        "summary": (
            "Proves a fresh-namespace write/retention/fresh-reader transaction, not just the "
            "configured runtime namespace warm-read surface."
        ),
    }


def _load_watchdog_observation(config: TwinrConfig) -> dict[str, object]:
    """Capture the latest watchdog attestation snapshot for canary comparison."""

    store = RemoteMemoryWatchdogStore.from_config(config)
    observed_at = _utc_now_iso()
    try:
        snapshot = store.load()
    except Exception as exc:
        return {
            "observed_at": observed_at,
            "artifact_path": str(store.path),
            "status": "artifact_unreadable",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    if snapshot is None:
        return {
            "observed_at": observed_at,
            "artifact_path": str(store.path),
            "status": "artifact_missing",
            "detail": "No remote-memory watchdog snapshot was available.",
        }

    probe_payload = snapshot.current.probe if isinstance(snapshot.current.probe, dict) else {}
    warm_result = probe_payload.get("warm_result") if isinstance(probe_payload, dict) else None
    warm_result_payload = warm_result if isinstance(warm_result, dict) else {}
    proof_contract = warm_result_payload.get("proof_contract")
    return {
        "observed_at": observed_at,
        "artifact_path": str(store.path),
        "status": "ok",
        "sample_status": snapshot.current.status,
        "sample_ready": snapshot.current.ready,
        "sample_captured_at": snapshot.current.captured_at,
        "sample_latency_ms": snapshot.current.latency_ms,
        "sample_detail": snapshot.current.detail,
        "probe_mode": warm_result_payload.get("probe_mode"),
        "archive_safe": warm_result_payload.get("archive_safe"),
        "health_tier": warm_result_payload.get("health_tier"),
        "proof_contract": proof_contract if isinstance(proof_contract, dict) else None,
    }


def _assess_watchdog_consistency(
    *,
    watchdog_observation: dict[str, object] | None,
    canary_ready: bool,
    failure_stage: str | None,
    error_message: str | None,
) -> dict[str, object]:
    """Classify how a canary result should be compared to the watchdog result."""

    canary_contract = _canary_proof_contract()
    if not isinstance(watchdog_observation, dict):
        return {
            "relation": "watchdog_unavailable",
            "equivalent_proofs": False,
            "summary": "The retention canary ran without a readable watchdog artifact to compare against.",
            "canary_contract": canary_contract,
        }

    observation_status = str(watchdog_observation.get("status") or "unknown")
    watchdog_ready = bool(watchdog_observation.get("sample_ready"))
    watchdog_contract = watchdog_observation.get("proof_contract")
    if not isinstance(watchdog_contract, dict):
        watchdog_contract = None
    if observation_status != "ok":
        return {
            "relation": "watchdog_observation_unavailable",
            "equivalent_proofs": False,
            "summary": "The retention canary could not compare itself to a readable watchdog sample.",
            "watchdog_status": observation_status,
            "watchdog_contract": watchdog_contract,
            "canary_contract": canary_contract,
        }
    if canary_ready and watchdog_ready:
        return {
            "relation": "consistent_success_different_scope",
            "equivalent_proofs": False,
            "summary": (
                "Both probes were green, but they still attest different surfaces: the watchdog covers "
                "configured-namespace warm reads, while the canary covers isolated-namespace retention transactions."
            ),
            "watchdog_contract": watchdog_contract,
            "canary_contract": canary_contract,
        }
    if (not canary_ready) and watchdog_ready:
        return {
            "relation": "watchdog_ready_canary_failed_non_equivalent",
            "equivalent_proofs": False,
            "summary": (
                "The watchdog was green because configured-namespace archive-inclusive warm reads were healthy, "
                "but the retention canary failed later on the stricter isolated-namespace write/retention/readback path."
            ),
            "failure_stage": _coerce_optional_text(failure_stage),
            "error_message": _coerce_optional_text(error_message),
            "watchdog_contract": watchdog_contract,
            "canary_contract": canary_contract,
        }
    if (not canary_ready) and (not watchdog_ready):
        return {
            "relation": "consistent_failure",
            "equivalent_proofs": False,
            "summary": "Both the watchdog and the retention canary were unhealthy, though on different proof surfaces.",
            "failure_stage": _coerce_optional_text(failure_stage),
            "error_message": _coerce_optional_text(error_message),
            "watchdog_contract": watchdog_contract,
            "canary_contract": canary_contract,
        }
    return {
        "relation": "watchdog_failed_canary_passed",
        "equivalent_proofs": False,
        "summary": (
            "The canary passed even though the watchdog sample was unhealthy. This suggests watchdog-specific timing "
            "or scope issues rather than a general remote-memory outage."
        ),
        "watchdog_contract": watchdog_contract,
        "canary_contract": canary_contract,
    }


def _source(event_id: str) -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="retention_canary",
        event_ids=(event_id,),
        speaker="system",
        modality="text",
    )


def _seed_retention_objects(*, now: datetime) -> tuple[LongTermMemoryObjectV1, ...]:
    """Return one deterministic object set that exercises keep/archive/prune."""

    old_time = now - timedelta(days=30)
    recent_time = now - timedelta(hours=1)
    return (
        LongTermMemoryObjectV1(
            memory_id="episode:retention_old_weather",
            kind="episode",
            summary="We talked about the weather a long time ago.",
            source=_source("retention:old_weather"),
            status="active",
            confidence=0.73,
            created_at=old_time,
            updated_at=old_time,
        ),
        LongTermMemoryObjectV1(
            memory_id="observation:retention_old_presence",
            kind="observation",
            summary="A motion observation was recorded a long time ago.",
            source=_source("retention:old_presence"),
            status="active",
            confidence=0.64,
            created_at=old_time,
            updated_at=old_time,
            attributes={"memory_domain": "presence"},
        ),
        LongTermMemoryObjectV1(
            memory_id="event:retention_future_appointment",
            kind="event",
            summary="There is still a newer doctor appointment coming up.",
            source=_source("retention:future_appointment"),
            status="active",
            confidence=0.97,
            confirmed_by_user=True,
            slot_key="event:retention_future_appointment",
            value_key="future_appointment",
            created_at=recent_time,
            updated_at=recent_time,
            attributes={"memory_domain": "planning"},
        ),
        LongTermMemoryObjectV1(
            memory_id="fact:retention_relationship_anchor",
            kind="relationship_fact",
            summary="Janina is the user's wife.",
            source=_source("retention:relationship_anchor"),
            status="active",
            confidence=0.99,
            confirmed_by_user=True,
            slot_key="relationship:user:main:wife",
            value_key="person:janina",
            created_at=recent_time,
            updated_at=recent_time,
            attributes={
                "fact_type": "relationship",
                "person_ref": "person:janina",
                "relation": "wife",
            },
        ),
    )


@dataclass(frozen=True, slots=True)
class LiveRetentionCanaryResult:
    """Describe one complete live retention canary run."""

    probe_id: str
    status: str
    started_at: str
    finished_at: str
    env_path: str
    base_project_root: str
    runtime_namespace: str
    writer_root: str | None = None
    fresh_reader_root: str | None = None
    projection_selected_ids: tuple[str, ...] = ()
    pruned_memory_ids: tuple[str, ...] = ()
    archived_memory_ids: tuple[str, ...] = ()
    fresh_kept_ids: tuple[str, ...] = ()
    fresh_archived_ids: tuple[str, ...] = ()
    error_message: str | None = None
    failure_stage: str | None = None
    root_cause_message: str | None = None
    remote_write_context: dict[str, object] | None = None
    exception_chain: tuple[dict[str, object], ...] = ()
    proof_contract: dict[str, object] | None = None
    watchdog_observations: tuple[dict[str, object], ...] = ()
    consistency_assessment: dict[str, object] | None = None
    steps: tuple[dict[str, object], ...] = ()
    artifact_path: str | None = None
    report_path: str | None = None
    accept_kind: str = field(default=_ACCEPT_KIND, init=False)
    schema_version: int = field(default=_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "probe_id", _coerce_text(self.probe_id))
        object.__setattr__(self, "status", _coerce_text(self.status).lower() or "unknown")
        object.__setattr__(self, "started_at", _coerce_text(self.started_at))
        object.__setattr__(self, "finished_at", _coerce_text(self.finished_at))
        object.__setattr__(self, "env_path", _coerce_text(self.env_path))
        object.__setattr__(self, "base_project_root", _coerce_text(self.base_project_root))
        object.__setattr__(self, "runtime_namespace", _coerce_text(self.runtime_namespace))
        object.__setattr__(self, "writer_root", _coerce_optional_text(self.writer_root))
        object.__setattr__(self, "fresh_reader_root", _coerce_optional_text(self.fresh_reader_root))
        object.__setattr__(self, "projection_selected_ids", _coerce_str_tuple(self.projection_selected_ids))
        object.__setattr__(self, "pruned_memory_ids", _coerce_str_tuple(self.pruned_memory_ids))
        object.__setattr__(self, "archived_memory_ids", _coerce_str_tuple(self.archived_memory_ids))
        object.__setattr__(self, "fresh_kept_ids", _coerce_str_tuple(self.fresh_kept_ids))
        object.__setattr__(self, "fresh_archived_ids", _coerce_str_tuple(self.fresh_archived_ids))
        object.__setattr__(self, "error_message", _coerce_optional_text(self.error_message))
        object.__setattr__(self, "failure_stage", _coerce_optional_text(self.failure_stage))
        object.__setattr__(self, "root_cause_message", _coerce_optional_text(self.root_cause_message))
        object.__setattr__(self, "remote_write_context", _coerce_object_dict(self.remote_write_context))
        object.__setattr__(self, "exception_chain", _coerce_object_tuple(self.exception_chain))
        object.__setattr__(self, "proof_contract", _coerce_object_dict(self.proof_contract))
        object.__setattr__(self, "watchdog_observations", _coerce_object_tuple(self.watchdog_observations))
        object.__setattr__(self, "consistency_assessment", _coerce_object_dict(self.consistency_assessment))
        object.__setattr__(self, "steps", _coerce_object_tuple(self.steps))
        object.__setattr__(self, "artifact_path", _coerce_optional_text(self.artifact_path))
        object.__setattr__(self, "report_path", _coerce_optional_text(self.report_path))

    @property
    def ready(self) -> bool:
        """Return whether the canary proved the selective retention contract."""

        return (
            self.status == "ok"
            and "episode:retention_old_weather" in self.archived_memory_ids
            and "observation:retention_old_presence" in self.pruned_memory_ids
            and "event:retention_future_appointment" in self.fresh_kept_ids
            and "episode:retention_old_weather" in self.fresh_archived_ids
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ready"] = self.ready
        return payload


def default_live_retention_canary_path(project_root: str | Path) -> Path:
    """Return the rolling ops artifact path for the latest retention canary."""

    return Path(project_root).expanduser().resolve() / "artifacts" / "stores" / "ops" / _OPS_ARTIFACT_NAME


def default_live_retention_canary_report_dir(project_root: str | Path) -> Path:
    """Return the report directory used for per-run retention canary snapshots."""

    return Path(project_root).expanduser().resolve() / "artifacts" / "reports" / _REPORT_DIR_NAME


def write_live_retention_canary_artifacts(
    result: LiveRetentionCanaryResult,
    *,
    project_root: str | Path,
) -> LiveRetentionCanaryResult:
    """Persist the rolling ops artifact plus a per-run report snapshot."""

    artifact_path = default_live_retention_canary_path(project_root)
    report_dir = default_live_retention_canary_report_dir(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{result.probe_id}.json"
    persisted = replace(
        result,
        artifact_path=str(artifact_path),
        report_path=str(report_path),
    )
    payload = persisted.to_dict()
    _atomic_write_json(report_path, payload)
    _atomic_write_json(artifact_path, payload)
    return persisted


def run_live_retention_canary(
    *,
    env_path: str | Path = ".env",
    probe_id: str | None = None,
    write_artifacts: bool = True,
) -> LiveRetentionCanaryResult:
    """Run the live retention canary against a fresh remote namespace."""

    resolved_env_path = Path(env_path).expanduser().resolve(strict=False)
    started_at = _utc_now_iso()
    effective_probe_id = " ".join(
        str(probe_id or f"retention_canary_{started_at.replace(':', '').replace('-', '')}").split()
    ).strip()
    base_config = TwinrConfig.from_env(resolved_env_path)
    base_project_root = _normalize_base_project_root(resolved_env_path, base_config)
    runtime_namespace = f"twinr_retention_canary_{_safe_namespace_suffix(effective_probe_id)}"

    writer_service: LongTermMemoryService | None = None
    fresh_reader_service: LongTermMemoryService | None = None
    result = LiveRetentionCanaryResult(
        probe_id=effective_probe_id,
        status="running",
        started_at=started_at,
        finished_at=started_at,
        env_path=str(resolved_env_path),
        base_project_root=str(base_project_root),
        runtime_namespace=runtime_namespace,
        proof_contract=_canary_proof_contract(),
    )
    steps: list[dict[str, object]] = []
    watchdog_observations: list[dict[str, object]] = [_load_watchdog_observation(base_config)]
    failure_stage: str | None = None

    try:
        if not base_config.chonkydb_base_url or not base_config.chonkydb_api_key:
            raise RuntimeError("ChonkyDB credentials are required for the live retention canary.")

        with ExitStack() as stack:
            writer_temp_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix=f"{effective_probe_id}_writer_"))
            fresh_reader_temp_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix=f"{effective_probe_id}_reader_"))
            writer_root = Path(writer_temp_dir).resolve(strict=False)
            fresh_reader_root = Path(fresh_reader_temp_dir).resolve(strict=False)
            result = replace(
                result,
                writer_root=str(writer_root),
                fresh_reader_root=str(fresh_reader_root),
            )

            writer_config = _build_isolated_config(
                base_config=base_config,
                base_project_root=base_project_root,
                runtime_root=writer_root,
                remote_namespace=runtime_namespace,
                background_store_turns=False,
            )
            fresh_reader_config = _build_isolated_config(
                base_config=base_config,
                base_project_root=base_project_root,
                runtime_root=fresh_reader_root,
                remote_namespace=runtime_namespace,
                background_store_turns=False,
            )

            writer_service = LongTermMemoryService.from_config(writer_config)
            stage_name = "writer_ensure_remote_ready"
            stage_started = time.monotonic()
            writer_service.ensure_remote_ready()
            steps.append(_step_payload(name=stage_name, started_monotonic=stage_started, status="ok"))

            now = datetime.now(timezone.utc)
            stage_name = "seed_retention_objects"
            stage_started = time.monotonic()
            writer_service.object_store.commit_active_delta(
                object_upserts=_seed_retention_objects(now=now),
            )
            steps.append(
                _step_payload(
                    name=stage_name,
                    started_monotonic=stage_started,
                    status="ok",
                    evidence={"seeded_object_count": 4},
                )
            )

            original_projection_loader = writer_service.object_store.load_objects_by_projection_filter
            projection_selected_ids: list[str] = []
            store_type = type(writer_service.object_store)

            def _recording_projection_loader(_self, *, predicate):
                objects = tuple(original_projection_loader(predicate=predicate))
                projection_selected_ids.extend(item.memory_id for item in objects)
                return objects

            with patch.object(
                store_type,
                "load_objects",
                side_effect=AssertionError("Retention canary must not hydrate the full object snapshot."),
            ), patch.object(
                store_type,
                "load_objects_fine_grained",
                side_effect=AssertionError("Retention canary must not run a broad object sweep."),
            ), patch.object(
                store_type,
                "load_current_state_fine_grained",
                side_effect=AssertionError("Retention canary must not materialize the full current state."),
            ), patch.object(
                store_type,
                "load_archived_objects",
                side_effect=AssertionError("Retention canary must not hydrate the full archive snapshot."),
            ), patch.object(
                store_type,
                "write_snapshot",
                side_effect=AssertionError("Retention canary must not rewrite the full current state."),
            ), patch.object(
                store_type,
                "load_objects_by_projection_filter",
                new=_recording_projection_loader,
            ):
                stage_name = "run_retention"
                stage_started = time.monotonic()
                retention = writer_service.run_retention()
                retained_archived_objects = tuple(retention.archived_objects)
                retained_pruned_ids = tuple(retention.pruned_memory_ids)
                steps.append(
                    _step_payload(
                        name=stage_name,
                        started_monotonic=stage_started,
                        status="ok",
                        evidence={
                            "projection_selected_ids": tuple(dict.fromkeys(projection_selected_ids)),
                            "archived_count": len(retained_archived_objects),
                            "pruned_count": len(retained_pruned_ids),
                        },
                    )
                )

            fresh_reader_service = LongTermMemoryService.from_config(fresh_reader_config)
            stage_name = "fresh_reader_ensure_remote_ready"
            stage_started = time.monotonic()
            fresh_reader_service.ensure_remote_ready()
            steps.append(_step_payload(name=stage_name, started_monotonic=stage_started, status="ok"))

            stage_name = "fresh_reader_load_current_state_fine_grained"
            stage_started = time.monotonic()
            fresh_state = fresh_reader_service.object_store.load_current_state_fine_grained()
            fresh_objects = tuple(fresh_state.objects)
            steps.append(
                _step_payload(
                    name=stage_name,
                    started_monotonic=stage_started,
                    status="ok",
                    evidence={"object_count": len(fresh_objects)},
                )
            )

            stage_name = "fresh_reader_load_archived_objects_fine_grained"
            stage_started = time.monotonic()
            fresh_archive = fresh_reader_service.object_store.load_archived_objects_fine_grained()
            fresh_archive_objects = tuple(fresh_archive)
            steps.append(
                _step_payload(
                    name=stage_name,
                    started_monotonic=stage_started,
                    status="ok",
                    evidence={"archive_count": len(fresh_archive_objects)},
                )
            )
            fresh_kept_ids = tuple(item.memory_id for item in fresh_objects)
            fresh_archived_ids = tuple(item.memory_id for item in fresh_archive_objects)

            stage_name = "validate_expected_retention_outcome"
            stage_started = time.monotonic()
            if "episode:retention_old_weather" not in {item.memory_id for item in retained_archived_objects}:
                raise RuntimeError("Retention canary did not archive the aged episode.")
            if "observation:retention_old_presence" not in set(retained_pruned_ids):
                raise RuntimeError("Retention canary did not prune the aged observation.")
            if "event:retention_future_appointment" not in fresh_kept_ids:
                raise RuntimeError("Fresh reader no longer sees the expected newer event.")
            if "episode:retention_old_weather" not in fresh_archived_ids:
                raise RuntimeError("Fresh reader did not see the archived aged episode.")
            steps.append(
                _step_payload(
                    name=stage_name,
                    started_monotonic=stage_started,
                    status="ok",
                    evidence={
                        "fresh_kept_ids": fresh_kept_ids,
                        "fresh_archived_ids": fresh_archived_ids,
                    },
                )
            )

            watchdog_observations.append(_load_watchdog_observation(base_config))
            result = replace(
                result,
                status="ok",
                finished_at=_utc_now_iso(),
                projection_selected_ids=tuple(dict.fromkeys(projection_selected_ids)),
                pruned_memory_ids=retained_pruned_ids,
                archived_memory_ids=tuple(item.memory_id for item in retained_archived_objects),
                fresh_kept_ids=fresh_kept_ids,
                fresh_archived_ids=fresh_archived_ids,
                watchdog_observations=tuple(watchdog_observations),
                consistency_assessment=_assess_watchdog_consistency(
                    watchdog_observation=watchdog_observations[-1] if watchdog_observations else None,
                    canary_ready=True,
                    failure_stage=None,
                    error_message=None,
                ),
                steps=tuple(steps),
            )
    except Exception as exc:
        failure_stage = failure_stage or locals().get("stage_name") or None
        stage_started_value = locals().get("stage_started")
        remote_write_context = _exception_remote_write_context(exc)
        root_cause_message = _format_exception_text(_root_cause_exception(exc))
        exception_chain_payload = _exception_chain_payload(exc)
        failure_evidence: dict[str, object] = {}
        if remote_write_context is not None:
            failure_evidence["remote_write_context"] = remote_write_context
        if root_cause_message is not None:
            failure_evidence["root_cause_message"] = root_cause_message
        if exception_chain_payload:
            failure_evidence["exception_chain"] = exception_chain_payload
        if isinstance(stage_started_value, (int, float)):
            steps.append(
                _step_payload(
                    name=str(failure_stage or "unknown_stage"),
                    started_monotonic=float(stage_started_value),
                    status="fail",
                    detail=f"{type(exc).__name__}: {exc}",
                    evidence=failure_evidence or None,
                )
            )
        watchdog_observations.append(_load_watchdog_observation(base_config))
        result = replace(
            result,
            status="failed",
            finished_at=_utc_now_iso(),
            error_message=f"{type(exc).__name__}: {exc}",
            failure_stage=str(failure_stage or "unknown_stage"),
            root_cause_message=root_cause_message,
            remote_write_context=remote_write_context,
            exception_chain=exception_chain_payload,
            watchdog_observations=tuple(watchdog_observations),
            consistency_assessment=_assess_watchdog_consistency(
                watchdog_observation=watchdog_observations[-1] if watchdog_observations else None,
                canary_ready=False,
                failure_stage=str(failure_stage or "unknown_stage"),
                error_message=f"{type(exc).__name__}: {exc}",
            ),
            steps=tuple(steps),
        )
    finally:
        _shutdown_service(writer_service)
        _shutdown_service(fresh_reader_service)

    if write_artifacts:
        result = write_live_retention_canary_artifacts(result, project_root=base_project_root)
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the live retention canary runner."""

    parser = argparse.ArgumentParser(description="Run the live remote-memory retention canary.")
    parser.add_argument("--env-file", default=".env", help="Path to the Twinr env file.")
    parser.add_argument("--probe-id", default=None, help="Optional stable probe id / namespace suffix.")
    parser.add_argument(
        "--no-write-artifacts",
        action="store_true",
        help="Skip writing the rolling ops artifact and per-run report snapshot.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entrypoint and print the structured result JSON."""

    args = _build_arg_parser().parse_args(argv)
    result = run_live_retention_canary(
        env_path=args.env_file,
        probe_id=args.probe_id,
        write_artifacts=not args.no_write_artifacts,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.ready else 1


if __name__ == "__main__":  # pragma: no cover - manual CLI execution
    raise SystemExit(main())


__all__ = [
    "LiveRetentionCanaryResult",
    "default_live_retention_canary_path",
    "default_live_retention_canary_report_dir",
    "run_live_retention_canary",
    "write_live_retention_canary_artifacts",
]
