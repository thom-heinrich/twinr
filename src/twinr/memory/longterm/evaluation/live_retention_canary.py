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
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermSourceRefV1
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.memory.longterm.evaluation.live_midterm_acceptance import (
    _build_isolated_config,
    _normalize_base_project_root,
    _shutdown_service,
)


_SCHEMA_VERSION = 1
_ACCEPT_KIND = "live_retention_canary"
_OPS_ARTIFACT_NAME = "retention_live_canary.json"
_REPORT_DIR_NAME = "retention_live_canary"


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
    )

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
            writer_service.ensure_remote_ready()

            now = datetime.now(timezone.utc)
            writer_service.object_store.commit_active_delta(
                object_upserts=_seed_retention_objects(now=now),
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
                retention = writer_service.run_retention()

            fresh_reader_service = LongTermMemoryService.from_config(fresh_reader_config)
            fresh_reader_service.ensure_remote_ready()
            fresh_state = fresh_reader_service.object_store.load_current_state_fine_grained()
            fresh_archive = fresh_reader_service.object_store.load_archived_objects_fine_grained()
            fresh_kept_ids = tuple(item.memory_id for item in fresh_state.objects)
            fresh_archived_ids = tuple(item.memory_id for item in fresh_archive)

            if "episode:retention_old_weather" not in {item.memory_id for item in retention.archived_objects}:
                raise RuntimeError("Retention canary did not archive the aged episode.")
            if "observation:retention_old_presence" not in set(retention.pruned_memory_ids):
                raise RuntimeError("Retention canary did not prune the aged observation.")
            if "event:retention_future_appointment" not in fresh_kept_ids:
                raise RuntimeError("Fresh reader no longer sees the expected newer event.")
            if "episode:retention_old_weather" not in fresh_archived_ids:
                raise RuntimeError("Fresh reader did not see the archived aged episode.")

            result = replace(
                result,
                status="ok",
                finished_at=_utc_now_iso(),
                projection_selected_ids=tuple(dict.fromkeys(projection_selected_ids)),
                pruned_memory_ids=tuple(retention.pruned_memory_ids),
                archived_memory_ids=tuple(item.memory_id for item in retention.archived_objects),
                fresh_kept_ids=fresh_kept_ids,
                fresh_archived_ids=fresh_archived_ids,
            )
    except Exception as exc:
        result = replace(
            result,
            status="failed",
            finished_at=_utc_now_iso(),
            error_message=f"{type(exc).__name__}: {exc}",
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
