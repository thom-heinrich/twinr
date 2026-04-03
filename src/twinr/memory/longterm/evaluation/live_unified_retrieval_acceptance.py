"""Run a live unified-retrieval acceptance against the real remote memory path.

This acceptance reuses the fixed unified-retrieval goldset cases but proves
them under the stronger remote-memory contract: one writer runtime seeds the
fixture into an isolated ChonkyDB namespace, then both the writer and a fresh
reader runtime must pass the same selected-id, join-anchor, context-section,
and access-path assertions. The result is persisted as an operator-facing
artifact so Pi validation can inspect the exact phase-by-phase outcome.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import tempfile

from twinr.agent.base_agent import TwinrConfig
from twinr.memory.longterm.evaluation._unified_retrieval_shared import (
    UnifiedRetrievalFixtureSeedStats,
    UnifiedRetrievalGoldsetCaseResult,
    ensure_unified_retrieval_remote_ready,
    seed_unified_retrieval_fixture,
    unified_retrieval_case_profile_memory_type_coverage,
    unified_retrieval_case_summary,
    unified_retrieval_goldset_cases,
    wait_for_unified_retrieval_cases,
)
from twinr.memory.longterm.evaluation.live_midterm_acceptance import (
    _build_isolated_config,
    _normalize_base_project_root,
    _safe_namespace_suffix,
    _shutdown_service,
)
from twinr.memory.longterm.runtime.service import LongTermMemoryService


_SCHEMA_VERSION = 1
_OPS_ARTIFACT_NAME = "unified_retrieval_live_acceptance.json"
_REPORT_DIR_NAME = "unified_retrieval_live_acceptance"
_WRITER_CASE_TIMEOUT_S = 90.0
_FRESH_READER_CASE_TIMEOUT_S = 120.0
_CASE_POLL_INTERVAL_S = 2.0
_DEFAULT_CASE_PROFILE = "core"


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in stable ISO-8601 form."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON payload atomically."""

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


@dataclass(frozen=True, slots=True)
class LiveUnifiedRetrievalAcceptanceResult:
    """Describe one complete live unified-retrieval acceptance run."""

    probe_id: str
    status: str
    started_at: str
    finished_at: str
    env_path: str
    base_project_root: str
    runtime_namespace: str
    case_profile: str = _DEFAULT_CASE_PROFILE
    profile_memory_type_coverage: tuple[tuple[str, int], ...] = ()
    writer_root: str | None = None
    fresh_reader_root: str | None = None
    seed_stats: UnifiedRetrievalFixtureSeedStats | None = None
    case_results: tuple[UnifiedRetrievalGoldsetCaseResult, ...] = ()
    artifact_path: str | None = None
    report_path: str | None = None
    error_message: str | None = None
    schema_version: int = _SCHEMA_VERSION

    @property
    def total_cases(self) -> int:
        """Return the total number of evaluated phase cases."""

        return len(self.case_results)

    @property
    def passed_cases(self) -> int:
        """Return the number of passing phase cases."""

        return sum(1 for item in self.case_results if item.passed)

    @property
    def ready(self) -> bool:
        """Return whether both writer and fresh-reader phases passed in full."""

        if self.status != "ok" or self.total_cases <= 0 or self.passed_cases != self.total_cases:
            return False
        phases = {item.phase for item in self.case_results}
        return {"writer", "fresh_reader"}.issubset(phases)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "probe_id": self.probe_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "env_path": self.env_path,
            "base_project_root": self.base_project_root,
            "runtime_namespace": self.runtime_namespace,
            "case_profile": self.case_profile,
            "profile_memory_type_coverage": {
                key: value for key, value in self.profile_memory_type_coverage
            },
            "writer_root": self.writer_root,
            "fresh_reader_root": self.fresh_reader_root,
            "seed_stats": self.seed_stats.to_dict() if self.seed_stats is not None else None,
            "case_results": [item.to_dict() for item in self.case_results],
            "artifact_path": self.artifact_path,
            "report_path": self.report_path,
            "error_message": self.error_message,
            "schema_version": self.schema_version,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "ready": self.ready,
        }


def default_live_unified_retrieval_acceptance_path(project_root: str | Path) -> Path:
    """Return the rolling ops artifact path for the latest live acceptance."""

    return Path(project_root).expanduser().resolve() / "artifacts" / "stores" / "ops" / _OPS_ARTIFACT_NAME


def default_live_unified_retrieval_acceptance_report_dir(project_root: str | Path) -> Path:
    """Return the per-run report directory for live acceptance snapshots."""

    return Path(project_root).expanduser().resolve() / "artifacts" / "reports" / _REPORT_DIR_NAME


def write_live_unified_retrieval_acceptance_artifacts(
    result: LiveUnifiedRetrievalAcceptanceResult,
    *,
    project_root: str | Path,
) -> LiveUnifiedRetrievalAcceptanceResult:
    """Persist the latest live-acceptance artifact plus a per-run snapshot."""

    artifact_path = default_live_unified_retrieval_acceptance_path(project_root)
    report_dir = default_live_unified_retrieval_acceptance_report_dir(project_root)
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


def run_live_unified_retrieval_acceptance(
    *,
    env_path: str | Path = ".env",
    probe_id: str | None = None,
    case_profile: str = _DEFAULT_CASE_PROFILE,
    write_artifacts: bool = True,
) -> LiveUnifiedRetrievalAcceptanceResult:
    """Run the live unified-retrieval acceptance against real ChonkyDB state."""

    resolved_env_path = Path(env_path).expanduser().resolve(strict=False)
    started_at = _utc_now_iso()
    effective_probe_id = " ".join(
        str(probe_id or f"unified_retrieval_live_{started_at.replace(':', '').replace('-', '')}").split()
    ).strip()
    base_config = TwinrConfig.from_env(resolved_env_path)
    base_project_root = _normalize_base_project_root(resolved_env_path, base_config)
    runtime_namespace = f"twinr_unified_retrieval_live_{_safe_namespace_suffix(effective_probe_id)}"

    result = LiveUnifiedRetrievalAcceptanceResult(
        probe_id=effective_probe_id,
        status="running",
        started_at=started_at,
        finished_at=started_at,
        env_path=str(resolved_env_path),
        base_project_root=str(base_project_root),
        runtime_namespace=runtime_namespace,
        case_profile=case_profile,
        profile_memory_type_coverage=unified_retrieval_case_profile_memory_type_coverage(profile=case_profile),
    )
    writer_service: LongTermMemoryService | None = None
    fresh_reader_service: LongTermMemoryService | None = None

    try:
        if not base_config.chonkydb_base_url or not base_config.chonkydb_api_key:
            raise RuntimeError("ChonkyDB credentials are required for live unified retrieval acceptance.")
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
            fresh_reader_service = LongTermMemoryService.from_config(fresh_reader_config)
            ensure_unified_retrieval_remote_ready(writer_service)
            ensure_unified_retrieval_remote_ready(fresh_reader_service)

            seed_stats = seed_unified_retrieval_fixture(writer_service)
            selected_cases = unified_retrieval_goldset_cases(profile=case_profile)
            writer_results = wait_for_unified_retrieval_cases(
                service=writer_service,
                cases=selected_cases,
                phase="writer",
                timeout_s=_WRITER_CASE_TIMEOUT_S,
                poll_interval_s=_CASE_POLL_INTERVAL_S,
            )
            fresh_reader_results = wait_for_unified_retrieval_cases(
                service=fresh_reader_service,
                cases=selected_cases,
                phase="fresh_reader",
                timeout_s=_FRESH_READER_CASE_TIMEOUT_S,
                poll_interval_s=_CASE_POLL_INTERVAL_S,
            )
            all_results = tuple((*writer_results, *fresh_reader_results))
            total_cases, passed_cases, failed_case_ids = unified_retrieval_case_summary(all_results)
            if total_cases <= 0 or passed_cases != total_cases:
                raise RuntimeError(
                    "Live unified retrieval acceptance failed cases: "
                    + ", ".join(failed_case_ids or ("unknown",))
                )
            result = replace(
                result,
                status="ok",
                finished_at=_utc_now_iso(),
                seed_stats=seed_stats,
                case_results=all_results,
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
        result = write_live_unified_retrieval_acceptance_artifacts(
            result,
            project_root=base_project_root,
        )
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the live acceptance runner."""

    parser = argparse.ArgumentParser(description="Run the live unified retrieval acceptance suite.")
    parser.add_argument("--env-file", default=".env", help="Path to the Twinr env file.")
    parser.add_argument("--probe-id", default=None, help="Optional stable probe id / namespace suffix.")
    parser.add_argument(
        "--case-profile",
        default=_DEFAULT_CASE_PROFILE,
        choices=("core", "expanded"),
        help="Which unified-retrieval case profile to run live against ChonkyDB.",
    )
    parser.add_argument(
        "--no-write-artifacts",
        action="store_true",
        help="Skip writing the rolling ops artifact and per-run report snapshot.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entrypoint and print the structured live-acceptance JSON."""

    args = _build_arg_parser().parse_args(argv)
    result = run_live_unified_retrieval_acceptance(
        env_path=args.env_file,
        probe_id=args.probe_id,
        case_profile=args.case_profile,
        write_artifacts=not args.no_write_artifacts,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.ready else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "LiveUnifiedRetrievalAcceptanceResult",
    "default_live_unified_retrieval_acceptance_path",
    "default_live_unified_retrieval_acceptance_report_dir",
    "run_live_unified_retrieval_acceptance",
    "write_live_unified_retrieval_acceptance_artifacts",
]
