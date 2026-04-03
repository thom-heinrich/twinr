"""Run the fixed unified-retrieval goldset against an isolated remote namespace.

The goldset seeds one deterministic multi-source fixture and then evaluates the
same explainable retrieval-plan assertions that matter in production:
candidate-source coverage, selected ids, join anchors, rendered sections, and
query-path access classes. Unlike the broader live Pi acceptance, this runner
uses a single isolated runtime root and is intended for repeatable developer
validation during retrieval work.
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
_REPORT_DIR_NAME = "unified_retrieval_goldset"
_OPS_ARTIFACT_NAME = "unified_retrieval_goldset.json"
_REMOTE_CASE_TIMEOUT_S = 90.0
_REMOTE_CASE_POLL_INTERVAL_S = 2.0
_DEFAULT_CASE_PROFILE = "expanded"


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
class UnifiedRetrievalGoldsetResult:
    """Describe one complete unified-retrieval goldset run."""

    probe_id: str
    status: str
    started_at: str
    finished_at: str
    env_path: str
    base_project_root: str
    runtime_namespace: str
    case_profile: str = _DEFAULT_CASE_PROFILE
    profile_memory_type_coverage: tuple[tuple[str, int], ...] = ()
    runtime_root: str | None = None
    seed_stats: UnifiedRetrievalFixtureSeedStats | None = None
    case_results: tuple[UnifiedRetrievalGoldsetCaseResult, ...] = ()
    artifact_path: str | None = None
    report_path: str | None = None
    error_message: str | None = None
    schema_version: int = _SCHEMA_VERSION

    @property
    def total_cases(self) -> int:
        """Return the number of evaluated goldset cases."""

        return len(self.case_results)

    @property
    def passed_cases(self) -> int:
        """Return the number of passing goldset cases."""

        return sum(1 for item in self.case_results if item.passed)

    @property
    def ready(self) -> bool:
        """Return whether the whole goldset run passed."""

        return self.status == "ok" and self.total_cases > 0 and self.passed_cases == self.total_cases

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
            "runtime_root": self.runtime_root,
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


def default_unified_retrieval_goldset_path(project_root: str | Path) -> Path:
    """Return the rolling ops artifact path for the latest goldset run."""

    return Path(project_root).expanduser().resolve() / "artifacts" / "stores" / "ops" / _OPS_ARTIFACT_NAME


def default_unified_retrieval_goldset_report_dir(project_root: str | Path) -> Path:
    """Return the per-run report directory for goldset snapshots."""

    return Path(project_root).expanduser().resolve() / "artifacts" / "reports" / _REPORT_DIR_NAME


def write_unified_retrieval_goldset_artifacts(
    result: UnifiedRetrievalGoldsetResult,
    *,
    project_root: str | Path,
) -> UnifiedRetrievalGoldsetResult:
    """Persist the rolling goldset artifact and per-run report snapshot."""

    artifact_path = default_unified_retrieval_goldset_path(project_root)
    report_dir = default_unified_retrieval_goldset_report_dir(project_root)
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


def run_unified_retrieval_goldset(
    *,
    env_path: str | Path = ".env",
    probe_id: str | None = None,
    case_profile: str = _DEFAULT_CASE_PROFILE,
    write_artifacts: bool = True,
) -> UnifiedRetrievalGoldsetResult:
    """Run the fixed unified-retrieval goldset inside one isolated remote namespace."""

    resolved_env_path = Path(env_path).expanduser().resolve(strict=False)
    started_at = _utc_now_iso()
    effective_probe_id = " ".join(
        str(probe_id or f"unified_retrieval_goldset_{started_at.replace(':', '').replace('-', '')}").split()
    ).strip()
    base_config = TwinrConfig.from_env(resolved_env_path)
    base_project_root = _normalize_base_project_root(resolved_env_path, base_config)
    runtime_namespace = f"twinr_unified_retrieval_goldset_{_safe_namespace_suffix(effective_probe_id)}"

    result = UnifiedRetrievalGoldsetResult(
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
    service: LongTermMemoryService | None = None

    try:
        if not base_config.chonkydb_base_url or not base_config.chonkydb_api_key:
            raise RuntimeError("ChonkyDB credentials are required for unified retrieval goldset runs.")
        with ExitStack() as stack:
            runtime_temp_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix=f"{effective_probe_id}_goldset_"))
            runtime_root = Path(runtime_temp_dir).resolve(strict=False)
            result = replace(result, runtime_root=str(runtime_root))
            runtime_config = _build_isolated_config(
                base_config=base_config,
                base_project_root=base_project_root,
                runtime_root=runtime_root,
                remote_namespace=runtime_namespace,
                background_store_turns=False,
            )
            service = LongTermMemoryService.from_config(runtime_config)
            ensure_unified_retrieval_remote_ready(service)
            seed_stats = seed_unified_retrieval_fixture(service)
            selected_cases = unified_retrieval_goldset_cases(profile=case_profile)
            case_results = wait_for_unified_retrieval_cases(
                service=service,
                cases=selected_cases,
                phase="goldset",
                timeout_s=_REMOTE_CASE_TIMEOUT_S,
                poll_interval_s=_REMOTE_CASE_POLL_INTERVAL_S,
            )
            total_cases, passed_cases, failed_case_ids = unified_retrieval_case_summary(case_results)
            if total_cases <= 0 or passed_cases != total_cases:
                raise RuntimeError(
                    "Unified retrieval goldset failed cases: "
                    + ", ".join(failed_case_ids or ("unknown",))
                )
            result = replace(
                result,
                status="ok",
                finished_at=_utc_now_iso(),
                seed_stats=seed_stats,
                case_results=case_results,
            )
    except Exception as exc:
        result = replace(
            result,
            status="failed",
            finished_at=_utc_now_iso(),
            error_message=f"{type(exc).__name__}: {exc}",
        )
    finally:
        _shutdown_service(service)

    if write_artifacts:
        result = write_unified_retrieval_goldset_artifacts(
            result,
            project_root=base_project_root,
        )
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the goldset runner."""

    parser = argparse.ArgumentParser(description="Run the unified retrieval goldset against an isolated namespace.")
    parser.add_argument("--env-file", default=".env", help="Path to the Twinr env file with ChonkyDB credentials.")
    parser.add_argument("--probe-id", default=None, help="Optional stable probe id / namespace suffix.")
    parser.add_argument(
        "--case-profile",
        default=_DEFAULT_CASE_PROFILE,
        choices=("core", "expanded"),
        help="Which unified-retrieval case profile to run.",
    )
    parser.add_argument(
        "--no-write-artifacts",
        action="store_true",
        help="Skip writing the rolling ops artifact and per-run report snapshot.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entrypoint and print the structured goldset result JSON."""

    args = _build_arg_parser().parse_args(argv)
    result = run_unified_retrieval_goldset(
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
    "UnifiedRetrievalGoldsetResult",
    "default_unified_retrieval_goldset_path",
    "default_unified_retrieval_goldset_report_dir",
    "run_unified_retrieval_goldset",
    "write_unified_retrieval_goldset_artifacts",
]
