from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig


@dataclass(frozen=True, slots=True)
class TwinrOpsPaths:
    project_root: Path
    artifacts_root: Path
    stores_root: Path
    ops_store_root: Path
    events_path: Path
    usage_path: Path
    self_tests_root: Path
    bundles_root: Path


def resolve_ops_paths(project_root: str | Path) -> TwinrOpsPaths:
    root = Path(project_root).resolve()
    artifacts_root = (root / "artifacts").resolve()
    stores_root = (artifacts_root / "stores").resolve()
    ops_store_root = (stores_root / "ops").resolve()
    return TwinrOpsPaths(
        project_root=root,
        artifacts_root=artifacts_root,
        stores_root=stores_root,
        ops_store_root=ops_store_root,
        events_path=(ops_store_root / "events.jsonl").resolve(),
        usage_path=(ops_store_root / "usage.jsonl").resolve(),
        self_tests_root=(artifacts_root / "ops" / "self_tests").resolve(),
        bundles_root=(artifacts_root / "ops" / "support_bundles").resolve(),
    )


def resolve_ops_paths_for_config(config: TwinrConfig) -> TwinrOpsPaths:
    raw_project_root = Path(config.project_root)
    if raw_project_root.is_absolute():
        return resolve_ops_paths(raw_project_root)
    if str(raw_project_root).strip() not in {"", "."}:
        return resolve_ops_paths(raw_project_root)

    runtime_state_path = Path(config.runtime_state_path)
    if runtime_state_path.is_absolute():
        return resolve_ops_paths(runtime_state_path.resolve().parent)
    return resolve_ops_paths(Path.cwd())
