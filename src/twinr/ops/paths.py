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


def _is_blank_config_path(value: str | Path | None) -> bool:
    # AUDIT-FIX(#3): Keep "unset" detection centralized so fallback behavior is deterministic for "" and "." config values.
    return value is None or str(value).strip() in {"", "."}


def _normalize_path(value: str | Path, *, field_name: str, base: Path | None = None) -> Path:
    # AUDIT-FIX(#4): Strip accidental whitespace and expand "~" so .env path values resolve predictably on Raspberry Pi Linux deployments.
    try:
        path = (
            Path(value.strip()).expanduser()
            if isinstance(value, str)
            else Path(value).expanduser()
        )
    except (TypeError, RuntimeError) as exc:
        # AUDIT-FIX(#2): Re-raise malformed path inputs as clear configuration errors instead of opaque pathlib exceptions.
        raise ValueError(f"Invalid path for {field_name}: {value!r}") from exc

    if not path.is_absolute() and base is not None:
        path = base / path

    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        # AUDIT-FIX(#2): Fail fast during resolution so bad config is caught at startup, not later during file-store I/O.
        raise ValueError(f"Unable to resolve {field_name}: {path!s}") from exc


def _resolve_descendant(root: Path, *parts: str) -> Path:
    candidate = root.joinpath(*parts)
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        # AUDIT-FIX(#2): Convert low-level resolution failures into actionable configuration errors.
        raise ValueError(f"Unable to resolve ops path: {candidate!s}") from exc

    # AUDIT-FIX(#1): Block symlink/path escapes so ops descendants cannot resolve outside the declared project_root.
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"Resolved ops path escapes project_root: {candidate!s} -> {resolved!s}"
        )

    return resolved


def resolve_ops_paths(project_root: str | Path) -> TwinrOpsPaths:
    root = _normalize_path(project_root, field_name="project_root")
    # AUDIT-FIX(#2): Reject existing files used as project_root; descendant directory creation would otherwise fail later and less clearly.
    if root.exists() and not root.is_dir():
        raise ValueError(
            f"project_root must be a directory path, got existing non-directory: {root!s}"
        )

    artifacts_root = _resolve_descendant(root, "artifacts")
    stores_root = _resolve_descendant(root, "artifacts", "stores")
    ops_store_root = _resolve_descendant(root, "artifacts", "stores", "ops")
    return TwinrOpsPaths(
        project_root=root,
        artifacts_root=artifacts_root,
        stores_root=stores_root,
        ops_store_root=ops_store_root,
        events_path=_resolve_descendant(
            root,
            "artifacts",
            "stores",
            "ops",
            "events.jsonl",
        ),
        usage_path=_resolve_descendant(
            root,
            "artifacts",
            "stores",
            "ops",
            "usage.jsonl",
        ),
        self_tests_root=_resolve_descendant(
            root,
            "artifacts",
            "ops",
            "self_tests",
        ),
        bundles_root=_resolve_descendant(
            root,
            "artifacts",
            "ops",
            "support_bundles",
        ),
    )


def resolve_ops_paths_for_config(config: TwinrConfig) -> TwinrOpsPaths:
    cwd = _normalize_path(Path.cwd(), field_name="cwd")
    raw_project_root = config.project_root
    if not _is_blank_config_path(raw_project_root):
        return resolve_ops_paths(
            _normalize_path(
                raw_project_root,
                field_name="config.project_root",
                base=cwd,
            )
        )

    raw_runtime_state_path = config.runtime_state_path
    if not _is_blank_config_path(raw_runtime_state_path):
        # AUDIT-FIX(#3): Use relative runtime_state_path as a real base instead of discarding it and drifting to cwd.
        runtime_state_path = _normalize_path(
            raw_runtime_state_path,
            field_name="config.runtime_state_path",
            base=cwd,
        )
        return resolve_ops_paths(runtime_state_path.parent)

    return resolve_ops_paths(cwd)