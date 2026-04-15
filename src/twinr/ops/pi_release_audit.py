# CHANGELOG: 2026-04-09
# BUG-1: Added one operator-facing Pi release audit that compares the current local
#        authoritative release against the persisted Pi release manifest and a live
#        checksum drift probe, so operators no longer have to mentally join three
#        separate tools to see whether /twinr is actually aligned.
# IMP-1: The audit summary intentionally strips the huge per-entry manifest payload
#        down to release metadata plus workspace status, making the report stable and
#        readable during live operations.

"""Collect one compact operator audit for the current Pi release state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Protocol, cast

from twinr.ops.pi_repo_mirror import (
    CURRENT_RELEASE_MANIFEST_RELATIVE_PATH,
    PiAuthoritativeReleaseManifest,
    PiAuthoritativeWorkspaceStatus,
    PiRepoMirrorCycleResult,
    PiRepoMirrorWatchdog,
    build_authoritative_release_manifest,
)
from twinr.ops.pi_runtime_deploy_remote import PiRemoteExecutor, load_remote_json_artifact
from twinr.ops.self_coding_pi import load_pi_connection_settings

_SubprocessRunner = Any


class _MirrorWatchdog(Protocol):
    """Describe the read-only watchdog surface used by the release audit."""

    def probe_once(
        self,
        *,
        apply_sync: bool = True,
        checksum: bool = True,
        max_change_lines: int = 40,
    ) -> PiRepoMirrorCycleResult:
        """Run one watchdog cycle."""


@dataclass(frozen=True, slots=True)
class PiReleaseManifestSummary:
    """Summarize one release manifest without the huge per-entry payload."""

    release_id: str | None
    repo_root: str | None
    source_commit: str | None
    source_dirty: bool | None
    generated_at_utc: str | None
    entry_count: int | None
    workspace_status: PiAuthoritativeWorkspaceStatus | None


@dataclass(frozen=True, slots=True)
class PiReleaseAuditResult:
    """Describe one combined local/remote Pi release audit."""

    generated_at_utc: str
    host: str
    remote_root: str
    remote_manifest_path: str
    status: str
    in_sync: bool
    local_release: PiReleaseManifestSummary
    remote_release_manifest: PiReleaseManifestSummary | None
    remote_manifest_present: bool
    remote_manifest_error: str | None
    release_id_match: bool
    source_commit_match: bool
    source_dirty_match: bool
    drift_probe: PiRepoMirrorCycleResult


def audit_pi_release(
    *,
    project_root: str | Path,
    pi_env_path: str | Path,
    remote_root: str = "/twinr",
    timeout_s: float = 120.0,
    max_change_lines: int = 40,
    subprocess_runner: _SubprocessRunner = subprocess.run,
    mirror_watchdog: _MirrorWatchdog | None = None,
    remote_executor: PiRemoteExecutor | None = None,
) -> PiReleaseAuditResult:
    """Collect one compact view of the local and deployed Pi release state."""

    if timeout_s <= 0:
        raise ValueError("timeout_s must be greater than zero")
    if max_change_lines <= 0:
        raise ValueError("max_change_lines must be greater than zero")

    resolved_root = Path(project_root).resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise ValueError(f"project root does not exist: {resolved_root}")

    settings = load_pi_connection_settings(pi_env_path)
    normalized_remote_root = _normalize_remote_root(remote_root)
    generated_at_utc = _utc_now_iso()
    local_manifest = build_authoritative_release_manifest(
        resolved_root,
        generated_at_utc=generated_at_utc,
    )
    local_release = _summarize_release_manifest(local_manifest)

    remote = (
        remote_executor
        if remote_executor is not None
        else PiRemoteExecutor(
            settings=settings,
            subprocess_runner=subprocess_runner,
            timeout_s=timeout_s,
        )
    )
    remote_manifest_path = (
        f"{normalized_remote_root.rstrip('/')}/{CURRENT_RELEASE_MANIFEST_RELATIVE_PATH.as_posix()}"
    )

    remote_manifest_payload: dict[str, object] | None = None
    remote_manifest_error: str | None = None
    try:
        remote_manifest_payload = load_remote_json_artifact(
            remote=remote,
            remote_path=remote_manifest_path,
        )
    except json.JSONDecodeError as exc:
        remote_manifest_error = f"{type(exc).__name__}: {exc}"

    remote_release_manifest = _summarize_remote_release_manifest(remote_manifest_payload)
    watchdog = (
        mirror_watchdog
        if mirror_watchdog is not None
        else cast(
            _MirrorWatchdog,
            PiRepoMirrorWatchdog.from_env(
                project_root=resolved_root,
                pi_env_path=pi_env_path,
                remote_root=normalized_remote_root,
                timeout_s=timeout_s,
                subprocess_runner=subprocess_runner,
            ),
        )
    )
    drift_probe = watchdog.probe_once(
        apply_sync=False,
        checksum=True,
        max_change_lines=max_change_lines,
    )

    release_id_match = bool(
        remote_release_manifest is not None
        and remote_release_manifest.release_id
        and remote_release_manifest.release_id == local_release.release_id
    )
    source_commit_match = bool(
        remote_release_manifest is not None
        and remote_release_manifest.source_commit
        and remote_release_manifest.source_commit == local_release.source_commit
    )
    source_dirty_match = bool(
        remote_release_manifest is not None
        and remote_release_manifest.source_dirty is not None
        and remote_release_manifest.source_dirty == local_release.source_dirty
    )
    status = _determine_audit_status(
        remote_manifest_error=remote_manifest_error,
        remote_release_manifest=remote_release_manifest,
        drift_probe=drift_probe,
        release_id_match=release_id_match,
        source_commit_match=source_commit_match,
        source_dirty_match=source_dirty_match,
    )
    return PiReleaseAuditResult(
        generated_at_utc=generated_at_utc,
        host=settings.host,
        remote_root=normalized_remote_root,
        remote_manifest_path=remote_manifest_path,
        status=status,
        in_sync=status == "in_sync",
        local_release=local_release,
        remote_release_manifest=remote_release_manifest,
        remote_manifest_present=remote_release_manifest is not None,
        remote_manifest_error=remote_manifest_error,
        release_id_match=release_id_match,
        source_commit_match=source_commit_match,
        source_dirty_match=source_dirty_match,
        drift_probe=drift_probe,
    )


def _summarize_release_manifest(manifest: PiAuthoritativeReleaseManifest) -> PiReleaseManifestSummary:
    """Return one compact summary for a typed release manifest."""

    return PiReleaseManifestSummary(
        release_id=manifest.release_id,
        repo_root=manifest.repo_root,
        source_commit=manifest.source_commit,
        source_dirty=manifest.source_dirty,
        generated_at_utc=manifest.generated_at_utc,
        entry_count=manifest.entry_count,
        workspace_status=manifest.workspace_status,
    )


def _summarize_remote_release_manifest(
    payload: dict[str, object] | None,
) -> PiReleaseManifestSummary | None:
    """Return one compact summary for the persisted Pi release manifest."""

    if not isinstance(payload, dict) or not payload:
        return None
    return PiReleaseManifestSummary(
        release_id=_coerce_optional_text(payload.get("release_id")),
        repo_root=_coerce_optional_text(payload.get("repo_root")),
        source_commit=_coerce_optional_text(payload.get("source_commit")),
        source_dirty=_coerce_optional_bool(payload.get("source_dirty")),
        generated_at_utc=_coerce_optional_text(payload.get("generated_at_utc")),
        entry_count=_coerce_optional_int(payload.get("entry_count")),
        workspace_status=_coerce_workspace_status(payload.get("workspace_status")),
    )


def _coerce_workspace_status(value: object) -> PiAuthoritativeWorkspaceStatus | None:
    """Rebuild one stored workspace-status snapshot when its core keys exist."""

    if not isinstance(value, dict):
        return None
    repo_root = _coerce_optional_text(value.get("repo_root"))
    head_commit = _coerce_optional_text(value.get("head_commit"))
    if repo_root is None or head_commit is None:
        return None
    return PiAuthoritativeWorkspaceStatus(
        repo_root=repo_root,
        head_commit=head_commit,
        source_dirty=bool(value.get("source_dirty")),
        tracked_dirty_count=_coerce_optional_int(value.get("tracked_dirty_count")) or 0,
        tracked_deleted_count=_coerce_optional_int(value.get("tracked_deleted_count")) or 0,
        untracked_count=_coerce_optional_int(value.get("untracked_count")) or 0,
        ignored_count=_coerce_optional_int(value.get("ignored_count")) or 0,
        sampled_tracked_dirty_paths=_coerce_str_tuple(value.get("sampled_tracked_dirty_paths")),
        sampled_tracked_deleted_paths=_coerce_str_tuple(value.get("sampled_tracked_deleted_paths")),
        sampled_untracked_paths=_coerce_str_tuple(value.get("sampled_untracked_paths")),
        sampled_ignored_paths=_coerce_str_tuple(value.get("sampled_ignored_paths")),
    )


def _coerce_optional_text(value: object) -> str | None:
    text = " ".join(str(value or "").split()).strip()
    return text or None


def _coerce_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _coerce_optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _coerce_str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    normalized: list[str] = []
    for item in value:
        text = _coerce_optional_text(item)
        if text is not None:
            normalized.append(text)
    return tuple(normalized)


def _determine_audit_status(
    *,
    remote_manifest_error: str | None,
    remote_release_manifest: PiReleaseManifestSummary | None,
    drift_probe: PiRepoMirrorCycleResult,
    release_id_match: bool,
    source_commit_match: bool,
    source_dirty_match: bool,
) -> str:
    """Classify one operator-facing release-audit status."""

    if remote_manifest_error is not None:
        return "remote_manifest_unparseable"
    if remote_release_manifest is None:
        return "remote_manifest_missing"
    if drift_probe.drift_detected:
        return "drift_detected"
    if not release_id_match:
        return "release_id_mismatch"
    if not source_commit_match:
        return "source_commit_mismatch"
    if not source_dirty_match:
        return "source_dirty_mismatch"
    return "in_sync"


def _normalize_remote_root(remote_root: str) -> str:
    """Return one normalized absolute remote root string for reporting."""

    normalized = str(remote_root or "").strip()
    if not normalized:
        raise ValueError("remote_root must not be empty")
    return normalized.rstrip("/") or "/"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "PiReleaseAuditResult",
    "PiReleaseManifestSummary",
    "audit_pi_release",
]
