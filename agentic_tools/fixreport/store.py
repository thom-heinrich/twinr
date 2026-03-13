"""
Contract
- Purpose: Persist FixReport entries under a repo-local store with locking and an index.
- Inputs (types, units): Validated fixreport dict + optional narrative/evidence.
- Outputs (types, units): YAML files on disk; index + events updated.
- Invariants: bf_id uniqueness; index reflects report files; writes are atomic.
- Error semantics: Fail-fast; partial writes avoided via tmp + os.replace.
- Time/Horizon: Timestamps are recorded as ISO-8601 UTC 'Z'.
- External boundaries: Filesystem under repo_root/artifacts/stores/fixreport (legacy: repo_root/state/fixreport).
"""

from __future__ import annotations

##REFACTOR: 2026-01-16##

import os
import re
import tempfile
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from agentic_tools._governance_locking import (
    FileLockTimeout,
    GovernanceFileLock,
    resolve_lock_settings,
)
from agentic_tools._governance_locking import (
    resolve_retry_budget,
    resolve_retry_poll,
    run_mutation_with_retry,
)
from agentic_tools.findings.store import FindingsStore
from common.links.tessairact_meta import (
    dedupe_link_tokens,
    ensure_tessairact_meta_header,
    extract_link_tokens_from_texts,
    infer_area_from_path,
)

from .vocab import Vocab, load_vocab, validate_fixreport_fields
from .failure_modes import (
    FailureModesCatalog,
    load_failure_modes_catalog,
    validate_failure_modes_block,
)
from agentic_tools._store_layout import resolve_store_dir


_BF_ID_RE = re.compile(r"^BF[0-9]{6}$")
_LIST_TOKEN_FIELDS = {
    "archetypes",
    "paths_touched",
    "contracts",
    "topics",
    "tables_views",
    "hypothesis_ids",
    "task_ids",
    "ops_msg_ids",
    "audit_ids",
    "links",
    "tags",
}

_POST_COMMIT_SYNC_KEY = "post_commit_sync"
_POST_COMMIT_FINDING_BACKLINKS_KEY = "finding_backlinks"

# Optional: enable durable writes (fsync) for atomic file replacement.
# Default off to preserve prior latency/throughput characteristics.
_DURABLE_WRITES_ENV = "FIXREPORT_DURABLE_WRITES"


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _durable_writes_enabled() -> bool:
    v = (os.environ.get(_DURABLE_WRITES_ENV) or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _collect_text_blobs(value: Any, *, max_items: int = 200) -> List[str]:
    out: List[str] = []

    def _walk(v: Any) -> None:
        if len(out) >= max_items:
            return
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s)
            return
        if isinstance(v, Mapping):
            for k, vv in v.items():
                if len(out) >= max_items:
                    return
                if str(k) in {"links", "metadata"}:
                    continue
                _walk(vv)
            return
        if isinstance(v, list):
            for vv in v:
                if len(out) >= max_items:
                    return
                _walk(vv)
            return

    _walk(value)
    return out


def _load_finding_allowed_fixreport_ids(
    *, repo_root: Path, finding_id: str
) -> tuple[set[str], bool]:
    if not str(finding_id or "").strip():
        return set(), False
    findings_dir = resolve_store_dir(
        legacy_rel="state/findings",
        canonical_rel="findings",
        repo_root=repo_root,
    )
    report_path = findings_dir / "reports" / f"{str(finding_id).strip()}.yml"
    if not report_path.exists():
        raise FileNotFoundError(f"missing finding backlink target: {finding_id}")
    try:
        raw = yaml.safe_load(report_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"invalid finding backlink target YAML: {finding_id}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"invalid finding backlink target document: {finding_id}")
    allowed: set[str] = set()
    finding = raw.get("finding")
    if isinstance(finding, Mapping):
        related = finding.get("related")
        if isinstance(related, Mapping):
            raw_ids = related.get("fixreport_ids")
            if isinstance(raw_ids, list):
                for raw_fix_id in raw_ids:
                    fix_id = str(raw_fix_id or "").strip()
                    if fix_id:
                        allowed.add(fix_id)
    resolution = raw.get("resolution")
    if not isinstance(resolution, Mapping):
        return allowed, True
    resolution_fix_id = str(resolution.get("fixreport_id") or "").strip()
    if resolution_fix_id:
        allowed.add(resolution_fix_id)
    return allowed, True


def _finding_ids_from_link_tokens(links: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw_token in list(links or []):
        token = str(raw_token or "").strip()
        if not token.startswith("finding:"):
            continue
        finding_id = token.split(":", 1)[1].strip()
        if finding_id:
            out.append(finding_id)
    return dedupe_link_tokens(out)


def _normalize_finding_ids(finding_ids: Sequence[str]) -> List[str]:
    return dedupe_link_tokens(
        [str(x or "").strip() for x in list(finding_ids or []) if str(x or "").strip()]
    )


class FixReportPostCommitSyncError(RuntimeError):
    """Raised when fixreport local commit succeeded but cross-store finding sync did not."""

    def __init__(
        self,
        *,
        action: str,
        bf_id: str,
        report_path: Path,
        finding_ids: Sequence[str],
        cause: BaseException,
    ) -> None:
        self.action = str(action or "").strip() or "fixreport_sync"
        self.bf_id = str(bf_id or "").strip()
        self.report_path = Path(report_path)
        self.finding_ids = [
            str(x or "").strip()
            for x in list(finding_ids or [])
            if str(x or "").strip()
        ]
        super().__init__(
            f"{self.action} committed {self.bf_id} at {self.report_path} but finding backlink sync failed "
            f"for {self.finding_ids}: {cause!r}"
        )


def _validate_explicit_finding_targets(
    *, repo_root: Path, finding_ids: Sequence[str]
) -> List[str]:
    clean_ids = _normalize_finding_ids(finding_ids)
    if not clean_ids:
        return []
    findings_store = FindingsStore(repo_root=repo_root)
    missing: List[str] = []
    for finding_id in clean_ids:
        try:
            findings_store.report_path(str(finding_id))
        except Exception as exc:
            raise ValueError(f"invalid explicit finding id: {finding_id}") from exc
        try:
            _load_finding_allowed_fixreport_ids(
                repo_root=repo_root, finding_id=str(finding_id)
            )
        except FileNotFoundError:
            missing.append(str(finding_id))
    if missing:
        raise FileNotFoundError(
            "missing explicit finding(s) for fixreport backlink sync: "
            + ", ".join(sorted(missing))
        )
    return clean_ids


def _fixreport_backlink_sync_retry_budget_sec() -> float:
    return resolve_retry_budget(
        retry_timeout_env="CAIA_FIXREPORT_BACKLINK_SYNC_RETRY_TIMEOUT_SEC",
        lock_timeout_envs=("CAIA_FINDINGS_LOCK_TIMEOUT_SEC", "FINDINGS_LOCK_TIMEOUT"),
        default_lock_timeout=10.0,
        minimum_budget_sec=15.0,
    )


def _fixreport_backlink_sync_retry_poll_sec() -> float:
    return resolve_retry_poll(
        retry_poll_env="CAIA_FIXREPORT_BACKLINK_SYNC_RETRY_POLL_SEC",
        default_poll_sec=0.25,
    )


def _classify_fixreport_backlink_sync_exception(exc: BaseException) -> Tuple[str, bool]:
    if isinstance(exc, FileLockTimeout) and exc.__cause__ is not None:
        return _classify_fixreport_backlink_sync_exception(exc.__cause__)
    if isinstance(exc, TimeoutError):
        return ("TIMEOUT", True)
    if isinstance(exc, OSError):
        return ("IO_ERROR", True)
    return (str(type(exc).__name__ or "UNKNOWN_ERROR").upper(), False)


def _emit_fixreport_backlink_sync_retry(
    *,
    action: str,
    bf_id: str,
    error_code: str,
    attempt: int,
    sleep_s: float,
    exc: BaseException,
) -> None:
    logging.warning(
        "Retrying fixreport post-commit finding sync action=%s bf_id=%s attempt=%s sleep_s=%.3f error_code=%s error=%s",
        action,
        bf_id,
        attempt,
        sleep_s,
        error_code,
        exc,
    )


def _sync_explicit_finding_links_after_commit(
    *,
    action: str,
    repo_root: Path,
    bf_id: str,
    report_path: Path,
    finding_ids: Sequence[str],
    actor: Optional[str],
) -> Dict[str, Any]:
    clean_ids = _normalize_finding_ids(finding_ids)
    if not clean_ids:
        return {}
    attempted_at_utc = _utc_now_z()
    try:
        _, retry_meta = run_mutation_with_retry(
            action=f"fixreport_{str(action or '').strip() or 'sync'}_finding_links",
            op=lambda: _sync_explicit_finding_links(
                repo_root=repo_root,
                bf_id=bf_id,
                finding_ids=clean_ids,
                actor=actor,
            ),
            classify_exception=_classify_fixreport_backlink_sync_exception,
            retry_budget_sec=_fixreport_backlink_sync_retry_budget_sec(),
            retry_poll_sec=_fixreport_backlink_sync_retry_poll_sec(),
            emit_retry=lambda error_code,
            attempt,
            sleep_s,
            exc: _emit_fixreport_backlink_sync_retry(
                action=action,
                bf_id=bf_id,
                error_code=error_code,
                attempt=attempt,
                sleep_s=sleep_s,
                exc=exc,
            ),
        )
    except Exception as exc:
        error_code, _ = _classify_fixreport_backlink_sync_exception(exc)
        logging.error(
            "Fixreport post-commit finding sync failed action=%s bf_id=%s path=%s error_code=%s finding_ids=%s error=%r",
            action,
            bf_id,
            report_path,
            error_code,
            clean_ids,
            exc,
        )
        return {
            _POST_COMMIT_FINDING_BACKLINKS_KEY: {
                "action": str(action or "").strip() or "update",
                "status": "failed",
                "finding_ids": list(clean_ids),
                "attempted_at_utc": attempted_at_utc,
                "updated_at_utc": _utc_now_z(),
                "error_code": error_code,
                "error": repr(exc),
            }
        }
    synced_at_utc = _utc_now_z()
    state: Dict[str, Any] = {
        _POST_COMMIT_FINDING_BACKLINKS_KEY: {
            "action": str(action or "").strip() or "update",
            "status": "synced",
            "finding_ids": list(clean_ids),
            "attempted_at_utc": attempted_at_utc,
            "synced_at_utc": synced_at_utc,
            "updated_at_utc": synced_at_utc,
        }
    }
    if retry_meta:
        state[_POST_COMMIT_FINDING_BACKLINKS_KEY]["retry"] = dict(retry_meta)
    return state


def _sync_explicit_finding_links(
    *,
    repo_root: Path,
    bf_id: str,
    finding_ids: Sequence[str],
    actor: Optional[str],
) -> None:
    clean_ids = [
        str(x or "").strip() for x in list(finding_ids or []) if str(x or "").strip()
    ]
    if not clean_ids:
        return
    findings_store = FindingsStore(repo_root=repo_root)
    for finding_id in clean_ids:
        findings_store.update(
            finding_id,
            set_finding={"related": {"fixreport_ids": [str(bf_id)]}},
            add_links=[f"fixreport:{str(bf_id)}"],
            actor=actor,
        )


def _canonicalize_fixreport_links(
    *,
    repo_root: Path,
    bf_id: str,
    links: Sequence[str],
    explicit_finding_ids: Sequence[str] = (),
) -> List[str]:
    explicit_ids = {
        str(x or "").strip()
        for x in list(explicit_finding_ids or [])
        if str(x or "").strip()
    }
    out: List[str] = []
    for raw_token in list(links or []):
        token = str(raw_token or "").strip()
        if not token:
            continue
        if token == f"fixreport:{bf_id}":
            continue
        if token.startswith("finding:"):
            finding_id = token.split(":", 1)[1].strip()
            allowed_fix_ids, finding_exists = _load_finding_allowed_fixreport_ids(
                repo_root=repo_root,
                finding_id=finding_id,
            )
            if (
                finding_exists
                and allowed_fix_ids
                and bf_id not in allowed_fix_ids
                and finding_id not in explicit_ids
            ):
                continue
        out.append(token)
    return dedupe_link_tokens(out)


def _add_exc_note(exc: BaseException, note: str) -> None:
    """
    Attach debugging context without changing exception type.
    Uses BaseException.add_note if available (Py>=3.11).
    """
    add_note = getattr(exc, "add_note", None)
    if callable(add_note):
        try:
            add_note(note)
        except Exception:
            pass


def _split_csv_tokens(raw: Any, *, field: str = "") -> List[str]:
    """
    Normalize list-token-like fields.

    Supports:
    - None -> []
    - "a,b,c" -> ["a","b","c"]
    - ["a,b", "c"] -> ["a","b","c"]

    Whitespace is trimmed, empties dropped, duplicates removed (stable).

    Strictness:
    - For list inputs, all items must be strings (fail-fast otherwise).
    - For other non-supported types, fail-fast (no silent dropping).
    """
    parts: List[str]
    if raw is None:
        parts = []
    elif isinstance(raw, str):
        parts = [raw]
    elif isinstance(raw, list):
        bad = [type(x).__name__ for x in raw if not isinstance(x, str)]
        if bad:
            fld = f" for field '{field}'" if field else ""
            raise ValueError(
                f"invalid token list{fld}: all items must be strings (got {bad[0]})"
            )
        parts = list(raw)
    else:
        fld = f" for field '{field}'" if field else ""
        raise ValueError(
            f"invalid token value{fld}: expected str|list[str]|None (got {type(raw).__name__})"
        )

    out: List[str] = []
    seen = set()
    for item in parts:
        for tok in item.split(","):
            t = tok.strip()
            if not t:
                continue
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
    return out


def _validate_create_archetypes(fixreport_fields: Mapping[str, Any]) -> None:
    raw = fixreport_fields.get("archetypes")
    if raw is None:
        raise ValueError(
            "validation_failed: missing mandatory field archetypes for create; "
            "set fixreport.archetypes (1..5 controlled tokens, e.g. stateful_agent_workflow, hnsw_manager)"
        )
    if not isinstance(raw, list):
        raise ValueError("validation_failed: invalid archetypes: expected list")
    items = [str(x).strip() for x in raw if str(x).strip()]
    if not items:
        raise ValueError(
            "validation_failed: invalid archetypes: expected non-empty list"
        )


def _normalize_fixreport_fields_inplace(fixreport_fields: Dict[str, Any]) -> None:
    for k in _LIST_TOKEN_FIELDS:
        if k not in fixreport_fields:
            continue
        fixreport_fields[k] = _split_csv_tokens(fixreport_fields.get(k), field=k)
        if not fixreport_fields[k]:
            fixreport_fields.pop(k, None)


def _validate_fixreport_doc_identity(
    *, doc: Mapping[str, Any], expected_bf_id: str, path: Path
) -> None:
    fr = doc.get("fixreport")
    if not isinstance(fr, Mapping):
        raise ValueError(f"invalid fixreport file: missing fixreport mapping in {path}")
    actual_bf_id = str(fr.get("bf_id") or "").strip()
    if not _BF_ID_RE.match(actual_bf_id):
        raise ValueError(f"invalid fixreport bf_id in {path}: {actual_bf_id!r}")
    file_bf_id = str(path.stem or "").strip()
    if file_bf_id != str(expected_bf_id or "").strip():
        raise ValueError(
            f"fixreport path identity mismatch: expected filename stem {expected_bf_id!r}, got {file_bf_id!r}"
        )
    if actual_bf_id != str(expected_bf_id or "").strip():
        raise ValueError(
            f"fixreport filename/content bf_id mismatch for {path}: filename={expected_bf_id!r} content={actual_bf_id!r}"
        )


def _prepare_fixreport_links_and_finding_targets(
    *,
    repo_root: Path,
    bf_id: str,
    fixreport_fields: Dict[str, Any],
    narrative: Optional[Mapping[str, Any]],
    evidence: Optional[Sequence[Mapping[str, Any]]],
    failure_modes: Optional[Mapping[str, Any]],
    explicit_finding_ids: Optional[Sequence[str]],
    explicit_link_finding_ids: Optional[Sequence[str]] = None,
) -> List[str]:
    auto_links = extract_link_tokens_from_texts(
        _collect_text_blobs(fixreport_fields)
        + _collect_text_blobs(narrative)
        + _collect_text_blobs(evidence)
        + _collect_text_blobs(failure_modes)
    )
    auto_links = [x for x in auto_links if x != f"fixreport:{bf_id}"]

    raw_links = fixreport_fields.get("links")
    merged_links: List[str] = []
    if raw_links is not None:
        if isinstance(raw_links, list):
            merged_links.extend(str(x) for x in raw_links)
        else:
            merged_links.append(str(raw_links))
    merged_links.extend(auto_links)
    if merged_links:
        fixreport_fields["links"] = merged_links
    else:
        fixreport_fields.pop("links", None)

    _normalize_fixreport_fields_inplace(fixreport_fields)
    normalized_links = (
        list(fixreport_fields.get("links") or [])
        if isinstance(fixreport_fields.get("links"), list)
        else []
    )

    explicit_requested_finding_ids = _normalize_finding_ids(
        list(explicit_finding_ids or []) + list(explicit_link_finding_ids or [])
    )
    explicit_requested_finding_ids = _validate_explicit_finding_targets(
        repo_root=repo_root,
        finding_ids=explicit_requested_finding_ids,
    )

    canonical_links = _canonicalize_fixreport_links(
        repo_root=repo_root,
        bf_id=bf_id,
        links=normalized_links,
        explicit_finding_ids=explicit_requested_finding_ids,
    )
    if canonical_links:
        fixreport_fields["links"] = canonical_links
    else:
        fixreport_fields.pop("links", None)

    return _normalize_finding_ids(
        list(explicit_finding_ids or [])
        + list(explicit_link_finding_ids or [])
        + _finding_ids_from_link_tokens(
            list(fixreport_fields.get("links") or [])
            if isinstance(fixreport_fields.get("links"), list)
            else []
        )
    )


def _set_pending_post_commit_finding_sync(
    *,
    doc: Dict[str, Any],
    action: str,
    finding_ids: Sequence[str],
) -> None:
    clean_ids = _normalize_finding_ids(finding_ids)
    if not clean_ids:
        doc.pop(_POST_COMMIT_SYNC_KEY, None)
        return
    now = _utc_now_z()
    doc[_POST_COMMIT_SYNC_KEY] = {
        _POST_COMMIT_FINDING_BACKLINKS_KEY: {
            "action": str(action or "").strip() or "update",
            "status": "pending",
            "finding_ids": list(clean_ids),
            "pending_since_utc": now,
            "updated_at_utc": now,
        }
    }


class FileLock(GovernanceFileLock):
    """Fixreport store lock backed by the shared governance lock primitive."""

    def __init__(self, path: Path, timeout_s: float = 15.0, poll_s: float = 0.05):
        settings = resolve_lock_settings(
            timeout_envs=("CAIA_FIXREPORT_LOCK_TIMEOUT_SEC", "FIXREPORT_LOCK_TIMEOUT"),
            timeout_default=float(timeout_s),
            poll_env="CAIA_FIXREPORT_LOCK_POLL_SEC",
            poll_default=float(poll_s),
            stale_env="CAIA_FIXREPORT_LOCK_STALE_SEC",
            stale_default=120,
            heartbeat_env="CAIA_FIXREPORT_LOCK_HEARTBEAT_SEC",
        )
        super().__init__(
            path=path,
            timeout_sec=settings.timeout_sec,
            poll_sec=settings.poll_sec,
            stale_after_sec=settings.stale_after_sec,
            heartbeat_sec=settings.heartbeat_sec,
        )


@dataclass(frozen=True)
class FixReportPaths:
    state_dir: Path
    reports_dir: Path
    index_path: Path
    events_path: Path
    lock_path: Path
    vocab_path: Path
    failure_modes_catalog_path: Path


def default_paths(repo_root: Path) -> FixReportPaths:
    state_dir = resolve_store_dir(
        legacy_rel="state/fixreport",
        canonical_rel="fixreport",
        repo_root=repo_root,
    )
    return FixReportPaths(
        state_dir=state_dir,
        reports_dir=state_dir / "reports",
        index_path=state_dir / "index.yml",
        events_path=state_dir / "events.yml",
        lock_path=state_dir / ".lock",
        vocab_path=repo_root / "agentic_tools" / "fixreport" / "vocab.yaml",
        failure_modes_catalog_path=repo_root
        / "agentic_tools"
        / "fixreport"
        / "failure_modes_catalog.yaml",
    )


def _fsync_dir_best_effort(dir_path: Path) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(str(dir_path), os.O_DIRECTORY)  # type: ignore[attr-defined]
    except Exception:
        return
    try:
        os.fsync(fd)
    except Exception:
        pass
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


def _atomic_write_text(path: Path, text: str) -> None:
    """
    Atomic replace write using a unique temp file in the same directory.

    - Avoids deterministic *.tmp collision under concurrency.
    - Uses os.replace for atomic replacement.
    - Optional durability (fsync) can be enabled via env var FIXREPORT_DURABLE_WRITES=1.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Important: tempfile.mkstemp creates files as 0o600. In this repo we commonly rely on
    # directory default ACLs (e.g. user:thh) to allow portal/service writers. If the file mode
    # has group bits = 0, the ACL mask becomes --- and *effective* ACL permissions for named
    # users/groups are stripped (portal then hits PermissionError on read/write).
    #
    # Therefore we force a group-writable mode on the temp file before it is replaced into
    # place. We keep "other" at 0 (no world access).
    target_mode = 0o660
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        try:
            os.fchmod(fd, target_mode)
        except Exception:
            # Best-effort: on non-POSIX platforms fchmod may be unavailable.
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            if _durable_writes_enabled():
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
        os.replace(str(tmp_path), str(path))
        if _durable_writes_enabled():
            _fsync_dir_best_effort(path.parent)
    finally:
        # Clean up leftover tmp if os.replace didn't happen.
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass


def _load_yaml_or_default(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        _add_exc_note(e, f"while loading YAML: {path}")
        raise
    return default if raw is None else raw


class FixReportStore:
    def __init__(
        self,
        repo_root: Path,
        *,
        vocab_path: Optional[Path] = None,
        failure_modes_catalog_path: Optional[Path] = None,
        state_dir: Optional[Path] = None,
    ):
        self.repo_root = repo_root.resolve()
        self.paths = default_paths(self.repo_root)
        if state_dir is not None:
            state_dir_r = state_dir.expanduser().resolve()
            self.paths = FixReportPaths(
                state_dir=state_dir_r,
                reports_dir=state_dir_r / "reports",
                index_path=state_dir_r / "index.yml",
                events_path=state_dir_r / "events.yml",
                lock_path=state_dir_r / ".lock",
                vocab_path=self.paths.vocab_path,
                failure_modes_catalog_path=self.paths.failure_modes_catalog_path,
            )
        if vocab_path is not None:
            self.paths = FixReportPaths(
                state_dir=self.paths.state_dir,
                reports_dir=self.paths.reports_dir,
                index_path=self.paths.index_path,
                events_path=self.paths.events_path,
                lock_path=self.paths.lock_path,
                vocab_path=vocab_path,
                failure_modes_catalog_path=self.paths.failure_modes_catalog_path,
            )
        if failure_modes_catalog_path is not None:
            self.paths = FixReportPaths(
                state_dir=self.paths.state_dir,
                reports_dir=self.paths.reports_dir,
                index_path=self.paths.index_path,
                events_path=self.paths.events_path,
                lock_path=self.paths.lock_path,
                vocab_path=self.paths.vocab_path,
                failure_modes_catalog_path=failure_modes_catalog_path,
            )
        self._vocab: Optional[Vocab] = None
        self._fm_catalog: Optional[FailureModesCatalog] = None

    @property
    def vocab(self) -> Vocab:
        if self._vocab is None:
            self._vocab = load_vocab(self.paths.vocab_path)
        return self._vocab

    @property
    def failure_modes_catalog(self) -> FailureModesCatalog:
        if self._fm_catalog is None:
            self._fm_catalog = load_failure_modes_catalog(
                self.paths.failure_modes_catalog_path
            )
        return self._fm_catalog

    def init(self, force: bool = False) -> None:
        # Ensure directories exist first; lockfile lives under state_dir.
        self.paths.reports_dir.mkdir(parents=True, exist_ok=True)

        # Serialize creation/rewrites of index/events to avoid races.
        lock = FileLock(self.paths.lock_path)
        with lock:
            if force or not self.paths.index_path.exists():
                payload: Dict[str, Any] = {"fixreports": []}
                ensure_tessairact_meta_header(
                    payload,
                    kind="fixreports_index",
                    area="ops",
                    uid="FIXREPORT_INDEX",
                    actor="fixreport",
                    links=["script:agentic_tools/fixreport/store.py"],
                    repo_root=self.repo_root,
                    tool="fixreport",
                )
                _atomic_write_text(
                    self.paths.index_path, yaml.safe_dump(payload, sort_keys=True)
                )
            if force or not self.paths.events_path.exists():
                payload2: Dict[str, Any] = {"events": []}
                ensure_tessairact_meta_header(
                    payload2,
                    kind="fixreports_events",
                    area="ops",
                    uid="FIXREPORT_EVENTS",
                    actor="fixreport",
                    links=["script:agentic_tools/fixreport/store.py"],
                    repo_root=self.repo_root,
                    tool="fixreport",
                )
                _atomic_write_text(
                    self.paths.events_path, yaml.safe_dump(payload2, sort_keys=True)
                )

    def _next_bf_id(self) -> str:
        existing = self.list_bf_ids()
        if not existing:
            return "BF000001"
        max_n = max(int(x[2:]) for x in existing)
        return f"BF{max_n + 1:06d}"

    def list_bf_ids(self) -> List[str]:
        if not self.paths.reports_dir.exists():
            return []
        ids: List[str] = []
        for p in self.paths.reports_dir.glob("BF*.yml"):
            m = _BF_ID_RE.match(p.stem)
            if m:
                ids.append(p.stem)
        return sorted(ids)

    def report_path(self, bf_id: str) -> Path:
        if not _BF_ID_RE.match(bf_id):
            raise ValueError("invalid bf_id")
        return self.paths.reports_dir / f"{bf_id}.yml"

    def load_report(self, bf_id: str) -> Dict[str, Any]:
        path = self.report_path(bf_id)
        if not path.exists():
            raise FileNotFoundError(f"missing fixreport: {bf_id}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as e:
            _add_exc_note(e, f"while loading fixreport YAML: {path}")
            raise
        if not isinstance(raw, dict):
            raise ValueError(f"invalid fixreport file: {path}")
        fr = raw.get("fixreport")
        if isinstance(fr, dict):
            try:
                _normalize_fixreport_fields_inplace(fr)
                _validate_fixreport_doc_identity(
                    doc=raw, expected_bf_id=bf_id, path=path
                )
            except Exception as e:
                _add_exc_note(e, f"while normalizing fixreport fields: {path}")
                raise
        return raw

    def validate_report_doc(self, raw: Mapping[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        fr = raw.get("fixreport")
        if not isinstance(fr, Mapping):
            return False, ["missing fixreport mapping"]
        ok_fr, verrs = validate_fixreport_fields(self.vocab, fr)
        if not ok_fr:
            errors.extend(verrs)
        # Optional failure_modes block (separate schema).
        fm = raw.get("failure_modes", None)
        ok_fm, fm_errs = validate_failure_modes_block(
            catalog=self.failure_modes_catalog, block=fm
        )
        if not ok_fm:
            errors.extend(fm_errs)
        return (len(errors) == 0), errors

    def _index_row_from_doc(
        self, doc: Mapping[str, Any], *, bf_id: str, updated_mtime_ns: int
    ) -> Dict[str, Any]:
        fr = doc.get("fixreport")
        if not isinstance(fr, Mapping):
            raise ValueError("invalid fixreport doc: missing fixreport mapping")
        path = self.report_path(str(bf_id))
        _validate_fixreport_doc_identity(doc=doc, expected_bf_id=str(bf_id), path=path)
        nar = doc.get("narrative")
        if not isinstance(nar, Mapping):
            nar = {}

        return {
            "bf_id": str(bf_id),
            "ts_utc": fr.get("ts_utc"),
            "commit": fr.get("commit"),
            "repo_area": fr.get("repo_area"),
            "target_kind": fr.get("target_kind"),
            "target_path": fr.get("target_path"),
            "scope": fr.get("scope"),
            "mode": fr.get("mode"),
            "bug_type": fr.get("bug_type"),
            "symptom": fr.get("symptom"),
            "root_cause": fr.get("root_cause"),
            "failure_mode": fr.get("failure_mode"),
            "fix_type": fr.get("fix_type"),
            "impact_area": fr.get("impact_area"),
            "severity": fr.get("severity"),
            "verification": fr.get("verification"),
            "signature": fr.get("signature"),
            "archetypes": fr.get("archetypes")
            if isinstance(fr.get("archetypes"), list)
            else None,
            "tags": fr.get("tags") if isinstance(fr.get("tags"), list) else None,
            # Optional narrative preview (supports fast semantic recall without loading every BF YAML).
            "title": nar.get("title"),
            "summary": nar.get("summary"),
            "updated_mtime_ns": int(updated_mtime_ns),
        }

    def _rebuild_index_locked(self) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        for bf_id in self.list_bf_ids():
            try:
                raw = self.load_report(bf_id)
            except Exception as e:
                _add_exc_note(e, f"while rebuilding index (bf_id={bf_id})")
                raise
            fr = raw.get("fixreport")
            if not isinstance(fr, dict):
                continue
            updated_mtime_ns = 0
            try:
                updated_mtime_ns = int(self.report_path(bf_id).stat().st_mtime_ns)
            except Exception:
                updated_mtime_ns = 0
            try:
                row = self._index_row_from_doc(
                    raw, bf_id=bf_id, updated_mtime_ns=updated_mtime_ns
                )
            except Exception as e:
                _add_exc_note(e, f"while computing index row (bf_id={bf_id})")
                raise
            rows.append(row)

        payload: Dict[str, Any] = {"fixreports": rows}
        ensure_tessairact_meta_header(
            payload,
            kind="fixreports_index",
            area="ops",
            uid="FIXREPORT_INDEX",
            actor="fixreport",
            links=["script:agentic_tools/fixreport/store.py"],
            repo_root=self.repo_root,
            tool="fixreport",
        )
        _atomic_write_text(
            self.paths.index_path, yaml.safe_dump(payload, sort_keys=True)
        )
        return payload

    def rebuild_index(self) -> Dict[str, Any]:
        lock = FileLock(self.paths.lock_path)
        with lock:
            return self._rebuild_index_locked()

    def _sync_index_after_report_write_locked(
        self, bf_id: str, doc: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """
        Prefer incremental index update (avoid full scan+parse of all reports) when safe;
        fall back to full rebuild if index is invalid or appears out-of-sync.
        """
        updated_mtime_ns = 0
        try:
            updated_mtime_ns = int(self.report_path(bf_id).stat().st_mtime_ns)
        except Exception:
            updated_mtime_ns = 0

        idx = _load_yaml_or_default(self.paths.index_path, {"fixreports": []})
        rows = idx.get("fixreports") if isinstance(idx, dict) else None
        if not isinstance(idx, dict) or not isinstance(rows, list):
            return self._rebuild_index_locked()

        idx_ids: List[str] = []
        mtime_by_id: Dict[str, int] = {}
        for r in rows:
            if not isinstance(r, Mapping):
                return self._rebuild_index_locked()
            rid = r.get("bf_id")
            if not isinstance(rid, str) or not _BF_ID_RE.match(rid):
                return self._rebuild_index_locked()
            idx_ids.append(rid)
            mtn = r.get("updated_mtime_ns")
            if isinstance(mtn, int):
                mtime_by_id[rid] = int(mtn)

        fs_ids = self.list_bf_ids()
        fs_set = set(fs_ids)
        idx_set = set(idx_ids)

        if bf_id in fs_set:
            if not idx_set.issubset(fs_set):
                return self._rebuild_index_locked()
            missing = fs_set - idx_set
            if missing and missing != {bf_id}:
                return self._rebuild_index_locked()
        else:
            return self._rebuild_index_locked()

        for rid in idx_ids:
            if rid == bf_id:
                continue
            expected = mtime_by_id.get(rid)
            if expected is None:
                return self._rebuild_index_locked()
            try:
                actual = int(self.report_path(rid).stat().st_mtime_ns)
            except Exception:
                return self._rebuild_index_locked()
            if actual != int(expected):
                return self._rebuild_index_locked()

        try:
            new_row = self._index_row_from_doc(
                doc, bf_id=bf_id, updated_mtime_ns=updated_mtime_ns
            )
        except Exception:
            return self._rebuild_index_locked()

        out_rows: List[Dict[str, Any]] = []
        replaced = False
        for r in rows:
            if isinstance(r, dict) and r.get("bf_id") == bf_id:
                out_rows.append(dict(new_row))
                replaced = True
            elif isinstance(r, dict):
                out_rows.append(r)
            else:
                return self._rebuild_index_locked()
        if not replaced:
            out_rows.append(dict(new_row))

        out_rows.sort(key=lambda x: str(x.get("bf_id") or ""))

        payload: Dict[str, Any] = dict(idx)
        payload["fixreports"] = out_rows
        ensure_tessairact_meta_header(
            payload,
            kind="fixreports_index",
            area="ops",
            uid="FIXREPORT_INDEX",
            actor="fixreport",
            links=["script:agentic_tools/fixreport/store.py"],
            repo_root=self.repo_root,
            tool="fixreport",
        )
        _atomic_write_text(
            self.paths.index_path, yaml.safe_dump(payload, sort_keys=True)
        )
        return payload

    def _sync_index_after_report_write(
        self, bf_id: str, doc: Mapping[str, Any]
    ) -> Dict[str, Any]:
        lock = FileLock(self.paths.lock_path)
        with lock:
            return self._sync_index_after_report_write_locked(bf_id, doc)

    def _append_event_locked(self, event: Dict[str, Any]) -> None:
        events = _load_yaml_or_default(self.paths.events_path, {"events": []})
        if (
            not isinstance(events, dict)
            or "events" not in events
            or not isinstance(events["events"], list)
        ):
            events = {"events": []}
        events["events"].append(event)
        ensure_tessairact_meta_header(
            events,
            kind="fixreports_events",
            area="ops",
            uid="FIXREPORT_EVENTS",
            actor=str(event.get("actor") or "fixreport").strip() or "fixreport",
            links=["script:agentic_tools/fixreport/store.py"],
            repo_root=self.repo_root,
            tool="fixreport",
        )
        _atomic_write_text(
            self.paths.events_path, yaml.safe_dump(events, sort_keys=True)
        )

    def _append_event(self, event: Dict[str, Any]) -> None:
        lock = FileLock(self.paths.lock_path)
        with lock:
            self._append_event_locked(event)

    def _persist_post_commit_finding_sync_state_locked(
        self,
        *,
        bf_id: str,
        sync_state: Mapping[str, Any],
        actor: Optional[str],
    ) -> None:
        if not sync_state:
            return
        path = self.report_path(bf_id)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"invalid fixreport file during post-commit sync persistence: {path}"
            )
        _validate_fixreport_doc_identity(doc=raw, expected_bf_id=bf_id, path=path)
        merged_sync = (
            dict(raw.get(_POST_COMMIT_SYNC_KEY) or {})
            if isinstance(raw.get(_POST_COMMIT_SYNC_KEY), Mapping)
            else {}
        )
        for key, value in dict(sync_state).items():
            merged_sync[str(key)] = value
        raw[_POST_COMMIT_SYNC_KEY] = merged_sync
        tm = raw.get("tessairact_meta")
        if isinstance(tm, dict):
            updated_at_utc = (
                str(
                    (
                        (sync_state.get(_POST_COMMIT_FINDING_BACKLINKS_KEY) or {})
                        if isinstance(
                            sync_state.get(_POST_COMMIT_FINDING_BACKLINKS_KEY), Mapping
                        )
                        else {}
                    ).get("updated_at_utc")
                    or _utc_now_z()
                ).strip()
                or _utc_now_z()
            )
            tm["updated_at_utc"] = updated_at_utc
            tm["updated_by"] = str(actor or "fixreport").strip() or "fixreport"
        _atomic_write_text(
            path, yaml.safe_dump(raw, sort_keys=True, allow_unicode=True)
        )

        finding_sync = sync_state.get(_POST_COMMIT_FINDING_BACKLINKS_KEY)
        if isinstance(finding_sync, Mapping):
            status = str(finding_sync.get("status") or "").strip() or "updated"
            event_ts = (
                str(finding_sync.get("updated_at_utc") or _utc_now_z()).strip()
                or _utc_now_z()
            )
            self._append_event_locked(
                {
                    "id": f"EV-{bf_id}-finding-backlinks-{status}-{int(time.time() * 1000)}",
                    "timestamp": event_ts,
                    "kind": f"fixreport_finding_backlinks_{status}",
                    "bf_id": bf_id,
                    "actor": actor or "",
                    "finding_ids": list(finding_sync.get("finding_ids") or []),
                    "error_code": str(finding_sync.get("error_code") or "").strip(),
                }
            )

    def _persist_post_commit_finding_sync_state(
        self,
        *,
        bf_id: str,
        sync_state: Mapping[str, Any],
        actor: Optional[str],
    ) -> None:
        if not sync_state:
            return
        lock = FileLock(self.paths.lock_path)
        try:
            with lock:
                self._persist_post_commit_finding_sync_state_locked(
                    bf_id=bf_id,
                    sync_state=sync_state,
                    actor=actor,
                )
        except Exception as exc:
            logging.error(
                "Failed to persist fixreport post-commit finding sync state bf_id=%s state=%s error=%r",
                bf_id,
                dict(sync_state),
                exc,
            )

    def create(
        self,
        fixreport_fields: Dict[str, Any],
        *,
        narrative: Optional[Dict[str, Any]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        failure_modes: Optional[Dict[str, Any]] = None,
        actor: Optional[str] = None,
        explicit_finding_ids: Optional[Sequence[str]] = None,
        allow_extra_fields: bool = True,
    ) -> Tuple[str, Path]:
        self.init(force=False)

        explicit_findings_norm: List[str] = []
        path: Optional[Path] = None
        bf_id = ""
        explicit_link_finding_ids: List[str] = []
        lock = FileLock(self.paths.lock_path)
        with lock:
            # Refresh vocab if needed
            _ = self.vocab
            _ = self.failure_modes_catalog

            bf_id = str(fixreport_fields.get("bf_id") or self._next_bf_id())
            fixreport_fields = dict(fixreport_fields)
            explicit_link_finding_ids = _finding_ids_from_link_tokens(
                _split_csv_tokens(fixreport_fields.get("links"), field="links")
            )
            fixreport_fields["bf_id"] = bf_id
            fixreport_fields.setdefault("ts_utc", _utc_now_z())
            if isinstance(fixreport_fields.get("links"), list):
                fixreport_fields["links"] = list(fixreport_fields.get("links") or [])
            try:
                explicit_findings_norm = _prepare_fixreport_links_and_finding_targets(
                    repo_root=self.repo_root,
                    bf_id=bf_id,
                    fixreport_fields=fixreport_fields,
                    narrative=narrative,
                    evidence=evidence,
                    failure_modes=failure_modes,
                    explicit_finding_ids=explicit_finding_ids,
                    explicit_link_finding_ids=explicit_link_finding_ids,
                )
            except Exception as e:
                _add_exc_note(e, f"while normalizing fixreport fields (bf_id={bf_id})")
                raise ValueError("validation_failed: " + str(e)) from e
            _validate_create_archetypes(fixreport_fields)

            ok, errors = validate_fixreport_fields(self.vocab, fixreport_fields)
            if not ok:
                raise ValueError("validation_failed: " + "; ".join(errors))

            doc: Dict[str, Any] = {"fixreport": fixreport_fields}
            if narrative is not None:
                doc["narrative"] = narrative
            if evidence is not None:
                doc["evidence"] = evidence
            if failure_modes is not None:
                ok_fm, fm_errs = validate_failure_modes_block(
                    catalog=self.failure_modes_catalog,
                    block=failure_modes,
                )
                if not ok_fm:
                    raise ValueError(
                        "failure_modes_validation_failed: " + "; ".join(fm_errs)
                    )
                doc["failure_modes"] = failure_modes
            if actor:
                doc.setdefault("metadata", {})["created_by"] = actor
            _set_pending_post_commit_finding_sync(
                doc=doc,
                action="create",
                finding_ids=explicit_findings_norm,
            )

            actor2 = (actor or "").strip() or "fixreport"
            target_path = (
                str(fixreport_fields.get("target_path") or "").strip().strip("/")
            )
            links: List[str] = []
            if target_path:
                links.append(f"script:{target_path}")
            raw_links = fixreport_fields.get("links")
            if isinstance(raw_links, list):
                links.extend(
                    [str(x) for x in raw_links if x is not None and str(x).strip()]
                )
            ts_utc = str(fixreport_fields.get("ts_utc") or "").strip()
            ensure_tessairact_meta_header(
                doc,
                kind="fixreport",
                area=infer_area_from_path(target_path) if target_path else "ops",
                uid=bf_id,
                actor=actor2,
                title=str((narrative or {}).get("title") or "").strip(),
                links=links,
                repo_root=self.repo_root,
                tool="fixreport",
                created_at_utc=ts_utc,
                updated_at_utc=ts_utc,
            )

            path = self.report_path(bf_id)
            if path.exists():
                raise ValueError(f"bf_id already exists: {bf_id}")

            # Transaction snapshot for fail-fast + avoid partial commits.
            prev_index_text: Optional[str] = None
            prev_events_text: Optional[str] = None
            try:
                if self.paths.index_path.exists():
                    prev_index_text = self.paths.index_path.read_text(encoding="utf-8")
                if self.paths.events_path.exists():
                    prev_events_text = self.paths.events_path.read_text(
                        encoding="utf-8"
                    )
            except Exception as e:
                _add_exc_note(
                    e, f"while snapshotting index/events before create (bf_id={bf_id})"
                )
                raise

            text = yaml.safe_dump(doc, sort_keys=True, allow_unicode=True)
            _atomic_write_text(path, text)

            try:
                # Prefer incremental index update; fall back to deterministic full rebuild.
                self._sync_index_after_report_write_locked(bf_id, doc)
                self._append_event_locked(
                    {
                        "id": f"EV-{bf_id}-created",
                        "timestamp": _utc_now_z(),
                        "kind": "fixreport_created",
                        "bf_id": bf_id,
                        "actor": actor or "",
                    }
                )
            except Exception as e:
                rollback_errors: List[str] = []
                try:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                except Exception as re:
                    rollback_errors.append(f"remove_report_failed: {re!r}")
                try:
                    if prev_index_text is None:
                        try:
                            self.paths.index_path.unlink()
                        except FileNotFoundError:
                            pass
                    else:
                        _atomic_write_text(self.paths.index_path, prev_index_text)
                except Exception as re:
                    rollback_errors.append(f"restore_index_failed: {re!r}")
                try:
                    if prev_events_text is None:
                        try:
                            self.paths.events_path.unlink()
                        except FileNotFoundError:
                            pass
                    else:
                        _atomic_write_text(self.paths.events_path, prev_events_text)
                except Exception as re:
                    rollback_errors.append(f"restore_events_failed: {re!r}")

                if rollback_errors:
                    _add_exc_note(e, "rollback errors: " + "; ".join(rollback_errors))
                raise

        if path is None:
            raise RuntimeError(
                f"fixreport create missing report path after commit for {bf_id}"
            )
        sync_state = _sync_explicit_finding_links_after_commit(
            action="create",
            repo_root=self.repo_root,
            bf_id=bf_id,
            report_path=path,
            finding_ids=explicit_findings_norm,
            actor=actor,
        )
        self._persist_post_commit_finding_sync_state(
            bf_id=bf_id,
            sync_state=sync_state,
            actor=actor,
        )
        return bf_id, path

    def update(
        self,
        bf_id: str,
        *,
        set_fixreport: Optional[Mapping[str, Any]] = None,
        set_narrative: Optional[Mapping[str, Any]] = None,
        set_failure_modes: Optional[Mapping[str, Any]] = None,
        actor: Optional[str] = None,
        validate: bool = True,
        explicit_finding_ids: Optional[Sequence[str]] = None,
    ) -> Path:
        """
        Update an existing fixreport file.

        - set_fixreport: shallow set/overwrite keys inside `fixreport`.
          For list fields: if both old and new are lists, we union them.
        - set_narrative: shallow set/overwrite keys inside `narrative`.
        """
        self.init(force=False)

        explicit_findings_norm: List[str] = []
        path: Optional[Path] = None
        explicit_link_finding_ids: List[str] = []
        lock = FileLock(self.paths.lock_path)
        with lock:
            path = self.report_path(bf_id)
            if not path.exists():
                raise FileNotFoundError(f"missing fixreport: {bf_id}")

            # Transaction snapshot for fail-fast + avoid partial commits.
            prev_report_text: str
            prev_index_text: Optional[str] = None
            prev_events_text: Optional[str] = None
            try:
                prev_report_text = path.read_text(encoding="utf-8")
                if self.paths.index_path.exists():
                    prev_index_text = self.paths.index_path.read_text(encoding="utf-8")
                if self.paths.events_path.exists():
                    prev_events_text = self.paths.events_path.read_text(
                        encoding="utf-8"
                    )
            except Exception as e:
                _add_exc_note(
                    e,
                    f"while snapshotting report/index/events before update (bf_id={bf_id})",
                )
                raise

            try:
                raw = yaml.safe_load(prev_report_text) or {}
            except Exception as e:
                _add_exc_note(
                    e,
                    f"while parsing existing report YAML before update (bf_id={bf_id}, path={path})",
                )
                raise
            if not isinstance(raw, dict):
                raise ValueError(f"invalid fixreport file: {path}")

            fr = raw.get("fixreport")
            if not isinstance(fr, dict):
                fr = {}
            nar = raw.get("narrative")
            if not isinstance(nar, dict):
                nar = {}
            fm = raw.get("failure_modes")
            if not isinstance(fm, dict):
                fm = {}
            _validate_fixreport_doc_identity(doc=raw, expected_bf_id=bf_id, path=path)

            if set_fixreport:
                if "links" in set_fixreport:
                    explicit_link_finding_ids = _finding_ids_from_link_tokens(
                        _split_csv_tokens(set_fixreport.get("links"), field="links")
                    )
                if "bf_id" in set_fixreport:
                    raise ValueError("validation_failed: fixreport.bf_id is immutable")
                for k, v in dict(set_fixreport).items():
                    if v is None:
                        fr.pop(k, None)
                        continue
                    if isinstance(fr.get(k), list) and isinstance(v, list):
                        # Links are special: the CLI already applies link-add/link-remove semantics and script-anchor
                        # normalization. Here we must overwrite, otherwise invalid anchors become "sticky" and cannot be
                        # removed (breaking KG repair and link hygiene).
                        if str(k) == "links":
                            fr[k] = list(v)
                        else:
                            fr[k] = list(dict.fromkeys([*fr.get(k, []), *v]))
                    else:
                        fr[k] = v

            if set_narrative:
                for k, v in dict(set_narrative).items():
                    if v is None:
                        nar.pop(k, None)
                        continue
                    nar[k] = v

            try:
                explicit_findings_norm = _prepare_fixreport_links_and_finding_targets(
                    repo_root=self.repo_root,
                    bf_id=bf_id,
                    fixreport_fields=fr,
                    narrative=nar,
                    evidence=None,
                    failure_modes=fm,
                    explicit_finding_ids=explicit_finding_ids,
                    explicit_link_finding_ids=explicit_link_finding_ids,
                )
            except Exception as e:
                _add_exc_note(
                    e,
                    f"while normalizing fixreport fields during update (bf_id={bf_id})",
                )
                raise ValueError("validation_failed: " + str(e)) from e

            ok, errors = validate_fixreport_fields(self.vocab, fr)
            if not ok:
                if validate:
                    raise ValueError("validation_failed: " + "; ".join(errors))
                # Relaxed mode: keep structural validation but ignore vocab-enum membership errors.
                kept = [e for e in errors if "not in allowed set" not in str(e)]
                if kept:
                    raise ValueError("validation_failed: " + "; ".join(kept))

            if set_failure_modes is not None:
                # Shallow merge for `failure_modes`. Treat `checks` specially if present.
                for k, v in dict(set_failure_modes).items():
                    if v is None:
                        fm.pop(k, None)
                        continue
                    if (
                        k == "checks"
                        and isinstance(fm.get("checks"), dict)
                        and isinstance(v, Mapping)
                    ):
                        merged = dict(fm.get("checks") or {})
                        for ck, cv in dict(v).items():
                            if cv is None:
                                merged.pop(str(ck), None)
                            else:
                                merged[str(ck)] = cv
                        fm["checks"] = merged
                    else:
                        fm[k] = v
                ok_fm, fm_errs = validate_failure_modes_block(
                    catalog=self.failure_modes_catalog, block=fm
                )
                if not ok_fm:
                    raise ValueError(
                        "failure_modes_validation_failed: " + "; ".join(fm_errs)
                    )

            out: Dict[str, Any] = dict(raw)
            out["fixreport"] = fr
            if nar:
                out["narrative"] = nar
            else:
                out.pop("narrative", None)
            if fm:
                out["failure_modes"] = fm
            else:
                out.pop("failure_modes", None)
            _set_pending_post_commit_finding_sync(
                doc=out,
                action="update",
                finding_ids=explicit_findings_norm,
            )

            actor2 = (actor or "").strip() or "fixreport"
            target_path = str(fr.get("target_path") or "").strip().strip("/")
            links: List[str] = []
            if target_path:
                links.append(f"script:{target_path}")
            raw_links = fr.get("links")
            if isinstance(raw_links, list):
                links.extend(
                    [str(x) for x in raw_links if x is not None and str(x).strip()]
                )

            # Update timestamp meta deterministically (created_at_utc preserved by header helper if present).
            updated_at_utc = _utc_now_z()
            # For FixReports, the meta header `links` should reflect the current normalized fixreport.links + target_path,
            # not accumulate historical/stale tokens forever (which can re-introduce invalid script anchors into KG scans).
            tm = out.get("tessairact_meta")
            if isinstance(tm, dict):
                tm["links"] = []
            ensure_tessairact_meta_header(
                out,
                kind="fixreport",
                area=infer_area_from_path(target_path) if target_path else "ops",
                uid=bf_id,
                actor=actor2,
                title=str(nar.get("title") or "").strip(),
                links=links,
                repo_root=self.repo_root,
                tool="fixreport",
                updated_at_utc=updated_at_utc,
            )

            text = yaml.safe_dump(out, sort_keys=True, allow_unicode=True)
            _atomic_write_text(path, text)

            try:
                self._sync_index_after_report_write_locked(bf_id, out)
                self._append_event_locked(
                    {
                        "id": f"EV-{bf_id}-updated",
                        "timestamp": _utc_now_z(),
                        "kind": "fixreport_updated",
                        "bf_id": bf_id,
                        "actor": actor or "",
                    }
                )
            except Exception as e:
                rollback_errors: List[str] = []
                try:
                    _atomic_write_text(path, prev_report_text)
                except Exception as re:
                    rollback_errors.append(f"restore_report_failed: {re!r}")
                try:
                    if prev_index_text is None:
                        try:
                            self.paths.index_path.unlink()
                        except FileNotFoundError:
                            pass
                    else:
                        _atomic_write_text(self.paths.index_path, prev_index_text)
                except Exception as re:
                    rollback_errors.append(f"restore_index_failed: {re!r}")
                try:
                    if prev_events_text is None:
                        try:
                            self.paths.events_path.unlink()
                        except FileNotFoundError:
                            pass
                    else:
                        _atomic_write_text(self.paths.events_path, prev_events_text)
                except Exception as re:
                    rollback_errors.append(f"restore_events_failed: {re!r}")

                if rollback_errors:
                    _add_exc_note(e, "rollback errors: " + "; ".join(rollback_errors))
                raise

        if path is None:
            raise RuntimeError(
                f"fixreport update missing report path after commit for {bf_id}"
            )
        sync_state = _sync_explicit_finding_links_after_commit(
            action="update",
            repo_root=self.repo_root,
            bf_id=bf_id,
            report_path=path,
            finding_ids=explicit_findings_norm,
            actor=actor,
        )
        self._persist_post_commit_finding_sync_state(
            bf_id=bf_id,
            sync_state=sync_state,
            actor=actor,
        )
        return path

    def normalize_reports(self, *, apply: bool) -> Dict[str, int]:
        """
        Normalize list-token fields across all reports.

        If apply=False, performs a dry-run count of would-change reports.
        If apply=True, rewrites report YAMLs and rebuilds index.
        """
        self.init(force=False)
        lock = FileLock(self.paths.lock_path)
        with lock:
            changed = 0
            total = 0
            for bf_id in self.list_bf_ids():
                path = self.report_path(bf_id)
                try:
                    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                except Exception as e:
                    _add_exc_note(
                        e,
                        f"while loading report during normalize_reports (bf_id={bf_id}, path={path})",
                    )
                    raise
                if not isinstance(raw, dict):
                    continue
                fr = raw.get("fixreport")
                if not isinstance(fr, dict):
                    continue
                before = yaml.safe_dump(raw, sort_keys=True, allow_unicode=True)
                try:
                    _normalize_fixreport_fields_inplace(fr)
                except Exception as e:
                    _add_exc_note(
                        e,
                        f"while normalizing fields during normalize_reports (bf_id={bf_id}, path={path})",
                    )
                    raise
                after = yaml.safe_dump(raw, sort_keys=True, allow_unicode=True)
                total += 1
                if before != after:
                    changed += 1
                    if apply:
                        _atomic_write_text(path, after)
            if apply:
                self._rebuild_index_locked()
            return {"total": total, "changed": changed}

    def search(
        self,
        *,
        filters: Mapping[str, Any],
        fm_filters: Optional[Mapping[str, str]] = None,
        limit: int = 50,
        since_ts_utc: str = "",
        until_ts_utc: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Search over index.yml (preferred) or fall back to scanning report files.
        Filters are exact-match on known fields; tags/signature support containment.
        """
        self.init(force=False)
        idx = _load_yaml_or_default(self.paths.index_path, {"fixreports": []})
        rows = idx.get("fixreports") if isinstance(idx, dict) else None
        if not isinstance(rows, list):
            rows = self.rebuild_index().get("fixreports", [])
        else:
            needs_rebuild = False
            for row in rows:
                if not isinstance(row, Mapping):
                    needs_rebuild = True
                    break
                row_bf_id = str(row.get("bf_id") or "").strip()
                if not _BF_ID_RE.match(row_bf_id):
                    needs_rebuild = True
                    break
                updated_mtime_ns = row.get("updated_mtime_ns")
                if not isinstance(updated_mtime_ns, int):
                    needs_rebuild = True
                    break
                try:
                    actual_mtime_ns = int(
                        self.report_path(row_bf_id).stat().st_mtime_ns
                    )
                except Exception:
                    needs_rebuild = True
                    break
                if actual_mtime_ns != int(updated_mtime_ns):
                    needs_rebuild = True
                    break
            if needs_rebuild:
                rows = self.rebuild_index().get("fixreports", [])

        def _match(row: Mapping[str, Any]) -> bool:
            ts = str(row.get("ts_utc") or "")
            if since_ts_utc and ts and ts < since_ts_utc:
                return False
            if until_ts_utc and ts and ts > until_ts_utc:
                return False
            for k, v in filters.items():
                if v is None:
                    continue
                if k == "tags_contains":
                    tags = row.get("tags") or []
                    if not isinstance(tags, list) or v not in tags:
                        return False
                    continue
                if k == "archetypes_contains":
                    archetypes = row.get("archetypes") or []
                    if not isinstance(archetypes, list) or v not in archetypes:
                        return False
                    continue
                if row.get(k) != v:
                    return False
            return True

        out: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if _match(row):
                if fm_filters:
                    bf_id2 = str(row.get("bf_id") or "").strip()
                    if not bf_id2:
                        continue
                    try:
                        raw = self.load_report(bf_id2)
                    except Exception:
                        continue
                    fm = raw.get("failure_modes")
                    if not isinstance(fm, Mapping):
                        continue
                    checks = fm.get("checks")
                    if not isinstance(checks, Mapping):
                        continue
                    ok = True
                    for fm_id, expected in fm_filters.items():
                        if str(checks.get(fm_id) or "") != str(expected):
                            ok = False
                            break
                    if not ok:
                        continue
                out.append(row)
                if len(out) >= limit:
                    break
        return out
