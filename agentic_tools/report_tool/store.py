from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from agentic_tools._governance_locking import FileLockTimeout
from agentic_tools._governance_locking import GovernanceFileLock
from agentic_tools._governance_locking import resolve_lock_settings
from agentic_tools._store_layout import ensure_dirs, resolve_repo_path, resolve_store_dir
from common.links.tessairact_meta import ensure_tessairact_meta_header, extract_links_from_doc

from .types import LINK_TOKEN_RE, REPORT_ID_RE, REPORT_KINDS, REPORT_STATUSES, utc_now_z, validate_report_payload_quality


class ReportStoreError(RuntimeError):
    pass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    raw = raw.strip()
    if not raw:
        return int(default)
    try:
        return int(raw, 10)
    except Exception:
        return int(default)


class FileLock(GovernanceFileLock):
    """Report store lock backed by the shared governance lock primitive."""

    def __init__(self, path: Path, *, stale_after_sec: int = 120) -> None:
        settings = resolve_lock_settings(
            timeout_envs=("CAIA_REPORT_LOCK_TIMEOUT_SEC", "REPORT_LOCK_TIMEOUT"),
            timeout_default=10.0,
            poll_env="CAIA_REPORT_LOCK_POLL_SEC",
            poll_default=0.05,
            stale_env="CAIA_REPORT_LOCK_STALE_SEC",
            stale_default=int(stale_after_sec),
            heartbeat_env="CAIA_REPORT_LOCK_HEARTBEAT_SEC",
        )
        super().__init__(
            path=Path(path),
            timeout_sec=settings.timeout_sec,
            poll_sec=settings.poll_sec,
            stale_after_sec=settings.stale_after_sec,
            heartbeat_sec=settings.heartbeat_sec,
        )


@dataclass(frozen=True)
class ReportPaths:
    store_dir: Path
    reports_dir: Path
    templates_dir: Path
    index_path: Path
    events_path: Path
    lock_path: Path
    shipped_data_template_path: Path
    shipped_body_template_path: Path
    store_data_template_path: Path
    store_body_template_path: Path


def default_paths(repo_root: Path) -> ReportPaths:
    store_dir = resolve_store_dir(legacy_rel=None, canonical_rel="report", repo_root=repo_root)
    templates_dir = store_dir / "templates"
    shipped_data = repo_root / "agentic_tools" / "report_tool" / "templates" / "report_data_template.yml"
    shipped_body = repo_root / "agentic_tools" / "report_tool" / "templates" / "report_body_template.md"
    return ReportPaths(
        store_dir=store_dir,
        reports_dir=store_dir / "reports",
        templates_dir=templates_dir,
        index_path=store_dir / "index.yml",
        events_path=store_dir / "events.yml",
        lock_path=store_dir / ".lock",
        shipped_data_template_path=shipped_data,
        shipped_body_template_path=shipped_body,
        store_data_template_path=templates_dir / "report_data_template.yml",
        store_body_template_path=templates_dir / "report_body_template.md",
    )


def _yaml_safe_load(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # local import
    except Exception as exc:  # pragma: no cover
        raise ReportStoreError("PyYAML is required to read report YAML.") from exc
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ReportStoreError(f"Failed to parse YAML at {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ReportStoreError(f"YAML root must be a mapping at {path}")
    return dict(data)


def _yaml_dump(doc: Mapping[str, Any]) -> str:
    try:
        import yaml  # local import
    except Exception as exc:  # pragma: no cover
        raise ReportStoreError("PyYAML is required to write report YAML.") from exc
    return yaml.safe_dump(dict(doc), sort_keys=False, allow_unicode=True)


def _durable_writes_enabled() -> bool:
    return os.getenv("CAIA_REPORT_DURABLE_WRITES", "").strip().lower() in {"1", "true", "yes", "y", "on"}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure group-writable atomic writes, consistent with other stores (FixReports, Findings):
    # In deployments that use directory default ACLs, temp files created with 0o600 can cause
    # the ACL mask to strip effective permissions for named users/groups.
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
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _ensure_text_file(path: Path, text: str) -> None:
    if path.exists():
        return
    _atomic_write_text(path, text)


def _normalize_report_doc(doc: MutableMapping[str, Any], *, actor: str, report_id: str) -> Tuple[MutableMapping[str, Any], List[str]]:
    warnings: List[str] = []
    report = doc.get("report")
    if not isinstance(report, Mapping):
        raise ReportStoreError("report doc must contain top-level 'report' mapping")
    rep = dict(report)
    doc["report"] = rep

    rid = str(rep.get("report_id") or "").strip() or str(report_id or "").strip()
    if not rid:
        raise ReportStoreError("report.report_id is required")
    if not REPORT_ID_RE.fullmatch(rid):
        raise ReportStoreError(f"invalid report_id: {rid!r}")
    rep["report_id"] = rid

    title = str(rep.get("title") or "").strip()
    if not title:
        raise ReportStoreError("report.title is required")

    kind = str(rep.get("kind") or "").strip().lower()
    if kind not in set(REPORT_KINDS):
        raise ReportStoreError(f"report.kind must be one of {list(REPORT_KINDS)}")
    rep["kind"] = kind

    status = str(rep.get("status") or "").strip().lower() or "draft"
    if status not in set(REPORT_STATUSES):
        raise ReportStoreError(f"report.status must be one of {list(REPORT_STATUSES)}")
    rep["status"] = status

    # Ensure timestamps.
    created_at = str(rep.get("created_at_utc") or "").strip() or utc_now_z()
    rep["created_at_utc"] = created_at
    rep["created_by"] = str(rep.get("created_by") or "").strip() or str(actor or "").strip() or "unknown"

    # Tessairact meta header.
    links, links_warn = extract_links_from_doc(doc)
    # Prevent historical link pollution: when a report already carries a meta header,
    # rewrite its links to the parsed canonical set so `ensure_tessairact_meta_header`
    # does not merge in invalid legacy items.
    tm = doc.get("tessairact_meta")
    if isinstance(tm, Mapping) and not isinstance(tm, MutableMapping):
        tm = dict(tm)
        doc["tessairact_meta"] = tm
    if isinstance(tm, MutableMapping):
        tm["links"] = list(links)
    doc2, hdr_warn = ensure_tessairact_meta_header(
        doc,
        kind="report",
        area="report",
        uid=rid,
        actor=str(actor or "").strip() or rep["created_by"],
        title=title,
        links=links,
        repo_root=None,
        tool="report",
    )
    warnings.extend(list(links_warn or []))
    warnings.extend(hdr_warn or [])
    return doc2, warnings


def _report_yaml_path(paths: ReportPaths, report_id: str) -> Path:
    return paths.reports_dir / f"{report_id}.yml"


def _read_index(paths: ReportPaths) -> Dict[str, Any]:
    if not paths.index_path.exists():
        return {}
    return _yaml_safe_load(paths.index_path)


def _write_index(paths: ReportPaths, doc: Mapping[str, Any]) -> None:
    _atomic_write_text(paths.index_path, _yaml_dump(doc))


def _read_events(paths: ReportPaths) -> Dict[str, Any]:
    if not paths.events_path.exists():
        return {}
    return _yaml_safe_load(paths.events_path)


def _write_events(paths: ReportPaths, doc: Mapping[str, Any]) -> None:
    _atomic_write_text(paths.events_path, _yaml_dump(doc))


def _canonicalize_summary_entry(doc: Mapping[str, Any], *, yaml_rel_path: str) -> Dict[str, Any]:
    report = doc.get("report") if isinstance(doc, Mapping) else None
    meta = doc.get("tessairact_meta") if isinstance(doc, Mapping) else None
    rep: Dict[str, Any] = dict(report) if isinstance(report, Mapping) else {}
    links, _links_warn = extract_links_from_doc(doc)
    title = str(rep.get("title") or "").strip()
    report_id = str(rep.get("report_id") or "").strip()
    created_at = str(rep.get("created_at_utc") or "").strip()
    status = str(rep.get("status") or "").strip().lower()
    kind = str(rep.get("kind") or "").strip().lower()
    tags = rep.get("tags") if isinstance(rep.get("tags"), list) else []
    content = rep.get("content") if isinstance(rep.get("content"), Mapping) else {}
    md_rel_path = str(content.get("md_rel_path") or "").strip()
    assets_rel_paths = content.get("assets_rel_paths") if isinstance(content.get("assets_rel_paths"), list) else []
    out: Dict[str, Any] = {
        "report_id": report_id,
        "created_at_utc": created_at,
        "status": status,
        "kind": kind,
        "title": title,
        "tags": [str(x).strip() for x in tags if str(x).strip()],
        "links": list(links),
        "yaml_rel_path": str(yaml_rel_path),
        "md_rel_path": md_rel_path,
        "assets_rel_paths": [str(x).strip() for x in assets_rel_paths if str(x).strip()],
    }
    if isinstance(meta, Mapping) and meta.get("updated_at_utc"):
        out["updated_at_utc"] = str(meta.get("updated_at_utc"))
    return out


def _ensure_index_skeleton(repo_root: Path, *, actor: str) -> Dict[str, Any]:
    doc: Dict[str, Any] = {"reports": []}
    ensure_tessairact_meta_header(
        doc,
        kind="report_index",
        area="report",
        uid="report_index",
        actor=str(actor or "").strip() or "unknown",
        title="Report index",
        links=[],
        repo_root=None,
        tool="report",
    )
    return doc


def _ensure_events_skeleton(repo_root: Path, *, actor: str) -> Dict[str, Any]:
    doc: Dict[str, Any] = {"events": []}
    ensure_tessairact_meta_header(
        doc,
        kind="report_events",
        area="report",
        uid="report_events",
        actor=str(actor or "").strip() or "unknown",
        title="Report events",
        links=[],
        repo_root=None,
        tool="report",
    )
    return doc


class ReportStore:
    def __init__(self, *, repo_root: Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.paths = default_paths(self.repo_root)

    def ensure_initialized(self, *, actor: str) -> Dict[str, Any]:
        ensure_dirs([self.paths.store_dir, self.paths.reports_dir, self.paths.templates_dir])

        # Copy templates into the store (non-destructive).
        if self.paths.shipped_data_template_path.is_file():
            if not self.paths.store_data_template_path.exists():
                shutil.copyfile(self.paths.shipped_data_template_path, self.paths.store_data_template_path)
        if self.paths.shipped_body_template_path.is_file():
            if not self.paths.store_body_template_path.exists():
                shutil.copyfile(self.paths.shipped_body_template_path, self.paths.store_body_template_path)

        lock = FileLock(self.paths.lock_path)
        with lock:
            if not self.paths.index_path.exists():
                _write_index(self.paths, _ensure_index_skeleton(self.repo_root, actor=actor))
            if not self.paths.events_path.exists():
                _write_events(self.paths, _ensure_events_skeleton(self.repo_root, actor=actor))

        return {
            "store_dir": str(self.paths.store_dir),
            "reports_dir": str(self.paths.reports_dir),
            "templates_dir": str(self.paths.templates_dir),
            "index_path": str(self.paths.index_path),
            "events_path": str(self.paths.events_path),
        }

    def list_report_ids(self) -> List[str]:
        if not self.paths.reports_dir.exists():
            return []
        out: List[str] = []
        for p in sorted(self.paths.reports_dir.glob("RPT*.yml"), key=lambda x: x.name):
            rid = p.stem
            if REPORT_ID_RE.fullmatch(rid):
                out.append(rid)
        return out

    def load_report(self, report_id: str) -> Tuple[Dict[str, Any], Path]:
        rid = str(report_id or "").strip()
        if not REPORT_ID_RE.fullmatch(rid):
            raise ReportStoreError(f"invalid report id: {rid!r}")
        p = _report_yaml_path(self.paths, rid)
        if not p.exists():
            raise ReportStoreError(f"report not found: {rid}")
        return _yaml_safe_load(p), p

    def write_report_doc(self, report_id: str, doc: MutableMapping[str, Any], *, actor: str) -> Tuple[Path, List[str]]:
        rid = str(report_id or "").strip()
        if not REPORT_ID_RE.fullmatch(rid):
            raise ReportStoreError(f"invalid report id: {rid!r}")

        lock = FileLock(self.paths.lock_path)
        with lock:
            doc2, warnings = _normalize_report_doc(doc, actor=actor, report_id=rid)
            out_path = _report_yaml_path(self.paths, rid)
            _atomic_write_text(out_path, _yaml_dump(doc2))

            # Update index.
            idx = _read_index(self.paths)
            reports = idx.get("reports")
            if not isinstance(reports, list):
                idx = _ensure_index_skeleton(self.repo_root, actor=actor)
                reports = idx.get("reports")
            assert isinstance(reports, list)
            yaml_rel = f"artifacts/stores/report/reports/{rid}.yml"
            entry = _canonicalize_summary_entry(doc2, yaml_rel_path=yaml_rel)
            reports = [x for x in reports if not (isinstance(x, Mapping) and str(x.get("report_id") or "").strip() == rid)]
            reports.append(entry)
            # Sort newest-first.
            reports.sort(key=lambda x: (str(x.get("created_at_utc") or ""), str(x.get("report_id") or "")), reverse=True)
            idx["reports"] = reports
            _write_index(self.paths, idx)

            return out_path, warnings

    def append_event(self, event: Mapping[str, Any], *, actor: str) -> None:
        lock = FileLock(self.paths.lock_path)
        with lock:
            ev = _read_events(self.paths)
            events = ev.get("events")
            if not isinstance(events, list):
                ev = _ensure_events_skeleton(self.repo_root, actor=actor)
                events = ev.get("events")
            assert isinstance(events, list)
            events.append(dict(event))
            # Keep bounded size (operators can archive if needed).
            max_events = max(1000, _env_int("REPORT_STORE_MAX_EVENTS", 20_000))
            if len(events) > int(max_events):
                events = events[-int(max_events) :]
            ev["events"] = events
            _write_events(self.paths, ev)

    def rebuild_index(self, *, actor: str) -> Dict[str, Any]:
        lock = FileLock(self.paths.lock_path)
        with lock:
            idx = _ensure_index_skeleton(self.repo_root, actor=actor)
            reports: List[Dict[str, Any]] = []
            for rid in self.list_report_ids():
                p = _report_yaml_path(self.paths, rid)
                try:
                    doc = _yaml_safe_load(p)
                except Exception:
                    continue
                yaml_rel = f"artifacts/stores/report/reports/{rid}.yml"
                reports.append(_canonicalize_summary_entry(doc, yaml_rel_path=yaml_rel))
            reports.sort(key=lambda x: (str(x.get("created_at_utc") or ""), str(x.get("report_id") or "")), reverse=True)
            idx["reports"] = reports
            _write_index(self.paths, idx)
            return idx

    def lint_report(self, report_id: str) -> Tuple[bool, List[str], Dict[str, Any]]:
        rid = str(report_id or "").strip()
        if not REPORT_ID_RE.fullmatch(rid):
            return (False, [f"invalid report id: {rid!r}"], {})
        doc, _ = self.load_report(rid)
        errs: List[str] = []

        report = doc.get("report")
        if not isinstance(report, Mapping):
            errs.append("missing report mapping")
            return (False, errs, doc)

        title = str(report.get("title") or "").strip()
        if not title:
            errs.append("report.title is required")

        kind = str(report.get("kind") or "").strip().lower()
        if kind not in set(REPORT_KINDS):
            errs.append(f"report.kind invalid: {kind!r}")

        status = str(report.get("status") or "").strip().lower()
        if status not in set(REPORT_STATUSES):
            errs.append(f"report.status invalid: {status!r}")

        errs.extend(validate_report_payload_quality(report))

        # Links
        links, _links_warn = extract_links_from_doc(doc)
        if not links:
            errs.append("missing tessairact_meta.links (at least one anchor required)")
        if not any(str(x).startswith(("script:", "task:", "hypothesis:", "service_unit:")) for x in links):
            errs.append("tessairact_meta.links must include at least one anchor (script:/task:/hypothesis:/service_unit:)")
        bad_links = [x for x in links if not LINK_TOKEN_RE.fullmatch(x)]
        if bad_links:
            errs.append(f"invalid link tokens: {bad_links[:5]}")

        # Content paths
        content = report.get("content")
        if isinstance(content, Mapping):
            md_rel = str(content.get("md_rel_path") or "").strip()
            if md_rel:
                md_abs = (self.repo_root / md_rel).resolve()
                if not md_abs.is_file():
                    errs.append(f"missing md file: {md_rel}")
            assets = content.get("assets_rel_paths")
            if isinstance(assets, list):
                for a in assets[:200]:
                    ar = str(a or "").strip()
                    if not ar:
                        continue
                    aa = (self.repo_root / ar).resolve()
                    if not aa.exists():
                        errs.append(f"missing asset: {ar}")

        return (len(errs) == 0, errs, doc)
