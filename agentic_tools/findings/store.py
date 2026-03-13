"""
Contract
- Purpose: Store + validate Findings (pre-fix issue notes) under a repo-local store.
- Inputs: Python dicts representing Finding YAML documents.
- Outputs: YAML files + index.yml + events.yml (append-only).
- Invariants: Finding IDs are unique; index is rebuildable from reports.
- Error semantics: Raises on validation or IO failures; caller (CLI) renders JSON.
- External boundaries: Filesystem under repo_root/artifacts/stores/findings (legacy repo_root/state/findings is rejected unless CAIA_STORES_ALLOW_LEGACY=1).
"""

##REFACTOR: 2026-01-16##

from __future__ import annotations

import os
import re
import socket
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml

from common.links.tessairact_meta import ensure_tessairact_meta_header, extract_link_tokens_from_texts, infer_area_from_path
from agentic_tools._governance_locking import GovernanceFileLock, resolve_lock_settings
from agentic_tools._store_layout import resolve_store_dir


LINK_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]{0,32}:[A-Za-z0-9._:/@+-]{1,256}$")
_FND_ID_RE = re.compile(r"^FND(?:[0-9]{6}|[0-9]{8}T[0-9]{6}Z(?:_[A-Za-z0-9]{1,16})?)$")
_REPORT_FILE_RE = re.compile(r"^(FND[0-9A-Za-z_]{6,40})\.ya?ml$")

_ISO_Z_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")

_ALLOWED_SEVERITIES = {"p0", "p1", "p2", "p3"}
_ALLOWED_STATUSES = {"open", "triaged", "in_progress", "blocked", "resolved", "wont_fix", "duplicate"}
_ALLOWED_SCOPES = {"trading", "analytics", "infra", "codebase", "ops", "portal", "research"}
_ALLOWED_MODES = {"live", "shadow", "backtest", "replay", "backfill", "train", "eval", "dev", "test"}
_ALLOWED_REPO_AREAS = {
    "services",
    "scripts",
    "tests",
    "clickhouse",
    "systemd",
    "config",
    "contracts",
    "analytics",
    "migration",
    "consumer",
    "common",
    "infra",
    "docs",
}

_ALLOWED_TARGET_KINDS = {
    "service",
    "script",
    "test",
    "sql",
    "proto",
    "systemd",
    "config",
    "lib",
    "contract",
    "doc",
    "topic",
    "table",
    "other",
}

# Public (frozen) views for tooling/UI/schema output. Keep these as the
# single source of truth for allowed values across CLI + validators.
ALLOWED_SEVERITIES = frozenset(_ALLOWED_SEVERITIES)
ALLOWED_STATUSES = frozenset(_ALLOWED_STATUSES)
ALLOWED_SCOPES = frozenset(_ALLOWED_SCOPES)
ALLOWED_MODES = frozenset(_ALLOWED_MODES)
ALLOWED_REPO_AREAS = frozenset(_ALLOWED_REPO_AREAS)
ALLOWED_TARGET_KINDS = frozenset(_ALLOWED_TARGET_KINDS)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


_MAX_YAML_BYTES_DEFAULT = 10 * 1024 * 1024  # generous default to avoid breaking legitimate repos
_MAX_YAML_ALIASES_DEFAULT = 200
_MAX_YAML_DEPTH_DEFAULT = 2000

_MAX_YAML_BYTES = _env_int("CAIA_FINDINGS_MAX_YAML_BYTES", _MAX_YAML_BYTES_DEFAULT)
_MAX_YAML_ALIASES = _env_int("CAIA_FINDINGS_MAX_YAML_ALIASES", _MAX_YAML_ALIASES_DEFAULT)
_MAX_YAML_DEPTH = _env_int("CAIA_FINDINGS_MAX_YAML_DEPTH", _MAX_YAML_DEPTH_DEFAULT)


_PREFERRED_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_PREFERRED_SAFE_DUMPER = getattr(yaml, "CSafeDumper", yaml.SafeDumper)


class _LimitedSafeLoader(_PREFERRED_SAFE_LOADER):
    """
    SafeLoader with coarse DoS hardening.

    - Limits number of alias events (anchors/references).
    - Limits compose recursion depth.

    Limits can be disabled by setting env vars to 0 or negative.
    """

    def __init__(self, stream) -> None:
        super().__init__(stream)
        self._alias_count = 0
        self._depth = 0

    def compose_node(self, parent, index):  # type: ignore[override]
        max_aliases = int(_MAX_YAML_ALIASES)
        max_depth = int(_MAX_YAML_DEPTH)

        if max_depth > 0:
            self._depth += 1
            if self._depth > max_depth:
                raise yaml.YAMLError(f"yaml_max_depth_exceeded: {self._depth} > {max_depth}")

        try:
            if max_aliases > 0 and self.check_event(yaml.events.AliasEvent):
                self._alias_count += 1
                if self._alias_count > max_aliases:
                    raise yaml.YAMLError(f"yaml_alias_limit_exceeded: {self._alias_count} > {max_aliases}")
            return super().compose_node(parent, index)
        finally:
            if max_depth > 0:
                self._depth -= 1


def _read_text_limited(path: Path) -> str:
    max_bytes = int(_MAX_YAML_BYTES)
    if max_bytes > 0:
        try:
            sz = int(path.stat().st_size)
        except Exception:
            sz = -1
        if sz >= 0 and sz > max_bytes:
            raise ValueError(f"yaml_too_large: {path} ({sz} bytes > {max_bytes})")
    return path.read_text(encoding="utf-8")


def _yaml_safe_load_text(text: str) -> Any:
    # If all limits are disabled, use yaml.safe_load directly.
    if int(_MAX_YAML_ALIASES) <= 0 and int(_MAX_YAML_DEPTH) <= 0:
        return yaml.load(text, Loader=_PREFERRED_SAFE_LOADER)
    return yaml.load(text, Loader=_LimitedSafeLoader)


def _yaml_safe_load_path(path: Path) -> Any:
    return _yaml_safe_load_text(_read_text_limited(path))


def _yaml_dump_text(data: Any, *, sort_keys: bool, allow_unicode: bool) -> str:
    return yaml.dump(
        data,
        Dumper=_PREFERRED_SAFE_DUMPER,
        sort_keys=sort_keys,
        allow_unicode=allow_unicode,
    )


def _yaml_dump_stream(stream, data: Any, *, sort_keys: bool, allow_unicode: bool) -> None:
    yaml.dump(
        data,
        stream,
        Dumper=_PREFERRED_SAFE_DUMPER,
        sort_keys=sort_keys,
        allow_unicode=allow_unicode,
    )


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fsync_dir_best_effort(dir_path: Path) -> None:
    # Best-effort directory fsync for crash-durability on POSIX.
    if os.name != "posix":
        return
    try:
        flags = os.O_RDONLY
        odir = getattr(os, "O_DIRECTORY", 0)
        fd = os.open(str(dir_path), flags | odir)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass


def _atomic_write_text(path: Path, text: str) -> None:
    """
    Atomic file replace with a unique temp file in the same directory.

    Notes:
    - Uses os.replace for atomicity.
    - Best-effort fsync for crash-durability (can be disabled via env).
    """
    do_fsync = _env_bool("CAIA_FINDINGS_ATOMIC_FSYNC", True)

    path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve current mode if the file exists.
    mode: Optional[int] = None
    try:
        st_mode = path.stat().st_mode
        mode = int(st_mode) & 0o777
    except FileNotFoundError:
        mode = None
    except Exception:
        mode = None
    # If the file does not exist yet, NamedTemporaryFile defaults to 0o600.
    # In repos that rely on directory default ACLs for additional writers
    # (e.g. portal/service user), a 0o600 file can result in an ACL mask of ---,
    # stripping effective permissions and causing PermissionError on updates.
    #
    # Keep "other" at 0 (no world access) but ensure group has rw by default.
    if mode is None:
        mode = 0o660
    elif mode == 0o600:
        mode = 0o660

    tmp_path: Optional[Path] = None
    try:
        prefix = f".{path.name}."
        suffix = f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=str(path.parent),
            prefix=prefix,
            suffix=suffix,
        ) as fp:
            tmp_path = Path(fp.name)
            fp.write(text)
            fp.flush()
            if do_fsync:
                try:
                    os.fsync(fp.fileno())
                except Exception:
                    pass
        if mode is not None:
            try:
                os.chmod(str(tmp_path), mode)
            except Exception:
                pass
        os.replace(str(tmp_path), str(path))
        if do_fsync:
            _fsync_dir_best_effort(path.parent)
    finally:
        if tmp_path is not None:
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _atomic_write_yaml(path: Path, data: Any, *, sort_keys: bool, allow_unicode: bool) -> None:
    """
    Atomic YAML dump without building the entire YAML string in memory.
    """
    do_fsync = _env_bool("CAIA_FINDINGS_ATOMIC_FSYNC", True)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode: Optional[int] = None
    try:
        st_mode = path.stat().st_mode
        mode = int(st_mode) & 0o777
    except FileNotFoundError:
        mode = None
    except Exception:
        mode = None
    if mode is None:
        mode = 0o660
    elif mode == 0o600:
        mode = 0o660

    tmp_path: Optional[Path] = None
    try:
        prefix = f".{path.name}."
        suffix = f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=str(path.parent),
            prefix=prefix,
            suffix=suffix,
        ) as fp:
            tmp_path = Path(fp.name)
            _yaml_dump_stream(fp, data, sort_keys=sort_keys, allow_unicode=allow_unicode)
            fp.flush()
            if do_fsync:
                try:
                    os.fsync(fp.fileno())
                except Exception:
                    pass
        if mode is not None:
            try:
                os.chmod(str(tmp_path), mode)
            except Exception:
                pass
        os.replace(str(tmp_path), str(path))
        if do_fsync:
            _fsync_dir_best_effort(path.parent)
    finally:
        if tmp_path is not None:
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _load_yaml_or_default(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    raw = _yaml_safe_load_path(path)
    return default if raw is None else raw


def _get_hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we might not have permission.
        return True
    except Exception:
        # On some platforms os.kill might not be fully supported; best-effort.
        return False


def _parse_lock_metadata(text: str) -> Tuple[Optional[str], Optional[int]]:
    host: Optional[str] = None
    pid: Optional[int] = None
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if k == "host":
            host = v or None
        elif k == "pid":
            try:
                pid = int(v)
            except Exception:
                pid = None
    return host, pid


class FileLock(GovernanceFileLock):
    """Findings store lock backed by the shared governance lock primitive."""

    def __init__(self, path: Path, *, stale_after_sec: int = 120) -> None:
        settings = resolve_lock_settings(
            timeout_envs=("CAIA_FINDINGS_LOCK_TIMEOUT_SEC", "FINDINGS_LOCK_TIMEOUT"),
            timeout_default=30.0,
            poll_env="CAIA_FINDINGS_LOCK_POLL_SEC",
            poll_default=0.05,
            stale_env="CAIA_FINDINGS_LOCK_STALE_SEC",
            stale_default=int(stale_after_sec),
            heartbeat_env="CAIA_FINDINGS_LOCK_HEARTBEAT_SEC",
        )
        super().__init__(
            path=path,
            timeout_sec=settings.timeout_sec,
            poll_sec=settings.poll_sec,
            stale_after_sec=settings.stale_after_sec,
            heartbeat_sec=settings.heartbeat_sec,
        )


@dataclass(frozen=True)
class FindingPaths:
    state_dir: Path
    reports_dir: Path
    templates_dir: Path
    index_path: Path
    events_path: Path
    lock_path: Path
    shipped_template_path: Path
    state_template_path: Path


def default_paths(repo_root: Path) -> FindingPaths:
    state_dir = resolve_store_dir(
        legacy_rel="state/findings",
        canonical_rel="findings",
        repo_root=repo_root,
    )
    templates_dir = state_dir / "templates"
    shipped = repo_root / "agentic_tools" / "findings" / "templates" / "bug_finding_template.yml"
    return FindingPaths(
        state_dir=state_dir,
        reports_dir=state_dir / "reports",
        templates_dir=templates_dir,
        index_path=state_dir / "index.yml",
        events_path=state_dir / "events.yml",
        lock_path=state_dir / ".lock",
        shipped_template_path=shipped,
        state_template_path=templates_dir / "bug_finding_template.yml",
    )


def is_safe_finding_id(finding_id: str) -> bool:
    fid = (finding_id or "").strip()
    return bool(_FND_ID_RE.fullmatch(fid))


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None and str(x).strip()]
    s = str(value).strip()
    return [s] if s else []


def _validate_links(links: Any) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    normalized: List[str] = []
    if links is None:
        return normalized, errors
    if not isinstance(links, list):
        return [], ["links must be a list of strings"]
    for i, item in enumerate(links):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"links[{i}] must be a non-empty string")
            continue
        tok = item.strip()
        if not LINK_TOKEN_RE.fullmatch(tok):
            errors.append(f"links[{i}] invalid token: {tok!r}")
            continue
        normalized.append(tok)
    # De-dup preserving order.
    seen: set[str] = set()
    out: List[str] = []
    for t in normalized:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out, errors


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def _validate_report_doc(doc: Any, *, path: Optional[Path] = None) -> Tuple[List[str], List[str]]:
    """
    Returns (errors, warnings).

    Keep warnings non-fatal so early-stage findings can be recorded.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(doc, Mapping):
        return [f"YAML root must be a mapping/object ({path})"], []

    f_raw = doc.get("finding")
    if not isinstance(f_raw, Mapping):
        return [f"missing or invalid finding mapping ({path})"], []

    fid = _safe_str(f_raw.get("id")).strip()
    if not fid:
        errors.append("finding.id is required")
    elif not is_safe_finding_id(fid):
        errors.append("finding.id must match ^FND(\\d{6}|YYYYMMDDTHHMMSSZ[_suffix])$")

    if path is not None:
        stem = path.stem
        if fid and stem != fid:
            errors.append(f"file name mismatch: {path.name} (stem={stem}) != finding.id={fid}")

    status = _safe_str(f_raw.get("status")).strip()
    if not status:
        errors.append("finding.status is required")
    elif status not in _ALLOWED_STATUSES:
        errors.append(f"finding.status invalid: {status!r}")

    addressed = f_raw.get("addressed")
    addressed_bool = False
    if addressed is not None:
        if not isinstance(addressed, bool):
            errors.append("finding.addressed must be a boolean when present")
        else:
            addressed_bool = bool(addressed)

    addressed_at = _safe_str(f_raw.get("addressed_at_utc")).strip()
    if addressed_at:
        if not _ISO_Z_RE.fullmatch(addressed_at):
            errors.append("finding.addressed_at_utc must be ISO8601 UTC '...Z' when set")
        if not addressed_bool:
            errors.append("finding.addressed_at_utc is set but finding.addressed is not true")
    elif addressed_bool:
        warnings.append("finding.addressed=true but finding.addressed_at_utc is empty (recommended to set)")

    sev = _safe_str(f_raw.get("severity")).strip().lower()
    if not sev:
        errors.append("finding.severity is required")
    elif sev not in _ALLOWED_SEVERITIES:
        errors.append(f"finding.severity invalid: {sev!r}")

    scope = _safe_str(f_raw.get("scope")).strip()
    if not scope:
        errors.append("finding.scope is required")
    elif scope not in _ALLOWED_SCOPES:
        errors.append(f"finding.scope invalid: {scope!r}")

    mode = _safe_str(f_raw.get("mode")).strip()
    if not mode:
        errors.append("finding.mode is required")
    elif mode not in _ALLOWED_MODES:
        errors.append(f"finding.mode invalid: {mode!r}")

    repo_area = _safe_str(f_raw.get("repo_area")).strip()
    if not repo_area:
        errors.append("finding.repo_area is required")
    elif repo_area not in _ALLOWED_REPO_AREAS:
        errors.append(f"finding.repo_area invalid: {repo_area!r}")

    title = _safe_str(f_raw.get("title")).strip()
    if not title:
        errors.append("finding.title is required")
    summary = _safe_str(f_raw.get("summary")).strip()
    if not summary:
        errors.append("finding.summary is required")

    target = f_raw.get("target")
    if not isinstance(target, Mapping):
        errors.append("finding.target must be a mapping")
    else:
        tk = _safe_str(target.get("kind")).strip()
        if not tk:
            errors.append("finding.target.kind is required")
        elif tk not in _ALLOWED_TARGET_KINDS:
            errors.append(f"finding.target.kind invalid: {tk!r}")
        tp = _safe_str(target.get("path")).strip()
        if tk in {"script", "service", "test", "lib", "config", "contract", "doc", "proto", "sql", "systemd"}:
            if not tp:
                errors.append(f"finding.target.path is required for target.kind={tk!r}")

    prov = f_raw.get("provenance")
    if not isinstance(prov, Mapping):
        errors.append("finding.provenance must be a mapping")
    else:
        ts = _safe_str(prov.get("discovered_at_utc")).strip()
        if not ts:
            errors.append("finding.provenance.discovered_at_utc is required")
        elif not _ISO_Z_RE.fullmatch(ts):
            errors.append("finding.provenance.discovered_at_utc must be ISO8601 UTC '...Z'")
        auth = _safe_str(prov.get("author")).strip()
        if not auth:
            warnings.append("finding.provenance.author is empty (recommended to set)")

    links_norm, link_errs = _validate_links(doc.get("links"))
    errors.extend(link_errs)
    if links_norm:
        # Recommend a script anchor for code-ish targets.
        has_script = any(t.startswith("script:") for t in links_norm)
        if not has_script and isinstance(target, Mapping) and str(target.get("kind") or "").strip() in {
            "script",
            "service",
            "test",
            "lib",
            "config",
            "contract",
            "doc",
            "proto",
            "sql",
            "systemd",
        }:
            warnings.append("missing script:<repo_path> link (recommended for Meta graph anchoring)")

    return errors, warnings


def _reordered_mapping(raw: Any, order: Sequence[str]) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    src = dict(raw)
    out: Dict[str, Any] = {}
    for k in order:
        if k in src:
            out[k] = src.pop(k)
    # Preserve any unknown keys (append at end, stable sorted for determinism).
    for k in sorted(src.keys()):
        out[k] = src[k]
    return out


def _canonical_fixreport_links_for_finding_doc(doc: Mapping[str, Any]) -> List[str]:
    allowed: set[str] = set()
    finding = doc.get("finding") if isinstance(doc.get("finding"), Mapping) else {}
    related = (
        finding.get("related")
        if isinstance(finding, Mapping) and isinstance(finding.get("related"), Mapping)
        else {}
    )
    for raw_fix_id in _as_list(related.get("fixreport_ids")):
        fix_id = str(raw_fix_id or "").strip()
        if fix_id:
            allowed.add(f"fixreport:{fix_id}")
    resolution = doc.get("resolution") if isinstance(doc.get("resolution"), Mapping) else {}
    resolution_fix_id = str(resolution.get("fixreport_id") or "").strip()
    if resolution_fix_id:
        allowed.add(f"fixreport:{resolution_fix_id}")
    links = _as_list(doc.get("links"))
    if not allowed:
        return links
    return [
        token
        for token in links
        if not str(token or "").strip().startswith("fixreport:") or str(token or "").strip() in allowed
    ]


def _reset_tessairact_meta_links(doc: MutableMapping[str, Any]) -> None:
    tm = doc.get("tessairact_meta")
    if isinstance(tm, dict):
        tm["links"] = []


def _promote_reserved_finding_fields(doc: MutableMapping[str, Any]) -> None:
    """
    Promote reserved report-level fields accidentally nested under ``finding``.

    ``findings update --set-finding-json`` historically merged arbitrary keys
    into ``finding``. If a caller supplied ``{"links": [...]}``, the report
    YAML could end up with fresh ``finding.links`` but stale top-level
    ``doc["links"]`` / ``tessairact_meta.links`` / ``index.yml`` summaries.
    Keep top-level fields authoritative and strip nested copies.
    """
    finding = doc.get("finding")
    if not isinstance(finding, MutableMapping):
        return

    if "links" in finding:
        normalized_links, errs = _validate_links(finding.get("links"))
        if errs:
            raise ValueError("invalid finding.links: " + "; ".join(errs))
        doc["links"] = normalized_links
        finding.pop("links", None)


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


def _canonicalize_report_doc(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of the Finding report doc with stable, human-friendly key order.

    This affects only YAML output ordering (no semantic changes).
    """
    src = dict(doc)
    finding_src = src.get("finding")
    if isinstance(finding_src, Mapping):
        src["finding"] = dict(finding_src)
    _promote_reserved_finding_fields(src)

    # Canonicalize nested mappings first.
    finding_raw = src.get("finding") if isinstance(src.get("finding"), Mapping) else {}
    finding = dict(finding_raw)
    finding["target"] = _reordered_mapping(
        finding.get("target"),
        [
            "kind",
            "path",
            "systemd_unit",
            "topics",
            "tables_views",
            "horizon_H_seconds",
            "asset",
            "venue",
            "instrument",
        ],
    )
    finding["provenance"] = _reordered_mapping(
        finding.get("provenance"),
        [
            "discovered_at_utc",
            "author",
            "skill",
            "run_id",
            "commit",
        ],
    )
    finding["classification"] = _reordered_mapping(
        finding.get("classification"),
        [
            "bug_type",
            "symptom",
            "root_cause",
            "failure_mode",
            "impact_area",
        ],
    )
    related = finding.get("related") if isinstance(finding.get("related"), Mapping) else {}
    related2 = dict(related)
    related2["quantaudit"] = _reordered_mapping(related2.get("quantaudit"), ["bundle", "report_id", "local_finding_id"])
    finding["related"] = _reordered_mapping(related2, ["task_ids", "hypothesis_ids", "fixreport_ids", "quantaudit"])

    finding = _reordered_mapping(
        finding,
        [
            "id",
            "status",
            "addressed",
            "addressed_at_utc",
            "severity",
            "scope",
            "mode",
            "repo_area",
            "target",
            "title",
            "summary",
            "provenance",
            "classification",
            "related",
        ],
    )

    # Canonicalize top-level keys.
    src["finding"] = finding
    out: Dict[str, Any] = {}
    for k in ["finding", "links", "narrative", "evidence", "actions", "resolution", "metadata"]:
        if k in src:
            out[k] = src.pop(k)
    for k in sorted(src.keys()):
        out[k] = src[k]
    return out


def _extract_summary(doc: Mapping[str, Any], *, updated_mtime_ns: int) -> Dict[str, Any]:
    f_raw = doc.get("finding") if isinstance(doc.get("finding"), Mapping) else {}
    target = f_raw.get("target") if isinstance(f_raw.get("target"), Mapping) else {}
    prov = f_raw.get("provenance") if isinstance(f_raw.get("provenance"), Mapping) else {}
    addressed_raw = f_raw.get("addressed")
    addressed = bool(addressed_raw) if isinstance(addressed_raw, bool) else False
    summary = {
        "finding_id": _safe_str(f_raw.get("id")).strip(),
        "status": _safe_str(f_raw.get("status")).strip(),
        "addressed": addressed,
        "addressed_at_utc": _safe_str(f_raw.get("addressed_at_utc")).strip(),
        "severity": _safe_str(f_raw.get("severity")).strip(),
        "scope": _safe_str(f_raw.get("scope")).strip(),
        "mode": _safe_str(f_raw.get("mode")).strip(),
        "repo_area": _safe_str(f_raw.get("repo_area")).strip(),
        "target_kind": _safe_str(target.get("kind")).strip(),
        "target_path": _safe_str(target.get("path")).strip(),
        "ts_utc": _safe_str(prov.get("discovered_at_utc")).strip(),
        "title": _safe_str(f_raw.get("title")).strip(),
        "summary": _safe_str(f_raw.get("summary")).strip(),
        "links": _as_list(doc.get("links")),
        "updated_mtime_ns": int(updated_mtime_ns),
    }
    return summary


def _report_ext_priority(p: Path) -> int:
    suf = p.suffix.lower()
    if suf == ".yml":
        return 0
    if suf == ".yaml":
        return 1
    return 2


class FindingsStore:
    def __init__(self, repo_root: Path, *, state_dir: Optional[Path] = None) -> None:
        self.repo_root = repo_root.resolve()
        self.paths = default_paths(self.repo_root)
        if state_dir is not None:
            sd = state_dir.expanduser().resolve()
            self.paths = FindingPaths(
                state_dir=sd,
                reports_dir=sd / "reports",
                templates_dir=sd / "templates",
                index_path=sd / "index.yml",
                events_path=sd / "events.yml",
                lock_path=sd / ".lock",
                shipped_template_path=self.paths.shipped_template_path,
                state_template_path=(sd / "templates" / "bug_finding_template.yml"),
            )

    def init(self, *, force: bool = False) -> None:
        self.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        self.paths.templates_dir.mkdir(parents=True, exist_ok=True)
        if force or not self.paths.index_path.exists():
            payload: Dict[str, Any] = {"findings": []}
            ensure_tessairact_meta_header(
                payload,
                kind="findings_index",
                area="ops",
                uid="FIN_INDEX",
                actor="findings",
                links=["script:agentic_tools/findings/store.py"],
                repo_root=self.repo_root,
                tool="findings",
            )
            _atomic_write_yaml(self.paths.index_path, payload, sort_keys=True, allow_unicode=True)
        if force or not self.paths.events_path.exists():
            payload2: Dict[str, Any] = {"events": []}
            ensure_tessairact_meta_header(
                payload2,
                kind="findings_events",
                area="ops",
                uid="FIN_EVENTS",
                actor="findings",
                links=["script:agentic_tools/findings/store.py"],
                repo_root=self.repo_root,
                tool="findings",
            )
            _atomic_write_yaml(self.paths.events_path, payload2, sort_keys=True, allow_unicode=True)
        # Copy shipped template into state (best-effort; never fails init).
        try:
            if force or not self.paths.state_template_path.exists():
                if self.paths.shipped_template_path.exists():
                    txt = self.paths.shipped_template_path.read_text(encoding="utf-8")
                    _atomic_write_text(self.paths.state_template_path, txt)
        except Exception:
            pass

    def _discover_report_files_with_duplicates(self) -> Tuple[List[Path], Dict[str, List[Path]]]:
        if not self.paths.reports_dir.exists() or not self.paths.reports_dir.is_dir():
            return [], {}
        buckets: Dict[str, List[Path]] = {}
        for p in self.paths.reports_dir.iterdir():
            try:
                if not p.is_file():
                    continue
            except Exception:
                continue
            m = _REPORT_FILE_RE.fullmatch(p.name)
            if not m:
                continue
            fid = m.group(1)
            if not is_safe_finding_id(fid):
                continue
            buckets.setdefault(fid, []).append(p)

        duplicates: Dict[str, List[Path]] = {fid: ps for fid, ps in buckets.items() if len(ps) > 1}

        canonical: List[Path] = []
        for fid, ps in buckets.items():
            ps_sorted = sorted(ps, key=lambda x: (_report_ext_priority(x), x.name))
            canonical.append(ps_sorted[0])

        canonical.sort(key=lambda x: x.name)
        return canonical, duplicates

    def _discover_report_files(self) -> List[Path]:
        # Backward-compatible wrapper (returns canonical file list, preferring .yml).
        files, _dups = self._discover_report_files_with_duplicates()
        return files

    def list_finding_ids(self) -> List[str]:
        ids: List[str] = []
        for p in self._discover_report_files():
            ids.append(p.stem)
        return sorted(set(ids))

    def report_path(self, finding_id: str) -> Path:
        fid = (finding_id or "").strip()
        if not is_safe_finding_id(fid):
            raise ValueError("invalid finding id")
        # Prefer existing file with either extension (avoid creating duplicates).
        p_yml = self.paths.reports_dir / f"{fid}.yml"
        p_yaml = self.paths.reports_dir / f"{fid}.yaml"
        if p_yml.exists():
            return p_yml
        if p_yaml.exists():
            return p_yaml
        return p_yml

    def load_report(self, finding_id: str) -> Dict[str, Any]:
        path = self.report_path(finding_id)
        if not path.exists():
            raise FileNotFoundError(f"missing finding: {finding_id}")
        doc = _yaml_safe_load_path(path) or {}
        if not isinstance(doc, dict):
            raise ValueError(f"invalid finding file: {path}")
        return doc

    def _next_finding_id(self) -> str:
        """
        Generate the next numeric ID (FND000001) based on existing numeric IDs.
        """
        existing = self.list_finding_ids()
        nums: List[int] = []
        for fid in existing:
            if re.fullmatch(r"^FND[0-9]{6}$", fid):
                try:
                    nums.append(int(fid[3:]))
                except Exception:
                    continue
        if not nums:
            return "FND000001"
        return f"FND{max(nums) + 1:06d}"

    def _rebuild_index_locked(self, *, include_warnings: bool = True) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Rebuild artifacts/stores/findings/index.yml from reports.

        Returns: (count, issues) where issues contain parse/validation failures and, optionally, warnings.
        """
        issues: List[Dict[str, Any]] = []
        items: List[Dict[str, Any]] = []

        files, duplicates = self._discover_report_files_with_duplicates()
        for fid, ps in sorted(duplicates.items()):
            issues.append(
                {
                    "path": str(ps[0]),
                    "errors": [f"duplicate_finding_files: {fid}: " + ", ".join(str(p) for p in ps)],
                    "warnings": [],
                }
            )

        for p in files:
            updated = 0
            try:
                updated = int(p.stat().st_mtime_ns)
            except Exception:
                updated = 0
            try:
                doc = _yaml_safe_load_path(p) or {}
                if isinstance(doc, dict):
                    _promote_reserved_finding_fields(doc)
                errs, warns = _validate_report_doc(doc, path=p)
                if errs:
                    issues.append({"path": str(p), "errors": errs, "warnings": warns})
                    continue
                if warns and include_warnings:
                    issues.append({"path": str(p), "errors": [], "warnings": warns})
                if not isinstance(doc, Mapping):
                    continue
                items.append(_extract_summary(_canonicalize_report_doc(doc), updated_mtime_ns=updated))
            except Exception as exc:
                issues.append({"path": str(p), "errors": [f"parse_error: {exc}"], "warnings": []})

        def _key(it: Mapping[str, Any]) -> Tuple[str, int]:
            return (str(it.get("ts_utc") or ""), int(it.get("updated_mtime_ns") or 0))

        items.sort(key=_key, reverse=True)
        payload: Dict[str, Any] = {"findings": items}
        ensure_tessairact_meta_header(
            payload,
            kind="findings_index",
            area="ops",
            uid="FIN_INDEX",
            actor="findings",
            links=["script:agentic_tools/findings/store.py"],
            repo_root=self.repo_root,
            tool="findings",
        )
        _atomic_write_yaml(self.paths.index_path, payload, sort_keys=True, allow_unicode=True)
        return len(items), issues

    def rebuild_index(self, *, include_warnings: bool = True) -> Tuple[int, List[Dict[str, Any]]]:
        self.init(force=False)
        lock = FileLock(self.paths.lock_path)
        with lock:
            return self._rebuild_index_locked(include_warnings=include_warnings)

    def _read_index_items(self) -> Optional[List[Dict[str, Any]]]:
        try:
            raw = _load_yaml_or_default(self.paths.index_path, {"findings": []})
        except Exception:
            return None
        if not isinstance(raw, dict):
            return None
        it = raw.get("findings")
        if not isinstance(it, list):
            return None
        out: List[Dict[str, Any]] = []
        for x in it:
            if isinstance(x, dict):
                out.append(x)
        return out

    def _write_index_items(self, items: List[Dict[str, Any]]) -> None:
        payload: Dict[str, Any] = {"findings": items}
        ensure_tessairact_meta_header(
            payload,
            kind="findings_index",
            area="ops",
            uid="FIN_INDEX",
            actor="findings",
            links=["script:agentic_tools/findings/store.py"],
            repo_root=self.repo_root,
            tool="findings",
        )
        _atomic_write_yaml(self.paths.index_path, payload, sort_keys=True, allow_unicode=True)

    def _update_index_incremental(self, *, report_doc: Mapping[str, Any], report_path: Path) -> None:
        """
        Update index.yml using existing index contents as a base (fast path).
        Falls back to rebuild_index() on corruption.
        """
        try:
            updated_ns = int(report_path.stat().st_mtime_ns)
        except Exception:
            updated_ns = 0
        summary = _extract_summary(_canonicalize_report_doc(report_doc), updated_mtime_ns=updated_ns)
        fid = str(summary.get("finding_id") or "").strip()
        if not fid or not is_safe_finding_id(fid):
            # Index is derived; do not write garbage.
            raise ValueError("invalid finding id for index update")

        items = self._read_index_items()
        if items is None:
            # Fallback: rebuild from reports.
            self._rebuild_index_locked()
            return

        # Filter to existing report IDs (cheap directory scan, no YAML parse).
        existing_ids = set(self.list_finding_ids())
        filtered: List[Dict[str, Any]] = []
        for it in items:
            it_id = str(it.get("finding_id") or "").strip()
            if it_id and it_id in existing_ids and it_id != fid:
                filtered.append(it)

        # Add updated summary
        filtered.append(summary)

        def _key(it: Mapping[str, Any]) -> Tuple[str, int]:
            return (str(it.get("ts_utc") or ""), int(it.get("updated_mtime_ns") or 0))

        filtered.sort(key=_key, reverse=True)
        self._write_index_items(filtered)

    def _append_event(self, event: Mapping[str, Any]) -> None:
        raw = _load_yaml_or_default(self.paths.events_path, {"events": []})
        if not isinstance(raw, dict):
            raw = {"events": []}
        evs = raw.get("events")
        if not isinstance(evs, list):
            evs = []
        evs = list(evs)
        evs.append(dict(event))
        raw["events"] = evs
        actor = str(event.get("actor") or "findings").strip() or "findings"
        ensure_tessairact_meta_header(
            raw,
            kind="findings_events",
            area="ops",
            uid="FIN_EVENTS",
            actor=actor,
            links=["script:agentic_tools/findings/store.py"],
            repo_root=self.repo_root,
            tool="findings",
        )
        _atomic_write_yaml(self.paths.events_path, raw, sort_keys=True, allow_unicode=True)

    def create(
        self,
        *,
        finding: Mapping[str, Any],
        links: Optional[Sequence[str]] = None,
        narrative: Optional[Mapping[str, Any]] = None,
        evidence: Optional[Mapping[str, Any]] = None,
        actions: Optional[Sequence[Mapping[str, Any]]] = None,
        resolution: Optional[Mapping[str, Any]] = None,
        actor: Optional[str] = None,
    ) -> Tuple[str, Path]:
        self.init(force=False)
        lock = FileLock(self.paths.lock_path)
        with lock:
            fid = _safe_str(finding.get("id")).strip() if isinstance(finding, Mapping) else ""
            if not fid:
                fid = self._next_finding_id()
            if not is_safe_finding_id(fid):
                raise ValueError(f"invalid finding id: {fid}")

            finding_fields = dict(finding)
            finding_fields["id"] = fid
            if not str(finding_fields.get("status") or "").strip():
                finding_fields["status"] = "open"
            if "addressed" not in finding_fields:
                finding_fields["addressed"] = False
            if "addressed_at_utc" not in finding_fields:
                finding_fields["addressed_at_utc"] = ""
            if not str(finding_fields.get("severity") or "").strip():
                finding_fields["severity"] = "p2"
            if not str(finding_fields.get("scope") or "").strip():
                finding_fields["scope"] = "analytics"
            if not str(finding_fields.get("mode") or "").strip():
                finding_fields["mode"] = "dev"
            if not str(finding_fields.get("repo_area") or "").strip():
                finding_fields["repo_area"] = "services"

            prov = finding_fields.get("provenance")
            if not isinstance(prov, Mapping):
                prov = {}
            prov = dict(prov)
            if not str(prov.get("discovered_at_utc") or "").strip():
                prov["discovered_at_utc"] = _utc_now_z()
            if actor and not prov.get("author"):
                prov["author"] = actor
            finding_fields["provenance"] = prov

            doc: Dict[str, Any] = {"finding": finding_fields}
            auto_links = extract_link_tokens_from_texts(
                _collect_text_blobs(finding_fields)
                + _collect_text_blobs(narrative)
                + _collect_text_blobs(evidence)
                + _collect_text_blobs(actions)
                + _collect_text_blobs(resolution)
            )
            # Avoid a self-referential link generated from finding.id (e.g. "FND000123" -> "finding:FND000123").
            auto_links = [x for x in auto_links if x != f"finding:{fid}"]
            if links is not None:
                merged_links = list(links) + list(auto_links)
                ln, errs = _validate_links(list(merged_links))
                if errs:
                    raise ValueError("invalid links: " + "; ".join(errs))
                doc["links"] = ln
            elif auto_links:
                ln, errs = _validate_links(list(auto_links))
                if errs:
                    raise ValueError("invalid links: " + "; ".join(errs))
                doc["links"] = ln
            if narrative is not None:
                doc["narrative"] = dict(narrative)
            if evidence is not None:
                doc["evidence"] = dict(evidence)
            if actions is not None:
                doc["actions"] = [dict(a) for a in actions]
            if resolution is not None:
                doc["resolution"] = dict(resolution)
            if actor:
                doc.setdefault("metadata", {})["created_by"] = actor

            # Validate before writing.
            errs, warns = _validate_report_doc(doc, path=None)
            if errs:
                raise ValueError("validation_failed: " + "; ".join(errs))
            if warns:
                doc.setdefault("metadata", {})["lint_warnings"] = warns

            doc = _canonicalize_report_doc(doc)
            doc["links"] = _canonical_fixreport_links_for_finding_doc(doc)
            _reset_tessairact_meta_links(doc)

            f2 = doc.get("finding") if isinstance(doc.get("finding"), Mapping) else {}
            target = f2.get("target") if isinstance(f2.get("target"), Mapping) else {}
            target_path = str(target.get("path") or "").strip()
            area = infer_area_from_path(target_path) if target_path else "ops"
            prov2 = f2.get("provenance") if isinstance(f2.get("provenance"), Mapping) else {}
            created_ts = str(prov2.get("discovered_at_utc") or "").strip()
            actor_s = str(actor or prov2.get("author") or "findings").strip() or "findings"
            title = str(f2.get("title") or "").strip()
            ensure_tessairact_meta_header(
                doc,
                kind="finding",
                area=area,
                uid=fid,
                actor=actor_s,
                title=title,
                links=_as_list(doc.get("links")),
                repo_root=self.repo_root,
                tool="findings",
                created_at_utc=created_ts,
                updated_at_utc=created_ts,
            )

            path = self.report_path(fid)
            if path.exists():
                raise ValueError(f"finding id already exists: {fid}")

            # Rollback plan (best-effort).
            rollback_actions: List[Tuple[str, Any]] = []

            def _rollback() -> None:
                for kind, payload in reversed(rollback_actions):
                    try:
                        if kind == "delete":
                            p: Path = payload
                            p.unlink(missing_ok=True)
                        elif kind == "restore_text":
                            p, txt = payload
                            _atomic_write_text(p, txt)
                    except Exception:
                        pass

            # Backup index/events before changes (text; index small, events potentially large).
            old_index_txt: Optional[str] = None
            try:
                if self.paths.index_path.exists():
                    old_index_txt = self.paths.index_path.read_text(encoding="utf-8")
            except Exception:
                old_index_txt = None
            if old_index_txt is not None:
                rollback_actions.append(("restore_text", (self.paths.index_path, old_index_txt)))

            old_events_txt: Optional[str] = None
            try:
                if self.paths.events_path.exists():
                    # Avoid unbounded memory usage: only keep backup if reasonably sized.
                    max_bak = _env_int("CAIA_FINDINGS_MAX_ROLLBACK_BYTES", 2 * 1024 * 1024)
                    sz = int(self.paths.events_path.stat().st_size)
                    if max_bak > 0 and sz <= max_bak:
                        old_events_txt = self.paths.events_path.read_text(encoding="utf-8")
            except Exception:
                old_events_txt = None
            if old_events_txt is not None:
                rollback_actions.append(("restore_text", (self.paths.events_path, old_events_txt)))

            try:
                _atomic_write_yaml(path, doc, sort_keys=False, allow_unicode=True)
                rollback_actions.append(("delete", path))

                # Fast incremental index update (fallback to rebuild).
                try:
                    self._update_index_incremental(report_doc=doc, report_path=path)
                except Exception:
                    self._rebuild_index_locked()

                self._append_event(
                    {
                        "id": f"EV-{fid}-created",
                        "timestamp": _utc_now_z(),
                        "kind": "finding_created",
                        "finding_id": fid,
                        "actor": actor or "",
                    }
                )
                return fid, path
            except Exception:
                _rollback()
                raise

    def update(
        self,
        finding_id: str,
        *,
        set_finding: Optional[Mapping[str, Any]] = None,
        set_narrative: Optional[Mapping[str, Any]] = None,
        set_evidence: Optional[Mapping[str, Any]] = None,
        set_resolution: Optional[Mapping[str, Any]] = None,
        add_links: Optional[Sequence[str]] = None,
        actor: Optional[str] = None,
    ) -> Path:
        self.init(force=False)
        fid = (finding_id or "").strip()
        if not is_safe_finding_id(fid):
            raise ValueError("invalid finding id")

        lock = FileLock(self.paths.lock_path)
        with lock:
            path = self.report_path(fid)
            if not path.exists():
                raise FileNotFoundError(f"missing finding: {fid}")

            old_report_txt: Optional[str] = None
            try:
                old_report_txt = path.read_text(encoding="utf-8")
            except Exception:
                old_report_txt = None

            raw = _yaml_safe_load_path(path) or {}
            if not isinstance(raw, dict):
                raise ValueError(f"invalid finding file: {path}")

            f = raw.get("finding")
            if not isinstance(f, dict):
                f = {}
            if set_finding:
                for k, v in dict(set_finding).items():
                    # Union-merge lists.
                    if isinstance(f.get(k), list) and isinstance(v, list):
                        prev = [str(x) for x in f.get(k) or [] if str(x).strip()]
                        nxt = [str(x) for x in v if str(x).strip()]
                        seen: set[str] = set()
                        merged: List[str] = []
                        for it in prev + nxt:
                            if it in seen:
                                continue
                            seen.add(it)
                            merged.append(it)
                        f[k] = merged
                    else:
                        f[k] = v
            f["id"] = fid
            raw["finding"] = f
            _promote_reserved_finding_fields(raw)

            if set_narrative is not None:
                nar = raw.get("narrative")
                if not isinstance(nar, dict):
                    nar = {}
                nar.update(dict(set_narrative))
                raw["narrative"] = nar

            if set_evidence is not None:
                ev = raw.get("evidence")
                if not isinstance(ev, dict):
                    ev = {}
                ev.update(dict(set_evidence))
                raw["evidence"] = ev

            if set_resolution is not None:
                res = raw.get("resolution")
                if not isinstance(res, dict):
                    res = {}
                res.update(dict(set_resolution))
                raw["resolution"] = res

            if add_links is not None:
                cur = _as_list(raw.get("links"))
                add = list(add_links)
                ln, errs = _validate_links(cur + add)
                if errs:
                    raise ValueError("invalid links: " + "; ".join(errs))
                raw["links"] = ln

            auto_links = extract_link_tokens_from_texts(
                _collect_text_blobs(raw.get("finding"))
                + _collect_text_blobs(raw.get("narrative"))
                + _collect_text_blobs(raw.get("evidence"))
                + _collect_text_blobs(raw.get("actions"))
                + _collect_text_blobs(raw.get("resolution"))
            )
            if auto_links:
                cur_links = _as_list(raw.get("links"))
                ln, errs = _validate_links(cur_links + list(auto_links))
                if errs:
                    raise ValueError("invalid links: " + "; ".join(errs))
                raw["links"] = ln

            errs, warns = _validate_report_doc(raw, path=None)
            if errs:
                raise ValueError("validation_failed: " + "; ".join(errs))
            if warns:
                raw.setdefault("metadata", {})["lint_warnings"] = warns

            raw2 = _canonicalize_report_doc(raw)
            raw2["links"] = _canonical_fixreport_links_for_finding_doc(raw2)
            _reset_tessairact_meta_links(raw2)

            f2 = raw2.get("finding") if isinstance(raw2.get("finding"), Mapping) else {}
            target = f2.get("target") if isinstance(f2.get("target"), Mapping) else {}
            target_path = str(target.get("path") or "").strip()
            area = infer_area_from_path(target_path) if target_path else "ops"
            prov2 = f2.get("provenance") if isinstance(f2.get("provenance"), Mapping) else {}
            created_ts = str(prov2.get("discovered_at_utc") or "").strip()
            actor_s = str(actor or prov2.get("author") or "findings").strip() or "findings"
            title = str(f2.get("title") or "").strip()
            ensure_tessairact_meta_header(
                raw2,
                kind="finding",
                area=area,
                uid=fid,
                actor=actor_s,
                title=title,
                links=_as_list(raw2.get("links")),
                repo_root=self.repo_root,
                tool="findings",
                created_at_utc=created_ts,
                updated_at_utc=_utc_now_z(),
            )

            rollback_actions: List[Tuple[str, Any]] = []

            def _rollback() -> None:
                for kind, payload in reversed(rollback_actions):
                    try:
                        if kind == "restore_text":
                            p, txt = payload
                            _atomic_write_text(p, txt)
                    except Exception:
                        pass

            if old_report_txt is not None:
                rollback_actions.append(("restore_text", (path, old_report_txt)))

            # Backup index/events (bounded) before changes
            old_index_txt: Optional[str] = None
            try:
                if self.paths.index_path.exists():
                    old_index_txt = self.paths.index_path.read_text(encoding="utf-8")
            except Exception:
                old_index_txt = None
            if old_index_txt is not None:
                rollback_actions.append(("restore_text", (self.paths.index_path, old_index_txt)))

            old_events_txt: Optional[str] = None
            try:
                if self.paths.events_path.exists():
                    max_bak = _env_int("CAIA_FINDINGS_MAX_ROLLBACK_BYTES", 2 * 1024 * 1024)
                    sz = int(self.paths.events_path.stat().st_size)
                    if max_bak > 0 and sz <= max_bak:
                        old_events_txt = self.paths.events_path.read_text(encoding="utf-8")
            except Exception:
                old_events_txt = None
            if old_events_txt is not None:
                rollback_actions.append(("restore_text", (self.paths.events_path, old_events_txt)))

            try:
                _atomic_write_yaml(path, raw2, sort_keys=False, allow_unicode=True)

                try:
                    self._update_index_incremental(report_doc=raw2, report_path=path)
                except Exception:
                    self._rebuild_index_locked()

                self._append_event(
                    {
                        "id": f"EV-{fid}-updated",
                        "timestamp": _utc_now_z(),
                        "kind": "finding_updated",
                        "finding_id": fid,
                        "actor": actor or "",
                    }
                )
                return path
            except Exception:
                _rollback()
                raise

    def set_addressed(
        self,
        finding_id: str,
        *,
        addressed: bool,
        actor: Optional[str] = None,
        add_links: Optional[Sequence[str]] = None,
    ) -> Path:
        """
        Mark a finding as addressed/unaddressed (checkmark state), recording a dedicated event.

        Addressed is distinct from `finding.status` (open/resolved/etc.).
        """
        self.init(force=False)
        fid = (finding_id or "").strip()
        if not is_safe_finding_id(fid):
            raise ValueError("invalid finding id")

        lock = FileLock(self.paths.lock_path)
        with lock:
            path = self.report_path(fid)
            if not path.exists():
                raise FileNotFoundError(f"missing finding: {fid}")

            old_report_txt: Optional[str] = None
            try:
                old_report_txt = path.read_text(encoding="utf-8")
            except Exception:
                old_report_txt = None

            raw = _yaml_safe_load_path(path) or {}
            if not isinstance(raw, dict):
                raise ValueError(f"invalid finding file: {path}")

            f = raw.get("finding")
            if not isinstance(f, dict):
                f = {}
            cur = f.get("addressed")
            cur_bool = bool(cur) if isinstance(cur, bool) else False
            want = bool(addressed)
            ts = _utc_now_z()

            changed = False
            if cur_bool != want:
                changed = True
            f["addressed"] = want
            if want:
                if not str(f.get("addressed_at_utc") or "").strip():
                    f["addressed_at_utc"] = ts
                    changed = True
            else:
                if str(f.get("addressed_at_utc") or "").strip():
                    f["addressed_at_utc"] = ""
                    changed = True

            f["id"] = fid
            raw["finding"] = f
            _promote_reserved_finding_fields(raw)

            if add_links is not None:
                cur_links = _as_list(raw.get("links"))
                ln, errs = _validate_links(cur_links + list(add_links))
                if errs:
                    raise ValueError("invalid links: " + "; ".join(errs))
                raw["links"] = ln
                # Even if `addressed` doesn't change, link updates should be written.
                if ln != cur_links:
                    changed = True

            errs, warns = _validate_report_doc(raw, path=None)
            if errs:
                raise ValueError("validation_failed: " + "; ".join(errs))
            if warns:
                raw.setdefault("metadata", {})["lint_warnings"] = warns

            rollback_actions: List[Tuple[str, Any]] = []

            def _rollback() -> None:
                for kind, payload in reversed(rollback_actions):
                    try:
                        if kind == "restore_text":
                            p, txt = payload
                            _atomic_write_text(p, txt)
                    except Exception:
                        pass

            if old_report_txt is not None:
                rollback_actions.append(("restore_text", (path, old_report_txt)))

            # Backup index/events (bounded) before changes if we will write.
            old_index_txt: Optional[str] = None
            old_events_txt: Optional[str] = None
            if changed:
                try:
                    if self.paths.index_path.exists():
                        old_index_txt = self.paths.index_path.read_text(encoding="utf-8")
                except Exception:
                    old_index_txt = None
                if old_index_txt is not None:
                    rollback_actions.append(("restore_text", (self.paths.index_path, old_index_txt)))

                try:
                    if self.paths.events_path.exists():
                        max_bak = _env_int("CAIA_FINDINGS_MAX_ROLLBACK_BYTES", 2 * 1024 * 1024)
                        sz = int(self.paths.events_path.stat().st_size)
                        if max_bak > 0 and sz <= max_bak:
                            old_events_txt = self.paths.events_path.read_text(encoding="utf-8")
                except Exception:
                    old_events_txt = None
                if old_events_txt is not None:
                    rollback_actions.append(("restore_text", (self.paths.events_path, old_events_txt)))

            if changed:
                try:
                    raw2 = _canonicalize_report_doc(raw)
                    raw2["links"] = _canonical_fixreport_links_for_finding_doc(raw2)
                    _reset_tessairact_meta_links(raw2)
                    f2 = raw2.get("finding") if isinstance(raw2.get("finding"), Mapping) else {}
                    target = f2.get("target") if isinstance(f2.get("target"), Mapping) else {}
                    target_path = str(target.get("path") or "").strip()
                    area = infer_area_from_path(target_path) if target_path else "ops"
                    prov2 = f2.get("provenance") if isinstance(f2.get("provenance"), Mapping) else {}
                    created_ts = str(prov2.get("discovered_at_utc") or "").strip()
                    actor_s = str(actor or prov2.get("author") or "findings").strip() or "findings"
                    title = str(f2.get("title") or "").strip()
                    ensure_tessairact_meta_header(
                        raw2,
                        kind="finding",
                        area=area,
                        uid=fid,
                        actor=actor_s,
                        title=title,
                        links=_as_list(raw2.get("links")),
                        repo_root=self.repo_root,
                        tool="findings",
                        created_at_utc=created_ts,
                        updated_at_utc=ts,
                    )
                    _atomic_write_yaml(path, raw2, sort_keys=False, allow_unicode=True)

                    try:
                        self._update_index_incremental(report_doc=raw2, report_path=path)
                    except Exception:
                        self._rebuild_index_locked()

                    self._append_event(
                        {
                            "id": f"EV-{fid}-{'addressed' if want else 'unaddressed'}",
                            "timestamp": ts,
                            "kind": "finding_addressed" if want else "finding_unaddressed",
                            "finding_id": fid,
                            "actor": actor or "",
                        }
                    )
                except Exception:
                    _rollback()
                    raise
            return path

    def lint(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate all finding YAML docs under the findings store (canonical: artifacts/stores/findings/reports/).
        """
        self.init(force=False)
        issues: List[Dict[str, Any]] = []

        files, duplicates = self._discover_report_files_with_duplicates()
        for fid, ps in sorted(duplicates.items()):
            issues.append(
                {
                    "path": str(ps[0]),
                    "errors": [f"duplicate_finding_files: {fid}: " + ", ".join(str(p) for p in ps)],
                    "warnings": [],
                }
            )

        for p in files:
            if not p.is_file():
                continue
            try:
                doc = _yaml_safe_load_path(p) or {}
            except Exception as exc:
                issues.append({"path": str(p), "errors": [f"parse_error: {exc}"], "warnings": []})
                continue
            errs, warns = _validate_report_doc(doc, path=p)
            if errs or warns:
                issues.append({"path": str(p), "errors": errs, "warnings": warns})
        ok = all(not it.get("errors") for it in issues)
        return ok, issues
