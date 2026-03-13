from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from agentic_tools._governance_locking import FileLockTimeout as _GovernanceFileLockTimeout
from agentic_tools._governance_locking import resolve_retry_budget as _resolve_governance_retry_budget
from agentic_tools._governance_locking import resolve_retry_poll as _resolve_governance_retry_poll
from agentic_tools._governance_locking import run_mutation_with_retry as _run_governance_mutation_with_retry
from agentic_tools.process_hints import enrich_payload_with_hints, hints_schema_fragment
from common.links.tessairact_meta import ensure_tessairact_meta_header, extract_links_from_doc

from .store import ReportStore, ReportStoreError
from .types import (
    LINK_TOKEN_RE,
    REPORT_QUALITY_POLICY,
    REPORT_ID_RE,
    REPORT_KINDS,
    REPORT_STATUSES,
    format_report_markdown,
    new_report_id,
    normalize_evidence,
    normalize_kind,
    normalize_lines,
    normalize_links,
    normalize_status,
    normalize_tags,
    safe_asset_name,
    utc_now_z,
    validate_report_payload_quality,
)


class _ArgparseError(Exception):
    def __init__(self, message: str, usage: str = "") -> None:
        super().__init__(message)
        self.message = str(message)
        self.usage = str(usage or "")


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # noqa: A003 (argparse API)
        raise _ArgparseError(str(message), self.format_usage())


def _json_dumps(payload: Any, *, pretty: bool) -> str:
    if pretty:
        return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False)
    return json.dumps(payload, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def _emit(payload: Mapping[str, Any], *, pretty: bool, action: str = "") -> None:
    out = enrich_payload_with_hints(payload, tool="report", default_action=action)
    sys.stdout.write(_json_dumps(out, pretty=pretty) + "\n")


def _error_payload(error: str, detail: str, *, error_code: str = "REPORT_ERROR", retryable: bool = False, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": False,
        "error": str(error or "error"),
        "error_code": str(error_code or "REPORT_ERROR"),
        "detail": str(detail or ""),
        "retryable": bool(retryable),
    }
    payload.update(extra)
    return payload


def _classify_report_exception(exc: BaseException) -> Tuple[str, Optional[bool]]:
    if isinstance(exc, _ArgparseError):
        return ("ARGPARSE_ERROR", False)
    if isinstance(exc, _GovernanceFileLockTimeout):
        return ("LOCK_TIMEOUT", True)
    if isinstance(exc, PermissionError):
        return ("PERMISSION_DENIED", False)
    if isinstance(exc, FileNotFoundError):
        return ("NOT_FOUND", False)
    if isinstance(exc, TimeoutError):
        return ("TIMEOUT", True)
    if isinstance(exc, (ReportStoreError, ValueError)):
        return ("REPORT_ERROR", False)
    if isinstance(exc, OSError):
        return ("IO_ERROR", True)
    return ("UNHANDLED_EXCEPTION", None)


def _report_cli_retry_budget_sec() -> float:
    return _resolve_governance_retry_budget(
        retry_timeout_env="CAIA_REPORT_CLI_RETRY_TIMEOUT_SEC",
        lock_timeout_envs=("CAIA_REPORT_LOCK_TIMEOUT_SEC", "REPORT_LOCK_TIMEOUT"),
        default_lock_timeout=10.0,
    )


def _report_cli_retry_poll_sec() -> float:
    return _resolve_governance_retry_poll(
        retry_poll_env="CAIA_REPORT_CLI_RETRY_POLL_SEC",
        default_poll_sec=0.25,
    )


def _emit_report_retry(action: str, error_code: str, attempt: int, sleep_s: float, exc: BaseException) -> None:
    print(
        f"report {action}: retrying after {error_code} (attempt={attempt}, sleep_s={sleep_s:.3f})",
        file=sys.stderr,
    )


def _run_report_mutation_with_retry(action: str, op: Any) -> Tuple[Any, Dict[str, Any]]:
    return _run_governance_mutation_with_retry(
        action=action,
        op=op,
        classify_exception=_classify_report_exception,
        retry_budget_sec=_report_cli_retry_budget_sec(),
        retry_poll_sec=_report_cli_retry_poll_sec(),
        emit_retry=lambda error_code, attempt, sleep_s, exc: _emit_report_retry(
            action,
            error_code,
            attempt,
            sleep_s,
            exc,
        ),
    )


def _read_text_limited(path: Path, *, max_bytes: int) -> str:
    st = path.stat()
    if int(getattr(st, "st_size", 0) or 0) > int(max_bytes):
        raise ValueError(f"file too large: {path} ({int(st.st_size)} bytes > {int(max_bytes)} bytes)")
    return path.read_text(encoding="utf-8")


def _load_data_file(path: Path) -> Dict[str, Any]:
    suf = path.suffix.lower()
    if suf in {".yml", ".yaml"}:
        try:
            import yaml  # local import
        except Exception as exc:
            raise ValueError("PyYAML is required to load YAML input files.") from exc
        obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(obj, Mapping):
            raise ValueError("data-file YAML root must be a mapping")
        return dict(obj)
    obj = json.loads(path.read_text(encoding="utf-8") or "{}")
    if not isinstance(obj, Mapping):
        raise ValueError("data-file JSON root must be a mapping")
    return dict(obj)


def _normalize_create_request(
    raw: Mapping[str, Any],
    *,
    links_add: Sequence[str],
    scripts_add: Sequence[str],
    tags_add: Sequence[str],
) -> Tuple[Dict[str, Any], List[str]]:
    errs: List[str] = []
    data: Dict[str, Any] = dict(raw or {})

    title = str(data.get("title") or "").strip()
    if not title:
        errs.append("title is required")
    if len(title) > 200:
        errs.append("title too long (>200 chars)")

    kind = normalize_kind(data.get("kind"))
    if kind not in set(REPORT_KINDS):
        errs.append(f"kind must be one of {list(REPORT_KINDS)}")
    data["kind"] = kind

    tags = normalize_tags(data.get("tags"))
    tags = normalize_tags(list(tags) + list(tags_add))
    data["tags"] = tags

    summary = normalize_lines(data.get("summary"))
    if not summary:
        errs.append("summary is required (string or list)")
    data["summary"] = summary

    insights = normalize_lines(data.get("insights"))
    if not insights:
        errs.append("insights is required (string or list; >=1)")
    data["insights"] = insights

    evidence_items, evidence_errs = normalize_evidence(data.get("evidence"))
    if evidence_errs:
        errs.extend(evidence_errs)
    if not evidence_items:
        errs.append("evidence is required (non-empty list of {kind, ref, note?})")
    data["evidence"] = [x.as_dict() for x in evidence_items]

    links0 = data.get("links")
    links, bad_links = normalize_links(list(links0 if isinstance(links0, list) else []) + list(links_add))
    links = normalize_links(list(links) + [f"script:{p}" for p in scripts_add if str(p or '').strip()])[0]
    if bad_links:
        errs.append(f"invalid link tokens: {bad_links[:10]}")
    data["links"] = links

    # Minimal gate: at least one anchor link.
    if not any(str(x).startswith(("script:", "task:", "hypothesis:", "service_unit:")) for x in (links or [])):
        errs.append("links must include at least one anchor (script:/task:/hypothesis:/service_unit:)")

    for key in ("limitations", "risks", "next_actions"):
        data[key] = normalize_lines(data.get(key))

    repro_raw = data.get("repro")
    if isinstance(repro_raw, str):
        data["repro"] = [x.strip() for x in str(repro_raw).splitlines() if x.strip()]
    elif isinstance(repro_raw, list):
        data["repro"] = [str(x).strip() for x in repro_raw if str(x).strip()]
    elif repro_raw is None:
        data["repro"] = []
    else:
        errs.append("repro must be a string or list of commands")
        data["repro"] = []

    context_raw = data.get("context")
    if context_raw is None:
        data["context"] = {}
    elif isinstance(context_raw, Mapping):
        data["context"] = dict(context_raw)
    else:
        errs.append("context must be a mapping")
        data["context"] = {}

    results_raw = data.get("results")
    if results_raw is None:
        data["results"] = {}
    elif isinstance(results_raw, Mapping):
        data["results"] = dict(results_raw)
    else:
        errs.append("results must be a mapping")
        data["results"] = {}

    errs.extend(validate_report_payload_quality(data))

    return data, errs


def _build_report_doc(
    *,
    report_id: str,
    actor: str,
    created_at_utc: str,
    status: str,
    create_data: Mapping[str, Any],
    md_rel_path: str,
    assets_rel_paths: Sequence[str],
) -> Dict[str, Any]:
    rep: Dict[str, Any] = {
        "report_id": report_id,
        "created_at_utc": created_at_utc,
        "created_by": actor,
        "status": status,
        "finalized_at_utc": None,
        "title": str(create_data.get("title") or "").strip(),
        "kind": str(create_data.get("kind") or "").strip().lower(),
        "tags": list(create_data.get("tags") or []),
        "summary": list(create_data.get("summary") or []),
        "insights": list(create_data.get("insights") or []),
        "evidence": list(create_data.get("evidence") or []),
        "results": dict(create_data.get("results") or {}) if isinstance(create_data.get("results"), Mapping) else {},
        "context": dict(create_data.get("context") or {}) if isinstance(create_data.get("context"), Mapping) else {},
        "limitations": list(create_data.get("limitations") or []),
        "risks": list(create_data.get("risks") or []),
        "next_actions": list(create_data.get("next_actions") or []),
        "repro": list(create_data.get("repro") or []),
        "content": {
            "md_rel_path": md_rel_path,
            "assets_rel_paths": list(assets_rel_paths or []),
        },
    }
    doc: Dict[str, Any] = {"report": rep}
    links = list(create_data.get("links") or [])
    ensure_tessairact_meta_header(
        doc,
        kind="report",
        area="report",
        uid=report_id,
        actor=actor,
        title=rep["title"],
        links=links,
        tool="report",
    )
    return doc


def _copy_attachments(
    *,
    repo_root: Path,
    report_id: str,
    attach_paths: Sequence[str],
    max_bytes: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Copy attachments to artifacts/reports/report/<id>/assets and return rel paths + audit info.
    """
    rels: List[str] = []
    copied: List[Dict[str, Any]] = []
    if not attach_paths:
        return rels, copied

    assets_rel_dir = f"artifacts/reports/report/{report_id}/assets"
    assets_abs_dir = (repo_root / assets_rel_dir).resolve()
    assets_abs_dir.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    for raw in attach_paths:
        src = Path(str(raw or "").strip()).expanduser()
        if not src.is_absolute():
            src = (repo_root / src).resolve()
        if not src.exists() or not src.is_file():
            raise ValueError(f"attachment not found: {raw}")
        st = src.stat()
        if int(getattr(st, "st_size", 0) or 0) > int(max_bytes):
            raise ValueError(f"attachment too large: {src} ({int(st.st_size)} bytes > {int(max_bytes)} bytes)")

        name = safe_asset_name(src.name)
        if name in used_names:
            stem = Path(name).stem
            suf = Path(name).suffix
            name = f"{stem}_{report_id[:14]}{suf}"
        used_names.add(name)
        dst = (assets_abs_dir / name).resolve()
        if assets_abs_dir not in dst.parents and dst != assets_abs_dir:
            raise ValueError("attachment path escape detected")
        shutil.copyfile(src, dst)

        rel_path = f"{assets_rel_dir}/{name}"
        rels.append(rel_path)
        copied.append({"src": str(src), "dst": str(dst), "rel": rel_path, "bytes": int(st.st_size)})

    return rels, copied


def _cmd_schema(*, pretty: bool) -> int:
    commands: Dict[str, str] = {
        "schema": "Emit JSON schema + command summaries.",
        "init": "Initialize report store dirs/templates/index/events.",
        "template": "Emit starter templates (data YAML or body Markdown).",
        "create": "Create a new report (YAML store + Markdown + assets).",
        "list": "List reports from index (filterable).",
        "get": "Get a report (summary + paths).",
        "finalize": "Finalize a report (immutable).",
        "revise": "Create a new report that references a prior report.",
        "rebuild-index": "Rebuild index.yml from report YAML files.",
        "lint": "Validate report invariants + referenced files + quality gates.",
    }

    schema = {
        "commands": commands,
        "quality_policy": REPORT_QUALITY_POLICY,
        "hint_envelope": hints_schema_fragment(),
        "create_request": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "title": {"type": "string", "minLength": 1, "maxLength": 200},
                "kind": {"type": "string", "enum": list(REPORT_KINDS)},
                "tags": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": ["string", "array"], "items": {"type": "string"}},
                "insights": {"type": ["string", "array"], "items": {"type": "string"}},
                "evidence": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            "kind": {"type": "string", "minLength": 1},
                            "ref": {"type": "string", "minLength": 1},
                            "note": {"type": "string"},
                        },
                        "required": ["kind", "ref"],
                    },
                },
                "context": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "objective": {"type": "string", "minLength": int(REPORT_QUALITY_POLICY["min_explanatory_chars"])},
                        "scope": {"type": "string", "minLength": int(REPORT_QUALITY_POLICY["min_explanatory_chars"])},
                        "methodology": {"type": "string", "minLength": int(REPORT_QUALITY_POLICY["min_explanatory_chars"])},
                        "data_window": {"type": "string", "minLength": int(REPORT_QUALITY_POLICY["min_explanatory_chars"])},
                    },
                    "required": list(REPORT_QUALITY_POLICY["context_required_keys"]),
                },
                "results": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {
                        "narrative": {"type": "string", "minLength": int(REPORT_QUALITY_POLICY["min_explanatory_chars"])},
                        "kpis": {
                            "type": "array",
                            "minItems": int(REPORT_QUALITY_POLICY["min_kpis_generic"]),
                            "items": {
                                "type": "object",
                                "additionalProperties": True,
                                "properties": {
                                    "name": {"type": "string", "minLength": 1},
                                    "value": {},
                                    "baseline": {"type": ["string", "number"]},
                                    "delta": {"type": ["string", "number"]},
                                    "explanation": {"type": "string", "minLength": int(REPORT_QUALITY_POLICY["min_explanatory_chars"])},
                                },
                                "required": ["name", "value", "explanation"],
                            },
                        },
                        "tests": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": True,
                                "properties": {
                                    "name": {"type": "string", "minLength": 1},
                                    "dataset_or_suite": {"type": "string", "minLength": 1},
                                    "status": {"type": "string", "minLength": 1},
                                    "sample_size_n": {"type": ["integer", "number", "string"]},
                                    "kpi_refs": {"type": "array", "items": {"type": "string"}},
                                    "notes": {"type": "string"},
                                },
                                "required": ["name", "dataset_or_suite", "status", "sample_size_n", "kpi_refs"],
                            },
                        },
                    },
                    "required": ["narrative"],
                },
                "limitations": {"type": ["string", "array"], "items": {"type": "string"}},
                "risks": {"type": ["string", "array"], "items": {"type": "string"}},
                "next_actions": {"type": ["string", "array"], "items": {"type": "string"}},
                "repro": {"type": ["string", "array"], "items": {"type": "string"}},
                "links": {"type": "array", "items": {"type": "string", "pattern": LINK_TOKEN_RE.pattern}},
            },
            "required": ["title", "kind", "summary", "insights", "evidence", "context", "results", "limitations", "risks", "next_actions", "repro"],
        },
        "report_doc": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "tessairact_meta": {"type": "object"},
                "report": {"type": "object"},
            },
            "required": ["tessairact_meta", "report"],
        },
    }

    payload = {"ok": True, "action": "schema", "commands": commands, "schema": schema}
    _emit(payload, pretty=pretty, action="schema")
    return 0


def _cmd_init(args: argparse.Namespace, *, repo_root: Path, pretty: bool) -> int:
    store = ReportStore(repo_root=repo_root)
    info, _ = _run_report_mutation_with_retry(
        "init",
        lambda: store.ensure_initialized(actor=str(args.actor or "unknown")),
    )
    payload = {"ok": True, "action": "init", "store": info}
    _emit(payload, pretty=pretty, action="init")
    return 0


def _read_template(path: Path, *, max_bytes: int = 256 * 1024) -> str:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return _read_text_limited(path, max_bytes=max_bytes)


def _cmd_template(args: argparse.Namespace, *, repo_root: Path, pretty: bool) -> int:
    store = ReportStore(repo_root=repo_root)
    shipped_data = store.paths.shipped_data_template_path
    shipped_body = store.paths.shipped_body_template_path
    fmt = str(args.format or "").strip().lower()
    if fmt not in {"yaml", "md"}:
        raise ValueError("--format must be yaml or md")
    text = _read_template(shipped_data if fmt == "yaml" else shipped_body)
    payload = {"ok": True, "action": "template", "format": fmt, "template_text": text}
    _emit(payload, pretty=pretty, action="template")
    return 0


def _cmd_create(args: argparse.Namespace, *, repo_root: Path, pretty: bool) -> int:
    actor = str(args.actor or "").strip() or "unknown"
    store = ReportStore(repo_root=repo_root)
    _run_report_mutation_with_retry("create.init", lambda: store.ensure_initialized(actor=actor))

    raw: Dict[str, Any] = {}
    if args.data_json:
        raw = json.loads(str(args.data_json))
        if not isinstance(raw, Mapping):
            raise ValueError("data-json must be a JSON object")
        raw = dict(raw)
    elif args.data_file:
        raw = _load_data_file(Path(str(args.data_file)).expanduser())
    else:
        raise ValueError("provide --data-json or --data-file")

    data, errs = _normalize_create_request(
        raw,
        links_add=list(args.link or []),
        scripts_add=list(args.script or []),
        tags_add=list(args.tag or []),
    )
    if errs:
        raise ValueError("; ".join(errs[:10]))

    report_id = new_report_id()
    created_at = utc_now_z()

    # Attachments first (ensures assets exist if YAML references them).
    max_attach = max(1, int(os.getenv("REPORT_MAX_ATTACHMENT_BYTES", "20000000") or "20000000"))
    assets_rel_paths, copied = _copy_attachments(
        repo_root=repo_root,
        report_id=report_id,
        attach_paths=list(args.attach or []),
        max_bytes=max_attach,
    )

    md_rel_path = f"artifacts/reports/report/{report_id}/report.md"
    md_abs_path = (repo_root / md_rel_path).resolve()
    md_abs_path.parent.mkdir(parents=True, exist_ok=True)

    body_md = ""
    if args.body_md_text:
        body_md = str(args.body_md_text)
    elif args.body_md_file:
        body_md = _read_text_limited(Path(str(args.body_md_file)).expanduser(), max_bytes=2 * 1024 * 1024)
    if not body_md.strip():
        # Deterministic derived body as a safe default.
        tmp_doc = _build_report_doc(
            report_id=report_id,
            actor=actor,
            created_at_utc=created_at,
            status="draft",
            create_data=data,
            md_rel_path=md_rel_path,
            assets_rel_paths=assets_rel_paths,
        )
        body_md = format_report_markdown(tmp_doc)

    md_abs_path.write_text(body_md, encoding="utf-8")

    doc = _build_report_doc(
        report_id=report_id,
        actor=actor,
        created_at_utc=created_at,
        status="draft",
        create_data=data,
        md_rel_path=md_rel_path,
        assets_rel_paths=assets_rel_paths,
    )

    (yaml_path, warnings), _ = _run_report_mutation_with_retry(
        "create.write-report",
        lambda: store.write_report_doc(report_id, doc, actor=actor),
    )
    _run_report_mutation_with_retry(
        "create.append-event",
        lambda: store.append_event(
            {
                "ts_utc": created_at,
                "kind": "report_created",
                "report_id": report_id,
                "actor": actor,
                "title": str(doc.get("report", {}).get("title", "")),
            },
            actor=actor,
        ),
    )

    payload = {
        "ok": True,
        "action": "create",
        "report_id": report_id,
        "status": "draft",
        "yaml_path": str(yaml_path),
        "md_rel_path": md_rel_path,
        "assets_rel_paths": assets_rel_paths,
        "warnings": warnings,
        "attachments": copied,
        "portal_urls": {
            "report_page": f"/reports/{report_id}",
            "repo_file_md": f"/repo/file?path={md_rel_path}",
        },
    }
    _emit(payload, pretty=pretty, action="create")
    return 0


def _cmd_list(args: argparse.Namespace, *, repo_root: Path, pretty: bool) -> int:
    store = ReportStore(repo_root=repo_root)
    _run_report_mutation_with_retry(
        "list.init",
        lambda: store.ensure_initialized(actor=str(args.actor or "unknown")),
    )
    idx = None
    if args.rebuild:
        idx, _ = _run_report_mutation_with_retry(
            "list.rebuild-index",
            lambda: store.rebuild_index(actor=str(args.actor or "unknown")),
        )
    # Prefer index.yml (rebuilt or existing)
    try:
        import yaml  # local import
    except Exception as exc:
        raise ReportStoreError("PyYAML is required to list reports.") from exc
    index_path = store.paths.index_path
    index_doc = yaml.safe_load(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    reports = index_doc.get("reports") if isinstance(index_doc, Mapping) else None
    if not isinstance(reports, list):
        reports = []

    kind = normalize_kind(args.kind) if args.kind else ""
    status = normalize_status(args.status) if args.status else ""
    tag = str(args.tag or "").strip()
    q = str(args.q or "").strip().lower()
    link_filter = str(args.link or "").strip()
    if link_filter and not LINK_TOKEN_RE.fullmatch(link_filter):
        raise ValueError("invalid --link token")

    out: List[Dict[str, Any]] = []
    for r in reports:
        if not isinstance(r, Mapping):
            continue
        if kind and str(r.get("kind") or "").strip().lower() != kind:
            continue
        if status and str(r.get("status") or "").strip().lower() != status:
            continue
        if tag:
            tags = r.get("tags") if isinstance(r.get("tags"), list) else []
            if tag not in [str(x).strip() for x in tags if str(x).strip()]:
                continue
        if link_filter:
            links = r.get("links") if isinstance(r.get("links"), list) else []
            if link_filter not in [str(x).strip() for x in links if str(x).strip()]:
                continue
        if q:
            blob = " ".join(
                [
                    str(r.get("report_id") or ""),
                    str(r.get("title") or ""),
                    str(r.get("kind") or ""),
                    " ".join(str(x) for x in (r.get("tags") or []) if str(x).strip()),
                ]
            ).lower()
            if q not in blob:
                continue
        out.append(dict(r))

    limit = max(1, min(int(args.limit or 100), 2000))
    out = out[:limit]
    payload = {"ok": True, "action": "list", "count": len(out), "reports": out, "index_path": str(store.paths.index_path)}
    _emit(payload, pretty=pretty, action="list")
    return 0


def _cmd_get(args: argparse.Namespace, *, repo_root: Path, pretty: bool) -> int:
    store = ReportStore(repo_root=repo_root)
    doc, path = store.load_report(str(args.id))
    report = doc.get("report") if isinstance(doc, Mapping) else {}
    links, links_warn = extract_links_from_doc(doc)
    payload = {
        "ok": True,
        "action": "get",
        "report_id": str(args.id),
        "yaml_path": str(path),
        "md_rel_path": str(report.get("content", {}).get("md_rel_path") if isinstance(report, Mapping) else ""),
        "links": list(links),
        "link_warnings": list(links_warn or []),
        "report": report,
    }
    _emit(payload, pretty=pretty, action="get")
    return 0


_MD_STATUS_LINE_RE = re.compile(r"^- \*\*Status:\*\* `([^`]*)`\s*$")


def _patch_markdown_status_line(md_text: str, *, status: str) -> Tuple[str, bool]:
    """
    Update (or insert) the `Status` line in a report.md header block.

    This intentionally only touches the metadata header so an operator-provided
    custom body (create supports --body-md-*) remains intact.
    """
    lines = (md_text or "").splitlines()
    if not lines:
        return (md_text, False)

    header_scan_limit = 80
    found_idxs: List[int] = []
    for idx, line in enumerate(lines[:header_scan_limit]):
        if _MD_STATUS_LINE_RE.match(line):
            found_idxs.append(idx)

    desired_line = f"- **Status:** `{status}`"
    if found_idxs:
        changed = False
        first_idx = found_idxs[0]
        if lines[first_idx] != desired_line:
            lines[first_idx] = desired_line
            changed = True
        # Remove duplicates (keep the first).
        for idx in reversed(found_idxs[1:]):
            del lines[idx]
            changed = True
        if not changed:
            return (md_text, False)
        return ("\n".join(lines) + ("\n" if md_text.endswith("\n") else ""), True)

    # Insert after Kind/Report ID when present; otherwise best-effort near the top.
    insert_at = None
    for idx, line in enumerate(lines[:header_scan_limit]):
        if line.startswith("- **Kind:**"):
            insert_at = idx + 1
            break
        if line.startswith("- **Report ID:**"):
            insert_at = idx + 1
    if insert_at is None:
        insert_at = min(2, len(lines))
    lines.insert(insert_at, desired_line)
    return ("\n".join(lines) + ("\n" if md_text.endswith("\n") else ""), True)


def _sync_report_md_status(*, repo_root: Path, report: Mapping[str, Any], status: str) -> Tuple[bool, str]:
    content = report.get("content")
    if not isinstance(content, Mapping):
        return (False, "report.content missing; cannot locate md_rel_path")
    md_rel_path = str(content.get("md_rel_path") or "").strip()
    if not md_rel_path:
        return (False, "report.content.md_rel_path missing; cannot update report.md status")

    rr = Path(repo_root).resolve()
    md_rel = Path(md_rel_path)
    if md_rel.is_absolute():
        return (False, f"md_rel_path must be repo-relative: {md_rel_path}")
    if ".." in md_rel.parts:
        return (False, f"md_rel_path contains '..': {md_rel_path}")
    # Do NOT resolve() here: artifacts/ is commonly a symlink to warm-offload paths,
    # and we still want to allow syncing the derived markdown under that symlink.
    md_abs = rr / md_rel

    if not md_abs.exists():
        return (False, f"report.md missing: {md_rel_path}")

    try:
        text = md_abs.read_text(encoding="utf-8")
    except Exception as exc:
        return (False, f"failed to read report.md: {md_rel_path}: {type(exc).__name__}: {exc}")

    patched, changed = _patch_markdown_status_line(text, status=status)
    if not changed:
        return (False, "")

    try:
        md_abs.write_text(patched, encoding="utf-8")
    except Exception as exc:
        return (False, f"failed to write report.md: {md_rel_path}: {type(exc).__name__}: {exc}")

    return (True, "")


def _cmd_finalize(args: argparse.Namespace, *, repo_root: Path, pretty: bool) -> int:
    actor = str(args.actor or "").strip() or "unknown"
    store = ReportStore(repo_root=repo_root)
    doc, _ = store.load_report(str(args.id))
    report = doc.get("report")
    if not isinstance(report, MutableMapping):
        report = dict(report) if isinstance(report, Mapping) else {}
        doc["report"] = report

    status = str(report.get("status") or "").strip().lower() or "draft"
    if status == "final":
        md_updated, md_warn = _sync_report_md_status(repo_root=repo_root, report=report, status="final")
        payload = {
            "ok": True,
            "action": "finalize",
            "report_id": str(args.id),
            "status": "final",
            "already_final": True,
            "md_updated": md_updated,
            "warnings": [md_warn] if md_warn else [],
        }
        _emit(payload, pretty=pretty, action="finalize")
        return 0

    lint_ok, lint_errs, _ = store.lint_report(str(args.id))
    if not lint_ok:
        raise ValueError(f"cannot finalize: report failed quality gate: {'; '.join(lint_errs[:12])}")

    report["status"] = "final"
    report["finalized_at_utc"] = utc_now_z()

    (yaml_path, warnings), _ = _run_report_mutation_with_retry(
        "finalize.write-report",
        lambda: store.write_report_doc(str(args.id), doc, actor=actor),
    )
    md_updated, md_warn = _sync_report_md_status(repo_root=repo_root, report=report, status="final")
    if md_warn:
        warnings = list(warnings) + [md_warn]
    _run_report_mutation_with_retry(
        "finalize.append-event",
        lambda: store.append_event(
            {
                "ts_utc": utc_now_z(),
                "kind": "report_finalized",
                "report_id": str(args.id),
                "actor": actor,
                "title": str(report.get("title") or ""),
            },
            actor=actor,
        ),
    )
    payload = {
        "ok": True,
        "action": "finalize",
        "report_id": str(args.id),
        "status": "final",
        "yaml_path": str(yaml_path),
        "md_updated": md_updated,
        "warnings": warnings,
    }
    _emit(payload, pretty=pretty, action="finalize")
    return 0


def _cmd_revise(args: argparse.Namespace, *, repo_root: Path, pretty: bool) -> int:
    prior = str(args.from_id or "").strip()
    if not REPORT_ID_RE.fullmatch(prior):
        raise ValueError("--from must be a report id")
    # Load prior for convenience defaults.
    store = ReportStore(repo_root=repo_root)
    doc, _ = store.load_report(prior)
    report = doc.get("report") if isinstance(doc, Mapping) else {}

    raw: Dict[str, Any] = {}
    if args.data_json:
        raw = json.loads(str(args.data_json))
        if not isinstance(raw, Mapping):
            raise ValueError("data-json must be a JSON object")
        raw = dict(raw)
    elif args.data_file:
        raw = _load_data_file(Path(str(args.data_file)).expanduser())
    else:
        raw = {}

    if "title" not in raw:
        raw["title"] = f"Revision of {str(report.get('title') or prior)}"
    if "kind" not in raw and isinstance(report, Mapping):
        raw["kind"] = str(report.get("kind") or "other")
    if "tags" not in raw and isinstance(report, Mapping):
        raw["tags"] = list(report.get("tags") or [])
    if "summary" not in raw:
        raw["summary"] = list(report.get("summary") or []) if isinstance(report, Mapping) else []
        if not raw["summary"]:
            raw["summary"] = [
                f"Revision scope: update prior report {prior} with new evidence and KPI interpretation.",
                "This revision preserves prior context but updates claims, metrics, and operational implications.",
            ]
    if "insights" not in raw:
        raw["insights"] = list(report.get("insights") or []) if isinstance(report, Mapping) else []
        if not raw["insights"]:
            raw["insights"] = [
                "Primary changes versus baseline are explicitly documented in KPI-level deltas.",
                "Evidence coverage is refreshed to prevent stale conclusions from prior runs.",
                "Operational implications and residual risk posture are re-assessed.",
            ]
    if "evidence" not in raw:
        raw["evidence"] = list(report.get("evidence") or []) if isinstance(report, Mapping) else []
        if not raw["evidence"]:
            raw["evidence"] = [
                {
                    "kind": "report",
                    "ref": f"report:{prior}",
                    "note": "Prior report reference used as baseline for this revision.",
                }
            ]
    if "context" not in raw and isinstance(report, Mapping) and isinstance(report.get("context"), Mapping):
        raw["context"] = dict(report.get("context") or {})
    if "results" not in raw and isinstance(report, Mapping) and isinstance(report.get("results"), Mapping):
        raw["results"] = dict(report.get("results") or {})
    if "limitations" not in raw and isinstance(report, Mapping):
        raw["limitations"] = list(report.get("limitations") or [])
    if "risks" not in raw and isinstance(report, Mapping):
        raw["risks"] = list(report.get("risks") or [])
    if "next_actions" not in raw and isinstance(report, Mapping):
        raw["next_actions"] = list(report.get("next_actions") or [])
    if "repro" not in raw and isinstance(report, Mapping):
        raw["repro"] = list(report.get("repro") or [])

    data, errs = _normalize_create_request(
        raw,
        links_add=list(args.link or []) + [f"report:{prior}"],
        scripts_add=list(args.script or []),
        tags_add=list(args.tag or []),
    )
    if errs:
        raise ValueError("; ".join(errs[:10]))

    # Delegate to create with derived body.
    ns = argparse.Namespace(
        actor=args.actor,
        data_json=json.dumps(data),
        data_file=None,
        body_md_text=args.body_md_text,
        body_md_file=args.body_md_file,
        attach=args.attach,
        link=[],
        script=[],
        tag=[],
    )
    return _cmd_create(ns, repo_root=repo_root, pretty=pretty)


def _cmd_rebuild_index(args: argparse.Namespace, *, repo_root: Path, pretty: bool) -> int:
    store = ReportStore(repo_root=repo_root)
    idx, _ = _run_report_mutation_with_retry(
        "rebuild-index",
        lambda: store.rebuild_index(actor=str(args.actor or "unknown")),
    )
    payload = {"ok": True, "action": "rebuild-index", "index_path": str(store.paths.index_path), "count": len(idx.get("reports") or [])}
    _emit(payload, pretty=pretty, action="rebuild-index")
    return 0


def _cmd_lint(args: argparse.Namespace, *, repo_root: Path, pretty: bool) -> int:
    store = ReportStore(repo_root=repo_root)
    ids: List[str] = []
    if args.id:
        ids = [str(args.id)]
    else:
        ids = store.list_report_ids()
    ok_all = True
    results: List[Dict[str, Any]] = []
    for rid in ids:
        ok, errs, _ = store.lint_report(rid)
        ok_all = ok_all and ok
        results.append({"report_id": rid, "ok": ok, "errors": errs[:50]})
    payload = {"ok": ok_all, "action": "lint", "count": len(results), "results": results}
    _emit(payload, pretty=pretty, action="lint")
    return 0 if ok_all else 2


def build_parser() -> argparse.ArgumentParser:
    p = _ArgumentParser(prog="report", add_help=True)
    p.add_argument("--json-pretty", action="store_true", help="Pretty-print JSON output.")
    p.add_argument("--json-compact", action="store_true", help="Compact JSON output (default).")
    p.add_argument("--actor", default=os.getenv("USER", "unknown"), help="Actor name (default: $USER).")

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("schema", help="Emit JSON schema.").set_defaults(_handler="schema")
    sub.add_parser("init", help="Initialize report store.").set_defaults(_handler="init")

    tp = sub.add_parser("template", help="Emit starter templates.")
    tp.add_argument("--format", required=True, choices=["yaml", "md"], help="Template format.")
    tp.set_defaults(_handler="template")

    cp = sub.add_parser("create", help="Create a report.")
    g = cp.add_mutually_exclusive_group(required=True)
    g.add_argument("--data-json", help="Create request as JSON object string.")
    g.add_argument("--data-file", help="Create request as JSON/YAML file.")
    cp.add_argument("--body-md-text", help="Optional report body (Markdown) as string.")
    cp.add_argument("--body-md-file", help="Optional report body (Markdown) from file.")
    cp.add_argument("--attach", action="append", default=[], help="Attachment file path (repeatable).")
    cp.add_argument("--link", action="append", default=[], help="Extra link token kind:id (repeatable).")
    cp.add_argument("--script", action="append", default=[], help="Convenience: add script:<path> link (repeatable).")
    cp.add_argument("--tag", action="append", default=[], help="Extra tag (repeatable).")
    cp.set_defaults(_handler="create")

    lp = sub.add_parser("list", help="List reports.")
    lp.add_argument("--kind", choices=list(REPORT_KINDS), help="Filter by kind.")
    lp.add_argument("--status", choices=list(REPORT_STATUSES), help="Filter by status.")
    lp.add_argument("--tag", help="Filter by tag.")
    lp.add_argument("--q", help="Substring query over id/title/kind/tags.")
    lp.add_argument("--link", help="Filter by exact link token.")
    lp.add_argument("--limit", type=int, default=100, help="Max results (default 100).")
    lp.add_argument("--rebuild", action="store_true", help="Rebuild index before listing.")
    lp.set_defaults(_handler="list")

    gp = sub.add_parser("get", help="Get one report.")
    gp.add_argument("--id", required=True, help="Report id.")
    gp.set_defaults(_handler="get")

    fp = sub.add_parser("finalize", help="Finalize (freeze) a report.")
    fp.add_argument("--id", required=True, help="Report id.")
    fp.set_defaults(_handler="finalize")

    rp = sub.add_parser("revise", help="Create a revision report referencing a prior report.")
    rp.add_argument("--from", dest="from_id", required=True, help="Prior report id.")
    rp.add_argument("--data-json", help="Optional override create request JSON.")
    rp.add_argument("--data-file", help="Optional override create request file.")
    rp.add_argument("--body-md-text", help="Optional report body (Markdown) as string.")
    rp.add_argument("--body-md-file", help="Optional report body (Markdown) from file.")
    rp.add_argument("--attach", action="append", default=[], help="Attachment file path (repeatable).")
    rp.add_argument("--link", action="append", default=[], help="Extra link token kind:id (repeatable).")
    rp.add_argument("--script", action="append", default=[], help="Convenience: add script:<path> link (repeatable).")
    rp.add_argument("--tag", action="append", default=[], help="Extra tag (repeatable).")
    rp.set_defaults(_handler="revise")

    sub.add_parser("rebuild-index", help="Rebuild index from reports.").set_defaults(_handler="rebuild-index")

    lint = sub.add_parser("lint", help="Validate report invariants.")
    lint.add_argument("--id", help="Report id (default: all).")
    lint.set_defaults(_handler="lint")

    return p


def main(*, argv: Sequence[str], repo_root: Path) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv))
        pretty = bool(args.json_pretty) if hasattr(args, "json_pretty") else False
        if bool(getattr(args, "json_compact", False)):
            pretty = False

        handler = str(getattr(args, "_handler", "") or "").strip()
        if handler == "schema":
            return _cmd_schema(pretty=pretty)
        if handler == "init":
            return _cmd_init(args, repo_root=repo_root, pretty=pretty)
        if handler == "template":
            return _cmd_template(args, repo_root=repo_root, pretty=pretty)
        if handler == "create":
            return _cmd_create(args, repo_root=repo_root, pretty=pretty)
        if handler == "list":
            return _cmd_list(args, repo_root=repo_root, pretty=pretty)
        if handler == "get":
            return _cmd_get(args, repo_root=repo_root, pretty=pretty)
        if handler == "finalize":
            return _cmd_finalize(args, repo_root=repo_root, pretty=pretty)
        if handler == "revise":
            return _cmd_revise(args, repo_root=repo_root, pretty=pretty)
        if handler == "rebuild-index":
            return _cmd_rebuild_index(args, repo_root=repo_root, pretty=pretty)
        if handler == "lint":
            return _cmd_lint(args, repo_root=repo_root, pretty=pretty)

        raise _ArgparseError("unknown command", parser.format_usage())
    except _ArgparseError as exc:
        payload = _error_payload("invalid_arguments", exc.message, error_code="ARGPARSE_ERROR", usage=exc.usage)
        sys.stdout.write(_json_dumps(payload, pretty=True) + "\n")
        return 2
    except _GovernanceFileLockTimeout as exc:
        payload = _error_payload("lock_timeout", str(exc), error_code="LOCK_TIMEOUT", retryable=True)
        _emit(payload, pretty=True, action="error")
        return 1
    except (ReportStoreError, ValueError) as exc:
        payload = _error_payload("error", str(exc), error_code="REPORT_ERROR", retryable=False)
        _emit(payload, pretty=True, action="error")
        return 1
    except json.JSONDecodeError as exc:
        payload = _error_payload("invalid_json", str(exc), error_code="INVALID_JSON", retryable=False)
        _emit(payload, pretty=True, action="error")
        return 1
    except Exception as exc:
        payload = _error_payload("exception", f"{type(exc).__name__}: {exc}", error_code="EXCEPTION", retryable=True)
        _emit(payload, pretty=True, action="error")
        return 1
