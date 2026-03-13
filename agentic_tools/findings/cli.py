"""
Contract
- Purpose: Agent-facing JSON CLI for creating/searching Findings entries.
- Inputs: CLI args; writes YAML under artifacts/stores/findings (legacy: state/findings).
- Outputs: JSON-only stdout for success/error.
- Invariants: All created findings validate against the store schema.
- Error semantics: Non-zero exit with {"ok":false,...} on stdout.
- External boundaries: artifacts/stores/ filesystem; optional git for commit SHA.
"""

##REFACTOR: 2026-01-16##

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from agentic_tools._governance_locking import FileLockTimeout as _GovernanceFileLockTimeout
from agentic_tools._governance_locking import resolve_retry_budget as _resolve_retry_budget
from agentic_tools._governance_locking import resolve_retry_poll as _resolve_retry_poll
from agentic_tools._governance_locking import run_mutation_with_retry as _run_governance_mutation_with_retry
from .store import (
    ALLOWED_MODES,
    ALLOWED_REPO_AREAS,
    ALLOWED_SCOPES,
    ALLOWED_SEVERITIES,
    ALLOWED_STATUSES,
    ALLOWED_TARGET_KINDS,
    FindingsStore,
    LINK_TOKEN_RE,
)

from agentic_tools.codexctx import append_codex_links
from agentic_tools.process_hints import enrich_payload_with_hints
from agentic_tools.process_hints import hints_schema_fragment
from agentic_tools.process_hints import next_steps_for_finding_doc


class _CliUsageError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = int(status)
        self.message = str(message or "")


class _StoreLockTimeout(Exception):
    pass


_ERROR_CODE_TOKEN_RE = re.compile(r"[^A-Za-z0-9]+")


def _git_toplevel(start: Path) -> Optional[Path]:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.5,
            check=False,
        )
        if p.returncode != 0:
            return None
        s = (p.stdout or "").strip()
        if not s:
            return None
        return Path(s).expanduser().resolve()
    except Exception:
        return None


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()

    # Prefer Git top-level when available (more robust than marker scanning),
    # but keep the historical marker contract as a guard against mis-detecting
    # an unrelated git repo.
    git_root = _git_toplevel(start)
    if git_root is not None:
        try:
            if (git_root / "AGENTS.md").exists() and (git_root / "agentic_tools" / "findings" / "cli.py").exists():
                return git_root
        except Exception:
            pass

    for cand in [start, *start.parents]:
        if (cand / "AGENTS.md").exists() and (cand / "agentic_tools" / "findings" / "cli.py").exists():
            return cand
    return start


def _json_out(payload: Dict[str, Any], pretty: bool) -> None:
    payload = enrich_payload_with_hints(payload, tool="findings")
    try:
        if pretty:
            sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        else:
            sys.stdout.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass


def _derive_error_code(error: str) -> str:
    token = _ERROR_CODE_TOKEN_RE.sub("_", str(error or "").strip()).strip("_").upper()
    return token or "UNKNOWN_ERROR"


def _default_retryable(error_code: str) -> bool:
    code = str(error_code or "").upper()
    if code in {"LOCK_TIMEOUT", "TIMEOUT", "IO_ERROR"}:
        return True
    if "TIMEOUT" in code or "LOCK" in code:
        return True
    return False


def _error_payload(
    error: str,
    detail: str,
    *,
    error_code: str = "",
    retryable: Optional[bool] = None,
) -> Dict[str, Any]:
    normalized_error = str(error or "")
    normalized_error_code = str(error_code or "").strip() or _derive_error_code(normalized_error)
    out: Dict[str, Any] = {
        "ok": False,
        "error": normalized_error,
        "error_code": normalized_error_code,
        "detail": str(detail or ""),
        "retryable": _default_retryable(normalized_error_code) if retryable is None else bool(retryable),
    }
    return out


def _classify_exception(exc: BaseException) -> Tuple[str, Optional[bool]]:
    # Keep stable `error` strings; add optional machine hints (`error_code`, `retryable`) only.
    if isinstance(exc, _CliUsageError):
        return ("ARGPARSE_ERROR", False)
    if isinstance(exc, (_StoreLockTimeout, _GovernanceFileLockTimeout)):
        return ("LOCK_TIMEOUT", True)
    if isinstance(exc, PermissionError):
        return ("PERMISSION_DENIED", False)
    if isinstance(exc, FileNotFoundError):
        return ("NOT_FOUND", False)
    if isinstance(exc, TimeoutError):
        return ("TIMEOUT", True)
    if isinstance(exc, ValueError):
        return ("VALIDATION_ERROR", False)
    if isinstance(exc, OSError):
        return ("IO_ERROR", True)
    return ("UNHANDLED_EXCEPTION", None)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _mutation_retry_budget_sec() -> float:
    return _resolve_retry_budget(
        retry_timeout_env="CAIA_FINDINGS_CLI_RETRY_TIMEOUT_SEC",
        lock_timeout_envs=("CAIA_FINDINGS_LOCK_TIMEOUT_SEC", "FINDINGS_LOCK_TIMEOUT"),
        default_lock_timeout=30.0,
    )


def _mutation_retry_poll_sec() -> float:
    return _resolve_retry_poll(
        retry_poll_env="CAIA_FINDINGS_CLI_RETRY_POLL_SEC",
        default_poll_sec=0.25,
    )


def _emit_retry(action: str, error_code: str, attempt: int, sleep_s: float, exc: BaseException) -> None:
    print(
        f"findings {action}: retrying after {error_code} (attempt={attempt}, sleep_s={sleep_s:.3f})",
        file=sys.stderr,
    )


def _run_mutation_with_retry(action: str, op: Callable[[], Any]) -> Tuple[Any, Dict[str, Any]]:
    return _run_governance_mutation_with_retry(
        action=action,
        op=op,
        classify_exception=_classify_exception,
        retry_budget_sec=_mutation_retry_budget_sec(),
        retry_poll_sec=_mutation_retry_poll_sec(),
        emit_retry=lambda error_code, attempt, sleep_s, exc: _emit_retry(
            action,
            error_code,
            attempt,
            sleep_s,
            exc,
        ),
    )


def _git_short_sha(repo_root: Path) -> str:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
            check=False,
        )
        sha = (p.stdout or "").strip()
        return sha
    except Exception:
        return ""


def _parse_json_arg(raw: str) -> Dict[str, Any]:
    s = (raw or "").strip()
    if not s:
        return {}
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object")
    return obj


def _as_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _pick_first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _normalize_finding_explain(doc: Dict[str, Any]) -> Dict[str, Any]:
    finding = doc.get("finding")
    if not isinstance(finding, dict):
        finding = {}
    target = finding.get("target")
    if not isinstance(target, dict):
        target = {}
    classification = finding.get("classification")
    if not isinstance(classification, dict):
        classification = {}
    narrative = doc.get("narrative")
    if not isinstance(narrative, dict):
        narrative = {}
    actions = doc.get("actions")
    if not isinstance(actions, list):
        actions = []
    resolution = doc.get("resolution")
    if not isinstance(resolution, dict):
        resolution = {}
    related = finding.get("related")
    if not isinstance(related, dict):
        related = {}

    links = _dedupe_keep_order(_as_string_list(doc.get("links")))
    how: List[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        title = str(action.get("title") or "").strip()
        desc = str(action.get("description") or "").strip()
        acceptance = _as_string_list(action.get("acceptance_criteria"))
        if title and desc:
            how.append(f"{title}: {desc}")
        elif title:
            how.append(title)
        elif desc:
            how.append(desc)
        how.extend(acceptance)
    if not how:
        how.extend(
            _as_string_list(
                [
                    narrative.get("expected"),
                    narrative.get("observed"),
                    resolution.get("notes"),
                ]
            )
        )
    if not how:
        how.append("Convert finding into actionable tasks and validate closure via fixreport.")

    guardrails: List[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        for step in action.get("validation_steps") or []:
            if not isinstance(step, dict):
                continue
            command = str(step.get("command") or "").strip()
            check = str(step.get("check") or "").strip()
            if command:
                guardrails.append(f"Validate via command: {command}")
            if check:
                guardrails.append(f"Validation check: {check}")
    status = str(finding.get("status") or "").strip().lower()
    if status == "blocked":
        guardrails.append("Do not close while finding.status=blocked.")
    if not guardrails:
        guardrails.append("Resolve with verifiable validation before marking as resolved.")

    evidence: List[str] = []
    evidence.extend(_as_string_list(related.get("task_ids")))
    evidence.extend(_as_string_list(related.get("hypothesis_ids")))
    evidence.extend(_as_string_list(related.get("fixreport_ids")))
    evidence.extend([x for x in links if any(x.startswith(prefix) for prefix in ("task:", "fixreport:", "hypothesis:", "script:", "audit_id:", "audit_file:"))])

    conflicts: List[str] = []
    if status in {"blocked", "duplicate", "wont_fix"}:
        conflicts.append(f"status={status}")
    root_cause = str(classification.get("root_cause") or "").strip()
    if not root_cause:
        conflicts.append("root_cause missing")

    return {
        "what": _pick_first_nonempty(finding.get("title"), finding.get("summary"), finding.get("id"), "Finding"),
        "why": _pick_first_nonempty(
            classification.get("root_cause"),
            classification.get("symptom"),
            narrative.get("impact"),
            finding.get("summary"),
            "Observed issue requiring mitigation.",
        ),
        "how": _dedupe_keep_order(how),
        "guardrails": _dedupe_keep_order(guardrails),
        "evidence": _dedupe_keep_order(evidence),
        "conflicts": _dedupe_keep_order(conflicts),
        "links": links,
        "target": {
            "kind": str(target.get("kind") or ""),
            "path": str(target.get("path") or ""),
            "systemd_unit": str(target.get("systemd_unit") or ""),
        },
    }


def _schema_payload() -> Dict[str, Any]:
    return {
        "semantics": {
            "finding": {
                "is": "Observed issue/bug/smell/risk that is actionable and should be addressed (pre-fix intake).",
                "is_not": "A theory about why something happens. The root cause can be UNKNOWN and should be validated elsewhere.",
                "addressed_checkmark": {
                    "meaning": "finding.addressed=true means the finding has been converted into a Task/UserTask/Work item (actioned), not that it is fixed.",
                    "fixed_is": "When fixed, link to a FixReport and set finding.status=resolved plus resolution.* fields.",
                },
            },
            "hypothesis": {
                "is": "Falsifiable statement/theory/prediction with evidence that can become supported/refuted over time.",
                "is_not": "The canonical list of bugs/smells to fix. Use Findings for the actionable pre-fix backlog.",
                "linking": "If a hypothesis explains a finding (or vice-versa), link them via links: ['finding:FND…', 'hypothesis:H…'].",
            },
        },
        "store": {
            "paths": {
                "reports_glob": "artifacts/stores/findings/reports/FND*.yml",
                "index": "artifacts/stores/findings/index.yml",
                "events": "artifacts/stores/findings/events.yml",
            }
        },
        "related_tools": {
            "tasks": "Convert findings into executable work and track lifecycle.",
            "fixreport": "Close findings with concrete remediation evidence.",
            "hypothesis": "Track explanatory theories linked to findings.",
            "rules": "Canonical governance constraints and long-term guardrails.",
            "ssot": "Canonical references for paths/subsystems.",
        },
        "recommended_workflow": [
            "findings overview",
            "findings list --status open --limit 50",
            "findings explain --id FNDxxxxx",
            "tasks add todo --link finding:FNDxxxxx --script <path> --title \"...\"",
        ],
        "finding": {
            "required": [
                "finding.id",
                "finding.status",
                "finding.severity",
                "finding.scope",
                "finding.mode",
                "finding.repo_area",
                "finding.target.kind",
                "finding.title",
                "finding.summary",
                "finding.provenance.discovered_at_utc",
            ],
            "optional": [
                "finding.addressed",
                "finding.addressed_at_utc",
            ],
            "notes": [
                "Use `links: [\"kind:id\", ...]` for Meta graph truth edges (see architecture/LINKS_CONTRACT.md).",
                "Prefer at least one script:<repo_path> link when the finding is about code.",
                "Use `findings address` / `findings unaddress` to toggle the checkmark state (distinct from finding.status).",
                "Boundary: Findings capture observed issues/smells; Hypotheses capture falsifiable theories/predictions + evidence.",
                "Index is rebuildable: `findings rebuild-index`.",
            ],
        },
        "allowed_values": {
            "finding.status": sorted(ALLOWED_STATUSES),
            "finding.severity": sorted(ALLOWED_SEVERITIES),
            "finding.scope": sorted(ALLOWED_SCOPES),
            "finding.mode": sorted(ALLOWED_MODES),
            "finding.repo_area": sorted(ALLOWED_REPO_AREAS),
            "finding.target.kind": sorted(ALLOWED_TARGET_KINDS),
        },
        "link_token_regex": LINK_TOKEN_RE.pattern,
    }


def _has_flag(argv: List[str], flag: str) -> bool:
    f = str(flag)
    for a in argv:
        if a == f:
            return True
        if a.startswith(f"{f}="):
            return True
    return False


def _validate_link_tokens(tokens: List[str]) -> None:
    for t in tokens:
        if not LINK_TOKEN_RE.fullmatch(t):
            raise ValueError(f"invalid link token: {t!r}")


def _validate_finding_id(fid: str) -> str:
    s = (fid or "").strip()
    if not s:
        raise ValueError("missing finding id")
    # Defense-in-depth against path traversal / path injection. Keep compatibility by only
    # rejecting path-like ids and obviously dangerous characters.
    if "\x00" in s:
        raise ValueError("invalid finding id")
    if "/" in s or "\\" in s:
        raise ValueError("invalid finding id")
    # Avoid dot-only / traversal-like ids.
    if s in {".", ".."}:
        raise ValueError("invalid finding id")
    return s


def _guard_report_path(store: FindingsStore, fid: str) -> None:
    # Best-effort guard: compute the canonical report path and ensure it stays inside reports dir.
    # This does not change store semantics; it blocks only traversal attempts.
    try:
        reports_dir = getattr(store.paths, "reports_dir", None)
        if not isinstance(reports_dir, Path):
            reports_dir = Path(store.paths.state_dir) / "reports"
        reports_dir = reports_dir.expanduser().resolve()

        # Expected naming from schema: reports/FND*.yml
        cand = (reports_dir / f"{fid}.yml").resolve()
        cand.relative_to(reports_dir)
    except Exception:
        raise ValueError("invalid finding id")


def _legacy_cli_lock_path_for_store(store: FindingsStore) -> Path:
    try:
        base = Path(store.paths.state_dir)
    except Exception:
        base = Path(".")
    return base / ".findings.lock"


def _cleanup_legacy_cli_lock(store: FindingsStore) -> bool:
    """
    Remove the deprecated CLI-local ``.findings.lock`` if and only if no process
    currently holds an advisory lock on it.

    The canonical findings store already serializes writes via
    ``agentic_tools.findings.store.FileLock`` on ``store.paths.lock_path``.
    Keeping a second lock file in the CLI creates a lock hierarchy the store
    never sees, which can strand governance commands behind a legacy file even
    though the canonical store lock would be recoverable.
    """
    lock_path = _legacy_cli_lock_path_for_store(store)
    if not lock_path.exists() or not lock_path.is_file():
        return False

    handle = None
    is_windows = os.name == "nt"
    acquired = False
    try:
        handle = open(lock_path, "a+b")
        try:
            if is_windows:
                import msvcrt  # type: ignore

                try:
                    handle.seek(0, os.SEEK_END)
                    if handle.tell() == 0:
                        handle.write(b"\0")
                        handle.flush()
                    handle.seek(0)
                except Exception:
                    pass
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # type: ignore

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            return False
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EBUSY):
                return False
            return False
    except FileNotFoundError:
        return False
    except Exception:
        return False
    finally:
        if acquired and handle is not None:
            try:
                if is_windows:
                    import msvcrt  # type: ignore

                    try:
                        handle.seek(0)
                    except Exception:
                        pass
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl  # type: ignore

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    try:
        lock_path.unlink()
        return True
    except FileNotFoundError:
        return True
    except Exception:
        return False


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _extract_summary_item(doc: Any, fallback_id: str = "") -> Optional[Dict[str, Any]]:
    if not isinstance(doc, dict):
        return None

    finding_obj = doc.get("finding")
    if isinstance(finding_obj, dict):
        finding = finding_obj
    else:
        # Some stores might flatten the finding object.
        finding = doc

    if not isinstance(finding, dict):
        return None

    target = finding.get("target") if isinstance(finding.get("target"), dict) else {}
    if not isinstance(target, dict):
        target = {}

    links_val = doc.get("links")
    if not isinstance(links_val, list):
        links_val = finding.get("links")
    if not isinstance(links_val, list):
        links_val = []

    finding_id = str(finding.get("id") or fallback_id or "").strip()
    if not finding_id:
        return None

    item: Dict[str, Any] = {
        "finding_id": finding_id,
        "status": str(finding.get("status") or "").strip(),
        "severity": str(finding.get("severity") or "").strip(),
        "scope": str(finding.get("scope") or "").strip(),
        "mode": str(finding.get("mode") or "").strip(),
        "repo_area": str(finding.get("repo_area") or "").strip(),
        "target_kind": str(target.get("kind") or "").strip(),
        "target_path": str(target.get("path") or "").strip(),
        "title": str(finding.get("title") or "").strip(),
        "summary": str(finding.get("summary") or "").strip(),
        "links": [str(x) for x in links_val if x],
    }

    # Preserve optional fields if present (non-breaking additive in list output).
    if "addressed" in finding:
        item["addressed"] = bool(finding.get("addressed"))
    if "addressed_at_utc" in finding and finding.get("addressed_at_utc") is not None:
        item["addressed_at_utc"] = str(finding.get("addressed_at_utc") or "")

    return item


def _scan_reports_summaries(store: FindingsStore) -> List[Dict[str, Any]]:
    reports_dir = getattr(store.paths, "reports_dir", None)
    if not isinstance(reports_dir, Path):
        reports_dir = Path(store.paths.state_dir) / "reports"

    try:
        reports_dir = reports_dir.expanduser().resolve()
    except Exception:
        reports_dir = Path(reports_dir)

    if not reports_dir.exists() or not reports_dir.is_dir():
        return []

    out: List[Dict[str, Any]] = []
    try:
        paths = sorted(reports_dir.glob("FND*.yml"))
    except Exception:
        paths = []

    for p in paths:
        try:
            raw = p.read_text(encoding="utf-8")
            doc = yaml.safe_load(raw) or {}
            item = _extract_summary_item(doc, fallback_id=p.stem)
            if item:
                out.append(item)
        except Exception:
            continue

    return out


def _new_argument_parser() -> argparse.ArgumentParser:
    # Python versions differ (exit_on_error was added in 3.9; color in 3.14).
    # We best-effort disable both "exit on error" and colored output for machine-friendly behavior.
    class _JsonArgumentParser(argparse.ArgumentParser):
        def error(self, message: str) -> None:
            raise _CliUsageError(2, message)

        def exit(self, status: int = 0, message: Optional[str] = None) -> None:
            if int(status) == 0:
                # Preserve historical help/version behavior.
                return super().exit(status=status, message=message)
            raise _CliUsageError(int(status), message or "")

    # Try most featureful ctor first.
    description = (
        "Findings CLI for bug/smell intake and lifecycle tracking.\n\n"
        "Related tools:\n"
        "  - tasks: convert findings into executable work.\n"
        "  - fixreport: close resolved findings with prevention evidence.\n"
        "  - hypothesis: track explanatory theories and supporting evidence.\n"
        "  - rules/ssot: govern mitigations and canonical constraints.\n\n"
        "Recommended workflow:\n"
        "  1) findings overview\n"
        "  2) findings list --status open --limit 50\n"
        "  3) findings explain --id FNDxxxxx\n"
        "  4) findings address --id FNDxxxxx after task conversion\n"
    )
    try:
        return _JsonArgumentParser(
            prog="findings",
            add_help=True,
            exit_on_error=False,
            color=False,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=description,
        )  # type: ignore[arg-type]
    except TypeError:
        pass
    try:
        return _JsonArgumentParser(
            prog="findings",
            add_help=True,
            exit_on_error=False,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=description,
        )  # type: ignore[arg-type]
    except TypeError:
        pass
    return _JsonArgumentParser(
        prog="findings",
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=description,
    )


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Allow global flags to appear after the subcommand (LLM/tool friendly).
    # Example: `findings schema --json-pretty`.
    _global_arity = {"--json-pretty": 0, "--repo-root": 1, "--state-dir": 1}
    moved: List[str] = []
    rest: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in _global_arity:
            n = _global_arity[a]
            if n == 0:
                moved.append(a)
                i += 1
                continue
            # If the value is missing, do NOT move the flag; keep it in-place so argparse
            # can report a coherent error (instead of mis-parsing the subcommand as the value).
            if i + n >= len(argv):
                rest.append(a)
                i += 1
                continue
            moved.append(a)
            moved.extend(argv[i + 1 : i + 1 + n])
            i += 1 + n
            continue
        if any(a.startswith(f"{k}=") for k in ("--repo-root", "--state-dir")):
            moved.append(a)
            i += 1
            continue
        rest.append(a)
        i += 1
    argv = moved + rest

    pretty_hint = _has_flag(argv, "--json-pretty")

    ap = _new_argument_parser()
    ap.add_argument("--repo-root", default="", help="Override repo root (default: auto-detect).")
    ap.add_argument(
        "--state-dir",
        default="",
        help="Override findings state dir (default: <repo>/artifacts/stores/findings; legacy: <repo>/state/findings).",
    )
    ap.add_argument("--json-pretty", action="store_true", help="Pretty-print JSON output.")

    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("schema", help="Print JSON schema and exit.")

    p_init = sub.add_parser("init", help="Initialize findings state store under artifacts/stores/findings/.")
    p_init.add_argument("--force", action="store_true", help="Overwrite index/events/template files.")

    sub.add_parser("rebuild-index", help="Rebuild artifacts/stores/findings/index.yml from reports.")

    sub.add_parser("lint", help="Validate all findings under artifacts/stores/findings/reports.")

    p_template = sub.add_parser("template", help="Return the shipped YAML template (as a string in JSON).")

    p_addr = sub.add_parser("address", help="Mark a finding as addressed (checkmark).")
    p_addr.add_argument("--id", required=True, help="Finding id (FND...).")
    p_addr.add_argument("--actor", default="", help="Actor/agent id (optional).")

    p_unaddr = sub.add_parser("unaddress", help="Mark a finding as unaddressed (remove checkmark).")
    p_unaddr.add_argument("--id", required=True, help="Finding id (FND...).")
    p_unaddr.add_argument("--actor", default="", help="Actor/agent id (optional).")

    p_create = sub.add_parser("create", help="Create a new Finding YAML file.")
    p_create.add_argument("--id", default="", help="Optional explicit finding id (default: auto).")
    p_create.add_argument(
        "--status",
        default="open",
        help=f"Finding status ({'|'.join(sorted(ALLOWED_STATUSES))}).",
    )
    p_create.add_argument(
        "--severity",
        default="p2",
        help=f"Severity tier ({'|'.join(sorted(ALLOWED_SEVERITIES))}).",
    )
    p_create.add_argument(
        "--scope",
        default="analytics",
        help=f"Scope ({'|'.join(sorted(ALLOWED_SCOPES))}).",
    )
    p_create.add_argument(
        "--mode",
        default="dev",
        help=f"Mode ({'|'.join(sorted(ALLOWED_MODES))}).",
    )
    p_create.add_argument(
        "--repo-area",
        default="services",
        help=f"Repo area ({'|'.join(sorted(ALLOWED_REPO_AREAS))}).",
    )
    p_create.add_argument(
        "--target-kind",
        default="script",
        help=f"Target kind ({'|'.join(sorted(ALLOWED_TARGET_KINDS))}).",
    )
    p_create.add_argument("--target-path", default="")
    p_create.add_argument("--systemd-unit", default="")
    p_create.add_argument("--title", required=True)
    p_create.add_argument("--summary", required=True)
    p_create.add_argument("--author", default="")
    p_create.add_argument("--skill", default="")
    p_create.add_argument("--run-id", default="")
    p_create.add_argument("--commit", default="")
    p_create.add_argument("--bug-type", default="")
    p_create.add_argument("--symptom", default="")
    p_create.add_argument("--root-cause", default="")
    p_create.add_argument("--failure-mode", default="")
    p_create.add_argument("--impact-area", default="")
    p_create.add_argument("--link", action="append", default=[], help="Add a links token 'kind:id' (repeatable).")
    p_create.add_argument("--narrative-json", default="", help="Optional JSON mapping for narrative section.")
    p_create.add_argument("--evidence-json", default="", help="Optional JSON mapping for evidence section.")

    p_get = sub.add_parser("get", help="Load a finding and return it as JSON.")
    p_get.add_argument("--id", required=True, help="Finding id (FND...).")

    p_explain = sub.add_parser(
        "explain",
        help="Explain a finding in normalized what/why/how form.",
    )
    p_explain.add_argument("--id", required=True, help="Finding id (FND...).")

    p_overview = sub.add_parser(
        "overview",
        help="Compact findings summary for agent orientation.",
    )
    p_overview.add_argument("--limit", type=int, default=300, help="Max findings to scan (default: %(default)s).")

    p_list = sub.add_parser("list", help="List finding summaries (from index.yml when available).")
    p_list.add_argument("--q", default="", help="Free-text match against id/title/summary/target_path/links.")
    p_list.add_argument("--severity", default="")
    p_list.add_argument("--status", default="")
    p_list.add_argument("--scope", default="")
    p_list.add_argument("--mode", default="")
    p_list.add_argument("--repo-area", default="")
    p_list.add_argument("--target-kind", default="")
    p_list.add_argument("--limit", type=int, default=200)

    p_update = sub.add_parser("update", help="Update an existing finding (shallow merge).")
    p_update.add_argument("--id", required=True, help="Finding id (FND...).")
    p_update.add_argument("--set-finding-json", default="", help="JSON mapping to merge into finding.")
    p_update.add_argument("--set-narrative-json", default="", help="JSON mapping to merge into narrative.")
    p_update.add_argument("--set-evidence-json", default="", help="JSON mapping to merge into evidence.")
    p_update.add_argument("--set-resolution-json", default="", help="JSON mapping to merge into resolution.")
    p_update.add_argument("--add-link", action="append", default=[], help="Add links token(s) (repeatable).")
    p_update.add_argument("--actor", default="")

    try:
        args = ap.parse_args(argv)
    except _CliUsageError as e:
        _json_out(
            _error_payload(
                "invalid_arguments",
                e.message,
                error_code="ARGPARSE_ERROR",
                retryable=False,
            ),
            pretty_hint,
        )
        return int(e.status) if int(e.status) != 0 else 2
    except argparse.ArgumentError as e:
        _json_out(
            _error_payload(
                "invalid_arguments",
                str(e),
                error_code="ARGPARSE_ERROR",
                retryable=False,
            ),
            pretty_hint,
        )
        return 2

    pretty = bool(args.json_pretty)
    repo_root = Path(args.repo_root).expanduser().resolve() if str(args.repo_root).strip() else _find_repo_root(Path.cwd())
    state_dir = Path(args.state_dir).expanduser().resolve() if str(args.state_dir).strip() else None

    try:
        store = FindingsStore(repo_root, state_dir=state_dir)
        _cleanup_legacy_cli_lock(store)

        if args.cmd == "schema":
            _json_out(
                {
                    "ok": True,
                    "schema": _schema_payload(),
                    "hint_envelope": hints_schema_fragment(),
                    "commands": {
                        "schema": "Print JSON schema and exit.",
                        "init": "Initialize findings state store.",
                        "template": "Return shipped YAML template (string in JSON).",
                        "create": "Create a new Finding YAML file.",
                        "overview": "Compact findings summary for quick orientation.",
                        "update": "Update an existing finding (shallow merge).",
                        "address": "Mark a finding as addressed (checkmark).",
                        "unaddress": "Mark a finding as unaddressed (remove checkmark).",
                        "get": "Get a finding as JSON.",
                        "explain": "Explain one finding in normalized what/why/how form.",
                        "list": "List finding summaries (index-backed).",
                        "lint": "Validate findings YAML files.",
                        "rebuild-index": "Rebuild index.yml from reports.",
                    },
                },
                pretty,
            )
            return 0

        if args.cmd == "init":
            _, retry_info = _run_mutation_with_retry("init", lambda: store.init(force=bool(args.force)))
            out = {"ok": True, "action": "init", "state_dir": str(store.paths.state_dir)}
            if retry_info:
                out["retry"] = retry_info
            _json_out(out, pretty)
            return 0

        if args.cmd == "template":
            tpl_path = store.paths.shipped_template_path
            tpl = tpl_path.read_text(encoding="utf-8") if tpl_path.exists() else ""
            _json_out({"ok": True, "action": "template", "template_path": str(tpl_path), "template_yaml": tpl}, pretty)
            return 0

        if args.cmd == "rebuild-index":
            (n, errors), retry_info = _run_mutation_with_retry("rebuild-index", store.rebuild_index)
            out = {"ok": True, "action": "rebuild-index", "count": n, "issues": errors}
            if retry_info:
                out["retry"] = retry_info
            _json_out(out, pretty)
            return 0

        if args.cmd == "lint":
            ok, issues = store.lint()
            _json_out({"ok": ok, "action": "lint", "issues": issues}, pretty)
            return 0 if ok else 1

        if args.cmd in {"address", "unaddress"}:
            fid = _validate_finding_id(str(args.id))
            _guard_report_path(store, fid)

            want = bool(args.cmd == "address")
            path, retry_info = _run_mutation_with_retry(
                str(args.cmd),
                lambda: store.set_addressed(
                    fid,
                    addressed=want,
                    actor=str(getattr(args, "actor", "")).strip() or None,
                ),
            )
            doc2: Dict[str, Any] = {}
            try:
                doc2 = store.load_report(fid) or {}
            except Exception:
                doc2 = {}
            next_steps = next_steps_for_finding_doc(action=str(args.cmd), finding_id=fid, doc=doc2) if doc2 else []
            out = {
                "ok": True,
                "action": args.cmd,
                "finding_id": fid,
                "addressed": want,
                "path": str(path),
                "next_steps": next_steps,
            }
            if retry_info:
                out["retry"] = retry_info
            _json_out(out, pretty)
            return 0

        if args.cmd == "create":
            links = [str(x).strip() for x in (args.link or []) if str(x).strip()]
            links = append_codex_links(links)
            _validate_link_tokens(links)

            explicit_id = str(args.id).strip()
            if explicit_id:
                _validate_finding_id(explicit_id)

            commit = (str(args.commit).strip() or _git_short_sha(repo_root) or "").strip()
            finding: Dict[str, Any] = {
                "id": explicit_id,
                "status": str(args.status).strip(),
                "severity": str(args.severity).strip(),
                "scope": str(args.scope).strip(),
                "mode": str(args.mode).strip(),
                "repo_area": str(args.repo_area).strip(),
                "target": {
                    "kind": str(args.target_kind).strip(),
                    "path": str(args.target_path).strip(),
                    "systemd_unit": str(args.systemd_unit).strip(),
                    "topics": [],
                    "tables_views": [],
                    "horizon_H_seconds": None,
                    "asset": "",
                    "venue": "",
                    "instrument": "",
                },
                "title": str(args.title).strip(),
                "summary": str(args.summary).strip(),
                "provenance": {
                    "discovered_at_utc": "",  # store fills
                    "author": str(args.author).strip(),
                    "skill": str(args.skill).strip(),
                    "run_id": str(args.run_id).strip(),
                    "commit": commit,
                },
                "classification": {
                    "bug_type": str(args.bug_type).strip(),
                    "symptom": str(args.symptom).strip(),
                    "root_cause": str(args.root_cause).strip(),
                    "failure_mode": str(args.failure_mode).strip(),
                    "impact_area": str(args.impact_area).strip(),
                },
                "related": {
                    "task_ids": [],
                    "hypothesis_ids": [],
                    "fixreport_ids": [],
                    "quantaudit": {"bundle": "", "report_id": "", "local_finding_id": ""},
                },
            }
            narrative = _parse_json_arg(str(args.narrative_json))
            evidence = _parse_json_arg(str(args.evidence_json))

            (fid, path), retry_info = _run_mutation_with_retry(
                "create",
                lambda: store.create(
                    finding=finding,
                    links=links or None,
                    narrative=narrative or None,
                    evidence=evidence or None,
                    actor=str(args.author).strip() or None,
                ),
            )
            doc2: Dict[str, Any] = {}
            try:
                doc2 = store.load_report(fid) or {}
            except Exception:
                doc2 = {}
            next_steps = next_steps_for_finding_doc(action="create", finding_id=fid, doc=doc2) if doc2 else []
            out = {"ok": True, "action": "create", "finding_id": fid, "path": str(path), "next_steps": next_steps}
            if retry_info:
                out["retry"] = retry_info
            _json_out(out, pretty)
            return 0

        if args.cmd == "update":
            fid = _validate_finding_id(str(args.id))
            _guard_report_path(store, fid)

            add_links = [str(x).strip() for x in (args.add_link or []) if str(x).strip()]
            add_links = append_codex_links(add_links)
            _validate_link_tokens(add_links)

            path, retry_info = _run_mutation_with_retry(
                "update",
                lambda: store.update(
                    fid,
                    set_finding=_parse_json_arg(str(args.set_finding_json)) or None,
                    set_narrative=_parse_json_arg(str(args.set_narrative_json)) or None,
                    set_evidence=_parse_json_arg(str(args.set_evidence_json)) or None,
                    set_resolution=_parse_json_arg(str(args.set_resolution_json)) or None,
                    add_links=add_links or None,
                    actor=str(args.actor).strip() or None,
                ),
            )
            doc2: Dict[str, Any] = {}
            try:
                doc2 = store.load_report(fid) or {}
            except Exception:
                doc2 = {}
            next_steps = next_steps_for_finding_doc(action="update", finding_id=fid, doc=doc2) if doc2 else []
            out = {"ok": True, "action": "update", "finding_id": fid, "path": str(path), "next_steps": next_steps}
            if retry_info:
                out["retry"] = retry_info
            _json_out(out, pretty)
            return 0

        if args.cmd == "get":
            fid = _validate_finding_id(str(args.id))
            _guard_report_path(store, fid)

            doc = store.load_report(fid)
            next_steps = next_steps_for_finding_doc(action="get", finding_id=fid, doc=doc or {}) if isinstance(doc, dict) else []
            _json_out({"ok": True, "action": "get", "finding_id": fid, "doc": doc, "next_steps": next_steps}, pretty)
            return 0

        if args.cmd == "explain":
            fid = _validate_finding_id(str(args.id))
            _guard_report_path(store, fid)

            doc = store.load_report(fid)
            next_steps = (
                next_steps_for_finding_doc(action="get", finding_id=fid, doc=doc or {})
                if isinstance(doc, dict)
                else []
            )
            _json_out(
                {
                    "ok": True,
                    "action": "explain",
                    "finding_id": fid,
                    "finding": (doc or {}).get("finding", {}) if isinstance(doc, dict) else {},
                    "explanation": _normalize_finding_explain(doc or {}) if isinstance(doc, dict) else {},
                    "next_steps": next_steps,
                },
                pretty,
            )
            return 0

        if args.cmd == "list":
            # Preserve historical behavior when writable (init + possibly rebuild index),
            # but ensure list can still succeed in read-only environments by falling back
            # to in-memory scanning if writes fail.
            try:
                store.init(force=False)
            except Exception:
                pass

            def _load_index_items() -> Optional[List[Dict[str, Any]]]:
                try:
                    raw = store.paths.index_path.read_text(encoding="utf-8")
                    idx_raw = yaml.safe_load(raw) or {}
                    items = idx_raw.get("findings") if isinstance(idx_raw, dict) else None
                    return items if isinstance(items, list) else None
                except Exception:
                    return None

            items = _load_index_items()

            if items is None:
                # Attempt to rebuild (historical behavior) but don't require it.
                try:
                    store.rebuild_index()
                    items = _load_index_items()
                except Exception:
                    items = None

            if items is None:
                # Read-only fallback: scan reports dir directly without writing index.yml.
                items = _scan_reports_summaries(store)

            out_items: List[Dict[str, Any]] = []
            q = (str(args.q) or "").strip().lower()

            for it in items or []:
                if not isinstance(it, dict):
                    continue
                if args.severity and str(it.get("severity") or "").strip() != str(args.severity).strip():
                    continue
                if args.status and str(it.get("status") or "").strip() != str(args.status).strip():
                    continue
                if args.scope and str(it.get("scope") or "").strip() != str(args.scope).strip():
                    continue
                if args.mode and str(it.get("mode") or "").strip() != str(args.mode).strip():
                    continue
                if args.repo_area and str(it.get("repo_area") or "").strip() != str(args.repo_area).strip():
                    continue
                if args.target_kind and str(it.get("target_kind") or "").strip() != str(args.target_kind).strip():
                    continue
                if q:
                    hay = " ".join(
                        [
                            str(it.get("finding_id") or ""),
                            str(it.get("title") or ""),
                            str(it.get("summary") or ""),
                            str(it.get("target_path") or ""),
                            " ".join(str(x) for x in (it.get("links") or []) if x),
                        ]
                    ).lower()
                    if q not in hay:
                        continue
                out_items.append(it)
                if len(out_items) >= max(1, min(int(args.limit or 200), 5000)):
                    break
            _json_out({"ok": True, "action": "list", "count": len(out_items), "items": out_items}, pretty)
            return 0

        if args.cmd == "overview":
            # Reuse list path with broad defaults, then aggregate.
            items = _scan_reports_summaries(store)
            items = sorted(
                [it for it in items if isinstance(it, dict)],
                key=lambda it: (str(it.get("discovered_at_utc") or ""), str(it.get("finding_id") or "")),
                reverse=True,
            )
            limit = max(1, min(int(getattr(args, "limit", 300) or 300), 5000))
            if len(items) > limit:
                items = items[:limit]

            counts_severity: Dict[str, int] = {}
            counts_status: Dict[str, int] = {}
            counts_scope: Dict[str, int] = {}
            counts_repo_area: Dict[str, int] = {}
            for item in items:
                sev = str(item.get("severity") or "unknown")
                sta = str(item.get("status") or "unknown")
                scope = str(item.get("scope") or "unknown")
                area = str(item.get("repo_area") or "unknown")
                counts_severity[sev] = int(counts_severity.get(sev, 0)) + 1
                counts_status[sta] = int(counts_status.get(sta, 0)) + 1
                counts_scope[scope] = int(counts_scope.get(scope, 0)) + 1
                counts_repo_area[area] = int(counts_repo_area.get(area, 0)) + 1

            _json_out(
                {
                    "ok": True,
                    "action": "overview",
                    "count": len(items),
                    "counts_by_severity": counts_severity,
                    "counts_by_status": counts_status,
                    "counts_by_scope": counts_scope,
                    "counts_by_repo_area": counts_repo_area,
                    "recent_items": items[: min(12, len(items))],
                    "recommended_next_calls": [
                        "findings list --status open --limit 50",
                        "findings explain --id <finding_id>",
                        "tasks add todo --title \"...\" --link finding:<id> --script <path>",
                    ],
                },
                pretty,
            )
            return 0

        _json_out(
            _error_payload(
                "unknown_command",
                f"Unknown command: {getattr(args, 'cmd', '')}",
                error_code="UNKNOWN_COMMAND",
                retryable=False,
            ),
            pretty,
        )
        return 2

    except _CliUsageError as e:
        _json_out(
            _error_payload(
                "invalid_arguments",
                e.message,
                error_code="ARGPARSE_ERROR",
                retryable=False,
            ),
            pretty,
        )
        return int(e.status) if int(e.status) != 0 else 2
    except Exception as exc:
        code, retryable = _classify_exception(exc)
        _json_out(
            _error_payload(
                "internal_error",
                str(exc),
                error_code=code,
                retryable=retryable,
            ),
            pretty,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
