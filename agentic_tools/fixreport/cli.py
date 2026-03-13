"""
Contract
- Purpose: Agent-facing JSON CLI for creating/searching FixReport entries.
- Inputs (types, units): CLI args; writes YAML under artifacts/stores/fixreport (legacy: state/fixreport).
- Outputs (types, units): JSON-only stdout for success/error.
- Invariants: All created fixreports validate against vocab.yaml.
- Error semantics: Non-zero exit with {"ok":false,...} on stdout.
- External boundaries: artifacts/stores/ filesystem; optional git for commit SHA.
"""

##REFACTOR: 2026-01-16##

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agentic_tools._governance_locking import resolve_retry_budget as _resolve_retry_budget
from agentic_tools._governance_locking import resolve_retry_poll as _resolve_retry_poll
from agentic_tools._governance_locking import run_mutation_with_retry as _run_governance_mutation_with_retry
from agentic_tools.codexctx import append_codex_links
from agentic_tools.process_hints import enrich_payload_with_hints
from agentic_tools.process_hints import hints_schema_fragment
from agentic_tools.process_hints import next_steps_for_fixreport_create
from agentic_tools.process_hints import next_steps_for_fixreport_doc

from .store import FixReportPostCommitSyncError, FixReportStore, _finding_ids_from_link_tokens
from .failure_modes import default_failure_modes_block, parse_fm_assignments, validate_failure_modes_block
from .vocab import load_vocab


LINK_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]{0,32}:[A-Za-z0-9._:/@+-]{1,256}$")
_ERROR_CODE_TOKEN_RE = re.compile(r"[^A-Za-z0-9]+")


class _ArgparseError(Exception):
    """Raised when argparse encounters a usage error (invalid arguments)."""

    def __init__(self, message: str, usage: str = "") -> None:
        super().__init__(message)
        self.message = str(message)
        self.usage = str(usage or "")


class _JsonArgumentParser(argparse.ArgumentParser):
    """
    ArgumentParser that does not call sys.exit() on errors.

    This enables us to honor the "JSON-only stdout" contract even for usage errors,
    while preserving argparse's help behavior (-h/--help).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Python 3.14+ adds colored help/errors via `color`. Disable when present for stable, non-ANSI output.
        # Note: This only affects help/usage on stderr/stdout; JSON remains on stdout via _json_out.
        if hasattr(self, "color"):
            try:
                setattr(self, "color", False)
            except Exception:
                pass

    def error(self, message: str) -> None:  # noqa: D401
        raise _ArgparseError(str(message), self.format_usage())


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for cand in [start, *start.parents]:
        if (cand / "AGENTS.md").exists() and (cand / "agentic_tools" / "fixreport" / "vocab.yaml").exists():
            return cand
    return start


def _json_out(payload: Dict[str, Any], pretty: bool) -> None:
    payload = enrich_payload_with_hints(payload, tool="fixreport")
    try:
        if pretty:
            sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        else:
            sys.stdout.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    except BrokenPipeError:
        # Common when piped into `head`/`rg` and downstream closes early.
        try:
            sys.stdout.close()
        except Exception:
            pass


def _read_json_payload_from_data_file(data_file: str) -> Any:
    """
    Read JSON payload from a file path or stdin.

    - `--data-file -` reads from stdin.
    - Other values are treated as filesystem paths (cwd-relative allowed).
    """
    src = str(data_file or "").strip()
    if not src:
        raise ValueError("missing_data_file: --data-file was empty")
    if src == "-":
        text = sys.stdin.read()
        origin = "<stdin>"
    else:
        p = Path(src).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p)
        origin = str(p)
        text = p.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid_json_payload: failed to parse JSON from {origin}: {e}") from e


def _git_short_sha(repo_root: Path) -> Optional[str]:
    """
    Best-effort git HEAD short SHA without invoking `git`.

    This repo is sometimes deployed without a full git worktree and our execution policy
    forbids invoking `git` as a subprocess. When `.git` data is unavailable, return None.
    """

    try:
        git_dir = repo_root / ".git"
        if not git_dir.exists():
            return None
        # Worktree `.git` may be a file containing "gitdir: <path>".
        if git_dir.is_file():
            head_txt = git_dir.read_text(encoding="utf-8", errors="replace").strip()
            if head_txt.startswith("gitdir:"):
                target = head_txt.split("gitdir:", 1)[1].strip()
                git_dir = (repo_root / target).resolve()
        head = (git_dir / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
        if head.startswith("ref:"):
            ref = head.split("ref:", 1)[1].strip()
            # 1) Loose ref file
            ref_path = (git_dir / ref).resolve()
            if ref_path.exists():
                head = ref_path.read_text(encoding="utf-8", errors="replace").strip()
            else:
                # 2) Packed refs (common on deployed machines / optimized repos)
                packed = (git_dir / "packed-refs").resolve()
                if packed.exists() and packed.is_file():
                    try:
                        for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
                            s = line.strip()
                            if not s or s.startswith("#") or s.startswith("^"):
                                continue
                            parts = s.split()
                            if len(parts) < 2:
                                continue
                            sha, ref_name = parts[0].strip(), parts[1].strip()
                            if ref_name == ref:
                                head = sha
                                break
                    except Exception:
                        # Best-effort: if packed-refs parsing fails, treat as missing.
                        return None

        sha = (head or "").strip()
        # Fail closed: only accept a plausible hex SHA.
        if not sha or len(sha) < 7:
            return None
        if any(ch not in "0123456789abcdefABCDEF" for ch in sha[:7]):
            return None
        return sha[:7]
    except Exception:
        return None


def _git_is_worktree(repo_root: Path) -> bool:
    """
    Best-effort check whether repo_root looks like a git work tree (no subprocess).
    """
    try:
        return (repo_root / ".git").exists()
    except Exception:
        return False


def _git_path_is_tracked(repo_root: Path, relpath: str, *, timeout_s: float = 1.0) -> Optional[bool]:
    """
    Deprecated: this tool no longer invokes `git` for tracking checks.
    """
    _ = (repo_root, relpath, timeout_s)
    return None


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    **extra: Any,
) -> Dict[str, Any]:
    normalized_error = str(error or "")
    normalized_error_code = str(error_code or "").strip() or _derive_error_code(normalized_error)
    payload: Dict[str, Any] = {
        "ok": False,
        "error": normalized_error,
        "error_code": normalized_error_code,
        "detail": str(detail or ""),
        "retryable": _default_retryable(normalized_error_code) if retryable is None else bool(retryable),
    }
    payload.update(extra)
    return payload


def _classify_exception(exc: BaseException) -> Tuple[str, bool]:
    if isinstance(exc, _ArgparseError):
        return ("ARGPARSE_ERROR", False)
    if isinstance(exc, FixReportPostCommitSyncError):
        return ("POST_COMMIT_SYNC_FAILED", False)
    if isinstance(exc, FileNotFoundError):
        return ("NOT_FOUND", False)
    if isinstance(exc, TimeoutError):
        return ("TIMEOUT", True)
    if isinstance(exc, PermissionError):
        return ("PERMISSION_DENIED", False)
    if isinstance(exc, ValueError):
        return ("VALIDATION_ERROR", False)
    if isinstance(exc, OSError):
        return ("IO_ERROR", True)
    return (_derive_error_code(type(exc).__name__), False)


def _fixreport_cli_retry_budget_sec() -> float:
    return _resolve_retry_budget(
        retry_timeout_env="CAIA_FIXREPORT_CLI_RETRY_TIMEOUT_SEC",
        lock_timeout_envs=("CAIA_FIXREPORT_LOCK_TIMEOUT_SEC", "FIXREPORT_LOCK_TIMEOUT"),
        default_lock_timeout=15.0,
    )


def _fixreport_cli_retry_poll_sec() -> float:
    return _resolve_retry_poll(
        retry_poll_env="CAIA_FIXREPORT_CLI_RETRY_POLL_SEC",
        default_poll_sec=0.25,
    )


def _emit_fixreport_retry(action: str, error_code: str, attempt: int, sleep_s: float, exc: BaseException) -> None:
    logging.warning(
        "Retrying fixreport mutation after %s action=%s attempt=%s sleep_s=%.3f error=%s",
        error_code,
        action,
        attempt,
        sleep_s,
        exc,
    )


def _run_mutation_with_retry(action: str, op: Callable[[], Any]) -> Tuple[Any, Dict[str, Any]]:
    return _run_governance_mutation_with_retry(
        action=action,
        op=op,
        classify_exception=_classify_exception,
        retry_budget_sec=_fixreport_cli_retry_budget_sec(),
        retry_poll_sec=_fixreport_cli_retry_poll_sec(),
        emit_retry=lambda error_code, attempt, sleep_s, exc: _emit_fixreport_retry(
            action,
            error_code,
            attempt,
            sleep_s,
            exc,
        ),
    )


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


def _dedupe_keep_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalize_fixreport_explain(doc: Dict[str, Any]) -> Dict[str, Any]:
    fr = doc.get("fixreport")
    if not isinstance(fr, dict):
        fr = {}
    nar = doc.get("narrative")
    if not isinstance(nar, dict):
        nar = {}
    fm = doc.get("failure_modes")
    if not isinstance(fm, dict):
        fm = {}
    checks = fm.get("checks")
    if not isinstance(checks, dict):
        checks = {}

    links = _dedupe_keep_order(_as_string_list(fr.get("links")))
    why = _pick_first_nonempty(
        fr.get("root_cause"),
        fr.get("failure_mode"),
        nar.get("repro"),
        nar.get("summary"),
        "Root-cause remediation record.",
    )
    how_values: List[str] = []
    how_values.extend(_as_string_list(nar.get("fix")))
    how_values.extend(_as_string_list(nar.get("validation")))
    how_values.extend(_as_string_list(nar.get("prevention")))
    if not how_values:
        how_values.append("Apply root-cause fix and verify against declared checks.")

    guardrails: List[str] = []
    verification = str(fr.get("verification") or "").strip()
    if verification:
        guardrails.append(f"verification={verification}")
    pass_checks: List[str] = []
    fail_checks: List[str] = []
    unknown_checks = 0
    for check_id, status in checks.items():
        status_text = str(status).strip()
        if not status_text:
            continue
        status_key = status_text.lower()
        if status_key == "pass":
            pass_checks.append(check_id)
            continue
        if status_key == "fail":
            fail_checks.append(check_id)
            continue
        if status_key == "unknown":
            unknown_checks += 1
            continue
    for check_id in fail_checks:
        guardrails.append(f"{check_id}=fail")
    if pass_checks:
        guardrails.append(f"failure_mode_checks_pass={len(pass_checks)}")
        for check_id in pass_checks[:8]:
            guardrails.append(f"{check_id}=pass")
        if len(pass_checks) > 8:
            guardrails.append(f"failure_mode_pass_checks_omitted={len(pass_checks) - 8}")
    if not guardrails:
        guardrails.append("Run deterministic validation before closure.")

    evidence: List[str] = []
    evidence.extend(_as_string_list(fr.get("task_ids")))
    evidence.extend(_as_string_list(fr.get("hypothesis_ids")))
    evidence.extend(_as_string_list(fr.get("ops_msg_ids")))
    evidence.extend(_as_string_list(fr.get("audit_ids")))
    evidence.extend(_as_string_list(fr.get("contracts")))
    evidence.extend([x for x in links if any(x.startswith(prefix) for prefix in ("finding:", "task:", "hypothesis:", "fixreport:", "audit_id:", "audit_file:", "script:"))])

    conflicts: List[str] = [f"{check_id}=fail" for check_id in fail_checks]
    if unknown_checks:
        conflicts.append(f"unknown_failure_mode_checks={unknown_checks}")

    return {
        "what": _pick_first_nonempty(nar.get("title"), nar.get("summary"), fr.get("bug_type"), fr.get("bf_id"), "FixReport"),
        "why": why,
        "how": _dedupe_keep_order(how_values),
        "guardrails": _dedupe_keep_order(guardrails),
        "evidence": _dedupe_keep_order(evidence),
        "conflicts": _dedupe_keep_order(conflicts),
        "links": links,
    }


def _scan_repo_files_for_script_anchors(repo_root: Path) -> List[str]:
    roots = [
        "services",
        "scripts",
        "systemd/units",
        "agentic_tools",
        "llm",
        "consumer",
        "common",
        "tests",
        "evaluations",
        "config",
        "docs",
        "audit",
    ]
    deny_prefixes = (
        "state/",
        "reports/",
        "research/",
        "artifacts/stores/",
        "artifacts/reports/",
        "logs/",
        "exports/",
        "kafka/",
        "clickhouse/",
    )
    deny_parts = {".git", "__pycache__"}

    out: List[str] = []
    for root_rel in roots:
        root_path = (repo_root / root_rel).resolve()
        if not root_path.exists():
            continue
        if not root_path.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
            dirnames[:] = [d for d in dirnames if d not in deny_parts]
            for fn in filenames:
                try:
                    p = (Path(dirpath) / fn).resolve()
                except Exception:
                    continue
                try:
                    rel = p.relative_to(repo_root).as_posix()
                except Exception:
                    continue
                if rel.startswith(deny_prefixes):
                    continue
                out.append(rel)

    # Also include repo-root files (e.g. .gitignore, README.md).
    try:
        for child in repo_root.iterdir():
            if child.is_file():
                out.append(child.name)
    except Exception:
        pass

    return sorted(set(out))


def _build_repo_file_index(repo_root: Path) -> Tuple[set[str], Dict[str, List[str]]]:
    repo_files = _scan_repo_files_for_script_anchors(repo_root)
    if not repo_files:
        return set(), {}
    repo_files_set = set(repo_files)
    by_basename: Dict[str, List[str]] = {}
    for p in repo_files:
        by_basename.setdefault(Path(p).name, []).append(p)
    return repo_files_set, by_basename


def _normalize_repo_path(raw: str, *, repo_root: Path) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if s.startswith("./"):
        s = s[2:]
    rp = str(repo_root) + "/"
    if s.startswith(rp):
        s = s[len(rp) :]
    if s.startswith("/etc/systemd/system/"):
        unit = s.split("/etc/systemd/system/", 1)[1]
        s = f"systemd/units/{unit}"
    return s


def _resolve_repo_path(
    raw: str,
    *,
    repo_root: Path,
    repo_files_set: set[str],
    by_basename: Dict[str, List[str]],
) -> Optional[str]:
    s = _normalize_repo_path(raw, repo_root=repo_root)
    if not s:
        return None
    if any(part == ".." for part in Path(s).parts):
        return None
    if repo_files_set:
        if s in repo_files_set:
            return s
    else:
        if (repo_root / s).is_file():
            return s
    # Basename-only fallback (unique only).
    if "/" not in s:
        candidates = sorted(set(by_basename.get(Path(s).name, [])))
        if len(candidates) == 1:
            return candidates[0]
    return None


def _is_reasonable_script_anchor_path(path: str) -> bool:
    p = (path or "").strip()
    if not p:
        return False
    if p.startswith(("state/", "reports/", "artifacts/reports/", "artifacts/stores/", "logs/", "exports/")):
        return False
    if p.startswith("__ignore__/"):
        return False
    if "/__pycache__/" in p or p.endswith("/__pycache__/"):
        return False
    return True


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        s = str(x or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _normalize_and_require_script_links(
    links: List[str],
    *,
    repo_root: Path,
    target_path: str,
    paths_touched: List[str],
    systemd_unit: str,
) -> List[str]:
    raw = _dedupe_keep_order([str(x).strip() for x in links if str(x).strip()])
    bad = [x for x in raw if not LINK_TOKEN_RE.fullmatch(x)]
    if bad:
        raise ValueError("invalid_link_token: " + ", ".join(bad[:10]))

    # Script anchors must be safe, repo-relative paths, but do not have to exist in the current checkout.
    # This supports post-factum reports where files were moved/deleted or git state is unavailable.
    #
    # We avoid invoking git; for basename-only anchors we fall back to a filesystem scan and require uniqueness.
    repo_files_set: Optional[set[str]] = None
    by_basename: Optional[Dict[str, List[str]]] = None

    def _ensure_index() -> Tuple[set[str], Dict[str, List[str]]]:
        nonlocal repo_files_set, by_basename
        if repo_files_set is not None and by_basename is not None:
            return repo_files_set, by_basename
        idx_set, idx_base = _build_repo_file_index(repo_root)
        repo_files_set = idx_set
        by_basename = idx_base
        return repo_files_set, by_basename

    def _normalize_script_path(raw_path: str) -> Optional[str]:
        s = _normalize_repo_path(raw_path, repo_root=repo_root)
        if not s:
            return None
        if any(part == ".." for part in Path(s).parts):
            return None
        if not _is_reasonable_script_anchor_path(s):
            return None
        # Script anchors are file-like paths; reject directory-like anchors deterministically.
        if s.endswith("/"):
            return None
        try:
            p = (repo_root / s)
            if p.exists() and p.is_dir():
                return None
        except Exception:
            pass

        # Basename-only anchors must either exist at repo root (e.g. ".gitignore") or resolve uniquely.
        if "/" not in s:
            if (repo_root / s).is_file():
                return s
            _, idx_base = _ensure_index()
            candidates = sorted(set(idx_base.get(Path(s).name, [])))
            if len(candidates) == 1:
                return candidates[0]
            return None

        return s

    normalized: List[str] = []
    invalid_scripts: List[str] = []
    for tok in raw:
        if not tok.startswith("script:"):
            normalized.append(tok)
            continue
        raw_path = tok.split("script:", 1)[1].strip()
        rp = _normalize_script_path(raw_path)
        if rp is None:
            invalid_scripts.append(raw_path)
            continue
        normalized.append(f"script:{rp}")

    if invalid_scripts:
        raise ValueError(
            "invalid_script_anchor: one or more script:<path> links do not resolve to a repo file: "
            + ", ".join(invalid_scripts[:10])
        )

    # If no script anchors were provided, infer from target_path / paths_touched / systemd_unit.
    if not any(x.startswith("script:") for x in normalized):
        candidates: List[str] = []
        if target_path:
            candidates.append(target_path)
        for p in paths_touched[:40]:
            if p:
                candidates.append(p)
        if systemd_unit:
            candidates.append(f"systemd/units/{systemd_unit}")

        inferred: List[str] = []
        for c in candidates:
            rp = _normalize_script_path(c)
            if rp is None:
                continue
            inferred.append(f"script:{rp}")

        normalized = _dedupe_keep_order(normalized + inferred)

    if not any(x.startswith("script:") for x in normalized):
        raise ValueError(
            "missing_script_anchor: FixReport creation requires at least one script:<repo_relative_path> in links. "
            "Provide --script <path> / --link script:<path> or ensure --target-path/--paths-touched resolve to repo files."
        )

    # Stable ordering: scripts sorted, then other links preserved.
    scripts_sorted = sorted({x for x in normalized if x.startswith("script:")})
    others = [x for x in normalized if not x.startswith("script:")]
    return _dedupe_keep_order(scripts_sorted + others)


def _enforce_strict_fm_policy(*, checks: Dict[str, str]) -> None:
    """
    Enforce that new FixReports explicitly set (pass/fail/na) at least one check
    in each high-leverage dimension: time, schema, cost, eval.

    This prevents "all unknown" checklists which are non-actionable for retrieval.
    """

    def _has_any(ids: List[str]) -> bool:
        for fm_id in ids:
            v = str(checks.get(fm_id) or "")
            if v and v != "unknown":
                return True
        return False

    time_ids = [
        "FM_TIME_SEMANTICS_EVENT_VS_INGEST",
        "FM_ASOF_JOINS_ENFORCED",
        "FM_LATE_EVENTS_NO_FORWARD_DATE",
        "FM_BACKFILL_CLOCK_CONSISTENCY",
    ]
    schema_ids = [
        "FM_SCHEMA_CONTRACT_DRIFT_GUARDED",
        "FM_DEFAULT_ZERO_DOMINANCE",
        "FM_TOPIC_FRAMING_DRIFT_GUARDED",
    ]
    cost_ids = [
        "FM_COST_SEMANTICS_DECLARED",
        "FM_COST_DOUBLE_COUNT_CHECK",
        "FM_DECISION_GATES_EX_ANTE",
    ]
    eval_ids = [
        "FM_HEARTBEATS_EXCLUDED",
        "FM_EVAL_WINDOW_AVAILABILITY_INTERSECTION",
        "FM_LABEL_DEGENERACY_GATED",
        "FM_PURGED_CV_EMBARGO_ENFORCED",
    ]

    missing_dims: List[str] = []
    if not _has_any(time_ids):
        missing_dims.append("time")
    if not _has_any(schema_ids):
        missing_dims.append("schema")
    if not _has_any(cost_ids):
        missing_dims.append("cost")
    if not _has_any(eval_ids):
        missing_dims.append("eval")

    if missing_dims:
        raise ValueError(
            "fm_policy_strict_failed: missing explicit failure-modes coverage for "
            + ",".join(missing_dims)
            + ". Provide at least one --fm per dimension (pass|fail|na). "
            + "Recommended: "
            + " --fm FM_TIME_SEMANTICS_EVENT_VS_INGEST=pass"
            + " --fm FM_SCHEMA_CONTRACT_DRIFT_GUARDED=pass"
            + " --fm FM_COST_SEMANTICS_DECLARED=pass"
            + " --fm FM_HEARTBEATS_EXCLUDED=pass"
        )


def _infer_fail_checks_from_root_cause(root_cause: str) -> List[str]:
    """
    Heuristic backfill for historical reports: infer a few high-signal FM checks
    that likely failed based on fixreport.root_cause.
    """

    rc = (root_cause or "").strip()
    mapping: Dict[str, List[str]] = {
        "lookahead": ["FM_ASOF_JOINS_ENFORCED", "FM_WITHIN_BUCKET_ARGMAX_LOOKAHEAD_GUARDED"],
        "leakage": ["FM_ASOF_JOINS_ENFORCED", "FM_PURGED_CV_EMBARGO_ENFORCED"],
        "asof_join_missing": ["FM_ASOF_JOINS_ENFORCED"],
        "join_window_wrong": ["FM_ASOF_JOINS_ENFORCED"],
        "join_key_wrong": ["FM_ASOF_JOINS_ENFORCED", "FM_VENUE_INSTRUMENT_JOIN_KEYS_STRICT"],
        "out_of_order": ["FM_LATE_EVENTS_NO_FORWARD_DATE", "FM_TIME_SEMANTICS_EVENT_VS_INGEST"],
        "ts_event_vs_ts_ingest": ["FM_TIME_SEMANTICS_EVENT_VS_INGEST"],
        "clock_skew": ["FM_TIME_SEMANTICS_EVENT_VS_INGEST"],
        "schema_mismatch": ["FM_SCHEMA_CONTRACT_DRIFT_GUARDED", "FM_DEFAULT_ZERO_DOMINANCE"],
        "proto_drift": ["FM_SCHEMA_CONTRACT_DRIFT_GUARDED", "FM_TOPIC_FRAMING_DRIFT_GUARDED"],
        "contract_mismatch": ["FM_SCHEMA_CONTRACT_DRIFT_GUARDED", "FM_TOPIC_FRAMING_DRIFT_GUARDED"],
        "missing_field": ["FM_SCHEMA_CONTRACT_DRIFT_GUARDED", "FM_DEFAULT_ZERO_DOMINANCE"],
        "double_count": ["FM_COST_DOUBLE_COUNT_CHECK", "FM_COST_SEMANTICS_DECLARED"],
        "wrong_cost_model": ["FM_COST_SEMANTICS_DECLARED", "FM_COST_TARGET_MATCH_DECLARED"],
        "nan_fill_policy": ["FM_MISSINGNESS_PARITY_TRAIN_LIVE"],
        "null_handling": ["FM_MISSINGNESS_PARITY_TRAIN_LIVE"],
        "ttl_drop": ["FM_TTL_LAG_DROPS_MONITORED"],
    }
    return list(mapping.get(rc, []))


def _build_parser() -> argparse.ArgumentParser:
    ap = _JsonArgumentParser(
        prog="fixreport",
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "FixReport CLI for bugfix memory and prevention.\n\n"
            "Related tools:\n"
            "  - findings: intake before a fix exists.\n"
            "  - tasks: execute remediation work linked to findings/fixreports.\n"
            "  - hypothesis: track disputed causes and evidence evolution.\n"
            "  - rules/ssot: canonize recurring mitigations into governance.\n\n"
            "Recommended workflow:\n"
            "  1) fixreport overview\n"
            "  2) fixreport search --root-cause <token>\n"
            "  3) fixreport explain --bf-id BFxxxxxx\n"
            "  4) fixreport create ... (for new fixes)\n"
        ),
    )
    ap.add_argument("--json-pretty", action="store_true")
    ap.add_argument("--repo-root", default="", help="Override repo root discovery (optional).")
    ap.add_argument(
        "--state-dir",
        default="",
        help="Override fixreport state directory (default: <repo>/artifacts/stores/fixreport; legacy: <repo>/state/fixreport).",
    )
    ap.add_argument("--actor", default="", help="Agent/human name for attribution (optional).")

    sub = ap.add_subparsers(dest="cmd", required=True, parser_class=_JsonArgumentParser)

    sub.add_parser("schema", help="Emit schema info from vocab.yaml.")
    sub.add_parser("vocab", help="Emit the full vocab (from vocab.yaml) as JSON.")
    sub.add_parser("failure-modes", help="Emit the failure-modes checklist catalog as JSON.")
    sub.add_parser("fm", help="Alias for failure-modes.")
    sub.add_parser("template", help="Emit a non-interactive JSON template for creating a FixReport.")

    p_init = sub.add_parser("init", help="Initialize fixreport state store under artifacts/stores/fixreport/.")
    p_init.add_argument("--force", action="store_true")

    p_rebuild = sub.add_parser("rebuild-index", help="Rebuild artifacts/stores/fixreport/index.yml from reports.")

    p_norm = sub.add_parser(
        "normalize",
        help="Normalize list-token fields in existing reports (split comma-separated values).",
    )
    p_norm.add_argument("--apply", action="store_true", help="Rewrite report YAMLs in-place (default: dry-run).")

    p_fm_backfill = sub.add_parser(
        "fm-backfill",
        help=(
            "Backfill/extend failure_modes blocks for existing fixreports "
            "(adds new FM ids as unknown; may infer a few obvious fails from root_cause)."
        ),
    )
    p_fm_backfill.add_argument("--apply", action="store_true", help="Rewrite report YAMLs in-place (default: dry-run).")
    p_fm_backfill.add_argument(
        "--limit", type=int, default=0, help="Optional limit on number of reports to modify (0 = no limit)."
    )

    def _add_create_subparser(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "--data-file",
            default="",
            help="Path to a JSON FixReport payload file for create/new, or '-' to read JSON from stdin.",
        )

        # Required vocab fields (required unless --data-file is used).
        p.add_argument("--repo-area", default="")
        p.add_argument("--target-kind", default="")
        p.add_argument("--target-path", default="")
        p.add_argument("--scope", default="")
        p.add_argument("--mode", default="")
        p.add_argument("--bug-type", default="")
        p.add_argument("--symptom", default="")
        p.add_argument("--root-cause", default="")
        p.add_argument("--failure-mode", default="")
        p.add_argument(
            "--archetype",
            "--archetypes",
            action="append",
            dest="archetypes",
            default=[],
            help="Repeatable archetype tag (mandatory for create/new), e.g. stateful_agent_workflow.",
        )
        p.add_argument("--fix-type", default="")
        p.add_argument("--impact-area", default="")
        p.add_argument("--severity", default="")
        p.add_argument("--verification", default="")
        p.add_argument("--commit", default="", help="Override commit sha (default: current git HEAD).")
        p.add_argument("--ts-utc", default="", help="Override timestamp (default: now UTC).")

        # Optional vocab fields
        p.add_argument("--paths-touched", action="append", default=[])
        p.add_argument("--systemd-unit", default="")
        p.add_argument("--contracts", action="append", default=[])
        p.add_argument("--topics", action="append", default=[])
        p.add_argument("--tables-views", action="append", default=[])
        p.add_argument("--asset", default="")
        p.add_argument("--venue", default="")
        p.add_argument("--instrument", default="")
        p.add_argument("--horizon", default="")
        p.add_argument("--tests-added", choices=["yes", "no"], default="")
        p.add_argument("--breaking", choices=["yes", "no"], default="")
        p.add_argument("--backfill-needed", choices=["yes", "no"], default="")
        p.add_argument("--hypothesis-ids", action="append", default=[])
        p.add_argument("--task-ids", action="append", default=[])
        p.add_argument("--ops-msg-ids", action="append", default=[])
        p.add_argument("--audit-ids", action="append", default=[])
        p.add_argument(
            "--link",
            "--links",
            action="append",
            dest="links",
            default=[],
            help="Repeatable outbound link token of the form kind:id (see architecture/LINKS_CONTRACT.md).",
        )
        p.add_argument(
            "--script",
            action="append",
            dest="scripts",
            default=[],
            help="Repo-relative path to anchor this FixReport to (repeatable). Stored as script:<path> in fixreport.links.",
        )
        p.add_argument("--signature", default="")
        p.add_argument("--tags", action="append", default=[])

        # Failure-modes checklist (optional but strongly recommended).
        # Use `fixreport failure-modes` to discover allowed IDs and meanings.
        p.add_argument(
            "--fm-policy",
            choices=["strict", "off"],
            default="strict",
            help="Policy for failure-modes checklist on create (default: strict).",
        )
        p.add_argument(
            "--fm",
            action="append",
            default=[],
            help="Failure-modes checklist entry like FM_TIME_SEMANTICS=pass (repeatable).",
        )
        p.add_argument("--fm-notes", default="", help="Optional notes for failure-modes checklist.")

        # Narrative (free text, not part of vocab strictness)
        p.add_argument("--title", default="")
        p.add_argument("--summary", default="")
        p.add_argument("--repro", default="")
        p.add_argument("--fix", default="")
        p.add_argument("--validation", default="")
        p.add_argument("--prevention", default="")
        p.add_argument("--notes", default="")
        p.set_defaults(_subcommand_usage=p.format_usage())
        return p

    _add_create_subparser("create", "Create a new fixreport entry.")
    _add_create_subparser("new", "Alias for create (LLM/tool friendly).")

    p_list = sub.add_parser("list", help="List fixreports (from index).")
    p_list.add_argument("--limit", type=int, default=50)
    p_ls = sub.add_parser("ls", help="Alias for list.")
    p_ls.add_argument("--limit", type=int, default=50)

    p_get = sub.add_parser("get", help="Load a single fixreport by bf_id.")
    p_get.add_argument("--bf-id", "--id", dest="bf_id", required=True, help="FixReport id (BFxxxxxx).")

    p_explain = sub.add_parser(
        "explain",
        help="Explain one FixReport in normalized what/why/how form.",
    )
    p_explain.add_argument("--bf-id", "--id", dest="bf_id", required=True, help="FixReport id (BFxxxxxx).")

    p_overview = sub.add_parser(
        "overview",
        help="Compact FixReport summary for agent orientation.",
    )
    p_overview.add_argument("--max-scan", type=int, default=400, help="Max reports to scan (default: %(default)s).")

    p_search = sub.add_parser("search", help="Search fixreports by exact-match filters.")
    p_search.add_argument("--limit", type=int, default=50)
    p_search.add_argument("--since", default="", help="Lower bound for ts_utc (inclusive, lexical ISO8601Z).")
    p_search.add_argument("--until", default="", help="Upper bound for ts_utc (inclusive, lexical ISO8601Z).")
    p_search.add_argument("--repo-area", default="")
    p_search.add_argument("--target-kind", default="")
    p_search.add_argument("--scope", default="")
    p_search.add_argument("--mode", default="")
    p_search.add_argument("--bug-type", default="")
    p_search.add_argument("--symptom", default="")
    p_search.add_argument("--root-cause", default="")
    p_search.add_argument("--failure-mode", default="")
    p_search.add_argument("--fix-type", default="")
    p_search.add_argument("--impact-area", default="")
    p_search.add_argument("--severity", default="")
    p_search.add_argument("--verification", default="")
    p_search.add_argument("--tags-contains", default="")
    p_search.add_argument("--archetypes-contains", default="")
    p_search.add_argument(
        "--fm",
        action="append",
        default=[],
        help="Failure-modes checklist filter like FM_TIME_SEMANTICS_EVENT_VS_INGEST=fail (repeatable).",
    )

    p_lint = sub.add_parser("lint", help="Validate all fixreports against vocab.yaml.")
    p_lint.add_argument("--limit", type=int, default=0, help="Stop after N errors (0 = no limit).")

    p_update = sub.add_parser("update", help="Update an existing fixreport (shallow merge) by bf_id.")
    p_update.add_argument("--bf-id", "--id", dest="bf_id", required=True, help="FixReport id (BFxxxxxx).")
    p_update.add_argument(
        "--allow-invalid-vocab",
        action="store_true",
        help="Relax vocab enum validation for backfills (still enforces required fields and structural checks).",
    )
    p_update.add_argument(
        "--set-fixreport-json",
        default="",
        help="JSON object to shallow-set into fixreport fields (list fields are union-merged).",
    )
    p_update.add_argument(
        "--set-narrative-json",
        default="",
        help="JSON object to shallow-set into narrative fields.",
    )
    p_update.add_argument(
        "--set-failure-modes-json",
        default="",
        help="JSON object to shallow-set into failure_modes (checks are merged by key).",
    )
    p_update.add_argument(
        "--link-add",
        "--links-add",
        dest="links_add",
        action="append",
        default=[],
        help="Append link token(s) into fixreport.links (repeatable).",
    )
    p_update.add_argument(
        "--link-remove",
        "--links-remove",
        dest="links_remove",
        action="append",
        default=[],
        help="Remove link token(s) from fixreport.links (repeatable).",
    )

    p_ask_det = sub.add_parser(
        "ask-analysis-det",
        help="Deterministic semantic analysis over FixReports (no LLM); writes a YAML analysis artifact.",
    )
    p_ask_det.add_argument("--question", required=True)
    p_ask_det.add_argument(
        "--analysis-dir",
        default="artifacts/stores/llm_specialists/fixreports/analysis",
        help="Output directory for analysis artifacts (relative to repo root unless absolute).",
    )
    p_ask_det.add_argument(
        "--include-archived",
        action="store_true",
        help="Include archived FixReports (default: active-only).",
    )
    p_ask_det.add_argument(
        "--full",
        action="store_true",
        help="Disable META preference (default: prefer META FixReports when relevant).",
    )
    p_ask_det.add_argument("--scan-all", action="store_true", help="Scan all FixReports (slower; default: false).")
    p_ask_det.add_argument("--global-scan", action="store_true", help="Use global scan mode in retrieval (default: false).")
    p_ask_det.add_argument("--candidate-limit", type=int, default=0, help="Override candidate limit (0 = no limit).")
    p_ask_det.add_argument("--artifact-id", default="", help="Optional explicit analysis id (FRAxxxxx).")

    p_ask = sub.add_parser(
        "ask-analysis",
        help="LLM-backed semantic FixReports analysis; writes a YAML analysis artifact grounded via BF citations.",
    )
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument(
        "--analysis-dir",
        default="artifacts/stores/llm_specialists/fixreports/analysis",
        help="Output directory for analysis artifacts (relative to repo root unless absolute).",
    )
    p_ask.add_argument(
        "--include-archived",
        action="store_true",
        help="Include archived FixReports (default: active-only).",
    )
    p_ask.add_argument(
        "--full",
        action="store_true",
        help="Disable META preference (default: prefer META FixReports when relevant).",
    )
    p_ask.add_argument("--global-scan", action="store_true", help="Use global scan mode in retrieval (default: false).")
    p_ask.add_argument("--candidate-limit", type=int, default=0, help="Override candidate limit (0 = default).")
    p_ask.add_argument("--artifact-id", default="", help="Optional explicit analysis id (FRAxxxxx).")

    p_ask_qa = sub.add_parser(
        "ask",
        help="Ask the FixReports Expert (LLM-backed) and return JSON (no analysis YAML artifact).",
    )
    p_ask_qa.add_argument("--question", required=True)
    p_ask_qa.add_argument(
        "--include-archived",
        action="store_true",
        help="Include archived FixReports (default: active-only).",
    )
    p_ask_qa.add_argument(
        "--full",
        action="store_true",
        help="Disable META preference (default: prefer META FixReports when relevant).",
    )
    p_ask_qa.add_argument("--scan-all", action="store_true", help="Scan all FixReports (slower; default: false).")
    p_ask_qa.add_argument("--global-scan", action="store_true", help="Use global scan mode in retrieval (default: false).")
    p_ask_qa.add_argument("--candidate-limit", type=int, default=0, help="Override candidate limit (0 = default).")

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Allow global flags to appear after the subcommand (LLM/tool friendly).
    # Example: `fixreport_cli schema --json-pretty`.
    _global_arity = {"--json-pretty": 0, "--repo-root": 1, "--state-dir": 1, "--actor": 1}
    moved: List[str] = []
    rest: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in _global_arity:
            n = _global_arity[a]
            if n:
                # Missing value: do not reorder (let argparse report).
                if i + n >= len(argv):
                    rest.append(a)
                    i += 1
                    continue
                moved.append(a)
                moved.extend(argv[i + 1 : i + 1 + n])
                i += 1 + n
                continue
            moved.append(a)
            i += 1
            continue
        if any(a.startswith(f"{k}=") for k in ("--repo-root", "--state-dir", "--actor")):
            moved.append(a)
            i += 1
            continue
        rest.append(a)
        i += 1
    argv = moved + rest

    def _global_prefix_len(args: List[str]) -> int:
        i2 = 0
        while i2 < len(args):
            a2 = args[i2]
            if a2 in _global_arity:
                n2 = _global_arity[a2]
                i2 += 1 + n2
                continue
            if any(a2.startswith(f"{k}=") for k in ("--repo-root", "--state-dir", "--actor")):
                i2 += 1
                continue
            break
        return i2

    # Convenience aliases: allow `fixreport --new ...` as well as subcommand `new`.
    if "--new" in argv and "create" not in argv and "new" not in argv:
        argv = [a for a in argv if a != "--new"]
        argv.insert(_global_prefix_len(argv), "new")
    if "--list" in argv and "list" not in argv and "ls" not in argv:
        argv = [a for a in argv if a != "--list"]
        argv.insert(_global_prefix_len(argv), "list")
    if "--ls" in argv and "list" not in argv and "ls" not in argv:
        argv = [a for a in argv if a != "--ls"]
        argv.insert(_global_prefix_len(argv), "ls")

    ap = _build_parser()

    # Use a cheap pre-parse guess for formatting any argparse errors.
    pretty_guess = "--json-pretty" in argv

    try:
        args = ap.parse_args(argv)
    except (_ArgparseError, argparse.ArgumentError) as e:
        usage = e.usage if isinstance(e, _ArgparseError) else ""
        detail = e.message if isinstance(e, _ArgparseError) else str(e)
        _json_out(
            _error_payload(
                "invalid_arguments",
                detail,
                error_code="ARGPARSE_ERROR",
                retryable=False,
                usage=(usage or "").strip(),
            ),
            pretty_guess,
        )
        return 2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03dZ | %(levelname)s | fixreport | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    pretty = bool(args.json_pretty)
    try:
        if (args.repo_root or "").strip():
            repo_root = Path(args.repo_root).expanduser().resolve()
        else:
            repo_root = _find_repo_root(Path.cwd())

        state_dir = Path(args.state_dir).expanduser() if (args.state_dir or "").strip() else None
        store = FixReportStore(repo_root=repo_root, state_dir=state_dir)

        if args.cmd == "schema":
            vocab = load_vocab(store.paths.vocab_path)
            required_fields = [k for k, f in vocab.fields.items() if f.required]
            enum_fields = {
                k: list(f.allowed or [])
                for k, f in vocab.fields.items()
                if f.type == "enum" and (f.allowed is not None)
            }
            payload = {
                "ok": True,
                "action": "schema",
                "vocab_version": vocab.version,
                "rules": dict(vocab.rules),
                "hint_envelope": hints_schema_fragment(),
                "required_fields": required_fields,
                "enum_fields": enum_fields,
                "fields": {
                    k: {
                        "type": f.type,
                        "required": f.required,
                        "free": f.free,
                        "allowed": f.allowed,
                        "pattern": f.pattern,
                    }
                    for k, f in vocab.fields.items()
                },
                "paths": {
                    "vocab_path": str(store.paths.vocab_path),
                    "state_dir": str(store.paths.state_dir),
                },
                "commands": {
                    "schema": "Emit vocab-derived schema and constraints.",
                    "overview": "Compact summary of fixreport corpus.",
                    "list": "List fixreports from index.",
                    "search": "Exact-match retrieval on structured fields.",
                    "get": "Load one fixreport document.",
                    "explain": "Normalize one fixreport into what/why/how/guardrails/evidence.",
                    "create": "Create a new fixreport entry.",
                    "update": "Patch an existing fixreport.",
                    "lint": "Validate fixreports against vocab/failure-modes constraints.",
                },
                "related_tools": {
                    "findings": "Pre-fix intake and tracking of observed issues.",
                    "tasks": "Execution lifecycle for remediation work.",
                    "hypothesis": "Evidence-backed root-cause reasoning.",
                    "rules": "Canonize recurring mitigations into governance.",
                    "ssot": "Canonical references for services/config/contracts.",
                },
                "recommended_workflow": [
                    "fixreport overview",
                    "fixreport search --root-cause <token>",
                    "fixreport explain --bf-id BFxxxxxx",
                    "rules search \"<mitigation>\"",
                ],
            }
            _json_out(payload, pretty)
            return 0

        if args.cmd == "vocab":
            import yaml

            raw = yaml.safe_load(store.paths.vocab_path.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                raise ValueError("vocab.yaml is not a mapping")
            _json_out(
                {
                    "ok": True,
                    "action": "vocab",
                    "vocab_path": str(store.paths.vocab_path),
                    "vocab": raw,
                },
                pretty,
            )
            return 0

        if args.cmd in {"failure-modes", "fm"}:
            import yaml

            raw = yaml.safe_load(store.paths.failure_modes_catalog_path.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                raise ValueError("failure_modes_catalog.yaml is not a mapping")
            _json_out(
                {
                    "ok": True,
                    "action": "failure-modes",
                    "catalog_path": str(store.paths.failure_modes_catalog_path),
                    "catalog": raw,
                },
                pretty,
            )
            return 0

        if args.cmd == "template":
            vocab = load_vocab(store.paths.vocab_path)
            required_fields = [k for k, f in vocab.fields.items() if f.required]
            enum_fields = {
                k: list(f.allowed or [])
                for k, f in vocab.fields.items()
                if f.type == "enum" and (f.allowed is not None)
            }

            # Provide a minimal, non-interactive skeleton that agents can fill.
            # NOTE: bf_id and ts_utc are normally auto-generated by `fixreport new`,
            # so the template leaves them empty on purpose.
            fr = {k: "" for k in required_fields}
            fr["bf_id"] = ""
            fr["ts_utc"] = ""
            fr["commit"] = _git_short_sha(repo_root) or ""

            # Fill enum-ish fields with a sane default when possible, to reduce friction.
            for k, allowed in enum_fields.items():
                if k in fr and not fr[k]:
                    fr[k] = allowed[0] if allowed else ""
            fr["archetypes"] = []

            template = {
                "fixreport": fr,
                "narrative": {
                    "title": "",
                    "summary": "",
                    "repro": "",
                    "fix": "",
                    "validation": "",
                    "prevention": "",
                    "notes": "",
                },
                "evidence": [],
                "failure_modes": default_failure_modes_block(catalog=store.failure_modes_catalog),
            }
            _json_out(
                {
                    "ok": True,
                    "action": "template",
                    "vocab_version": vocab.version,
                    "vocab_path": str(store.paths.vocab_path),
                    "rules": dict(vocab.rules),
                    "required_fields": required_fields,
                    "enum_fields": enum_fields,
                    "template": template,
                    "usage": {
                        "create_cmd": (
                            "fixreport new --repo-area <...> --target-kind <...> --target-path <...> "
                            "--scope <...> --mode <...> --bug-type <...> --symptom <...> --root-cause <...> "
                            "--failure-mode <...> --archetype <...> --fix-type <...> --impact-area <...> --severity <...> "
                            "--verification <...> --title \"...\" --repro \"...\" --fix \"...\" --validation \"...\""
                        ),
                        "note": "Allowed enum values are in enum_fields; full source-of-truth is vocab.yaml.",
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

        if args.cmd == "rebuild-index":
            def _rebuild_index() -> Dict[str, Any]:
                store.init(force=False)
                return store.rebuild_index()

            idx, retry_info = _run_mutation_with_retry("rebuild-index", _rebuild_index)
            out = {"ok": True, "action": "rebuild-index", "count": len(idx.get("fixreports", []))}
            if retry_info:
                out["retry"] = retry_info
            _json_out(out, pretty)
            return 0

        if args.cmd == "normalize":
            retry_info: Dict[str, Any] = {}
            if bool(args.apply):
                res, retry_info = _run_mutation_with_retry(
                    "normalize",
                    lambda: store.normalize_reports(apply=True),
                )
            else:
                res = store.normalize_reports(apply=False)
            out = {"ok": True, "action": "normalize", "apply": bool(args.apply), "total": res["total"], "changed": res["changed"]}
            if retry_info:
                out["retry"] = retry_info
            _json_out(out, pretty)
            return 0

        if args.cmd == "fm-backfill":
            store.init(force=False)
            actor = args.actor.strip() or None
            catalog_ids = sorted(store.failure_modes_catalog.checks.keys())
            changed: List[Dict[str, Any]] = []
            for bf_id in store.list_bf_ids():
                raw = store.load_report(bf_id)
                fr = raw.get("fixreport") if isinstance(raw, dict) else None
                if not isinstance(fr, dict):
                    continue
                fm = raw.get("failure_modes") if isinstance(raw, dict) else None
                if not isinstance(fm, dict):
                    fm = {}
                checks = fm.get("checks")
                if not isinstance(checks, dict):
                    checks = {}

                patch_checks: Dict[str, str] = {}
                for fm_id in catalog_ids:
                    if fm_id not in checks:
                        patch_checks[fm_id] = "unknown"

                inferred_fail = _infer_fail_checks_from_root_cause(str(fr.get("root_cause") or ""))
                applied_fail: List[str] = []
                for fm_id in inferred_fail:
                    cur = str(checks.get(fm_id) or "")
                    if not cur or cur == "unknown":
                        patch_checks[fm_id] = "fail"
                        applied_fail.append(fm_id)

                if not patch_checks:
                    continue

                if args.apply:
                    note = (
                        f"auto_backfill {_utc_now_z()} "
                        f"root_cause={str(fr.get('root_cause') or '')} "
                        f"inferred_fail={','.join(applied_fail) or 'none'}"
                    )
                    existing_notes = str(fm.get("notes") or "").strip()
                    new_notes = note if not existing_notes else (existing_notes + "\n" + note)
                    _run_mutation_with_retry(
                        "fm-backfill",
                        lambda: store.update(
                            bf_id,
                            set_failure_modes={
                                "catalog_version": int(store.failure_modes_catalog.version),
                                "checks": patch_checks,
                                "notes": new_notes,
                            },
                            actor=actor,
                        ),
                    )

                changed.append(
                    {
                        "bf_id": bf_id,
                        "added_checks": sorted([k for k, v in patch_checks.items() if v == "unknown"]),
                        "failed_checks": sorted([k for k, v in patch_checks.items() if v == "fail"]),
                    }
                )
                if args.limit and len(changed) >= int(args.limit):
                    break

            _json_out(
                {
                    "ok": True,
                    "action": "fm-backfill",
                    "apply": bool(args.apply),
                    "changed": len(changed),
                    "items": changed,
                },
                pretty,
            )
            return 0

        if args.cmd in {"create", "new"}:
            using_data_file = bool(str(getattr(args, "data_file", "") or "").strip())
            if using_data_file:
                raw = _read_json_payload_from_data_file(str(getattr(args, "data_file", "") or ""))
                if not isinstance(raw, dict):
                    _json_out(
                        _error_payload(
                            "invalid_arguments",
                            "--data-file payload must be a JSON object.",
                            error_code="ARGPARSE_ERROR",
                            retryable=False,
                        ),
                        pretty,
                    )
                    return 2

                # Accept either a full document with {"fixreport": {...}, "narrative": {...}, ...}
                # or a direct fixreport fields object {...}.
                if "fixreport" in raw:
                    fixreport = raw.get("fixreport")
                    narrative = raw.get("narrative")
                    evidence = raw.get("evidence")
                    failure_modes = raw.get("failure_modes")
                    payload_scripts = raw.get("scripts")
                else:
                    fixreport = raw
                    narrative = None
                    evidence = None
                    failure_modes = None
                    payload_scripts = None

                if not isinstance(fixreport, dict):
                    _json_out(
                        _error_payload(
                            "invalid_arguments",
                            "FixReport payload must contain a JSON object at key fixreport (or be the object itself).",
                            error_code="ARGPARSE_ERROR",
                            retryable=False,
                        ),
                        pretty,
                    )
                    return 2
                if narrative is not None and not isinstance(narrative, dict):
                    raise ValueError("invalid_payload: narrative must be a JSON object when provided")
                if evidence is not None:
                    if not isinstance(evidence, list) or any(not isinstance(x, dict) for x in evidence):
                        raise ValueError("invalid_payload: evidence must be a list of JSON objects when provided")
                if failure_modes is not None and not isinstance(failure_modes, dict):
                    raise ValueError("invalid_payload: failure_modes must be a JSON object when provided")

                fixreport = dict(fixreport)

                # Many production deployments run from a non-git checkout; default to a sentinel
                # commit instead of hard-failing the FixReport workflow.
                if str(getattr(args, "commit", "") or "").strip():
                    fixreport["commit"] = str(getattr(args, "commit", "") or "").strip()
                else:
                    commit = str(fixreport.get("commit") or "").strip() or (_git_short_sha(repo_root) or "0000000")
                    fixreport["commit"] = commit

                if str(getattr(args, "ts_utc", "") or "").strip():
                    fixreport["ts_utc"] = str(getattr(args, "ts_utc", "") or "").strip()

                # Optional CLI patch-style additions (kept for LLM/tool friendliness).
                if str(getattr(args, "systemd_unit", "") or "").strip() and not str(fixreport.get("systemd_unit") or "").strip():
                    fixreport["systemd_unit"] = str(getattr(args, "systemd_unit", "") or "").strip()
                if getattr(args, "paths_touched", None):
                    existing_pt = fixreport.get("paths_touched")
                    existing_pt_list = list(existing_pt) if isinstance(existing_pt, list) else []
                    existing_pt_list.extend([str(x).strip() for x in (args.paths_touched or []) if str(x).strip()])
                    fixreport["paths_touched"] = existing_pt_list

                links_raw: List[str] = []
                existing_links = fixreport.get("links")
                if isinstance(existing_links, list):
                    links_raw.extend([str(x).strip() for x in existing_links if str(x or "").strip()])
                elif isinstance(existing_links, str) and existing_links.strip():
                    links_raw.append(existing_links.strip())
                if getattr(args, "links", None):
                    links_raw.extend([str(x).strip() for x in (args.links or []) if str(x or "").strip()])
                if getattr(args, "scripts", None):
                    links_raw.extend([f"script:{x}" for x in (args.scripts or []) if str(x or "").strip()])
                if isinstance(payload_scripts, list):
                    links_raw.extend([f"script:{x}" for x in payload_scripts if str(x or "").strip()])

                eff_target_path = str(fixreport.get("target_path") or "").strip()
                eff_paths_touched_raw = fixreport.get("paths_touched")
                eff_paths_touched = (
                    [str(x).strip() for x in list(eff_paths_touched_raw or []) if str(x or "").strip()]
                    if isinstance(eff_paths_touched_raw, list)
                    else []
                )
                eff_systemd_unit = str(fixreport.get("systemd_unit") or "").strip()

                fixreport["links"] = _normalize_and_require_script_links(
                    links_raw,
                    repo_root=repo_root,
                    target_path=eff_target_path,
                    paths_touched=eff_paths_touched,
                    systemd_unit=eff_systemd_unit,
                )
                fixreport["links"] = append_codex_links(list(fixreport.get("links") or []))

                actor = args.actor.strip() or None
                fm_assignments = [x for x in (args.fm or []) if isinstance(x, str) and x.strip()]
                fm_notes = (args.fm_notes or "").strip()
                if failure_modes is None:
                    failure_modes = default_failure_modes_block(catalog=store.failure_modes_catalog)
                else:
                    failure_modes = dict(failure_modes)
                fm_checks_raw = failure_modes.get("checks")
                fm_checks = dict(fm_checks_raw) if isinstance(fm_checks_raw, dict) else {}
                fm_updates, fm_errors = parse_fm_assignments(assignments=fm_assignments)
                if fm_errors:
                    raise ValueError("--fm parse error: " + "; ".join(fm_errors))
                if fm_updates:
                    fm_checks.update(fm_updates)
                if fm_notes:
                    failure_modes["notes"] = fm_notes
                failure_modes["checks"] = fm_checks
                if str(getattr(args, "fm_policy", "strict") or "strict") == "strict":
                    _enforce_strict_fm_policy(checks=fm_checks)

                (bf_id, path), retry_info = _run_mutation_with_retry(
                    "create",
                    lambda: store.create(
                        fixreport,
                        narrative=narrative if isinstance(narrative, dict) else None,
                        evidence=evidence if isinstance(evidence, list) else None,
                        failure_modes=failure_modes,
                        actor=actor,
                    ),
                )
                next_steps = next_steps_for_fixreport_create(
                    bf_id=str(bf_id),
                    target_path=str(fixreport.get("target_path") or "").strip(),
                    links=[str(x) for x in (list(fixreport.get("links") or [])) if str(x).strip()],
                )
                out = {"ok": True, "action": "create", "bf_id": bf_id, "path": str(path), "next_steps": next_steps}
                if retry_info:
                    out["retry"] = retry_info
                _json_out(out, pretty)
                return 0

            # CLI mode (no --data-file): enforce required vocab fields explicitly.
            required = [
                ("--repo-area", "repo_area"),
                ("--target-kind", "target_kind"),
                ("--target-path", "target_path"),
                ("--scope", "scope"),
                ("--mode", "mode"),
                ("--bug-type", "bug_type"),
                ("--symptom", "symptom"),
                ("--root-cause", "root_cause"),
                ("--failure-mode", "failure_mode"),
                ("--archetype", "archetypes"),
                ("--fix-type", "fix_type"),
                ("--impact-area", "impact_area"),
                ("--severity", "severity"),
                ("--verification", "verification"),
            ]
            missing_flags: List[str] = []
            for flag, attr in required:
                v = getattr(args, attr, "")
                if isinstance(v, list):
                    if not any(str(x or "").strip() for x in v):
                        missing_flags.append(flag)
                    continue
                if not str(v or "").strip():
                    missing_flags.append(flag)
            if missing_flags:
                _json_out(
                    _error_payload(
                        "invalid_arguments",
                        "the following arguments are required: " + ", ".join(missing_flags),
                        error_code="ARGPARSE_ERROR",
                        retryable=False,
                        usage=str(getattr(args, "_subcommand_usage", "") or "").strip(),
                    ),
                    pretty,
                )
                return 2

            commit = args.commit.strip() or (_git_short_sha(repo_root) or "0000000")
            fixreport = {
                "commit": commit,
                "repo_area": args.repo_area,
                "target_kind": args.target_kind,
                "target_path": args.target_path,
                "scope": args.scope,
                "mode": args.mode,
                "bug_type": args.bug_type,
                "symptom": args.symptom,
                "root_cause": args.root_cause,
                "failure_mode": args.failure_mode,
                "archetypes": [str(x).strip() for x in (args.archetypes or []) if str(x).strip()],
                "fix_type": args.fix_type,
                "impact_area": args.impact_area,
                "severity": args.severity,
                "verification": args.verification,
            }
            if args.ts_utc.strip():
                fixreport["ts_utc"] = args.ts_utc.strip()

            if args.paths_touched:
                fixreport["paths_touched"] = [p for p in args.paths_touched if p]
            if args.systemd_unit:
                fixreport["systemd_unit"] = args.systemd_unit
            if args.contracts:
                fixreport["contracts"] = [c for c in args.contracts if c]
            if args.topics:
                fixreport["topics"] = [t for t in args.topics if t]
            if args.tables_views:
                fixreport["tables_views"] = [t for t in args.tables_views if t]
            if args.asset:
                fixreport["asset"] = args.asset
            if args.venue:
                fixreport["venue"] = args.venue
            if args.instrument:
                fixreport["instrument"] = args.instrument
            if args.horizon:
                fixreport["horizon"] = args.horizon
            if args.tests_added:
                fixreport["tests_added"] = args.tests_added
            if args.breaking:
                fixreport["breaking"] = args.breaking
            if args.backfill_needed:
                fixreport["backfill_needed"] = args.backfill_needed
            if args.hypothesis_ids:
                fixreport["hypothesis_ids"] = [x for x in args.hypothesis_ids if x]
            if args.task_ids:
                fixreport["task_ids"] = [x for x in args.task_ids if x]
            if args.ops_msg_ids:
                fixreport["ops_msg_ids"] = [x for x in args.ops_msg_ids if x]
            if args.audit_ids:
                fixreport["audit_ids"] = [x for x in args.audit_ids if x]

            # Cross-tool links: MUST include at least one explicit script anchor.
            # This enables the Meta graph/UI to treat FixReports as truthy nodes
            # in the same namespace as tasks/hypotheses/audits.
            links_raw: List[str] = []
            if args.links:
                links_raw.extend([x for x in args.links if x])
            if getattr(args, "scripts", None):
                links_raw.extend([f"script:{x}" for x in args.scripts if x])

            fixreport["links"] = _normalize_and_require_script_links(
                links_raw,
                repo_root=repo_root,
                target_path=str(args.target_path or "").strip(),
                paths_touched=[str(p or "").strip() for p in (args.paths_touched or []) if str(p or "").strip()],
                systemd_unit=str(args.systemd_unit or "").strip(),
            )
            fixreport["links"] = append_codex_links(list(fixreport.get("links") or []))
            if args.signature:
                fixreport["signature"] = args.signature
            if args.tags:
                fixreport["tags"] = [x for x in args.tags if x]

            narrative: Dict[str, Any] = {}
            if args.title:
                narrative["title"] = args.title
            if args.summary:
                narrative["summary"] = args.summary
            if args.repro:
                narrative["repro"] = args.repro
            if args.fix:
                narrative["fix"] = args.fix
            if args.validation:
                narrative["validation"] = args.validation
            if args.prevention:
                narrative["prevention"] = args.prevention
            if args.notes:
                narrative["notes"] = args.notes

            actor = args.actor.strip() or None
            fm_assignments = [x for x in (args.fm or []) if isinstance(x, str) and x.strip()]
            fm_notes = (args.fm_notes or "").strip()
            fm = default_failure_modes_block(catalog=store.failure_modes_catalog)
            fm_updates, fm_errors = parse_fm_assignments(assignments=fm_assignments)
            if fm_errors:
                raise ValueError("--fm parse error: " + "; ".join(fm_errors))
            fm["checks"].update(fm_updates or {})
            if fm_notes:
                fm["notes"] = fm_notes
            failure_modes = fm
            if str(getattr(args, "fm_policy", "strict") or "strict") == "strict":
                _enforce_strict_fm_policy(checks=failure_modes["checks"])

            (bf_id, path), retry_info = _run_mutation_with_retry(
                "create",
                lambda: store.create(
                    fixreport,
                    narrative=narrative or None,
                    evidence=None,
                    failure_modes=failure_modes,
                    actor=actor,
                ),
            )
            next_steps = next_steps_for_fixreport_create(
                bf_id=str(bf_id),
                target_path=str(args.target_path or "").strip(),
                links=[str(x) for x in (list(fixreport.get("links") or [])) if str(x).strip()],
            )
            out = {"ok": True, "action": "create", "bf_id": bf_id, "path": str(path), "next_steps": next_steps}
            if retry_info:
                out["retry"] = retry_info
            _json_out(out, pretty)
            return 0

        if args.cmd == "list":
            rows = store.search(filters={}, limit=int(args.limit))
            _json_out({"ok": True, "action": "list", "count": len(rows), "items": rows}, pretty)
            return 0

        if args.cmd == "ls":
            rows = store.search(filters={}, limit=int(args.limit) if hasattr(args, "limit") else 50)
            _json_out({"ok": True, "action": "list", "count": len(rows), "items": rows}, pretty)
            return 0

        if args.cmd == "get":
            raw = store.load_report(args.bf_id)
            next_steps = next_steps_for_fixreport_doc(action="get", bf_id=str(args.bf_id), doc=raw or {}) if isinstance(raw, dict) else []
            _json_out({"ok": True, "action": "get", "bf_id": args.bf_id, "item": raw, "next_steps": next_steps}, pretty)
            return 0

        if args.cmd == "explain":
            raw = store.load_report(args.bf_id)
            next_steps = (
                next_steps_for_fixreport_doc(action="get", bf_id=str(args.bf_id), doc=raw or {})
                if isinstance(raw, dict)
                else []
            )
            _json_out(
                {
                    "ok": True,
                    "action": "explain",
                    "bf_id": str(args.bf_id),
                    "item": raw,
                    "explanation": _normalize_fixreport_explain(raw or {}) if isinstance(raw, dict) else {},
                    "next_steps": next_steps,
                },
                pretty,
            )
            return 0

        if args.cmd == "overview":
            max_scan = max(1, min(int(getattr(args, "max_scan", 400) or 400), 5000))
            bf_ids = store.list_bf_ids()
            bf_ids = sorted(bf_ids, reverse=True)[:max_scan]
            count_by_severity: Dict[str, int] = {}
            count_by_bug_type: Dict[str, int] = {}
            count_by_root_cause: Dict[str, int] = {}
            latest: List[Dict[str, Any]] = []
            for bf_id in bf_ids:
                try:
                    raw = store.load_report(bf_id)
                except Exception:
                    continue
                fr = raw.get("fixreport")
                if not isinstance(fr, dict):
                    continue
                severity = str(fr.get("severity") or "unknown")
                bug_type = str(fr.get("bug_type") or "unknown")
                root_cause = str(fr.get("root_cause") or "unknown")
                count_by_severity[severity] = int(count_by_severity.get(severity, 0)) + 1
                count_by_bug_type[bug_type] = int(count_by_bug_type.get(bug_type, 0)) + 1
                count_by_root_cause[root_cause] = int(count_by_root_cause.get(root_cause, 0)) + 1
                if len(latest) < 12:
                    latest.append(
                        {
                            "bf_id": str(fr.get("bf_id") or bf_id),
                            "ts_utc": str(fr.get("ts_utc") or ""),
                            "severity": severity,
                            "bug_type": bug_type,
                            "root_cause": root_cause,
                            "target_path": str(fr.get("target_path") or ""),
                        }
                    )
            _json_out(
                {
                    "ok": True,
                    "action": "overview",
                    "scanned": len(bf_ids),
                    "total_reports": len(store.list_bf_ids()),
                    "counts_by_severity": count_by_severity,
                    "counts_by_bug_type": count_by_bug_type,
                    "counts_by_root_cause": count_by_root_cause,
                    "latest": latest,
                    "recommended_next_calls": [
                        "fixreport search --severity p1 --limit 50",
                        "fixreport explain --bf-id <bf_id>",
                        "rules search \"<root_cause_or_mitigation>\"",
                    ],
                },
                pretty,
            )
            return 0

        if args.cmd == "search":
            filters: Dict[str, Any] = {
                "repo_area": args.repo_area or None,
                "target_kind": args.target_kind or None,
                "scope": args.scope or None,
                "mode": args.mode or None,
                "bug_type": args.bug_type or None,
                "symptom": args.symptom or None,
                "root_cause": args.root_cause or None,
                "failure_mode": args.failure_mode or None,
                "fix_type": args.fix_type or None,
                "impact_area": args.impact_area or None,
                "severity": args.severity or None,
                "verification": args.verification or None,
                "tags_contains": args.tags_contains or None,
                "archetypes_contains": args.archetypes_contains or None,
            }
            fm_assignments = [x for x in (args.fm or []) if isinstance(x, str) and x.strip()]
            fm_filters, fm_errors = parse_fm_assignments(assignments=fm_assignments)
            if fm_errors:
                raise ValueError("--fm parse error: " + "; ".join(fm_errors))
            if fm_filters:
                ok_fm, fm_errs = validate_failure_modes_block(
                    catalog=store.failure_modes_catalog,
                    block={
                        "catalog_version": int(store.failure_modes_catalog.version),
                        "checks": fm_filters,
                        "notes": "",
                    },
                )
                if not ok_fm:
                    raise ValueError("--fm validation error: " + "; ".join(fm_errs))
            rows = store.search(
                filters=filters,
                fm_filters=fm_filters or None,
                limit=int(args.limit),
                since_ts_utc=(args.since or "").strip(),
                until_ts_utc=(args.until or "").strip(),
            )
            _json_out({"ok": True, "action": "search", "count": len(rows), "items": rows}, pretty)
            return 0

        if args.cmd in {"ask-analysis-det", "ask-analysis", "ask"}:
            actor = args.actor.strip() or "fixreport"
            store.init(force=False)

            import yaml  # local import

            def _load_index_rows() -> List[Dict[str, Any]]:
                raw = yaml.safe_load(store.paths.index_path.read_text(encoding="utf-8")) or {}
                if not isinstance(raw, dict):
                    return []
                rows = raw.get("fixreports")
                return list(rows) if isinstance(rows, list) else []

            rows = _load_index_rows()
            # If the index is an older shape (missing narrative previews), rebuild once so we can
            # do fast semantic recall without parsing every BF YAML on each ask invocation.
            if rows and (("title" not in rows[0]) or ("summary" not in rows[0]) or ("commit" not in rows[0])):
                store.rebuild_index()
                rows = _load_index_rows()

            from consumer.portal.fixreports_store import FixReportSummary  # noqa: WPS433

            summaries: List[FixReportSummary] = []
            for r in rows:
                if not isinstance(r, dict):
                    continue
                summaries.append(
                    FixReportSummary(
                        bf_id=str(r.get("bf_id") or ""),
                        ts_utc=str(r.get("ts_utc") or ""),
                        commit=str(r.get("commit") or ""),
                        repo_area=str(r.get("repo_area") or ""),
                        target_kind=str(r.get("target_kind") or ""),
                        target_path=str(r.get("target_path") or ""),
                        scope=str(r.get("scope") or ""),
                        mode=str(r.get("mode") or ""),
                        bug_type=str(r.get("bug_type") or ""),
                        symptom=str(r.get("symptom") or ""),
                        root_cause=str(r.get("root_cause") or ""),
                        failure_mode=str(r.get("failure_mode") or ""),
                        fix_type=str(r.get("fix_type") or ""),
                        impact_area=str(r.get("impact_area") or ""),
                        severity=str(r.get("severity") or ""),
                        verification=str(r.get("verification") or ""),
                        signature=str(r.get("signature") or ""),
                        tags=tuple(sorted(set([str(x) for x in (r.get("tags") or []) if x is not None and str(x).strip()]))),
                        title=str(r.get("title") or ""),
                        summary=str(r.get("summary") or ""),
                        parse_error=None,
                        updated_mtime_ns=int(r.get("updated_mtime_ns") or 0),
                    )
                )

            def _is_archived(tags: Tuple[str, ...]) -> bool:
                for t in tags or ():
                    if str(t or "").strip().lower() == "archived":
                        return True
                return False

            if not bool(getattr(args, "include_archived", False)):
                summaries = [s for s in summaries if not _is_archived(s.tags)]

            artifact_id = str(getattr(args, "artifact_id", "") or "").strip()
            prefer_meta = not bool(getattr(args, "full", False))

            if args.cmd == "ask":
                from llm.specialists.fixreports.ask import AskConfig, AskLimits, default_ask_config, run_fixreports_ask  # noqa: WPS433
                from llm.specialists.fixreports.specialist import build_candidate_pool, compute_store_stats, compute_top_counts  # noqa: WPS433

                base_cfg = default_ask_config()
                lim_raw = int(getattr(args, "candidate_limit", 0) or 0)
                lim = lim_raw if lim_raw > 0 else int(base_cfg.limits.candidate_limit)
                lim = max(40, min(lim, 400))
                cfg = AskConfig(
                    backend=base_cfg.backend,
                    base_url=base_cfg.base_url,
                    model=base_cfg.model,
                    timeout_s=base_cfg.timeout_s,
                    max_retries=base_cfg.max_retries,
                    seed=base_cfg.seed,
                    limits=AskLimits(
                        candidate_limit=int(lim),
                        candidate_preview_chars=base_cfg.limits.candidate_preview_chars,
                        max_selected_fixreports=base_cfg.limits.max_selected_fixreports,
                        max_iters=base_cfg.limits.max_iters,
                        max_answer_chars=base_cfg.limits.max_answer_chars,
                        max_prompt_chars=base_cfg.limits.max_prompt_chars,
                        max_synth_tokens=base_cfg.limits.max_synth_tokens,
                    ),
                )

                idx_like = type(
                    "Idx",
                    (),
                    {"summaries": summaries, "reports_dir": store.paths.reports_dir, "state_dir": store.paths.state_dir},
                )()
                scored = build_candidate_pool(
                    index=idx_like,
                    question=str(args.question or ""),
                    scan_all=bool(getattr(args, "scan_all", False)),
                    global_scan=bool(getattr(args, "global_scan", False)),
                    prefer_meta=bool(prefer_meta),
                )
                store_stats = compute_store_stats(summaries=summaries)
                top_counts = compute_top_counts(summaries=summaries)
                res = run_fixreports_ask(
                    cfg=cfg,
                    question=str(args.question or ""),
                    summaries_scored=scored,
                    reports_dir=store.paths.reports_dir,
                    store_stats=store_stats,
                    top_counts=top_counts,
                )
                _json_out(
                    {
                        "ok": bool(res.ok),
                        "action": "ask",
                        "prefer_meta": bool(prefer_meta),
                        "include_archived": bool(getattr(args, "include_archived", False)),
                        "answer": str(res.answer_markdown or ""),
                        "answer_json": dict(res.answer_json or {}),
                        "used_fixreport_ids": list(res.used_fixreport_ids or ()),
                        "uncertainties": list(res.uncertainties or ()),
                        "followup_questions": list(res.followup_questions or ()),
                        "trace": [t.__dict__ for t in (res.trace or tuple())],
                        "error": str(res.error or ""),
                    },
                    pretty,
                )
                return 0 if bool(res.ok) else 2

            out_raw = Path(
                str(getattr(args, "analysis_dir", "") or "").strip()
                or "artifacts/stores/llm_specialists/fixreports/analysis"
            )
            out_dir = out_raw if out_raw.is_absolute() else (repo_root / out_raw)
            out_dir = out_dir.resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            if args.cmd == "ask-analysis-det":
                from llm.specialists.fixreports.analysis import build_fixreports_analysis_doc, write_analysis_artifact  # noqa: WPS433

                doc = build_fixreports_analysis_doc(
                    index_summaries=summaries,
                    reports_dir=store.paths.reports_dir,
                    question=str(args.question or ""),
                    scan_all=bool(getattr(args, "scan_all", False)),
                    global_scan=bool(getattr(args, "global_scan", False)),
                    candidate_limit=int(getattr(args, "candidate_limit", 0) or 0),
                    prefer_meta=bool(prefer_meta),
                    cfg=None,
                    created_by=actor,
                    source_state_dir=str(store.paths.state_dir),
                )
                aid, path = write_analysis_artifact(doc=doc, out_dir=out_dir, artifact_id=artifact_id)
                _json_out(
                    {"ok": True, "action": "ask-analysis-det", "analysis_id": aid, "analysis_path": str(path)},
                    pretty,
                )
                return 0

            from llm.specialists.fixreports import analysis_llm  # noqa: WPS433
            from llm.specialists.fixreports.analysis import write_analysis_artifact  # noqa: WPS433

            cfg = analysis_llm.default_analysis_config()
            doc, trace = analysis_llm.build_fixreports_analysis_doc_llm(
                cfg=cfg,
                index_summaries=summaries,
                reports_dir=store.paths.reports_dir,
                question=str(args.question or ""),
                global_scan=bool(getattr(args, "global_scan", False)),
                candidate_limit=int(getattr(args, "candidate_limit", 0) or 0),
                prefer_meta=bool(prefer_meta),
                created_by=actor,
                source_state_dir=str(store.paths.state_dir),
                progress_cb=None,
            )
            aid, path = write_analysis_artifact(doc=doc, out_dir=out_dir, artifact_id=artifact_id)
            _json_out(
                {"ok": True, "action": "ask-analysis", "analysis_id": aid, "analysis_path": str(path), "trace": trace},
                pretty,
            )
            return 0

        if args.cmd == "lint":
            store.init(force=False)
            errors: List[Dict[str, Any]] = []
            count = 0
            for bf_id in store.list_bf_ids():
                raw = store.load_report(bf_id)
                ok, verrs = store.validate_report_doc(raw)
                if not ok:
                    errors.append({"bf_id": bf_id, "errors": verrs})
                    if args.limit and len(errors) >= int(args.limit):
                        break
                count += 1
            _json_out(
                {"ok": len(errors) == 0, "action": "lint", "checked": count, "error_count": len(errors), "errors": errors},
                pretty,
            )
            return 0 if len(errors) == 0 else 2

        if args.cmd == "update":
            bf_id = str(args.bf_id or "").strip()
            raw_existing = store.load_report(bf_id)
            existing_fixreport = raw_existing.get("fixreport") if isinstance(raw_existing, dict) else {}
            if not isinstance(existing_fixreport, dict):
                existing_fixreport = {}
            existing_links_raw = existing_fixreport.get("links")
            existing_links: List[str] = (
                [str(x).strip() for x in list(existing_links_raw or []) if str(x or "").strip()]
                if isinstance(existing_links_raw, list)
                else []
            )

            set_fixreport: Optional[Dict[str, Any]] = None
            set_narrative: Optional[Dict[str, Any]] = None
            set_failure_modes: Optional[Dict[str, Any]] = None
            if (args.set_fixreport_json or "").strip():
                set_fixreport = json.loads(args.set_fixreport_json)
                if not isinstance(set_fixreport, dict):
                    raise ValueError("--set-fixreport-json must be a JSON object")
                # Preserve historical behavior for direct links overrides, but keep codex links appended.
                if isinstance(set_fixreport.get("links"), list):
                    set_fixreport["links"] = append_codex_links(list(set_fixreport.get("links") or []))
            if (args.set_narrative_json or "").strip():
                set_narrative = json.loads(args.set_narrative_json)
                if not isinstance(set_narrative, dict):
                    raise ValueError("--set-narrative-json must be a JSON object")
            if (args.set_failure_modes_json or "").strip():
                set_failure_modes = json.loads(args.set_failure_modes_json)
                if not isinstance(set_failure_modes, dict):
                    raise ValueError("--set-failure-modes-json must be a JSON object")

            links_add = [str(x).strip() for x in (args.links_add or []) if str(x or "").strip()]
            links_remove = [str(x).strip() for x in (args.links_remove or []) if str(x or "").strip()]
            if links_add or links_remove:
                bad = [x for x in (links_add + links_remove) if not LINK_TOKEN_RE.fullmatch(x)]
                if bad:
                    raise ValueError("invalid_link_token: " + ", ".join(bad[:10]))

            has_link_mutations = bool(links_add or links_remove)
            has_links_override = bool(set_fixreport is not None and "links" in set_fixreport)
            if has_link_mutations or has_links_override:
                base_links: List[str] = []
                if has_links_override and isinstance((set_fixreport or {}).get("links"), list):
                    base_links = [str(x).strip() for x in list((set_fixreport or {}).get("links") or []) if str(x or "").strip()]
                else:
                    base_links = list(existing_links)

                remove_l = {x.lower() for x in links_remove}
                merged = [x for x in base_links if x.lower() not in remove_l]
                seen_l = {x.lower() for x in merged}
                for tok in links_add:
                    tl = tok.lower()
                    if tl in seen_l:
                        continue
                    merged.append(tok)
                    seen_l.add(tl)

                merged = append_codex_links(merged)

                # Normalize/enforce script anchors only when links were touched explicitly.
                eff_target_path = str((set_fixreport or {}).get("target_path") or existing_fixreport.get("target_path") or "").strip()
                eff_paths_touched_raw = (set_fixreport or {}).get("paths_touched") or existing_fixreport.get("paths_touched") or []
                eff_paths_touched = [str(x).strip() for x in list(eff_paths_touched_raw or []) if str(x or "").strip()] if isinstance(eff_paths_touched_raw, list) else []
                eff_systemd_unit = str((set_fixreport or {}).get("systemd_unit") or existing_fixreport.get("systemd_unit") or "").strip()
                merged = _normalize_and_require_script_links(
                    merged,
                    repo_root=repo_root,
                    target_path=eff_target_path,
                    paths_touched=eff_paths_touched,
                    systemd_unit=eff_systemd_unit,
                )
                if set_fixreport is None:
                    set_fixreport = {}
                set_fixreport["links"] = merged

            explicit_finding_ids: List[str] = _finding_ids_from_link_tokens(links_add)
            if has_links_override and isinstance((set_fixreport or {}).get("links"), list):
                explicit_finding_ids = _finding_ids_from_link_tokens(
                    list(explicit_finding_ids) + list((set_fixreport or {}).get("links") or [])
                )

            actor = args.actor.strip() or None
            path, retry_info = _run_mutation_with_retry(
                "update",
                lambda: store.update(
                    bf_id,
                    set_fixreport=set_fixreport,
                    set_narrative=set_narrative,
                    set_failure_modes=set_failure_modes,
                    actor=actor,
                    validate=not bool(getattr(args, "allow_invalid_vocab", False)),
                    explicit_finding_ids=explicit_finding_ids,
                ),
            )
            raw = store.load_report(bf_id)
            next_steps = next_steps_for_fixreport_doc(action="update", bf_id=str(bf_id), doc=raw or {}) if isinstance(raw, dict) else []
            out = {"ok": True, "action": "update", "bf_id": bf_id, "path": str(path), "next_steps": next_steps}
            if retry_info:
                out["retry"] = retry_info
            _json_out(out, pretty)
            return 0

        _json_out(
            _error_payload(
                "unknown_command",
                f"unknown cmd: {args.cmd}",
                error_code="UNKNOWN_COMMAND",
                retryable=False,
            ),
            pretty,
        )
        return 2
    except ValueError as e:
        usage = str(getattr(args, "_subcommand_usage", "") or "").strip()
        extra: Dict[str, Any] = {}
        if usage:
            extra["usage"] = usage
        _json_out(
            _error_payload(
                "invalid_arguments",
                str(e),
                error_code="ARGPARSE_ERROR",
                retryable=False,
                **extra,
            ),
            pretty,
        )
        return 2
    except Exception as e:
        error_code, retryable = _classify_exception(e)
        extra: Dict[str, Any] = {}
        if isinstance(e, FixReportPostCommitSyncError):
            extra = {
                "committed": True,
                "bf_id": str(e.bf_id),
                "path": str(e.report_path),
                "finding_ids": list(e.finding_ids),
            }
        _json_out(
            _error_payload(
                "internal_error",
                repr(e),
                error_code=error_code,
                retryable=retryable,
                **extra,
            ),
            pretty,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
