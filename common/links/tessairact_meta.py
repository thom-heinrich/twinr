from __future__ import annotations

##REFACTOR: 2026-01-16##

import getpass
import hashlib
import os
import re
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)


TESSAIRACT_META_KEY = "tessairact_meta"
TESSAIRACT_META_HEADER_VERSION = 1

_GIT_TIMEOUT_ENV = "TESSAIRACT_GIT_TIMEOUT_S"


_KIND_RE = re.compile(r"^[a-z][a-z0-9_]{0,32}$")
_STRICT_LINK_RE = re.compile(r"^[a-z][a-z0-9_]{0,32}:[A-Za-z0-9._:/@+-]{1,256}$")
_LOOSE_LINK_RE = re.compile(r"^[a-z][a-z0-9_]{0,32}:\S+$")
_EMBEDDED_LINK_RE = re.compile(
    r"(?<![A-Za-z0-9_])([a-z][a-z0-9_]{0,32}:[A-Za-z0-9._:/@+-]{1,256})"
)
_BF_BARE_RE = re.compile(r"\bBF\d{6}\b")
_H_BARE_RE = re.compile(r"\bH\d{5}\b")
_E_BARE_RE = re.compile(r"\bE\d{5}\b")
_M_BARE_RE = re.compile(r"\bM\d{5}\b")
_FND_BARE_RE = re.compile(r"\bFND\d{6}\b")
_BF_ID_RE = re.compile(r"^BF\d{6}$")
_H_ID_RE = re.compile(r"^H\d{5}$")
_E_ID_RE = re.compile(r"^E\d{5}$")
_M_ID_RE = re.compile(r"^M\d{5}$")
_FND_ID_RE = re.compile(r"^FND\d{6}$")

# Trailing punctuation commonly attached to IDs in prose (e.g. "BF000123,", "evidence:E00002.").
# We only apply this normalization to known ID-like kinds (fixreport/hypothesis/evidence/chat_msg/finding).
_TRAILING_ID_PUNCT = ".,;:!?)]}>\"'"

_CODEX_SESSION_ID_RE = re.compile(r"^(session|thread)/[A-Za-z0-9._:@+-]{1,256}$")


def _strip_trailing_id_punct(s: str) -> str:
    return str(s or "").rstrip(_TRAILING_ID_PUNCT)


def _normalize_id_like_link(kind: str, ident: str) -> Optional[str]:
    """
    Normalize ID-like link tokens that often appear in prose with trailing punctuation.

    Examples:
      - evidence:E00002.  -> evidence:E00002
      - fixreport:BF000123) -> fixreport:BF000123

    Only normalizes when the stripped ident matches the strict ID pattern for the kind.
    Returns the normalized token string, or None if no safe normalization applies.
    """
    k = str(kind or "").strip().lower()
    raw = str(ident or "")
    stripped = _strip_trailing_id_punct(raw)
    if stripped == raw:
        return None
    if k == "fixreport" and _BF_ID_RE.fullmatch(stripped):
        return f"fixreport:{stripped}"
    if k == "hypothesis" and _H_ID_RE.fullmatch(stripped):
        return f"hypothesis:{stripped}"
    if k == "evidence" and _E_ID_RE.fullmatch(stripped):
        return f"evidence:{stripped}"
    if k == "chat_msg" and _M_ID_RE.fullmatch(stripped):
        return f"chat_msg:{stripped}"
    if k == "finding" and _FND_ID_RE.fullmatch(stripped):
        return f"finding:{stripped}"
    return None


_DEFAULT_TEXT_LINK_KINDS = {
    "script",
    "task",
    "finding",
    "fixreport",
    "hypothesis",
    "evidence",
    "chat_msg",
    "topic",
    "table",
    "proto",
    "contract",
    "service_unit",
    "audit_id",
    "audit_file",
    "quantaudit",
    "codex_session",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class LinkParseResult:
    links: Tuple[Tuple[str, str], ...]
    warnings: Tuple[str, ...]


def _normalize_repo_path(path_s: str) -> str:
    p = (path_s or "").strip()
    if not p:
        return ""
    p2 = Path(p).as_posix()
    if p2.startswith("./"):
        p2 = p2[2:]
    return p2.strip("/")


def _iter_legacy_meta_keys(doc: Mapping[str, Any]) -> Iterable[str]:
    """
    Yield legacy meta header keys found at the top-level of a work-artefact doc.

    Historically, some artefacts used a project-specific `<project>_meta` key
    instead of the canonical `tessairact_meta` key. To avoid hard-coding old
    project names (or redacted placeholders), accept any top-level key
    ending in `_meta` that looks like a meta header mapping.
    """
    for k, v in doc.items():
        if not isinstance(k, str):
            continue
        if k == TESSAIRACT_META_KEY:
            continue
        if not k.endswith("_meta"):
            continue
        if not isinstance(v, Mapping):
            continue
        if any(
            x in v
            for x in ("links", "header_version", "kind", "area", "uid", "provenance")
        ):
            yield k


def find_meta_header(doc: Mapping[str, Any]) -> Tuple[str, Optional[Mapping[str, Any]]]:
    """
    Return (key, mapping) for the best-effort meta header.

    Preference order:
      1) `tessairact_meta`
      2) legacy `*_meta` keys (see `_iter_legacy_meta_keys`)
      3) none
    """
    tm = doc.get(TESSAIRACT_META_KEY)
    if isinstance(tm, Mapping):
        return TESSAIRACT_META_KEY, tm
    for k in _iter_legacy_meta_keys(doc):
        tm2 = doc.get(k)
        if isinstance(tm2, Mapping):
            return k, tm2
    return "", None


def parse_link_tokens(raw: Any) -> LinkParseResult:
    """
    Parse link tokens into normalized (kind, id) tuples.

    Backward compatible:
    - strict grammar per architecture/LINKS_CONTRACT.md is accepted without warnings.
    - a looser grammar (kind:\\S+) is accepted with a warning.
    - invalid items are ignored (never hard-fail).
    """
    if raw is None:
        return LinkParseResult(links=tuple(), warnings=tuple())
    if isinstance(raw, (str, int, float, bool)):
        items: List[Any] = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        return LinkParseResult(links=tuple(), warnings=("links_not_list",))

    out: List[Tuple[str, str]] = []
    warnings: List[str] = []
    for item in items:
        s = str(item or "").strip()
        if not s:
            continue
        if ":" not in s:
            continue
        kind_raw, ident_raw = s.split(":", 1)
        kind = kind_raw.strip().lower()
        ident = ident_raw.strip()
        if not kind or not ident:
            continue
        if kind == "codex_session" and not _CODEX_SESSION_ID_RE.fullmatch(ident):
            continue
        token_norm = f"{kind}:{ident}"
        if _STRICT_LINK_RE.fullmatch(token_norm):
            out.append((kind, ident))
            continue
        if _LOOSE_LINK_RE.fullmatch(token_norm):
            if not _KIND_RE.fullmatch(kind):
                continue
            # Defensive: do not accept backslash escapes (e.g. "\\n") as link IDs.
            # These show up in recovered/serialized text blobs and can pollute durable stores.
            if "\\" in ident:
                continue
            out.append((kind, ident))
            warnings.append("noncanonical_link_token")
            continue
        continue

    return LinkParseResult(links=tuple(out), warnings=tuple(sorted(set(warnings))))


def dedupe_link_tokens(tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for t in tokens:
        s = str(t or "").strip()
        if not s:
            continue
        key = s.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def extract_link_tokens_from_text(
    text: str, *, allow_kinds: Optional[Iterable[str]] = None
) -> List[str]:
    """
    Extract link tokens from free-text without mutating the input.

    - Pulls explicit kind:id tokens embedded in text.
    - Canonicalizes bare IDs (BF/H/E/M/FND) into fixreport/hypothesis/evidence/chat_msg/finding links.
    - Returns de-duplicated link tokens (strict kind:id).
    """
    s = str(text or "")
    if not s:
        return []
    found = _EMBEDDED_LINK_RE.findall(s)
    allow = {
        str(x).strip().lower()
        for x in (allow_kinds or _DEFAULT_TEXT_LINK_KINDS)
        if str(x).strip()
    }
    parsed = parse_link_tokens(found)
    out: List[str] = []
    no_leading_slash_kinds = {
        "script",
        "task",
        "topic",
        "table",
        "proto",
        "contract",
        "service_unit",
        "quantaudit",
        "codex_session",
    }
    for k, i in parsed.links:
        if allow and k not in allow:
            continue
        # Fail-closed: these kinds are never expected to use absolute IDs/paths.
        if k in no_leading_slash_kinds and str(i or "").startswith("/"):
            continue
        # Normalize common prose punctuation attached to tokens (e.g. "task:abc123.", "script:foo.py)").
        # Prefer strict ID-like normalization for known kinds, then fall back to a safe trailing-punct strip.
        if k in {"fixreport", "hypothesis", "evidence", "chat_msg", "finding"}:
            norm = _normalize_id_like_link(k, i)
            if norm:
                out.append(norm)
                continue
            # Fail-closed for ID-like kinds: only accept canonical IDs (prevents false positives like "finding:FND").
            if k == "fixreport" and _BF_ID_RE.fullmatch(i):
                out.append(f"fixreport:{i}")
                continue
            if k == "hypothesis" and _H_ID_RE.fullmatch(i):
                out.append(f"hypothesis:{i}")
                continue
            if k == "evidence" and _E_ID_RE.fullmatch(i):
                out.append(f"evidence:{i}")
                continue
            if k == "chat_msg" and _M_ID_RE.fullmatch(i):
                out.append(f"chat_msg:{i}")
                continue
            if k == "finding" and _FND_ID_RE.fullmatch(i):
                out.append(f"finding:{i}")
                continue
            continue
        stripped = _strip_trailing_id_punct(i)
        if k == "script" and stripped.endswith("/"):
            continue
        if stripped != i and _STRICT_LINK_RE.fullmatch(f"{k}:{stripped}"):
            out.append(f"{k}:{stripped}")
        else:
            out.append(f"{k}:{i}")
    if not allow or "fixreport" in allow:
        for bf in _BF_BARE_RE.findall(s):
            out.append(f"fixreport:{bf}")
    if not allow or "hypothesis" in allow:
        for h in _H_BARE_RE.findall(s):
            out.append(f"hypothesis:{h}")
    if not allow or "evidence" in allow:
        for e in _E_BARE_RE.findall(s):
            out.append(f"evidence:{e}")
    if not allow or "chat_msg" in allow:
        for m in _M_BARE_RE.findall(s):
            out.append(f"chat_msg:{m}")
    if not allow or "finding" in allow:
        for fnd in _FND_BARE_RE.findall(s):
            out.append(f"finding:{fnd}")
    return dedupe_link_tokens(out)


def extract_link_tokens_from_texts(
    texts: Iterable[str],
    *,
    allow_kinds: Optional[Iterable[str]] = None,
) -> List[str]:
    out: List[str] = []
    for t in texts:
        out.extend(extract_link_tokens_from_text(t, allow_kinds=allow_kinds))
    return dedupe_link_tokens(out)


def extract_links_from_doc(doc: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Extract link tokens from a YAML/JSON-like mapping.

    Precedence:
      1) tessairact_meta.links
      2) legacy: *_meta.links (backward compatible)
      2) legacy locations:
         - top-level links
         - fixreport.links
         - audit_manifest.links
         - audit_versions[*].links
         - module_manifest.links / module_manifest.module.links
    Returns (links, warnings).
    """
    warnings: List[str] = []

    meta_keys = [TESSAIRACT_META_KEY, *_iter_legacy_meta_keys(doc)]
    for meta_key in meta_keys:
        tm = doc.get(meta_key)
        if not isinstance(tm, Mapping):
            continue
        links_raw = tm.get("links")
        if links_raw is None:
            continue
        parsed = parse_link_tokens(links_raw)
        return (
            dedupe_link_tokens([f"{k}:{i}" for k, i in parsed.links]),
            list(parsed.warnings),
        )

    # legacy: top-level links
    parsed_all: List[str] = []

    parsed = parse_link_tokens(doc.get("links"))
    parsed_all.extend([f"{k}:{i}" for k, i in parsed.links])
    warnings.extend(list(parsed.warnings))

    fr = doc.get("fixreport")
    if isinstance(fr, Mapping):
        parsed = parse_link_tokens(fr.get("links"))
        parsed_all.extend([f"{k}:{i}" for k, i in parsed.links])
        warnings.extend(list(parsed.warnings))

    am = doc.get("audit_manifest")
    if isinstance(am, Mapping):
        parsed = parse_link_tokens(am.get("links"))
        parsed_all.extend([f"{k}:{i}" for k, i in parsed.links])
        warnings.extend(list(parsed.warnings))

    avs = doc.get("audit_versions")
    if isinstance(avs, list):
        for av in avs[:200]:
            if not isinstance(av, Mapping):
                continue
            parsed = parse_link_tokens(av.get("links"))
            parsed_all.extend([f"{k}:{i}" for k, i in parsed.links])
            warnings.extend(list(parsed.warnings))

    mm = doc.get("module_manifest")
    if isinstance(mm, Mapping):
        parsed = parse_link_tokens(mm.get("links"))
        parsed_all.extend([f"{k}:{i}" for k, i in parsed.links])
        warnings.extend(list(parsed.warnings))
        mod = mm.get("module")
        if isinstance(mod, Mapping):
            parsed = parse_link_tokens(mod.get("links"))
            parsed_all.extend([f"{k}:{i}" for k, i in parsed.links])
            warnings.extend(list(parsed.warnings))

    return (dedupe_link_tokens(parsed_all), sorted(set(warnings)))


def validate_link_tokens(tokens: Sequence[str]) -> List[str]:
    """
    Return warnings for tokens that do not match strict LINKS_CONTRACT grammar.
    This is non-blocking by design.
    """
    warnings: List[str] = []
    for t in tokens:
        s = str(t or "").strip()
        if not s:
            continue
        if ":" not in s:
            warnings.append("invalid_link_token")
            continue
        kind_raw, ident_raw = s.split(":", 1)
        kind = kind_raw.strip().lower()
        ident = ident_raw.strip()
        if not kind or not ident:
            warnings.append("invalid_link_token")
            continue
        token_norm = f"{kind}:{ident}"
        if _STRICT_LINK_RE.fullmatch(token_norm):
            continue
        if _LOOSE_LINK_RE.fullmatch(token_norm):
            warnings.append("noncanonical_link_token")
            continue
        warnings.append("invalid_link_token")
    return sorted(set(warnings))


def make_uid(*, prefix: str, created_at_utc: str, salt: str) -> str:
    p = (prefix or "").strip().upper()[:4] or "ID"
    ts = (created_at_utc or "").strip().replace(":", "").replace("-", "")
    ts = ts.replace("T", "T").replace("Z", "Z")
    h = hashlib.sha256(
        f"{salt}|{p}|{created_at_utc}".encode("utf-8", errors="replace")
    ).hexdigest()[:8]
    return f"{p}_{ts}_{h}"


def _parse_timeout_env(name: str) -> Optional[float]:
    """
    Parse a timeout (seconds) from an environment variable.

    Accepted values:
      - positive float/int -> timeout seconds
      - "0", "none", "null", "off", "false" -> disables timeout (None)
      - invalid/unset -> None
    """
    v = os.environ.get(name)
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    if s in {"0", "none", "null", "off", "false", "no"}:
        return None
    try:
        f = float(s)
    except Exception:
        return None
    if f <= 0:
        return None
    return f


def _effective_git_timeout(timeout_s: Optional[float]) -> Optional[float]:
    if timeout_s is not None:
        try:
            f = float(timeout_s)
        except Exception:
            return None
        if f <= 0:
            return None
        return f
    return _parse_timeout_env(_GIT_TIMEOUT_ENV)


def _run_git(
    repo_root: str,
    args: Sequence[str],
    *,
    timeout_s: Optional[float],
    capture_stdout: bool,
) -> Tuple[Optional[int], str]:
    """
    Run a git command and return (returncode, stdout_stripped).

    - returncode is None if the process could not be started or timed out.
    - stdout is empty on errors/timeouts or if capture_stdout is False.
    """
    cmd = ["git", "-C", repo_root, *list(args)]
    try:
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, ""
    except Exception:
        return None, ""
    if not capture_stdout:
        return int(cp.returncode), ""
    try:
        out = (cp.stdout or b"").decode("utf-8", errors="replace").strip()
    except Exception:
        out = ""
    return int(cp.returncode), out


def _git_dir_present(repo_root: str) -> bool:
    try:
        return (Path(repo_root) / ".git").exists()
    except OSError:
        return False


def _git_dirty_via_status(
    repo_root: str, *, timeout_s: Optional[float], include_untracked: bool
) -> Optional[bool]:
    args: List[str] = ["status", "--porcelain=v1"]
    if not include_untracked:
        args.append("--untracked-files=no")
    rc, out = _run_git(repo_root, args, timeout_s=timeout_s, capture_stdout=True)
    if rc is None:
        return None
    if rc != 0:
        return None
    return bool((out or "").strip())


def _git_dirty_via_diff(
    repo_root: str, *, timeout_s: Optional[float]
) -> Optional[bool]:
    rc, _ = _run_git(
        repo_root, ["diff", "--quiet"], timeout_s=timeout_s, capture_stdout=False
    )
    if rc is None:
        return None
    # Preserve historical semantics: any non-zero return code is treated as "dirty".
    return bool(rc != 0)


@lru_cache(maxsize=32)
def _git_info_configurable(
    repo_root: str,
    timeout_s: Optional[float],
    dirty_mode: str,
    status_include_untracked: bool,
) -> Dict[str, Any]:
    if not _git_dir_present(repo_root):
        return {}

    timeout_eff = _effective_git_timeout(timeout_s)

    commit = ""
    rc, out = _run_git(
        repo_root, ["rev-parse", "HEAD"], timeout_s=timeout_eff, capture_stdout=True
    )
    if rc == 0:
        commit = out.strip()

    branch = ""
    rc, out = _run_git(
        repo_root,
        ["rev-parse", "--abbrev-ref", "HEAD"],
        timeout_s=timeout_eff,
        capture_stdout=True,
    )
    if rc == 0:
        branch = out.strip()

    mode = str(dirty_mode or "").strip().lower()
    if mode not in {"status", "legacy"}:
        mode = "status"

    dirty_opt: Optional[bool] = None
    if mode == "status":
        dirty_opt = _git_dirty_via_status(
            repo_root,
            timeout_s=timeout_eff,
            include_untracked=bool(status_include_untracked),
        )
        if dirty_opt is None:
            dirty_opt = _git_dirty_via_diff(repo_root, timeout_s=timeout_eff)
    else:
        dirty_opt = _git_dirty_via_diff(repo_root, timeout_s=timeout_eff)

    # Preserve historical behavior for "cannot run git" cases: default to clean.
    dirty = bool(dirty_opt) if dirty_opt is not None else False

    out_d: Dict[str, Any] = {}
    if commit:
        out_d["commit"] = commit
    if branch:
        out_d["branch"] = branch
    out_d["dirty"] = dirty
    return out_d


@lru_cache(maxsize=8)
def _git_info(repo_root: str) -> Dict[str, Any]:
    # Backwards compatible default: no explicit timeout; dirty detection via status (more complete),
    # with fallback to legacy diff when status fails.
    return _git_info_configurable(repo_root, None, "status", True)


def _safe_get_user() -> str:
    try:
        return getpass.getuser()
    except OSError:
        return ""
    except Exception:
        # Keep best-effort behavior: provenance is non-critical metadata.
        return ""


def _redact_value(value: str, *, salt: str) -> str:
    v = str(value or "").strip()
    if not v:
        return ""
    s = str(salt or "")
    h = hashlib.sha256(f"{s}|{v}".encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"sha256:{h}"


def build_provenance(
    *,
    repo_root: Optional[Path],
    tool: str,
    tool_run_id: str = "",
    tool_version: str = "",
    include_git: bool = True,
    # Optional hardening knobs (backwards compatible defaults preserve old output).
    git_timeout_s: Optional[float] = None,
    git_dirty_mode: str = "status",
    git_status_include_untracked: bool = True,
    include_host: bool = True,
    include_user: bool = True,
    include_repo_root: bool = True,
    redact_host: bool = False,
    redact_user: bool = False,
    redact_repo_root: bool = False,
    redaction_salt: str = "",
) -> Dict[str, Any]:
    rr_full = ""
    try:
        rr_full = str(repo_root.resolve()) if repo_root is not None else ""
    except Exception:
        rr_full = str(repo_root) if repo_root is not None else ""

    host_val = ""
    if include_host:
        try:
            host_val = socket.gethostname()
        except Exception:
            host_val = ""

    user_val = ""
    if include_user:
        user_val = _safe_get_user()

    repo_root_val = rr_full if include_repo_root else ""

    if redact_host and host_val:
        host_val = _redact_value(host_val, salt=redaction_salt)
    if redact_user and user_val:
        user_val = _redact_value(user_val, salt=redaction_salt)
    if redact_repo_root and repo_root_val:
        repo_root_val = _redact_value(repo_root_val, salt=redaction_salt)

    out: Dict[str, Any] = {
        "tool": str(tool or "").strip(),
        "tool_run_id": str(tool_run_id or "").strip(),
        "tool_version": str(tool_version or "").strip(),
        "host": host_val,
        "user": user_val,
        "repo_root": repo_root_val,
    }
    # remove empties
    out = {k: v for k, v in out.items() if v}
    if include_git and rr_full:
        gi = _git_info_configurable(
            rr_full,
            git_timeout_s,
            str(git_dirty_mode or "status"),
            bool(git_status_include_untracked),
        )
        if gi:
            out["git"] = gi
    return out


def _legacy_meta_migration_plan(
    doc: Mapping[str, Any],
) -> Tuple[str, Optional[Mapping[str, Any]], bool]:
    """
    Determine if we should migrate from a legacy `*_meta` key.

    Returns (legacy_key, legacy_mapping, destructive_pop).

    - destructive_pop is True only if we are sufficiently confident the legacy mapping
      is truly a tessairact-style meta header (reduces risk of accidental data loss).
    """
    for k in _iter_legacy_meta_keys(doc):
        v = doc.get(k)
        if not isinstance(v, Mapping):
            continue
        keys = set(str(x) for x in v.keys())
        strong_indicators = {
            "header_version",
            "kind",
            "area",
            "uid",
            "created_at_utc",
            "updated_at_utc",
            "created_by",
            "updated_by",
            "provenance",
            "title",
        }
        destructive = bool(keys.intersection(strong_indicators))
        return k, v, destructive
    return "", None, False


def ensure_tessairact_meta_header(
    doc: MutableMapping[str, Any],
    *,
    kind: str,
    area: str,
    uid: str,
    actor: str,
    title: str = "",
    links: Optional[Sequence[str]] = None,
    repo_root: Optional[Path] = None,
    tool: str = "",
    tool_run_id: str = "",
    created_at_utc: str = "",
    updated_at_utc: str = "",
    # Optional hardening knobs (defaults preserve historical behavior as closely as possible).
    provenance_include_host: bool = True,
    provenance_include_user: bool = True,
    provenance_include_repo_root: bool = True,
    provenance_redact_host: bool = False,
    provenance_redact_user: bool = False,
    provenance_redact_repo_root: bool = False,
    provenance_redaction_salt: str = "",
    git_timeout_s: Optional[float] = None,
    git_dirty_mode: str = "status",
    git_status_include_untracked: bool = True,
) -> Tuple[MutableMapping[str, Any], List[str]]:
    """
    Ensure a minimal tessairact_meta header exists and is updated.

    - If missing, creates it and sets created_at_utc and created_by.
    - Always updates updated_at_utc and updated_by.
    - If links are provided, merges and de-dupes them (preserving order).
    Returns (doc, warnings).
    """
    warnings: List[str] = []
    now = utc_now_iso()
    ts_created = (created_at_utc or "").strip() or now
    ts_updated = (updated_at_utc or "").strip() or now
    header: Dict[str, Any]
    raw = doc.get(TESSAIRACT_META_KEY)

    if not isinstance(raw, Mapping):
        legacy_key, legacy_map, destructive = _legacy_meta_migration_plan(doc)
        if legacy_key and isinstance(legacy_map, Mapping):
            # Best-effort migration from legacy key to canonical key.
            legacy = dict(legacy_map or {})

            # Avoid destructive mutation on weak matches to reduce risk of accidental data loss.
            if destructive:
                doc.pop(legacy_key, None)

            # If the legacy mapping is a weak match, only migrate "links" to avoid polluting the header
            # with unrelated metadata.
            if not destructive and "links" in legacy and len(legacy.keys()) == 1:
                doc[TESSAIRACT_META_KEY] = {"links": legacy.get("links")}
                raw = doc[TESSAIRACT_META_KEY]
            else:
                doc[TESSAIRACT_META_KEY] = legacy
                raw = legacy

    if isinstance(raw, Mapping):
        header = dict(raw)
    else:
        header = {}
        doc[TESSAIRACT_META_KEY] = header
        header["created_at_utc"] = ts_created
        header["created_by"] = actor

    if not str(header.get("created_at_utc") or "").strip():
        header["created_at_utc"] = ts_created

    header["header_version"] = int(TESSAIRACT_META_HEADER_VERSION)
    header["kind"] = str(kind or "").strip().lower()
    header["area"] = str(area or "").strip().lower()
    header["uid"] = str(uid or "").strip()
    if title:
        header["title"] = str(title)

    header["updated_at_utc"] = ts_updated
    header["updated_by"] = actor

    if links is not None:
        existing = header.get("links")
        merged: List[str] = []
        if isinstance(existing, list):
            merged.extend([str(x) for x in existing if x is not None])
        merged.extend([str(x) for x in links if x is not None])
        merged = dedupe_link_tokens(merged)
        header["links"] = merged
        warnings.extend(validate_link_tokens(merged))

    if tool:
        header["provenance"] = build_provenance(
            repo_root=repo_root,
            tool=tool,
            tool_run_id=tool_run_id,
            include_git=True,
            git_timeout_s=git_timeout_s,
            git_dirty_mode=git_dirty_mode,
            git_status_include_untracked=git_status_include_untracked,
            include_host=provenance_include_host,
            include_user=provenance_include_user,
            include_repo_root=provenance_include_repo_root,
            redact_host=provenance_redact_host,
            redact_user=provenance_redact_user,
            redact_repo_root=provenance_redact_repo_root,
            redaction_salt=provenance_redaction_salt,
        )

    doc[TESSAIRACT_META_KEY] = header
    return doc, sorted(set(warnings))


def infer_area_from_path(path: str) -> str:
    p = _normalize_repo_path(path)
    top = p.split("/", 1)[0] if p else ""
    if top == "artifacts":
        parts = p.split("/")
        sub = parts[1] if len(parts) >= 2 else ""
        if sub in {"audit", "evaluations", "reports"}:
            return "research"
        if sub == "llm":
            return "portal"
        if sub == "stores":
            # Some stores are portal-facing (portal/wiki, llm_specialists), most are ops governance.
            third = parts[2] if len(parts) >= 3 else ""
            if third in {
                "portal",
                "llm_specialists",
                "llm_tools",
                "llm_usage",
                "review_agent",
            }:
                return "portal"
            if third in {
                "health",
                "healthstream",
                "service_health_check",
                "systemd_agent",
            }:
                return "infra"
            return "ops"
    mapping = {
        "consumer": "portal",
        "services": "infra",
        "systemd": "infra",
        "clickhouse": "infra",
        "scripts": "ops",
        "evaluations": "research",
        "audit": "research",
        "reports": "research",
        "state": "ops",
        "analytics": "analytics",
        "llm": "portal",
        "agentic_tools": "ops",
    }
    return mapping.get(top, "other")
