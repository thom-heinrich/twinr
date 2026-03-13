"""
codexctx: Codex session/thread correlation helper.

Goal
----
Provide a best-effort way for repo tooling (findings/fixreport/chat/tasks/healthstream/checkreview)
to automatically attach high-precision join keys to artifacts created from Codex runs.

This enables downstream analysis (anti-pattern mining, outcome joins, lead-time metrics) without
relying on noisy time-window heuristics.

Link contract
-------------
Emits link tokens compatible with `architecture/LINKS_CONTRACT.md`:
- `codex_session:session/<uuid>` and/or
- `codex_session:thread/<uuid>`

Extraction sources (priority)
-----------------------------
1) Explicit env (preferred, deterministic):
   - `CAIA_CODEX_SESSION_ID` / `CAIA_CODEX_THREAD_ID`
   - (also accepts `CODEX_SESSION_ID` / `CODEX_THREAD_ID` for compatibility)
2) Best-effort fallback:
   - parse `SUDO_COMMAND` for `... resume <id> ...` (works for many interactive Codex invocations)
3) Best-effort fallback (proc tree):
   - walk parent process cmdlines to find a `resume <id>` token (useful when env is sanitized)
4) Best-effort fallback (Codex sessions dir):
   - find the newest `rollout-...-<session_id>.jsonl` under `CODEX_HOME/sessions/**`
5) Best-effort fallback (Codex history):
   - tail `history.jsonl` under CODEX_HOME (or common homes) and use the most recent session_id
6) Best-effort fallback (Codex shell snapshots):
   - take the newest file stem under `CODEX_HOME/shell_snapshots` as session_id

Safety / compatibility
----------------------
- Pure stdlib; reads `/proc` only as a last resort.
- Never raises on parse failures (returns empty context/tokens).
- Can be disabled via `CAIA_CODEXCTX_DISABLE=1`.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class CodexContext:
    session_id: str
    thread_id: str
    source: str  # "env" | "sudo_command" | "proc_tree" | "sessions_dir" | "history_jsonl" | "shell_snapshots" | "none"


_CODEX_SESSION_ID_RE = re.compile(
    r"(?P<sid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    flags=re.IGNORECASE,
)
_CODEX_SESSION_ID_FULL_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    flags=re.IGNORECASE,
)
_CODEX_SESSION_TOKEN_PREFIX = "codex_session:session/"
_CODEX_THREAD_TOKEN_PREFIX = "codex_session:thread/"


def _clean_id(raw: object, *, max_len: int = 128) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if any(ch.isspace() for ch in s):
        # Defense-in-depth: ids should never contain whitespace; this also prevents token injection.
        return ""
    if len(s) > int(max_len):
        return s[: int(max_len)]
    return s


def _normalize_codex_id(raw: object, *, max_len: int = 128) -> str:
    """
    Normalize a Codex session/thread id to canonical lower-case UUID form.

    Accepted inputs:
    - `<uuid>`
    - `session/<uuid>`
    - `thread/<uuid>`
    - `codex_session:session/<uuid>`
    - `codex_session:thread/<uuid>`
    """
    s = str(raw or "").strip()
    if not s:
        return ""
    if any(ch.isspace() for ch in s):
        return ""
    if len(s) > int(max_len):
        return ""
    low = s.lower()
    for pfx in (
        _CODEX_SESSION_TOKEN_PREFIX,
        _CODEX_THREAD_TOKEN_PREFIX,
        "session/",
        "thread/",
    ):
        if low.startswith(pfx):
            s = s[len(pfx) :].strip()
            break
    if not _CODEX_SESSION_ID_FULL_RE.fullmatch(s):
        return ""
    return s.lower()


def _is_codex_link_token(raw: object) -> bool:
    s = str(raw or "").strip().lower()
    return s.startswith("codex_session:")


def normalize_codex_link_token(raw: object) -> str:
    """
    Return canonical codex_session link token or empty string if invalid/non-codex.
    """
    s = str(raw or "").strip()
    if not s:
        return ""
    low = s.lower()
    if low.startswith(_CODEX_SESSION_TOKEN_PREFIX):
        sid = _normalize_codex_id(s[len(_CODEX_SESSION_TOKEN_PREFIX) :])
        return f"{_CODEX_SESSION_TOKEN_PREFIX}{sid}" if sid else ""
    if low.startswith(_CODEX_THREAD_TOKEN_PREFIX):
        tid = _normalize_codex_id(s[len(_CODEX_THREAD_TOKEN_PREFIX) :])
        return f"{_CODEX_THREAD_TOKEN_PREFIX}{tid}" if tid else ""
    return ""


def codex_session_id_from_link_token(raw: object) -> str:
    """
    Return canonical session id from `codex_session:session/<id>` token, else "".
    """
    norm = normalize_codex_link_token(raw)
    if not norm.startswith(_CODEX_SESSION_TOKEN_PREFIX):
        return ""
    return norm[len(_CODEX_SESSION_TOKEN_PREFIX) :]


def _env_bool(environ: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw = environ.get(key)
    if raw is None:
        return default
    v = str(raw).strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "y", "on"}


def _first_env(environ: Mapping[str, str], keys: Sequence[str]) -> str:
    for k in keys:
        v = _normalize_codex_id(environ.get(k))
        if v:
            return v
    return ""


def _has_explicit_invalid_id_value(
    environ: Mapping[str, str], keys: Sequence[str]
) -> bool:
    """
    Return True when an explicitly provided Codex ID value is non-empty but invalid.

    Empty placeholders (e.g. VAR=\"\") are treated as "not provided" and should not
    suppress non-env fallbacks.
    """
    for k in keys:
        if k not in environ:
            continue
        raw_obj = environ.get(k)
        raw = "" if raw_obj is None else str(raw_obj)
        if not raw.strip():
            continue
        if not _normalize_codex_id(raw):
            return True
    return False


def _parse_resume_id_from_sudo_command(sudo_command: str) -> str:
    """
    Extract the first token after a literal `resume` subcommand.

    Examples
    --------
    - `/usr/local/bin/codex resume 019bea... --yolo`
    - `/usr/local/bin/codex exec resume 4251... --json ...`
    """
    raw = str(sudo_command or "").strip()
    if not raw:
        return ""
    try:
        argv = shlex.split(raw)
    except Exception:
        argv = raw.split()
    resume_id, _ = _parse_resume_id_and_kind_from_argv(argv)
    return resume_id


def _parse_resume_id_from_argv(argv: Sequence[str]) -> str:
    """
    Extract the first token after a literal `resume` subcommand from an argv list.

    This is shared by the SUDO_COMMAND parser and the /proc cmdline fallback.
    """
    resume_id, _ = _parse_resume_id_and_kind_from_argv(argv)
    return resume_id


def _parse_resume_id_and_kind_from_argv(argv: Sequence[str]) -> tuple[str, str]:
    """
    Extract the first token after a literal `resume` subcommand from an argv list.

    Returns
    -------
    (resume_id, kind)

    kind:
    - "resume": `codex resume <id>`
    - "exec_resume": `codex exec resume <id>` (commonly used for thread resumes)
    """
    for i, tok in enumerate(argv):
        if tok != "resume":
            continue
        if i + 1 >= len(argv):
            return ("", "")
        cand = argv[i + 1]
        if cand.startswith("-"):
            # Defensive: avoid treating flags as ids.
            return ("", "")
        rid = _normalize_codex_id(cand)
        if not rid:
            return ("", "")
        kind = "exec_resume" if i > 0 and argv[i - 1] == "exec" else "resume"
        return (rid, kind)
    return ("", "")


def _read_proc_cmdline(pid: int) -> list[str]:
    """
    Read `/proc/<pid>/cmdline` as an argv list.

    Never raises (returns []).
    """
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as handle:
            raw = handle.read()
    except Exception:
        return []
    if not raw:
        return []
    parts = [p for p in raw.split(b"\0") if p]
    out: list[str] = []
    for p in parts:
        try:
            out.append(p.decode("utf-8", errors="ignore"))
        except Exception:
            continue
    return out


def _read_proc_ppid(pid: int) -> int:
    """
    Read `/proc/<pid>/status` and return PPid.

    Never raises (returns 0).
    """
    try:
        with open(f"/proc/{int(pid)}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("PPid:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    return 0
                try:
                    return int(parts[1])
                except Exception:
                    return 0
    except Exception:
        return 0
    return 0


def _tail_jsonl_dicts(
    path: Path, *, max_bytes: int = 256 * 1024, max_lines: int = 200
) -> list[dict]:
    """
    Best-effort: read the last `max_lines` JSONL records from `path`.

    Never raises (returns []).
    """
    try:
        p = Path(path)
    except Exception:
        return []
    try:
        if not p.exists() or not p.is_file():
            return []
    except Exception:
        return []

    try:
        size = int(p.stat().st_size)
    except Exception:
        size = 0
    try:
        with p.open("rb") as handle:
            if size > int(max_bytes):
                handle.seek(max(0, size - int(max_bytes)))
            raw = handle.read()
    except Exception:
        return []
    if not raw:
        return []

    # Decode forgivingly (history.jsonl is UTF-8, but we prefer robustness over strictness).
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        return []

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    out: list[dict] = []
    for ln in lines[-max(1, int(max_lines)) :]:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _candidate_codex_homes(environ: Mapping[str, str]) -> list[Path]:
    homes: list[Path] = []
    for key in ("CAIA_CODEX_HOME", "CODEX_HOME"):
        raw = str(environ.get(key) or "").strip()
        if not raw:
            continue
        try:
            homes.append(Path(raw))
        except Exception:
            continue

    # If CODEX_HOME was provided explicitly, treat it as authoritative.
    if homes:
        return homes

    # Common defaults (best-effort; existence checked later).
    homes.append(Path("/root/.codex"))

    # If running under sudo, the original user may have the relevant CODEX_HOME.
    sudo_user = str(environ.get("SUDO_USER") or "").strip()
    if sudo_user:
        homes.append(Path("/home") / sudo_user / ".codex")

    user = str(environ.get("USER") or "").strip()
    if user:
        homes.append(Path("/home") / user / ".codex")

    # Repo default location (common on this host).
    homes.append(Path("/home/thh/.codex"))

    # Dedupe while preserving order.
    seen: set[str] = set()
    out: list[Path] = []
    for h in homes:
        key = str(h)
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _infer_session_id_from_history_jsonl(environ: Mapping[str, str]) -> str:
    """
    Best-effort: infer the most recent Codex session_id from `history.jsonl`.

    This is a fallback for runs started without `resume <id>` (no SUDO_COMMAND hint)
    and without explicit env propagation.

    Never raises (returns "").
    """
    best: tuple[int, str] | None = None
    for home in _candidate_codex_homes(environ):
        p = home / "history.jsonl"
        for obj in _tail_jsonl_dicts(p):
            sid = _normalize_codex_id(obj.get("session_id"))
            if not sid:
                continue
            try:
                ts = int(obj.get("ts") or 0)
            except Exception:
                ts = 0
            if best is None or ts > best[0]:
                best = (ts, sid)
    return best[1] if best else ""


def _infer_session_id_from_shell_snapshots(environ: Mapping[str, str]) -> str:
    """
    Best-effort: infer the most recent Codex session_id from CODEX_HOME/shell_snapshots.

    Rationale: some Codex runners may not persist `history.jsonl` in environments where
    the repo tools still run. Shell snapshots are lightweight and often available even
    when history is pruned.

    Never raises (returns "").
    """
    best: tuple[int, str] | None = None
    for home in _candidate_codex_homes(environ):
        p = home / "shell_snapshots"
        try:
            if (not p.exists()) or (not p.is_dir()):
                continue
        except Exception:
            continue
        try:
            entries = [x for x in p.iterdir()]
        except Exception:
            continue
        for fp in entries:
            try:
                if (not fp.is_file()) or fp.is_symlink():
                    continue
            except Exception:
                continue
            # Codex uses `<session_id>.sh` for snapshots. Be permissive and use the stem.
            sid = _normalize_codex_id(fp.stem)
            if not sid:
                continue
            try:
                mtime_ns = int(fp.stat().st_mtime_ns)
            except Exception:
                mtime_ns = 0
            if best is None or mtime_ns > best[0]:
                best = (mtime_ns, sid)
    return best[1] if best else ""


def _infer_session_id_from_sessions_dir(environ: Mapping[str, str]) -> str:
    """
    Best-effort: infer the most recent Codex session_id from CODEX_HOME/sessions rollouts.

    Rationale: for some runners, `history.jsonl` can lag or be pruned, while rollouts are the
    canonical trace store. We keep this bounded by only scanning a small set of recent date
    folders.

    Never raises (returns "").
    """

    def _sorted_numeric_dirs(parent: Path, *, limit: int) -> list[Path]:
        try:
            entries = [x for x in parent.iterdir()]
        except Exception:
            return []
        dirs: list[Path] = []
        for p in entries:
            try:
                if (not p.is_dir()) or p.is_symlink():
                    continue
            except Exception:
                continue
            if not p.name.isdigit():
                continue
            dirs.append(p)
        dirs.sort(key=lambda p: int(p.name), reverse=True)
        return dirs[: max(1, int(limit))]

    best: tuple[int, str] | None = None
    for home in _candidate_codex_homes(environ):
        base = home / "sessions"
        try:
            if (not base.exists()) or (not base.is_dir()):
                continue
        except Exception:
            continue
        # sessions/<YYYY>/<MM>/<DD>/rollout-...-<sid>.jsonl
        for year_dir in _sorted_numeric_dirs(base, limit=2):
            for month_dir in _sorted_numeric_dirs(year_dir, limit=3):
                for day_dir in _sorted_numeric_dirs(month_dir, limit=5):
                    try:
                        files = [x for x in day_dir.iterdir()]
                    except Exception:
                        continue
                    for fp in files:
                        try:
                            if (not fp.is_file()) or fp.is_symlink():
                                continue
                        except Exception:
                            continue
                        name = fp.name
                        if (not name.startswith("rollout-")) or (
                            not name.endswith(".jsonl")
                        ):
                            continue
                        m = _CODEX_SESSION_ID_RE.search(name)
                        if not m:
                            continue
                        sid = _normalize_codex_id(m.group("sid"))
                        if not sid:
                            continue
                        try:
                            mtime_ns = int(fp.stat().st_mtime_ns)
                        except Exception:
                            mtime_ns = 0
                        if best is None or mtime_ns > best[0]:
                            best = (mtime_ns, sid)
    return best[1] if best else ""


def _pid_is_alive(pid: int) -> bool:
    try:
        p = int(pid)
    except Exception:
        return False
    if p <= 0:
        return False
    try:
        os.kill(p, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _infer_session_id_from_ccodex_registry(
    environ: Mapping[str, str],
    *,
    max_records: int = 2000,
) -> str:
    """
    Best-effort: use the local ccodex session registry as a deterministic hint.

    When running under `ccodex`, tool processes typically inherit:
    - `CCODEX_SESSION_NAME=<human_name>`
    - `CCODEX_NAME=<human_name>` (compat)

    We map name -> conversation_id (Codex threadId) and return it as a session_id so
    downstream artifacts can be joined to Codex traces via `codex_session:*` tokens.
    """
    name = _clean_id(environ.get("CCODEX_SESSION_NAME") or environ.get("CCODEX_NAME"))
    if not name:
        return ""
    root = (
        str(
            environ.get("CCODEX_SESSIONS_ROOT_DIR") or "/tmp/tessairact_codex_sessions"
        ).strip()
        or "/tmp/tessairact_codex_sessions"
    )
    try:
        entries = list(os.scandir(root))
    except Exception:
        return ""

    matches: list[str] = []
    for ent in entries[: max(1, int(max_records))]:
        if not ent.is_file() or not ent.name.endswith(".json"):
            continue
        try:
            raw = Path(ent.path).read_text(encoding="utf-8")
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if str(obj.get("name") or "").strip() != name:
            continue
        try:
            pid = int(obj.get("pid") or 0)
        except Exception:
            pid = 0
        if pid and not _pid_is_alive(pid):
            continue
        cid = _normalize_codex_id(obj.get("conversation_id"))
        if not cid:
            continue
        matches.append(cid)
        if len({*matches}) > 1:
            # Ambiguous: multiple live sessions with same name.
            return ""
    if not matches:
        return ""
    uniq = list({*matches})
    return uniq[0] if len(uniq) == 1 else ""


def _infer_ccodex_session_id_from_thread_id(
    environ: Mapping[str, str],
    thread_id: str,
) -> str:
    """
    Best-effort: if CODEX_THREAD_ID points at a ccodex registry record, treat it as a session id.

    Motivation
    ----------
    In ccodex (Codex app-server bridge) sessions, the "thread id" exposed to tool processes is the
    app-server conversation_id. That id is the best join key we have and should be acceptable as a
    `codex_session:session/<id>` token.
    """
    tid = _normalize_codex_id(thread_id)
    if not tid:
        return ""
    root = (
        str(
            environ.get("CCODEX_SESSIONS_ROOT_DIR") or "/tmp/tessairact_codex_sessions"
        ).strip()
        or "/tmp/tessairact_codex_sessions"
    )
    try:
        path = Path(root) / f"{tid}.json"
    except Exception:
        return ""
    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""
    if _normalize_codex_id(obj.get("conversation_id")) != tid:
        return ""
    try:
        pid = int(obj.get("pid") or 0)
    except Exception:
        pid = 0
    if pid and not _pid_is_alive(pid):
        return ""
    return tid


def _infer_resume_from_proc_tree(
    start_pid: int, *, max_depth: int = 8
) -> tuple[str, str]:
    """
    Best-effort: walk the parent process chain and try to find a `resume <id>` argv.

    Never raises (returns ("","")).
    """
    try:
        pid = int(start_pid)
    except Exception:
        pid = 0
    seen: set[int] = set()
    for _ in range(max(1, int(max_depth))):
        if pid <= 0 or pid in seen:
            break
        seen.add(pid)
        argv = _read_proc_cmdline(pid)
        rid, kind = _parse_resume_id_and_kind_from_argv(argv)
        if rid:
            return (rid, kind)
        ppid = _read_proc_ppid(pid)
        if ppid <= 1 or ppid == pid:
            break
        pid = ppid
    return ("", "")


def _infer_session_id_from_non_env_fallbacks(
    env: Mapping[str, str],
    *,
    allow_proc_tree: bool,
) -> tuple[str, str]:
    """
    Infer a Codex session id without using explicit CAIA/CODEX session env vars.

    Returns
    -------
    (session_id, source)
    """
    sudo_cmd = str(env.get("SUDO_COMMAND") or "").strip()
    if sudo_cmd:
        try:
            argv = shlex.split(sudo_cmd)
        except Exception:
            argv = sudo_cmd.split()
        resume_id, kind = _parse_resume_id_and_kind_from_argv(argv)
        if resume_id:
            _ = kind
            return (resume_id, "sudo_command")

    if allow_proc_tree:
        resume_id, kind = _infer_resume_from_proc_tree(os.getpid())
        if resume_id:
            _ = kind
            return (resume_id, "proc_tree")

    sid = _infer_session_id_from_sessions_dir(env)
    if sid:
        return (sid, "sessions_dir")

    sid = _infer_session_id_from_history_jsonl(env)
    if sid:
        return (sid, "history_jsonl")

    sid = _infer_session_id_from_shell_snapshots(env)
    if sid:
        return (sid, "shell_snapshots")

    return ("", "none")


def get_codex_context(environ: Optional[Mapping[str, str]] = None) -> CodexContext:
    env = os.environ if environ is None else environ
    if _env_bool(env, "CAIA_CODEXCTX_DISABLE", default=False):
        return CodexContext(session_id="", thread_id="", source="none")

    session_keys = ("CAIA_CODEX_SESSION_ID", "CODEX_SESSION_ID")
    thread_keys = ("CAIA_CODEX_THREAD_ID", "CODEX_THREAD_ID")
    codex_id_keys = (*session_keys, *thread_keys)

    raw_env_has_codex_keys = any(k in env for k in codex_id_keys)
    session_id = _first_env(env, session_keys)
    thread_id = _first_env(env, thread_keys)
    if raw_env_has_codex_keys and not (session_id or thread_id):
        # Explicit non-empty invalid IDs should fail closed. Empty placeholders
        # are treated as "not provided" and may use fallbacks.
        if _has_explicit_invalid_id_value(env, codex_id_keys):
            return CodexContext(session_id="", thread_id="", source="none")

    if thread_id and not session_id:
        # Some ccodex runners only propagate CODEX_THREAD_ID to tool processes. In our
        # app-server bridge, conversation_id == thread_id is also the best session join key.
        sid = _infer_ccodex_session_id_from_thread_id(env, thread_id)
        if sid:
            return CodexContext(
                session_id=sid, thread_id=thread_id, source="ccodex_thread_id"
            )

        # Secondary ccodex hint: map via CCODEX_SESSION_NAME -> conversation_id.
        sid = _infer_session_id_from_ccodex_registry(env)
        if sid and sid == thread_id:
            return CodexContext(
                session_id=sid, thread_id=thread_id, source="ccodex_registry"
            )

        sid, sid_source = _infer_session_id_from_non_env_fallbacks(
            env,
            allow_proc_tree=(environ is None),
        )
        if sid:
            if sid == thread_id:
                return CodexContext(
                    session_id=sid, thread_id=thread_id, source=sid_source
                )
            # If a non-env fallback yields a different id than the explicit thread
            # id, prefer the explicit thread id as deterministic join key to avoid
            # cross-session misattribution.
            return CodexContext(
                session_id=thread_id, thread_id=thread_id, source="thread_env"
            )

        return CodexContext(
            session_id=thread_id, thread_id=thread_id, source="thread_env"
        )

    if session_id or thread_id:
        return CodexContext(session_id=session_id, thread_id=thread_id, source="env")

    # ccodex bridge sessions: deterministic mapping via local registry.
    sid = _infer_session_id_from_ccodex_registry(env)
    if sid:
        # In our app-server bridge, conversation_id is also the threadId.
        return CodexContext(session_id=sid, thread_id=sid, source="ccodex_registry")

    sid, sid_source = _infer_session_id_from_non_env_fallbacks(
        env,
        allow_proc_tree=(environ is None),
    )
    if sid:
        return CodexContext(session_id=sid, thread_id="", source=sid_source)

    return CodexContext(session_id="", thread_id="", source="none")


def codex_link_tokens(ctx: CodexContext) -> list[str]:
    out: list[str] = []
    sid = _normalize_codex_id(ctx.session_id)
    tid = _normalize_codex_id(ctx.thread_id)
    if sid:
        out.append(f"{_CODEX_SESSION_TOKEN_PREFIX}{sid}")
    if tid:
        out.append(f"{_CODEX_THREAD_TOKEN_PREFIX}{tid}")
    return out


def append_codex_links(
    existing: Optional[Sequence[object]] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> list[str]:
    """
    Return `existing` (as strings) with missing `codex_session:*` tokens appended.

    - Preserves original order.
    - Appends in stable order: session token first, then thread token.
    - Dedupe is exact-string based.
    """
    base: list[str] = []
    if existing:
        for x in existing:
            s = str(x or "").strip()
            if not s:
                continue
            # Keep non-codex tokens as-is; codex_session tokens are canonicalized
            # and malformed variants are dropped to preserve join correctness.
            if _is_codex_link_token(s):
                s_norm = normalize_codex_link_token(s)
                if not s_norm:
                    continue
                s = s_norm
            base.append(s)

    ctx = get_codex_context(environ=environ)
    tokens = codex_link_tokens(ctx)
    if not tokens:
        return base

    seen = set(base)
    out = list(base)
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out
