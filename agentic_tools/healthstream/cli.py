"""
Contract
- Purpose:
  - JSON-only CLI for `healthstream` (central store for health/DQ/ops events).
- Inputs:
  - Subcommands: schema|init|emit|list|prune.
- Outputs:
  - JSON on stdout (always). Logs to stderr.
- Invariants:
  - Store uses file locking; safe for concurrent agents/services.
  - Link tokens follow `architecture/LINKS_CONTRACT.md`.
- Error semantics:
  - On errors: prints `{ok:false,error,detail}` JSON and exits non-zero.
"""

##REFACTOR: 2026-01-16##

from __future__ import annotations

import argparse
import errno
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agentic_tools._store_layout import resolve_repo_path
from agentic_tools._governance_locking import resolve_retry_budget as _resolve_governance_retry_budget
from agentic_tools._governance_locking import resolve_retry_poll as _resolve_governance_retry_poll
from agentic_tools._governance_locking import run_mutation_with_retry as _run_governance_mutation_with_retry
from agentic_tools.codexctx import append_codex_links
from agentic_tools.healthstream.store import (
    FileLockTimeout as _HealthstreamFileLockTimeout,
    default_store,
    emit_event,
    get_store_path,
    list_events,
    load_store,
    prune_events,
    save_store,
)
from agentic_tools.healthstream.telemetry import setup_logging


REPO_ROOT = Path(__file__).resolve().parents[2]

_LOG = logging.getLogger(__name__)

# Contract-adjacent defaults.
_DEFAULT_LOCK_TIMEOUT_SEC = 10.0

# Safety caps (configurable via env). Defaults are intentionally permissive to avoid breaking
# existing consumers; operators can tighten via env.
_HARD_MAX_LIST_LIMIT = 1_000_000  # absolute safety net to avoid pathological stdout/memory blowups
_WARN_LIST_LIMIT = 10_000  # warn on stderr when above this and no explicit cap is set
_HARD_MAX_DATA_JSON_BYTES = 10_000_000  # 10MB safety net
_WARN_DATA_JSON_BYTES = 1_000_000  # warn when above this and no explicit cap is set


class _CLIError(Exception):
    """Internal exception used to convert CLI/usage problems into JSON errors."""

    def __init__(self, message: str, *, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = int(exit_code)


class _StdoutBrokenPipe(Exception):
    """Raised when stdout is closed (e.g., piping into `head`)."""


class _JsonArgumentParser(argparse.ArgumentParser):
    """
    ArgumentParser that never prints to stdout/stderr on errors.

    We emit all user-facing output as JSON from main().
    """

    def error(self, message: str) -> None:  # type: ignore[override]
        raise _CLIError(message, exit_code=2)

    def exit(self, status: int = 0, message: Optional[str] = None) -> None:  # type: ignore[override]
        # argparse might call exit() in some scenarios; convert to exception so main() can emit JSON.
        msg = (message or "").strip()
        raise _CLIError(msg or f"exit {status}", exit_code=status)


def _argument_parser_kwargs() -> Dict[str, Any]:
    """
    Build kwargs compatible across Python versions.

    - We prefer deterministic help formatting (no ANSI colors).
    - We avoid `argparse` exiting by itself when possible.
    """
    kw: Dict[str, Any] = {"allow_abbrev": False, "add_help": False}

    # exit_on_error added in Python 3.9
    try:
        argparse.ArgumentParser(exit_on_error=False)  # type: ignore[call-arg]
        kw["exit_on_error"] = False
    except TypeError:
        pass

    # color and suggest_on_error added in Python 3.14
    try:
        argparse.ArgumentParser(color=False)  # type: ignore[call-arg]
        kw["color"] = False
    except TypeError:
        pass
    try:
        argparse.ArgumentParser(suggest_on_error=False)  # type: ignore[call-arg]
        kw["suggest_on_error"] = False
    except TypeError:
        pass

    return kw


def _print(payload: Dict[str, Any], *, pretty: bool) -> None:
    """
    Print JSON to stdout, handling BrokenPipe robustly.
    """
    try:
        out = json.dumps(payload, indent=2, sort_keys=True) if pretty else json.dumps(payload, sort_keys=True)
        sys.stdout.write(out)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except BrokenPipeError as exc:
        # Downstream consumer closed the pipe (e.g., `| head`).
        # Silence further writes and signal to main() to exit with a SIGPIPE-like code.
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull_fd, sys.stdout.fileno())
            finally:
                os.close(devnull_fd)
        except Exception:
            pass
        raise _StdoutBrokenPipe() from exc


def _schema() -> Dict[str, Any]:
    commands = {
        "schema": "Emit JSON schema for tool discovery.",
        "init": "Initialize the store (idempotent).",
        "emit": "Append an event (optionally deduped).",
        "list": "List recent events with filters.",
        "prune": "Prune old events from the store (retention).",
    }
    return {
        "ok": True,
        "schema": {
            "summary": "Central health/DQ/ops event stream (file-backed, JSON-only).",
            "env": {
                "HEALTHSTREAM_FILE": "Override store path (default: .healthstream.json).",
                "HEALTHSTREAM_LOCK_TIMEOUT_SEC": "Lock acquisition timeout (default: 10).",
            },
        },
        "commands": commands,
    }


def _nonneg_int(value: str) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from exc
    if v < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return v


def _nonneg_float(value: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}") from exc
    if v < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return v


def _default_lock_timeout_sec() -> float:
    raw = os.environ.get("HEALTHSTREAM_LOCK_TIMEOUT_SEC", "").strip()
    if not raw:
        return _DEFAULT_LOCK_TIMEOUT_SEC
    try:
        v = float(raw)
    except ValueError as exc:
        raise ValueError(f"invalid HEALTHSTREAM_LOCK_TIMEOUT_SEC={raw!r}") from exc
    if v < 0:
        raise ValueError("HEALTHSTREAM_LOCK_TIMEOUT_SEC must be >= 0")
    return v


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        v = int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid {name}={raw!r}") from exc
    return v


def _env_bytes_limit(name: str, default: int = 0) -> int:
    v = _env_int(name, default=default)
    if v < 0:
        raise ValueError(f"{name} must be >= 0")
    return v


def _argv_wants_pretty(argv: List[str]) -> bool:
    return "--json-pretty" in argv


def _argv_requests_help(argv: List[str]) -> bool:
    return "-h" in argv or "--help" in argv


def _detect_cmd(argv: List[str]) -> Optional[str]:
    """
    Best-effort detection of the subcommand from argv without full parsing.

    We ignore global flags (with and without '=') and their values.
    """
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("-h", "--help", "--json-pretty"):
            i += 1
            continue
        if tok == "--log-level":
            i += 2
            continue
        if tok.startswith("--log-level="):
            i += 1
            continue
        if tok.startswith("-"):
            # Other options (likely subcommand options) — skip only token.
            i += 1
            continue
        # First non-flag token is the subcommand.
        return tok
    return None


def _build_parser() -> Tuple[argparse.ArgumentParser, Dict[str, argparse.ArgumentParser]]:
    global_parent = argparse.ArgumentParser(add_help=False)
    global_parent.add_argument("--json-pretty", action="store_true", help="Pretty-print JSON output.")
    global_parent.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"), help="stderr log level.")
    # Preserve -h/--help flags but handle them ourselves (JSON-only).
    global_parent.add_argument("-h", "--help", action="store_true", help="Show help (JSON) and exit.")

    p = _JsonArgumentParser(prog="healthstream", parents=[global_parent], **_argument_parser_kwargs())
    sub = p.add_subparsers(dest="cmd", required=False)

    subparsers: Dict[str, argparse.ArgumentParser] = {}

    schema_p = sub.add_parser("schema", parents=[global_parent], **_argument_parser_kwargs())
    subparsers["schema"] = schema_p

    init_p = sub.add_parser("init", parents=[global_parent], **_argument_parser_kwargs())
    init_p.add_argument("--file", help="Optional store path; defaults to HEALTHSTREAM_FILE env or .healthstream.json.")
    subparsers["init"] = init_p

    emit_p = sub.add_parser("emit", parents=[global_parent], **_argument_parser_kwargs())
    emit_p.add_argument(
        "--kind",
        required=True,
        choices=["health", "dq", "metric", "alert", "ops"],
        help="Event kind: health|dq|metric|alert|ops.",
    )
    emit_p.add_argument(
        "--status",
        required=True,
        choices=["ok", "degraded", "error", "unknown", "skipped"],
        help="Event status: ok|degraded|error|unknown|skipped.",
    )
    emit_p.add_argument("--source", required=True, help="Event source (script/service/unit name).")
    emit_p.add_argument(
        "--severity",
        choices=["info", "warning", "critical"],
        help="Optional severity: info|warning|critical (default inferred from status).",
    )
    emit_p.add_argument("--channel", help="Optional logical channel (e.g. ops).")
    emit_p.add_argument("--actor", help="Optional actor identifier (agent/service).")
    emit_p.add_argument(
        "--origin",
        choices=["systemd", "cron", "manual", "agent"],
        help="Optional origin (systemd|cron|manual|agent).",
    )
    emit_p.add_argument("--text", "--description", dest="text", help="Optional human-readable message.")
    emit_p.add_argument("--data-json", help="Optional JSON object payload for structured fields.")
    emit_p.add_argument("--artifact", action="append", dest="artifacts", help="Repeatable: artifact path(s).")
    emit_p.add_argument("--tag", action="append", dest="tags", help="Repeatable: free-form tag(s).")
    emit_p.add_argument("--link", action="append", dest="links", help="Repeatable: explicit link token kind:id.")
    emit_p.add_argument("--script", action="append", dest="scripts", help="Repeatable: add link script:<path>.")
    emit_p.add_argument("--topic", action="append", dest="topics", help="Repeatable: add link topic:<name>.")
    emit_p.add_argument("--table", action="append", dest="tables", help="Repeatable: add link table:<db.table>.")
    emit_p.add_argument(
        "--service-unit",
        action="append",
        dest="service_units",
        help="Repeatable: add link service_unit:<unit>.",
    )
    emit_p.add_argument("--dedupe-key", help="Optional dedupe key; if used with --dedupe-window-sec may skip emits.")
    emit_p.add_argument(
        "--dedupe-window-sec",
        type=_nonneg_int,
        default=0,
        help="Dedupe window in seconds (default: 0).",
    )
    emit_p.add_argument("--file", help="Optional store path; defaults to HEALTHSTREAM_FILE env or .healthstream.json.")
    emit_p.add_argument("--lock-timeout-sec", type=_nonneg_float, help="Override lock timeout seconds.")
    subparsers["emit"] = emit_p

    list_p = sub.add_parser("list", parents=[global_parent], **_argument_parser_kwargs())
    list_p.add_argument("--kind", choices=["health", "dq", "metric", "alert", "ops"], help="Filter by kind.")
    list_p.add_argument(
        "--status",
        choices=["ok", "degraded", "error", "unknown", "skipped"],
        help="Filter by status.",
    )
    list_p.add_argument("--source", help="Filter by substring match on source.")
    list_p.add_argument("--since", help="Lower bound created_at (ISO8601, inclusive).")
    list_p.add_argument("--contains", help="Substring match across text/source/kind/status.")
    list_p.add_argument("--limit", type=_nonneg_int, default=50, help="Max events returned.")
    list_p.add_argument("--file", help="Optional store path; defaults to HEALTHSTREAM_FILE env or .healthstream.json.")
    subparsers["list"] = list_p

    prune_p = sub.add_parser("prune", parents=[global_parent], **_argument_parser_kwargs())
    prune_p.add_argument("--file", help="Optional store path; defaults to HEALTHSTREAM_FILE env or .healthstream.json.")
    prune_p.add_argument("--keep-days", type=_nonneg_int, help="Drop events older than N days (based on created_at).")
    prune_p.add_argument("--max-events", type=_nonneg_int, help="Keep only the most recent N events.")
    prune_p.add_argument("--lock-timeout-sec", type=_nonneg_float, help="Override lock timeout seconds.")
    subparsers["prune"] = prune_p

    return p, subparsers


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p, _ = _build_parser()
    return p.parse_args(argv)


def _normalize_global_flags(argv: List[str]) -> List[str]:
    """
    Allow global flags to appear after the subcommand.

    Example:
      healthstream list --kind dq --json-pretty

    NOTE: This is retained for backward-compatibility and for callers that relied on the old
    behavior. The parser now supports global flags after subcommands natively via `parents=`.
    """
    out: List[str] = []
    rest: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--json-pretty":
            out.append(tok)
            i += 1
            continue
        if tok == "--log-level":
            if i + 1 >= len(argv):
                # Let argparse surface the correct error.
                rest.append(tok)
                i += 1
                continue
            out.extend([tok, argv[i + 1]])
            i += 2
            continue
        if tok.startswith("--log-level="):
            out.append(tok)
            i += 1
            continue
        rest.append(tok)
        i += 1
    return out + rest


def _resolve_store_path(file_arg: Optional[str]) -> Path:
    if file_arg:
        return resolve_repo_path(file_arg, repo_root=REPO_ROOT)
    return get_store_path()


def _atomic_init_store(path: Path) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Atomically initialize the store file if it does not exist.

    Returns:
      (created, store_if_existing)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return False, load_store(path)

    store = default_store()

    # Write store to a temp file in the same directory (same filesystem).
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.init.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        save_store(tmp_path, store)
        try:
            os.link(tmp_path, path)  # Atomic, fails if destination exists.
            return True, None
        except FileExistsError:
            # Someone else created it between our exists() and link().
            return False, load_store(path)
        except OSError as exc:
            # Fallback when hardlinks are unsupported/restricted.
            if exc.errno in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP}:
                created = _exclusive_create_and_write(path, store)
                if created:
                    return True, None
                return False, load_store(path)
            raise
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            # Best-effort cleanup only.
            pass


def _exclusive_create_and_write(path: Path, store: Dict[str, Any]) -> bool:
    """
    Fallback init: exclusive-create final path and write JSON directly.

    This is less ideal than link-based publication because the file becomes visible while being
    written, but it avoids overwriting an existing store and is used only when hardlinks fail.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(str(path), flags, 0o644)
    except FileExistsError:
        return False

    # Best-effort advisory locking while writing, to cooperate with store readers/writers.
    try:
        _lock_fd_exclusive(fd)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(store, sort_keys=True))
            f.write("\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
    finally:
        try:
            _unlock_fd(fd)
        except Exception:
            pass
    return True


def _lock_fd_exclusive(fd: int) -> None:
    if os.name == "nt":
        # Best-effort on Windows; keep minimal.
        try:
            import msvcrt  # type: ignore

            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        except Exception:
            return
    else:
        try:
            import fcntl  # type: ignore

            fcntl.flock(fd, fcntl.LOCK_EX)
        except Exception:
            return


def _unlock_fd(fd: int) -> None:
    if os.name == "nt":
        try:
            import msvcrt  # type: ignore

            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except Exception:
            return
    else:
        try:
            import fcntl  # type: ignore

            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            return


def _help_payload(argv: List[str]) -> Dict[str, Any]:
    parser, subparsers = _build_parser()
    cmd = _detect_cmd(argv)
    help_text = subparsers[cmd].format_help() if cmd and cmd in subparsers else parser.format_help()
    return {"ok": True, "action": "help", "cmd": cmd, "help": help_text}


def _classify_healthstream_exception(exc: BaseException) -> Tuple[str, Optional[bool]]:
    if isinstance(exc, _CLIError):
        return ("CLI_ERROR", False)
    if isinstance(exc, _HealthstreamFileLockTimeout):
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


def _healthstream_cli_retry_budget_sec() -> float:
    return _resolve_governance_retry_budget(
        retry_timeout_env="CAIA_HEALTHSTREAM_CLI_RETRY_TIMEOUT_SEC",
        lock_timeout_envs=("HEALTHSTREAM_LOCK_TIMEOUT_SEC",),
        default_lock_timeout=_DEFAULT_LOCK_TIMEOUT_SEC,
    )


def _healthstream_cli_retry_poll_sec() -> float:
    return _resolve_governance_retry_poll(
        retry_poll_env="CAIA_HEALTHSTREAM_CLI_RETRY_POLL_SEC",
        default_poll_sec=0.25,
    )


def _emit_healthstream_retry(action: str, error_code: str, attempt: int, sleep_s: float, exc: BaseException) -> None:
    _LOG.warning(
        "Retrying healthstream mutation after %s action=%s attempt=%s sleep_s=%.3f error=%s",
        error_code,
        action,
        attempt,
        sleep_s,
        exc,
    )


def _run_healthstream_mutation_with_retry(action: str, op: Any) -> Any:
    result, _retry = _run_governance_mutation_with_retry(
        action=action,
        op=op,
        classify_exception=_classify_healthstream_exception,
        retry_budget_sec=_healthstream_cli_retry_budget_sec(),
        retry_poll_sec=_healthstream_cli_retry_poll_sec(),
        emit_retry=lambda error_code, attempt, sleep_s, exc: _emit_healthstream_retry(
            action,
            error_code,
            attempt,
            sleep_s,
            exc,
        ),
    )
    return result


def main(argv: Optional[List[str]] = None) -> int:
    raw_argv = list(sys.argv[1:]) if argv is None else list(argv)

    # NOTE: The parser supports global flags after subcommands natively (via `parents=`).
    # `_normalize_global_flags` is retained for backward-compatibility for any external callers,
    # but is no longer required for correct parsing.
    argv_to_parse = raw_argv

    pretty = _argv_wants_pretty(argv_to_parse)
    detected_cmd = _detect_cmd(argv_to_parse)

    if _argv_requests_help(argv_to_parse):
        try:
            _print(_help_payload(argv_to_parse), pretty=pretty)
            return 0
        except _StdoutBrokenPipe:
            return 141

    args: Optional[argparse.Namespace] = None
    try:
        args = parse_args(argv_to_parse)

        # Enforce "command required" ourselves to allow `--help` without subcommand.
        if not getattr(args, "cmd", None):
            raise _CLIError("missing command (expected one of: schema|init|emit|list|prune)", exit_code=2)

        setup_logging(args.log_level)
        pretty = bool(getattr(args, "json_pretty", False))

        if args.cmd == "schema":
            _print(_schema(), pretty=pretty)
            return 0

        if args.cmd == "init":
            path = _resolve_store_path(getattr(args, "file", None))
            if path.exists():
                store = load_store(path)
                _print(
                    {"ok": True, "action": "init", "created": False, "store_path": str(path), "revision": store.get("revision")},
                    pretty=pretty,
                )
                return 0
            created, existing_store = _atomic_init_store(path)
            if not created:
                # Another process created it concurrently; mirror the "exists" behavior.
                store = existing_store or load_store(path)
                _print(
                    {"ok": True, "action": "init", "created": False, "store_path": str(path), "revision": store.get("revision")},
                    pretty=pretty,
                )
                return 0
            _print({"ok": True, "action": "init", "created": True, "store_path": str(path)}, pretty=pretty)
            return 0

        if args.cmd == "emit":
            links: List[str] = []
            for raw in (args.links or []):
                links.append(str(raw))
            for sp in (args.scripts or []):
                links.append(f"script:{sp}")
            for tp in (args.topics or []):
                links.append(f"topic:{tp}")
            for tb in (args.tables or []):
                links.append(f"table:{tb}")
            for un in (args.service_units or []):
                links.append(f"service_unit:{un}")
            links = append_codex_links(links)

            data = None
            if args.data_json:
                raw_bytes = args.data_json.encode("utf-8", errors="replace")
                max_bytes = _env_bytes_limit("HEALTHSTREAM_DATA_JSON_MAX_BYTES", default=0)
                if max_bytes and len(raw_bytes) > max_bytes:
                    raise ValueError(f"--data-json too large ({len(raw_bytes)} bytes > {max_bytes} bytes)")
                if len(raw_bytes) > _HARD_MAX_DATA_JSON_BYTES:
                    raise ValueError(f"--data-json too large ({len(raw_bytes)} bytes > {_HARD_MAX_DATA_JSON_BYTES} bytes)")
                if not max_bytes and len(raw_bytes) > _WARN_DATA_JSON_BYTES:
                    _LOG.warning(
                        "large --data-json payload (%d bytes); consider limiting via HEALTHSTREAM_DATA_JSON_MAX_BYTES",
                        len(raw_bytes),
                    )
                try:
                    data = json.loads(args.data_json)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid --data-json: {exc}") from exc
                if data is not None and not isinstance(data, dict):
                    raise ValueError("--data-json must be a JSON object")

            store_path = _resolve_store_path(getattr(args, "file", None))
            lock_timeout_sec = args.lock_timeout_sec if args.lock_timeout_sec is not None else _default_lock_timeout_sec()

            payload = _run_healthstream_mutation_with_retry(
                "emit",
                lambda: emit_event(
                    kind=args.kind,
                    status=args.status,
                    source=args.source,
                    text=args.text,
                    severity=args.severity,
                    channel=args.channel,
                    actor=args.actor,
                    origin=args.origin,
                    dedupe_key=args.dedupe_key,
                    dedupe_window_sec=int(args.dedupe_window_sec or 0),
                    artifacts=args.artifacts,
                    tags=args.tags,
                    links=links,
                    data=data,
                    store_path=store_path,
                    lock_timeout_sec=lock_timeout_sec,
                ),
            )
            payload["action"] = "emit"
            _print(payload, pretty=pretty)
            return 0 if payload.get("ok") else 1

        if args.cmd == "list":
            store_path = _resolve_store_path(getattr(args, "file", None))
            limit = int(args.limit or 0)

            max_limit = _env_int("HEALTHSTREAM_LIST_LIMIT_MAX", default=0)
            if max_limit and limit > max_limit:
                raise ValueError(f"--limit too large ({limit} > {max_limit}); adjust HEALTHSTREAM_LIST_LIMIT_MAX if needed")
            if limit > _HARD_MAX_LIST_LIMIT:
                raise ValueError(f"--limit too large ({limit} > {_HARD_MAX_LIST_LIMIT})")
            if not max_limit and limit > _WARN_LIST_LIMIT:
                _LOG.warning("large --limit=%d; consider limiting via HEALTHSTREAM_LIST_LIMIT_MAX", limit)

            payload = list_events(
                kind=args.kind,
                status=args.status,
                source=args.source,
                since=args.since,
                contains=args.contains,
                limit=limit,
                store_path=store_path,
            )
            payload["action"] = "list"
            _print(payload, pretty=pretty)
            return 0 if payload.get("ok") else 1

        if args.cmd == "prune":
            store_path = _resolve_store_path(getattr(args, "file", None))
            lock_timeout_sec = args.lock_timeout_sec if args.lock_timeout_sec is not None else _default_lock_timeout_sec()
            payload = _run_healthstream_mutation_with_retry(
                "prune",
                lambda: prune_events(
                    keep_days=args.keep_days,
                    max_events=args.max_events,
                    store_path=store_path,
                    lock_timeout_sec=lock_timeout_sec,
                ),
            )
            payload["action"] = "prune"
            _print(payload, pretty=pretty)
            return 0 if payload.get("ok") else 1

        raise ValueError(f"unknown command: {args.cmd}")
    except _StdoutBrokenPipe:
        return 141
    except _CLIError as exc:
        try:
            _print({"ok": False, "error": "exception", "detail": str(exc), "action": getattr(args, "cmd", detected_cmd)}, pretty=pretty)
        except _StdoutBrokenPipe:
            return 141
        return int(exc.exit_code)
    except _HealthstreamFileLockTimeout as exc:
        try:
            _print(
                {
                    "ok": False,
                    "error": "lock_timeout",
                    "error_code": "LOCK_TIMEOUT",
                    "detail": str(exc),
                    "retryable": True,
                    "action": getattr(args, "cmd", detected_cmd),
                },
                pretty=pretty,
            )
        except _StdoutBrokenPipe:
            return 141
        return 1
    except Exception as exc:
        try:
            _print({"ok": False, "error": "exception", "detail": str(exc), "action": getattr(args, "cmd", detected_cmd)}, pretty=pretty)
        except _StdoutBrokenPipe:
            return 141
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
