"""
Contract
- Purpose:
  - File-backed append-only store for healthstream events with safe locking.
- Inputs (types, units):
  - Store path on disk; lock timeout in seconds.
- Outputs (types, units):
  - Atomic persisted JSON store containing events indexed by id.
- Invariants:
  - Atomic writes via tmp+rename.
  - Cross-process lock via lockfile create/unlink.
  - UTC timestamps in ISO8601 with 'Z'.
- Error semantics:
  - Raises `RuntimeError` on I/O problems and `ValueError` on schema violations.
- External boundaries:
  - Filesystem only.
- Telemetry:
  - Uses stdlib logging (caller configures).
"""

##REFACTOR: 2026-01-16##

import calendar
import errno
import json
import logging
import os
import random
import socket
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from agentic_tools.healthstream.config import lock_timeout_sec_from_env, store_path_from_env
from agentic_tools.healthstream.types import HealthStreamEvent, JsonObject
from agentic_tools.healthstream.validation import normalize_kind, normalize_links, normalize_severity, normalize_status


LOG = logging.getLogger("healthstream")

_HOSTNAME = socket.gethostname()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_iso(ts: str) -> float:
    try:
        return calendar.timegm(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid ISO8601 timestamp: {ts!r}") from exc


class FileLockTimeout(RuntimeError):
    pass


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if not hasattr(os, "kill"):
        # Conservative: if we can't check, assume it's alive.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        # Conservative: unknown error, assume alive to avoid false stale cleanup.
        return True
    return True


@dataclass
class FileLock:
    path: Path
    timeout: float
    poll_interval: float = 0.05

    def _ensure_parent_dir(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"I/O error creating lock directory: {self.path.parent} ({exc})") from exc

    def _write_lock_metadata(self, file_obj) -> None:
        # Best-effort permissions hardening: allow both `root` and
        # the repo group user (typically `thh`) to clean up a stale
        # lock file without losing write access to the directory.
        try:
            os.fchmod(file_obj.fileno(), 0o664)
        except Exception:
            pass
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            try:
                gid = int(self.path.parent.stat().st_gid)
                os.fchown(file_obj.fileno(), -1, gid)
            except Exception:
                pass
        meta = {
            "created_at": now_iso(),
            "pid": os.getpid(),
            "rand": random.random(),
            "hostname": _HOSTNAME,
        }
        file_obj.write(json.dumps(meta, sort_keys=True))
        # For locks we treat fsync as best-effort; the lock's existence is the mutex.
        try:
            file_obj.flush()
        except Exception:
            pass
        try:
            os.fsync(file_obj.fileno())
        except Exception:
            pass

    def _try_acquire_via_link(self) -> Optional[bool]:
        """
        Attempt a NFS-safer lock acquisition using the unique-file + link(2) pattern.

        Returns:
          - True: acquired
          - False: lock already exists
          - None: linking not supported/allowed; caller should fall back.
        """
        self._ensure_parent_dir()
        # Create the candidate in the same directory so link() is atomic.
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=str(self.path.parent),
                text=True,
            )
        except OSError as exc:
            raise RuntimeError(f"I/O error creating lock temp file in {self.path.parent} ({exc})") from exc
        tmp_path = Path(tmp_name)
        try:
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    self._write_lock_metadata(handle)
            except OSError as exc:
                raise RuntimeError(f"I/O error writing lock temp file {tmp_path} ({exc})") from exc

            try:
                os.link(str(tmp_path), str(self.path))
                return True
            except FileExistsError:
                return False
            except OSError as exc:
                # Hard links may be unsupported (or restricted). Allow a safe fallback.
                if exc.errno in {
                    errno.EPERM,
                    errno.EOPNOTSUPP,
                    getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
                    errno.EACCES,
                }:
                    return None
                raise RuntimeError(f"I/O error acquiring lock {self.path} via link ({exc})") from exc
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                # Best-effort cleanup
                pass

    def _try_acquire_via_excl(self) -> bool:
        """
        Attempt a simple lock acquisition using O_CREAT|O_EXCL.

        Returns:
          - True: acquired
          - False: lock already exists
        """
        self._ensure_parent_dir()
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        except OSError as exc:
            raise RuntimeError(f"I/O error acquiring lock {self.path} ({exc})") from exc
        try:
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    self._write_lock_metadata(handle)
            except OSError as exc:
                raise RuntimeError(f"I/O error writing lock file {self.path} ({exc})") from exc
        except Exception:
            # Avoid wedging on partially-created lock files.
            try:
                self.path.unlink()
            except Exception:
                pass
            raise
        return True

    def _read_lock_metadata(self) -> Optional[Dict[str, Any]]:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = handle.read()
        except FileNotFoundError:
            return None
        except OSError:
            return None
        try:
            obj = json.loads(data)
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    def _maybe_cleanup_stale_lock(self) -> bool:
        """
        Best-effort stale lock cleanup.

        Safety properties:
          - Only attempts PID-liveness-based cleanup when hostname matches.
          - Never removes locks it cannot confidently attribute to this host.
        """
        meta = self._read_lock_metadata()
        if not meta:
            return False
        host = meta.get("hostname")
        if not isinstance(host, str) or host != _HOSTNAME:
            return False
        pid_raw = meta.get("pid")
        try:
            pid = int(pid_raw)
        except Exception:
            return False
        # If PID is alive, do not touch.
        if _is_pid_alive(pid):
            return False
        # PID not alive: stale.
        try:
            self.path.unlink()
            LOG.warning("removed stale lock file %s (pid=%s host=%s)", self.path, pid, host)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False

    def acquire(self) -> None:
        deadline = time.monotonic() + float(self.timeout)
        use_link = True
        while True:
            try:
                acquired: Optional[bool]
                if use_link:
                    acquired = self._try_acquire_via_link()
                    if acquired is None:
                        use_link = False
                        continue
                else:
                    acquired = self._try_acquire_via_excl()

                if acquired:
                    return
                # Not acquired: lock exists.
            except FileLockTimeout:
                raise
            except RuntimeError:
                # Re-raise with context already attached.
                raise

            # Lock exists: attempt stale cleanup (best-effort).
            if self._maybe_cleanup_stale_lock():
                continue

            if time.monotonic() > deadline:
                raise FileLockTimeout(f"timed out acquiring lock {self.path}")

            time.sleep(self.poll_interval + random.random() * self.poll_interval)

    def release(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise RuntimeError(f"I/O error releasing lock {self.path} ({exc})") from exc

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self.release()
        except Exception:
            # Never mask an exception from the with-body.
            if exc_type is not None:
                LOG.error("error releasing lock %s", self.path, exc_info=True)
                return
            raise


def default_store() -> JsonObject:
    now = now_iso()
    return {
        "version": 1,
        "created_at": now,
        "updated_at": now,
        "revision": 0,
        "next_event_id": 1,
        "events": {},  # id -> event dict
    }


def _max_event_seq(events: Mapping[Any, Any]) -> int:
    max_seq = 0
    for eid, ev in events.items():
        # Prefer dict key, but also consider embedded id in event dict.
        candidates: List[Any] = [eid]
        if isinstance(ev, dict):
            candidates.append(ev.get("id"))
        for cand in candidates:
            try:
                seq = _event_seq(str(cand))
            except Exception:
                continue
            if seq > max_seq:
                max_seq = seq
    return max_seq


def ensure_store_shape(store: JsonObject) -> JsonObject:
    if "version" not in store:
        store["version"] = 1
    if "created_at" not in store:
        store["created_at"] = now_iso()
    if "updated_at" not in store:
        store["updated_at"] = now_iso()
    if "revision" not in store:
        store["revision"] = 0

    events = store.get("events")
    if not isinstance(events, dict):
        events = {}
        store["events"] = events

    # next_event_id: derive from existing events when missing/invalid/too small to avoid collisions.
    raw_next = store.get("next_event_id")
    next_id: Optional[int]
    try:
        next_id = int(raw_next)
    except Exception:
        next_id = None
    computed_next = _max_event_seq(events) + 1 if events else 1
    if next_id is None or next_id < 1:
        store["next_event_id"] = computed_next
    else:
        if next_id < computed_next:
            store["next_event_id"] = computed_next

    return store


def load_store(path: Path) -> JsonObject:
    try:
        exists = path.exists()
    except OSError as exc:
        raise RuntimeError(f"I/O error checking store: {path} ({exc})") from exc
    if not exists:
        return default_store()
    try:
        with path.open("r", encoding="utf-8") as handle:
            obj = json.load(handle)
    except FileNotFoundError:
        # Race: the file may have been replaced/removed after exists() check.
        return default_store()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON store: {path} ({exc})") from exc
    except OSError as exc:
        raise RuntimeError(f"I/O error reading store: {path} ({exc})") from exc
    if not isinstance(obj, dict):
        raise RuntimeError(f"invalid store root (expected object): {path}")
    return ensure_store_shape(obj)


def _fsync_dir_best_effort(dir_path: Path) -> None:
    # Best-effort: do not raise after a successful commit (os.replace).
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        dir_fd = os.open(str(dir_path), os.O_RDONLY | os.O_DIRECTORY)
    except Exception:
        return
    try:
        os.fsync(dir_fd)
    except Exception:
        return
    finally:
        try:
            os.close(dir_fd)
        except Exception:
            pass


def save_store(path: Path, store: JsonObject) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"I/O error creating store directory: {path.parent} ({exc})") from exc

    store["updated_at"] = now_iso()
    store["revision"] = int(store.get("revision", 0)) + 1

    tmp_fd: Optional[int] = None
    tmp_path: Optional[Path] = None
    try:
        try:
            tmp_fd, tmp_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=f"{path.suffix}.tmp",
                dir=str(path.parent),
                text=True,
            )
            tmp_path = Path(tmp_name)
        except OSError as exc:
            raise RuntimeError(f"I/O error creating temp store file in {path.parent} ({exc})") from exc

        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                tmp_fd = None  # ownership transferred to handle
                json.dump(store, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise RuntimeError(f"I/O error writing temp store file {tmp_path} ({exc})") from exc

        try:
            os.replace(str(tmp_path), str(path))
        except OSError as exc:
            raise RuntimeError(f"I/O error replacing store file: {path} ({exc})") from exc

        _fsync_dir_best_effort(path.parent)

    finally:
        # Ensure temp file is cleaned up on failure paths.
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except Exception:
                pass
        if tmp_path is not None:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    # Best-effort: make the store group-writable and aligned with the parent
    # directory's group. This prevents a root-run unit from creating a
    # root-owned file that blocks thh-run units/agents.
    try:
        os.chmod(str(path), 0o664)
    except Exception:
        pass
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        try:
            gid = int(path.parent.stat().st_gid)
            os.chown(str(path), -1, gid)
        except Exception:
            pass


def get_store_path() -> Path:
    return store_path_from_env()


def _format_event_id(n: int) -> str:
    return f"HS{int(n):06d}"


def _event_sort_key(event: Mapping[str, Any]) -> Tuple[float, int]:
    created_at = str(event.get("created_at") or "")
    ts = 0.0
    if created_at:
        try:
            ts = parse_iso(created_at)
        except Exception:
            ts = 0.0
    idv = str(event.get("id") or "")
    digits = "".join(ch for ch in idv if ch.isdigit())
    seq = int(digits) if digits else 0
    return (ts, seq)


def _event_seq(event_id: str) -> int:
    """
    Parse a stable monotonic sequence number from a HealthStream event id.

    Expected shape is "HS000123" but we accept any id with digits.
    """
    s = str(event_id or "").strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        raise ValueError(f"invalid event_id (no digits): {event_id!r}")
    return int(digits)


def get_event(*, event_id: str, store_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Fetch a single event by id.

    Returns a JSON-friendly dict with {ok, found, event?, store_path, meta}.
    """
    p = store_path or get_store_path()
    store = load_store(p)
    events = store.get("events")
    if not isinstance(events, dict):
        raise RuntimeError("invalid store: events must be a dict")
    eid = str(event_id or "").strip()
    if not eid:
        raise ValueError("event_id is required")
    ev = events.get(eid)
    found = isinstance(ev, dict)
    return {
        "ok": True,
        "store_path": str(p),
        "found": found,
        "event": dict(ev) if found else None,
        "meta": {
            "store_version": store.get("version"),
            "store_revision": store.get("revision"),
            "updated_at": store.get("updated_at"),
        },
    }


def _prune_events_inplace(
    events: Dict[str, Any],
    *,
    keep_days: Optional[int],
    max_events: Optional[int],
) -> Set[str]:
    removed: Set[str] = set()
    now_ts = time.time()

    if keep_days is not None:
        kd = int(keep_days)
        if kd < 0:
            raise ValueError("keep_days must be >= 0")
        cutoff = now_ts - float(kd) * 86400.0
        for eid, ev in list(events.items()):
            if not isinstance(ev, dict):
                continue
            created_at = str(ev.get("created_at") or "").strip()
            if not created_at:
                continue
            try:
                ev_ts = parse_iso(created_at)
            except Exception:
                # Conservative: keep events with invalid timestamps.
                continue
            if ev_ts < cutoff:
                removed.add(str(eid))

    if max_events is not None:
        me = int(max_events)
        if me < 0:
            raise ValueError("max_events must be >= 0")
        rows: List[Tuple[Tuple[float, int], str]] = []
        for eid, ev in events.items():
            if not isinstance(ev, dict):
                continue
            try:
                key = _event_sort_key(ev)
            except Exception:
                key = (0.0, 0)
            rows.append((key, str(eid)))
        rows.sort(key=lambda t: t[0], reverse=True)
        keep_ids = {eid for _key, eid in rows[:me]} if me > 0 else set()
        for _key, eid in rows[me:]:
            removed.add(eid)
        removed = {eid for eid in removed if eid not in keep_ids}

    return removed


def prune_events(
    *,
    keep_days: Optional[int] = None,
    max_events: Optional[int] = None,
    store_path: Optional[Path] = None,
    lock_timeout_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Prune old events from the store.

    - keep_days: drop events older than N days (based on created_at).
    - max_events: keep only the most recent N events.
    """
    p = store_path or get_store_path()
    timeout = float(lock_timeout_sec) if lock_timeout_sec is not None else lock_timeout_sec_from_env()
    lock_path = p.with_suffix(p.suffix + ".lock")
    with FileLock(lock_path, timeout=float(timeout)):
        store = load_store(p)
        events = store.get("events")
        if not isinstance(events, dict):
            raise RuntimeError("invalid store: events must be a dict")

        removed = _prune_events_inplace(events, keep_days=keep_days, max_events=max_events)

        if not removed:
            return {
                "ok": True,
                "pruned": False,
                "store_path": str(p),
                "removed": 0,
                "meta": {
                    "store_version": store.get("version"),
                    "store_revision": store.get("revision"),
                    "updated_at": store.get("updated_at"),
                },
            }

        for eid in removed:
            events.pop(eid, None)
        save_store(p, store)
        return {
            "ok": True,
            "pruned": True,
            "store_path": str(p),
            "removed": len(removed),
            "meta": {
                "store_version": store.get("version"),
                "store_revision": store.get("revision"),
                "updated_at": store.get("updated_at"),
            },
        }


def _int_from_env(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    return int(s)


def _auto_prune_settings_from_env() -> Tuple[Optional[int], Optional[int]]:
    """
    Optional, backward-compatible safety valve for unbounded growth.
    Defaults to disabled (returns (None, None)).
    """
    keep_days = _int_from_env("HEALTHSTREAM_AUTO_PRUNE_KEEP_DAYS")
    max_events = _int_from_env("HEALTHSTREAM_AUTO_PRUNE_MAX_EVENTS")
    return keep_days, max_events


def emit_event(
    *,
    kind: str,
    status: str,
    source: str,
    text: Optional[str] = None,
    severity: Optional[str] = None,
    channel: Optional[str] = None,
    actor: Optional[str] = None,
    origin: Optional[str] = None,
    dedupe_key: Optional[str] = None,
    dedupe_window_sec: int = 0,
    artifacts: Optional[Iterable[str]] = None,
    tags: Optional[Iterable[str]] = None,
    links: Optional[Iterable[str]] = None,
    data: Optional[Mapping[str, Any]] = None,
    store_path: Optional[Path] = None,
    lock_timeout_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Append an event to the healthstream store.

    Returns a JSON-friendly dict with {ok, emitted, event_id, store_path, ...}.
    """
    norm_kind = normalize_kind(kind)
    norm_status = normalize_status(status)
    norm_sev = normalize_severity(severity, status=norm_status)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source is required")
    norm_source = source.strip()
    norm_links = normalize_links(links)

    p = store_path or get_store_path()
    timeout = float(lock_timeout_sec) if lock_timeout_sec is not None else lock_timeout_sec_from_env()
    lock_path = p.with_suffix(p.suffix + ".lock")
    with FileLock(lock_path, timeout=timeout):
        store = load_store(p)
        events = store.get("events")
        if not isinstance(events, dict):
            raise RuntimeError("invalid store: events must be a dict")

        # Dedupe: short-circuit as soon as we find any event within the window.
        if dedupe_key and dedupe_window_sec > 0:
            dk = str(dedupe_key).strip()
            if dk:
                cutoff = time.time() - float(dedupe_window_sec)
                for ev in events.values():
                    if not isinstance(ev, dict):
                        continue
                    if str(ev.get("dedupe_key") or "") != dk:
                        continue
                    try:
                        ts = parse_iso(str(ev.get("created_at") or ""))
                    except Exception:
                        continue
                    if ts >= cutoff:
                        return {
                            "ok": True,
                            "emitted": False,
                            "reason": "deduped",
                            "dedupe_key": dk,
                            "dedupe_window_sec": int(dedupe_window_sec),
                            "store_path": str(p),
                        }

        # Optional safety valve: auto-prune under lock (disabled by default).
        try:
            ap_keep_days, ap_max_events = _auto_prune_settings_from_env()
        except Exception:
            ap_keep_days, ap_max_events = (None, None)
        if ap_keep_days is not None or ap_max_events is not None:
            removed = _prune_events_inplace(
                events,
                keep_days=ap_keep_days,
                max_events=ap_max_events,
            )
            for eid in removed:
                events.pop(eid, None)

        # Robust next id assignment (avoid collisions if store metadata is missing/invalid).
        next_id_raw = store.get("next_event_id") or 1
        try:
            next_id = int(next_id_raw)
        except Exception:
            next_id = 1
        if next_id < 1:
            next_id = 1

        event_id = _format_event_id(next_id)
        if event_id in events:
            # Recover from metadata corruption/manual edits by bumping past max existing seq.
            next_id = _max_event_seq(events) + 1
            event_id = _format_event_id(next_id)

        event = HealthStreamEvent(
            id=event_id,
            created_at=now_iso(),
            kind=norm_kind,
            status=norm_status,
            severity=norm_sev,
            source=norm_source,
            text=text.strip() if isinstance(text, str) and text.strip() else None,
            channel=channel.strip() if isinstance(channel, str) and channel.strip() else None,
            actor=actor.strip() if isinstance(actor, str) and actor.strip() else None,
            origin=origin.strip() if isinstance(origin, str) and origin.strip() else None,
            dedupe_key=dedupe_key.strip() if isinstance(dedupe_key, str) and dedupe_key.strip() else None,
            artifacts=[str(a).strip() for a in artifacts] if artifacts else None,
            tags=[str(t).strip() for t in tags] if tags else None,
            links=norm_links if norm_links else None,
            data=dict(data) if data else None,
        )
        events[event_id] = event.to_dict()
        store["next_event_id"] = int(next_id) + 1
        save_store(p, store)
        LOG.info("emit event_id=%s kind=%s status=%s source=%s", event_id, norm_kind, norm_status, norm_source)
        return {"ok": True, "emitted": True, "event_id": event_id, "store_path": str(p), "event": event.to_dict()}


def list_events(
    *,
    kind: Optional[str] = None,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    source: Optional[str] = None,
    source_exact: Optional[str] = None,
    channel: Optional[str] = None,
    origin: Optional[str] = None,
    actor: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    since_id: Optional[str] = None,
    after_id: Optional[str] = None,
    contains: Optional[str] = None,
    limit: int = 50,
    store_path: Optional[Path] = None,
) -> Dict[str, Any]:
    p = store_path or get_store_path()
    store = load_store(p)
    events = store.get("events")
    if not isinstance(events, dict):
        raise RuntimeError("invalid store: events must be a dict")
    since_ts = None
    if since:
        since_ts = parse_iso(since)
    until_ts = None
    if until:
        until_ts = parse_iso(until)
    out: List[JsonObject] = []
    norm_kind = normalize_kind(kind) if kind else None
    norm_status = normalize_status(status) if status else None
    norm_sev = None
    if severity is not None and str(severity).strip():
        # status is only used for defaulting when raw is empty; it's irrelevant when filtering by explicit severity.
        norm_sev = normalize_severity(severity, status="ok")
    source_q = source.strip() if isinstance(source, str) and source.strip() else None
    source_exact_q = source_exact.strip() if isinstance(source_exact, str) and source_exact.strip() else None
    channel_q = channel.strip() if isinstance(channel, str) and channel.strip() else None
    origin_q = origin.strip() if isinstance(origin, str) and origin.strip() else None
    actor_q = actor.strip() if isinstance(actor, str) and actor.strip() else None
    contains_q = contains.lower().strip() if isinstance(contains, str) and contains.strip() else None
    since_seq = _event_seq(since_id) if since_id else None
    after_seq = _event_seq(after_id) if after_id else None
    for ev in events.values():
        if not isinstance(ev, dict):
            continue
        try:
            ev_id = str(ev.get("id") or "").strip()
        except Exception:
            ev_id = ""
        if since_seq is not None:
            try:
                if _event_seq(ev_id) < since_seq:
                    continue
            except Exception:
                continue
        if after_seq is not None:
            try:
                if _event_seq(ev_id) >= after_seq:
                    continue
            except Exception:
                continue
        if norm_kind and str(ev.get("kind") or "") != norm_kind:
            continue
        if norm_status and str(ev.get("status") or "") != norm_status:
            continue
        if norm_sev and str(ev.get("severity") or "") != norm_sev:
            continue
        if channel_q and str(ev.get("channel") or "") != channel_q:
            continue
        if origin_q and str(ev.get("origin") or "") != origin_q:
            continue
        if actor_q and str(ev.get("actor") or "") != actor_q:
            continue
        if source_exact_q and str(ev.get("source") or "") != source_exact_q:
            continue
        if source_q and source_q not in str(ev.get("source") or ""):
            continue
        if since_ts is not None:
            try:
                ev_ts = parse_iso(str(ev.get("created_at") or ""))
            except Exception:
                continue
            if ev_ts < since_ts:
                continue
        if until_ts is not None:
            try:
                ev_ts2 = parse_iso(str(ev.get("created_at") or ""))
            except Exception:
                continue
            if ev_ts2 > until_ts:
                continue
        if contains_q:
            blob = " ".join(
                [
                    str(ev.get("text") or ""),
                    str(ev.get("source") or ""),
                    str(ev.get("kind") or ""),
                    str(ev.get("status") or ""),
                ]
            ).lower()
            if contains_q not in blob:
                continue
        out.append(dict(ev))
    out.sort(key=_event_sort_key, reverse=True)
    if limit > 0:
        out = out[: int(limit)]
    return {
        "ok": True,
        "store_path": str(p),
        "count": len(out),
        "events": out,
        "meta": {
            "store_version": store.get("version"),
            "store_revision": store.get("revision"),
            "updated_at": store.get("updated_at"),
        },
    }
