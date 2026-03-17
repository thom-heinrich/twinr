"""Run one validated skill handler in a process-isolated sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
import io
import math
import os
import pickle
import signal
import stat
import time
from typing import Any, Final

from twinr.agent.self_coding.sandbox.broker import ParentSkillContextBroker, SkillBrokerPolicy
from twinr.agent.self_coding.sandbox.loader_process import sandbox_loader_child_main


_MAX_IPC_MESSAGE_BYTES: Final[int] = 1 * 1024 * 1024
_COMPLETION_GRACE_SECONDS: Final[float] = 1.0
_TERMINATION_GRACE_SECONDS: Final[float] = 0.5


@dataclass(frozen=True, slots=True)
class SelfCodingSandboxLimits:
    """Bound the wall-clock and resource footprint of one sandboxed handler run."""

    timeout_seconds: float = 180.0
    cpu_seconds: int = 20
    address_space_bytes: int = 512 * 1024 * 1024
    max_open_files: int = 64

    def __post_init__(self) -> None:
        # AUDIT-FIX(#9): Coerce and validate limit values eagerly so invalid config fails fast and safely.
        object.__setattr__(self, "timeout_seconds", _coerce_positive_float(self.timeout_seconds, field_name="timeout_seconds"))
        # AUDIT-FIX(#9): Coerce and validate limit values eagerly so invalid config fails fast and safely.
        object.__setattr__(self, "cpu_seconds", _coerce_positive_int(self.cpu_seconds, field_name="cpu_seconds"))
        # AUDIT-FIX(#9): Coerce and validate limit values eagerly so invalid config fails fast and safely.
        object.__setattr__(
            self,
            "address_space_bytes",
            _coerce_positive_int(self.address_space_bytes, field_name="address_space_bytes"),
        )
        # AUDIT-FIX(#9): Coerce and validate limit values eagerly so invalid config fails fast and safely.
        object.__setattr__(self, "max_open_files", _coerce_positive_int(self.max_open_files, field_name="max_open_files"))


@dataclass(frozen=True, slots=True)
class SelfCodingSandboxResult:
    """Return the observable outcome and hardening report of one sandbox run."""

    spoken_count: int
    hardening: dict[str, Any]
    child_pid: int | None = None


class SelfCodingSandboxExecutionError(RuntimeError):
    """Raised when a sandboxed handler cannot complete successfully."""

    def __init__(self, message: str, *, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata = dict(metadata or {})


class SelfCodingSandboxTimeoutError(SelfCodingSandboxExecutionError):
    """Raised when the sandboxed child exceeds its wall-clock budget."""


class SelfCodingSandboxRunner:
    """Execute one skill handler in a child process and broker ctx calls back to the parent."""

    def __init__(self, *, limits: SelfCodingSandboxLimits | None = None) -> None:
        self.limits = limits if limits is not None else SelfCodingSandboxLimits()

    def run_handler(
        self,
        *,
        owner: Any,
        context: Any,
        materialized_root: Path,
        entry_module: str,
        handler_name: str,
        policy: SkillBrokerPolicy,
        event_name: str | None = None,
    ) -> SelfCodingSandboxResult:
        """Run one handler in a child process and return the resulting spoken-count delta."""

        # AUDIT-FIX(#1): Resolve and read the entry module without allowing absolute paths, traversal, or symlink escapes.
        validated_root = _resolve_materialized_root(materialized_root)
        # AUDIT-FIX(#1): Resolve and read the entry module without allowing absolute paths, traversal, or symlink escapes.
        source_path, source_text = _load_entry_source(validated_root, entry_module)
        # AUDIT-FIX(#4): Capture the baseline so the returned spoken_count is the handler delta, not the absolute counter.
        starting_spoken_count = _read_spoken_count(context)

        spawn_context = get_context("spawn")
        parent_connection = None
        child_connection = None
        process = None
        try:
            parent_connection, child_connection = spawn_context.Pipe(duplex=True)
            process = spawn_context.Process(
                # AUDIT-FIX(#2): Start the child in its own POSIX session so teardown can reap descendant processes.
                target=_sandbox_loader_child_entry,
                kwargs={
                    "connection": child_connection,
                    "source_text": source_text,
                    "entry_module": entry_module,
                    "handler_name": handler_name,
                    "event_name": event_name,
                    "materialized_root": str(validated_root),
                    "limits": self.limits,
                },
                name=f"twinr-self-coding-{handler_name}",
            )
            broker = ParentSkillContextBroker(context=context, policy=policy)
            process.start()
        except Exception as exc:
            # AUDIT-FIX(#5): Wrap startup failures and close partially initialized IPC/process resources.
            _safe_close_connection(child_connection)
            # AUDIT-FIX(#5): Wrap startup failures and close partially initialized IPC/process resources.
            _safe_close_connection(parent_connection)
            # AUDIT-FIX(#5): Wrap startup failures and close partially initialized IPC/process resources.
            _cleanup_process(process)
            raise _sandbox_error(
                "failed to start sandbox child",
                extra_metadata={"entry_module": entry_module, "source_path": str(source_path)},
            ) from exc
        else:
            _safe_close_connection(child_connection)

        deadline = time.monotonic() + self.limits.timeout_seconds
        hardening_payload: dict[str, Any] = {}
        try:
            while True:
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0.0:
                    raise _sandbox_error(
                        "sandbox child timed out before completing the handler",
                        error_type=SelfCodingSandboxTimeoutError,
                        child_pid=process.pid,
                        hardening=hardening_payload,
                        extra_metadata={"timeout_seconds": self.limits.timeout_seconds},
                    )
                try:
                    # AUDIT-FIX(#6): Translate poll/transport faults into sandbox errors instead of leaking raw IPC exceptions.
                    has_message = parent_connection.poll(min(0.1, remaining_seconds))
                except (EOFError, OSError) as exc:
                    raise _sandbox_error(
                        "sandbox IPC poll failed",
                        child_pid=process.pid,
                        hardening=hardening_payload,
                    ) from exc
                if has_message:
                    # AUDIT-FIX(#3): Receive bounded IPC bytes and unpickle them with a restricted decoder.
                    message = _recv_ipc_message(
                        parent_connection,
                        child_pid=process.pid,
                        hardening=hardening_payload,
                    )
                    hardening_payload = _hardening_payload(message, fallback=hardening_payload)
                    kind = str(message.get("kind") or "").strip()
                    if kind == "loader_ready":
                        continue
                    if kind == "rpc_call":
                        self._handle_rpc_call(
                            parent_connection,
                            broker,
                            message,
                            deadline=deadline,
                            child_pid=process.pid,
                            hardening=hardening_payload,
                        )
                        continue
                    if kind == "completed":
                        # AUDIT-FIX(#7): Require the child to exit after reporting completion instead of returning success early.
                        if not _wait_for_process_exit(process, timeout=_COMPLETION_GRACE_SECONDS):
                            raise _sandbox_error(
                                "sandbox child reported completion but did not exit cleanly",
                                child_pid=process.pid,
                                hardening=hardening_payload,
                            )
                        if process.exitcode != 0:
                            raise _sandbox_error(
                                f"sandbox child exited with status {process.exitcode}",
                                child_pid=process.pid,
                                hardening=hardening_payload,
                            )
                        return SelfCodingSandboxResult(
                            spoken_count=max(0, _read_spoken_count(context) - starting_spoken_count),
                            hardening=hardening_payload,
                            child_pid=process.pid,
                        )
                    if kind == "failed":
                        raise _sandbox_error(
                            str(message.get("error") or "sandbox child failed"),
                            child_pid=process.pid,
                            hardening=hardening_payload,
                        )
                    raise _sandbox_error(
                        f"unexpected sandbox child message: {kind or 'unknown'}",
                        child_pid=process.pid,
                        hardening=hardening_payload,
                    )
                if process.exitcode is not None:
                    if process.exitcode == 0:
                        raise _sandbox_error(
                            "sandbox child exited before completing its handler",
                            child_pid=process.pid,
                            hardening=hardening_payload,
                        )
                    raise _sandbox_error(
                        f"sandbox child exited with status {process.exitcode}",
                        child_pid=process.pid,
                        hardening=hardening_payload,
                    )
        finally:
            _safe_close_connection(parent_connection)
            # AUDIT-FIX(#10): Always reap and close the Process object so repeated sandbox runs do not leak OS resources.
            _cleanup_process(process)

    @staticmethod
    def _handle_rpc_call(
        connection: Any,
        broker: ParentSkillContextBroker,
        message: dict[str, Any],
        *,
        deadline: float,
        child_pid: int | None,
        hardening: dict[str, Any],
    ) -> None:
        call_id = str(message.get("call_id") or "").strip()
        if time.monotonic() >= deadline:
            raise _sandbox_error(
                "sandbox child timed out before completing the handler",
                error_type=SelfCodingSandboxTimeoutError,
                child_pid=child_pid,
                hardening=hardening,
            )
        try:
            response = broker.dispatch(message)
        except Exception:
            try:
                # AUDIT-FIX(#8): Avoid leaking parent-side exception details back into the untrusted child process.
                _send_ipc_message(
                    connection,
                    {"kind": "rpc_error", "call_id": call_id, "error": "sandbox broker failure"},
                    child_pid=child_pid,
                    hardening=hardening,
                )
            except SelfCodingSandboxExecutionError:
                raise
            except Exception as send_exc:
                raise _sandbox_error(
                    "failed to send sandbox broker error response",
                    child_pid=child_pid,
                    hardening=hardening,
                    extra_metadata={"call_id": call_id},
                ) from send_exc
            return
        if not isinstance(response, dict):
            raise _sandbox_error(
                "sandbox broker returned an invalid response",
                child_pid=child_pid,
                hardening=hardening,
                extra_metadata={"call_id": call_id},
            )
        # AUDIT-FIX(#6): Serialize broker responses through the same bounded IPC path and wrap transport errors consistently.
        _send_ipc_message(
            connection,
            response,
            child_pid=child_pid,
            hardening=hardening,
            extra_metadata={"call_id": call_id},
        )


def _hardening_payload(message: dict[str, Any], *, fallback: dict[str, Any]) -> dict[str, Any]:
    payload = message.get("hardening")
    if not isinstance(payload, dict):
        return dict(fallback)
    return dict(payload)


# AUDIT-FIX(#2): Bootstrap the loader inside a dedicated POSIX session so process-group teardown can target the sandbox only.
def _sandbox_loader_child_entry(
    *,
    connection: Any,
    source_text: str,
    entry_module: str,
    handler_name: str,
    event_name: str | None,
    materialized_root: str,
    limits: SelfCodingSandboxLimits,
) -> None:
    if os.name == "posix":
        try:
            os.setsid()
        except OSError:
            pass
    sandbox_loader_child_main(
        connection=connection,
        source_text=source_text,
        entry_module=entry_module,
        handler_name=handler_name,
        event_name=event_name,
        materialized_root=materialized_root,
        limits=limits,
    )


# AUDIT-FIX(#1): Validate the sandbox root eagerly so all file reads are anchored to a real directory.
def _resolve_materialized_root(materialized_root: Path) -> Path:
    try:
        root = Path(materialized_root).resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise _sandbox_error(
            "sandbox materialized_root does not exist or is not accessible",
            extra_metadata={"materialized_root": str(materialized_root)},
        ) from exc
    if not root.is_dir():
        raise _sandbox_error(
            "sandbox materialized_root is not a directory",
            extra_metadata={"materialized_root": str(root)},
        )
    return root


# AUDIT-FIX(#1): Read the entry module via safe relative path traversal that rejects symlinks and escapes.
def _load_entry_source(materialized_root: Path, entry_module: str) -> tuple[Path, str]:
    parts = _validated_entry_module_parts(entry_module)
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        return _load_entry_source_fallback(materialized_root, parts)
    root_fd: int | None = None
    opened_fds: list[int] = []
    try:
        root_fd = os.open(materialized_root, os.O_RDONLY | os.O_DIRECTORY)
        current_fd = root_fd
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current_fd)
            opened_fds.append(next_fd)
            current_fd = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current_fd)
        opened_fds.append(file_fd)
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise _sandbox_error(
                "sandbox entry module is not a regular file",
                extra_metadata={"entry_module": entry_module},
            )
        with os.fdopen(os.dup(file_fd), "r", encoding="utf-8", closefd=True) as source_handle:
            source_text = source_handle.read()
        return materialized_root.joinpath(*parts), source_text
    except SelfCodingSandboxExecutionError:
        raise
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise _sandbox_error(
            "sandbox entry module was not found inside materialized_root",
            extra_metadata={"entry_module": entry_module, "materialized_root": str(materialized_root)},
        ) from exc
    except IsADirectoryError as exc:
        raise _sandbox_error(
            "sandbox entry module points to a directory, not a file",
            extra_metadata={"entry_module": entry_module},
        ) from exc
    except PermissionError as exc:
        raise _sandbox_error(
            "sandbox entry module is not readable",
            extra_metadata={"entry_module": entry_module},
        ) from exc
    except UnicodeDecodeError as exc:
        raise _sandbox_error(
            "sandbox entry module is not valid UTF-8 text",
            extra_metadata={"entry_module": entry_module},
        ) from exc
    except OSError as exc:
        raise _sandbox_error(
            "sandbox entry module could not be read safely",
            extra_metadata={"entry_module": entry_module},
        ) from exc
    finally:
        for fd in reversed(opened_fds):
            try:
                os.close(fd)
            except OSError:
                pass
        if root_fd is not None:
            try:
                os.close(root_fd)
            except OSError:
                pass


# AUDIT-FIX(#1): Keep a strict resolved-path containment check on non-POSIX platforms where openat-style hardening is unavailable.
def _load_entry_source_fallback(materialized_root: Path, parts: tuple[str, ...]) -> tuple[Path, str]:
    entry_module = str(Path(*parts))
    source_path = materialized_root.joinpath(*parts)
    try:
        resolved_source = source_path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise _sandbox_error(
            "sandbox entry module was not found inside materialized_root",
            extra_metadata={"entry_module": entry_module},
        ) from exc
    try:
        resolved_source.relative_to(materialized_root)
    except ValueError as exc:
        raise _sandbox_error(
            "sandbox entry module escapes materialized_root",
            extra_metadata={"entry_module": entry_module},
        ) from exc
    if not resolved_source.is_file():
        raise _sandbox_error(
            "sandbox entry module is not a regular file",
            extra_metadata={"entry_module": entry_module},
        )
    try:
        return source_path, resolved_source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise _sandbox_error(
            "sandbox entry module is not valid UTF-8 text",
            extra_metadata={"entry_module": entry_module},
        ) from exc
    except OSError as exc:
        raise _sandbox_error(
            "sandbox entry module could not be read safely",
            extra_metadata={"entry_module": entry_module},
        ) from exc


# AUDIT-FIX(#1): Reject empty, absolute, traversal, and NUL-containing entry module paths before any filesystem access.
def _validated_entry_module_parts(entry_module: str) -> tuple[str, ...]:
    if "\x00" in entry_module:
        raise _sandbox_error("sandbox entry_module contains a NUL byte")
    path = Path(entry_module)
    if path.is_absolute():
        raise _sandbox_error(
            "sandbox entry_module must be a relative path",
            extra_metadata={"entry_module": entry_module},
        )
    parts = path.parts
    if not parts or entry_module.strip() == "":
        raise _sandbox_error("sandbox entry_module must not be empty")
    invalid_parts = {"", ".", ".."}
    if any(part in invalid_parts for part in parts):
        raise _sandbox_error(
            "sandbox entry_module contains an invalid path segment",
            extra_metadata={"entry_module": entry_module},
        )
    return tuple(parts)


# AUDIT-FIX(#3): Consume IPC as bounded raw bytes and decode with a restricted unpickler before trusting the payload.
def _recv_ipc_message(
    connection: Any,
    *,
    child_pid: int | None,
    hardening: dict[str, Any],
) -> dict[str, Any]:
    try:
        payload = connection.recv_bytes(_MAX_IPC_MESSAGE_BYTES)
    except EOFError as exc:
        raise _sandbox_error(
            "sandbox child closed the IPC channel unexpectedly",
            child_pid=child_pid,
            hardening=hardening,
        ) from exc
    except OSError as exc:
        raise _sandbox_error(
            "sandbox child sent an invalid or oversized IPC payload",
            child_pid=child_pid,
            hardening=hardening,
        ) from exc
    try:
        message = _restricted_pickle_loads(payload)
    except Exception as exc:
        raise _sandbox_error(
            "sandbox child returned an unsafe IPC payload",
            child_pid=child_pid,
            hardening=hardening,
        ) from exc
    if not isinstance(message, dict):
        raise _sandbox_error(
            "sandbox child returned an invalid message",
            child_pid=child_pid,
            hardening=hardening,
        )
    _assert_simple_ipc_value(message)
    return message


# AUDIT-FIX(#6): Send only validated protocol dictionaries so parent-side IPC failures stay explicit and contained.
def _send_ipc_message(
    connection: Any,
    message: dict[str, Any],
    *,
    child_pid: int | None,
    hardening: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    if not isinstance(message, dict):
        raise _sandbox_error(
            "sandbox IPC response must be a dictionary",
            child_pid=child_pid,
            hardening=hardening,
            extra_metadata=extra_metadata,
        )
    _assert_simple_ipc_value(message)
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > _MAX_IPC_MESSAGE_BYTES:
        raise _sandbox_error(
            "sandbox IPC response is too large",
            child_pid=child_pid,
            hardening=hardening,
            extra_metadata=extra_metadata,
        )
    try:
        connection.send_bytes(payload)
    except (EOFError, OSError) as exc:
        raise _sandbox_error(
            "failed to send sandbox IPC response",
            child_pid=child_pid,
            hardening=hardening,
            extra_metadata=extra_metadata,
        ) from exc


# AUDIT-FIX(#3): Keep unpickling on a deny-by-default allowlist so child-controlled IPC cannot instantiate globals in the parent.
def _restricted_pickle_loads(payload: bytes) -> Any:
    return _RestrictedUnpickler(io.BytesIO(payload)).load()


# AUDIT-FIX(#3): Ban GLOBAL/STACK_GLOBAL imports during IPC deserialization.
class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"global '{module}.{name}' is not allowed in sandbox IPC")


# AUDIT-FIX(#3): Reject exotic, cyclic, or oversized-complexity IPC structures even after restricted unpickling.
def _assert_simple_ipc_value(value: Any, *, _depth: int = 0, _seen: set[int] | None = None) -> None:
    if _depth > 32:
        raise _sandbox_error("sandbox IPC payload is too deeply nested")
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _sandbox_error("sandbox IPC payload contains a non-finite float")
        return

    if _seen is None:
        _seen = set()
    container_id = id(value)
    if container_id in _seen:
        raise _sandbox_error("sandbox IPC payload contains cyclic or shared container references")
    _seen.add(container_id)
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise _sandbox_error("sandbox IPC payload contains a non-string dictionary key")
                _assert_simple_ipc_value(item, _depth=_depth + 1, _seen=_seen)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                _assert_simple_ipc_value(item, _depth=_depth + 1, _seen=_seen)
            return
    finally:
        _seen.remove(container_id)
    raise _sandbox_error(
        f"sandbox IPC payload contains unsupported type: {type(value).__name__}",
    )


# AUDIT-FIX(#4): Normalize spoken_count reads through one helper so delta calculation stays stable.
def _read_spoken_count(context: Any) -> int:
    try:
        return _coerce_non_negative_int(getattr(context, "spoken_count", 0) or 0, field_name="spoken_count")
    except ValueError as exc:
        raise _sandbox_error("context spoken_count is invalid") from exc


# AUDIT-FIX(#9): Centralize numeric limit coercion and validation for .env-driven values.
def _coerce_positive_float(value: Any, *, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite positive float") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a finite positive float")
    return number


# AUDIT-FIX(#9): Centralize numeric limit coercion and validation for .env-driven values.
def _coerce_positive_int(value: Any, *, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if number <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return number


# AUDIT-FIX(#4): Clamp count-like values to non-negative integers before computing deltas.
def _coerce_non_negative_int(value: Any, *, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    return max(0, number)


# AUDIT-FIX(#7): Confirm a reported completion is followed by an actual process exit.
def _wait_for_process_exit(process: Any, *, timeout: float) -> bool:
    try:
        process.join(timeout=max(0.0, timeout))
    except (AssertionError, ValueError):
        return False
    return process.exitcode is not None


# AUDIT-FIX(#10): Reap and close Process objects consistently to prevent descriptor/resource buildup.
def _cleanup_process(process: Any) -> None:
    if process is None:
        return
    try:
        alive = process.is_alive()
    except (AssertionError, ValueError):
        alive = False
    if alive:
        _terminate_process_tree(process.pid)
    try:
        process.join(timeout=1.0)
    except (AssertionError, ValueError):
        pass
    _safe_close_process(process)


# AUDIT-FIX(#5): Close partially initialized pipe endpoints defensively during all failure paths.
def _safe_close_connection(connection: Any) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except OSError:
        return


# AUDIT-FIX(#10): Release multiprocessing.Process bookkeeping once the child has been reaped.
def _safe_close_process(process: Any) -> None:
    if process is None:
        return
    try:
        process.close()
    except (AttributeError, OSError, ValueError):
        return


# AUDIT-FIX(#2): Terminate the isolated child process-group first, then fall back to the single PID when needed.
def _terminate_process_tree(pid: int | None) -> None:
    if pid is None:
        return
    terminate_signal = getattr(signal, "SIGTERM", signal.SIGINT)
    kill_signal = getattr(signal, "SIGKILL", terminate_signal)

    if os.name == "posix":
        try:
            child_pgid = os.getpgid(pid)
            parent_pgid = os.getpgrp()
        except OSError:
            child_pgid = None
            parent_pgid = None
        if child_pgid is not None and child_pgid != parent_pgid:
            try:
                os.killpg(child_pgid, terminate_signal)
            except OSError:
                return
            time.sleep(_TERMINATION_GRACE_SECONDS)
            try:
                os.killpg(child_pgid, kill_signal)
            except OSError:
                return
            return
    try:
        os.kill(pid, terminate_signal)
    except OSError:
        return
    time.sleep(_TERMINATION_GRACE_SECONDS)
    try:
        os.kill(pid, kill_signal)
    except OSError:
        return


# AUDIT-FIX(#5): Normalize metadata-rich sandbox exceptions so callers never see raw startup/IPC internals.
def _sandbox_error(
    message: str,
    *,
    error_type: type[SelfCodingSandboxExecutionError] = SelfCodingSandboxExecutionError,
    child_pid: int | None = None,
    hardening: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> SelfCodingSandboxExecutionError:
    metadata: dict[str, Any] = {}
    if hardening is not None:
        metadata["hardening"] = dict(hardening)
    if child_pid is not None:
        metadata["child_pid"] = child_pid
    if extra_metadata:
        metadata.update(extra_metadata)
    return error_type(message, metadata=metadata)


__all__ = [
    "SelfCodingSandboxExecutionError",
    "SelfCodingSandboxLimits",
    "SelfCodingSandboxResult",
    "SelfCodingSandboxRunner",
    "SelfCodingSandboxTimeoutError",
]