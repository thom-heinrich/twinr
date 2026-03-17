"""Run one validated skill handler in a process-isolated sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
import asyncio
import inspect
import os
import stat
import threading
import time
from typing import Any

from twinr.agent.self_coding.sandbox.broker import ParentSkillContextBroker, SandboxSkillContextProxy
from twinr.agent.self_coding.sandbox.trusted_loader import load_trusted_skill_module

try:
    import resource
except ImportError:  # pragma: no cover - only relevant on non-POSIX platforms.
    resource = None


_MAX_ERROR_TEXT_LENGTH = 512


@dataclass(frozen=True, slots=True)
class SelfCodingSandboxLimits:
    """Bound the wall-clock and resource footprint of one sandboxed handler run."""

    timeout_seconds: float = 180.0
    cpu_seconds: int = 20
    address_space_bytes: int = 512 * 1024 * 1024
    max_open_files: int = 64
    max_source_bytes: int = 1 * 1024 * 1024  # AUDIT-FIX(#6): Cap parent-side source reads to avoid pre-sandbox memory exhaustion.

    def __post_init__(self) -> None:
        # AUDIT-FIX(#7): Reject invalid or non-positive limits so a bad config cannot silently disable sandbox ceilings.
        if not isinstance(self.timeout_seconds, (int, float)) or isinstance(self.timeout_seconds, bool):
            raise ValueError("timeout_seconds must be a positive number")
        if not (self.timeout_seconds > 0):
            raise ValueError("timeout_seconds must be greater than zero")
        for field_name in ("cpu_seconds", "address_space_bytes", "max_open_files", "max_source_bytes"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class SelfCodingSandboxResult:
    """Return the minimal observable outcome of one sandboxed execution."""

    spoken_count: int


class SelfCodingSandboxExecutionError(RuntimeError):
    """Raised when a sandboxed handler cannot complete successfully."""


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
        event_name: str | None = None,
    ) -> SelfCodingSandboxResult:
        """Run one handler in a child process and return the resulting spoken-count delta."""

        parent_connection: Any | None = None
        child_connection: Any | None = None
        process: Any | None = None
        initial_spoken_count = _read_spoken_count(context)  # AUDIT-FIX(#5): Snapshot the pre-run counter so the result is a delta, not the cumulative total.
        ctx = get_context("spawn")
        try:
            source_text = _read_entry_source(
                materialized_root=materialized_root,
                entry_module=entry_module,
                max_source_bytes=self.limits.max_source_bytes,
            )  # AUDIT-FIX(#1, #6): Validate the path stays inside the materialized root and bound the file size before reading it.
            parent_connection, child_connection = ctx.Pipe(duplex=True)
            broker = ParentSkillContextBroker(context=context)
            process = ctx.Process(
                target=_sandbox_child_main,
                args=(child_connection, source_text, entry_module, handler_name, event_name, self.limits),
                name=f"twinr-self-coding-{handler_name}",
            )
            try:
                process.start()
            except Exception as exc:  # AUDIT-FIX(#4): Convert child-start failures into a stable sandbox error and ensure cleanup still runs.
                raise SelfCodingSandboxExecutionError("failed to start sandbox child process") from exc
            finally:
                _close_connection_quietly(child_connection)
                child_connection = None
            deadline = time.monotonic() + float(self.limits.timeout_seconds)  # AUDIT-FIX(#2): Start the wall-clock budget once the child is actually running.
            while True:
                remaining_timeout = deadline - time.monotonic()
                if remaining_timeout <= 0:
                    raise SelfCodingSandboxTimeoutError("sandbox child timed out before completing the handler")
                try:
                    has_message = parent_connection.poll(min(0.1, remaining_timeout))
                except (EOFError, OSError) as exc:  # AUDIT-FIX(#4): Broken IPC must not escape as a raw multiprocessing exception.
                    raise SelfCodingSandboxExecutionError("sandbox IPC channel failed while waiting for child output") from exc
                if has_message:
                    try:
                        message = parent_connection.recv()
                    except (EOFError, OSError) as exc:  # AUDIT-FIX(#4): Handle half-closed pipes and child crashes deterministically.
                        if process.exitcode is not None:
                            raise SelfCodingSandboxExecutionError(_format_exit_status(process.exitcode)) from exc
                        raise SelfCodingSandboxExecutionError("sandbox child closed the IPC channel unexpectedly") from exc
                    if not isinstance(message, dict):
                        raise SelfCodingSandboxExecutionError("sandbox child returned an invalid message")
                    kind = str(message.get("kind") or "").strip()
                    if kind == "rpc_call":
                        self._handle_rpc_call(
                            parent_connection,
                            broker,
                            message,
                            remaining_timeout_seconds=deadline - time.monotonic(),
                        )  # AUDIT-FIX(#2, #4): Bound parent-side RPC dispatch by the remaining wall-clock budget and normalize transport failures.
                        continue
                    if kind == "completed":
                        _join_process_quietly(process, timeout=min(1.0, max(0.0, deadline - time.monotonic())))
                        if process.exitcode is None and process.is_alive():
                            raise SelfCodingSandboxExecutionError(
                                "sandbox child did not exit cleanly after reporting completion"
                            )  # AUDIT-FIX(#4): Do not report success while the child is still running in the background.
                        if process.exitcode not in {0, None}:
                            raise SelfCodingSandboxExecutionError(_format_exit_status(process.exitcode))
                        final_spoken_count = _read_spoken_count(context)
                        return SelfCodingSandboxResult(
                            spoken_count=max(0, final_spoken_count - initial_spoken_count)
                        )  # AUDIT-FIX(#5): Return the spoken-count delta promised by the public API contract.
                    if kind == "failed":
                        raise SelfCodingSandboxExecutionError(
                            _normalize_error_text(message.get("error"), default="sandbox child failed")
                        )  # AUDIT-FIX(#8): Sanitize child-controlled error text before surfacing it outside the sandbox boundary.
                    raise SelfCodingSandboxExecutionError(f"unexpected sandbox child message: {kind or 'unknown'}")
                if process.exitcode is not None:
                    if process.exitcode == 0:
                        raise SelfCodingSandboxExecutionError("sandbox child exited before completing its handler")
                    raise SelfCodingSandboxExecutionError(_format_exit_status(process.exitcode))
        except SelfCodingSandboxExecutionError:
            raise
        except Exception as exc:
            raise SelfCodingSandboxExecutionError(
                "sandbox runner failed before the child could complete"
            ) from exc  # AUDIT-FIX(#4): Normalize unexpected local setup/runtime exceptions into the module's public error contract.
        finally:
            _close_connection_quietly(parent_connection)
            _close_connection_quietly(child_connection)
            _stop_process(process)  # AUDIT-FIX(#4): Always terminate/kill/join the child robustly, even after partial setup failures.

    @staticmethod
    def _handle_rpc_call(
        connection: Any,
        broker: ParentSkillContextBroker,
        message: dict[str, Any],
        *,
        remaining_timeout_seconds: float,
    ) -> None:
        call_id = str(message.get("call_id") or "").strip()
        if remaining_timeout_seconds <= 0:
            raise SelfCodingSandboxTimeoutError("sandbox child timed out while waiting for a broker response")
        dispatch_state: dict[str, Any] = {"response": None, "error": None}

        def _dispatch() -> None:
            try:
                response = broker.dispatch(message)
                if not isinstance(response, dict):
                    raise SelfCodingSandboxExecutionError("sandbox broker returned an invalid response")
                dispatch_state["response"] = response
            except Exception as exc:  # AUDIT-FIX(#2, #4): Capture broker failures without letting raw parent exceptions or IPC errors escape unpredictably.
                dispatch_state["error"] = exc

        dispatch_thread = threading.Thread(
            target=_dispatch,
            name="twinr-sandbox-broker-dispatch",
            daemon=True,
        )
        dispatch_thread.start()
        dispatch_thread.join(timeout=remaining_timeout_seconds)
        if dispatch_thread.is_alive():
            raise SelfCodingSandboxTimeoutError("sandbox broker call exceeded the wall-clock budget")  # AUDIT-FIX(#2): Enforce timeout_seconds across parent-side RPC work as well.
        if dispatch_state["error"] is not None:
            _send_connection_message(
                connection,
                {"kind": "rpc_error", "call_id": call_id, "error": "sandbox broker failure"},
            )  # AUDIT-FIX(#8): Do not leak raw parent-side exception text back into the sandbox.
            return
        response = dispatch_state["response"]
        if not isinstance(response, dict):
            raise SelfCodingSandboxExecutionError("sandbox broker returned an invalid response")
        _send_connection_message(connection, response)  # AUDIT-FIX(#4): Normalize broken-pipe and pickling failures into sandbox execution errors.


def _sandbox_child_main(
    connection: Any,
    source_text: str,
    entry_module: str,
    handler_name: str,
    event_name: str | None,
    limits: SelfCodingSandboxLimits,
) -> None:
    """Run inside the spawned child process under restricted OS and Python limits."""

    try:
        _apply_resource_limits(limits)
        proxy = SandboxSkillContextProxy(connection=connection)
        module = load_trusted_skill_module(source_text=source_text, filename=entry_module)
        handler = module.get_handler(handler_name)
        if not callable(handler):
            raise SelfCodingSandboxExecutionError("sandbox handler is not callable")  # AUDIT-FIX(#3): Fail clearly if the trusted loader returns a non-callable handler.
        call_args, call_kwargs = _build_handler_call(handler, proxy, event_name)
        result = handler(*call_args, **call_kwargs)
        if inspect.isawaitable(result):
            asyncio.run(result)  # AUDIT-FIX(#3): Support async skill handlers instead of silently returning an un-awaited coroutine.
        _try_send_child_message(connection, {"kind": "completed"})
    except BaseException as exc:  # AUDIT-FIX(#4): Report broad child failures (including runtime resource errors) back to the parent when possible.
        _try_send_child_message(
            connection,
            {"kind": "failed", "error": _normalize_error_text(exc, default="sandbox child failure")},
        )  # AUDIT-FIX(#8): Sanitize sandbox-to-parent error text to reduce detail leakage and log injection.
    finally:
        _close_connection_quietly(connection)


def sandbox_loader_child_main(
    connection: Any,
    source_text: str,
    entry_module: str,
    handler_name: str,
    event_name: str | None,
    limits: SelfCodingSandboxLimits,
) -> None:
    """Public compatibility wrapper for the spawned sandbox child entrypoint."""

    _sandbox_child_main(
        connection,
        source_text,
        entry_module,
        handler_name,
        event_name,
        limits,
    )


def _apply_resource_limits(limits: SelfCodingSandboxLimits) -> None:
    """Apply best-effort POSIX resource ceilings inside the spawned child."""

    if resource is None:  # pragma: no cover - only relevant on non-POSIX platforms.
        return
    _set_limit(getattr(resource, "RLIMIT_CPU", None), limits.cpu_seconds)
    _set_limit(getattr(resource, "RLIMIT_AS", None), limits.address_space_bytes)
    _set_limit(getattr(resource, "RLIMIT_NOFILE", None), limits.max_open_files)
    if hasattr(os, "nice"):
        try:
            os.nice(10)
        except OSError:
            pass


def _set_limit(limit_name: int | None, value: int) -> None:
    if resource is None or limit_name is None:
        return
    try:
        sanitized_value = int(value)
    except (TypeError, ValueError):
        return
    if sanitized_value <= 0:
        return  # AUDIT-FIX(#7): Never attempt to apply non-positive RLIMIT values.
    try:
        resource.setrlimit(limit_name, (sanitized_value, sanitized_value))
    except (OSError, ValueError):
        return


def _read_entry_source(*, materialized_root: Path, entry_module: str, max_source_bytes: int) -> str:
    entry_module_text = str(entry_module or "").strip()
    if not entry_module_text:
        raise SelfCodingSandboxExecutionError("sandbox entry module path is missing")
    if "\x00" in entry_module_text:
        raise SelfCodingSandboxExecutionError("sandbox entry module path contains invalid characters")
    try:
        root_path = Path(materialized_root).resolve(strict=True)
    except OSError as exc:
        raise SelfCodingSandboxExecutionError("sandbox materialized root is unavailable") from exc
    if not root_path.is_dir():
        raise SelfCodingSandboxExecutionError("sandbox materialized root is not a directory")  # AUDIT-FIX(#1): Refuse non-directory roots before resolving the requested module path.
    entry_path = Path(entry_module_text)
    if entry_path.is_absolute():
        raise SelfCodingSandboxExecutionError("sandbox entry module must be a relative path")  # AUDIT-FIX(#1): Reject absolute paths so the caller cannot bypass the materialized root.
    if any(part in {"", ".", ".."} for part in entry_path.parts):
        raise SelfCodingSandboxExecutionError(
            "sandbox entry module path must not contain traversal segments"
        )  # AUDIT-FIX(#1): Reject traversal tokens before path resolution.
    try:
        source_path = (root_path / entry_path).resolve(strict=True)
    except OSError as exc:
        raise SelfCodingSandboxExecutionError("sandbox entry module could not be resolved") from exc
    try:
        source_path.relative_to(root_path)
    except ValueError as exc:
        raise SelfCodingSandboxExecutionError("sandbox entry module escapes the materialized root") from exc  # AUDIT-FIX(#1): Enforce that symlinks and relative segments cannot escape the trusted root.
    file_descriptor = _open_readonly_file(source_path)
    try:
        source_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise SelfCodingSandboxExecutionError("sandbox entry module is not a regular file")  # AUDIT-FIX(#1): Only regular files are valid module sources.
        if source_stat.st_size > max_source_bytes:
            raise SelfCodingSandboxExecutionError(
                f"sandbox entry module exceeds the {max_source_bytes}-byte size limit"
            )  # AUDIT-FIX(#6): Bound parent memory use before loading sandbox source into RAM.
        with os.fdopen(file_descriptor, "r", encoding="utf-8", closefd=True) as source_file:
            file_descriptor = None
            return source_file.read()
    except UnicodeDecodeError as exc:
        raise SelfCodingSandboxExecutionError("sandbox entry module is not valid UTF-8 text") from exc
    except OSError as exc:
        raise SelfCodingSandboxExecutionError("sandbox entry module could not be read") from exc
    finally:
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except OSError:
                pass


def _open_readonly_file(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW  # AUDIT-FIX(#1): Refuse to follow a swapped final symlink during the actual file open.
    try:
        return os.open(path, flags)
    except OSError as exc:
        raise SelfCodingSandboxExecutionError("sandbox entry module could not be opened securely") from exc


def _build_handler_call(handler: Any, proxy: SandboxSkillContextProxy, event_name: str | None) -> tuple[list[Any], dict[str, Any]]:
    call_args: list[Any] = [proxy]
    call_kwargs: dict[str, Any] = {}
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return call_args, call_kwargs
    event_parameter = signature.parameters.get("event_name")
    if event_parameter is None:
        return call_args, call_kwargs
    if event_parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
        call_args.append(event_name)  # AUDIT-FIX(#3): Support positional-only event_name parameters instead of forcing a keyword call that can fail.
    else:
        call_kwargs["event_name"] = event_name
    return call_args, call_kwargs


def _read_spoken_count(context: Any) -> int:
    value = getattr(context, "spoken_count", 0)
    if value is None:
        return 0
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise SelfCodingSandboxExecutionError("sandbox context exposed an invalid spoken_count value") from exc
    return max(0, count)


def _normalize_error_text(value: Any, *, default: str) -> str:
    raw_text = str(value or "").strip()
    if not raw_text:
        return default
    normalized = " ".join(raw_text.split())
    if not normalized:
        return default
    if len(normalized) > _MAX_ERROR_TEXT_LENGTH:
        return f"{normalized[:_MAX_ERROR_TEXT_LENGTH - 1]}…"
    return normalized


def _send_connection_message(connection: Any, message: dict[str, Any]) -> None:
    try:
        connection.send(message)
    except Exception as exc:
        raise SelfCodingSandboxExecutionError("sandbox IPC channel failed while sending a response") from exc


def _try_send_child_message(connection: Any, message: dict[str, Any]) -> None:
    try:
        connection.send(message)
    except Exception:
        return


def _close_connection_quietly(connection: Any | None) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except Exception:
        return


def _join_process_quietly(process: Any | None, *, timeout: float) -> None:
    if process is None:
        return
    try:
        process.join(timeout=max(0.0, timeout))
    except Exception:
        return


def _stop_process(process: Any | None) -> None:
    if process is None:
        return
    try:
        alive = process.is_alive()
    except Exception:
        alive = False
    if alive:
        try:
            process.terminate()
        except Exception:
            pass
        _join_process_quietly(process, timeout=0.5)
        try:
            alive = process.is_alive()
        except Exception:
            alive = False
        if alive:
            try:
                process.kill()
            except Exception:
                pass
            _join_process_quietly(process, timeout=1.0)
        return
    _join_process_quietly(process, timeout=0.0)


def _format_exit_status(exitcode: int | None) -> str:
    if exitcode is None:
        return "sandbox child exited unexpectedly"
    if exitcode < 0:
        return f"sandbox child terminated by signal {-exitcode}"
    return f"sandbox child exited with status {exitcode}"
