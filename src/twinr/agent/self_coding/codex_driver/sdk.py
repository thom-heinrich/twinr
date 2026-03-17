"""Run self_coding compile jobs through a pinned local Codex SDK bridge.

The TypeScript Codex SDK is the supported integration path for single-shot
automation and backend jobs. This module keeps Twinr on that path while
reusing the existing JSONL event normalization and compile-result parsing used
by the local fallback driver.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import threading
import time
from queue import Empty, Queue
from typing import Any

from twinr.agent.self_coding.codex_driver.exec_fallback import (
    CodexExecRunCollector,
    _BoundedTextBuffer,
    _enqueue_stream_lines,
    _normalize_error_message,
)
from twinr.agent.self_coding.codex_driver.config import (
    codex_optional_model,
    codex_reasoning_effort,
    codex_timeout_seconds,
)
from twinr.agent.self_coding.codex_driver.environment import (
    assert_codex_sdk_environment_ready,
    collect_codex_sdk_environment_report,
    default_bridge_script_path,
)
from twinr.agent.self_coding.codex_driver.types import (
    CodexCompileEvent,
    CodexCompileProgress,
    CodexCompileRequest,
    CodexCompileResult,
    CodexDriverProtocolError,
    CodexDriverUnavailableError,
    compile_result_from_text,
)

_MAX_ERROR_TEXT_CHARS = 16_384
_QUEUE_POLL_SECONDS = 0.25
_TERMINATION_GRACE_SECONDS = 5.0
_STREAM_QUEUE_MAX_RECORDS = 1_024  # AUDIT-FIX(#4): Bound queued bridge output to prevent unbounded RAM growth on the RPi.
_StreamRecord = tuple[str, str, str | BaseException | None]
_EventSink = Callable[[CodexCompileEvent, CodexCompileProgress], None]


class CodexSdkDriver:
    """Run one compile request through the pinned local `@openai/codex-sdk`.

    The driver shells out to a tiny repo-local Node bridge script that imports
    `@openai/codex-sdk` from a pinned package directory, streams SDK events as
    JSONL, and lets the Python side keep owning validation, status persistence,
    and artifact parsing.
    """

    def __init__(
        self,
        *,
        command: tuple[str, ...] = ("node",),
        bridge_script: str | Path | None = None,
        timeout_seconds: float | None = None,
        model: str | None = None,
        model_reasoning_effort: str | None = None,
    ) -> None:
        if not command or any(not str(part).strip() for part in command):
            raise ValueError("command must contain at least one non-empty argument")
        resolved_timeout = codex_timeout_seconds(
            "TWINR_SELF_CODING_CODEX_SDK_TIMEOUT_SECONDS",
            "TWINR_SELF_CODING_CODEX_TIMEOUT_SECONDS",
        )
        if timeout_seconds is not None:
            resolved_timeout = float(timeout_seconds)
        normalized_timeout = float(resolved_timeout)
        if not math.isfinite(normalized_timeout) or normalized_timeout <= 0.0:
            raise ValueError("timeout_seconds must be a finite positive number")
        self.command = tuple(str(part).strip() for part in command)  # AUDIT-FIX(#8): Persist the normalized command actually validated above.
        self.timeout_seconds = normalized_timeout
        self.model = model if model is not None else codex_optional_model("TWINR_SELF_CODING_CODEX_MODEL")
        self.model_reasoning_effort = (
            model_reasoning_effort
            if model_reasoning_effort is not None
            else codex_reasoning_effort("TWINR_SELF_CODING_CODEX_MODEL_REASONING_EFFORT", default="high")
        )

        default_bridge = default_bridge_script_path().resolve(strict=False)
        resolved_bridge = default_bridge if bridge_script is None else Path(bridge_script).expanduser().resolve(strict=False)
        self._uses_default_bridge = resolved_bridge == default_bridge  # AUDIT-FIX(#9): Preserve default-bridge safety checks even when the caller passes the default path explicitly.
        self._startup_self_test_passed = False
        self._startup_self_test_lock = threading.Lock()  # AUDIT-FIX(#6): Serialize first-run self-test so concurrent compile calls cannot race.
        self.bridge_script = resolved_bridge

    def run_compile(self, request: CodexCompileRequest, *, event_sink: _EventSink | None = None) -> CodexCompileResult:
        """Run one bounded SDK-backed compile turn and return the normalized result."""

        try:
            workspace_root = self._resolve_workspace_root(request.workspace_root)  # AUDIT-FIX(#5): Normalize request input failures into deterministic driver errors.
        except AttributeError as exc:
            raise CodexDriverProtocolError("compile request is missing `workspace_root`") from exc

        bridge_script = self._resolve_bridge_script()
        resolved_command = self._resolve_command()
        payload = self._build_bridge_payload(request, workspace_root)
        self._ensure_startup_self_test(resolved_command=resolved_command, bridge_script=bridge_script)
        process = self._start_process([*resolved_command, str(bridge_script)], workspace_root)

        stream_queue: Queue[_StreamRecord] = Queue(maxsize=_STREAM_QUEUE_MAX_RECORDS)  # AUDIT-FIX(#4): Apply backpressure instead of allowing unlimited queued output.
        collector = CodexExecRunCollector()
        stderr_buffer = _BoundedTextBuffer(max_chars=_MAX_ERROR_TEXT_CHARS)
        deadline = time.monotonic() + self.timeout_seconds
        stdout_done = False
        stderr_done = False
        stdout_thread: threading.Thread | None = None
        stderr_thread: threading.Thread | None = None
        stdout_thread_started = False
        stderr_thread_started = False

        try:  # AUDIT-FIX(#2): Guard the entire child-process lifecycle so every failure path tears the process tree down cleanly.
            stdout_thread = threading.Thread(
                target=_enqueue_stream_lines,
                args=("stdout", process.stdout, stream_queue),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=_enqueue_stream_lines,
                args=("stderr", process.stderr, stream_queue),
                daemon=True,
            )
            stdout_thread.start()
            stdout_thread_started = True
            stderr_thread.start()
            stderr_thread_started = True

            self._write_payload_to_stdin(process, payload)

            while not (stdout_done and stderr_done and process.poll() is not None):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self._terminate_process_tree(process)
                    raise CodexDriverUnavailableError("the local codex-sdk bridge timed out before the compile finished")
                try:
                    stream_name, record_type, payload_record = stream_queue.get(timeout=min(_QUEUE_POLL_SECONDS, remaining))
                except Empty:
                    continue

                if record_type == "line":
                    if not isinstance(payload_record, str):
                        self._terminate_process_tree(process)
                        raise CodexDriverProtocolError("internal stream reader returned a non-text line")
                    if stream_name == "stderr":
                        stderr_buffer.append(payload_record)
                        continue
                    self._consume_stdout_line(payload_record, collector, event_sink, process)
                    continue

                if record_type == "error":
                    self._terminate_process_tree(process)
                    if not isinstance(payload_record, BaseException):  # AUDIT-FIX(#7): Do not rely on `assert`, which disappears under `python -O`.
                        raise CodexDriverProtocolError("internal stream reader returned a non-exception error payload")
                    if stream_name == "stdout":
                        raise CodexDriverProtocolError("failed to decode the local codex-sdk bridge stdout stream") from payload_record
                    raise CodexDriverUnavailableError("failed to read stderr from the local codex-sdk bridge") from payload_record

                if record_type == "eof":
                    if stream_name == "stdout":
                        stdout_done = True
                    else:
                        stderr_done = True
                    continue

                self._terminate_process_tree(process)
                raise CodexDriverProtocolError("internal stream reader returned an unknown record type")
        except BaseException:
            self._terminate_process_tree(process)
            raise
        finally:
            self._safe_close_pipe(process.stdin)
            self._safe_close_pipe(process.stdout)
            self._safe_close_pipe(process.stderr)
            if stdout_thread_started and stdout_thread is not None:
                stdout_thread.join(timeout=1.0)
            if stderr_thread_started and stderr_thread is not None:
                stderr_thread.join(timeout=1.0)

        try:
            transcript = collector.build_result()  # AUDIT-FIX(#2): Normalize malformed collector state instead of leaking raw internal exceptions.
        except Exception as exc:
            raise CodexDriverProtocolError("failed to build the local codex-sdk transcript") from exc

        stderr_text = stderr_buffer.render()
        returncode = process.returncode
        if returncode is None:  # AUDIT-FIX(#2): Fail closed if the child somehow exited the loop without a stable return code.
            raise CodexDriverProtocolError("the local codex-sdk bridge exited without a final return code")
        if collector.turn_failed:
            raise CodexDriverUnavailableError(
                _normalize_error_message(
                    collector.error_message or stderr_text,
                    fallback="the local codex-sdk bridge reported a failed turn",
                )
            )
        if returncode != 0:
            raise CodexDriverUnavailableError(
                _normalize_error_message(stderr_text or collector.error_message, fallback="unknown local codex-sdk failure")
            )
        if not collector.turn_completed:
            raise CodexDriverProtocolError("the local codex-sdk bridge exited without a `turn.completed` event")
        if not transcript.final_message:
            raise CodexDriverProtocolError("the local codex-sdk bridge completed without a final assistant message")
        try:
            return compile_result_from_text(transcript.final_message, events=transcript.events)
        except Exception as exc:  # pragma: no cover - defensive wrapper around downstream parser contract.
            raise CodexDriverProtocolError("failed to parse the final assistant message from the local codex-sdk bridge") from exc

    def _resolve_command(self) -> tuple[str, ...]:
        executable = self.command[0]
        if os.path.sep in executable or (os.path.altsep and os.path.altsep in executable):
            candidate = Path(executable).expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve(strict=False)
            else:
                candidate = candidate.resolve(strict=False)
            if not candidate.exists() or not candidate.is_file() or not os.access(candidate, os.X_OK):
                raise CodexDriverUnavailableError("configured codex-sdk command is not executable")
            resolved_executable = str(candidate)
        else:
            resolved_executable = shutil.which(executable) or ""
            if not resolved_executable:
                raise CodexDriverUnavailableError("the configured codex-sdk bridge command is unavailable on this machine")
        return (resolved_executable, *self.command[1:])

    def _resolve_bridge_script(self) -> Path:
        bridge_script = self.bridge_script.resolve(strict=False)
        if not bridge_script.exists() or not bridge_script.is_file():
            raise CodexDriverUnavailableError(f"codex-sdk bridge script is missing: {bridge_script}")
        if self._uses_default_bridge:
            install_root = bridge_script.parent
            sdk_package = install_root / "node_modules" / "@openai" / "codex-sdk"
            if not sdk_package.exists():
                raise CodexDriverUnavailableError(
                    f"codex-sdk dependencies are missing under {install_root}; run `npm ci` there first"
                )
        return bridge_script

    def _ensure_startup_self_test(
        self,
        *,
        resolved_command: tuple[str, ...],
        bridge_script: Path,
    ) -> None:
        if self._startup_self_test_passed:
            return
        with self._startup_self_test_lock:  # AUDIT-FIX(#6): Make the one-time self-test idempotent under concurrent callers.
            if self._startup_self_test_passed:
                return
            report = collect_codex_sdk_environment_report(
                bridge_script=bridge_script,
                bridge_command=resolved_command,
                which_resolver=shutil.which,
                subprocess_runner=subprocess.run,
                run_local_self_test=True,
                run_live_auth_check=False,
                self_test_timeout_seconds=min(self.timeout_seconds, 15.0),
                require_bridge_dependencies=self._uses_default_bridge,
                require_codex_auth=self._uses_default_bridge,
            )
            assert_codex_sdk_environment_ready(report)
            self._startup_self_test_passed = True

    @staticmethod
    def _coerce_path_text(raw_path: str | os.PathLike[str], *, field_name: str) -> str:
        try:
            path_text = os.fspath(raw_path)
        except TypeError as exc:
            raise CodexDriverProtocolError(f"{field_name} must be a path-like string") from exc
        if not isinstance(path_text, str):
            raise CodexDriverProtocolError(f"{field_name} must be a text path")
        if not path_text.strip():
            raise CodexDriverProtocolError(f"{field_name} must not be empty")
        return path_text

    @classmethod
    def _normalize_workspace_relative_path(
        cls,
        raw_path: str | os.PathLike[str],
        *,
        field_name: str,
        workspace_root: Path,
    ) -> str:
        path_text = cls._coerce_path_text(raw_path, field_name=field_name)  # AUDIT-FIX(#1): Validate bridge file-path fields before serializing them into the Node job payload.
        candidate = Path(path_text)
        if candidate.is_absolute():
            raise CodexDriverProtocolError(f"{field_name} must be a relative path inside workspace_root")
        if any(part == ".." for part in candidate.parts):
            raise CodexDriverProtocolError(f"{field_name} must not escape workspace_root")
        try:
            resolved_candidate = (workspace_root / candidate).resolve(strict=False)
        except OSError as exc:
            raise CodexDriverProtocolError(f"failed to resolve {field_name}: {exc}") from exc
        try:
            resolved_candidate.relative_to(workspace_root)
        except ValueError as exc:
            raise CodexDriverProtocolError(f"{field_name} must stay inside workspace_root") from exc
        return str(candidate)

    @classmethod
    def _resolve_workspace_root(cls, raw_workspace_root: str | os.PathLike[str]) -> Path:
        workspace_root_text = cls._coerce_path_text(raw_workspace_root, field_name="workspace_root")  # AUDIT-FIX(#5): Reject empty and non-path-like workspace roots instead of silently falling back to the service CWD.
        workspace_root = Path(workspace_root_text).expanduser()
        try:
            resolved = workspace_root.resolve(strict=True)
        except FileNotFoundError as exc:
            raise CodexDriverUnavailableError("workspace_root does not exist") from exc
        except OSError as exc:
            raise CodexDriverUnavailableError(f"failed to resolve workspace_root: {exc}") from exc
        if not resolved.is_dir():
            raise CodexDriverUnavailableError("workspace_root is not a directory")
        return resolved

    def _build_bridge_payload(self, request: CodexCompileRequest, workspace_root: Path) -> str:
        try:
            request_path = self._normalize_workspace_relative_path(
                request.request_path,
                field_name="request_path",
                workspace_root=workspace_root,
            )
            output_schema_path = self._normalize_workspace_relative_path(
                request.output_schema_path,
                field_name="output_schema_path",
                workspace_root=workspace_root,
            )
            prompt = request.prompt
            output_schema = request.output_schema
        except AttributeError as exc:
            raise CodexDriverProtocolError("compile request is missing one or more required fields") from exc

        try:
            return json.dumps(
                {
                    "workspaceRoot": str(workspace_root),
                    "requestPath": request_path,
                    "outputSchemaPath": output_schema_path,
                    "prompt": prompt,
                    "outputSchema": output_schema,
                    "model": self.model,
                    "modelReasoningEffort": self.model_reasoning_effort,
                },
                ensure_ascii=False,
            )
        except (TypeError, ValueError) as exc:
            raise CodexDriverProtocolError("failed to serialize the local codex-sdk request payload") from exc

    def _start_process(self, command: list[str], workspace_root: Path) -> subprocess.Popen[str]:
        kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "strict",
            "cwd": str(workspace_root),
            "bufsize": 1,
        }
        if os.name == "posix":
            kwargs["start_new_session"] = True
        elif os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            return subprocess.Popen(command, **kwargs)
        except FileNotFoundError as exc:
            raise CodexDriverUnavailableError("the configured codex-sdk bridge command is unavailable on this machine") from exc
        except OSError as exc:
            raise CodexDriverUnavailableError(f"failed to start the local codex-sdk bridge: {exc}") from exc

    def _write_payload_to_stdin(self, process: subprocess.Popen[str], payload: str) -> None:
        stdin = process.stdin
        if stdin is None:
            self._terminate_process_tree(process)
            raise CodexDriverUnavailableError("the local codex-sdk bridge stdin pipe is unavailable")
        try:
            stdin.write(payload)
            stdin.write("\n")  # AUDIT-FIX(#3): Force a flushable record boundary for bridges that consume stdin line-by-line.
            stdin.flush()  # AUDIT-FIX(#3): Surface short writes/BrokenPipe here instead of suppressing them during close().
        except BrokenPipeError as exc:
            self._terminate_process_tree(process)
            raise CodexDriverUnavailableError("the local codex-sdk bridge closed stdin before receiving the compile payload") from exc
        except OSError as exc:
            self._terminate_process_tree(process)
            raise CodexDriverUnavailableError("failed to send the compile payload to the local codex-sdk bridge") from exc
        finally:
            self._safe_close_pipe(stdin)

    def _consume_stdout_line(
        self,
        line: str,
        collector: CodexExecRunCollector,
        event_sink: _EventSink | None,
        process: subprocess.Popen[str],
    ) -> None:
        text = line.strip()
        if not text:
            return
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            self._terminate_process_tree(process)
            raise CodexDriverProtocolError("the local codex-sdk bridge returned invalid JSONL output") from exc
        if not isinstance(payload, dict):
            self._terminate_process_tree(process)
            raise CodexDriverProtocolError("the local codex-sdk bridge returned a non-object event")

        try:
            emitted = collector.consume(payload)  # AUDIT-FIX(#2): Treat malformed event sequences as protocol failures and terminate the child promptly.
            snapshot = collector.snapshot() if event_sink is not None else None
        except Exception as exc:
            self._terminate_process_tree(process)
            if isinstance(exc, (CodexDriverProtocolError, CodexDriverUnavailableError)):
                raise
            raise CodexDriverProtocolError("the local codex-sdk bridge emitted an invalid event sequence") from exc

        if event_sink is not None and snapshot is not None:
            for event in emitted:
                try:
                    event_sink(event, snapshot)
                except Exception:
                    self._terminate_process_tree(process)
                    raise

    def _terminate_process_tree(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            return
        except OSError:
            try:
                process.terminate()
            except OSError:
                return
        try:
            process.wait(timeout=_TERMINATION_GRACE_SECONDS)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            return
        except OSError:
            return
        try:
            process.wait(timeout=_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            return

    @staticmethod
    def _safe_close_pipe(pipe: Any) -> None:
        if pipe is None:
            return
        try:
            pipe.close()
        except Exception:
            return