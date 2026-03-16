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
from twinr.agent.self_coding.codex_driver.types import (
    CodexCompileEvent,
    CodexCompileProgress,
    CodexCompileRequest,
    CodexCompileResult,
    CodexDriverProtocolError,
    CodexDriverUnavailableError,
    compile_result_from_text,
)

_DEFAULT_TIMEOUT_SECONDS = 180.0
_MAX_ERROR_TEXT_CHARS = 16_384
_QUEUE_POLL_SECONDS = 0.25
_TERMINATION_GRACE_SECONDS = 5.0
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
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not command or any(not str(part).strip() for part in command):
            raise ValueError("command must contain at least one non-empty argument")
        normalized_timeout = float(timeout_seconds)
        if not math.isfinite(normalized_timeout) or normalized_timeout <= 0.0:
            raise ValueError("timeout_seconds must be a finite positive number")
        self.command = tuple(str(part) for part in command)
        self.timeout_seconds = normalized_timeout
        self._uses_default_bridge = bridge_script is None
        self._startup_self_test_passed = False
        self.bridge_script = (
            _default_bridge_script_path() if bridge_script is None else Path(bridge_script).expanduser().resolve(strict=False)
        )

    def run_compile(self, request: CodexCompileRequest, *, event_sink: _EventSink | None = None) -> CodexCompileResult:
        """Run one bounded SDK-backed compile turn and return the normalized result."""

        workspace_root = self._resolve_workspace_root(request.workspace_root)
        bridge_script = self._resolve_bridge_script()
        resolved_command = self._resolve_command()
        self._ensure_startup_self_test(resolved_command=resolved_command, bridge_script=bridge_script)
        payload = self._build_bridge_payload(request, workspace_root)
        process = self._start_process([*resolved_command, str(bridge_script)], workspace_root)

        stream_queue: Queue[_StreamRecord] = Queue()
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
        stderr_thread.start()

        self._write_payload_to_stdin(process, payload)

        collector = CodexExecRunCollector()
        stderr_buffer = _BoundedTextBuffer(max_chars=_MAX_ERROR_TEXT_CHARS)
        deadline = time.monotonic() + self.timeout_seconds
        stdout_done = False
        stderr_done = False

        try:
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
                    assert isinstance(payload_record, BaseException)
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
        finally:
            self._safe_close_pipe(process.stdin)
            self._safe_close_pipe(process.stdout)
            self._safe_close_pipe(process.stderr)
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)

        transcript = collector.build_result()
        stderr_text = stderr_buffer.render()
        returncode = process.returncode if process.returncode is not None else process.wait(timeout=0)
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

        command = [*resolved_command, str(bridge_script), "--self-test"]
        timeout_seconds = min(self.timeout_seconds, 15.0)
        try:
            completed = subprocess.run(
                command,
                cwd=str(bridge_script.parent),
                text=True,
                encoding="utf-8",
                errors="strict",
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise CodexDriverUnavailableError("the configured codex-sdk bridge command is unavailable on this machine") from exc
        except subprocess.TimeoutExpired as exc:
            raise CodexDriverUnavailableError("the codex-sdk bridge startup self-test timed out") from exc
        except OSError as exc:
            raise CodexDriverUnavailableError(f"failed to start the codex-sdk bridge startup self-test: {exc}") from exc

        stdout_text = str(completed.stdout or "").strip()
        stderr_text = str(completed.stderr or "").strip()
        if completed.returncode != 0:
            detail = _normalize_error_message(
                stderr_text or stdout_text,
                fallback="the codex-sdk bridge startup self-test failed",
            )
            raise CodexDriverUnavailableError(f"codex-sdk bridge startup self-test failed: {detail}")

        if not stdout_text:
            raise CodexDriverUnavailableError("codex-sdk bridge startup self-test returned no result payload")
        last_line = stdout_text.splitlines()[-1]
        try:
            payload = json.loads(last_line)
        except json.JSONDecodeError as exc:
            raise CodexDriverUnavailableError("codex-sdk bridge startup self-test returned invalid JSON") from exc
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise CodexDriverUnavailableError("codex-sdk bridge startup self-test did not confirm a healthy runtime")
        self._startup_self_test_passed = True

    @staticmethod
    def _resolve_workspace_root(raw_workspace_root: str) -> Path:
        workspace_root = Path(raw_workspace_root).expanduser()
        try:
            resolved = workspace_root.resolve(strict=True)
        except FileNotFoundError as exc:
            raise CodexDriverUnavailableError("workspace_root does not exist") from exc
        except OSError as exc:
            raise CodexDriverUnavailableError(f"failed to resolve workspace_root: {exc}") from exc
        if not resolved.is_dir():
            raise CodexDriverUnavailableError("workspace_root is not a directory")
        return resolved

    @staticmethod
    def _build_bridge_payload(request: CodexCompileRequest, workspace_root: Path) -> str:
        try:
            return json.dumps(
                {
                    "workspaceRoot": str(workspace_root),
                    "requestPath": request.request_path,
                    "outputSchemaPath": request.output_schema_path,
                    "prompt": request.prompt,
                    "outputSchema": request.output_schema,
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
        except BrokenPipeError:
            return
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
        emitted = collector.consume(payload)
        if event_sink is not None:
            snapshot = collector.snapshot()
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


def _default_bridge_script_path() -> Path:
    return Path(__file__).with_name("sdk_bridge") / "run_compile.mjs"
