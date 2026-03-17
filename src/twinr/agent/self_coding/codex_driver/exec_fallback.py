"""Fallback local compile driver built on `codex exec --json`."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
from queue import Empty, Queue
from typing import Any

from twinr.agent.self_coding.codex_driver.config import (
    codex_optional_model,
    codex_reasoning_effort,
    codex_timeout_seconds,
)
from twinr.agent.self_coding.codex_driver.types import (
    CodexCompileEvent,
    CodexCompileProgress,
    CodexCompileRequest,
    CodexCompileResult,
    CodexCompileRunTranscript,
    CodexDriverProtocolError,
    CodexDriverUnavailableError,
    compile_result_from_text,
)

_MAX_ERROR_TEXT_CHARS = 16_384
_MAX_METADATA_DEPTH = 4
_MAX_METADATA_DICT_ITEMS = 24
_MAX_METADATA_LIST_ITEMS = 16
_MAX_METADATA_STRING_CHARS = 2_048
_QUEUE_POLL_SECONDS = 0.25
_TERMINATION_GRACE_SECONDS = 5.0
_StreamRecord = tuple[str, str, str | BaseException | None]
_EventSink = Callable[[CodexCompileEvent, CodexCompileProgress], None]


@dataclass(slots=True)
class CodexExecRunCollector:
    """Collect normalized state from `codex exec --json` events."""

    thread_id: str | None = None
    final_message: str | None = None
    turn_completed: bool = False
    error_message: str | None = None
    turn_failed: bool = False
    events: list[CodexCompileEvent] = field(default_factory=list)

    def consume(self, payload: dict[str, Any]) -> tuple[CodexCompileEvent, ...]:
        event_type = str(payload.get("type", "")).strip()
        emitted: list[CodexCompileEvent] = []
        if event_type == "thread.started":
            self.thread_id = str(payload.get("thread_id", "") or "").strip() or None
            emitted.append(CodexCompileEvent(kind="thread_started", metadata={"thread_id": self.thread_id}))
            self.events.extend(emitted)
            return tuple(emitted)
        if event_type == "turn.started":
            emitted.append(CodexCompileEvent(kind="turn_started"))
            self.events.extend(emitted)
            return tuple(emitted)
        if event_type == "item.completed":
            item = payload.get("item", {})
            if isinstance(item, dict) and str(item.get("type", "")).strip() == "agent_message":
                self.final_message = str(item.get("text", "") or "").strip() or self.final_message
                emitted.append(
                    CodexCompileEvent(
                        kind="assistant_message",
                        message=str(item.get("text", "") or ""),
                        metadata={"item_id": item.get("id")},
                    )
                )
                self.events.extend(emitted)
                return tuple(emitted)
        if event_type == "turn.completed":
            usage = payload.get("usage", {})
            metadata = dict(usage) if isinstance(usage, dict) else {}
            self.turn_completed = True
            emitted.append(CodexCompileEvent(kind="turn_completed", metadata=_compact_payload(metadata)))
            self.events.extend(emitted)
            return tuple(emitted)
        if event_type in {"turn.failed", "error"}:
            self.turn_failed = True
            self.error_message = _extract_error_message(payload) or self.error_message
            emitted.append(
                CodexCompileEvent(
                    kind=event_type.replace(".", "_"),
                    metadata={
                        "error_message": self.error_message,
                        "raw": _compact_payload(payload),
                    },
                )
            )
            self.events.extend(emitted)
            return tuple(emitted)
        if event_type:
            emitted.append(
                CodexCompileEvent(
                    kind=event_type.replace(".", "_"),
                    metadata={"raw": _compact_payload(payload)},
                )
            )
            self.events.extend(emitted)
        return tuple(emitted)

    def snapshot(self) -> CodexCompileProgress:
        """Return a live progress snapshot for status-store persistence."""

        last_event_kind = self.events[-1].kind if self.events else None
        return CodexCompileProgress(
            thread_id=self.thread_id,
            event_count=len(self.events),
            last_event_kind=last_event_kind,
            final_message_seen=bool(self.final_message),
            turn_completed=self.turn_completed,
        )

    def build_result(self) -> CodexCompileRunTranscript:
        return CodexCompileRunTranscript(
            thread_id=self.thread_id,
            final_message=self.final_message,
            events=tuple(self.events),
        )


class CodexExecFallbackDriver:
    """Run one compile request through `codex exec --json`."""

    def __init__(
        self,
        *,
        command: tuple[str, ...] = ("codex", "exec"),
        timeout_seconds: float | None = None,
        model: str | None = None,
        model_reasoning_effort: str | None = None,
    ) -> None:
        # AUDIT-FIX(#9): Reject empty command vectors and invalid timeout values during construction.
        if not command or any(not str(part).strip() for part in command):
            raise ValueError("command must contain at least one non-empty argument")
        resolved_timeout = codex_timeout_seconds(
            "TWINR_SELF_CODING_CODEX_EXEC_TIMEOUT_SECONDS",
            "TWINR_SELF_CODING_CODEX_TIMEOUT_SECONDS",
        )
        if timeout_seconds is not None:
            resolved_timeout = float(timeout_seconds)
        normalized_timeout = float(resolved_timeout)
        if not math.isfinite(normalized_timeout) or normalized_timeout <= 0.0:
            raise ValueError("timeout_seconds must be a finite positive number")
        self.command = tuple(str(part) for part in command)
        self.timeout_seconds = normalized_timeout
        self.model = model if model is not None else codex_optional_model("TWINR_SELF_CODING_CODEX_MODEL")
        self.model_reasoning_effort = (
            model_reasoning_effort
            if model_reasoning_effort is not None
            else codex_reasoning_effort("TWINR_SELF_CODING_CODEX_MODEL_REASONING_EFFORT", default="high")
        )

    def run_compile(self, request: CodexCompileRequest, *, event_sink: _EventSink | None = None) -> CodexCompileResult:
        # AUDIT-FIX(#2): Canonicalize and validate all filesystem paths before handing them to the CLI.
        workspace_root = self._resolve_workspace_root(request.workspace_root)
        output_schema_path = self._resolve_output_schema_path(request.output_schema_path, workspace_root)
        prompt_text = request.prompt if isinstance(request.prompt, str) else str(request.prompt)
        resolved_command = self._resolve_command()
        command = [
            *resolved_command,
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--color",
            "never",
            "-s",
            "workspace-write",
            "-C",
            str(workspace_root),
            "--output-schema",
            str(output_schema_path),
            "-",  # AUDIT-FIX(#3): Always read the prompt from stdin to avoid argv-length limits and accidental "-" sentinel hangs.
        ]
        insertion_index = len(resolved_command)
        if self.model:
            command[insertion_index:insertion_index] = ["--model", self.model]
            insertion_index += 2
        if self.model_reasoning_effort:
            command[insertion_index:insertion_index] = [
                "--config",
                f'model_reasoning_effort="{self.model_reasoning_effort}"',
            ]

        # AUDIT-FIX(#6): Stream JSONL incrementally so progress events are emitted while the run is still active.
        # AUDIT-FIX(#4): Normalize launch-time OS failures into the driver's domain exception contract.
        # AUDIT-FIX(#5): Run in a dedicated process group so timeout/error cleanup can terminate descendants too.
        process = self._start_process(command, workspace_root)
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

        self._write_prompt_to_stdin(process, prompt_text)

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
                    raise CodexDriverUnavailableError("`codex exec` timed out before the compile finished")
                try:
                    stream_name, record_type, payload = stream_queue.get(timeout=min(_QUEUE_POLL_SECONDS, remaining))
                except Empty:
                    continue

                if record_type == "line":
                    if not isinstance(payload, str):
                        self._terminate_process_tree(process)
                        raise CodexDriverProtocolError("internal stream reader returned a non-text line")
                    if stream_name == "stderr":
                        stderr_buffer.append(payload)
                        continue
                    self._consume_stdout_line(payload, collector, event_sink, process)
                    continue

                if record_type == "error":
                    self._terminate_process_tree(process)
                    assert isinstance(payload, BaseException)
                    if stream_name == "stdout":
                        raise CodexDriverProtocolError("failed to decode `codex exec --json` stdout stream") from payload
                    raise CodexDriverUnavailableError("failed to read stderr from `codex exec`") from payload

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
            # AUDIT-FIX(#1): Prefer structured driver errors emitted in JSON events over stderr noise.
            raise CodexDriverUnavailableError(
                _normalize_error_message(
                    collector.error_message or stderr_text,
                    fallback="`codex exec` reported a failed turn",
                )
            )
        if returncode != 0:
            # AUDIT-FIX(#1): Avoid relying on transcript attributes that are not guaranteed by this file's builder contract.
            raise CodexDriverUnavailableError(
                _normalize_error_message(stderr_text or collector.error_message, fallback="unknown `codex exec` failure")
            )
        if not collector.turn_completed:
            # AUDIT-FIX(#10): Require a terminal success event so truncated JSONL streams do not look successful.
            raise CodexDriverProtocolError("`codex exec --json` exited without a `turn.completed` event")
        if not transcript.final_message:
            raise CodexDriverProtocolError("`codex exec --json` completed without a final assistant message")
        try:
            # AUDIT-FIX(#7): Normalize final-message parse failures into the driver's protocol error domain.
            return compile_result_from_text(transcript.final_message, events=transcript.events)
        except Exception as exc:  # pragma: no cover - defensive wrapper around downstream parser contract.
            raise CodexDriverProtocolError("failed to parse the final assistant message from `codex exec`") from exc

    def _resolve_command(self) -> tuple[str, ...]:
        executable = self.command[0]
        if os.path.sep in executable or (os.path.altsep and os.path.altsep in executable):
            candidate = Path(executable).expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve(strict=False)
            else:
                candidate = candidate.resolve(strict=False)
            if not candidate.exists() or not candidate.is_file() or not os.access(candidate, os.X_OK):
                raise CodexDriverUnavailableError("configured `codex exec` command is not executable")
            resolved_executable = str(candidate)
        else:
            resolved_executable = shutil.which(executable) or ""
            if not resolved_executable:
                raise CodexDriverUnavailableError("`codex exec` is unavailable on this machine")
        return (resolved_executable, *self.command[1:])

    def _resolve_workspace_root(self, raw_workspace_root: str) -> Path:
        # AUDIT-FIX(#2): Resolve workspace_root to a canonical directory before passing it across the process boundary.
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

    def _resolve_output_schema_path(self, raw_output_schema_path: str, workspace_root: Path) -> Path:
        # AUDIT-FIX(#2): Reject schema paths that escape the workspace after symlink and ".." resolution.
        output_schema_path = Path(raw_output_schema_path).expanduser()
        if not output_schema_path.is_absolute():
            output_schema_path = workspace_root / output_schema_path
        try:
            resolved = output_schema_path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise CodexDriverUnavailableError("output_schema_path does not exist") from exc
        except OSError as exc:
            raise CodexDriverUnavailableError(f"failed to resolve output_schema_path: {exc}") from exc
        if not resolved.is_file():
            raise CodexDriverUnavailableError("output_schema_path is not a file")
        if not resolved.is_relative_to(workspace_root):
            raise CodexDriverUnavailableError("output_schema_path must stay inside workspace_root")
        return resolved

    def _start_process(self, command: list[str], workspace_root: Path) -> subprocess.Popen[str]:
        # AUDIT-FIX(#4): Wrap generic OS launch failures instead of leaking raw OSError subclasses.
        # AUDIT-FIX(#5): Start the child in its own process group for whole-tree termination on timeout/failure.
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
            raise CodexDriverUnavailableError("`codex exec` is unavailable on this machine") from exc
        except OSError as exc:
            raise CodexDriverUnavailableError(f"failed to start `codex exec`: {exc}") from exc

    def _write_prompt_to_stdin(self, process: subprocess.Popen[str], prompt_text: str) -> None:
        # AUDIT-FIX(#3): Feed the prompt via stdin so the payload is not constrained by argv parsing/length rules.
        stdin = process.stdin
        if stdin is None:
            self._terminate_process_tree(process)
            raise CodexDriverUnavailableError("`codex exec` stdin pipe is unavailable")
        try:
            stdin.write(prompt_text)
        except BrokenPipeError:
            # The process exited before consuming stdin; stderr/return code handling below will surface the real cause.
            return
        except OSError as exc:
            self._terminate_process_tree(process)
            raise CodexDriverUnavailableError("failed to send the compile prompt to `codex exec`") from exc
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
            raise CodexDriverProtocolError("`codex exec --json` returned invalid JSONL output") from exc
        if not isinstance(payload, dict):
            self._terminate_process_tree(process)
            raise CodexDriverProtocolError("`codex exec --json` returned a non-object event")
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
        # AUDIT-FIX(#5): Terminate the full process tree, not just the top-level CLI wrapper.
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


class _BoundedTextBuffer:
    __slots__ = ("_chunks", "_size", "_max_chars")

    def __init__(self, *, max_chars: int) -> None:
        self._chunks: deque[str] = deque()
        self._size = 0
        self._max_chars = max_chars

    def append(self, text: str) -> None:
        if not text:
            return
        self._chunks.append(text)
        self._size += len(text)
        while self._size > self._max_chars and self._chunks:
            removed = self._chunks.popleft()
            self._size -= len(removed)

    def render(self) -> str:
        return "".join(self._chunks).strip()


def _enqueue_stream_lines(
    stream_name: str,
    pipe: Any,
    stream_queue: Queue[_StreamRecord],
) -> None:
    try:
        if pipe is not None:
            for line in iter(pipe.readline, ""):
                stream_queue.put((stream_name, "line", line))
    except Exception as exc:  # pragma: no cover - depends on platform/encoding failures.
        stream_queue.put((stream_name, "error", exc))
    finally:
        try:
            if pipe is not None:
                pipe.close()
        finally:
            stream_queue.put((stream_name, "eof", None))


def _compact_payload(value: Any, *, depth: int = 0) -> Any:
    # AUDIT-FIX(#8): Keep event metadata bounded so large command/file/search payloads do not explode memory or status-store size.
    if depth >= _MAX_METADATA_DEPTH:
        return _truncate_text(_safe_repr(value), limit=_MAX_METADATA_STRING_CHARS)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _truncate_text(value, limit=_MAX_METADATA_STRING_CHARS)
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        items = list(value.items())
        for index, (key, item_value) in enumerate(items):
            if index >= _MAX_METADATA_DICT_ITEMS:
                compacted["__truncated__"] = f"{len(items) - _MAX_METADATA_DICT_ITEMS} more keys omitted"
                break
            compacted[str(key)] = _compact_payload(item_value, depth=depth + 1)
        return compacted
    if isinstance(value, (list, tuple)):
        compacted_list = [_compact_payload(item, depth=depth + 1) for item in value[:_MAX_METADATA_LIST_ITEMS]]
        if len(value) > _MAX_METADATA_LIST_ITEMS:
            compacted_list.append(f"... {len(value) - _MAX_METADATA_LIST_ITEMS} more items omitted")
        return compacted_list
    return _truncate_text(str(value), limit=_MAX_METADATA_STRING_CHARS)


def _extract_error_message(payload: dict[str, Any]) -> str | None:
    candidates: list[Any] = [
        payload.get("message"),
        payload.get("error"),
        payload.get("reason"),
        payload.get("details"),
    ]
    item = payload.get("item")
    if isinstance(item, dict):
        candidates.extend([item.get("message"), item.get("error"), item.get("reason")])

    for candidate in candidates:
        message = _extract_nested_text(candidate)
        if message:
            return message
    return None


def _extract_nested_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if isinstance(value, dict):
        for key in ("message", "error", "reason", "details", "detail"):
            nested = _extract_nested_text(value.get(key))
            if nested:
                return nested
        return _truncate_text(_safe_repr(value), limit=_MAX_METADATA_STRING_CHARS) or None
    if isinstance(value, list):
        for item in value:
            nested = _extract_nested_text(item)
            if nested:
                return nested
        return None
    return str(value).strip() or None


def _normalize_error_message(message: str | None, *, fallback: str) -> str:
    if not message:
        return fallback
    normalized = " ".join(str(message).split())
    if not normalized:
        return fallback
    return _truncate_text(normalized, limit=_MAX_METADATA_STRING_CHARS)


def _truncate_text(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: max(limit - 1, 0)]}…"


def _safe_repr(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return repr(value)
