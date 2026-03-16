"""Keep the legacy local `codex app-server` compile probe available for debug use."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import selectors
import subprocess
import time
from typing import Any

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


@dataclass(slots=True)
class CodexAppServerRunCollector:
    """Collect normalized state from app-server JSON-RPC messages."""

    thread_id: str | None = None
    turn_id: str | None = None
    final_message: str | None = None
    error_message: str | None = None
    turn_completed: bool = False
    _message_parts: list[str] = field(default_factory=list)
    events: list[CodexCompileEvent] = field(default_factory=list)

    def consume(self, message: dict[str, Any]) -> tuple[CodexCompileEvent, ...]:
        emitted: list[CodexCompileEvent] = []
        result = message.get("result")
        if isinstance(result, dict):
            thread = result.get("thread")
            if isinstance(thread, dict) and thread.get("id"):
                self.thread_id = str(thread.get("id"))
            turn = result.get("turn")
            if isinstance(turn, dict) and turn.get("id"):
                self.turn_id = str(turn.get("id"))

        method = str(message.get("method", "")).strip()
        params = message.get("params", {})
        if not isinstance(params, dict):
            params = {}
        thread_id = str(params.get("threadId", "") or "").strip() or None
        turn_id = str(params.get("turnId", "") or "").strip() or None
        if thread_id:
            self.thread_id = thread_id
        if turn_id:
            self.turn_id = turn_id

        if method == "thread/started":
            thread = params.get("thread", {})
            if isinstance(thread, dict) and thread.get("id"):
                self.thread_id = str(thread.get("id"))
            emitted.append(CodexCompileEvent(kind="thread_started", metadata={"thread_id": self.thread_id}))
            self.events.extend(emitted)
            return tuple(emitted)
        if method == "turn/started":
            turn = params.get("turn", {})
            if isinstance(turn, dict) and turn.get("id"):
                self.turn_id = str(turn.get("id"))
            emitted.append(CodexCompileEvent(kind="turn_started", metadata={"turn_id": self.turn_id}))
            self.events.extend(emitted)
            return tuple(emitted)
        if method == "item/started":
            item = params.get("item", {})
            if isinstance(item, dict):
                emitted.append(
                    CodexCompileEvent(
                        kind="item_started",
                        metadata={
                            "item_id": item.get("id"),
                            "item_type": item.get("type"),
                        },
                    )
                )
            self.events.extend(emitted)
            return tuple(emitted)
        if method == "item/agentMessage/delta":
            delta = str(params.get("delta", "") or "")
            self._message_parts.append(delta)
            emitted.append(
                CodexCompileEvent(
                    kind="assistant_delta",
                    message=delta,
                    metadata={"item_id": params.get("itemId")},
                )
            )
            self.events.extend(emitted)
            return tuple(emitted)
        if method == "item/completed":
            item = params.get("item", {})
            if isinstance(item, dict) and str(item.get("type", "")).strip() == "agentMessage":
                text = str(item.get("text", "") or "")
                if text:
                    self.final_message = text
                emitted.append(
                    CodexCompileEvent(
                        kind="assistant_message",
                        message=text,
                        metadata={
                            "item_id": item.get("id"),
                            "phase": item.get("phase"),
                        },
                    )
                )
            self.events.extend(emitted)
            return tuple(emitted)
        if method == "codex/event/error":
            message = "Local Codex app-server returned an error."
            payload = params.get("msg", {})
            if isinstance(payload, dict) and payload.get("message"):
                message = str(payload.get("message"))
            self.error_message = message
            emitted.append(CodexCompileEvent(kind="driver_error", message=message))
            self.events.extend(emitted)
            return tuple(emitted)
        if method == "turn/completed":
            turn = params.get("turn", {})
            metadata = {}
            if isinstance(turn, dict):
                self.turn_id = str(turn.get("id") or self.turn_id or "") or self.turn_id
                metadata["turn_id"] = self.turn_id
                metadata["status"] = turn.get("status")
                if turn.get("error") is not None:
                    self.error_message = str(turn.get("error"))
            self.turn_completed = True
            emitted.append(CodexCompileEvent(kind="turn_completed", metadata=metadata))
            self.events.extend(emitted)
            return tuple(emitted)

        if method:
            emitted.append(
                CodexCompileEvent(
                    kind=_normalize_event_kind(method),
                    message=str(params.get("delta", "") or ""),
                    metadata={
                        "thread_id": self.thread_id,
                        "turn_id": self.turn_id,
                        "item_id": params.get("itemId"),
                        "raw_method": method,
                    },
                )
            )
            self.events.extend(emitted)
        return tuple(emitted)

    def snapshot(self) -> CodexCompileProgress:
        """Return a live progress snapshot for status-store persistence."""

        last_event_kind = self.events[-1].kind if self.events else None
        return CodexCompileProgress(
            thread_id=self.thread_id,
            turn_id=self.turn_id,
            event_count=len(self.events),
            last_event_kind=last_event_kind,
            final_message_seen=bool(self.final_message or "".join(self._message_parts).strip()),
            turn_completed=self.turn_completed,
            error_message=self.error_message,
        )

    def build_result(self) -> CodexCompileRunTranscript:
        final_message = self.final_message or "".join(self._message_parts).strip() or None
        return CodexCompileRunTranscript(
            thread_id=self.thread_id,
            turn_id=self.turn_id,
            final_message=final_message,
            error_message=self.error_message,
            events=tuple(self.events),
        )

    def infer_completion(self, *, reason: str) -> CodexCompileEvent | None:
        """Mark one run as completed when app-server stalls after the final message."""

        if self.turn_completed:
            return None
        self.turn_completed = True
        event = CodexCompileEvent(
            kind="turn_completed_inferred",
            metadata={
                "thread_id": self.thread_id,
                "turn_id": self.turn_id,
                "reason": reason,
            },
        )
        self.events.append(event)
        return event


class CodexAppServerDriver:
    """Run one compile request through `codex app-server` for legacy probing."""

    def __init__(
        self,
        *,
        command: tuple[str, ...] = ("codex", "app-server"),
        timeout_seconds: float = 180.0,
        final_message_grace_seconds: float = 1.5,
        client_name: str = "twinr-self-coding",
        client_version: str = "0.1",
    ) -> None:
        self.command = tuple(command)
        self.timeout_seconds = float(timeout_seconds)
        self.final_message_grace_seconds = max(0.1, float(final_message_grace_seconds))
        self.client_name = client_name
        self.client_version = client_version

    def run_compile(self, request: CodexCompileRequest, *, event_sink=None) -> CodexCompileResult:
        try:
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise CodexDriverUnavailableError("`codex app-server` is unavailable on this machine") from exc

        try:
            return self._run_with_process(process, request, event_sink=event_sink)
        finally:
            self._terminate_process(process)

    def _run_with_process(
        self,
        process: subprocess.Popen[str],
        request: CodexCompileRequest,
        *,
        event_sink=None,
    ) -> CodexCompileResult:
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise CodexDriverUnavailableError("`codex app-server` did not expose stdio pipes")

        collector = CodexAppServerRunCollector()
        request_id = 1
        initialize_id = request_id
        self._send_request(
            process,
            request_id=initialize_id,
            method="initialize",
            params={
                "clientInfo": {"name": self.client_name, "version": self.client_version},
                "capabilities": {"experimentalApi": True},
            },
        )
        request_id += 1

        thread_request_id: int | None = None
        turn_request_id: int | None = None
        completed_turn = False
        stderr_lines: list[str] = []

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        selector.register(process.stderr, selectors.EVENT_READ)
        deadline = time.monotonic() + self.timeout_seconds
        last_stdout_at = time.monotonic()

        while time.monotonic() < deadline:
            if process.poll() is not None and completed_turn:
                break
            for key, _mask in selector.select(timeout=0.25):
                line = key.fileobj.readline()
                if not line:
                    continue
                if key.fileobj is process.stderr:
                    stderr_lines.append(line.rstrip())
                    continue
                last_stdout_at = time.monotonic()
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CodexDriverProtocolError("`codex app-server` returned invalid JSON-RPC output") from exc
                if not isinstance(message, dict):
                    raise CodexDriverProtocolError("`codex app-server` returned a non-object JSON-RPC message")
                emitted = collector.consume(message)
                if event_sink is not None:
                    snapshot = collector.snapshot()
                    for event in emitted:
                        event_sink(event, snapshot)
                if "error" in message:
                    error = message.get("error", {})
                    if isinstance(error, dict):
                        raise CodexDriverUnavailableError(str(error.get("message", "Unknown app-server error")))
                    raise CodexDriverUnavailableError(str(error))
                if collector.error_message:
                    raise CodexDriverUnavailableError(collector.error_message)
                if message.get("id") == initialize_id:
                    thread_request_id = request_id
                    self._send_request(
                        process,
                        request_id=thread_request_id,
                        method="thread/start",
                        params={
                            "cwd": request.workspace_root,
                            "approvalPolicy": "never",
                            "sandbox": "workspace-write",
                            "ephemeral": True,
                            "developerInstructions": (
                                "Return only JSON that matches the requested output schema. "
                                "Do not ask follow-up questions."
                            ),
                        },
                    )
                    request_id += 1
                    continue
                if thread_request_id is not None and message.get("id") == thread_request_id:
                    thread = message.get("result", {}).get("thread", {})
                    thread_id = thread.get("id") if isinstance(thread, dict) else collector.thread_id
                    if not thread_id:
                        raise CodexDriverProtocolError("`thread/start` did not return a thread id")
                    turn_request_id = request_id
                    self._send_request(
                        process,
                        request_id=turn_request_id,
                        method="turn/start",
                        params={
                            "threadId": thread_id,
                            "input": [{"type": "text", "text": request.prompt}],
                            "outputSchema": request.output_schema,
                            "effort": "low",
                            "summary": "none",
                        },
                    )
                    request_id += 1
                    continue
                if message.get("method") == "turn/completed":
                    completed_turn = True
                    break
            if completed_turn:
                break
            snapshot = collector.snapshot()
            if process.poll() is not None and snapshot.final_message_seen and not snapshot.turn_completed:
                inferred_event = collector.infer_completion(reason="process_exited_after_final_message")
                if inferred_event is not None and event_sink is not None:
                    event_sink(inferred_event, collector.snapshot())
                completed_turn = True
                break
            idle_seconds = time.monotonic() - last_stdout_at
            if self._should_infer_completion(
                progress=snapshot,
                idle_seconds=idle_seconds,
                final_message_grace_seconds=self.final_message_grace_seconds,
            ):
                inferred_event = collector.infer_completion(reason="final_message_idle")
                if inferred_event is not None and event_sink is not None:
                    event_sink(inferred_event, collector.snapshot())
                completed_turn = True
                break

        if not completed_turn:
            raise CodexDriverUnavailableError(self._timeout_message(collector=collector, stderr_lines=stderr_lines))

        transcript = collector.build_result()
        if transcript.error_message:
            raise CodexDriverUnavailableError(transcript.error_message)
        if not transcript.final_message:
            raise CodexDriverProtocolError("`codex app-server` completed without a final assistant message")
        return compile_result_from_text(transcript.final_message, events=transcript.events)

    @staticmethod
    def _send_request(
        process: subprocess.Popen[str],
        *,
        request_id: int,
        method: str,
        params: dict[str, Any],
    ) -> None:
        if process.stdin is None:
            raise CodexDriverUnavailableError("`codex app-server` stdin is unavailable")
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()

    @staticmethod
    def _timeout_message(*, collector: CodexAppServerRunCollector, stderr_lines: list[str]) -> str:
        snapshot = collector.snapshot()
        stderr = "\n".join(line for line in stderr_lines if line.strip()).strip()
        details = [
            f"events={snapshot.event_count}",
            f"last_event={snapshot.last_event_kind or 'none'}",
            f"thread_id={snapshot.thread_id or 'none'}",
            f"turn_id={snapshot.turn_id or 'none'}",
            f"final_message_seen={'yes' if snapshot.final_message_seen else 'no'}",
            f"turn_completed={'yes' if snapshot.turn_completed else 'no'}",
        ]
        if stderr:
            details.append(f"stderr={stderr}")
        return "`codex app-server` timed out before turn completion (" + ", ".join(details) + ")"

    @staticmethod
    def _should_infer_completion(
        *,
        progress: CodexCompileProgress,
        idle_seconds: float,
        final_message_grace_seconds: float,
    ) -> bool:
        return (
            progress.final_message_seen
            and not progress.turn_completed
            and not progress.error_message
            and idle_seconds >= max(0.0, final_message_grace_seconds)
        )

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)


def _normalize_event_kind(method: str) -> str:
    return str(method or "").strip().lower().replace("/", "_").replace(".", "_")
