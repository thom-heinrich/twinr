"""Bridge the Python runtime to the Baileys worker over JSONL stdio."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import uuid

from twinr.channels.contracts import ChannelInboundMessage, ChannelTransportError

from .config import WhatsAppChannelConfig
from .worker_dependencies import ensure_whatsapp_worker_dependencies


LOGGER = logging.getLogger(__name__)
_NODE_VERSION_PATTERN = re.compile(r"v?(?P<major>\d+)")


@dataclass(frozen=True, slots=True)
class WhatsAppWorkerStatusEvent:
    """Represent one status or fatal event emitted by the Baileys worker."""

    connection: str
    detail: str | None = None
    account_jid: str | None = None
    status_code: int | None = None
    reconnect_in_ms: int | None = None
    qr_available: bool = False
    qr_svg: str | None = None
    fatal: bool = False


@dataclass(frozen=True, slots=True)
class WhatsAppWorkerSendResult:
    """Represent the result of one worker ``send_text`` command."""

    request_id: str
    ok: bool
    message_id: str | None = None
    error: str | None = None


class WhatsAppWorkerExitedError(ChannelTransportError):
    """Raise when the Baileys worker exits while the channel loop is active."""

    def __init__(self, exit_code: int | None) -> None:
        self.exit_code = exit_code
        super().__init__(f"WhatsApp worker exited with code {exit_code}")


class WhatsAppWorkerBridge:
    """Own the Baileys worker process and expose typed inbound events."""

    def __init__(self, config: WhatsAppChannelConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._event_queue: queue.Queue[ChannelInboundMessage | WhatsAppWorkerStatusEvent] = queue.Queue()
        self._pending_results: dict[str, queue.Queue[WhatsAppWorkerSendResult]] = {}
        self._lock = threading.RLock()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stop_requested = False

    @property
    def worker_entry(self) -> Path:
        """Return the absolute path to the Node worker entrypoint."""

        return self.config.worker_root / "index.mjs"

    def start(self) -> None:
        """Launch the worker when it is not already running."""

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            self._assert_supported_node_runtime()
            ensure_whatsapp_worker_dependencies(
                worker_root=self.config.worker_root,
                node_binary=self.config.node_binary,
            )
            self._stop_requested = False
            self.config.auth_dir.mkdir(parents=True, exist_ok=True)
            self._event_queue = queue.Queue()
            self._pending_results = {}
            environment = {
                **dict(os.environ),
                "TWINR_WHATSAPP_AUTH_DIR": str(self.config.auth_dir),
                "TWINR_WHATSAPP_RECONNECT_BASE_MS": str(int(self.config.reconnect_base_delay_s * 1000)),
                "TWINR_WHATSAPP_RECONNECT_MAX_MS": str(int(self.config.reconnect_max_delay_s * 1000)),
            }
            self._process = subprocess.Popen(
                [self.config.node_binary, str(self.worker_entry)],
                cwd=str(self.config.worker_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=environment,
            )
            self._stdout_thread = threading.Thread(target=self._pump_stdout, name="whatsapp-worker-stdout", daemon=True)
            self._stderr_thread = threading.Thread(target=self._pump_stderr, name="whatsapp-worker-stderr", daemon=True)
            self._stdout_thread.start()
            self._stderr_thread.start()

    def _assert_supported_node_runtime(self) -> None:
        """Fail closed when the configured Node runtime is too old for Baileys."""

        try:
            completed = subprocess.run(
                [self.config.node_binary, "--version"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except OSError as exc:
            raise ChannelTransportError(
                f"Could not execute WhatsApp node runtime {self.config.node_binary!r}: {exc}"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise ChannelTransportError(
                f"Could not query WhatsApp node runtime {self.config.node_binary!r}: {detail or completed.returncode}"
            )
        version_text = (completed.stdout or completed.stderr or "").strip()
        match = _NODE_VERSION_PATTERN.search(version_text)
        if match is None:
            raise ChannelTransportError(f"Could not parse node version from {version_text!r}")
        major = int(match.group("major"))
        if major < 20:
            raise ChannelTransportError(
                "The WhatsApp Baileys worker requires Node.js 20+; "
                f"{self.config.node_binary!r} reported {version_text!r}"
            )

    def stop(self) -> None:
        """Stop the worker process and clear any pending waiters."""

        with self._lock:
            self._stop_requested = True
            process = self._process
            self._process = None
        if process is None:
            return
        try:
            self._write_command({"type": "shutdown"}, process=process)
        except Exception:
            LOGGER.debug("WhatsApp worker did not accept shutdown command; terminating.", exc_info=True)
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        finally:
            self._release_pending_results("worker stopped")

    def next_event(self, *, timeout_s: float | None = None) -> ChannelInboundMessage | WhatsAppWorkerStatusEvent | None:
        """Return the next typed worker event or ``None`` on timeout."""

        try:
            return self._event_queue.get(timeout=timeout_s)
        except queue.Empty:
            process = self._process
            if process is not None and process.poll() is not None and not self._stop_requested:
                raise WhatsAppWorkerExitedError(process.returncode)
            return None

    def send_text(
        self,
        *,
        chat_jid: str,
        text: str,
        reply_to_message_id: str | None = None,
    ) -> WhatsAppWorkerSendResult:
        """Ask the worker to send one text message into the given WhatsApp chat."""

        process = self._require_process()
        request_id = uuid.uuid4().hex
        waiter: queue.Queue[WhatsAppWorkerSendResult] = queue.Queue(maxsize=1)
        with self._lock:
            self._pending_results[request_id] = waiter
        try:
            self._write_command(
                {
                    "type": "send_text",
                    "request_id": request_id,
                    "chat_jid": chat_jid,
                    "text": text,
                    "reply_to_message_id": reply_to_message_id,
                },
                process=process,
            )
            result = waiter.get(timeout=self.config.send_timeout_s)
        except queue.Empty as exc:
            raise ChannelTransportError("Timed out while waiting for WhatsApp worker send result") from exc
        finally:
            with self._lock:
                self._pending_results.pop(request_id, None)

        if not result.ok:
            raise ChannelTransportError(result.error or "WhatsApp worker send failed")
        return result

    def _require_process(self) -> subprocess.Popen[str]:
        process = self._process
        if process is None or process.poll() is not None:
            raise ChannelTransportError("WhatsApp worker is not running")
        return process

    def _write_command(self, payload: dict[str, object], *, process: subprocess.Popen[str]) -> None:
        stdin = process.stdin
        if stdin is None:
            raise ChannelTransportError("WhatsApp worker stdin is unavailable")
        stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
        stdin.flush()

    def _pump_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Ignoring malformed WhatsApp worker JSON line: %s", line)
                continue
            self._dispatch_worker_payload(payload)
        if not self._stop_requested:
            self._release_pending_results("worker stdout closed")

    def _pump_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            sys.stderr.write(line)
        sys.stderr.flush()

    def _dispatch_worker_payload(self, payload: dict[str, object]) -> None:
        payload_type = str(payload.get("type", "") or "").strip()
        if payload_type == "incoming_message":
            message_id = str(payload.get("message_id", "") or "").strip()
            conversation_id = str(payload.get("conversation_id", "") or "").strip()
            sender_id = str(payload.get("sender_id", "") or "").strip()
            text = str(payload.get("text", "") or "").strip()
            if not all((message_id, conversation_id, sender_id, text)):
                LOGGER.warning("Ignoring incomplete WhatsApp worker message payload: %r", payload)
                return
            message = ChannelInboundMessage(
                channel="whatsapp",
                message_id=message_id,
                conversation_id=conversation_id,
                sender_id=sender_id,
                text=text,
                sender_display_name=self._optional_string(payload.get("sender_display_name")),
                received_at=self._optional_string(payload.get("received_at")),
                is_group=bool(payload.get("is_group", False)),
                is_from_self=bool(payload.get("is_from_self", False)),
                metadata={
                    "account_jid": self._optional_string(payload.get("account_jid")) or "",
                },
            )
            self._event_queue.put(message)
            return

        if payload_type in {"status", "fatal", "worker_ready"}:
            event = WhatsAppWorkerStatusEvent(
                connection=str(payload.get("connection", payload_type) or payload_type),
                detail=self._optional_string(payload.get("detail")),
                account_jid=self._optional_string(payload.get("account_jid")),
                status_code=self._optional_int(payload.get("status_code")),
                reconnect_in_ms=self._optional_int(payload.get("reconnect_in_ms")),
                qr_available=bool(payload.get("qr_available", False)),
                qr_svg=self._optional_string(payload.get("qr_svg")),
                fatal=payload_type == "fatal" or bool(payload.get("fatal", False)),
            )
            self._event_queue.put(event)
            return

        if payload_type == "send_result":
            request_id = str(payload.get("request_id", "") or "").strip()
            if not request_id:
                LOGGER.warning("Ignoring WhatsApp worker send_result without request_id: %r", payload)
                return
            result = WhatsAppWorkerSendResult(
                request_id=request_id,
                ok=bool(payload.get("ok", False)),
                message_id=self._optional_string(payload.get("message_id")),
                error=self._optional_string(payload.get("error")),
            )
            with self._lock:
                waiter = self._pending_results.get(request_id)
            if waiter is not None:
                waiter.put(result)
            return

        LOGGER.debug("Ignoring unsupported WhatsApp worker payload type %r.", payload_type)

    def _release_pending_results(self, reason: str) -> None:
        result = WhatsAppWorkerSendResult(
            request_id="worker_shutdown",
            ok=False,
            error=reason,
        )
        with self._lock:
            pending = list(self._pending_results.values())
            self._pending_results.clear()
        for waiter in pending:
            try:
                waiter.put_nowait(result)
            except queue.Full:
                continue

    @staticmethod
    def _optional_string(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
