"""Run the long-lived WhatsApp listener loop backed by a Baileys worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
import math
import time
from typing import Protocol
from uuid import uuid4

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import AgentTextProvider, ToolCallingAgentProvider
from twinr.agent.base_agent.runtime.runtime import TwinrRuntime  # pylint: disable=no-name-in-module
from twinr.channels.contracts import ChannelInboundMessage, ChannelOutboundMessage
from twinr.channels.runtime import TwinrTextChannelTurnService
from twinr.integrations import (
    SOCIAL_HISTORY_LEARNING_INTEGRATION_ID,
    TwinrIntegrationStore,
    social_history_record_with_import_state,
)

from .config import WhatsAppChannelConfig
from .history_import import WhatsAppHistoryImportQueue, WhatsAppHistoryImportResult
from .history_learning import WhatsAppHistoryLearningService
from .inbound_event_log import WhatsAppInboundEventLog
from .outbound import WhatsAppOutboundQueue, WhatsAppOutboundResult
from .policy import WhatsAppMessagePolicy
from .worker_bridge import (
    WhatsAppWorkerBridge,
    WhatsAppWorkerExitedError,
    WhatsAppWorkerStatusEvent,
)


LOGGER = logging.getLogger(__name__)


class _ChannelTurnService(Protocol):
    def handle_inbound(self, message: ChannelInboundMessage) -> "ChannelOutboundMessage":
        ...


@dataclass(slots=True)
class TwinrWhatsAppChannelLoop:
    """Listen for allowlisted WhatsApp messages and answer through Twinr."""

    config: TwinrConfig = field(repr=False)
    runtime: TwinrRuntime = field(repr=False)
    backend: AgentTextProvider = field(repr=False)
    tool_agent_provider: ToolCallingAgentProvider | None = field(default=None, repr=False)
    print_backend: AgentTextProvider | None = field(default=None, repr=False)
    whatsapp_config: WhatsAppChannelConfig = field(init=False, repr=False)
    policy: WhatsAppMessagePolicy = field(init=False, repr=False)
    transport: WhatsAppWorkerBridge = field(init=False, repr=False)
    turn_service: _ChannelTurnService = field(init=False, repr=False)
    outbound_queue: WhatsAppOutboundQueue = field(init=False, repr=False)
    history_import_queue: WhatsAppHistoryImportQueue = field(init=False, repr=False)
    inbound_event_log: WhatsAppInboundEventLog = field(init=False, repr=False)
    _account_jid: str | None = field(init=False, repr=False, default=None)
    _transport_open: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        self.whatsapp_config = WhatsAppChannelConfig.from_twinr_config(self.config)
        self.policy = WhatsAppMessagePolicy(self.whatsapp_config)
        self.transport = WhatsAppWorkerBridge(self.whatsapp_config)
        if self.tool_agent_provider is not None and self.print_backend is not None:
            from twinr.channels.tool_runtime import TwinrToolTextChannelTurnService

            self.turn_service = TwinrToolTextChannelTurnService(
                runtime=self.runtime,
                tool_agent_provider=self.tool_agent_provider,
                print_backend=self.print_backend,
                whatsapp_outbound_dispatcher=self._dispatch_in_channel_outbound_message,
            )
        else:
            self.turn_service = TwinrTextChannelTurnService(runtime=self.runtime, backend=self.backend)
        self.outbound_queue = WhatsAppOutboundQueue.from_whatsapp_config(self.whatsapp_config)
        self.history_import_queue = WhatsAppHistoryImportQueue.from_whatsapp_config(self.whatsapp_config)
        self.inbound_event_log = WhatsAppInboundEventLog(self.whatsapp_config.auth_dir.parent / "debug")

    def run(self, *, duration_s: float | None = None) -> int:
        """Run the listener until interrupted, fatal logout, or optional timeout."""

        deadline = self._deadline(duration_s)
        restart_delay_s = self.whatsapp_config.reconnect_base_delay_s
        self.outbound_queue.fail_stale_processing_requests()
        self.history_import_queue.fail_stale_processing_requests()
        if not self._start_transport():
            return 1
        try:
            while True:
                remaining = self._remaining_seconds(deadline)
                if remaining is not None and remaining <= 0.0:
                    return 0
                try:
                    event = self.transport.next_event(timeout_s=self._poll_timeout(remaining))
                except WhatsAppWorkerExitedError as exc:
                    LOGGER.warning("WhatsApp worker exited unexpectedly: %s", exc)
                    self._transport_open = False
                    if remaining is not None and remaining <= restart_delay_s:
                        return 1
                    time.sleep(restart_delay_s)
                    restart_delay_s = min(
                        restart_delay_s * 2.0,
                        self.whatsapp_config.reconnect_max_delay_s,
                    )
                    if not self._start_transport():
                        return 1
                    continue

                if event is None:
                    while self._process_history_import_queue_once():
                        pass
                    while self._process_outbound_queue_once():
                        pass
                    continue
                if isinstance(event, WhatsAppWorkerStatusEvent):
                    self._handle_status_event(event)
                    if event.fatal:
                        return 1
                    if event.connection == "open":
                        restart_delay_s = self.whatsapp_config.reconnect_base_delay_s
                    self._process_history_import_queue_once()
                    self._process_outbound_queue_once()
                    continue
                if not isinstance(event, ChannelInboundMessage):
                    LOGGER.debug("Ignoring unexpected WhatsApp worker event outside history mode: %r", event)
                    self._process_history_import_queue_once()
                    self._process_outbound_queue_once()
                    continue

                self.inbound_event_log.record_received(event)
                decision = self.policy.evaluate(event, account_jid=self._account_jid)
                self.inbound_event_log.record_policy_decision(
                    event,
                    accepted=decision.accepted,
                    reason=decision.reason,
                )
                if not decision.accepted:
                    LOGGER.info(
                        "Ignoring WhatsApp message %s from %s: %s",
                        event.message_id,
                        event.sender_id,
                        decision.reason,
                    )
                    self._process_history_import_queue_once()
                    self._process_outbound_queue_once()
                    continue

                LOGGER.info(
                    "Processing WhatsApp message %s from %s in %s.",
                    event.message_id,
                    event.sender_id,
                    event.conversation_id,
                )
                reply = self.turn_service.handle_inbound(event)
                try:
                    send_result = self.transport.send_text(
                        chat_jid=reply.conversation_id,
                        text=reply.text,
                        reply_to_message_id=reply.reply_to_message_id,
                    )
                    self.policy.remember_outbound(send_result.message_id)
                    self.inbound_event_log.record_delivery(
                        event,
                        outbound_message_id=send_result.message_id,
                        reply_text=reply.text,
                    )
                    LOGGER.info(
                        "Delivered WhatsApp reply %s for inbound %s.",
                        send_result.message_id,
                        event.message_id,
                    )
                except Exception as exc:
                    LOGGER.exception("Failed to deliver WhatsApp reply for %s.", event.message_id)
                    self.inbound_event_log.record_delivery_error(event, error=str(exc))
                    self.runtime.fail(f"WhatsApp delivery failed: {exc}")
                self._process_history_import_queue_once()
                self._process_outbound_queue_once()
        finally:
            self.transport.stop()

    def _start_transport(self) -> bool:
        """Start the worker once and fail the runtime state clearly on hard startup errors."""

        try:
            self.transport.start()
        except Exception as exc:
            LOGGER.exception("WhatsApp worker startup failed.")
            self.runtime.fail(f"WhatsApp startup failed: {exc}")
            return False
        self._transport_open = False
        return True

    def _handle_status_event(self, event: WhatsAppWorkerStatusEvent) -> None:
        if event.account_jid:
            self._account_jid = event.account_jid
        self._transport_open = event.connection == "open" and not event.fatal
        if event.qr_available:
            LOGGER.info("WhatsApp worker emitted a new pairing QR.")
        if event.fatal:
            LOGGER.error(
                "WhatsApp worker entered a fatal state: connection=%s detail=%s status_code=%s",
                event.connection,
                event.detail,
                event.status_code,
            )
            return
        LOGGER.info(
            "WhatsApp worker status: connection=%s detail=%s account_jid=%s",
            event.connection,
            event.detail,
            event.account_jid,
        )

    def _dispatch_in_channel_outbound_message(
        self,
        *,
        chat_jid: str,
        text: str,
        recipient_label: str,
        reply_to_message_id: str | None = None,
        wait_timeout_s: float | None = None,
    ) -> WhatsAppOutboundResult:
        """Send immediately when the active WhatsApp loop itself owns the turn.

        The on-disk outbound queue is required for cross-process callers. An
        inbound WhatsApp turn running inside this same loop must instead use
        the already-open worker transport directly, otherwise it would enqueue
        work and block while waiting for the very loop that is currently busy.
        """

        del wait_timeout_s
        request_id = f"direct-{uuid4().hex}"
        if not self._transport_open:
            return WhatsAppOutboundResult.failed(
                request_id=request_id,
                error_code="channel_not_ready",
                error="The running WhatsApp channel is not connected right now.",
            )
        try:
            send_result = self.transport.send_text(
                chat_jid=chat_jid,
                text=text,
                reply_to_message_id=reply_to_message_id,
            )
            self.policy.remember_outbound(send_result.message_id)
            LOGGER.info(
                "Delivered direct in-channel WhatsApp outbound %s to %s (%s).",
                request_id,
                chat_jid,
                recipient_label,
            )
            return WhatsAppOutboundResult.sent(
                request_id=request_id,
                message_id=send_result.message_id,
            )
        except Exception as exc:
            LOGGER.exception(
                "Failed direct in-channel WhatsApp outbound %s to %s (%s).",
                request_id,
                chat_jid,
                recipient_label,
            )
            return WhatsAppOutboundResult.failed(
                request_id=request_id,
                error_code="delivery_failed",
                error=str(exc),
            )

    def _process_outbound_queue_once(self) -> bool:
        """Claim and deliver at most one queued outbound WhatsApp message."""

        if not self._transport_open:
            return False

        request = self.outbound_queue.claim_next_request()
        if request is None:
            return False
        if request.is_expired():
            self.outbound_queue.complete_request(
                WhatsAppOutboundResult.failed(
                    request_id=request.request_id,
                    error_code="expired",
                    error="The outbound WhatsApp send request expired before delivery.",
                )
            )
            return True

        try:
            send_result = self.transport.send_text(
                chat_jid=request.chat_jid,
                text=request.text,
                reply_to_message_id=request.reply_to_message_id,
            )
            self.policy.remember_outbound(send_result.message_id)
            self.outbound_queue.complete_request(
                WhatsAppOutboundResult.sent(
                    request_id=request.request_id,
                    message_id=send_result.message_id,
                )
            )
            LOGGER.info(
                "Delivered queued WhatsApp outbound %s to %s.",
                request.request_id,
                request.chat_jid,
            )
        except Exception as exc:
            LOGGER.exception("Failed to deliver queued WhatsApp outbound %s.", request.request_id)
            self.outbound_queue.complete_request(
                WhatsAppOutboundResult.failed(
                    request_id=request.request_id,
                    error_code="delivery_failed",
                    error=str(exc),
                )
            )
        return True

    def _process_history_import_queue_once(self) -> bool:
        """Claim and execute at most one queued WhatsApp history import."""

        if not self._transport_open:
            return False

        request = self.history_import_queue.claim_next_request()
        if request is None:
            return False
        integration_store = TwinrIntegrationStore.from_project_root(self.config.project_root)
        current_record = integration_store.get(SOCIAL_HISTORY_LEARNING_INTEGRATION_ID)
        if request.is_expired():
            failed = WhatsAppHistoryImportResult.failed(
                request_id=request.request_id,
                error_code="expired",
                error="The queued WhatsApp history import expired before the channel service could start it.",
            )
            self.history_import_queue.complete_request(failed)
            integration_store.save(
                social_history_record_with_import_state(
                    current_record,
                    status="failed",
                    request_id=request.request_id,
                    finished_at=failed.completed_at,
                    error=failed.error,
                )
            )
            return True

        started_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        integration_store.save(
            social_history_record_with_import_state(
                current_record,
                status="running",
                request_id=request.request_id,
                started_at=started_at,
                detail="Twinr is importing approved WhatsApp history into shared memory.",
            )
        )
        self.transport.stop()
        self._transport_open = False
        try:
            memory_service = getattr(self.runtime, "long_term_memory", None)
            if memory_service is None:
                raise RuntimeError("Twinr long-term memory is unavailable for WhatsApp history import.")
            summary = WhatsAppHistoryLearningService(
                whatsapp_config=self.whatsapp_config,
                memory_service=memory_service,
            ).run_import(lookback_key=request.lookback_key)
            result = WhatsAppHistoryImportResult.completed(
                request_id=request.request_id,
                status=summary.status,
                detail=summary.detail,
                imported_messages=summary.imported_messages,
                imported_turns=summary.imported_turns,
                imported_chats=summary.imported_chats,
                oldest_at=summary.oldest_at,
                newest_at=summary.newest_at,
            )
            integration_store.save(
                social_history_record_with_import_state(
                    integration_store.get(SOCIAL_HISTORY_LEARNING_INTEGRATION_ID),
                    status=summary.status,
                    request_id=request.request_id,
                    started_at=started_at,
                    finished_at=result.completed_at,
                    detail=summary.detail,
                    messages=summary.imported_messages,
                    turns=summary.imported_turns,
                    chats=summary.imported_chats,
                    oldest_at=summary.oldest_at,
                    newest_at=summary.newest_at,
                )
            )
            LOGGER.info(
                "Imported WhatsApp history request %s: %s messages, %s turns, %s chats.",
                request.request_id,
                summary.imported_messages,
                summary.imported_turns,
                summary.imported_chats,
            )
        except Exception as exc:
            LOGGER.exception("WhatsApp history import %s failed.", request.request_id)
            result = WhatsAppHistoryImportResult.failed(
                request_id=request.request_id,
                error_code="import_failed",
                error=str(exc),
            )
            integration_store.save(
                social_history_record_with_import_state(
                    integration_store.get(SOCIAL_HISTORY_LEARNING_INTEGRATION_ID),
                    status="failed",
                    request_id=request.request_id,
                    started_at=started_at,
                    finished_at=result.completed_at,
                    error=result.error,
                )
            )
        finally:
            self.history_import_queue.complete_request(result)
            self._start_transport()
        return True

    @staticmethod
    def _deadline(duration_s: float | None) -> float | None:
        if duration_s is None:
            return None
        timeout = float(duration_s)
        if not math.isfinite(timeout) or timeout < 0.0:
            return None
        return time.monotonic() + timeout

    @staticmethod
    def _remaining_seconds(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return deadline - time.monotonic()

    @staticmethod
    def _poll_timeout(remaining: float | None) -> float:
        if remaining is None:
            return 1.0
        return min(1.0, max(0.05, remaining))
