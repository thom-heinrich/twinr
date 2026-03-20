"""Run the long-lived WhatsApp listener loop backed by a Baileys worker."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import AgentTextProvider
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.channels.runtime import TwinrTextChannelTurnService

from .config import WhatsAppChannelConfig
from .policy import WhatsAppMessagePolicy
from .worker_bridge import (
    WhatsAppWorkerBridge,
    WhatsAppWorkerExitedError,
    WhatsAppWorkerStatusEvent,
)


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TwinrWhatsAppChannelLoop:
    """Listen for allowlisted WhatsApp messages and answer through Twinr."""

    config: TwinrConfig = field(repr=False)
    runtime: TwinrRuntime = field(repr=False)
    backend: AgentTextProvider = field(repr=False)
    whatsapp_config: WhatsAppChannelConfig = field(init=False, repr=False)
    policy: WhatsAppMessagePolicy = field(init=False, repr=False)
    transport: WhatsAppWorkerBridge = field(init=False, repr=False)
    turn_service: TwinrTextChannelTurnService = field(init=False, repr=False)
    _account_jid: str | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.whatsapp_config = WhatsAppChannelConfig.from_twinr_config(self.config)
        self.policy = WhatsAppMessagePolicy(self.whatsapp_config)
        self.transport = WhatsAppWorkerBridge(self.whatsapp_config)
        self.turn_service = TwinrTextChannelTurnService(runtime=self.runtime, backend=self.backend)

    def run(self, *, duration_s: float | None = None) -> int:
        """Run the listener until interrupted, fatal logout, or optional timeout."""

        deadline = self._deadline(duration_s)
        restart_delay_s = self.whatsapp_config.reconnect_base_delay_s
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
                    continue
                if isinstance(event, WhatsAppWorkerStatusEvent):
                    self._handle_status_event(event)
                    if event.fatal:
                        return 1
                    if event.connection == "open":
                        restart_delay_s = self.whatsapp_config.reconnect_base_delay_s
                    continue

                decision = self.policy.evaluate(event, account_jid=self._account_jid)
                if not decision.accepted:
                    LOGGER.info(
                        "Ignoring WhatsApp message %s from %s: %s",
                        event.message_id,
                        event.sender_id,
                        decision.reason,
                    )
                    continue

                reply = self.turn_service.handle_inbound(event)
                try:
                    send_result = self.transport.send_text(
                        chat_jid=reply.conversation_id,
                        text=reply.text,
                        reply_to_message_id=reply.reply_to_message_id,
                    )
                    self.policy.remember_outbound(send_result.message_id)
                except Exception as exc:
                    LOGGER.exception("Failed to deliver WhatsApp reply for %s.", event.message_id)
                    self.runtime.fail(f"WhatsApp delivery failed: {exc}")
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
        return True

    def _handle_status_event(self, event: WhatsAppWorkerStatusEvent) -> None:
        if event.account_jid:
            self._account_jid = event.account_jid
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
