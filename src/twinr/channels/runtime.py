"""Convert normalized external text messages into one Twinr runtime turn.

This module intentionally stays transport-agnostic. Channel listeners hand it a
``ChannelInboundMessage`` and receive a ``ChannelOutboundMessage`` back without
needing to know about Twinr's internal turn orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from threading import RLock

from twinr.agent.base_agent.contracts import AgentTextProvider
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.agent.base_agent.state.machine import TwinrStatus

from .contracts import ChannelInboundMessage, ChannelOutboundMessage


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TwinrTextChannelTurnService:
    """Handle one text turn for an external channel with serialized runtime use."""

    runtime: TwinrRuntime = field(repr=False)
    backend: AgentTextProvider = field(repr=False)
    allow_web_search: bool | None = None
    busy_reply_text: str = "Ich bearbeite gerade schon etwas. Schreib bitte gleich noch einmal."
    error_reply_text: str = "Ich hatte gerade ein internes Problem. Versuch es bitte noch einmal."
    _lock: RLock = field(init=False, repr=False, default_factory=RLock)

    def handle_inbound(self, message: ChannelInboundMessage) -> ChannelOutboundMessage:
        """Run one inbound text message through the Twinr runtime and provider."""

        with self._lock:
            self._prepare_runtime()
            if self.runtime.status != TwinrStatus.WAITING:
                return self._reply_for(message, self.busy_reply_text)

            try:
                self.runtime.begin_listening(request_source=message.channel)
                self.runtime.submit_transcript(message.text)
                response = self.backend.respond_with_metadata(
                    message.text,
                    conversation=self.runtime.provider_conversation_context(),
                    allow_web_search=self.allow_web_search,
                )
                answer = self.runtime.complete_agent_turn(response.text)
                self.runtime.finish_speaking()
                metadata: dict[str, str] = {}
                if getattr(response, "response_id", None):
                    metadata["provider_response_id"] = str(response.response_id)
                if getattr(response, "request_id", None):
                    metadata["provider_request_id"] = str(response.request_id)
                if getattr(response, "model", None):
                    metadata["provider_model"] = str(response.model)
                return self._reply_for(message, answer, metadata=metadata)
            except Exception as exc:
                LOGGER.exception("Failed to process %s channel message %s.", message.channel, message.message_id)
                self.runtime.fail(str(exc))
                return self._reply_for(message, self.error_reply_text)

    def _prepare_runtime(self) -> None:
        """Reset a stale error state before the next external text turn starts."""

        if self.runtime.status == TwinrStatus.ERROR:
            self.runtime.reset_error()

    @staticmethod
    def _reply_for(
        message: ChannelInboundMessage,
        text: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> ChannelOutboundMessage:
        """Build one outbound reply addressed back to the inbound sender."""

        return ChannelOutboundMessage(
            channel=message.channel,
            conversation_id=message.conversation_id,
            recipient_id=message.sender_id,
            text=text,
            reply_to_message_id=message.message_id,
            metadata=metadata or {},
        )
