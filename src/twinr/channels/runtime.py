"""Convert normalized external text messages into one Twinr runtime turn.

This module intentionally stays transport-agnostic. Channel listeners hand it a
``ChannelInboundMessage`` and receive a ``ChannelOutboundMessage`` back without
needing to know about Twinr's internal turn orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from threading import RLock
import time
from uuid import uuid4

from twinr.agent.base_agent.contracts import AgentTextProvider
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.agent.base_agent.state.machine import TwinrStatus
from twinr.agent.workflows.forensics import (
    WorkflowForensics,
    bind_workflow_forensics,
    workflow_decision,
    workflow_event,
    workflow_span,
)

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
    _workflow_forensics: WorkflowForensics = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Synchronously warm remote-backed memory caches before live traffic."""

        project_root = Path(
            getattr(
                getattr(self.runtime, "config", None),
                "project_root",
                Path(__file__).resolve().parents[3],
            )
        ).expanduser().resolve(strict=False)
        self._workflow_forensics = WorkflowForensics.from_env(
            project_root=project_root,
            service=self.__class__.__name__,
        )
        warmup_callback = getattr(getattr(self.runtime, "long_term_memory", None), "prewarm_foreground_read_cache", None)
        if not callable(warmup_callback):
            return
        try:
            with bind_workflow_forensics(self._workflow_forensics, trace_id=uuid4().hex):
                with workflow_span(
                    name="text_channel_prewarm_foreground_read_cache",
                    kind="cache",
                    details={"channel_service": self.__class__.__name__},
                ):
                    warmup_callback()
        except Exception:
            LOGGER.exception("Text-channel warmup failed.")
            raise

    def handle_inbound(self, message: ChannelInboundMessage) -> ChannelOutboundMessage:
        """Run one inbound text message through the Twinr runtime and provider."""

        turn_started = time.monotonic()
        trace_id = uuid4().hex
        with bind_workflow_forensics(self._workflow_forensics, trace_id=trace_id):
            workflow_event(
                kind="turn_start",
                msg="text_channel_turn_started",
                details=self._message_trace_details(message),
            )
            workflow_decision(
                msg="text_channel_web_search_policy",
                question="Should this text-channel turn allow provider-side web search?",
                selected={
                    "id": "inherit_default" if self.allow_web_search is None else str(bool(self.allow_web_search)).lower(),
                    "summary": "Use the configured text-channel web-search policy.",
                },
                options=[
                    {"id": "inherit_default", "summary": "Defer to backend/runtime default"},
                    {"id": "true", "summary": "Force web search on"},
                    {"id": "false", "summary": "Force web search off"},
                ],
                context={"allow_web_search": self.allow_web_search},
                guardrails=["channel_turn_policy"],
            )
            with workflow_span(
                name="text_channel_handle_inbound",
                kind="turn",
                details=self._message_trace_details(message),
            ):
                with self._lock:
                    with workflow_span(name="text_channel_prepare_runtime", kind="mutation"):
                        self._prepare_runtime()
                    if self.runtime.status != TwinrStatus.WAITING:
                        workflow_event(
                            kind="branch",
                            msg="text_channel_busy_reply_returned",
                            details={"runtime_status": getattr(self.runtime.status, "value", str(self.runtime.status))},
                        )
                        return self._reply_for(message, self.busy_reply_text)

                    try:
                        with workflow_span(
                            name="text_channel_begin_listening",
                            kind="mutation",
                            details={"request_source": message.channel},
                        ):
                            self.runtime.begin_listening(request_source=message.channel)
                        with workflow_span(name="text_channel_submit_transcript", kind="mutation"):
                            self.runtime.submit_transcript(message.text)
                        with workflow_span(
                            name="text_channel_provider_context",
                            kind="retrieval",
                            details={"query_chars": len((message.text or "").strip())},
                        ):
                            conversation = self.runtime.provider_conversation_context()
                        with workflow_span(
                            name="text_channel_backend_respond",
                            kind="llm_call",
                            details={
                                "conversation_messages": len(conversation),
                                "allow_web_search": self.allow_web_search,
                            },
                        ):
                            response = self.backend.respond_with_metadata(
                                message.text,
                                conversation=conversation,
                                allow_web_search=self.allow_web_search,
                            )
                        with workflow_span(name="text_channel_complete_agent_turn", kind="mutation"):
                            answer = self.runtime.complete_agent_turn(
                                response.text,
                                source=message.channel,
                                modality="text",
                            )
                        with workflow_span(name="text_channel_finish_speaking", kind="mutation"):
                            self.runtime.finish_speaking()
                        metadata: dict[str, str] = {}
                        if getattr(response, "response_id", None):
                            metadata["provider_response_id"] = str(response.response_id)
                        if getattr(response, "request_id", None):
                            metadata["provider_request_id"] = str(response.request_id)
                        if getattr(response, "model", None):
                            metadata["provider_model"] = str(response.model)
                        if getattr(response, "used_web_search", None) is not None:
                            metadata["provider_used_web_search"] = str(bool(response.used_web_search)).lower()
                        workflow_event(
                            kind="turn_end",
                            msg="text_channel_turn_completed",
                            details={
                                "runtime_status": getattr(self.runtime.status, "value", str(self.runtime.status)),
                                "provider_model": getattr(response, "model", None),
                                "response_chars": len(answer),
                            },
                            kpi={"duration_ms": round(max(0.0, (time.monotonic() - turn_started) * 1000.0), 3)},
                        )
                        return self._reply_for(message, answer, metadata=metadata)
                    except Exception as exc:
                        workflow_event(
                            kind="exception",
                            msg="text_channel_turn_failed",
                            details={"error_type": type(exc).__name__},
                            level="ERROR",
                            kpi={"duration_ms": round(max(0.0, (time.monotonic() - turn_started) * 1000.0), 3)},
                        )
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

    @staticmethod
    def _message_trace_details(message: ChannelInboundMessage) -> dict[str, object]:
        """Summarize one inbound message without storing raw user text."""

        normalized_text = " ".join((message.text or "").split()).strip()
        return {
            "channel": message.channel,
            "message_id": message.message_id,
            "conversation_id": message.conversation_id,
            "sender_id": message.sender_id,
            "text_chars": len(normalized_text),
            "text_words": len(normalized_text.split()) if normalized_text else 0,
            "is_group": bool(getattr(message, "is_group", False)),
            "is_from_self": bool(getattr(message, "is_from_self", False)),
        }
