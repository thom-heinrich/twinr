"""Run external text-channel turns through Twinr's shared tool-calling path.

This module keeps long-lived messaging channels like WhatsApp on the same
tool-capable Twinr agent lane that voice and operator text surfaces already
use, while remaining transport-agnostic at the channel boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
from threading import RLock
import time
from typing import cast
from uuid import uuid4

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.conversation.thread_resolution import (
    focus_recent_thread_conversation,
    maybe_rewrite_prompt_against_recent_thread,
)
from twinr.agent.base_agent.contracts import AgentTextProvider, ToolCallingAgentProvider
from twinr.agent.base_agent.runtime.runtime import TwinrRuntime  # pylint: disable=no-name-in-module
from twinr.agent.base_agent.state.machine import TwinrStatus
from twinr.agent.tools.runtime.executor import RealtimeToolExecutor
from twinr.agent.tools.runtime.streaming_loop import ToolCallingStreamingLoop, ToolHandler
from twinr.agent.tools.runtime.availability import (
    available_realtime_tool_names,
    bind_available_realtime_tool_handlers,
)
from twinr.agent.tools.schemas import build_agent_tool_schemas, build_compact_agent_tool_schemas
from twinr.agent.workflows.forensics import (
    WorkflowForensics,
    bind_workflow_forensics,
    workflow_decision,
    workflow_event,
    workflow_span,
)

from .contracts import ChannelInboundMessage, ChannelOutboundMessage
from .pending_actions import PendingTextChannelAction, PendingTextChannelActionStore
from .pending_whatsapp_followup import (
    PendingWhatsAppFollowUpResult,
    resolve_pending_whatsapp_follow_up,
)
from .turn_instructions import build_tool_text_channel_turn_instructions
from .whatsapp.outbound import WhatsAppOutboundResult, dispatch_whatsapp_outbound_message


LOGGER = logging.getLogger(__name__)
_TEXT_CHANNEL_TOOL_NAMES: tuple[str, ...] = (
    "search_live_info",
    "browser_automation",
    "connect_service_integration",
    "schedule_reminder",
    "list_automations",
    "create_time_automation",
    "create_sensor_automation",
    "update_time_automation",
    "update_sensor_automation",
    "delete_automation",
    "list_smart_home_entities",
    "read_smart_home_state",
    "control_smart_home_entities",
    "read_smart_home_sensor_stream",
    "remember_memory",
    "remember_contact",
    "lookup_contact",
    "send_whatsapp_message",
    "get_memory_conflicts",
    "resolve_memory_conflict",
    "remember_preference",
    "remember_plan",
    "update_user_profile",
    "update_personality",
    "manage_user_discovery",
    "configure_world_intelligence",
    "update_simple_setting",
    "manage_voice_quiet_mode",
    "end_conversation",
)


@dataclass
class _TextChannelToolOwner:
    """Expose the focused owner surface required by the shared tool executor."""

    config: TwinrConfig
    runtime: TwinrRuntime = field(repr=False)
    print_backend: AgentTextProvider = field(repr=False)
    whatsapp_outbound_dispatcher: Callable[..., WhatsAppOutboundResult] | None = field(default=None, repr=False)
    _configurable_providers: tuple[object, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._configurable_providers = (self.print_backend,)

    def emit(self, payload: str) -> None:
        LOGGER.debug("Text-channel tool telemetry: %s", payload)

    def _record_event(self, event_name: str, message: str, **data: object) -> None:
        LOGGER.info(
            "Text-channel tool event %s: %s | %s",
            event_name,
            message,
            {key: value for key, value in data.items() if value is not None},
        )

    def _record_usage(self, **data: object) -> None:
        LOGGER.debug(
            "Text-channel tool usage: %s",
            {key: value for key, value in data.items() if value is not None},
        )

    def _reload_live_config_from_env(self, env_path: Path) -> None:
        updated_config = TwinrConfig.from_env(env_path)
        self.config = updated_config
        apply_live_config = getattr(self.runtime, "apply_live_config", None)
        if callable(apply_live_config):
            apply_live_config(updated_config)
        for provider in self._configurable_providers:
            if hasattr(provider, "config"):
                provider.config = updated_config

    def dispatch_whatsapp_outbound_message(
        self,
        *,
        chat_jid: str,
        text: str,
        recipient_label: str,
        reply_to_message_id: str | None = None,
        wait_timeout_s: float | None = None,
    ) -> WhatsAppOutboundResult:
        """Deliver WhatsApp sends directly when the active channel owns transport.

        External callers still fall back to the shared outbound queue, but the
        long-lived WhatsApp channel must not enqueue a request and then wait for
        its own event loop to deliver it.
        """

        if self.whatsapp_outbound_dispatcher is not None:
            return self.whatsapp_outbound_dispatcher(
                chat_jid=chat_jid,
                text=text,
                recipient_label=recipient_label,
                reply_to_message_id=reply_to_message_id,
                wait_timeout_s=wait_timeout_s,
            )
        return dispatch_whatsapp_outbound_message(
            self.config,
            chat_jid=chat_jid,
            text=text,
            recipient_label=recipient_label,
            reply_to_message_id=reply_to_message_id,
            wait_timeout_s=wait_timeout_s,
        )


@dataclass(slots=True)
class TwinrToolTextChannelTurnService:
    """Handle one external text turn on Twinr's shared tool-capable agent lane."""

    runtime: TwinrRuntime = field(repr=False)
    tool_agent_provider: ToolCallingAgentProvider = field(repr=False)
    print_backend: AgentTextProvider = field(repr=False)
    allow_web_search: bool | None = False
    busy_reply_text: str = "Ich bearbeite gerade schon etwas. Schreib bitte gleich noch einmal."
    error_reply_text: str = "Ich hatte gerade ein internes Problem. Versuch es bitte noch einmal."
    max_rounds: int = 6
    whatsapp_outbound_dispatcher: Callable[..., WhatsAppOutboundResult] | None = field(default=None, repr=False)
    _lock: RLock = field(init=False, repr=False, default_factory=RLock)
    _workflow_forensics: WorkflowForensics = field(init=False, repr=False)
    _owner: _TextChannelToolOwner = field(init=False, repr=False)
    _pending_actions: PendingTextChannelActionStore = field(init=False, repr=False, default_factory=PendingTextChannelActionStore)

    def __post_init__(self) -> None:
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
        runtime_config = getattr(self.runtime, "config", None)
        if not isinstance(runtime_config, TwinrConfig):
            raise RuntimeError("Tool text-channel turns require runtime.config to be a TwinrConfig")
        self._owner = _TextChannelToolOwner(
            config=runtime_config,
            runtime=self.runtime,
            print_backend=self.print_backend,
            whatsapp_outbound_dispatcher=self.whatsapp_outbound_dispatcher,
        )
        self._pending_actions = PendingTextChannelActionStore(
            path=self._pending_action_store_path(runtime_config)
        )
        warmup_callback = getattr(getattr(self.runtime, "long_term_memory", None), "prewarm_foreground_read_cache", None)
        if not callable(warmup_callback):
            return
        try:
            with bind_workflow_forensics(self._workflow_forensics, trace_id=uuid4().hex):
                with workflow_span(
                    name="tool_text_channel_prewarm_foreground_read_cache",
                    kind="cache",
                    details={"channel_service": self.__class__.__name__},
                ):
                    warmup_callback()
        except Exception:
            LOGGER.exception("Tool text-channel warmup failed.")
            raise

    def handle_inbound(self, message: ChannelInboundMessage) -> ChannelOutboundMessage:
        """Run one inbound text message through the shared tool-calling agent."""

        turn_started = time.monotonic()
        trace_id = uuid4().hex
        with bind_workflow_forensics(self._workflow_forensics, trace_id=trace_id):
            workflow_event(
                kind="turn_start",
                msg="tool_text_channel_turn_started",
                details=self._message_trace_details(message),
            )
            workflow_decision(
                msg="tool_text_channel_web_search_policy",
                question="Should this tool text-channel turn allow provider-side web search?",
                selected={
                    "id": "inherit_default" if self.allow_web_search is None else str(bool(self.allow_web_search)).lower(),
                    "summary": "Use the configured tool text-channel web-search policy.",
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
                name="tool_text_channel_handle_inbound",
                kind="turn",
                details=self._message_trace_details(message),
            ):
                with self._lock:
                    with workflow_span(name="tool_text_channel_prepare_runtime", kind="mutation"):
                        self._prepare_runtime()
                    if self.runtime.status != TwinrStatus.WAITING:
                        workflow_event(
                            kind="branch",
                            msg="tool_text_channel_busy_reply_returned",
                            details={"runtime_status": getattr(self.runtime.status, "value", str(self.runtime.status))},
                        )
                        return self._reply_for(message, self.busy_reply_text)

                    try:
                        pending_action = self._pending_actions.get(message.conversation_id)
                        with workflow_span(
                            name="tool_text_channel_begin_listening",
                            kind="mutation",
                            details={"request_source": message.channel},
                        ):
                            self.runtime.begin_listening(request_source=message.channel)
                        with workflow_span(name="tool_text_channel_submit_transcript", kind="mutation"):
                            self.runtime.submit_transcript(message.text)
                        conversation: tuple[tuple[str, str], ...] | None = None
                        effective_prompt = message.text
                        if pending_action is not None:
                            with workflow_span(
                                name="tool_text_channel_provider_context",
                                kind="retrieval",
                                details={"query_chars": len((message.text or "").strip())},
                            ):
                                conversation = focus_recent_thread_conversation(
                                    self.runtime.tool_provider_text_surface_conversation_context(),
                                    user_transcript=message.text,
                                )
                                conversation, effective_prompt = self._apply_pending_action_context(
                                    conversation,
                                    message.text,
                                    pending_action=pending_action,
                                )
                            direct_follow_up = self._resolve_pending_whatsapp_follow_up(
                                message,
                                pending_action=pending_action,
                                conversation=conversation,
                            )
                            if direct_follow_up is not None:
                                workflow_event(
                                    kind="branch",
                                    msg="tool_text_channel_pending_whatsapp_follow_up_resolved_with_model_guidance",
                                    details={
                                        "pending_state": pending_action.state,
                                        "tool_results": len(direct_follow_up.tool_results),
                                        "provider_model": direct_follow_up.provider_model,
                                    },
                                )
                                return self._complete_local_pending_follow_up(
                                    message,
                                    direct_follow_up,
                                    turn_started=turn_started,
                                )
                        if conversation is None:
                            with workflow_span(
                                name="tool_text_channel_provider_context",
                                kind="retrieval",
                                details={"query_chars": len((message.text or "").strip())},
                            ):
                                conversation = focus_recent_thread_conversation(
                                    self.runtime.tool_provider_text_surface_conversation_context(),
                                    user_transcript=message.text,
                                )
                                resolution = maybe_rewrite_prompt_against_recent_thread(
                                    getattr(self.tool_agent_provider, "backend", None),
                                    conversation=conversation,
                                    user_transcript=message.text,
                                )
                                conversation, effective_prompt = self._apply_pending_action_context(
                                    conversation,
                                    resolution.effective_prompt,
                                    pending_action=pending_action,
                                )
                        available_tool_names, loop = self._build_turn_loop()
                        with workflow_span(
                            name="tool_text_channel_agent_turn",
                            kind="llm_call",
                            details={
                                "conversation_messages": len(conversation),
                                "allow_web_search": self.allow_web_search,
                                "tool_count": len(available_tool_names),
                            },
                        ):
                            result = loop.run(
                                effective_prompt,
                                conversation=conversation,
                                instructions=self._turn_instructions(
                                    channel=message.channel,
                                    tool_names=available_tool_names,
                                    pending_action=pending_action,
                                ),
                                allow_web_search=self.allow_web_search,
                                on_text_delta=None,
                            )
                        self._pending_actions.update_from_tool_results(message.conversation_id, result.tool_results)
                        with workflow_span(name="tool_text_channel_complete_agent_turn", kind="mutation"):
                            answer = self.runtime.complete_agent_turn(
                                result.text,
                                source=message.channel,
                                modality="text",
                            )
                        with workflow_span(name="tool_text_channel_finish_speaking", kind="mutation"):
                            self.runtime.finish_speaking()
                        metadata: dict[str, str] = {}
                        if result.response_id:
                            metadata["provider_response_id"] = str(result.response_id)
                        if result.request_id:
                            metadata["provider_request_id"] = str(result.request_id)
                        if result.model:
                            metadata["provider_model"] = str(result.model)
                        metadata["provider_used_web_search"] = str(bool(result.used_web_search)).lower()
                        metadata["tool_rounds"] = str(result.rounds)
                        metadata["tool_calls"] = str(len(result.tool_calls))
                        workflow_event(
                            kind="turn_end",
                            msg="tool_text_channel_turn_completed",
                            details={
                                "runtime_status": getattr(self.runtime.status, "value", str(self.runtime.status)),
                                "provider_model": result.model,
                                "response_chars": len(answer),
                                "tool_count": len(result.tool_calls),
                            },
                            kpi={"duration_ms": round(max(0.0, (time.monotonic() - turn_started) * 1000.0), 3)},
                        )
                        return self._reply_for(message, answer, metadata=metadata)
                    except Exception as exc:
                        workflow_event(
                            kind="exception",
                            msg="tool_text_channel_turn_failed",
                            details={"error_type": type(exc).__name__},
                            level="ERROR",
                            kpi={"duration_ms": round(max(0.0, (time.monotonic() - turn_started) * 1000.0), 3)},
                        )
                        LOGGER.exception("Failed to process %s channel message %s.", message.channel, message.message_id)
                        self.runtime.fail(str(exc))
                        return self._reply_for(message, self.error_reply_text)

    def _resolve_pending_whatsapp_follow_up(
        self,
        message: ChannelInboundMessage,
        *,
        pending_action: PendingTextChannelAction | None,
        conversation: tuple[tuple[str, str], ...],
    ) -> PendingWhatsAppFollowUpResult | None:
        if pending_action is None:
            return None
        tool_names = ("send_whatsapp_message",)
        tool_executor = RealtimeToolExecutor(self._owner)
        with workflow_span(
            name="tool_text_channel_pending_whatsapp_follow_up",
            kind="llm_call",
            details={
                "channel": message.channel,
                "conversation_id": message.conversation_id,
                "pending_tool": pending_action.tool_name,
                "pending_state": pending_action.state,
            },
        ):
            return resolve_pending_whatsapp_follow_up(
                pending_action,
                message.text,
                provider=self.tool_agent_provider,
                conversation=conversation,
                instructions=self._turn_instructions(
                    channel=message.channel,
                    tool_names=tool_names,
                    pending_action=pending_action,
                ),
                tool_schemas=self._build_tool_schemas(tool_names),
                allow_web_search=self.allow_web_search,
                execute_tool=tool_executor.handle_send_whatsapp_message,
                max_rounds=self.max_rounds,
            )

    def _complete_local_pending_follow_up(
        self,
        message: ChannelInboundMessage,
        result: PendingWhatsAppFollowUpResult,
        *,
        turn_started: float,
    ) -> ChannelOutboundMessage:
        if result.clear_pending:
            self._pending_actions.clear(message.conversation_id)
        elif result.tool_results:
            self._pending_actions.update_from_tool_results(
                message.conversation_id,
                result.tool_results,
            )
        with workflow_span(name="tool_text_channel_complete_local_pending_follow_up", kind="mutation"):
            answer = self.runtime.complete_agent_turn(
                result.reply_text,
                source=message.channel,
                modality="text",
            )
        with workflow_span(name="tool_text_channel_finish_local_pending_follow_up", kind="mutation"):
            self.runtime.finish_speaking()
        workflow_event(
            kind="turn_end",
            msg="tool_text_channel_turn_completed_from_local_pending_follow_up",
            details={
                "runtime_status": getattr(self.runtime.status, "value", str(self.runtime.status)),
                "provider_model": result.provider_model or "pending_whatsapp_follow_up",
                "response_chars": len(answer),
                "tool_count": len(result.tool_results),
            },
            kpi={"duration_ms": round(max(0.0, (time.monotonic() - turn_started) * 1000.0), 3)},
        )
        metadata = {
            "provider_model": result.provider_model or "pending_whatsapp_follow_up",
            "provider_used_web_search": str(bool(result.used_web_search)).lower(),
            "tool_rounds": str(max(0, int(result.tool_rounds))),
            "tool_calls": str(len(result.tool_results)),
        }
        if result.provider_response_id:
            metadata["provider_response_id"] = result.provider_response_id
        if result.provider_request_id:
            metadata["provider_request_id"] = result.provider_request_id
        return self._reply_for(message, answer, metadata=metadata)

    def _build_turn_loop(self) -> tuple[tuple[str, ...], ToolCallingStreamingLoop]:
        config = self._owner.config
        available_tool_names = available_realtime_tool_names(
            config,
            tool_names=_TEXT_CHANNEL_TOOL_NAMES,
        )
        tool_executor = RealtimeToolExecutor(self._owner)
        tool_handlers = bind_available_realtime_tool_handlers(
            tool_executor,
            config=config,
            tool_names=available_tool_names,
        )
        tool_schemas = self._build_tool_schemas(available_tool_names)
        return (
            available_tool_names,
            ToolCallingStreamingLoop(
                provider=self.tool_agent_provider,
                tool_handlers=cast(dict[str, ToolHandler], tool_handlers),
                tool_schemas=tuple(tool_schemas),
                max_rounds=self.max_rounds,
                stream_final_only=(self._provider_name() == "groq"),
            ),
        )

    def _build_tool_schemas(self, tool_names: tuple[str, ...]) -> tuple[dict[str, object], ...]:
        if self._provider_name() == "groq":
            return tuple(build_compact_agent_tool_schemas(tool_names))
        return tuple(build_agent_tool_schemas(tool_names))

    def _provider_name(self) -> str:
        return str(getattr(self._owner.config, "llm_provider", "") or "").strip().lower()

    def _pending_action_store_path(self, config: TwinrConfig) -> Path:
        project_root = Path(config.project_root).expanduser().resolve(strict=False)
        runtime_state_path = Path(config.runtime_state_path).expanduser()
        if not runtime_state_path.is_absolute():
            runtime_state_path = project_root / runtime_state_path
        return runtime_state_path.resolve(strict=False).parent / "pending-text-channel-actions.json"

    def _turn_instructions(
        self,
        *,
        channel: str,
        tool_names: tuple[str, ...],
        pending_action: PendingTextChannelAction | None,
    ) -> str:
        return build_tool_text_channel_turn_instructions(
            self._owner.config,
            channel=channel,
            tool_names=tool_names,
            pending_action_message=(
                pending_action.system_message()
                if pending_action is not None
                else None
            ),
        )

    @staticmethod
    def _apply_pending_action_context(
        conversation: tuple[tuple[str, str], ...],
        prompt: str,
        *,
        pending_action: PendingTextChannelAction | None,
    ) -> tuple[tuple[tuple[str, str], ...], str]:
        if pending_action is None:
            return conversation, prompt
        return (
            tuple(conversation) + (("system", pending_action.system_message()),),
            pending_action.rewrite_prompt(prompt),
        )

    def _prepare_runtime(self) -> None:
        if self.runtime.status == TwinrStatus.ERROR:
            self.runtime.reset_error()

    @staticmethod
    def _reply_for(
        message: ChannelInboundMessage,
        text: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> ChannelOutboundMessage:
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
