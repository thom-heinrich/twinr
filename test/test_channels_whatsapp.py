# mypy: disable-error-code="arg-type,assignment,attr-defined,method-assign,index"
import os
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import sys
import tempfile
import time
from unittest.mock import Mock, call, patch
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult, ToolCallingTurnResponse
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime.runtime import TwinrRuntime
from twinr.agent.base_agent.state.machine import TwinrStatus
from twinr.agent.tools.handlers.whatsapp import handle_send_whatsapp_message
from twinr.agent.tools.runtime.availability import available_realtime_tool_names
from twinr.agent.tools.schemas import build_agent_tool_schemas
from twinr.channels import ChannelInboundMessage, ChannelTransportError, TwinrTextChannelTurnService
from twinr.channels.pending_actions import PendingTextChannelActionStore
from twinr.channels.tool_runtime import TwinrToolTextChannelTurnService
from twinr.channels.whatsapp import (
    WhatsAppHistoryImportResult,
    WhatsAppHistoryImportQueue,
    TwinrWhatsAppChannelLoop,
    WhatsAppChannelConfig,
    WhatsAppMessagePolicy,
    WhatsAppOutboundQueue,
    WhatsAppOutboundResult,
)
from twinr.channels.whatsapp.inbound_event_log import WhatsAppInboundEventLog
from twinr.channels.whatsapp.history_learning import build_import_turns
from twinr.channels.whatsapp.node_runtime import detect_whatsapp_node_runtime_spec, resolve_whatsapp_node_binary
from twinr.channels.whatsapp.worker_dependencies import (
    ensure_whatsapp_worker_dependencies,
    probe_whatsapp_worker_dependencies,
)
from twinr.channels.whatsapp.worker_bridge import (
    WhatsAppWorkerBridge,
    WhatsAppWorkerHistoryBatchEvent,
    WhatsAppWorkerHistoryMessage,
    WhatsAppWorkerStatusEvent,
)
from twinr.channels.onboarding import FileBackedChannelOnboardingStore, InProcessChannelPairingRegistry
from twinr.channels.whatsapp.pairing import WhatsAppPairingCoordinator, probe_whatsapp_runtime
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermSourceRefV1

_TEST_ALLOW_FROM_DISPLAY = "+1 555 555 4567"
_TEST_ALLOW_FROM = "+15555554567"
_TEST_ACCOUNT_JID = "15555554567@s.whatsapp.net"
_TEST_LOOKUP_PHONE = "+15555551223"
_TEST_CONTACT_PHONE_A = "+15555552234"
_TEST_CONTACT_PHONE_B = "+15555555678"
_TEST_CONTACT_PHONE_ALT_1 = "+15555550001"
_TEST_CONTACT_PHONE_ALT_2 = "+15555550002"


def _test_twinr_whatsapp_config(root: Path, *, runtime_state_path: str | None = None) -> TwinrConfig:
    values: dict[str, str] = {
        "project_root": str(root),
        "whatsapp_allow_from": _TEST_ALLOW_FROM,
        "whatsapp_auth_dir": "state/channels/whatsapp/auth",
        "whatsapp_worker_root": "src/twinr/channels/whatsapp/worker",
    }
    if runtime_state_path is not None:
        values["runtime_state_path"] = runtime_state_path
    return TwinrConfig(**values)


def _test_whatsapp_channel_config(
    *,
    auth_dir: Path,
    worker_root: Path,
    node_binary: str = "node",
    groups_enabled: bool = False,
    self_chat_mode: bool = False,
    reconnect_base_delay_s: float = 2.0,
    reconnect_max_delay_s: float = 30.0,
    send_timeout_s: float = 20.0,
    sent_cache_ttl_s: float = 180.0,
    sent_cache_max_entries: int = 256,
) -> WhatsAppChannelConfig:
    return WhatsAppChannelConfig(
        allow_from=_TEST_ALLOW_FROM_DISPLAY,
        allow_from_jid=_TEST_ACCOUNT_JID,
        auth_dir=auth_dir,
        worker_root=worker_root,
        node_binary=node_binary,
        groups_enabled=groups_enabled,
        self_chat_mode=self_chat_mode,
        reconnect_base_delay_s=reconnect_base_delay_s,
        reconnect_max_delay_s=reconnect_max_delay_s,
        send_timeout_s=send_timeout_s,
        sent_cache_ttl_s=sent_cache_ttl_s,
        sent_cache_max_entries=sent_cache_max_entries,
    )


class _FakeRuntime:
    def __init__(self) -> None:
        self.status = TwinrStatus.WAITING
        self.events: list[tuple[object, ...]] = []
        self.long_term_memory = None
        self.last_transcript = ""
        self._conversation_turns: list[tuple[str, str]] = []

    def begin_listening(self, *, request_source: str, button=None, proactive_trigger=None):
        self.events.append(("begin_listening", request_source))
        self.status = TwinrStatus.LISTENING
        return self.status

    def submit_transcript(self, transcript: str):
        self.events.append(("submit_transcript", transcript))
        self.last_transcript = transcript
        self.status = TwinrStatus.PROCESSING
        return self.status

    def provider_conversation_context(self):
        return (("system", "memory context"), *tuple(self._conversation_turns))

    def complete_agent_turn(self, answer: str, *, source: str = "conversation", modality: str = "voice"):
        self.events.append(("complete_agent_turn", answer, source, modality))
        if self.last_transcript:
            self._conversation_turns.append(("user", self.last_transcript))
        self._conversation_turns.append(("assistant", answer))
        self.status = TwinrStatus.ANSWERING
        return answer

    def finish_speaking(self):
        self.events.append(("finish_speaking", "done"))
        self.status = TwinrStatus.WAITING
        return self.status

    def fail(self, message: str):
        self.events.append(("fail", message))
        self.status = TwinrStatus.ERROR
        return self.status

    def reset_error(self):
        self.events.append(("reset_error", "done"))
        self.status = TwinrStatus.WAITING
        return self.status


class _FakeToolRuntime(_FakeRuntime):
    def __init__(self, config: TwinrConfig) -> None:
        super().__init__()
        self.config = config
        self.lookup_calls: list[tuple[str, str | None, str | None]] = []

    def provider_conversation_context(self):
        raise AssertionError("Tool text-channel turns must use tool_provider_conversation_context")

    def tool_provider_conversation_context(self):
        self.events.append(("tool_provider_conversation_context", "done"))
        return (("system", "tool memory context"), *tuple(self._conversation_turns))

    def lookup_contact(
        self,
        *,
        name: str,
        family_name: str | None = None,
        role: str | None = None,
        contact_label: str | None = None,
    ):
        self.lookup_calls.append((name, family_name, role))
        return SimpleNamespace(
            status="ok",
            match=SimpleNamespace(
                label=contact_label or "Anna",
                role="friend",
                phones=[_TEST_LOOKUP_PHONE],
            ),
        )

    def apply_live_config(self, config: TwinrConfig):
        self.config = config
        self.events.append(("apply_live_config", str(config.project_root)))
        return self.config


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[tuple[str, str], ...], bool | None]] = []

    def respond_with_metadata(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions=None,
        allow_web_search=None,
    ):
        self.calls.append((prompt, conversation, allow_web_search))
        return SimpleNamespace(
            text="Hallo aus Twinr.",
            response_id="resp_123",
            request_id="req_123",
            model="gpt-test",
        )


class _FakeToolAgentProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.start_calls: list[dict[str, object]] = []
        self.continue_calls: list[dict[str, object]] = []

    def start_turn_streaming(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del on_text_delta
        self.start_calls.append(
            {
                "prompt": prompt,
                "conversation": conversation,
                "instructions": instructions,
                "tool_schemas": tuple(tool_schemas),
                "allow_web_search": allow_web_search,
            }
        )
        return ToolCallingTurnResponse(
            text="",
            tool_calls=(
                AgentToolCall(
                    name="send_whatsapp_message",
                    call_id="call_1",
                    arguments={
                        "name": "Anna",
                        "message": "Bring bitte den Schlüssel mit.",
                        "confirmed": True,
                    },
                ),
            ),
            response_id="resp_tool_start",
            request_id="req_tool_start",
            model="gpt-tool-test",
            continuation_token="cont_1",
        )

    def continue_turn_streaming(
        self,
        *,
        continuation_token: str,
        tool_results,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del on_text_delta
        self.continue_calls.append(
            {
                "continuation_token": continuation_token,
                "tool_results": tuple(tool_results),
                "instructions": instructions,
                "tool_schemas": tuple(tool_schemas),
                "allow_web_search": allow_web_search,
            }
        )
        return ToolCallingTurnResponse(
            text="Ich habe Anna geschrieben.",
            response_id="resp_tool_final",
            request_id="req_tool_final",
            model="gpt-tool-test",
        )


def _fake_whatsapp_reply_text(result_output: dict[str, object]) -> str:
    question = str(result_output.get("question", "") or "").strip()
    if question:
        return question
    message = str(result_output.get("message", "") or "").strip()
    if message:
        return message
    status = str(result_output.get("status", "") or "").strip()
    recipient_label = (
        str(result_output.get("recipient_label", "") or "").strip()
        or str(result_output.get("name", "") or "").strip()
    )
    if status == "sent" and recipient_label:
        return f"Ich habe {recipient_label} geschrieben."
    if status == "delivery_failed":
        return "Die WhatsApp konnte gerade nicht gesendet werden."
    if status == "not_found":
        return "Ich konnte diesen Kontakt gerade nicht finden."
    return ""


class _FakeWhatsAppPendingProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.start_calls: list[dict[str, object]] = []
        self.continue_calls: list[dict[str, object]] = []

    def start_turn_streaming(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del instructions, tool_schemas, allow_web_search, on_text_delta
        call_index = len(self.start_calls)
        call: dict[str, object] = {
            "prompt": prompt,
            "conversation": tuple(conversation or ()),
        }
        self.start_calls.append(call)
        if call_index == 0:
            return ToolCallingTurnResponse(
                text="",
                tool_calls=(
                    AgentToolCall(
                        name="send_whatsapp_message",
                        call_id="call_1",
                        arguments={"name": "Anna"},
                    ),
                ),
                response_id="resp_msg_start",
                request_id="req_msg_start",
                model="gpt-tool-test",
                continuation_token="cont_msg",
            )
        if call_index == 1:
            return ToolCallingTurnResponse(
                text="",
                tool_calls=(
                    AgentToolCall(
                        name="send_whatsapp_message",
                        call_id="call_2",
                        arguments={
                            "name": "Anna",
                            "message": "Bring bitte den Schlüssel mit.",
                        },
                    ),
                ),
                response_id="resp_msg_pending",
                request_id="req_msg_pending",
                model="gpt-tool-test",
                continuation_token="cont_msg_pending",
            )
        if call_index == 2:
            return ToolCallingTurnResponse(
                text="",
                tool_calls=(
                    AgentToolCall(
                        name="send_whatsapp_message",
                        call_id="call_3",
                        arguments={
                            "name": "Anna",
                            "message": "Bring bitte den Schlüssel mit.",
                            "confirmed": True,
                        },
                    ),
                ),
                response_id="resp_confirm_pending",
                request_id="req_confirm_pending",
                model="gpt-tool-test",
                continuation_token="cont_confirm_pending",
            )
        raise AssertionError("unexpected extra start_turn_streaming call")

    def continue_turn_streaming(
        self,
        *,
        continuation_token: str,
        tool_results,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del instructions, tool_schemas, allow_web_search, on_text_delta
        normalized_results = tuple(tool_results)
        self.continue_calls.append(
            {
                "continuation_token": continuation_token,
                "tool_results": normalized_results,
            }
        )
        result_output = normalized_results[0].output
        assert isinstance(result_output, dict)
        if continuation_token in {"cont_msg", "cont_msg_pending", "cont_confirm_pending"}:
            response_id = {
                "cont_msg": "resp_msg_final",
                "cont_msg_pending": "resp_msg_pending_final",
                "cont_confirm_pending": "resp_confirm_pending_final",
            }[continuation_token]
            request_id = {
                "cont_msg": "req_msg_final",
                "cont_msg_pending": "req_msg_pending_final",
                "cont_confirm_pending": "req_confirm_pending_final",
            }[continuation_token]
            return ToolCallingTurnResponse(
                text=_fake_whatsapp_reply_text(result_output),
                response_id=response_id,
                request_id=request_id,
                model="gpt-tool-test",
            )
        raise AssertionError(f"unexpected continuation token {continuation_token!r}")


class _FakeWhatsAppClarificationProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.start_calls: list[dict[str, object]] = []
        self.continue_calls: list[dict[str, object]] = []

    def start_turn_streaming(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del instructions, tool_schemas, allow_web_search, on_text_delta
        call_index = len(self.start_calls)
        call: dict[str, object] = {
            "prompt": prompt,
            "conversation": tuple(conversation or ()),
        }
        self.start_calls.append(call)
        if call_index == 0:
            return ToolCallingTurnResponse(
                text="",
                tool_calls=(
                    AgentToolCall(
                        name="send_whatsapp_message",
                        call_id="call_1",
                        arguments={
                            "name": "Janina",
                            "message": "Kannst du mich spaeter anrufen?",
                        },
                    ),
                ),
                response_id="resp_clarify_start",
                request_id="req_clarify_start",
                model="gpt-tool-test",
                continuation_token="cont_clarify",
            )
        if call_index == 1:
            return ToolCallingTurnResponse(
                text="",
                tool_calls=(
                    AgentToolCall(
                        name="send_whatsapp_message",
                        call_id="call_2",
                        arguments={
                            "name": "Janina",
                            "family_name": "Weber",
                            "role": "Nachbarin",
                            "message": "Kannst du mich spaeter anrufen?",
                        },
                    ),
                ),
                response_id="resp_clarify_pending",
                request_id="req_clarify_pending",
                model="gpt-tool-test",
                continuation_token="cont_clarify_pending",
            )
        if call_index == 2:
            return ToolCallingTurnResponse(
                text="",
                tool_calls=(
                    AgentToolCall(
                        name="send_whatsapp_message",
                        call_id="call_3",
                        arguments={
                            "name": "Janina",
                            "family_name": "Weber",
                            "role": "Nachbarin",
                            "message": "Kannst du mich spaeter anrufen?",
                            "confirmed": True,
                        },
                    ),
                ),
                response_id="resp_confirm_pending",
                request_id="req_confirm_pending",
                model="gpt-tool-test",
                continuation_token="cont_clarify_confirm_pending",
            )
        raise AssertionError("unexpected extra start_turn_streaming call")

    def continue_turn_streaming(
        self,
        *,
        continuation_token: str,
        tool_results,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del instructions, tool_schemas, allow_web_search, on_text_delta
        normalized_results = tuple(tool_results)
        self.continue_calls.append(
            {
                "continuation_token": continuation_token,
                "tool_results": normalized_results,
            }
        )
        result_output = normalized_results[0].output
        assert isinstance(result_output, dict)
        if continuation_token in {
            "cont_clarify",
            "cont_clarify_pending",
            "cont_clarify_confirm_pending",
        }:
            response_id = {
                "cont_clarify": "resp_clarify_final",
                "cont_clarify_pending": "resp_clarify_pending_final",
                "cont_clarify_confirm_pending": "resp_confirm_pending_final",
            }[continuation_token]
            request_id = {
                "cont_clarify": "req_clarify_final",
                "cont_clarify_pending": "req_clarify_pending_final",
                "cont_clarify_confirm_pending": "req_confirm_pending_final",
            }[continuation_token]
            return ToolCallingTurnResponse(
                text=_fake_whatsapp_reply_text(result_output),
                response_id=response_id,
                request_id=request_id,
                model="gpt-tool-test",
            )
        raise AssertionError(f"unexpected continuation token {continuation_token!r}")


class _FakeWhatsAppNameClarificationProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.start_calls: list[dict[str, object]] = []
        self.continue_calls: list[dict[str, object]] = []

    def start_turn_streaming(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del instructions, tool_schemas, allow_web_search, on_text_delta
        call_index = len(self.start_calls)
        call: dict[str, object] = {
            "prompt": prompt,
            "conversation": tuple(conversation or ()),
        }
        self.start_calls.append(call)
        if call_index == 0:
            return ToolCallingTurnResponse(
                text="",
                tool_calls=(
                    AgentToolCall(
                        name="send_whatsapp_message",
                        call_id="call_1",
                        arguments={"name": "Janina"},
                    ),
                ),
                response_id="resp_name_clarify_start",
                request_id="req_name_clarify_start",
                model="gpt-tool-test",
                continuation_token="cont_name_clarify",
            )
        if call_index == 1:
            return ToolCallingTurnResponse(
                text="",
                tool_calls=(
                    AgentToolCall(
                        name="send_whatsapp_message",
                        call_id="call_2",
                        arguments={
                            "name": "Janina",
                            "family_name": "Werner privat",
                            "role": "Privat",
                        },
                    ),
                ),
                response_id="resp_name_pending",
                request_id="req_name_pending",
                model="gpt-tool-test",
                continuation_token="cont_name_pending",
            )
        if call_index == 2:
            return ToolCallingTurnResponse(
                text="",
                tool_calls=(
                    AgentToolCall(
                        name="send_whatsapp_message",
                        call_id="call_3",
                        arguments={
                            "name": "Janina",
                            "family_name": "Werner privat",
                            "role": "Privat",
                            "message": "Ein Gedicht über Skyla und Uschi",
                        },
                    ),
                ),
                response_id="resp_name_message_pending",
                request_id="req_name_message_pending",
                model="gpt-tool-test",
                continuation_token="cont_name_message_pending",
            )
        if call_index == 3:
            return ToolCallingTurnResponse(
                text="",
                tool_calls=(
                    AgentToolCall(
                        name="send_whatsapp_message",
                        call_id="call_4",
                        arguments={
                            "name": "Janina",
                            "family_name": "Werner privat",
                            "role": "Privat",
                            "message": "Ein Gedicht über Skyla und Uschi",
                            "confirmed": True,
                        },
                    ),
                ),
                response_id="resp_name_confirm_pending",
                request_id="req_name_confirm_pending",
                model="gpt-tool-test",
                continuation_token="cont_name_confirm_pending",
            )
        raise AssertionError("unexpected extra start_turn_streaming call")

    def continue_turn_streaming(
        self,
        *,
        continuation_token: str,
        tool_results,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del instructions, tool_schemas, allow_web_search, on_text_delta
        normalized_results = tuple(tool_results)
        self.continue_calls.append(
            {
                "continuation_token": continuation_token,
                "tool_results": normalized_results,
            }
        )
        result_output = normalized_results[0].output
        assert isinstance(result_output, dict)
        if continuation_token in {
            "cont_name_clarify",
            "cont_name_pending",
            "cont_name_message_pending",
            "cont_name_confirm_pending",
        }:
            response_id = {
                "cont_name_clarify": "resp_name_clarify_final",
                "cont_name_pending": "resp_name_pending_final",
                "cont_name_message_pending": "resp_name_message_pending_final",
                "cont_name_confirm_pending": "resp_name_confirm_pending_final",
            }[continuation_token]
            request_id = {
                "cont_name_clarify": "req_name_clarify_final",
                "cont_name_pending": "req_name_pending_final",
                "cont_name_message_pending": "req_name_message_pending_final",
                "cont_name_confirm_pending": "req_name_confirm_pending_final",
            }[continuation_token]
            return ToolCallingTurnResponse(
                text=_fake_whatsapp_reply_text(result_output),
                response_id=response_id,
                request_id=request_id,
                model="gpt-tool-test",
            )
        raise AssertionError(f"unexpected continuation token {continuation_token!r}")


class _FakeWhatsAppExactLabelClarificationProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.start_calls: list[dict[str, object]] = []
        self.continue_calls: list[dict[str, object]] = []

    def start_turn_streaming(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del instructions, tool_schemas, allow_web_search, on_text_delta
        call_index = len(self.start_calls)
        call: dict[str, object] = {
            "prompt": prompt,
            "conversation": tuple(conversation or ()),
        }
        self.start_calls.append(call)
        if call_index == 0:
            return ToolCallingTurnResponse(
                text="",
                tool_calls=(
                    AgentToolCall(
                        name="send_whatsapp_message",
                        call_id="call_1",
                        arguments={"name": "Janina"},
                    ),
                ),
                response_id="resp_exact_label_start",
                request_id="req_exact_label_start",
                model="gpt-tool-test",
                continuation_token="cont_exact_label",
            )
        if call_index == 1:
            return ToolCallingTurnResponse(
                text="",
                tool_calls=(
                    AgentToolCall(
                        name="send_whatsapp_message",
                        call_id="call_2",
                        arguments={
                            "name": "Janina",
                            "contact_label": "Janina Werner privat",
                        },
                    ),
                ),
                response_id="resp_exact_label_pending",
                request_id="req_exact_label_pending",
                model="gpt-tool-test",
                continuation_token="cont_exact_label_pending",
            )
        if call_index == 2:
            return ToolCallingTurnResponse(
                text="",
                tool_calls=(
                    AgentToolCall(
                        name="send_whatsapp_message",
                        call_id="call_3",
                        arguments={
                            "name": "Janina",
                            "contact_label": "Janina Werner privat",
                            "message": "Ein Gedicht über Skyla und Uschi",
                        },
                    ),
                ),
                response_id="resp_exact_label_message_pending",
                request_id="req_exact_label_message_pending",
                model="gpt-tool-test",
                continuation_token="cont_exact_label_message_pending",
            )
        if call_index == 3:
            return ToolCallingTurnResponse(
                text="",
                tool_calls=(
                    AgentToolCall(
                        name="send_whatsapp_message",
                        call_id="call_4",
                        arguments={
                            "name": "Janina",
                            "contact_label": "Janina Werner privat",
                            "message": "Ein Gedicht über Skyla und Uschi",
                            "confirmed": True,
                        },
                    ),
                ),
                response_id="resp_exact_label_confirm_pending",
                request_id="req_exact_label_confirm_pending",
                model="gpt-tool-test",
                continuation_token="cont_exact_label_confirm_pending",
            )
        raise AssertionError("unexpected extra start_turn_streaming call")

    def continue_turn_streaming(
        self,
        *,
        continuation_token: str,
        tool_results,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del instructions, tool_schemas, allow_web_search, on_text_delta
        normalized_results = tuple(tool_results)
        self.continue_calls.append(
            {
                "continuation_token": continuation_token,
                "tool_results": normalized_results,
            }
        )
        result_output = normalized_results[0].output
        assert isinstance(result_output, dict)
        if continuation_token in {
            "cont_exact_label",
            "cont_exact_label_pending",
            "cont_exact_label_message_pending",
            "cont_exact_label_confirm_pending",
        }:
            response_id = {
                "cont_exact_label": "resp_exact_label_final",
                "cont_exact_label_pending": "resp_exact_label_pending_final",
                "cont_exact_label_message_pending": "resp_exact_label_message_pending_final",
                "cont_exact_label_confirm_pending": "resp_exact_label_confirm_pending_final",
            }[continuation_token]
            request_id = {
                "cont_exact_label": "req_exact_label_final",
                "cont_exact_label_pending": "req_exact_label_pending_final",
                "cont_exact_label_message_pending": "req_exact_label_message_pending_final",
                "cont_exact_label_confirm_pending": "req_exact_label_confirm_pending_final",
            }[continuation_token]
            return ToolCallingTurnResponse(
                text=_fake_whatsapp_reply_text(result_output),
                response_id=response_id,
                request_id=request_id,
                model="gpt-tool-test",
            )
        raise AssertionError(f"unexpected continuation token {continuation_token!r}")


class _FakeSingleToolCallProvider:
    def __init__(self, config: TwinrConfig, *, tool_call: AgentToolCall) -> None:
        self.config = config
        self._tool_call = tool_call
        self.start_calls: list[dict[str, object]] = []
        self.continue_calls: list[dict[str, object]] = []

    def start_turn_streaming(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del instructions, tool_schemas, allow_web_search, on_text_delta
        self.start_calls.append(
            {
                "prompt": prompt,
                "conversation": tuple(conversation or ()),
            }
        )
        return ToolCallingTurnResponse(
            text="",
            tool_calls=(self._tool_call,),
            response_id="resp_single",
            request_id="req_single",
            model="gpt-tool-test",
            continuation_token="cont_single",
        )

    def continue_turn_streaming(
        self,
        *,
        continuation_token: str,
        tool_results,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del instructions, tool_schemas, allow_web_search, on_text_delta
        if continuation_token != "cont_single":
            raise AssertionError(f"unexpected continuation token {continuation_token!r}")
        normalized_results = tuple(tool_results)
        self.continue_calls.append(
            {
                "continuation_token": continuation_token,
                "tool_results": normalized_results,
            }
        )
        result_output = normalized_results[0].output
        assert isinstance(result_output, dict)
        return ToolCallingTurnResponse(
            text=_fake_whatsapp_reply_text(result_output),
            response_id="resp_single_final",
            request_id="req_single_final",
            model="gpt-tool-test",
        )


class _SequencedWorkerBridge:
    def __init__(self, _config, events):
        self._events = list(events)

    def start(self) -> None:
        return None

    def next_event(self, *, timeout_s: float | None = None):
        if not self._events:
            return None
        next_item = self._events.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item

    def stop(self) -> None:
        return None


class WhatsAppChannelTests(unittest.TestCase):
    def _message(self, **overrides):
        payload = {
            "channel": "whatsapp",
            "message_id": "msg-1",
            "conversation_id": _TEST_ACCOUNT_JID,
            "sender_id": _TEST_ACCOUNT_JID,
            "text": "Hallo Twinr",
        }
        payload.update(overrides)
        return ChannelInboundMessage(**payload)

    def _policy(self, *, self_chat_mode: bool = False, groups_enabled: bool = False):
        return WhatsAppMessagePolicy(
            _test_whatsapp_channel_config(
                auth_dir=Path("/tmp/twinr-whatsapp-auth"),
                worker_root=Path("/tmp/twinr-whatsapp-worker"),
                groups_enabled=groups_enabled,
                self_chat_mode=self_chat_mode,
            )
        )

    def test_text_channel_turn_service_runs_one_turn_and_returns_reply(self) -> None:
        runtime = _FakeRuntime()
        backend = _FakeBackend()
        service = TwinrTextChannelTurnService(runtime=runtime, backend=backend)

        reply = service.handle_inbound(self._message())

        self.assertEqual(reply.text, "Hallo aus Twinr.")
        self.assertEqual(reply.reply_to_message_id, "msg-1")
        self.assertEqual(reply.metadata["provider_response_id"], "resp_123")
        self.assertEqual(runtime.events[0], ("begin_listening", "whatsapp"))
        self.assertEqual(runtime.events[2], ("complete_agent_turn", "Hallo aus Twinr.", "whatsapp", "text"))
        self.assertEqual(runtime.events[-1], ("finish_speaking", "done"))
        self.assertEqual(
            backend.calls,
            [("Hallo Twinr", (("system", "memory context"),), None)],
        )

    def test_text_channel_turn_service_warms_memory_before_live_turns_when_available(self) -> None:
        runtime = _FakeRuntime()
        backend = _FakeBackend()
        warmed = Event()
        runtime.long_term_memory = SimpleNamespace(prewarm_foreground_read_cache=lambda: warmed.set())

        TwinrTextChannelTurnService(runtime=runtime, backend=backend)

        self.assertTrue(warmed.wait(timeout=1.0))

    def test_tool_text_channel_turn_service_uses_tool_loop_for_whatsapp_send(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            auth_dir = root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            auth_dir.mkdir(parents=True, exist_ok=True)
            worker_root.mkdir(parents=True, exist_ok=True)
            (auth_dir / "creds.json").write_text("{\"me\":{}}\n", encoding="utf-8")
            (worker_root / "package.json").write_text("{\"name\":\"worker\"}\n", encoding="utf-8")
            config = _test_twinr_whatsapp_config(root)
            runtime = _FakeToolRuntime(config)
            provider = _FakeToolAgentProvider(config)
            service = TwinrToolTextChannelTurnService(
                runtime=runtime,
                tool_agent_provider=provider,
                print_backend=SimpleNamespace(config=config),
            )

            with patch(
                "twinr.channels.tool_runtime.dispatch_whatsapp_outbound_message",
                return_value=SimpleNamespace(
                    ok=True,
                    status="sent",
                    message_id="wa-tool-1",
                    error_code=None,
                    error=None,
                ),
            ) as dispatch_mock:
                reply = service.handle_inbound(
                    self._message(text="Schreib Anna auf WhatsApp, dass sie den Schlüssel mitbringen soll.")
                )

        self.assertEqual(reply.text, "Ich habe Anna geschrieben.")
        self.assertEqual(reply.metadata["provider_response_id"], "resp_tool_final")
        self.assertEqual(reply.metadata["tool_calls"], "1")
        self.assertEqual(runtime.events[0], ("begin_listening", "whatsapp"))
        self.assertIn(("tool_provider_conversation_context", "done"), runtime.events)
        self.assertEqual(
            runtime.events[-2],
            ("complete_agent_turn", "Ich habe Anna geschrieben.", "whatsapp", "text"),
        )
        self.assertEqual(runtime.events[-1], ("finish_speaking", "done"))
        self.assertEqual(runtime.lookup_calls, [("Anna", None, None)])
        self.assertEqual(provider.start_calls[0]["conversation"], (("system", "tool memory context"),))
        self.assertFalse(provider.start_calls[0]["allow_web_search"])
        self.assertIn(
            "send_whatsapp_message",
            {
                schema["name"]
                for schema in provider.start_calls[0]["tool_schemas"]
                if isinstance(schema, dict) and "name" in schema
            },
        )
        self.assertEqual(provider.continue_calls[0]["continuation_token"], "cont_1")
        self.assertEqual(provider.continue_calls[0]["tool_results"][0].name, "send_whatsapp_message")
        self.assertIn("Because this turn came through WhatsApp", str(provider.start_calls[0]["instructions"]))
        self.assertIn("more detailed and explicit than in voice", str(provider.start_calls[0]["instructions"]))
        dispatch_mock.assert_called_once()

    def test_tool_text_channel_turn_service_uses_direct_whatsapp_dispatcher_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            auth_dir = root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            auth_dir.mkdir(parents=True, exist_ok=True)
            worker_root.mkdir(parents=True, exist_ok=True)
            (auth_dir / "creds.json").write_text("{\"me\":{}}\n", encoding="utf-8")
            (worker_root / "package.json").write_text("{\"name\":\"worker\"}\n", encoding="utf-8")
            config = _test_twinr_whatsapp_config(root)
            runtime = _FakeToolRuntime(config)
            provider = _FakeToolAgentProvider(config)
            direct_dispatch = Mock(
                return_value=WhatsAppOutboundResult.sent(
                    request_id="direct-1",
                    message_id="wa-direct-1",
                )
            )
            service = TwinrToolTextChannelTurnService(
                runtime=runtime,
                tool_agent_provider=provider,
                print_backend=SimpleNamespace(config=config),
                whatsapp_outbound_dispatcher=direct_dispatch,
            )

            with patch(
                "twinr.channels.tool_runtime.dispatch_whatsapp_outbound_message",
                side_effect=AssertionError("queue dispatch must not be used for in-channel WhatsApp sends"),
            ):
                reply = service.handle_inbound(
                    self._message(text="Schreib Anna auf WhatsApp, dass sie den Schlüssel mitbringen soll.")
                )

        self.assertEqual(reply.text, "Ich habe Anna geschrieben.")
        direct_dispatch.assert_called_once_with(
            chat_jid="15555551223@s.whatsapp.net",
            text="Bring bitte den Schlüssel mit.",
            recipient_label="Anna",
            reply_to_message_id=None,
            wait_timeout_s=None,
        )

    def test_tool_text_channel_turn_instructions_describe_shared_memory_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            auth_dir = root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            auth_dir.mkdir(parents=True, exist_ok=True)
            worker_root.mkdir(parents=True, exist_ok=True)
            (auth_dir / "creds.json").write_text("{\"me\":{}}\n", encoding="utf-8")
            (worker_root / "package.json").write_text("{\"name\":\"worker\"}\n", encoding="utf-8")
            config = _test_twinr_whatsapp_config(root)
            runtime = _FakeToolRuntime(config)
            service = TwinrToolTextChannelTurnService(
                runtime=runtime,
                tool_agent_provider=_FakeToolAgentProvider(config),
                print_backend=SimpleNamespace(config=config),
            )

            instructions = service._turn_instructions(
                channel="whatsapp",
                tool_names=("send_whatsapp_message", "schedule_reminder"),
                pending_action=None,
            )

        self.assertIn("shared Twinr long-term memory", instructions)
        self.assertIn("managed user and personality context", instructions)
        self.assertIn("do not claim you only know this chat", instructions)
        self.assertIn("current chat plus relevant shared Twinr memory", instructions)
        self.assertIn("Because this turn came through WhatsApp", instructions)
        self.assertIn("more detailed and explicit than in voice", instructions)
        self.assertIn("Prefer 2 to 5 short sentences", instructions)
        self.assertIn("Do not claim a capability is unavailable", instructions)

    def test_fresh_tool_runtime_can_recall_voice_origin_memory_in_tool_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_recall_limit=4,
                long_term_memory_path=str(root / "state" / "chonkydb"),
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                user_display_name="Erika",
            )
            writer_runtime = TwinrRuntime(config=config)
            try:
                writer_runtime.long_term_memory.object_store.write_snapshot(
                    objects=(
                        LongTermMemoryObjectV1(
                            memory_id="fact:blue_emergency_folder_location",
                            kind="fact",
                            summary="Die blaue Notfallmappe liegt im Flurschrank.",
                            details="Voice-origin fact: Die blaue Notfallmappe liegt im Flurschrank neben den Schluesseln.",
                            source=LongTermSourceRefV1(
                                source_type="conversation_turn",
                                event_ids=("turn:voice:blue-folder",),
                                speaker="user",
                                modality="voice",
                            ),
                            status="active",
                            confidence=0.99,
                            confirmed_by_user=True,
                            slot_key="object:blue_emergency_folder:location",
                            value_key="hall_cupboard",
                        ),
                    ),
                    conflicts=(),
                    archived_objects=(),
                )
            finally:
                writer_runtime.shutdown(timeout_s=1.0)

            reader_runtime = TwinrRuntime(config=config)
            try:
                reader_runtime.last_transcript = "Wo liegt meine blaue Notfallmappe?"
                context = reader_runtime.tool_provider_conversation_context()
            finally:
                reader_runtime.shutdown(timeout_s=1.0)

        system_messages = [content for role, content in context if role == "system"]
        self.assertTrue(
            any(
                "blaue Notfallmappe" in message and "Flurschrank" in message
                for message in system_messages
            )
        )

    def test_send_whatsapp_schema_allows_missing_message_for_bounded_follow_up(self) -> None:
        schema = next(
            item
            for item in build_agent_tool_schemas(("send_whatsapp_message",))
            if item.get("name") == "send_whatsapp_message"
        )

        self.assertEqual(schema["parameters"]["required"], ["name"])
        self.assertIn(
            "If the user has not told you the exact message text yet",
            schema["description"],
        )

    def test_send_whatsapp_handler_ignores_invalid_optional_phone_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            auth_dir = root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            auth_dir.mkdir(parents=True, exist_ok=True)
            worker_root.mkdir(parents=True, exist_ok=True)
            (auth_dir / "creds.json").write_text("{\"me\":{}}\n", encoding="utf-8")
            (worker_root / "package.json").write_text("{\"name\":\"worker\"}\n", encoding="utf-8")
            config = _test_twinr_whatsapp_config(root)
            owner = SimpleNamespace(
                config=config,
                runtime=_FakeToolRuntime(config),
            )

            result = handle_send_whatsapp_message(
                owner,
                {
                    "name": "Anna",
                    "message": "Bring bitte den Schlüssel mit.",
                    "phone_last4": "unknown",
                },
            )

        self.assertEqual(result["status"], "confirmation_required")
        self.assertEqual(result["recipient_label"], "Anna")
        self.assertNotIn("phone_last4", result)

    def test_send_whatsapp_handler_keeps_message_preview_for_contact_clarification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            auth_dir = root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            auth_dir.mkdir(parents=True, exist_ok=True)
            worker_root.mkdir(parents=True, exist_ok=True)
            (auth_dir / "creds.json").write_text("{\"me\":{}}\n", encoding="utf-8")
            (worker_root / "package.json").write_text("{\"name\":\"worker\"}\n", encoding="utf-8")
            config = _test_twinr_whatsapp_config(root)
            runtime = _FakeToolRuntime(config)
            runtime.lookup_contact = Mock(  # type: ignore[method-assign]
                return_value=SimpleNamespace(
                    status="needs_clarification",
                    question="Welche Janina meinst du?",
                    options=(
                        SimpleNamespace(
                            label="Janina Weber",
                            role="Nachbarin",
                            phones=(_TEST_CONTACT_PHONE_A,),
                        ),
                        SimpleNamespace(
                            label="Janina Scholz",
                            role="Physiotherapeutin",
                            phones=(_TEST_CONTACT_PHONE_B,),
                        ),
                    ),
                )
            )
            owner = SimpleNamespace(
                config=config,
                runtime=runtime,
            )

            result = handle_send_whatsapp_message(
                owner,
                {
                    "name": "Janina",
                    "message": "Kannst du mich spaeter anrufen?",
                },
            )

        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["message_preview"], "Kannst du mich spaeter anrufen?")
        self.assertEqual(result["options"][0]["label"], "Janina Weber")

    def test_send_whatsapp_handler_accepts_lookup_role_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            auth_dir = root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            auth_dir.mkdir(parents=True, exist_ok=True)
            worker_root.mkdir(parents=True, exist_ok=True)
            (auth_dir / "creds.json").write_text("{\"me\":{}}\n", encoding="utf-8")
            (worker_root / "package.json").write_text("{\"name\":\"worker\"}\n", encoding="utf-8")
            config = _test_twinr_whatsapp_config(root)
            runtime = _FakeToolRuntime(config)
            runtime.lookup_contact = Mock(  # type: ignore[method-assign]
                return_value=SimpleNamespace(
                    status="ok",
                    match=SimpleNamespace(
                        label="Janina Weber",
                        role="Nachbarin",
                        phones=[_TEST_CONTACT_PHONE_A],
                    ),
                )
            )
            owner = SimpleNamespace(
                config=config,
                runtime=runtime,
            )

            result = handle_send_whatsapp_message(
                owner,
                {
                    "name": "Janina",
                    "lookup_role": "Nachbarin",
                    "message": "Kannst du mich spaeter anrufen?",
                },
            )

        self.assertEqual(result["status"], "confirmation_required")
        runtime.lookup_contact.assert_called_once_with(  # type: ignore[union-attr]
            name="Janina",
            family_name=None,
            role="Nachbarin",
        )

    def test_send_whatsapp_handler_accepts_exact_contact_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            auth_dir = root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            auth_dir.mkdir(parents=True, exist_ok=True)
            worker_root.mkdir(parents=True, exist_ok=True)
            (auth_dir / "creds.json").write_text("{\"me\":{}}\n", encoding="utf-8")
            (worker_root / "package.json").write_text("{\"name\":\"worker\"}\n", encoding="utf-8")
            config = _test_twinr_whatsapp_config(root)
            runtime = _FakeToolRuntime(config)

            def lookup_contact(**kwargs):
                if kwargs.get("contact_label") == "Janina Werner privat":
                    return SimpleNamespace(
                        status="ok",
                        match=SimpleNamespace(
                            label="Janina Werner privat",
                            role="Privat",
                            phones=[_TEST_CONTACT_PHONE_ALT_2],
                        ),
                    )
                return SimpleNamespace(
                    status="needs_clarification",
                    question="Welche Janina meinst du?",
                    options=(
                        SimpleNamespace(
                            label="Janina Werner",
                            role="Bekannte",
                            phones=(_TEST_CONTACT_PHONE_ALT_1,),
                        ),
                        SimpleNamespace(
                            label="Janina Werner privat",
                            role="Privat",
                            phones=(_TEST_CONTACT_PHONE_ALT_2,),
                        ),
                    ),
                )

            runtime.lookup_contact = Mock(side_effect=lookup_contact)  # type: ignore[method-assign]
            owner = SimpleNamespace(
                config=config,
                runtime=runtime,
            )

            result = handle_send_whatsapp_message(
                owner,
                {
                    "name": "Janina",
                    "contact_label": "Janina Werner privat",
                    "message": "Kannst du mich spaeter anrufen?",
                },
            )

        self.assertEqual(result["status"], "confirmation_required")
        self.assertEqual(result["recipient_label"], "Janina Werner privat")
        runtime.lookup_contact.assert_called_once_with(  # type: ignore[union-attr]
            name="Janina",
            family_name=None,
            role=None,
            contact_label="Janina Werner privat",
        )

    def test_whatsapp_loop_direct_in_channel_dispatcher_sends_without_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            auth_dir = root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            auth_dir.mkdir(parents=True, exist_ok=True)
            worker_root.mkdir(parents=True, exist_ok=True)
            (auth_dir / "creds.json").write_text("{\"me\":{}}\n", encoding="utf-8")
            (worker_root / "package.json").write_text("{\"name\":\"worker\"}\n", encoding="utf-8")
            config = _test_twinr_whatsapp_config(root)
            loop = TwinrWhatsAppChannelLoop(
                config=config,
                runtime=_FakeToolRuntime(config),
                backend=_FakeBackend(),
                tool_agent_provider=_FakeToolAgentProvider(config),
                print_backend=SimpleNamespace(config=config),
            )
            loop._transport_open = True
            loop.transport.send_text = Mock(return_value=SimpleNamespace(message_id="wa-direct-2"))  # type: ignore[method-assign]
            result = loop._dispatch_in_channel_outbound_message(
                chat_jid="15555551223@s.whatsapp.net",
                text="Bring bitte den Schlüssel mit.",
                recipient_label="Anna",
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "sent")
        self.assertEqual(result.message_id, "wa-direct-2")
        loop.transport.send_text.assert_called_once_with(  # type: ignore[union-attr]
            chat_jid="15555551223@s.whatsapp.net",
            text="Bring bitte den Schlüssel mit.",
            reply_to_message_id=None,
        )
        self.assertTrue(loop.policy._sent_cache.contains("wa-direct-2"))  # pylint: disable=protected-access

    def test_pending_store_keeps_contact_clarification_options_for_whatsapp_draft(self) -> None:
        store = PendingTextChannelActionStore()
        store.update_from_tool_results(
            "chat-clarify",
            (
                AgentToolResult(
                    call_id="call_1",
                    name="send_whatsapp_message",
                    output={
                        "status": "needs_clarification",
                        "name": "Janina",
                        "question": "Welche Janina meinst du?",
                        "message_preview": "Kannst du mich spaeter anrufen?",
                        "options": [
                            {
                                "label": "Janina Weber",
                                "role": "Nachbarin",
                                "phone_last4_options": ["...2234"],
                            },
                            {
                                "label": "Janina Scholz",
                                "role": "Physiotherapeutin",
                                "phone_last4_options": ["...5678"],
                            },
                        ],
                    },
                    serialized_output="{}",
                ),
            ),
        )

        pending = store.get("chat-clarify")
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.state, "awaiting_contact_clarification")
        self.assertEqual(pending.message_preview, "Kannst du mich spaeter anrufen?")
        self.assertEqual(
            pending.clarification_options,
            (
                "label=Janina Weber; role=Nachbarin; phone_last4_options=...2234",
                "label=Janina Scholz; role=Physiotherapeutin; phone_last4_options=...5678",
            ),
        )
        self.assertIn("Welche Janina meinst du?", pending.system_message())
        self.assertIn("family_name, role, or phone_last4", pending.system_message())
        self.assertIn("Janina Weber", pending.system_message())
        self.assertIn("Stored contact options:", pending.rewrite_prompt("die Nachbarin"))

    def test_pending_store_preserves_whatsapp_draft_on_tool_execution_failure(self) -> None:
        store = PendingTextChannelActionStore()
        store.update_from_tool_results(
            "chat-1",
            (
                AgentToolResult(
                    call_id="call_1",
                    name="send_whatsapp_message",
                    output={
                        "status": "message_required",
                        "name": "Anna",
                        "recipient_label": "Anna",
                        "question": "What should the WhatsApp to Anna say?",
                    },
                    serialized_output="{}",
                ),
            ),
        )

        store.update_from_tool_results(
            "chat-1",
            (
                AgentToolResult(
                    call_id="call_2",
                    name="send_whatsapp_message",
                    output={
                        "status": "error",
                        "error_code": "tool_execution_failed",
                    },
                    serialized_output="{}",
                ),
            ),
        )

        pending = store.get("chat-1")
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.tool_name, "send_whatsapp_message")
        self.assertEqual(pending.state, "awaiting_message")
        self.assertEqual(pending.recipient_label, "Anna")

    def test_pending_store_round_trips_file_backed_contact_clarification_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "pending-actions.json"
            store = PendingTextChannelActionStore(path=state_path)
            store.update_from_tool_results(
                "chat-clarify",
                (
                    AgentToolResult(
                        call_id="call_1",
                        name="send_whatsapp_message",
                        output={
                            "status": "needs_clarification",
                            "name": "Janina",
                            "question": "Welche Janina meinst du?",
                            "options": (
                                {
                                    "label": "Janina Werner",
                                    "role": "Bekannte",
                                    "phone_last4_options": ("0001",),
                                },
                                {
                                    "label": "Janina Werner privat",
                                    "role": "Privat",
                                    "phone_last4_options": ("0002",),
                                },
                            ),
                        },
                        serialized_output="{}",
                    ),
                ),
            )

            reloaded = PendingTextChannelActionStore(path=state_path)
            pending = reloaded.get("chat-clarify")

        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.state, "awaiting_contact_clarification")
        self.assertEqual(pending.name, "Janina")
        self.assertEqual(pending.question, "Welche Janina meinst du?")
        self.assertEqual(
            pending.clarification_options,
            (
                "label=Janina Werner; role=Bekannte; phone_last4_options=0001",
                "label=Janina Werner privat; role=Privat; phone_last4_options=0002",
            ),
        )

    def test_tool_text_channel_turn_service_keeps_pending_whatsapp_draft_across_follow_up_turns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            auth_dir = root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            auth_dir.mkdir(parents=True, exist_ok=True)
            worker_root.mkdir(parents=True, exist_ok=True)
            (auth_dir / "creds.json").write_text("{\"me\":{}}\n", encoding="utf-8")
            (worker_root / "package.json").write_text("{\"name\":\"worker\"}\n", encoding="utf-8")
            config = _test_twinr_whatsapp_config(root)
            runtime = _FakeToolRuntime(config)
            provider = _FakeWhatsAppPendingProvider(config)
            service = TwinrToolTextChannelTurnService(
                runtime=runtime,
                tool_agent_provider=provider,
                print_backend=SimpleNamespace(config=config),
            )

            with patch(
                "twinr.channels.tool_runtime.dispatch_whatsapp_outbound_message",
                return_value=SimpleNamespace(
                    ok=True,
                    status="sent",
                    message_id="wa-tool-2",
                    error_code=None,
                    error=None,
                ),
            ) as dispatch_mock:
                first = service.handle_inbound(
                    self._message(message_id="msg-1", text="Schreib Anna auf WhatsApp.")
                )
                second = service.handle_inbound(
                    self._message(message_id="msg-2", text="Bring bitte den Schlüssel mit.")
                )
                third = service.handle_inbound(
                    self._message(message_id="msg-3", text="Ja")
                )

        self.assertEqual(first.text, "What should the WhatsApp to Anna say?")
        self.assertEqual(second.text, "Should I send this WhatsApp to Anna: Bring bitte den Schlüssel mit.")
        self.assertEqual(third.text, "Ich habe Anna geschrieben.")
        self.assertEqual(len(provider.start_calls), 3)
        self.assertEqual(len(provider.continue_calls), 3)
        self.assertEqual(second.metadata["provider_model"], "gpt-tool-test")
        self.assertEqual(second.metadata["provider_response_id"], "resp_msg_pending_final")
        self.assertEqual(second.metadata["tool_calls"], "1")
        self.assertEqual(third.metadata["provider_model"], "gpt-tool-test")
        self.assertEqual(third.metadata["provider_response_id"], "resp_confirm_pending_final")
        self.assertEqual(third.metadata["tool_calls"], "1")
        dispatch_mock.assert_called_once()

    def test_tool_text_channel_turn_service_restores_pending_whatsapp_draft_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            auth_dir = root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            auth_dir.mkdir(parents=True, exist_ok=True)
            worker_root.mkdir(parents=True, exist_ok=True)
            (auth_dir / "creds.json").write_text("{\"me\":{}}\n", encoding="utf-8")
            (worker_root / "package.json").write_text("{\"name\":\"worker\"}\n", encoding="utf-8")
            config = _test_twinr_whatsapp_config(root, runtime_state_path="state/runtime-state.json")

            first_runtime = _FakeToolRuntime(config)
            first_provider = _FakeWhatsAppPendingProvider(config)
            first_service = TwinrToolTextChannelTurnService(
                runtime=first_runtime,
                tool_agent_provider=first_provider,
                print_backend=SimpleNamespace(config=config),
            )
            first = first_service.handle_inbound(
                self._message(message_id="msg-1", text="Schreib Anna auf WhatsApp.")
            )

            second_runtime = _FakeToolRuntime(config)
            second_provider = _FakeSingleToolCallProvider(
                config,
                tool_call=AgentToolCall(
                    name="send_whatsapp_message",
                    call_id="call_2",
                    arguments={
                        "name": "Anna",
                        "message": "Bring bitte den Schlüssel mit.",
                    },
                ),
            )
            second_service = TwinrToolTextChannelTurnService(
                runtime=second_runtime,
                tool_agent_provider=second_provider,
                print_backend=SimpleNamespace(config=config),
            )
            second = second_service.handle_inbound(
                self._message(message_id="msg-2", text="Bring bitte den Schlüssel mit.")
            )

            third_runtime = _FakeToolRuntime(config)
            third_provider = _FakeSingleToolCallProvider(
                config,
                tool_call=AgentToolCall(
                    name="send_whatsapp_message",
                    call_id="call_3",
                    arguments={
                        "name": "Anna",
                        "message": "Bring bitte den Schlüssel mit.",
                        "confirmed": True,
                    },
                ),
            )
            third_service = TwinrToolTextChannelTurnService(
                runtime=third_runtime,
                tool_agent_provider=third_provider,
                print_backend=SimpleNamespace(config=config),
            )
            with patch(
                "twinr.channels.tool_runtime.dispatch_whatsapp_outbound_message",
                return_value=SimpleNamespace(
                    ok=True,
                    status="sent",
                    message_id="wa-tool-2",
                    error_code=None,
                    error=None,
                ),
            ) as dispatch_mock:
                third = third_service.handle_inbound(
                    self._message(message_id="msg-3", text="Ja")
                )

        self.assertEqual(first.text, "What should the WhatsApp to Anna say?")
        self.assertEqual(second.text, "Should I send this WhatsApp to Anna: Bring bitte den Schlüssel mit.")
        self.assertEqual(third.text, "Ich habe Anna geschrieben.")
        self.assertEqual(len(second_provider.start_calls), 1)
        self.assertEqual(len(third_provider.start_calls), 1)
        self.assertEqual(second.metadata["provider_model"], "gpt-tool-test")
        self.assertEqual(third.metadata["provider_model"], "gpt-tool-test")
        dispatch_mock.assert_called_once()

    def test_tool_text_channel_turn_service_restores_contact_clarification_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            auth_dir = root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            auth_dir.mkdir(parents=True, exist_ok=True)
            worker_root.mkdir(parents=True, exist_ok=True)
            (auth_dir / "creds.json").write_text("{\"me\":{}}\n", encoding="utf-8")
            (worker_root / "package.json").write_text("{\"name\":\"worker\"}\n", encoding="utf-8")
            config = _test_twinr_whatsapp_config(root, runtime_state_path="state/runtime-state.json")

            first_runtime = _FakeToolRuntime(config)
            first_runtime.lookup_contact = Mock(  # type: ignore[method-assign]
                side_effect=(
                    SimpleNamespace(
                        status="needs_clarification",
                        question="Welche Janina meinst du?",
                        options=(
                            SimpleNamespace(
                                label="Janina Werner",
                                role="Bekannte",
                                phones=(_TEST_CONTACT_PHONE_ALT_1,),
                            ),
                            SimpleNamespace(
                                label="Janina Werner privat",
                                role="Privat",
                                phones=(_TEST_CONTACT_PHONE_ALT_2,),
                            ),
                        ),
                    ),
                )
            )
            first_provider = _FakeWhatsAppNameClarificationProvider(config)
            first_service = TwinrToolTextChannelTurnService(
                runtime=first_runtime,
                tool_agent_provider=first_provider,
                print_backend=SimpleNamespace(config=config),
            )
            first = first_service.handle_inbound(
                self._message(message_id="msg-1", text="Schreib Janina auf WhatsApp.")
            )

            second_runtime = _FakeToolRuntime(config)
            second_runtime.lookup_contact = Mock(  # type: ignore[method-assign]
                side_effect=(
                    SimpleNamespace(
                        status="ok",
                        match=SimpleNamespace(
                            label="Janina Werner privat",
                            role="Privat",
                            phones=[_TEST_CONTACT_PHONE_ALT_2],
                        ),
                    ),
                )
            )
            second_provider = _FakeSingleToolCallProvider(
                config,
                tool_call=AgentToolCall(
                    name="send_whatsapp_message",
                    call_id="call_2",
                    arguments={
                        "name": "Janina",
                        "family_name": "Werner privat",
                        "role": "Privat",
                    },
                ),
            )
            second_service = TwinrToolTextChannelTurnService(
                runtime=second_runtime,
                tool_agent_provider=second_provider,
                print_backend=SimpleNamespace(config=config),
            )
            second = second_service.handle_inbound(
                self._message(message_id="msg-2", text="Janina Werner privat")
            )

            third_runtime = _FakeToolRuntime(config)
            third_runtime.lookup_contact = Mock(  # type: ignore[method-assign]
                side_effect=(
                    SimpleNamespace(
                        status="ok",
                        match=SimpleNamespace(
                            label="Janina Werner privat",
                            role="Privat",
                            phones=[_TEST_CONTACT_PHONE_ALT_2],
                        ),
                    ),
                )
            )
            third_provider = _FakeSingleToolCallProvider(
                config,
                tool_call=AgentToolCall(
                    name="send_whatsapp_message",
                    call_id="call_3",
                    arguments={
                        "name": "Janina",
                        "family_name": "Werner privat",
                        "role": "Privat",
                        "message": "Ein Gedicht über Skyla und Uschi",
                    },
                ),
            )
            third_service = TwinrToolTextChannelTurnService(
                runtime=third_runtime,
                tool_agent_provider=third_provider,
                print_backend=SimpleNamespace(config=config),
            )
            third = third_service.handle_inbound(
                self._message(message_id="msg-3", text="Ein Gedicht über Skyla und Uschi")
            )

        self.assertEqual(first.text, "Welche Janina meinst du?")
        self.assertEqual(second.text, "What should the WhatsApp to Janina Werner privat say?")
        self.assertEqual(
            third.text,
            "Should I send this WhatsApp to Janina Werner privat: Ein Gedicht über Skyla und Uschi",
        )
        self.assertEqual(len(second_provider.start_calls), 1)
        self.assertEqual(len(third_provider.start_calls), 1)
        second_runtime.lookup_contact.assert_called_once_with(  # type: ignore[union-attr]
            name="Janina",
            family_name="Werner privat",
            role="Privat",
        )
        third_runtime.lookup_contact.assert_called_once_with(  # type: ignore[union-attr]
            name="Janina",
            family_name="Werner privat",
            role="Privat",
        )

    def test_tool_text_channel_turn_service_resolves_contact_clarification_without_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            auth_dir = root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            auth_dir.mkdir(parents=True, exist_ok=True)
            worker_root.mkdir(parents=True, exist_ok=True)
            (auth_dir / "creds.json").write_text("{\"me\":{}}\n", encoding="utf-8")
            (worker_root / "package.json").write_text("{\"name\":\"worker\"}\n", encoding="utf-8")
            config = _test_twinr_whatsapp_config(root)
            runtime = _FakeToolRuntime(config)
            runtime.lookup_contact = Mock(  # type: ignore[method-assign]
                side_effect=(
                    SimpleNamespace(
                        status="needs_clarification",
                        question="Welche Janina meinst du?",
                        options=(
                            SimpleNamespace(
                                label="Janina Weber",
                                role="Nachbarin",
                                phones=(_TEST_CONTACT_PHONE_A,),
                            ),
                            SimpleNamespace(
                                label="Janina Scholz",
                                role="Physiotherapeutin",
                                phones=(_TEST_CONTACT_PHONE_B,),
                            ),
                        ),
                    ),
                    SimpleNamespace(
                        status="ok",
                        match=SimpleNamespace(
                            label="Janina Weber",
                            role="Nachbarin",
                            phones=[_TEST_CONTACT_PHONE_A],
                        ),
                    ),
                    SimpleNamespace(
                        status="ok",
                        match=SimpleNamespace(
                            label="Janina Weber",
                            role="Nachbarin",
                            phones=[_TEST_CONTACT_PHONE_A],
                        ),
                    ),
                )
            )
            provider = _FakeWhatsAppClarificationProvider(config)
            service = TwinrToolTextChannelTurnService(
                runtime=runtime,
                tool_agent_provider=provider,
                print_backend=SimpleNamespace(config=config),
            )

            with patch(
                "twinr.channels.tool_runtime.dispatch_whatsapp_outbound_message",
                return_value=SimpleNamespace(
                    ok=True,
                    status="sent",
                    message_id="wa-tool-clarify",
                    error_code=None,
                    error=None,
                ),
            ) as dispatch_mock:
                first = service.handle_inbound(
                    self._message(
                        message_id="msg-1",
                        text="Schreib Janina auf WhatsApp, dass sie mich spaeter anrufen soll.",
                    )
                )
                second = service.handle_inbound(
                    self._message(
                        message_id="msg-2",
                        text="die Nachbarin",
                    )
                )
                third = service.handle_inbound(
                    self._message(
                        message_id="msg-3",
                        text="Ja",
                    )
                )

        self.assertEqual(first.text, "Welche Janina meinst du?")
        self.assertEqual(
            second.text,
            "Should I send this WhatsApp to Janina Weber: Kannst du mich spaeter anrufen?",
        )
        self.assertEqual(third.text, "Ich habe Janina Weber geschrieben.")
        self.assertEqual(len(provider.start_calls), 3)
        self.assertEqual(len(provider.continue_calls), 3)
        self.assertEqual(second.metadata["provider_model"], "gpt-tool-test")
        self.assertEqual(third.metadata["provider_model"], "gpt-tool-test")
        runtime.lookup_contact.assert_has_calls(  # type: ignore[union-attr]
            [
                call(name="Janina", family_name=None, role=None),
                call(name="Janina", family_name="Weber", role="Nachbarin"),
                call(name="Janina", family_name="Weber", role="Nachbarin"),
            ]
        )
        dispatch_mock.assert_called_once()

    def test_tool_text_channel_turn_service_prefers_specific_full_name_during_contact_clarification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            auth_dir = root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            auth_dir.mkdir(parents=True, exist_ok=True)
            worker_root.mkdir(parents=True, exist_ok=True)
            (auth_dir / "creds.json").write_text("{\"me\":{}}\n", encoding="utf-8")
            (worker_root / "package.json").write_text("{\"name\":\"worker\"}\n", encoding="utf-8")
            config = _test_twinr_whatsapp_config(root)
            runtime = _FakeToolRuntime(config)
            runtime.lookup_contact = Mock(  # type: ignore[method-assign]
                side_effect=(
                    SimpleNamespace(
                        status="needs_clarification",
                        question="Welche Janina meinst du?",
                        options=(
                            SimpleNamespace(
                                label="Janina Werner",
                                role="Bekannte",
                                phones=(_TEST_CONTACT_PHONE_ALT_1,),
                            ),
                            SimpleNamespace(
                                label="Janina Werner privat",
                                role="Privat",
                                phones=(_TEST_CONTACT_PHONE_ALT_2,),
                            ),
                        ),
                    ),
                    SimpleNamespace(
                        status="ok",
                        match=SimpleNamespace(
                            label="Janina Werner privat",
                            role="Privat",
                            phones=[_TEST_CONTACT_PHONE_ALT_2],
                        ),
                    ),
                    SimpleNamespace(
                        status="ok",
                        match=SimpleNamespace(
                            label="Janina Werner privat",
                            role="Privat",
                            phones=[_TEST_CONTACT_PHONE_ALT_2],
                        ),
                    ),
                    SimpleNamespace(
                        status="ok",
                        match=SimpleNamespace(
                            label="Janina Werner privat",
                            role="Privat",
                            phones=[_TEST_CONTACT_PHONE_ALT_2],
                        ),
                    ),
                )
            )
            provider = _FakeWhatsAppNameClarificationProvider(config)
            service = TwinrToolTextChannelTurnService(
                runtime=runtime,
                tool_agent_provider=provider,
                print_backend=SimpleNamespace(config=config),
            )

            with patch(
                "twinr.channels.tool_runtime.dispatch_whatsapp_outbound_message",
                return_value=SimpleNamespace(
                    ok=True,
                    status="sent",
                    message_id="wa-tool-name-clarify",
                    error_code=None,
                    error=None,
                ),
            ) as dispatch_mock:
                first = service.handle_inbound(
                    self._message(
                        message_id="msg-1",
                        text="Schreib Janina auf WhatsApp.",
                    )
                )
                second = service.handle_inbound(
                    self._message(
                        message_id="msg-2",
                        text="Janina Werner privat bitte",
                    )
                )
                third = service.handle_inbound(
                    self._message(
                        message_id="msg-3",
                        text="Ein Gedicht über Skyla und Uschi",
                    )
                )
                fourth = service.handle_inbound(
                    self._message(
                        message_id="msg-4",
                        text="Du sollst es ihr senden!",
                    )
                )

        self.assertEqual(first.text, "Welche Janina meinst du?")
        self.assertEqual(second.text, "What should the WhatsApp to Janina Werner privat say?")
        self.assertEqual(
            third.text,
            "Should I send this WhatsApp to Janina Werner privat: Ein Gedicht über Skyla und Uschi",
        )
        self.assertEqual(fourth.text, "Ich habe Janina Werner privat geschrieben.")
        self.assertEqual(len(provider.start_calls), 4)
        self.assertEqual(len(provider.continue_calls), 4)
        self.assertEqual(second.metadata["provider_model"], "gpt-tool-test")
        self.assertEqual(third.metadata["provider_model"], "gpt-tool-test")
        self.assertEqual(fourth.metadata["provider_model"], "gpt-tool-test")
        runtime.lookup_contact.assert_has_calls(  # type: ignore[union-attr]
            [
                call(name="Janina", family_name=None, role=None),
                call(name="Janina", family_name="Werner privat", role="Privat"),
                call(name="Janina", family_name="Werner privat", role="Privat"),
                call(name="Janina", family_name="Werner privat", role="Privat"),
            ]
        )
        dispatch_mock.assert_called_once()

    def test_tool_text_channel_turn_service_uses_exact_contact_label_during_clarification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            auth_dir = root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            auth_dir.mkdir(parents=True, exist_ok=True)
            worker_root.mkdir(parents=True, exist_ok=True)
            (auth_dir / "creds.json").write_text("{\"me\":{}}\n", encoding="utf-8")
            (worker_root / "package.json").write_text("{\"name\":\"worker\"}\n", encoding="utf-8")
            config = _test_twinr_whatsapp_config(root)
            runtime = _FakeToolRuntime(config)

            def lookup_contact(**kwargs):
                if kwargs.get("contact_label") == "Janina Werner privat":
                    return SimpleNamespace(
                        status="ok",
                        match=SimpleNamespace(
                            label="Janina Werner privat",
                            role="Privat",
                            phones=[_TEST_CONTACT_PHONE_ALT_2],
                        ),
                    )
                return SimpleNamespace(
                    status="needs_clarification",
                    question="Welche Janina meinst du?",
                    options=(
                        SimpleNamespace(
                            label="Janina Werner",
                            role="Bekannte",
                            phones=(_TEST_CONTACT_PHONE_ALT_1,),
                        ),
                        SimpleNamespace(
                            label="Janina Werner privat",
                            role="Privat",
                            phones=(_TEST_CONTACT_PHONE_ALT_2,),
                        ),
                    ),
                )

            runtime.lookup_contact = Mock(side_effect=lookup_contact)  # type: ignore[method-assign]
            provider = _FakeWhatsAppExactLabelClarificationProvider(config)
            service = TwinrToolTextChannelTurnService(
                runtime=runtime,
                tool_agent_provider=provider,
                print_backend=SimpleNamespace(config=config),
            )

            with patch(
                "twinr.channels.tool_runtime.dispatch_whatsapp_outbound_message",
                return_value=SimpleNamespace(
                    ok=True,
                    status="sent",
                    message_id="wa-tool-exact-label",
                    error_code=None,
                    error=None,
                ),
            ) as dispatch_mock:
                first = service.handle_inbound(
                    self._message(
                        message_id="msg-1",
                        text="Schreib Janina auf WhatsApp.",
                    )
                )
                second = service.handle_inbound(
                    self._message(
                        message_id="msg-2",
                        text="Janina Werner privat",
                    )
                )
                third = service.handle_inbound(
                    self._message(
                        message_id="msg-3",
                        text="Ein Gedicht über Skyla und Uschi",
                    )
                )
                fourth = service.handle_inbound(
                    self._message(
                        message_id="msg-4",
                        text="Ja",
                    )
                )

        self.assertEqual(first.text, "Welche Janina meinst du?")
        self.assertEqual(second.text, "What should the WhatsApp to Janina Werner privat say?")
        self.assertEqual(
            third.text,
            "Should I send this WhatsApp to Janina Werner privat: Ein Gedicht über Skyla und Uschi",
        )
        self.assertEqual(fourth.text, "Ich habe Janina Werner privat geschrieben.")
        runtime.lookup_contact.assert_has_calls(  # type: ignore[union-attr]
            [
                call(name="Janina", family_name=None, role=None),
                call(
                    name="Janina",
                    family_name=None,
                    role=None,
                    contact_label="Janina Werner privat",
                ),
                call(
                    name="Janina",
                    family_name=None,
                    role=None,
                    contact_label="Janina Werner privat",
                ),
                call(
                    name="Janina",
                    family_name=None,
                    role=None,
                    contact_label="Janina Werner privat",
                ),
            ]
        )
        dispatch_mock.assert_called_once()

    def test_policy_rejects_group_messages_when_groups_are_disabled(self) -> None:
        policy = self._policy()

        decision = policy.evaluate(
            self._message(
                conversation_id="12345@g.us",
                sender_id=_TEST_ACCOUNT_JID,
                is_group=True,
            ),
            account_jid=None,
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "groups_disabled")

    def test_policy_rejects_echoed_self_chat_reply(self) -> None:
        policy = self._policy(self_chat_mode=True)
        policy.remember_outbound("msg-echo")

        decision = policy.evaluate(
            self._message(
                message_id="msg-echo",
                is_from_self=True,
            ),
            account_jid=_TEST_ACCOUNT_JID,
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "outbound_echo")

    def test_policy_accepts_new_self_chat_message_when_enabled(self) -> None:
        policy = self._policy(self_chat_mode=True)

        decision = policy.evaluate(
            self._message(
                message_id="msg-self",
                is_from_self=True,
            ),
            account_jid=_TEST_ACCOUNT_JID,
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "self_chat_inbound")

    def test_policy_rejects_duplicate_inbound_message_id(self) -> None:
        policy = self._policy(self_chat_mode=True)
        message = self._message(
            message_id="msg-dup",
            is_from_self=True,
        )

        first = policy.evaluate(
            message,
            account_jid=_TEST_ACCOUNT_JID,
        )
        duplicate = policy.evaluate(
            message,
            account_jid=_TEST_ACCOUNT_JID,
        )

        self.assertTrue(first.accepted)
        self.assertFalse(duplicate.accepted)
        self.assertEqual(duplicate.reason, "duplicate_inbound")

    def test_worker_bridge_rejects_node_versions_below_twenty(self) -> None:
        config = _test_whatsapp_channel_config(
            auth_dir=Path("/tmp/twinr-whatsapp-auth"),
            worker_root=Path("/tmp/twinr-whatsapp-worker"),
        )
        bridge = WhatsAppWorkerBridge(config)

        with patch("twinr.channels.whatsapp.worker_bridge.subprocess.run") as run_mock:
            run_mock.return_value = SimpleNamespace(
                returncode=0,
                stdout="v18.20.8\n",
                stderr="",
            )
            with self.assertRaises(ChannelTransportError):
                bridge._assert_supported_node_runtime()

    def test_probe_worker_dependencies_requires_npm_install_when_node_modules_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            worker_root = Path(temp_dir)
            (worker_root / "package.json").write_text(
                '{"name":"worker","dependencies":{"@whiskeysockets/baileys":"7.0.0-rc.9"}}\n',
                encoding="utf-8",
            )
            (worker_root / "package-lock.json").write_text('{"name":"worker","lockfileVersion":3}\n', encoding="utf-8")

            probe = probe_whatsapp_worker_dependencies(worker_root)

        self.assertFalse(probe.ready)
        self.assertIn("npm ci", probe.detail)
        self.assertIn("@whiskeysockets/baileys", probe.detail)

    def test_ensure_worker_dependencies_runs_npm_ci_and_writes_lock_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            worker_root = Path(temp_dir)
            package_json = worker_root / "package.json"
            package_lock = worker_root / "package-lock.json"
            package_json.write_text(
                '{"name":"worker","dependencies":{"@whiskeysockets/baileys":"7.0.0-rc.9","pino":"^9.7.0"}}\n',
                encoding="utf-8",
            )
            package_lock.write_text('{"name":"worker","lockfileVersion":3}\n', encoding="utf-8")
            npm_binary = worker_root / "npm"
            npm_binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

            def _fake_run(command, **kwargs):
                self.assertEqual(command[:2], [str(npm_binary), "ci"])
                self.assertEqual(
                    kwargs["env"]["PATH"].split(os.pathsep)[0],
                    str((worker_root / "node").parent),
                )
                node_modules = worker_root / "node_modules"
                (node_modules / "@whiskeysockets" / "baileys").mkdir(parents=True, exist_ok=True)
                (node_modules / "pino").mkdir(parents=True, exist_ok=True)
                ((node_modules / "@whiskeysockets" / "baileys") / "package.json").write_text("{}", encoding="utf-8")
                ((node_modules / "pino") / "package.json").write_text("{}", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            probe = ensure_whatsapp_worker_dependencies(
                worker_root=worker_root,
                node_binary=str(worker_root / "node"),
                subprocess_runner=_fake_run,
            )

            marker_path = worker_root / "node_modules" / ".twinr-package-lock.sha256"
            self.assertTrue(probe.ready)
            self.assertTrue(marker_path.is_file())

    def test_worker_bridge_start_ensures_dependencies_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            worker_root = Path(temp_dir)
            auth_dir = worker_root / "auth"
            auth_dir.mkdir(parents=True, exist_ok=True)
            (worker_root / "index.mjs").write_text("console.log('ok')\n", encoding="utf-8")
            config = _test_whatsapp_channel_config(
                auth_dir=auth_dir,
                worker_root=worker_root,
            )
            bridge = WhatsAppWorkerBridge(config)
            fake_process = SimpleNamespace(
                poll=lambda: None,
                stdout=[],
                stderr=[],
                stdin=SimpleNamespace(write=lambda _value: None, flush=lambda: None),
            )

            with patch.object(bridge, "_assert_supported_node_runtime") as assert_runtime_mock, patch(
                "twinr.channels.whatsapp.worker_bridge.ensure_whatsapp_worker_dependencies"
            ) as ensure_mock, patch(
                "twinr.channels.whatsapp.worker_bridge.subprocess.Popen",
                return_value=fake_process,
            ) as popen_mock, patch(
                "twinr.channels.whatsapp.worker_bridge.threading.Thread"
            ) as thread_mock:
                thread_mock.return_value = SimpleNamespace(start=lambda: None)
                bridge.start()

        assert_runtime_mock.assert_called_once_with()
        ensure_mock.assert_called_once()
        popen_mock.assert_called_once()

    def test_from_twinr_config_resolves_relative_paths_against_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            auth_dir = project_root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = project_root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            node_binary = project_root / "state" / "tools" / "node-v20" / "bin" / "node"
            worker_root.mkdir(parents=True, exist_ok=True)
            auth_dir.mkdir(parents=True, exist_ok=True)
            node_binary.parent.mkdir(parents=True, exist_ok=True)
            (worker_root / "package.json").write_text("{\"name\": \"worker\"}\n", encoding="utf-8")
            node_binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

            focused = WhatsAppChannelConfig.from_twinr_config(
                SimpleNamespace(
                    project_root=str(project_root),
                    whatsapp_allow_from=_TEST_ALLOW_FROM,
                    whatsapp_auth_dir="state/channels/whatsapp/auth",
                    whatsapp_worker_root="src/twinr/channels/whatsapp/worker",
                    whatsapp_node_binary="state/tools/node-v20/bin/node",
                    whatsapp_groups_enabled=False,
                    whatsapp_self_chat_mode=True,
                    whatsapp_reconnect_base_delay_s=2.0,
                    whatsapp_reconnect_max_delay_s=30.0,
                    whatsapp_send_timeout_s=20.0,
                    whatsapp_sent_cache_ttl_s=180.0,
                    whatsapp_sent_cache_max_entries=256,
                )
            )

        self.assertEqual(focused.auth_dir, auth_dir)
        self.assertEqual(focused.worker_root, worker_root)
        self.assertEqual(focused.node_binary, str(node_binary))

    def test_from_twinr_config_prefers_staged_local_node_runtime_when_config_uses_default_node(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            auth_dir = project_root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = project_root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            local_runtime = detect_whatsapp_node_runtime_spec(project_root)
            worker_root.mkdir(parents=True, exist_ok=True)
            auth_dir.mkdir(parents=True, exist_ok=True)
            local_runtime.binary_path.parent.mkdir(parents=True, exist_ok=True)
            (worker_root / "package.json").write_text("{\"name\": \"worker\"}\n", encoding="utf-8")
            local_runtime.binary_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

            focused = WhatsAppChannelConfig.from_twinr_config(
                SimpleNamespace(
                    project_root=str(project_root),
                    whatsapp_allow_from=_TEST_ALLOW_FROM,
                    whatsapp_auth_dir="state/channels/whatsapp/auth",
                    whatsapp_worker_root="src/twinr/channels/whatsapp/worker",
                    whatsapp_node_binary="node",
                    whatsapp_groups_enabled=False,
                    whatsapp_self_chat_mode=True,
                    whatsapp_reconnect_base_delay_s=2.0,
                    whatsapp_reconnect_max_delay_s=30.0,
                    whatsapp_send_timeout_s=20.0,
                    whatsapp_sent_cache_ttl_s=180.0,
                    whatsapp_sent_cache_max_entries=256,
                )
            )

        self.assertEqual(focused.node_binary, str(local_runtime.binary_path))

    def test_resolve_whatsapp_node_binary_returns_explicit_relative_project_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            resolved = resolve_whatsapp_node_binary(project_root, "state/tools/node-v20/bin/node")

        self.assertEqual(str(project_root / "state" / "tools" / "node-v20" / "bin" / "node"), resolved)

    def test_worker_bridge_parses_qr_svg_status_payload(self) -> None:
        config = _test_whatsapp_channel_config(
            auth_dir=Path("/tmp/twinr-whatsapp-auth"),
            worker_root=Path("/tmp/twinr-whatsapp-worker"),
        )
        bridge = WhatsAppWorkerBridge(config)

        bridge._dispatch_worker_payload(
            {
                "type": "status",
                "connection": "qr",
                "qr_available": True,
                "qr_svg": "<svg viewBox='0 0 10 10'></svg>",
                "qr_image_data_url": "data:image/png;base64,AAAA",
            }
        )

        event = bridge.next_event(timeout_s=0.01)
        self.assertIsInstance(event, WhatsAppWorkerStatusEvent)
        assert isinstance(event, WhatsAppWorkerStatusEvent)
        self.assertTrue(event.qr_available)
        self.assertEqual(event.qr_svg, "<svg viewBox='0 0 10 10'></svg>")
        self.assertEqual(event.qr_image_data_url, "data:image/png;base64,AAAA")

    def test_outbound_queue_round_trips_request_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = WhatsAppOutboundQueue(Path(temp_dir) / "outbound")

            request = queue.submit_request(
                chat_jid=_TEST_ACCOUNT_JID,
                text="Hallo Anna",
                recipient_label="Anna",
                expires_in_s=15.0,
            )
            claimed = queue.claim_next_request()
            assert claimed is not None
            queue.complete_request(
                WhatsAppOutboundResult.sent(
                    request_id=request.request_id,
                    message_id="wa-msg-1",
                )
            )
            result = queue.wait_for_result(request.request_id, timeout_s=0.01)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(claimed.request_id, request.request_id)
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "sent")
        self.assertEqual(result.message_id, "wa-msg-1")

    def test_history_import_queue_round_trips_request_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = WhatsAppHistoryImportQueue(Path(temp_dir) / "history_import")

            request = queue.submit_request(
                source="whatsapp",
                lookback_key="3m",
                expires_in_s=45.0,
            )
            claimed = queue.claim_next_request()
            assert claimed is not None
            queue.complete_request(
                WhatsAppHistoryImportResult.completed(
                    request_id=request.request_id,
                    status="completed",
                    detail="done",
                    imported_messages=12,
                    imported_turns=5,
                    imported_chats=2,
                    oldest_at="2026-03-01T10:00:00+00:00",
                    newest_at="2026-03-25T10:00:00+00:00",
                )
            )
            result = queue.wait_for_result(request.request_id, timeout_s=0.01)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(claimed.request_id, request.request_id)
        self.assertTrue(result.ok)
        self.assertEqual(result.imported_turns, 5)

    def test_worker_bridge_parses_history_batch_payload(self) -> None:
        config = _test_whatsapp_channel_config(
            auth_dir=Path("/tmp/twinr-whatsapp-auth"),
            worker_root=Path("/tmp/twinr-whatsapp-worker"),
        )
        bridge = WhatsAppWorkerBridge(config, history_sync_enabled=True)

        bridge._dispatch_worker_payload(
            {
                "type": "history_batch",
                "sync_type": "FULL",
                "progress": 88,
                "is_latest": True,
                "messages": [
                    {
                        "message_id": "hist-1",
                        "conversation_id": _TEST_ACCOUNT_JID,
                        "sender_id": _TEST_ACCOUNT_JID,
                        "text": "Hallo von früher",
                        "timestamp_ms": 1710000000000,
                        "is_group": False,
                        "is_from_self": True,
                        "chat_label": "Self chat",
                        "sender_label": "Thomas",
                    }
                ],
            }
        )

        event = bridge.next_event(timeout_s=0.01)
        self.assertIsInstance(event, WhatsAppWorkerHistoryBatchEvent)
        assert isinstance(event, WhatsAppWorkerHistoryBatchEvent)
        self.assertEqual(event.sync_type, "FULL")
        self.assertEqual(event.progress, 88)
        self.assertEqual(len(event.messages), 1)
        self.assertEqual(event.messages[0].sender_label, "Thomas")

    def test_worker_bridge_parses_incoming_worker_provenance_fields(self) -> None:
        config = _test_whatsapp_channel_config(
            auth_dir=Path("/tmp/twinr-whatsapp-auth"),
            worker_root=Path("/tmp/twinr-whatsapp-worker"),
        )
        bridge = WhatsAppWorkerBridge(config)

        bridge._dispatch_worker_payload(
            {
                "type": "incoming_message",
                "message_id": "msg-debug",
                "conversation_id": _TEST_ACCOUNT_JID,
                "sender_id": _TEST_ACCOUNT_JID,
                "text": "Hallo Twinr",
                "received_at": "2026-03-26T15:00:00Z",
                "is_group": False,
                "is_from_self": False,
                "account_jid": _TEST_ACCOUNT_JID,
                "upsert_type": "append",
                "worker_request_id": "req-1",
                "raw_remote_jid": "15555554567:18@s.whatsapp.net",
                "raw_remote_jid_alt": "15555554567@lid",
                "raw_participant": "15555554567:18@s.whatsapp.net",
                "raw_participant_alt": "15555554567@lid",
                "message_timestamp": "1774539523",
                "context_stanza_id": "stanza-1",
            }
        )

        event = bridge.next_event(timeout_s=0.01)
        self.assertIsInstance(event, ChannelInboundMessage)
        assert isinstance(event, ChannelInboundMessage)
        self.assertEqual(event.metadata["worker_upsert_type"], "append")
        self.assertEqual(event.metadata["worker_request_id"], "req-1")
        self.assertEqual(event.metadata["worker_remote_jid"], "15555554567:18@s.whatsapp.net")
        self.assertEqual(event.metadata["worker_remote_jid_alt"], "15555554567@lid")
        self.assertEqual(event.metadata["worker_participant"], "15555554567:18@s.whatsapp.net")
        self.assertEqual(event.metadata["worker_participant_alt"], "15555554567@lid")
        self.assertEqual(event.metadata["worker_message_timestamp"], "1774539523")
        self.assertEqual(event.metadata["worker_context_stanza_id"], "stanza-1")

    def test_inbound_event_log_records_redacted_worker_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit = WhatsAppInboundEventLog(Path(temp_dir) / "debug")
            message = ChannelInboundMessage(
                channel="whatsapp",
                message_id="msg-1",
                conversation_id=_TEST_ACCOUNT_JID,
                sender_id=_TEST_ACCOUNT_JID,
                text="Bitte schreibe Janina Werner privat.",
                received_at="2026-03-26T15:38:43Z",
                is_group=False,
                is_from_self=False,
                metadata={
                    "account_jid": _TEST_ACCOUNT_JID,
                    "worker_upsert_type": "append",
                    "worker_request_id": "req-123",
                    "worker_remote_jid": "15555554567:18@s.whatsapp.net",
                    "worker_remote_jid_alt": "15555554567@lid",
                    "worker_participant": "15555554567:18@s.whatsapp.net",
                    "worker_participant_alt": "15555554567@lid",
                    "worker_message_timestamp": "1774539523",
                    "worker_context_stanza_id": "stanza-123",
                },
            )

            audit.record_received(message)
            audit.record_policy_decision(message, accepted=True, reason="self_chat_inbound")
            audit.record_delivery(message, outbound_message_id="wa-1", reply_text="Erledigt.")

            lines = (Path(temp_dir) / "debug" / "inbound_events.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 3)
        self.assertNotIn("Bitte schreibe Janina Werner privat.", lines[0])
        self.assertIn("\"text_sha256\":", lines[0])
        self.assertIn("\"worker_upsert_type\": \"append\"", lines[0])
        self.assertIn("\"worker_request_id\": \"req-123\"", lines[0])
        self.assertIn("\"worker_message_timestamp\": \"1774539523\"", lines[0])

    def test_build_import_turns_groups_user_authored_history_and_skips_contact_only_leads(self) -> None:
        now_ms = int(time.time() * 1000)
        turns = build_import_turns(
            (
                WhatsAppWorkerHistoryMessage(
                    message_id="c1",
                    conversation_id="anna@s.whatsapp.net",
                    sender_id="anna@s.whatsapp.net",
                    text="Bist du schon unterwegs?",
                    timestamp_ms=now_ms - 3000,
                    is_from_self=False,
                    sender_label="Anna",
                ),
                WhatsAppWorkerHistoryMessage(
                    message_id="u1",
                    conversation_id="anna@s.whatsapp.net",
                    sender_id="me@s.whatsapp.net",
                    text="Ich fahre in zehn Minuten los.",
                    timestamp_ms=now_ms - 2000,
                    is_from_self=True,
                    sender_label="Ich",
                ),
                WhatsAppWorkerHistoryMessage(
                    message_id="u2",
                    conversation_id="anna@s.whatsapp.net",
                    sender_id="me@s.whatsapp.net",
                    text="Bring bitte den Schlüssel mit.",
                    timestamp_ms=now_ms - 1900,
                    is_from_self=True,
                    sender_label="Ich",
                ),
                WhatsAppWorkerHistoryMessage(
                    message_id="a1",
                    conversation_id="anna@s.whatsapp.net",
                    sender_id="anna@s.whatsapp.net",
                    text="Mach ich.",
                    timestamp_ms=now_ms - 1000,
                    is_from_self=False,
                    sender_label="Anna",
                ),
            ),
            lookback_key="4w",
        )

        self.assertEqual(len(turns), 1)
        self.assertIn("Ich fahre in zehn Minuten los.", turns[0].transcript)
        self.assertIn("Bring bitte den Schlüssel mit.", turns[0].transcript)
        self.assertIn("WhatsApp contact Anna replied: Mach ich.", turns[0].response)

    def test_outbound_queue_marks_stale_processing_requests_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = WhatsAppOutboundQueue(Path(temp_dir) / "outbound")
            request = queue.submit_request(
                chat_jid=_TEST_ACCOUNT_JID,
                text="Hallo Anna",
                recipient_label="Anna",
                expires_in_s=15.0,
            )
            claimed = queue.claim_next_request()
            assert claimed is not None

            failed = queue.fail_stale_processing_requests()
            result = queue.wait_for_result(request.request_id, timeout_s=0.01)

        self.assertEqual(failed, 1)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "unknown_after_restart")

    def test_loop_processes_queued_outbound_message_through_transport(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            auth_dir = project_root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = project_root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            auth_dir.mkdir(parents=True, exist_ok=True)
            worker_root.mkdir(parents=True, exist_ok=True)
            (worker_root / "package.json").write_text("{\"name\":\"worker\"}\n", encoding="utf-8")

            loop = TwinrWhatsAppChannelLoop(
                config=SimpleNamespace(
                    project_root=str(project_root),
                    whatsapp_allow_from=_TEST_ALLOW_FROM,
                    whatsapp_auth_dir=str(auth_dir),
                    whatsapp_worker_root=str(worker_root),
                    whatsapp_node_binary="node",
                    whatsapp_groups_enabled=False,
                    whatsapp_self_chat_mode=True,
                    whatsapp_reconnect_base_delay_s=2.0,
                    whatsapp_reconnect_max_delay_s=30.0,
                    whatsapp_send_timeout_s=20.0,
                    whatsapp_sent_cache_ttl_s=180.0,
                    whatsapp_sent_cache_max_entries=256,
                ),
                runtime=_FakeRuntime(),
                backend=_FakeBackend(),
            )
            loop.transport.send_text = Mock(
                return_value=SimpleNamespace(
                    message_id="wa-msg-2",
                )
            )
            loop._transport_open = True
            request = loop.outbound_queue.submit_request(
                chat_jid=_TEST_ACCOUNT_JID,
                text="Hallo von Twinr",
                recipient_label="Anna",
                expires_in_s=15.0,
            )

            processed = loop._process_outbound_queue_once()
            result = loop.outbound_queue.wait_for_result(request.request_id, timeout_s=0.01)

        self.assertTrue(processed)
        loop.transport.send_text.assert_called_once_with(
            chat_jid=_TEST_ACCOUNT_JID,
            text="Hallo von Twinr",
            reply_to_message_id=None,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.ok)
        self.assertEqual(result.message_id, "wa-msg-2")

    def test_runtime_availability_hides_whatsapp_send_tool_until_session_is_paired(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        f"TWINR_WHATSAPP_ALLOW_FROM={_TEST_ALLOW_FROM}",
                        "TWINR_WHATSAPP_AUTH_DIR=state/channels/whatsapp/auth",
                        "TWINR_WHATSAPP_WORKER_ROOT=src/twinr/channels/whatsapp/worker",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            worker_root = root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            auth_dir = root / "state" / "channels" / "whatsapp" / "auth"
            worker_root.mkdir(parents=True, exist_ok=True)
            auth_dir.mkdir(parents=True, exist_ok=True)
            (worker_root / "package.json").write_text("{\"name\": \"worker\"}\n", encoding="utf-8")

            config = TwinrConfig.from_env(env_path)
            hidden = available_realtime_tool_names(
                config,
                tool_names=("lookup_contact", "send_whatsapp_message"),
            )

            (auth_dir / "creds.json").write_text("{\"me\":{}}\n", encoding="utf-8")
            shown = available_realtime_tool_names(
                config,
                tool_names=("lookup_contact", "send_whatsapp_message"),
            )

        self.assertEqual(hidden, ("lookup_contact",))
        self.assertEqual(shown, ("lookup_contact", "send_whatsapp_message"))

    def test_pairing_coordinator_marks_open_session_as_paired(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        project_root = Path(temp_dir.name)
        auth_dir = project_root / "state" / "channels" / "whatsapp" / "auth"
        worker_root = project_root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
        worker_root.mkdir(parents=True, exist_ok=True)
        (worker_root / "package.json").write_text("{\"name\": \"worker\"}\n", encoding="utf-8")

        coordinator = WhatsAppPairingCoordinator(
            store=FileBackedChannelOnboardingStore.from_project_root(project_root, channel_id="whatsapp"),
            registry=InProcessChannelPairingRegistry(),
        )
        config = SimpleNamespace(
            whatsapp_allow_from=_TEST_ALLOW_FROM,
            whatsapp_auth_dir=str(auth_dir),
            whatsapp_worker_root=str(worker_root),
            whatsapp_node_binary="node",
            whatsapp_groups_enabled=False,
            whatsapp_self_chat_mode=True,
            whatsapp_reconnect_base_delay_s=2.0,
            whatsapp_reconnect_max_delay_s=30.0,
            whatsapp_send_timeout_s=20.0,
            whatsapp_sent_cache_ttl_s=180.0,
            whatsapp_sent_cache_max_entries=256,
        )
        events = [
            WhatsAppWorkerStatusEvent(connection="booting", detail="worker_ready"),
            WhatsAppWorkerStatusEvent(connection="connecting"),
            WhatsAppWorkerStatusEvent(connection="open", account_jid=_TEST_ACCOUNT_JID),
        ]

        with patch(
            "twinr.channels.whatsapp.pairing.WhatsAppWorkerBridge",
            side_effect=lambda bridge_config: _SequencedWorkerBridge(bridge_config, events),
        ):
            coordinator._run_pairing(config)

        snapshot = coordinator.store.load()
        self.assertTrue(snapshot.paired)
        self.assertFalse(snapshot.running)
        self.assertEqual(snapshot.summary, "Paired")
        self.assertEqual(snapshot.account_id, _TEST_ACCOUNT_JID)

    def test_pairing_coordinator_flags_bad_session_for_auth_repair(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        project_root = Path(temp_dir.name)
        auth_dir = project_root / "state" / "channels" / "whatsapp" / "auth"
        worker_root = project_root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
        worker_root.mkdir(parents=True, exist_ok=True)
        (worker_root / "package.json").write_text("{\"name\": \"worker\"}\n", encoding="utf-8")

        coordinator = WhatsAppPairingCoordinator(
            store=FileBackedChannelOnboardingStore.from_project_root(project_root, channel_id="whatsapp"),
            registry=InProcessChannelPairingRegistry(),
        )
        config = SimpleNamespace(
            whatsapp_allow_from=_TEST_ALLOW_FROM,
            whatsapp_auth_dir=str(auth_dir),
            whatsapp_worker_root=str(worker_root),
            whatsapp_node_binary="node",
            whatsapp_groups_enabled=False,
            whatsapp_self_chat_mode=True,
            whatsapp_reconnect_base_delay_s=2.0,
            whatsapp_reconnect_max_delay_s=30.0,
            whatsapp_send_timeout_s=20.0,
            whatsapp_sent_cache_ttl_s=180.0,
            whatsapp_sent_cache_max_entries=256,
        )
        events = [
            WhatsAppWorkerStatusEvent(connection="booting", detail="worker_ready"),
            WhatsAppWorkerStatusEvent(connection="close", detail="badSession", fatal=True),
        ]

        with patch(
            "twinr.channels.whatsapp.pairing.WhatsAppWorkerBridge",
            side_effect=lambda bridge_config: _SequencedWorkerBridge(bridge_config, events),
        ):
            coordinator._run_pairing(config)

        snapshot = coordinator.store.load()
        self.assertTrue(snapshot.fatal)
        self.assertTrue(snapshot.auth_repair_needed)
        self.assertEqual(snapshot.summary, "Auth repair needed")
        self.assertEqual(snapshot.last_worker_detail, "badSession")

    def test_probe_whatsapp_runtime_reports_missing_worker_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            auth_dir = project_root / "state" / "channels" / "whatsapp" / "auth"
            worker_root = project_root / "src" / "twinr" / "channels" / "whatsapp" / "worker"
            worker_root.mkdir(parents=True, exist_ok=True)
            auth_dir.mkdir(parents=True, exist_ok=True)
            (worker_root / "index.mjs").write_text("console.log('worker')\n", encoding="utf-8")
            (worker_root / "package.json").write_text(
                '{"name":"worker","dependencies":{"@whiskeysockets/baileys":"7.0.0-rc.9"}}\n',
                encoding="utf-8",
            )
            (worker_root / "package-lock.json").write_text('{"name":"worker","lockfileVersion":3}\n', encoding="utf-8")

            probe = probe_whatsapp_runtime(
                SimpleNamespace(
                    project_root=str(project_root),
                    whatsapp_worker_root=str(worker_root),
                    whatsapp_node_binary="node",
                    whatsapp_auth_dir=str(auth_dir),
                ),
                env_path=project_root / ".env",
            )

        self.assertFalse(probe.worker_ready)
        self.assertIn("npm ci", probe.worker_detail)
