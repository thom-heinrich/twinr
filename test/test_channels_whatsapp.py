from pathlib import Path
from threading import Event
from types import SimpleNamespace
import sys
import tempfile
from unittest.mock import patch
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.state.machine import TwinrStatus
from twinr.channels import ChannelInboundMessage, ChannelTransportError, TwinrTextChannelTurnService
from twinr.channels.whatsapp import WhatsAppChannelConfig, WhatsAppMessagePolicy
from twinr.channels.whatsapp.node_runtime import detect_whatsapp_node_runtime_spec, resolve_whatsapp_node_binary
from twinr.channels.whatsapp.worker_bridge import WhatsAppWorkerBridge, WhatsAppWorkerStatusEvent
from twinr.web.support.channel_onboarding import FileBackedChannelOnboardingStore, InProcessChannelPairingRegistry
from twinr.web.support.whatsapp import WhatsAppPairingCoordinator


class _FakeRuntime:
    def __init__(self) -> None:
        self.status = TwinrStatus.WAITING
        self.events: list[tuple[str, object]] = []
        self.long_term_memory = None

    def begin_listening(self, *, request_source: str, button=None, proactive_trigger=None):
        self.events.append(("begin_listening", request_source))
        self.status = TwinrStatus.LISTENING
        return self.status

    def submit_transcript(self, transcript: str):
        self.events.append(("submit_transcript", transcript))
        self.status = TwinrStatus.PROCESSING
        return self.status

    def provider_conversation_context(self):
        return (("system", "memory context"),)

    def complete_agent_turn(self, answer: str):
        self.events.append(("complete_agent_turn", answer))
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
            "conversation_id": "491711234567@s.whatsapp.net",
            "sender_id": "491711234567@s.whatsapp.net",
            "text": "Hallo Twinr",
        }
        payload.update(overrides)
        return ChannelInboundMessage(**payload)

    def _policy(self, *, self_chat_mode: bool = False, groups_enabled: bool = False):
        return WhatsAppMessagePolicy(
            WhatsAppChannelConfig(
                allow_from="+49 171 1234567",
                allow_from_jid="491711234567@s.whatsapp.net",
                auth_dir=Path("/tmp/twinr-whatsapp-auth"),
                worker_root=Path("/tmp/twinr-whatsapp-worker"),
                node_binary="node",
                groups_enabled=groups_enabled,
                self_chat_mode=self_chat_mode,
                reconnect_base_delay_s=2.0,
                reconnect_max_delay_s=30.0,
                send_timeout_s=20.0,
                sent_cache_ttl_s=180.0,
                sent_cache_max_entries=256,
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

    def test_policy_rejects_group_messages_when_groups_are_disabled(self) -> None:
        policy = self._policy()

        decision = policy.evaluate(
            self._message(
                conversation_id="12345@g.us",
                sender_id="491711234567@s.whatsapp.net",
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
            account_jid="491711234567@s.whatsapp.net",
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
            account_jid="491711234567@s.whatsapp.net",
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "self_chat_inbound")

    def test_worker_bridge_rejects_node_versions_below_twenty(self) -> None:
        config = WhatsAppChannelConfig(
            allow_from="+49 171 1234567",
            allow_from_jid="491711234567@s.whatsapp.net",
            auth_dir=Path("/tmp/twinr-whatsapp-auth"),
            worker_root=Path("/tmp/twinr-whatsapp-worker"),
            node_binary="node",
            groups_enabled=False,
            self_chat_mode=False,
            reconnect_base_delay_s=2.0,
            reconnect_max_delay_s=30.0,
            send_timeout_s=20.0,
            sent_cache_ttl_s=180.0,
            sent_cache_max_entries=256,
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
                    whatsapp_allow_from="+491711234567",
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
                    whatsapp_allow_from="+491711234567",
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
        config = WhatsAppChannelConfig(
            allow_from="+49 171 1234567",
            allow_from_jid="491711234567@s.whatsapp.net",
            auth_dir=Path("/tmp/twinr-whatsapp-auth"),
            worker_root=Path("/tmp/twinr-whatsapp-worker"),
            node_binary="node",
            groups_enabled=False,
            self_chat_mode=False,
            reconnect_base_delay_s=2.0,
            reconnect_max_delay_s=30.0,
            send_timeout_s=20.0,
            sent_cache_ttl_s=180.0,
            sent_cache_max_entries=256,
        )
        bridge = WhatsAppWorkerBridge(config)

        bridge._dispatch_worker_payload(
            {
                "type": "status",
                "connection": "qr",
                "qr_available": True,
                "qr_svg": "<svg viewBox='0 0 10 10'></svg>",
            }
        )

        event = bridge.next_event(timeout_s=0.01)
        self.assertIsInstance(event, WhatsAppWorkerStatusEvent)
        assert isinstance(event, WhatsAppWorkerStatusEvent)
        self.assertTrue(event.qr_available)
        self.assertEqual(event.qr_svg, "<svg viewBox='0 0 10 10'></svg>")

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
            whatsapp_allow_from="+491711234567",
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
            WhatsAppWorkerStatusEvent(connection="open", account_jid="491711234567@s.whatsapp.net"),
        ]

        with patch(
            "twinr.web.support.whatsapp.WhatsAppWorkerBridge",
            side_effect=lambda bridge_config: _SequencedWorkerBridge(bridge_config, events),
        ):
            coordinator._run_pairing(config)

        snapshot = coordinator.store.load()
        self.assertTrue(snapshot.paired)
        self.assertFalse(snapshot.running)
        self.assertEqual(snapshot.summary, "Paired")
        self.assertEqual(snapshot.account_id, "491711234567@s.whatsapp.net")

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
            whatsapp_allow_from="+491711234567",
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
            "twinr.web.support.whatsapp.WhatsAppWorkerBridge",
            side_effect=lambda bridge_config: _SequencedWorkerBridge(bridge_config, events),
        ):
            coordinator._run_pairing(config)

        snapshot = coordinator.store.load()
        self.assertTrue(snapshot.fatal)
        self.assertTrue(snapshot.auth_repair_needed)
        self.assertEqual(snapshot.summary, "Auth repair needed")
        self.assertEqual(snapshot.last_worker_detail, "badSession")
