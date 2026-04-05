from pathlib import Path
from tempfile import TemporaryFile
from threading import Event, Lock, Thread
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.playback_coordinator import PlaybackActivityEvent
from twinr.agent.workflows.respeaker_duplex_keepalive import (
    build_respeaker_duplex_keepalive,
    respeaker_duplex_keepalive_supported,
)
from twinr.agent.workflows.voice_orchestrator import (
    EdgeVoiceOrchestrator,
    _CaptureStreamStalledError,
)
from twinr.orchestrator.voice_contracts import (
    OrchestratorVoiceErrorEvent,
    OrchestratorVoiceFollowUpClosedEvent,
    OrchestratorVoiceIdentityProfile,
    OrchestratorVoiceIdentityProfilesEvent,
    OrchestratorVoiceReadyEvent,
    OrchestratorVoiceTranscriptCommittedEvent,
)


class _FakeVoiceClient:
    def __init__(self) -> None:
        self.open_calls = 0
        self.close_calls = 0
        self.hello_requests: list[Any] = []
        self.runtime_states: list[Any] = []
        self.audio_frames: list[Any] = []
        self.identity_profiles: list[Any] = []
        self.fail_audio_sends = 0
        self.fail_open_calls = 0

    def open(self):
        self.open_calls += 1
        if self.fail_open_calls > 0:
            self.fail_open_calls -= 1
            raise ConnectionError("unavailable")
        return self

    def close(self) -> None:
        self.close_calls += 1

    def send_hello(self, request) -> None:
        self.hello_requests.append(request)

    def send_runtime_state(self, event) -> None:
        self.runtime_states.append(event)

    def send_audio_frame(self, frame) -> None:
        if self.fail_audio_sends > 0:
            self.fail_audio_sends -= 1
            raise ConnectionError("closed")
        self.audio_frames.append(frame)

    def send_identity_profiles(self, event) -> None:
        self.identity_profiles.append(event)


class _FakeCaptureProcess:
    def __init__(self, *, returncode: int | None = None) -> None:
        self.stdout = TemporaryFile()
        self.stderr = TemporaryFile()
        self.returncode = returncode
        self.terminate_calls = 0
        self.wait_calls = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls += 1
        del timeout
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class EdgeVoiceOrchestratorTests(unittest.TestCase):
    def _make_orchestrator(self):
        lines: list[str] = []
        committed: list[tuple[str, str]] = []
        orchestrator = EdgeVoiceOrchestrator(
            TwinrConfig(
                voice_orchestrator_ws_url="ws://127.0.0.1:8797/ws/orchestrator/voice",
            ),
            emit=lines.append,
            on_voice_activation=lambda match: True,
            on_transcript_committed=lambda transcript, source: committed.append((transcript, source)) or True,
            on_barge_in_interrupt=lambda: True,
        )
        fake_client = _FakeVoiceClient()
        orchestrator._client = cast(Any, fake_client)
        return orchestrator, fake_client, lines, committed

    def test_generic_voice_device_alias_resolves_to_specific_capture_device(self) -> None:
        orchestrator = EdgeVoiceOrchestrator(
            TwinrConfig(
                voice_orchestrator_ws_url="ws://127.0.0.1:8797/ws/orchestrator/voice",
                voice_orchestrator_audio_device="default",
                audio_input_device="sysdefault:CARD=Array",
            ),
            emit=lambda _msg: None,
            on_voice_activation=lambda match: True,
            on_transcript_committed=lambda transcript, source: True,
            on_barge_in_interrupt=lambda: True,
        )

        self.assertEqual(orchestrator._device, "sysdefault:CARD=Array")

    def test_specific_voice_device_is_still_respected(self) -> None:
        orchestrator = EdgeVoiceOrchestrator(
            TwinrConfig(
                voice_orchestrator_ws_url="ws://127.0.0.1:8797/ws/orchestrator/voice",
                voice_orchestrator_audio_device="hw:CARD=Array,DEV=0",
                audio_input_device="sysdefault:CARD=Array",
            ),
            emit=lambda _msg: None,
            on_voice_activation=lambda match: True,
            on_transcript_committed=lambda transcript, source: True,
            on_barge_in_interrupt=lambda: True,
        )

        self.assertEqual(orchestrator._device, "hw:CARD=Array,DEV=0")

    def test_missing_voice_device_still_falls_back_to_specific_input_device(self) -> None:
        orchestrator = EdgeVoiceOrchestrator(
            TwinrConfig(
                voice_orchestrator_ws_url="ws://127.0.0.1:8797/ws/orchestrator/voice",
                voice_orchestrator_audio_device="",
                audio_input_device="sysdefault:CARD=Array",
            ),
            emit=lambda _msg: None,
            on_voice_activation=lambda match: True,
            on_transcript_committed=lambda transcript, source: True,
            on_barge_in_interrupt=lambda: True,
        )

        self.assertEqual(orchestrator._device, "sysdefault:CARD=Array")

    def test_send_frame_attaches_speech_probability_side_channel(self) -> None:
        orchestrator, fake_client, _lines, _committed = self._make_orchestrator()
        orchestrator._connected = True
        orchestrator._frame_speech_annotator = cast(
            Any,
            SimpleNamespace(
                classify_frame=lambda pcm_bytes, stream_ended=False: SimpleNamespace(
                    speech_probability=0.81
                )
            ),
        )

        sent = orchestrator._send_frame_now(b"\x01\x00" * 1600)

        self.assertTrue(sent)
        self.assertEqual(len(fake_client.audio_frames), 1)
        self.assertEqual(fake_client.audio_frames[0].speech_probability, 0.81)

    def test_respeaker_duplex_keepalive_supports_twinr_owned_playback_pcm(self) -> None:
        self.assertTrue(
            respeaker_duplex_keepalive_supported(
                capture_device="plughw:CARD=Array,DEV=0",
                playback_device="twinr_playback_softvol",
            )
        )

    def test_respeaker_duplex_keepalive_skips_non_respeaker_capture(self) -> None:
        self.assertFalse(
            respeaker_duplex_keepalive_supported(
                capture_device="default",
                playback_device="twinr_playback_softvol",
            )
        )

    def test_build_duplex_keepalive_prefers_direct_guard_for_softvol_without_coordinator(self) -> None:
        keepalive = build_respeaker_duplex_keepalive(
            config=TwinrConfig(audio_output_device="twinr_playback_softvol"),
            capture_device="plughw:CARD=Array,DEV=0",
            playback_coordinator=None,
        )

        self.assertIsNotNone(keepalive)
        assert keepalive is not None
        self.assertTrue(keepalive._prefers_direct_guard)
        self.assertIsNone(keepalive._playback_coordinator)

    def test_direct_guard_keepalive_opens_and_closes_guard_process(self) -> None:
        entered = Event()
        exited = Event()

        class _Guard:
            def __enter__(self):
                entered.set()
                return self

            def __exit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback
                exited.set()
                return False

        lines: list[str] = []
        keepalive = build_respeaker_duplex_keepalive(
            config=TwinrConfig(audio_output_device="twinr_playback_softvol"),
            capture_device="plughw:CARD=Array,DEV=0",
            playback_coordinator=None,
            emit=lines.append,
        )
        self.assertIsNotNone(keepalive)
        assert keepalive is not None

        with mock.patch(
            "twinr.agent.workflows.respeaker_duplex_keepalive.maybe_open_respeaker_duplex_playback_guard",
            return_value=_Guard(),
        ) as guard_factory:
            keepalive.open()
            self.assertTrue(entered.wait(timeout=1.0))
            keepalive.close()

        self.assertTrue(exited.wait(timeout=1.0))
        guard_factory.assert_called_once_with(
            capture_device="plughw:CARD=Array,DEV=0",
            playback_device="twinr_playback_softvol",
            sample_rate_hz=24000,
        )
        self.assertIn("voice_orchestrator_duplex_keepalive=started", lines)
        self.assertIn("voice_orchestrator_duplex_keepalive=stopped", lines)

    def test_direct_guard_keepalive_releases_guard_while_foreground_playback_runs(self) -> None:
        first_entered = Event()
        first_exited = Event()
        second_entered = Event()
        enter_count = 0
        exit_count = 0
        count_lock = Lock()

        class _Guard:
            active = True

            def __enter__(self):
                nonlocal enter_count
                with count_lock:
                    enter_count += 1
                    current = enter_count
                if current == 1:
                    first_entered.set()
                elif current == 2:
                    second_entered.set()
                return self

            def __exit__(self, exc_type, exc, traceback):
                nonlocal exit_count
                del exc_type, exc, traceback
                with count_lock:
                    exit_count += 1
                    current = exit_count
                if current == 1:
                    first_exited.set()
                return False

        keepalive = build_respeaker_duplex_keepalive(
            config=TwinrConfig(audio_output_device="twinr_playback_softvol"),
            capture_device="plughw:CARD=Array,DEV=0",
            playback_coordinator=None,
        )
        self.assertIsNotNone(keepalive)
        assert keepalive is not None

        with mock.patch(
            "twinr.agent.workflows.respeaker_duplex_keepalive.maybe_open_respeaker_duplex_playback_guard",
            side_effect=lambda **_kwargs: _Guard(),
        ):
            keepalive.open()
            self.assertTrue(first_entered.wait(timeout=1.0))
            keepalive.handle_playback_activity(
                PlaybackActivityEvent(
                    phase="starting",
                    owner="streaming_tts",
                    priority=20,
                    category="wav_chunks",
                )
            )
            self.assertTrue(first_exited.wait(timeout=1.0))
            keepalive.handle_playback_activity(
                PlaybackActivityEvent(
                    phase="finished",
                    owner="streaming_tts",
                    priority=20,
                    category="wav_chunks",
                )
            )
            self.assertTrue(second_entered.wait(timeout=1.0))
            keepalive.close()

    def test_direct_guard_keepalive_waits_before_reopening_after_foreground_playback(self) -> None:
        first_entered = Event()
        first_exited = Event()
        second_entered = Event()
        enter_count = 0
        exit_count = 0
        count_lock = Lock()

        class _Guard:
            active = True

            def __enter__(self):
                nonlocal enter_count
                with count_lock:
                    enter_count += 1
                    current = enter_count
                if current == 1:
                    first_entered.set()
                elif current == 2:
                    second_entered.set()
                return self

            def __exit__(self, exc_type, exc, traceback):
                nonlocal exit_count
                del exc_type, exc, traceback
                with count_lock:
                    exit_count += 1
                    current = exit_count
                if current == 1:
                    first_exited.set()
                return False

        keepalive = build_respeaker_duplex_keepalive(
            config=TwinrConfig(audio_output_device="twinr_playback_softvol"),
            capture_device="plughw:CARD=Array,DEV=0",
            playback_coordinator=None,
        )
        self.assertIsNotNone(keepalive)
        assert keepalive is not None

        with mock.patch(
            "twinr.agent.workflows.respeaker_duplex_keepalive.maybe_open_respeaker_duplex_playback_guard",
            side_effect=lambda **_kwargs: _Guard(),
        ), mock.patch(
            "twinr.agent.workflows.respeaker_duplex_keepalive._DIRECT_GUARD_RESUME_GRACE_S",
            0.20,
        ):
            keepalive.open()
            self.assertTrue(first_entered.wait(timeout=1.0))
            keepalive.handle_playback_activity(
                PlaybackActivityEvent(
                    phase="starting",
                    owner="streaming_tts",
                    priority=20,
                    category="wav_chunks",
                )
            )
            self.assertTrue(first_exited.wait(timeout=1.0))
            keepalive.handle_playback_activity(
                PlaybackActivityEvent(
                    phase="finished",
                    owner="streaming_tts",
                    priority=20,
                    category="wav_chunks",
                )
            )
            time.sleep(0.08)
            self.assertFalse(second_entered.is_set())
            self.assertTrue(second_entered.wait(timeout=0.5))
            keepalive.close()

    def test_orchestrator_registers_duplex_keepalive_with_playback_coordinator(self) -> None:
        class _FakePlaybackCoordinator:
            def __init__(self) -> None:
                self.listeners: list[object] = []

            def add_activity_listener(self, listener) -> None:
                self.listeners.append(listener)

        coordinator = _FakePlaybackCoordinator()
        orchestrator = EdgeVoiceOrchestrator(
            TwinrConfig(
                voice_orchestrator_ws_url="ws://127.0.0.1:8797/ws/orchestrator/voice",
                audio_input_device="plughw:CARD=Array,DEV=0",
                audio_output_device="twinr_playback_softvol",
            ),
            emit=lambda _msg: None,
            playback_coordinator=coordinator,
            on_voice_activation=lambda match: True,
            on_transcript_committed=lambda transcript, source: True,
            on_barge_in_interrupt=lambda: True,
        )

        self.assertIsNotNone(orchestrator._duplex_keepalive)
        self.assertEqual(len(coordinator.listeners), 1)

    def test_non_loopback_ws_requires_explicit_opt_in(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Insecure ws:// voice orchestrator endpoint rejected",
        ):
            EdgeVoiceOrchestrator(
                TwinrConfig(
                    voice_orchestrator_ws_url="ws://192.168.1.154:8797/ws/orchestrator/voice",
                ),
                emit=lambda _msg: None,
                on_voice_activation=lambda match: True,
                on_transcript_committed=lambda transcript, source: True,
                on_barge_in_interrupt=lambda: True,
            )

    def test_explicit_opt_in_allows_non_loopback_ws_bridge(self) -> None:
        orchestrator = EdgeVoiceOrchestrator(
            TwinrConfig(
                voice_orchestrator_ws_url="ws://192.168.1.154:8797/ws/orchestrator/voice",
                voice_orchestrator_allow_insecure_ws=True,
            ),
            emit=lambda _msg: None,
            on_voice_activation=lambda match: True,
            on_transcript_committed=lambda transcript, source: True,
            on_barge_in_interrupt=lambda: True,
        )

        self.assertEqual(
            orchestrator._client.url,
            "ws://192.168.1.154:8797/ws/orchestrator/voice",
        )

    def test_reconnects_and_replays_runtime_state_after_remote_close(self) -> None:
        orchestrator, fake_client, lines, _committed = self._make_orchestrator()

        orchestrator.notify_runtime_state(
            state="speaking",
            detail="voice_activation",
            follow_up_allowed=True,
        )
        self.assertEqual(fake_client.open_calls, 1)
        self.assertEqual(len(fake_client.hello_requests), 1)
        self.assertEqual(fake_client.runtime_states[-1].state, "speaking")

        orchestrator._handle_server_event(
            OrchestratorVoiceErrorEvent(error="Voice orchestrator websocket closed unexpectedly.")
        )

        orchestrator._send_frame(b"\x00" * orchestrator._frame_bytes)

        self.assertEqual(fake_client.open_calls, 2)
        self.assertEqual(len(fake_client.hello_requests), 2)
        self.assertEqual(fake_client.runtime_states[-1].state, "speaking")
        self.assertEqual(fake_client.runtime_states[-1].detail, "voice_activation")
        self.assertEqual(len(fake_client.audio_frames), 1)
        self.assertIn("voice_orchestrator_error=Voice orchestrator websocket closed unexpectedly.", lines)
        self.assertIn("voice_orchestrator_reconnected=true", lines)

    def test_notify_runtime_state_includes_intent_context_fields(self) -> None:
        orchestrator, fake_client, _lines, _committed = self._make_orchestrator()

        orchestrator.notify_runtime_state(
            state="follow_up_open",
            detail="voice_activation",
            follow_up_allowed=True,
            attention_state="attending_to_device",
            interaction_intent_state="showing_intent",
            person_visible=True,
            presence_active=True,
            interaction_ready=True,
            targeted_inference_blocked=False,
            recommended_channel="speech",
        )

        event = fake_client.runtime_states[-1]
        self.assertEqual(event.state, "follow_up_open")
        self.assertEqual(event.attention_state, "attending_to_device")
        self.assertEqual(event.interaction_intent_state, "showing_intent")
        self.assertTrue(event.person_visible)
        self.assertTrue(event.presence_active)
        self.assertTrue(event.interaction_ready)
        self.assertFalse(event.targeted_inference_blocked)
        self.assertEqual(event.recommended_channel, "speech")

    def test_send_frame_embeds_latest_runtime_state_snapshot(self) -> None:
        orchestrator, fake_client, _lines, _committed = self._make_orchestrator()

        orchestrator.notify_runtime_state(
            state="waiting",
            detail="idle",
            follow_up_allowed=False,
            attention_state="attending_to_device",
            interaction_intent_state="showing_intent",
            person_visible=True,
            presence_active=True,
            interaction_ready=True,
            targeted_inference_blocked=False,
            recommended_channel="speech",
        )

        orchestrator._send_frame(b"\x00" * orchestrator._frame_bytes)

        frame = fake_client.audio_frames[-1]
        self.assertIsNotNone(frame.runtime_state)
        self.assertEqual(frame.runtime_state.state, "waiting")
        self.assertEqual(frame.runtime_state.detail, "idle")
        self.assertEqual(frame.runtime_state.attention_state, "attending_to_device")
        self.assertEqual(frame.runtime_state.interaction_intent_state, "showing_intent")
        self.assertTrue(frame.runtime_state.person_visible)
        self.assertTrue(frame.runtime_state.presence_active)
        self.assertTrue(frame.runtime_state.interaction_ready)
        self.assertFalse(frame.runtime_state.targeted_inference_blocked)
        self.assertEqual(frame.runtime_state.recommended_channel, "speech")

    def test_connect_client_embeds_cached_runtime_state_in_hello(self) -> None:
        orchestrator, fake_client, _lines, _committed = self._make_orchestrator()

        orchestrator.seed_runtime_state(
            state="waiting",
            detail="idle",
            follow_up_allowed=False,
            person_visible=False,
            presence_active=True,
            interaction_ready=False,
            targeted_inference_blocked=True,
            recommended_channel="display",
        )

        orchestrator._connect_client()

        hello = fake_client.hello_requests[-1]
        self.assertEqual(hello.trace_id, orchestrator._session_id)
        self.assertEqual(hello.initial_state, "waiting")
        self.assertEqual(hello.detail, "idle")
        self.assertFalse(hello.follow_up_allowed)
        self.assertTrue(hello.state_attested)
        self.assertFalse(hello.person_visible)
        self.assertTrue(hello.presence_active)
        self.assertFalse(hello.interaction_ready)
        self.assertTrue(hello.targeted_inference_blocked)
        self.assertEqual(hello.recommended_channel, "display")
        self.assertEqual(fake_client.runtime_states[-1].state, "waiting")

    def test_connect_client_does_not_replay_stale_runtime_state_after_newer_update(self) -> None:
        orchestrator, _fake_client, _lines, _committed = self._make_orchestrator()

        class _ConcurrentRuntimeStateClient(_FakeVoiceClient):
            def __init__(self) -> None:
                super().__init__()
                self._send_lock = Lock()
                self._triggered = False

            def send_hello(self, request) -> None:
                super().send_hello(request)
                if self._triggered:
                    return
                self._triggered = True

                def concurrent_notify() -> None:
                    while not orchestrator._connected:
                        time.sleep(0.001)
                    orchestrator.notify_runtime_state(state="listening", detail="voice_activation")

                Thread(target=concurrent_notify, daemon=True).start()

            def send_runtime_state(self, event) -> None:
                if event.state == "waiting":
                    time.sleep(0.05)
                with self._send_lock:
                    self.runtime_states.append(event)

        fake_client = _ConcurrentRuntimeStateClient()
        orchestrator._client = cast(Any, fake_client)
        orchestrator.seed_runtime_state(state="waiting", detail="idle")

        orchestrator._connect_client()
        for _ in range(100):
            if len(fake_client.runtime_states) >= 2:
                break
            time.sleep(0.01)

        observed_states = [(event.state, event.detail) for event in fake_client.runtime_states]
        listening_index = observed_states.index(("listening", "voice_activation"))
        self.assertNotIn(("waiting", "idle"), observed_states[listening_index + 1 :])

    def test_open_starts_capture_thread_even_when_initial_connect_fails(self) -> None:
        orchestrator, fake_client, lines, _committed = self._make_orchestrator()
        fake_client.fail_open_calls = 1
        started = mock.Mock()

        def fake_capture_loop() -> None:
            started()

        with mock.patch.object(orchestrator, "_capture_loop", side_effect=fake_capture_loop):
            orchestrator.open()
            thread = orchestrator._thread
            self.assertIsNotNone(thread)
            assert thread is not None
            thread.join(timeout=1.0)

        started.assert_called_once_with()
        self.assertEqual(fake_client.open_calls, 1)
        self.assertIn("voice_orchestrator_unavailable=ConnectionError", lines)

    def test_open_and_close_manage_respeaker_duplex_keepalive(self) -> None:
        lines: list[str] = []
        keepalive = mock.Mock()
        orchestrator = EdgeVoiceOrchestrator(
            TwinrConfig(
                voice_orchestrator_ws_url="ws://127.0.0.1:8797/ws/orchestrator/voice",
            ),
            emit=lines.append,
            on_voice_activation=lambda match: True,
            on_transcript_committed=lambda transcript, source: True,
            on_barge_in_interrupt=lambda: True,
        )
        fake_client = _FakeVoiceClient()
        orchestrator._client = cast(Any, fake_client)
        orchestrator._duplex_keepalive = keepalive

        with (
            mock.patch.object(orchestrator, "_capture_loop"),
            mock.patch.object(orchestrator, "_sender_loop"),
        ):
            orchestrator.open()
            orchestrator.close()

        keepalive.open.assert_called_once_with()
        keepalive.close.assert_called_once_with()

    def test_next_frame_reconnects_after_send_failure(self) -> None:
        orchestrator, fake_client, lines, _committed = self._make_orchestrator()
        orchestrator._connect_client()
        fake_client.fail_audio_sends = 1

        orchestrator._send_frame(b"\x00" * orchestrator._frame_bytes)
        self.assertFalse(orchestrator._connected)
        self.assertIn("voice_orchestrator_send_failed=ConnectionError", lines)

        orchestrator._send_frame(b"\x00" * orchestrator._frame_bytes)

        self.assertTrue(orchestrator._connected)
        self.assertEqual(fake_client.open_calls, 2)
        self.assertEqual(len(fake_client.audio_frames), 1)
        self.assertIn("voice_orchestrator_reconnected=true", lines)

    def test_ready_backend_tracks_remote_asr_gateway(self) -> None:
        orchestrator, _fake_client, _lines, _committed = self._make_orchestrator()

        self.assertTrue(orchestrator.supports_remote_follow_up())

        orchestrator._handle_server_event(
            OrchestratorVoiceReadyEvent(session_id="voice-1", backend="remote_asr")
        )
        self.assertEqual(orchestrator.ready_backend, "remote_asr")
        self.assertTrue(orchestrator.supports_remote_follow_up())

    def test_unsupported_identity_profiles_disable_optional_sync_without_disconnect(self) -> None:
        orchestrator, fake_client, lines, _committed = self._make_orchestrator()
        orchestrator._connect_client()

        event = OrchestratorVoiceIdentityProfilesEvent(
            revision="rev-1",
            profiles=(
                OrchestratorVoiceIdentityProfile(
                    user_id="main_user",
                    embedding=(0.1, 0.2, 0.3),
                    primary_user=True,
                ),
            ),
        )

        orchestrator.notify_identity_profiles(event)
        self.assertEqual(len(fake_client.identity_profiles), 1)
        self.assertTrue(orchestrator._connected)

        orchestrator._handle_server_event(
            OrchestratorVoiceErrorEvent(error="Unsupported message type.")
        )

        self.assertTrue(orchestrator._connected)
        self.assertFalse(orchestrator._identity_profiles_supported)
        self.assertIn("voice_orchestrator_identity_profiles_unsupported=true", lines)

        orchestrator.notify_identity_profiles(event)
        self.assertEqual(len(fake_client.identity_profiles), 1)

    def test_transcript_committed_event_dispatches_same_stream_transcript(self) -> None:
        orchestrator, _fake_client, lines, committed = self._make_orchestrator()

        orchestrator._handle_server_event(
            OrchestratorVoiceTranscriptCommittedEvent(
                transcript="wie geht es dir",
                source="follow_up",
            )
        )
        orchestrator._handle_server_event(
            OrchestratorVoiceFollowUpClosedEvent(reason="timeout")
        )

        self.assertEqual(committed, [("wie geht es dir", "follow_up")])
        self.assertIn("voice_orchestrator_transcript_committed=follow_up", lines)
        self.assertIn("voice_orchestrator_follow_up_closed=timeout", lines)

    def test_follow_up_closed_event_dispatches_local_callback(self) -> None:
        closed: list[str] = []
        orchestrator = EdgeVoiceOrchestrator(
            TwinrConfig(
                voice_orchestrator_ws_url="ws://127.0.0.1:8797/ws/orchestrator/voice",
            ),
            emit=lambda _msg: None,
            on_voice_activation=lambda match: True,
            on_transcript_committed=lambda transcript, source: True,
            on_follow_up_closed=closed.append,
            on_barge_in_interrupt=lambda: True,
        )

        orchestrator._handle_server_event(
            OrchestratorVoiceFollowUpClosedEvent(reason="timeout")
        )

        self.assertEqual(closed, ["timeout"])

    def test_capture_loop_retries_transient_respeaker_process_exit_before_first_frame(self) -> None:
        orchestrator, _fake_client, lines, _committed = self._make_orchestrator()
        failed_process = _FakeCaptureProcess(returncode=1)
        recovered_process = _FakeCaptureProcess()
        sent_frames: list[bytes] = []
        select_call_count = 0

        def fake_select(read_fds, _write_fds, _error_fds, _timeout):
            nonlocal select_call_count
            select_call_count += 1
            if select_call_count == 1:
                return [], [], []
            return [read_fds[0]], [], []

        def fake_enqueue_frame(frame: bytes) -> None:
            sent_frames.append(frame)
            orchestrator._stop_event.set()

        with (
            mock.patch.object(
                orchestrator,
                "_start_process",
                side_effect=[failed_process, recovered_process],
            ) as start_process,
            mock.patch(
                "twinr.agent.workflows.voice_orchestrator.recover_stalled_respeaker_capture",
                return_value=True,
            ) as recover_capture,
            mock.patch("twinr.agent.workflows.voice_orchestrator.select.select", side_effect=fake_select),
            mock.patch(
                "twinr.agent.workflows.voice_orchestrator.os.read",
                return_value=b"\x01\x00" * (orchestrator._frame_bytes // 2),
            ),
            mock.patch.object(orchestrator, "_drain_stderr"),
            mock.patch.object(orchestrator, "_enqueue_frame", side_effect=fake_enqueue_frame),
        ):
            orchestrator._capture_loop()

        recover_capture.assert_called_once()
        self.assertEqual(start_process.call_count, 2)
        self.assertEqual(len(sent_frames), 1)
        self.assertNotIn("voice_orchestrator_capture_failed=RuntimeError", lines)

    def test_capture_loop_recovers_when_first_frame_never_arrives(self) -> None:
        orchestrator, _fake_client, lines, _committed = self._make_orchestrator()
        stalled_process = _FakeCaptureProcess()
        recovered_process = _FakeCaptureProcess()
        sent_frames: list[bytes] = []
        select_call_count = 0

        def fake_select(read_fds, _write_fds, _error_fds, _timeout):
            nonlocal select_call_count
            select_call_count += 1
            if select_call_count <= 2:
                return [], [], []
            return [read_fds[0]], [], []

        def fake_enqueue_frame(frame: bytes) -> None:
            sent_frames.append(frame)
            orchestrator._stop_event.set()

        with (
            mock.patch.object(
                orchestrator,
                "_start_process",
                side_effect=[stalled_process, recovered_process],
            ) as start_process,
            mock.patch.object(orchestrator, "_first_frame_timed_out", side_effect=[False, True]),
            mock.patch(
                "twinr.agent.workflows.voice_orchestrator.recover_stalled_respeaker_capture",
                return_value=True,
            ) as recover_capture,
            mock.patch("twinr.agent.workflows.voice_orchestrator.select.select", side_effect=fake_select),
            mock.patch(
                "twinr.agent.workflows.voice_orchestrator.os.read",
                return_value=b"\x01\x00" * (orchestrator._frame_bytes // 2),
            ),
            mock.patch.object(orchestrator, "_drain_stderr"),
            mock.patch.object(orchestrator, "_enqueue_frame", side_effect=fake_enqueue_frame),
        ):
            orchestrator._capture_loop()

        recover_capture.assert_called_once()
        self.assertEqual(start_process.call_count, 2)
        self.assertEqual(len(sent_frames), 1)
        self.assertIn("voice_orchestrator_capture_recovered=respeaker_recovery", lines)

    def test_capture_loop_recovers_when_mid_stream_bytes_stop_arriving(self) -> None:
        orchestrator, _fake_client, lines, _committed = self._make_orchestrator()
        stalled_process = _FakeCaptureProcess()
        recovered_process = _FakeCaptureProcess()
        sent_frames: list[bytes] = []
        select_call_count = 0

        def fake_select(read_fds, _write_fds, _error_fds, _timeout):
            nonlocal select_call_count
            select_call_count += 1
            if select_call_count == 1:
                return [read_fds[0]], [], []
            if select_call_count in (2, 3):
                return [], [], []
            return [read_fds[0]], [], []

        def fake_enqueue_frame(frame: bytes) -> None:
            sent_frames.append(frame)
            if len(sent_frames) >= 2:
                orchestrator._stop_event.set()

        with (
            mock.patch.object(
                orchestrator,
                "_start_process",
                side_effect=[stalled_process, recovered_process],
            ) as start_process,
            mock.patch.object(orchestrator, "_first_frame_timed_out", return_value=False),
            mock.patch.object(
                orchestrator,
                "_raise_for_capture_activity_timeout",
                side_effect=[None, _CaptureStreamStalledError(stalled_ms=1600), None],
            ) as raise_for_stall,
            mock.patch(
                "twinr.agent.workflows.voice_orchestrator.recover_stalled_respeaker_capture",
                return_value=True,
            ) as recover_capture,
            mock.patch("twinr.agent.workflows.voice_orchestrator.select.select", side_effect=fake_select),
            mock.patch(
                "twinr.agent.workflows.voice_orchestrator.os.read",
                return_value=b"\x01\x00" * (orchestrator._frame_bytes // 2),
            ),
            mock.patch.object(orchestrator, "_drain_stderr"),
            mock.patch.object(orchestrator, "_enqueue_frame", side_effect=fake_enqueue_frame),
        ):
            orchestrator._capture_loop()

        self.assertEqual(raise_for_stall.call_count, 2)
        recover_capture.assert_called_once()
        self.assertEqual(start_process.call_count, 2)
        self.assertEqual(len(sent_frames), 2)
        self.assertIn("voice_orchestrator_capture_recovered=respeaker_recovery", lines)

    def test_start_process_uses_sanitized_audio_env(self) -> None:
        orchestrator, _fake_client, lines, _committed = self._make_orchestrator()
        fake_process = _FakeCaptureProcess()
        sanitized_env = {"PATH": "/usr/bin", "HOME": "/root"}

        with (
            mock.patch("twinr.agent.workflows.voice_orchestrator.shutil.which", return_value="/usr/bin/arecord"),
            mock.patch(
                "twinr.agent.workflows.voice_orchestrator.build_audio_subprocess_env_for_mode",
                return_value=sanitized_env,
            ) as build_env,
            mock.patch(
                "twinr.agent.workflows.voice_orchestrator.subprocess.Popen",
                return_value=fake_process,
            ) as popen,
            mock.patch("twinr.agent.workflows.voice_orchestrator.os.set_blocking"),
        ):
            process = orchestrator._start_process()

        self.assertIs(process, fake_process)
        build_env.assert_called_once_with(allow_root_borrowed_session_audio=True)
        popen.assert_called_once()
        self.assertEqual(popen.call_args.kwargs["env"], sanitized_env)
        self.assertTrue(popen.call_args.kwargs["close_fds"])
        self.assertNotIn("start_new_session", popen.call_args.kwargs)
        self.assertIn("voice_orchestrator_capture=started", lines)


if __name__ == "__main__":
    unittest.main()
