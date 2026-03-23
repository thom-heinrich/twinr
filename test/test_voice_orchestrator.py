from pathlib import Path
from tempfile import TemporaryFile
from unittest import mock
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.voice_orchestrator import EdgeVoiceOrchestrator
from twinr.orchestrator.voice_contracts import OrchestratorVoiceErrorEvent, OrchestratorVoiceReadyEvent


class _FakeVoiceClient:
    def __init__(self) -> None:
        self.open_calls = 0
        self.close_calls = 0
        self.hello_requests = []
        self.runtime_states = []
        self.audio_frames = []
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
        orchestrator = EdgeVoiceOrchestrator(
            TwinrConfig(
                voice_orchestrator_ws_url="ws://127.0.0.1:8797/ws/orchestrator/voice",
            ),
            emit=lines.append,
            on_wakeword_match=lambda match: True,
            on_follow_up_capture_requested=lambda: True,
            on_barge_in_interrupt=lambda: True,
        )
        fake_client = _FakeVoiceClient()
        orchestrator._client = fake_client
        return orchestrator, fake_client, lines

    def test_reconnects_and_replays_runtime_state_after_remote_close(self) -> None:
        orchestrator, fake_client, lines = self._make_orchestrator()

        orchestrator.notify_runtime_state(
            state="speaking",
            detail="wakeword",
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
        self.assertEqual(fake_client.runtime_states[-1].detail, "wakeword")
        self.assertEqual(len(fake_client.audio_frames), 1)
        self.assertIn("voice_orchestrator_error=Voice orchestrator websocket closed unexpectedly.", lines)
        self.assertIn("voice_orchestrator_reconnected=true", lines)

    def test_open_starts_capture_thread_even_when_initial_connect_fails(self) -> None:
        orchestrator, fake_client, lines = self._make_orchestrator()
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

    def test_next_frame_reconnects_after_send_failure(self) -> None:
        orchestrator, fake_client, lines = self._make_orchestrator()
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

    def test_ready_backend_controls_remote_follow_up_capability(self) -> None:
        orchestrator, _fake_client, _lines = self._make_orchestrator()

        self.assertFalse(orchestrator.supports_remote_follow_up())

        orchestrator._handle_server_event(
            OrchestratorVoiceReadyEvent(session_id="voice-1", backend="local_stt")
        )
        self.assertEqual(orchestrator.ready_backend, "local_stt")
        self.assertTrue(orchestrator.supports_remote_follow_up())

        orchestrator._handle_server_event(
            OrchestratorVoiceReadyEvent(session_id="voice-1", backend="openwakeword")
        )
        self.assertEqual(orchestrator.ready_backend, "openwakeword")
        self.assertFalse(orchestrator.supports_remote_follow_up())

    def test_capture_loop_retries_transient_respeaker_process_exit_before_first_frame(self) -> None:
        orchestrator, _fake_client, lines = self._make_orchestrator()
        failed_process = _FakeCaptureProcess(returncode=1)
        recovered_process = _FakeCaptureProcess()
        sent_frames: list[bytes] = []

        def fake_select(read_fds, _write_fds, _error_fds, _timeout):
            fake_select.calls += 1
            if fake_select.calls == 1:
                return [], [], []
            return [read_fds[0]], [], []

        fake_select.calls = 0

        def fake_send_frame(frame: bytes) -> None:
            sent_frames.append(frame)
            orchestrator._stop_event.set()

        with (
            mock.patch.object(
                orchestrator,
                "_start_process",
                side_effect=[failed_process, recovered_process],
            ) as start_process,
            mock.patch(
                "twinr.agent.workflows.voice_orchestrator.wait_for_transient_respeaker_capture_ready",
                return_value=True,
            ) as recover_capture,
            mock.patch("twinr.agent.workflows.voice_orchestrator.select.select", side_effect=fake_select),
            mock.patch(
                "twinr.agent.workflows.voice_orchestrator.os.read",
                return_value=b"\x01\x00" * (orchestrator._frame_bytes // 2),
            ),
            mock.patch.object(orchestrator, "_drain_stderr"),
            mock.patch.object(orchestrator, "_send_frame", side_effect=fake_send_frame),
        ):
            orchestrator._capture_loop()

        recover_capture.assert_called_once()
        self.assertEqual(start_process.call_count, 2)
        self.assertEqual(len(sent_frames), 1)
        self.assertNotIn("voice_orchestrator_capture_failed=RuntimeError", lines)


if __name__ == "__main__":
    unittest.main()
