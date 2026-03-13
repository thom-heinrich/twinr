from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.hardware import VoiceAssessment
from twinr.memory.reminders import now_in_timezone
from twinr.proactive import SocialTriggerDecision, SocialTriggerPriority
from twinr.providers.openai import OpenAIImageInput, OpenAITextResponse
from twinr.runner import TwinrHardwareLoop
from twinr.runtime import TwinrRuntime


class FakeBackend:
    def __init__(self) -> None:
        self.transcribe_calls: list[tuple[bytes, str, str]] = []
        self.respond_calls: list[tuple[str, tuple[tuple[str, str], ...] | None, bool | None]] = []
        self.respond_to_images_calls: list[
            tuple[str, list[OpenAIImageInput], tuple[tuple[str, str], ...] | None, bool | None]
        ] = []
        self.synthesize_calls: list[str] = []
        self.print_calls: list[tuple[tuple[tuple[str, str], ...] | None, str | None, str | None, str]] = []
        self.reminder_calls: list[object] = []
        self.transcript = "Hello Twinr"
        self.answer = "Hello back."

    def transcribe(self, audio_bytes: bytes, *, filename: str, content_type: str) -> str:
        self.transcribe_calls.append((audio_bytes, filename, content_type))
        return self.transcript

    def respond_streaming(self, prompt: str, *, conversation=None, allow_web_search=None, on_text_delta=None) -> OpenAITextResponse:
        self.respond_calls.append((prompt, conversation, allow_web_search))
        if on_text_delta is not None:
            for chunk in ("Hello ", "back."):
                on_text_delta(chunk)
        return OpenAITextResponse(
            text=self.answer,
            response_id="resp_123",
            request_id="req_123",
            used_web_search=True,
        )

    def respond_to_images_with_metadata(
        self,
        prompt: str,
        *,
        images,
        conversation=None,
        allow_web_search=None,
        instructions=None,
    ) -> OpenAITextResponse:
        self.respond_to_images_calls.append((prompt, list(images), conversation, allow_web_search))
        return OpenAITextResponse(
            text=self.answer,
            response_id="resp_vision_123",
            request_id="req_vision_123",
            used_web_search=False,
        )

    def synthesize(self, text: str) -> bytes:
        self.synthesize_calls.append(text)
        return b"RIFF"

    def synthesize_stream(self, text: str):
        self.synthesize_calls.append(text)
        yield b"RI"
        yield b"FF"

    def compose_print_job_with_metadata(
        self,
        *,
        conversation=None,
        focus_hint: str | None = None,
        direct_text: str | None = None,
        request_source: str = "button",
    ) -> OpenAITextResponse:
        self.print_calls.append((conversation, focus_hint, direct_text, request_source))
        return OpenAITextResponse(text="HELLO BACK")

    def phrase_due_reminder_with_metadata(self, reminder) -> OpenAITextResponse:
        self.reminder_calls.append(reminder)
        return OpenAITextResponse(
            text=f"Erinnerung: {reminder.summary}",
            response_id="resp_reminder_1",
            request_id="req_reminder_1",
            used_web_search=False,
        )


class FakeRecorder:
    def __init__(self) -> None:
        self.pause_values: list[int] = []

    def record_until_pause(self, *, pause_ms: int) -> bytes:
        self.pause_values.append(pause_ms)
        return b"WAVDATA"


class FakePlayer:
    def __init__(self) -> None:
        self.played: list[bytes] = []

    def play_wav_bytes(self, audio_bytes: bytes) -> None:
        self.played.append(audio_bytes)

    def play_wav_chunks(self, chunks) -> None:
        self.played.append(b"".join(chunks))


class FakePrinter:
    def __init__(self) -> None:
        self.printed: list[str] = []

    def print_text(self, text: str) -> str:
        self.printed.append(text)
        return "request id is Test-1 (1 file(s))"


class FakeCamera:
    def __init__(self) -> None:
        self.capture_calls = 0

    def capture_photo(self, *, output_path=None, filename: str = "camera-capture.png"):
        self.capture_calls += 1
        return SimpleNamespace(
            data=b"\x89PNG\r\n\x1a\ncamera",
            content_type="image/png",
            filename=filename,
            source_device="/dev/video0",
            input_format="bayer_grbg8",
        )


class FakeIdleButtonMonitor:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.poll_calls = 0

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True

    def poll(self, timeout=None):
        self.poll_calls += 1
        if timeout:
            time.sleep(min(timeout, 0.001))
        return None


class FakeProactiveMonitor:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True


class HardwareLoopTests(unittest.TestCase):
    def make_loop(
        self,
        *,
        backend=None,
        config: TwinrConfig | None = None,
        camera=None,
        button_monitor=None,
        voice_profile_monitor=None,
        proactive_monitor=None,
    ) -> tuple[TwinrHardwareLoop, list[str], FakeRecorder, FakePlayer, FakePrinter]:
        lines: list[str] = []
        recorder = FakeRecorder()
        player = FakePlayer()
        printer = FakePrinter()
        config = config or TwinrConfig()
        loop = TwinrHardwareLoop(
            config=config,
            runtime=TwinrRuntime(config=config),
            backend=backend or FakeBackend(),
            button_monitor=button_monitor or SimpleNamespace(__enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: None),
            recorder=recorder,
            player=player,
            printer=printer,
            camera=camera or FakeCamera(),
            voice_profile_monitor=voice_profile_monitor,
            proactive_monitor=proactive_monitor,
            emit=lines.append,
            sleep=lambda _seconds: None,
            error_reset_seconds=0.0,
        )
        return loop, lines, recorder, player, printer

    def test_green_button_runs_full_audio_turn(self) -> None:
        backend = FakeBackend()
        loop, lines, recorder, player, _printer = self.make_loop(backend=backend)

        loop.handle_button_press("green")

        self.assertEqual(recorder.pause_values, [1200])
        self.assertEqual(backend.transcribe_calls[0][0], b"WAVDATA")
        self.assertEqual(backend.transcribe_calls[0][1], "twinr-listen.wav")
        self.assertEqual(backend.respond_calls[0][0], "Hello Twinr")
        self.assertEqual(backend.respond_to_images_calls, [])
        self.assertFalse(backend.respond_calls[0][2])
        self.assertEqual(player.played, [b"RIFF"])
        self.assertEqual(loop.runtime.last_response, "Hello back.")
        self.assertIn("status=listening", lines)
        self.assertIn("status=processing", lines)
        self.assertIn("status=answering", lines)
        self.assertIn("status=waiting", lines)
        self.assertIn("timing_playback_ms=streamed", lines)

    def test_yellow_button_formats_and_prints_last_answer(self) -> None:
        backend = FakeBackend()
        loop, lines, _recorder, _player, printer = self.make_loop(backend=backend)
        loop.runtime.last_response = "Hello back"

        loop.handle_button_press("yellow")

        self.assertEqual(
            backend.print_calls,
            [((), None, "Hello back", "button")],
        )
        self.assertEqual(printer.printed, ["HELLO BACK"])
        self.assertIn("status=printing", lines)
        self.assertEqual(lines[-1], "status=waiting")

    def test_idle_loop_delivers_due_reminder(self) -> None:
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                reminder_store_path=str(Path(temp_dir) / "state" / "reminders.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                reminder_poll_interval_s=0.0,
            )
            loop, lines, _recorder, player, _printer = self.make_loop(backend=backend, config=config)
            loop.runtime.schedule_reminder(
                due_at=now_in_timezone(config.local_timezone_name).isoformat(),
                summary="Tabletten nehmen",
                kind="medication",
                source="test",
            )

            delivered = loop._maybe_deliver_due_reminder()
            stored_entries = loop.runtime.reminder_store.load_entries()

        self.assertTrue(delivered)
        self.assertEqual(len(backend.reminder_calls), 1)
        self.assertEqual(backend.synthesize_calls, ["Erinnerung: Tabletten nehmen"])
        self.assertEqual(player.played, [b"RIFF"])
        self.assertIn("reminder_delivered=true", lines)
        self.assertTrue(stored_entries[0].delivered)

    def test_errors_reset_runtime_to_waiting(self) -> None:
        loop, lines, _recorder, _player, _printer = self.make_loop(backend=FakeBackend())

        loop.handle_button_press("yellow")

        self.assertIn("status=error", lines)
        self.assertTrue(any(line.startswith("error=") for line in lines))
        self.assertEqual(loop.runtime.status.value, "waiting")

    def test_auto_web_search_enables_search_for_freshness_queries(self) -> None:
        backend = FakeBackend()
        backend.transcript = "What is the weather today in Berlin?"
        loop, _lines, _recorder, _player, _printer = self.make_loop(backend=backend)

        loop.handle_button_press("green")

        self.assertTrue(backend.respond_calls[0][2])
        self.assertEqual(len(loop.runtime.memory.search_results), 1)
        self.assertEqual(loop.runtime.memory.search_results[0].question, "What is the weather today in Berlin?")
        self.assertEqual(loop.runtime.memory.search_results[0].answer, "Hello back.")

    def test_visual_queries_use_camera_and_multimodal_request(self) -> None:
        backend = FakeBackend()
        backend.transcript = "Schau mich mal an"
        camera = FakeCamera()
        loop, lines, _recorder, player, _printer = self.make_loop(backend=backend, camera=camera)

        loop.handle_button_press("green")

        self.assertEqual(camera.capture_calls, 1)
        self.assertEqual(len(backend.respond_to_images_calls), 1)
        prompt, images, conversation, allow_web_search = backend.respond_to_images_calls[0]
        self.assertIn("Image 1 is the current live camera frame", prompt)
        self.assertEqual(conversation, ())
        self.assertFalse(allow_web_search)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].label, "Image 1: live camera frame from the device.")
        self.assertEqual(player.played, [b"RIFF"])
        self.assertIn("camera_used=true", lines)
        self.assertIn("vision_image_count=1", lines)

    def test_visual_queries_attach_reference_image_when_configured(self) -> None:
        backend = FakeBackend()
        backend.transcript = "Wie sehe ich heute aus?"
        camera = FakeCamera()

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "user-reference.jpg"
            reference_path.write_bytes(b"\xff\xd8\xffreference")
            config = TwinrConfig(vision_reference_image_path=str(reference_path))
            loop, lines, _recorder, _player, _printer = self.make_loop(
                backend=backend,
                config=config,
                camera=camera,
            )

            loop.handle_button_press("green")

        self.assertEqual(len(backend.respond_to_images_calls), 1)
        prompt, images, _conversation, _allow_web_search = backend.respond_to_images_calls[0]
        self.assertIn("Image 2 is a stored reference image of the main user.", prompt)
        self.assertEqual(len(images), 2)
        self.assertEqual(images[1].label, "Image 2: stored reference image of the main user. Use it only for person or identity comparison.")
        self.assertTrue(any(line.startswith("vision_reference_image=") for line in lines))

    def test_green_button_updates_runtime_voice_assessment(self) -> None:
        class FakeVoiceProfileMonitor:
            def assess_wav_bytes(self, audio_bytes: bytes) -> VoiceAssessment:
                self.audio_bytes = audio_bytes
                return VoiceAssessment(
                    status="likely_user",
                    label="Likely user",
                    detail="Close to the enrolled template.",
                    confidence=0.81,
                    checked_at="2026-03-13T12:00:00+00:00",
                )

        monitor = FakeVoiceProfileMonitor()
        backend = FakeBackend()
        loop, lines, _recorder, _player, _printer = self.make_loop(
            backend=backend,
            voice_profile_monitor=monitor,
        )

        loop.handle_button_press("green")

        self.assertEqual(monitor.audio_bytes, b"WAVDATA")
        self.assertEqual(loop.runtime.user_voice_status, "likely_user")
        self.assertEqual(loop.runtime.user_voice_confidence, 0.81)
        self.assertEqual(loop.runtime.user_voice_checked_at, "2026-03-13T12:00:00+00:00")
        self.assertIsNotNone(backend.respond_calls[0][1])
        self.assertEqual(backend.respond_calls[0][1][0][0], "system")
        self.assertIn("Speaker signal: likely match", backend.respond_calls[0][1][0][1])
        self.assertIn("voice_profile_status=likely_user", lines)
        self.assertIn("voice_profile_confidence=0.81", lines)

    def test_social_trigger_speaks_proactive_prompt(self) -> None:
        backend = FakeBackend()
        loop, lines, _recorder, player, _printer = self.make_loop(backend=backend)

        spoke = loop.handle_social_trigger(
            SocialTriggerDecision(
                trigger_id="attention_window",
                prompt="Kann ich dir bei etwas helfen?",
                reason="Person looked toward the device and stayed quiet.",
                observed_at=42.0,
                priority=SocialTriggerPriority.ATTENTION_WINDOW,
            )
        )

        self.assertTrue(spoke)
        self.assertEqual(loop.runtime.status.value, "waiting")
        self.assertEqual(player.played, [b"RIFF"])
        self.assertEqual(loop.runtime.last_response, None)
        self.assertIn("status=answering", lines)
        self.assertIn("social_trigger=attention_window", lines)
        self.assertIn("social_prompt=Kann ich dir bei etwas helfen?", lines)
        social_events = [entry for entry in loop.runtime.ops_events.tail(limit=20) if entry["event"] == "social_trigger_prompted"]
        self.assertEqual(len(social_events), 1)
        self.assertEqual(social_events[0]["data"]["prompt"], "Kann ich dir bei etwas helfen?")

    def test_social_trigger_is_skipped_when_runtime_is_busy(self) -> None:
        backend = FakeBackend()
        loop, lines, _recorder, player, _printer = self.make_loop(backend=backend)
        loop.runtime.press_green_button()

        spoke = loop.handle_social_trigger(
            SocialTriggerDecision(
                trigger_id="showing_intent",
                prompt="Möchtest du mir etwas zeigen?",
                reason="Hand or object near the camera.",
                observed_at=42.0,
                priority=SocialTriggerPriority.SHOWING_INTENT,
            )
        )

        self.assertFalse(spoke)
        self.assertEqual(player.played, [])
        self.assertIn("social_trigger_skipped=busy", lines)
        social_events = [entry for entry in loop.runtime.ops_events.tail(limit=20) if entry["event"] == "social_trigger_skipped"]
        self.assertEqual(len(social_events), 1)
        self.assertEqual(social_events[0]["data"]["prompt"], "Möchtest du mir etwas zeigen?")

    def test_run_opens_and_closes_proactive_monitor(self) -> None:
        backend = FakeBackend()
        button_monitor = FakeIdleButtonMonitor()
        proactive_monitor = FakeProactiveMonitor()
        loop, _lines, _recorder, _player, _printer = self.make_loop(
            backend=backend,
            button_monitor=button_monitor,
            proactive_monitor=proactive_monitor,
        )

        result = loop.run(duration_s=0.01, poll_timeout=0.001)

        self.assertEqual(result, 0)
        self.assertTrue(button_monitor.entered)
        self.assertTrue(button_monitor.exited)
        self.assertTrue(proactive_monitor.entered)
        self.assertTrue(proactive_monitor.exited)


if __name__ == "__main__":
    unittest.main()
