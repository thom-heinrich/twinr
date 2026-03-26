from pathlib import Path
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.ambient_impulse_cues import (
    DisplayAmbientImpulseCue,
    DisplayAmbientImpulseCueStore,
)
from twinr.display import service as display_service_mod
from twinr.display.debug_signals import (
    DisplayDebugSignal,
    DisplayDebugSignalSnapshot,
    DisplayDebugSignalStore,
)
from twinr.display.emoji_cues import DisplayEmojiCue, DisplayEmojiCueStore
from twinr.display.face_cues import DisplayFaceCue, DisplayFaceCueStore
from twinr.display.heartbeat import DisplayHeartbeatStore
from twinr.display.presentation_cues import DisplayPresentationCue, DisplayPresentationStore
from twinr.display.service import TwinrStatusDisplayLoop
from twinr.display.service_connect_cues import DisplayServiceConnectCue, DisplayServiceConnectCueStore
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshot, RuntimeSnapshotStore
from twinr.ops.health import ServiceHealth, TwinrSystemHealth


class FakeDisplay:
    def __init__(self) -> None:
        self.calls: list[
            tuple[
                str,
                str | None,
                str | None,
                tuple[str, ...],
                tuple[tuple[str, str], ...],
                tuple[tuple[str, tuple[str, ...]], ...],
                int,
                DisplayFaceCue | None,
                DisplayEmojiCue | None,
                DisplayAmbientImpulseCue | None,
                DisplayServiceConnectCue | None,
                DisplayPresentationCue | None,
                tuple[DisplayDebugSignal, ...],
            ]
        ] = []

    def show_status(
        self,
        status: str,
        *,
        headline: str | None = None,
        ticker_text: str | None = None,
        details: tuple[str, ...] = (),
        state_fields: tuple[tuple[str, str], ...] = (),
        log_sections: tuple[tuple[str, tuple[str, ...]], ...] = (),
        animation_frame: int = 0,
        face_cue: DisplayFaceCue | None = None,
        emoji_cue: DisplayEmojiCue | None = None,
        ambient_impulse_cue: DisplayAmbientImpulseCue | None = None,
        service_connect_cue: DisplayServiceConnectCue | None = None,
        presentation_cue: DisplayPresentationCue | None = None,
        debug_signals: tuple[DisplayDebugSignal, ...] = (),
    ) -> None:
        self.calls.append(
            (
                status,
                headline,
                ticker_text,
                details,
                state_fields,
                log_sections,
                animation_frame,
                face_cue,
                emoji_cue,
                ambient_impulse_cue,
                service_connect_cue,
                presentation_cue,
                debug_signals,
            )
        )


class IdleAnimatedDisplay(FakeDisplay):
    def supports_idle_waiting_animation(self) -> bool:
        return True


class FakeDebugLogBuilder:
    def __init__(self, sections: tuple[tuple[str, tuple[str, ...]], ...]) -> None:
        self.sections = sections

    def build_sections(self, **_kwargs: object) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return self.sections


class ReopenableDisplay:
    created_emits: list[object] = []

    def __init__(self, *, fail: bool, emit=None) -> None:
        self.fail = fail
        self.emit = emit
        self._last_rendered_status = "waiting"

    @classmethod
    def from_config(cls, _config, *, emit=None) -> "ReopenableDisplay":
        cls.created_emits.append(emit)
        return cls(fail=False, emit=emit)

    def show_status(
        self,
        status: str,
        *,
        headline: str | None = None,
        ticker_text: str | None = None,
        details: tuple[str, ...] = (),
        state_fields: tuple[tuple[str, str], ...] = (),
        log_sections: tuple[tuple[str, tuple[str, ...]], ...] = (),
        animation_frame: int = 0,
        face_cue: DisplayFaceCue | None = None,
        emoji_cue: DisplayEmojiCue | None = None,
        ambient_impulse_cue: DisplayAmbientImpulseCue | None = None,
        service_connect_cue: DisplayServiceConnectCue | None = None,
        presentation_cue: DisplayPresentationCue | None = None,
        debug_signals: tuple[DisplayDebugSignal, ...] = (),
    ) -> None:
        del (
            status,
            headline,
            ticker_text,
            details,
            state_fields,
            log_sections,
            animation_frame,
            face_cue,
            emoji_cue,
            ambient_impulse_cue,
            service_connect_cue,
            presentation_cue,
            debug_signals,
        )
        if self.fail:
            raise RuntimeError("boom")
        if self.emit is not None:
            self.emit("reopened_emit=true")

    def close(self) -> None:
        return None


class TickingDisplay(FakeDisplay):
    def __init__(self) -> None:
        super().__init__()
        self.tick_calls = 0

    def tick(self) -> None:
        self.tick_calls += 1


class FakeReSpeakerHciStore:
    def __init__(self, state) -> None:
        self.state = state

    def load(self):
        return self.state


class DisplayServiceTests(unittest.TestCase):
    def make_clock(self, hour: int = 12, minute: int = 34):
        return lambda: datetime(2026, 3, 13, hour, minute)

    def test_display_heartbeat_store_uses_ops_artifact_path(self) -> None:
        config = TwinrConfig(project_root="/tmp/twinr")

        store = DisplayHeartbeatStore.from_config(config)

        self.assertEqual(store.path, Path("/tmp/twinr/artifacts/stores/ops/display_heartbeat.json"))

    def make_health(
        self,
        *,
        status: str = "ok",
        cpu_temperature_c: float | None = 53.8,
        runtime_error: str | None = None,
        conversation_running: bool = True,
        memory_used_percent: float | None = 42.0,
        disk_used_percent: float | None = 34.0,
    ) -> TwinrSystemHealth:
        return TwinrSystemHealth(
            status=status,
            captured_at="2026-03-13T12:00:00+00:00",
            hostname="twinr-test",
            cpu_temperature_c=cpu_temperature_c,
            memory_used_percent=memory_used_percent,
            disk_used_percent=disk_used_percent,
            runtime_error=runtime_error,
            services=(
                ServiceHealth(
                    key="conversation_loop",
                    label="Conversation loop",
                    running=conversation_running,
                    count=1 if conversation_running else 0,
                    detail="ok" if conversation_running else "missing",
                ),
            ),
        )

    def test_display_loop_renders_when_snapshot_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "state" / "runtime-state.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            store = RuntimeSnapshotStore(snapshot_path)
            store.save(
                status="waiting",
                memory_turns=(),
                last_transcript=None,
                last_response="Hallo Thom",
            )
            display = FakeDisplay()
            loop = TwinrStatusDisplayLoop(
                config=TwinrConfig(
                    runtime_state_path=str(snapshot_path),
                    display_poll_interval_s=0.0,
                    openai_api_key="sk-test",
                ),
                display=display,
                snapshot_store=store,
                emit=lambda _line: None,
                sleep=lambda _seconds: None,
                health_collector=lambda _config, *, snapshot=None: self.make_health(),
                internet_probe=lambda: True,
                clock=self.make_clock(),
            )

            loop.run(max_cycles=1)
            store.save(
                status="printing",
                memory_turns=(),
                last_transcript=None,
                last_response="Wichtige Info",
            )
            loop.run(max_cycles=1)

            self.assertEqual(
                [
                    (status, headline, details)
                    for status, headline, _ticker, details, _fields, _logs, _frame, _cue, _emoji, _ambient, _service_connect, _presentation, _signals in display.calls
                ],
                [
                    ("waiting", "Waiting", ("Internet ok", "AI ok", "System ok", "Zeit 12:34")),
                    ("printing", "Printing", ("Internet ok", "AI ok", "System ok", "Zeit 12:34")),
                ],
            )

    def test_display_loop_persists_display_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = root / "state" / "runtime-state.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            store = RuntimeSnapshotStore(snapshot_path)
            store.save(
                status="waiting",
                memory_turns=(),
                last_transcript=None,
                last_response="Hallo Thom",
            )
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(snapshot_path),
                display_poll_interval_s=0.0,
                openai_api_key="sk-test",
            )
            loop = TwinrStatusDisplayLoop(
                config=config,
                display=FakeDisplay(),
                snapshot_store=store,
                emit=lambda _line: None,
                sleep=lambda _seconds: None,
                health_collector=lambda _config, *, snapshot=None: self.make_health(),
                internet_probe=lambda: True,
                clock=self.make_clock(),
                heartbeat_store=DisplayHeartbeatStore.from_config(config),
            )

            loop.run(max_cycles=1)
            heartbeat = DisplayHeartbeatStore.from_config(config).load()

        self.assertIsNotNone(heartbeat)
        assert heartbeat is not None
        self.assertEqual(heartbeat.runtime_status, "waiting")
        self.assertEqual(heartbeat.phase, "stopping")
        self.assertGreaterEqual(heartbeat.seq, 3)
        self.assertIsNotNone(heartbeat.last_render_started_at)
        self.assertIsNotNone(heartbeat.last_render_completed_at)

    def test_display_loop_ticks_visible_surface_between_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "state" / "runtime-state.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            store = RuntimeSnapshotStore(snapshot_path)
            store.save(
                status="waiting",
                memory_turns=(),
                last_transcript=None,
                last_response="Hallo Thom",
            )
            display = TickingDisplay()
            loop = TwinrStatusDisplayLoop(
                config=TwinrConfig(
                    runtime_state_path=str(snapshot_path),
                    display_poll_interval_s=0.0,
                    openai_api_key="sk-test",
                ),
                display=display,
                snapshot_store=store,
                emit=lambda _line: None,
                sleep=lambda _seconds: None,
                health_collector=lambda _config, *, snapshot=None: self.make_health(),
                internet_probe=lambda: True,
                clock=self.make_clock(),
            )

            loop.run(max_cycles=2)

        self.assertEqual(display.tick_calls, 2)

    def test_display_loop_refreshes_heartbeat_while_idle_without_rerender(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = root / "state" / "runtime-state.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            store = RuntimeSnapshotStore(snapshot_path)
            store.save(
                status="waiting",
                memory_turns=(),
                last_transcript=None,
                last_response="Hallo Thom",
            )
            display = FakeDisplay()
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(snapshot_path),
                display_poll_interval_s=0.0,
                openai_api_key="sk-test",
            )
            loop = TwinrStatusDisplayLoop(
                config=config,
                display=display,
                snapshot_store=store,
                emit=lambda _line: None,
                sleep=lambda _seconds: None,
                health_collector=lambda _config, *, snapshot=None: self.make_health(),
                internet_probe=lambda: True,
                clock=self.make_clock(),
                heartbeat_store=DisplayHeartbeatStore.from_config(config),
            )

            with mock.patch.object(display_service_mod, "_HEARTBEAT_IDLE_REFRESH_S", 0.0):
                loop.run(max_cycles=2)
            heartbeat = DisplayHeartbeatStore.from_config(config).load()

        self.assertEqual(len(display.calls), 1)
        self.assertIsNotNone(heartbeat)
        assert heartbeat is not None
        self.assertGreaterEqual(heartbeat.seq, 4)

    def test_reopen_display_preserves_emit_sink_and_last_status(self) -> None:
        emit_lines: list[str] = []
        emit = emit_lines.append
        ReopenableDisplay.created_emits.clear()
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(display_poll_interval_s=0.0, openai_api_key="sk-test"),
            display=ReopenableDisplay(fail=True, emit=emit),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=emit,
            sleep=lambda _seconds: None,
            health_collector=lambda _config, *, snapshot=None: self.make_health(),
            internet_probe=lambda: True,
            clock=self.make_clock(),
        )

        rendered = loop._show_status(
            "waiting",
            headline="Waiting",
            ticker_text=None,
            details=(),
            state_fields=(),
            log_sections=(),
            animation_frame=0,
            face_cue=None,
            emoji_cue=None,
            ambient_impulse_cue=None,
            service_connect_cue=None,
            presentation_cue=None,
            debug_signals=(),
        )

        self.assertTrue(rendered)
        self.assertEqual(ReopenableDisplay.created_emits, [emit])
        self.assertIs(loop.display.emit, emit)
        self.assertEqual(getattr(loop.display, "_last_rendered_status", None), "waiting")
        self.assertIn("reopened_emit=true", emit_lines)

    def test_build_status_content_uses_error_footer(self) -> None:
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(display_poll_interval_s=0.0, openai_api_key="sk-test"),
            display=FakeDisplay(),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=lambda _line: None,
            sleep=lambda _seconds: None,
            health_collector=lambda _config, *, snapshot=None: self.make_health(runtime_error="Printer offline"),
            internet_probe=lambda: True,
            clock=self.make_clock(),
        )

        headline, details = loop._build_status_content(
            RuntimeSnapshot(status="error", error_message="Printer offline")
        )

        self.assertEqual(headline, "Error")
        self.assertEqual(details, ("Internet ok", "AI ok", "System Fehler", "Zeit 12:34"))

    def test_build_status_content_warns_when_conversation_loop_is_missing(self) -> None:
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(display_poll_interval_s=0.0, openai_api_key="sk-test"),
            display=FakeDisplay(),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=lambda _line: None,
            sleep=lambda _seconds: None,
            health_collector=lambda _config, *, snapshot=None: self.make_health(
                status="warn",
                conversation_running=False,
            ),
            internet_probe=lambda: False,
            clock=self.make_clock(),
        )

        _headline, details = loop._build_status_content(RuntimeSnapshot(status="waiting"))

        self.assertEqual(details, ("Internet fehlt", "AI wartet", "System Achtung", "Zeit 12:34"))

    def test_build_state_fields_exposes_runtime_and_health_values(self) -> None:
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(display_poll_interval_s=0.0, openai_api_key="sk-test"),
            display=FakeDisplay(),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=lambda _line: None,
            sleep=lambda _seconds: None,
            health_collector=lambda _config, *, snapshot=None: self.make_health(),
            internet_probe=lambda: True,
            clock=self.make_clock(),
        )

        state_fields = loop._build_state_fields(RuntimeSnapshot(status="waiting"))

        self.assertEqual(
            state_fields,
            (
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
        )

    def test_build_state_fields_appends_respeaker_hci_status_when_available(self) -> None:
        state = type(
            "FakeState",
            (),
            {
                "state_fields": lambda self: (("ReSpeaker", "DFU"), ("Mikrofon", "stumm")),
            },
        )()
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(display_poll_interval_s=0.0, openai_api_key="sk-test"),
            display=FakeDisplay(),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=lambda _line: None,
            sleep=lambda _seconds: None,
            health_collector=lambda _config, *, snapshot=None: self.make_health(),
            internet_probe=lambda: True,
            clock=self.make_clock(),
            respeaker_hci_store=FakeReSpeakerHciStore(state),
        )

        state_fields = loop._build_state_fields(RuntimeSnapshot(status="waiting"))

        self.assertEqual(
            state_fields,
            (
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("ReSpeaker", "DFU"),
                ("Mikrofon", "stumm"),
                ("Zeit", "12:34"),
            ),
        )

    def test_debug_log_layout_passes_log_sections_and_suppresses_animation_churn(self) -> None:
        display = FakeDisplay()
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(
                display_layout="debug_log",
                display_poll_interval_s=0.0,
                openai_api_key="sk-test",
            ),
            display=display,
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=lambda _line: None,
            sleep=lambda _seconds: None,
            health_collector=lambda _config, *, snapshot=None: self.make_health(),
            internet_probe=lambda: True,
            clock=self.make_clock(),
            debug_log_builder=FakeDebugLogBuilder(
                (
                    ("System Log", ("12:34 now Waiting",)),
                    ("LLM Log", ("user Hallo",)),
                    ("Hardware Log", ("button green",)),
                )
            ),
        )

        loop.run(max_cycles=2)

        self.assertEqual(len(display.calls), 1)
        self.assertEqual(
            display.calls[0][5],
            (
                ("System Log", ("12:34 now Waiting",)),
                ("LLM Log", ("user Hallo",)),
                ("Hardware Log", ("button green",)),
            ),
        )

    def test_debug_log_layout_signature_ignores_non_rendered_footer_churn(self) -> None:
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(
                display_layout="debug_log",
                display_poll_interval_s=0.0,
                openai_api_key="sk-test",
            ),
            display=FakeDisplay(),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=lambda _line: None,
            sleep=lambda _seconds: None,
            health_collector=lambda _config, *, snapshot=None: self.make_health(),
            internet_probe=lambda: True,
            clock=self.make_clock(),
            debug_log_builder=FakeDebugLogBuilder(
                (
                    ("System Log", ("12:34 now Waiting",)),
                    ("LLM Log", ("user Hallo",)),
                    ("Hardware Log", ("button green",)),
                )
            ),
        )

        first_signature = loop._render_signature(
            status="waiting",
            headline="Waiting",
            ticker_text=None,
            details=("Internet ok", "AI ok", "System ok", "Zeit 12:34"),
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:34"),
            ),
            log_sections=(
                ("System Log", ("12:34 now Waiting",)),
                ("LLM Log", ("user Hallo",)),
                ("Hardware Log", ("button green",)),
            ),
            animation_frame=0,
            face_cue=None,
            emoji_cue=None,
            ambient_impulse_cue=None,
            service_connect_cue=None,
            presentation_cue=None,
        )
        second_signature = loop._render_signature(
            status="waiting",
            headline="Waiting",
            ticker_text="Tagesschau · Headline",
            details=("Internet ok", "AI ok", "System ok", "Zeit 12:35"),
            state_fields=(
                ("Status", "Waiting"),
                ("Internet", "ok"),
                ("AI", "ok"),
                ("System", "ok"),
                ("Zeit", "12:35"),
            ),
            log_sections=(
                ("System Log", ("12:34 now Waiting",)),
                ("LLM Log", ("user Hallo",)),
                ("Hardware Log", ("button green",)),
            ),
            animation_frame=0,
            face_cue=DisplayFaceCue(mouth="smile"),
            emoji_cue=DisplayEmojiCue(symbol="sparkles"),
            ambient_impulse_cue=DisplayAmbientImpulseCue(
                topic_key="ai companions",
                headline="AI companions",
                body="Da tut sich etwas.",
            ),
            service_connect_cue=None,
            presentation_cue=None,
        )

        self.assertEqual(first_signature, second_signature)

    def test_debug_log_layout_defers_rapid_rerender_when_status_is_stable(self) -> None:
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(
                display_layout="debug_log",
                display_poll_interval_s=0.0,
                openai_api_key="sk-test",
            ),
            display=FakeDisplay(),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=lambda _line: None,
            sleep=lambda _seconds: None,
            health_collector=lambda _config, *, snapshot=None: self.make_health(),
            internet_probe=lambda: True,
            clock=self.make_clock(),
        )
        last_signature = ("debug_log", "waiting", "Waiting", (("System Log", ("a",)),))
        next_signature = ("debug_log", "waiting", "Waiting", (("System Log", ("b",)),))
        loop._last_render_status = "waiting"
        loop._last_render_monotonic_s = 100.0

        with mock.patch.object(display_service_mod.time, "monotonic", return_value=110.0):
            self.assertFalse(
                loop._should_render_signature(
                    status="waiting",
                    signature=next_signature,
                    last_signature=last_signature,
                )
            )

        with mock.patch.object(display_service_mod.time, "monotonic", return_value=131.0):
            self.assertTrue(
                loop._should_render_signature(
                    status="waiting",
                    signature=next_signature,
                    last_signature=last_signature,
                )
            )

        with mock.patch.object(display_service_mod.time, "monotonic", return_value=110.0):
            self.assertTrue(
                loop._should_render_signature(
                    status="answering",
                    signature=("debug_log", "answering", "Answering", (("System Log", ("b",)),)),
                    last_signature=last_signature,
                )
            )

    def test_build_status_content_shows_missing_ai_when_no_key_is_configured(self) -> None:
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(display_poll_interval_s=0.0),
            display=FakeDisplay(),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=lambda _line: None,
            sleep=lambda _seconds: None,
            health_collector=lambda _config, *, snapshot=None: self.make_health(),
            internet_probe=lambda: True,
            clock=self.make_clock(),
        )

        _headline, details = loop._build_status_content(RuntimeSnapshot(status="waiting"))

        self.assertEqual(details, ("Internet ok", "AI fehlt", "System ok", "Zeit 12:34"))

    def test_system_state_ignores_temperature_only_warn_for_primary_status_card(self) -> None:
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(display_poll_interval_s=0.0),
            display=FakeDisplay(),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=lambda _line: None,
            sleep=lambda _seconds: None,
            health_collector=lambda _config, *, snapshot=None: self.make_health(
                status="warn",
                cpu_temperature_c=73.5,
            ),
            internet_probe=lambda: True,
            clock=self.make_clock(),
        )

        system_value = loop._system_state_value(RuntimeSnapshot(status="waiting"), self.make_health(status="warn", cpu_temperature_c=73.5))

        self.assertEqual(system_value, "ok")

    def test_waiting_animation_frame_changes_over_time(self) -> None:
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(display_poll_interval_s=0.0),
            display=FakeDisplay(),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=lambda _line: None,
            sleep=lambda _seconds: None,
        )

        original_monotonic = time.monotonic
        try:
            time.monotonic = lambda: 0.0
            first = loop._animation_frame("waiting")
            time.monotonic = lambda: 24.5
            second = loop._animation_frame("waiting")
        finally:
            time.monotonic = original_monotonic

        self.assertNotEqual(first, second)

    def test_default_layout_suppresses_waiting_animation_frame_for_static_backends(self) -> None:
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(display_poll_interval_s=0.0, display_driver="waveshare_4in2_v2"),
            display=FakeDisplay(),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=lambda _line: None,
            sleep=lambda _seconds: None,
        )

        original_monotonic = time.monotonic
        try:
            time.monotonic = lambda: 0.0
            first = loop._display_animation_frame("waiting")
            time.monotonic = lambda: 24.5
            second = loop._display_animation_frame("waiting")
        finally:
            time.monotonic = original_monotonic

        self.assertEqual(first, 0)
        self.assertEqual(second, 0)

    def test_default_layout_animates_waiting_frame_for_hdmi_idle_backends(self) -> None:
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(display_poll_interval_s=0.0, display_driver="hdmi_wayland"),
            display=IdleAnimatedDisplay(),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=lambda _line: None,
            sleep=lambda _seconds: None,
        )

        original_monotonic = time.monotonic
        try:
            time.monotonic = lambda: 0.0
            first = loop._display_animation_frame("waiting")
            time.monotonic = lambda: 4.5
            second = loop._display_animation_frame("waiting")
        finally:
            time.monotonic = original_monotonic

        self.assertNotEqual(first, second)

    def test_display_status_telemetry_suppresses_duplicate_idle_lines(self) -> None:
        emit_lines: list[str] = []
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(display_poll_interval_s=0.0, display_driver="hdmi_wayland"),
            display=IdleAnimatedDisplay(),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=emit_lines.append,
            sleep=lambda _seconds: None,
        )

        loop._emit_display_status(status="waiting", presentation_cue=None)
        loop._emit_display_status(status="waiting", presentation_cue=None)

        self.assertEqual(emit_lines, ["display_status=waiting"])

    def test_display_status_telemetry_emits_when_presentation_stage_changes(self) -> None:
        emit_lines: list[str] = []
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(display_poll_interval_s=0.0, display_driver="hdmi_wayland"),
            display=IdleAnimatedDisplay(),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=emit_lines.append,
            sleep=lambda _seconds: None,
        )
        now = datetime.now(timezone.utc)
        lifting = DisplayPresentationCue(
            kind="image",
            title="Photo",
            updated_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=10)).isoformat(),
        )
        focused = DisplayPresentationCue(
            kind="image",
            title="Photo",
            updated_at=(now - timedelta(seconds=2)).isoformat(),
            expires_at=(now + timedelta(seconds=10)).isoformat(),
        )

        loop._emit_display_status(status="waiting", presentation_cue=lifting)
        loop._emit_display_status(status="waiting", presentation_cue=focused)

        self.assertEqual(len(emit_lines), 2)
        self.assertTrue(all(line.startswith("display_status=waiting") for line in emit_lines))
        self.assertNotEqual(emit_lines[0], emit_lines[1])
        self.assertIn("presentation=image:", emit_lines[0])
        self.assertIn("presentation=image:focused", emit_lines[1])

    def test_display_loop_forwards_active_face_cue_to_default_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = root / "state" / "runtime-state.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            store = RuntimeSnapshotStore(snapshot_path)
            store.save(
                status="waiting",
                memory_turns=(),
                last_transcript=None,
                last_response="Hallo Thom",
            )
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(snapshot_path),
                display_poll_interval_s=0.0,
                openai_api_key="sk-test",
            )
            face_cue_store = DisplayFaceCueStore.from_config(config)
            face_cue_store.save(
                DisplayFaceCue(gaze_x=2, gaze_y=-1, mouth="smile", brows="focus"),
                hold_seconds=15.0,
            )
            display = FakeDisplay()
            loop = TwinrStatusDisplayLoop(
                config=config,
                display=display,
                snapshot_store=store,
                emit=lambda _line: None,
                sleep=lambda _seconds: None,
                health_collector=lambda _config, *, snapshot=None: self.make_health(),
                internet_probe=lambda: True,
                clock=self.make_clock(),
                face_cue_store=face_cue_store,
            )

            loop.run(max_cycles=1)

        self.assertEqual(len(display.calls), 1)
        cue = display.calls[0][7]
        self.assertIsNotNone(cue)
        assert cue is not None
        self.assertEqual(cue.gaze_x, 2)
        self.assertEqual(cue.gaze_y, -1)
        self.assertEqual(cue.mouth, "smile")
        self.assertEqual(cue.brows, "inward_tilt")

    def test_display_loop_forwards_active_emoji_cue_to_default_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = root / "state" / "runtime-state.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            store = RuntimeSnapshotStore(snapshot_path)
            store.save(
                status="waiting",
                memory_turns=(),
                last_transcript=None,
                last_response="Hallo Thom",
            )
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(snapshot_path),
                display_poll_interval_s=0.0,
                openai_api_key="sk-test",
            )
            emoji_cue_store = DisplayEmojiCueStore.from_config(config)
            emoji_cue_store.save(
                DisplayEmojiCue(symbol="thumbs_up", accent="success"),
                hold_seconds=15.0,
            )
            display = FakeDisplay()
            loop = TwinrStatusDisplayLoop(
                config=config,
                display=display,
                snapshot_store=store,
                emit=lambda _line: None,
                sleep=lambda _seconds: None,
                health_collector=lambda _config, *, snapshot=None: self.make_health(),
                internet_probe=lambda: True,
                clock=self.make_clock(),
                emoji_cue_store=emoji_cue_store,
            )

            loop.run(max_cycles=1)

        self.assertEqual(len(display.calls), 1)
        cue = display.calls[0][8]
        self.assertIsNotNone(cue)
        assert cue is not None
        self.assertEqual(cue.symbol, "thumbs_up")
        self.assertEqual(cue.accent, "success")

    def test_display_loop_forwards_active_ambient_impulse_cue_to_default_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = root / "state" / "runtime-state.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            store = RuntimeSnapshotStore(snapshot_path)
            store.save(
                status="waiting",
                memory_turns=(),
                last_transcript=None,
                last_response="Hallo Thom",
            )
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(snapshot_path),
                display_poll_interval_s=0.0,
                openai_api_key="sk-test",
            )
            ambient_store = DisplayAmbientImpulseCueStore.from_config(config)
            ambient_store.save(
                DisplayAmbientImpulseCue(
                    source="proactive_ambient_impulse",
                    topic_key="ai companions",
                    eyebrow="GEMEINSAMER FADEN",
                    headline="AI companions",
                    body="Wenn du magst, schauen wir spaeter kurz darauf.",
                    symbol="heart",
                    accent="warm",
                    action="invite_follow_up",
                    attention_state="shared_thread",
                ),
                hold_seconds=15.0,
            )
            display = FakeDisplay()
            loop = TwinrStatusDisplayLoop(
                config=config,
                display=display,
                snapshot_store=store,
                emit=lambda _line: None,
                sleep=lambda _seconds: None,
                health_collector=lambda _config, *, snapshot=None: self.make_health(),
                internet_probe=lambda: True,
                clock=self.make_clock(),
                ambient_impulse_cue_store=ambient_store,
            )

            loop.run(max_cycles=1)

        self.assertEqual(len(display.calls), 1)
        cue = display.calls[0][9]
        self.assertIsNotNone(cue)
        assert cue is not None
        self.assertEqual(cue.topic_key, "ai companions")
        self.assertEqual(cue.symbol, "heart")
        self.assertEqual(cue.action, "invite_follow_up")
        self.assertEqual(cue.attention_state, "shared_thread")

    def test_display_loop_forwards_active_service_connect_cue_to_default_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = root / "state" / "runtime-state.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            store = RuntimeSnapshotStore(snapshot_path)
            store.save(
                status="waiting",
                memory_turns=(),
                last_transcript=None,
                last_response="Hallo Thom",
            )
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(snapshot_path),
                display_poll_interval_s=0.0,
                openai_api_key="sk-test",
            )
            service_connect_store = DisplayServiceConnectCueStore.from_config(config)
            service_connect_store.save(
                DisplayServiceConnectCue(
                    service_id="whatsapp",
                    service_label="WhatsApp",
                    phase="qr",
                    summary="Scan the QR",
                    detail="Open Linked Devices in WhatsApp.",
                    qr_image_data_url="data:image/png;base64,AAAA",
                    accent="warm",
                ),
                hold_seconds=15.0,
            )
            display = FakeDisplay()
            loop = TwinrStatusDisplayLoop(
                config=config,
                display=display,
                snapshot_store=store,
                emit=lambda _line: None,
                sleep=lambda _seconds: None,
                health_collector=lambda _config, *, snapshot=None: self.make_health(),
                internet_probe=lambda: True,
                clock=self.make_clock(),
                service_connect_cue_store=service_connect_store,
            )

            loop.run(max_cycles=1)

        self.assertEqual(len(display.calls), 1)
        cue = display.calls[0][10]
        self.assertIsNotNone(cue)
        assert cue is not None
        self.assertEqual(cue.service_id, "whatsapp")
        self.assertEqual(cue.phase, "qr")
        self.assertEqual(cue.qr_image_data_url, "data:image/png;base64,AAAA")
        self.assertEqual(cue.accent, "warm")

    def test_display_loop_forwards_news_ticker_text_to_default_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "state" / "runtime-state.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            store = RuntimeSnapshotStore(snapshot_path)
            store.save(
                status="waiting",
                memory_turns=(),
                last_transcript=None,
                last_response="Hallo Thom",
            )
            display = FakeDisplay()
            ticker_runtime = mock.Mock()
            ticker_runtime.current_text.return_value = "Tagesschau · Major headline"
            loop = TwinrStatusDisplayLoop(
                config=TwinrConfig(
                    runtime_state_path=str(snapshot_path),
                    display_poll_interval_s=0.0,
                    openai_api_key="sk-test",
                ),
                display=display,
                snapshot_store=store,
                emit=lambda _line: None,
                sleep=lambda _seconds: None,
                health_collector=lambda _config, *, snapshot=None: self.make_health(),
                internet_probe=lambda: True,
                clock=self.make_clock(),
                news_ticker_runtime=ticker_runtime,
            )

            loop.run(max_cycles=1)

        self.assertEqual(len(display.calls), 1)
        self.assertEqual(display.calls[0][2], "Tagesschau · Major headline")
        ticker_runtime.current_text.assert_called_once()

    def test_display_loop_forwards_active_presentation_cue_to_default_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = root / "state" / "runtime-state.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            store = RuntimeSnapshotStore(snapshot_path)
            store.save(
                status="waiting",
                memory_turns=(),
                last_transcript=None,
                last_response="Hallo Thom",
            )
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(snapshot_path),
                display_poll_interval_s=0.0,
                openai_api_key="sk-test",
            )
            presentation_store = DisplayPresentationStore.from_config(config)
            presentation_store.save(
                DisplayPresentationCue(
                    kind="rich_card",
                    title="Family Call",
                    subtitle="Marta is waiting",
                    body_lines=("Tap green and answer",),
                    accent="warm",
                ),
                hold_seconds=15.0,
            )
            display = FakeDisplay()
            loop = TwinrStatusDisplayLoop(
                config=config,
                display=display,
                snapshot_store=store,
                emit=lambda _line: None,
                sleep=lambda _seconds: None,
                health_collector=lambda _config, *, snapshot=None: self.make_health(),
                internet_probe=lambda: True,
                clock=self.make_clock(),
                presentation_cue_store=presentation_store,
            )

            loop.run(max_cycles=1)

        self.assertEqual(len(display.calls), 1)
        cue = display.calls[0][11]
        self.assertIsNotNone(cue)
        assert cue is not None
        self.assertEqual(cue.kind, "rich_card")
        self.assertEqual(cue.title, "Family Call")
        self.assertEqual(cue.body_lines, ("Tap green and answer",))
        self.assertEqual(cue.accent, "warm")

    def test_display_loop_forwards_active_debug_signals_to_default_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = root / "state" / "runtime-state.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            store = RuntimeSnapshotStore(snapshot_path)
            store.save(
                status="waiting",
                memory_turns=(),
                last_transcript=None,
                last_response="Hallo Thom",
            )
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(snapshot_path),
                display_poll_interval_s=0.0,
                openai_api_key="sk-test",
            )
            debug_signal_store = DisplayDebugSignalStore.from_config(config)
            save_now = datetime.now(timezone.utc)
            debug_signal_store.save(
                DisplayDebugSignalSnapshot(
                    source="test",
                    signals=(
                        DisplayDebugSignal(
                            key="motion_state",
                            label="MOTION_STILL",
                            accent="neutral",
                            priority=90,
                        ),
                        DisplayDebugSignal(
                            key="person_visible",
                            label="PERSON_1",
                            accent="info",
                            priority=70,
                        ),
                    ),
                ),
                hold_seconds=15.0,
                now=save_now,
            )
            display = FakeDisplay()
            loop = TwinrStatusDisplayLoop(
                config=config,
                display=display,
                snapshot_store=store,
                emit=lambda _line: None,
                sleep=lambda _seconds: None,
                health_collector=lambda _config, *, snapshot=None: self.make_health(),
                internet_probe=lambda: True,
                clock=self.make_clock(),
                debug_signal_store=debug_signal_store,
            )

            loop.run(max_cycles=1)

        self.assertEqual(len(display.calls), 1)
        signals = display.calls[0][12]
        self.assertEqual(
            tuple(signal.label for signal in signals),
            ("MOTION_STILL", "PERSON_1"),
        )

    def test_all_statuses_have_six_to_twelve_frames(self) -> None:
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(display_poll_interval_s=0.0),
            display=FakeDisplay(),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=lambda _line: None,
            sleep=lambda _seconds: None,
        )

        for status in ("waiting", "listening", "processing", "answering", "printing", "error"):
            frame_count, _frame_seconds = loop._animation_spec(status)
            self.assertGreaterEqual(frame_count, 6)
            self.assertLessEqual(frame_count, 12)


if __name__ == "__main__":
    unittest.main()
