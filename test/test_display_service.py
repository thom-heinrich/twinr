from pathlib import Path
import sys
import tempfile
import time
import unittest
from datetime import datetime
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.display import service as display_service_mod
from twinr.display.heartbeat import DisplayHeartbeatStore
from twinr.display.service import TwinrStatusDisplayLoop
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshot, RuntimeSnapshotStore
from twinr.ops.health import ServiceHealth, TwinrSystemHealth


class FakeDisplay:
    def __init__(self) -> None:
        self.calls: list[
            tuple[
                str,
                str | None,
                tuple[str, ...],
                tuple[tuple[str, str], ...],
                tuple[tuple[str, tuple[str, ...]], ...],
                int,
            ]
        ] = []

    def show_status(
        self,
        status: str,
        *,
        headline: str | None = None,
        details: tuple[str, ...] = (),
        state_fields: tuple[tuple[str, str], ...] = (),
        log_sections: tuple[tuple[str, tuple[str, ...]], ...] = (),
        animation_frame: int = 0,
    ) -> None:
        self.calls.append((status, headline, details, state_fields, log_sections, animation_frame))


class FakeDebugLogBuilder:
    def __init__(self, sections: tuple[tuple[str, tuple[str, ...]], ...]) -> None:
        self.sections = sections

    def build_sections(self, **_kwargs: object) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return self.sections


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
                [(status, headline, details) for status, headline, details, _fields, _logs, _frame in display.calls],
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
            display.calls[0][4],
            (
                ("System Log", ("12:34 now Waiting",)),
                ("LLM Log", ("user Hallo",)),
                ("Hardware Log", ("button green",)),
            ),
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

    def test_all_statuses_have_four_to_six_frames(self) -> None:
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(display_poll_interval_s=0.0),
            display=FakeDisplay(),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=lambda _line: None,
            sleep=lambda _seconds: None,
        )

        for status in ("waiting", "listening", "processing", "answering", "printing", "error"):
            frame_count, _frame_seconds = loop._animation_spec(status)
            self.assertGreaterEqual(frame_count, 4)
            self.assertLessEqual(frame_count, 6)


if __name__ == "__main__":
    unittest.main()
