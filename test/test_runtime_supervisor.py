from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshot
from twinr.agent.workflows.required_remote_snapshot import RequiredRemoteWatchdogAssessment
from twinr.ops.health import ServiceHealth, TwinrSystemHealth
from twinr.ops.remote_memory_watchdog import RemoteMemoryWatchdogStore
from twinr.ops.runtime_supervisor import RUNTIME_SUPERVISOR_ENV_KEY, TwinrRuntimeSupervisor


class _FakeClock:
    def __init__(self) -> None:
        self._monotonic = 0.0
        self._utcnow = datetime(2026, 3, 16, 21, 0, tzinfo=timezone.utc)

    def monotonic(self) -> float:
        return self._monotonic

    def sleep(self, seconds: float) -> None:
        delta = max(0.0, float(seconds))
        self._monotonic += delta
        self._utcnow += timedelta(seconds=delta)

    def utcnow(self) -> datetime:
        return self._utcnow


class _FakeProcess:
    _next_pid = 3200

    def __init__(self, key: str) -> None:
        self.key = key
        self.pid = _FakeProcess._next_pid
        _FakeProcess._next_pid += 1
        self.exit_code: int | None = None

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.exit_code = 0

    def kill(self) -> None:
        self.exit_code = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.exit_code is None:
            self.exit_code = 0
        return self.exit_code


class _RecordingProcessFactory:
    def __init__(self, clock: _FakeClock) -> None:
        self.clock = clock
        self.calls: list[dict[str, object]] = []
        self.processes: list[_FakeProcess] = []

    def __call__(self, argv, *, cwd, env):
        key = "watchdog" if "--watch-remote-memory" in argv else "streaming"
        process = _FakeProcess(key)
        self.processes.append(process)
        self.calls.append(
            {
                "key": key,
                "argv": tuple(argv),
                "cwd": Path(cwd),
                "env": dict(env),
                "started_at": self.clock.monotonic(),
                "process": process,
            }
        )
        return process


class _FreshSnapshotStore:
    def __init__(self, clock: _FakeClock) -> None:
        self.clock = clock

    def load(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            status="waiting",
            updated_at=self.clock.utcnow().isoformat(),
        )


class _StaleSnapshotStore:
    def load(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            status="waiting",
            updated_at="2026-03-16T20:00:00+00:00",
        )


class _RuntimeSnapshotStore:
    def __init__(self, snapshot_factory) -> None:
        self.snapshot_factory = snapshot_factory

    def load(self) -> RuntimeSnapshot:
        return self.snapshot_factory()


def _build_assessment(
    *,
    ready: bool,
    detail: str = "ok",
    pid_alive: bool = True,
    sample_status: str | None = "ok",
    sample_ready: bool | None = None,
    sample_required: bool | None = True,
    sample_age_s: float | None = 1.0,
    max_sample_age_s: float = 10.0,
    watchdog_pid: int | None = None,
    snapshot_updated_at: str | None = None,
) -> RequiredRemoteWatchdogAssessment:
    return RequiredRemoteWatchdogAssessment(
        ready=ready,
        detail=detail,
        artifact_path="/tmp/remote-memory-watchdog.json",
        pid_alive=pid_alive,
        sample_age_s=sample_age_s,
        max_sample_age_s=max_sample_age_s,
        sample_status=sample_status,
        sample_ready=ready if sample_ready is None else sample_ready,
        sample_required=sample_required,
        sample_latency_ms=1000.0,
        watchdog_pid=watchdog_pid,
        snapshot_updated_at=snapshot_updated_at,
    )


def _build_health(*, display_running: bool, display_count: int = 1) -> TwinrSystemHealth:
    return TwinrSystemHealth(
        status="ok",
        captured_at="2026-03-16T21:00:00Z",
        hostname="test-host",
        runtime_status="waiting",
        runtime_updated_at="2026-03-16T21:00:00Z",
        services=(
            ServiceHealth(
                key="conversation_loop",
                label="Conversation loop",
                running=True,
                count=1,
                detail="pid=123 python --run-streaming-loop",
            ),
            ServiceHealth(
                key="display",
                label="Display loop",
                running=display_running,
                count=display_count,
                detail="pid=123 display-companion" if display_running else "Service not detected.",
            ),
        ),
    )


class RuntimeSupervisorTests(unittest.TestCase):
    def _build_config(self, root: Path) -> TwinrConfig:
        return TwinrConfig(
            project_root=str(root),
            runtime_state_path=str(root / "runtime-state.json"),
        )

    def test_run_starts_watchdog_and_streaming_when_watchdog_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=_FreshSnapshotStore(clock),
                health_collector=lambda *_args, **_kwargs: _build_health(display_running=True),
                watchdog_assessor=lambda _config: _build_assessment(ready=True),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                streaming_health_grace_s=999.0,
                restart_backoff_s=0.0,
            )

            exit_code = supervisor.run(duration_s=0.0)

        self.assertEqual(exit_code, 0)
        self.assertEqual([call["key"] for call in factory.calls], ["watchdog", "streaming"])
        self.assertEqual(factory.calls[0]["cwd"], root)
        self.assertEqual(factory.calls[0]["env"][RUNTIME_SUPERVISOR_ENV_KEY], "1")
        self.assertIn("src", factory.calls[0]["env"]["PYTHONPATH"].split(":"))

    def test_run_writes_watchdog_bootstrap_snapshot_for_fresh_child(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=_FreshSnapshotStore(clock),
                health_collector=lambda *_args, **_kwargs: _build_health(display_running=True),
                watchdog_assessor=lambda _config: _build_assessment(ready=True),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                streaming_health_grace_s=999.0,
                restart_backoff_s=0.0,
            )

            supervisor.run(duration_s=0.0)
            snapshot = RemoteMemoryWatchdogStore.from_config(config).load()

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.current.status, "starting")
        self.assertTrue(snapshot.probe_inflight)
        self.assertEqual(snapshot.pid, factory.calls[0]["process"].pid)

    def test_run_primes_audio_session_env_before_spawning_children(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            original = {
                key: os.environ.get(key)
                for key in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "PULSE_SERVER")
            }
            for key in original:
                os.environ.pop(key, None)

            def fake_prime() -> dict[str, str]:
                os.environ["XDG_RUNTIME_DIR"] = "/run/user/1234"
                os.environ["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=/run/user/1234/bus"
                os.environ["PULSE_SERVER"] = "unix:/run/user/1234/pulse/native"
                return {
                    "XDG_RUNTIME_DIR": os.environ["XDG_RUNTIME_DIR"],
                    "DBUS_SESSION_BUS_ADDRESS": os.environ["DBUS_SESSION_BUS_ADDRESS"],
                    "PULSE_SERVER": os.environ["PULSE_SERVER"],
                }

            try:
                with mock.patch(
                    "twinr.ops.runtime_supervisor.prime_user_session_audio_env",
                    side_effect=fake_prime,
                ):
                    supervisor = TwinrRuntimeSupervisor(
                        config=config,
                        env_file=root / ".env",
                        process_factory=factory,
                        snapshot_store=_FreshSnapshotStore(clock),
                        health_collector=lambda *_args, **_kwargs: _build_health(display_running=True),
                        watchdog_assessor=lambda _config: _build_assessment(ready=True),
                        monotonic=clock.monotonic,
                        sleep=clock.sleep,
                        utcnow=clock.utcnow,
                        streaming_health_grace_s=999.0,
                        restart_backoff_s=0.0,
                    )

                    supervisor.run(duration_s=0.0)
            finally:
                for key, value in original.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        self.assertEqual(factory.calls[0]["env"]["XDG_RUNTIME_DIR"], "/run/user/1234")
        self.assertEqual(
            factory.calls[0]["env"]["DBUS_SESSION_BUS_ADDRESS"],
            "unix:path=/run/user/1234/bus",
        )
        self.assertEqual(
            factory.calls[0]["env"]["PULSE_SERVER"],
            "unix:/run/user/1234/pulse/native",
        )

    def test_run_delays_streaming_until_watchdog_startup_grace_expires(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=_FreshSnapshotStore(clock),
                health_collector=lambda *_args, **_kwargs: _build_health(display_running=True),
                watchdog_assessor=lambda _config: _build_assessment(
                    ready=False,
                    detail="Remote memory watchdog snapshot is missing.",
                    sample_status=None,
                    sample_ready=False,
                    sample_age_s=None,
                ),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                watchdog_startup_grace_s=5.0,
                streaming_health_grace_s=999.0,
                restart_backoff_s=0.0,
            )

            supervisor.run(duration_s=6.0)

        self.assertEqual([call["key"] for call in factory.calls], ["watchdog", "streaming"])
        streaming_start = next(call["started_at"] for call in factory.calls if call["key"] == "streaming")
        self.assertGreaterEqual(streaming_start, 5.0)

    def test_run_can_consume_external_watchdog_without_spawning_watchdog_child(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=_FreshSnapshotStore(clock),
                health_collector=lambda *_args, **_kwargs: _build_health(display_running=True),
                watchdog_assessor=lambda _config: _build_assessment(ready=True, watchdog_pid=4321),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                streaming_health_grace_s=999.0,
                restart_backoff_s=0.0,
                manage_watchdog=False,
            )

            exit_code = supervisor.run(duration_s=0.0)

        self.assertEqual(exit_code, 0)
        self.assertEqual([call["key"] for call in factory.calls], ["streaming"])

    def test_run_adopts_existing_streaming_lock_owner_after_supervisor_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            external_pid = 4242
            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=_FreshSnapshotStore(clock),
                health_collector=lambda *_args, **_kwargs: _build_health(display_running=True),
                watchdog_assessor=lambda _config: _build_assessment(ready=True, watchdog_pid=4321),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                loop_owner=lambda _config, name: external_pid if name == "streaming-loop" else None,
                pid_alive=lambda pid: pid == external_pid,
                pid_signal=lambda _pid, _sig: None,
                pid_cmdline=lambda pid: ("python", "-m", "twinr", "--run-streaming-loop") if pid == external_pid else (),
                streaming_health_grace_s=999.0,
                restart_backoff_s=0.0,
                manage_watchdog=False,
            )

            exit_code = supervisor.run(duration_s=0.0)

        self.assertEqual(exit_code, 0)
        self.assertEqual(factory.calls, [])

    def test_run_starts_streaming_when_existing_lock_owner_is_not_alive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=_FreshSnapshotStore(clock),
                health_collector=lambda *_args, **_kwargs: _build_health(display_running=True),
                watchdog_assessor=lambda _config: _build_assessment(ready=True, watchdog_pid=4321),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                loop_owner=lambda _config, name: 4242 if name == "streaming-loop" else None,
                pid_alive=lambda _pid: False,
                pid_signal=lambda _pid, _sig: None,
                pid_cmdline=lambda _pid: (),
                streaming_health_grace_s=999.0,
                restart_backoff_s=0.0,
                manage_watchdog=False,
            )

            exit_code = supervisor.run(duration_s=0.0)

        self.assertEqual(exit_code, 0)
        self.assertEqual([call["key"] for call in factory.calls], ["streaming"])

    def test_run_does_not_adopt_live_non_streaming_lock_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=_FreshSnapshotStore(clock),
                health_collector=lambda *_args, **_kwargs: _build_health(display_running=True),
                watchdog_assessor=lambda _config: _build_assessment(ready=True, watchdog_pid=4321),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                loop_owner=lambda _config, name: 4242 if name == "streaming-loop" else None,
                pid_alive=lambda pid: pid == 4242,
                pid_signal=lambda _pid, _sig: None,
                pid_cmdline=lambda _pid: ("python", "holder.py"),
                streaming_health_grace_s=999.0,
                restart_backoff_s=0.0,
                manage_watchdog=False,
            )

            exit_code = supervisor.run(duration_s=0.0)

        self.assertEqual(exit_code, 0)
        self.assertEqual([call["key"] for call in factory.calls], ["streaming"])

    def test_run_delays_streaming_until_external_watchdog_startup_grace_expires(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=_FreshSnapshotStore(clock),
                health_collector=lambda *_args, **_kwargs: _build_health(display_running=True),
                watchdog_assessor=lambda _config: _build_assessment(
                    ready=False,
                    detail="Remote memory watchdog snapshot is missing.",
                    sample_status=None,
                    sample_ready=False,
                    sample_age_s=None,
                ),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                watchdog_startup_grace_s=5.0,
                streaming_health_grace_s=999.0,
                restart_backoff_s=0.0,
                manage_watchdog=False,
            )

            supervisor.run(duration_s=6.0)

        self.assertEqual([call["key"] for call in factory.calls], ["streaming"])
        self.assertGreaterEqual(factory.calls[0]["started_at"], 5.0)

    def test_run_attempts_external_watchdog_recovery_when_external_owner_is_dead(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            recovery_calls: list[dict[str, object]] = []

            def _starter(start_config: TwinrConfig, env_file: str, emit) -> int | None:
                recovery_calls.append(
                    {
                        "project_root": start_config.project_root,
                        "env_file": env_file,
                    }
                )
                emit("remote_memory_watchdog=spawned")
                return 9876

            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=_FreshSnapshotStore(clock),
                health_collector=lambda *_args, **_kwargs: _build_health(display_running=True),
                watchdog_assessor=lambda _config: _build_assessment(
                    ready=False,
                    detail="Remote memory watchdog process 485052 is not alive.",
                    pid_alive=False,
                    sample_ready=False,
                    watchdog_pid=485052,
                ),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                streaming_health_grace_s=999.0,
                restart_backoff_s=0.0,
                manage_watchdog=False,
                external_watchdog_starter=_starter,
            )

            exit_code = supervisor.run(duration_s=0.0)

        self.assertEqual(exit_code, 0)
        self.assertEqual(factory.calls, [])
        self.assertEqual(
            recovery_calls,
            [{"project_root": str(root), "env_file": str(root / ".env")}],
        )

    def test_run_starts_streaming_after_external_watchdog_recovery_restores_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            recovery_calls: list[dict[str, object]] = []
            assessments = [
                _build_assessment(
                    ready=False,
                    detail="Remote memory watchdog process 485052 is not alive.",
                    pid_alive=False,
                    sample_ready=False,
                    watchdog_pid=485052,
                ),
                _build_assessment(
                    ready=True,
                    detail="ok",
                    pid_alive=True,
                    watchdog_pid=9876,
                ),
            ]

            def _assessor(_config: TwinrConfig) -> RequiredRemoteWatchdogAssessment:
                if assessments:
                    return assessments.pop(0)
                return _build_assessment(ready=True, detail="ok", pid_alive=True, watchdog_pid=9876)

            def _starter(start_config: TwinrConfig, env_file: str, emit) -> int | None:
                recovery_calls.append(
                    {
                        "project_root": start_config.project_root,
                        "env_file": env_file,
                    }
                )
                emit("remote_memory_watchdog=spawned")
                return 9876

            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=_FreshSnapshotStore(clock),
                health_collector=lambda *_args, **_kwargs: _build_health(display_running=True),
                watchdog_assessor=_assessor,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                streaming_health_grace_s=999.0,
                restart_backoff_s=0.0,
                manage_watchdog=False,
                external_watchdog_starter=_starter,
            )

            exit_code = supervisor.run(duration_s=1.0)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            recovery_calls,
            [{"project_root": str(root), "env_file": str(root / ".env")}],
        )
        self.assertEqual([call["key"] for call in factory.calls], ["streaming"])
        self.assertGreaterEqual(factory.calls[0]["started_at"], 1.0)

    def test_run_does_not_restart_streaming_for_persistent_missing_display_health(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            health_call_count = 0

            def _health_collector(*_args, **_kwargs):
                nonlocal health_call_count
                health_call_count += 1
                if health_call_count <= 2:
                    return _build_health(display_running=False, display_count=0)
                return _build_health(display_running=True)

            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=_FreshSnapshotStore(clock),
                health_collector=_health_collector,
                watchdog_assessor=lambda _config: _build_assessment(ready=True),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                loop_owner=lambda _config, name: (
                    next(
                        (
                            process.pid
                            for process in reversed(factory.processes)
                            if process.key == "streaming"
                        ),
                        None,
                    )
                    if name == "streaming-loop"
                    else None
                ),
                streaming_health_grace_s=0.0,
                restart_backoff_s=0.0,
            )

            supervisor.run(duration_s=2.0)

        self.assertEqual([call["key"] for call in factory.calls].count("streaming"), 1)

    def test_run_waits_for_streaming_startup_progress_before_enforcing_display_health(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            health_call_count = 0

            def _health_collector(*_args, **_kwargs):
                nonlocal health_call_count
                health_call_count += 1
                return _build_health(display_running=False, display_count=0)

            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=_StaleSnapshotStore(),
                health_collector=_health_collector,
                watchdog_assessor=lambda _config: _build_assessment(ready=True),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                loop_owner=lambda _config, _name: None,
                streaming_startup_timeout_s=30.0,
                streaming_health_grace_s=0.0,
                restart_backoff_s=0.0,
            )

            supervisor.run(duration_s=12.0)

        self.assertEqual([call["key"] for call in factory.calls].count("streaming"), 1)
        self.assertEqual(health_call_count, 0)

    def test_run_does_not_treat_fresh_runtime_snapshot_as_streaming_lock_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            health_call_count = 0

            def _health_collector(*_args, **_kwargs):
                nonlocal health_call_count
                health_call_count += 1
                return _build_health(display_running=False, display_count=0)

            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=_FreshSnapshotStore(clock),
                health_collector=_health_collector,
                watchdog_assessor=lambda _config: _build_assessment(ready=True),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                loop_owner=lambda _config, _name: None,
                streaming_startup_timeout_s=30.0,
                streaming_health_grace_s=0.0,
                restart_backoff_s=0.0,
            )

            supervisor.run(duration_s=12.0)

        self.assertEqual([call["key"] for call in factory.calls].count("streaming"), 1)
        self.assertEqual(health_call_count, 0)

    def test_run_requires_current_streaming_child_to_own_streaming_lock_before_health_enforcement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            health_call_count = 0

            def _health_collector(*_args, **_kwargs):
                nonlocal health_call_count
                health_call_count += 1
                return _build_health(display_running=False, display_count=0)

            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=_FreshSnapshotStore(clock),
                health_collector=_health_collector,
                watchdog_assessor=lambda _config: _build_assessment(ready=True),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                loop_owner=lambda _config, name: 999999 if name == "streaming-loop" else None,
                streaming_startup_timeout_s=30.0,
                streaming_health_grace_s=0.0,
                restart_backoff_s=0.0,
            )

            supervisor.run(duration_s=12.0)

        self.assertEqual([call["key"] for call in factory.calls].count("streaming"), 1)
        self.assertEqual(health_call_count, 0)

    def test_run_does_not_require_display_heartbeat_before_enforcing_health(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            health_call_count = 0

            def _health_collector(*_args, **_kwargs):
                nonlocal health_call_count
                health_call_count += 1
                return _build_health(display_running=False, display_count=0)

            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=_FreshSnapshotStore(clock),
                health_collector=_health_collector,
                watchdog_assessor=lambda _config: _build_assessment(ready=True),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                loop_owner=lambda _config, name: (
                    next(
                        (
                            process.pid
                            for process in reversed(factory.processes)
                            if process.key == "streaming"
                        ),
                        None,
                    )
                    if name == "streaming-loop"
                    else None
                ),
                streaming_startup_timeout_s=30.0,
                streaming_health_grace_s=0.0,
                restart_backoff_s=0.0,
            )

            supervisor.run(duration_s=12.0)

        self.assertEqual([call["key"] for call in factory.calls].count("streaming"), 1)
        self.assertGreaterEqual(health_call_count, 2)

    def test_run_does_not_restart_streaming_when_display_heartbeat_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            health_call_count = 0

            def _health_collector(*_args, **_kwargs):
                nonlocal health_call_count
                health_call_count += 1
                return _build_health(display_running=False, display_count=0)

            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=_FreshSnapshotStore(clock),
                health_collector=_health_collector,
                watchdog_assessor=lambda _config: _build_assessment(ready=True),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                loop_owner=lambda _config, name: (
                    next(
                        (
                            process.pid
                            for process in reversed(factory.processes)
                            if process.key == "streaming"
                        ),
                        None,
                    )
                    if name == "streaming-loop"
                    else None
                ),
                streaming_health_grace_s=0.0,
                restart_backoff_s=0.0,
            )

            supervisor.run(duration_s=70.0)

        self.assertGreaterEqual(health_call_count, 2)
        self.assertEqual([call["key"] for call in factory.calls].count("streaming"), 1)

    def test_run_waits_for_current_watchdog_startup_progress_before_restarting_stale_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=_FreshSnapshotStore(clock),
                health_collector=lambda *_args, **_kwargs: _build_health(display_running=True),
                watchdog_assessor=lambda _config: _build_assessment(
                    ready=False,
                    detail="Remote memory watchdog process 999999 is not alive.",
                    pid_alive=False,
                    sample_status="fail",
                    sample_ready=False,
                    watchdog_pid=999999,
                    snapshot_updated_at="2026-03-16T20:00:00+00:00",
                ),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                watchdog_startup_grace_s=5.0,
                watchdog_startup_timeout_s=30.0,
                streaming_health_grace_s=999.0,
                restart_backoff_s=0.0,
            )

            supervisor.run(duration_s=12.0)

        self.assertEqual([call["key"] for call in factory.calls], ["watchdog", "streaming"])

    def test_run_waits_for_current_child_to_refresh_runtime_snapshot_before_enforcing_snapshot_health(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)
            health_call_count = 0

            def _health_collector(*_args, **_kwargs):
                nonlocal health_call_count
                health_call_count += 1
                return _build_health(display_running=True)

            snapshot_store = _RuntimeSnapshotStore(
                lambda: RuntimeSnapshot(
                    status="waiting",
                    updated_at="2026-03-16T20:00:00+00:00",
                )
            )
            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=snapshot_store,
                health_collector=_health_collector,
                watchdog_assessor=lambda _config: _build_assessment(ready=True),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                loop_owner=lambda _config, name: (
                    next(
                        (
                            process.pid
                            for process in reversed(factory.processes)
                            if process.key == "streaming"
                        ),
                        None,
                    )
                    if name == "streaming-loop"
                    else None
                ),
                streaming_startup_timeout_s=30.0,
                streaming_health_grace_s=0.0,
                restart_backoff_s=0.0,
            )

            supervisor.run(duration_s=12.0)

        self.assertEqual([call["key"] for call in factory.calls].count("streaming"), 1)
        self.assertEqual(health_call_count, 0)

    def test_waiting_snapshot_age_does_not_restart_an_otherwise_healthy_child(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)

            snapshot_store = _RuntimeSnapshotStore(
                lambda: RuntimeSnapshot(
                    status="waiting",
                    updated_at="2026-03-16T21:00:00+00:00",
                )
            )
            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=snapshot_store,
                health_collector=lambda *_args, **_kwargs: _build_health(display_running=True),
                watchdog_assessor=lambda _config: _build_assessment(ready=True),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                loop_owner=lambda _config, name: (
                    next(
                        (
                            process.pid
                            for process in reversed(factory.processes)
                            if process.key == "streaming"
                        ),
                        None,
                    )
                    if name == "streaming-loop"
                    else None
                ),
                streaming_health_grace_s=0.0,
                max_snapshot_age_s=5.0,
                restart_backoff_s=0.0,
            )

            supervisor.run(duration_s=12.0)

        self.assertEqual([call["key"] for call in factory.calls].count("streaming"), 1)

    def test_required_remote_error_does_not_restart_streaming_while_watchdog_is_unready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)

            snapshot_store = _RuntimeSnapshotStore(
                lambda: RuntimeSnapshot(
                    status="error",
                    updated_at=clock.utcnow().isoformat(),
                    error_message="Remote memory watchdog snapshot is stale.",
                )
            )
            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=snapshot_store,
                health_collector=lambda *_args, **_kwargs: _build_health(display_running=True),
                watchdog_assessor=lambda _config: _build_assessment(
                    ready=False,
                    detail="Remote memory watchdog snapshot is stale.",
                    sample_status="fail",
                    sample_ready=False,
                    sample_age_s=55.0,
                    max_sample_age_s=45.0,
                ),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                loop_owner=lambda _config, name: (
                    next(
                        (
                            process.pid
                            for process in reversed(factory.processes)
                            if process.key == "streaming"
                        ),
                        None,
                    )
                    if name == "streaming-loop"
                    else None
                ),
                streaming_health_grace_s=0.0,
                restart_backoff_s=0.0,
            )

            supervisor.run(duration_s=12.0)

        self.assertEqual([call["key"] for call in factory.calls].count("streaming"), 1)

    def test_required_remote_error_restarts_streaming_once_watchdog_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)

            snapshot_store = _RuntimeSnapshotStore(
                lambda: RuntimeSnapshot(
                    status="error",
                    updated_at=clock.utcnow().isoformat(),
                    error_message="LongTermRemoteUnavailableError: Remote long-term memory is temporarily cooling down after recent failures.",
                )
            )
            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=snapshot_store,
                health_collector=lambda *_args, **_kwargs: _build_health(display_running=True),
                watchdog_assessor=lambda _config: _build_assessment(ready=True),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                loop_owner=lambda _config, name: (
                    next(
                        (
                            process.pid
                            for process in reversed(factory.processes)
                            if process.key == "streaming"
                        ),
                        None,
                    )
                    if name == "streaming-loop"
                    else None
                ),
                streaming_health_grace_s=0.0,
                restart_backoff_s=0.0,
            )

            supervisor.run(duration_s=1.5)

        self.assertEqual([call["key"] for call in factory.calls].count("streaming"), 2)

    def test_foreign_error_does_not_restart_streaming_while_watchdog_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._build_config(root)
            clock = _FakeClock()
            factory = _RecordingProcessFactory(clock)

            snapshot_store = _RuntimeSnapshotStore(
                lambda: RuntimeSnapshot(
                    status="error",
                    updated_at=clock.utcnow().isoformat(),
                    error_message="ReSpeaker XVF3800 is not visible to the Pi.",
                )
            )
            supervisor = TwinrRuntimeSupervisor(
                config=config,
                env_file=root / ".env",
                process_factory=factory,
                snapshot_store=snapshot_store,
                health_collector=lambda *_args, **_kwargs: _build_health(display_running=True),
                watchdog_assessor=lambda _config: _build_assessment(ready=True),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                utcnow=clock.utcnow,
                loop_owner=lambda _config, name: (
                    next(
                        (
                            process.pid
                            for process in reversed(factory.processes)
                            if process.key == "streaming"
                        ),
                        None,
                    )
                    if name == "streaming-loop"
                    else None
                ),
                streaming_health_grace_s=0.0,
                restart_backoff_s=0.0,
            )

            supervisor.run(duration_s=2.0)

        self.assertEqual([call["key"] for call in factory.calls].count("streaming"), 1)

    def test_base_agent_runtime_import_stays_free_of_workflow_init_cycles(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from twinr.agent.base_agent import TwinrRuntime; print(TwinrRuntime.__name__)",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env={"PYTHONPATH": "src", **dict(os.environ)},
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "TwinrRuntime")


if __name__ == "__main__":
    unittest.main()
