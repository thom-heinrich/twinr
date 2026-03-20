from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.ops.pi_repo_mirror import PiRepoMirrorWatchdog
from twinr.ops.self_coding_pi import PiConnectionSettings


def _completed(
    args: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


class _FakeClock:
    def __init__(self) -> None:
        self.current = 0.0

    def monotonic(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.current += seconds


class PiRepoMirrorWatchdogTests(unittest.TestCase):
    def test_probe_once_reports_clean_repo_when_rsync_finds_no_changes(self) -> None:
        commands: list[list[str]] = []
        envs: list[dict[str, str] | None] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("Twinr\n", encoding="utf-8")

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                envs.append(kwargs.get("env"))
                return _completed(command, stdout="")

            watchdog = PiRepoMirrorWatchdog(
                project_root=root,
                connection_settings=PiConnectionSettings(
                    host="192.168.1.95",
                    user="thh",
                    password="chaos",
                ),
                subprocess_runner=_runner,
            )

            result = watchdog.probe_once()

        self.assertFalse(result.drift_detected)
        self.assertFalse(result.sync_applied)
        self.assertTrue(result.verified_clean)
        self.assertFalse(result.checksum_used)
        self.assertEqual(result.change_count, 0)
        self.assertEqual(len(commands), 1)
        joined = " ".join(commands[0])
        self.assertIn("sshpass -e rsync", joined)
        self.assertIn("--delete", joined)
        self.assertIn("--itemize-changes", joined)
        self.assertIn("--exclude=.env", joined)
        self.assertNotIn("--dry-run", commands[0])
        self.assertEqual(envs[0]["SSHPASS"], "chaos")

    def test_probe_once_heals_drift_and_runs_checksum_verification(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            responses = iter(
                (
                    _completed(
                        ["rsync"],
                        stdout=">f+++++++++ src/app.py\n*deleting   stale.py\n",
                    ),
                    _completed(["rsync"], stdout=""),
                )
            )

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                return next(responses)

            watchdog = PiRepoMirrorWatchdog(
                project_root=root,
                connection_settings=PiConnectionSettings(
                    host="192.168.1.95",
                    user="thh",
                    password="chaos",
                ),
                subprocess_runner=_runner,
            )

            result = watchdog.probe_once()

        self.assertTrue(result.drift_detected)
        self.assertTrue(result.sync_applied)
        self.assertTrue(result.verified_clean)
        self.assertEqual(result.change_count, 2)
        self.assertEqual(
            result.sampled_change_lines,
            (">f+++++++++ src/app.py", "*deleting   stale.py"),
        )
        self.assertEqual(len(commands), 2)
        self.assertNotIn("--checksum", commands[0])
        self.assertIn("--dry-run", commands[1])
        self.assertIn("--checksum", commands[1])

    def test_probe_once_supports_checksum_only_dry_run_mode(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "pyproject.toml").write_text("[project]\nname='twinr'\n", encoding="utf-8")

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                return _completed(
                    command,
                    stdout=">f..t...... pyproject.toml\n",
                )

            watchdog = PiRepoMirrorWatchdog(
                project_root=root,
                connection_settings=PiConnectionSettings(
                    host="192.168.1.95",
                    user="thh",
                    password="chaos",
                ),
                subprocess_runner=_runner,
            )

            result = watchdog.probe_once(apply_sync=False, checksum=True, max_change_lines=1)

        self.assertTrue(result.drift_detected)
        self.assertFalse(result.sync_applied)
        self.assertIsNone(result.verified_clean)
        self.assertTrue(result.checksum_used)
        self.assertEqual(result.change_count, 1)
        self.assertEqual(result.sampled_change_lines, (">f..t...... pyproject.toml",))
        self.assertEqual(len(commands), 1)
        self.assertIn("--dry-run", commands[0])
        self.assertIn("--checksum", commands[0])

    def test_run_continues_after_transient_failure(self) -> None:
        commands: list[list[str]] = []
        cycles = []
        errors = []
        clock = _FakeClock()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("Twinr\n", encoding="utf-8")
            responses = iter(
                (
                    _completed(["rsync"], returncode=1, stderr="ssh timeout"),
                    _completed(["rsync"], stdout=""),
                )
            )

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                return next(responses)

            watchdog = PiRepoMirrorWatchdog(
                project_root=root,
                connection_settings=PiConnectionSettings(
                    host="192.168.1.95",
                    user="thh",
                    password="chaos",
                ),
                subprocess_runner=_runner,
                sleep_fn=clock.sleep,
                monotonic_fn=clock.monotonic,
            )

            result = watchdog.run(
                interval_s=1.0,
                checksum_always=True,
                max_cycles=2,
                on_cycle=cycles.append,
                on_error=lambda exc, failure_count: errors.append((str(exc), failure_count)),
            )

        self.assertEqual(result.cycles, 1)
        self.assertEqual(result.failures, 1)
        self.assertEqual(result.syncs_applied, 0)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(errors, [("ssh timeout", 1)])
        self.assertTrue(cycles[0].checksum_used)
        self.assertEqual(len(commands), 2)
        self.assertIn("--checksum", commands[0])
        self.assertIn("--checksum", commands[1])


if __name__ == "__main__":
    unittest.main()
