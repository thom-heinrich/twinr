from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.ops.pi_repo_mirror import PiRepoMirrorWatchdog
from twinr.ops.self_coding_pi import PiConnectionSettings


_TEST_PI_HOST = "192.0.2.10"
_TEST_PI_SSH_USER = "pi-test-user"
_TEST_PI_SSH_PASSWORD = "placeholder-password"


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
                    host=_TEST_PI_HOST,
                    user=_TEST_PI_SSH_USER,
                    password=_TEST_PI_SSH_PASSWORD,
                ),
                subprocess_runner=_runner,
            )

            result = watchdog.probe_once()

        self.assertFalse(result.drift_detected)
        self.assertFalse(result.sync_applied)
        self.assertTrue(result.verified_clean)
        self.assertTrue(result.checksum_used)
        self.assertEqual(result.change_count, 0)
        self.assertEqual(len(commands), 1)
        joined = " ".join(commands[0])
        self.assertIn("sshpass -e rsync", joined)
        self.assertIn("--checksum", joined)
        self.assertIn("--delete", joined)
        self.assertIn("--itemize-changes", joined)
        self.assertIn("--no-specials", joined)
        self.assertIn("--no-devices", joined)
        self.assertIn("--exclude=**/__pycache__/", commands[0])
        self.assertIn("--exclude=**/*.pyc", commands[0])
        self.assertIn("--exclude=**/*.pyo", commands[0])
        self.assertIn("--exclude=**/node_modules/", commands[0])
        self.assertIn("--exclude=**/browser_automation/artifacts/", commands[0])
        self.assertIn("--filter=-p /.env", joined)
        self.assertNotIn("--dry-run", commands[0])
        env = envs[0]
        self.assertIsNotNone(env)
        assert env is not None
        self.assertEqual(env["SSHPASS"], _TEST_PI_SSH_PASSWORD)

    def test_probe_once_excludes_local_special_files_from_rsync_args(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            os.mkfifo(root / ".lgd-nfy0")
            (root / "README.md").write_text("Twinr\n", encoding="utf-8")

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                return _completed(command, stdout="")

            watchdog = PiRepoMirrorWatchdog(
                project_root=root,
                connection_settings=PiConnectionSettings(
                    host=_TEST_PI_HOST,
                    user=_TEST_PI_SSH_USER,
                    password=_TEST_PI_SSH_PASSWORD,
                ),
                subprocess_runner=_runner,
            )

            watchdog.probe_once()

        self.assertEqual(len(commands), 1)
        self.assertIn("--exclude=/.lgd-nfy0", commands[0])

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
                    host=_TEST_PI_HOST,
                    user=_TEST_PI_SSH_USER,
                    password=_TEST_PI_SSH_PASSWORD,
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
        self.assertIn("--checksum", commands[0])
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
                    host=_TEST_PI_HOST,
                    user=_TEST_PI_SSH_USER,
                    password=_TEST_PI_SSH_PASSWORD,
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

    def test_probe_once_prunes_cache_only_stale_dirs_and_retries_sync(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("Twinr\n", encoding="utf-8")
            responses = iter(
                (
                    _completed(["rsync"], stdout=">f..t...... README.md\n"),
                    _completed(["rsync"], stdout="*deleting   common/links/\n"),
                    _completed(["ssh"], stdout=""),
                    _completed(["rsync"], stdout="*deleting   common/links/\n"),
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
                    host=_TEST_PI_HOST,
                    user=_TEST_PI_SSH_USER,
                    password=_TEST_PI_SSH_PASSWORD,
                ),
                subprocess_runner=_runner,
            )

            result = watchdog.probe_once()

        self.assertTrue(result.drift_detected)
        self.assertTrue(result.sync_applied)
        self.assertTrue(result.verified_clean)
        self.assertEqual(len(commands), 5)
        self.assertIn("sshpass", commands[2][0])
        self.assertIn("ssh", commands[2][2])
        rendered_prune = " ".join(commands[2])
        self.assertIn("common/links", rendered_prune)
        self.assertIn("__pycache__", rendered_prune)
        self.assertIn("rm -rf", rendered_prune)
        self.assertIn("--dry-run", commands[1])
        self.assertNotIn("--dry-run", commands[3])
        self.assertIn("--dry-run", commands[4])

    def test_run_continues_after_transient_failure(self) -> None:
        commands: list[list[str]] = []
        cycles: list[Any] = []
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
                    host=_TEST_PI_HOST,
                    user=_TEST_PI_SSH_USER,
                    password=_TEST_PI_SSH_PASSWORD,
                ),
                subprocess_runner=_runner,
                sleep_fn=clock.sleep,
                monotonic_fn=clock.monotonic,
            )

            result = watchdog.run(
                interval_s=1.0,
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

    @unittest.skipUnless(shutil.which("rsync"), "rsync required")
    def test_perishable_filters_preserve_root_env_but_allow_nested_tree_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as dest_dir:
            source = Path(source_dir)
            destination = Path(dest_dir)
            (destination / ".env").write_text("PI_ONLY=1\n", encoding="utf-8")
            nested_root = destination / "home" / "thh" / "twinr"
            nested_root.mkdir(parents=True)
            (nested_root / ".env").write_text("STALE=1\n", encoding="utf-8")
            subprocess.run(
                [
                    "rsync",
                    "-ai",
                    "--delete",
                    "--filter=-p .env",
                    f"{source}/",
                    f"{destination}/",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue((destination / ".env").exists())
            self.assertFalse((destination / "home").exists())

    @unittest.skipUnless(shutil.which("rsync"), "rsync required")
    def test_rsync_contract_excludes_transient_special_files_from_repo_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as dest_dir:
            source = Path(source_dir)
            destination = Path(dest_dir)
            fifo_path = source / ".lgd-nfy0"
            (source / "README.md").write_text("Twinr\n", encoding="utf-8")
            fifo_path.unlink(missing_ok=True)
            fifo_path.parent.mkdir(parents=True, exist_ok=True)
            fifo_path.touch(exist_ok=True)
            fifo_path.unlink()
            fifo_path.parent.mkdir(parents=True, exist_ok=True)
            import os

            os.mkfifo(fifo_path)
            try:
                subprocess.run(
                    [
                        "rsync",
                        "-ai",
                        "--delete",
                        "--no-specials",
                        "--no-devices",
                        f"{source}/",
                        f"{destination}/",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            finally:
                fifo_path.unlink(missing_ok=True)

            self.assertTrue((destination / "README.md").exists())
            self.assertFalse((destination / ".lgd-nfy0").exists())


if __name__ == "__main__":
    unittest.main()
