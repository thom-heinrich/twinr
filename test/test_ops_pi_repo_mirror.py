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

from twinr.ops.pi_repo_mirror import (
    PiRepoMirrorCapabilities,
    PiRepoMirrorWatchdog,
    build_authoritative_repo_entry_digests,
    materialize_authoritative_repo_snapshot,
)
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
    @staticmethod
    def _seed_capabilities(watchdog: PiRepoMirrorWatchdog) -> None:
        watchdog._capabilities = PiRepoMirrorCapabilities(
            local_rsync_version="3.2.7",
            remote_rsync_version="3.2.7",
            supports_delete_delay=False,
            supports_delay_updates=False,
            supports_mkpath=False,
            supports_checksum_choice=False,
            supports_compress_choice=False,
            supports_xxh128=False,
            supports_zstd=False,
            supports_fsync=False,
        )

    def test_build_authoritative_repo_entry_digests_uses_mirror_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("Twinr\n", encoding="utf-8")
            (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
            (root / "state").mkdir()
            (root / "state" / "runtime-state.json").write_text("{}", encoding="utf-8")
            (root / "pkg.egg-info").mkdir()
            (root / "pkg.egg-info" / "PKG-INFO").write_text("ignored\n", encoding="utf-8")
            (root / "browser_automation" / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "browser_automation" / "artifacts" / "trace.json").write_text(
                "{}",
                encoding="utf-8",
            )
            (root / "node_modules").mkdir()
            (root / "node_modules" / "index.js").write_text("console.log('x')\n", encoding="utf-8")
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "link-to-app").symlink_to("src/app.py")

            entries = build_authoritative_repo_entry_digests(root)

        entries_by_path = {entry.relative_path: entry for entry in entries}
        self.assertIn("README.md", entries_by_path)
        self.assertIn("src/app.py", entries_by_path)
        self.assertIn("link-to-app", entries_by_path)
        self.assertEqual(entries_by_path["link-to-app"].kind, "symlink")
        self.assertEqual(entries_by_path["link-to-app"].link_target, "src/app.py")
        self.assertNotIn(".env", entries_by_path)
        self.assertNotIn("state/runtime-state.json", entries_by_path)
        self.assertNotIn("pkg.egg-info/PKG-INFO", entries_by_path)
        self.assertNotIn("browser_automation/artifacts/trace.json", entries_by_path)
        self.assertNotIn("node_modules/index.js", entries_by_path)

    def test_materialize_authoritative_repo_snapshot_copies_only_mirror_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            snapshot = Path(temp_dir) / "snapshot"
            root.mkdir(parents=True, exist_ok=True)
            (root / "README.md").write_text("Twinr\n", encoding="utf-8")
            (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
            (root / "state").mkdir()
            (root / "state" / "runtime-state.json").write_text("{}", encoding="utf-8")
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "link-to-app").symlink_to("src/app.py")

            entries = materialize_authoritative_repo_snapshot(root, snapshot)

            snapshot_entries = build_authoritative_repo_entry_digests(snapshot)
            self.assertEqual(entries, snapshot_entries)
            self.assertTrue((snapshot / "README.md").is_file())
            self.assertTrue((snapshot / "src" / "app.py").is_file())
            self.assertTrue((snapshot / "link-to-app").is_symlink())
            self.assertEqual(os.readlink(snapshot / "link-to-app"), "src/app.py")
            self.assertFalse((snapshot / ".env").exists())
            self.assertFalse((snapshot / "state" / "runtime-state.json").exists())

    def test_probe_once_reports_clean_repo_when_rsync_finds_no_changes(self) -> None:
        commands: list[list[str]] = []
        envs: list[dict[str, str] | None] = []
        inputs: list[str | None] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("Twinr\n", encoding="utf-8")

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                envs.append(kwargs.get("env"))
                inputs.append(kwargs.get("input"))
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
            self._seed_capabilities(watchdog)

            result = watchdog.probe_once()

        self.assertFalse(result.drift_detected)
        self.assertFalse(result.sync_applied)
        self.assertTrue(result.verified_clean)
        self.assertTrue(result.checksum_used)
        self.assertEqual(result.change_count, 0)
        self.assertEqual(len(commands), 1)
        joined = " ".join(commands[0])
        self.assertIn("sshpass -d 0 rsync", joined)
        self.assertIn("--checksum", joined)
        self.assertIn("--delete", joined)
        self.assertIn("--itemize-changes", joined)
        self.assertIn("--no-specials", joined)
        self.assertIn("--no-devices", joined)
        self.assertIn("--exclude=**/__pycache__/", commands[0])
        self.assertIn("--exclude=**/*.pyc", commands[0])
        self.assertIn("--exclude=**/*.pyo", commands[0])
        self.assertIn("--exclude=**/*.egg-info/", commands[0])
        self.assertIn("--exclude=**/node_modules/", commands[0])
        self.assertIn("--exclude=**/browser_automation/artifacts/", commands[0])
        self.assertIn("--exclude=/hardware/bitcraze/twinr_on_device_failsafe/build/", commands[0])
        self.assertIn("--filter=-p /.cache/", joined)
        self.assertIn("--filter=-p /.env", joined)
        self.assertNotIn("--dry-run", commands[0])
        env = envs[0]
        self.assertFalse(env and "SSHPASS" in env)
        self.assertEqual(inputs[0], _TEST_PI_SSH_PASSWORD + "\n")

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
            self._seed_capabilities(watchdog)

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
            self._seed_capabilities(watchdog)

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
            self._seed_capabilities(watchdog)

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
            self._seed_capabilities(watchdog)

            result = watchdog.probe_once()

        self.assertTrue(result.drift_detected)
        self.assertTrue(result.sync_applied)
        self.assertTrue(result.verified_clean)
        self.assertEqual(len(commands), 5)
        self.assertIn("sshpass", commands[2][0])
        self.assertIn("ssh", commands[2])
        rendered_prune = " ".join(commands[2])
        self.assertIn("common/links", rendered_prune)
        self.assertIn("__pycache__", rendered_prune)
        self.assertIn("sudo rm -rf", rendered_prune)
        self.assertIn("--dry-run", commands[1])
        self.assertNotIn("--dry-run", commands[3])
        self.assertIn("--dry-run", commands[4])

    def test_probe_once_retries_sync_when_post_sync_verification_still_sees_drift(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("Twinr\n", encoding="utf-8")
            responses = iter(
                (
                    _completed(["rsync"], stdout="<fcst...... src/app.py\n"),
                    _completed(["rsync"], stdout="<fcst...... src/app.py\n"),
                    _completed(["rsync"], stdout="<fcst...... src/app.py\n"),
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
            self._seed_capabilities(watchdog)

            result = watchdog.probe_once()

        self.assertTrue(result.drift_detected)
        self.assertTrue(result.sync_applied)
        self.assertTrue(result.verified_clean)
        self.assertEqual(result.change_count, 1)
        self.assertEqual(result.sampled_change_lines, ("<fcst...... src/app.py",))
        self.assertEqual(len(commands), 4)
        self.assertNotIn("--dry-run", commands[0])
        self.assertIn("--dry-run", commands[1])
        self.assertNotIn("--dry-run", commands[2])
        self.assertIn("--dry-run", commands[3])

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
            self._seed_capabilities(watchdog)

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
        self.assertFalse(cycles[0].checksum_used)
        self.assertEqual(len(commands), 2)
        self.assertIn("--checksum", commands[0])
        self.assertNotIn("--checksum", commands[1])

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
