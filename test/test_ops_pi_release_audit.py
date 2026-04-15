from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.ops.pi_release_audit import audit_pi_release
from twinr.ops.pi_repo_mirror import PiRepoMirrorCycleResult, build_authoritative_release_manifest

_TEST_PI_HOST = "192.0.2.10"
_TEST_PI_SSH_USER = "pi-test-user"
_TEST_PI_SSH_PASSWORD = "placeholder-password"


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Twinr Tests"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.com"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True)


class _FakeMirrorWatchdog:
    def __init__(self, result: PiRepoMirrorCycleResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def probe_once(
        self,
        *,
        apply_sync: bool = True,
        checksum: bool = True,
        max_change_lines: int = 40,
    ) -> PiRepoMirrorCycleResult:
        self.calls.append(
            {
                "apply_sync": apply_sync,
                "checksum": checksum,
                "max_change_lines": max_change_lines,
            }
        )
        return self.result


class _FakeRemoteExecutor:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.calls: list[str] = []

    def run_ssh(self, script: str, *, timeout_s: float | None = None):
        del timeout_s
        self.calls.append(script)
        return subprocess.CompletedProcess(
            args=["ssh"],
            returncode=0,
            stdout=self.stdout,
            stderr="",
        )


class PiReleaseAuditTests(unittest.TestCase):
    def test_audit_reports_in_sync_when_manifest_and_drift_probe_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("Twinr\n", encoding="utf-8")
            _init_git_repo(root)
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            local_manifest = build_authoritative_release_manifest(root, generated_at_utc="2026-04-09T08:30:00Z")
            remote = _FakeRemoteExecutor(json.dumps(asdict(local_manifest)) + "\n")
            mirror = _FakeMirrorWatchdog(
                PiRepoMirrorCycleResult(
                    host=_TEST_PI_HOST,
                    remote_root="/twinr",
                    drift_detected=False,
                    sync_applied=False,
                    checksum_used=True,
                    verified_clean=None,
                    change_count=0,
                    sampled_change_lines=(),
                    duration_s=0.25,
                )
            )

            result = audit_pi_release(
                project_root=root,
                pi_env_path=pi_env_path,
                remote_root="/twinr",
                mirror_watchdog=mirror,
                remote_executor=cast(Any, remote),
            )

        self.assertTrue(result.in_sync)
        self.assertEqual(result.status, "in_sync")
        self.assertTrue(result.release_id_match)
        self.assertTrue(result.source_commit_match)
        self.assertTrue(result.source_dirty_match)
        self.assertTrue(result.remote_manifest_present)
        self.assertIsNotNone(result.remote_release_manifest)
        self.assertEqual(mirror.calls, [{"apply_sync": False, "checksum": True, "max_change_lines": 40}])
        self.assertTrue(any("current_release_manifest.json" in script for script in remote.calls))

    def test_audit_reports_missing_remote_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("Twinr\n", encoding="utf-8")
            _init_git_repo(root)
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            remote = _FakeRemoteExecutor("{}\n")
            mirror = _FakeMirrorWatchdog(
                PiRepoMirrorCycleResult(
                    host=_TEST_PI_HOST,
                    remote_root="/twinr",
                    drift_detected=False,
                    sync_applied=False,
                    checksum_used=True,
                    verified_clean=None,
                    change_count=0,
                    sampled_change_lines=(),
                    duration_s=0.19,
                )
            )

            result = audit_pi_release(
                project_root=root,
                pi_env_path=pi_env_path,
                remote_root="/twinr",
                mirror_watchdog=mirror,
                remote_executor=cast(Any, remote),
            )

        self.assertFalse(result.in_sync)
        self.assertEqual(result.status, "remote_manifest_missing")
        self.assertFalse(result.remote_manifest_present)
        self.assertIsNone(result.remote_release_manifest)
        self.assertFalse(result.release_id_match)

    def test_audit_reports_drift_even_when_remote_manifest_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("Twinr\n", encoding="utf-8")
            _init_git_repo(root)
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            local_manifest = build_authoritative_release_manifest(root, generated_at_utc="2026-04-09T08:31:00Z")
            remote = _FakeRemoteExecutor(json.dumps(asdict(local_manifest)) + "\n")
            mirror = _FakeMirrorWatchdog(
                PiRepoMirrorCycleResult(
                    host=_TEST_PI_HOST,
                    remote_root="/twinr",
                    drift_detected=True,
                    sync_applied=False,
                    checksum_used=True,
                    verified_clean=None,
                    change_count=1,
                    sampled_change_lines=(">f+++++++++ README.md",),
                    duration_s=0.33,
                )
            )

            result = audit_pi_release(
                project_root=root,
                pi_env_path=pi_env_path,
                remote_root="/twinr",
                mirror_watchdog=mirror,
                remote_executor=cast(Any, remote),
            )

        self.assertFalse(result.in_sync)
        self.assertEqual(result.status, "drift_detected")
        self.assertTrue(result.release_id_match)
        self.assertTrue(result.remote_manifest_present)
        self.assertEqual(result.drift_probe.change_count, 1)


if __name__ == "__main__":
    unittest.main()
