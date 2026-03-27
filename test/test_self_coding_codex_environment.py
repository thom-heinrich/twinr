from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.self_coding.codex_driver.environment import (
    CodexSdkEnvironmentReport,
    assert_codex_sdk_environment_ready,
    collect_codex_sdk_environment_report,
)
from twinr.agent.self_coding.codex_driver.types import CodexDriverUnavailableError


def _completed(
    args: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


class CodexSdkEnvironmentReportTests(unittest.TestCase):
    def test_report_fails_when_required_commands_and_auth_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = collect_codex_sdk_environment_report(
                bridge_script=Path(temp_dir) / "missing_bridge.mjs",
                codex_home=Path(temp_dir) / "missing_codex_home",
                which_resolver=lambda _name: None,
                subprocess_runner=lambda *args, **kwargs: _completed(list(args[0])),
                run_local_self_test=True,
                run_live_auth_check=False,
            )

        self.assertFalse(report.ready)
        self.assertEqual(report.status, "fail")
        self.assertIn("node", " ".join(report.issues).lower())
        self.assertIn("codex", " ".join(report.issues).lower())
        self.assertIn("auth", " ".join(report.issues).lower())

    def test_report_passes_when_bridge_and_auth_are_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bridge_root = root / "sdk_bridge"
            sdk_package = bridge_root / "node_modules" / "@openai" / "codex-sdk"
            sdk_package.mkdir(parents=True)
            bridge_script = bridge_root / "run_compile.mjs"
            bridge_script.write_text("// bridge\n", encoding="utf-8")
            (bridge_root / "package.json").write_text('{"name":"bridge"}\n', encoding="utf-8")
            codex_home = root / ".codex"
            codex_home.mkdir()
            (codex_home / "auth.json").write_text('{"access_token":"token"}\n', encoding="utf-8")
            (codex_home / "config.toml").write_text("model = 'gpt-5-codex'\n", encoding="utf-8")

            def _which(name: str) -> str | None:
                return {
                    "node": "/usr/bin/node",
                    "npm": "/usr/bin/npm",
                    "codex": "/usr/bin/codex",
                }.get(name)

            def _runner(args, **kwargs):
                command = list(args)
                if command[-1] == "--self-test":
                    return _completed(
                        command,
                        stdout='{"ok":true,"nodeVersion":"v18.20.4","codexPath":"/usr/bin/codex","codexVersion":"codex-cli 0.114.0"}\n',
                    )
                if command[:2] == ["/usr/bin/node", "--version"]:
                    return _completed(command, stdout="v18.20.4\n")
                if command[:2] == ["/usr/bin/npm", "--version"]:
                    return _completed(command, stdout="9.2.0\n")
                if command[:2] == ["/usr/bin/codex", "--version"]:
                    return _completed(command, stdout="codex-cli 0.114.0\n")
                raise AssertionError(f"unexpected command: {command}")

            report = collect_codex_sdk_environment_report(
                bridge_script=bridge_script,
                codex_home=codex_home,
                which_resolver=_which,
                subprocess_runner=_runner,
                run_local_self_test=True,
                run_live_auth_check=False,
            )

        self.assertTrue(report.ready)
        self.assertEqual(report.status, "ok")
        self.assertTrue(report.local_self_test_ok)
        self.assertEqual(report.node_version, "v18.20.4")
        self.assertEqual(report.codex_version, "codex-cli 0.114.0")

    def test_report_surfaces_live_auth_probe_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bridge_root = root / "sdk_bridge"
            sdk_package = bridge_root / "node_modules" / "@openai" / "codex-sdk"
            sdk_package.mkdir(parents=True)
            bridge_script = bridge_root / "run_compile.mjs"
            bridge_script.write_text("// bridge\n", encoding="utf-8")
            (bridge_root / "package.json").write_text('{"name":"bridge"}\n', encoding="utf-8")
            codex_home = root / ".codex"
            codex_home.mkdir()
            (codex_home / "auth.json").write_text('{"access_token":"token"}\n', encoding="utf-8")
            (codex_home / "config.toml").write_text("model = 'gpt-5-codex'\n", encoding="utf-8")

            def _which(name: str) -> str | None:
                return {
                    "node": "/usr/bin/node",
                    "npm": "/usr/bin/npm",
                    "codex": "/usr/bin/codex",
                }.get(name)

            def _runner(args, **kwargs):
                command = list(args)
                if command[-1] == "--self-test":
                    return _completed(
                        command,
                        stdout='{"ok":true,"nodeVersion":"v18.20.4","codexPath":"/usr/bin/codex","codexVersion":"codex-cli 0.114.0"}\n',
                    )
                if command[:2] == ["/usr/bin/node", "--version"]:
                    return _completed(command, stdout="v18.20.4\n")
                if command[:2] == ["/usr/bin/npm", "--version"]:
                    return _completed(command, stdout="9.2.0\n")
                if command[:2] == ["/usr/bin/codex", "--version"]:
                    return _completed(command, stdout="codex-cli 0.114.0\n")
                if command[:4] == ["/usr/bin/codex", "exec", "--json", "--skip-git-repo-check"]:
                    return _completed(
                        command,
                        returncode=1,
                        stdout="",
                        stderr="unexpected status 401 Unauthorized: Missing Bearer token\n",
                    )
                raise AssertionError(f"unexpected command: {command}")

            report = collect_codex_sdk_environment_report(
                bridge_script=bridge_script,
                codex_home=codex_home,
                which_resolver=_which,
                subprocess_runner=_runner,
                run_local_self_test=True,
                run_live_auth_check=True,
            )

        self.assertFalse(report.ready)
        self.assertEqual(report.status, "fail")
        self.assertFalse(report.live_auth_check_ok)
        self.assertIn("401", report.detail)

    def test_live_auth_probe_skips_git_repo_check(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bridge_root = root / "sdk_bridge"
            sdk_package = bridge_root / "node_modules" / "@openai" / "codex-sdk"
            sdk_package.mkdir(parents=True)
            bridge_script = bridge_root / "run_compile.mjs"
            bridge_script.write_text("// bridge\n", encoding="utf-8")
            (bridge_root / "package.json").write_text('{"name":"bridge"}\n', encoding="utf-8")
            codex_home = root / ".codex"
            codex_home.mkdir()
            (codex_home / "auth.json").write_text('{"access_token":"token"}\n', encoding="utf-8")
            (codex_home / "config.toml").write_text("model = 'gpt-5-codex'\n", encoding="utf-8")
            seen_commands: list[list[str]] = []

            def _which(name: str) -> str | None:
                return {
                    "node": "/usr/bin/node",
                    "npm": "/usr/bin/npm",
                    "codex": "/usr/bin/codex",
                }.get(name)

            def _runner(args, **kwargs):
                command = list(args)
                seen_commands.append(command)
                if command[-1] == "--self-test":
                    return _completed(
                        command,
                        stdout='{"ok":true,"nodeVersion":"v18.20.4","codexPath":"/usr/bin/codex","codexVersion":"codex-cli 0.114.0"}\n',
                    )
                if command[:2] == ["/usr/bin/node", "--version"]:
                    return _completed(command, stdout="v18.20.4\n")
                if command[:2] == ["/usr/bin/npm", "--version"]:
                    return _completed(command, stdout="9.2.0\n")
                if command[:2] == ["/usr/bin/codex", "--version"]:
                    return _completed(command, stdout="codex-cli 0.114.0\n")
                if command[:4] == ["/usr/bin/codex", "exec", "--json", "--skip-git-repo-check"]:
                    return _completed(command, stdout='{"content":"READY"}\n')
                raise AssertionError(f"unexpected command: {command}")

            report = collect_codex_sdk_environment_report(
                bridge_script=bridge_script,
                codex_home=codex_home,
                which_resolver=_which,
                subprocess_runner=_runner,
                run_local_self_test=True,
                run_live_auth_check=True,
            )

        self.assertTrue(report.ready)
        self.assertTrue(report.live_auth_check_ok)
        self.assertIn(
            ["/usr/bin/codex", "exec", "--json", "--skip-git-repo-check"],
            [command[:4] for command in seen_commands if len(command) >= 4],
        )

    def test_assert_ready_raises_driver_error_for_failed_report(self) -> None:
        report = CodexSdkEnvironmentReport(
            status="fail",
            ready=False,
            detail="codex auth is missing",
            issues=("codex auth is missing",),
        )

        with self.assertRaisesRegex(CodexDriverUnavailableError, "codex auth is missing"):
            assert_codex_sdk_environment_ready(report)


if __name__ == "__main__":
    unittest.main()
