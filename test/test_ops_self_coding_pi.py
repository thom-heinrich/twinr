from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.ops.self_coding_pi import (
    PiConnectionSettings,
    bootstrap_self_coding_pi,
    load_pi_connection_settings,
)


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


class SelfCodingPiBootstrapTests(unittest.TestCase):
    def test_load_pi_connection_settings_reads_dotenv_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env.pi"
            env_path.write_text(
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

            settings = load_pi_connection_settings(env_path)

        self.assertEqual(
            settings,
            PiConnectionSettings(
                host=_TEST_PI_HOST,
                user=_TEST_PI_SSH_USER,
                password=_TEST_PI_SSH_PASSWORD,
            ),
        )

    def test_bootstrap_runs_remote_install_sync_and_self_test_steps(self) -> None:
        commands: list[tuple[list[str], dict[str, str] | None]] = []
        env_overrides: list[dict[str, str] | None] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            codex_home = root / ".codex"
            codex_home.mkdir()
            (codex_home / "auth.json").write_text('{"access_token":"token"}\n', encoding="utf-8")
            (codex_home / "config.toml").write_text("model = 'gpt-5-codex'\n", encoding="utf-8")
            bridge_root = root / "src" / "twinr" / "agent" / "self_coding" / "codex_driver" / "sdk_bridge"
            bridge_root.mkdir(parents=True)
            (bridge_root / "package.json").write_text('{"name":"bridge"}\n', encoding="utf-8")
            (bridge_root / "package-lock.json").write_text('{"name":"bridge","lockfileVersion":3}\n', encoding="utf-8")

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

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                env_overrides.append(kwargs.get("env"))
                if command[:2] == ["codex", "--version"]:
                    return _completed(command, stdout="codex-cli 0.114.0\n")
                return _completed(command, stdout="ok\n")

            result = bootstrap_self_coding_pi(
                project_root=root,
                pi_env_path=pi_env_path,
                local_codex_home=codex_home,
                subprocess_runner=_runner,
            )

        self.assertTrue(result.ready)
        self.assertEqual(result.codex_cli_version, "0.114.0")
        joined = "\n".join(" ".join(command) for command in commands)
        self.assertIn("sshpass -d", joined)
        self.assertIn("sshpass -d 11 ssh", joined)
        self.assertIn("sshpass -d 11 rsync", joined)
        self.assertIn("npm install --global --prefix", joined)
        self.assertIn("@openai/codex@$CODEX_VERSION", joined)
        self.assertIn("--self-coding-codex-self-test", joined)
        self.assertFalse(any(env and "SSHPASS" in env for env in env_overrides))


if __name__ == "__main__":
    unittest.main()
