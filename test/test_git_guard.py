"""Targeted regression tests for the repo-local git guard support scripts."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import json
import subprocess
import sys
import tempfile
from typing import Any, cast
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from git_guard_tool.cli import main as git_guard_main


_POLICY_DATA = {
    "guard": {"max_issues": 50},
    "paths": {
        "ignore_path_prefixes": [".git/", ".venv/", "artifacts/", "state/"],
        "blocked_exact_names": [
            ".env",
            ".env.pi",
            ".env.chonkydb",
            ".env.twinr-proxy",
            ".voice_gateway_alias.env",
        ],
        "blocked_suffixes": [".pem", ".key", ".p12", ".pfx", ".kdbx", ".mobileprovision"],
    },
    "content": {
        "blocked_terms": ["chaos", "warhammer"],
        "secret_prefixes": ["sk-proj-", "sk-", "ghp_", "github_pat_", "glpat-", "xoxb-", "xoxp-", "xapp-", "AIza"],
        "sensitive_key_fragments": [
            "api_key",
            "apikey",
            "secret",
            "token",
            "password",
            "passwd",
            "pwd",
            "private_key",
            "access_key",
            "session_key",
            "auth_key",
        ],
        "placeholder_values": [
            "",
            "example",
            "example-value",
            "placeholder",
            "placeholder-value",
            "dummy",
            "test",
            "fake",
            "secret-key",
            "token",
            "redacted",
            "<redacted>",
            "your-value-here",
            "changeme",
            "replace-me",
            "...",
        ],
        "secret_min_length": 12,
    },
    "phones": {"min_digits": 7, "max_digits": 15},
}


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )


class GitGuardCliTests(unittest.TestCase):
    def _init_repo(self, repo_root: Path) -> None:
        _run_git(repo_root, "init", "-q")
        _run_git(repo_root, "config", "user.name", "Twinr Test")
        _run_git(repo_root, "config", "user.email", "test@example.com")
        _run_git(repo_root, "branch", "-M", "main")
        (repo_root / ".git_guard.json").write_text(json.dumps(_POLICY_DATA), encoding="utf-8")

    def _commit_file(self, repo_root: Path, relative_path: str, content: str, message: str) -> None:
        target = repo_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        _run_git(repo_root, "add", relative_path)
        _run_git(repo_root, "commit", "-q", "-m", message)

    def _commit_bytes(self, repo_root: Path, relative_path: str, content: bytes, message: str) -> None:
        target = repo_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        _run_git(repo_root, "add", relative_path)
        _run_git(repo_root, "commit", "-q", "-m", message)

    def _run_cli(self, repo_root: Path, *args: str) -> tuple[int, dict[str, Any], str]:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = git_guard_main(["--repo-root", str(repo_root), "--json", *args])
        payload = cast(dict[str, Any], json.loads(stdout.getvalue()))
        return exit_code, payload, stderr.getvalue()

    def test_install_sets_core_hooks_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._init_repo(repo_root)
            hooks_dir = repo_root / "scripts" / "git_hooks"
            hooks_dir.mkdir(parents=True, exist_ok=True)
            (hooks_dir / "pre-commit").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (hooks_dir / "pre-push").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

            exit_code, payload, _ = self._run_cli(repo_root, "install")

            configured = _run_git(repo_root, "config", "--local", "--get", "core.hooksPath").stdout.strip()
            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(configured, "scripts/git_hooks")

    def test_scan_staged_blocks_terms_secret_values_and_phone_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._init_repo(repo_root)
            self._commit_file(repo_root, "notes.txt", "baseline\n", "base")
            (repo_root / "notes.txt").write_text(
                "baseline\nchaos should never ship\nOPENAI_API_KEY=live-real-secret-value\nCall me at +49 30 12345678\n",
                encoding="utf-8",
            )
            _run_git(repo_root, "add", "notes.txt")

            exit_code, payload, _ = self._run_cli(repo_root, "scan-staged")

            issues = cast(list[dict[str, Any]], payload["issues"])
            rule_ids = {issue["rule_id"] for issue in issues}
            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["ok"])
            self.assertIn("blocked-term-content", rule_ids)
            self.assertIn("sensitive-assignment", rule_ids)
            self.assertIn("phone-number", rule_ids)

    def test_scan_staged_blocks_env_files_by_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._init_repo(repo_root)
            (repo_root / ".env").write_text("OPENAI_API_KEY=placeholder\n", encoding="utf-8")
            _run_git(repo_root, "add", ".env")

            exit_code, payload, _ = self._run_cli(repo_root, "scan-staged")

            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["ok"])
            issues = cast(list[dict[str, Any]], payload["issues"])
            self.assertEqual(issues[0]["rule_id"], "blocked-path-name")

    def test_scan_staged_allows_code_metadata_for_sensitive_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._init_repo(repo_root)
            (repo_root / "module.py").write_text(
                (
                    "secret_label: str = \"Mailbox password\"\n"
                    "auth_password_value = str(env.get(\"TWINR_WEB_PASSWORD\", \"\"))\n"
                    "password = secret_value\n"
                    "async def auth_password(request):\n"
                    "    return request\n"
                ),
                encoding="utf-8",
            )
            _run_git(repo_root, "add", "module.py")

            exit_code, payload, _ = self._run_cli(repo_root, "scan-staged")

            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["issues"], [])

    def test_scan_staged_allows_sensitive_name_references_and_non_secret_literals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._init_repo(repo_root)
            (repo_root / "module.py").write_text(
                (
                    'ssl_keyfile_password=args.ssl_keyfile_password\n'
                    '_SECRET_NAME_FRAGMENT = r"(?:api[_-]?key|token|password)"\n'
                    '_PROCESS_TOKEN = f"{pid}:{clock}"\n'
                    'token = "".join(normalized).strip("_")\n'
                    'target_tokens = target.tokens\n'
                    'token_usage=turn.token_usage\n'
                    'api_key_header=config.chonkydb_api_key_header\n'
                    'password_path = "/auth/password"\n'
                ),
                encoding="utf-8",
            )
            _run_git(repo_root, "add", "module.py")

            exit_code, payload, _ = self._run_cli(repo_root, "scan-staged")

            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["issues"], [])

    def test_scan_staged_allows_custom_header_name_metadata_literals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._init_repo(repo_root)
            (repo_root / "module.py").write_text(
                (
                    'shared_secret_header_name = "x-twinr-secret"\n'
                    'api_key_header_name = "x-api-key"\n'
                ),
                encoding="utf-8",
            )
            _run_git(repo_root, "add", "module.py")

            exit_code, payload, _ = self._run_cli(repo_root, "scan-staged")

            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["issues"], [])

    def test_scan_staged_still_blocks_non_header_literals_for_header_name_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._init_repo(repo_root)
            (repo_root / "module.py").write_text(
                'shared_secret_header_name = "live-real-secret-value"\n',
                encoding="utf-8",
            )
            _run_git(repo_root, "add", "module.py")

            exit_code, payload, _ = self._run_cli(repo_root, "scan-staged")

            issues = cast(list[dict[str, Any]], payload["issues"])
            rule_ids = {issue["rule_id"] for issue in issues}
            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["ok"])
            self.assertIn("sensitive-assignment", rule_ids)

    def test_scan_staged_allows_prefix_metadata_literals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._init_repo(repo_root)
            (repo_root / "module.py").write_text(
                (
                    '_CURSOR_TOKEN_PREFIX = "smarthome-route-cursor:"\n'
                    'EVENT_NAMESPACE = "route-event:"\n'
                ),
                encoding="utf-8",
            )
            _run_git(repo_root, "add", "module.py")

            exit_code, payload, _ = self._run_cli(repo_root, "scan-staged")

            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["issues"], [])

    def test_scan_staged_still_blocks_non_prefix_literals_for_prefix_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._init_repo(repo_root)
            (repo_root / "module.py").write_text(
                'token_prefix = "live-real-secret-value"\n',
                encoding="utf-8",
            )
            _run_git(repo_root, "add", "module.py")

            exit_code, payload, _ = self._run_cli(repo_root, "scan-staged")

            issues = cast(list[dict[str, Any]], payload["issues"])
            rule_ids = {issue["rule_id"] for issue in issues}
            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["ok"])
            self.assertIn("sensitive-assignment", rule_ids)

    def test_scan_staged_allows_blocked_term_policy_definitions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._init_repo(repo_root)
            (repo_root / "policy.py").write_text(
                'blocked_terms=("chaos", "warhammer")\n',
                encoding="utf-8",
            )
            _run_git(repo_root, "add", "policy.py")

            exit_code, payload, _ = self._run_cli(repo_root, "scan-staged")

            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["issues"], [])

    def test_scan_staged_ignores_weak_heuristics_in_test_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._init_repo(repo_root)
            test_file = repo_root / "worker.test.mjs"
            test_file.write_text(
                (
                    'const blocked = "warhammer sneaked in here";\n'
                    'const password = "Password.123";\n'
                    "const ts = 1774539523;\n"
                ),
                encoding="utf-8",
            )
            _run_git(repo_root, "add", "worker.test.mjs")

            exit_code, payload, _ = self._run_cli(repo_root, "scan-staged")

            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["issues"], [])

    def test_scan_staged_allows_placeholder_secret_prefixes_in_test_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._init_repo(repo_root)
            (repo_root / "test" / "test_openai_backend.py").parent.mkdir(parents=True, exist_ok=True)
            (repo_root / "test" / "test_openai_backend.py").write_text(
                'openai_api_key = "sk-proj-example"\n',
                encoding="utf-8",
            )
            _run_git(repo_root, "add", "test/test_openai_backend.py")

            exit_code, payload, _ = self._run_cli(repo_root, "scan-staged")

            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["issues"], [])

    def test_scan_staged_still_blocks_real_secret_prefixes_in_test_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._init_repo(repo_root)
            secret_token = "sk-" + "live-real-secret-value"
            (repo_root / "test" / "test_openai_backend.py").parent.mkdir(parents=True, exist_ok=True)
            (repo_root / "test" / "test_openai_backend.py").write_text(
                f'openai_api_key = "{secret_token}"\n',
                encoding="utf-8",
            )
            _run_git(repo_root, "add", "test/test_openai_backend.py")

            exit_code, payload, _ = self._run_cli(repo_root, "scan-staged")

            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["ok"])
            issues = cast(list[dict[str, Any]], payload["issues"])
            self.assertEqual(issues[0]["rule_id"], "secret-prefix")

    def test_scan_staged_ignores_numeric_ranges_that_are_not_phone_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._init_repo(repo_root)
            (repo_root / "notes.md").write_text(
                "servo range is 500-2500 us with 1500 us center\n",
                encoding="utf-8",
            )
            _run_git(repo_root, "add", "notes.md")

            exit_code, payload, _ = self._run_cli(repo_root, "scan-staged")

            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["issues"], [])

    def test_scan_staged_ignores_dates_math_and_resolution_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._init_repo(repo_root)
            (repo_root / "notes.md").write_text(
                (
                    "schema version is 2026-03-27.1\n"
                    "timeout uses max(100, ceil(value * 100)) / 1000.0\n"
                    "center pulse is clamp(2500) + 1500\n"
                    "supported sizes: 420/640/720\n"
                    "normalized gain is 2.000000 and floor is 0.500000\n"
                ),
                encoding="utf-8",
            )
            _run_git(repo_root, "add", "notes.md")

            exit_code, payload, _ = self._run_cli(repo_root, "scan-staged")

            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["issues"], [])

    def test_scan_staged_still_blocks_real_phone_number_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._init_repo(repo_root)
            (repo_root / "notes.md").write_text(
                "Call me at +49 30 12345678\n",
                encoding="utf-8",
            )
            _run_git(repo_root, "add", "notes.md")

            exit_code, payload, _ = self._run_cli(repo_root, "scan-staged")

            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["ok"])
            issues = cast(list[dict[str, Any]], payload["issues"])
            self.assertEqual(issues[0]["rule_id"], "phone-number")

    def test_scan_staged_ignores_binary_patch_content_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._init_repo(repo_root)
            (repo_root / "artifact.o").write_bytes(b"\x7fELF\x01\x01\x01\x00\xd0bad\x00\xff")
            _run_git(repo_root, "add", "artifact.o")

            exit_code, payload, _ = self._run_cli(repo_root, "scan-staged")

            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["issues"], [])

    def test_scan_push_blocks_new_commit_against_remote_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as remote_dir:
            repo_root = Path(temp_dir)
            remote_root = Path(remote_dir)
            subprocess.run(["git", "init", "--bare", "-q", str(remote_root)], check=True, text=True, capture_output=True)
            self._init_repo(repo_root)
            _run_git(repo_root, "remote", "add", "origin", str(remote_root))
            self._commit_file(repo_root, "notes.txt", "baseline\n", "base")
            _run_git(repo_root, "push", "-u", "origin", "main")
            (repo_root / "notes.txt").write_text("baseline\nWarhammer sneaked in here\n", encoding="utf-8")
            _run_git(repo_root, "add", "notes.txt")
            _run_git(repo_root, "commit", "-q", "-m", "add forbidden content")

            local_oid = _run_git(repo_root, "rev-parse", "HEAD").stdout.strip()
            remote_oid = _run_git(repo_root, "rev-parse", "origin/main").stdout.strip()
            updates_path = repo_root / "pre_push_updates.txt"
            updates_path.write_text(
                f"refs/heads/main {local_oid} refs/heads/main {remote_oid}\n",
                encoding="utf-8",
            )

            exit_code, payload, _ = self._run_cli(
                repo_root,
                "scan-push",
                "--remote",
                "origin",
                "--stdin-file",
                str(updates_path),
            )

            self.assertEqual(exit_code, 1)
            self.assertFalse(payload["ok"])
            issues = cast(list[dict[str, Any]], payload["issues"])
            self.assertEqual(issues[0]["rule_id"], "blocked-term-content")
            self.assertEqual(issues[0]["commit"], local_oid)

    def test_scan_push_ignores_binary_commit_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as remote_dir:
            repo_root = Path(temp_dir)
            remote_root = Path(remote_dir)
            subprocess.run(["git", "init", "--bare", "-q", str(remote_root)], check=True, text=True, capture_output=True)
            self._init_repo(repo_root)
            _run_git(repo_root, "remote", "add", "origin", str(remote_root))
            self._commit_file(repo_root, "notes.txt", "baseline\n", "base")
            _run_git(repo_root, "push", "-u", "origin", "main")
            self._commit_bytes(repo_root, "artifact.o", b"\x7fELF\x01\x01\x01\x00\xd0bad\x00\xff", "add binary")

            local_oid = _run_git(repo_root, "rev-parse", "HEAD").stdout.strip()
            remote_oid = _run_git(repo_root, "rev-parse", "origin/main").stdout.strip()
            updates_path = repo_root / "pre_push_updates.txt"
            updates_path.write_text(
                f"refs/heads/main {local_oid} refs/heads/main {remote_oid}\n",
                encoding="utf-8",
            )

            exit_code, payload, _ = self._run_cli(
                repo_root,
                "scan-push",
                "--remote",
                "origin",
                "--stdin-file",
                str(updates_path),
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["issues"], [])


if __name__ == "__main__":
    unittest.main()
