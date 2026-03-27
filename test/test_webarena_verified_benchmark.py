# ruff: noqa: E402
"""Regression coverage for the WebArena Verified benchmark runner."""

from pathlib import Path
import sys
import tempfile
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from test.run_webarena_verified_benchmark import resolve_benchmark_auth_state_root


class WebArenaVerifiedBenchmarkRunnerTests(unittest.TestCase):
    def test_resolve_benchmark_auth_state_root_creates_unique_default_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            first = resolve_benchmark_auth_state_root(raw_auth_state_root="", repo_root=repo_root)
            second = resolve_benchmark_auth_state_root(raw_auth_state_root="", repo_root=repo_root)

        self.assertTrue(first.is_dir())
        self.assertTrue(second.is_dir())
        self.assertNotEqual(first, second)

    def test_resolve_benchmark_auth_state_root_preserves_explicit_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            explicit_root = Path(temp_dir) / "explicit-auth-root"
            resolved = resolve_benchmark_auth_state_root(
                raw_auth_state_root=str(explicit_root),
                repo_root=Path(temp_dir),
            )
            self.assertEqual(resolved, explicit_root.resolve())
            self.assertTrue(resolved.is_dir())


if __name__ == "__main__":
    unittest.main()
