from pathlib import Path
from unittest.mock import patch
import os
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.ops.runtime_env import prime_user_session_audio_env


class RuntimeEnvTests(unittest.TestCase):
    def test_prime_user_session_audio_env_sets_missing_runtime_vars(self) -> None:
        original = {key: os.environ.get(key) for key in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "PULSE_SERVER")}
        for key in original:
            os.environ.pop(key, None)
        runtime_dir = Path("/run/user/1234")

        def fake_is_dir(path: Path) -> bool:
            return path == runtime_dir

        def fake_exists(path: Path) -> bool:
            return path in {runtime_dir / "bus", runtime_dir / "pulse" / "native"}

        try:
            with patch("os.getuid", return_value=1234):
                with patch("pathlib.Path.is_dir", fake_is_dir):
                    with patch("pathlib.Path.exists", fake_exists):
                        updates = prime_user_session_audio_env()
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(updates["XDG_RUNTIME_DIR"], "/run/user/1234")
        self.assertEqual(updates["DBUS_SESSION_BUS_ADDRESS"], "unix:path=/run/user/1234/bus")
        self.assertEqual(updates["PULSE_SERVER"], "unix:/run/user/1234/pulse/native")

    def test_prime_user_session_audio_env_does_not_override_existing_values(self) -> None:
        original = {key: os.environ.get(key) for key in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "PULSE_SERVER")}
        os.environ["XDG_RUNTIME_DIR"] = "/already/set"
        os.environ["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=/already/bus"
        os.environ["PULSE_SERVER"] = "unix:/already/pulse"
        try:
            with patch("os.getuid", return_value=1234):
                updates = prime_user_session_audio_env()
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(updates, {})


if __name__ == "__main__":
    unittest.main()
