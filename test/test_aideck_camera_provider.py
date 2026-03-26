from pathlib import Path
from typing import Any, cast
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.social.aideck_camera_provider import AIDeckOpenAIVisionObservationProvider


class AIDeckCameraProviderTests(unittest.TestCase):
    def test_from_config_requires_aideck_camera_device(self) -> None:
        config = TwinrConfig(camera_device="/dev/video0")

        with self.assertRaises(ValueError):
            AIDeckOpenAIVisionObservationProvider.from_config(
                config,
                backend=cast(Any, object()),
                camera=cast(Any, object()),
            )

    def test_provider_delegates_to_nested_openai_observer(self) -> None:
        snapshot = object()

        class _FakeObserver:
            def __init__(self) -> None:
                self.calls = 0

            def observe(self):
                self.calls += 1
                return snapshot

        observer = _FakeObserver()
        provider = AIDeckOpenAIVisionObservationProvider(observer=observer)

        result = provider.observe()

        self.assertIs(result, snapshot)
        self.assertEqual(observer.calls, 1)


if __name__ == "__main__":
    unittest.main()
