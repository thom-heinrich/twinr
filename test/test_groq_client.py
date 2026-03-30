from pathlib import Path
from types import ModuleType
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.providers.groq.client import default_groq_client


class GroqClientFactoryTests(unittest.TestCase):
    def _restore_module(self, name: str, original: ModuleType | None) -> None:
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

    def test_default_groq_client_prefers_official_sdk_and_strips_openai_suffix(self) -> None:
        captured_kwargs: dict[str, object] = {}

        class FakeGroq:
            def __init__(self, **kwargs) -> None:
                captured_kwargs.update(kwargs)

        original_module = sys.modules.get("groq")
        fake_groq_module = ModuleType("groq")
        setattr(fake_groq_module, "Groq", FakeGroq)
        sys.modules["groq"] = fake_groq_module
        self.addCleanup(self._restore_module, "groq", original_module)

        default_groq_client(
            TwinrConfig(
                groq_api_key="groq-key",
                groq_base_url="https://localhost:8443/openai/v1",
            )
        )

        self.assertEqual(captured_kwargs["api_key"], "groq-key")
        self.assertEqual(captured_kwargs["base_url"], "https://localhost:8443")
        self.assertEqual(captured_kwargs["max_retries"], 2)

    def test_default_groq_client_can_force_openai_compat_backend(self) -> None:
        captured_kwargs: dict[str, object] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                captured_kwargs.update(kwargs)

        original_module = sys.modules.get("openai")
        fake_openai_module = ModuleType("openai")
        setattr(fake_openai_module, "OpenAI", FakeOpenAI)
        sys.modules["openai"] = fake_openai_module
        self.addCleanup(self._restore_module, "openai", original_module)

        default_groq_client(
            TwinrConfig(
                groq_api_key="groq-key",
                groq_base_url="https://localhost:11434",
                groq_sdk_backend="openai_compat",
            )
        )

        self.assertEqual(captured_kwargs["api_key"], "groq-key")
        self.assertEqual(
            captured_kwargs["base_url"],
            "https://localhost:11434/openai/v1",
        )
        self.assertEqual(captured_kwargs["max_retries"], 2)

    def test_default_groq_client_rejects_keepalive_pool_misconfiguration(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "groq_max_keepalive_connections must be <= groq_max_connections",
        ):
            default_groq_client(
                TwinrConfig(
                    groq_api_key="groq-key",
                    groq_sdk_backend="openai_compat",
                    groq_max_connections=4,
                    groq_max_keepalive_connections=5,
                )
            )


if __name__ == "__main__":
    unittest.main()
