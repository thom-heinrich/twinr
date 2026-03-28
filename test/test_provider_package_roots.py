from pathlib import Path
import importlib
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _purge_modules(*prefixes: str) -> None:
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            sys.modules.pop(name, None)


class ProviderPackageRootImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_provider_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "twinr.providers" or name.startswith("twinr.providers.")
        }
        self.addCleanup(self._restore_provider_modules)

    def _restore_provider_modules(self) -> None:
        _purge_modules("twinr.providers")
        sys.modules.update(self._original_provider_modules)

    def test_providers_root_keeps_factory_and_openai_surfaces_lazy(self) -> None:
        _purge_modules("twinr.providers")

        module = importlib.import_module("twinr.providers")

        self.assertNotIn("twinr.providers.factory", sys.modules)
        self.assertNotIn("twinr.providers.openai", sys.modules)
        self.assertNotIn("twinr.providers.openai.api", sys.modules)

        builder = module.build_streaming_provider_bundle
        self.assertTrue(callable(builder))
        self.assertIn("twinr.providers.factory", sys.modules)

        backend_type = module.OpenAIBackend
        self.assertEqual(backend_type.__name__, "OpenAIBackend")
        self.assertIn("twinr.providers.openai", sys.modules)
        self.assertIn("twinr.providers.openai.api", sys.modules)
        self.assertIn("twinr.providers.openai.api.backend", sys.modules)

    def test_deepgram_root_import_is_lazy(self) -> None:
        _purge_modules("twinr.providers.deepgram")

        module = importlib.import_module("twinr.providers.deepgram")

        self.assertNotIn("twinr.providers.deepgram.speech", sys.modules)

        provider_type = module.DeepgramSpeechToTextProvider
        self.assertEqual(provider_type.__name__, "DeepgramSpeechToTextProvider")
        self.assertIn("twinr.providers.deepgram.speech", sys.modules)

    def test_groq_root_import_is_lazy(self) -> None:
        _purge_modules("twinr.providers.groq")

        module = importlib.import_module("twinr.providers.groq")

        self.assertNotIn("twinr.providers.groq.adapters", sys.modules)

        provider_type = module.GroqAgentTextProvider
        self.assertEqual(provider_type.__name__, "GroqAgentTextProvider")
        self.assertIn("twinr.providers.groq.adapters", sys.modules)

    def test_openai_root_keeps_api_core_and_realtime_lazy(self) -> None:
        _purge_modules("twinr.providers.openai")

        module = importlib.import_module("twinr.providers.openai")

        self.assertNotIn("twinr.providers.openai.api", sys.modules)
        self.assertNotIn("twinr.providers.openai.core", sys.modules)
        self.assertNotIn("twinr.providers.openai.realtime", sys.modules)

        image_type = module.OpenAIImageInput
        self.assertEqual(image_type.__name__, "OpenAIImageInput")
        self.assertIn("twinr.providers.openai.core", sys.modules)
        self.assertNotIn("twinr.providers.openai.api", sys.modules)
        self.assertNotIn("twinr.providers.openai.realtime", sys.modules)

        realtime_type = module.OpenAIRealtimeSession
        self.assertEqual(realtime_type.__name__, "OpenAIRealtimeSession")
        self.assertIn("twinr.providers.openai.realtime", sys.modules)
        self.assertNotIn("twinr.providers.openai.api", sys.modules)

    def test_openai_api_root_keeps_backend_and_adapters_split(self) -> None:
        _purge_modules("twinr.providers.openai.api")

        module = importlib.import_module("twinr.providers.openai.api")

        self.assertNotIn("twinr.providers.openai.api.backend", sys.modules)
        self.assertNotIn("twinr.providers.openai.api.adapters", sys.modules)

        backend_type = module.OpenAIBackend
        self.assertEqual(backend_type.__name__, "OpenAIBackend")
        self.assertIn("twinr.providers.openai.api.backend", sys.modules)
        self.assertNotIn("twinr.providers.openai.api.adapters", sys.modules)

        bundle_type = module.OpenAIProviderBundle
        self.assertEqual(bundle_type.__name__, "OpenAIProviderBundle")
        self.assertIn("twinr.providers.openai.api.adapters", sys.modules)

    def test_openai_capabilities_root_keeps_sibling_mixins_lazy(self) -> None:
        _purge_modules("twinr.providers.openai.capabilities")

        module = importlib.import_module("twinr.providers.openai.capabilities")

        self.assertNotIn("twinr.providers.openai.capabilities.printing", sys.modules)
        self.assertNotIn("twinr.providers.openai.capabilities.search", sys.modules)

        print_mixin = module.OpenAIPrintMixin
        self.assertEqual(print_mixin.__name__, "OpenAIPrintMixin")
        self.assertIn("twinr.providers.openai.capabilities.printing", sys.modules)
        self.assertNotIn("twinr.providers.openai.capabilities.search", sys.modules)

    def test_openai_core_root_keeps_client_and_types_split(self) -> None:
        _purge_modules("twinr.providers.openai.core")

        module = importlib.import_module("twinr.providers.openai.core")

        self.assertNotIn("twinr.providers.openai.core.client", sys.modules)
        self.assertNotIn("twinr.providers.openai.core.types", sys.modules)

        default_client_factory = module._default_client_factory
        self.assertTrue(callable(default_client_factory))
        self.assertIn("twinr.providers.openai.core.client", sys.modules)
        self.assertNotIn("twinr.providers.openai.core.types", sys.modules)

        image_type = module.OpenAIImageInput
        self.assertEqual(image_type.__name__, "OpenAIImageInput")
        self.assertIn("twinr.providers.openai.core.types", sys.modules)

    def test_openai_realtime_root_import_is_lazy(self) -> None:
        _purge_modules("twinr.providers.openai.realtime")

        module = importlib.import_module("twinr.providers.openai.realtime")

        self.assertNotIn("twinr.providers.openai.realtime.session", sys.modules)

        session_type = module.OpenAIRealtimeSession
        self.assertEqual(session_type.__name__, "OpenAIRealtimeSession")
        self.assertIn("twinr.providers.openai.realtime.session", sys.modules)


if __name__ == "__main__":
    unittest.main()
