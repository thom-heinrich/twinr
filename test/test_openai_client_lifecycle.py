from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.providers.openai.core import client as openai_client_module


class _FakeHttpClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeOpenAIClient:
    def __init__(self, *, http_client: _FakeHttpClient | None = None) -> None:
        self._client = http_client if http_client is not None else _FakeHttpClient()
        self.close_calls = 0
        self.option_kwargs: dict[str, object] = {}

    def with_options(self, **kwargs):
        clone = type(self)(http_client=self._client)
        clone.option_kwargs = dict(kwargs)
        return clone

    def close(self) -> None:
        self.close_calls += 1
        self._client.close()


class OpenAIClientLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        openai_client_module.close_cached_openai_clients()

    def tearDown(self) -> None:
        openai_client_module.close_cached_openai_clients()

    def test_close_openai_client_skips_borrowed_with_options_wrapper(self) -> None:
        base_client = _FakeOpenAIClient()
        borrowed = openai_client_module.openai_client_with_options(
            base_client,
            timeout_s=5.0,
            max_retries=1,
        )

        self.assertIsNot(borrowed, base_client)
        self.assertIs(borrowed._client, base_client._client)
        self.assertFalse(openai_client_module.close_openai_client(borrowed))
        self.assertFalse(base_client._client.closed)
        self.assertEqual(base_client.close_calls, 0)

    def test_close_openai_client_skips_cached_transport_until_cache_shutdown(self) -> None:
        created_client = _FakeOpenAIClient()

        with patch.object(openai_client_module, "_create_openai_client", return_value=created_client):
            cached_client = openai_client_module._default_client_factory(
                TwinrConfig(openai_api_key="test-key")
            )

        borrowed = openai_client_module.openai_client_with_options(
            cached_client,
            timeout_s=3.0,
            max_retries=0,
        )

        self.assertFalse(openai_client_module.close_openai_client(borrowed))
        self.assertFalse(openai_client_module.close_openai_client(cached_client))
        self.assertFalse(created_client._client.closed)
        self.assertEqual(created_client.close_calls, 0)

        openai_client_module.close_cached_openai_clients()

        self.assertTrue(created_client._client.closed)
        self.assertEqual(created_client.close_calls, 1)

    def test_close_openai_client_closes_owned_client(self) -> None:
        owned_client = _FakeOpenAIClient()

        self.assertTrue(openai_client_module.close_openai_client(owned_client))
        self.assertTrue(owned_client._client.closed)
        self.assertEqual(owned_client.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
