from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality._display_utils import (
    normalized_text,
    stable_fraction,
    truncate_text,
)
from twinr.agent.personality._payload_utils import (
    normalize_mapping,
    normalize_string_tuple,
    required_mapping_text,
)
from twinr.agent.personality._remote_state_utils import resolve_remote_state
from twinr.agent.personality.profile_defaults import (
    default_humor_profile,
    default_style_profile,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore


class _DummyRemoteState:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled


class PersonalitySharedHelperTests(unittest.TestCase):
    def test_payload_utils_normalize_mapping_trims_keys_and_rejects_blank_ones(self) -> None:
        normalized = normalize_mapping({" topic ": 1, "other": 2}, field_name="meta")

        self.assertEqual(normalized, {"topic": 1, "other": 2})
        with self.assertRaises(ValueError):
            normalize_mapping({"   ": 1}, field_name="meta")

    def test_payload_utils_required_mapping_text_respects_aliases(self) -> None:
        value = required_mapping_text(
            {"description": "  hello world  "},
            field_name="summary",
            aliases=("description",),
        )

        self.assertEqual(value, "hello world")
        self.assertEqual(
            normalize_string_tuple((" one ", "two"), field_name="items"),
            ("one", "two"),
        )

    def test_display_utils_keep_text_and_variation_deterministic(self) -> None:
        self.assertEqual(normalized_text("  hello   world  "), "hello world")
        self.assertEqual(truncate_text("abcdef", max_len=5), "abcd…")
        self.assertEqual(stable_fraction("topic", "seed"), stable_fraction("topic", "seed"))
        self.assertNotEqual(stable_fraction("topic", "seed"), stable_fraction("topic", "other"))

    def test_profile_defaults_return_expected_baselines(self) -> None:
        style = default_style_profile()
        humor = default_humor_profile()

        self.assertEqual(style.verbosity, 0.5)
        self.assertEqual(style.initiative, 0.45)
        self.assertEqual(humor.intensity, 0.25)
        self.assertIn("never mocking", humor.boundaries)

    def test_resolve_remote_state_prefers_explicit_adapter_and_filters_disabled_default(self) -> None:
        config = TwinrConfig(project_root=".")
        explicit = _DummyRemoteState(enabled=True)

        self.assertIs(
            resolve_remote_state(
                config=config,
                remote_state=cast(LongTermRemoteStateStore, explicit),
            ),
            explicit,
        )

        with patch(
            "twinr.agent.personality._remote_state_utils.LongTermRemoteStateStore.from_config",
            return_value=_DummyRemoteState(enabled=False),
        ):
            self.assertIsNone(resolve_remote_state(config=config, remote_state=None))

        enabled = _DummyRemoteState(enabled=True)
        with patch(
            "twinr.agent.personality._remote_state_utils.LongTermRemoteStateStore.from_config",
            return_value=enabled,
        ):
            self.assertIs(resolve_remote_state(config=config, remote_state=None), enabled)


if __name__ == "__main__":
    unittest.main()
