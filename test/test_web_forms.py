from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.web.support.forms import _collect_standard_updates, _text_field


class WebFormsTests(unittest.TestCase):
    def test_text_field_accepts_safe_lowercase_managed_keys(self) -> None:
        field = _text_field("managed_enabled_test", "Managed enabled", {}, "false")

        self.assertEqual(field.key, "managed_enabled_test")

    def test_collect_standard_updates_keeps_registered_lowercase_keys(self) -> None:
        _text_field("managed_profile_test", "Managed profile", {}, "")

        updates = _collect_standard_updates(
            {
                "managed_profile_test": "  gmail  ",
                "managed-profile-test": "ignored",
            }
        )

        self.assertEqual(updates["managed_profile_test"], "gmail")
        self.assertNotIn("managed-profile-test", updates)

    def test_text_field_rejects_unsafe_key_characters(self) -> None:
        with self.assertRaises(ValueError):
            _text_field("managed-profile-test", "Managed profile", {}, "")


if __name__ == "__main__":
    unittest.main()
