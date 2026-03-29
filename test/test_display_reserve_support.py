from datetime import datetime, time as LocalTime, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.proactive.runtime.display_reserve_support import (
    compact_text,
    format_timestamp,
    parse_local_time,
    parse_timestamp,
)


class DisplayReserveSupportTests(unittest.TestCase):
    def test_compact_text_collapses_whitespace_and_bounds_length(self) -> None:
        self.assertEqual(compact_text("  hello   world  ", max_len=32), "hello world")
        self.assertEqual(compact_text("abcdefghij", max_len=7), "abcdef…")

    def test_parse_timestamp_accepts_z_suffix_and_normalizes_to_utc(self) -> None:
        parsed = parse_timestamp("2026-03-24T08:15:00Z")

        self.assertEqual(parsed, datetime(2026, 3, 24, 8, 15, tzinfo=timezone.utc))

    def test_parse_timestamp_returns_none_for_invalid_text(self) -> None:
        self.assertIsNone(parse_timestamp("not-a-timestamp"))

    def test_format_timestamp_serializes_as_utc_isoformat(self) -> None:
        value = datetime(2026, 3, 24, 9, 15, tzinfo=timezone.utc)

        self.assertEqual(format_timestamp(value), "2026-03-24T09:15:00Z")

    def test_parse_local_time_falls_back_for_invalid_values(self) -> None:
        self.assertEqual(
            parse_local_time("25:99", fallback="07:30"),
            LocalTime(hour=7, minute=30),
        )
        self.assertEqual(
            parse_local_time("06:45", fallback="07:30"),
            LocalTime(hour=6, minute=45),
        )


if __name__ == "__main__":
    unittest.main()
