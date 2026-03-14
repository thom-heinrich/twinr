from __future__ import annotations

from datetime import datetime
import unittest
from zoneinfo import ZoneInfo

from twinr.temporal import parse_local_date_text


class ParseLocalDateTextTests(unittest.TestCase):
    def test_resolves_relative_day_terms_without_inline_keyword_logic(self) -> None:
        reference = datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin"))

        self.assertEqual(
            parse_local_date_text("today", timezone_name="Europe/Berlin", reference_datetime=reference),
            reference.date(),
        )
        self.assertEqual(
            parse_local_date_text("heute", timezone_name="Europe/Berlin", reference_datetime=reference),
            reference.date(),
        )
        self.assertEqual(
            parse_local_date_text("tomorrow", timezone_name="Europe/Berlin", reference_datetime=reference),
            datetime(2026, 3, 15, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")).date(),
        )
        self.assertEqual(
            parse_local_date_text("morgen", timezone_name="Europe/Berlin", reference_datetime=reference),
            datetime(2026, 3, 15, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")).date(),
        )

    def test_accepts_iso_dates_and_rejects_non_date_text(self) -> None:
        reference = datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin"))

        self.assertEqual(
            parse_local_date_text("2026-03-20", timezone_name="Europe/Berlin", reference_datetime=reference),
            datetime(2026, 3, 20, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")).date(),
        )
        self.assertEqual(
            parse_local_date_text(
                "2026-03-20T10:30:00+01:00",
                timezone_name="Europe/Berlin",
                reference_datetime=reference,
            ),
            datetime(2026, 3, 20, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")).date(),
        )
        self.assertIsNone(
            parse_local_date_text("go for a walk", timezone_name="Europe/Berlin", reference_datetime=reference)
        )


if __name__ == "__main__":
    unittest.main()
