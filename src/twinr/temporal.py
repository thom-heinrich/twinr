from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import dateparser

from twinr.text_utils import collapse_whitespace


def parse_local_date_text(
    value: str | None,
    *,
    timezone_name: str,
    reference_datetime: datetime | None = None,
) -> date | None:
    text = collapse_whitespace(value)
    if not text:
        return None
    zone = ZoneInfo(timezone_name)
    if reference_datetime is None:
        relative_base = datetime.now(zone)
    elif reference_datetime.tzinfo is None:
        relative_base = reference_datetime.replace(tzinfo=zone)
    else:
        relative_base = reference_datetime.astimezone(zone)
    parsed = dateparser.parse(
        text,
        settings={
            "RELATIVE_BASE": relative_base,
            "REQUIRE_PARTS": ["day"],
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TIMEZONE": timezone_name,
            "TO_TIMEZONE": timezone_name,
        },
    )
    if parsed is None:
        return None
    return parsed.astimezone(zone).date()
