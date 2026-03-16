from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import dateparser

from twinr.text_utils import collapse_whitespace

# AUDIT-FIX(#2): Use module logging so parser/config failures are observable
# without echoing raw user utterances into logs.
logger = logging.getLogger(__name__)


def _is_aware_datetime(value: datetime) -> bool:
    # AUDIT-FIX(#4): Python defines awareness as tzinfo set and utcoffset()
    # returning a non-None offset.
    try:
        return value.tzinfo is not None and value.utcoffset() is not None
    except Exception:
        return False


def _normalize_reference_datetime(
    reference_datetime: datetime | None,
    zone: ZoneInfo,
) -> datetime:
    if reference_datetime is None:
        return datetime.now(zone)
    if _is_aware_datetime(reference_datetime):
        return reference_datetime.astimezone(zone)

    # AUDIT-FIX(#5): replace(tzinfo=...) only attaches a zone and does not
    # validate/normalize DST-gap wall times, so normalize through UTC first.
    localized = reference_datetime.replace(tzinfo=zone)
    return localized.astimezone(timezone.utc).astimezone(zone)


def parse_local_date_text(
    value: str | None,
    *,
    timezone_name: str,
    reference_datetime: datetime | None = None,
) -> date | None:
    # AUDIT-FIX(#6): Respect the optional-input contract before delegating to a
    # helper whose None-handling is not guaranteed by this file.
    if value is None:
        return None

    text = collapse_whitespace(value)
    if not text:
        return None

    # AUDIT-FIX(#1): Fail closed on blank timezone keys instead of bubbling a
    # ZoneInfo constructor error up the stack.
    timezone_key = timezone_name.strip()
    if not timezone_key:
        logger.warning("parse_local_date_text: blank timezone key")
        return None

    try:
        # AUDIT-FIX(#1): Invalid keys and missing tzdata currently raise here.
        zone = ZoneInfo(timezone_key)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        logger.warning(
            "parse_local_date_text: invalid timezone key %r (%s)",
            timezone_name,
            exc.__class__.__name__,
        )
        return None

    relative_base = _normalize_reference_datetime(reference_datetime, zone)

    try:
        parsed = dateparser.parse(
            text,
            settings={
                "RELATIVE_BASE": relative_base,
                "REQUIRE_PARTS": ["day"],
                "RETURN_AS_TIMEZONE_AWARE": True,
                "TIMEZONE": timezone_key,
                "TO_TIMEZONE": timezone_key,
            },
        )
    except (OverflowError, TypeError, ValueError) as exc:
        # AUDIT-FIX(#2): Third-party parser/config errors should degrade to
        # None, not crash a request path.
        logger.warning(
            "parse_local_date_text: date parsing failed with %s",
            exc.__class__.__name__,
        )
        return None

    if parsed is None:
        return None

    if not _is_aware_datetime(parsed):
        # AUDIT-FIX(#3): astimezone() on a naive datetime assumes the host's
        # system timezone, which can silently shift the parsed local date.
        return parsed.date()

    return parsed.astimezone(zone).date()