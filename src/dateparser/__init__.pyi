from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

def parse(
    date_string: str,
    date_formats: Sequence[str] | None = ...,
    languages: Sequence[str] | None = ...,
    locales: Sequence[str] | None = ...,
    region: str | None = ...,
    settings: Mapping[str, object] | None = ...,
    detect_languages_function: Any = ...,
) -> datetime | None: ...
