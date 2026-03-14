from __future__ import annotations

from twinr.web.store import FileBackedSetting


def _text_field(
    key: str,
    label: str,
    env_values: dict[str, str],
    default: str,
    *,
    help_text: str = "",
    tooltip_text: str = "",
    placeholder: str = "",
    wide: bool = False,
) -> FileBackedSetting:
    return FileBackedSetting(
        key=key,
        label=label,
        value=env_values.get(key, default),
        help_text=help_text,
        tooltip_text=tooltip_text,
        placeholder=placeholder,
        wide=wide,
    )


def _select_field(
    key: str,
    label: str,
    env_values: dict[str, str],
    options: tuple[tuple[str, str], ...],
    default: str,
    *,
    help_text: str = "",
    tooltip_text: str = "",
) -> FileBackedSetting:
    return FileBackedSetting(
        key=key,
        label=label,
        value=env_values.get(key, default),
        help_text=help_text,
        tooltip_text=tooltip_text,
        input_type="select",
        options=options,
    )


def _textarea_field(
    key: str,
    label: str,
    env_values: dict[str, str],
    default: str,
    *,
    help_text: str = "",
    tooltip_text: str = "",
    placeholder: str = "",
    rows: int = 4,
) -> FileBackedSetting:
    return FileBackedSetting(
        key=key,
        label=label,
        value=env_values.get(key, default),
        help_text=help_text,
        tooltip_text=tooltip_text,
        input_type="textarea",
        placeholder=placeholder,
        rows=rows,
        wide=True,
    )


def _collect_standard_updates(form: dict[str, str], *, exclude: set[str] | None = None) -> dict[str, str]:
    blocked = exclude or set()
    return {key: value.strip() for key, value in form.items() if key and key.isupper() and key not in blocked}
