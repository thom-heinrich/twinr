from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs


def parse_urlencoded_form(body: bytes) -> dict[str, str]:
    parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in parsed.items()}


def read_env_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def write_env_updates(path: Path, updates: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    result: list[str] = []
    seen: set[str] = set()
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            result.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            if key not in seen:
                result.append(f"{key}={_quote_env_value(updates[key])}")
                seen.add(key)
            continue
        result.append(line)

    for key, value in updates.items():
        if key in seen:
            continue
        result.append(f"{key}={_quote_env_value(value)}")

    path.write_text("\n".join(result).rstrip() + "\n", encoding="utf-8")


def read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"
    path.write_text(normalized, encoding="utf-8")


def mask_secret(value: str | None) -> str:
    if not value:
        return "Not configured"
    if len(value) <= 8:
        return "Configured"
    return f"{value[:4]}…{value[-4:]}"


@dataclass(frozen=True, slots=True)
class FileBackedSetting:
    key: str
    label: str
    value: str
    help_text: str = ""
    tooltip_text: str = ""
    input_type: str = "text"
    options: tuple[tuple[str, str], ...] = ()
    placeholder: str = ""
    rows: int = 4
    wide: bool = False
    secret: bool = False


def _quote_env_value(value: str) -> str:
    normalized = value.strip()
    if normalized == "":
        return '""'
    if any(char.isspace() for char in normalized) or any(char in normalized for char in ['#', '"', "'"]):
        escaped = normalized.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return normalized
