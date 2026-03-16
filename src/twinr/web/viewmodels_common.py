from __future__ import annotations

from datetime import datetime, timedelta, timezone  # AUDIT-FIX(#4): Needed for UTC-normalized reminder defaults and fail-safe timezone fallback.
from pathlib import Path
from stat import S_ISREG  # AUDIT-FIX(#1,#3): Use lstat-based regular-file checks without following symlinks.
from typing import Any
from urllib.parse import quote  # AUDIT-FIX(#5): Encode filenames safely when constructing download URLs.

from fastapi import HTTPException

from twinr.agent.base_agent import TwinrConfig
from twinr.integrations import EMAIL_APP_PASSWORD_ENV_KEY
from twinr.memory.reminders import format_due_label, now_in_timezone
from twinr.web.store import mask_secret

_PROVIDER_OPTIONS = (
    ("openai", "OpenAI"),
    ("deepinfra", "DeepInfra"),
    ("openrouter", "OpenRouter"),
)
_BOOL_OPTIONS = (("true", "Enabled"), ("false", "Disabled"))
_TRISTATE_BOOL_OPTIONS = (("", "Auto"), ("true", "Always send"), ("false", "Never send"))
_YES_NO_OPTIONS = (("true", "Yes"), ("false", "No"))
_REASONING_EFFORT_OPTIONS = (("low", "Low"), ("medium", "Medium"), ("high", "High"))
_SEARCH_CONTEXT_OPTIONS = (("low", "Low"), ("medium", "Medium"), ("high", "High"))
_CONVERSATION_WEB_SEARCH_OPTIONS = (("auto", "Auto"), ("always", "Always"), ("never", "Never"))
_VISION_DETAIL_OPTIONS = (("auto", "Auto"), ("low", "Low"), ("high", "High"))
_GPIO_BIAS_OPTIONS = (("pull-up", "Pull-up"), ("pull-down", "Pull-down"), ("disabled", "Off"))
_WAKEWORD_BACKEND_OPTIONS = (("stt", "STT"), ("openwakeword", "openWakeWord"))
_WAKEWORD_PRIMARY_BACKEND_OPTIONS = (("openwakeword", "openWakeWord"), ("stt", "STT"))
_WAKEWORD_FALLBACK_BACKEND_OPTIONS = (("stt", "STT"), ("disabled", "Disabled"))
_WAKEWORD_VERIFIER_MODE_OPTIONS = (
    ("ambiguity_only", "Only ambiguous hits"),
    ("always", "Always verify"),
    ("disabled", "Disabled"),
)
_EMAIL_PROFILE_OPTIONS = (("gmail", "Gmail"), ("generic_imap_smtp", "Generic IMAP/SMTP"))
_CALENDAR_SOURCE_OPTIONS = (("ics_file", "ICS file"), ("ics_url", "ICS URL"))
_EMAIL_SECRET_KEY = EMAIL_APP_PASSWORD_ENV_KEY

_FILE_NOT_FOUND_DETAIL = "File not found"  # AUDIT-FIX(#1,#3): Standardize fail-closed file rejection behavior.
_UNAVAILABLE_LABEL = "Unavailable"  # AUDIT-FIX(#2): Degrade visibly instead of crashing on malformed reminder state.


def _nav_items() -> tuple[tuple[str, str, str], ...]:
    return (
        ("dashboard", "Dashboard", "/"),
        ("ops_self_test", "Self-Test", "/ops/self-test"),
        ("ops_devices", "Devices", "/ops/devices"),
        ("ops_usage", "LLM Usage", "/ops/usage"),
        ("ops_health", "System Health", "/ops/health"),
        ("ops_logs", "Ops Logs", "/ops/logs"),
        ("ops_config", "Config Checks", "/ops/config"),
        ("ops_support", "Support", "/ops/support"),
        ("integrations", "Integrations", "/integrations"),
        ("voice_profile", "Voice Profile", "/voice-profile"),
        ("automations", "Automations", "/automations"),
        ("personality", "Personality", "/personality"),
        ("memory", "Memory", "/memory"),
        ("connect", "Connect", "/connect"),
        ("settings", "Settings", "/settings"),
        ("user", "User", "/user"),
    )


def _provider_status(env_values: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return (
        ("OpenAI key", mask_secret(env_values.get("OPENAI_API_KEY"))),
        ("OpenAI project", env_values.get("OPENAI_PROJ_ID", "Not configured") or "Not configured"),
        ("DeepInfra key", mask_secret(env_values.get("DEEPINFRA_API_KEY"))),
        ("OpenRouter key", mask_secret(env_values.get("OPENROUTER_API_KEY"))),
    )


def _entry_value(entry: Any, key: str, default: Any = None) -> Any:
    if isinstance(entry, dict):  # AUDIT-FIX(#2): Support dict-backed reminder entries from file-backed or legacy state.
        return entry.get(key, default)
    try:  # AUDIT-FIX(#2): Prevent property/attribute access failures from taking down the whole page.
        return getattr(entry, key)
    except Exception:
        return default


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)  # AUDIT-FIX(#2): Normalize arbitrary persisted values for safe rendering.


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):  # AUDIT-FIX(#2): Handle legacy stringified booleans from persisted state.
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    if value is None:
        return default
    return bool(value)


def _coerce_int(value: Any, *, default: int = 0) -> int:
    try:
        parsed = int(value)  # AUDIT-FIX(#2): Guard rendering against malformed numeric fields.
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):  # AUDIT-FIX(#2): Accept ISO-8601 strings from JSON/file-backed reminder state.
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _safe_isoformat(value: Any) -> str | None:
    parsed = _coerce_datetime(value)
    if parsed is not None:  # AUDIT-FIX(#2): Avoid isoformat() attribute errors on malformed reminder rows.
        return parsed.isoformat()
    return _coerce_text(value)


def _safe_due_label(value: Any, *, timezone_name: str) -> str | None:
    parsed = _coerce_datetime(value)
    if parsed is None:
        return _coerce_text(value)
    try:
        return format_due_label(parsed, timezone_name=timezone_name)  # AUDIT-FIX(#2): Keep UI alive even when timezone conversion/formatting fails.
    except Exception:
        return parsed.isoformat()


def _raise_file_not_found() -> None:
    raise HTTPException(status_code=404, detail=_FILE_NOT_FOUND_DETAIL)  # AUDIT-FIX(#1,#3): Fail closed for all rejected file accesses.


def _default_reminder_due_at(config: TwinrConfig) -> str:
    try:
        now_local = now_in_timezone(config.local_timezone_name)
    except Exception:  # AUDIT-FIX(#4): Broken timezone config must not kill the reminder form.
        now_local = datetime.now(timezone.utc)
    if now_local.tzinfo is None:  # AUDIT-FIX(#4): Normalize to aware datetime before UTC arithmetic.
        now_local = now_local.replace(tzinfo=timezone.utc)
    target_timezone = now_local.tzinfo or timezone.utc
    now_utc = now_local.astimezone(timezone.utc)
    due_at_utc = now_utc.replace(second=0, microsecond=0)
    if due_at_utc < now_utc:
        due_at_utc += timedelta(minutes=1)  # AUDIT-FIX(#4): Round up instead of defaulting to a past/immediate minute.
    return due_at_utc.astimezone(target_timezone).strftime("%Y-%m-%dT%H:%M")


def _reminder_rows(entries: tuple[Any, ...], *, timezone_name: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        delivered = _coerce_bool(_entry_value(entry, "delivered", False))  # AUDIT-FIX(#2): Tolerate malformed reminder payloads without crashing.
        last_error = _coerce_text(_entry_value(entry, "last_error", None))
        status_key = "delivered" if delivered else ("retry" if last_error else "pending")
        if status_key == "delivered":
            status_label = "Delivered"
        elif status_key == "retry":
            status_label = "Retrying"
        else:
            status_label = "Pending"
        rows.append(
            {
                "reminder_id": _coerce_text(_entry_value(entry, "reminder_id", "unknown")) or "unknown",
                "kind": _coerce_text(_entry_value(entry, "kind", "")) or "",
                "summary": _coerce_text(_entry_value(entry, "summary", "")) or "",
                "details": _coerce_text(_entry_value(entry, "details", None)),
                "source": _coerce_text(_entry_value(entry, "source", None)),
                "original_request": _coerce_text(_entry_value(entry, "original_request", None)),
                "delivery_attempts": _coerce_int(_entry_value(entry, "delivery_attempts", 0)),
                "last_error": last_error,
                "created_at": _safe_isoformat(_entry_value(entry, "created_at", None)) or "",
                "updated_at": _safe_isoformat(_entry_value(entry, "updated_at", None)) or "",
                "due_label": _safe_due_label(_entry_value(entry, "due_at", None), timezone_name=timezone_name)
                or _UNAVAILABLE_LABEL,
                "next_attempt_label": _safe_due_label(
                    _entry_value(entry, "next_attempt_at", None),
                    timezone_name=timezone_name,
                ),
                "delivered_at_label": _safe_due_label(
                    _entry_value(entry, "delivered_at", None),
                    timezone_name=timezone_name,
                ),
                "delivered": delivered,
                "status_key": status_key,
                "status_label": status_label,
            }
        )
    return tuple(rows)


def _credential_state_label(value: str | None) -> str:
    return "Configured" if value else "Not configured"


def _format_seconds_label(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") + " s"


def _format_millis_label(value: int) -> str:
    return f"{int(value)} ms"


def _recent_named_files(directory: Path, *, suffix: str) -> tuple[dict[str, str], ...]:
    if not suffix or "/" in suffix or "\\" in suffix:  # AUDIT-FIX(#3): Refuse invalid suffix filters and fail closed.
        return ()
    try:
        if not directory.exists() or not directory.is_dir():  # AUDIT-FIX(#3): Avoid NotADirectoryError and similar filesystem surprises.
            return ()
    except OSError:
        return ()

    files: list[tuple[float, Path]] = []
    try:
        for path in directory.iterdir():
            try:
                path_stat = path.lstat()  # AUDIT-FIX(#3): Use a single metadata snapshot and do not follow symlinks while listing.
            except OSError:
                continue
            if not S_ISREG(path_stat.st_mode):
                continue
            if not path.name.endswith(suffix):
                continue
            files.append((path_stat.st_mtime, path))
    except OSError:
        return ()

    files.sort(key=lambda item: item[0], reverse=True)
    base_href = "/ops/support/download" if suffix == ".zip" else "/ops/self-test/artifacts"
    return tuple(
        {
            "name": path.name,
            "path": path.name,  # AUDIT-FIX(#5): Do not leak absolute server filesystem paths into the web layer.
            "download_href": f"{base_href}/{quote(path.name, safe='')}",  # AUDIT-FIX(#5): URL-encode filenames to keep links correct and unambiguous.
        }
        for _, path in files[:8]
    )


def _resolve_named_file(root: Path, name: str) -> Path:
    safe_name = Path(name).name
    if not name or safe_name != name or safe_name in {".", ".."}:  # AUDIT-FIX(#1): Reject traversal attempts and malformed file names early.
        _raise_file_not_found()

    try:
        resolved_root = root.resolve(strict=True)  # AUDIT-FIX(#1): Anchor validation against a strict, trusted root directory.
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail=_FILE_NOT_FOUND_DETAIL) from exc
    if not resolved_root.is_dir():
        _raise_file_not_found()

    candidate = resolved_root / safe_name
    try:
        candidate_stat = candidate.lstat()  # AUDIT-FIX(#1): Validate the on-disk entry without following symlinks.
    except OSError as exc:
        raise HTTPException(status_code=404, detail=_FILE_NOT_FOUND_DETAIL) from exc
    if not S_ISREG(candidate_stat.st_mode):
        _raise_file_not_found()

    try:
        resolved_candidate = candidate.resolve(strict=True)
        resolved_candidate.relative_to(resolved_root)
        resolved_stat = resolved_candidate.stat()  # AUDIT-FIX(#1): Re-stat after resolution to detect inode swaps during validation.
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=_FILE_NOT_FOUND_DETAIL) from exc

    if not S_ISREG(resolved_stat.st_mode):
        _raise_file_not_found()
    if (resolved_stat.st_dev, resolved_stat.st_ino) != (candidate_stat.st_dev, candidate_stat.st_ino):
        _raise_file_not_found()

    return resolved_candidate