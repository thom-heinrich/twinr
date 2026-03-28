"""Build redacted Twinr support bundles from local ops artifacts.

This module gathers config checks, ops events, runtime snapshots, usage data,
health snapshots, and recent self-test artifacts into a bounded ZIP export.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import json
import logging
import os
import re
import stat
import tempfile
from typing import Any, Callable, TypeGuard, TypeVar
from urllib.parse import SplitResult, urlsplit, urlunsplit
from uuid import uuid4

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.checks import check_summary, run_config_checks
from twinr.ops.events import TwinrOpsEventStore
from twinr.ops.health import collect_system_health
from twinr.ops.paths import resolve_ops_paths_for_config
from twinr.ops.usage import TwinrUsageStore

_SECRET_MARKERS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSCODE",
    "COOKIE",
    "SESSION",
    "CREDENTIAL",
    "PRIVATE",
    "CERT",
    "DSN",
)  # AUDIT-FIX(#2): Broaden secret detection for support-bundle exports.

_MAX_EVENT_LIMIT = 1000  # AUDIT-FIX(#5): Clamp user-supplied limits to protect RPi resources.
_MAX_SELF_TEST_ARTIFACTS = 4
_MAX_SELF_TEST_ARTIFACT_BYTES = 5 * 1024 * 1024  # AUDIT-FIX(#10): Skip oversized artifacts that could bloat the bundle.

_LOGGER = logging.getLogger(__name__)

_RUNTIME_SNAPSHOT_DROP_KEYS = {
    "user_voice_status",
    "user_voice_confidence",
    "user_voice_checked_at",
    "conversation_history",
    "message_history",
    "messages",
    "last_transcript",
    "transcript",
    "audio_bytes",
    "audio_buffer",
    "recording_path",
    "camera_frame",
    "image_bytes",
    "face_encoding",
    "voice_sample",
}
_RUNTIME_PRIVATE_KEY_MARKERS = (
    "transcript",
    "utterance",
    "recording",
    "audio_bytes",
    "pcm",
    "wav",
    "camera_frame",
    "image_bytes",
    "face_encoding",
    "voice_sample",
    "email",
    "phone",
    "address",
    "contact",
)
_RUNTIME_SNAPSHOT_DROP_KEYS_UPPER = {key.upper() for key in _RUNTIME_SNAPSHOT_DROP_KEYS}

T = TypeVar("T")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")  # AUDIT-FIX(#3): Microseconds reduce name collisions during concurrent bundle builds.


@dataclass(frozen=True, slots=True)
class SupportBundleInfo:
    """Describe one generated Twinr support bundle archive."""

    bundle_name: str
    bundle_path: str
    created_at: str
    file_count: int
    includes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_support_bundle(
    config: TwinrConfig,
    *,
    env_path: str | Path,
    event_limit: int = 100,
) -> SupportBundleInfo:
    """Build a redacted support bundle from local Twinr ops evidence.

    Args:
        config: Twinr runtime configuration that points at the local ops
            stores and runtime snapshot.
        env_path: Path to the environment file whose relevant values should be
            redacted into the bundle.
        event_limit: Maximum number of recent ops events to include.

    Returns:
        Metadata describing the created support bundle archive.
    """

    paths = resolve_ops_paths_for_config(config)
    paths.bundles_root.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc).isoformat()
    bundle_name = f"twinr-support-{_utc_stamp()}-{uuid4().hex[:8]}.zip"  # AUDIT-FIX(#3): Add a random suffix so same-second calls cannot overwrite each other.
    bundle_path = paths.bundles_root / bundle_name
    temp_bundle_path = _create_temp_bundle_path(paths.bundles_root)  # AUDIT-FIX(#3): Build the archive off to the side and promote it atomically when complete.

    safe_event_limit = _sanitize_limit(event_limit, default=100, maximum=_MAX_EVENT_LIMIT)  # AUDIT-FIX(#5): Reject pathological negative/huge limits without crashing.
    generation_errors: dict[str, str] = {}  # AUDIT-FIX(#4): Preserve partial diagnostics instead of aborting the full support flow on the first collector failure.

    env_source_path = Path(env_path)
    env_values, env_error = _read_env_values(env_source_path)  # AUDIT-FIX(#8): Read .env defensively and tolerate absent/unreadable files.
    if env_error is not None:
        generation_errors["env_file"] = env_error
    redacted_env = redact_env_values(env_values)

    checks: list[object] = _collect_or_default(
        "config_checks",
        lambda: list(run_config_checks(config)),
        [],
        generation_errors,
    )  # AUDIT-FIX(#4): Keep bundle generation alive when config checks fail.
    summary: object
    if "config_checks" in generation_errors:
        summary = {"status": "unavailable", "reason": generation_errors["config_checks"]}
    else:
        summary = _collect_or_default(
            "config_check_summary",
            lambda: check_summary(checks),
            {"status": "unavailable"},
            generation_errors,
        )  # AUDIT-FIX(#4): Summary generation should degrade gracefully too.

    event_store = _collect_or_default(
        "event_store",
        lambda: TwinrOpsEventStore.from_config(config),
        None,
        generation_errors,
    )
    events: list[object] = []
    if event_store is not None and safe_event_limit > 0:
        events = _collect_or_default(
            "events",
            lambda: list(event_store.tail(limit=safe_event_limit)),
            [],
            generation_errors,
        )  # AUDIT-FIX(#4): Event tail failures should not prevent support export.
    errors = [entry for entry in events if _event_level(entry) == "error"][-20:]

    snapshot_path = Path(config.runtime_state_path)
    snapshot_payload_raw, snapshot_error = _read_json(snapshot_path)  # AUDIT-FIX(#9): Surface snapshot read/parse failures instead of silently pretending the file does not exist.
    if snapshot_error is not None:
        generation_errors["runtime_snapshot"] = snapshot_error
    snapshot_payload = _redact_runtime_snapshot_payload(snapshot_payload_raw)  # AUDIT-FIX(#2): Recursively scrub nested secrets/PII from runtime state exports.

    usage_store = _collect_or_default(
        "usage_store",
        lambda: TwinrUsageStore.from_config(config),
        None,
        generation_errors,
    )
    usage_summary: object = {}
    recent_usage: list[object] = []
    if usage_store is not None:
        usage_summary = _collect_or_default(
            "usage_summary",
            lambda: usage_store.summary(),
            {},
            generation_errors,
        )  # AUDIT-FIX(#4): Usage summary failures should degrade to an empty payload.
        recent_usage = _collect_or_default(
            "recent_usage",
            lambda: list(usage_store.tail(limit=50)),
            [],
            generation_errors,
        )  # AUDIT-FIX(#4): Recent usage export should not take down the whole bundle.

    health_payload: object = _collect_or_default(
        "system_health",
        lambda: collect_system_health(config),
        {},
        generation_errors,
    )  # AUDIT-FIX(#4): Health collection is diagnostically useful but non-fatal.

    self_test_entries: list[tuple[str, bytes]] = []
    for artifact_path in _latest_self_test_artifacts(paths.self_tests_root, limit=_MAX_SELF_TEST_ARTIFACTS):
        artifact_bytes, artifact_error = _read_bytes_file(
            artifact_path,
            max_bytes=_MAX_SELF_TEST_ARTIFACT_BYTES,
        )  # AUDIT-FIX(#1, #10): Re-open artifacts as bounded regular files so symlinks and giant payloads are skipped safely.
        if artifact_bytes is None:
            if artifact_error is not None:
                generation_errors[f"self_tests/{artifact_path.name}"] = artifact_error
            continue
        arcname = f"self_tests/{_sanitize_archive_component(artifact_path.name)}"  # AUDIT-FIX(#1): Sanitize archive names so file names cannot escape their intended ZIP directory.
        self_test_entries.append((arcname, artifact_bytes))

    archive_entries: list[tuple[str, object]] = [
        (
            "summary.json",
            {
                "created_at": created_at,
                "bundle_name": bundle_name,
                "env_path": _path_for_report(env_source_path),
                "runtime_state_path": _path_for_report(snapshot_path),
                "event_limit_requested": event_limit,
                "event_limit_used": safe_event_limit,
                "check_summary": summary,
                "generation_error_count": len(generation_errors),
            },
        ),
        ("redacted_env.json", redacted_env),
        ("config_checks.json", checks),
        ("events.json", events),
        ("errors.json", errors),
        ("system_health.json", health_payload),
        ("usage_summary.json", usage_summary),
        ("recent_usage.json", recent_usage),
    ]
    if snapshot_payload is not None:
        archive_entries.append(("runtime_snapshot.json", snapshot_payload))
    if generation_errors:
        archive_entries.append(("generation_errors.json", generation_errors))  # AUDIT-FIX(#4, #9): Ship explicit collection failures to support instead of hiding them.

    includes: list[str] = []
    try:
        with ZipFile(temp_bundle_path, "w", compression=ZIP_DEFLATED) as archive:
            for archive_name, payload in archive_entries:
                archive.writestr(archive_name, _json_text(payload))  # AUDIT-FIX(#6): Serialize through a JSON-safe normalizer so datetime/path/custom objects do not explode at runtime.
                includes.append(archive_name)
            for archive_name, payload_bytes in self_test_entries:
                archive.writestr(archive_name, payload_bytes)
                includes.append(archive_name)
        os.replace(temp_bundle_path, bundle_path)  # AUDIT-FIX(#3): Atomic replacement prevents partially-written bundles from appearing as valid artifacts.
    except Exception:
        _unlink_quietly(temp_bundle_path)  # AUDIT-FIX(#3, #4): Clean up temp files on failure so low-disk conditions do not snowball.
        raise

    if event_store is not None:
        try:
            event_store.append(
                event="support_bundle_created",
                message="Support bundle created.",
                data={"bundle_name": bundle_name, "file_count": len(includes)},
            )
        except Exception:
            _LOGGER.warning(
                "Support bundle telemetry append failed after bundle creation.",
                exc_info=True,
            )  # AUDIT-FIX(#7): Telemetry logging is best-effort and must not convert a successful bundle build into an exception.

    return SupportBundleInfo(
        bundle_name=bundle_name,
        bundle_path=str(bundle_path),
        created_at=created_at,
        file_count=len(includes),
        includes=tuple(includes),
    )


def redact_env_values(values: dict[str, str]) -> dict[str, str]:
    """Redact secret-like environment values before export."""

    redacted: dict[str, str] = {}
    for key, value in sorted(values.items()):
        if not _is_relevant_key(key):
            continue
        redacted[key] = _redact_env_value(key, value)  # AUDIT-FIX(#2): Apply stronger secret masking heuristics before shipping env values off-device.
    return redacted


def _read_env_values(path: Path) -> tuple[dict[str, str], str | None]:
    text, error = _read_text_file(path)  # AUDIT-FIX(#8): Centralize safe file reading so unreadable/symlinked files become structured diagnostics.
    if text is None:
        return {}, error
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        parsed = _parse_env_assignment(raw_line)  # AUDIT-FIX(#8): Tolerate export prefixes and inline comments in .env files.
        if parsed is None:
            continue
        key, value = parsed
        values[key] = value
    return values, None


def _read_json(path: Path) -> tuple[object | None, str | None]:
    text, error = _read_text_file(path)  # AUDIT-FIX(#9): Use the same safe file-open path and propagate concrete read errors up to the bundle summary.
    if text is None:
        return None, error
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, _format_exception(exc)


def _redact_runtime_snapshot_payload(payload: object | None) -> object | None:
    return _redact_nested_payload(
        _unwrap_runtime_snapshot_payload(payload)
    )  # AUDIT-FIX(#2): Redact the logical snapshot payload even when the on-disk file uses the schema-v2 integrity envelope.


def _unwrap_runtime_snapshot_payload(payload: object | None) -> object | None:
    if not isinstance(payload, Mapping):
        return payload
    nested_payload = payload.get("payload")
    if payload.get("format") == "twinr.runtime_snapshot" and isinstance(nested_payload, Mapping):
        return nested_payload
    return payload


def _latest_self_test_artifacts(root: Path, *, limit: int = 4) -> tuple[Path, ...]:
    if limit <= 0:
        return ()
    try:
        root_path = root.resolve(strict=True)
    except OSError:
        return ()
    if not root_path.is_dir():
        return ()
    files: list[tuple[float, Path]] = []
    try:
        with os.scandir(root_path) as entries:
            for entry in entries:
                try:
                    if entry.is_symlink():
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    entry_stat = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if entry_stat.st_size > _MAX_SELF_TEST_ARTIFACT_BYTES:
                    continue  # AUDIT-FIX(#10): Skip oversized artifacts before archive assembly.
                files.append((entry_stat.st_mtime, Path(entry.path)))
    except OSError:
        return ()
    files.sort(key=lambda item: item[0], reverse=True)
    return tuple(path for _, path in files[:limit])


def _is_relevant_key(key: str) -> bool:
    upper = key.upper()
    return upper.startswith("TWINR_") or upper.startswith("OPENAI_") or upper in {
        "DEEPINFRA_API_KEY",
        "OPENROUTER_API_KEY",
    }


def _key_tokens(key: str) -> tuple[str, ...]:
    return tuple(token for token in re.split(r"[^A-Z0-9]+", key.upper()) if token)



def _is_secret_key(key: str) -> bool:
    tokens = _key_tokens(key)  # AUDIT-FIX(#2): Match secret markers on key tokens, not arbitrary substrings like KEYBOARD.
    return any(token in _SECRET_MARKERS for token in tokens)


def _mask_secret(value: str | None) -> str:
    if not value:
        return "Not configured"
    if len(value) <= 8:
        return "Configured"
    return f"{value[:4]}…{value[-4:]}"


def _sanitize_limit(value: int, *, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, min(parsed, maximum))


def _collect_or_default(
    label: str,
    collector: Callable[[], T],
    default: T,
    errors: dict[str, str],
) -> T:
    try:
        return collector()
    except Exception as exc:
        errors[label] = _format_exception(exc)
        return default


def _format_exception(exc: BaseException) -> str:
    detail = str(exc).strip() or repr(exc)
    return f"{exc.__class__.__name__}: {detail}"


def _create_temp_bundle_path(root: Path) -> Path:
    fd, temp_path = tempfile.mkstemp(
        prefix=".tmp-twinr-support-",
        suffix=".zip",
        dir=str(root),
    )  # AUDIT-FIX(#3): Secure temp-file creation avoids write races while the ZIP is being assembled.
    os.close(fd)
    return Path(temp_path)


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _read_text_file(path: Path) -> tuple[str | None, str | None]:
    raw_bytes, error = _read_bytes_file(path)
    if raw_bytes is None:
        return None, error
    try:
        return raw_bytes.decode("utf-8"), None
    except UnicodeDecodeError as exc:
        return None, _format_exception(exc)


def _read_bytes_file(path: Path, *, max_bytes: int | None = None) -> tuple[bytes | None, str | None]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)  # AUDIT-FIX(#1): Refuse the final path component if it is a symlink on platforms that support O_NOFOLLOW.
    fd = -1
    try:
        fd = os.open(path, flags)
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            return None, f"Refused non-regular file: {path}"
        if max_bytes is not None and file_stat.st_size > max_bytes:
            return None, f"File exceeds size limit ({file_stat.st_size} > {max_bytes} bytes): {path.name}"
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            return handle.read(), None
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        return None, _format_exception(exc)
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def _parse_env_assignment(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None

    value = value.strip()
    if not value:
        return key, ""

    if value[0] in {"'", '"'} and len(value) >= 2 and value[-1] == value[0]:
        return key, value[1:-1]

    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()

    return key, value


def _redact_env_value(key: str, value: str) -> str:
    if _is_secret_key(key):
        return _mask_secret(value)

    masked_url = _mask_url_credentials(value)
    if masked_url != value:
        return masked_url

    if _looks_like_secret_value(value):
        return _mask_secret(value)

    return value


def _looks_like_secret_value(value: str) -> bool:
    stripped = value.strip()
    if len(stripped) < 16 or any(character.isspace() for character in stripped):
        return False
    return stripped.startswith(
        (
            "sk-",
            "rk-",
            "rk_",
            "srk_",
            "sess-",
            "ghp_",
            "glpat-",
            "opk_",
        )
    )


def _mask_url_credentials(value: str) -> str:
    parts = urlsplit(value)
    if not parts.scheme or not parts.netloc:
        return value
    if parts.username is None and parts.password is None:
        return value

    hostname = parts.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    netloc = f"Configured@{netloc}" if netloc else "Configured"
    masked_parts = SplitResult(parts.scheme, netloc, parts.path, parts.query, parts.fragment)
    return urlunsplit(masked_parts)


def _redact_nested_payload(payload: object | None) -> object | None:
    value = _coerce_for_json(payload)
    if isinstance(value, dict):
        redacted: dict[str, object] = {}
        for key, nested_value in value.items():
            key_text = str(key)
            if key_text.upper() in _RUNTIME_SNAPSHOT_DROP_KEYS_UPPER:
                continue
            if _is_secret_key(key_text):
                redacted[key_text] = _mask_secret(_to_text(nested_value))  # AUDIT-FIX(#2): Mask nested secret-bearing values instead of exporting them raw.
                continue
            if _is_runtime_private_key(key_text):
                redacted[key_text] = "[REDACTED]"  # AUDIT-FIX(#2): Explicitly hide high-risk PII and biometric payloads from support bundles.
                continue
            redacted[key_text] = _redact_nested_payload(nested_value)
        return redacted
    if isinstance(value, list):
        return [_redact_nested_payload(item) for item in value]
    return value


def _is_runtime_private_key(key: str) -> bool:
    upper = key.upper()
    tokens = _key_tokens(key)
    for marker in _RUNTIME_PRIVATE_KEY_MARKERS:
        marker_upper = marker.upper()
        if "_" in marker and marker_upper in upper:
            return True
        if "_" not in marker and marker_upper in tokens:
            return True
    return False


def _to_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _event_level(entry: object) -> str:
    normalized = _coerce_for_json(entry)
    if isinstance(normalized, dict):
        return str(normalized.get("level", "")).lower()
    return ""


def _json_text(payload: object) -> str:
    return json.dumps(
        _coerce_for_json(payload),
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ) + "\n"


def _is_dataclass_instance(value: object) -> TypeGuard[Any]:
    """Return whether ``value`` is a dataclass instance, not a dataclass type."""

    return is_dataclass(value) and not isinstance(value, type)


def _coerce_for_json(value: object) -> object:
    try:
        to_dict = getattr(value, "to_dict", None)
    except Exception:
        to_dict = None
    if callable(to_dict):
        try:
            return _coerce_for_json(to_dict())
        except Exception:
            return str(value)
    if _is_dataclass_instance(value):
        try:
            return _coerce_for_json(asdict(value))
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(key): _coerce_for_json(item) for key, item in value.items()}  # AUDIT-FIX(#6): Normalize mapping keys to strings so sort_keys cannot crash on mixed key types.
    if isinstance(value, list):
        return [_coerce_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_coerce_for_json(item) for item in value]
    if isinstance(value, set):
        return [_coerce_for_json(item) for item in sorted(value, key=str)]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"__type__": "bytes", "length": len(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sanitize_archive_component(name: str) -> str:
    sanitized = name.replace("/", "_").replace("\\", "_").strip()
    sanitized = "".join(character if character.isprintable() and character not in {":", "\n", "\r", "\t"} else "_" for character in sanitized)
    return sanitized or "artifact.bin"


def _path_for_report(path: Path) -> str:
    try:
        return str(path.resolve(strict=False))
    except OSError:
        return str(path)
