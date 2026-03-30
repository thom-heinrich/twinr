"""Build redacted Twinr support bundles from local ops artifacts.

This module gathers config checks, ops events, runtime snapshots, usage data,
health snapshots, and recent self-test artifacts into a bounded ZIP export.
"""

from __future__ import annotations

# CHANGELOG: 2026-03-30
# BUG-1: Fixed raw export of unredacted events/usage/health/check/error payloads and generation errors.
# BUG-2: Fixed unbounded memory amplification from large JSON/text payloads by adding hard caps, truncation, and streaming ZIP writes.
# BUG-3: Fixed raw binary self-test exfiltration; only redacted text artifacts are exported, binary artifacts become metadata stubs by default.  # BREAKING:
# SEC-1: Added bundle-wide content redaction with per-bundle pseudonymization for PII/secrets found in free-form text, URLs, and structured fields.
# SEC-2: Hardened archive creation with private file permissions, safer path reporting, and bounded artifact ingestion.
# IMP-1: Upgraded archive writing to 2026-style streaming ZIP members with ZIP_ZSTANDARD when available and safe fallback to DEFLATE.
# IMP-2: Added manifest, redaction report, and export warnings so support can audit what was collected, redacted, truncated, or skipped.
# IMP-3: Preserved forensic utility better by using stable per-bundle pseudonyms for repeated sensitive strings instead of blind "[REDACTED]".

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import hmac
import importlib
import json
import logging
import os
import re
import secrets
import stat
import tempfile
import zipfile
from typing import Any, Callable, TypeGuard, TypeVar
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit, urlunsplit
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
    "BEARER",
    "AUTH",
    "ACCESS",
    "REFRESH",
    "JWT",
    "CLIENT_SECRET",
)

_SENSITIVE_FIELD_MARKERS = (
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "CONTACT",
    "IP",
    "TRANSCRIPT",
    "UTTERANCE",
    "VOICE",
    "AUDIO",
    "RECORDING",
    "IMAGE",
    "CAMERA",
    "FACE",
    "BIOMETRIC",
    "PROMPT",
    "COMPLETION",
    "CONVERSATION",
    "MESSAGE_HISTORY",
    "USER_INPUT",
    "TOOL_INPUT",
    "TOOL_OUTPUT",
    "RETRIEVAL",
)

_BINARY_BLOB_FIELD_MARKERS = (
    "AUDIO_BYTES",
    "AUDIO_BUFFER",
    "PCM",
    "WAV",
    "CAMERA_FRAME",
    "IMAGE_BYTES",
    "FACE_ENCODING",
    "VOICE_SAMPLE",
)

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
_RUNTIME_SNAPSHOT_DROP_KEYS_UPPER = {key.upper() for key in _RUNTIME_SNAPSHOT_DROP_KEYS}

_MAX_EVENT_LIMIT = 1000
_MAX_RECENT_USAGE_LIMIT = 100
_MAX_SELF_TEST_ARTIFACTS = 4
_MAX_SELF_TEST_ARTIFACT_BYTES = 5 * 1024 * 1024
_MAX_RUNTIME_SNAPSHOT_BYTES = 8 * 1024 * 1024
_MAX_TEXT_FILE_BYTES = 2 * 1024 * 1024
_MAX_JSON_STRING_CHARS = 4096
_MAX_COLLECTION_ITEMS = 250
_MAX_RECURSION_DEPTH = 12
_MAX_ARCHIVE_PAYLOAD_BYTES = 64 * 1024 * 1024

_ALLOWED_TEXT_ARTIFACT_SUFFIXES = {
    ".txt",
    ".log",
    ".json",
    ".jsonl",
    ".ndjson",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".csv",
    ".md",
}

_SECRET_QUERY_PARAM_MARKERS = {
    "api_key",
    "apikey",
    "token",
    "access_token",
    "refresh_token",
    "client_secret",
    "password",
    "passwd",
    "passcode",
    "key",
    "secret",
    "sig",
    "signature",
    "session",
    "cookie",
    "auth",
    "authorization",
    "jwt",
    "dsn",
}

_SECRET_VALUE_PREFIXES = (
    "sk-",
    "rk-",
    "rk_",
    "srk_",
    "sess-",
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "glpat-",
    "opk_",
    "xoxb-",
    "xoxp-",
    "xapp-",
    "ya29.",
)

_EMAIL_RE = re.compile(r"(?<![\w.+-])([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})(?![\w.-])", re.IGNORECASE)
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9._-]{10,}\.[A-Za-z0-9._-]{10,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+([A-Za-z0-9._=-]{16,})", re.IGNORECASE)
_LONG_SECRET_RE = re.compile(r"\b(?:sk|rk|sess|gh[pous]|glpat|opk|xox[abps])[-_A-Za-z0-9]{12,}\b", re.IGNORECASE)
_PEM_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ][A-Z0-9 ]+-----.*?-----END [A-Z0-9 ][A-Z0-9 ]+-----",
    re.DOTALL,
)
_URL_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s<>'\"]+", re.IGNORECASE)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

T = TypeVar("T")

_LOGGER = logging.getLogger(__name__)

_ERROR_LEVELS = {"error", "critical", "fatal"}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


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


@dataclass(slots=True)
class RedactionStats:
    secret_values_redacted: int = 0
    pii_values_pseudonymized: int = 0
    strings_truncated: int = 0
    collections_truncated: int = 0
    fields_dropped: int = 0
    artifact_text_exports: int = 0
    artifact_metadata_exports: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "secret_values_redacted": self.secret_values_redacted,
            "pii_values_pseudonymized": self.pii_values_pseudonymized,
            "strings_truncated": self.strings_truncated,
            "collections_truncated": self.collections_truncated,
            "fields_dropped": self.fields_dropped,
            "artifact_text_exports": self.artifact_text_exports,
            "artifact_metadata_exports": self.artifact_metadata_exports,
            "warning_count": len(self.warnings),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class ArchiveEntryRecord:
    archive_name: str
    content_type: str
    sha256: str
    uncompressed_bytes: int


class _BundleRedactor:
    def __init__(self, *, pseudonym_key: bytes, stats: RedactionStats) -> None:
        self._pseudonym_key = pseudonym_key
        self.stats = stats

    def redact_env_value(self, key: str, value: str) -> str:
        if _is_secret_key(key):
            return self._secret_placeholder(value)
        masked_url = _mask_url_credentials(value, redactor=self)
        if masked_url != value:
            return masked_url
        if _looks_like_secret_value(value):
            return self._secret_placeholder(value)
        return self.redact_text(value, key_hint=key)

    def redact_payload(
        self,
        payload: object | None,
        *,
        key_hint: str | None = None,
        drop_runtime_snapshot_keys: bool = False,
        depth: int = 0,
    ) -> object | None:
        value = _coerce_for_json(payload)

        if depth >= _MAX_RECURSION_DEPTH:
            self.stats.collections_truncated += 1
            return {
                "__truncated__": "max_depth",
                "__type__": type(value).__name__,
            }

        if isinstance(value, dict):
            redacted: dict[str, object] = {}
            item_count = 0
            for raw_key, nested_value in value.items():
                key_text = str(raw_key)
                item_count += 1
                if item_count > _MAX_COLLECTION_ITEMS:
                    self.stats.collections_truncated += 1
                    redacted["__truncated__"] = f"max_items:{_MAX_COLLECTION_ITEMS}"
                    break

                upper_key = key_text.upper()
                if drop_runtime_snapshot_keys and upper_key in _RUNTIME_SNAPSHOT_DROP_KEYS_UPPER:
                    self.stats.fields_dropped += 1
                    continue

                if _is_secret_key(key_text):
                    redacted[key_text] = self._secret_placeholder(_to_text(nested_value))
                    continue

                if _is_binary_blob_key(key_text):
                    self.stats.fields_dropped += 1
                    redacted[key_text] = {"__redacted__": "binary_blob"}
                    continue

                if _is_sensitive_field_key(key_text):
                    redacted[key_text] = self._redact_sensitive_field(nested_value, key_text)
                    continue

                redacted[key_text] = self.redact_payload(
                    nested_value,
                    key_hint=key_text,
                    drop_runtime_snapshot_keys=False,
                    depth=depth + 1,
                )
            return redacted

        if isinstance(value, list):
            redacted_items: list[object] = []
            for index, item in enumerate(value):
                if index >= _MAX_COLLECTION_ITEMS:
                    self.stats.collections_truncated += 1
                    redacted_items.append({"__truncated__": f"max_items:{_MAX_COLLECTION_ITEMS}"})
                    break
                redacted_items.append(
                    self.redact_payload(
                        item,
                        key_hint=key_hint,
                        drop_runtime_snapshot_keys=False,
                        depth=depth + 1,
                    )
                )
            return redacted_items

        if isinstance(value, str):
            if key_hint and _is_secret_key(key_hint):
                return self._secret_placeholder(value)
            if key_hint and _is_sensitive_field_key(key_hint):
                return self._sensitive_placeholder(key_hint, value)
            return self.redact_text(value, key_hint=key_hint)

        return value

    def redact_text(self, value: str, *, key_hint: str | None = None) -> str:
        text = value
        if key_hint and _is_secret_key(key_hint):
            return self._secret_placeholder(text)
        if key_hint and _is_sensitive_field_key(key_hint):
            return self._sensitive_placeholder(key_hint, text)

        text = _URL_RE.sub(
            lambda match: _mask_url_query_secrets(
                _mask_url_credentials(match.group(0), redactor=self),
                redactor=self,
            ),
            text,
        )
        text = _EMAIL_RE.sub(lambda match: self._pseudonym("EMAIL", match.group(1)), text)
        text = _IPV4_RE.sub(lambda match: self._pseudonym("IP", match.group(0)), text)
        text = _JWT_RE.sub(lambda match: self._secret_placeholder(match.group(0)), text)
        text = _BEARER_RE.sub(lambda match: f"Bearer {self._secret_placeholder(match.group(1))}", text)
        text = _LONG_SECRET_RE.sub(lambda match: self._secret_placeholder(match.group(0)), text)
        text = _PEM_RE.sub(lambda match: self._secret_placeholder(match.group(0)), text)

        if self._looks_like_phone(text):
            return self._sensitive_placeholder("PHONE", text)

        if _looks_like_secret_value(text):
            return self._secret_placeholder(text)

        if len(text) > _MAX_JSON_STRING_CHARS:
            fingerprint = self._fingerprint("STRING", text)
            self.stats.strings_truncated += 1
            head = text[:2048]
            tail = text[-512:]
            text = (
                f"{head}…[TRUNCATED:{len(value)}:{fingerprint}]…{tail}"
                if len(value) > 2560
                else text[:_MAX_JSON_STRING_CHARS]
            )

        return text

    def _redact_sensitive_field(self, value: object, key_hint: str) -> object:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return self._sensitive_placeholder(key_hint, value)
        if isinstance(value, bytes):
            self.stats.fields_dropped += 1
            return {"__redacted__": "binary_blob", "length": len(value)}
        if isinstance(value, Mapping):
            self.stats.fields_dropped += 1
            return {"__redacted__": "structured_sensitive", "keys": sorted(str(key) for key in list(value.keys())[:10])}
        if isinstance(value, list):
            self.stats.fields_dropped += 1
            return {"__redacted__": "list_sensitive", "items": min(len(value), _MAX_COLLECTION_ITEMS)}
        text = _to_text(value)
        return self._sensitive_placeholder(key_hint, text)

    def _secret_placeholder(self, value: str | None) -> str:
        if not value:
            return "Not configured"
        token = self._fingerprint("SECRET", value)
        self.stats.secret_values_redacted += 1
        return f"[REDACTED:SECRET:{token}]"

    def _sensitive_placeholder(self, kind: str, value: str | None) -> str:
        if not value:
            return "[REDACTED]"
        token = self._fingerprint(kind.upper(), value)
        self.stats.pii_values_pseudonymized += 1
        return f"[REDACTED:{kind.upper()}:{token}]"

    def _pseudonym(self, kind: str, value: str) -> str:
        self.stats.pii_values_pseudonymized += 1
        return f"[{kind}:{self._fingerprint(kind, value)}]"

    def _safe_token(self, kind: str, value: str) -> str:
        self.stats.pii_values_pseudonymized += 1
        return f"redacted-{kind.lower()}-{self._fingerprint(kind, value)}"

    def _safe_secret_token(self, value: str) -> str:
        self.stats.secret_values_redacted += 1
        return f"redacted-secret-{self._fingerprint('SECRET', value)}"

    def _fingerprint(self, kind: str, value: str) -> str:
        digest = hmac.new(
            self._pseudonym_key,
            msg=f"{kind}\x1f{value}".encode("utf-8", errors="replace"),
            digestmod=hashlib.sha256,
        ).hexdigest()
        return digest[:12]

    @staticmethod
    def _looks_like_phone(text: str) -> bool:
        stripped = text.strip()
        if len(stripped) < 7 or len(stripped) > 32:
            return False
        if "@" in stripped or "/" in stripped or "." in stripped and not stripped.startswith("+"):
            return False
        digits = sum(character.isdigit() for character in stripped)
        return digits >= 7 and all(
            character.isdigit() or character in "+-() ."
            for character in stripped
        )


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
    bundle_name = f"twinr-support-{_utc_stamp()}-{uuid4().hex[:8]}.zip"
    bundle_path = paths.bundles_root / bundle_name
    temp_bundle_path = _create_temp_bundle_path(paths.bundles_root)

    safe_event_limit = _sanitize_limit(event_limit, default=100, maximum=_MAX_EVENT_LIMIT)
    generation_errors: dict[str, str] = {}
    generation_warnings: list[str] = []

    redaction_stats = RedactionStats()
    redactor = _BundleRedactor(
        pseudonym_key=secrets.token_bytes(32),
        stats=redaction_stats,
    )

    env_source_path = Path(env_path)
    env_values, env_error = _read_env_values(env_source_path)
    if env_error is not None:
        generation_errors["env_file"] = redactor.redact_text(env_error, key_hint="error")
    elif not env_source_path.exists():
        generation_warnings.append(f"env_file missing: {_path_for_report(env_source_path)}")
    redacted_env = redact_env_values(env_values, redactor=redactor)

    checks: list[object] = _collect_or_default(
        "config_checks",
        lambda: list(run_config_checks(config)),
        [],
        generation_errors,
    )
    summary: object
    if "config_checks" in generation_errors:
        summary = {"status": "unavailable", "reason": generation_errors["config_checks"]}
    else:
        summary = _collect_or_default(
            "config_check_summary",
            lambda: check_summary(checks),
            {"status": "unavailable"},
            generation_errors,
        )

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
        )
    errors = [entry for entry in events if _event_level(entry) in _ERROR_LEVELS][-20:]

    snapshot_path = Path(config.runtime_state_path)
    snapshot_payload_raw, snapshot_error = _read_json(
        snapshot_path,
        max_bytes=_MAX_RUNTIME_SNAPSHOT_BYTES,
    )
    if snapshot_error is not None:
        generation_errors["runtime_snapshot"] = redactor.redact_text(snapshot_error, key_hint="error")
    elif not snapshot_path.exists():
        generation_warnings.append(f"runtime_snapshot missing: {_path_for_report(snapshot_path)}")
    snapshot_payload = _redact_runtime_snapshot_payload(snapshot_payload_raw, redactor=redactor)

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
        )
        recent_usage = _collect_or_default(
            "recent_usage",
            lambda: list(usage_store.tail(limit=_MAX_RECENT_USAGE_LIMIT)),
            [],
            generation_errors,
        )

    health_payload: object = _collect_or_default(
        "system_health",
        lambda: collect_system_health(config),
        {},
        generation_errors,
    )

    redacted_checks = redactor.redact_payload(checks)
    redacted_summary = redactor.redact_payload(summary)
    redacted_events = redactor.redact_payload(events)
    redacted_errors = redactor.redact_payload(errors)
    redacted_usage_summary = redactor.redact_payload(usage_summary)
    redacted_recent_usage = redactor.redact_payload(recent_usage)
    redacted_health_payload = redactor.redact_payload(health_payload)

    archive_entries: list[tuple[str, object, str]] = [
        (
            "summary.json",
            {
                "schema_version": 2,
                "created_at": created_at,
                "bundle_name": bundle_name,
                "env_path": _path_for_report(env_source_path),
                "runtime_state_path": _path_for_report(snapshot_path),
                "event_limit_requested": event_limit,
                "event_limit_used": safe_event_limit,
                "check_summary": redacted_summary,
            },
            "application/json",
        ),
        ("redacted_env.json", redacted_env, "application/json"),
        ("config_checks.json", redacted_checks, "application/json"),
        ("events.json", redacted_events, "application/json"),
        ("errors.json", redacted_errors, "application/json"),
        ("system_health.json", redacted_health_payload, "application/json"),
        ("usage_summary.json", redacted_usage_summary, "application/json"),
        ("recent_usage.json", redacted_recent_usage, "application/json"),
    ]
    if snapshot_payload is not None:
        archive_entries.append(("runtime_snapshot.json", snapshot_payload, "application/json"))

    self_test_entries, self_test_warnings = _collect_self_test_entries(
        root=paths.self_tests_root,
        redactor=redactor,
    )
    generation_warnings.extend(self_test_warnings)

    if generation_errors:
        archive_entries.append(
            (
                "generation_errors.json",
                redactor.redact_payload(generation_errors),
                "application/json",
            )
        )

    compression, compresslevel, compression_name = _zip_compression_settings()
    includes: list[str] = []
    manifest_records: list[ArchiveEntryRecord] = []
    total_uncompressed_bytes = 0

    try:
        with zipfile.ZipFile(
            temp_bundle_path,
            "w",
            compression=compression,
            compresslevel=compresslevel,
            allowZip64=True,
        ) as archive:
            for archive_name, payload, content_type in archive_entries:
                archive_name = _sanitize_archive_name(archive_name)
                bytes_written = _estimate_json_size(payload)
                if total_uncompressed_bytes + bytes_written > _MAX_ARCHIVE_PAYLOAD_BYTES:
                    warning = (
                        f"Skipped {archive_name}: bundle payload budget exceeded "
                        f"({_MAX_ARCHIVE_PAYLOAD_BYTES} bytes)"
                    )
                    generation_warnings.append(warning)
                    redaction_stats.warnings.append(warning)
                    continue
                record = _write_json_entry(archive, archive_name, payload, content_type=content_type)
                manifest_records.append(record)
                includes.append(archive_name)
                total_uncompressed_bytes += record.uncompressed_bytes

            for archive_name, payload_bytes, content_type in self_test_entries:
                archive_name = _sanitize_archive_name(archive_name)
                if total_uncompressed_bytes + len(payload_bytes) > _MAX_ARCHIVE_PAYLOAD_BYTES:
                    warning = (
                        f"Skipped {archive_name}: bundle payload budget exceeded "
                        f"({_MAX_ARCHIVE_PAYLOAD_BYTES} bytes)"
                    )
                    generation_warnings.append(warning)
                    redaction_stats.warnings.append(warning)
                    continue
                record = _write_bytes_entry(archive, archive_name, payload_bytes, content_type=content_type)
                manifest_records.append(record)
                includes.append(archive_name)
                total_uncompressed_bytes += record.uncompressed_bytes

            if generation_warnings:
                warnings_record = _write_json_entry(
                    archive,
                    "generation_warnings.json",
                    redactor.redact_payload(generation_warnings),
                    content_type="application/json",
                )
                manifest_records.append(warnings_record)
                includes.append(warnings_record.archive_name)
                total_uncompressed_bytes += warnings_record.uncompressed_bytes

            redaction_report_payload = {
                "correlation_scope": "per_bundle",
                "stats": redaction_stats.to_dict(),
                "compression": compression_name,
                "payload_budget_bytes": _MAX_ARCHIVE_PAYLOAD_BYTES,
                "runtime_snapshot_cap_bytes": _MAX_RUNTIME_SNAPSHOT_BYTES,
                "text_artifact_cap_bytes": _MAX_TEXT_FILE_BYTES,
                "binary_artifact_cap_bytes": _MAX_SELF_TEST_ARTIFACT_BYTES,
            }
            redaction_report = _write_json_entry(
                archive,
                "redaction_report.json",
                redaction_report_payload,
                content_type="application/json",
            )
            manifest_records.append(redaction_report)
            includes.append(redaction_report.archive_name)
            total_uncompressed_bytes += redaction_report.uncompressed_bytes

            manifest_payload = {
                "schema_version": 2,
                "bundle_name": bundle_name,
                "created_at": created_at,
                "compression": compression_name,
                "entry_count": len(manifest_records),
                "entries": [asdict(record) for record in manifest_records],
            }
            manifest = _write_json_entry(
                archive,
                "bundle_manifest.json",
                manifest_payload,
                content_type="application/json",
            )
            manifest_records.append(manifest)
            includes.append(manifest.archive_name)
            total_uncompressed_bytes += manifest.uncompressed_bytes

        os.replace(temp_bundle_path, bundle_path)
        _chmod_private(bundle_path)
    except Exception:
        _unlink_quietly(temp_bundle_path)
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
            )

    return SupportBundleInfo(
        bundle_name=bundle_name,
        bundle_path=str(bundle_path),
        created_at=created_at,
        file_count=len(includes),
        includes=tuple(includes),
    )


def redact_env_values(
    values: dict[str, str],
    *,
    redactor: _BundleRedactor | None = None,
) -> dict[str, str]:
    """Redact secret-like environment values before export."""

    bundle_redactor = redactor or _BundleRedactor(
        pseudonym_key=secrets.token_bytes(32),
        stats=RedactionStats(),
    )

    redacted: dict[str, str] = {}
    for key, value in sorted(values.items()):
        if not _is_relevant_key(key):
            continue
        redacted[key] = bundle_redactor.redact_env_value(key, value)
    return redacted


def _read_env_values(path: Path) -> tuple[dict[str, str], str | None]:
    text, error = _read_text_file(path, max_bytes=_MAX_TEXT_FILE_BYTES)
    if text is None:
        return {}, error
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        parsed = _parse_env_assignment(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        values[key] = value
    return values, None


def _read_json(path: Path, *, max_bytes: int) -> tuple[object | None, str | None]:
    text, error = _read_text_file(path, max_bytes=max_bytes)
    if text is None:
        return None, error
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, _format_exception(exc)


def _redact_runtime_snapshot_payload(
    payload: object | None,
    *,
    redactor: _BundleRedactor,
) -> object | None:
    return redactor.redact_payload(
        _unwrap_runtime_snapshot_payload(payload),
        drop_runtime_snapshot_keys=True,
    )


def _unwrap_runtime_snapshot_payload(payload: object | None) -> object | None:
    if not isinstance(payload, Mapping):
        return payload
    nested_payload = payload.get("payload")
    if payload.get("format") == "twinr.runtime_snapshot" and isinstance(nested_payload, Mapping):
        return nested_payload
    return payload


def _collect_self_test_entries(
    *,
    root: Path,
    redactor: _BundleRedactor,
) -> tuple[list[tuple[str, bytes, str]], list[str]]:
    entries: list[tuple[str, bytes, str]] = []
    warnings: list[str] = []

    for artifact_path in _latest_self_test_artifacts(root, limit=_MAX_SELF_TEST_ARTIFACTS):
        artifact_name = _sanitize_archive_component(artifact_path.name)
        artifact_bytes, artifact_error = _read_bytes_file(
            artifact_path,
            max_bytes=_MAX_SELF_TEST_ARTIFACT_BYTES,
        )
        if artifact_bytes is None:
            if artifact_error is not None:
                warnings.append(f"{artifact_name}: {artifact_error}")
                redactor.stats.warnings.append(f"{artifact_name}: {artifact_error}")
            continue

        suffix = artifact_path.suffix.lower()
        if suffix in _ALLOWED_TEXT_ARTIFACT_SUFFIXES:
            text = _decode_text(artifact_bytes)
            if text is None:
                entries.append(
                    (
                        f"self_tests/{artifact_name}.metadata.json",
                        _json_bytes(
                            {
                                "artifact_name": artifact_path.name,
                                "reason": "text_decode_failed",
                                "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                                "size_bytes": len(artifact_bytes),
                            }
                        ),
                        "application/json",
                    )
                )
                redactor.stats.artifact_metadata_exports += 1
                continue

            if suffix in {".json", ".jsonl", ".ndjson"}:
                redacted_bytes = _redact_text_artifact_jsonish(
                    artifact_name=artifact_path.name,
                    text=text,
                    redactor=redactor,
                    suffix=suffix,
                )
            else:
                redacted_text = redactor.redact_text(text, key_hint="artifact")
                redacted_bytes = redacted_text.encode("utf-8")

            entries.append((f"self_tests/{artifact_name}", redacted_bytes, "text/plain"))
            redactor.stats.artifact_text_exports += 1
            continue

        # BREAKING: binary self-test artifacts are no longer exported raw by default.
        metadata_payload = {
            "artifact_name": artifact_path.name,
            "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "size_bytes": len(artifact_bytes),
            "content_type": "application/octet-stream",
            "exported": False,
            "reason": "binary_artifact_redacted",
        }
        entries.append(
            (
                f"self_tests/{artifact_name}.metadata.json",
                _json_bytes(metadata_payload),
                "application/json",
            )
        )
        redactor.stats.artifact_metadata_exports += 1

    return entries, warnings


def _redact_text_artifact_jsonish(
    *,
    artifact_name: str,
    text: str,
    redactor: _BundleRedactor,
    suffix: str,
) -> bytes:
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return redactor.redact_text(text, key_hint="artifact_json").encode("utf-8")
        return _json_bytes(redactor.redact_payload(payload))

    lines = text.splitlines()
    redacted_lines: list[str] = []
    for index, line in enumerate(lines):
        if index >= _MAX_COLLECTION_ITEMS:
            redactor.stats.collections_truncated += 1
            redacted_lines.append(
                json.dumps({"__truncated__": f"max_lines:{_MAX_COLLECTION_ITEMS}"}, ensure_ascii=False)
            )
            break
        if not line.strip():
            redacted_lines.append("")
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            redacted_lines.append(redactor.redact_text(line, key_hint="artifact_jsonl"))
            continue
        redacted_lines.append(json.dumps(redactor.redact_payload(payload), ensure_ascii=False, sort_keys=True))
    return ("\n".join(redacted_lines) + ("\n" if redacted_lines else "")).encode("utf-8")


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
                    continue
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


def _key_matches_markers(key: str, markers: Iterable[str]) -> bool:
    upper = key.upper()
    tokens = set(_key_tokens(key))
    for marker in markers:
        marker_upper = marker.upper()
        if "_" in marker_upper:
            if marker_upper in upper:
                return True
        elif marker_upper in tokens:
            return True
    return False


def _is_secret_key(key: str) -> bool:
    return _key_matches_markers(key, _SECRET_MARKERS)


def _is_sensitive_field_key(key: str) -> bool:
    return _key_matches_markers(key, _SENSITIVE_FIELD_MARKERS)


def _is_binary_blob_key(key: str) -> bool:
    return _key_matches_markers(key, _BINARY_BLOB_FIELD_MARKERS)


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
    )
    try:
        os.fchmod(fd, 0o600)
    except (AttributeError, OSError):
        pass
    finally:
        os.close(fd)
    return Path(temp_path)


def _chmod_private(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError:
        return


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _read_text_file(path: Path, *, max_bytes: int) -> tuple[str | None, str | None]:
    raw_bytes, error = _read_bytes_file(path, max_bytes=max_bytes)
    if raw_bytes is None:
        return None, error
    text = _decode_text(raw_bytes)
    if text is None:
        return None, "UnicodeDecodeError: file is not valid UTF-8 text"
    return text, None


def _decode_text(raw_bytes: bytes) -> str | None:
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _read_bytes_file(path: Path, *, max_bytes: int | None = None) -> tuple[bytes | None, str | None]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = -1
    try:
        fd = os.open(path, flags)
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            return None, f"Refused non-regular file: {path.name}"
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


def _looks_like_secret_value(value: str) -> bool:
    stripped = value.strip()
    if len(stripped) < 16 or any(character.isspace() for character in stripped):
        return False
    if stripped.startswith(_SECRET_VALUE_PREFIXES):
        return True
    if _JWT_RE.fullmatch(stripped):
        return True
    if len(stripped) >= 32 and re.fullmatch(r"[A-Fa-f0-9]{32,}", stripped):
        return True
    return False


def _mask_url_credentials(value: str, *, redactor: _BundleRedactor) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
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
    user_token = redactor._safe_token("URL_USER", parts.username or "configured")
    netloc = f"{user_token}@{netloc}" if netloc else user_token
    masked_parts = SplitResult(parts.scheme, netloc, parts.path, parts.query, parts.fragment)
    return urlunsplit(masked_parts)


def _mask_url_query_secrets(value: str, *, redactor: _BundleRedactor) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if not parts.scheme or not parts.netloc or not parts.query:
        return value

    changed = False
    query_pairs: list[tuple[str, str]] = []
    for key, query_value in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() in _SECRET_QUERY_PARAM_MARKERS or _looks_like_secret_value(query_value):
            query_pairs.append((key, redactor._safe_secret_token(query_value or key)))
            changed = True
        else:
            query_pairs.append((key, redactor.redact_text(query_value, key_hint=key)))
            changed = changed or (query_pairs[-1][1] != query_value)

    if not changed:
        return value

    masked_parts = SplitResult(
        parts.scheme,
        parts.netloc,
        parts.path,
        urlencode(query_pairs, doseq=True),
        parts.fragment,
    )
    return urlunsplit(masked_parts)


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


def _json_bytes(payload: object) -> bytes:
    return (_json_text(payload)).encode("utf-8")


def _json_text(payload: object) -> str:
    return json.dumps(
        _coerce_for_json(payload),
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ) + "\n"


def _estimate_json_size(payload: object) -> int:
    encoder = json.JSONEncoder(indent=2, ensure_ascii=False, sort_keys=True)
    total = 1  # newline
    for chunk in encoder.iterencode(_coerce_for_json(payload)):
        total += len(chunk.encode("utf-8"))
    return total


def _write_json_entry(
    archive: zipfile.ZipFile,
    archive_name: str,
    payload: object,
    *,
    content_type: str,
) -> ArchiveEntryRecord:
    hasher = hashlib.sha256()
    byte_count = 0
    encoder = json.JSONEncoder(indent=2, ensure_ascii=False, sort_keys=True)
    coerced = _coerce_for_json(payload)

    with archive.open(archive_name, mode="w", force_zip64=True) as handle:
        for chunk in encoder.iterencode(coerced):
            encoded = chunk.encode("utf-8")
            handle.write(encoded)
            hasher.update(encoded)
            byte_count += len(encoded)
        handle.write(b"\n")
        hasher.update(b"\n")
        byte_count += 1

    return ArchiveEntryRecord(
        archive_name=archive_name,
        content_type=content_type,
        sha256=hasher.hexdigest(),
        uncompressed_bytes=byte_count,
    )


def _write_bytes_entry(
    archive: zipfile.ZipFile,
    archive_name: str,
    payload: bytes,
    *,
    content_type: str,
) -> ArchiveEntryRecord:
    with archive.open(archive_name, mode="w", force_zip64=True) as handle:
        handle.write(payload)
    return ArchiveEntryRecord(
        archive_name=archive_name,
        content_type=content_type,
        sha256=hashlib.sha256(payload).hexdigest(),
        uncompressed_bytes=len(payload),
    )


def _zip_compression_settings() -> tuple[int, int | None, str]:
    zstd = getattr(zipfile, "ZIP_ZSTANDARD", None)
    if zstd is not None:
        try:
            importlib.import_module("compression.zstd")
        except Exception:
            pass
        else:
            return zstd, 3, "ZIP_ZSTANDARD"
    return zipfile.ZIP_DEFLATED, 6, "ZIP_DEFLATED"


def _is_dataclass_instance(value: object) -> TypeGuard[Any]:
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
        return {str(key): _coerce_for_json(item) for key, item in value.items()}
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


def _sanitize_archive_name(name: str) -> str:
    parts = [
        _sanitize_archive_component(part)
        for part in re.split(r"[\\/]+", name)
        if part and part not in {".", ".."}
    ]
    return "/".join(parts) or "artifact.bin"


def _sanitize_archive_component(name: str) -> str:
    sanitized = name.replace("/", "_").replace("\\", "_").strip()
    sanitized = "".join(
        character if character.isprintable() and character not in {":", "\n", "\r", "\t"} else "_"
        for character in sanitized
    )
    return sanitized or "artifact.bin"


def _path_for_report(path: Path) -> str:
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        resolved = path
    home = Path.home()
    try:
        relative_to_home = resolved.relative_to(home)
    except ValueError:
        relative_to_home = None
    if relative_to_home is not None:
        return f"~/{relative_to_home}"
    parts = resolved.parts
    if len(parts) <= 3:
        return str(resolved)
    return str(Path("…") / Path(*parts[-3:]))