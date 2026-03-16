"""Persist normalized model-usage telemetry for Twinr runtime paths.

This module records request metadata and token usage from provider calls and
offers bounded summaries for dashboards, support bundles, and runtime analysis.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import fcntl
import json
import logging
import os
import stat

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.paths import resolve_ops_paths, resolve_ops_paths_for_config


LOGGER = logging.getLogger(__name__)
_PRIVATE_DIR_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_SENSITIVE_METADATA_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "password",
    "secret",
    "session",
    "token",
)


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_str(value: object) -> str:
    try:
        return str(value)
    except Exception:  # pragma: no cover - ultra-defensive for foreign SDK objects
        return repr(type(value))


def _normalize_optional_text(value: object, *, default: str | None = None) -> str | None:
    text = _safe_str(value).strip() if value is not None else ""
    return text or default


def _field_value(source: object, key: str) -> object | None:
    if source is None:
        return None
    if isinstance(source, Mapping):  # AUDIT-FIX(#5): Support generic mappings, not only plain dict.
        return source.get(key)
    try:
        return getattr(source, key)  # AUDIT-FIX(#5): Guard foreign SDK/property access from bubbling exceptions.
    except AttributeError:
        return None
    except Exception:
        LOGGER.debug("Failed to read attribute %r from %r", key, type(source), exc_info=True)
        return None


def _first_non_none(*values: object) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):  # AUDIT-FIX(#6): Reject bool so token counters never silently become 0/1.
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    if isinstance(value, float):
        if not value.is_integer():  # AUDIT-FIX(#6): Reject lossy float-to-int truncation.
            return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    if coerced < 0:  # AUDIT-FIX(#6): Token counts are never valid when negative.
        return None
    return coerced


def _coerce_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _parse_iso_datetime(value: str | None) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        normalized = raw.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_sensitive_metadata_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_METADATA_KEY_PARTS)


def _sanitize_metadata(metadata: object | None) -> dict[str, str] | None:
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):  # AUDIT-FIX(#2): Invalid metadata must not crash append().
        LOGGER.debug("Ignoring non-mapping usage metadata of type %r", type(metadata))
        return None
    sanitized: dict[str, str] = {}
    for key, value in metadata.items():
        normalized_key = _normalize_optional_text(key)
        if not normalized_key:
            continue
        if _is_sensitive_metadata_key(normalized_key):  # AUDIT-FIX(#8): Prevent credential-like metadata from hitting disk verbatim.
            sanitized[normalized_key] = "[REDACTED]"
            continue
        normalized_value = _normalize_optional_text(value)
        if normalized_value:
            sanitized[normalized_key] = normalized_value
    return sanitized or None


def _assert_no_symlink_components(path: Path) -> None:
    for component in (path, *path.parents):
        try:
            if component.is_symlink():  # AUDIT-FIX(#1): Reject symlinked file or parent components to avoid path redirection.
                raise ValueError(f"Unsafe usage store path component: {component}")
        except OSError as exc:
            raise ValueError(f"Unable to validate usage store path {path}: {exc}") from exc


def _assert_regular_file_target(path: Path) -> None:
    try:
        target_stat = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(f"Unable to stat usage store path {path}: {exc}") from exc
    if not stat.S_ISREG(target_stat.st_mode):  # AUDIT-FIX(#1): Refuse special files such as FIFOs/devices/sockets for the JSONL store.
        raise ValueError(f"Usage store path must be a regular file: {path}")


def _normalize_store_path(path: str | Path) -> Path:
    raw_path = Path(path).expanduser()
    if not raw_path.is_absolute():
        raw_path = Path.cwd() / raw_path
    _assert_no_symlink_components(raw_path)  # AUDIT-FIX(#1): Validate before normalization to catch symlink components.
    normalized = Path(os.path.normpath(os.fspath(raw_path)))
    _assert_no_symlink_components(normalized)  # AUDIT-FIX(#1): Re-check normalized path used for all later I/O.
    _assert_regular_file_target(normalized)  # AUDIT-FIX(#1): Existing targets must be regular files, not FIFOs/devices/directories.
    return normalized


def _ensure_parent_dir(path: Path) -> None:
    _assert_no_symlink_components(path.parent)  # AUDIT-FIX(#1): Re-validate before mkdir to reduce TOCTOU window.
    path.parent.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIR_MODE)  # AUDIT-FIX(#8): New directories default to private mode.
    _assert_no_symlink_components(path.parent)  # AUDIT-FIX(#1): Re-check after mkdir in case the path changed concurrently.
    _assert_regular_file_target(path)  # AUDIT-FIX(#1): Keep the append target constrained to a normal file.


def _open_locked_append_text(path: Path):
    _ensure_parent_dir(path)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW  # AUDIT-FIX(#1): Refuse to follow a symlink on the final file path.
    fd = os.open(path, flags, _PRIVATE_FILE_MODE)
    try:
        try:
            os.fchmod(fd, _PRIVATE_FILE_MODE)  # AUDIT-FIX(#8): Tighten file permissions even when umask is permissive.
        except OSError:
            LOGGER.debug("Failed to tighten permissions on %s", path, exc_info=True)
        fcntl.flock(fd, fcntl.LOCK_EX)  # AUDIT-FIX(#1): Serialize writers/readers that cooperate via flock.
        return os.fdopen(fd, "a", encoding="utf-8", closefd=True)
    except Exception:
        os.close(fd)
        raise


def _open_locked_read_text(path: Path):
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW  # AUDIT-FIX(#1): Refuse to read through a symlinked log file.
    fd = os.open(path, flags)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)  # AUDIT-FIX(#1): Coordinate readers with writers to avoid torn JSON lines.
        return os.fdopen(fd, "r", encoding="utf-8", errors="replace", closefd=True)
    except Exception:
        os.close(fd)
        raise


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Represent normalized token usage extracted from a provider response."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    audio_input_tokens: int | None = None
    audio_output_tokens: int | None = None

    @property
    def has_values(self) -> bool:
        return any(
            value is not None
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.total_tokens,
                self.cached_input_tokens,
                self.reasoning_tokens,
                self.audio_input_tokens,
                self.audio_output_tokens,
            )
        )

    @property
    def total_tokens_estimate(self) -> int | None:
        if self.total_tokens is not None:
            return self.total_tokens
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return int(self.input_tokens or 0) + int(self.output_tokens or 0)

    def to_dict(self) -> dict[str, int]:
        payload = {key: value for key, value in asdict(self).items() if value is not None}
        estimated_total = self.total_tokens_estimate
        if "total_tokens" not in payload and estimated_total is not None:
            payload["total_tokens"] = estimated_total
        return payload


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """Represent one persisted Twinr usage telemetry record."""

    created_at: str
    source: str
    request_kind: str
    model: str | None = None
    response_id: str | None = None
    request_id: str | None = None
    used_web_search: bool | None = None
    token_usage: TokenUsage | None = None
    metadata: dict[str, str] | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.token_usage is None:
            return None
        return self.token_usage.total_tokens_estimate

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "created_at": self.created_at,
            "source": self.source,
            "request_kind": self.request_kind,
            "model": self.model,
            "response_id": self.response_id,
            "request_id": self.request_id,
            "used_web_search": self.used_web_search,
            "metadata": dict(self.metadata or {}),
        }
        if self.token_usage is not None and self.token_usage.has_values:
            payload["token_usage"] = self.token_usage.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "UsageRecord":
        """Build a normalized usage record from persisted JSON data."""

        usage_payload = payload.get("token_usage")
        token_usage = None
        if isinstance(usage_payload, Mapping):
            token_usage = TokenUsage(
                input_tokens=_coerce_int(usage_payload.get("input_tokens")),
                output_tokens=_coerce_int(usage_payload.get("output_tokens")),
                total_tokens=_coerce_int(usage_payload.get("total_tokens")),
                cached_input_tokens=_coerce_int(usage_payload.get("cached_input_tokens")),
                reasoning_tokens=_coerce_int(usage_payload.get("reasoning_tokens")),
                audio_input_tokens=_coerce_int(usage_payload.get("audio_input_tokens")),
                audio_output_tokens=_coerce_int(usage_payload.get("audio_output_tokens")),
            )
        return cls(
            created_at=_normalize_optional_text(payload.get("created_at")) or "",
            source=_normalize_optional_text(payload.get("source"), default="runtime") or "runtime",  # AUDIT-FIX(#2): Strip and default malformed source values.
            request_kind=_normalize_optional_text(payload.get("request_kind"), default="unknown") or "unknown",  # AUDIT-FIX(#2): Strip and default malformed request kinds.
            model=_normalize_optional_text(payload.get("model")),
            response_id=_normalize_optional_text(payload.get("response_id")),
            request_id=_normalize_optional_text(payload.get("request_id")),
            used_web_search=_coerce_bool(payload.get("used_web_search")),  # AUDIT-FIX(#7): Parse booleans explicitly; "false" must stay False.
            token_usage=token_usage if token_usage is not None and token_usage.has_values else None,
            metadata=_sanitize_metadata(payload.get("metadata")),
        )


@dataclass(frozen=True, slots=True)
class UsageSummary:
    """Aggregate persisted usage records for dashboards and support exports."""

    requests_total: int = 0
    requests_with_token_data: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    latest_model: str | None = None
    latest_request_kind: str | None = None
    latest_created_at: str | None = None
    by_kind: dict[str, int] | None = None
    by_model: dict[str, int] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "requests_total": self.requests_total,
            "requests_with_token_data": self.requests_with_token_data,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "latest_model": self.latest_model,
            "latest_request_kind": self.latest_request_kind,
            "latest_created_at": self.latest_created_at,
            "by_kind": dict(self.by_kind or {}),
            "by_model": dict(self.by_model or {}),
        }


class TwinrUsageStore:
    """Persist and summarize Twinr usage telemetry in a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = _normalize_store_path(path)  # AUDIT-FIX(#1): Normalize once and reject unsafe paths up front.

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "TwinrUsageStore":
        """Build a usage store rooted in Twinr's configured project."""

        return cls(resolve_ops_paths_for_config(config).usage_path)

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "TwinrUsageStore":
        """Build a usage store rooted in a given Twinr project tree."""

        return cls(resolve_ops_paths(project_root).usage_path)

    def _iter_records(self) -> Iterator[UsageRecord]:
        if not self.path.exists():
            return
        try:
            with _open_locked_read_text(self.path) as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        LOGGER.warning(
                            "Skipping invalid usage JSON line %s in %s",
                            line_number,
                            self.path,
                        )  # AUDIT-FIX(#3): Corrupt lines must not crash reads; recover best-effort.
                        continue
                    if isinstance(parsed, Mapping):
                        yield UsageRecord.from_dict(dict(parsed))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            LOGGER.warning("Failed to read usage records from %s: %s", self.path, exc)  # AUDIT-FIX(#3): Storage failures should degrade gracefully.
            return

    def append(
        self,
        *,
        source: str,
        request_kind: str,
        model: str | None = None,
        response_id: str | None = None,
        request_id: str | None = None,
        used_web_search: bool | None = None,
        token_usage: TokenUsage | None = None,
        metadata: dict[str, object] | None = None,
    ) -> UsageRecord:
        """Append one normalized usage record and return the stored entry."""

        record = UsageRecord(
            created_at=_utc_now_iso_z(),
            source=_normalize_optional_text(source, default="runtime") or "runtime",  # AUDIT-FIX(#2): Never trust runtime callers to respect type hints.
            request_kind=_normalize_optional_text(request_kind, default="unknown") or "unknown",  # AUDIT-FIX(#2): Normalize request kind instead of calling .strip() blindly.
            model=_normalize_optional_text(model),
            response_id=_normalize_optional_text(response_id),
            request_id=_normalize_optional_text(request_id),
            used_web_search=_coerce_bool(used_web_search),  # AUDIT-FIX(#7): Normalize truthy/falsy values consistently at write-time too.
            token_usage=token_usage if token_usage is not None and token_usage.has_values else None,
            metadata=_sanitize_metadata(metadata),  # AUDIT-FIX(#2): Invalid metadata must not break the request path.
        )
        try:
            with _open_locked_append_text(self.path) as handle:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
        except (OSError, ValueError) as exc:
            LOGGER.warning("Failed to append usage record to %s: %s", self.path, exc)  # AUDIT-FIX(#3): Usage logging failure must not take down the voice flow.
        return record

    def tail(self, *, limit: int = 100) -> list[UsageRecord]:
        """Return the most recent usage records in chronological order."""

        if limit <= 0:
            return []
        records: deque[UsageRecord] = deque(maxlen=limit)  # AUDIT-FIX(#3): Stream the file instead of loading it all into memory.
        for record in self._iter_records():
            records.append(record)
        return list(records)

    def summary(self, *, within_hours: int | None = None) -> UsageSummary:
        """Summarize stored usage records for the optional recent window."""

        cutoff = None
        if within_hours is not None and within_hours > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=within_hours)

        by_kind: dict[str, int] = {}
        by_model: dict[str, int] = {}
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        cached_input_tokens = 0
        reasoning_tokens = 0
        requests_total = 0
        requests_with_token_data = 0
        latest: UsageRecord | None = None
        latest_dt: datetime | None = None

        for record in self._iter_records():  # AUDIT-FIX(#4): Summarize the full file, not a silently truncated 10k-line tail.
            created_at_dt = _parse_iso_datetime(record.created_at)
            if cutoff is not None and (created_at_dt is None or created_at_dt < cutoff):
                continue

            requests_total += 1
            by_kind[record.request_kind] = by_kind.get(record.request_kind, 0) + 1
            model_key = record.model or "unknown"
            by_model[model_key] = by_model.get(model_key, 0) + 1

            if latest is None:
                latest = record
                latest_dt = created_at_dt
            elif created_at_dt is not None and (latest_dt is None or created_at_dt >= latest_dt):
                latest = record
                latest_dt = created_at_dt
            elif created_at_dt is None and latest_dt is None:
                latest = record

            if record.token_usage is None:
                continue
            requests_with_token_data += 1
            input_tokens += int(record.token_usage.input_tokens or 0)
            output_tokens += int(record.token_usage.output_tokens or 0)
            total_tokens += int(record.token_usage.total_tokens_estimate or 0)
            cached_input_tokens += int(record.token_usage.cached_input_tokens or 0)
            reasoning_tokens += int(record.token_usage.reasoning_tokens or 0)

        if requests_total == 0:
            return UsageSummary(by_kind={}, by_model={})

        return UsageSummary(
            requests_total=requests_total,
            requests_with_token_data=requests_with_token_data,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_input_tokens,
            reasoning_tokens=reasoning_tokens,
            latest_model=latest.model if latest is not None else None,
            latest_request_kind=latest.request_kind if latest is not None else None,
            latest_created_at=latest.created_at if latest is not None else None,
            by_kind=by_kind,
            by_model=by_model,
        )


def extract_model_name(source: object, fallback: str | None = None) -> str | None:
    """Extract a normalized model name from a provider object or mapping."""

    value = _field_value(source, "model")
    if value is None or not _normalize_optional_text(value):
        return _normalize_optional_text(fallback)
    return _normalize_optional_text(value)


def extract_token_usage(source: object) -> TokenUsage | None:
    """Extract normalized token usage from a provider object or mapping."""

    usage = _field_value(source, "usage")
    if usage is None and any(
        _field_value(source, field_name) is not None
        for field_name in ("input_tokens", "output_tokens", "total_tokens")
    ):
        usage = source  # AUDIT-FIX(#6): Accept callers that already pass a usage object/dict directly.
    if usage is None:
        return None
    input_details = _first_non_none(
        _field_value(usage, "input_tokens_details"),
        _field_value(usage, "input_token_details"),
    )
    output_details = _first_non_none(
        _field_value(usage, "output_tokens_details"),
        _field_value(usage, "output_token_details"),
    )
    token_usage = TokenUsage(
        input_tokens=_coerce_int(_field_value(usage, "input_tokens")),
        output_tokens=_coerce_int(_field_value(usage, "output_tokens")),
        total_tokens=_coerce_int(_field_value(usage, "total_tokens")),
        cached_input_tokens=_coerce_int(_field_value(input_details, "cached_tokens")),
        reasoning_tokens=_coerce_int(_field_value(output_details, "reasoning_tokens")),
        audio_input_tokens=_coerce_int(
            _first_non_none(  # AUDIT-FIX(#6): Preserve explicit zero token counts instead of dropping them via `or`.
                _field_value(input_details, "audio_tokens"),
                _field_value(input_details, "input_audio_tokens"),
            )
        ),
        audio_output_tokens=_coerce_int(
            _first_non_none(  # AUDIT-FIX(#6): Preserve explicit zero token counts instead of dropping them via `or`.
                _field_value(output_details, "audio_tokens"),
                _field_value(output_details, "output_audio_tokens"),
            )
        ),
    )
    if not token_usage.has_values:
        return None
    return token_usage
