"""Persist health counters and bounded auto-pause policy for learned skills."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import logging  # AUDIT-FIX(#3): Log dependency failures instead of crashing the runtime path.
import math  # AUDIT-FIX(#8): Validate duration_seconds against NaN/inf and negatives.
import re  # AUDIT-FIX(#2): Enforce safe identifier boundaries for file-backed stores.
from threading import RLock  # AUDIT-FIX(#1): Serialize read-modify-write health updates within the process.
from typing import Any

from twinr.agent.self_coding.activation import SelfCodingActivationService
from twinr.agent.self_coding.contracts import LiveE2EStatusRecord, SkillHealthRecord
from twinr.agent.self_coding.status import LearnedSkillStatus
from twinr.agent.self_coding.store import SelfCodingStore

_LOGGER = logging.getLogger(__name__)
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")  # AUDIT-FIX(#2): Reject path-like or ambiguous identifiers at the service boundary.
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]+\b"),
    re.compile(r"(?i)\b(authorization|api[_-]?key|token|secret|password|passwd)\b(\s*[:=]\s*)(\S+)"),
)  # AUDIT-FIX(#7): Redact common credential patterns before persisting error text.
_MAX_ERROR_MESSAGE_LENGTH = 512
_MAX_TEXT_FIELD_LENGTH = 256
_MAX_DETAILS_LENGTH = 2048
_MAX_METADATA_DEPTH = 5
_MAX_METADATA_ITEMS = 100


class SelfCodingHealthService:
    """Track runtime health for learned skills and pause unstable versions."""

    def __init__(
        self,
        *,
        store: SelfCodingStore,
        activation_service: SelfCodingActivationService | None = None,
        auto_pause_failure_threshold: int = 3,
    ) -> None:
        self.store = store
        self.activation_service = activation_service
        self.auto_pause_failure_threshold = self._validated_int(
            "auto_pause_failure_threshold",
            auto_pause_failure_threshold,
            minimum=1,
        )  # AUDIT-FIX(#5): Reject lossy threshold coercions and invalid config values.
        self._lock = RLock()  # AUDIT-FIX(#1): Protect in-process read-modify-write health transitions.

    def record_success(
        self,
        *,
        skill_id: str,
        version: int,
        delivered: bool,
        triggered_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SkillHealthRecord:
        """Record one successful trigger execution."""

        skill_id = self._validated_identifier("skill_id", skill_id)  # AUDIT-FIX(#2): Stop unsafe identifiers before they reach file-backed dependencies.
        version = self._validated_int("version", version, minimum=0)  # AUDIT-FIX(#2): Reject negative or malformed version values.
        delivered = self._validated_bool("delivered", delivered)  # AUDIT-FIX(#7): Avoid truthiness-based counter corruption.
        now = self._aware_now(triggered_at)
        with self._lock:  # AUDIT-FIX(#1): Serialize load-update-save for file-backed state.
            current = self._load_or_default(skill_id=skill_id, version=version)
            preserve_auto_paused = self._should_preserve_auto_paused(
                current=current,
                skill_id=skill_id,
                version=version,
            )  # AUDIT-FIX(#4): Keep auto-paused state until activation is active again.
            updated = replace(
                current,
                status="auto_paused" if preserve_auto_paused else "healthy",
                trigger_count=self._counter_value(getattr(current, "trigger_count", 0)) + 1,  # AUDIT-FIX(#8): Tolerate corrupt or missing counters in persisted records.
                delivered_count=self._counter_value(getattr(current, "delivered_count", 0)) + (1 if delivered else 0),
                consecutive_error_count=0,
                last_triggered_at=now,
                last_delivered_at=now if delivered else getattr(current, "last_delivered_at", None),
                updated_at=now,
                metadata=self._merged_metadata(getattr(current, "metadata", {}), metadata),  # AUDIT-FIX(#5): Persist sanitized JSON-safe metadata only.
            )
            return self._save_skill_health_best_effort(updated)

    def record_failure(
        self,
        *,
        skill_id: str,
        version: int,
        error: Exception | str,
        triggered_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SkillHealthRecord:
        """Record one failed trigger execution and auto-pause when needed."""

        skill_id = self._validated_identifier("skill_id", skill_id)  # AUDIT-FIX(#2): Stop unsafe identifiers before they reach file-backed dependencies.
        version = self._validated_int("version", version, minimum=0)  # AUDIT-FIX(#2): Reject negative or malformed version values.
        now = self._aware_now(triggered_at)
        safe_message = self._error_message(error)  # AUDIT-FIX(#5): Persist bounded, redacted, printable error summaries only.
        with self._lock:  # AUDIT-FIX(#1): Serialize load-update-pause-save state transitions.
            current = self._load_or_default(skill_id=skill_id, version=version)
            preserve_auto_paused = self._should_preserve_auto_paused(
                current=current,
                skill_id=skill_id,
                version=version,
            )  # AUDIT-FIX(#4): Do not overwrite a paused activation back to failing.
            updated = replace(
                current,
                status="auto_paused" if preserve_auto_paused else "failing",
                trigger_count=self._counter_value(getattr(current, "trigger_count", 0)) + 1,  # AUDIT-FIX(#8): Tolerate corrupt or missing counters in persisted records.
                error_count=self._counter_value(getattr(current, "error_count", 0)) + 1,
                consecutive_error_count=self._counter_value(getattr(current, "consecutive_error_count", 0)) + 1,
                last_triggered_at=now,
                last_error_at=now,
                last_error_message=safe_message,
                updated_at=now,
                metadata=self._merged_metadata(getattr(current, "metadata", {}), metadata),  # AUDIT-FIX(#5): Persist sanitized JSON-safe metadata only.
            )
            persisted_failure = self._save_skill_health_best_effort(
                updated,
            )  # AUDIT-FIX(#1): Persist the failure record before attempting the safety pause.

            if (
                persisted_failure.consecutive_error_count >= self.auto_pause_failure_threshold
                and self._can_auto_pause(skill_id, version)
            ):
                pause_error = self._pause_activation(skill_id=skill_id, version=version)
                if pause_error is not None:
                    pause_failed_record = replace(
                        persisted_failure,
                        updated_at=now,
                        metadata=self._merged_metadata(
                            getattr(persisted_failure, "metadata", {}),
                            {
                                "auto_pause_failed": True,
                                "auto_pause_error": pause_error,
                            },
                        ),
                    )  # AUDIT-FIX(#1): Keep the failure persisted even when the pause operation fails.
                    return self._save_skill_health_best_effort(pause_failed_record)

                auto_paused = replace(
                    persisted_failure,
                    status="auto_paused",
                    auto_pause_count=self._counter_value(getattr(persisted_failure, "auto_pause_count", 0)) + 1,
                    updated_at=now,
                    metadata=self._merged_metadata(
                        getattr(persisted_failure, "metadata", {}),
                        {"pause_reason": "auto_pause"},
                    ),
                )
                return self._save_skill_health_best_effort(auto_paused)

            return persisted_failure

    def record_live_e2e_status(
        self,
        *,
        suite_id: str,
        environment: str,
        status: str,
        duration_seconds: float | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        details: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LiveE2EStatusRecord:
        """Persist the latest explicit live end-to-end proof result."""

        suite_id = self._validated_identifier("suite_id", suite_id)  # AUDIT-FIX(#2): Stop unsafe identifiers before they reach file-backed dependencies.
        environment = self._validated_identifier(
            "environment",
            environment,
        )  # AUDIT-FIX(#2): Stop unsafe identifiers before they reach file-backed dependencies.
        record = LiveE2EStatusRecord(
            suite_id=suite_id,
            environment=environment,
            status=self._validated_label("status", status, max_length=64),  # AUDIT-FIX(#7): Normalize and reject blank status labels.
            duration_seconds=self._validated_duration_seconds(duration_seconds),  # AUDIT-FIX(#7): Reject negative/NaN/inf durations.
            model=self._optional_text(model, max_length=_MAX_TEXT_FIELD_LENGTH),
            reasoning_effort=self._optional_text(reasoning_effort, max_length=64),
            details=self._optional_text(details, max_length=_MAX_DETAILS_LENGTH),
            metadata=self._metadata_dict(metadata),  # AUDIT-FIX(#5): Persist sanitized JSON-safe metadata only.
        )
        return self._save_live_e2e_status_best_effort(record)

    def _load_or_default(self, *, skill_id: str, version: int) -> SkillHealthRecord:
        try:
            return self.store.load_skill_health(skill_id, version=version)
        except FileNotFoundError:
            return SkillHealthRecord(skill_id=skill_id, version=version, status="unknown")
        except Exception as exc:  # AUDIT-FIX(#3): Corrupt or unreadable health files must not crash runtime execution.
            _LOGGER.warning(
                "Failed to load skill health for %s v%s; using a default record. error=%s",
                skill_id,
                version,
                self._error_message(exc),
                exc_info=True,
            )
            return SkillHealthRecord(skill_id=skill_id, version=version, status="unknown")

    def _save_skill_health_best_effort(self, record: SkillHealthRecord) -> SkillHealthRecord:
        try:
            return self.store.save_skill_health(record)
        except Exception as exc:  # AUDIT-FIX(#3): Health persistence should degrade gracefully instead of taking down the caller.
            _LOGGER.exception(
                "Failed to save skill health for %s v%s; returning an unsaved record.",
                getattr(record, "skill_id", "<unknown>"),
                getattr(record, "version", "<unknown>"),
            )
            return replace(
                record,
                metadata=self._merged_metadata(
                    getattr(record, "metadata", {}),
                    {
                        "health_persist_failed": True,
                        "health_persist_error": self._error_message(exc),
                    },
                ),
            )

    def _save_live_e2e_status_best_effort(self, record: LiveE2EStatusRecord) -> LiveE2EStatusRecord:
        try:
            return self.store.save_live_e2e_status(record)
        except Exception as exc:  # AUDIT-FIX(#3): Live E2E telemetry persistence should not crash the runtime path.
            _LOGGER.exception(
                "Failed to save live E2E status for suite=%s environment=%s; returning an unsaved record.",
                getattr(record, "suite_id", "<unknown>"),
                getattr(record, "environment", "<unknown>"),
            )
            return record

    def _load_activation(self, *, skill_id: str, version: int) -> Any | None:
        if self.activation_service is None:
            return None
        try:
            return self.activation_service.load_activation(skill_id=skill_id, version=version)
        except FileNotFoundError:
            return None
        except Exception as exc:  # AUDIT-FIX(#3): Activation-state read failures should degrade safely.
            _LOGGER.warning(
                "Failed to load activation for %s v%s; assuming inactive. error=%s",
                skill_id,
                version,
                self._error_message(exc),
                exc_info=True,
            )
            return None

    def _pause_activation(self, *, skill_id: str, version: int) -> str | None:
        if self.activation_service is None:
            return "activation service unavailable"
        try:
            self.activation_service.pause_activation(
                skill_id=skill_id,
                version=version,
                reason="auto_pause",
            )
        except Exception as exc:  # AUDIT-FIX(#1): Never let the safety pause crash the failure-accounting path.
            _LOGGER.exception(
                "Failed to auto-pause activation for %s v%s after repeated failures.",
                skill_id,
                version,
            )
            return self._error_message(exc)
        return None

    def _can_auto_pause(self, skill_id: str, version: int) -> bool:
        activation = self._load_activation(skill_id=skill_id, version=version)
        return activation is not None and activation.status == LearnedSkillStatus.ACTIVE

    def _should_preserve_auto_paused(
        self,
        *,
        current: SkillHealthRecord,
        skill_id: str,
        version: int,
    ) -> bool:
        if getattr(current, "status", None) != "auto_paused":
            return False
        activation = self._load_activation(skill_id=skill_id, version=version)
        if activation is None:
            return True
        return activation.status != LearnedSkillStatus.ACTIVE

    @staticmethod
    def _aware_now(value: datetime | None) -> datetime:
        if value is None:
            return datetime.now(UTC)
        if not isinstance(value, datetime):
            raise TypeError("triggered_at must be datetime | None")
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)  # AUDIT-FIX(#9): Normalize all aware datetimes to UTC before persisting.

    @classmethod
    def _error_message(cls, error: Exception | str) -> str:
        return cls._sanitize_text(
            error,
            max_length=_MAX_ERROR_MESSAGE_LENGTH,
            default="unknown skill execution failure",
        )

    @classmethod
    def _merged_metadata(cls, current: Any, extra: Any | None) -> dict[str, Any]:
        merged = cls._metadata_dict(current)
        if extra is None:
            return merged
        merged.update(cls._metadata_dict(extra))
        return merged

    @classmethod
    def _metadata_dict(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        try:
            mapping = dict(value)
        except Exception:  # AUDIT-FIX(#5): Drop invalid metadata payloads instead of failing persistence.
            _LOGGER.warning(
                "Ignoring non-mapping metadata payload of type %s.",
                type(value).__name__,
            )
            return {}
        safe: dict[str, Any] = {}
        for index, (key, item) in enumerate(mapping.items()):
            if index >= _MAX_METADATA_ITEMS:
                break
            safe_key = cls._sanitize_text(key, max_length=128, default="key")
            safe[safe_key] = cls._json_safe_value(item, depth=0)
        return safe

    @classmethod
    def _json_safe_value(cls, value: Any, *, depth: int) -> Any:
        if depth >= _MAX_METADATA_DEPTH:
            return cls._sanitize_text(value, max_length=_MAX_TEXT_FIELD_LENGTH, default="truncated")
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, str):
            return cls._sanitize_text(value, max_length=_MAX_TEXT_FIELD_LENGTH, default="")
        if isinstance(value, datetime):
            return cls._aware_now(value).isoformat()
        if isinstance(value, Exception):
            return cls._error_message(value)
        if isinstance(value, dict):
            nested: dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= _MAX_METADATA_ITEMS:
                    break
                safe_key = cls._sanitize_text(key, max_length=128, default="key")
                nested[safe_key] = cls._json_safe_value(item, depth=depth + 1)
            return nested
        if isinstance(value, (list, tuple, set, frozenset)):
            return [
                cls._json_safe_value(item, depth=depth + 1)
                for item in list(value)[:_MAX_METADATA_ITEMS]
            ]
        return cls._sanitize_text(value, max_length=_MAX_TEXT_FIELD_LENGTH, default=type(value).__name__)

    @staticmethod
    def _stringify(value: Any) -> str:
        try:
            return str(value)
        except Exception:
            return f"<unprintable {type(value).__name__}>"

    @classmethod
    def _sanitize_text(cls, value: Any, *, max_length: int, default: str) -> str:
        text = cls._stringify(value)
        printable = "".join(character if character.isprintable() and character != "\x00" else " " for character in text)
        normalized = " ".join(printable.split())
        if not normalized:
            return default
        redacted = cls._redact_secrets(normalized)
        return redacted[:max_length]

    @staticmethod
    def _redact_secrets(text: str) -> str:
        redacted = text
        redacted = _SECRET_PATTERNS[0].sub(r"\1 [REDACTED]", redacted)
        redacted = _SECRET_PATTERNS[1].sub(r"\1\2[REDACTED]", redacted)
        return redacted

    @staticmethod
    def _validated_bool(field_name: str, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        raise TypeError(f"{field_name} must be bool")

    @staticmethod
    def _validated_int(field_name: str, value: Any, *, minimum: int) -> int:
        parsed: int
        if isinstance(value, bool):
            raise TypeError(f"{field_name} must be int")
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"{field_name} must be an integer value")
            parsed = int(value)
        elif isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                raise ValueError(f"{field_name} must not be empty")
            try:
                parsed = int(candidate, 10)
            except ValueError as exc:
                raise ValueError(f"{field_name} must be an integer value") from exc
        else:
            raise TypeError(f"{field_name} must be int")
        if parsed < minimum:
            raise ValueError(f"{field_name} must be >= {minimum}")
        return parsed

    @classmethod
    def _validated_identifier(cls, field_name: str, value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be str")
        if value != value.strip():
            raise ValueError(f"{field_name} must not contain leading or trailing whitespace")
        if not value:
            raise ValueError(f"{field_name} must not be empty")
        if "/" in value or "\\" in value or ".." in value or "\x00" in value:
            raise ValueError(f"{field_name} contains unsafe path characters")
        if not _SAFE_IDENTIFIER_RE.fullmatch(value):
            raise ValueError(
                f"{field_name} may contain only letters, digits, '.', '_', ':', and '-' and must start with an alphanumeric character",
            )
        return value

    @classmethod
    def _validated_label(cls, field_name: str, value: Any, *, max_length: int) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be str")
        normalized = cls._sanitize_text(value, max_length=max_length, default="")
        if not normalized:
            raise ValueError(f"{field_name} must not be empty")
        return normalized

    @staticmethod
    def _validated_duration_seconds(value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError("duration_seconds must be a real number or None")
        if isinstance(value, (int, float)):
            duration_seconds = float(value)
        elif isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                raise ValueError("duration_seconds must not be empty")
            try:
                duration_seconds = float(candidate)
            except ValueError as exc:
                raise ValueError("duration_seconds must be a real number") from exc
        else:
            raise TypeError("duration_seconds must be a real number or None")
        if not math.isfinite(duration_seconds) or duration_seconds < 0:
            raise ValueError("duration_seconds must be finite and >= 0")
        return duration_seconds

    @classmethod
    def _optional_text(cls, value: Any, *, max_length: int) -> str | None:
        if value is None:
            return None
        normalized = cls._sanitize_text(value, max_length=max_length, default="")
        return normalized or None

    @classmethod
    def _counter_value(cls, value: Any) -> int:
        try:
            return cls._validated_int("counter", value, minimum=0)
        except (TypeError, ValueError):
            return 0


__all__ = ["SelfCodingHealthService"]