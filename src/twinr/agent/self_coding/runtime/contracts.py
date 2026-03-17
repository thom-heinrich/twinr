"""Define validated runtime contracts for compiled self-coding skill packages."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
import json
import keyword
import math
from pathlib import PurePosixPath
from typing import Any, ClassVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.agent.self_coding.sandbox.policy import (
    CapabilityBrokerManifest,
    build_capability_broker_manifest,
)
from twinr.text_utils import extract_json_object

from twinr.agent.self_coding.contracts import CompileJobRecord, RequirementsDialogueSession

_SKILL_PACKAGE_SCHEMA = "twinr_self_coding_skill_package_v1"
_SKILL_PACKAGE_TARGET = "skill_package"  # AUDIT-FIX(#7): validate artifact target explicitly when loading persisted documents.
_MAX_FILES = 16
_MAX_FILE_BYTES = 128_000
_MAX_TRIGGERS_PER_FAMILY = 8
_MAX_TEXT_LENGTH = 240
_SHA256_HEX_LENGTH = 64
_SUPPORTED_SCHEDULES: ClassVar[frozenset[str]] = frozenset({"once", "daily", "weekly"})


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _text(value: object, *, field_name: str, allow_empty: bool = False, limit: int = _MAX_TEXT_LENGTH) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > limit:
        raise ValueError(f"{field_name} must be <= {limit} characters")
    return normalized


def _stable_identifier(value: object, *, field_name: str) -> str:
    text = _text(value, field_name=field_name, limit=128)
    if not text.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"{field_name} must be a stable identifier")
    return text.lower()


def _python_identifier(value: object, *, field_name: str) -> str:
    text = _text(value, field_name=field_name, limit=128)
    if not text.isidentifier() or keyword.iskeyword(text):  # AUDIT-FIX(#2): reject Python keywords that pass isidentifier() but cannot name handlers safely.
        raise ValueError(f"{field_name} must be a valid Python identifier")
    return text


def _safe_relative_python_path(value: object, *, field_name: str) -> str:
    raw_text = _text(value, field_name=field_name, limit=240)
    path = PurePosixPath(raw_text)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} must be a safe relative path")
    if path.suffix != ".py":
        raise ValueError(f"{field_name} must point to a Python file")
    return str(path)


def _positive_float(value: object, *, field_name: str, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a non-negative finite number")
    try:
        number = float(value)  # AUDIT-FIX(#3): normalize numeric-like inputs but trap invalid conversions deterministically.
    except TypeError as exc:  # AUDIT-FIX(#3): expose contract errors instead of leaking float() implementation details.
        raise TypeError(f"{field_name} must be a non-negative finite number") from exc
    except (ValueError, OverflowError) as exc:  # AUDIT-FIX(#3): reject malformed, overflowed, and non-parsable numeric text.
        raise ValueError(f"{field_name} must be a non-negative finite number") from exc
    if not math.isfinite(number) or number < 0.0:  # AUDIT-FIX(#3): block NaN/Inf values that later break timers and JSON serialization.
        raise ValueError(f"{field_name} must be a non-negative finite number")
    return number


def _weekday_tuple(value: object | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError("weekdays must be a list")
    normalized: set[int] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):  # AUDIT-FIX(#4): reject silent truncation/coercion from floats and numeric strings.
            raise TypeError("weekdays must contain integers between 0 and 6")
        if item < 0 or item > 6:
            raise ValueError("weekdays entries must be between 0 and 6")
        normalized.add(item)
    return tuple(sorted(normalized))  # AUDIT-FIX(#4): canonicalize weekday order for deterministic persisted artifacts.


def _json_roundtrip(value: object, *, field_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be JSON serializable") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{field_name} must be a JSON object")
    return payload


def _json_object_payload(value: object, *, field_name: str) -> dict[str, Any]:
    candidate = value
    if isinstance(value, str):
        try:
            candidate = extract_json_object(value)  # AUDIT-FIX(#8): support string payloads without double-encoding and surface extraction failures clearly.
        except Exception as exc:  # AUDIT-FIX(#8): normalize parser failures into contract-level validation errors.
            raise ValueError(f"{field_name} must contain a JSON object") from exc
    return _json_roundtrip(candidate, field_name=field_name)


def _json_object_list(value: object, *, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):  # AUDIT-FIX(#8): reject malformed list entries instead of silently dropping them.
            raise TypeError(f"{field_name}[{index}] must be a JSON object")
        normalized.append(item)
    return normalized


def _normalized_time_of_day(value: object, *, field_name: str) -> str:
    text = _text(value, field_name=field_name, limit=16)
    try:
        parsed = datetime.strptime(text, "%H:%M")  # AUDIT-FIX(#5): require a canonical HH:MM format for recurring schedules.
    except ValueError as exc:
        raise ValueError(f"{field_name} must be in HH:MM 24-hour format") from exc
    return parsed.strftime("%H:%M")


def _normalized_due_at(value: object, *, field_name: str) -> str:
    text = _text(value, field_name=field_name, limit=64)
    candidate = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)  # AUDIT-FIX(#5): validate once-trigger datetimes as real ISO 8601 timestamps.
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO 8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")  # AUDIT-FIX(#5): block naive datetimes that misfire across DST and device locale changes.
    return parsed.isoformat().replace("+00:00", "Z")


def _validated_timezone_name(value: object, *, field_name: str) -> str:
    text = _text(value, field_name=field_name, limit=64)
    try:
        ZoneInfo(text)  # AUDIT-FIX(#5): ensure timezone names are real IANA zones before accepting recurring schedules.
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"{field_name} must be a valid IANA timezone name") from exc
    return text


def _validated_sha256_hex(value: object, *, field_name: str) -> str:
    text = _text(value, field_name=field_name, limit=_SHA256_HEX_LENGTH).lower()
    if len(text) != _SHA256_HEX_LENGTH or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field_name} must be a 64-character hex digest")
    return text


def _parse_python_module(source: str, *, path: str) -> ast.Module:
    try:
        return ast.parse(source, filename=path)  # AUDIT-FIX(#2): validate package file syntax before a malformed skill reaches runtime activation.
    except SyntaxError as exc:
        location = f"line {exc.lineno}, column {exc.offset}" if exc.lineno and exc.offset else "unknown location"
        raise ValueError(f"invalid Python syntax in {path}: {exc.msg} ({location})") from exc


def _top_level_handler_names(module: ast.Module) -> set[str]:
    return {
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _ensure_unique(values: tuple[str, ...], *, field_name: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {field_name}: {value!r}")  # AUDIT-FIX(#6): fail closed on ambiguous file paths and trigger IDs.
        seen.add(value)


@dataclass(frozen=True, slots=True)
class SkillPackageFile:
    """One source file inside a compiled skill package."""

    path: str
    content: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _safe_relative_python_path(self.path, field_name="path"))
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if not self.content.strip():
            raise ValueError("content must not be empty")
        if len(self.content.encode("utf-8")) > _MAX_FILE_BYTES:
            raise ValueError(f"content for {self.path!r} exceeds {_MAX_FILE_BYTES} bytes")

    @classmethod
    def from_payload(cls, payload: object, *, field_name: str = "skill_package.files") -> SkillPackageFile:
        if not isinstance(payload, dict):
            raise TypeError(f"{field_name} must be a JSON object")
        instance = cls(
            path=payload.get("path", ""),
            content=payload.get("content", ""),
        )
        expected_sha256 = payload.get("sha256")
        if expected_sha256 is not None:
            actual_sha256 = _validated_sha256_hex(expected_sha256, field_name=f"{field_name}.sha256")  # AUDIT-FIX(#9): verify embedded file digests when present.
            if actual_sha256 != instance.sha256:
                raise ValueError(f"{field_name}.sha256 does not match content")  # AUDIT-FIX(#9): detect corrupted or partially overwritten persisted artifacts.
        return instance

    @property
    def sha256(self) -> str:
        return sha256(self.content.encode("utf-8")).hexdigest()

    def to_payload(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content": self.content,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SkillPackageScheduledTrigger:
    """A time-based trigger that calls into the generated skill package."""

    trigger_id: str
    schedule: str
    time_of_day: str | None = None
    due_at: str | None = None
    weekdays: tuple[int, ...] = field(default_factory=tuple)
    timezone_name: str = "Europe/Berlin"
    handler: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "trigger_id", _stable_identifier(self.trigger_id, field_name="trigger_id"))
        schedule = _text(self.schedule, field_name="schedule", limit=32).lower()
        if schedule not in _SUPPORTED_SCHEDULES:
            raise ValueError(f"schedule must be one of {sorted(_SUPPORTED_SCHEDULES)!r}")
        object.__setattr__(self, "schedule", schedule)
        normalized_time = None if self.time_of_day is None else _normalized_time_of_day(self.time_of_day, field_name="time_of_day")
        normalized_due_at = None if self.due_at is None else _normalized_due_at(self.due_at, field_name="due_at")
        weekdays = _weekday_tuple(self.weekdays)
        timezone_name = _validated_timezone_name(self.timezone_name, field_name="timezone_name")
        if schedule == "once":
            if normalized_due_at is None:
                raise ValueError("once scheduled triggers require due_at")  # AUDIT-FIX(#5): prevent ambiguous one-shot triggers.
            if normalized_time is not None:
                raise ValueError("once scheduled triggers do not support time_of_day")  # AUDIT-FIX(#5): keep schedule semantics unambiguous.
            if weekdays:
                raise ValueError("once scheduled triggers do not support weekdays")  # AUDIT-FIX(#5): weekdays are meaningless for one-shot triggers.
        elif schedule == "daily":
            if normalized_time is None:
                raise ValueError("daily scheduled triggers require time_of_day")  # AUDIT-FIX(#5): recurring daily triggers need a local clock time.
            if normalized_due_at is not None:
                raise ValueError("daily scheduled triggers do not support due_at")  # AUDIT-FIX(#5): avoid mixing one-shot and recurring semantics.
            if weekdays:
                raise ValueError("daily scheduled triggers do not support weekdays")  # AUDIT-FIX(#5): weekdays belong to weekly schedules only.
        else:
            if normalized_time is None:
                raise ValueError("weekly scheduled triggers require time_of_day")  # AUDIT-FIX(#5): weekly triggers need a local clock time.
            if normalized_due_at is not None:
                raise ValueError("weekly scheduled triggers do not support due_at")  # AUDIT-FIX(#5): avoid ambiguous scheduling precedence.
            if not weekdays:
                raise ValueError("weekly scheduled triggers require weekdays")  # AUDIT-FIX(#5): weekly recurrence is invalid without explicit days.
        object.__setattr__(self, "time_of_day", normalized_time)
        object.__setattr__(self, "due_at", normalized_due_at)
        object.__setattr__(self, "weekdays", weekdays)
        object.__setattr__(self, "timezone_name", timezone_name)
        object.__setattr__(self, "handler", _python_identifier(self.handler, field_name="handler"))

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        field_name: str = "skill_package.scheduled_triggers",
    ) -> SkillPackageScheduledTrigger:
        if not isinstance(payload, dict):
            raise TypeError(f"{field_name} must be a JSON object")
        return cls(
            trigger_id=payload.get("trigger_id", ""),
            schedule=payload.get("schedule", ""),
            time_of_day=payload.get("time_of_day"),
            due_at=payload.get("due_at"),
            weekdays=payload.get("weekdays", ()),
            timezone_name=payload.get("timezone_name", "Europe/Berlin"),
            handler=payload.get("handler", ""),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "trigger_id": self.trigger_id,
            "schedule": self.schedule,
            "time_of_day": self.time_of_day,
            "due_at": self.due_at,
            "weekdays": list(self.weekdays),
            "timezone_name": self.timezone_name,
            "handler": self.handler,
        }


@dataclass(frozen=True, slots=True)
class SkillPackageSensorTrigger:
    """A sensor-based trigger that calls into the generated skill package."""

    trigger_id: str
    sensor_trigger_kind: str
    handler: str
    hold_seconds: float = 0.0
    cooldown_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "trigger_id", _stable_identifier(self.trigger_id, field_name="trigger_id"))
        object.__setattr__(
            self,
            "sensor_trigger_kind",
            _stable_identifier(self.sensor_trigger_kind, field_name="sensor_trigger_kind"),
        )
        object.__setattr__(self, "handler", _python_identifier(self.handler, field_name="handler"))
        object.__setattr__(self, "hold_seconds", _positive_float(self.hold_seconds, field_name="hold_seconds"))
        object.__setattr__(
            self,
            "cooldown_seconds",
            _positive_float(self.cooldown_seconds, field_name="cooldown_seconds"),
        )

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        field_name: str = "skill_package.sensor_triggers",
    ) -> SkillPackageSensorTrigger:
        if not isinstance(payload, dict):
            raise TypeError(f"{field_name} must be a JSON object")
        return cls(
            trigger_id=payload.get("trigger_id", ""),
            sensor_trigger_kind=payload.get("sensor_trigger_kind", ""),
            handler=payload.get("handler", ""),
            hold_seconds=payload.get("hold_seconds", 0.0),
            cooldown_seconds=payload.get("cooldown_seconds", 0.0),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "trigger_id": self.trigger_id,
            "sensor_trigger_kind": self.sensor_trigger_kind,
            "handler": self.handler,
            "hold_seconds": self.hold_seconds,
            "cooldown_seconds": self.cooldown_seconds,
        }


@dataclass(frozen=True, slots=True)
class SkillPackage:
    """The validated runtime envelope for one compiled skill package."""

    name: str
    description: str
    entry_module: str
    files: tuple[SkillPackageFile, ...]
    scheduled_triggers: tuple[SkillPackageScheduledTrigger, ...] = field(default_factory=tuple)
    sensor_triggers: tuple[SkillPackageSensorTrigger, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, field_name="name", limit=160))
        object.__setattr__(self, "description", _text(self.description, field_name="description", limit=240))
        object.__setattr__(self, "entry_module", _safe_relative_python_path(self.entry_module, field_name="entry_module"))
        files = tuple(self.files)
        if not files:
            raise ValueError("skill package requires at least one file")
        if len(files) > _MAX_FILES:
            raise ValueError(f"skill package supports at most {_MAX_FILES} files")
        if not all(isinstance(item, SkillPackageFile) for item in files):
            raise TypeError("files must contain SkillPackageFile items only")
        _ensure_unique(tuple(item.path for item in files), field_name="file path")
        file_paths = {item.path for item in files}
        if self.entry_module not in file_paths:
            raise ValueError("entry_module must point to one of the package files")
        scheduled = tuple(self.scheduled_triggers)
        sensor = tuple(self.sensor_triggers)
        if len(scheduled) > _MAX_TRIGGERS_PER_FAMILY:
            raise ValueError(f"skill package supports at most {_MAX_TRIGGERS_PER_FAMILY} scheduled triggers")
        if len(sensor) > _MAX_TRIGGERS_PER_FAMILY:
            raise ValueError(f"skill package supports at most {_MAX_TRIGGERS_PER_FAMILY} sensor triggers")
        if not all(isinstance(item, SkillPackageScheduledTrigger) for item in scheduled):
            raise TypeError("scheduled_triggers must contain SkillPackageScheduledTrigger items only")
        if not all(isinstance(item, SkillPackageSensorTrigger) for item in sensor):
            raise TypeError("sensor_triggers must contain SkillPackageSensorTrigger items only")
        _ensure_unique(tuple(item.trigger_id for item in scheduled), field_name="scheduled trigger_id")
        _ensure_unique(tuple(item.trigger_id for item in sensor), field_name="sensor trigger_id")
        if not scheduled and not sensor:
            raise ValueError("skill package requires at least one trigger")
        parsed_modules = {
            item.path: _parse_python_module(item.content, path=item.path)
            for item in files
        }
        entry_handlers = _top_level_handler_names(parsed_modules[self.entry_module])  # AUDIT-FIX(#2): inspect actual top-level callables instead of fragile substring matching.
        required_handlers = {trigger.handler for trigger in scheduled} | {trigger.handler for trigger in sensor}
        missing_handlers = sorted(handler_name for handler_name in required_handlers if handler_name not in entry_handlers)
        if missing_handlers:
            missing_text = ", ".join(repr(name) for name in missing_handlers)
            raise ValueError(f"entry module is missing handler(s): {missing_text}")  # AUDIT-FIX(#2): fail validation before runtime dispatch crashes.
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "scheduled_triggers", scheduled)
        object.__setattr__(self, "sensor_triggers", sensor)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "entry_module": self.entry_module,
            "files": [item.to_payload() for item in self.files],
            "scheduled_triggers": [item.to_payload() for item in self.scheduled_triggers],
            "sensor_triggers": [item.to_payload() for item in self.sensor_triggers],
        }

    def scheduled_trigger(self, trigger_id: str) -> SkillPackageScheduledTrigger:
        normalized = _stable_identifier(trigger_id, field_name="trigger_id")
        for trigger in self.scheduled_triggers:
            if trigger.trigger_id == normalized:
                return trigger
        raise KeyError(f"Unknown scheduled trigger: {trigger_id}")

    def sensor_trigger(self, trigger_id: str) -> SkillPackageSensorTrigger:
        normalized = _stable_identifier(trigger_id, field_name="trigger_id")
        for trigger in self.sensor_triggers:
            if trigger.trigger_id == normalized:
                return trigger
        raise KeyError(f"Unknown sensor trigger: {trigger_id}")


@dataclass(frozen=True, slots=True)
class CompiledSkillPackage:
    """Carry a canonical skill package document ready for persistence."""

    package: SkillPackage
    content: str
    summary: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SkillPackageDocument:
    """Hold the parsed package plus its explicit sandbox policy manifest."""

    package: SkillPackage
    policy_manifest: CapabilityBrokerManifest
    raw_payload: dict[str, Any]


def canonical_skill_package_document(
    *,
    job: CompileJobRecord,
    session: RequirementsDialogueSession,
    package: SkillPackage,
) -> CompiledSkillPackage:
    """Build the canonical persisted document for one validated skill package."""

    compiled_at = _utc_now()
    required_capabilities = (
        job.required_capabilities
        if job.required_capabilities is not None
        else session.capabilities
    )  # AUDIT-FIX(#1): distinguish "explicitly no capabilities" from "no override supplied" to avoid sandbox privilege escalation.
    policy_manifest = build_capability_broker_manifest(required_capabilities)
    document = {
        "schema": _SKILL_PACKAGE_SCHEMA,
        "target": _SKILL_PACKAGE_TARGET,
        "job_id": job.job_id,
        "skill_id": job.skill_id,
        "skill_name": session.skill_name,
        "compiled_at": compiled_at.isoformat().replace("+00:00", "Z"),
        "activation_policy": {
            "requires_confirmation": True,
            "initial_enabled": False,
        },
        "sandbox": {
            "policy_manifest": policy_manifest.to_payload(),
        },
        "package": package.to_payload(),
    }
    metadata = {
        "artifact_kind": "skill_package",
        "manifest_schema": _SKILL_PACKAGE_SCHEMA,
        "scheduled_trigger_count": len(package.scheduled_triggers),
        "sensor_trigger_count": len(package.sensor_triggers),
        "activatable": True,
        "sandbox_policy_manifest_schema": policy_manifest.schema,
        "sandbox_allowed_methods": sorted(policy_manifest.allowed_methods),  # AUDIT-FIX(#10): keep canonical metadata deterministic across runs.
    }
    return CompiledSkillPackage(
        package=package,
        content=json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        summary=f"Validated skill package for {session.skill_name}.",
        metadata=metadata,
    )


def skill_package_document_from_document(
    raw_text: str,
    *,
    fallback_capabilities: tuple[str, ...] = (),
) -> SkillPackageDocument:
    """Load one compiled skill package document plus its sandbox policy manifest."""

    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    try:
        payload = json.loads(raw_text)  # AUDIT-FIX(#7): return a stable validation error for malformed persisted artifacts.
    except json.JSONDecodeError as exc:
        raise ValueError("skill package artifact must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("skill package artifact must be a JSON object")
    if payload.get("schema") != _SKILL_PACKAGE_SCHEMA:
        raise ValueError("skill package artifact schema mismatch")
    if payload.get("target") != _SKILL_PACKAGE_TARGET:
        raise ValueError("skill package artifact target mismatch")  # AUDIT-FIX(#7): reject other artifact kinds that happen to share a schema envelope.
    package_payload = payload.get("package")
    if not isinstance(package_payload, dict):
        raise ValueError("skill package artifact must include a package object")
    sandbox_payload = payload.get("sandbox")
    if sandbox_payload is None:
        sandbox_payload = {}
    elif not isinstance(sandbox_payload, dict):
        raise ValueError("skill package artifact sandbox must be a JSON object")  # AUDIT-FIX(#7): malformed sandbox envelopes must not fall through to permissive fallback behavior.
    manifest_payload = sandbox_payload.get("policy_manifest")
    if manifest_payload is None:
        policy_manifest = build_capability_broker_manifest(fallback_capabilities)
    else:
        try:
            policy_manifest = CapabilityBrokerManifest.from_payload(manifest_payload)  # AUDIT-FIX(#7): fail closed on malformed sandbox manifests.
        except Exception as exc:
            raise ValueError("skill package artifact includes an invalid policy manifest") from exc
    return SkillPackageDocument(
        package=skill_package_from_payload(package_payload),
        policy_manifest=policy_manifest,
        raw_payload=payload,
    )


def skill_package_from_document(raw_text: str) -> SkillPackage:
    """Load one compiled skill package document from persisted artifact text."""

    return skill_package_document_from_document(raw_text).package


def skill_package_from_payload(payload: object) -> SkillPackage:
    """Validate one package payload into a runtime-ready SkillPackage object."""

    mapping = _json_object_payload(payload, field_name="skill_package")  # AUDIT-FIX(#8): validate string/dict payloads without lossy double-serialization.
    files_payload = _json_object_list(mapping.get("files", ()), field_name="skill_package.files")
    scheduled_payload = _json_object_list(mapping.get("scheduled_triggers", ()), field_name="skill_package.scheduled_triggers")
    sensor_payload = _json_object_list(mapping.get("sensor_triggers", ()), field_name="skill_package.sensor_triggers")
    return SkillPackage(
        name=mapping.get("name", ""),
        description=mapping.get("description", ""),
        entry_module=mapping.get("entry_module", ""),
        files=tuple(
            SkillPackageFile.from_payload(item, field_name=f"skill_package.files[{index}]")
            for index, item in enumerate(files_payload)
        ),
        scheduled_triggers=tuple(
            SkillPackageScheduledTrigger.from_payload(
                item,
                field_name=f"skill_package.scheduled_triggers[{index}]",
            )
            for index, item in enumerate(scheduled_payload)
        ),
        sensor_triggers=tuple(
            SkillPackageSensorTrigger.from_payload(
                item,
                field_name=f"skill_package.sensor_triggers[{index}]",
            )
            for index, item in enumerate(sensor_payload)
        ),
    )


__all__ = [
    "CompiledSkillPackage",
    "SkillPackageDocument",
    "SkillPackage",
    "SkillPackageFile",
    "SkillPackageScheduledTrigger",
    "SkillPackageSensorTrigger",
    "canonical_skill_package_document",
    "skill_package_document_from_document",
    "skill_package_from_document",
    "skill_package_from_payload",
]