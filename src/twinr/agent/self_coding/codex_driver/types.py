"""Shared types for the local Codex-backed self-coding compile drivers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math  # AUDIT-FIX(#6): Needed for strict rejection of non-finite JSON numbers.
import os  # AUDIT-FIX(#1): Needed for path-like validation before workspace anchoring.
from pathlib import Path
from typing import Any

from twinr.agent.self_coding.contracts import CompileJobRecord, RequirementsDialogueSession
from twinr.agent.self_coding.status import ArtifactKind

_COMPILE_RESULT_STATUSES = {"ok", "unsupported", "failed"}
_COMPILE_RESULT_SCHEMA_REQUIRED_FIELDS = frozenset({"status", "summary", "review", "artifacts"})  # AUDIT-FIX(#4): Enforce the advertised strict top-level schema.
_COMPILE_RESULT_SCHEMA_ALLOWED_FIELDS = frozenset({"status", "summary", "review", "artifacts"})  # AUDIT-FIX(#4): Reject unexpected top-level fields instead of silently ignoring them.
_COMPILE_ARTIFACT_REQUIRED_FIELDS = frozenset({"kind", "artifact_name", "media_type", "content", "summary"})  # AUDIT-FIX(#4): Enforce required artifact fields before normalization.
_COMPILE_ARTIFACT_ALLOWED_FIELDS = frozenset({"kind", "artifact_name", "media_type", "content", "summary", "metadata"})  # AUDIT-FIX(#8): Keep schema/runtime parity for optional artifact metadata.


def _normalize_json_value(value: object, *, field_name: str, path: str = "") -> Any:  # AUDIT-FIX(#6): Recursively validate JSON-compatible values with field-path context.
    location = field_name if not path else f"{field_name}{path}"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{location} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} keys must be strings")
            child_path = f"{path}.{key}" if path else f".{key}"
            normalized[key] = _normalize_json_value(item, field_name=field_name, path=child_path)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        normalized_items: list[Any] = []
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]"
            normalized_items.append(_normalize_json_value(item, field_name=field_name, path=child_path))
        return normalized_items
    raise TypeError(f"{location} must be JSON serializable")


def _json_mapping(value: object | None, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized = _normalize_json_value(value, field_name=field_name)  # AUDIT-FIX(#6): Fail closed on malformed nested metadata/schema values.
    if not isinstance(normalized, dict):
        raise TypeError(f"{field_name} must serialize to a JSON object")
    return normalized


def _text(value: object, *, field_name: str, allow_empty: bool = False, trim: bool = True) -> str:  # AUDIT-FIX(#2): Require real strings and allow exact-text preservation for code/log payloads.
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if value.strip() or allow_empty:
        return value.strip() if trim else value
    raise ValueError(f"{field_name} must not be empty")


def _optional_text(value: object | None, *, field_name: str, trim: bool = True) -> str | None:  # AUDIT-FIX(#2): Normalize optional strings without coercing arbitrary objects.
    if value is None:
        return None
    normalized = _text(value, field_name=field_name, allow_empty=True, trim=trim)
    return normalized if normalized.strip() else None


def _bool(value: object, *, field_name: str) -> bool:  # AUDIT-FIX(#3): Prevent truthy string coercion such as "false" -> True.
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _non_negative_int(value: object, *, field_name: str) -> int:  # AUDIT-FIX(#3): Prevent int()/bool() coercion from silently changing progress state.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def _pathlike_text(value: object, *, field_name: str) -> str:  # AUDIT-FIX(#1): Reject empty and non-string path inputs before filesystem normalization.
    try:
        raw_path = os.fspath(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be path-like") from exc
    if isinstance(raw_path, bytes):
        raise TypeError(f"{field_name} must be a string path")
    if not isinstance(raw_path, str):
        raise TypeError(f"{field_name} must be a string path")
    if not raw_path.strip():
        raise ValueError(f"{field_name} must not be empty")
    return raw_path


def _resolve_path(path: Path, *, field_name: str) -> Path:  # AUDIT-FIX(#1): Normalize paths safely and fail predictably on symlink loops or OS errors.
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{field_name} could not be resolved safely") from exc


def _resolve_workspace_path(path_value: object, *, field_name: str, workspace_root: Path) -> str:  # AUDIT-FIX(#1): Anchor relative paths to workspace_root and block traversal escapes.
    raw_path = Path(_pathlike_text(path_value, field_name=field_name)).expanduser()
    candidate = raw_path if raw_path.is_absolute() else workspace_root / raw_path
    normalized = _resolve_path(candidate, field_name=field_name)
    if normalized == workspace_root:
        raise ValueError(f"{field_name} must reference a file inside workspace_root")
    if not normalized.is_relative_to(workspace_root):
        raise ValueError(f"{field_name} must stay within workspace_root")
    return str(normalized)


def _validate_required_and_allowed_keys(
    payload: Mapping[str, Any],
    *,
    field_name: str,
    required_keys: frozenset[str],
    allowed_keys: frozenset[str],
) -> None:  # AUDIT-FIX(#4): Enforce required fields and reject unexpected properties before object construction.
    payload_keys = set(payload.keys())
    missing = sorted(required_keys - payload_keys)
    if missing:
        raise CodexDriverProtocolError(f"{field_name} is missing required field(s): {', '.join(missing)}")
    unexpected = sorted(payload_keys - allowed_keys)
    if unexpected:
        raise CodexDriverProtocolError(f"{field_name} contains unexpected field(s): {', '.join(unexpected)}")


@dataclass(frozen=True, slots=True)
class CodexCompileEvent:
    """One normalized event emitted by a local Codex driver run."""

    kind: str
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _text(self.kind, field_name="kind"))
        object.__setattr__(self, "message", _text(self.message, field_name="message", allow_empty=True, trim=False))  # AUDIT-FIX(#2): Preserve exact event text instead of stripping driver output.
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, field_name="metadata"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class CodexCompileProgress:
    """Describe the latest observable state of one live Codex compile run."""

    driver_name: str | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    event_count: int = 0
    last_event_kind: str | None = None
    final_message_seen: bool = False
    turn_completed: bool = False
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        driver_name = _optional_text(self.driver_name, field_name="driver_name")
        object.__setattr__(self, "driver_name", driver_name)
        thread_id = _optional_text(self.thread_id, field_name="thread_id")
        object.__setattr__(self, "thread_id", thread_id)
        turn_id = _optional_text(self.turn_id, field_name="turn_id")
        object.__setattr__(self, "turn_id", turn_id)
        object.__setattr__(self, "event_count", _non_negative_int(self.event_count, field_name="event_count"))  # AUDIT-FIX(#3): Reject coerced numerics that can poison progress tracking.
        last_event_kind = _optional_text(self.last_event_kind, field_name="last_event_kind")
        object.__setattr__(self, "last_event_kind", last_event_kind)
        object.__setattr__(self, "final_message_seen", _bool(self.final_message_seen, field_name="final_message_seen"))  # AUDIT-FIX(#3): Keep completion flags type-stable.
        object.__setattr__(self, "turn_completed", _bool(self.turn_completed, field_name="turn_completed"))  # AUDIT-FIX(#3): Keep completion flags type-stable.
        error_message = _optional_text(self.error_message, field_name="error_message", trim=False)
        object.__setattr__(self, "error_message", error_message)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, field_name="metadata"))


@dataclass(frozen=True, slots=True)
class CodexCompileArtifact:
    """A compile artifact returned by a local Codex driver."""

    kind: ArtifactKind
    artifact_name: str
    media_type: str
    content: str
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.kind, ArtifactKind):
            normalized_kind = self.kind
        else:
            try:
                normalized_kind = ArtifactKind(_text(self.kind, field_name="kind"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"kind must be one of {sorted(item.value for item in ArtifactKind)}") from exc
        object.__setattr__(self, "kind", normalized_kind)
        object.__setattr__(self, "artifact_name", _text(self.artifact_name, field_name="artifact_name"))
        object.__setattr__(self, "media_type", _text(self.media_type, field_name="media_type"))
        object.__setattr__(self, "content", _text(self.content, field_name="content", allow_empty=True, trim=False))  # AUDIT-FIX(#2): Preserve exact artifact bytes-as-text instead of trimming them.
        summary = _optional_text(self.summary, field_name="summary", trim=False)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, field_name="metadata"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "artifact_name": self.artifact_name,
            "media_type": self.media_type,
            "content": self.content,
            "summary": self.summary,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class CodexCompileRunTranscript:
    """Capture the normalized output of one raw Codex driver run."""

    thread_id: str | None = None
    turn_id: str | None = None
    final_message: str | None = None
    error_message: str | None = None
    events: tuple[CodexCompileEvent, ...] = ()

    def __post_init__(self) -> None:  # AUDIT-FIX(#7): Enforce the "normalized output" invariant on transcript instances too.
        thread_id = _optional_text(self.thread_id, field_name="thread_id")
        object.__setattr__(self, "thread_id", thread_id)
        turn_id = _optional_text(self.turn_id, field_name="turn_id")
        object.__setattr__(self, "turn_id", turn_id)
        final_message = _optional_text(self.final_message, field_name="final_message", trim=False)
        object.__setattr__(self, "final_message", final_message)
        error_message = _optional_text(self.error_message, field_name="error_message", trim=False)
        object.__setattr__(self, "error_message", error_message)
        events = tuple(self.events)
        if not all(isinstance(item, CodexCompileEvent) for item in events):
            raise TypeError("events must contain CodexCompileEvent items only")
        object.__setattr__(self, "events", events)


@dataclass(frozen=True, slots=True)
class CodexCompileResult:
    """Structured compile output returned by a local Codex driver."""

    status: str
    summary: str
    review: str | None = None
    final_message: str | None = None
    artifacts: tuple[CodexCompileArtifact, ...] = ()
    events: tuple[CodexCompileEvent, ...] = ()

    def __post_init__(self) -> None:
        normalized_status = _text(self.status, field_name="status").lower()
        if normalized_status not in _COMPILE_RESULT_STATUSES:
            raise ValueError(f"status must be one of {sorted(_COMPILE_RESULT_STATUSES)}")
        object.__setattr__(self, "status", normalized_status)
        object.__setattr__(self, "summary", _text(self.summary, field_name="summary"))
        review = _optional_text(self.review, field_name="review", trim=False)
        object.__setattr__(self, "review", review)
        final_message = _optional_text(self.final_message, field_name="final_message", trim=False)  # AUDIT-FIX(#2): Preserve the exact raw driver payload for diagnostics/replay.
        object.__setattr__(self, "final_message", final_message)
        artifacts = tuple(self.artifacts)
        if not all(isinstance(item, CodexCompileArtifact) for item in artifacts):
            raise TypeError("artifacts must contain CodexCompileArtifact items only")
        object.__setattr__(self, "artifacts", artifacts)
        events = tuple(self.events)
        if not all(isinstance(item, CodexCompileEvent) for item in events):
            raise TypeError("events must contain CodexCompileEvent items only")
        object.__setattr__(self, "events", events)


@dataclass(frozen=True, slots=True)
class CodexCompileRequest:
    """Describe one compile job handed to a local Codex driver."""

    job: CompileJobRecord
    session: RequirementsDialogueSession
    prompt: str
    output_schema: dict[str, Any]
    workspace_root: str
    request_path: str
    output_schema_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.job, CompileJobRecord):
            raise TypeError("job must be a CompileJobRecord")
        if not isinstance(self.session, RequirementsDialogueSession):
            raise TypeError("session must be a RequirementsDialogueSession")
        object.__setattr__(self, "prompt", _text(self.prompt, field_name="prompt", trim=False))  # AUDIT-FIX(#2): Preserve exact prompt formatting passed to the local driver.
        object.__setattr__(self, "output_schema", _json_mapping(self.output_schema, field_name="output_schema"))
        workspace_root = _resolve_path(  # AUDIT-FIX(#1): Resolve workspace_root once and use it as the only anchor for child paths.
            Path(_pathlike_text(self.workspace_root, field_name="workspace_root")).expanduser(),
            field_name="workspace_root",
        )
        if workspace_root.exists() and not workspace_root.is_dir():
            raise ValueError("workspace_root must be a directory path")
        object.__setattr__(self, "workspace_root", str(workspace_root))
        request_path = _resolve_workspace_path(self.request_path, field_name="request_path", workspace_root=workspace_root)  # AUDIT-FIX(#1): Block traversal/absolute escapes for request_path.
        object.__setattr__(self, "request_path", request_path)
        output_schema_path = _resolve_workspace_path(  # AUDIT-FIX(#1): Block traversal/absolute escapes for output_schema_path.
            self.output_schema_path,
            field_name="output_schema_path",
            workspace_root=workspace_root,
        )
        if request_path == output_schema_path:
            raise ValueError("request_path and output_schema_path must be different files")  # AUDIT-FIX(#1): Prevent self-overwrite/collision between request and schema files.
        object.__setattr__(self, "output_schema_path", output_schema_path)


class CodexDriverError(RuntimeError):
    """Base error for local Codex driver failures."""


class CodexDriverUnavailableError(CodexDriverError):
    """Raised when a local Codex driver cannot be started or found."""


class CodexDriverProtocolError(CodexDriverError):
    """Raised when a local Codex driver returns malformed output."""


def compile_output_schema() -> dict[str, Any]:
    """Return the strict JSON schema used for Codex compile responses."""

    return {
        "type": "object",
        "required": ["status", "summary", "review", "artifacts"],
        "additionalProperties": False,
        "properties": {
            "status": {
                "type": "string",
                "enum": sorted(_COMPILE_RESULT_STATUSES),
            },
            "summary": {"type": "string", "minLength": 1},  # AUDIT-FIX(#8): Align schema with runtime validation for non-empty summary text.
            "review": {"type": ["string", "null"]},
            "artifacts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["kind", "artifact_name", "media_type", "content", "summary"],
                    "additionalProperties": False,
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": sorted(item.value for item in ArtifactKind),
                        },
                        "artifact_name": {"type": "string", "minLength": 1},  # AUDIT-FIX(#8): Align schema with runtime validation for required names.
                        "media_type": {"type": "string", "minLength": 1},  # AUDIT-FIX(#8): Align schema with runtime validation for required media types.
                        "content": {"type": "string"},
                        "summary": {"type": ["string", "null"]},
                        "metadata": {
                            "type": "object",
                            "additionalProperties": False,
                        },  # AUDIT-FIX(#11): Structured outputs require nested objects to be explicitly closed.
                    },
                },
            },
        },
    }


def compile_result_from_text(
    raw_text: str,
    *,
    events: Sequence[CodexCompileEvent] = (),
) -> CodexCompileResult:
    """Parse a strict JSON compile result from one final Codex message."""

    if not isinstance(raw_text, str):  # AUDIT-FIX(#5): Keep parser failure mode explicit before attempting JSON decode.
        raise TypeError("raw_text must be a string")

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise CodexDriverProtocolError("Codex compile output was not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise CodexDriverProtocolError("Codex compile output must be a JSON object")

    _validate_required_and_allowed_keys(  # AUDIT-FIX(#4): Enforce strict root-field parity with compile_output_schema().
        payload,
        field_name="Codex compile output",
        required_keys=_COMPILE_RESULT_SCHEMA_REQUIRED_FIELDS,
        allowed_keys=_COMPILE_RESULT_SCHEMA_ALLOWED_FIELDS,
    )

    artifacts_payload = payload["artifacts"]
    if not isinstance(artifacts_payload, list):
        raise CodexDriverProtocolError("Codex compile artifacts must be a JSON array")

    artifacts: list[CodexCompileArtifact] = []
    for index, item in enumerate(artifacts_payload):
        if not isinstance(item, Mapping):
            raise CodexDriverProtocolError(f"Codex compile artifact at index {index} must be a JSON object")
        _validate_required_and_allowed_keys(  # AUDIT-FIX(#4): Enforce strict artifact-field parity with compile_output_schema().
            item,
            field_name=f"Codex compile artifact at index {index}",
            required_keys=_COMPILE_ARTIFACT_REQUIRED_FIELDS,
            allowed_keys=_COMPILE_ARTIFACT_ALLOWED_FIELDS,
        )
        try:
            artifacts.append(
                CodexCompileArtifact(
                    kind=item["kind"],
                    artifact_name=item["artifact_name"],
                    media_type=item["media_type"],
                    content=item["content"],
                    summary=item["summary"],
                    metadata=item.get("metadata", {}),
                )
            )
        except (TypeError, ValueError) as exc:
            raise CodexDriverProtocolError(f"Codex compile artifact at index {index} failed validation: {exc}") from exc  # AUDIT-FIX(#5): Translate constructor failures into the module's typed protocol error.

    try:
        return CodexCompileResult(
            status=payload["status"],
            summary=payload["summary"],
            review=payload["review"],
            final_message=raw_text,
            artifacts=tuple(artifacts),
            events=tuple(events),
        )
    except (TypeError, ValueError) as exc:
        raise CodexDriverProtocolError(f"Codex compile output failed validation: {exc}") from exc  # AUDIT-FIX(#5): Keep malformed driver output recoverable for callers.
