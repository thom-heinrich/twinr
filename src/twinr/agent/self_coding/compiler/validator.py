"""Validate target artifacts deterministically before persistence/activation."""

from __future__ import annotations

import json
import os
from typing import Any

from twinr.agent.self_coding.codex_driver import CodexCompileArtifact
from twinr.agent.self_coding.compiler.automation_target import compile_automation_manifest_content
from twinr.agent.self_coding.compiler.skill_target import compile_skill_package_content
from twinr.agent.self_coding.contracts import CompileJobRecord, RequirementsDialogueSession
from twinr.agent.self_coding.status import ArtifactKind, CompileTarget

_DEFAULT_MAX_COMPILE_ARTIFACT_BYTES = 1_048_576


class CompileArtifactValidationError(ValueError):
    """Raised when a compile artifact does not match the requested target."""


def validate_compile_artifact(
    *,
    job: CompileJobRecord,
    session: RequirementsDialogueSession,
    artifact: CodexCompileArtifact,
) -> CodexCompileArtifact:
    """Validate and canonicalize one target artifact for the current compile job."""

    requested_target = _normalize_compile_target(getattr(job, "requested_target", None))  # AUDIT-FIX(#2): Normalize enum-like target values before any `.value` access.
    expected_kind = _target_artifact_kind(requested_target)
    artifact_kind = _normalize_artifact_kind(getattr(artifact, "kind", None))  # AUDIT-FIX(#2): Normalize enum-like artifact kinds and fail with a domain error on malformed input.
    if artifact_kind != expected_kind:
        raise CompileArtifactValidationError(
            f"compile target {requested_target.value!r} requires artifact kind {expected_kind.value!r}, "
            f"got {artifact_kind.value!r}"
        )

    raw_content = _coerce_text("artifact.content", getattr(artifact, "content", None), allow_bytes=True)  # AUDIT-FIX(#5): Validate textual content before delegating to downstream compilers.
    _ensure_artifact_size_within_limit("artifact.content", raw_content)  # AUDIT-FIX(#6): Bound untrusted artifact size to protect the single-process RPi runtime.
    compiled = _compile_target_content(
        expected_kind=expected_kind,
        job=job,
        session=session,
        raw_content=raw_content,
    )  # AUDIT-FIX(#3): Wrap downstream compiler failures in a stable domain error.

    compiled_content = _coerce_text("compiled.content", getattr(compiled, "content", None), allow_bytes=True)  # AUDIT-FIX(#5): Validate compiler output before persistence/activation.
    _ensure_artifact_size_within_limit("compiled.content", compiled_content)  # AUDIT-FIX(#6): Reject oversized compiled artifacts before JSON parsing/persistence.
    canonical_content = _canonicalize_json_content(compiled_content)  # AUDIT-FIX(#7): Enforce strict valid JSON and deterministic serialization.
    compiled_summary = _coerce_optional_text("compiled.summary", getattr(compiled, "summary", None))  # AUDIT-FIX(#5): Preserve explicit empty summaries instead of truthiness-based fallback.
    artifact_summary = _coerce_optional_text("artifact.summary", getattr(artifact, "summary", None))
    artifact_name = _canonical_artifact_name(getattr(artifact, "artifact_name", None), expected_kind)  # AUDIT-FIX(#1): Reject traversal/control-character filenames and supply a safe default name.
    artifact_metadata = _coerce_metadata_mapping("artifact.metadata", getattr(artifact, "metadata", None))  # AUDIT-FIX(#4): Validate metadata shape before merging.
    compiled_metadata = _coerce_metadata_mapping("compiled.metadata", getattr(compiled, "metadata", None))

    return CodexCompileArtifact(
        kind=expected_kind,
        artifact_name=artifact_name,
        media_type="application/json",
        content=canonical_content,
        summary=compiled_summary if compiled_summary is not None else artifact_summary,
        metadata={
            **artifact_metadata,
            **compiled_metadata,
            "artifact_kind": expected_kind.value,
        },
    )


def _compile_target_content(
    *,
    expected_kind: ArtifactKind,
    job: CompileJobRecord,
    session: RequirementsDialogueSession,
    raw_content: str,
) -> Any:
    # AUDIT-FIX(#3): Convert downstream compiler failures into a stable validation error without leaking raw internal exception text.
    try:
        if expected_kind == ArtifactKind.AUTOMATION_MANIFEST:
            return compile_automation_manifest_content(
                job=job,
                session=session,
                raw_content=raw_content,
            )
        if expected_kind == ArtifactKind.SKILL_PACKAGE:
            return compile_skill_package_content(
                job=job,
                session=session,
                raw_content=raw_content,
            )
    except CompileArtifactValidationError:
        raise
    except Exception as exc:
        human_kind = expected_kind.value.replace("_", " ")
        raise CompileArtifactValidationError(f"failed to compile {human_kind} content") from exc

    raise CompileArtifactValidationError(f"unsupported compile target artifact kind: {expected_kind.value}")


def _target_artifact_kind(target: CompileTarget) -> ArtifactKind:
    normalized_target = _normalize_compile_target(target)  # AUDIT-FIX(#2): Guarantee predictable enum handling even for deserialized/raw target values.
    if normalized_target == CompileTarget.AUTOMATION_MANIFEST:
        return ArtifactKind.AUTOMATION_MANIFEST
    if normalized_target == CompileTarget.SKILL_PACKAGE:
        return ArtifactKind.SKILL_PACKAGE
    raise CompileArtifactValidationError(f"unsupported compile target: {normalized_target!r}")


def _normalize_compile_target(target: object) -> CompileTarget:
    # AUDIT-FIX(#2): Accept enum instances, raw values, or enum names while rejecting malformed targets with a domain-specific error.
    if isinstance(target, CompileTarget):
        return target
    if isinstance(target, str):
        try:
            return CompileTarget(target)
        except ValueError:
            try:
                return CompileTarget[target]
            except KeyError:
                pass
    try:
        return CompileTarget(target)
    except Exception as exc:
        raise CompileArtifactValidationError(f"unsupported compile target: {target!r}") from exc


def _normalize_artifact_kind(kind: object) -> ArtifactKind:
    # AUDIT-FIX(#2): Accept enum instances, raw values, or enum names while rejecting malformed artifact kinds cleanly.
    if isinstance(kind, ArtifactKind):
        return kind
    if isinstance(kind, str):
        try:
            return ArtifactKind(kind)
        except ValueError:
            try:
                return ArtifactKind[kind]
            except KeyError:
                pass
    try:
        return ArtifactKind(kind)
    except Exception as exc:
        raise CompileArtifactValidationError(f"unsupported artifact kind: {kind!r}") from exc


def _coerce_text(field_name: str, value: object, *, allow_bytes: bool = False) -> str:
    # AUDIT-FIX(#5): Validate text-like fields explicitly instead of relying on downstream implicit conversions or crashes.
    if isinstance(value, str):
        return value
    if allow_bytes and isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CompileArtifactValidationError(f"{field_name} must be valid UTF-8 text") from exc
    raise CompileArtifactValidationError(f"{field_name} must be text, got {type(value).__name__}")


def _coerce_optional_text(field_name: str, value: object) -> str | None:
    # AUDIT-FIX(#5): Preserve `None` distinctly so callers can tell the difference between missing and intentionally empty text.
    if value is None:
        return None
    return _coerce_text(field_name, value, allow_bytes=True)


def _coerce_metadata_mapping(field_name: str, value: object) -> dict[str, Any]:
    # AUDIT-FIX(#4): Coerce mapping-like metadata deterministically and fail with a validation error on incompatible shapes.
    if value is None:
        return {}
    try:
        mapping = dict(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise CompileArtifactValidationError(
            f"{field_name} must be a mapping-compatible object, got {type(value).__name__}"
        ) from exc
    return {str(key): item for key, item in mapping.items()}


def _json_object_pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # AUDIT-FIX(#7): Reject duplicate JSON object keys instead of silently discarding earlier values during canonicalization.
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CompileArtifactValidationError(f"compiled artifact content contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonicalize_json_content(content: str) -> str:
    # AUDIT-FIX(#7): Parse and re-serialize JSON to guarantee strict validity and deterministic persistence bytes.
    try:
        parsed = json.loads(content, object_pairs_hook=_json_object_pairs_no_duplicates)
        return json.dumps(parsed, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except CompileArtifactValidationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CompileArtifactValidationError("compiled artifact content must be valid JSON") from exc


def _canonical_artifact_name(value: object, expected_kind: ArtifactKind) -> str:
    # AUDIT-FIX(#1): Reject traversal/control-character filenames while providing a safe default when the name is missing.
    default_name = f"{expected_kind.value}.json"
    if value is None:
        return default_name

    name = _coerce_text("artifact.artifact_name", value, allow_bytes=True).strip()
    if not name:
        return default_name
    if name in {".", ".."}:
        raise CompileArtifactValidationError("artifact.artifact_name must not be '.' or '..'")
    if os.path.isabs(name) or "/" in name or "\\" in name:
        raise CompileArtifactValidationError("artifact.artifact_name must be a simple file name, not a path")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
        raise CompileArtifactValidationError("artifact.artifact_name contains control characters")
    if len(name) > 255:
        raise CompileArtifactValidationError("artifact.artifact_name exceeds 255 characters")
    return name


def _ensure_artifact_size_within_limit(field_name: str, text: str) -> None:
    # AUDIT-FIX(#6): Fail fast on oversized artifacts so validation cannot starve the single-process device runtime.
    limit = _max_compile_artifact_bytes()
    if len(text) > limit:
        raise CompileArtifactValidationError(f"{field_name} exceeds size limit of {limit} bytes")
    if len(text.encode("utf-8")) > limit:
        raise CompileArtifactValidationError(f"{field_name} exceeds size limit of {limit} bytes")


def _max_compile_artifact_bytes() -> int:
    # AUDIT-FIX(#6): New env knob is optional and falls back safely, preserving backward compatibility with existing `.env` files.
    raw_value = os.getenv("TWINR_MAX_COMPILE_ARTIFACT_BYTES")
    if raw_value is None:
        return _DEFAULT_MAX_COMPILE_ARTIFACT_BYTES
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_COMPILE_ARTIFACT_BYTES
    return parsed if parsed > 0 else _DEFAULT_MAX_COMPILE_ARTIFACT_BYTES


__all__ = ["CompileArtifactValidationError", "validate_compile_artifact"]
