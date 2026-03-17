"""Validate and canonicalize raw Codex skill-package drafts."""

from __future__ import annotations

import json
from typing import Any, Mapping

from twinr.agent.self_coding.contracts import CompileJobRecord, RequirementsDialogueSession
from twinr.agent.self_coding.runtime import (
    CompiledSkillPackage,
    canonical_skill_package_document,
    skill_package_from_payload,
)
from twinr.text_utils import extract_json_object

_MAX_RAW_CONTENT_CHARS = 128_000
_MAX_JSON_NESTING_DEPTH = 64  # AUDIT-FIX(#5): Bound structural complexity beyond raw character count.
_MAX_JSON_CONTAINER_ITEMS = 10_000  # AUDIT-FIX(#5): Bound total mapping/list items to cap CPU on malformed drafts.
_MISSING: Any = object()  # AUDIT-FIX(#2): Sentinel distinguishes absent wrapper keys from explicit nulls.


class SkillPackageCompilerError(ValueError):
    """Raised when a raw skill-package draft cannot be compiled safely."""


def _select_package_payload(payload: Mapping[str, Any]) -> Any:
    """Select the actual package payload from a direct or wrapped draft."""

    # AUDIT-FIX(#2): Only genuinely missing wrapper keys may fall back to the root payload.
    skill_package_payload = payload.get("skill_package", _MISSING)
    if skill_package_payload is not _MISSING and skill_package_payload is not None:
        return skill_package_payload

    package_payload = payload.get("package", _MISSING)
    if package_payload is not _MISSING and package_payload is not None:
        return package_payload

    if skill_package_payload is None:
        raise SkillPackageCompilerError('"skill_package" must not be null')
    if package_payload is None:
        raise SkillPackageCompilerError('"package" must not be null')

    return payload


def _validate_json_structure(value: Any) -> None:
    """Reject structurally abusive JSON-like payloads before downstream validation."""

    # AUDIT-FIX(#5): Use an iterative walk so the guard itself does not introduce recursive failure modes.
    stack: list[tuple[Any, int]] = [(value, 1)]
    container_items = 0

    while stack:
        current, depth = stack.pop()
        if depth > _MAX_JSON_NESTING_DEPTH:
            raise SkillPackageCompilerError(
                f"skill package draft exceeds maximum nesting depth of {_MAX_JSON_NESTING_DEPTH}"
            )

        if isinstance(current, Mapping):
            container_items += len(current)
            if container_items > _MAX_JSON_CONTAINER_ITEMS:
                raise SkillPackageCompilerError(
                    f"skill package draft exceeds maximum structural size of {_MAX_JSON_CONTAINER_ITEMS} items"
                )

            for key, nested_value in current.items():
                if not isinstance(key, str):
                    raise SkillPackageCompilerError("skill package draft contains a non-string object key")
                stack.append((nested_value, depth + 1))
        elif isinstance(current, list):
            container_items += len(current)
            if container_items > _MAX_JSON_CONTAINER_ITEMS:
                raise SkillPackageCompilerError(
                    f"skill package draft exceeds maximum structural size of {_MAX_JSON_CONTAINER_ITEMS} items"
                )

            for nested_value in current:
                stack.append((nested_value, depth + 1))


def compile_skill_package_content(
    *,
    job: CompileJobRecord,
    session: RequirementsDialogueSession,
    raw_content: str,
) -> CompiledSkillPackage:
    """Validate and canonicalize one raw skill-package draft."""

    # AUDIT-FIX(#3): Fail fast on missing context instead of misreporting deep canonicalization errors as payload issues.
    if job is None:
        raise SkillPackageCompilerError("compile job is required")
    # AUDIT-FIX(#3): Fail fast on missing context instead of misreporting deep canonicalization errors as payload issues.
    if session is None:
        raise SkillPackageCompilerError("requirements dialogue session is required")

    if not isinstance(raw_content, str):
        raise SkillPackageCompilerError("skill package draft must be a string")
    if len(raw_content) > _MAX_RAW_CONTENT_CHARS:
        raise SkillPackageCompilerError(f"skill package draft exceeds {_MAX_RAW_CONTENT_CHARS} characters")
    # AUDIT-FIX(#4): Reject blank drafts deterministically before JSON extraction to avoid parser-dependent error surfaces.
    if not raw_content.strip():
        raise SkillPackageCompilerError("skill package draft must not be empty")

    # AUDIT-FIX(#1): Separate JSON extraction from later stages so internal canonicalization faults are not mislabeled as invalid input.
    try:
        payload_object = extract_json_object(raw_content)
    except json.JSONDecodeError as exc:
        raise SkillPackageCompilerError("skill package draft must contain a valid JSON object") from exc
    # AUDIT-FIX(#5): Convert deep-recursion parser failures into deterministic validation errors on constrained hardware.
    except RecursionError as exc:
        raise SkillPackageCompilerError("skill package draft is too deeply nested to compile safely") from exc
    except (TypeError, ValueError) as exc:
        raise SkillPackageCompilerError("skill package draft could not be parsed safely") from exc

    if not isinstance(payload_object, Mapping):
        raise SkillPackageCompilerError("skill package draft must be a JSON object")

    # AUDIT-FIX(#1): Materialize the parsed mapping inside the validated stage boundary to keep error attribution precise.
    try:
        payload = dict(payload_object)
    except (TypeError, ValueError) as exc:
        raise SkillPackageCompilerError("skill package draft could not be materialized safely") from exc

    # AUDIT-FIX(#5): Bound nested/container-heavy payloads before downstream validation to cap CPU and stack usage.
    _validate_json_structure(payload)
    # AUDIT-FIX(#2): Distinguish missing wrapper keys from explicit null wrapper values to avoid mis-parsing malformed drafts.
    package_payload = _select_package_payload(payload)

    # AUDIT-FIX(#1): Keep payload-validation failures stage-specific and preserve exception chaining for operator diagnostics.
    try:
        package = skill_package_from_payload(package_payload)
    except json.JSONDecodeError as exc:
        raise SkillPackageCompilerError("skill package payload contains invalid JSON") from exc
    # AUDIT-FIX(#5): Protect the compiler boundary from recursion-driven failures in downstream validators.
    except RecursionError as exc:
        raise SkillPackageCompilerError("skill package payload is too deeply nested to validate safely") from exc
    except (TypeError, ValueError) as exc:
        raise SkillPackageCompilerError("skill package payload is invalid") from exc

    # AUDIT-FIX(#1): Canonicalization faults are reported as canonicalization failures, not as generic invalid-input errors.
    try:
        return canonical_skill_package_document(job=job, session=session, package=package)
    except RecursionError as exc:
        raise SkillPackageCompilerError("skill package is too deeply nested to canonicalize safely") from exc
    except (TypeError, ValueError) as exc:
        raise SkillPackageCompilerError("skill package failed canonicalization") from exc


__all__ = [
    "SkillPackageCompilerError",
    "compile_skill_package_content",
]