"""Define the shared metadata model for self_coding module shims.

The first ASE module-library slice is intentionally metadata-first. Each module
file exposes importable placeholder functions for future sandboxed skills plus
one structured spec object that the deterministic capability registry and
compile prompt builder can read today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

from twinr.text_utils import is_valid_stable_identifier

from twinr.agent.self_coding.contracts import CapabilityDefinition
from twinr.agent.self_coding.status import CapabilityRiskClass


_MAX_SIGNATURE_LENGTH = 160  # AUDIT-FIX(#2): Centralize inline-text limits so validation stays deterministic across constructors.
_MAX_SUMMARY_LENGTH = 220  # AUDIT-FIX(#2): Centralize inline-text limits so validation stays deterministic across constructors.
_MAX_RETURNS_LENGTH = 160  # AUDIT-FIX(#2): Centralize inline-text limits so validation stays deterministic across constructors.
_MAX_DOC_HEADER_LENGTH = 4000  # AUDIT-FIX(#3): Bound generated/custom headers to protect prompt size and RPi memory headroom.
_MAX_API_NAME_LENGTH = 160  # AUDIT-FIX(#5): Keep placeholder error messages short and log-safe.


def _coerce_iterable(value: object, *, field_name: str) -> tuple[object, ...]:
    """Normalize iterable metadata fields and reject unstable container types."""

    if value is None:
        raise TypeError(f"{field_name} must be an iterable")
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable, not a string-like value")
    if isinstance(value, (dict, set, frozenset)):  # AUDIT-FIX(#1): Reject unordered/bag-like containers so metadata order stays deterministic.
        raise TypeError(f"{field_name} must preserve order; use a list or tuple")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable") from exc


def _require_identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):  # AUDIT-FIX(#2): Fail fast on non-string metadata instead of silently stringifying authoring mistakes.
        raise TypeError(f"{field_name} must be a string")
    text = value.strip().lower()
    if not is_valid_stable_identifier(text):
        raise ValueError(f"{field_name} must be a stable identifier")
    return text


def _require_text(value: object, *, field_name: str, limit: int) -> str:
    if not isinstance(value, str):  # AUDIT-FIX(#2): Fail fast on non-string metadata instead of silently stringifying authoring mistakes.
        raise TypeError(f"{field_name} must be a string")
    text = " ".join(value.split())  # AUDIT-FIX(#2): Collapse multiline/control whitespace so prompt-facing metadata stays one-line and deterministic.
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    if len(text) > limit:
        raise ValueError(f"{field_name} must be <= {limit} characters")
    return text


def _normalize_doc_header(value: object, *, field_name: str, limit: int) -> str:
    """Accept only bounded string doc headers while preserving intentional line breaks."""

    if value is None:
        return ""
    if not isinstance(value, str):  # AUDIT-FIX(#3): Reject silent fallback on invalid doc_header types so bad metadata is caught at import time.
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        return ""
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()  # AUDIT-FIX(#3): Preserve authored multiline headers but strip noisy trailing whitespace.
    if len(normalized) > limit:
        raise ValueError(f"{field_name} must be <= {limit} characters")
    return normalized


def _normalize_tags(values: object) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in _coerce_iterable(values, field_name="tags"):  # AUDIT-FIX(#1): Reject None, string-like, and unordered containers before tag normalization.
        value = _require_identifier(raw_value, field_name="tags")
        if value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return tuple(normalized)


class SelfCodingModuleRuntimeUnavailableError(RuntimeError):
    """Raised when a placeholder self_coding module is called outside the skill runner."""


def runtime_unavailable(api_name: str) -> NoReturn:
    """Raise the standard error for placeholder module functions."""

    normalized_api_name = _require_text(api_name, field_name="api_name", limit=_MAX_API_NAME_LENGTH)  # AUDIT-FIX(#5): Sanitize placeholder API names before surfacing them in exceptions/logs.
    raise SelfCodingModuleRuntimeUnavailableError(
        f"{normalized_api_name} is only available inside the future sandboxed self_coding runtime."
    )


@dataclass(frozen=True, slots=True)
class SelfCodingModuleFunction:
    """Describe one public function exported by a self_coding module shim."""

    name: str
    signature: str
    summary: str
    returns: str = ""
    effectful: bool = False
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_identifier(self.name, field_name="name"))
        object.__setattr__(self, "signature", _require_text(self.signature, field_name="signature", limit=_MAX_SIGNATURE_LENGTH))
        object.__setattr__(self, "summary", _require_text(self.summary, field_name="summary", limit=_MAX_SUMMARY_LENGTH))
        if not isinstance(self.returns, str):  # AUDIT-FIX(#2): Prevent falsey non-string values from silently becoming an empty returns description.
            raise TypeError("returns must be a string")
        object.__setattr__(self, "returns", _require_text(self.returns, field_name="returns", limit=_MAX_RETURNS_LENGTH) if self.returns else "")
        if not isinstance(self.effectful, bool):
            raise TypeError("effectful must be a boolean")
        object.__setattr__(self, "tags", _normalize_tags(self.tags))  # AUDIT-FIX(#1): Validate iterable tags through the shared coercion path.

    def descriptor_line(self) -> str:
        """Render one Codex-readable API line for prompts and docs."""

        details = self.summary
        if self.returns:
            details = f"{details} Returns {self.returns}."
        if self.effectful:
            details = f"{details} Effectful."
        return f"- {self.signature}: {details}"


def _normalize_public_api(values: object) -> tuple[SelfCodingModuleFunction, ...]:
    """Validate the public API collection and enforce unique function names."""

    normalized_api: list[SelfCodingModuleFunction] = []
    seen_names: set[str] = set()
    raw_values = _coerce_iterable(values, field_name="public_api")  # AUDIT-FIX(#1): Reject None, string-like, and unordered containers before API normalization.
    if not raw_values:
        raise ValueError("public_api must not be empty")
    for function in raw_values:
        if not isinstance(function, SelfCodingModuleFunction):
            raise TypeError("public_api items must be SelfCodingModuleFunction")
        if function.name in seen_names:
            raise ValueError(f"public_api function names must be unique: {function.name}")  # AUDIT-FIX(#4): Fail fast on ambiguous API metadata that would confuse the registry and prompt builder.
        normalized_api.append(function)
        seen_names.add(function.name)
    return tuple(normalized_api)


@dataclass(frozen=True, slots=True)
class SelfCodingModuleSpec:
    """Describe one curated self_coding module and its public API."""

    capability_id: str
    module_name: str
    summary: str
    risk_class: CapabilityRiskClass
    public_api: tuple[SelfCodingModuleFunction, ...]
    doc_header: str = ""
    requires_configuration: bool = False
    integration_id: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability_id", _require_identifier(self.capability_id, field_name="capability_id"))
        object.__setattr__(self, "module_name", _require_identifier(self.module_name, field_name="module_name"))
        object.__setattr__(self, "summary", _require_text(self.summary, field_name="summary", limit=_MAX_SUMMARY_LENGTH))
        if not isinstance(self.risk_class, CapabilityRiskClass):
            raise TypeError("risk_class must be a CapabilityRiskClass")
        object.__setattr__(self, "public_api", _normalize_public_api(self.public_api))  # AUDIT-FIX(#1,#4): Normalize iterable handling and reject duplicate exported function names.
        if not isinstance(self.requires_configuration, bool):
            raise TypeError("requires_configuration must be a boolean")
        object.__setattr__(
            self,
            "integration_id",
            None if self.integration_id is None else _require_identifier(self.integration_id, field_name="integration_id"),
        )
        object.__setattr__(self, "tags", _normalize_tags(self.tags))  # AUDIT-FIX(#1): Validate iterable tags through the shared coercion path.
        doc_header = _normalize_doc_header(self.doc_header, field_name="doc_header", limit=_MAX_DOC_HEADER_LENGTH)  # AUDIT-FIX(#3): Bound and type-check custom doc headers instead of silently accepting unbounded blobs.
        object.__setattr__(self, "doc_header", doc_header or build_module_doc_header(self))

    def capability_definition(self) -> CapabilityDefinition:
        """Build the matching registry capability definition."""

        return CapabilityDefinition(
            capability_id=self.capability_id,
            module_name=self.module_name,
            summary=self.summary,
            risk_class=self.risk_class,
            requires_configuration=self.requires_configuration,
            integration_id=self.integration_id,
            tags=self.tags,
        )


def build_module_doc_header(spec: SelfCodingModuleSpec) -> str:
    """Render the Codex-readable module header for one module spec."""

    if not isinstance(spec, SelfCodingModuleSpec):  # AUDIT-FIX(#6): Raise a predictable API-level error instead of an AttributeError on accidental misuse.
        raise TypeError("spec must be a SelfCodingModuleSpec")
    lines = [
        f"Module `{spec.module_name}`: {spec.summary}",
        "",
        "Public API:",
        *(function.descriptor_line() for function in spec.public_api),
    ]
    doc_header = "\n".join(lines)
    if len(doc_header) > _MAX_DOC_HEADER_LENGTH:  # AUDIT-FIX(#3): Cap generated headers too, so oversized public APIs fail fast before prompt construction.
        raise ValueError(f"generated doc_header must be <= {_MAX_DOC_HEADER_LENGTH} characters")
    return doc_header


__all__ = [
    "SelfCodingModuleFunction",
    "SelfCodingModuleRuntimeUnavailableError",
    "SelfCodingModuleSpec",
    "build_module_doc_header",
    "runtime_unavailable",
]