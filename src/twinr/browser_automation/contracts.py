"""Define the stable Twinr browser-automation request/response contracts.

Concrete browser stacks stay outside the committed repo tree and are loaded
through the local workspace bridge in ``loader.py``. This module owns the
versioned in/out surface that Twinr code may depend on safely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Protocol, runtime_checkable

_MAX_SHORT_TEXT = 240
_MAX_LONG_TEXT = 4_000
_MAX_PATH_TEXT = 512
_MAX_DOMAIN_TEXT = 253
_MAX_ARTIFACTS = 16
_MAX_ALLOWED_DOMAINS = 16

BROWSER_AUTOMATION_STATUSES = frozenset({"completed", "blocked", "failed", "cancelled"})


def _text(
    value: object,
    *,
    field_name: str,
    allow_empty: bool = False,
    limit: int = _MAX_SHORT_TEXT,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > limit:
        raise ValueError(f"{field_name} must be <= {limit} characters")
    return normalized


def _optional_text(value: object | None, *, field_name: str, limit: int = _MAX_SHORT_TEXT) -> str | None:
    if value is None:
        return None
    normalized = _text(value, field_name=field_name, allow_empty=True, limit=limit)
    return normalized or None


def _stable_identifier(value: object, *, field_name: str) -> str:
    text = _text(value, field_name=field_name, limit=64)
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if any(char not in allowed for char in text):
        raise ValueError(f"{field_name} must use only letters, digits, '_' or '-'")
    return text


def _positive_int(value: object, *, field_name: str, minimum: int = 1, maximum: int = 128) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}")
    return value


def _positive_float(value: object, *, field_name: str, minimum: float = 0.1, maximum: float = 600.0) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a finite number")
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}")
    return number


def _json_object(value: object, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        payload = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be JSON serializable") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{field_name} must be a JSON object")
    return payload


def _allowed_domains(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (tuple, list)):
        raise TypeError("allowed_domains must be a list or tuple")
    normalized: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        domain = _text(item, field_name=f"allowed_domains[{index}]", limit=_MAX_DOMAIN_TEXT).lower()
        if any(char.isspace() for char in domain) or "/" in domain:
            raise ValueError(f"allowed_domains[{index}] must look like a host name")
        if domain in seen:
            continue
        seen.add(domain)
        normalized.append(domain)
    if len(normalized) > _MAX_ALLOWED_DOMAINS:
        raise ValueError(f"allowed_domains must contain at most {_MAX_ALLOWED_DOMAINS} entries")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class BrowserAutomationArtifact:
    """Describe one artifact produced by a browser automation run."""

    kind: str
    path: str
    content_type: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _stable_identifier(self.kind, field_name="kind"))
        object.__setattr__(self, "path", _text(self.path, field_name="path", limit=_MAX_PATH_TEXT))
        object.__setattr__(
            self,
            "content_type",
            _optional_text(self.content_type, field_name="content_type", limit=128),
        )
        object.__setattr__(
            self,
            "description",
            _optional_text(self.description, field_name="description", limit=_MAX_SHORT_TEXT),
        )


@dataclass(frozen=True, slots=True)
class BrowserAutomationRequest:
    """Describe one bounded browser task Twinr may hand to an external driver."""

    task_id: str
    goal: str
    start_url: str | None = None
    allowed_domains: tuple[str, ...] = ()
    max_steps: int = 6
    max_runtime_s: float = 45.0
    capture_screenshot: bool = True
    capture_html: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _stable_identifier(self.task_id, field_name="task_id"))
        object.__setattr__(self, "goal", _text(self.goal, field_name="goal", limit=_MAX_LONG_TEXT))
        object.__setattr__(self, "start_url", _optional_text(self.start_url, field_name="start_url", limit=_MAX_LONG_TEXT))
        object.__setattr__(self, "allowed_domains", _allowed_domains(self.allowed_domains))
        object.__setattr__(self, "max_steps", _positive_int(self.max_steps, field_name="max_steps", minimum=1, maximum=32))
        object.__setattr__(
            self,
            "max_runtime_s",
            _positive_float(self.max_runtime_s, field_name="max_runtime_s", minimum=0.1, maximum=900.0),
        )
        object.__setattr__(self, "capture_screenshot", bool(self.capture_screenshot))
        object.__setattr__(self, "capture_html", bool(self.capture_html))
        object.__setattr__(self, "metadata", _json_object(self.metadata, field_name="metadata"))


@dataclass(frozen=True, slots=True)
class BrowserAutomationResult:
    """Normalize the small result payload returned by a browser automation run."""

    ok: bool
    status: str
    summary: str
    final_url: str | None = None
    error_code: str | None = None
    artifacts: tuple[BrowserAutomationArtifact, ...] = ()
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ok", bool(self.ok))
        status = _stable_identifier(self.status, field_name="status").lower()
        if status not in BROWSER_AUTOMATION_STATUSES:
            raise ValueError(f"status must be one of {sorted(BROWSER_AUTOMATION_STATUSES)}")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "summary", _text(self.summary, field_name="summary", limit=_MAX_LONG_TEXT))
        object.__setattr__(self, "final_url", _optional_text(self.final_url, field_name="final_url", limit=_MAX_LONG_TEXT))
        object.__setattr__(
            self,
            "error_code",
            _optional_text(self.error_code, field_name="error_code", limit=64),
        )
        artifacts = tuple(self.artifacts)
        if len(artifacts) > _MAX_ARTIFACTS:
            raise ValueError(f"artifacts must contain at most {_MAX_ARTIFACTS} entries")
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, BrowserAutomationArtifact):
                raise TypeError(f"artifacts[{index}] must be BrowserAutomationArtifact")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "data", _json_object(self.data, field_name="data"))


@dataclass(frozen=True, slots=True)
class BrowserAutomationAvailability:
    """Summarize whether the optional browser automation backend is usable."""

    enabled: bool
    available: bool
    reason: str = ""
    workspace_path: str | None = None
    entry_module: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "available", bool(self.available))
        object.__setattr__(
            self,
            "reason",
            _text(self.reason or "", field_name="reason", allow_empty=True, limit=_MAX_LONG_TEXT),
        )
        object.__setattr__(
            self,
            "workspace_path",
            _optional_text(self.workspace_path, field_name="workspace_path", limit=_MAX_PATH_TEXT),
        )
        object.__setattr__(
            self,
            "entry_module",
            _optional_text(self.entry_module, field_name="entry_module", limit=_MAX_PATH_TEXT),
        )


@runtime_checkable
class BrowserAutomationDriver(Protocol):
    """Runtime protocol implemented by local browser automation drivers."""

    def availability(self) -> BrowserAutomationAvailability:
        """Return one small readiness snapshot for operator/debug surfaces."""

    def execute(self, request: BrowserAutomationRequest) -> BrowserAutomationResult:
        """Execute one bounded browser automation request."""


__all__ = [
    "BROWSER_AUTOMATION_STATUSES",
    "BrowserAutomationArtifact",
    "BrowserAutomationAvailability",
    "BrowserAutomationDriver",
    "BrowserAutomationRequest",
    "BrowserAutomationResult",
]
