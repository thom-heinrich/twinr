"""Expose managed email primitives for future self_coding skills."""

from __future__ import annotations

from typing import Final, NoReturn

from twinr.agent.self_coding.status import CapabilityRiskClass

from .base import SelfCodingModuleFunction, SelfCodingModuleSpec, runtime_unavailable

# AUDIT-FIX(#2): Centralize explicit bounds for the documented "bounded" unread mailbox surface.
_DEFAULT_UNREAD_LIMIT: Final[int] = 10
_MAX_UNREAD_LIMIT: Final[int] = 50

# AUDIT-FIX(#3): Bound effectful draft inputs before a real mailbox backend is attached.
_MAX_RECIPIENT_LENGTH: Final[int] = 320
_MAX_SUBJECT_LENGTH: Final[int] = 998
_MAX_BODY_LENGTH: Final[int] = 20_000


# AUDIT-FIX(#1): Make runtime-unavailable behavior fail-closed even if the imported helper unexpectedly returns.
def _raise_runtime_unavailable(operation: str) -> NoReturn:
    """Raise the runtime-unavailable error in a deterministic, fail-closed way."""

    outcome = runtime_unavailable(operation)
    if isinstance(outcome, BaseException):
        raise outcome
    raise RuntimeError(f"{operation} is unavailable in this runtime.")


# AUDIT-FIX(#2): Validate the documented limit contract up front instead of accepting invalid or unbounded inputs.
def _validate_limit(limit: int) -> int:
    """Validate a bounded unread-mail limit."""

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if limit < 0:
        raise ValueError("limit must be >= 0")
    if limit > _MAX_UNREAD_LIMIT:
        raise ValueError(f"limit must be <= {_MAX_UNREAD_LIMIT}")
    return limit


# AUDIT-FIX(#3): Reject malformed header-like text early so future email backends do not fail deep in the stack.
def _validate_header_like_text(
    value: str,
    field_name: str,
    *,
    allow_empty: bool,
    max_length: int,
) -> str:
    """Validate recipient/subject-style text."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain NUL bytes")
    if "\r" in value or "\n" in value:
        raise ValueError(f"{field_name} must not contain CR or LF characters")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    if len(value) > max_length:
        raise ValueError(f"{field_name} must be <= {max_length} characters")
    return value


# AUDIT-FIX(#3): Bound and sanity-check free-form draft content before it reaches any effectful integration.
def _validate_body(body: str) -> str:
    """Validate free-form email body text."""

    if not isinstance(body, str):
        raise TypeError("body must be a string")
    if "\x00" in body:
        raise ValueError("body must not contain NUL bytes")
    if len(body) > _MAX_BODY_LENGTH:
        raise ValueError(f"body must be <= {_MAX_BODY_LENGTH} characters")
    return body


def get_unread(limit: int = _DEFAULT_UNREAD_LIMIT) -> list[dict[str, str]]:
    """Return a bounded list of unread emails."""

    # AUDIT-FIX(#2): Enforce the documented bounded input contract.
    _validate_limit(limit)
    # AUDIT-FIX(#1): Ensure this stub always fails deterministically instead of returning None.
    _raise_runtime_unavailable("email.get_unread")


def summarize_unread(limit: int = _DEFAULT_UNREAD_LIMIT) -> str:
    """Summarize a bounded set of unread emails in plain language."""

    # AUDIT-FIX(#2): Enforce the documented bounded input contract.
    _validate_limit(limit)
    # AUDIT-FIX(#1): Ensure this stub always fails deterministically instead of returning None.
    _raise_runtime_unavailable("email.summarize_unread")


def draft_reply(recipient: str, subject: str, body: str) -> dict[str, str]:
    """Prepare one bounded email draft for later review or confirmation."""

    # AUDIT-FIX(#3): Validate effectful draft parameters before a mailbox backend exists.
    _validate_header_like_text(
        recipient,
        "recipient",
        allow_empty=False,
        max_length=_MAX_RECIPIENT_LENGTH,
    )
    # AUDIT-FIX(#3): Keep subject input within a mailbox-safe header bound.
    _validate_header_like_text(
        subject,
        "subject",
        allow_empty=True,
        max_length=_MAX_SUBJECT_LENGTH,
    )
    # AUDIT-FIX(#3): Bound and sanity-check the free-form draft body.
    _validate_body(body)
    # AUDIT-FIX(#1): Ensure this stub always fails deterministically instead of returning None.
    _raise_runtime_unavailable("email.draft_reply")


# AUDIT-FIX(#2): Reuse the validated default limit in metadata so runtime and declared signatures stay aligned.
MODULE_SPEC = SelfCodingModuleSpec(
    capability_id="email",
    module_name="email",
    summary="Read and prepare email through the managed email mailbox integration.",
    risk_class=CapabilityRiskClass.HIGH,
    public_api=(
        SelfCodingModuleFunction(
            name="get_unread",
            signature=(
                f"get_unread(limit: int = {_DEFAULT_UNREAD_LIMIT}) -> "
                "list[dict[str, str]]"
            ),
            summary="Return a bounded list of unread emails with sender, subject, and timestamp fields.",
            returns="a list of JSON-safe unread email records",
            tags=("read_only", "integration"),
        ),
        SelfCodingModuleFunction(
            name="summarize_unread",
            signature=f"summarize_unread(limit: int = {_DEFAULT_UNREAD_LIMIT}) -> str",
            summary="Summarize the bounded unread mailbox in plain language.",
            returns="a short mailbox summary string",
            tags=("read_only", "integration"),
        ),
        SelfCodingModuleFunction(
            name="draft_reply",
            signature="draft_reply(recipient: str, subject: str, body: str) -> dict[str, str]",
            summary="Prepare one bounded email draft for later human review or confirmation.",
            returns="a JSON-safe draft description",
            effectful=True,
            tags=("effectful", "integration"),
        ),
    ),
    requires_configuration=True,
    integration_id="email_mailbox",
    tags=("integration", "email", "configured"),
)

__all__ = ["MODULE_SPEC", "draft_reply", "get_unread", "summarize_unread"]