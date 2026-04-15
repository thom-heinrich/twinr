"""Fail-closed contracts for graph-contact fixture seeding.

The evaluation harnesses rely on deterministic graph fixtures. If a contact
write returns ``needs_clarification`` or otherwise fails to produce a concrete
person node, the seed must abort immediately instead of silently continuing
with a corrupted graph.
"""

from __future__ import annotations

from typing import Iterable
import logging

from twinr.memory.chonkydb.personal_graph import TwinrGraphContactOption, TwinrGraphWriteResult


logger = logging.getLogger(__name__)

_SUCCESSFUL_CONTACT_SEED_STATUSES = frozenset({"created", "updated"})


def _format_contact_label(*, given_name: str, family_name: str | None) -> str:
    parts = [str(given_name or "").strip(), str(family_name or "").strip()]
    return " ".join(part for part in parts if part).strip()


def _summarize_options(options: Iterable[TwinrGraphContactOption]) -> tuple[str, ...]:
    summary: list[str] = []
    for option in options:
        label = " ".join(str(option.label or "").split()).strip()
        detail = " ".join(str(option.detail or "").split()).strip()
        if label and detail:
            summary.append(f"{label} ({detail})")
        elif label:
            summary.append(label)
    return tuple(summary)


def require_successful_contact_seed_write(
    *,
    result: TwinrGraphWriteResult,
    given_name: str,
    family_name: str | None,
    role: str | None,
    phone: str | None,
    email: str | None,
    seed_context: str,
) -> TwinrGraphWriteResult:
    """Fail closed when one seeded contact did not produce a concrete node."""

    if result.status in _SUCCESSFUL_CONTACT_SEED_STATUSES and str(result.node_id or "").strip():
        return result
    requested_label = _format_contact_label(given_name=given_name, family_name=family_name)
    option_summary = _summarize_options(result.options)
    message = (
        f"{seed_context} failed to seed graph contact '{requested_label or given_name}': "
        f"remember_contact returned status={result.status!r} node_id={result.node_id!r} "
        f"role={role!r} phone={phone!r} email={email!r} "
        f"question={result.question!r} options={option_summary!r}"
    )
    logger.error(message)
    raise RuntimeError(message)
