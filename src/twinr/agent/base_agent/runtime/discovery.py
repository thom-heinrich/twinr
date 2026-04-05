# CHANGELOG: 2026-03-27
# BUG-1: Normalize `action` before dispatch; the previous code normalized only for feedback,
# BUG-1: so variants like "start-or-resume" could be tracked as valid feedback while failing in `service.manage`.
# BUG-2: Make invite snapshot + `service.manage(...)` atomic under the runtime memory lock to prevent stale feedback attribution.
# BUG-3: Treat invite-feedback persistence as non-critical; analytics write failures must not abort committed discovery changes.
# BUG-4: Naive `now` values are now interpreted deterministically as UTC instead of silently inheriting the host timezone.
# SEC-1: Bound externally supplied identifiers, batch sizes, and feedback summaries to prevent memory/disk amplification on Raspberry Pi deployments.
# IMP-1: Replace the global service cold-start lock with a per-instance reentrant init lock for lower contention and better free-threaded safety.
# IMP-2: Snapshot iterables to bounded tuples and harden callback `source` injection for forward compatibility with upstream callback payloads.
# IMP-3: Add lightweight structured logging and optional OpenTelemetry spans for edge-runtime observability.

"""Expose runtime helpers for Twinr's guided user-discovery flow."""


from __future__ import annotations

import logging
from contextlib import nullcontext
from datetime import datetime, timezone
from threading import Lock, RLock
from time import perf_counter
from typing import Any, Final, Iterable, TypeVar

try:  # Optional frontier-grade observability, no hard dependency.
    # pylint: disable=import-error
    from opentelemetry import trace as _otel_trace
except Exception:  # pragma: no cover - optional dependency
    _DISCOVERY_TRACER = None
else:  # pragma: no cover - optional dependency
    _DISCOVERY_TRACER = _otel_trace.get_tracer(__name__)

from twinr.memory.user_discovery import (
    UserDiscoveryCommitCallbacks,
    UserDiscoveryFact,
    UserDiscoveryMemoryRoute,
    UserDiscoveryResult,
    UserDiscoveryService,
)
from twinr.proactive.runtime.display_reserve_user_discovery_feedback import (
    record_user_discovery_invite_feedback,
)

T = TypeVar("T")

_LOGGER: Final = logging.getLogger(__name__)

_DISCOVERY_LOCK_GUARD: Final = Lock()
_DISCOVERY_SERVICE_LOCK_ATTR: Final = "_twinr_user_discovery_service_lock"

_MAX_ACTION_CHARS: Final = 64
_MAX_TOPIC_ID_CHARS: Final = 128
_MAX_FACT_ID_CHARS: Final = 128
_MAX_FEEDBACK_SUMMARY_CHARS: Final = 512
# BREAKING: discovery write batches are now capped to keep edge deployments memory-bounded.
_MAX_LEARNED_FACTS: Final = 64
_MAX_MEMORY_ROUTES: Final = 64
_MAX_SNOOZE_DAYS: Final = 3650

_DIRECT_DISCOVERY_FEEDBACK_STATUS: Final = {
    "start_or_resume": "engaged",
    "answer": "engaged",
    "review_profile": "engaged",
    "replace_fact": "engaged",
    "delete_fact": "engaged",
    "pause_session": "cooled",
    "snooze": "cooled",
    "skip_topic": "avoided",
}


def _normalize_slug(
    value: str | None,
    *,
    field_name: str,
    max_chars: int,
    allow_none: bool = True,
) -> str | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} must not be None")

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        if allow_none:
            return None
        raise ValueError(f"{field_name} must not be empty")

    if len(normalized) > max_chars:
        raise ValueError(f"{field_name} exceeds max length {max_chars}")

    return normalized


def _normalize_action(action: str) -> str:
    normalized = _normalize_slug(
        action,
        field_name="action",
        max_chars=_MAX_ACTION_CHARS,
        allow_none=False,
    )
    if normalized is None:  # pragma: no cover - unreachable, keeps static typing precise.
        raise RuntimeError("normalized action unexpectedly missing")
    return normalized


def _sanitize_identifier(value: str | None, *, field_name: str, max_chars: int) -> str | None:
    if value is None:
        return None

    cleaned = str(value).strip()
    if not cleaned:
        return None

    if len(cleaned) > max_chars:
        raise ValueError(f"{field_name} exceeds max length {max_chars}")

    return cleaned


def _coerce_bounded_tuple(values: Iterable[T] | None, *, field_name: str, max_items: int) -> tuple[T, ...]:
    if values is None:
        return ()

    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be an iterable of items, not {type(values).__name__}")

    if isinstance(values, tuple):
        if len(values) > max_items:
            raise ValueError(f"{field_name} exceeds max size {max_items}")
        return values

    if hasattr(values, "__len__") and len(values) > max_items:
        raise ValueError(f"{field_name} exceeds max size {max_items}")

    items: list[T] = []
    for item in values:
        if len(items) >= max_items:
            raise ValueError(f"{field_name} exceeds max size {max_items}")
        items.append(item)
    return tuple(items)


def _coerce_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)

    # BREAKING: naive datetimes are treated as UTC for deterministic scheduling across hosts.
    if now.tzinfo is None or now.utcoffset() is None:
        return now.replace(tzinfo=timezone.utc)

    return now.astimezone(timezone.utc)


def _coerce_snooze_days(value: int | None) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool):
        raise TypeError("snooze_days must be an int or None")

    days = int(value)
    if days < 0:
        raise ValueError("snooze_days must be >= 0")
    if days > _MAX_SNOOZE_DAYS:
        raise ValueError(f"snooze_days exceeds max value {_MAX_SNOOZE_DAYS}")

    return days


def _with_source(kwargs: dict[str, Any], *, source: str) -> dict[str, Any]:
    merged = dict(kwargs)
    merged["source"] = source
    return merged


def _feedback_summary(result: UserDiscoveryResult, *, normalized_action: str) -> str:
    fallback = f"user_discovery:{normalized_action}"
    summary = getattr(result, "assistant_brief", None)
    if summary is None:
        safe_summary = fallback
    else:
        safe_summary = str(summary).strip() or fallback

    if len(safe_summary) > _MAX_FEEDBACK_SUMMARY_CHARS:
        return safe_summary[: _MAX_FEEDBACK_SUMMARY_CHARS - 1] + "…"

    return safe_summary


def _discovery_span():
    if _DISCOVERY_TRACER is None:  # pragma: no cover - optional dependency
        return nullcontext(None)
    return _DISCOVERY_TRACER.start_as_current_span("twinr.user_discovery.manage")


class TwinrRuntimeDiscoveryMixin:
    """Provide runtime-owned access to the user-discovery service."""

    def _user_discovery_service_init_lock(self) -> RLock:
        lock = getattr(self, _DISCOVERY_SERVICE_LOCK_ATTR, None)
        if lock is None:
            with _DISCOVERY_LOCK_GUARD:
                lock = getattr(self, _DISCOVERY_SERVICE_LOCK_ATTR, None)
                if lock is None:
                    lock = RLock()
                    setattr(self, _DISCOVERY_SERVICE_LOCK_ATTR, lock)
        return lock

    def _user_discovery_service(self) -> UserDiscoveryService:
        service = getattr(self, "_twinr_user_discovery_service", None)
        if service is None:
            with self._user_discovery_service_init_lock():
                service = getattr(self, "_twinr_user_discovery_service", None)
                if service is None:
                    service = UserDiscoveryService.from_config(self.config)
                    setattr(self, "_twinr_user_discovery_service", service)
        return service

    def _user_discovery_commit_callbacks(self) -> UserDiscoveryCommitCallbacks:
        return UserDiscoveryCommitCallbacks(
            update_user_profile=lambda category, instruction: self.update_user_profile_context(
                category=category,
                instruction=instruction,
            ),
            delete_user_profile=lambda category: self.remove_user_profile_context(category=category),
            update_personality=lambda category, instruction: self.update_personality_context(
                category=category,
                instruction=instruction,
            ),
            delete_personality=lambda category: self.remove_personality_context(category=category),
            remember_contact=lambda **kwargs: self.remember_contact(
                **_with_source(kwargs, source="user_discovery"),
            ),
            delete_contact=lambda node_id: self.delete_contact(node_id=node_id),
            remember_preference=lambda **kwargs: self.remember_preference(
                **_with_source(kwargs, source="user_discovery"),
            ),
            delete_preference=lambda node_id, edge_type: self.delete_preference(
                node_id=node_id,
                edge_type=edge_type,
            ),
            remember_plan=lambda **kwargs: self.remember_plan(
                **_with_source(kwargs, source="user_discovery"),
            ),
            delete_plan=lambda node_id: self.delete_plan(node_id=node_id),
            store_durable_memory=lambda **kwargs: self.store_durable_memory(**kwargs),
            delete_durable_memory=lambda entry_id: self.delete_durable_memory_entry(entry_id=entry_id),
        )

    def manage_user_discovery(
        self,
        *,
        action: str,
        topic_id: str | None = None,
        learned_facts: tuple[UserDiscoveryFact, ...] = (),
        memory_routes: tuple[UserDiscoveryMemoryRoute, ...] = (),
        fact_id: str | None = None,
        topic_complete: bool | None = None,
        permission_granted: bool | None = None,
        snooze_days: int | None = None,
        now: datetime | None = None,
    ) -> UserDiscoveryResult:
        """Advance or inspect the bounded guided user-discovery flow."""

        effective_now = _coerce_now(now)
        normalized_action = _normalize_action(action)
        normalized_topic_id = _normalize_slug(
            topic_id,
            field_name="topic_id",
            max_chars=_MAX_TOPIC_ID_CHARS,
            allow_none=True,
        )
        safe_fact_id = _sanitize_identifier(
            fact_id,
            field_name="fact_id",
            max_chars=_MAX_FACT_ID_CHARS,
        )
        safe_learned_facts = _coerce_bounded_tuple(
            learned_facts,
            field_name="learned_facts",
            max_items=_MAX_LEARNED_FACTS,
        )
        safe_memory_routes = _coerce_bounded_tuple(
            memory_routes,
            field_name="memory_routes",
            max_items=_MAX_MEMORY_ROUTES,
        )
        safe_snooze_days = _coerce_snooze_days(snooze_days)

        service = self._user_discovery_service()
        callbacks = self._user_discovery_commit_callbacks()
        feedback_status = _DIRECT_DISCOVERY_FEEDBACK_STATUS.get(normalized_action)
        pending_invite = None

        started_at = perf_counter()
        with _discovery_span() as span:
            if span is not None:  # pragma: no branch - optional dependency
                span.set_attribute("twinr.user_discovery.action", normalized_action)
                span.set_attribute("twinr.user_discovery.learned_fact_count", len(safe_learned_facts))
                span.set_attribute("twinr.user_discovery.memory_route_count", len(safe_memory_routes))
                if normalized_topic_id is not None:
                    span.set_attribute("twinr.user_discovery.topic_id", normalized_topic_id)

            try:
                with self._memory_runtime_lock():
                    if feedback_status is not None:
                        try:
                            pending_invite = service.build_invitation(now=effective_now)
                        except Exception:
                            _LOGGER.warning(
                                "Twinr user discovery invite snapshot failed; continuing without feedback.",
                                exc_info=True,
                            )

                    result = service.manage(
                        action=normalized_action,
                        topic_id=normalized_topic_id,
                        learned_facts=safe_learned_facts,
                        memory_routes=safe_memory_routes,
                        fact_id=safe_fact_id,
                        topic_complete=topic_complete,
                        permission_granted=permission_granted,
                        snooze_days=safe_snooze_days,
                        callbacks=callbacks,
                        now=effective_now,
                    )
            except Exception as exc:
                if span is not None:  # pragma: no branch - optional dependency
                    span.record_exception(exc)
                _LOGGER.exception(
                    "Twinr user discovery manage failed action=%s learned_facts=%d memory_routes=%d",
                    normalized_action,
                    len(safe_learned_facts),
                    len(safe_memory_routes),
                )
                raise
            finally:
                duration_ms = (perf_counter() - started_at) * 1000.0
                if span is not None:  # pragma: no branch - optional dependency
                    span.set_attribute("twinr.user_discovery.duration_ms", duration_ms)

                _LOGGER.debug(
                    "Twinr user discovery action=%s duration_ms=%.2f learned_facts=%d memory_routes=%d feedback_action=%s",
                    normalized_action,
                    duration_ms,
                    len(safe_learned_facts),
                    len(safe_memory_routes),
                    feedback_status,
                )

        if (
            pending_invite is not None
            and feedback_status is not None
            and (normalized_topic_id is None or normalized_topic_id == pending_invite.topic_id)
        ):
            try:
                record_user_discovery_invite_feedback(
                    self.config,
                    invite=pending_invite,
                    status=feedback_status,
                    occurred_at=effective_now,
                    summary=_feedback_summary(result, normalized_action=normalized_action),
                )
            except Exception:
                _LOGGER.warning(
                    "Twinr user discovery feedback persistence failed action=%s topic_id=%s",
                    normalized_action,
                    getattr(pending_invite, "topic_id", None),
                    exc_info=True,
                )

        return result


__all__ = ["TwinrRuntimeDiscoveryMixin"]
