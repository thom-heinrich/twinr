"""Proactive-planning entry points for the long-term runtime service."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from twinr.agent.workflows.forensics import workflow_event
from twinr.memory.longterm.core.models import (
    LongTermProactiveCandidateV1,
    LongTermProactivePlanV1,
)
from twinr.memory.longterm.proactive.state import LongTermProactiveReservationV1
from twinr.memory.longterm.runtime.live_object_selectors import (
    select_proactive_planner_objects as _select_proactive_planner_objects,
)
from twinr.memory.longterm.storage.remote_state import (
    LongTermRemoteReadFailedError,
    LongTermRemoteUnavailableError,
)

from ._typing import ServiceMixinBase
from .compat import _normalize_datetime, logger


class LongTermMemoryServiceProactiveMixin(ServiceMixinBase):
    """Plan and track proactive candidates under policy control."""

    @staticmethod
    def _record_fast_topic_proactive_skip(
        *,
        action: str,
        exc: LongTermRemoteReadFailedError,
    ) -> None:
        """Record one non-fatal proactive fast-topic failure."""

        workflow_event(
            kind="warning",
            msg="longterm_proactive_fast_topic_skipped",
            details={
                "action": action,
                **dict(getattr(exc, "details", {}) or {}),
            },
        )

    def plan_proactive_candidates(
        self,
        *,
        now: datetime | None = None,
        live_facts: Mapping[str, object] | None = None,
    ) -> LongTermProactivePlanV1:
        """Plan proactive candidates from stored memory and live facts."""

        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:
                return self.planner.plan(
                    objects=_select_proactive_planner_objects(self.object_store),
                    now=normalized_now,
                    live_facts=live_facts,
                )
        except LongTermRemoteReadFailedError as exc:
            self._record_fast_topic_proactive_skip(action="plan", exc=exc)
            return LongTermProactivePlanV1(candidates=())
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Proactive planning failed.")
            return LongTermProactivePlanV1(candidates=())

    def reserve_proactive_candidate(
        self,
        *,
        now: datetime | None = None,
        live_facts: Mapping[str, object] | None = None,
    ) -> LongTermProactiveReservationV1 | None:
        """Plan and reserve the next eligible proactive candidate."""

        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:
                plan = self.planner.plan(
                    objects=_select_proactive_planner_objects(self.object_store),
                    now=normalized_now,
                    live_facts=live_facts,
                )
                return self.proactive_policy.reserve_candidate(plan=plan, now=normalized_now)
        except LongTermRemoteReadFailedError as exc:
            self._record_fast_topic_proactive_skip(action="reserve", exc=exc)
            return None
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Proactive candidate reservation failed.")
            return None

    def reserve_specific_proactive_candidate(
        self,
        candidate: LongTermProactiveCandidateV1,
        *,
        now: datetime | None = None,
    ) -> LongTermProactiveReservationV1:
        """Reserve one specific proactive candidate under policy control."""

        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        with self._store_lock:
            return self.proactive_policy.reserve_specific_candidate(candidate, now=normalized_now)

    def preview_proactive_candidate(
        self,
        *,
        now: datetime | None = None,
        live_facts: Mapping[str, object] | None = None,
    ) -> LongTermProactiveCandidateV1 | None:
        """Preview the next eligible proactive candidate without reserving it."""

        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:
                plan = self.planner.plan(
                    objects=_select_proactive_planner_objects(self.object_store),
                    now=normalized_now,
                    live_facts=live_facts,
                )
                return self.proactive_policy.preview_candidate(plan=plan, now=normalized_now)
        except LongTermRemoteReadFailedError as exc:
            self._record_fast_topic_proactive_skip(action="preview", exc=exc)
            return None
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Proactive candidate preview failed.")
            return None

    def mark_proactive_candidate_delivered(
        self,
        reservation: LongTermProactiveReservationV1,
        *,
        delivered_at: datetime | None = None,
        prompt_text: str | None = None,
    ) -> Any | None:
        """Record that a reserved proactive candidate was delivered."""

        normalized_delivered_at = _normalize_datetime(delivered_at, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:
                return self.proactive_policy.mark_delivered(
                    reservation,
                    delivered_at=normalized_delivered_at,
                    prompt_text=prompt_text,
                )
        except Exception:
            logger.exception("Failed to mark proactive candidate as delivered.")
            return None

    def mark_proactive_candidate_skipped(
        self,
        reservation: LongTermProactiveReservationV1,
        *,
        reason: str,
        skipped_at: datetime | None = None,
    ) -> Any | None:
        """Record that a reserved proactive candidate was skipped."""

        normalized_skipped_at = _normalize_datetime(skipped_at, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:
                return self.proactive_policy.mark_skipped(
                    reservation,
                    reason=reason,
                    skipped_at=normalized_skipped_at,
                )
        except Exception:
            logger.exception("Failed to mark proactive candidate as skipped.")
            return None
