"""Nightly orchestration hook for the realtime background loop."""

# mypy: ignore-errors

from __future__ import annotations

from datetime import datetime

from twinr.agent.workflows.realtime_runtime.nightly import TwinrNightlyOrchestrator
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError

_DEFAULT_NIGHTLY_POLL_INTERVAL_S = 300.0


class BackgroundNightlyMixin:
    """Run the explicit overnight preparation pass during quiet idle windows."""

    def _maybe_run_nightly_orchestration(self) -> bool:
        """Run one bounded nightly pass when Twinr is idle and the day is due."""

        if not self._longterm_deadline_due(
            "_next_nightly_orchestration_check_at",
            self._config_interval_seconds(
                "nightly_orchestration_poll_interval_s",
                _DEFAULT_NIGHTLY_POLL_INTERVAL_S,
            ),
        ):
            return False
        if not self._background_work_allowed():
            return False

        orchestrator = TwinrNightlyOrchestrator(
            config=self.config,
            runtime=self.runtime,
            text_backend=getattr(self, "agent_provider", None),
            search_backend=getattr(self, "print_backend", None),
            print_backend=getattr(self, "print_backend", None),
            display_planner=self._display_reserve_companion_planner(),
            remote_ready=self._required_remote_dependency_current_ready,
            background_allowed=self._background_work_allowed,
        )
        try:
            result = orchestrator.maybe_run(
                local_now=datetime.now(self._local_timezone()),
            )
        except LongTermRemoteUnavailableError as exc:
            error_summary = self._longterm_exception_summary(exc)
            if self._enter_required_remote_error(exc):
                self._safe_record_event(
                    "nightly_required_remote_failed",
                    "The nightly orchestration hit a required remote-memory failure and forced Twinr into fail-closed error state.",
                    level="error",
                    error_type=type(exc).__name__,
                    error=error_summary,
                )
                return False
            self._remember_background_fault("nightly_orchestration_required_remote", exc)
            return False
        except RuntimeError as exc:
            reason = self._longterm_normalize_telemetry_text(exc, limit=96) or "interrupted"
            if reason.startswith("background_not_idle"):
                return False
            self._remember_background_fault("nightly_orchestration_runtime", exc)
            self._safe_record_event(
                "nightly_orchestration_failed",
                "Nightly orchestration failed inside the realtime background loop.",
                level="error",
                error_type=type(exc).__name__,
                error=self._longterm_exception_summary(exc),
            )
            return False
        except Exception as exc:
            self._remember_background_fault("nightly_orchestration", exc)
            self._longterm_safe_emit_kv("nightly_orchestration_error", self._longterm_exception_summary(exc))
            self._safe_record_event(
                "nightly_orchestration_failed",
                "Nightly orchestration failed inside the realtime background loop.",
                level="error",
                error_type=type(exc).__name__,
                error=self._longterm_exception_summary(exc),
            )
            return False

        if result.action != "prepared":
            return False

        state = result.state
        summary = result.summary
        digest = result.digest
        self._longterm_safe_emit_kv("nightly_orchestration_prepared", result.target_local_day)
        self._safe_record_event(
            "nightly_orchestration_prepared",
            "Prepared Twinr's overnight memory and morning artifacts.",
            level="info",
            target_local_day=result.target_local_day,
            status=getattr(state, "last_status", None),
            digest_ready=bool(getattr(state, "digest_ready", False)),
            display_reserve_status=getattr(state, "display_reserve_status", None),
            reflection_reflected_object_count=getattr(
                state,
                "reflection_reflected_object_count",
                0,
            ),
            reflection_created_summary_count=getattr(
                state,
                "reflection_created_summary_count",
                0,
            ),
            world_refresh_status=getattr(state, "world_refresh_status", None),
            live_search_queries=getattr(state, "live_search_queries", 0),
            reminder_lines=self._longterm_sanitize_event_value(
                list(getattr(digest, "reminder_lines", ())) if digest is not None else []
            ),
            headline_lines=self._longterm_sanitize_event_value(
                list(getattr(digest, "headline_lines", ())) if digest is not None else []
            ),
            summary_errors=self._longterm_sanitize_event_value(
                list(getattr(summary, "errors", ())) if summary is not None else []
            ),
        )
        return True
