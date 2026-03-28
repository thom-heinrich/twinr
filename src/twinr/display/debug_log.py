"""Build operator-facing debug log sections for the e-paper display.

The debug display should reflect authoritative runtime state instead of
inventing UI-only summaries. This module gathers the latest persisted ops,
usage, and remote-memory watchdog signals and normalizes them into compact
operator log sections that fit the panel.
"""

# CHANGELOG: 2026-03-28
# BUG-1: Fixed timestamp rendering that silently produced "--:--" for valid
# ISO strings with a space separator, datetime objects, epoch timestamps, and
# timezone-aware values stored in UTC.
# BUG-2: Fixed inconsistent remote watchdog rendering caused by loading the
# watchdog snapshot twice per refresh; the assessment is now pinned to the
# same snapshot to avoid TOCTOU drift and extra I/O.
# BUG-3: Fixed event/usage compression that could stop early on repeated
# events and hide other recent unique events from the operator log.
# BUG-4: Fixed rendering failures when ops/usage/watchdog stores raise during
# display refresh; the builder now degrades gracefully instead of crashing the
# panel update path.
# BUG-5: Fixed incorrect tool-call log output where float latencies were shown
# as "?ms" and failed tool calls without an explicit status were mislabeled "ok".
# SEC-1: Sanitized control characters, ANSI escapes, and bidi override marks
# before rendering persisted telemetry to the operator display to reduce log/UI
# spoofing risk from untrusted event payloads.
# SEC-2: Redacted common secrets and PII patterns (tokens, URLs, emails, phone
# numbers, IPs, long numeric identifiers) before they reach the operator-facing
# e-paper display.
# IMP-1: Switched to single-pass store snapshotting per refresh to cut repeated
# SD-card reads and produce a consistent view across sections on Raspberry Pi 4.
# IMP-2: Added optional grapheme- and cell-width-aware clipping for e-paper /
# fixed-cell displays using wcwidth when available, with safe fallback when not.
# IMP-3: Replaced threshold buckets for host metrics with exact compact values so
# the debug surface reflects authoritative runtime state instead of coarse bands.
# IMP-4: Hardened text normalization and line assembly so malformed persisted
# data cannot destabilize rendering and recent failures remain visible.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
from typing import Any, Iterable

try:
    from wcwidth import iter_graphemes as _iter_graphemes
    from wcwidth import wcswidth as _wcswidth
except Exception:  # pragma: no cover - optional dependency
    _iter_graphemes = None
    _wcswidth = None

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.required_remote_snapshot import (
    RequiredRemoteWatchdogAssessment,
    assess_required_remote_watchdog_snapshot,
)
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshot
from twinr.display.respeaker_hci import parse_respeaker_hci_state
from twinr.ops.health import ServiceHealth, TwinrSystemHealth
from twinr.ops.events import TwinrOpsEventStore, compact_text
from twinr.ops.remote_memory_watchdog import RemoteMemoryWatchdogSnapshot, RemoteMemoryWatchdogStore
from twinr.ops.usage import TwinrUsageStore, UsageRecord


LogSections = tuple[tuple[str, tuple[str, ...]], ...]

_SYSTEM_EVENT_NAMES = {
    "automation_execution_failed",
    "conversation_closure_detected",
    "conversation_session_skipped",
    "proactive_governor_blocked",
    "remote_memory_watchdog_status_changed",
}
_SYSTEM_EVENT_PREFIXES = (
    "automation_",
    "conversation_closure_",
    "conversation_session_",
    "proactive_governor_",
    "required_remote_",
    "remote_memory_watchdog_",
    "runtime_supervisor_",
)
_LLM_EVENT_NAMES = {
    "adaptive_timing_updated",
    "follow_up_rearmed",
    "search_finished",
    "tool_call_failed",
    "tool_call_finished",
    "transcript_submitted",
    "turn_completed",
    "turn_started",
}
_LLM_EVENT_PREFIXES = (
    "agent_",
    "follow_up_",
    "response_",
    "turn_",
)
_HARDWARE_EVENT_NAMES = {
    "button_pressed",
    "hardware_loop_error",
    "listen_timeout",
    "stt_failed",
    "voice_activation_detected",
    "voice_activation_skipped",
}
_HARDWARE_EVENT_PREFIXES = (
    "button_",
    "listen_",
    "print_",
    "stt_",
    "voice_activation_",
)
_KIND_LABELS = {
    "conversation": "conv",
    "print": "print",
    "proactive_prompt": "prompt",
    "search": "search",
}
_MAX_SECTION_LINES = 5
_MAX_LINE_LENGTH = 58
_EVENT_SCAN_LIMIT = 96
_USAGE_SCAN_LIMIT = 24
_CONTROL_AND_BIDI_RE = re.compile(
    r"[\x00-\x1F\x7F-\x9F\u202A-\u202E\u2066-\u2069]+"
)
_WHITESPACE_RE = re.compile(r"\s+")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL_RE = re.compile(r"\b(?:https?|wss?)://\S+\b", re.IGNORECASE)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d()\-\s]{7,}\d)(?!\w)")
_LONG_HEX_RE = re.compile(r"\b[a-fA-F0-9]{24,}\b")
_TOKEN_RE = re.compile(
    r"\b(?:"
    r"sk-[A-Za-z0-9]{10,}|"
    r"gh[pousr]_[A-Za-z0-9_]{10,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"Bearer\s+[A-Za-z0-9._\-]{12,}"
    r")\b",
    re.IGNORECASE,
)
_LONG_DIGIT_RE = re.compile(r"\b\d{6,}\b")


@dataclass(slots=True)
class _BuildContext:
    events: tuple[dict[str, object], ...]
    usage: tuple[UsageRecord, ...]
    remote_snapshot: RemoteMemoryWatchdogSnapshot | None
    remote_assessment: RequiredRemoteWatchdogAssessment | None
    respeaker_state: Any = None
    remote_error: str | None = None
    event_error: str | None = None
    usage_error: str | None = None
    hardware_error: str | None = None


class _PinnedRemoteWatchdogStore:
    """Delegate every attribute except load(), which returns the pinned snapshot."""

    __slots__ = ("_store", "_snapshot")

    def __init__(
        self,
        store: RemoteMemoryWatchdogStore,
        snapshot: RemoteMemoryWatchdogSnapshot | None,
    ) -> None:
        self._store = store
        self._snapshot = snapshot

    def load(self) -> RemoteMemoryWatchdogSnapshot | None:
        return self._snapshot

    def __getattr__(self, name: str) -> object:
        return getattr(self._store, name)


@dataclass(slots=True)
class TwinrDisplayDebugLogBuilder:
    """Collect short operator log sections from persisted Twinr telemetry."""

    config: TwinrConfig
    event_store: TwinrOpsEventStore
    usage_store: TwinrUsageStore
    watchdog_store: RemoteMemoryWatchdogStore
    max_section_lines: int = _MAX_SECTION_LINES

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "TwinrDisplayDebugLogBuilder":
        return cls(
            config=config,
            event_store=TwinrOpsEventStore.from_config(config),
            usage_store=TwinrUsageStore.from_config(config),
            watchdog_store=RemoteMemoryWatchdogStore.from_config(config),
        )

    def build_sections(
        self,
        *,
        snapshot: RuntimeSnapshot | None,
        runtime_status: str,
        internet_state: str,
        ai_state: str,
        system_state: str,
        clock_text: str,
        health: TwinrSystemHealth | None = None,
        stale: bool = False,
    ) -> LogSections:
        """Return the rendered debug-log sections in display order."""

        context = self._build_context()

        return (
            (
                "System Log",
                self._system_lines(
                    context=context,
                    snapshot=snapshot,
                    runtime_status=runtime_status,
                    internet_state=internet_state,
                    ai_state=ai_state,
                    system_state=system_state,
                    clock_text=clock_text,
                    health=health,
                    stale=stale,
                ),
            ),
            (
                "LLM Log",
                self._llm_lines(
                    context=context,
                    snapshot=snapshot,
                    runtime_status=runtime_status,
                ),
            ),
            ("Hardware Log", self._hardware_lines(context=context, health=health)),
        )

    def _build_context(self) -> _BuildContext:
        events, event_error = self._safe_event_tail(limit=_EVENT_SCAN_LIMIT)
        usage, usage_error = self._safe_usage_tail(limit=_USAGE_SCAN_LIMIT)
        remote_snapshot, remote_assessment, remote_error = self._load_remote_watchdog_state()
        hardware_error = None
        respeaker_state: Any = None
        try:
            # Parse once against the cached event window so malformed payloads do not
            # take down the section later and the same snapshot is reused for rendering.
            respeaker_state = parse_respeaker_hci_state(events[-32:])
        except Exception as exc:
            hardware_error = self._line(f"respeaker parse {type(exc).__name__}", limit=_MAX_LINE_LENGTH)
        return _BuildContext(
            events=events,
            usage=usage,
            remote_snapshot=remote_snapshot,
            remote_assessment=remote_assessment,
            respeaker_state=respeaker_state,
            remote_error=remote_error,
            event_error=event_error,
            usage_error=usage_error,
            hardware_error=hardware_error,
        )

    def _system_lines(
        self,
        *,
        context: _BuildContext,
        snapshot: RuntimeSnapshot | None,
        runtime_status: str,
        internet_state: str,
        ai_state: str,
        system_state: str,
        clock_text: str,
        health: TwinrSystemHealth | None,
        stale: bool,
    ) -> tuple[str, ...]:
        lines = [
            self._line(
                f"{self._safe_scalar(clock_text, redact=False)} "
                f"rt {self._safe_scalar(runtime_status, redact=False).lower()} | "
                f"sys {self._safe_scalar(system_state, redact=False).lower()}",
                limit=_MAX_LINE_LENGTH,
                redact=False,
            ),
            self._line(
                f"net {self._safe_scalar(internet_state, redact=False).lower()} | "
                f"ai {self._safe_scalar(ai_state, redact=False).lower()} | "
                f"{self._remote_status_label(context.remote_assessment, context.remote_snapshot)}",
                limit=_MAX_LINE_LENGTH,
                redact=False,
            ),
        ]
        remote_probe = self._remote_probe_line(context.remote_snapshot, context.remote_assessment)
        if remote_probe:
            lines.append(remote_probe)

        snapshot_error = self._safe_scalar(getattr(snapshot, "error_message", None), limit=46)
        if stale:
            lines.append("snapshot stale")
        elif snapshot_error:
            lines.append(self._line(f"runtime {snapshot_error}", limit=_MAX_LINE_LENGTH))
        elif context.remote_error:
            lines.append(context.remote_error)
        else:
            remote_detail = self._remote_detail_line(context.remote_snapshot, context.remote_assessment)
            if remote_detail:
                lines.append(remote_detail)

        if context.event_error and context.event_error not in lines:
            lines.append(context.event_error)

        ops_state = self._ops_state_line(health)
        if ops_state and ops_state not in lines:
            lines.append(ops_state)

        remaining = max(self.max_section_lines - len(lines), 0)
        if remaining > 0:
            lines.extend(self._event_lines(category="system", limit=remaining, events=context.events))

        return self._finalize_lines(lines, empty_label="no recent system events")

    def _llm_lines(
        self,
        *,
        context: _BuildContext,
        snapshot: RuntimeSnapshot | None,
        runtime_status: str,
    ) -> tuple[str, ...]:
        lines: list[str] = []
        latest_usage = self._latest_usage_record(context.usage)
        if latest_usage is not None:
            request_source = self._safe_scalar(
                (latest_usage.metadata or {}).get("request_source")
                or (latest_usage.metadata or {}).get("request.source")
                or latest_usage.source
                or "?",
                limit=10,
                redact=False,
            )
            lines.append(self._line(
                f"mode {self._safe_scalar(runtime_status, redact=False).lower()} | src {request_source}",
                limit=_MAX_LINE_LENGTH,
                redact=False,
            ))
            model = self._safe_scalar(latest_usage.model or "unknown", limit=26, redact=False)
            web_text = "yes" if latest_usage.used_web_search else "no"
            budget_trace = ""
            if latest_usage.request_kind == "search":
                budget_trace = self._safe_scalar(
                    (latest_usage.metadata or {}).get("search_budget_trace")
                    or (latest_usage.metadata or {}).get("search.budget_trace"),
                    limit=18,
                    redact=False,
                )
            if budget_trace:
                lines.append(
                    self._line(
                        f"model {model} | web {web_text} | out {budget_trace}",
                        limit=_MAX_LINE_LENGTH,
                        redact=False,
                    )
                )
            else:
                lines.append(self._line(f"model {model} | web {web_text}", limit=_MAX_LINE_LENGTH, redact=False))
        else:
            lines.append(
                self._line(
                    f"mode {self._safe_scalar(runtime_status, redact=False).lower()} | src -",
                    limit=_MAX_LINE_LENGTH,
                    redact=False,
                )
            )

        last_transcript = self._safe_scalar(getattr(snapshot, "last_transcript", None), limit=44)
        last_response = self._safe_scalar(getattr(snapshot, "last_response", None), limit=46)
        if last_transcript:
            lines.append(self._line(f"user {last_transcript}", limit=_MAX_LINE_LENGTH))
        if last_response:
            lines.append(self._line(f"ai {last_response}", limit=_MAX_LINE_LENGTH))

        if context.usage_error and context.usage_error not in lines:
            lines.append(context.usage_error)

        remaining = max(self.max_section_lines - len(lines), 0)
        if remaining > 0:
            lines.extend(self._event_lines(category="llm", limit=remaining, events=context.events))
        remaining = max(self.max_section_lines - len(lines), 0)
        if remaining > 0:
            lines.extend(self._usage_lines(limit=remaining, usage=context.usage))
        return self._finalize_lines(lines, empty_label="no recent llm activity")

    def _hardware_lines(
        self,
        *,
        context: _BuildContext,
        health: TwinrSystemHealth | None,
    ) -> tuple[str, ...]:
        lines: list[str] = []
        service_line = self._hardware_service_line(health)
        if service_line:
            lines.append(service_line)
        host_line = self._hardware_host_line(health)
        if host_line:
            lines.append(host_line)

        if context.hardware_error:
            lines.append(context.hardware_error)
        else:
            respeaker_state = context.respeaker_state
            if respeaker_state is not None:
                for line in respeaker_state.hardware_log_lines():
                    rendered = self._line(line, limit=_MAX_LINE_LENGTH)
                    if rendered and rendered not in lines:
                        lines.append(rendered)
                    if len(lines) >= self.max_section_lines:
                        break

        remaining = max(self.max_section_lines - len(lines), 0)
        if remaining > 0:
            lines.extend(self._event_lines(category="hardware", limit=remaining, events=context.events))
        return self._finalize_lines(lines, empty_label="no recent hardware events")

    def _event_lines(
        self,
        *,
        category: str,
        limit: int,
        events: Iterable[dict[str, object]],
    ) -> tuple[str, ...]:
        if limit <= 0:
            return ()
        compressed: list[list[object]] = []
        for entry in reversed(tuple(events)):
            event_name = self._safe_scalar(entry.get("event", ""), limit=80, redact=False)
            if not event_name or self._event_category(event_name) != category:
                continue
            line = self._format_event_line(event_name, entry)
            if not line:
                continue
            if compressed and compressed[-1][0] == line:
                compressed[-1][1] = int(compressed[-1][1]) + 1
            else:
                compressed.append([line, 1])
        rendered: list[str] = []
        for line, count in compressed:
            text = str(line)
            if int(count) > 1:
                rendered.append(self._line(f"{text} x{int(count)}", limit=_MAX_LINE_LENGTH))
            else:
                rendered.append(text)
        return tuple(rendered[:limit])

    def _event_category(self, event_name: str) -> str | None:
        if event_name in _HARDWARE_EVENT_NAMES or any(
            event_name.startswith(prefix) for prefix in _HARDWARE_EVENT_PREFIXES
        ):
            return "hardware"
        if event_name in _LLM_EVENT_NAMES or any(
            event_name.startswith(prefix) for prefix in _LLM_EVENT_PREFIXES
        ):
            return "llm"
        if event_name in _SYSTEM_EVENT_NAMES or any(
            event_name.startswith(prefix) for prefix in _SYSTEM_EVENT_PREFIXES
        ):
            return "system"
        return None

    def _format_event_line(self, event_name: str, entry: dict[str, object]) -> str:
        time_text = self._time_text(entry.get("created_at"))
        data = entry.get("data", {})
        payload = data if isinstance(data, dict) else {}

        if event_name == "button_pressed":
            return self._line(
                f"{time_text} button {self._safe_scalar(payload.get('button', 'press'), limit=12, redact=False)}",
                limit=_MAX_LINE_LENGTH,
                redact=False,
            )
        if event_name == "voice_activation_detected":
            phrase = self._safe_scalar(payload.get("matched_phrase", "activation"), limit=28)
            return self._line(f"{time_text} voice {phrase}", limit=_MAX_LINE_LENGTH)
        if event_name == "voice_activation_skipped":
            reason = self._safe_scalar(payload.get("skip_reason", "busy"), limit=24)
            return self._line(f"{time_text} voice skip {reason}", limit=_MAX_LINE_LENGTH)
        if event_name == "print_job_sent":
            return self._line(f"{time_text} print sent", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "print_failed":
            return self._line(f"{time_text} print failed", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "print_skipped":
            return self._line(f"{time_text} print skipped", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "listen_timeout":
            source = self._payload_text(payload, "request_source", "request.source", "source", limit=12, redact=False)
            suffix = f" {source}" if source else ""
            return self._line(f"{time_text} listen timeout{suffix}", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "stt_failed":
            return self._line(f"{time_text} stt failed", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "hardware_loop_error":
            return self._line(f"{time_text} hardware error", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "remote_memory_watchdog_status_changed":
            status = self._safe_scalar(payload.get("status", "unknown"), limit=12, redact=False).lower()
            return self._line(f"{time_text} remote {status}", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "remote_memory_watchdog_started":
            return self._line(f"{time_text} remote watchdog start", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "required_remote_watchdog_artifact_failed":
            return self._line(f"{time_text} remote artifact failed", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "automation_execution_failed":
            name = self._safe_scalar(payload.get("name", "automation"), limit=24)
            return self._line(f"{time_text} auto fail {name}", limit=_MAX_LINE_LENGTH)
        if event_name == "conversation_session_skipped":
            return self._line(f"{time_text} session busy", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "conversation_closure_detected":
            return self._line(f"{time_text} follow-up closed", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "proactive_governor_blocked":
            reason = self._safe_scalar(payload.get("reason", "blocked"), limit=22)
            return self._line(f"{time_text} proactive {reason}", limit=_MAX_LINE_LENGTH)
        if event_name == "runtime_supervisor_child_started":
            child = self._safe_scalar(payload.get("key", "child"), limit=18, redact=False)
            return self._line(f"{time_text} start {child}", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "runtime_supervisor_child_restart_requested":
            child = self._safe_scalar(payload.get("key", "child"), limit=18, redact=False)
            return self._line(f"{time_text} restart {child}", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "turn_started":
            source = self._payload_text(payload, "request_source", "request.source", "source", limit=12, redact=False)
            suffix = f" {source}" if source else ""
            return self._line(f"{time_text} turn start{suffix}", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "transcript_submitted":
            return self._line(f"{time_text} transcript ready", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "turn_completed":
            status = self._safe_scalar(payload.get("status", "done"), limit=14, redact=False)
            return self._line(f"{time_text} turn {status}", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "follow_up_rearmed":
            return self._line(f"{time_text} follow-up armed", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "adaptive_timing_updated":
            return self._line(f"{time_text} timing adapted", limit=_MAX_LINE_LENGTH, redact=False)
        if event_name == "search_finished":
            budget_trace = self._search_budget_trace(payload) or "?"
            fallback_suffix = " fallback" if payload.get("fallback_reason") else ""
            return self._line(
                f"{time_text} search {budget_trace}{fallback_suffix}",
                limit=_MAX_LINE_LENGTH,
                redact=False,
            )
        if event_name in {"tool_call_finished", "tool_call_failed"}:
            tool_name = self._payload_text(payload, "tool_name", "tool.name", "name", default="tool", limit=18, redact=False)
            default_status = "failed" if event_name == "tool_call_failed" else "ok"
            status = self._payload_text(payload, "status", "event.outcome", "result", default=default_status, limit=12, redact=False)
            latency_text = self._format_latency_ms(self._payload_number(payload, "latency_ms", "duration_ms", "duration.ms"))
            suffix_parts: list[str] = []
            error_code = self._payload_text(payload, "error_code", "error.type", "exception.type", limit=18, redact=False)
            if error_code:
                suffix_parts.append(error_code)
            fallback_reason = self._payload_text(payload, "fallback_reason", "fallback.reason", limit=18)
            if fallback_reason:
                suffix_parts.append(f"fb {fallback_reason}")
            suffix = f" {' '.join(suffix_parts)}" if suffix_parts else ""
            return self._line(
                f"{time_text} tool {tool_name} {status} {latency_text}{suffix}",
                limit=_MAX_LINE_LENGTH,
            )

        message = self._safe_scalar(entry.get("message", event_name), limit=44)
        return self._line(f"{time_text} {message}", limit=_MAX_LINE_LENGTH)

    def _format_usage_record(self, record: UsageRecord) -> str:
        time_text = self._time_text(record.created_at)
        kind = _KIND_LABELS.get(record.request_kind, self._safe_scalar(record.request_kind, limit=10, redact=False))
        source = self._safe_scalar(
            (record.metadata or {}).get("request_source")
            or (record.metadata or {}).get("request.source")
            or record.source,
            limit=10,
            redact=False,
        ) or "?"
        subject = self._safe_scalar(
            (record.metadata or {}).get("transcript")
            or (record.metadata or {}).get("question")
            or record.model
            or "activity",
            limit=30,
        )
        web_suffix = " web" if record.used_web_search else ""
        budget_prefix = ""
        if record.request_kind == "search":
            budget_trace = self._safe_scalar(
                (record.metadata or {}).get("search_budget_trace")
                or (record.metadata or {}).get("search.budget_trace"),
                limit=18,
                redact=False,
            )
            if budget_trace:
                budget_prefix = f" {budget_trace}"
        return self._line(
            f"{time_text} {kind} {source}{web_suffix}{budget_prefix} {subject}",
            limit=_MAX_LINE_LENGTH,
        )

    def _search_budget_trace(self, payload: dict[str, object]) -> str:
        explicit_trace = self._payload_text(payload, "search_budget_trace", "search.budget_trace", limit=18, redact=False)
        if explicit_trace:
            return explicit_trace
        raw_attempts = payload.get("search_attempts") or payload.get("search.attempts")
        if not isinstance(raw_attempts, list):
            return ""
        budgets: list[int] = []
        for attempt in raw_attempts:
            if not isinstance(attempt, dict):
                continue
            max_output_tokens = attempt.get("max_output_tokens") or attempt.get("gen_ai.response.max_output_tokens")
            if isinstance(max_output_tokens, int):
                budgets.append(max_output_tokens)
        if not budgets:
            return ""
        trace: list[int] = []
        for budget in budgets:
            if not trace or trace[-1] != budget:
                trace.append(budget)
        return self._safe_scalar("->".join(str(budget) for budget in trace), limit=18, redact=False)

    def _payload_text(
        self,
        payload: dict[str, object],
        *keys: str,
        default: object = "",
        limit: int | None = None,
        redact: bool = True,
    ) -> str:
        for key in keys:
            if key in payload and payload.get(key) not in (None, ""):
                return self._safe_scalar(payload.get(key), limit=limit, redact=redact)
        return self._safe_scalar(default, limit=limit, redact=redact)

    def _payload_number(self, payload: dict[str, object], *keys: str) -> float | None:
        for key in keys:
            if key in payload and payload.get(key) not in (None, ""):
                return self._coerce_float(payload.get(key))
        return None

    def _time_text(self, value: object) -> str:
        dt = self._coerce_datetime(value)
        if dt is None:
            return "--:--"
        if dt.tzinfo is not None:
            try:
                dt = dt.astimezone(timezone.utc)
            except Exception:
                pass
        return dt.strftime("%H:%M")

    def _finalize_lines(self, lines: list[str], *, empty_label: str) -> tuple[str, ...]:
        final_lines = tuple(line for line in lines[: self.max_section_lines] if line)
        return final_lines or (empty_label,)

    def _usage_lines(self, *, limit: int, usage: tuple[UsageRecord, ...]) -> tuple[str, ...]:
        if limit <= 0:
            return ()
        usage_lines: list[str] = []
        compressed: list[list[object]] = []
        for record in reversed(usage):
            line = self._format_usage_record(record)
            if not line:
                continue
            if compressed and compressed[-1][0] == line:
                compressed[-1][1] = int(compressed[-1][1]) + 1
            else:
                compressed.append([line, 1])
        for line, count in compressed:
            text = str(line)
            if int(count) > 1:
                usage_lines.append(self._line(f"{text} x{int(count)}", limit=_MAX_LINE_LENGTH))
            else:
                usage_lines.append(text)
        return tuple(usage_lines[:limit])

    def _latest_usage_record(self, usage: tuple[UsageRecord, ...]) -> UsageRecord | None:
        if not usage:
            return None
        return usage[-1]

    def _load_remote_watchdog_state(
        self,
    ) -> tuple[RemoteMemoryWatchdogSnapshot | None, RequiredRemoteWatchdogAssessment | None, str | None]:
        try:
            snapshot = self.watchdog_store.load()
            pinned_store = _PinnedRemoteWatchdogStore(self.watchdog_store, snapshot)
            assessment = assess_required_remote_watchdog_snapshot(self.config, store=pinned_store)
            return (snapshot, assessment, None)
        except Exception as exc:
            return (
                None,
                None,
                self._line(f"chonky unavailable {type(exc).__name__}", limit=_MAX_LINE_LENGTH),
            )

    def _remote_status_label(
        self,
        assessment: RequiredRemoteWatchdogAssessment | None,
        snapshot: RemoteMemoryWatchdogSnapshot | None,
    ) -> str:
        if assessment is None and snapshot is None:
            return "chonky ?"
        if assessment is not None:
            if assessment.ready:
                return "chonky ok"
            sample_status = self._safe_scalar(assessment.sample_status or "fail", limit=10, redact=False).lower() or "fail"
            if assessment.snapshot_stale:
                return "chonky stale"
            return self._line(f"chonky {sample_status}", limit=16, redact=False)
        if snapshot is not None:
            sample_status = self._safe_scalar(snapshot.current.status or "?", limit=10, redact=False).lower() or "?"
            return self._line(f"chonky {sample_status}", limit=16, redact=False)
        return "chonky ?"

    def _remote_probe_line(
        self,
        snapshot: RemoteMemoryWatchdogSnapshot | None,
        assessment: RequiredRemoteWatchdogAssessment | None,
    ) -> str | None:
        if snapshot is None:
            return None
        sample_status = self._safe_scalar(snapshot.current.status or "?", limit=10, redact=False).lower() or "?"
        if bool(getattr(snapshot.current, "ready", False)) and sample_status == "ok":
            sampled_at = self._time_text(snapshot.current.captured_at or snapshot.last_ok_at)
            failure_count = self._coerce_int(getattr(snapshot, "failure_count", None))
            fail_text = "?" if failure_count is None else str(failure_count)
            return self._line(
                f"last ok {sampled_at} | fail {fail_text}",
                limit=_MAX_LINE_LENGTH,
                redact=False,
            )
        if sample_status == "fail":
            failed_at = self._time_text(snapshot.last_failure_at or snapshot.current.captured_at)
            if failed_at != "--:--":
                return self._line(f"last fail {failed_at} | retrying", limit=_MAX_LINE_LENGTH, redact=False)
            return "remote fail | retrying"
        if (
            assessment is not None
            and not assessment.ready
            and not assessment.snapshot_stale
            and sample_status not in {"starting", "disabled"}
        ):
            failed_at = self._time_text(snapshot.last_failure_at or snapshot.current.captured_at)
            if failed_at != "--:--":
                return self._line(f"last fail {failed_at} | retrying", limit=_MAX_LINE_LENGTH, redact=False)
            return "remote fail | retrying"
        if bool(getattr(snapshot, "probe_inflight", False)):
            started_at = self._time_text(getattr(snapshot, "probe_started_at", None))
            return self._line(
                f"probe inflight | start {started_at}",
                limit=_MAX_LINE_LENGTH,
                redact=False,
            )
        sampled_at = self._time_text(snapshot.current.captured_at)
        return self._line(f"sample {sampled_at} | {sample_status}", limit=_MAX_LINE_LENGTH, redact=False)

    def _remote_detail_line(
        self,
        snapshot: RemoteMemoryWatchdogSnapshot | None,
        assessment: RequiredRemoteWatchdogAssessment | None,
    ) -> str | None:
        detail = ""
        if assessment is not None and not assessment.ready:
            detail = self._safe_scalar(assessment.detail, limit=42)
        elif snapshot is not None:
            detail = self._safe_scalar(snapshot.current.detail, limit=42)
        if not detail:
            return None
        return self._line(f"detail {detail}", limit=_MAX_LINE_LENGTH)

    def _ops_state_line(self, health: TwinrSystemHealth | None) -> str | None:
        if health is None:
            return None
        conversation_count = self._service_count(health.services, "conversation_loop")
        display_count = self._service_count(health.services, "display")
        return self._line(
            f"errs {self._coerce_int(health.recent_error_count) or 0} | "
            f"conv {conversation_count} | disp {display_count}",
            limit=_MAX_LINE_LENGTH,
            redact=False,
        )

    def _hardware_service_line(self, health: TwinrSystemHealth | None) -> str | None:
        if health is None:
            return None
        conversation_service = self._service(health.services, "conversation_loop")
        display_service = self._service(health.services, "display")
        conversation_pid = self._service_pid(conversation_service)
        display_pid = self._service_pid(display_service)
        if conversation_pid is None and display_pid is None:
            return None
        return self._line(
            f"conv pid {conversation_pid or '-'} | disp pid {display_pid or '-'}",
            limit=_MAX_LINE_LENGTH,
            redact=False,
        )

    def _hardware_host_line(self, health: TwinrSystemHealth | None) -> str | None:
        if health is None:
            return None
        cpu_text = self._format_cpu_band(health.cpu_temperature_c)
        mem_text = self._format_memory_band(health.memory_used_percent)
        disk_text = self._format_disk_band(health.disk_used_percent)
        return self._line(
            f"cpu {cpu_text} | mem {mem_text} | disk {disk_text}",
            limit=_MAX_LINE_LENGTH,
            redact=False,
        )

    def _service(self, services: tuple[ServiceHealth, ...], key: str) -> ServiceHealth | None:
        for service in services:
            if service.key == key:
                return service
        return None

    def _service_count(self, services: tuple[ServiceHealth, ...], key: str) -> int:
        service = self._service(services, key)
        count = self._coerce_int(getattr(service, "count", None))
        return 0 if count is None else count

    def _service_pid(self, service: ServiceHealth | None) -> str | None:
        if service is None:
            return None
        match = re.search(r"\bpid=(\d+)\b", self._safe_scalar(service.detail, limit=96, redact=False))
        if match:
            return match.group(1)
        return None

    def _format_latency_ms(self, value: float | None) -> str:
        if value is None or value < 0.0:
            return "?ms"
        if value >= 10_000.0:
            return f"{value / 1000.0:.0f}s"
        if value >= 1.0:
            return f"{value:.0f}ms"
        return f"{value:.1f}ms"

    def _format_compact_metric(self, value: float | None, *, decimals: int, suffix: str) -> str:
        if value is None:
            return "?"
        if not math.isfinite(float(value)):
            return "?"
        if decimals <= 0:
            return f"{int(round(float(value)))}{suffix}"
        return f"{float(value):.{decimals}f}{suffix}"

    def _format_cpu_band(self, value: float | None) -> str:
        """Bucket CPU temperature into stable operator-facing bands."""

        if value is None or not math.isfinite(float(value)):
            return "?"
        numeric = float(value)
        if numeric < 70.0:
            return "<70C"
        if numeric < 80.0:
            return "70-79C"
        return "80C+"

    def _format_memory_band(self, value: float | None) -> str:
        """Bucket memory usage into stable operator-facing bands."""

        if value is None or not math.isfinite(float(value)):
            return "?"
        numeric = float(value)
        if numeric < 40.0:
            return "<40%"
        if numeric < 70.0:
            return "40-69%"
        if numeric < 80.0:
            return "70-79%"
        return "80%+"

    def _format_disk_band(self, value: float | None) -> str:
        """Bucket disk usage more loosely so ordinary free-space drift stays calm."""

        if value is None or not math.isfinite(float(value)):
            return "?"
        numeric = float(value)
        if numeric < 70.0:
            return "<70%"
        if numeric < 85.0:
            return "70-84%"
        return "85%+"

    def _safe_event_tail(self, *, limit: int) -> tuple[tuple[dict[str, object], ...], str | None]:
        try:
            events = tuple(self.event_store.tail(limit=limit))
            return events, None
        except Exception as exc:
            return (), self._line(f"ops events unavailable {type(exc).__name__}", limit=_MAX_LINE_LENGTH)

    def _safe_usage_tail(self, *, limit: int) -> tuple[tuple[UsageRecord, ...], str | None]:
        try:
            usage = tuple(self.usage_store.tail(limit=limit))
            return usage, None
        except Exception as exc:
            return (), self._line(f"usage unavailable {type(exc).__name__}", limit=_MAX_LINE_LENGTH)

    def _safe_scalar(
        self,
        value: object,
        *,
        limit: int | None = None,
        redact: bool = True,
    ) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return ""
            text = f"{value:g}"
        else:
            text = str(value)
        text = _CONTROL_AND_BIDI_RE.sub(" ", text)
        text = _WHITESPACE_RE.sub(" ", text).strip()
        if not text:
            return ""
        if redact:
            text = self._redact_sensitive(text)
        if limit is not None and limit > 0:
            text = self._clip_display_text(compact_text(text, limit=max(limit * 2, limit + 8)), limit=limit)
        return text

    def _line(self, value: object, *, limit: int, redact: bool = True) -> str:
        return self._safe_scalar(value, limit=limit, redact=redact)

    def _redact_sensitive(self, text: str) -> str:
        # BREAKING: operator-facing previews now redact obvious secrets / PII before
        # rendering, so raw transcript/response/detail text may no longer be shown verbatim.
        redacted = text
        redacted = _TOKEN_RE.sub("[secret]", redacted)
        redacted = _EMAIL_RE.sub("[email]", redacted)
        redacted = _URL_RE.sub("[url]", redacted)
        redacted = _IPV4_RE.sub("[ip]", redacted)
        redacted = _PHONE_RE.sub("[phone]", redacted)
        redacted = _LONG_HEX_RE.sub("[hex]", redacted)
        redacted = _LONG_DIGIT_RE.sub("[id]", redacted)
        return redacted

    def _clip_display_text(self, text: str, *, limit: int) -> str:
        if limit <= 0 or not text:
            return ""
        if _iter_graphemes is None or _wcswidth is None:
            if len(text) <= limit:
                return text
            if limit == 1:
                return "…"
            return text[: limit - 1].rstrip() + "…"

        if self._display_width(text) <= limit:
            return text

        ellipsis_width = self._display_width("…")
        target_width = max(limit - ellipsis_width, 0)
        chunks: list[str] = []
        width = 0
        for grapheme in _iter_graphemes(text):
            grapheme_width = self._display_width(grapheme)
            if grapheme_width < 0:
                continue
            if width + grapheme_width > target_width:
                break
            chunks.append(grapheme)
            width += grapheme_width

        clipped = "".join(chunks).rstrip()
        if not clipped:
            return "…" if limit >= ellipsis_width else ""
        return f"{clipped}…"

    def _display_width(self, text: str) -> int:
        if _wcswidth is None:
            return len(text)
        width = _wcswidth(text)
        return width if width >= 0 else len(text)

    def _coerce_datetime(self, value: object) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            try:
                return datetime.fromtimestamp(float(value))
            except Exception:
                return None
        text = self._safe_scalar(value, redact=False)
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            pass
        match = re.search(r"\b(\d{2}):(\d{2})(?::\d{2})?\b", text)
        if match:
            try:
                return datetime(1900, 1, 1, int(match.group(1)), int(match.group(2)))
            except Exception:
                return None
        return None

    def _coerce_float(self, value: object) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            number = float(value)
            return number if math.isfinite(number) else None
        if isinstance(value, str):
            try:
                number = float(value.strip())
            except ValueError:
                return None
            return number if math.isfinite(number) else None
        return None

    def _coerce_int(self, value: object) -> int | None:
        number = self._coerce_float(value)
        if number is None:
            return None
        return int(number)
