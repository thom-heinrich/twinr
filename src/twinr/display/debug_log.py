"""Build operator-facing debug log sections for the e-paper display.

The debug display should reflect authoritative runtime state instead of
inventing UI-only summaries. This module gathers the latest persisted ops,
usage, and remote-memory watchdog signals and normalizes them into compact
operator log sections that fit the panel.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    "wakeword_detected",
    "wakeword_skipped",
}
_HARDWARE_EVENT_PREFIXES = (
    "button_",
    "listen_",
    "print_",
    "stt_",
    "wakeword_",
)
_KIND_LABELS = {
    "conversation": "conv",
    "print": "print",
    "proactive_prompt": "prompt",
    "search": "search",
}
_MAX_SECTION_LINES = 5
_MAX_LINE_LENGTH = 58


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

        return (
            ("System Log", self._system_lines(
                snapshot=snapshot,
                runtime_status=runtime_status,
                internet_state=internet_state,
                ai_state=ai_state,
                system_state=system_state,
                clock_text=clock_text,
                health=health,
                stale=stale,
            )),
            ("LLM Log", self._llm_lines(snapshot=snapshot, runtime_status=runtime_status)),
            ("Hardware Log", self._hardware_lines(health=health)),
        )

    def _system_lines(
        self,
        *,
        snapshot: RuntimeSnapshot | None,
        runtime_status: str,
        internet_state: str,
        ai_state: str,
        system_state: str,
        clock_text: str,
        health: TwinrSystemHealth | None,
        stale: bool,
    ) -> tuple[str, ...]:
        remote_snapshot, remote_assessment, remote_error = self._load_remote_watchdog_state()
        lines = [
            compact_text(f"{clock_text} rt {runtime_status.lower()} | sys {system_state.lower()}", limit=_MAX_LINE_LENGTH),
            compact_text(
                f"net {internet_state.lower()} | ai {ai_state.lower()} | {self._remote_status_label(remote_assessment, remote_snapshot)}",
                limit=_MAX_LINE_LENGTH,
            ),
        ]
        remote_probe = self._remote_probe_line(remote_snapshot, remote_assessment)
        if remote_probe:
            lines.append(remote_probe)

        snapshot_error = compact_text(getattr(snapshot, "error_message", None), limit=46)
        if stale:
            lines.append("snapshot stale")
        elif snapshot_error:
            lines.append(compact_text(f"runtime {snapshot_error}", limit=_MAX_LINE_LENGTH))
        elif remote_error:
            lines.append(compact_text(remote_error, limit=_MAX_LINE_LENGTH))
        else:
            remote_detail = self._remote_detail_line(remote_snapshot, remote_assessment)
            if remote_detail:
                lines.append(remote_detail)

        ops_state = self._ops_state_line(health)
        if ops_state and ops_state not in lines:
            lines.append(ops_state)
        lines.extend(self._event_lines(category="system", limit=self.max_section_lines - len(lines)))
        return self._finalize_lines(lines, empty_label="no recent system events")

    def _llm_lines(
        self,
        *,
        snapshot: RuntimeSnapshot | None,
        runtime_status: str,
    ) -> tuple[str, ...]:
        lines: list[str] = []
        latest_usage = self._latest_usage_record()
        if latest_usage is not None:
            request_source = compact_text(
                (latest_usage.metadata or {}).get("request_source") or latest_usage.source or "?",
                limit=10,
            )
            lines.append(compact_text(f"mode {runtime_status.lower()} | src {request_source}", limit=_MAX_LINE_LENGTH))
            model = compact_text(latest_usage.model or "unknown", limit=26)
            web_text = "yes" if latest_usage.used_web_search else "no"
            lines.append(compact_text(f"model {model} | web {web_text}", limit=_MAX_LINE_LENGTH))
        else:
            lines.append(compact_text(f"mode {runtime_status.lower()} | src -", limit=_MAX_LINE_LENGTH))

        last_transcript = compact_text(getattr(snapshot, "last_transcript", None), limit=44)
        last_response = compact_text(getattr(snapshot, "last_response", None), limit=46)
        if last_transcript:
            lines.append(compact_text(f"user {last_transcript}", limit=_MAX_LINE_LENGTH))
        if last_response:
            lines.append(compact_text(f"ai {last_response}", limit=_MAX_LINE_LENGTH))

        remaining = max(self.max_section_lines - len(lines), 0)
        if remaining > 0:
            llm_event_lines = self._event_lines(category="llm", limit=remaining)
            lines.extend(llm_event_lines)
        remaining = max(self.max_section_lines - len(lines), 0)
        if remaining > 0:
            lines.extend(self._usage_lines(limit=remaining))
        return self._finalize_lines(lines, empty_label="no recent llm activity")

    def _hardware_lines(self, *, health: TwinrSystemHealth | None) -> tuple[str, ...]:
        lines: list[str] = []
        service_line = self._hardware_service_line(health)
        if service_line:
            lines.append(service_line)
        host_line = self._hardware_host_line(health)
        if host_line:
            lines.append(host_line)
        respeaker_state = parse_respeaker_hci_state(self.event_store.tail(limit=32))
        if respeaker_state is not None:
            for line in respeaker_state.hardware_log_lines():
                if line not in lines:
                    lines.append(line)
                if len(lines) >= self.max_section_lines:
                    break
        lines.extend(self._event_lines(category="hardware", limit=self.max_section_lines - len(lines)))
        return self._finalize_lines(lines, empty_label="no recent hardware events")

    def _event_lines(self, *, category: str, limit: int) -> tuple[str, ...]:
        if limit <= 0:
            return ()
        raw_lines: list[str] = []
        for entry in reversed(self.event_store.tail(limit=48)):
            event_name = compact_text(str(entry.get("event", "")), limit=80)
            if self._event_category(event_name) != category:
                continue
            raw_lines.append(self._format_event_line(event_name, entry))
            if len(raw_lines) >= max(limit * 6, limit + 4):
                break
        compressed = self._compress_repeated_lines(raw_lines)
        return tuple(compressed[:limit])

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
            return compact_text(f"{time_text} button {payload.get('button', 'press')}", limit=_MAX_LINE_LENGTH)
        if event_name == "wakeword_detected":
            phrase = compact_text(str(payload.get("matched_phrase", "wakeword")), limit=28)
            return compact_text(f"{time_text} wakeword {phrase}", limit=_MAX_LINE_LENGTH)
        if event_name == "wakeword_skipped":
            reason = compact_text(str(payload.get("skip_reason", "busy")), limit=24)
            return compact_text(f"{time_text} wakeword skip {reason}", limit=_MAX_LINE_LENGTH)
        if event_name == "print_job_sent":
            return compact_text(f"{time_text} print sent", limit=_MAX_LINE_LENGTH)
        if event_name == "print_failed":
            return compact_text(f"{time_text} print failed", limit=_MAX_LINE_LENGTH)
        if event_name == "print_skipped":
            return compact_text(f"{time_text} print skipped", limit=_MAX_LINE_LENGTH)
        if event_name == "listen_timeout":
            source = compact_text(str(payload.get("request_source", "")), limit=12)
            suffix = f" {source}" if source else ""
            return compact_text(f"{time_text} listen timeout{suffix}", limit=_MAX_LINE_LENGTH)
        if event_name == "stt_failed":
            return compact_text(f"{time_text} stt failed", limit=_MAX_LINE_LENGTH)
        if event_name == "hardware_loop_error":
            return compact_text(f"{time_text} hardware error", limit=_MAX_LINE_LENGTH)
        if event_name == "remote_memory_watchdog_status_changed":
            status = compact_text(str(payload.get("status", "unknown")), limit=12)
            return compact_text(f"{time_text} remote {status}", limit=_MAX_LINE_LENGTH)
        if event_name == "remote_memory_watchdog_started":
            return compact_text(f"{time_text} remote watchdog start", limit=_MAX_LINE_LENGTH)
        if event_name == "required_remote_watchdog_artifact_failed":
            return compact_text(f"{time_text} remote artifact failed", limit=_MAX_LINE_LENGTH)
        if event_name == "automation_execution_failed":
            name = compact_text(str(payload.get("name", "automation")), limit=24)
            return compact_text(f"{time_text} auto fail {name}", limit=_MAX_LINE_LENGTH)
        if event_name == "conversation_session_skipped":
            return compact_text(f"{time_text} session busy", limit=_MAX_LINE_LENGTH)
        if event_name == "conversation_closure_detected":
            return compact_text(f"{time_text} follow-up closed", limit=_MAX_LINE_LENGTH)
        if event_name == "proactive_governor_blocked":
            reason = compact_text(str(payload.get("reason", "blocked")), limit=22)
            return compact_text(f"{time_text} proactive {reason}", limit=_MAX_LINE_LENGTH)
        if event_name == "runtime_supervisor_child_started":
            child = compact_text(str(payload.get("key", "child")), limit=18)
            return compact_text(f"{time_text} start {child}", limit=_MAX_LINE_LENGTH)
        if event_name == "runtime_supervisor_child_restart_requested":
            child = compact_text(str(payload.get("key", "child")), limit=18)
            return compact_text(f"{time_text} restart {child}", limit=_MAX_LINE_LENGTH)
        if event_name == "turn_started":
            source = compact_text(str(payload.get("request_source", "")), limit=12)
            suffix = f" {source}" if source else ""
            return compact_text(f"{time_text} turn start{suffix}", limit=_MAX_LINE_LENGTH)
        if event_name == "transcript_submitted":
            return compact_text(f"{time_text} transcript ready", limit=_MAX_LINE_LENGTH)
        if event_name == "turn_completed":
            status = compact_text(str(payload.get("status", "done")), limit=14)
            return compact_text(f"{time_text} turn {status}", limit=_MAX_LINE_LENGTH)
        if event_name == "follow_up_rearmed":
            return compact_text(f"{time_text} follow-up armed", limit=_MAX_LINE_LENGTH)
        if event_name == "adaptive_timing_updated":
            return compact_text(f"{time_text} timing adapted", limit=_MAX_LINE_LENGTH)

        message = compact_text(str(entry.get("message", event_name)), limit=44)
        return compact_text(f"{time_text} {message}", limit=_MAX_LINE_LENGTH)

    def _format_usage_record(self, record: UsageRecord) -> str:
        time_text = self._time_text(record.created_at)
        kind = _KIND_LABELS.get(record.request_kind, compact_text(record.request_kind, limit=10))
        source = compact_text((record.metadata or {}).get("request_source") or record.source, limit=10)
        subject = compact_text(
            (record.metadata or {}).get("transcript")
            or (record.metadata or {}).get("question")
            or record.model
            or "activity",
            limit=30,
        )
        web_suffix = " web" if record.used_web_search else ""
        return compact_text(f"{time_text} {kind} {source}{web_suffix} {subject}", limit=_MAX_LINE_LENGTH)

    def _time_text(self, value: object) -> str:
        text = str(value or "").strip()
        if len(text) >= 16 and "T" in text:
            return text[11:16]
        return "--:--"

    def _compress_repeated_lines(self, lines: list[str]) -> list[str]:
        compressed: list[tuple[str, int]] = []
        for line in lines:
            if not line:
                continue
            if compressed and compressed[-1][0] == line:
                previous_line, previous_count = compressed[-1]
                compressed[-1] = (previous_line, previous_count + 1)
                continue
            compressed.append((line, 1))
        result: list[str] = []
        for line, count in compressed:
            if count > 1:
                result.append(compact_text(f"{line} x{count}", limit=_MAX_LINE_LENGTH))
            else:
                result.append(line)
        return result

    def _finalize_lines(self, lines: list[str], *, empty_label: str) -> tuple[str, ...]:
        final_lines = tuple(line for line in lines[: self.max_section_lines] if line)
        return final_lines or (empty_label,)

    def _usage_lines(self, *, limit: int) -> tuple[str, ...]:
        if limit <= 0:
            return ()
        usage_lines: list[str] = []
        for record in reversed(self.usage_store.tail(limit=12)):
            usage_lines.append(self._format_usage_record(record))
            if len(usage_lines) >= limit * 2:
                break
        compressed = self._compress_repeated_lines(usage_lines)
        return tuple(compressed[:limit])

    def _latest_usage_record(self) -> UsageRecord | None:
        recent = self.usage_store.tail(limit=1)
        if not recent:
            return None
        return recent[-1]

    def _load_remote_watchdog_state(
        self,
    ) -> tuple[RemoteMemoryWatchdogSnapshot | None, RequiredRemoteWatchdogAssessment | None, str | None]:
        try:
            snapshot = self.watchdog_store.load()
            assessment = assess_required_remote_watchdog_snapshot(self.config, store=self.watchdog_store)
            return (snapshot, assessment, None)
        except Exception as exc:
            return (None, None, compact_text(f"chonky unknown {type(exc).__name__}", limit=_MAX_LINE_LENGTH))

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
            sample_status = compact_text(assessment.sample_status or "fail", limit=10).lower() or "fail"
            if assessment.snapshot_stale:
                return "chonky stale"
            return compact_text(f"chonky {sample_status}", limit=16)
        if snapshot is not None:
            sample_status = compact_text(snapshot.current.status or "?", limit=10).lower() or "?"
            return compact_text(f"chonky {sample_status}", limit=16)
        return "chonky ?"

    def _remote_probe_line(
        self,
        snapshot: RemoteMemoryWatchdogSnapshot | None,
        assessment: RequiredRemoteWatchdogAssessment | None,
    ) -> str | None:
        if snapshot is None:
            return None
        sample_status = compact_text(snapshot.current.status or "?", limit=10).lower() or "?"
        if bool(getattr(snapshot.current, "ready", False)) and sample_status == "ok":
            sampled_at = self._time_text(snapshot.current.captured_at or snapshot.last_ok_at)
            return compact_text(
                f"last ok {sampled_at} | fail {snapshot.failure_count}",
                limit=_MAX_LINE_LENGTH,
            )
        if sample_status == "fail" or (
            assessment is not None
            and not assessment.ready
            and not assessment.snapshot_stale
            and sample_status not in {"starting", "disabled"}
        ):
            failed_at = self._time_text(snapshot.last_failure_at or snapshot.current.captured_at)
            if failed_at != "--:--":
                return compact_text(f"last fail {failed_at} | retrying", limit=_MAX_LINE_LENGTH)
            return "remote fail | retrying"
        if bool(getattr(snapshot, "probe_inflight", False)):
            started_at = self._time_text(getattr(snapshot, "probe_started_at", None))
            return compact_text(
                f"probe inflight | start {started_at}",
                limit=_MAX_LINE_LENGTH,
            )
        sampled_at = self._time_text(snapshot.current.captured_at)
        return compact_text(f"sample {sampled_at} | {sample_status}", limit=_MAX_LINE_LENGTH)

    def _remote_detail_line(
        self,
        snapshot: RemoteMemoryWatchdogSnapshot | None,
        assessment: RequiredRemoteWatchdogAssessment | None,
    ) -> str | None:
        detail = ""
        if assessment is not None and not assessment.ready:
            detail = compact_text(assessment.detail, limit=42)
        elif snapshot is not None:
            detail = compact_text(snapshot.current.detail, limit=42)
        if not detail:
            return None
        return compact_text(f"detail {detail}", limit=_MAX_LINE_LENGTH)

    def _ops_state_line(self, health: TwinrSystemHealth | None) -> str | None:
        if health is None:
            return None
        conversation_count = self._service_count(health.services, "conversation_loop")
        display_count = self._service_count(health.services, "display")
        return compact_text(
            f"errs {health.recent_error_count} | conv {conversation_count} | disp {display_count}",
            limit=_MAX_LINE_LENGTH,
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
        return compact_text(
            f"conv pid {conversation_pid or '-'} | disp pid {display_pid or '-'}",
            limit=_MAX_LINE_LENGTH,
        )

    def _hardware_host_line(self, health: TwinrSystemHealth | None) -> str | None:
        if health is None:
            return None
        cpu_text = self._format_threshold_band(
            health.cpu_temperature_c,
            thresholds=(
                (70.0, "<70C"),
                (80.0, "70-79C"),
            ),
            fallback="80C+",
        )
        mem_text = self._format_threshold_band(
            health.memory_used_percent,
            thresholds=(
                (40.0, "<40%"),
                (60.0, "40-59%"),
                (80.0, "60-79%"),
            ),
            fallback="80%+",
        )
        disk_text = self._format_threshold_band(
            health.disk_used_percent,
            thresholds=(
                (70.0, "<70%"),
                (85.0, "70-84%"),
            ),
            fallback="85%+",
        )
        return compact_text(f"cpu {cpu_text} | mem {mem_text} | disk {disk_text}", limit=_MAX_LINE_LENGTH)

    def _service(self, services: tuple[ServiceHealth, ...], key: str) -> ServiceHealth | None:
        for service in services:
            if service.key == key:
                return service
        return None

    def _service_count(self, services: tuple[ServiceHealth, ...], key: str) -> int:
        service = self._service(services, key)
        if service is None:
            return 0
        return int(service.count)

    def _service_pid(self, service: ServiceHealth | None) -> str | None:
        if service is None:
            return None
        for token in str(service.detail or "").split():
            if token.startswith("pid="):
                pid = token.split("=", 1)[1].strip()
                if pid:
                    return pid
        return None

    def _format_latency_ms(self, value: float | None) -> str:
        if value is None:
            return "?"
        if value >= 1000.0:
            return f"{value:.0f}ms"
        return f"{value:.1f}ms"

    def _format_bucketed_metric(self, value: float | None, *, bucket: float, suffix: str) -> str:
        if value is None:
            return "?"
        if bucket <= 0.0:
            return f"{value:.1f}{suffix}"
        bucketed = round(float(value) / bucket) * bucket
        if suffix == "C":
            return f"{bucketed:.0f}{suffix}"
        return f"{bucketed:.0f}{suffix}"

    def _format_range_band(self, value: float | None, *, bucket: float, suffix: str) -> str:
        if value is None:
            return "?"
        if bucket <= 0.0:
            return f"{value:.1f}{suffix}"
        bucket_value = float(bucket)
        lower = int((float(value) // bucket_value) * bucket_value)
        upper = lower + int(bucket_value) - 1
        return f"{lower}-{upper}{suffix}"

    def _format_threshold_band(
        self,
        value: float | None,
        *,
        thresholds: tuple[tuple[float, str], ...],
        fallback: str,
    ) -> str:
        if value is None:
            return "?"
        resolved = float(value)
        for upper_bound, label in thresholds:
            if resolved < upper_bound:
                return label
        return fallback
