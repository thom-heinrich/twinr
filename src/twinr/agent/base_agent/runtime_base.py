from __future__ import annotations

from dataclasses import dataclass, field

from twinr.agent.base_agent.adaptive_timing import AdaptiveTimingStore
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime_state import RuntimeSnapshotStore
from twinr.agent.base_agent.state_machine import TwinrStateMachine, TwinrStatus
from twinr.automations import AutomationStore
from twinr.memory import LongTermMemoryService, OnDeviceMemory, TwinrPersonalGraphStore
from twinr.memory.reminders import ReminderStore
from twinr.ops.events import TwinrOpsEventStore


@dataclass(slots=True)
class TwinrRuntimeBase:
    config: TwinrConfig
    state_machine: TwinrStateMachine = field(default_factory=TwinrStateMachine)
    memory: OnDeviceMemory = field(init=False)
    graph_memory: TwinrPersonalGraphStore = field(init=False)
    long_term_memory: LongTermMemoryService = field(init=False)
    reminder_store: ReminderStore = field(init=False)
    automation_store: AutomationStore = field(init=False)
    adaptive_timing_store: AdaptiveTimingStore = field(init=False)
    snapshot_store: RuntimeSnapshotStore = field(init=False)
    ops_events: TwinrOpsEventStore = field(init=False)
    last_transcript: str | None = None
    last_response: str | None = None
    user_voice_status: str | None = None
    user_voice_confidence: float | None = None
    user_voice_checked_at: str | None = None

    def __post_init__(self) -> None:
        self.memory = OnDeviceMemory(
            max_turns=self.config.memory_max_turns,
            keep_recent=self.config.memory_keep_recent,
        )
        self.graph_memory = TwinrPersonalGraphStore.from_config(self.config)
        self.long_term_memory = LongTermMemoryService.from_config(
            self.config,
            graph_store=self.graph_memory,
        )
        self.reminder_store = ReminderStore(
            self.config.reminder_store_path,
            timezone_name=self.config.local_timezone_name,
            retry_delay_s=self.config.reminder_retry_delay_s,
            max_entries=self.config.reminder_max_entries,
        )
        self.automation_store = AutomationStore(
            self.config.automation_store_path,
            timezone_name=self.config.local_timezone_name,
            max_entries=self.config.automation_max_entries,
        )
        self.adaptive_timing_store = AdaptiveTimingStore(
            self.config.adaptive_timing_store_path,
            config=self.config,
        )
        if self.config.adaptive_timing_enabled:
            self.adaptive_timing_store.ensure_saved()
        self.snapshot_store = RuntimeSnapshotStore(self.config.runtime_state_path)
        self.ops_events = TwinrOpsEventStore.from_config(self.config)
        if self.config.restore_runtime_state_on_startup:
            self._restore_snapshot_context()
        self._persist_snapshot()

    @property
    def status(self) -> TwinrStatus:
        return self.state_machine.status

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        self.long_term_memory.shutdown(timeout_s=timeout_s)
