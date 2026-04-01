"""Tests for the live voice-turn latency tracker."""

from __future__ import annotations

from types import SimpleNamespace

from twinr.agent.workflows.voice_turn_latency import (
    bind_voice_turn_trace,
    clear_voice_turn_latency,
    emit_voice_turn_latency_breakdown,
    mark_voice_turn_commit,
    mark_voice_turn_supervisor_ready,
    mark_voice_turn_tts_started,
    mark_voice_turn_wake_confirmed,
    record_voice_turn_remote_memory_ready,
    snapshot_voice_turn_latency,
)


def test_voice_turn_latency_breakdown_emits_expected_stage_metrics() -> None:
    loop = SimpleNamespace()
    emitted: list[str] = []
    trace_events: list[tuple[str, dict[str, object]]] = []

    mark_voice_turn_wake_confirmed(loop, source="voice_activation")
    mark_voice_turn_commit(loop, source="voice_activation")
    bind_voice_turn_trace(loop, trace_id="trace-direct", initial_source="voice_activation")
    mark_voice_turn_supervisor_ready(trace_id="trace-direct")
    record_voice_turn_remote_memory_ready(duration_ms=16450.2, trace_id="trace-direct")
    mark_voice_turn_tts_started(trace_id="trace-direct")

    snapshot = snapshot_voice_turn_latency(trace_id="trace-direct")
    assert snapshot is not None
    assert snapshot.remote_memory_reads == 1
    assert snapshot.remote_memory_total_ms == 16450

    emit_voice_turn_latency_breakdown(
        emit=emitted.append,
        trace_event=lambda name, **kwargs: trace_events.append((name, kwargs)),
        trace_id="trace-direct",
    )

    assert any(line.startswith("timing_wake_to_commit_ms=") for line in emitted)
    assert any(line.startswith("timing_commit_to_supervisor_ms=") for line in emitted)
    assert any(line.startswith("timing_supervisor_to_remote_memory_ms=") for line in emitted)
    assert any(line.startswith("timing_remote_memory_to_tts_ms=") for line in emitted)
    assert "timing_remote_memory_reads=1" in emitted
    assert "timing_remote_memory_total_ms=16450" in emitted
    assert trace_events[0][0] == "voice_turn_latency_breakdown"
    assert snapshot_voice_turn_latency(trace_id="trace-direct") is None


def test_clear_voice_turn_latency_drops_pending_and_traced_state() -> None:
    loop = SimpleNamespace()

    mark_voice_turn_wake_confirmed(loop, source="voice_activation")
    mark_voice_turn_commit(loop, source="voice_activation")
    bind_voice_turn_trace(loop, trace_id="trace-clear", initial_source="voice_activation")

    assert snapshot_voice_turn_latency(trace_id="trace-clear") is not None
    clear_voice_turn_latency(loop, trace_id="trace-clear")
    assert snapshot_voice_turn_latency(trace_id="trace-clear") is None
