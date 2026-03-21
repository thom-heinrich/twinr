"""Run a bounded multi-call eval for personality formation and reflection.

Purpose
-------
Exercise Twinr's current long-horizon personality stack against a bounded
multi-day scenario and report which parts are already proven end-to-end:

- raw conversation turns persisted through long-term memory and reflection
- raw free-conversation style/humor cues reaching personality learning
- structured downstream personality learning persisted through remote state
- world-intelligence refresh/recalibration operating inside the same isolated
  remote namespace
- prompt-facing downstream behavior changing after the accumulated evidence

The eval is intentionally explicit about current boundaries. It separates the
``raw conversation -> memory/reflection`` path from the ``structured extracted
evidence -> personality drift`` path so the report can show where the system is
already fully connected and where the stack still depends on upstream extractors
becoming richer.

Usage
-----
Command-line invocation examples::

    PYTHONPATH=src python3 test/run_personality_formation_eval.py --env-file .env
    PYTHONPATH=src python3 test/run_personality_formation_eval.py --env-file /twinr/.env --output artifacts/reports/personality_formation_eval_latest.json

Inputs
------
- ``--env-file``: Twinr env file used to build the runtime config.
- ``--run-id``: Optional suffix for the isolated remote namespace.
- ``--clone-live-baseline``: Copy the current live personality/world baseline
  into the isolated eval namespace before the scenario runs.
- ``--output``: Optional JSON report path.

Outputs
-------
- JSON summary written to stdout.
- Optional JSON artifact written to ``--output``.

Notes
-----
The script uses a unique remote namespace so it can exercise real ChonkyDB-
backed persistence without mutating Twinr's live durable state. Reflection,
personality learning, and world-intelligence refresh all run through the real
runtime services used in production.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult
from twinr.agent.personality.intelligence import (
    RemoteStateWorldIntelligenceStore,
    WorldFeedSubscription,
    WorldIntelligenceState,
)
from twinr.agent.personality.models import PersonalitySnapshot
from twinr.agent.personality.positive_engagement import build_positive_engagement_policies
from twinr.agent.personality.self_expression import build_mindshare_items
from twinr.agent.personality.signals import PersonalitySignalBatch
from twinr.agent.personality.steering import build_turn_steering_cues
from twinr.agent.personality.store import RemoteStatePersonalitySnapshotStore
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermConversationTurn,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore


def _utcnow() -> datetime:
    """Return the current aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    """Return the current UTC timestamp as ISO-8601 text."""

    return _utcnow().isoformat()


def _isoformat(value: datetime) -> str:
    """Render one aware timestamp as UTC ISO-8601 text."""

    return value.astimezone(timezone.utc).isoformat()


def _clean_topic_key(value: object | None) -> str:
    """Normalize one topic-like value into a stable comparison key."""

    return " ".join(str(value or "").split()).strip().casefold()


def _emit_progress(*, enabled: bool, message: str) -> None:
    """Write one bounded progress line to stderr when requested."""

    if not enabled:
        return
    sys.stderr.write(f"[personality-eval] {message}\n")
    sys.stderr.flush()


@dataclass(frozen=True, slots=True)
class RawTurnSpec:
    """Describe one raw conversation turn persisted through long-term memory."""

    name: str
    transcript: str
    response: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class StructuredLearningSpec:
    """Describe one structured downstream personality-learning event."""

    name: str
    transcript: str
    response: str
    occurred_at: datetime
    durable_objects: tuple[LongTermMemoryObjectV1, ...]


@dataclass(frozen=True, slots=True)
class ToolHistorySpec:
    """Describe one structured tool-history event for world/personality learning."""

    name: str
    tool_call: AgentToolCall
    tool_result: AgentToolResult


@dataclass(frozen=True, slots=True)
class EvalDaySpec:
    """Describe one simulated eval day."""

    name: str
    raw_turns: tuple[RawTurnSpec, ...]
    structured_learning: tuple[StructuredLearningSpec, ...]
    tool_history: tuple[ToolHistorySpec, ...]


def _source_ref(turn_id: str) -> LongTermSourceRefV1:
    """Build one canonical conversation-turn source reference."""

    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=(turn_id,),
        speaker="user",
        modality="voice",
    )


def _structured_memory(
    *,
    memory_id: str,
    turn_id: str,
    kind: str,
    summary: str,
    confidence: float,
    attributes: dict[str, object],
    status: str = "active",
    slot_key: str | None = None,
    value_key: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    sensitivity: str = "normal",
) -> LongTermMemoryObjectV1:
    """Build one long-term object fixture for structured learning."""

    return LongTermMemoryObjectV1(
        memory_id=memory_id,
        kind=kind,
        summary=summary,
        source=_source_ref(turn_id),
        status=status,
        confidence=confidence,
        slot_key=slot_key,
        value_key=value_key,
        valid_from=valid_from,
        valid_to=valid_to,
        sensitivity=sensitivity,
        attributes=attributes,
    )


def _default_eval_days(*, now: datetime | None = None) -> tuple[EvalDaySpec, ...]:
    """Build the bounded multi-day scenario used by the eval.

    The raw-turn appointment thread must stay temporally fresh relative to the
    actual eval runtime so retention does not immediately archive the very
    summary/continuity chain this harness is trying to prove.
    """

    anchor = (now or _utcnow()).astimezone(timezone.utc)
    day_1 = (anchor + timedelta(days=2)).replace(hour=9, minute=0, second=0, microsecond=0)
    day_2 = day_1 + timedelta(days=1)
    janina_appointment_day = (day_1 + timedelta(days=1)).date().isoformat()
    return (
        EvalDaySpec(
            name="day_1_starting_threads",
            raw_turns=(
                RawTurnSpec(
                    name="janina_relationship",
                    transcript="Janina is my wife.",
                    response="I will keep Janina as an important person in mind.",
                    occurred_at=day_1,
                ),
                RawTurnSpec(
                    name="janina_appointment",
                    transcript=f"Janina has eye laser treatment at the eye doctor on {janina_appointment_day}.",
                    response="I will remember that Janina has that appointment.",
                    occurred_at=day_1 + timedelta(minutes=5),
                ),
                RawTurnSpec(
                    name="style_verbosity_free_1",
                    transcript="When you answer me, shorter and calmer usually helps me more than a lot at once.",
                    response="I will keep answers shorter and calmer by default.",
                    occurred_at=day_1 + timedelta(minutes=10),
                ),
                RawTurnSpec(
                    name="style_initiative_free_1",
                    transcript="If you sometimes ask one small follow-up when it really helps, that usually works well for me.",
                    response="I can occasionally bring one small helpful follow-up.",
                    occurred_at=day_1 + timedelta(minutes=15),
                ),
                RawTurnSpec(
                    name="humor_feedback_free_1",
                    transcript="That small dry joke actually worked for me.",
                    response="I will remember that this kind of quiet dry humor can land well here.",
                    occurred_at=day_1 + timedelta(minutes=20),
                ),
            ),
            structured_learning=(
                StructuredLearningSpec(
                    name="topic_follow_up_ai",
                    transcript="Tell me more about AI companions, I want to come back to that often.",
                    response="I will keep AI companions in focus.",
                    occurred_at=day_1 + timedelta(minutes=25),
                    durable_objects=(
                        _structured_memory(
                            memory_id="pref:topic_follow_up:ai_companions",
                            turn_id="turn:eval:day1:ai_followup",
                            kind="fact",
                            summary="The user explicitly wants to keep talking about AI companions.",
                            confidence=0.91,
                            attributes={
                                "memory_domain": "preference",
                                "fact_type": "preference",
                                "preference_type": "topic_follow_up",
                                "topic": "AI companions",
                                "support_count": 2,
                            },
                        ),
                    ),
                ),
            ),
            tool_history=(
                ToolHistorySpec(
                    name="tool_ai_companions_search",
                    tool_call=AgentToolCall(
                        name="search_live_info",
                        call_id="call:eval:day1:ai",
                        arguments={"question": "What changed in AI companions this week?"},
                    ),
                    tool_result=AgentToolResult(
                        call_id="call:eval:day1:ai",
                        name="search_live_info",
                        output={
                            "status": "ok",
                            "answer": "A few AI companion products focused more on calm daily usefulness.",
                        },
                        serialized_output='{"status":"ok"}',
                    ),
                ),
            ),
        ),
        EvalDaySpec(
            name="day_2_personality_forming_and_boundaries",
            raw_turns=(
                RawTurnSpec(
                    name="style_verbosity_free_2",
                    transcript="The short and clear way from earlier still fits me best.",
                    response="I will keep that shorter style active.",
                    occurred_at=day_2 + timedelta(minutes=9),
                ),
                RawTurnSpec(
                    name="style_initiative_free_2",
                    transcript="You can occasionally suggest one next step when it is genuinely useful.",
                    response="I can stay gently proactive in relaxed moments.",
                    occurred_at=day_2 + timedelta(minutes=12),
                ),
                RawTurnSpec(
                    name="humor_feedback_free_2",
                    transcript="That second understated dry joke also landed well with me.",
                    response="I will keep that humor available sparingly.",
                    occurred_at=day_2 + timedelta(minutes=15),
                ),
            ),
            structured_learning=(
                StructuredLearningSpec(
                    name="topic_follow_up_world_politics",
                    transcript="I want to come back to diplomacy and world politics as well.",
                    response="I will keep that thread in view too.",
                    occurred_at=day_2 + timedelta(minutes=18),
                    durable_objects=(
                        _structured_memory(
                            memory_id="pref:topic_follow_up:world_politics",
                            turn_id="turn:eval:day2:world_followup",
                            kind="fact",
                            summary="The user explicitly wants to keep talking about world politics and diplomacy.",
                            confidence=0.9,
                            attributes={
                                "memory_domain": "preference",
                                "fact_type": "preference",
                                "preference_type": "topic_follow_up",
                                "topic": "world politics",
                                "support_count": 2,
                            },
                        ),
                    ),
                ),
                StructuredLearningSpec(
                    name="topic_follow_up_ai_again",
                    transcript="Keep AI companions on the list, I want to return to that.",
                    response="I will keep that topic active.",
                    occurred_at=day_2 + timedelta(minutes=22),
                    durable_objects=(
                        _structured_memory(
                            memory_id="pref:topic_follow_up:ai_companions:2",
                            turn_id="turn:eval:day2:ai_followup_repeat",
                            kind="fact",
                            summary="The user again wants to come back to AI companions.",
                            confidence=0.9,
                            attributes={
                                "memory_domain": "preference",
                                "fact_type": "preference",
                                "preference_type": "topic_follow_up",
                                "topic": "AI companions",
                                "support_count": 2,
                            },
                        ),
                    ),
                ),
                StructuredLearningSpec(
                    name="topic_aversion_celebrity_gossip",
                    transcript="Please do not bring me celebrity gossip.",
                    response="Understood, I will leave that alone.",
                    occurred_at=day_2 + timedelta(minutes=26),
                    durable_objects=(
                        _structured_memory(
                            memory_id="pref:topic_aversion:celebrity_gossip",
                            turn_id="turn:eval:day2:avoid_gossip",
                            kind="fact",
                            summary="The user does not want celebrity gossip.",
                            confidence=0.88,
                            attributes={
                                "memory_domain": "preference",
                                "fact_type": "preference",
                                "preference_type": "topic_aversion",
                                "topic": "celebrity gossip",
                                "support_count": 2,
                            },
                        ),
                    ),
                ),
                StructuredLearningSpec(
                    name="local_politics_cooling",
                    transcript="Local politics is not something I want too often unless I ask.",
                    response="All right, I will keep local politics quieter unless you pull it forward.",
                    occurred_at=day_2 + timedelta(minutes=30),
                    durable_objects=(
                        _structured_memory(
                            memory_id="pattern:topic_non_reengagement:local_politics",
                            turn_id="turn:eval:day2:local_cooling",
                            kind="pattern",
                            summary="Local politics did not pull the user back in after repeated exposure.",
                            confidence=0.8,
                            attributes={
                                "memory_domain": "pattern",
                                "pattern_type": "topic_non_reengagement",
                                "topic": "local politics",
                                "support_count": 2,
                                "exposure_count": 3,
                                "non_reengagement_count": 2,
                                "exposure_aware": True,
                            },
                        ),
                    ),
                ),
            ),
            tool_history=(
                ToolHistorySpec(
                    name="tool_world_politics_search",
                    tool_call=AgentToolCall(
                        name="search_live_info",
                        call_id="call:eval:day2:world",
                        arguments={"question": "What changed in world politics and diplomacy today?"},
                    ),
                    tool_result=AgentToolResult(
                        call_id="call:eval:day2:world",
                        name="search_live_info",
                        output={
                            "status": "ok",
                            "answer": "Several diplomatic talks focused on de-escalation.",
                        },
                        serialized_output='{"status":"ok"}',
                    ),
                ),
                ToolHistorySpec(
                    name="tool_local_politics_search",
                    tool_call=AgentToolCall(
                        name="search_live_info",
                        call_id="call:eval:day2:local",
                        arguments={
                            "question": "What changed in local politics around Hamburg today?",
                            "location_hint": "Hamburg",
                        },
                    ),
                    tool_result=AgentToolResult(
                        call_id="call:eval:day2:local",
                        name="search_live_info",
                        output={
                            "status": "ok",
                            "answer": "A local transit budget update was discussed in Hamburg.",
                        },
                        serialized_output='{"status":"ok"}',
                    ),
                ),
            ),
        ),
    )


def _with_isolated_namespace(config: TwinrConfig, *, namespace: str, temp_root: Path) -> TwinrConfig:
    """Return an eval config that writes into isolated remote and local paths."""

    repo_root = Path(config.project_root).resolve()
    return replace(
        config,
        project_root=str(repo_root),
        personality_dir=str((repo_root / "personality").resolve()),
        memory_markdown_path=str((temp_root / "state" / "MEMORY.md").resolve()),
        reminder_store_path=str((temp_root / "state" / "reminders.json").resolve()),
        long_term_memory_path=str((temp_root / "state" / "chonkydb").resolve()),
        long_term_memory_remote_namespace=namespace,
        long_term_memory_enabled=True,
        long_term_memory_mode="remote_primary",
    )


def _clone_live_baseline(*, base_config: TwinrConfig, eval_config: TwinrConfig) -> dict[str, object]:
    """Clone the current live snapshot/subscriptions into the isolated eval namespace."""

    base_remote = LongTermRemoteStateStore.from_config(base_config)
    eval_remote = LongTermRemoteStateStore.from_config(eval_config)
    personality_store = RemoteStatePersonalitySnapshotStore()
    world_store = RemoteStateWorldIntelligenceStore()

    copied_snapshot = False
    copied_subscription_count = 0
    copied_interest_signal_count = 0

    live_snapshot = personality_store.load_snapshot(config=base_config, remote_state=base_remote)
    if live_snapshot is not None:
        personality_store.save_snapshot(
            config=eval_config,
            snapshot=live_snapshot,
            remote_state=eval_remote,
        )
        copied_snapshot = True

    live_subscriptions = world_store.load_subscriptions(config=base_config, remote_state=base_remote)
    if live_subscriptions:
        refreshed = tuple(
            replace(
                item,
                last_checked_at=None,
                last_refreshed_at=None,
                last_error=None,
                last_item_ids=(),
            )
            for item in live_subscriptions
        )
        world_store.save_subscriptions(
            config=eval_config,
            subscriptions=refreshed,
            remote_state=eval_remote,
        )
        copied_subscription_count = len(refreshed)

    live_state = world_store.load_state(config=base_config, remote_state=base_remote)
    copied_interest_signal_count = len(live_state.interest_signals)
    world_store.save_state(
        config=eval_config,
        state=replace(
            live_state,
            last_refreshed_at=None,
        ),
        remote_state=eval_remote,
    )
    return {
        "copied_snapshot": copied_snapshot,
        "copied_subscription_count": copied_subscription_count,
        "copied_interest_signal_count": copied_interest_signal_count,
    }


def _persist_raw_turn_with_preview(
    service: LongTermMemoryService,
    spec: RawTurnSpec,
) -> dict[str, object]:
    """Persist one raw turn and return the inline reflection/learning metrics.

    This keeps the eval bounded: the expensive structured-turn extractor runs
    only once per raw turn, while the helper still returns the same reflection
    and learning metrics the report needs.
    """

    item = LongTermConversationTurn(
        transcript=spec.transcript,
        response=spec.response,
        created_at=spec.occurred_at,
    )
    with service._store_lock:
        existing_objects = tuple(service.object_store.load_objects())
        existing_conflicts = tuple(service.object_store.load_conflicts())
        existing_archived = tuple(service.object_store.load_archived_objects())
        extraction = service.extractor.extract_conversation_turn(
            transcript=spec.transcript,
            response=spec.response,
            occurred_at=spec.occurred_at,
        )
        consolidation = service.consolidator.consolidate(
            extraction=extraction,
            existing_objects=existing_objects,
        )
        current_objects, current_conflicts = LongTermMemoryService._merge_consolidation_state(
            object_store=service.object_store,
            existing_objects=existing_objects,
            existing_conflicts=existing_conflicts,
            result=consolidation,
        )
        reflection = service.reflector.reflect(objects=current_objects)
        if LongTermMemoryService._has_reflection_payload(reflection):
            current_objects = LongTermMemoryService._merge_reflection_objects(
                object_store=service.object_store,
                current_objects=current_objects,
                reflection=reflection,
            )
        sensor_reflection = service.sensor_memory.compile(
            objects=current_objects,
            now=spec.occurred_at,
        )
        if LongTermMemoryService._has_reflection_payload(sensor_reflection):
            current_objects = LongTermMemoryService._merge_reflection_objects(
                object_store=service.object_store,
                current_objects=current_objects,
                reflection=sensor_reflection,
            )
        learning_consolidation = LongTermMemoryService._merge_reflection_into_consolidation(
            result=consolidation,
            reflection_batches=(reflection, sensor_reflection),
        )
        learning_batch = PersonalitySignalBatch()
        if service.personality_learning is not None:
            learning_batch = service.personality_learning.extractor.extract_from_consolidation(
                turn=item,
                consolidation=learning_consolidation,
            )
        interaction_counts = Counter(
            signal.signal_kind for signal in learning_batch.interaction_signals
        )
        retention = LongTermMemoryService._apply_retention_or_keep(
            retention_policy=service.retention_policy,
            objects=current_objects,
        )
        service.object_store.write_snapshot(
            objects=retention.kept_objects,
            conflicts=current_conflicts,
            archived_objects=LongTermMemoryService._merge_archived_objects(
                existing_archived=existing_archived,
                archived_updates=retention.archived_objects,
            ),
        )
        try:
            service.graph_store.apply_candidate_edges(consolidation.graph_edges)
        except Exception:
            pass
        try:
            if LongTermMemoryService._has_reflection_payload(reflection):
                service.midterm_store.apply_reflection(reflection)
            if LongTermMemoryService._has_reflection_payload(sensor_reflection):
                service.midterm_store.apply_reflection(sensor_reflection)
        except Exception:
            pass
        if service.personality_learning is not None:
            service.personality_learning.record_conversation_consolidation(
                turn=item,
                consolidation=learning_consolidation,
            )
        return {
            "reflected_objects": len(reflection.reflected_objects) + len(sensor_reflection.reflected_objects),
            "created_summaries": len(reflection.created_summaries) + len(sensor_reflection.created_summaries),
            "midterm_packets": len(reflection.midterm_packets) + len(sensor_reflection.midterm_packets),
            "interaction_signal_counts": {
                kind: int(count)
                for kind, count in sorted(interaction_counts.items())
            },
            "style_signal_counts": {
                "verbosity_preference": int(interaction_counts.get("verbosity_preference", 0)),
                "initiative_preference": int(interaction_counts.get("initiative_preference", 0)),
                "humor_feedback": int(interaction_counts.get("humor_feedback", 0)),
            },
        }


def _consolidation_result_from_structured_spec(spec: StructuredLearningSpec) -> LongTermConsolidationResultV1:
    """Build one typed consolidation result for a structured learning spec."""

    turn_id = f"turn:eval:{spec.name}"
    return LongTermConsolidationResultV1(
        turn_id=turn_id,
        occurred_at=spec.occurred_at,
        episodic_objects=(),
        durable_objects=spec.durable_objects,
        deferred_objects=(),
        conflicts=(),
        graph_edges=(),
    )


def _persist_structured_learning_batch(
    service: LongTermMemoryService,
    specs: Sequence[StructuredLearningSpec],
) -> None:
    """Persist one simulated day's structured learning in a single commit.

    The eval keeps the underlying conversation evidence separate per spec, but
    batches their downstream signal commits per day. That preserves the
    day-level shape of repeated user interactions while avoiding an unrealistic
    snapshot storm against remote-primary state during the acceptance run.
    """

    if service.personality_learning is None:
        raise RuntimeError("personality learning is not configured")
    if not specs:
        return
    learning = service.personality_learning
    combined_batch = PersonalitySignalBatch()
    combined_interest_signals = ()
    existing_interest_signals = ()
    if learning.world_intelligence is not None:
        existing_interest_signals = learning.world_intelligence.store.load_state(
            config=learning.world_intelligence.config,
            remote_state=learning.world_intelligence.remote_state,
        ).interest_signals
    running_interest_signals = tuple(existing_interest_signals)

    for spec in specs:
        consolidation = _consolidation_result_from_structured_spec(spec)
        turn = LongTermConversationTurn(
            transcript=spec.transcript,
            response=spec.response,
            created_at=spec.occurred_at,
        )
        batch = learning.extractor.extract_from_consolidation(
            turn=turn,
            consolidation=consolidation,
        )
        combined_batch = combined_batch.merged(batch)
        world_interest_batch = learning.world_interest_extractor.extract_from_personality_batch(
            turn_id=consolidation.turn_id,
            batch=batch,
            occurred_at=consolidation.occurred_at,
            existing_interest_signals=running_interest_signals,
        )
        if world_interest_batch.interest_signals:
            combined_interest_signals = (
                *combined_interest_signals,
                *world_interest_batch.interest_signals,
            )
            running_interest_signals = (
                *running_interest_signals,
                *world_interest_batch.interest_signals,
            )

    if combined_interest_signals:
        learning._record_world_interest_signals(combined_interest_signals)
    learning._commit(combined_batch)


def _record_tool_history(service: LongTermMemoryService, tool_specs: Sequence[ToolHistorySpec]) -> None:
    """Persist one batch of structured tool-history learning signals."""

    if not tool_specs:
        return
    service.record_personality_tool_history(
        tool_calls=tuple(item.tool_call for item in tool_specs),
        tool_results=tuple(item.tool_result for item in tool_specs),
    )


def _snapshot_metrics(snapshot: PersonalitySnapshot | None) -> dict[str, object]:
    """Project a personality snapshot into a compact eval-friendly mapping."""

    if snapshot is None:
        return {
            "exists": False,
            "humor_intensity": None,
            "verbosity": None,
            "initiative": None,
            "relationship_topics": [],
            "place_focuses": [],
            "continuity_titles": [],
        }
    humor_profile = snapshot.humor_profile
    style_profile = snapshot.style_profile
    return {
        "exists": True,
        "humor_intensity": None if humor_profile is None else round(float(humor_profile.intensity), 4),
        "verbosity": None if style_profile is None else round(float(style_profile.verbosity), 4),
        "initiative": None if style_profile is None else round(float(style_profile.initiative), 4),
        "relationship_topics": sorted(signal.topic for signal in snapshot.relationship_signals),
        "place_focuses": sorted(
            str(getattr(focus, "name", getattr(focus, "place_name", "")))
            for focus in snapshot.place_focuses
        ),
        "continuity_titles": sorted(thread.title for thread in snapshot.continuity_threads),
    }


def _behavior_projection(
    snapshot: PersonalitySnapshot | None,
    *,
    world_state: WorldIntelligenceState,
    selected_topics: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Project prompt-facing behavior for the selected topics at one checkpoint."""

    mindshare = build_mindshare_items(
        snapshot,
        engagement_signals=world_state.interest_signals,
        max_items=max(6, len(selected_topics)),
    )
    policies = build_positive_engagement_policies(
        snapshot,
        engagement_signals=world_state.interest_signals,
        max_items=max(6, len(selected_topics)),
    )
    cues = build_turn_steering_cues(
        snapshot,
        engagement_signals=world_state.interest_signals,
        max_items=max(6, len(selected_topics)),
    )
    mindshare_by_topic = {_clean_topic_key(item.title): item for item in mindshare}
    policy_by_topic = {_clean_topic_key(item.title): item for item in policies}
    cue_by_topic = {_clean_topic_key(item.title): item for item in cues}
    projections: dict[str, dict[str, object]] = {}
    for topic in selected_topics:
        key = _clean_topic_key(topic)
        item = mindshare_by_topic.get(key)
        policy = policy_by_topic.get(key)
        cue = cue_by_topic.get(key)
        projections[topic] = {
            "in_mindshare": item is not None,
            "appetite_state": None if item is None else item.appetite.state,
            "appetite_interest": None if item is None else item.appetite.interest,
            "appetite_follow_up": None if item is None else item.appetite.follow_up,
            "positive_engagement_action": None if policy is None else policy.action,
            "attention_state": None if cue is None else cue.attention_state,
            "open_offer": None if cue is None else cue.open_offer,
            "user_pull": None if cue is None else cue.user_pull,
        }
    return projections


def _behavior_changes(
    *,
    initial: Mapping[str, Mapping[str, object]],
    final: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return the bounded set of behavior changes between two checkpoints."""

    changes: list[dict[str, object]] = []
    for topic, before in initial.items():
        after = final.get(topic, {})
        changed_fields: dict[str, dict[str, object]] = {}
        for field_name in (
            "in_mindshare",
            "appetite_state",
            "appetite_interest",
            "appetite_follow_up",
            "positive_engagement_action",
            "attention_state",
            "open_offer",
            "user_pull",
        ):
            before_value = before.get(field_name)
            after_value = after.get(field_name)
            if before_value == after_value:
                continue
            changed_fields[field_name] = {"before": before_value, "after": after_value}
        if changed_fields:
            changes.append({"topic": topic, "fields": changed_fields})
    return changes


def _reflection_metrics(service: LongTermMemoryService) -> dict[str, object]:
    """Summarize the current long-term reflection-bearing object state."""

    objects = tuple(service.object_store.load_objects())
    summaries = [item for item in objects if item.kind == "summary"]
    reflected_active = [item for item in objects if item.status == "active" and item.kind != "summary"]
    return {
        "object_count": len(objects),
        "summary_count": len(summaries),
        "active_object_count": len(reflected_active),
        "summary_titles": [item.summary for item in summaries[:4]],
    }


def _world_state_metrics(state: WorldIntelligenceState) -> dict[str, object]:
    """Project world-intelligence state into a compact eval-friendly mapping."""

    return {
        "interest_signal_count": len(state.interest_signals),
        "awareness_thread_count": len(state.awareness_threads),
        "last_refreshed_at": state.last_refreshed_at,
        "last_recalibrated_at": state.last_recalibrated_at,
    }


def _build_evaluation_summary(
    *,
    initial_snapshot_metrics: Mapping[str, object],
    final_snapshot_metrics: Mapping[str, object],
    initial_memory_state: Mapping[str, object],
    final_memory_state: Mapping[str, object],
    initial_world_state: WorldIntelligenceState,
    final_world_state: WorldIntelligenceState,
    initial_behavior: Mapping[str, Mapping[str, object]],
    final_behavior: Mapping[str, Mapping[str, object]],
    total_reflected_objects: int,
    total_created_summaries: int,
    raw_style_signal_counts: Mapping[str, int],
) -> dict[str, object]:
    """Derive one explicit eval verdict from the initial/final checkpoints.

    The summary keeps the critical proof questions separate instead of hiding
    them behind one boolean:

    - Did raw long-term reflection produce any activity at all?
    - Did reflection become semantically richer by producing summaries?
    - Did the promptable personality snapshot actually drift?
    - Did that drift become visible in downstream mindshare/engagement policy?
    """

    humor_shift = (
        None
        if initial_snapshot_metrics["humor_intensity"] is None or final_snapshot_metrics["humor_intensity"] is None
        else round(
            float(final_snapshot_metrics["humor_intensity"])
            - float(initial_snapshot_metrics["humor_intensity"]),
            4,
        )
    )
    verbosity_shift = (
        None
        if initial_snapshot_metrics["verbosity"] is None or final_snapshot_metrics["verbosity"] is None
        else round(
            float(final_snapshot_metrics["verbosity"])
            - float(initial_snapshot_metrics["verbosity"]),
            4,
        )
    )
    initiative_shift = (
        None
        if initial_snapshot_metrics["initiative"] is None or final_snapshot_metrics["initiative"] is None
        else round(
            float(final_snapshot_metrics["initiative"])
            - float(initial_snapshot_metrics["initiative"]),
            4,
        )
    )
    behavior_changes = _behavior_changes(initial=initial_behavior, final=final_behavior)
    relationship_topics_changed = (
        initial_snapshot_metrics["relationship_topics"]
        != final_snapshot_metrics["relationship_topics"]
    )
    continuity_titles_changed = (
        initial_snapshot_metrics["continuity_titles"]
        != final_snapshot_metrics["continuity_titles"]
    )
    inline_summary_growth = max(
        0,
        int(final_memory_state.get("summary_count", 0))
        - int(initial_memory_state.get("summary_count", 0)),
    )
    reflection_activity_nonzero = bool(total_reflected_objects > 0 or inline_summary_growth > 0)
    reflection_semantic_richness_nonzero = bool(
        total_created_summaries > 0 or inline_summary_growth > 0
    )
    humor_evolution_nonzero = humor_shift not in (None, 0.0)
    style_evolution_nonzero = any(
        value not in (None, 0.0) for value in (verbosity_shift, initiative_shift)
    )
    personality_drift_nonzero = bool(
        humor_evolution_nonzero
        or style_evolution_nonzero
        or relationship_topics_changed
        or continuity_titles_changed
    )
    world_intelligence_growth = len(final_world_state.interest_signals) > len(initial_world_state.interest_signals)
    downstream_behavior_changed = len(behavior_changes) > 0
    behavior_change_topics = [str(item["topic"]) for item in behavior_changes]
    raw_verbosity_signal_count = int(raw_style_signal_counts.get("verbosity_preference", 0))
    raw_initiative_signal_count = int(raw_style_signal_counts.get("initiative_preference", 0))
    raw_humor_signal_count = int(raw_style_signal_counts.get("humor_feedback", 0))
    verbosity_forming_proven = bool(raw_verbosity_signal_count > 0 and (verbosity_shift or 0.0) < 0.0)
    initiative_forming_proven = bool(raw_initiative_signal_count > 0 and (initiative_shift or 0.0) > 0.0)
    humor_forming_proven = bool(raw_humor_signal_count > 0 and (humor_shift or 0.0) > 0.0)
    style_axes_forming_proven = bool(
        verbosity_forming_proven and initiative_forming_proven and humor_forming_proven
    )
    personality_forming_proven = bool(
        personality_drift_nonzero and downstream_behavior_changed
    )
    return {
        "reflection_activity_nonzero": reflection_activity_nonzero,
        "reflection_semantic_richness_nonzero": reflection_semantic_richness_nonzero,
        "inline_summary_growth": inline_summary_growth,
        "total_reflected_objects": total_reflected_objects,
        "total_created_summaries": total_created_summaries,
        "personality_drift_nonzero": personality_drift_nonzero,
        "personality_forming_proven": personality_forming_proven,
        "humor_evolution_nonzero": humor_evolution_nonzero,
        "humor_shift": humor_shift,
        "style_evolution_nonzero": style_evolution_nonzero,
        "verbosity_shift": verbosity_shift,
        "initiative_shift": initiative_shift,
        "relationship_topics_changed": relationship_topics_changed,
        "continuity_titles_changed": continuity_titles_changed,
        "world_intelligence_growth": world_intelligence_growth,
        "downstream_behavior_changed": downstream_behavior_changed,
        "behavior_change_count": len(behavior_changes),
        "behavior_change_topics": behavior_change_topics,
        "behavior_changes": behavior_changes,
        "raw_style_signal_counts": {
            "verbosity_preference": raw_verbosity_signal_count,
            "initiative_preference": raw_initiative_signal_count,
            "humor_feedback": raw_humor_signal_count,
        },
        "verbosity_forming_proven": verbosity_forming_proven,
        "initiative_forming_proven": initiative_forming_proven,
        "humor_forming_proven": humor_forming_proven,
        "style_axes_forming_proven": style_axes_forming_proven,
        "snapshot_exists_after_run": bool(final_snapshot_metrics["exists"]),
    }


def run_eval(
    *,
    env_file: Path,
    run_id: str | None = None,
    clone_live_baseline: bool = True,
    emit_progress: bool = False,
) -> dict[str, object]:
    """Run the bounded personality-formation eval and return one JSON-safe report."""

    base_config = TwinrConfig.from_env(env_file)
    effective_run_id = run_id or _utcnow().strftime("%Y%m%dT%H%M%SZ")
    namespace = f"twinr_personality_eval_{effective_run_id.lower()}"
    selected_topics = (
        "Janina",
        "AI companions",
        "world politics",
        "local politics",
        "celebrity gossip",
    )
    day_reports: list[dict[str, object]] = []
    total_reflected_objects = 0
    total_created_summaries = 0
    raw_style_signal_counts: Counter[str] = Counter()
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="twinr_personality_eval_") as temp_dir:
        eval_config = _with_isolated_namespace(
            base_config,
            namespace=namespace,
            temp_root=Path(temp_dir),
        )
        _emit_progress(
            enabled=emit_progress,
            message=f"starting run_id={effective_run_id} namespace={namespace}",
        )
        baseline_copy: dict[str, object] = {
            "copied_snapshot": False,
            "copied_subscription_count": 0,
            "copied_interest_signal_count": 0,
        }
        if clone_live_baseline:
            _emit_progress(enabled=emit_progress, message="cloning live baseline")
            baseline_copy = _clone_live_baseline(
                base_config=base_config,
                eval_config=eval_config,
            )

        remote_state = LongTermRemoteStateStore.from_config(eval_config)
        service = LongTermMemoryService.from_config(eval_config)
        if service.writer is not None:
            service.writer.shutdown(timeout_s=1.0)
            service.writer = None
        if service.multimodal_writer is not None:
            service.multimodal_writer.shutdown(timeout_s=1.0)
            service.multimodal_writer = None

        snapshot_store = RemoteStatePersonalitySnapshotStore()
        world_store = RemoteStateWorldIntelligenceStore()

        try:
            _emit_progress(enabled=emit_progress, message="loading initial snapshot and world state")
            initial_snapshot = snapshot_store.load_snapshot(
                config=eval_config,
                remote_state=remote_state,
            )
            initial_world_state = world_store.load_state(
                config=eval_config,
                remote_state=remote_state,
            )
            initial_memory_state = _reflection_metrics(service)
            initial_behavior = _behavior_projection(
                initial_snapshot,
                world_state=initial_world_state,
                selected_topics=selected_topics,
            )
            days = _default_eval_days(now=_utcnow())
            for day in days:
                day_started = time.monotonic()
                _emit_progress(enabled=emit_progress, message=f"{day.name}: starting")
                raw_turn_names: list[str] = []
                structured_names: list[str] = []
                tool_names: list[str] = []
                day_reflected_objects = 0
                day_created_summaries = 0
                day_midterm_packets = 0
                day_interaction_signal_counts: Counter[str] = Counter()
                day_style_signal_counts: Counter[str] = Counter()
                for turn in day.raw_turns:
                    _emit_progress(
                        enabled=emit_progress,
                        message=f"{day.name}: raw_turn {turn.name}",
                    )
                    preview = _persist_raw_turn_with_preview(service, turn)
                    day_reflected_objects += preview["reflected_objects"]
                    day_created_summaries += preview["created_summaries"]
                    day_midterm_packets += preview["midterm_packets"]
                    day_interaction_signal_counts.update(preview["interaction_signal_counts"])
                    day_style_signal_counts.update(preview["style_signal_counts"])
                    raw_turn_names.append(turn.name)
                for structured in day.structured_learning:
                    _emit_progress(
                        enabled=emit_progress,
                        message=f"{day.name}: structured {structured.name}",
                    )
                    structured_names.append(structured.name)
                _persist_structured_learning_batch(service, day.structured_learning)
                if day.tool_history:
                    _emit_progress(
                        enabled=emit_progress,
                        message=f"{day.name}: tool_history batch size={len(day.tool_history)}",
                    )
                _record_tool_history(service, day.tool_history)
                tool_names.extend(item.name for item in day.tool_history)

                _emit_progress(enabled=emit_progress, message=f"{day.name}: loading post-day snapshot")
                day_snapshot = snapshot_store.load_snapshot(
                    config=eval_config,
                    remote_state=remote_state,
                )
                day_world_state = world_store.load_state(
                    config=eval_config,
                    remote_state=remote_state,
                )
                day_memory_state = _reflection_metrics(service)
                total_reflected_objects += day_reflected_objects
                total_created_summaries += day_created_summaries
                raw_style_signal_counts.update(day_style_signal_counts)
                day_reports.append(
                    {
                        "day": day.name,
                        "raw_turns": raw_turn_names,
                        "structured_learning": structured_names,
                        "tool_history": tool_names,
                        "reflection": {
                            "reflected_objects": day_reflected_objects,
                            "created_summaries": day_created_summaries,
                            "midterm_packets": day_midterm_packets,
                            "mode": "inline_turn_reflection",
                        },
                        "raw_learning": {
                            "interaction_signal_counts": {
                                key: int(value)
                                for key, value in sorted(day_interaction_signal_counts.items())
                            },
                            "style_signal_counts": {
                                key: int(value)
                                for key, value in sorted(day_style_signal_counts.items())
                            },
                        },
                        "snapshot": _snapshot_metrics(day_snapshot),
                        "world_state": _world_state_metrics(day_world_state),
                        "behavior": _behavior_projection(
                            day_snapshot,
                            world_state=day_world_state,
                            selected_topics=selected_topics,
                        ),
                        "memory_state": day_memory_state,
                        "elapsed_ms": round(max(0.0, (time.monotonic() - day_started) * 1000.0), 3),
                    }
                )
                _emit_progress(
                    enabled=emit_progress,
                    message=(
                        f"{day.name}: done inline_summaries={day_memory_state['summary_count']} "
                        f"elapsed_ms={day_reports[-1]['elapsed_ms']}"
                    ),
                )

            _emit_progress(enabled=emit_progress, message="loading final snapshot and world state")
            final_snapshot = snapshot_store.load_snapshot(
                config=eval_config,
                remote_state=remote_state,
            )
            final_world_state = world_store.load_state(
                config=eval_config,
                remote_state=remote_state,
            )
            final_memory_state = _reflection_metrics(service)
            final_behavior = _behavior_projection(
                final_snapshot,
                world_state=final_world_state,
                selected_topics=selected_topics,
            )
        finally:
            service.shutdown(timeout_s=1.0)

    initial_metrics = _snapshot_metrics(initial_snapshot)
    final_metrics = _snapshot_metrics(final_snapshot)
    evaluation = _build_evaluation_summary(
        initial_snapshot_metrics=initial_metrics,
        final_snapshot_metrics=final_metrics,
        initial_memory_state=initial_memory_state,
        final_memory_state=final_memory_state,
        initial_world_state=initial_world_state,
        final_world_state=final_world_state,
        initial_behavior=initial_behavior,
        final_behavior=final_behavior,
        total_reflected_objects=total_reflected_objects,
        total_created_summaries=total_created_summaries,
        raw_style_signal_counts={
            key: int(value)
            for key, value in sorted(raw_style_signal_counts.items())
        },
    )
    report = {
        "recorded_at": _utcnow_iso(),
        "env_file": str(env_file.resolve()),
        "run_id": effective_run_id,
        "namespace": namespace,
        "elapsed_ms": round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
        "baseline_copy": baseline_copy,
        "days": day_reports,
        "initial": {
            "snapshot": initial_metrics,
            "world_state": _world_state_metrics(initial_world_state),
            "behavior": initial_behavior,
            "memory_state": initial_memory_state,
        },
        "final": {
            "snapshot": final_metrics,
            "world_state": _world_state_metrics(final_world_state),
            "behavior": final_behavior,
            "memory_state": final_memory_state,
        },
        "evaluation": evaluation,
        "raw_learning": {
            "style_signal_counts": {
                key: int(value)
                for key, value in sorted(raw_style_signal_counts.items())
            },
        },
    }
    report["ok"] = bool(
        evaluation["reflection_semantic_richness_nonzero"]
        and evaluation["personality_forming_proven"]
        and evaluation["style_axes_forming_proven"]
    )
    return report


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the bounded eval script."""

    parser = argparse.ArgumentParser(
        description="Run a bounded multi-call personality formation eval."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Twinr env file used to build the runtime config.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional suffix for the isolated remote namespace.",
    )
    parser.add_argument(
        "--no-clone-live-baseline",
        action="store_true",
        help="Start from an empty isolated eval namespace instead of cloning the live baseline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the eval report.",
    )
    parser.add_argument(
        "--emit-progress",
        action="store_true",
        help="Emit bounded progress lines to stderr while the eval runs.",
    )
    return parser


def main() -> int:
    """Run the CLI entrypoint and emit one JSON report."""

    parser = _build_argument_parser()
    args = parser.parse_args()
    report = run_eval(
        env_file=args.env_file,
        run_id=args.run_id,
        clone_live_baseline=not args.no_clone_live_baseline,
        emit_progress=args.emit_progress,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
