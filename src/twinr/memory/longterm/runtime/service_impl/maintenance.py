# mypy: disable-error-code=attr-defined
"""Maintenance flows for the long-term runtime service."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime

from twinr.memory.longterm.ingestion.backfill import LongTermOpsBackfillRunResult
from twinr.memory.longterm.core.models import (
    LongTermReflectionResultV1,
    LongTermRetentionResultV1,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError

from ._typing import ServiceMixinBase
from .compat import _normalize_datetime, _sort_conflicts, _sort_objects_by_memory_id, logger


class LongTermMemoryServiceMaintenanceMixin(ServiceMixinBase):
    """Reflection, sensor-memory, backfill, and retention orchestration."""

    def run_reflection(
        self,
        *,
        search_backend: object | None = None,
    ) -> LongTermReflectionResultV1:
        """Run object reflection plus sensor-memory summaries over stored state."""

        try:
            with self._store_lock:
                result = self.reflector.reflect(objects=self.object_store.load_objects())
                if self._has_reflection_payload(result):
                    self.object_store.apply_reflection(result)
                    self.midterm_store.apply_reflection(result)
                sensor_memory_result = self.sensor_memory.compile(objects=self.object_store.load_objects())
                if self._has_reflection_payload(sensor_memory_result):
                    self.object_store.apply_reflection(sensor_memory_result)
                    self.midterm_store.apply_reflection(sensor_memory_result)
                    combined_result = LongTermReflectionResultV1(
                        reflected_objects=tuple((*result.reflected_objects, *sensor_memory_result.reflected_objects)),
                        created_summaries=tuple((*result.created_summaries, *sensor_memory_result.created_summaries)),
                        midterm_packets=tuple((*result.midterm_packets, *sensor_memory_result.midterm_packets)),
                    )
                else:
                    combined_result = result
                if self.personality_learning is not None:
                    self.personality_learning.maybe_refresh_world_intelligence(
                        search_backend=search_backend,
                    )
                return combined_result
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Long-term reflection failed.")
            return self._empty_reflection_result()

    def run_sensor_memory(self, *, now: datetime | None = None) -> LongTermReflectionResultV1:
        """Compile sensor-memory summaries from current long-term objects."""

        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:
                result = self.sensor_memory.compile(objects=self.object_store.load_objects(), now=normalized_now)
                if self._has_reflection_payload(result):
                    self.object_store.apply_reflection(result)
                    self.midterm_store.apply_reflection(result)
                return result
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Sensor-memory compilation failed.")
            return self._empty_reflection_result()

    def backfill_ops_multimodal_history(
        self,
        *,
        entries: Iterable[Mapping[str, object]] | None = None,
        now: datetime | None = None,
    ) -> LongTermOpsBackfillRunResult:
        """Replay ops-event history into multimodal long-term evidence."""

        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        scanned_events = 0
        generated_evidence = 0
        sensor_observations = 0
        button_interactions = 0
        print_completions = 0
        applied_evidence = 0
        skipped_existing = 0
        reflected_objects = 0
        created_summaries = 0
        reflection_error: str | None = None

        try:
            raw_entries = tuple(entries) if entries is not None else self._load_backfill_entries()
            build = self.ops_backfiller.build_evidence(raw_entries)
            scanned_events = build.scanned_events
            generated_evidence = build.generated_evidence
            sensor_observations = build.sensor_observations
            button_interactions = build.button_interactions
            print_completions = build.print_completions

            with self._store_lock:
                existing_objects = tuple(self.object_store.load_objects())
                existing_conflicts = tuple(self.object_store.load_conflicts())
                existing_archived = tuple(self.object_store.load_archived_objects())
                objects_by_id = {item.memory_id: item for item in existing_objects}
                conflicts_by_slot = {item.slot_key: item for item in existing_conflicts}
                seen_turn_ids = {
                    event_id
                    for item in objects_by_id.values()
                    for event_id in item.source.event_ids
                }

                for evidence in build.evidence:
                    try:
                        extraction = self.multimodal_extractor.extract_evidence(evidence)
                        if extraction.turn_id in seen_turn_ids:
                            skipped_existing += 1
                            continue
                        result = self.consolidator.consolidate(
                            extraction=extraction,
                            existing_objects=tuple(objects_by_id.values()),
                        )
                        for item in (*result.episodic_objects, *result.durable_objects, *result.deferred_objects):
                            objects_by_id[item.memory_id] = self.object_store._merge_object(
                                existing=objects_by_id.get(item.memory_id),
                                incoming=item,
                                increment_support=True,
                            )
                        for conflict in result.conflicts:
                            conflicts_by_slot[conflict.slot_key] = conflict
                        seen_turn_ids.add(extraction.turn_id)
                        applied_evidence += 1
                    except Exception as exc:
                        logger.exception("Failed to backfill one multimodal evidence item.")
                        if reflection_error is None:
                            reflection_error = f"{type(exc).__name__}: {exc}"

                current_objects = _sort_objects_by_memory_id(objects_by_id.values())
                current_conflicts = _sort_conflicts(conflicts_by_slot.values())

                reflection_result = self._empty_reflection_result()
                if applied_evidence:
                    try:
                        reflection_result = self.reflector.reflect(objects=current_objects)
                    except Exception as exc:
                        logger.exception("Long-term backfill reflection failed.")
                        if reflection_error is None:
                            reflection_error = f"{type(exc).__name__}: {exc}"
                    else:
                        current_objects = self._merge_reflection_objects(
                            object_store=self.object_store,
                            current_objects=current_objects,
                            reflection=reflection_result,
                        )
                        reflected_objects += len(reflection_result.reflected_objects)
                        created_summaries += len(reflection_result.created_summaries)

                sensor_result = self._empty_reflection_result()
                try:
                    sensor_result = self.sensor_memory.compile(objects=current_objects, now=normalized_now)
                except Exception as exc:
                    logger.exception("Long-term backfill sensor-memory compilation failed.")
                    if reflection_error is None:
                        reflection_error = f"{type(exc).__name__}: {exc}"
                else:
                    if self._has_reflection_payload(sensor_result):
                        current_objects = self._merge_reflection_objects(
                            object_store=self.object_store,
                            current_objects=current_objects,
                            reflection=sensor_result,
                        )
                    reflected_objects += len(sensor_result.reflected_objects)
                    created_summaries += len(sensor_result.created_summaries)

                retention = self._apply_retention_or_keep(
                    retention_policy=self.retention_policy,
                    objects=current_objects,
                )
                should_write_snapshot = (
                    applied_evidence > 0
                    or self._has_reflection_payload(reflection_result)
                    or self._has_reflection_payload(sensor_result)
                    or bool(retention.expired_objects or retention.pruned_memory_ids or retention.archived_objects)
                )
                if should_write_snapshot:
                    self.object_store.write_snapshot(
                        objects=_sort_objects_by_memory_id(retention.kept_objects),
                        conflicts=current_conflicts,
                        archived_objects=self._merge_archived_objects(
                            existing_archived=existing_archived,
                            archived_updates=retention.archived_objects,
                        ),
                    )
                if self._has_reflection_payload(reflection_result):
                    self.midterm_store.apply_reflection(reflection_result)
                if self._has_reflection_payload(sensor_result):
                    self.midterm_store.apply_reflection(sensor_result)
        except LongTermRemoteUnavailableError:
            raise
        except Exception as exc:
            logger.exception("Ops multimodal backfill failed.")
            reflection_error = f"{type(exc).__name__}: {exc}"

        return LongTermOpsBackfillRunResult(
            scanned_events=scanned_events,
            generated_evidence=generated_evidence,
            applied_evidence=applied_evidence,
            skipped_existing=skipped_existing,
            sensor_observations=sensor_observations,
            button_interactions=button_interactions,
            print_completions=print_completions,
            reflected_objects=reflected_objects,
            created_summaries=created_summaries,
            reflection_error=reflection_error,
        )

    def run_retention(self) -> LongTermRetentionResultV1:
        """Apply retention and archive policy to stored long-term objects."""

        try:
            with self._store_lock:
                result = self.retention_policy.apply(objects=self.object_store.load_objects())
                self.object_store.apply_retention(result)
                return result
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Long-term retention run failed.")
            return LongTermRetentionResultV1(
                kept_objects=(),
                expired_objects=(),
                pruned_memory_ids=(),
                archived_objects=(),
            )
