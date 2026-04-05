"""Sensor observation and live-context helpers for background delivery."""

# CHANGELOG: 2026-03-27
# BUG-1: Normalize sensor.observed_at to a monotonic event time instead of treating every numeric value as monotonic.
# BUG-2: Use the current incoming observation's effective time, not the merged snapshot's stale timestamp, for context tracking and person-state derivation.
# BUG-3: Stop sharing mutable snapshot objects between _latest_sensor_observation_facts and the automation queue.
# BUG-4: Replace one-slot drop-oldest overflow handling with latest-wins queue coalescing to prevent stale automation processing under bursty input.
# BUG-5: Normalize single-string event_names as one event instead of exploding them into per-character tuples.
# BUG-6: Expire live facts from receipt/apply time fallback when sensor.observed_at is missing, malformed, or poisoned.
# SEC-1: Add depth/item/node caps and cycle detection for sensor payload cloning/merging to prevent practical DoS on Raspberry Pi-class deployments.
# SEC-2: Clamp future clock skew and reject malformed/implausible timestamps so sensor input cannot poison freshness and live context indefinitely.
# IMP-1: Store per-snapshot temporal provenance (applied_at / source_observed_at) for auditable freshness decisions.
# IMP-2: Use copy-on-write internal merges and clone only when crossing thread boundaries, cutting memory churn on Pi 4.
# IMP-3: Degrade gracefully when context tracking or person-state derivation fails, instead of killing the background observation path.


from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Empty, Full
import math
import time
from typing import Any

from twinr.proactive.runtime.person_state import derive_person_state


@dataclass(frozen=True)
class _ObservationTiming:
    effective_monotonic: float
    received_monotonic: float
    received_epoch: float
    source_epoch: float | None


class BackgroundObservationPayloadError(ValueError):
    """Raised when one sensor payload exceeds local safety/resource limits."""


class BackgroundObservationMixin:
    """Own sensor-state merging, freshness, and long-term live snapshots."""

    _BACKGROUND_STALE_AFTER_S = 45.0
    _BACKGROUND_MAX_FUTURE_SKEW_S = 300.0
    _BACKGROUND_MIN_EPOCH_S = 946684800.0  # 2000-01-01T00:00:00Z
    _BACKGROUND_MAX_CLONE_DEPTH = 16
    _BACKGROUND_MAX_CONTAINER_ITEMS = 256
    _BACKGROUND_MAX_TOTAL_NODES = 4096
    _BACKGROUND_MAX_SNAPSHOT_TOTAL_NODES = 16384
    _BACKGROUND_MAX_EVENT_NAMES = 32
    _BACKGROUND_MAX_EVENT_NAME_LENGTH = 128
    # BREAKING: emitted fact snapshots now carry a reserved hidden metadata key for temporal provenance.
    _BACKGROUND_META_KEY = "_twinr_observation_meta"

    def _background_float_setting(self, name: str, default: float) -> float:
        value = getattr(self, name, default)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(number):
            return default
        return number

    def _background_int_setting(self, name: str, default: int) -> int:
        value = getattr(self, name, default)
        try:
            number = int(value)
        except (TypeError, ValueError):
            return default
        return max(1, number)

    def _background_max_future_skew_seconds(self) -> float:
        return max(
            0.0,
            self._background_float_setting(
                "_background_observation_max_future_skew_s",
                self._BACKGROUND_MAX_FUTURE_SKEW_S,
            ),
        )

    def _background_stale_after_seconds(self) -> float:
        return max(
            0.0,
            self._background_float_setting(
                "_background_observation_stale_after_s",
                self._BACKGROUND_STALE_AFTER_S,
            ),
        )

    def _background_meta_key(self) -> str:
        return str(
            getattr(self, "_background_observation_meta_key", self._BACKGROUND_META_KEY)
        )

    def _background_now(self) -> tuple[float, float]:
        return time.monotonic(), datetime.now(timezone.utc).timestamp()

    def _new_background_clone_state(self, *, kind: str = "payload") -> dict[str, Any]:
        node_attr = "_background_observation_max_total_nodes"
        node_default = self._BACKGROUND_MAX_TOTAL_NODES
        if kind == "snapshot":
            node_attr = "_background_snapshot_max_total_nodes"
            node_default = self._BACKGROUND_MAX_SNAPSHOT_TOTAL_NODES
        return {
            "depth_limit": self._background_int_setting(
                "_background_observation_max_depth",
                self._BACKGROUND_MAX_CLONE_DEPTH,
            ),
            "item_limit": self._background_int_setting(
                "_background_observation_max_container_items",
                self._BACKGROUND_MAX_CONTAINER_ITEMS,
            ),
            "node_limit": self._background_int_setting(node_attr, node_default),
            "nodes": 0,
            "active_ids": set(),
        }

    def _clone_background_value(
        self,
        value: object,
        *,
        _depth: int = 0,
        _state: dict[str, Any] | None = None,
    ) -> object:
        if _state is None:
            _state = self._new_background_clone_state()
        if _depth > _state["depth_limit"]:
            raise BackgroundObservationPayloadError(
                "Sensor payload nesting exceeded the configured limit."
            )

        def _tick() -> None:
            _state["nodes"] += 1
            if _state["nodes"] > _state["node_limit"]:
                raise BackgroundObservationPayloadError(
                    "Sensor payload exceeded the configured node budget."
                )

        if isinstance(value, dict):
            container_id = id(value)
            if container_id in _state["active_ids"]:
                raise BackgroundObservationPayloadError(
                    "Sensor payload contains a recursive container."
                )
            items = tuple(value.items())
            if len(items) > _state["item_limit"]:
                raise BackgroundObservationPayloadError(
                    "Sensor mapping exceeded the configured item limit."
                )
            _state["active_ids"].add(container_id)
            try:
                cloned: dict[object, object] = {}
                for key, item in items:
                    _tick()
                    cloned[key] = self._clone_background_value(
                        item,
                        _depth=_depth + 1,
                        _state=_state,
                    )
                return cloned
            finally:
                _state["active_ids"].remove(container_id)

        if isinstance(value, list):
            container_id = id(value)
            if container_id in _state["active_ids"]:
                raise BackgroundObservationPayloadError(
                    "Sensor payload contains a recursive container."
                )
            if len(value) > _state["item_limit"]:
                raise BackgroundObservationPayloadError(
                    "Sensor list exceeded the configured item limit."
                )
            _state["active_ids"].add(container_id)
            try:
                cloned_list: list[object] = []
                for item in value:
                    _tick()
                    cloned_list.append(
                        self._clone_background_value(
                            item,
                            _depth=_depth + 1,
                            _state=_state,
                        )
                    )
                return cloned_list
            finally:
                _state["active_ids"].remove(container_id)

        if isinstance(value, tuple):
            container_id = id(value)
            if container_id in _state["active_ids"]:
                raise BackgroundObservationPayloadError(
                    "Sensor payload contains a recursive container."
                )
            if len(value) > _state["item_limit"]:
                raise BackgroundObservationPayloadError(
                    "Sensor tuple exceeded the configured item limit."
                )
            _state["active_ids"].add(container_id)
            try:
                cloned_tuple: list[object] = []
                for item in value:
                    _tick()
                    cloned_tuple.append(
                        self._clone_background_value(
                            item,
                            _depth=_depth + 1,
                            _state=_state,
                        )
                    )
                return tuple(cloned_tuple)
            finally:
                _state["active_ids"].remove(container_id)

        if isinstance(value, set):
            container_id = id(value)
            if container_id in _state["active_ids"]:
                raise BackgroundObservationPayloadError(
                    "Sensor payload contains a recursive container."
                )
            if len(value) > _state["item_limit"]:
                raise BackgroundObservationPayloadError(
                    "Sensor set exceeded the configured item limit."
                )
            _state["active_ids"].add(container_id)
            try:
                cloned_set: set[object] = set()
                for item in value:
                    _tick()
                    cloned_set.add(
                        self._clone_background_value(
                            item,
                            _depth=_depth + 1,
                            _state=_state,
                        )
                    )
                return cloned_set
            finally:
                _state["active_ids"].remove(container_id)

        _tick()
        return value

    def _merge_background_mapping_values(
        self,
        base: object,
        incoming: object,
        *,
        _depth: int = 0,
        _state: dict[str, Any] | None = None,
    ) -> object:
        """Merge nested observation mappings while replacing scalar/list values."""

        if _state is None:
            _state = self._new_background_clone_state()
        if _depth > _state["depth_limit"]:
            raise BackgroundObservationPayloadError(
                "Sensor payload nesting exceeded the configured limit."
            )

        if isinstance(base, dict) and isinstance(incoming, dict):
            items = tuple(incoming.items())
            if len(items) > _state["item_limit"]:
                raise BackgroundObservationPayloadError(
                    "Sensor mapping exceeded the configured item limit."
                )
            merged = dict(base)
            for key, value in items:
                _state["nodes"] += 1
                if _state["nodes"] > _state["node_limit"]:
                    raise BackgroundObservationPayloadError(
                        "Sensor payload exceeded the configured node budget."
                    )
                if key in base:
                    merged[key] = self._merge_background_mapping_values(
                        base[key],
                        value,
                        _depth=_depth + 1,
                        _state=_state,
                    )
                else:
                    merged[key] = self._clone_background_value(
                        value,
                        _depth=_depth + 1,
                        _state=_state,
                    )
            return merged
        return self._clone_background_value(incoming, _depth=_depth, _state=_state)

    def _merge_sensor_observation_facts(
        self,
        *,
        existing: dict[str, object] | None,
        incoming: dict[str, object],
    ) -> dict[str, object]:
        """Merge one partial sensor observation into the latest live fact map."""

        state = self._new_background_clone_state(kind="payload")
        if not isinstance(existing, dict) or not existing:
            cloned = self._clone_background_value(incoming, _state=state)
            return cloned if isinstance(cloned, dict) else dict(incoming)
        merged = self._merge_background_mapping_values(existing, incoming, _state=state)
        return merged if isinstance(merged, dict) else dict(incoming)

    def _extract_sensor_observed_at(self, facts: dict[str, object]) -> object:
        sensor = facts.get("sensor")
        if isinstance(sensor, dict):
            return sensor.get("observed_at")
        return None

    def _coerce_numeric_timestamp(self, value: object) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            numeric = float(value)
        else:
            text = self._coerce_text(value)
            if not text:
                return None
            try:
                numeric = float(text.strip())
            except (TypeError, ValueError):
                return None
        if not math.isfinite(numeric):
            return None
        return numeric

    def _coerce_epoch_seconds_from_numeric(
        self,
        numeric: float,
        *,
        received_epoch: float | None = None,
    ) -> float | None:
        if not math.isfinite(numeric):
            return None
        if received_epoch is None:
            received_epoch = datetime.now(timezone.utc).timestamp()
        future_skew = self._background_max_future_skew_seconds()
        if self._BACKGROUND_MIN_EPOCH_S <= numeric <= received_epoch + future_skew:
            return numeric
        for scale in (1_000.0, 1_000_000.0, 1_000_000_000.0):
            scaled = numeric / scale
            if self._BACKGROUND_MIN_EPOCH_S <= scaled <= received_epoch + future_skew:
                return scaled
        return None

    def _parse_event_timestamp(self, value: object) -> float | None:
        numeric = self._coerce_numeric_timestamp(value)
        epoch_now = datetime.now(timezone.utc).timestamp()
        if numeric is not None:
            epoch_seconds = self._coerce_epoch_seconds_from_numeric(
                numeric,
                received_epoch=epoch_now,
            )
            if epoch_seconds is not None:
                return epoch_seconds
            return None

        if isinstance(value, datetime):
            when = value
        else:
            text = self._coerce_text(value)
            if not text:
                return None
            try:
                when = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return when.astimezone(timezone.utc).timestamp()

    def _resolve_observation_timing(self, facts: dict[str, object]) -> _ObservationTiming:
        received_monotonic, received_epoch = self._background_now()
        observed_at = self._extract_sensor_observed_at(facts)
        future_skew = self._background_max_future_skew_seconds()
        numeric = self._coerce_numeric_timestamp(observed_at)

        if numeric is not None:
            if 0.0 <= numeric <= received_monotonic + future_skew:
                applied_monotonic = min(
                    max(0.0, numeric),
                    received_monotonic + future_skew,
                )
                source_epoch = received_epoch - max(
                    0.0,
                    received_monotonic - applied_monotonic,
                )
                if applied_monotonic > received_monotonic:
                    source_epoch = received_epoch + (
                        applied_monotonic - received_monotonic
                    )
                return _ObservationTiming(
                    effective_monotonic=applied_monotonic,
                    received_monotonic=received_monotonic,
                    received_epoch=received_epoch,
                    source_epoch=source_epoch,
                )

            epoch_seconds = self._coerce_epoch_seconds_from_numeric(
                numeric,
                received_epoch=received_epoch,
            )
            if epoch_seconds is not None:
                clamped_epoch = min(
                    max(self._BACKGROUND_MIN_EPOCH_S, epoch_seconds),
                    received_epoch + future_skew,
                )
                if clamped_epoch >= received_epoch:
                    applied_monotonic = min(
                        received_monotonic + (clamped_epoch - received_epoch),
                        received_monotonic + future_skew,
                    )
                else:
                    applied_monotonic = max(
                        0.0,
                        received_monotonic - (received_epoch - clamped_epoch),
                    )
                return _ObservationTiming(
                    effective_monotonic=applied_monotonic,
                    received_monotonic=received_monotonic,
                    received_epoch=received_epoch,
                    source_epoch=clamped_epoch,
                )

        parsed_epoch = None
        if observed_at is not None:
            parsed_epoch = self._parse_event_timestamp(observed_at)
        if parsed_epoch is not None:
            clamped_epoch = min(
                max(self._BACKGROUND_MIN_EPOCH_S, parsed_epoch),
                received_epoch + future_skew,
            )
            if clamped_epoch >= received_epoch:
                applied_monotonic = min(
                    received_monotonic + (clamped_epoch - received_epoch),
                    received_monotonic + future_skew,
                )
            else:
                applied_monotonic = max(
                    0.0,
                    received_monotonic - (received_epoch - clamped_epoch),
                )
            return _ObservationTiming(
                effective_monotonic=applied_monotonic,
                received_monotonic=received_monotonic,
                received_epoch=received_epoch,
                source_epoch=clamped_epoch,
            )

        return _ObservationTiming(
            effective_monotonic=received_monotonic,
            received_monotonic=received_monotonic,
            received_epoch=received_epoch,
            source_epoch=None,
        )

    def _sensor_observation_monotonic_now(
        self,
        *,
        facts: dict[str, object],
    ) -> float:
        """Return a safe monotonic timestamp for one observation."""

        return self._resolve_observation_timing(facts).effective_monotonic

    def _sanitize_event_name(self, value: object) -> str:
        text = self._coerce_text(value)
        if not text:
            return ""
        cleaned = "".join(
            ch if ch.isprintable() and ch not in "\r\n\t\x0b\x0c" else " "
            for ch in text
        )
        normalized = " ".join(cleaned.split())
        if not normalized:
            return ""
        max_length = self._background_int_setting(
            "_background_observation_max_event_name_length",
            self._BACKGROUND_MAX_EVENT_NAME_LENGTH,
        )
        return normalized[:max_length]

    def _normalize_event_names(self, event_names: object) -> tuple[str, ...]:
        max_events = self._background_int_setting(
            "_background_observation_max_event_names",
            self._BACKGROUND_MAX_EVENT_NAMES,
        )
        if max_events <= 0:
            return ()

        if isinstance(event_names, (str, bytes)):
            iterator = iter((event_names,))
        else:
            try:
                iterator = iter(event_names or ())
            except TypeError:
                return ()

        normalized: list[str] = []
        seen: set[str] = set()
        for item in iterator:
            text = self._sanitize_event_name(item)
            if not text or text in seen:
                continue
            normalized.append(text)
            seen.add(text)
            if len(normalized) >= max_events:
                break
        return tuple(normalized)

    def _observation_age_seconds(self, observed_at: object) -> float | None:
        if isinstance(observed_at, datetime):
            timestamp = self._parse_event_timestamp(observed_at)
            if timestamp is None:
                return None
            return max(0.0, datetime.now(timezone.utc).timestamp() - timestamp)

        if isinstance(observed_at, str):
            stripped = observed_at.strip()
            if not stripped:
                return None
            try:
                return self._observation_age_seconds(
                    datetime.fromisoformat(stripped.replace("Z", "+00:00"))
                )
            except ValueError:
                observed_at = stripped

        value = self._coerce_numeric_timestamp(observed_at)
        if value is None:
            return None

        monotonic_now, epoch_now = self._background_now()
        future_skew = self._background_max_future_skew_seconds()
        if 0.0 <= value <= monotonic_now + future_skew:
            return max(0.0, monotonic_now - min(value, monotonic_now + future_skew))

        epoch_seconds = self._coerce_epoch_seconds_from_numeric(
            value,
            received_epoch=epoch_now,
        )
        if epoch_seconds is not None:
            return max(0.0, epoch_now - min(epoch_seconds, epoch_now + future_skew))

        return None

    def _observation_meta_dict(self, timing: _ObservationTiming) -> dict[str, object]:
        metadata: dict[str, object] = {
            "received_at": datetime.fromtimestamp(
                timing.received_epoch,
                tz=timezone.utc,
            ).isoformat().replace("+00:00", "Z"),
            "received_at_monotonic": timing.received_monotonic,
            "effective_observed_at_monotonic": timing.effective_monotonic,
        }
        if timing.source_epoch is not None:
            metadata["source_observed_at"] = datetime.fromtimestamp(
                timing.source_epoch,
                tz=timezone.utc,
            ).isoformat().replace("+00:00", "Z")
        return metadata

    def _attach_observation_meta(
        self,
        facts: dict[str, object],
        timing: _ObservationTiming,
    ) -> dict[str, object]:
        facts[self._background_meta_key()] = self._observation_meta_dict(timing)
        return facts

    def _record_latest_observation_timing(self, timing: _ObservationTiming) -> None:
        self._latest_sensor_observation_effective_monotonic = (
            timing.effective_monotonic
        )
        self._latest_sensor_observation_received_monotonic = (
            timing.received_monotonic
        )
        self._latest_sensor_observation_received_epoch = timing.received_epoch
        self._latest_sensor_observation_source_epoch = timing.source_epoch

    def _reject_sensor_payload(
        self,
        *,
        reason: str,
        exc: Exception,
        queue_for_automation: bool,
    ) -> None:
        fault_name = "sensor_observation_payload"
        try:
            self._remember_background_fault(fault_name, exc)
        except Exception:
            pass
        try:
            self._safe_emit(f"{fault_name}_rejected={type(exc).__name__}")
        except Exception:
            pass
        try:
            self._safe_record_event(
                f"{fault_name}_rejected",
                reason,
                level="warning",
                event_names=[
                    "sensor_payload_rejected"
                ] if queue_for_automation else [
                    "live_context_payload_rejected"
                ],
            )
        except Exception:
            pass

    def _prepare_sensor_observation_facts(self, facts: dict[str, object]) -> dict[str, object]:
        if not isinstance(facts, dict):
            raise BackgroundObservationPayloadError("Sensor payload must be a mapping.")
        cloned = self._clone_background_value(
            facts,
            _state=self._new_background_clone_state(kind="payload"),
        )
        copied = cloned if isinstance(cloned, dict) else dict(facts)
        copied.pop(self._background_meta_key(), None)
        return copied

    def handle_sensor_observation(self, facts: dict[str, object], event_names: tuple[str, ...]) -> None:
        try:
            copied_facts = self._prepare_sensor_observation_facts(facts)
        except Exception as exc:
            self._reject_sensor_payload(
                reason="Twinr rejected a sensor observation payload because it exceeded local safety or validation limits.",
                exc=exc,
                queue_for_automation=True,
            )
            return
        normalized_event_names = self._normalize_event_names(event_names)
        self._apply_sensor_observation_facts(
            copied_facts,
            normalized_event_names=normalized_event_names,
            queue_for_automation=True,
        )

    def handle_live_sensor_context(self, facts: dict[str, object]) -> None:
        """Refresh live multimodal state without queueing sensor automations."""

        try:
            copied_facts = self._prepare_sensor_observation_facts(facts)
        except Exception as exc:
            self._reject_sensor_payload(
                reason="Twinr rejected a live sensor context payload because it exceeded local safety or validation limits.",
                exc=exc,
                queue_for_automation=False,
            )
            return
        self._apply_sensor_observation_facts(
            copied_facts,
            normalized_event_names=(),
            queue_for_automation=False,
        )

    # BREAKING: queue backpressure now coalesces to the latest snapshot instead of preserving stale intermediate snapshots.
    def _enqueue_sensor_observation_snapshot(
        self,
        facts: dict[str, object],
        event_names: tuple[str, ...],
    ) -> None:
        try:
            snapshot = self._clone_background_value(
                facts,
                _state=self._new_background_clone_state(kind="snapshot"),
            )
        except Exception as exc:
            self._remember_background_fault("sensor_observation_snapshot", exc)
            self._safe_emit(f"sensor_observation_snapshot_failed={type(exc).__name__}")
            return

        queue_payload = snapshot if isinstance(snapshot, dict) else dict(facts)
        queue_event_names = event_names

        try:
            self._sensor_observation_queue.put_nowait((queue_payload, queue_event_names))
            return
        except Full:
            pass

        dropped_event_names: list[str] = list(queue_event_names)
        while True:
            try:
                dropped_payload = self._sensor_observation_queue.get_nowait()
            except Empty:
                break
            if isinstance(dropped_payload, tuple) and len(dropped_payload) >= 2:
                dropped_event_names.extend(
                    self._normalize_event_names(dropped_payload[1])
                )

        queue_event_names = self._normalize_event_names(dropped_event_names)
        try:
            self._sensor_observation_queue.put_nowait((queue_payload, queue_event_names))
        except Full as exc:
            self._remember_background_fault("sensor_observation_queue", exc)
            self._safe_emit("sensor_observation_queue_overflow=true")
            self._safe_record_event(
                "sensor_observation_queue_overflow",
                "Twinr dropped sensor observations because the queue stayed full after coalescing.",
                level="warning",
                event_names=list(queue_event_names),
            )

    def _apply_sensor_observation_facts(
        self,
        facts: dict[str, object],
        *,
        normalized_event_names: tuple[str, ...],
        queue_for_automation: bool,
    ) -> None:
        """Merge one sensor snapshot into live state and optionally queue it."""

        timing = self._resolve_observation_timing(facts)
        with self._get_lock("_sensor_observation_state_lock"):
            try:
                merged_facts = self._merge_sensor_observation_facts(
                    existing=getattr(self, "_latest_sensor_observation_facts", None),
                    incoming=facts,
                )
            except Exception as exc:
                self._reject_sensor_payload(
                    reason="Twinr rejected a sensor observation during merge because it exceeded local safety or validation limits.",
                    exc=exc,
                    queue_for_automation=queue_for_automation,
                )
                return

            context_event_names: tuple[str, ...] = ()
            tracker = getattr(self, "_smart_home_context_tracker", None)
            if callable(tracker):
                try:
                    context_update = tracker().observe(  # pylint: disable=not-callable
                        observed_at=timing.effective_monotonic,
                        live_facts=merged_facts,
                        incoming_facts=facts,
                    )
                except Exception as exc:
                    self._remember_background_fault("smart_home_context_tracker", exc)
                    self._safe_emit(
                        f"smart_home_context_tracker_failed={type(exc).__name__}"
                    )
                else:
                    try:
                        snapshot = getattr(context_update, "snapshot", None)
                        if snapshot is not None:
                            updated_facts = snapshot.apply_to_facts(merged_facts)
                            if not isinstance(updated_facts, dict):
                                raise TypeError(
                                    "Smart-home context snapshot must return a dict of facts."
                                )
                            merged_facts = updated_facts
                    except Exception as exc:
                        self._remember_background_fault("smart_home_context_snapshot", exc)
                        self._safe_emit(
                            f"smart_home_context_snapshot_failed={type(exc).__name__}"
                        )
                    else:
                        context_event_names = self._normalize_event_names(
                            getattr(context_update, "event_names", ())
                        )

            try:
                merged_facts["person_state"] = derive_person_state(
                    observed_at=timing.effective_monotonic,
                    live_facts=merged_facts,
                ).to_automation_facts()
            except Exception as exc:
                self._remember_background_fault("person_state_derivation", exc)
                self._safe_emit(
                    f"person_state_derivation_failed={type(exc).__name__}"
                )

            normalized_event_names = self._normalize_event_names(
                (*normalized_event_names, *context_event_names)
            )
            merged_facts = self._attach_observation_meta(merged_facts, timing)
            self._latest_sensor_observation_facts = merged_facts
            self._record_latest_observation_timing(timing)
            if queue_for_automation:
                self._enqueue_sensor_observation_snapshot(
                    merged_facts,
                    normalized_event_names,
                )

        refresh_voice_context = getattr(
            self,
            "_refresh_voice_orchestrator_sensor_context",
            None,
        )
        if callable(refresh_voice_context):
            try:
                refresh_voice_context()  # pylint: disable=not-callable
            except Exception as exc:
                self._safe_emit(
                    f"voice_orchestrator_context_refresh_failed={type(exc).__name__}"
                )
        if not queue_for_automation:
            return

    def _current_longterm_live_facts(self) -> dict[str, object] | None:
        facts = getattr(self, "_latest_sensor_observation_facts", None)
        if not isinstance(facts, dict):
            return None

        meta_key = self._background_meta_key()
        meta = facts.get(meta_key)
        age_s: float | None = None
        if isinstance(meta, dict):
            age_s = self._observation_age_seconds(
                meta.get("effective_observed_at_monotonic")
            )
            if age_s is None:
                age_s = self._observation_age_seconds(meta.get("received_at_monotonic"))
            if age_s is None:
                age_s = self._observation_age_seconds(meta.get("received_at"))

        if age_s is None:
            age_s = self._observation_age_seconds(
                getattr(self, "_latest_sensor_observation_effective_monotonic", None)
            )
        if age_s is None:
            age_s = self._observation_age_seconds(
                getattr(self, "_latest_sensor_observation_received_monotonic", None)
            )
        if age_s is None:
            age_s = self._observation_age_seconds(
                getattr(self, "_latest_sensor_observation_received_epoch", None)
            )

        if age_s is not None and age_s > self._background_stale_after_seconds():
            return None

        try:
            cloned_facts = self._clone_background_value(
                facts,
                _state=self._new_background_clone_state(kind="snapshot"),
            )
        except Exception as exc:
            self._remember_background_fault("current_longterm_live_facts", exc)
            self._safe_emit(f"current_longterm_live_facts_failed={type(exc).__name__}")
            return None

        copied = cloned_facts if isinstance(cloned_facts, dict) else dict(facts)
        copied["last_response_available"] = bool(getattr(self.runtime, "last_response", None))
        copied["recent_print_completed"] = self._recent_print_completed()
        return copied

    def _recent_print_completed(self, *, within_s: float = 900.0) -> bool:
        cutoff = datetime.now(timezone.utc).timestamp() - max(0.0, within_s)
        for entry in reversed(tuple(self.runtime.ops_events.tail(limit=40) or ())):
            if not isinstance(entry, dict):
                continue
            if entry.get("event") not in {"print_finished", "print_completed"}:
                continue
            when = self._parse_event_timestamp(entry.get("created_at", ""))
            if when is None:
                continue
            return when >= cutoff
        return False
