"""Map smart-home events into Twinr observations and run bounded workers."""

from __future__ import annotations

# CHANGELOG: 2026-03-30
# BUG-1: Fixed incremental stream processing. The worker now passes the last cursor
#        back into read_sensor_stream(...) and persists successful checkpoints, so it
#        no longer replays the same batch forever.
# BUG-2: Fixed adapter-loader crash behavior. Exceptions raised by adapter_loader()
#        used to kill the worker thread because they happened outside the protected
#        try/except block.
# BUG-3: Fixed sticky button state. smart_home["button_pressed"] is now an edge flag
#        for the current observation instead of staying True forever after one press.
# BUG-4: Fixed recent_events semantics. The snapshot now retains a bounded rolling
#        history across batches instead of only the current batch.
# SEC-1: Bounded and sanitized untrusted event payloads to prevent oversized or deeply
#        nested smart-home details from turning the Pi into a memory/CPU sink.
# SEC-2: Marked external smart-home text/details as untrusted so downstream LLM or
#        policy layers can segregate sensor content from trusted system instructions.
# IMP-1: Added recent-event-id deduplication and optional crash-safe cursor
#        checkpointing for at-least-once streams.
# IMP-2: Added capped exponential backoff with jitter and richer bounded worker state.
# IMP-3: Added optional push-stream fast path via subscribe_sensor_stream(...), while
#        preserving polling fallback for existing adapters.

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import queue
import random
import re
import tempfile
from threading import Event, Lock, Thread
import time
from typing import Protocol, runtime_checkable

from twinr.integrations.smarthome.adapter import SmartHomeIntegrationAdapter
from twinr.integrations.smarthome.models import SmartHomeEvent, SmartHomeEventBatch, SmartHomeEventKind

logger = logging.getLogger(__name__)

_EVENT_NAME_BY_KIND: dict[SmartHomeEventKind, str] = {
    SmartHomeEventKind.MOTION_DETECTED: "smart_home.motion_detected",
    SmartHomeEventKind.MOTION_CLEARED: "smart_home.motion_cleared",
    SmartHomeEventKind.BUTTON_PRESSED: "smart_home.button_pressed",
    SmartHomeEventKind.DEVICE_ONLINE: "smart_home.device_online",
    SmartHomeEventKind.DEVICE_OFFLINE: "smart_home.device_offline",
    SmartHomeEventKind.ALARM_TRIGGERED: "smart_home.alarm_triggered",
    SmartHomeEventKind.ALARM_CLEARED: "smart_home.alarm_cleared",
    SmartHomeEventKind.STATE_CHANGED: "smart_home.state_changed",
}

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def _positive_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive number.") from exc
    if parsed <= 0.0:
        raise ValueError(f"{field_name} must be a positive number.")
    return parsed


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive whole number.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive whole number.") from exc
    if parsed < 1:
        raise ValueError(f"{field_name} must be a positive whole number.")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field_name} must be a positive whole number.")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or any(character in stripped for character in ".eE"):
            raise ValueError(f"{field_name} must be a positive whole number.")
    return parsed


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _sanitize_text(value: object, *, max_chars: int) -> str:
    text = _coerce_text(value)
    if not text:
        return ""
    normalized = _CONTROL_CHARS_RE.sub(" ", text).replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 1:
        return normalized[:max_chars]
    return normalized[: max_chars - 1] + "…"


def _clone_json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _clone_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_json_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_json_value(item) for item in value)
    return value


def _clone_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _clone_json_value(item) for key, item in value.items()}


def _bounded_json_value(
    value: object,
    *,
    max_depth: int,
    max_items: int,
    max_text_chars: int,
    _depth: int = 0,
) -> object:
    if _depth >= max_depth:
        return {"_truncated": True, "_reason": "max_depth_exceeded"}
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, str):
        return _sanitize_text(value, max_chars=max_text_chars)
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        remaining_items = 0
        for index, (raw_key, raw_value) in enumerate(value.items()):
            if index >= max_items:
                try:
                    remaining_items = len(value) - max_items
                except Exception:
                    remaining_items = 1
                normalized["_truncated"] = True
                normalized["_truncated_items"] = max(remaining_items, 1)
                break
            key = _sanitize_text(raw_key, max_chars=max_text_chars) or f"key_{index}"
            normalized[key] = _bounded_json_value(
                raw_value,
                max_depth=max_depth,
                max_items=max_items,
                max_text_chars=max_text_chars,
                _depth=_depth + 1,
            )
        return normalized
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        sequence = list(value)
        normalized_items = [
            _bounded_json_value(
                item,
                max_depth=max_depth,
                max_items=max_items,
                max_text_chars=max_text_chars,
                _depth=_depth + 1,
            )
            for item in sequence[:max_items]
        ]
        if len(sequence) > max_items:
            normalized_items.append({"_truncated": True, "_truncated_items": len(sequence) - max_items})
        return normalized_items
    return _sanitize_text(value, max_chars=max_text_chars)


def _bounded_mapping(
    value: object,
    *,
    max_depth: int,
    max_items: int,
    max_text_chars: int,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    bounded = _bounded_json_value(
        value,
        max_depth=max_depth,
        max_items=max_items,
        max_text_chars=max_text_chars,
    )
    return bounded if isinstance(bounded, dict) else {}


def _parse_bool_like(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


def _bool_mapping(value: object) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, bool] = {}
    for raw_key, raw_value in value.items():
        key = _sanitize_text(raw_key, max_chars=256)
        if key:
            normalized[key] = _parse_bool_like(raw_value)
    return normalized


@dataclass(frozen=True, slots=True)
class SmartHomeObservation:
    """Describe one normalized Twinr observation derived from smart-home events."""

    facts: dict[str, object]
    event_names: tuple[str, ...]
    events: tuple[SmartHomeEvent, ...]
    next_cursor: str | None = None


class SmartHomeObservationBuilder:
    """Keep a compact fact snapshot while translating smart-home events."""

    def __init__(
        self,
        *,
        initial_facts: Mapping[str, object] | None = None,
        recent_events_limit: int = 8,
        max_detail_depth: int = 6,
        max_collection_items: int = 32,
        max_text_chars: int = 256,
    ) -> None:
        self.recent_events_limit = _positive_int(recent_events_limit, field_name="recent_events_limit")
        self.max_detail_depth = _positive_int(max_detail_depth, field_name="max_detail_depth")
        self.max_collection_items = _positive_int(max_collection_items, field_name="max_collection_items")
        self.max_text_chars = _positive_int(max_text_chars, field_name="max_text_chars")

        self._facts = _clone_mapping(initial_facts)
        smart_home = _clone_mapping(self._facts.get("smart_home"))
        existing_recent = smart_home.get("recent_events")
        self._recent_events: deque[dict[str, object]] = deque(maxlen=self.recent_events_limit)
        if isinstance(existing_recent, list):
            for item in existing_recent[-self.recent_events_limit :]:
                if isinstance(item, Mapping):
                    self._recent_events.append(
                        _bounded_mapping(
                            item,
                            max_depth=self.max_detail_depth,
                            max_items=self.max_collection_items,
                            max_text_chars=self.max_text_chars,
                        )
                    )
        smart_home["recent_events"] = list(self._recent_events)
        smart_home["button_pressed"] = False
        smart_home["external_content_untrusted"] = True
        self._facts["smart_home"] = smart_home

    def build(self, batch: SmartHomeEventBatch) -> SmartHomeObservation | None:
        """Translate one smart-home event batch into a Twinr observation."""

        if not isinstance(batch, SmartHomeEventBatch):
            raise TypeError("batch must be a SmartHomeEventBatch.")
        if not batch.events:
            return None

        facts = self._facts
        smart_home = _clone_mapping(facts.get("smart_home"))
        motion_active = _bool_mapping(smart_home.get("motion_active_by_entity"))
        device_online = _bool_mapping(smart_home.get("device_online_by_entity"))
        alarm_active = _bool_mapping(smart_home.get("alarm_active_by_entity"))
        event_names: list[str] = []
        button_pressed = False

        smart_home["sensor_stream_live"] = batch.stream_live
        smart_home["external_content_untrusted"] = True
        if batch.next_cursor is not None:
            smart_home["sensor_stream_cursor"] = batch.next_cursor

        for event in batch.events:
            event_name = _EVENT_NAME_BY_KIND.get(event.event_kind)
            if event_name is not None:
                event_names.append(event_name)

            bounded_details = _bounded_mapping(
                event.details,
                max_depth=self.max_detail_depth,
                max_items=self.max_collection_items,
                max_text_chars=self.max_text_chars,
            )
            smart_home["last_event_kind"] = event.event_kind.value
            smart_home["last_event_entity_id"] = event.entity_id
            smart_home["last_event_provider"] = _sanitize_text(event.provider, max_chars=self.max_text_chars)
            smart_home["last_event_at"] = event.observed_at
            smart_home["last_event_area"] = _sanitize_text(event.area, max_chars=self.max_text_chars)
            smart_home["last_event_label"] = _sanitize_text(event.label, max_chars=self.max_text_chars)
            smart_home["last_event_details"] = bounded_details
            smart_home["last_event_details_untrusted"] = True

            event_payload = {
                "event_id": event.event_id,
                "provider": _sanitize_text(event.provider, max_chars=self.max_text_chars),
                "entity_id": event.entity_id,
                "event_kind": event.event_kind.value,
                "observed_at": event.observed_at,
                "label": _sanitize_text(event.label, max_chars=self.max_text_chars),
                "area": _sanitize_text(event.area, max_chars=self.max_text_chars),
                "details": bounded_details,
                "details_untrusted": True,
            }
            self._recent_events.append(event_payload)

            if event.event_kind is SmartHomeEventKind.MOTION_DETECTED:
                motion_active[event.entity_id] = True
                smart_home["last_motion_entity_id"] = event.entity_id
                smart_home["last_motion_area"] = _sanitize_text(event.area, max_chars=self.max_text_chars)
                smart_home["last_motion_detected_at"] = event.observed_at
            elif event.event_kind is SmartHomeEventKind.MOTION_CLEARED:
                motion_active[event.entity_id] = False
                smart_home["last_motion_entity_id"] = event.entity_id
                smart_home["last_motion_area"] = _sanitize_text(event.area, max_chars=self.max_text_chars)
                smart_home["last_motion_cleared_at"] = event.observed_at
            elif event.event_kind is SmartHomeEventKind.BUTTON_PRESSED:
                button_pressed = True
                smart_home["last_button_entity_id"] = event.entity_id
                smart_home["last_button_area"] = _sanitize_text(event.area, max_chars=self.max_text_chars)
                smart_home["last_button_pressed_at"] = event.observed_at
            elif event.event_kind is SmartHomeEventKind.DEVICE_ONLINE:
                device_online[event.entity_id] = True
                smart_home["last_device_online_entity_id"] = event.entity_id
                smart_home["last_device_online_at"] = event.observed_at
            elif event.event_kind is SmartHomeEventKind.DEVICE_OFFLINE:
                device_online[event.entity_id] = False
                smart_home["last_device_offline_entity_id"] = event.entity_id
                smart_home["last_device_offline_at"] = event.observed_at
            elif event.event_kind is SmartHomeEventKind.ALARM_TRIGGERED:
                alarm_active[event.entity_id] = True
                smart_home["last_alarm_entity_id"] = event.entity_id
                smart_home["last_alarm_triggered_at"] = event.observed_at
            elif event.event_kind is SmartHomeEventKind.ALARM_CLEARED:
                alarm_active[event.entity_id] = False
                smart_home["last_alarm_entity_id"] = event.entity_id
                smart_home["last_alarm_cleared_at"] = event.observed_at

        active_motion_ids = sorted(entity_id for entity_id, active in motion_active.items() if active)
        offline_device_ids = sorted(entity_id for entity_id, online in device_online.items() if not online)
        active_alarm_ids = sorted(entity_id for entity_id, active in alarm_active.items() if active)

        smart_home["button_pressed"] = button_pressed
        smart_home["motion_detected"] = bool(active_motion_ids)
        smart_home["motion_entity_ids"] = active_motion_ids
        smart_home["motion_active_by_entity"] = motion_active
        smart_home["device_online_by_entity"] = device_online
        smart_home["offline_entity_ids"] = offline_device_ids
        smart_home["device_offline"] = bool(offline_device_ids)
        smart_home["alarm_active_by_entity"] = alarm_active
        smart_home["alarm_entity_ids"] = active_alarm_ids
        smart_home["alarm_triggered"] = bool(active_alarm_ids)
        smart_home["recent_events"] = list(self._recent_events)

        facts["smart_home"] = smart_home
        self._facts = facts
        return SmartHomeObservation(
            facts=_clone_mapping(facts),
            event_names=tuple(dict.fromkeys(event_names)),
            events=batch.events,
            next_cursor=batch.next_cursor,
        )


AdapterLoader = Callable[[], SmartHomeIntegrationAdapter | None]
ObservationCallback = Callable[[SmartHomeObservation], None]
EmitCallback = Callable[[str], None]
RecordEventCallback = Callable[[str, str], None]


@runtime_checkable
class SmartHomePushSensorStream(Protocol):
    """Optional push-stream fast path for providers that support subscriptions."""

    def subscribe_sensor_stream(
        self,
        *,
        callback: Callable[[SmartHomeEventBatch], None],
        cursor: str | None = None,
    ) -> object:
        """Start streaming event batches to callback and return an unsubscribe handle."""

        ...


class SmartHomeSensorWorker:
    """Poll one managed smart-home stream and publish Twinr observations."""

    def __init__(
        self,
        *,
        adapter_loader: AdapterLoader,
        observation_callback: ObservationCallback,
        idle_sleep_s: float = 1.0,
        retry_delay_s: float = 2.0,
        batch_limit: int = 8,
        emit: EmitCallback | None = None,
        record_event: RecordEventCallback | None = None,
        initial_cursor: str | None = None,
        checkpoint_path: str | Path | None = None,
        dedupe_window: int = 256,
        retry_delay_max_s: float = 30.0,
        queue_capacity: int = 32,
        recent_events_limit: int = 8,
        max_detail_depth: int = 6,
        max_collection_items: int = 32,
        max_text_chars: int = 256,
        initial_facts: Mapping[str, object] | None = None,
    ) -> None:
        self._adapter_loader = adapter_loader
        self._observation_callback = observation_callback
        self.idle_sleep_s = _positive_float(idle_sleep_s, field_name="idle_sleep_s")
        self.retry_delay_s = _positive_float(retry_delay_s, field_name="retry_delay_s")
        self.retry_delay_max_s = _positive_float(retry_delay_max_s, field_name="retry_delay_max_s")
        if self.retry_delay_max_s < self.retry_delay_s:
            raise ValueError("retry_delay_max_s must be greater than or equal to retry_delay_s.")
        self.batch_limit = _positive_int(batch_limit, field_name="batch_limit")
        self.queue_capacity = _positive_int(queue_capacity, field_name="queue_capacity")
        self.dedupe_window = _positive_int(dedupe_window, field_name="dedupe_window")
        self._emit = emit
        self._record_event = record_event
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._lifecycle_lock = Lock()
        self._builder = SmartHomeObservationBuilder(
            initial_facts=initial_facts,
            recent_events_limit=recent_events_limit,
            max_detail_depth=max_detail_depth,
            max_collection_items=max_collection_items,
            max_text_chars=max_text_chars,
        )
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._cursor = initial_cursor if initial_cursor is not None else self._load_checkpoint()
        self._delivered_event_ids: deque[str] = deque(maxlen=self.dedupe_window)
        self._delivered_event_id_set: set[str] = set()
        self._rng = random.Random()
        self._consecutive_failures = 0

    def start(self) -> None:
        """Start the background worker exactly once."""

        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = Thread(
                target=self._run,
                daemon=True,
                name="twinr-smart-home-sensor",
            )
            self._thread.start()

    def stop(self, *, timeout_s: float = 3.0) -> None:
        """Stop the worker and join the thread for a bounded period."""

        self._stop_event.set()
        with self._lifecycle_lock:
            thread = self._thread
        if thread is None:
            return
        thread.join(timeout=max(0.1, timeout_s))
        if not thread.is_alive():
            with self._lifecycle_lock:
                if self._thread is thread:
                    self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                adapter = self._adapter_loader()
                if not isinstance(adapter, SmartHomeIntegrationAdapter) or adapter.sensor_stream is None:
                    self._consecutive_failures = 0
                    if self._stop_event.wait(self.idle_sleep_s):
                        return
                    continue

                sensor_stream = adapter.sensor_stream
                if isinstance(sensor_stream, SmartHomePushSensorStream):
                    self._run_push_stream(sensor_stream)
                else:
                    self._run_poll_iteration(sensor_stream)
                self._consecutive_failures = 0
            except Exception as exc:
                logger.warning("Smart-home sensor worker iteration failed.", exc_info=True)
                self._emit_safe("smart_home_sensor_worker_error=true")
                self._record_event_safe(
                    "smart_home_sensor_worker_failed",
                    f"{type(exc).__name__}: {exc}",
                )
                self._consecutive_failures += 1
                if self._stop_event.wait(self._next_retry_delay()):
                    return

    def _run_poll_iteration(self, sensor_stream: object) -> None:
        batch = sensor_stream.read_sensor_stream(cursor=self._cursor, limit=self.batch_limit)
        if not isinstance(batch, SmartHomeEventBatch):
            raise TypeError("read_sensor_stream() must return SmartHomeEventBatch.")
        delivered = self._process_batch(batch)
        if delivered == 0 and self._stop_event.wait(self.idle_sleep_s):
            return

    def _run_push_stream(self, sensor_stream: SmartHomePushSensorStream) -> None:
        batches: queue.Queue[SmartHomeEventBatch] = queue.Queue(maxsize=self.queue_capacity)

        def on_batch(batch: SmartHomeEventBatch) -> None:
            if not isinstance(batch, SmartHomeEventBatch):
                logger.warning("Ignored malformed smart-home sensor batch %r.", type(batch).__name__)
                return
            try:
                batches.put_nowait(batch)
            except queue.Full:
                dropped: SmartHomeEventBatch | None = None
                try:
                    dropped = batches.get_nowait()
                except queue.Empty:
                    dropped = None
                try:
                    batches.put_nowait(batch)
                except queue.Full:
                    return
                finally:
                    dropped_events = len(dropped.events) if dropped is not None else 0
                    self._emit_safe(f"smart_home_sensor_queue_overflow_dropped_events={dropped_events}")

        unsubscribe_handle = sensor_stream.subscribe_sensor_stream(callback=on_batch, cursor=self._cursor)
        self._emit_safe("smart_home_sensor_mode=push")
        try:
            while not self._stop_event.is_set():
                try:
                    batch = batches.get(timeout=self.idle_sleep_s)
                except queue.Empty:
                    continue
                self._process_batch(batch)
        finally:
            self._close_subscription(unsubscribe_handle)

    def _process_batch(self, batch: SmartHomeEventBatch) -> int:
        next_cursor = batch.next_cursor if batch.next_cursor is not None else self._cursor
        pending_events = tuple(
            event for event in batch.events if event.event_id not in self._delivered_event_id_set
        )

        if not pending_events:
            self._update_cursor(next_cursor)
            self._emit_safe("smart_home_sensor_events=0")
            return 0

        filtered_batch = SmartHomeEventBatch(
            events=pending_events,
            next_cursor=batch.next_cursor,
            stream_live=batch.stream_live,
        )
        observation = self._builder.build(filtered_batch)
        if observation is None:
            self._update_cursor(next_cursor)
            self._emit_safe("smart_home_sensor_events=0")
            return 0

        self._observation_callback(observation)
        for event in pending_events:
            self._remember_delivered_event_id(event.event_id)
        self._update_cursor(next_cursor)
        self._emit_safe(f"smart_home_sensor_events={len(observation.events)}")
        return len(observation.events)

    def _remember_delivered_event_id(self, event_id: str) -> None:
        if event_id in self._delivered_event_id_set:
            return
        if len(self._delivered_event_ids) == self._delivered_event_ids.maxlen:
            oldest = self._delivered_event_ids.popleft()
            self._delivered_event_id_set.discard(oldest)
        self._delivered_event_ids.append(event_id)
        self._delivered_event_id_set.add(event_id)

    def _next_retry_delay(self) -> float:
        exponent = max(0, min(self._consecutive_failures - 1, 8))
        capped = min(self.retry_delay_max_s, self.retry_delay_s * (2 ** exponent))
        if capped <= self.retry_delay_s:
            return self.retry_delay_s
        return self._rng.uniform(self.retry_delay_s, capped)

    def _update_cursor(self, cursor: str | None) -> None:
        if cursor is None or cursor == self._cursor:
            return
        self._cursor = cursor
        self._persist_checkpoint(cursor)

    def _load_checkpoint(self) -> str | None:
        path = self._checkpoint_path
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read smart-home checkpoint file %s.", path, exc_info=True)
            return None
        cursor = payload.get("cursor")
        return cursor if isinstance(cursor, str) and cursor.strip() else None

    def _persist_checkpoint(self, cursor: str) -> None:
        path = self._checkpoint_path
        if path is None:
            return
        payload = {
            "cursor": cursor,
            "updated_at_unix_s": time.time(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", closefd=True) as handle:
                json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            Path(temporary_path).replace(path)
        except Exception:
            logger.warning("Failed to persist smart-home checkpoint file %s.", path, exc_info=True)
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _close_subscription(handle: object) -> None:
        if handle is None:
            return
        for attribute in ("close", "cancel", "unsubscribe", "stop"):
            close_callable = getattr(handle, attribute, None)
            if callable(close_callable):
                try:
                    close_callable()
                except Exception:
                    logger.debug("Subscription handle %r.%s() failed.", type(handle).__name__, attribute, exc_info=True)
                return
        if callable(handle):
            try:
                handle()
            except Exception:
                logger.debug("Subscription callable failed.", exc_info=True)

    def _emit_safe(self, message: str) -> None:
        if not callable(self._emit):
            return
        try:
            self._emit(message)
        except Exception:
            return

    def _record_event_safe(self, event_name: str, detail: str) -> None:
        if not callable(self._record_event):
            return
        try:
            self._record_event(event_name, detail)
        except Exception:
            return


__all__ = [
    "SmartHomeObservation",
    "SmartHomeObservationBuilder",
    "SmartHomePushSensorStream",
    "SmartHomeSensorWorker",
]