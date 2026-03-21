"""Map smart-home events into Twinr observations and run bounded workers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import logging
from threading import Event, Thread
import time

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
    return parsed


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


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


def _bool_mapping(value: object) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, bool] = {}
    for raw_key, raw_value in value.items():
        key = _coerce_text(raw_key).strip()
        if key:
            normalized[key] = bool(raw_value)
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

    def __init__(self, *, initial_facts: Mapping[str, object] | None = None) -> None:
        self._facts = _clone_mapping(initial_facts)

    def build(self, batch: SmartHomeEventBatch) -> SmartHomeObservation | None:
        """Translate one smart-home event batch into a Twinr observation."""

        if not batch.events:
            return None
        facts = _clone_mapping(self._facts)
        smart_home = _clone_mapping(facts.get("smart_home"))
        motion_active = _bool_mapping(smart_home.get("motion_active_by_entity"))
        device_online = _bool_mapping(smart_home.get("device_online_by_entity"))
        alarm_active = _bool_mapping(smart_home.get("alarm_active_by_entity"))
        event_names: list[str] = []
        events_payload: list[dict[str, object]] = [event.as_dict() for event in batch.events]

        smart_home["sensor_stream_live"] = batch.stream_live
        if batch.next_cursor is not None:
            smart_home["sensor_stream_cursor"] = batch.next_cursor

        for event in batch.events:
            event_name = _EVENT_NAME_BY_KIND.get(event.event_kind)
            if event_name is not None:
                event_names.append(event_name)
            smart_home["last_event_kind"] = event.event_kind.value
            smart_home["last_event_entity_id"] = event.entity_id
            smart_home["last_event_provider"] = event.provider
            smart_home["last_event_at"] = event.observed_at
            smart_home["last_event_area"] = event.area
            smart_home["last_event_label"] = event.label
            smart_home["last_event_details"] = _clone_mapping(event.details)

            if event.event_kind is SmartHomeEventKind.MOTION_DETECTED:
                motion_active[event.entity_id] = True
                smart_home["last_motion_entity_id"] = event.entity_id
                smart_home["last_motion_area"] = event.area
                smart_home["last_motion_detected_at"] = event.observed_at
            elif event.event_kind is SmartHomeEventKind.MOTION_CLEARED:
                motion_active[event.entity_id] = False
                smart_home["last_motion_entity_id"] = event.entity_id
                smart_home["last_motion_area"] = event.area
                smart_home["last_motion_cleared_at"] = event.observed_at
            elif event.event_kind is SmartHomeEventKind.BUTTON_PRESSED:
                smart_home["button_pressed"] = True
                smart_home["last_button_entity_id"] = event.entity_id
                smart_home["last_button_area"] = event.area
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

        smart_home["motion_detected"] = bool(active_motion_ids)
        smart_home["motion_entity_ids"] = active_motion_ids
        smart_home["motion_active_by_entity"] = motion_active
        smart_home["device_online_by_entity"] = device_online
        smart_home["offline_entity_ids"] = offline_device_ids
        smart_home["device_offline"] = bool(offline_device_ids)
        smart_home["alarm_active_by_entity"] = alarm_active
        smart_home["alarm_entity_ids"] = active_alarm_ids
        smart_home["alarm_triggered"] = bool(active_alarm_ids)
        smart_home["recent_events"] = events_payload[-8:]

        facts["smart_home"] = smart_home
        self._facts = _clone_mapping(facts)
        return SmartHomeObservation(
            facts=facts,
            event_names=tuple(dict.fromkeys(event_names)),
            events=batch.events,
            next_cursor=batch.next_cursor,
        )


AdapterLoader = Callable[[], SmartHomeIntegrationAdapter | None]
ObservationCallback = Callable[[SmartHomeObservation], None]
EmitCallback = Callable[[str], None]
RecordEventCallback = Callable[[str, str], None]


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
    ) -> None:
        self._adapter_loader = adapter_loader
        self._observation_callback = observation_callback
        self.idle_sleep_s = _positive_float(idle_sleep_s, field_name="idle_sleep_s")
        self.retry_delay_s = _positive_float(retry_delay_s, field_name="retry_delay_s")
        self.batch_limit = _positive_int(batch_limit, field_name="batch_limit")
        self._emit = emit
        self._record_event = record_event
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._builder = SmartHomeObservationBuilder()

    def start(self) -> None:
        """Start the background worker exactly once."""

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
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=max(0.1, timeout_s))

    def _run(self) -> None:
        while not self._stop_event.is_set():
            adapter = self._adapter_loader()
            if not isinstance(adapter, SmartHomeIntegrationAdapter) or adapter.sensor_stream is None:
                if self._stop_event.wait(self.idle_sleep_s):
                    return
                continue
            try:
                batch = adapter.sensor_stream.read_sensor_stream(limit=self.batch_limit)
                observation = self._builder.build(batch)
                if observation is not None:
                    self._observation_callback(observation)
                    self._emit_safe(f"smart_home_sensor_events={len(observation.events)}")
                elif self._stop_event.wait(self.idle_sleep_s):
                    return
            except Exception as exc:
                logger.warning("Smart-home sensor worker iteration failed.", exc_info=True)
                self._emit_safe("smart_home_sensor_worker_error=true")
                self._record_event_safe(
                    "smart_home_sensor_worker_failed",
                    f"{type(exc).__name__}: {exc}",
                )
                if self._stop_event.wait(self.retry_delay_s):
                    return

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
    "SmartHomeSensorWorker",
]
