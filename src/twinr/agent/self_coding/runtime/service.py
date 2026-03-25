"""Execute compiled self-coding skill packages through a brokered sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
import logging
import math
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Callable, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.agent.self_coding.execution_runs import SelfCodingExecutionRunService
from twinr.agent.self_coding.sandbox import SelfCodingSandboxRunner, SelfCodingSandboxTimeoutError, SkillBrokerPolicy
from twinr.agent.self_coding.store import SelfCodingStore
from twinr.agent.self_coding.status import LearnedSkillStatus

from .contracts import SkillPackageDocument, skill_package_document_from_document
from .state import SelfCodingSkillRuntimeStore

if TYPE_CHECKING:
    from twinr.agent.self_coding.health import SelfCodingHealthService


logger = logging.getLogger(__name__)

_DEFAULT_TIMEZONE_NAME = "Europe/Berlin"
_FALLBACK_TIMEZONE_NAME = "UTC"

# AUDIT-FIX(#5): Bound brokered payload sizes and speech fan-out so generated skills cannot
# create runaway token cost, oversized prompts, or abusive TTS queues inside one handler run.
_MAX_SEARCH_QUESTION_CHARS = 4_000
_MAX_SUMMARY_TEXT_CHARS = 50_000
_MAX_SUMMARY_INSTRUCTIONS_CHARS = 2_000
_MAX_SAY_TEXT_CHARS = 2_000
_MAX_SAY_CALLS = 3
_MAX_TOTAL_SPOKEN_CHARS = 4_000
_MAX_EMAIL_LIST_LIMIT = 25
_MAX_CALENDAR_LIST_LIMIT = 25
_MAX_CALENDAR_DAYS = 31

# AUDIT-FIX(#4): Serialize state flushes per skill/version inside the single-process runtime so
# overlapping executions do not overwrite each other's persisted state with stale snapshots.
_STATE_LOCKS_GUARD = RLock()
_STATE_LOCKS: dict[tuple[str, int], RLock] = {}

_ALLOWED_EVENT_SEVERITIES = {"debug", "info", "warning", "error", "critical"}
_LOGGER_LEVEL_BY_SEVERITY = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


@dataclass(frozen=True, slots=True)
class SkillSearchResult:
    """Return the normalized result of one live search call inside a skill."""

    answer: str
    sources: tuple[str, ...]
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None


# AUDIT-FIX(#8): Load activation only once per materialized package to avoid TOCTOU drift between
# artifact selection and skill-name lookup.
@dataclass(frozen=True, slots=True)
class _LoadedSkillPackage:
    document: SkillPackageDocument
    materialized_root: Path
    skill_name: str


class SkillContext:
    """Provide the bounded runtime API exposed to generated skill code."""

    def __init__(
        self,
        *,
        owner: Any,
        runtime_store: SelfCodingSkillRuntimeStore,
        skill_id: str,
        version: int,
        skill_name: str,
        now: datetime,
    ) -> None:
        self.owner = owner
        self.runtime_store = runtime_store
        self.skill_id = _validate_identifier(skill_id, field_name="skill_id")  # AUDIT-FIX(#8): Reject traversal/control characters before touching file-backed stores or automation IDs.
        self.version = _normalize_version(version)  # AUDIT-FIX(#8): Normalize version once so all state paths use the same validated value.
        self.skill_name = str(skill_name or "").strip() or self.skill_id
        self._timezone = _resolve_timezone(self._timezone_name(), owner=owner, context="skill_context")  # AUDIT-FIX(#3): Invalid timezone config must degrade to a safe fallback instead of crashing execution.
        self.now = _normalize_datetime(now, assume_tz=self._timezone, field_name="now").astimezone(self._timezone)  # AUDIT-FIX(#3): Naive datetimes are interpreted in configured local time, not host-default or implicit UTC.
        self._state = self._load_initial_state()  # AUDIT-FIX(#4): Corrupted/non-dict state falls back safely instead of aborting the whole run.
        self._dirty = False
        self._dirty_keys: set[str] = set()  # AUDIT-FIX(#4): Track changed keys so flush can merge against latest persisted state.
        self._deleted_keys: set[str] = set()  # AUDIT-FIX(#4): Track deletions explicitly to avoid resurrecting removed keys on merge.
        self._spoken_count = 0
        self._spoken_char_count = 0  # AUDIT-FIX(#5): Bound cumulative speech volume per handler execution.
        self._managed_integrations_runtime: Any | None = None
        self._managed_integrations_lock = RLock()  # AUDIT-FIX(#6): Prevent duplicate lazy initialization under concurrent broker calls.

    def search_web(self, question: str, *, location_hint: str | None = None, date_context: str | None = None) -> SkillSearchResult:
        backend = getattr(self.owner, "print_backend", None)
        if backend is None or not callable(getattr(backend, "search_live_info_with_metadata", None)):
            raise RuntimeError("search backend is unavailable for self-coding skill execution")
        question_text = self._bounded_text(question, field_name="question", max_chars=_MAX_SEARCH_QUESTION_CHARS)  # AUDIT-FIX(#5): Reject oversized search prompts before they hit paid/networked backends.
        try:
            result = backend.search_live_info_with_metadata(
                question_text,
                conversation=self._search_conversation_context(),
                location_hint=self._optional_text(location_hint),
                date_context=self._optional_text(date_context),
            )
        except Exception as exc:  # AUDIT-FIX(#5): Do not leak raw backend exceptions, URLs, or vendor-specific diagnostics into skill/runtime surfaces.
            raise RuntimeError("web search failed during self-coding skill execution") from exc
        answer = str(getattr(result, "answer", "") or "").strip()
        if not answer:
            raise RuntimeError("self-coding web search returned no answer")
        raw_sources = getattr(result, "sources", ()) or ()
        sources = tuple(str(item).strip() for item in raw_sources if str(item).strip())
        return SkillSearchResult(
            answer=answer,
            sources=sources,
            response_id=self._optional_text(getattr(result, "response_id", None)),
            request_id=self._optional_text(getattr(result, "request_id", None)),
            model=self._optional_text(getattr(result, "model", None)),
        )

    def summarize_text(self, text: str, instructions: str | None = None) -> str:
        backend = getattr(self.owner, "agent_provider", None)
        if backend is None or not callable(getattr(backend, "respond_with_metadata", None)):
            backend = getattr(self.owner, "print_backend", None)
        if backend is None or not callable(getattr(backend, "respond_with_metadata", None)):
            raise RuntimeError("summary backend is unavailable for self-coding skill execution")
        prompt = self._bounded_text(text, field_name="text", max_chars=_MAX_SUMMARY_TEXT_CHARS)  # AUDIT-FIX(#5): Bound brokered summarization payload size.
        if instructions:
            instructions_text = self._bounded_text(
                instructions,
                field_name="instructions",
                max_chars=_MAX_SUMMARY_INSTRUCTIONS_CHARS,
            )
            prompt = f"{instructions_text}\n\n{prompt}"
        try:
            result = backend.respond_with_metadata(prompt, instructions=None, allow_web_search=False)
        except Exception as exc:  # AUDIT-FIX(#5): Surface a generic execution error instead of leaking provider internals.
            raise RuntimeError("text summarization failed during self-coding skill execution") from exc
        summary = str(getattr(result, "text", "") or "").strip()
        if not summary:
            raise RuntimeError("self-coding summary backend returned no text")
        return summary

    def say(self, text: str) -> None:
        spoken_text = self._bounded_text(text, field_name="text", max_chars=_MAX_SAY_TEXT_CHARS)  # AUDIT-FIX(#5): Refuse oversized speech payloads that would monopolize the speaker.
        if self._spoken_count >= _MAX_SAY_CALLS or (self._spoken_char_count + len(spoken_text)) > _MAX_TOTAL_SPOKEN_CHARS:
            raise RuntimeError("speech output limit reached for self-coding skill execution")  # AUDIT-FIX(#5): Bound TTS fan-out per handler run.
        speaker = getattr(self.owner, "_speak_automation_text", None)
        if not callable(speaker):
            raise RuntimeError("speech output is unavailable for self-coding skill execution")
        synthetic_entry = type(
            "_SyntheticSelfCodingEntry",
            (),
            {"automation_id": f"ase_{self.skill_id}_v{self.version}", "name": self.skill_name},
        )()
        try:
            speaker(synthetic_entry, spoken_text)
        except Exception as exc:  # AUDIT-FIX(#5): Hide raw speaker/backend exceptions from generated skill code.
            raise RuntimeError("speech output failed during self-coding skill execution") from exc
        self._spoken_count += 1
        self._spoken_char_count += len(spoken_text)

    def store_json(self, key: str, value: Any) -> None:
        normalized_key = self._state_key(key)
        normalized_value = self._json_value(value)
        self._state[normalized_key] = normalized_value
        self._dirty = True
        self._dirty_keys.add(normalized_key)  # AUDIT-FIX(#4): Persist only the keys this run actually changed.
        self._deleted_keys.discard(normalized_key)  # AUDIT-FIX(#4): A later write must win over an earlier delete inside the same run.

    def load_json(self, key: str, default: Any | None = None) -> Any:
        normalized_key = self._state_key(key)
        if normalized_key not in self._state:
            return default
        return self._json_value(self._state[normalized_key])

    def delete_json(self, key: str) -> None:
        normalized_key = self._state_key(key)
        if normalized_key in self._state:
            self._state.pop(normalized_key, None)
            self._dirty = True
            self._deleted_keys.add(normalized_key)  # AUDIT-FIX(#4): Track deletions explicitly for merged flushes.
            self._dirty_keys.discard(normalized_key)  # AUDIT-FIX(#4): Deleted keys must not be re-written from the current snapshot.

    def list_json_keys(self, prefix: str | None = None) -> tuple[str, ...]:
        normalized_prefix = "" if prefix is None else str(prefix).strip()
        keys = sorted(str(item) for item in self._state.keys())
        if not normalized_prefix:
            return tuple(keys)
        return tuple(item for item in keys if item.startswith(normalized_prefix))

    def merge_json(self, key: str, patch: Any) -> Any:
        normalized_key = self._state_key(key)
        current = self._state.get(normalized_key)
        merged = self._merge_json_value(current, patch)
        normalized_value = self._json_value(merged)
        self._state[normalized_key] = normalized_value
        self._dirty = True
        self._dirty_keys.add(normalized_key)  # AUDIT-FIX(#4): Record merged keys for conflict-minimized flush.
        self._deleted_keys.discard(normalized_key)  # AUDIT-FIX(#4): A merge is a write and therefore cancels a pending delete.
        return self._json_value(normalized_value)

    def today_local_date(self) -> str:
        return self.now.date().isoformat()

    def now_iso(self) -> str:
        return self.now.astimezone(UTC).isoformat().replace("+00:00", "Z")

    def current_presence_session_id(self) -> int | None:
        callback = getattr(self.owner, "_current_presence_session_id", None)
        if not callable(callback):
            return None
        try:
            value = callback()
        except Exception:
            return None
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def current_sensor_facts(self) -> dict[str, Any]:
        facts = getattr(self.owner, "_latest_sensor_observation_facts", None)
        if not isinstance(facts, dict):
            return {}
        try:
            cloned = self._json_value(facts)
        except ValueError:  # AUDIT-FIX(#9): Sensor payloads may contain NaN/datetime/object values; degrade to an empty fact set.
            self.log_event("failed_to_clone_sensor_facts", severity="warning")
            return {}
        return cloned if isinstance(cloned, dict) else {}

    def is_night_mode(self) -> bool:
        try:
            return bool(getattr(self.owner, "_night_mode", False))
        except Exception:
            return False

    def is_private_for_speech(self) -> bool:
        facts = getattr(self.owner, "_latest_sensor_observation_facts", None)
        if not isinstance(facts, dict):
            return False  # AUDIT-FIX(#1): Unknown sensor state is not private enough for sensitive speech.
        explicit_private = facts.get("is_private_for_speech", facts.get("private_for_speech"))
        if isinstance(explicit_private, bool):
            return explicit_private  # AUDIT-FIX(#1): Honor an explicit privacy decision from upstream sensing if available.
        camera = facts.get("camera")
        if not isinstance(camera, dict):
            return False  # AUDIT-FIX(#1): Missing camera details must not be treated as safe for private speech.
        explicit_camera_private = camera.get("is_private_for_speech", camera.get("private_for_speech"))
        if isinstance(explicit_camera_private, bool):
            return explicit_camera_private  # AUDIT-FIX(#1): Prefer explicit privacy decisions over heuristic fallbacks.
        for key in ("count_persons", "person_count", "visible_persons"):
            if key in camera:
                try:
                    return int(camera.get(key) or 0) <= 1
                except (TypeError, ValueError):
                    return False
        if "person_visible" in camera:
            return False  # AUDIT-FIX(#1): A boolean "someone is visible" signal is insufficient to prove privacy.
        return False

    def log_event(self, event: str, *, severity: str = "info", **data: object) -> None:
        safe_event = _safe_control_text(event, fallback="self_coding_event", max_chars=120)  # AUDIT-FIX(#9): Strip control characters from log/event names to avoid log forging.
        _emit_owner_event(
            self.owner,
            _normalize_severity(severity),  # AUDIT-FIX(#9): Restrict severity values to known log channels.
            safe_event,
            skill_id=self.skill_id,
            version=self.version,
            **data,
        )

    def list_recent_emails(self, *, limit: int = 5, unread_only: bool = True) -> tuple[dict[str, Any], ...]:
        runtime = self._managed_integrations()
        adapter = None if runtime is None else getattr(runtime, "email_mailbox", None)
        if adapter is None or not callable(getattr(adapter, "list_recent", None)):
            raise RuntimeError("email integration is unavailable for self-coding skill execution")
        bounded_limit = _coerce_positive_int(limit, field_name="limit", maximum=_MAX_EMAIL_LIST_LIMIT)  # AUDIT-FIX(#5): Cap mailbox fan-out for cost/latency control.
        try:
            items = adapter.list_recent(limit=bounded_limit, unread_only=bool(unread_only))
        except Exception as exc:  # AUDIT-FIX(#5): Hide raw mailbox adapter failures from generated skills and run metadata.
            raise RuntimeError("email integration request failed during self-coding skill execution") from exc
        normalized: list[dict[str, Any]] = []
        for item in items or ():
            normalized.append(
                {
                    "message_id": self._optional_text(getattr(item, "message_id", None)),
                    "subject": self._optional_text(getattr(item, "subject", None)),
                    "from_display": self._optional_text(getattr(item, "from_display", None)),
                    "from_address": self._optional_text(getattr(item, "from_address", None)),
                    "received_at": self._optional_text(getattr(item, "received_at", None)),
                    "snippet": self._optional_text(getattr(item, "snippet", None)),
                    "unread": bool(getattr(item, "unread", False)),
                }
            )
        return tuple(normalized)

    def list_calendar_events(
        self,
        *,
        days: int = 1,
        limit: int = 5,
        start_iso: str | None = None,
        end_iso: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        runtime = self._managed_integrations()
        adapter = None if runtime is None else getattr(runtime, "calendar_agenda", None)
        if adapter is None or not callable(getattr(adapter, "list_events", None)):
            raise RuntimeError("calendar integration is unavailable for self-coding skill execution")
        bounded_limit = _coerce_positive_int(limit, field_name="limit", maximum=_MAX_CALENDAR_LIST_LIMIT)  # AUDIT-FIX(#5): Cap agenda fan-out for deterministic latency.
        start_at, end_at = self._calendar_range(days=days, start_iso=start_iso, end_iso=end_iso)
        try:
            items = adapter.list_events(start_at=start_at, end_at=end_at, limit=bounded_limit)
        except Exception as exc:  # AUDIT-FIX(#5): Avoid exposing adapter-specific failures through the skill surface.
            raise RuntimeError("calendar integration request failed during self-coding skill execution") from exc
        normalized: list[dict[str, Any]] = []
        for item in items or ():
            normalized.append(
                {
                    "title": self._optional_text(getattr(item, "title", None)),
                    "start_at": self._optional_text(getattr(item, "start_at", None)),
                    "end_at": self._optional_text(getattr(item, "end_at", None)),
                    "location": self._optional_text(getattr(item, "location", None)),
                    "description": self._optional_text(getattr(item, "description", None)),
                }
            )
        return tuple(normalized)

    def flush(self) -> None:
        if not self._dirty:
            return
        state_lock = _state_lock_for(self.skill_id, self.version)
        with state_lock:
            latest_state = self._load_initial_state()  # AUDIT-FIX(#4): Merge against current persisted state to avoid lost updates from concurrent executions.
            for key in self._deleted_keys:
                latest_state.pop(key, None)
            for key in self._dirty_keys:
                latest_state[key] = self._json_value(self._state[key])
            self.runtime_store.save_state(skill_id=self.skill_id, version=self.version, payload=latest_state)
            self._state = latest_state
            self._dirty = False
            self._dirty_keys.clear()
            self._deleted_keys.clear()

    def close(self) -> None:
        runtime = self._managed_integrations_runtime
        self._managed_integrations_runtime = None
        if runtime is None:
            return
        for callback_name in ("close", "shutdown", "dispose"):
            callback = getattr(runtime, callback_name, None)
            if callable(callback):
                close_callback = cast(Callable[[], object], callback)
                try:
                    close_callback()  # pylint: disable=not-callable  # AUDIT-FIX(#6): getattr + callable guard makes this runtime-safe.
                except Exception:
                    self.log_event("failed_to_close_managed_integrations", severity="warning")
                return

    @property
    def spoken_count(self) -> int:
        return self._spoken_count

    def _search_conversation_context(self):
        runtime = getattr(self.owner, "runtime", None)
        callback = None if runtime is None else getattr(runtime, "search_provider_conversation_context", None)
        if callable(callback):
            try:
                value = callback()
            except Exception:
                self.log_event("failed_to_build_search_conversation_context", severity="warning")
                return ()
            return value if value is not None else ()
        return ()

    def _timezone_name(self) -> str:
        config = getattr(self.owner, "config", None)
        timezone_name = _DEFAULT_TIMEZONE_NAME if config is None else str(getattr(config, "local_timezone_name", "") or "").strip()
        return timezone_name or _DEFAULT_TIMEZONE_NAME

    def _managed_integrations(self) -> Any:
        if self._managed_integrations_runtime is not None:
            return self._managed_integrations_runtime
        with self._managed_integrations_lock:
            if self._managed_integrations_runtime is not None:
                return self._managed_integrations_runtime
            from twinr.integrations.runtime import build_managed_integrations

            config = getattr(self.owner, "config", None)
            project_root_raw = "." if config is None else str(getattr(config, "project_root", ".") or ".").strip() or "."
            env_path_raw = None
            if config is not None:
                env_path_raw = getattr(config, "env_path", None)
            if env_path_raw is None:
                env_path_raw = getattr(self.owner, "env_path", None)
            try:
                project_root = str(Path(project_root_raw).expanduser())
                env_path = None if env_path_raw is None else str(Path(str(env_path_raw)).expanduser())
                self._managed_integrations_runtime = build_managed_integrations(project_root, env_path=env_path)
            except Exception as exc:  # AUDIT-FIX(#6): Integration bootstrap failures must not leak backend-specific details.
                raise RuntimeError("managed integrations are unavailable for self-coding skill execution") from exc
            return self._managed_integrations_runtime

    def _calendar_range(
        self,
        *,
        days: int,
        start_iso: str | None,
        end_iso: str | None,
    ) -> tuple[datetime, datetime]:
        if start_iso:
            start_at = self._parse_datetime_input(start_iso, field_name="start_iso")
        else:
            start_at = self.now
        if end_iso:
            end_at = self._parse_datetime_input(end_iso, field_name="end_iso")
        else:
            end_at = start_at + self._calendar_delta(days=days)
        if end_at <= start_at:
            raise ValueError("calendar range end must be after start")
        return start_at, end_at

    @staticmethod
    def _calendar_delta(*, days: int):
        bounded_days = _coerce_positive_int(days, field_name="days", maximum=_MAX_CALENDAR_DAYS)  # AUDIT-FIX(#5): Keep calendar windows small and deterministic on RPi-class hardware.
        return timedelta(days=bounded_days)

    @staticmethod
    def _state_key(value: object) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("skill state key must not be empty")
        return text

    @staticmethod
    def _optional_text(value: object | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _bounded_text(value: object, *, field_name: str, max_chars: int) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"{field_name} must not be empty")
        if len(text) > max_chars:
            raise ValueError(f"{field_name} must be at most {max_chars} characters")
        return text

    @staticmethod
    def _json_value(value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("skill runtime values must be JSON-serializable finite values") from exc

    @classmethod
    def _merge_json_value(cls, current: Any, patch: Any) -> Any:
        current_json = None if current is None else cls._json_value(current)
        patch_json = cls._json_value(patch)
        if isinstance(current_json, dict) and isinstance(patch_json, dict):
            merged = dict(current_json)
            for key, value in patch_json.items():
                merged[str(key)] = value
            return merged
        if isinstance(current_json, list) and isinstance(patch_json, list):
            return list((*current_json, *patch_json))
        return patch_json

    def _load_initial_state(self) -> dict[str, Any]:
        try:
            payload = self.runtime_store.load_state(skill_id=self.skill_id, version=self.version)
        except FileNotFoundError:
            return {}
        except Exception:
            self.log_event("failed_to_load_skill_state", severity="warning")
            return {}
        if not isinstance(payload, dict):
            self.log_event("invalid_skill_state_payload_type", severity="warning", payload_type=type(payload).__name__)
            return {}
        normalized: dict[str, Any] = {}
        dropped_entries = 0
        for raw_key, raw_value in payload.items():
            key_text = str(raw_key or "").strip()
            if not key_text:
                dropped_entries += 1
                continue
            try:
                normalized[key_text] = self._json_value(raw_value)
            except ValueError:
                dropped_entries += 1
        if dropped_entries:
            self.log_event("dropped_invalid_skill_state_entries", severity="warning", dropped_entries=dropped_entries)
        return normalized

    def _parse_datetime_input(self, value: str, *, field_name: str) -> datetime:
        text = self._bounded_text(value, field_name=field_name, max_chars=64)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a valid ISO-8601 datetime") from exc
        return _normalize_datetime(parsed, assume_tz=self._timezone, field_name=field_name)  # AUDIT-FIX(#3): Naive local calendar datetimes are interpreted in configured local time instead of UTC.


class SelfCodingSkillExecutionService:
    """Load compiled skill packages, materialize them, and execute handlers in a child sandbox."""

    def __init__(
        self,
        *,
        store: SelfCodingStore,
        runtime_store: SelfCodingSkillRuntimeStore | None = None,
        health_service: "SelfCodingHealthService | None" = None,
        sandbox_runner: SelfCodingSandboxRunner | None = None,
        execution_run_service: SelfCodingExecutionRunService | None = None,
    ) -> None:
        self.store = store
        self.runtime_store = runtime_store if runtime_store is not None else SelfCodingSkillRuntimeStore(store.root)
        self.health_service = health_service
        self.sandbox_runner = sandbox_runner if sandbox_runner is not None else SelfCodingSandboxRunner()
        self.execution_run_service = (
            execution_run_service if execution_run_service is not None else SelfCodingExecutionRunService(store=store)
        )

    def execute_scheduled(
        self,
        owner: Any,
        *,
        skill_id: str,
        version: int,
        trigger_id: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        normalized_skill_id = _validate_identifier(skill_id, field_name="skill_id")  # AUDIT-FIX(#8): Validate identifiers before they reach file-backed stores.
        normalized_version = _normalize_version(version)
        normalized_trigger_id = _validate_identifier(trigger_id, field_name="trigger_id")
        effective_now = self._effective_now(owner, now=now)
        run_record = self._safe_start_run(  # AUDIT-FIX(#2): Run tracking is best-effort; it must not block the actual skill execution.
            owner,
            run_kind="scheduled_trigger",
            skill_id=normalized_skill_id,
            version=normalized_version,
            timeout_seconds=float(self.sandbox_runner.limits.timeout_seconds),
            metadata={"trigger_family": "scheduled", "trigger_id": normalized_trigger_id},
        )
        context: SkillContext | None = None
        try:
            loaded_package = self._load_active_skill_package(skill_id=normalized_skill_id, version=normalized_version)
            trigger = loaded_package.document.package.scheduled_trigger(normalized_trigger_id)
            context = SkillContext(
                owner=owner,
                runtime_store=self.runtime_store,
                skill_id=normalized_skill_id,
                version=normalized_version,
                skill_name=loaded_package.skill_name,
                now=effective_now,
            )
            sandbox_result = self.sandbox_runner.run_handler(
                owner=owner,
                context=context,
                materialized_root=loaded_package.materialized_root,
                entry_module=loaded_package.document.package.entry_module,
                handler_name=trigger.handler,
                policy=SkillBrokerPolicy.from_manifest(loaded_package.document.policy_manifest),
            )
            context.flush()
            result = {"status": "ok", "trigger_id": trigger.trigger_id, "delivered": context.spoken_count > 0}
            self._safe_finish_run(  # AUDIT-FIX(#7): Telemetry completion failures must not turn a successful handler into a failed one.
                owner,
                run_record,
                status="completed",
                metadata={
                    "trigger_family": "scheduled",
                    "trigger_id": normalized_trigger_id,
                    "delivered": bool(result["delivered"]),
                    "spoken_count": int(context.spoken_count),
                    "timeout_seconds": float(self.sandbox_runner.limits.timeout_seconds),
                    "hardening": dict(getattr(sandbox_result, "hardening", {}) or {}),
                    "child_pid": getattr(sandbox_result, "child_pid", None),
                    "policy_manifest": loaded_package.document.policy_manifest.to_payload(),
                },
            )
        except Exception as exc:
            self._safe_finish_run(  # AUDIT-FIX(#7): Preserve the original execution failure even if run-finalization fails.
                owner,
                run_record,
                status="timed_out" if isinstance(exc, SelfCodingSandboxTimeoutError) else "failed",
                reason=_safe_exception_reason(exc),
                metadata={
                    "trigger_family": "scheduled",
                    "trigger_id": normalized_trigger_id,
                    **_exception_metadata(exc),
                },
            )
            self._record_health_failure(
                owner=owner,
                skill_id=normalized_skill_id,
                version=normalized_version,
                triggered_at=effective_now,
                error=exc,
                metadata={"trigger_family": "scheduled", "trigger_id": normalized_trigger_id},
            )
            raise
        finally:
            if context is not None:
                context.close()  # AUDIT-FIX(#6): Best-effort runtime cleanup belongs in finally so it also runs on sandbox errors/timeouts.
        self._record_health_success(
            owner=owner,
            skill_id=normalized_skill_id,
            version=normalized_version,
            delivered=bool(result["delivered"]),
            triggered_at=effective_now,
            metadata={"trigger_family": "scheduled", "trigger_id": normalized_trigger_id},
        )
        return result

    def execute_sensor_event(
        self,
        owner: Any,
        *,
        skill_id: str,
        version: int,
        trigger_id: str,
        event_name: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        normalized_skill_id = _validate_identifier(skill_id, field_name="skill_id")  # AUDIT-FIX(#8): Validate identifiers before file/log usage.
        normalized_version = _normalize_version(version)
        normalized_trigger_id = _validate_identifier(trigger_id, field_name="trigger_id")
        effective_now = self._effective_now(owner, now=now)
        run_record = self._safe_start_run(  # AUDIT-FIX(#2): Telemetry start is non-critical and must not suppress sensor-trigger handling.
            owner,
            run_kind="sensor_trigger",
            skill_id=normalized_skill_id,
            version=normalized_version,
            timeout_seconds=float(self.sandbox_runner.limits.timeout_seconds),
            metadata={
                "trigger_family": "sensor",
                "trigger_id": normalized_trigger_id,
                "event_name": event_name,
            },
        )
        context: SkillContext | None = None
        try:
            loaded_package = self._load_active_skill_package(skill_id=normalized_skill_id, version=normalized_version)
            trigger = loaded_package.document.package.sensor_trigger(normalized_trigger_id)
            context = SkillContext(
                owner=owner,
                runtime_store=self.runtime_store,
                skill_id=normalized_skill_id,
                version=normalized_version,
                skill_name=loaded_package.skill_name,
                now=effective_now,
            )
            sandbox_result = self.sandbox_runner.run_handler(
                owner=owner,
                context=context,
                materialized_root=loaded_package.materialized_root,
                entry_module=loaded_package.document.package.entry_module,
                handler_name=trigger.handler,
                policy=SkillBrokerPolicy.from_manifest(loaded_package.document.policy_manifest),
                event_name=event_name,
            )
            context.flush()
            result = {
                "status": "ok",
                "trigger_id": trigger.trigger_id,
                "event_name": event_name,
                "delivered": context.spoken_count > 0,
            }
            self._safe_finish_run(  # AUDIT-FIX(#7): Run-finalization must never mask a successful sensor-trigger execution.
                owner,
                run_record,
                status="completed",
                metadata={
                    "trigger_family": "sensor",
                    "trigger_id": normalized_trigger_id,
                    "event_name": event_name,
                    "delivered": bool(result["delivered"]),
                    "spoken_count": int(context.spoken_count),
                    "timeout_seconds": float(self.sandbox_runner.limits.timeout_seconds),
                    "hardening": dict(getattr(sandbox_result, "hardening", {}) or {}),
                    "child_pid": getattr(sandbox_result, "child_pid", None),
                    "policy_manifest": loaded_package.document.policy_manifest.to_payload(),
                },
            )
        except Exception as exc:
            self._safe_finish_run(  # AUDIT-FIX(#7): Preserve the real handler exception even if telemetry serialization/finalization fails.
                owner,
                run_record,
                status="timed_out" if isinstance(exc, SelfCodingSandboxTimeoutError) else "failed",
                reason=_safe_exception_reason(exc),
                metadata={
                    "trigger_family": "sensor",
                    "trigger_id": normalized_trigger_id,
                    "event_name": event_name,
                    **_exception_metadata(exc),
                },
            )
            self._record_health_failure(
                owner=owner,
                skill_id=normalized_skill_id,
                version=normalized_version,
                triggered_at=effective_now,
                error=exc,
                metadata={
                    "trigger_family": "sensor",
                    "trigger_id": normalized_trigger_id,
                    "event_name": event_name,
                },
            )
            raise
        finally:
            if context is not None:
                context.close()  # AUDIT-FIX(#6): Always release integration resources after sensor-trigger execution.
        self._record_health_success(
            owner=owner,
            skill_id=normalized_skill_id,
            version=normalized_version,
            delivered=bool(result["delivered"]),
            triggered_at=effective_now,
            metadata={
                "trigger_family": "sensor",
                "trigger_id": normalized_trigger_id,
                "event_name": event_name,
            },
        )
        return result

    def _load_active_skill_package(self, *, skill_id: str, version: int) -> _LoadedSkillPackage:
        normalized_skill_id = _validate_identifier(skill_id, field_name="skill_id")  # AUDIT-FIX(#8): Reject invalid path/log identifiers before store access.
        normalized_version = _normalize_version(version)
        activation = self.store.load_activation(normalized_skill_id, version=normalized_version)
        if activation.status != LearnedSkillStatus.ACTIVE:
            raise RuntimeError(f"self-coding skill {normalized_skill_id!r} version {normalized_version} is not active")
        artifact_id = _validate_identifier(activation.artifact_id, field_name="artifact_id")  # AUDIT-FIX(#8): Local validation narrows traversal/filename abuse even if downstream stores are lax.
        artifact_text = self.store.read_text_artifact(artifact_id)
        fallback_capabilities: tuple[str, ...] = ()
        raw_job_id = str(activation.job_id or "").strip()
        if raw_job_id:
            job_id = _validate_identifier(raw_job_id, field_name="job_id")  # AUDIT-FIX(#8): Validate secondary store keys used for job lookups.
            try:
                required_capabilities = self.store.load_job(job_id).required_capabilities
                fallback_capabilities = tuple(str(item).strip() for item in required_capabilities if str(item).strip())
            except FileNotFoundError:
                fallback_capabilities = ()
        document = skill_package_document_from_document(
            artifact_text,
            fallback_capabilities=fallback_capabilities,
        )
        materialized_root = self.runtime_store.materialize_package(
            skill_id=normalized_skill_id,
            version=normalized_version,
            package=document.package,
        )
        return _LoadedSkillPackage(
            document=document,
            materialized_root=materialized_root,
            skill_name=str(getattr(activation, "skill_name", "") or "").strip() or normalized_skill_id,
        )

    @staticmethod
    def _effective_now(owner: Any, *, now: datetime | None) -> datetime:
        config = getattr(owner, "config", None)
        timezone_name = (
            _DEFAULT_TIMEZONE_NAME
            if config is None
            else str(getattr(config, "local_timezone_name", "") or "").strip() or _DEFAULT_TIMEZONE_NAME
        )
        timezone = _resolve_timezone(timezone_name, owner=owner, context="execution_now")  # AUDIT-FIX(#3): Bad timezone config must not abort trigger execution.
        if now is not None:
            return _normalize_datetime(now, assume_tz=timezone, field_name="now")  # AUDIT-FIX(#3): Interpret naive trigger times in configured local time for human-facing scheduling.
        return datetime.now(timezone)

    def _safe_start_run(
        self,
        owner: Any,
        *,
        run_kind: str,
        skill_id: str,
        version: int,
        timeout_seconds: float,
        metadata: dict[str, Any],
    ) -> Any | None:
        try:
            return self.execution_run_service.start_run(
                run_kind=run_kind,
                skill_id=skill_id,
                version=version,
                timeout_seconds=timeout_seconds,
                metadata=_json_metadata(metadata),  # AUDIT-FIX(#7): Normalize metadata before it hits file-backed run tracking.
            )
        except Exception:
            _emit_owner_event(owner, "warning", "self_coding_execution_run_start_failed", run_kind=run_kind, skill_id=skill_id, version=version)
            return None

    def _safe_finish_run(
        self,
        owner: Any,
        run_record: Any | None,
        *,
        status: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if run_record is None:
            return
        kwargs: dict[str, Any] = {
            "status": status,
            "metadata": _json_metadata(metadata),  # AUDIT-FIX(#7): Telemetry payloads must be JSON-safe and non-fatal.
        }
        if reason is not None:
            kwargs["reason"] = reason
        try:
            self.execution_run_service.finish_run(run_record, **kwargs)
        except Exception:
            _emit_owner_event(owner, "warning", "self_coding_execution_run_finish_failed", status=status)

    def _record_health_success(
        self,
        *,
        owner: Any,
        skill_id: str,
        version: int,
        delivered: bool,
        triggered_at: datetime,
        metadata: dict[str, Any],
    ) -> None:
        if self.health_service is None:
            return
        try:
            self.health_service.record_success(
                skill_id=skill_id,
                version=version,
                delivered=delivered,
                triggered_at=triggered_at,
                metadata=_json_metadata(metadata),  # AUDIT-FIX(#7): Health metadata must not crash success accounting.
            )
        except Exception:
            _emit_owner_event(owner, "warning", "self_coding_health_record_success_failed", skill_id=skill_id, version=version)

    def _record_health_failure(
        self,
        *,
        owner: Any,
        skill_id: str,
        version: int,
        triggered_at: datetime,
        error: Exception | str,
        metadata: dict[str, Any],
    ) -> None:
        if self.health_service is None:
            return
        try:
            self.health_service.record_failure(
                skill_id=skill_id,
                version=version,
                error=error,
                triggered_at=triggered_at,
                metadata=_json_metadata(metadata),  # AUDIT-FIX(#7): Failure telemetry must not replace the original execution error.
            )
        except Exception:
            _emit_owner_event(owner, "warning", "self_coding_health_record_failure_failed", skill_id=skill_id, version=version)


def _normalize_severity(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if text in _ALLOWED_EVENT_SEVERITIES else "info"


def _emit_owner_event(owner: Any, severity: str, event: str, **data: object) -> None:
    callback = getattr(owner, "_safe_record_event", None)
    safe_event = _safe_control_text(event, fallback="self_coding_event", max_chars=120)  # AUDIT-FIX(#9): Strip control characters from log/event names to avoid log forging.
    safe_data = _json_metadata(data)
    if callable(callback):
        try:
            callback(f"self_coding_skill_{_normalize_severity(severity)}", safe_event, **safe_data)  # AUDIT-FIX(#9): Logging is best-effort and metadata is sanitized before emission.
            return
        except Exception:
            logger.warning("self-coding owner event callback failed", exc_info=True)
    logger.log(_LOGGER_LEVEL_BY_SEVERITY.get(_normalize_severity(severity), logging.INFO), "self-coding event %s: %s", safe_event, safe_data)


def _safe_control_text(value: object, *, fallback: str, max_chars: int) -> str:
    raw = str(value or "").strip()
    if not raw:
        return fallback
    sanitized = "".join(char if ord(char) >= 32 else " " for char in raw)
    return sanitized[:max_chars]


def _validate_identifier(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    if "\x00" in text:
        raise ValueError(f"{field_name} must not contain NUL bytes")
    if any(ord(char) < 32 for char in text):
        raise ValueError(f"{field_name} must not contain control characters")
    if "/" in text or "\\" in text:
        raise ValueError(f"{field_name} must not contain path separators")
    if text in {".", ".."}:
        raise ValueError(f"{field_name} must not contain path traversal segments")
    return text


def _normalize_version(value: object) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("skill version must be an integer") from exc
    if normalized < 0:
        raise ValueError("skill version must be >= 0")
    return normalized


def _state_lock_for(skill_id: str, version: int) -> RLock:
    key = (skill_id, version)
    with _STATE_LOCKS_GUARD:
        lock = _STATE_LOCKS.get(key)
        if lock is None:
            lock = RLock()
            _STATE_LOCKS[key] = lock
        return lock


def _coerce_positive_int(value: object, *, field_name: str, maximum: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if normalized < 1:
        raise ValueError(f"{field_name} must be >= 1")
    if normalized > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}")
    return normalized


def _resolve_timezone(timezone_name: str, *, owner: Any | None = None, context: str) -> ZoneInfo:
    candidate = str(timezone_name or "").strip() or _DEFAULT_TIMEZONE_NAME
    try:
        return ZoneInfo(candidate)
    except ZoneInfoNotFoundError:
        _emit_owner_event(
            owner,
            "warning",
            "self_coding_invalid_timezone",
            configured_timezone=candidate,
            context=context,
            fallback_timezone=_DEFAULT_TIMEZONE_NAME,
        )  # AUDIT-FIX(#3): Invalid configured timezones are downgraded to a safe default instead of aborting execution.
    try:
        return ZoneInfo(_DEFAULT_TIMEZONE_NAME)
    except ZoneInfoNotFoundError:
        return ZoneInfo(_FALLBACK_TIMEZONE_NAME)


def _normalize_datetime(value: datetime, *, assume_tz: ZoneInfo, field_name: str = "datetime") -> datetime:
    if value.tzinfo is not None:
        return value
    candidate_fold_0 = value.replace(tzinfo=assume_tz, fold=0)
    candidate_fold_1 = value.replace(tzinfo=assume_tz, fold=1)
    roundtrip_fold_0 = candidate_fold_0.astimezone(UTC).astimezone(assume_tz).replace(tzinfo=None)
    roundtrip_fold_1 = candidate_fold_1.astimezone(UTC).astimezone(assume_tz).replace(tzinfo=None)
    if roundtrip_fold_0 != value and roundtrip_fold_1 != value:
        raise ValueError(f"{field_name} falls into a nonexistent local time in timezone {assume_tz.key}")
    if roundtrip_fold_0 == value and roundtrip_fold_1 == value and candidate_fold_0.utcoffset() != candidate_fold_1.utcoffset():
        return value.replace(tzinfo=assume_tz, fold=getattr(value, "fold", 0))  # AUDIT-FIX(#3): Preserve DST-fold information for ambiguous local times.
    return candidate_fold_0 if roundtrip_fold_0 == value else candidate_fold_1


def _json_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    sanitized = _json_sanitize(metadata or {})
    return sanitized if isinstance(sanitized, dict) else {"metadata_serialization_error": True}


def _json_sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}  # AUDIT-FIX(#7): Preserve serializable metadata fields while coercing unsupported values.
    if isinstance(value, (list, tuple, set)):
        return [_json_sanitize(item) for item in value]
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return f"<{type(value).__name__}>"  # AUDIT-FIX(#7): Unsupported metadata objects are redacted to their type, not stringified verbatim.
    return value


def _safe_exception_reason(exc: Exception) -> str:
    return _safe_control_text(str(exc).strip() or type(exc).__name__, fallback=type(exc).__name__, max_chars=512)


def _exception_metadata(exc: Exception) -> dict[str, Any]:
    metadata = getattr(exc, "metadata", {})
    return _json_metadata(metadata) if isinstance(metadata, dict) else {}


__all__ = [
    "SelfCodingSkillExecutionService",
    "SkillContext",
    "SkillSearchResult",
]
