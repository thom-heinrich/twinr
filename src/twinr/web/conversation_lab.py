"""Run the portal conversation lab against Twinr's real text/tool path.

This module owns the file-backed session store plus the synchronous execution
path for one operator text turn. It intentionally stays web-specific: sessions
persist template-facing, human-readable trace snapshots so the portal can load
previous runs without replaying provider calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from inspect import signature
from pathlib import Path
import json
import secrets
import threading

from twinr.agent.base_agent.prompting.personality import load_supervisor_loop_instructions
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime.runtime import TwinrRuntime
from twinr.agent.tools import (
    DualLaneToolLoop,
    RealtimeToolExecutor,
    ToolCallingStreamingLoop,
    build_agent_tool_schemas,
    build_compact_agent_tool_schemas,
    build_specialist_tool_agent_instructions,
    build_supervisor_decision_instructions,
    build_tool_agent_instructions,
    bind_realtime_tool_handlers,
)
from twinr.agent.tools.runtime.availability import available_realtime_tool_names
from twinr.memory.longterm.retrieval.operator_search import run_long_term_operator_search
from twinr.ops import TwinrUsageStore
from twinr.ops.paths import TwinrOpsPaths
from twinr.providers.factory import build_streaming_provider_bundle
from twinr.providers.openai import OpenAIBackend, OpenAISupervisorDecisionProvider, OpenAIToolCallingAgentProvider
from twinr.web.conversation_lab_vision import build_vision_images, build_vision_prompt, owner_camera
from twinr.web.presenters.memory_search import build_memory_search_panel_context
from twinr.web.support.store import read_text_file, write_text_file


_CONVERSATION_LAB_TOOL_NAMES: tuple[str, ...] = (
    "search_live_info",
    "browser_automation",
    "schedule_reminder",
    "list_automations",
    "create_time_automation",
    "create_sensor_automation",
    "update_time_automation",
    "update_sensor_automation",
    "delete_automation",
    "remember_memory",
    "remember_contact",
    "lookup_contact",
    "get_memory_conflicts",
    "resolve_memory_conflict",
    "remember_preference",
    "remember_plan",
    "update_user_profile",
    "update_personality",
    "configure_world_intelligence",
    "update_simple_setting",
    "manage_voice_quiet_mode",
    "list_smart_home_entities",
    "read_smart_home_state",
    "control_smart_home_entities",
    "read_smart_home_sensor_stream",
    "enroll_portrait_identity",
    "get_portrait_identity_status",
    "reset_portrait_identity",
    "manage_household_identity",
    "inspect_camera",
    "end_conversation",
)
_MAX_SESSION_LIST = 12
_MAX_SESSION_TITLE_CHARS = 72
_MAX_TURN_PREVIEW_CHARS = 220
_MAX_DETAIL_VALUE_CHARS = 220
_CONVERSATION_LAB_SOURCE = "web_conversation_lab"
_CONVERSATION_LAB_REQUEST_KIND = "conversation_lab"
_CONVERSATION_LAB_FLUSH_TIMEOUT_MAX_S = 5.0
_CONVERSATION_LAB_SEARCH_TIMEOUT_S = 3.0
_TRUNCATE_SUFFIX = "..."


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{secrets.token_hex(4)}"


def _lab_root(ops_paths: TwinrOpsPaths) -> Path:
    return ops_paths.ops_store_root / "conversation_lab"


def _session_path(ops_paths: TwinrOpsPaths, session_id: str) -> Path:
    return _lab_root(ops_paths) / f"{session_id}.json"


def _ensure_session_id(value: object | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if not all(character.isalnum() or character in {"-", "_"} for character in text):
        return None
    return text


def _truncate_text(value: object, *, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(_TRUNCATE_SUFFIX))].rstrip() + _TRUNCATE_SUFFIX


def _humanize_name(value: object) -> str:
    text = str(value or "").strip().replace("_", " ").replace("-", " ")
    return " ".join(part.capitalize() for part in text.split()) or "Unknown"


def _status_class(value: object | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"ok", "ready", "active", "updated", "created", "stored", "printed", "true"}:
        return "ok"
    if normalized in {"warn", "warning", "unchanged", "needs_clarification", "candidate"}:
        return "warn"
    if normalized in {"error", "fail", "failed", "invalid", "rejected", "false"}:
        return "fail"
    return "muted"


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except Exception:
            return str(value)
    return str(value)


def _compact_lines(mapping: Mapping[str, object], *, max_items: int = 8) -> tuple[str, ...]:
    lines: list[str] = []
    for index, (key, value) in enumerate(mapping.items()):
        if index >= max_items:
            lines.append(f"+{len(mapping) - max_items} more fields")
            break
        if isinstance(value, Mapping):
            nested = ", ".join(
                f"{nested_key}={_truncate_text(nested_value, limit=48)}"
                for nested_key, nested_value in list(value.items())[:4]
            )
            if nested:
                lines.append(f"{key}: {nested}")
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            preview = " | ".join(_truncate_text(item, limit=48) for item in list(value)[:4])
            if preview:
                lines.append(f"{key}: {preview}")
            continue
        rendered = _truncate_text(value, limit=_MAX_DETAIL_VALUE_CHARS)
        if rendered:
            lines.append(f"{key}: {rendered}")
    return tuple(lines)


def _detail_item(
    label: str,
    value: object,
    *,
    detail: str | None = None,
    status: str = "muted",
    copy: bool = False,
    wide: bool = False,
) -> dict[str, object]:
    return {
        "label": label,
        "value": str(value),
        "detail": detail,
        "status": status,
        "copy": copy,
        "wide": wide,
    }


def _turn_summary_rows(
    *,
    result: object | None,
    response_text: str,
) -> tuple[dict[str, object], ...]:
    if result is None:
        return ()
    rows = [
        _detail_item("Response chars", len(response_text)),
        _detail_item("Rounds", getattr(result, "rounds", 0)),
        _detail_item("Model", getattr(result, "model", None) or "—", copy=True),
        _detail_item(
            "Web search",
            "yes" if bool(getattr(result, "used_web_search", False)) else "no",
        ),
        _detail_item(
            "Response ID",
            getattr(result, "response_id", None) or "—",
            copy=True,
            wide=True,
        ),
        _detail_item(
            "Request ID",
            getattr(result, "request_id", None) or "—",
            copy=True,
            wide=True,
        ),
    ]
    token_usage = _token_usage_rows(getattr(result, "token_usage", None))
    return tuple(rows + list(token_usage))


def _token_usage_rows(token_usage: object | None) -> tuple[dict[str, object], ...]:
    if token_usage is None:
        return ()
    source = _json_safe(token_usage)
    if not isinstance(source, Mapping):
        return (_detail_item("Token usage", _truncate_text(source, limit=80), wide=True),)
    rows: list[dict[str, object]] = []
    for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens"):
        value = source.get(key)
        if value not in {None, ""}:
            rows.append(_detail_item(_humanize_name(key), value))
    return tuple(rows)


def _writer_state_snapshot(runtime: TwinrRuntime) -> dict[str, object]:
    writer = getattr(getattr(runtime, "long_term_memory", None), "writer", None)
    if writer is None or not hasattr(writer, "snapshot_state"):
        return {
            "pending_count": 0,
            "inflight_count": 0,
            "dropped_count": 0,
            "last_error_message": None,
            "accepting": False,
            "worker_alive": False,
        }
    state = writer.snapshot_state()
    return dict(_json_safe(state)) if isinstance(_json_safe(state), dict) else {}


def _has_background_long_term_writers(runtime: TwinrRuntime) -> bool:
    service = getattr(runtime, "long_term_memory", None)
    if service is None:
        return False
    return any(
        getattr(service, attribute_name, None) is not None
        for attribute_name in ("writer", "multimodal_writer")
    )


def _memory_rows(
    *,
    before_enqueue: dict[str, object],
    after_enqueue: dict[str, object] | None,
    after_flush: dict[str, object] | None,
    flush_ok: bool | None,
) -> tuple[dict[str, object], ...]:
    rows = [
        _detail_item("Writer pending before", before_enqueue.get("pending_count", 0)),
        _detail_item("Writer inflight before", before_enqueue.get("inflight_count", 0)),
    ]
    if after_enqueue is not None:
        rows.extend(
            [
                _detail_item("Writer pending after enqueue", after_enqueue.get("pending_count", 0)),
                _detail_item("Writer inflight after enqueue", after_enqueue.get("inflight_count", 0)),
            ]
        )
    if flush_ok is not None:
        rows.append(
            _detail_item(
                "Flush result",
                "ok" if flush_ok else "failed",
                status="ok" if flush_ok else "fail",
            )
        )
    if after_flush is not None:
        rows.extend(
            [
                _detail_item("Writer pending after flush", after_flush.get("pending_count", 0)),
                _detail_item("Writer dropped", after_flush.get("dropped_count", 0)),
                _detail_item(
                    "Writer error",
                    after_flush.get("last_error_message") or "—",
                    status="fail" if after_flush.get("last_error_message") else "muted",
                    wide=True,
                    copy=True,
                ),
            ]
        )
    return tuple(rows)


def _tool_items(result: object | None) -> tuple[dict[str, object], ...]:
    if result is None:
        return ()
    tool_calls = tuple(getattr(result, "tool_calls", ()) or ())
    tool_results = {
        str(getattr(item, "call_id", "")): item
        for item in tuple(getattr(result, "tool_results", ()) or ())
    }
    items: list[dict[str, object]] = []
    for call in tool_calls:
        call_id = str(getattr(call, "call_id", "") or "")
        arguments = getattr(call, "arguments", {}) or {}
        tool_result = tool_results.get(call_id)
        output = getattr(tool_result, "output", {}) if tool_result is not None else {}
        output_mapping = output if isinstance(output, Mapping) else {"output": output}
        status_value = output_mapping.get("status") or output_mapping.get("error") or "completed"
        title = _humanize_name(getattr(call, "name", "tool"))
        body = _tool_body(output_mapping)
        meta_lines = list(_compact_lines(arguments if isinstance(arguments, Mapping) else {}, max_items=6))
        meta_lines.extend(_compact_lines(output_mapping, max_items=8))
        items.append(
            {
                "badge": getattr(call, "name", "tool"),
                "level": _status_class(status_value),
                "title": title,
                "time": None,
                "body": body,
                "meta_lines": tuple(meta_lines),
            }
        )
    return tuple(items)


def _route_items(result: object | None, collected_items: Sequence[dict[str, object]]) -> tuple[dict[str, object], ...]:
    items: list[dict[str, object]] = []
    if result is not None:
        rounds = int(getattr(result, "rounds", 0) or 0)
        items.append(
            {
                "badge": "route",
                "level": "ok",
                "title": "Text turn completed",
                "time": None,
                "body": f"Twinr completed this portal turn in {rounds} model round(s).",
                "meta_lines": tuple(
                    line
                    for line in (
                        f"model: {getattr(result, 'model', None) or '—'}",
                        f"used_web_search: {bool(getattr(result, 'used_web_search', False))}",
                        f"response_id: {getattr(result, 'response_id', None) or '—'}",
                        f"request_id: {getattr(result, 'request_id', None) or '—'}",
                    )
                    if line
                ),
            }
        )
    items.extend(_collector_section_items("route", collected_items))
    return tuple(items)


def _tool_body(output: Mapping[str, object]) -> str:
    for key in ("answer", "spoken_answer", "summary", "message", "text", "label"):
        value = output.get(key)
        rendered = _truncate_text(value, limit=_MAX_TURN_PREVIEW_CHARS)
        if rendered:
            return rendered
    status_value = _truncate_text(output.get("status") or "Tool finished.", limit=80)
    return status_value or "Tool finished."


def _collector_section_items(section_name: str, items: Sequence[dict[str, object]]) -> tuple[dict[str, object], ...]:
    rendered: list[dict[str, object]] = []
    for item in items:
        body = _truncate_text(item.get("body"), limit=_MAX_TURN_PREVIEW_CHARS)
        detail_lines = _compact_lines(item.get("details", {}) if isinstance(item.get("details"), Mapping) else {}, max_items=8)
        rendered.append(
            {
                "badge": item.get("badge", section_name),
                "level": _status_class(item.get("status")),
                "title": str(item.get("title", "Trace item")),
                "time": item.get("created_at"),
                "body": body or "No extra summary available.",
                "meta_lines": detail_lines,
            }
        )
    return tuple(rendered)


def _search_snapshot(config: TwinrConfig, query_text: str) -> dict[str, object]:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="conversation_lab_search")
    future = executor.submit(run_long_term_operator_search, config, query_text)
    try:
        result = future.result(timeout=_CONVERSATION_LAB_SEARCH_TIMEOUT_S)
        return build_memory_search_panel_context(
            query_text=query_text,
            result=result,
            error_message=None,
        )
    except FutureTimeoutError:
        future.cancel()
        return build_memory_search_panel_context(
            query_text=query_text,
            result=None,
            error_message=(
                f"TimeoutError: operator memory search exceeded "
                f"{_CONVERSATION_LAB_SEARCH_TIMEOUT_S:.1f}s."
            ),
        )
    except Exception as exc:
        return build_memory_search_panel_context(
            query_text=query_text,
            result=None,
            error_message=f"{type(exc).__name__}: {_truncate_text(exc, limit=120)}",
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _close_if_possible(component: object | None) -> None:
    close = getattr(component, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            return


def _seed_runtime_conversation(runtime: TwinrRuntime, session: Mapping[str, object]) -> None:
    for turn in tuple(session.get("turns", ()) or ()):
        if not isinstance(turn, Mapping):
            continue
        prompt = str(turn.get("prompt", "") or "").strip()
        response = str(turn.get("response", "") or "").strip()
        if prompt:
            runtime.memory.remember("user", prompt)
        if response:
            runtime.memory.remember("assistant", response)


def _result_text(result: object) -> str:
    text = str(getattr(result, "text", "") or "").strip()
    if not text:
        raise RuntimeError("The provider returned no final answer text.")
    return text


def _turn_status_badge(status: str) -> dict[str, str]:
    label = {
        "ok": "Completed",
        "error": "Failed",
    }.get(status, _humanize_name(status))
    return {"label": label, "status": _status_class(status)}


class _TurnCollector:
    """Collect human-readable trace artifacts for one portal turn."""

    def __init__(self) -> None:
        self.route_items: list[dict[str, object]] = []
        self.tool_events: list[dict[str, object]] = []
        self.telemetry_items: list[dict[str, object]] = []

    def trace_event(
        self,
        name: str,
        *,
        kind: str,
        details: dict[str, object] | None = None,
        level: str | None = None,
        kpi: dict[str, object] | None = None,
    ) -> None:
        merged_details = dict(details or {})
        if kpi:
            merged_details["kpi"] = dict(kpi)
        self.route_items.append(
            {
                "badge": kind,
                "status": str(level or kind or "info").lower(),
                "title": _humanize_name(name),
                "body": _trace_body(kind=kind, details=merged_details),
                "details": merged_details,
                "created_at": None,
            }
        )

    def trace_decision(
        self,
        name: str,
        *,
        question: str,
        selected: dict[str, object],
        options: Sequence[dict[str, object]],
        context: dict[str, object] | None = None,
        guardrails: Sequence[str] | None = None,
    ) -> None:
        merged_details = {
            "question": question,
            "selected": dict(selected),
            "options": [dict(option) for option in options],
        }
        if context:
            merged_details["context"] = dict(context)
        if guardrails:
            merged_details["guardrails"] = list(guardrails)
        self.route_items.append(
            {
                "badge": "decision",
                "status": "ok",
                "title": _humanize_name(name),
                "body": _truncate_text(selected.get("summary") or question, limit=_MAX_TURN_PREVIEW_CHARS),
                "details": merged_details,
                "created_at": None,
            }
        )

    def emit(self, payload: str) -> None:
        key, separator, value = str(payload or "").partition("=")
        if separator:
            title = _humanize_name(key)
            body = _truncate_text(value, limit=_MAX_TURN_PREVIEW_CHARS)
            details = {"key": key, "value": value}
        else:
            title = "Signal"
            body = _truncate_text(payload, limit=_MAX_TURN_PREVIEW_CHARS)
            details = {"payload": payload}
        self.telemetry_items.append(
            {
                "badge": "signal",
                "status": "muted",
                "title": title,
                "body": body,
                "details": details,
                "created_at": None,
            }
        )

    def record_event(self, event_name: str, message: str, **data: object) -> None:
        self.tool_events.append(
            {
                "badge": "event",
                "status": _status_class(data.get("status") or "muted"),
                "title": _humanize_name(event_name),
                "body": _truncate_text(message, limit=_MAX_TURN_PREVIEW_CHARS),
                "details": dict(data),
                "created_at": None,
            }
        )

    def record_usage(self, **data: object) -> None:
        self.telemetry_items.append(
            {
                "badge": "usage",
                "status": "muted",
                "title": _humanize_name(data.get("request_kind") or "usage"),
                "body": _truncate_text(data.get("model") or "Usage tracked", limit=120),
                "details": dict(data),
                "created_at": None,
            }
        )


def _trace_body(*, kind: str, details: Mapping[str, object]) -> str:
    if kind == "decision":
        selected = details.get("selected")
        if isinstance(selected, Mapping):
            summary = _truncate_text(selected.get("summary"), limit=_MAX_TURN_PREVIEW_CHARS)
            if summary:
                return summary
    for key in ("reason", "decision", "reply", "source"):
        rendered = _truncate_text(details.get(key), limit=_MAX_TURN_PREVIEW_CHARS)
        if rendered:
            return rendered
    return _humanize_name(kind)


class _ConversationLabToolOwner:
    """Expose the runtime/tool contract expected by the realtime handlers."""

    def __init__(
        self,
        *,
        config: TwinrConfig,
        env_path: Path,
        runtime: TwinrRuntime,
        print_backend: object,
        usage_store: TwinrUsageStore,
        collector: _TurnCollector,
        configurable_providers: Sequence[object],
    ) -> None:
        self.config = config
        self.env_path = env_path
        self.runtime = runtime
        self.print_backend = print_backend
        self.usage_store = usage_store
        self.collector = collector
        self._configurable_providers = tuple(configurable_providers)
        self._current_turn_audio_pcm = b""
        self._camera_instance = None
        self._camera_lock = threading.Lock()

    @property
    def camera(self):
        """Expose one lazily initialized still camera for camera-backed tools."""

        return owner_camera(self)

    def emit(self, payload: str) -> None:
        self.collector.emit(payload)

    def _record_event(self, event_name: str, message: str, **data: object) -> None:
        self.collector.record_event(event_name, message, **data)

    def _record_usage(self, **data: object) -> None:
        metadata = {
            key: value
            for key, value in data.items()
            if key not in {"source", "request_kind", "model", "response_id", "request_id", "used_web_search", "token_usage"}
        }
        self.usage_store.append(
            source=str(data.get("source") or _CONVERSATION_LAB_SOURCE),
            request_kind=str(data.get("request_kind") or _CONVERSATION_LAB_REQUEST_KIND),
            model=str(data.get("model") or "") or None,
            response_id=str(data.get("response_id") or "") or None,
            request_id=str(data.get("request_id") or "") or None,
            used_web_search=bool(data.get("used_web_search", False)),
            token_usage=data.get("token_usage"),
            metadata={key: _truncate_text(value, limit=120) for key, value in metadata.items()},
        )
        self.collector.record_usage(**data)

    def _reload_live_config_from_env(self, env_path: Path) -> None:
        updated_config = TwinrConfig.from_env(env_path)
        self.config = updated_config
        self.runtime.apply_live_config(updated_config)
        seen: set[int] = set()
        for provider in self._configurable_providers:
            provider_id = id(provider)
            if provider_id in seen:
                continue
            seen.add(provider_id)
            if hasattr(provider, "config"):
                provider.config = updated_config

    def _build_vision_images(self):
        """Capture the normalized vision-image list for one tool call."""

        return build_vision_images(self)

    def _build_vision_prompt(self, question: str, *, include_reference: bool) -> str:
        """Build the stable camera-question prompt used by inspect_camera."""

        return build_vision_prompt(question, include_reference=include_reference)


def _build_tool_loop(
    *,
    config: TwinrConfig,
    owner: _ConversationLabToolOwner,
) -> tuple[object, tuple[object, ...]]:
    provider_bundle = build_streaming_provider_bundle(config)
    tool_executor = RealtimeToolExecutor(owner)
    available_tool_names = available_realtime_tool_names(
        config,
        tool_names=_CONVERSATION_LAB_TOOL_NAMES,
    )
    tool_handlers = bind_realtime_tool_handlers(tool_executor)
    tool_handlers = {
        name: handler
        for name, handler in tool_handlers.items()
        if name in available_tool_names
    }
    tool_schemas = (
        build_compact_agent_tool_schemas(available_tool_names)
        if (config.llm_provider or "").strip().lower() == "groq"
        else build_agent_tool_schemas(available_tool_names)
    )
    loop_resources: list[object] = [provider_bundle.print_backend, provider_bundle.tool_agent, provider_bundle.support_backend]
    if (
        (config.llm_provider or "").strip().lower() == "openai"
        and bool(config.streaming_dual_lane_enabled)
        and isinstance(provider_bundle.tool_agent, OpenAIToolCallingAgentProvider)
    ):
        supervisor_backend = OpenAIBackend(config=config)
        specialist_backend = OpenAIBackend(config=config)
        supervisor_decision_backend = OpenAIBackend(config=config)
        supervisor_provider = OpenAIToolCallingAgentProvider(
            supervisor_backend,
            model_override=config.streaming_supervisor_model,
            reasoning_effort_override=config.streaming_supervisor_reasoning_effort,
            base_instructions_override=load_supervisor_loop_instructions(config),
            replace_base_instructions=True,
        )
        supervisor_decision_provider = OpenAISupervisorDecisionProvider(
            supervisor_decision_backend,
            model_override=config.streaming_supervisor_model,
            reasoning_effort_override=config.streaming_supervisor_reasoning_effort,
            base_instructions_override=load_supervisor_loop_instructions(config),
            replace_base_instructions=True,
        )
        specialist_provider = OpenAIToolCallingAgentProvider(
            specialist_backend,
            model_override=config.streaming_specialist_model,
            reasoning_effort_override=config.streaming_specialist_reasoning_effort,
        )
        owner.print_backend = provider_bundle.print_backend
        owner._configurable_providers = tuple(
            list(owner._configurable_providers)
            + [
                provider_bundle.print_backend,
                provider_bundle.tool_agent,
                provider_bundle.support_backend,
                supervisor_provider,
                supervisor_decision_provider,
                specialist_provider,
            ]
        )
        loop_resources.extend([supervisor_backend, specialist_backend, supervisor_decision_backend, supervisor_provider, supervisor_decision_provider, specialist_provider])
        dual_lane_kwargs: dict[str, object] = {
            "supervisor_provider": supervisor_provider,
            "specialist_provider": specialist_provider,
            "tool_handlers": tool_handlers,
            "tool_schemas": tool_schemas,
            "supervisor_decision_provider": supervisor_decision_provider,
            "supervisor_instructions": build_supervisor_decision_instructions(
                config,
                extra_instructions=config.openai_realtime_instructions,
            ),
            "specialist_instructions": build_specialist_tool_agent_instructions(
                config,
                extra_instructions=config.openai_realtime_instructions,
            ),
            "max_rounds": 6,
        }
        try:
            dual_lane_parameters = signature(DualLaneToolLoop.__init__).parameters
        except (TypeError, ValueError):
            dual_lane_parameters = {}
        if "trace_event" in dual_lane_parameters:
            dual_lane_kwargs["trace_event"] = owner.collector.trace_event
        if "trace_decision" in dual_lane_parameters:
            dual_lane_kwargs["trace_decision"] = owner.collector.trace_decision
        return (
            DualLaneToolLoop(**dual_lane_kwargs),
            tuple(loop_resources),
        )

    owner.print_backend = provider_bundle.print_backend
    owner._configurable_providers = tuple(
        list(owner._configurable_providers)
        + [provider_bundle.print_backend, provider_bundle.tool_agent, provider_bundle.support_backend]
    )
    return (
        ToolCallingStreamingLoop(
            provider=provider_bundle.tool_agent,
            tool_handlers=tool_handlers,
            tool_schemas=tool_schemas,
            stream_final_only=((config.llm_provider or "").strip().lower() == "groq"),
        ),
        tuple(loop_resources),
    )


def _turn_instructions(config: TwinrConfig) -> str:
    return build_tool_agent_instructions(
        config,
        extra_instructions=config.openai_realtime_instructions,
    )


def _run_text_turn(
    *,
    config: TwinrConfig,
    runtime: TwinrRuntime,
    loop: object,
    prompt: str,
) -> object:
    instructions = _turn_instructions(config)
    if isinstance(loop, DualLaneToolLoop):
        return loop.run(
            prompt,
            conversation=runtime.tool_provider_conversation_context(),
            supervisor_conversation=runtime.supervisor_provider_conversation_context(),
            instructions=instructions,
            allow_web_search=False,
            on_text_delta=None,
            on_lane_text_delta=None,
        )
    if isinstance(loop, ToolCallingStreamingLoop):
        return loop.run(
            prompt,
            conversation=runtime.tool_provider_conversation_context(),
            instructions=instructions,
            allow_web_search=False,
            on_text_delta=None,
        )
    raise TypeError("Unsupported conversation-lab loop type")


def _build_turn_record(
    *,
    prompt: str,
    response_text: str,
    result: object | None,
    status: str,
    error_message: str | None,
    retrieval_before: dict[str, object],
    retrieval_after: dict[str, object] | None,
    collector: _TurnCollector,
    before_enqueue: dict[str, object],
    after_enqueue: dict[str, object] | None,
    after_flush: dict[str, object] | None,
    flush_ok: bool | None,
) -> dict[str, object]:
    return {
        "turn_id": _new_id("turn"),
        "created_at": _utc_now_iso_z(),
        "prompt": prompt,
        "response": response_text,
        "status": status,
        "status_badge": _turn_status_badge(status),
        "error_message": error_message,
        "result_rows": _turn_summary_rows(result=result, response_text=response_text),
        "route_items": _route_items(result, collector.route_items),
        "tool_items": _tool_items(result) + _collector_section_items("event", collector.tool_events),
        "telemetry_items": _collector_section_items("telemetry", collector.telemetry_items),
        "memory_rows": _memory_rows(
            before_enqueue=before_enqueue,
            after_enqueue=after_enqueue,
            after_flush=after_flush,
            flush_ok=flush_ok,
        ),
        "retrieval_before": retrieval_before,
        "retrieval_after": retrieval_after,
    }


def _default_session_title() -> str:
    return "Portal Conversation"


def _session_title_from_prompt(prompt: str) -> str:
    title = _truncate_text(prompt, limit=_MAX_SESSION_TITLE_CHARS)
    return title or _default_session_title()


def _session_summary(session: Mapping[str, object]) -> dict[str, object]:
    turns = tuple(session.get("turns", ()) or ())
    last_turn = turns[-1] if turns else {}
    status = str(getattr(last_turn, "get", lambda *_args, **_kwargs: "muted")("status", "muted"))
    return {
        "session_id": str(session.get("session_id", "") or ""),
        "title": str(session.get("title", "") or _default_session_title()),
        "updated_at": str(session.get("updated_at", "") or ""),
        "turn_count": len(turns),
        "status": _status_class(status),
    }


def _load_session(path: Path) -> dict[str, object] | None:
    raw_content = read_text_file(path).strip()
    if not raw_content:
        return None
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    parsed.setdefault("title", _default_session_title())
    parsed.setdefault("turns", [])
    parsed.setdefault("updated_at", parsed.get("created_at") or _utc_now_iso_z())
    return parsed


def _save_session(ops_paths: TwinrOpsPaths, session: Mapping[str, object]) -> None:
    session_id = _ensure_session_id(session.get("session_id"))
    if session_id is None:
        raise ValueError("Conversation-lab session is missing a valid session_id.")
    payload = dict(session)
    payload["updated_at"] = str(payload.get("updated_at") or _utc_now_iso_z())
    payload["title"] = str(payload.get("title") or _default_session_title())
    path = _session_path(ops_paths, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_file(path, json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True))


def _load_or_create_session(ops_paths: TwinrOpsPaths, session_id: str | None) -> dict[str, object]:
    normalized_id = _ensure_session_id(session_id)
    if normalized_id is not None:
        loaded = _load_session(_session_path(ops_paths, normalized_id))
        if loaded is not None:
            return loaded
    created = {
        "session_id": _new_id("session"),
        "created_at": _utc_now_iso_z(),
        "updated_at": _utc_now_iso_z(),
        "title": _default_session_title(),
        "turns": [],
    }
    _save_session(ops_paths, created)
    return created


def create_conversation_lab_session(ops_paths: TwinrOpsPaths) -> str:
    """Create one empty conversation-lab session and return its id."""

    session = _load_or_create_session(ops_paths, None)
    return str(session["session_id"])


def load_conversation_lab_state(
    ops_paths: TwinrOpsPaths,
    *,
    session_id: str | None,
) -> dict[str, object]:
    """Load the recent portal conversation sessions plus the active one."""

    root = _lab_root(ops_paths)
    sessions: list[dict[str, object]] = []
    if root.exists():
        for path in sorted(root.glob("*.json")):
            session = _load_session(path)
            if session is None:
                continue
            sessions.append(session)
    sessions.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    active_id = _ensure_session_id(session_id)
    active_session = next((session for session in sessions if session.get("session_id") == active_id), None)
    missing_session = active_id is not None and active_session is None
    if active_session is None and sessions:
        active_session = sessions[0]
    summaries = tuple(_session_summary(session) for session in sessions[:_MAX_SESSION_LIST])
    return {
        "sessions": summaries,
        "active_session": active_session,
        "missing_session": missing_session,
    }


def _conversation_lab_runtime_config(config: TwinrConfig) -> TwinrConfig:
    """Return the text-only runtime config used by the portal conversation lab.

    Portal text turns do not open microphone captures, so adaptive listening
    state is irrelevant there and should not touch the shared Pi audio timing
    store. Conversation Lab turns also stay out of the background long-term
    turn writer so operator debugging does not create bounded flush/shutdown
    tail work.
    """

    return replace(
        config,
        adaptive_timing_enabled=False,
        long_term_memory_background_store_turns=False,
    )


def run_conversation_lab_turn(
    config: TwinrConfig,
    env_path: Path,
    ops_paths: TwinrOpsPaths,
    *,
    session_id: str | None,
    prompt: str,
) -> str:
    """Run one real Twinr text turn and persist its human-readable traces."""

    normalized_prompt = str(prompt or "").strip()
    if not normalized_prompt:
        raise ValueError("Please enter a prompt first.")

    session = _load_or_create_session(ops_paths, session_id)
    collector = _TurnCollector()
    runtime: TwinrRuntime | None = None
    loop_resources: tuple[object, ...] = ()
    response_text = ""
    result: object | None = None
    error_message: str | None = None
    retrieval_before = _search_snapshot(config, normalized_prompt)
    retrieval_after: dict[str, object] | None = None
    before_enqueue: dict[str, object] = {}
    after_enqueue: dict[str, object] | None = None
    after_flush: dict[str, object] | None = None
    flush_ok: bool | None = None
    status = "ok"

    try:
        runtime_config = _conversation_lab_runtime_config(config)
        runtime = TwinrRuntime(runtime_config)
        runtime.user_voice_status = "portal_operator_authenticated"
        _seed_runtime_conversation(runtime, session)
        runtime.last_transcript = normalized_prompt
        before_enqueue = _writer_state_snapshot(runtime)
        usage_store = TwinrUsageStore.from_config(config)
        owner = _ConversationLabToolOwner(
            config=runtime_config,
            env_path=env_path,
            runtime=runtime,
            print_backend=None,
            usage_store=usage_store,
            collector=collector,
            configurable_providers=(),
        )
        loop, loop_resources = _build_tool_loop(config=runtime_config, owner=owner)
        result = _run_text_turn(
            config=runtime_config,
            runtime=runtime,
            loop=loop,
            prompt=normalized_prompt,
        )
        response_text = runtime.finalize_agent_turn(_result_text(result))
        usage_store.append(
            source=_CONVERSATION_LAB_SOURCE,
            request_kind=_CONVERSATION_LAB_REQUEST_KIND,
            model=str(getattr(result, "model", "") or "") or None,
            response_id=str(getattr(result, "response_id", "") or "") or None,
            request_id=str(getattr(result, "request_id", "") or "") or None,
            used_web_search=bool(getattr(result, "used_web_search", False)),
            token_usage=getattr(result, "token_usage", None),
            metadata={"turn_type": "portal_operator_lab"},
        )
        after_enqueue = _writer_state_snapshot(runtime)
        if _has_background_long_term_writers(runtime):
            flush_timeout_s = min(
                _CONVERSATION_LAB_FLUSH_TIMEOUT_MAX_S,
                max(2.0, float(getattr(config, "long_term_memory_remote_flush_timeout_s", 2.0))),
            )
            flush_ok = runtime.flush_long_term_memory(timeout_s=flush_timeout_s)
            after_flush = _writer_state_snapshot(runtime)
        else:
            flush_ok = True
            after_flush = dict(after_enqueue)
        if flush_ok:
            retrieval_after = _search_snapshot(config, normalized_prompt)
        else:
            status = "warn"
            error_message = (
                f"Long-term memory flush did not complete within {flush_timeout_s:.1f}s; "
                "the turn finished, but durable memory updates may still be pending."
            )
    except Exception as exc:
        status = "error"
        error_message = f"{type(exc).__name__}: {_truncate_text(exc, limit=160)}"
        if not response_text and result is not None:
            response_text = _truncate_text(getattr(result, "text", ""), limit=600)
    finally:
        if runtime is not None:
            try:
                runtime.shutdown(timeout_s=2.0)
            except Exception:
                pass
        seen: set[int] = set()
        for component in loop_resources:
            component_id = id(component)
            if component_id in seen:
                continue
            seen.add(component_id)
            _close_if_possible(component)

    turn_record = _build_turn_record(
        prompt=normalized_prompt,
        response_text=response_text,
        result=result,
        status=status,
        error_message=error_message,
        retrieval_before=retrieval_before,
        retrieval_after=retrieval_after,
        collector=collector,
        before_enqueue=before_enqueue,
        after_enqueue=after_enqueue,
        after_flush=after_flush,
        flush_ok=flush_ok,
    )
    turns = list(tuple(session.get("turns", ()) or ()))
    turns.append(turn_record)
    session["turns"] = turns
    if len(turns) == 1 or str(session.get("title", "") or "").strip() == _default_session_title():
        session["title"] = _session_title_from_prompt(normalized_prompt)
    session["updated_at"] = _utc_now_iso_z()
    _save_session(ops_paths, session)
    return str(session["session_id"])
