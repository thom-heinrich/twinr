# CHANGELOG: 2026-03-28
# BUG-1: Prevent gated-mode crashes when the router emits an unsupported authoritative label;
#        abstain cleanly and fall back instead of raising ValueError.
# BUG-2: Make live reload fail-safe; keep the last known-good router when a rebuild fails.
# BUG-3: Make tracing and ID generation robust to missing/non-finite score fields and fast-LLM
#        failures, so successful local routing does not crash in observability or bridge code.
# BUG-4: Degrade gracefully to single-stage routing when the optional user-intent bundle is
#        unavailable instead of disabling the whole local router.
# BUG-5: Make synthesized supervisor IDs transcript-scoped; the previous IDs collided across
#        unrelated utterances that shared the same route label / score tuple.
# SEC-1: Normalize and cap routing transcripts to mitigate practical CPU / latency DoS on a
#        Raspberry Pi 4 from oversized or control-character-heavy inputs.
# IMP-1: Add a small TTL/LRU resolution cache for repeated streaming transcripts.
# IMP-2: Add policy-gated abstention, optional model-dir root validation, and richer traces.

"""Bridge local semantic router decisions into the streaming workflow path."""

from __future__ import annotations

import math
import stat
import threading
import time
import unicodedata
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

from twinr.agent.base_agent.contracts import FirstWordReply, SupervisorDecision
from twinr.agent.tools.prompting import build_local_route_first_word_instructions
from twinr.agent.routing import (
    LocalSemanticRouter,
    SemanticRouteDecision,
    TwoStageLocalSemanticRouter,
    UserIntentDecision,
    load_semantic_router_bundle,
    load_user_intent_bundle,
)


SUPPORTED_HANDOFF_LABELS: tuple[str, ...] = ("web", "memory", "tool")
DEFAULT_MAX_TRANSCRIPT_CHARS = 1024
DEFAULT_CACHE_SIZE = 128
DEFAULT_CACHE_TTL_S = 10.0


@dataclass(frozen=True, slots=True)
class LocalSemanticRouteResolution:
    """Store one local-route result plus any synthesized workflow artifacts."""

    route_decision: SemanticRouteDecision
    user_intent_decision: UserIntentDecision | None = None
    supervisor_decision: SupervisorDecision | None = None
    bridge_reply: FirstWordReply | None = None


@dataclass(frozen=True, slots=True)
class _RuntimePolicy:
    mode: str
    trace_enabled: bool
    first_word_enabled: bool
    supported_labels: tuple[str, ...]
    max_transcript_chars: int
    cache_size: int
    cache_ttl_s: float
    min_confidence: float | None
    min_margin: float | None
    allowed_model_root: str | None


@dataclass(frozen=True, slots=True)
class _RouterBuildResult:
    status: str
    router: LocalSemanticRouter | TwoStageLocalSemanticRouter | None = None
    model_dir: str | None = None
    user_intent_model_dir: str | None = None
    model_id: str | None = None
    authoritative_labels: tuple[str, ...] = ()
    two_stage: bool = False
    reason: str | None = None


@dataclass(slots=True)
class _ResolutionCacheEntry:
    expires_at: float
    resolution: LocalSemanticRouteResolution


class StreamingSemanticRouterRuntime:
    """Own local semantic-router loading and streaming-path adaptation."""

    def __init__(self, loop) -> None:
        self._loop = loop
        self._state_lock = threading.RLock()
        self._router: LocalSemanticRouter | TwoStageLocalSemanticRouter | None = None
        self._router_epoch = 0
        self._instance_scope_token = _short_hash(f"runtime:{time.time_ns()}:{id(loop)}")
        self._resolution_cache: OrderedDict[tuple[object, ...], _ResolutionCacheEntry] = OrderedDict()
        build_result = self._build_router_candidate(self._runtime_policy())
        with self._state_lock:
            self._apply_build_result_locked(build_result, initial=True)

    def reload(self) -> None:
        """Rebuild the local semantic router after live config reloads."""

        build_result = self._build_router_candidate(self._runtime_policy())
        with self._state_lock:
            previous_router = self._router
            applied = self._apply_build_result_locked(build_result, initial=False)
        if not applied and previous_router is not None:
            self._trace_event(
                "streaming_local_semantic_router_reload_kept_previous",
                kind="cache",
                level="WARN",
                details={
                    "reason": build_result.reason or "build_failed",
                    "mode": self._router_mode(),
                },
            )

    def resolve_transcript(self, transcript: str) -> LocalSemanticRouteResolution | None:
        """Classify one transcript locally and synthesize workflow state when safe."""

        started_at = time.monotonic()
        policy = self._runtime_policy()
        cleaned, normalization = _normalize_transcript(transcript, policy.max_transcript_chars)
        if policy.mode == "off" or not cleaned:
            return None

        with self._state_lock:
            router = self._router
            router_epoch = self._router_epoch
        if router is None:
            return None

        if policy.trace_enabled and normalization["truncated"]:
            self._trace_event(
                "streaming_local_semantic_router_input_truncated",
                kind="guardrail",
                level="WARN",
                details={
                    "mode": policy.mode,
                    "raw_chars": normalization["raw_chars"],
                    "normalized_chars": normalization["normalized_chars"],
                    "kept_chars": len(cleaned),
                    "transcript": _text_summary(cleaned),
                },
            )

        cache_key = self._resolution_cache_key(router_epoch, policy, cleaned)
        cached = self._cache_get(cache_key, policy)
        if cached is not None:
            if policy.trace_enabled:
                self._trace_event(
                    "streaming_local_semantic_router_cache_hit",
                    kind="cache",
                    details={
                        "mode": policy.mode,
                        "route": _route_summary(cached.route_decision),
                        "user_intent": _user_intent_summary(cached.user_intent_decision),
                        "transcript": _text_summary(cleaned),
                    },
                )
            return cached

        try:
            two_stage_decision = None
            user_intent_decision = None
            if isinstance(router, TwoStageLocalSemanticRouter):
                two_stage_decision = router.classify(cleaned)
                decision = two_stage_decision.route_decision
                user_intent_decision = two_stage_decision.user_intent
            else:
                decision = router.classify(cleaned)
        except Exception as exc:
            self._loop.emit(f"local_semantic_router_failed={type(exc).__name__}")
            self._trace_event(
                "streaming_local_semantic_router_failed",
                kind="exception",
                level="WARN",
                details={
                    "error_type": type(exc).__name__,
                    "mode": policy.mode,
                    "transcript": _text_summary(cleaned),
                },
            )
            return None

        if policy.trace_enabled:
            self._trace_event(
                "streaming_local_semantic_router_decision",
                kind="decision",
                details={
                    "mode": policy.mode,
                    "route": _route_summary(decision),
                    "user_intent": _user_intent_summary(user_intent_decision),
                    "allowed_route_labels": _safe_labels(
                        getattr(two_stage_decision, "allowed_route_labels", None)
                    )
                    if two_stage_decision is not None
                    else None,
                    "transcript": _text_summary(cleaned),
                },
            )

        route_is_authoritative = bool(getattr(decision, "authoritative", False))
        resolution = LocalSemanticRouteResolution(
            route_decision=decision,
            user_intent_decision=user_intent_decision,
        )

        if policy.mode == "gated" and route_is_authoritative:
            abstain_reason = _policy_abstain_reason(decision, policy)
            if abstain_reason is None:
                bridge_reply = self._build_bridge_reply(
                    cleaned,
                    decision,
                    first_word_enabled=policy.first_word_enabled,
                    trace_enabled=policy.trace_enabled,
                )
                synthesized_decision = None
                if _normalized_label(getattr(decision, "label", "")) != "tool":
                    synthesized_decision = _synthesize_supervisor_decision(
                        route_decision=decision,
                        bridge_reply=bridge_reply,
                        transcript_hash=_short_hash(cleaned),
                        scope_token=_decision_scope_token(self._loop) or self._instance_scope_token,
                    )
                resolution = LocalSemanticRouteResolution(
                    route_decision=decision,
                    user_intent_decision=user_intent_decision,
                    supervisor_decision=synthesized_decision,
                    bridge_reply=bridge_reply,
                )
            else:
                if policy.trace_enabled:
                    self._trace_event(
                        "streaming_local_semantic_router_abstained",
                        kind="guardrail",
                        level="WARN",
                        details={
                            "mode": policy.mode,
                            "reason": abstain_reason,
                            "route": _route_summary(decision),
                            "transcript": _text_summary(cleaned),
                        },
                    )

        self._cache_put(cache_key, resolution, policy)

        if policy.trace_enabled:
            self._trace_event(
                "streaming_local_semantic_router_resolved",
                kind="decision",
                details={
                    "mode": policy.mode,
                    "route": _route_summary(resolution.route_decision),
                    "has_supervisor_decision": resolution.supervisor_decision is not None,
                    "has_bridge_reply": resolution.bridge_reply is not None,
                    "latency_ms_total": round((time.monotonic() - started_at) * 1000.0, 3),
                    "transcript": _text_summary(cleaned),
                },
            )
        return resolution

    def _build_router_candidate(self, policy: _RuntimePolicy) -> _RouterBuildResult:
        """Load the configured local semantic router bundle, if enabled."""

        mode = policy.mode
        if mode == "off":
            return _RouterBuildResult(status="disabled", reason="mode_off")

        model_dir_raw = str(getattr(self._loop.config, "local_semantic_router_model_dir", "") or "").strip()
        user_intent_model_dir_raw = str(
            getattr(self._loop.config, "local_semantic_router_user_intent_model_dir", "") or ""
        ).strip()
        if not model_dir_raw:
            self._trace_event(
                "streaming_local_semantic_router_disabled",
                kind="branch",
                details={"reason": "missing_model_dir", "mode": mode},
            )
            return _RouterBuildResult(status="disabled", reason="missing_model_dir")

        try:
            model_dir = self._resolve_model_dir(model_dir_raw, policy.allowed_model_root, kind="route_model")
            self._trace_if_world_writable(model_dir, kind="route_model")
            bundle = load_semantic_router_bundle(model_dir)
        except Exception as exc:
            self._loop.emit(f"local_semantic_router_unavailable={type(exc).__name__}")
            self._trace_event(
                "streaming_local_semantic_router_unavailable",
                kind="exception",
                level="WARN",
                details={
                    "error_type": type(exc).__name__,
                    "mode": mode,
                    "model_dir": str(model_dir_raw),
                },
            )
            return _RouterBuildResult(status="failed", reason="route_bundle_unavailable")

        router: LocalSemanticRouter | TwoStageLocalSemanticRouter
        two_stage = False
        user_intent_model_dir: Path | None = None

        if user_intent_model_dir_raw:
            try:
                user_intent_model_dir = self._resolve_model_dir(
                    user_intent_model_dir_raw,
                    policy.allowed_model_root,
                    kind="user_intent_model",
                )
                self._trace_if_world_writable(user_intent_model_dir, kind="user_intent_model")
                user_intent_bundle = load_user_intent_bundle(user_intent_model_dir)
                router = TwoStageLocalSemanticRouter(user_intent_bundle, bundle)
                two_stage = True
            except Exception as exc:
                self._loop.emit(f"local_semantic_router_user_intent_unavailable={type(exc).__name__}")
                self._trace_event(
                    "streaming_local_semantic_router_user_intent_unavailable",
                    kind="exception",
                    level="WARN",
                    details={
                        "error_type": type(exc).__name__,
                        "mode": mode,
                        "user_intent_model_dir": str(user_intent_model_dir_raw),
                        "fallback": "single_stage_router",
                    },
                )
                router = LocalSemanticRouter(bundle)
        else:
            router = LocalSemanticRouter(bundle)

        self._maybe_warmup_router(router)

        self._trace_event(
            "streaming_local_semantic_router_ready",
            kind="cache",
            details={
                "mode": mode,
                "model_dir": str(model_dir),
                "model_id": getattr(getattr(bundle, "metadata", None), "model_id", None),
                "authoritative_labels": _safe_labels(
                    getattr(getattr(bundle, "metadata", None), "authoritative_labels", None)
                ),
                "supported_route_labels": list(policy.supported_labels),
                "two_stage": two_stage,
                "user_intent_model_dir": str(user_intent_model_dir) if user_intent_model_dir is not None else None,
            },
        )
        return _RouterBuildResult(
            status="ready",
            router=router,
            model_dir=str(model_dir),
            user_intent_model_dir=str(user_intent_model_dir) if user_intent_model_dir is not None else None,
            model_id=getattr(getattr(bundle, "metadata", None), "model_id", None),
            authoritative_labels=tuple(
                _safe_labels(getattr(getattr(bundle, "metadata", None), "authoritative_labels", None))
            ),
            two_stage=two_stage,
        )

    def _apply_build_result_locked(self, build_result: _RouterBuildResult, *, initial: bool) -> bool:
        """Swap runtime state when the build result is usable.

        Returns True when the runtime state changed.
        """

        if build_result.status == "ready":
            self._router = build_result.router
            self._router_epoch += 1
            self._resolution_cache.clear()
            return True
        if build_result.status == "disabled":
            self._router = None
            self._router_epoch += 1
            self._resolution_cache.clear()
            return True
        if initial:
            self._router = None
        return False

    def _build_bridge_reply(
        self,
        transcript: str,
        decision: SemanticRouteDecision,
        *,
        first_word_enabled: bool | None = None,
        trace_enabled: bool | None = None,
    ) -> FirstWordReply | None:
        """Return one filler-style bridge reply only when the fast LLM provides one."""

        if first_word_enabled is None or trace_enabled is None:
            policy = self._runtime_policy()
            if first_word_enabled is None:
                first_word_enabled = policy.first_word_enabled
            if trace_enabled is None:
                trace_enabled = policy.trace_enabled

        route_label = str(getattr(decision, "label", "") or "").strip()
        try:
            route_first_word_instructions = build_local_route_first_word_instructions(
                route_label,
                handoff_goal=_handoff_goal_for_label(route_label),
                language_hint=_language_hint(self._loop),
            )
        except Exception as exc:
            self._loop.emit(f"local_semantic_router_bridge_instructions_failed={type(exc).__name__}")
            self._trace_event(
                "streaming_local_semantic_router_bridge_reply_failed",
                kind="exception",
                level="WARN",
                details={
                    "error_type": type(exc).__name__,
                    "route_label": route_label,
                    "stage": "instructions",
                },
            )
            return None

        reply = None
        if first_word_enabled:
            try:
                reply = self._loop._generate_first_word_reply(
                    transcript,
                    instructions=route_first_word_instructions,
                )
            except Exception as exc:
                self._loop.emit(f"local_semantic_router_first_word_failed={type(exc).__name__}")
                self._trace_event(
                    "streaming_local_semantic_router_bridge_reply_failed",
                    kind="exception",
                    level="WARN",
                    details={
                        "error_type": type(exc).__name__,
                        "route_label": route_label,
                        "stage": "first_word_llm",
                    },
                )
                reply = None

        spoken_text = str(getattr(reply, "spoken_text", "") or "").strip() if reply is not None else ""
        if spoken_text:
            reply_source = "first_word_llm"
        elif first_word_enabled:
            reply_source = "first_word_unavailable"
        else:
            reply_source = "first_word_disabled"

        if trace_enabled:
            self._trace_event(
                "streaming_local_semantic_router_bridge_reply",
                kind="decision",
                details={
                    "route_label": route_label,
                    "source": reply_source,
                    "text": _text_summary(spoken_text),
                },
            )
        if not spoken_text:
            return None
        return FirstWordReply(
            mode="filler",
            spoken_text=spoken_text,
            response_id=getattr(reply, "response_id", None) if reply is not None else None,
            request_id=getattr(reply, "request_id", None) if reply is not None else None,
            model=getattr(reply, "model", None) if reply is not None else None,
            token_usage=getattr(reply, "token_usage", None) if reply is not None else None,
        )

    def _runtime_policy(self) -> _RuntimePolicy:
        mode = self._router_mode()
        supported_labels = _parse_supported_labels(
            getattr(self._loop.config, "local_semantic_router_supported_labels", None),
            default=SUPPORTED_HANDOFF_LABELS,
        )
        return _RuntimePolicy(
            mode=mode,
            trace_enabled=bool(getattr(self._loop.config, "local_semantic_router_trace", True)),
            first_word_enabled=bool(getattr(self._loop.config, "streaming_first_word_enabled", False)),
            supported_labels=supported_labels,
            max_transcript_chars=_coerce_int(
                getattr(self._loop.config, "local_semantic_router_max_chars", DEFAULT_MAX_TRANSCRIPT_CHARS),
                default=DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=8192,
            ),
            cache_size=_coerce_int(
                getattr(self._loop.config, "local_semantic_router_cache_size", DEFAULT_CACHE_SIZE),
                default=DEFAULT_CACHE_SIZE,
                minimum=0,
                maximum=4096,
            ),
            cache_ttl_s=_coerce_float(
                getattr(self._loop.config, "local_semantic_router_cache_ttl_s", DEFAULT_CACHE_TTL_S),
                default=DEFAULT_CACHE_TTL_S,
                minimum=0.0,
                maximum=3600.0,
            ),
            min_confidence=_optional_float(
                getattr(self._loop.config, "local_semantic_router_min_confidence", None),
                minimum=0.0,
                maximum=1.0,
            ),
            min_margin=_optional_float(
                getattr(self._loop.config, "local_semantic_router_min_margin", None),
                minimum=0.0,
                maximum=None,
            ),
            allowed_model_root=str(
                getattr(self._loop.config, "local_semantic_router_allowed_model_root", "") or ""
            ).strip()
            or None,
        )

    def _router_mode(self) -> str:
        mode = str(getattr(self._loop.config, "local_semantic_router_mode", "off") or "off").strip().lower()
        if mode not in {"off", "shadow", "gated"}:
            return "off"
        return mode

    def _resolution_cache_key(
        self,
        router_epoch: int,
        policy: _RuntimePolicy,
        transcript: str,
    ) -> tuple[object, ...]:
        return (
            router_epoch,
            policy.mode,
            policy.first_word_enabled,
            policy.supported_labels,
            policy.min_confidence,
            policy.min_margin,
            transcript,
        )

    def _cache_get(
        self,
        key: tuple[object, ...],
        policy: _RuntimePolicy,
    ) -> LocalSemanticRouteResolution | None:
        if policy.cache_size <= 0 or policy.cache_ttl_s <= 0:
            return None
        now = time.monotonic()
        with self._state_lock:
            self._evict_expired_locked(now)
            entry = self._resolution_cache.get(key)
            if entry is None or entry.expires_at <= now:
                if entry is not None:
                    self._resolution_cache.pop(key, None)
                return None
            self._resolution_cache.move_to_end(key)
            try:
                return deepcopy(entry.resolution)
            except Exception:
                return entry.resolution

    def _cache_put(
        self,
        key: tuple[object, ...],
        resolution: LocalSemanticRouteResolution,
        policy: _RuntimePolicy,
    ) -> None:
        if policy.cache_size <= 0 or policy.cache_ttl_s <= 0:
            return
        cached_resolution = _sanitize_resolution_for_cache(resolution)
        if cached_resolution is None:
            return
        now = time.monotonic()
        expires_at = now + policy.cache_ttl_s
        with self._state_lock:
            self._evict_expired_locked(now)
            self._resolution_cache[key] = _ResolutionCacheEntry(
                expires_at=expires_at,
                resolution=cached_resolution,
            )
            self._resolution_cache.move_to_end(key)
            while len(self._resolution_cache) > policy.cache_size:
                self._resolution_cache.popitem(last=False)

    def _evict_expired_locked(self, now: float) -> None:
        expired_keys = [key for key, entry in self._resolution_cache.items() if entry.expires_at <= now]
        for key in expired_keys:
            self._resolution_cache.pop(key, None)

    def _resolve_model_dir(self, value: str, allowed_root: str | None, *, kind: str) -> Path:
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise NotADirectoryError(f"{kind} path is not a directory: {value!r}")
        if allowed_root:
            root = Path(allowed_root).expanduser().resolve(strict=True)
            if not _path_is_within(path, root):
                raise PermissionError(f"{kind} path {path!s} escapes allowed root {root!s}")
        return path

    def _trace_if_world_writable(self, path: Path, *, kind: str) -> None:
        try:
            mode = path.stat().st_mode
        except OSError:
            return
        if mode & stat.S_IWOTH:
            self._trace_event(
                "streaming_local_semantic_router_model_dir_world_writable",
                kind="guardrail",
                level="WARN",
                details={"kind": kind, "path": str(path)},
            )

    def _maybe_warmup_router(self, router: LocalSemanticRouter | TwoStageLocalSemanticRouter) -> None:
        warmup_enabled = bool(getattr(self._loop.config, "local_semantic_router_warmup_enabled", False))
        if not warmup_enabled:
            return
        warmup_probe = str(getattr(self._loop.config, "local_semantic_router_warmup_probe", "") or "").strip()
        try:
            warmup = getattr(router, "warmup", None)
            if callable(warmup):
                warmup()
                self._trace_event(
                    "streaming_local_semantic_router_warmup",
                    kind="cache",
                    details={"method": "router.warmup"},
                )
                return
            if warmup_probe:
                router.classify(warmup_probe)
                self._trace_event(
                    "streaming_local_semantic_router_warmup",
                    kind="cache",
                    details={"method": "classify_probe", "probe": _text_summary(warmup_probe)},
                )
        except Exception as exc:
            self._trace_event(
                "streaming_local_semantic_router_warmup_failed",
                kind="exception",
                level="WARN",
                details={"error_type": type(exc).__name__},
            )

    def _trace_event(
        self,
        name: str,
        *,
        kind: str,
        details: dict[str, object],
        level: str | None = None,
    ) -> None:
        trace = getattr(self._loop, "_trace_event", None)
        if not callable(trace):
            return
        payload: dict[str, object] = {"kind": kind, "details": details}
        if level is not None:
            payload["level"] = level
        trace(name, **payload)


def _synthesize_supervisor_decision(
    route_decision: SemanticRouteDecision,
    bridge_reply: FirstWordReply | None,
    *,
    transcript_hash: str | None = None,
    scope_token: str | None = None,
) -> SupervisorDecision:
    """Map one authoritative local route into the supervisor-decision contract."""

    route_label = str(getattr(route_decision, "label", "") or "").strip()
    route_key = _normalized_label(route_label)
    if transcript_hash is None:
        transcript_hash = _short_hash(f"{route_key}:{_route_id_suffix(route_decision)}")
    scope_prefix = f"{scope_token}:" if scope_token else ""
    response_id = f"local_semantic_router:{scope_prefix}{route_key}:{transcript_hash}"
    request_id = (
        f"local_semantic_router:{scope_prefix}{route_key}:"
        f"{transcript_hash}:{_route_id_suffix(route_decision)}"
    )
    # BREAKING: response_id/request_id are now transcript-scoped to avoid collisions
    # across unrelated utterances that happened to share the same routed score tuple.
    return SupervisorDecision(
        action="handoff",
        spoken_ack=bridge_reply.spoken_text if bridge_reply is not None else None,
        spoken_reply=None,
        kind=_handoff_kind_for_label(route_label),
        goal=_handoff_goal_for_label(route_label),
        allow_web_search=(route_key == "web"),
        context_scope=(
            "full_context"
            if route_key == "memory"
            else "tiny_recent"
            if route_key == "tool"
            else None
        ),
        response_id=response_id,
        request_id=request_id,
        model=getattr(route_decision, "model_id", None),
        token_usage=None,
    )
def _handoff_kind_for_label(route_label: str) -> str:
    route_key = _normalized_label(route_label)
    if route_key == "web":
        return "search"
    if route_key == "memory":
        return "memory"
    if route_key == "tool":
        return "general"
    raise ValueError(f"Local semantic router cannot synthesize handoff for {route_label!r}.")


def _handoff_goal_for_label(route_label: str) -> str:
    route_key = _normalized_label(route_label)
    if route_key == "web":
        return "Use fresh external information and reply clearly."
    if route_key == "memory":
        return "Answer using the user's persisted or recent Twinr memory."
    if route_key == "tool":
        return "Use the appropriate Twinr tools or device actions to handle the request."
    raise ValueError(f"Local semantic router cannot synthesize goal for {route_label!r}.")


def _language_hint(loop) -> str | None:
    return (
        str(getattr(loop.config, "deepgram_stt_language", "") or "").strip()
        or str(getattr(loop.config, "openai_realtime_language", "") or "").strip()
        or None
    )


def _route_id_suffix(route_decision: SemanticRouteDecision) -> str:
    payload = (
        f"{getattr(route_decision, 'label', '')}|{_float_fragment(getattr(route_decision, 'confidence', None))}|"
        f"{_float_fragment(getattr(route_decision, 'margin', None))}|"
        f"{getattr(route_decision, 'model_id', None)}"
    )
    return _short_hash(payload)


def _normalize_transcript(value: object, max_chars: int) -> tuple[str, dict[str, object]]:
    # BREAKING: routing now uses normalized, bounded text to keep edge latency predictable
    # and to avoid oversized transcripts from monopolizing a Raspberry Pi 4 CPU.
    raw = str(value or "")
    normalized = unicodedata.normalize("NFKC", raw)
    normalized = "".join(ch if (ch.isprintable() or ch.isspace()) else " " for ch in normalized)
    normalized = " ".join(normalized.split())
    normalized_before_truncation = normalized
    if len(normalized) > max_chars:
        candidate = normalized[:max_chars]
        if " " in candidate:
            candidate = candidate.rsplit(" ", 1)[0]
        normalized = candidate.strip() or normalized[:max_chars].strip()
    return normalized, {
        "raw_chars": len(raw),
        "normalized_chars": len(normalized_before_truncation),
        "truncated": len(normalized_before_truncation) > len(normalized),
    }


def _text_summary(value: object) -> dict[str, object]:
    """Describe text safely for workflow tracing without leaking raw content."""

    normalized = str(value or "").strip()
    if not normalized:
        return {"present": False, "chars": 0, "words": 0, "sha12": None}
    return {
        "present": True,
        "chars": len(normalized),
        "words": len(normalized.split()),
        "sha12": _short_hash(normalized),
    }


def _route_summary(route_decision: SemanticRouteDecision) -> dict[str, object]:
    """Return one redacted summary of a scored route decision."""

    return {
        "label": getattr(route_decision, "label", None),
        "confidence": _rounded_float(getattr(route_decision, "confidence", None), 6),
        "margin": _rounded_float(getattr(route_decision, "margin", None), 6),
        "authoritative": bool(getattr(route_decision, "authoritative", False)),
        "fallback_reason": getattr(route_decision, "fallback_reason", None),
        "model_id": getattr(route_decision, "model_id", None),
        "latency_ms": _rounded_float(getattr(route_decision, "latency_ms", None), 3),
    }


def _user_intent_summary(user_intent_decision: UserIntentDecision | None) -> dict[str, object] | None:
    """Return one redacted summary of a scored user-intent decision."""

    if user_intent_decision is None:
        return None
    return {
        "label": getattr(user_intent_decision, "label", None),
        "confidence": _rounded_float(getattr(user_intent_decision, "confidence", None), 6),
        "margin": _rounded_float(getattr(user_intent_decision, "margin", None), 6),
        "model_id": getattr(user_intent_decision, "model_id", None),
        "latency_ms": _rounded_float(getattr(user_intent_decision, "latency_ms", None), 3),
    }


def _sanitize_resolution_for_cache(
    resolution: LocalSemanticRouteResolution,
) -> LocalSemanticRouteResolution | None:
    try:
        cached = deepcopy(resolution)
    except Exception:
        return None
    bridge_reply = cached.bridge_reply
    if bridge_reply is not None:
        cached = LocalSemanticRouteResolution(
            route_decision=cached.route_decision,
            user_intent_decision=cached.user_intent_decision,
            supervisor_decision=cached.supervisor_decision,
            bridge_reply=FirstWordReply(
                mode=str(getattr(bridge_reply, "mode", "filler") or "filler"),
                spoken_text=str(getattr(bridge_reply, "spoken_text", "") or "").strip(),
                response_id=None,
                request_id=None,
                model=getattr(bridge_reply, "model", None),
                token_usage=getattr(bridge_reply, "token_usage", None),
            ),
        )
    return cached


def _policy_abstain_reason(
    route_decision: SemanticRouteDecision,
    policy: _RuntimePolicy,
) -> str | None:
    route_label = str(getattr(route_decision, "label", "") or "").strip()
    route_key = _normalized_label(route_label)
    if route_key not in SUPPORTED_HANDOFF_LABELS or route_key not in policy.supported_labels:
        return f"unsupported_label:{route_label or 'unknown'}"
    confidence = _finite_float(getattr(route_decision, "confidence", None))
    if policy.min_confidence is not None:
        if confidence is None:
            return "confidence_missing"
        if confidence < policy.min_confidence:
            return f"confidence_below_floor:{confidence:.6f}"
    margin = _finite_float(getattr(route_decision, "margin", None))
    if policy.min_margin is not None:
        if margin is None:
            return "margin_missing"
        if margin < policy.min_margin:
            return f"margin_below_floor:{margin:.6f}"
    return None


def _decision_scope_token(loop) -> str | None:
    for attr_name in ("session_id", "conversation_id", "stream_id", "connection_id", "request_id"):
        value = str(getattr(loop, attr_name, "") or "").strip()
        if value:
            return _short_hash(f"{attr_name}:{value}")
    return None


def _parse_supported_labels(value: object, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    labels: list[str] = []
    if isinstance(value, str):
        raw_labels = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw_labels = list(value)
    else:
        return default
    seen: set[str] = set()
    for item in raw_labels:
        label = _normalized_label(item)
        if not label or label not in SUPPORTED_HANDOFF_LABELS or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return tuple(labels) or default


def _safe_labels(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            return None
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    return cleaned or None


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _short_hash(value: object) -> str:
    return sha1(str(value).encode("utf-8")).hexdigest()[:12]


def _normalized_label(value: object) -> str:
    return str(value or "").strip().lower()


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _rounded_float(value: object, digits: int) -> float | None:
    number = _finite_float(value)
    if number is None:
        return None
    return round(number, digits)


def _float_fragment(value: object) -> str:
    number = _finite_float(value)
    if number is None:
        return "na"
    return f"{number:.6f}"


def _coerce_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _coerce_float(value: object, *, default: float, minimum: float, maximum: float) -> float:
    number = _finite_float(value)
    if number is None:
        return default
    return max(minimum, min(maximum, number))


def _optional_float(value: object, *, minimum: float | None, maximum: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    number = _finite_float(value)
    if number is None:
        return None
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number
