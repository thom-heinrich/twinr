"""Bridge local semantic router decisions into the streaming workflow path."""

from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class LocalSemanticRouteResolution:
    """Store one local-route result plus any synthesized workflow artifacts."""

    route_decision: SemanticRouteDecision
    user_intent_decision: UserIntentDecision | None = None
    supervisor_decision: SupervisorDecision | None = None
    bridge_reply: FirstWordReply | None = None


class StreamingSemanticRouterRuntime:
    """Own local semantic-router loading and streaming-path adaptation."""

    def __init__(self, loop) -> None:
        self._loop = loop
        self._router = self._build_router()

    def reload(self) -> None:
        """Rebuild the local semantic router after live config reloads."""

        self._router = self._build_router()

    def resolve_transcript(self, transcript: str) -> LocalSemanticRouteResolution | None:
        """Classify one transcript locally and synthesize workflow state when safe."""

        router = self._router
        cleaned = str(transcript or "").strip()
        if router is None or not cleaned:
            return None
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
            self._loop._trace_event(
                "streaming_local_semantic_router_failed",
                kind="exception",
                level="WARN",
                details={
                    "error_type": type(exc).__name__,
                    "mode": self._router_mode(),
                    "transcript": _text_summary(cleaned),
                },
            )
            return None
        if bool(getattr(self._loop.config, "local_semantic_router_trace", True)):
            self._loop._trace_event(
                "streaming_local_semantic_router_decision",
                kind="decision",
                details={
                    "mode": self._router_mode(),
                    "route": _route_summary(decision),
                    "user_intent": _user_intent_summary(user_intent_decision),
                    "allowed_route_labels": list(two_stage_decision.allowed_route_labels)
                    if two_stage_decision is not None
                    else None,
                    "transcript": _text_summary(cleaned),
                },
            )
        if self._router_mode() != "gated" or not decision.authoritative:
            return LocalSemanticRouteResolution(
                route_decision=decision,
                user_intent_decision=user_intent_decision,
            )
        bridge_reply = self._build_bridge_reply(cleaned, decision)
        synthesized = _synthesize_supervisor_decision(decision, bridge_reply)
        return LocalSemanticRouteResolution(
            route_decision=decision,
            user_intent_decision=user_intent_decision,
            supervisor_decision=synthesized,
            bridge_reply=bridge_reply,
        )

    def _build_router(self) -> LocalSemanticRouter | None:
        """Load the configured local semantic router bundle, if enabled."""

        mode = self._router_mode()
        if mode == "off":
            return None
        model_dir = str(getattr(self._loop.config, "local_semantic_router_model_dir", "") or "").strip()
        user_intent_model_dir = str(
            getattr(self._loop.config, "local_semantic_router_user_intent_model_dir", "") or ""
        ).strip()
        if not model_dir:
            self._loop._trace_event(
                "streaming_local_semantic_router_disabled",
                kind="branch",
                details={"reason": "missing_model_dir", "mode": mode},
            )
            return None
        try:
            bundle = load_semantic_router_bundle(Path(model_dir))
            if user_intent_model_dir:
                user_intent_bundle = load_user_intent_bundle(Path(user_intent_model_dir))
                router = TwoStageLocalSemanticRouter(user_intent_bundle, bundle)
            else:
                router = LocalSemanticRouter(bundle)
        except Exception as exc:
            self._loop.emit(f"local_semantic_router_unavailable={type(exc).__name__}")
            self._loop._trace_event(
                "streaming_local_semantic_router_unavailable",
                kind="exception",
                level="WARN",
                details={
                    "error_type": type(exc).__name__,
                    "mode": mode,
                    "model_dir": str(model_dir),
                },
            )
            return None
        self._loop._trace_event(
            "streaming_local_semantic_router_ready",
            kind="cache",
            details={
                "mode": mode,
                "model_dir": str(model_dir),
                "model_id": bundle.metadata.model_id,
                "authoritative_labels": list(bundle.metadata.authoritative_labels),
                "two_stage": bool(user_intent_model_dir),
                "user_intent_model_dir": str(user_intent_model_dir) or None,
            },
        )
        return router

    def _build_bridge_reply(
        self,
        transcript: str,
        decision: SemanticRouteDecision,
    ) -> FirstWordReply | None:
        """Return one filler-style bridge reply only when the fast LLM provides one."""

        reply = None
        route_first_word_instructions = build_local_route_first_word_instructions(
            decision.label,
            handoff_goal=_handoff_goal_for_label(decision.label),
            language_hint=_language_hint(self._loop),
        )
        if bool(getattr(self._loop.config, "streaming_first_word_enabled", False)):
            reply = self._loop._generate_first_word_reply(
                transcript,
                instructions=route_first_word_instructions,
            )
        spoken_text = (
            str(getattr(reply, "spoken_text", "") or "").strip()
            if reply is not None
            else ""
        )
        if spoken_text:
            reply_source = "first_word_llm"
        elif bool(getattr(self._loop.config, "streaming_first_word_enabled", False)):
            reply_source = "first_word_unavailable"
        else:
            reply_source = "first_word_disabled"
        if bool(getattr(self._loop.config, "local_semantic_router_trace", True)):
            self._loop._trace_event(
                "streaming_local_semantic_router_bridge_reply",
                kind="decision",
                details={
                    "route_label": decision.label,
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

    def _router_mode(self) -> str:
        mode = str(getattr(self._loop.config, "local_semantic_router_mode", "off") or "off").strip().lower()
        if mode not in {"off", "shadow", "gated"}:
            return "off"
        return mode


def _synthesize_supervisor_decision(
    route_decision: SemanticRouteDecision,
    bridge_reply: FirstWordReply | None,
) -> SupervisorDecision:
    """Map one authoritative local route into the supervisor-decision contract."""

    route_label = route_decision.label
    return SupervisorDecision(
        action="handoff",
        spoken_ack=bridge_reply.spoken_text if bridge_reply is not None else None,
        spoken_reply=None,
        kind=_handoff_kind_for_label(route_label),
        goal=_handoff_goal_for_label(route_label),
        allow_web_search=(route_label == "web"),
        context_scope=(
            "full_context"
            if route_label == "memory"
            else "tiny_recent"
            if route_label == "tool"
            else None
        ),
        response_id=f"local_semantic_router:{route_label}",
        request_id=f"local_semantic_router:{route_label}:{_route_id_suffix(route_decision)}",
        model=route_decision.model_id,
        token_usage=None,
    )


def _handoff_kind_for_label(route_label: str) -> str:
    if route_label == "web":
        return "search"
    if route_label == "memory":
        return "memory"
    if route_label == "tool":
        return "general"
    raise ValueError(f"Local semantic router cannot synthesize handoff for {route_label!r}.")


def _handoff_goal_for_label(route_label: str) -> str:
    if route_label == "web":
        return "Use fresh external information and reply clearly."
    if route_label == "memory":
        return "Answer using the user's persisted or recent Twinr memory."
    if route_label == "tool":
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
        f"{route_decision.label}|{route_decision.confidence:.6f}|"
        f"{route_decision.margin:.6f}|{route_decision.model_id}"
    )
    return sha1(payload.encode("utf-8")).hexdigest()[:12]


def _text_summary(value: object) -> dict[str, object]:
    """Describe text safely for workflow tracing without leaking raw content."""

    normalized = str(value or "").strip()
    if not normalized:
        return {"present": False, "chars": 0, "words": 0, "sha12": None}
    return {
        "present": True,
        "chars": len(normalized),
        "words": len(normalized.split()),
        "sha12": sha1(normalized.encode("utf-8")).hexdigest()[:12],
    }


def _route_summary(route_decision: SemanticRouteDecision) -> dict[str, object]:
    """Return one redacted summary of a scored route decision."""

    return {
        "label": route_decision.label,
        "confidence": round(route_decision.confidence, 6),
        "margin": round(route_decision.margin, 6),
        "authoritative": route_decision.authoritative,
        "fallback_reason": route_decision.fallback_reason,
        "model_id": route_decision.model_id,
        "latency_ms": round(route_decision.latency_ms, 3),
    }


def _user_intent_summary(user_intent_decision: UserIntentDecision | None) -> dict[str, object] | None:
    """Return one redacted summary of a scored user-intent decision."""

    if user_intent_decision is None:
        return None
    return {
        "label": user_intent_decision.label,
        "confidence": round(user_intent_decision.confidence, 6),
        "margin": round(user_intent_decision.margin, 6),
        "model_id": user_intent_decision.model_id,
        "latency_ms": round(user_intent_decision.latency_ms, 3),
    }
