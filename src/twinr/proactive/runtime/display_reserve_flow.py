"""Orchestrate the full ambient companion candidate flow for the right lane.

The HDMI reserve lane should be planned from one coherent companion flow
instead of multiple partially overlapping helper paths. This module is that
integration layer. It blends:

- structured personality/mindshare candidates
- durable memory clarification and continuity prompts
- slower reflection-derived summaries and midterm packets
- long-horizon reserve-lane learning from prior visible card outcomes

The result is a bounded, source-diverse card-surface pool rather than only raw
topic seeds. Scheduling, publishing, and LLM copy rewriting remain in their
dedicated modules, and the flow now exposes the selected-card seam as well so
eval harnesses can exercise the same upstream selection path before copy
rewrite. Broad seed-family diversity is normalized before semantic-topic
expansion, and a dedicated latent-snapshot backfill path widens semantic topic
breadth when stronger explicit loaders stay sparse.
"""

from __future__ import annotations

# CHANGELOG: 2026-03-29
# BUG-1: `_dedupe()` existed but was never used, so duplicate topics from memory /
#        world / snapshot loaders could crowd out the bounded right-lane surface.
# BUG-2: `max_items` was not enforced after expansion or after copy rewrite,
#        allowing practical card-budget overruns at the public API seam.
# BUG-3: Any exception in snapshot/world/memory/reflection/copy-rewrite loading
#        could abort the whole flow and blank the lane instead of degrading
#        gracefully to the remaining sources.
# SEC-1: Untrusted external text from world/discovery loaders reached the copy
#        rewrite path unsanitized, enabling practical indirect prompt-injection
#        and markup-injection attempts against downstream LLM rewriting.
# SEC-2: Arbitrary raw `generation_context` payloads could be forwarded further
#        than necessary, increasing privacy leakage and secret/PII propagation
#        risk from memory/world sources.
# IMP-1: Added source-aware fusion (RRF-style) plus MMR-like diversity reranking
#        so the selected seam maximizes coverage instead of raw salience only.
# IMP-2: Added optional OpenTelemetry spans and structured logging so production
#        deployments can measure loader failures, cardinality, and selection.
# BREAKING: `generation_context` is now minimized, scalarized, and sanitized
#           before returning candidates from this module. Downstream code that
#           relied on raw nested payloads must read them from the original
#           loaders instead of this orchestration layer.

import logging
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from time import perf_counter
from typing import Any

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import (
    AmbientDisplayImpulseCandidate,
    build_ambient_display_impulse_candidates,
)
from twinr.agent.personality.intelligence.store import RemoteStateWorldIntelligenceStore
from twinr.agent.personality.models import PersonalitySnapshot
from twinr.agent.personality.service import PersonalityContextService
from twinr.memory.longterm.runtime.service import LongTermMemoryService

from .display_reserve_expansion import expand_display_reserve_candidates
from .display_reserve_generation import DisplayReserveCopyGenerator
from .display_reserve_learning import (
    DisplayReserveLearningProfile,
    DisplayReserveLearningProfileBuilder,
)
from .display_reserve_memory import load_display_reserve_memory_candidates
from .display_reserve_reflection import load_display_reserve_reflection_candidates
from .display_reserve_snapshot_topics import load_display_reserve_snapshot_candidates
from .display_reserve_user_discovery import load_display_reserve_user_discovery_candidates
from .display_reserve_world import load_display_reserve_world_candidates

_LOG = logging.getLogger(__name__)

try:  # Optional frontier observability. Safe no-op when OTel is absent.
    # pylint: disable=import-error
    from opentelemetry import trace as _otel_trace
except Exception:  # pragma: no cover - optional dependency
    _otel_trace = None

_TRACER = _otel_trace.get_tracer(__name__) if _otel_trace is not None else None

_RRF_K = 60.0
_DIVERSITY_PENALTY = 0.38
_SAME_SOURCE_PENALTY = 0.06
_DUPLICATE_SIMILARITY_CUTOFF = 0.82
_MAX_TEXT_LEN = 220
_MAX_REASON_LEN = 160
_MAX_CONTEXT_ITEMS = 16
_MAX_CONTEXT_VALUE_LEN = 180
_MAX_CONTEXT_KEY_LEN = 64
_MAX_TOKEN_COUNT = 24
_MAX_SOURCE_EXPANSION = 4
_SOURCE_WEIGHTS: dict[str, float] = {
    "personality": 1.00,
    "world": 0.95,
    "discovery": 0.75,
    "memory": 0.95,
    "reflection": 0.85,
    "snapshot": 0.80,
}
_UNTRUSTED_SOURCES = frozenset({"world", "discovery"})
_SENSITIVE_KEY_RE = re.compile(
    r"(pass(word)?|secret|token|api[_-]?key|auth(orization)?|cookie|session|credential|"
    r"bearer|private|ssn|social[_-]?security|iban|credit|card|cvv|pin|email|phone|address)",
    re.IGNORECASE,
)
_PROMPT_INJECTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"follow\s+these\s+instructions",
        r"you\s+are\s+now\s+(in\s+)?developer\s+mode",
        r"system\s+prompt",
        r"reveal\s+(the\s+)?prompt",
        r"disregard\s+(all\s+)?prior\s+messages?",
        r"act\s+as\s+",
        r"bypass\s+safety",
        r"jailbreak",
        r"do\s+anything\s+now",
        r"<script\b",
        r"```",
    )
)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_TAG_RE = re.compile(r"<[^>]+>")


def _compact_text(value: object | None) -> str:
    """Collapse arbitrary text into one trimmed single line."""

    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = _CONTROL_CHARS_RE.sub(" ", normalized)
    return " ".join(normalized.split()).strip()


def _truncate_text(value: str, *, limit: int) -> str:
    """Truncate text without returning blank padding or trailing separators."""

    compact = _compact_text(value)
    if len(compact) <= limit:
        return compact
    return compact[: max(1, limit - 1)].rstrip(" ,;:-") + "…"


def _topic_key(value: object | None) -> str:
    """Return one stable dedupe/cooldown key."""

    return _compact_text(value).casefold()


def _candidate_sort_key(candidate: AmbientDisplayImpulseCandidate) -> tuple[float, str, str, str]:
    """Return one stable ranking key across all candidate families."""

    return (
        float(getattr(candidate, "salience", 0.0)),
        _compact_text(getattr(candidate, "attention_state", None)).casefold(),
        _compact_text(getattr(candidate, "action", None)).casefold(),
        _topic_key(getattr(candidate, "topic_key", None) or getattr(candidate, "headline", None)),
    )


def _coerce_candidate_sequence(value: object) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Convert supported loader outputs into one candidate tuple."""

    if value is None:
        return ()
    as_tuple = getattr(value, "as_tuple", None)
    if callable(as_tuple):
        value = as_tuple()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(item for item in value if isinstance(item, AmbientDisplayImpulseCandidate))
    return ()


@contextmanager
def _telemetry_span(name: str, /, **attributes: object):
    """Create an optional OpenTelemetry span without hard dependency."""

    if _TRACER is None:
        yield None
        return
    with _TRACER.start_as_current_span(name) as span:  # pragma: no branch - tiny wrapper
        for key, value in attributes.items():
            if value is None:
                continue
            try:
                span.set_attribute(key, value)
            except Exception:
                continue
        yield span


@dataclass(frozen=True, slots=True)
class DisplayReserveCompanionFlowContext:
    """Collect the slower companion-flow inputs for one candidate pass."""

    snapshot: PersonalitySnapshot | None
    learning_profile: DisplayReserveLearningProfile


@dataclass(slots=True)
class _NoopLearningProfile:
    """Fallback learning profile used when the real builder fails."""

    def candidate_adjustment(self, candidate: AmbientDisplayImpulseCandidate) -> float:
        return 0.0

    def context_for_candidate(self, candidate: AmbientDisplayImpulseCandidate) -> dict[str, object]:
        return {}


@dataclass(slots=True)
class _SelectionRecord:
    candidate: AmbientDisplayImpulseCandidate
    source: str
    fusion_score: float = 0.0
    contributing_sources: set[str] = field(default_factory=set)
    source_rank: int = 0
    source_weight: float = 0.0


@dataclass(slots=True)
class DisplayReserveCompanionFlow:
    """Blend ambient reserve candidates from personality, memory, and reflection."""

    personality_service: PersonalityContextService = field(default_factory=PersonalityContextService)
    world_store: RemoteStateWorldIntelligenceStore = field(default_factory=RemoteStateWorldIntelligenceStore)
    learning_builder_factory: type[DisplayReserveLearningProfileBuilder] = DisplayReserveLearningProfileBuilder
    copy_generator: DisplayReserveCopyGenerator = field(default_factory=DisplayReserveCopyGenerator)

    def load_candidates(
        self,
        config: TwinrConfig,
        *,
        local_now: datetime,
        max_items: int,
    ) -> tuple[AmbientDisplayImpulseCandidate, ...]:
        """Return rewritten reserve candidates for the visible right-lane plan."""

        limited_max = max(1, int(max_items))
        with _telemetry_span(
            "twinr.display_reserve_companion.load_candidates",
            max_items=limited_max,
        ):
            snapshot, selected = self.load_raw_candidates(
                config,
                local_now=local_now,
                max_items=limited_max,
            )
            rewritten = self._safe_call(
                "copy_rewrite",
                self.copy_generator.rewrite_candidates,
                config=config,
                snapshot=snapshot,
                candidates=selected,
                local_now=local_now,
                default=selected,
            )
            final_candidates = self._finalize_candidates(
                candidates=self._sanitize_candidates(
                    _coerce_candidate_sequence(rewritten),
                    source="rewrite",
                    trust="trusted",
                ),
                max_items=limited_max,
            )
            _LOG.debug(
                "display reserve companion flow produced %s candidates after rewrite",
                len(final_candidates),
            )
            return final_candidates

    def load_raw_candidates(
        self,
        config: TwinrConfig,
        *,
        local_now: datetime,
        max_items: int,
    ) -> tuple[PersonalitySnapshot | None, tuple[AmbientDisplayImpulseCandidate, ...]]:
        """Return snapshot plus selected reserve candidates before copy rewrite."""

        limited_max = max(1, int(max_items))
        with _telemetry_span(
            "twinr.display_reserve_companion.load_raw_candidates",
            max_items=limited_max,
        ):
            started_at = perf_counter()

            snapshot = self._safe_call(
                "load_snapshot",
                self.personality_service.load_snapshot,
                config=config,
                default=None,
            )
            engagement_signals = self._safe_call(
                "load_engagement_signals",
                self.personality_service.load_engagement_signals,
                config=config,
                default=None,
            )
            world_state = self._safe_call(
                "load_world_state",
                self.world_store.load_state,
                config=config,
                default=None,
            )
            world_subscriptions = self._safe_call(
                "load_world_subscriptions",
                self.world_store.load_subscriptions,
                config=config,
                default=(),
            )
            learning_builder = self._safe_call(
                "learning_builder_from_config",
                self.learning_builder_factory.from_config,
                config,
                default=None,
            )
            learning_profile = (
                self._safe_call(
                    "build_learning_profile",
                    learning_builder.build,
                    now=local_now,
                    default=_NoopLearningProfile(),
                )
                if learning_builder is not None
                else _NoopLearningProfile()
            )
            memory_service = self._safe_call(
                "create_memory_service",
                LongTermMemoryService.from_config,
                config,
                default=None,
            )

            source_families: dict[str, tuple[AmbientDisplayImpulseCandidate, ...]] = {}

            source_families["personality"] = self._sanitize_candidates(
                _coerce_candidate_sequence(
                    self._safe_call(
                        "load_personality_candidates",
                        build_ambient_display_impulse_candidates,
                        snapshot,
                        engagement_signals=engagement_signals,
                        local_now=local_now,
                        max_items=max(limited_max, 6),
                        default=(),
                    )
                ),
                source="personality",
                trust="trusted",
            )

            source_families["world"] = self._sanitize_candidates(
                _coerce_candidate_sequence(
                    self._safe_call(
                        "load_world_candidates",
                        load_display_reserve_world_candidates,
                        subscriptions=world_subscriptions,
                        state=world_state,
                        max_items=max(limited_max, 8),
                        default=(),
                    )
                ),
                source="world",
                trust="untrusted",
            )

            source_families["discovery"] = self._sanitize_candidates(
                _coerce_candidate_sequence(
                    self._safe_call(
                        "load_user_discovery_candidates",
                        load_display_reserve_user_discovery_candidates,
                        config,
                        local_now=local_now,
                        max_items=1,
                        default=(),
                    )
                ),
                source="discovery",
                trust="untrusted",
            )

            if memory_service is not None:
                conflict_queue = self._safe_call(
                    "memory_conflict_queue",
                    memory_service.select_conflict_queue,
                    query_text=None,
                    limit=min(limited_max, 4),
                    default=(),
                )
                proactive_plan = self._safe_call(
                    "memory_proactive_plan",
                    memory_service.plan_proactive_candidates,
                    now=local_now,
                    default=None,
                )
                proactive_candidates = getattr(proactive_plan, "candidates", ()) if proactive_plan is not None else ()
                memory_loaded = self._safe_call(
                    "load_memory_candidates",
                    load_display_reserve_memory_candidates,
                    conflicts=conflict_queue,
                    proactive_candidates=proactive_candidates,
                    max_items=max(limited_max, 4),
                    default=(),
                )
                source_families["memory"] = self._sanitize_candidates(
                    _coerce_candidate_sequence(memory_loaded),
                    source="memory",
                    trust="trusted",
                )
                source_families["reflection"] = self._sanitize_candidates(
                    _coerce_candidate_sequence(
                        self._safe_call(
                            "load_reflection_candidates",
                            load_display_reserve_reflection_candidates,
                            memory_service,
                            config=config,
                            local_now=local_now,
                            max_items=max(limited_max, 4),
                            default=(),
                        )
                    ),
                    source="reflection",
                    trust="trusted",
                )
            else:
                source_families["memory"] = ()
                source_families["reflection"] = ()

            exclude_topic_keys = tuple(
                key
                for key in (
                    self._semantic_candidate_key(candidate)
                    for candidate in self._flatten_families(source_families)
                )
                if key
            )
            source_families["snapshot"] = self._sanitize_candidates(
                _coerce_candidate_sequence(
                    self._safe_call(
                        "load_snapshot_candidates",
                        load_display_reserve_snapshot_candidates,
                        snapshot,
                        engagement_signals=engagement_signals,
                        exclude_topic_keys=exclude_topic_keys,
                        max_items=max(limited_max, 12),
                        default=(),
                    )
                ),
                source="snapshot",
                trust="trusted",
            )

            learned_families = {
                source: tuple(
                    self._apply_learning_profile(candidate, profile=learning_profile)
                    for candidate in candidates
                )
                for source, candidates in source_families.items()
            }

            selected = self._select_candidates(
                source_families=learned_families,
                max_items=limited_max,
            )

            elapsed_ms = round((perf_counter() - started_at) * 1000.0, 2)
            _LOG.debug(
                "display reserve companion flow loaded %s raw candidates across %s sources in %sms",
                len(selected),
                sum(1 for items in learned_families.values() if items),
                elapsed_ms,
            )
            return snapshot, selected

    def _safe_call(
        self,
        label: str,
        fn: Any,
        /,
        *args: object,
        default: Any,
        **kwargs: object,
    ) -> Any:
        """Execute one loader without letting one failure blank the full lane."""

        started_at = perf_counter()
        try:
            with _telemetry_span(f"twinr.display_reserve_companion.{label}"):
                result = fn(*args, **kwargs)
        except Exception:
            _LOG.exception("display reserve companion flow step failed: %s", label)
            return default

        elapsed_ms = round((perf_counter() - started_at) * 1000.0, 2)
        _LOG.debug("display reserve companion flow step %s completed in %sms", label, elapsed_ms)
        return result

    def _apply_learning_profile(
        self,
        candidate: AmbientDisplayImpulseCandidate,
        *,
        profile: DisplayReserveLearningProfile,
    ) -> AmbientDisplayImpulseCandidate:
        """Bias one candidate by long-horizon reserve-lane learning."""

        adjustment = float(profile.candidate_adjustment(candidate))
        updated_context = dict(candidate.generation_context or {})
        learning_context = profile.context_for_candidate(candidate)
        learning_summary = self._summarize_learning_context(learning_context)
        if learning_summary:
            updated_context["ambient_learning"] = learning_summary
        updated_context["ambient_learning_adjustment"] = round(adjustment, 4)
        updated_reason = _truncate_text(
            f"{candidate.reason}; ambient_learning={adjustment:+.2f}",
            limit=_MAX_REASON_LEN,
        )
        return replace(
            candidate,
            salience=max(0.0, min(1.25, float(candidate.salience) + adjustment)),
            reason=updated_reason or candidate.reason,
            generation_context=updated_context,
        )

    def _summarize_learning_context(self, value: object) -> str:
        """Convert learning context into a bounded scalar summary."""

        if isinstance(value, Mapping):
            parts: list[str] = []
            for raw_key, raw_value in list(value.items())[:4]:
                key = _truncate_text(_compact_text(raw_key), limit=24)
                scalar = self._scalarize_context_value(raw_value)
                if not key or scalar is None:
                    continue
                parts.append(f"{key}={_truncate_text(_compact_text(scalar), limit=36)}")
            return _truncate_text(" | ".join(parts), limit=_MAX_CONTEXT_VALUE_LEN)
        scalar = self._scalarize_context_value(value)
        return _truncate_text(_compact_text(scalar), limit=_MAX_CONTEXT_VALUE_LEN) if scalar is not None else ""

    def _sanitize_candidates(
        self,
        candidates: Sequence[AmbientDisplayImpulseCandidate],
        *,
        source: str,
        trust: str,
    ) -> tuple[AmbientDisplayImpulseCandidate, ...]:
        """Normalize candidate text and minimize context before selection/rewrite."""

        sanitized: list[AmbientDisplayImpulseCandidate] = []
        for rank, candidate in enumerate(candidates):
            sanitized.append(
                self._sanitize_candidate(
                    candidate,
                    source=source,
                    trust=trust,
                    source_rank=rank,
                )
            )
        return tuple(sanitized)

    def _sanitize_candidate(
        self,
        candidate: AmbientDisplayImpulseCandidate,
        *,
        source: str,
        trust: str,
        source_rank: int,
    ) -> AmbientDisplayImpulseCandidate:
        """Sanitize one candidate and tag it with source-aware metadata."""

        trust = trust if trust in {"trusted", "untrusted"} else "trusted"
        raw_text = " | ".join(
            (
                _compact_text(getattr(candidate, "headline", None)),
                _compact_text(getattr(candidate, "action", None)),
                _compact_text(getattr(candidate, "reason", None)),
            )
        )
        injection_flag = trust == "untrusted" and self._looks_like_prompt_injection(raw_text)
        salience_penalty = 0.12 if injection_flag else 0.0
        updated_context = self._sanitize_generation_context(
            candidate.generation_context,
            source=source,
            trust=trust,
            injection_flag=injection_flag,
            source_rank=source_rank,
        )

        headline = self._sanitize_text(
            getattr(candidate, "headline", None),
            limit=_MAX_TEXT_LEN,
            aggressive=injection_flag,
        ) or _sanitize_topic_fallback(candidate)
        topic_key = self._sanitize_text(
            getattr(candidate, "topic_key", None),
            limit=_MAX_TEXT_LEN,
            aggressive=injection_flag,
        ) or headline
        attention_state = self._sanitize_text(
            getattr(candidate, "attention_state", None),
            limit=96,
            aggressive=injection_flag,
        ) or "ambient"
        action = self._sanitize_text(
            getattr(candidate, "action", None),
            limit=96,
            aggressive=injection_flag,
        ) or "notice"
        reason = self._sanitize_text(
            getattr(candidate, "reason", None),
            limit=_MAX_REASON_LEN,
            aggressive=injection_flag,
        )
        if injection_flag:
            reason = _truncate_text(
                f"{reason or 'sanitized external content'}; prompt_injection_filtered",
                limit=_MAX_REASON_LEN,
            )

        return replace(
            candidate,
            headline=headline,
            topic_key=topic_key,
            attention_state=attention_state,
            action=action,
            reason=reason or "selected",
            salience=max(0.0, min(1.25, float(candidate.salience) - salience_penalty)),
            generation_context=updated_context,
        )

    def _sanitize_generation_context(
        self,
        context: Mapping[str, object] | None,
        *,
        source: str,
        trust: str,
        injection_flag: bool,
        source_rank: int,
    ) -> dict[str, object]:
        """Strip nested/raw payloads and keep only bounded scalar context."""

        existing_source = ""
        existing_trust = ""
        existing_source_rank = source_rank
        if isinstance(context, Mapping):
            existing_source = _compact_text(context.get("ambient_source"))
            existing_trust = _compact_text(context.get("ambient_trust"))
            try:
                existing_source_rank = int(context.get("ambient_source_rank", source_rank))
            except Exception:
                existing_source_rank = source_rank

        sanitized: dict[str, object] = {
            "ambient_source": existing_source or source,
            "ambient_trust": existing_trust or trust,
            "ambient_source_rank": existing_source_rank,
            "ambient_stage": source,
        }
        if injection_flag:
            sanitized["ambient_prompt_injection_filtered"] = True

        if not isinstance(context, Mapping):
            return sanitized

        kept = 0
        for raw_key, raw_value in context.items():
            if kept >= _MAX_CONTEXT_ITEMS:
                break
            key = _truncate_text(_compact_text(raw_key), limit=_MAX_CONTEXT_KEY_LEN)
            if not key or _SENSITIVE_KEY_RE.search(key):
                continue
            scalar = self._scalarize_context_value(raw_value)
            if scalar is None:
                continue
            if isinstance(scalar, str):
                scalar = self._sanitize_text(
                    scalar,
                    limit=_MAX_CONTEXT_VALUE_LEN,
                    aggressive=source in _UNTRUSTED_SOURCES,
                )
                if not scalar:
                    continue
            sanitized[key] = scalar
            kept += 1
        return sanitized

    def _scalarize_context_value(self, value: object) -> str | int | float | bool | None:
        """Convert context values to bounded scalars so prompts stay small and safe."""

        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return round(value, 6)
        if isinstance(value, str):
            return _truncate_text(value, limit=_MAX_CONTEXT_VALUE_LEN)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            parts: list[str] = []
            for item in value[:4]:
                compact = _truncate_text(_compact_text(item), limit=48)
                if compact:
                    parts.append(compact)
            joined = " | ".join(parts)
            return _truncate_text(joined, limit=_MAX_CONTEXT_VALUE_LEN) if joined else None
        return None

    def _sanitize_text(self, value: object | None, *, limit: int, aggressive: bool) -> str:
        """Normalize text, remove basic markup, and redact prompt-injection phrases."""

        text = _compact_text(value)
        if not text:
            return ""
        text = _TAG_RE.sub(" ", text)
        text = text.replace("```", " ").replace("`", "'")
        text = text.replace("<<", " ").replace(">>", " ")
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(text):
                replacement = "[filtered external instruction]" if aggressive else " "
                text = pattern.sub(replacement, text)
        return _truncate_text(text, limit=limit)

    def _looks_like_prompt_injection(self, value: str) -> bool:
        """Detect high-signal indirect prompt-injection phrases in untrusted text."""

        text = _compact_text(value)
        if not text:
            return False
        return any(pattern.search(text) for pattern in _PROMPT_INJECTION_PATTERNS)

    def _semantic_candidate_key(self, candidate: AmbientDisplayImpulseCandidate) -> str:
        """Return the grouped semantic key used to fuse raw topic seeds."""

        semantic_key_fn = getattr(candidate, "semantic_key", None)
        semantic_key = semantic_key_fn() if callable(semantic_key_fn) else None
        return (
            _topic_key(semantic_key)
            or _topic_key(getattr(candidate, "topic_key", None))
            or _topic_key(getattr(candidate, "headline", None))
            or _topic_key(getattr(candidate, "action", None))
        )

    def _card_identity_key(self, candidate: AmbientDisplayImpulseCandidate) -> str:
        """Return the concrete card key used for expanded/output identities."""

        topic_key = _topic_key(getattr(candidate, "topic_key", None))
        return topic_key or self._semantic_candidate_key(candidate)

    def _flatten_families(
        self,
        source_families: Mapping[str, Sequence[AmbientDisplayImpulseCandidate]],
    ) -> tuple[AmbientDisplayImpulseCandidate, ...]:
        """Flatten source families while preserving their internal order."""

        flattened: list[AmbientDisplayImpulseCandidate] = []
        for candidates in source_families.values():
            flattened.extend(candidates)
        return tuple(flattened)

    def _select_candidates(
        self,
        *,
        source_families: Mapping[str, Sequence[AmbientDisplayImpulseCandidate]],
        max_items: int,
    ) -> tuple[AmbientDisplayImpulseCandidate, ...]:
        """Fuse, dedupe, expand, diversify, and clip candidates to the public seam."""

        max_items = max(1, int(max_items))
        fused_records = self._fuse_source_families(source_families)
        if not fused_records:
            return ()

        seed_pool = tuple(record.candidate for record in fused_records.values())
        expanded = _coerce_candidate_sequence(
            self._safe_call(
                "expand_candidates",
                expand_display_reserve_candidates,
                seed_pool,
                target_cards=max(max_items, min(len(seed_pool), max_items * _MAX_SOURCE_EXPANSION)),
                default=seed_pool,
            )
        )
        if not expanded:
            expanded = seed_pool

        expanded = self._sanitize_candidates(
            expanded,
            source="expanded",
            trust="trusted",
        )

        selection_records = self._expanded_selection_records(
            expanded=expanded,
            fused_records=fused_records,
        )
        diversified = self._diversify_records(
            records=self._semantic_topic_representatives(selection_records),
            max_items=min(max_items, len(selection_records)),
        )
        if len(diversified) < max_items:
            selected_card_keys = {
                self._card_identity_key(record.candidate)
                for record in diversified
                if self._card_identity_key(record.candidate)
            }
            overflow_records = tuple(
                record
                for record in selection_records
                if self._card_identity_key(record.candidate) not in selected_card_keys
            )
            diversified = self._diversify_records(
                records=overflow_records,
                max_items=max_items,
                initial_selected=diversified,
            )
        return tuple(record.candidate for record in diversified[:max_items])

    def _expanded_selection_records(
        self,
        *,
        expanded: Sequence[AmbientDisplayImpulseCandidate],
        fused_records: Mapping[str, _SelectionRecord],
    ) -> tuple[_SelectionRecord, ...]:
        """Carry fused topic metadata onto concrete expanded reserve cards."""

        if not expanded:
            return tuple(fused_records.values())

        expanded_records: dict[str, _SelectionRecord] = {}
        represented_semantics: set[str] = set()
        for candidate in expanded:
            semantic_key = self._semantic_candidate_key(candidate)
            card_key = self._card_identity_key(candidate)
            if not card_key:
                continue
            base_record = fused_records.get(semantic_key)
            if base_record is None:
                source = str((candidate.generation_context or {}).get("ambient_source", "expanded"))
                record = _SelectionRecord(
                    candidate=candidate,
                    source=source,
                    fusion_score=0.0,
                    contributing_sources={source},
                    source_rank=int((candidate.generation_context or {}).get("ambient_source_rank", 0)),
                    source_weight=_SOURCE_WEIGHTS.get(source, 0.5),
                )
            else:
                represented_semantics.add(semantic_key)
                record = replace(base_record, candidate=candidate)
            current = expanded_records.get(card_key)
            if current is None or _candidate_sort_key(candidate) > _candidate_sort_key(current.candidate):
                expanded_records[card_key] = record

        for semantic_key, record in fused_records.items():
            if semantic_key in represented_semantics:
                continue
            fallback_candidate = self._primary_fallback_candidate(record.candidate)
            fallback_key = self._card_identity_key(fallback_candidate)
            if not fallback_key or fallback_key in expanded_records:
                continue
            expanded_records[fallback_key] = replace(record, candidate=fallback_candidate)

        return tuple(expanded_records.values())

    def _primary_fallback_candidate(
        self,
        candidate: AmbientDisplayImpulseCandidate,
    ) -> AmbientDisplayImpulseCandidate:
        """Promote one unexpanded topic seed into a primary-card fallback."""

        if _compact_text(getattr(candidate, "expansion_angle", None)).casefold() == "primary":
            return candidate
        context = dict(candidate.generation_context or {})
        context.setdefault("semantic_topic_key", candidate.semantic_key())
        context["expansion_angle"] = "primary"
        return replace(
            candidate,
            expansion_angle="primary",
            generation_context=context,
        )

    def _semantic_representative_sort_key(
        self,
        candidate: AmbientDisplayImpulseCandidate,
    ) -> tuple[int, float, str, str]:
        """Prefer one breadth-first representative card for each semantic topic."""

        angle = _compact_text(getattr(candidate, "expansion_angle", None)).casefold()
        angle_priority = 0 if angle == "primary" else 1 if not angle else 2
        return (
            angle_priority,
            -float(getattr(candidate, "salience", 0.0)),
            angle,
            self._card_identity_key(candidate),
        )

    def _semantic_topic_representatives(
        self,
        records: Sequence[_SelectionRecord],
    ) -> tuple[_SelectionRecord, ...]:
        """Keep one breadth-first representative card per semantic topic."""

        representatives: dict[str, _SelectionRecord] = {}
        for record in records:
            semantic_key = self._semantic_candidate_key(record.candidate)
            if not semantic_key:
                semantic_key = self._card_identity_key(record.candidate)
            if not semantic_key:
                continue
            current = representatives.get(semantic_key)
            if current is None or self._semantic_representative_sort_key(record.candidate) < self._semantic_representative_sort_key(current.candidate):
                representatives[semantic_key] = record
        return tuple(representatives.values())

    def _fuse_source_families(
        self,
        source_families: Mapping[str, Sequence[AmbientDisplayImpulseCandidate]],
    ) -> dict[str, _SelectionRecord]:
        """Apply RRF-style source fusion across ranked candidate families."""

        fused: dict[str, _SelectionRecord] = {}
        for source, family in source_families.items():
            if not family:
                continue
            source_weight = _SOURCE_WEIGHTS.get(source, 0.5)
            ranked = sorted(family, key=_candidate_sort_key, reverse=True)
            for rank, candidate in enumerate(ranked, start=1):
                key = self._semantic_candidate_key(candidate)
                if not key:
                    continue
                fusion_score = source_weight / (_RRF_K + float(rank))
                current = fused.get(key)
                if current is None:
                    fused[key] = _SelectionRecord(
                        candidate=self._annotate_candidate_for_fusion(
                            candidate,
                            fusion_score=fusion_score,
                            source=source,
                            contributing_sources={source},
                        ),
                        source=source,
                        fusion_score=fusion_score,
                        contributing_sources={source},
                        source_rank=rank,
                        source_weight=source_weight,
                    )
                    continue
                best_candidate = current.candidate
                primary_source = current.source
                if _candidate_sort_key(candidate) > _candidate_sort_key(current.candidate):
                    best_candidate = candidate
                    primary_source = source
                contributing_sources = set(current.contributing_sources)
                contributing_sources.add(source)
                total_fusion = current.fusion_score + fusion_score
                fused[key] = _SelectionRecord(
                    candidate=self._annotate_candidate_for_fusion(
                        best_candidate,
                        fusion_score=total_fusion,
                        source=primary_source,
                        contributing_sources=contributing_sources,
                    ),
                    source=primary_source,
                    fusion_score=total_fusion,
                    contributing_sources=contributing_sources,
                    source_rank=min(current.source_rank, rank),
                    source_weight=max(current.source_weight, source_weight),
                )
        return fused

    def _annotate_candidate_for_fusion(
        self,
        candidate: AmbientDisplayImpulseCandidate,
        *,
        fusion_score: float,
        source: str,
        contributing_sources: set[str],
    ) -> AmbientDisplayImpulseCandidate:
        """Persist source-fusion metadata on the candidate for downstream use."""

        context = dict(candidate.generation_context or {})
        context["ambient_fusion_score"] = round(fusion_score, 6)
        context["ambient_contributing_sources"] = tuple(sorted(contributing_sources))
        context.setdefault("ambient_source", source)
        return replace(candidate, generation_context=context)

    def _diversify_records(
        self,
        *,
        records: Sequence[_SelectionRecord],
        max_items: int,
        initial_selected: Sequence[_SelectionRecord] = (),
    ) -> tuple[_SelectionRecord, ...]:
        """Apply MMR-like greedy diversification over the fused candidate pool."""

        max_items = max(0, int(max_items))
        if max_items <= 0:
            return ()

        pending = list(records)
        selected: list[_SelectionRecord] = list(initial_selected[:max_items])
        source_usage: Counter[str] = Counter(record.source for record in selected)

        while pending and len(selected) < max_items:
            best_index = max(
                range(len(pending)),
                key=lambda index: self._selection_score(
                    pending[index],
                    selected=selected,
                    source_usage=source_usage,
                ),
            )
            chosen = pending.pop(best_index)
            if self._is_near_duplicate(chosen, selected):
                continue
            selected.append(chosen)
            source_usage[chosen.source] += 1

        if len(selected) < max_items:
            leftovers = [
                record
                for record in pending
                if not self._is_near_duplicate(record, selected)
            ]
            leftovers.sort(
                key=lambda record: self._selection_score(record, selected=selected, source_usage=source_usage),
                reverse=True,
            )
            selected.extend(leftovers[: max_items - len(selected)])

        return tuple(selected)

    def _selection_score(
        self,
        record: _SelectionRecord,
        *,
        selected: Sequence[_SelectionRecord],
        source_usage: Mapping[str, int],
    ) -> float:
        """Compute one diversified selection score for a candidate."""

        candidate = record.candidate
        context = candidate.generation_context or {}
        base = float(getattr(candidate, "salience", 0.0))
        fusion_bonus = float(context.get("ambient_fusion_score", record.fusion_score or 0.0)) * 8.0
        multi_source_bonus = 0.04 * max(0, len(record.contributing_sources) - 1)
        source_penalty = _SAME_SOURCE_PENALTY * float(source_usage.get(record.source, 0))
        trust_penalty = 0.08 if context.get("ambient_prompt_injection_filtered") else 0.0
        diversity_penalty = 0.0
        if selected:
            diversity_penalty = _DIVERSITY_PENALTY * max(
                self._candidate_similarity(candidate, prior.candidate) for prior in selected
            )
        return base + fusion_bonus + multi_source_bonus - source_penalty - trust_penalty - diversity_penalty

    def _is_near_duplicate(
        self,
        record: _SelectionRecord,
        selected: Sequence[_SelectionRecord],
    ) -> bool:
        """Reject candidates that are effectively the same surface as an already kept one."""

        candidate = record.candidate
        return any(
            self._candidate_similarity(candidate, prior.candidate) >= _DUPLICATE_SIMILARITY_CUTOFF
            for prior in selected
        )

    def _candidate_similarity(
        self,
        left: AmbientDisplayImpulseCandidate,
        right: AmbientDisplayImpulseCandidate,
    ) -> float:
        """Estimate lexical/semantic overlap cheaply enough for Pi 4 reranking."""

        left_key = self._card_identity_key(left)
        right_key = self._card_identity_key(right)
        if left_key and left_key == right_key:
            return 1.0

        left_tokens = self._candidate_tokens(left)
        right_tokens = self._candidate_tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0

        overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        if _topic_key(getattr(left, "attention_state", None)) == _topic_key(getattr(right, "attention_state", None)):
            overlap += 0.05
        if _topic_key(getattr(left, "action", None)) == _topic_key(getattr(right, "action", None)):
            overlap += 0.05
        return max(0.0, min(1.0, overlap))

    def _candidate_tokens(
        self,
        candidate: AmbientDisplayImpulseCandidate,
    ) -> frozenset[str]:
        """Build a bounded token set for cheap similarity comparisons."""

        text = " ".join(
            part
            for part in (
                self._semantic_candidate_key(candidate),
                _compact_text(getattr(candidate, "headline", None)),
                _compact_text(getattr(candidate, "action", None)),
            )
            if part
        )
        tokens = [
            token.casefold()
            for token in re.findall(r"\w+", text, flags=re.UNICODE)
            if len(token) >= 3
        ]
        return frozenset(tokens[:_MAX_TOKEN_COUNT])

    def _finalize_candidates(
        self,
        *,
        candidates: Sequence[AmbientDisplayImpulseCandidate],
        max_items: int,
    ) -> tuple[AmbientDisplayImpulseCandidate, ...]:
        """Dedupe and clip the public output seam one final time."""

        ordered: list[AmbientDisplayImpulseCandidate] = []
        seen: set[str] = set()
        overflow: list[AmbientDisplayImpulseCandidate] = []

        for candidate in candidates:
            key = self._card_identity_key(candidate)
            if not key:
                continue
            if key in seen:
                overflow.append(candidate)
                continue
            seen.add(key)
            ordered.append(candidate)
            if len(ordered) >= max(1, int(max_items)):
                return tuple(ordered)

        if len(ordered) < max(1, int(max_items)):
            deduped_overflow = self._dedupe(overflow)
            for candidate in sorted(deduped_overflow.values(), key=_candidate_sort_key, reverse=True):
                key = self._card_identity_key(candidate)
                if not key or key in seen:
                    continue
                seen.add(key)
                ordered.append(candidate)
                if len(ordered) >= max(1, int(max_items)):
                    break

        return tuple(ordered)

    def _dedupe(
        self,
        candidates: Sequence[AmbientDisplayImpulseCandidate],
    ) -> dict[str, AmbientDisplayImpulseCandidate]:
        """Keep the strongest candidate for each stable topic key."""

        deduped: dict[str, AmbientDisplayImpulseCandidate] = {}
        for candidate in candidates:
            key = self._card_identity_key(candidate)
            if not key:
                continue
            current = deduped.get(key)
            if current is None or _candidate_sort_key(candidate) > _candidate_sort_key(current):
                deduped[key] = candidate
        return deduped


def _sanitize_topic_fallback(candidate: AmbientDisplayImpulseCandidate) -> str:
    """Produce one safe fallback headline when sanitization removes unsafe text."""

    return (
        _truncate_text(_compact_text(getattr(candidate, "topic_key", None)), limit=_MAX_TEXT_LEN)
        or _truncate_text(_compact_text(getattr(candidate, "headline", None)), limit=_MAX_TEXT_LEN)
        or "Ambient update"
    )


__all__ = ["DisplayReserveCompanionFlow", "DisplayReserveCompanionFlowContext"]
