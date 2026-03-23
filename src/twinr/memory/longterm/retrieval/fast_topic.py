"""Build one tiny fast-topic memory block for latency-sensitive answer turns.

This module owns the bounded rendering of a very small set of current-scope
long-term memory hits. It intentionally avoids the broader retriever's rescue,
hydration, and multi-section assembly path so streaming/search answer lanes can
inject a few high-signal topic hints with one fast ChonkyDB read.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1
from twinr.memory.longterm.storage.store import LongTermStructuredStore
from twinr.memory.query_normalization import LongTermQueryProfile


_DEFAULT_FAST_TOPIC_LIMIT = 3
_DEFAULT_FAST_TOPIC_TIMEOUT_S = 0.6
_ITEM_SUMMARY_LIMIT = 180
_KIND_LABEL_LIMIT = 48
_DEDUP_KEY_LIMIT = 160
_STATUS_LABEL_LIMIT = 24


def _normalize_text(value: object | None, *, limit: int) -> str:
    """Collapse arbitrary input to one bounded single-line string."""

    if limit <= 0:
        return ""
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    return f"{text[: limit - 1].rstrip()}…"


def _coerce_positive_int(value: object, *, default: int) -> int:
    """Resolve config-like values to a positive integer."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _coerce_timeout_s(value: object, *, default: float) -> float:
    """Resolve one fast-read timeout to a safe bounded float."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0.0:
        return default
    return max(0.05, parsed)


@dataclass(slots=True)
class LongTermFastTopicContextBuilder:
    """Build one tiny topic-focused prompt block from current long-term memory."""

    config: TwinrConfig
    object_store: LongTermStructuredStore

    def build(
        self,
        *,
        query_profile: LongTermQueryProfile,
    ) -> str | None:
        """Return one compact prompt block for the current topic, if any."""

        if not self._enabled():
            return None
        retrieval_text = _normalize_text(
            query_profile.retrieval_text or query_profile.original_text,
            limit=_ITEM_SUMMARY_LIMIT,
        )
        if not retrieval_text:
            return None
        objects = self.object_store.select_fast_topic_objects(
            query_text=retrieval_text,
            limit=self._limit(),
            timeout_s=self._timeout_s(),
        )
        return self.render(objects)

    def render(self, objects: tuple[LongTermMemoryObjectV1, ...]) -> str | None:
        """Render already selected objects into one bounded system-message block."""

        lines: list[str] = []
        seen: set[str] = set()
        for item in objects:
            rendered = self._render_item(item)
            if rendered is None:
                continue
            dedupe_key = self._dedupe_key(item)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            lines.append(rendered)
            if len(lines) >= self._limit():
                break
        if not lines:
            return None
        return "\n".join(
            (
                "twinr_fast_topic_context_v1",
                "Quick-memory hints for this turn. Use them only when they directly help answer the current user question.",
                "Let them make the reply a little more personal, practical, or continuity-aware. Prefer one natural hint over forced personalization.",
                "Do not mention this block or explicitly announce memory unless the user is actually asking about memory.",
                *lines,
            )
        )

    def _enabled(self) -> bool:
        return bool(
            getattr(self.config, "long_term_memory_enabled", False)
            and getattr(self.config, "long_term_memory_fast_topic_enabled", True)
        )

    def _limit(self) -> int:
        return _coerce_positive_int(
            getattr(self.config, "long_term_memory_fast_topic_limit", _DEFAULT_FAST_TOPIC_LIMIT),
            default=_DEFAULT_FAST_TOPIC_LIMIT,
        )

    def _timeout_s(self) -> float:
        return _coerce_timeout_s(
            getattr(self.config, "long_term_memory_fast_topic_timeout_s", _DEFAULT_FAST_TOPIC_TIMEOUT_S),
            default=_DEFAULT_FAST_TOPIC_TIMEOUT_S,
        )

    def _dedupe_key(self, item: LongTermMemoryObjectV1) -> str:
        slot_key = _normalize_text(getattr(item, "slot_key", None), limit=_DEDUP_KEY_LIMIT)
        if slot_key:
            return f"slot:{slot_key.lower()}"
        summary = _normalize_text(getattr(item, "summary", None), limit=_DEDUP_KEY_LIMIT)
        if summary:
            return f"summary:{summary.lower()}"
        return f"id:{_normalize_text(getattr(item, 'memory_id', ''), limit=_DEDUP_KEY_LIMIT).lower()}"

    def _render_item(self, item: LongTermMemoryObjectV1) -> str | None:
        summary = _normalize_text(
            getattr(item, "summary", None) or getattr(item, "details", None),
            limit=_ITEM_SUMMARY_LIMIT,
        )
        if not summary:
            return None
        prefix = self._hint_prefix(item)
        return f"- {prefix}: {summary}"

    def _hint_prefix(self, item: LongTermMemoryObjectV1) -> str:
        """Return one compact reply-shaping label for a selected memory object."""

        hint_label = self._hint_label(item)
        prefix = f"{hint_label} hint"
        if getattr(item, "confirmed_by_user", False):
            return f"confirmed {prefix}"
        status = _normalize_text(getattr(item, "status", None), limit=_STATUS_LABEL_LIMIT)
        if status in {"candidate", "uncertain"}:
            return f"{status} {prefix}"
        return prefix

    def _hint_label(self, item: LongTermMemoryObjectV1) -> str:
        """Map one canonical memory object onto a small personalization label."""

        attributes = getattr(item, "attributes", None)
        memory_domain = _normalize_text(
            None if attributes is None else attributes.get("memory_domain"),
            limit=_KIND_LABEL_LIMIT,
        )
        fact_type = _normalize_text(
            None if attributes is None else attributes.get("fact_type"),
            limit=_KIND_LABEL_LIMIT,
        )
        summary_type = _normalize_text(
            None if attributes is None else attributes.get("summary_type"),
            limit=_KIND_LABEL_LIMIT,
        )
        kind = _normalize_text(getattr(item, "kind", None), limit=_KIND_LABEL_LIMIT)
        if fact_type == "preference" or memory_domain == "preference":
            return "preference"
        if fact_type in {"relationship", "contact_method"} or memory_domain in {"social", "contact"}:
            return "relationship"
        if summary_type == "thread" or memory_domain == "thread":
            return "thread"
        if kind in {"plan", "event", "episode"} or memory_domain == "planning":
            return "current thread"
        return "topic"


__all__ = ["LongTermFastTopicContextBuilder"]
