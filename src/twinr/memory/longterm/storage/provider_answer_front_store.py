"""Remote current-head storage for materialized live provider answer fronts.

The live provider path should consume prompt-ready long-term sections from a
small remote-authoritative current-head collection instead of rebuilding the
full retriever output inline. This store persists one bounded set of provider
answer fronts, where each front is split into explicit rendered section blocks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib

from twinr.memory.longterm.core.models import LongTermMemoryContext
from twinr.memory.longterm.storage._remote_current_records import LongTermRemoteCurrentRecordStore
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore

_PROVIDER_ANSWER_FRONT_SNAPSHOT_KIND = "provider_answer_fronts"
_FRONT_STATE_BLOCK_KIND = "front_state"
_CONTEXT_BLOCK_FIELDS: tuple[tuple[str, str], ...] = (
    ("subtext", "subtext_context"),
    ("durable", "durable_context"),
    ("episodic", "episodic_context"),
    ("midterm", "midterm_context"),
    ("graph", "graph_context"),
    ("conflict", "conflict_context"),
)


def _utc_now_iso() -> str:
    """Return a compact UTC timestamp for persisted front metadata."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_query_key(value: str | None) -> str:
    """Normalize one provider-front key while preserving semantic wording."""

    return " ".join(str(value or "").split()).strip()


def _front_id(front_key: str) -> str:
    """Return a deterministic identifier that stays URI-safe across writes."""

    return hashlib.blake2s(front_key.encode("utf-8"), digest_size=16).hexdigest()


def _coerce_generation(value: object) -> int:
    """Normalize persisted generation values to a non-negative integer."""

    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _coerce_head_order(value: object) -> tuple[str, ...]:
    """Normalize persisted front ordering metadata from one current head."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    ordered: list[str] = []
    for item in value:
        normalized = str(item or "").strip()
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class MaterializedProviderAnswerFrontRecord:
    """Describe one rendered live provider answer front."""

    front_id: str
    front_key: str
    aliases: tuple[str, ...]
    generation: int
    built_at: str
    context: LongTermMemoryContext


@dataclass(frozen=True, slots=True)
class _PersistedProviderFrontPayload:
    """Carry one stored block payload while rebuilding the current collection."""

    front_id: str
    front_key: str
    aliases: tuple[str, ...]
    generation: int
    built_at: str
    item_payloads: tuple[dict[str, object], ...]


class LongTermProviderAnswerFrontStore:
    """Persist bounded live-provider answer fronts through current-head records."""

    def __init__(
        self,
        remote_state: LongTermRemoteStateStore | None,
        *,
        max_fronts: int = 6,
    ) -> None:
        self._current_store = LongTermRemoteCurrentRecordStore(remote_state)
        self._max_fronts = max(1, int(max_fronts))

    def enabled(self) -> bool:
        """Return whether remote current-head persistence is available."""

        return self._current_store.enabled()

    def current_generation(self) -> int:
        """Return the current invalidation generation from the remote head."""

        head = self._current_store.load_current_head(snapshot_kind=_PROVIDER_ANSWER_FRONT_SNAPSHOT_KIND)
        if not isinstance(head, Mapping):
            return 0
        return _coerce_generation(head.get("generation"))

    def load_front(
        self,
        *,
        query_keys: Sequence[str],
    ) -> MaterializedProviderAnswerFrontRecord | None:
        """Load one matching materialized provider front for the given keys."""

        normalized_keys = tuple(
            normalized
            for raw_value in query_keys
            if (normalized := _normalize_query_key(raw_value))
        )
        if not normalized_keys or not self.enabled():
            return None
        head = self._current_store.load_current_head(snapshot_kind=_PROVIDER_ANSWER_FRONT_SNAPSHOT_KIND)
        if not isinstance(head, Mapping):
            return None
        generation = _coerce_generation(head.get("generation"))
        ordered_fronts = _coerce_head_order(head.get("front_order"))
        payloads = self._current_store.load_collection_payloads(
            snapshot_kind=_PROVIDER_ANSWER_FRONT_SNAPSHOT_KIND,
            head_payload=head,
        )
        grouped = self._group_payloads(payloads=payloads, generation=generation)
        if not grouped:
            return None
        remaining_order = list(ordered_fronts)
        remaining_order.extend(front_id for front_id in grouped if front_id not in remaining_order)
        for front_id in remaining_order:
            candidate = grouped.get(front_id)
            if candidate is None:
                continue
            candidate_keys = {candidate.front_key, *candidate.aliases}
            if not candidate_keys.intersection(normalized_keys):
                continue
            return MaterializedProviderAnswerFrontRecord(
                front_id=candidate.front_id,
                front_key=candidate.front_key,
                aliases=candidate.aliases,
                generation=candidate.generation,
                built_at=candidate.built_at,
                context=self._context_from_payloads(candidate.item_payloads),
            )
        return None

    def save_front(
        self,
        *,
        query_keys: Sequence[str],
        context: LongTermMemoryContext,
        built_at: str | None = None,
    ) -> MaterializedProviderAnswerFrontRecord:
        """Persist one provider answer front and publish it in the current head."""

        if not self.enabled():
            raise RuntimeError("provider answer front store is unavailable")
        aliases = tuple(
            normalized
            for raw_value in query_keys
            if (normalized := _normalize_query_key(raw_value))
        )
        if not aliases:
            raise ValueError("query_keys must include at least one non-empty key")
        front_key = aliases[0]
        front_id = _front_id(front_key)
        built_at_value = str(built_at or _utc_now_iso()).strip() or _utc_now_iso()
        head = self._current_store.load_current_head(snapshot_kind=_PROVIDER_ANSWER_FRONT_SNAPSHOT_KIND)
        generation = _coerce_generation(head.get("generation")) if isinstance(head, Mapping) else 0
        ordered_fronts = _coerce_head_order(head.get("front_order")) if isinstance(head, Mapping) else ()
        payloads = self._current_store.load_collection_payloads(
            snapshot_kind=_PROVIDER_ANSWER_FRONT_SNAPSHOT_KIND,
            head_payload=head,
        ) if isinstance(head, Mapping) else ()
        grouped = self._group_payloads(payloads=payloads, generation=generation)
        grouped[front_id] = self._serialize_front(
            front_id=front_id,
            front_key=front_key,
            aliases=aliases,
            generation=generation,
            built_at=built_at_value,
            context=context,
        )
        next_order: list[str] = [front_id]
        for existing_front_id in ordered_fronts:
            if existing_front_id in grouped and existing_front_id not in next_order:
                next_order.append(existing_front_id)
        for existing_front_id in grouped:
            if existing_front_id not in next_order:
                next_order.append(existing_front_id)
        kept_front_ids = tuple(next_order[: self._max_fronts])
        persisted_payloads = tuple(
            payload
            for existing_front_id in kept_front_ids
            for payload in grouped[existing_front_id].item_payloads
        )
        self._current_store.save_collection(
            snapshot_kind=_PROVIDER_ANSWER_FRONT_SNAPSHOT_KIND,
            item_payloads=persisted_payloads,
            item_id_getter=lambda payload: str(payload["item_id"]),
            metadata_builder=self._metadata_builder,
            content_builder=self._content_builder,
            head_fields={
                "generation": generation,
                "front_order": list(kept_front_ids),
                "front_count": len(kept_front_ids),
                "profile": "provider",
            },
            written_at=built_at_value,
        )
        return MaterializedProviderAnswerFrontRecord(
            front_id=front_id,
            front_key=front_key,
            aliases=aliases,
            generation=generation,
            built_at=built_at_value,
            context=context,
        )

    def invalidate(self, *, reason: str | None = None) -> int:
        """Drop all persisted provider fronts by advancing the current generation."""

        if not self.enabled():
            return 0
        next_generation = self.current_generation() + 1
        self._current_store.save_collection(
            snapshot_kind=_PROVIDER_ANSWER_FRONT_SNAPSHOT_KIND,
            item_payloads=(),
            item_id_getter=lambda payload: str(payload["item_id"]),
            metadata_builder=self._metadata_builder,
            content_builder=self._content_builder,
            head_fields={
                "generation": next_generation,
                "front_order": [],
                "front_count": 0,
                "profile": "provider",
                "invalidated_reason": str(reason or "").strip() or None,
            },
            written_at=_utc_now_iso(),
        )
        return next_generation

    def _serialize_front(
        self,
        *,
        front_id: str,
        front_key: str,
        aliases: Sequence[str],
        generation: int,
        built_at: str,
        context: LongTermMemoryContext,
    ) -> _PersistedProviderFrontPayload:
        """Convert one provider context into remote-answer front block records."""

        normalized_aliases = tuple(
            normalized
            for raw_value in aliases
            if (normalized := _normalize_query_key(raw_value))
        )
        item_payloads: list[dict[str, object]] = [
            {
                "item_id": f"{front_id}:{_FRONT_STATE_BLOCK_KIND}",
                "front_id": front_id,
                "front_key": front_key,
                "aliases": list(normalized_aliases),
                "block_kind": _FRONT_STATE_BLOCK_KIND,
                "generation": generation,
                "built_at": built_at,
                "rendered_text": "",
            }
        ]
        for block_kind, field_name in _CONTEXT_BLOCK_FIELDS:
            text = getattr(context, field_name, None)
            if not isinstance(text, str) or not text.strip():
                continue
            item_payloads.append(
                {
                    "item_id": f"{front_id}:{block_kind}",
                    "front_id": front_id,
                    "front_key": front_key,
                    "aliases": list(normalized_aliases),
                    "block_kind": block_kind,
                    "generation": generation,
                    "built_at": built_at,
                    "rendered_text": text,
                }
            )
        return _PersistedProviderFrontPayload(
            front_id=front_id,
            front_key=front_key,
            aliases=normalized_aliases,
            generation=generation,
            built_at=built_at,
            item_payloads=tuple(item_payloads),
        )

    def _group_payloads(
        self,
        *,
        payloads: Iterable[Mapping[str, object]],
        generation: int,
    ) -> dict[str, _PersistedProviderFrontPayload]:
        """Group raw remote payloads by front id for one active generation."""

        grouped: dict[str, list[dict[str, object]]] = {}
        front_keys: dict[str, str] = {}
        aliases_by_front: dict[str, tuple[str, ...]] = {}
        built_at_by_front: dict[str, str] = {}
        for raw_payload in payloads:
            payload = dict(raw_payload)
            front_id = str(payload.get("front_id") or "").strip()
            if not front_id:
                continue
            if _coerce_generation(payload.get("generation")) != generation:
                continue
            grouped.setdefault(front_id, []).append(payload)
            front_keys[front_id] = _normalize_query_key(payload.get("front_key"))
            aliases = payload.get("aliases")
            if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes, bytearray)):
                aliases_by_front[front_id] = tuple(
                    normalized
                    for value in aliases
                    if (normalized := _normalize_query_key(value))
                )
            built_at_by_front[front_id] = str(payload.get("built_at") or "").strip()
        return {
            front_id: _PersistedProviderFrontPayload(
                front_id=front_id,
                front_key=front_keys.get(front_id) or front_id,
                aliases=aliases_by_front.get(front_id, (front_keys.get(front_id) or front_id,)),
                generation=generation,
                built_at=built_at_by_front.get(front_id) or _utc_now_iso(),
                item_payloads=tuple(front_payloads),
            )
            for front_id, front_payloads in grouped.items()
        }

    def _context_from_payloads(self, payloads: Iterable[Mapping[str, object]]) -> LongTermMemoryContext:
        """Rebuild one provider context from the persisted answer-front blocks."""

        fields: dict[str, str] = {}
        for payload in payloads:
            block_kind = str(payload.get("block_kind") or "").strip()
            if block_kind == _FRONT_STATE_BLOCK_KIND:
                continue
            rendered_text = payload.get("rendered_text")
            if not isinstance(rendered_text, str) or not rendered_text.strip():
                continue
            for expected_kind, field_name in _CONTEXT_BLOCK_FIELDS:
                if block_kind == expected_kind:
                    fields[field_name] = rendered_text
                    break
        return LongTermMemoryContext(**fields)

    def _metadata_builder(self, payload: Mapping[str, object]) -> dict[str, object]:
        """Build small catalog metadata that keeps answer fronts inspectable."""

        rendered_text = payload.get("rendered_text")
        return {
            "front_id": str(payload.get("front_id") or "").strip(),
            "front_key": str(payload.get("front_key") or "").strip(),
            "aliases": list(payload.get("aliases") or ()),
            "block_kind": str(payload.get("block_kind") or "").strip(),
            "generation": _coerce_generation(payload.get("generation")),
            "updated_at": str(payload.get("built_at") or "").strip() or _utc_now_iso(),
            "has_rendered_text": isinstance(rendered_text, str) and bool(rendered_text.strip()),
        }

    def _content_builder(self, payload: Mapping[str, object]) -> str:
        """Provide searchable content for answer-front records when needed."""

        rendered_text = payload.get("rendered_text")
        if isinstance(rendered_text, str) and rendered_text.strip():
            return rendered_text
        aliases = payload.get("aliases")
        if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes, bytearray)):
            return " ".join(str(value or "").strip() for value in aliases if str(value or "").strip())
        return str(payload.get("front_key") or "").strip()

