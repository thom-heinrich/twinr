# CHANGELOG: 2026-03-27
# BUG-1: Stop treating malformed subscription envelopes as "empty success"; corrupt payloads are now detected and quarantined instead of silently erasing subscriptions.
# BUG-2: Stop failing the entire subscription restore on one bad item; invalid records are isolated and skipped so healthy subscriptions still load.
# BUG-3: Stop letting malformed state payloads abort scheduler recovery; invalid state now falls back to a safe default instead of crashing restore.
# SEC-1: Bound remote snapshot ingestion (item count, nesting, mapping width, string size) to reduce practical Pi-4 denial-of-service risk from poisoned or corrupted remote snapshots.
# IMP-1: Add version-aware, metadata-rich subscription envelopes (schema_version, saved_at, item_count, payload_sha256) with backward-compatible loading from legacy schema_version=1 payloads.
# IMP-2: Add forward-compatible state-envelope loading, JSON-Schema contracts, structured logging, and process-local save locking; preserve legacy state write format for drop-in compatibility.

"""Persist RSS/world-intelligence state through remote-primary snapshots."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Final, Protocol

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality._remote_state_utils import (
    resolve_remote_state as _resolve_remote_state,
)
from twinr.agent.personality.intelligence.models import (
    DEFAULT_WORLD_INTELLIGENCE_STATE_KIND,
    DEFAULT_WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND,
    WorldFeedSubscription,
    WorldIntelligenceState,
)
from twinr.memory.longterm.storage._remote_current_records import LongTermRemoteCurrentRecordStore
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore

_LOGGER = logging.getLogger(__name__)

_SUBSCRIPTIONS_SCHEMA_VERSION: Final[int] = 2
_SUPPORTED_SUBSCRIPTIONS_SCHEMA_VERSIONS: Final[frozenset[int]] = frozenset({1, 2})
_FUTURE_STATE_ENVELOPE_KEY: Final[str] = "data"

_DEFAULT_MAX_SUBSCRIPTIONS: Final[int] = 256
_DEFAULT_MAX_CONTAINER_ITEMS: Final[int] = 512
_DEFAULT_MAX_STRING_CHARS: Final[int] = 8_192
_DEFAULT_MAX_NESTING_DEPTH: Final[int] = 12


class WorldIntelligenceStore(Protocol):
    """Describe remote-backed persistence for subscriptions and timing state."""

    def load_subscriptions(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[WorldFeedSubscription, ...]:
        """Load persisted world-intelligence subscriptions."""

    def save_subscriptions(
        self,
        *,
        config: TwinrConfig,
        subscriptions: Sequence[WorldFeedSubscription],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist world-intelligence subscriptions."""

    def load_state(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> WorldIntelligenceState:
        """Load persisted global refresh/discovery timing state."""

    def save_state(
        self,
        *,
        config: TwinrConfig,
        state: WorldIntelligenceState,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist global refresh/discovery timing state."""


def _utc_now_iso8601() -> str:
    """Return a stable UTC timestamp suitable for remote snapshot metadata."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json_sha256(value: object) -> str | None:
    """Hash JSON-compatible data deterministically for corruption detection."""
    try:
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(canonical).hexdigest()


def _ensure_mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping, got {type(value).__name__}.")
    return value


def _ensure_sequence(value: object, *, path: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{path} must be a sequence.")
    return value


def _bounded_json_clone(
    value: object,
    *,
    path: str,
    max_container_items: int,
    max_string_chars: int,
    max_nesting_depth: int,
) -> object:
    """Validate that a decoded payload is JSON-like and bounded for Pi-class devices."""
    if max_nesting_depth < 0:
        raise ValueError(f"{path} exceeds the maximum nesting depth.")

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        if len(value) > max_string_chars:
            raise ValueError(f"{path} exceeds the maximum string length of {max_string_chars}.")
        return value

    if isinstance(value, Mapping):
        if len(value) > max_container_items:
            raise ValueError(f"{path} exceeds the maximum mapping size of {max_container_items}.")
        cloned: dict[str, object] = {}
        for key, inner_value in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string mapping key {key!r}.")
            if len(key) > max_string_chars:
                raise ValueError(f"{path} contains an oversized key.")
            cloned[key] = _bounded_json_clone(
                inner_value,
                path=f"{path}.{key}",
                max_container_items=max_container_items,
                max_string_chars=max_string_chars,
                max_nesting_depth=max_nesting_depth - 1,
            )
        return cloned

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > max_container_items:
            raise ValueError(f"{path} exceeds the maximum sequence size of {max_container_items}.")
        return [
            _bounded_json_clone(
                inner_value,
                path=f"{path}[{index}]",
                max_container_items=max_container_items,
                max_string_chars=max_string_chars,
                max_nesting_depth=max_nesting_depth - 1,
            )
            for index, inner_value in enumerate(value)
        ]

    raise ValueError(f"{path} contains unsupported type {type(value).__name__}.")


def _normalized_record_text(value: object, *, limit: int = _DEFAULT_MAX_STRING_CHARS) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _record_json_content(payload: Mapping[str, object]) -> str:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _subscriptions_written_at(payloads: Sequence[Mapping[str, object]]) -> str | None:
    candidates: list[str] = []
    for payload in payloads:
        for field_name in ("updated_at", "last_refreshed_at", "last_checked_at", "created_at"):
            value = _normalized_record_text(payload.get(field_name), limit=80)
            if value:
                candidates.append(value)
                break
    return max(candidates) if candidates else None


def _subscription_metadata(payload: Mapping[str, object]) -> Mapping[str, object]:
    created_at = _normalized_record_text(payload.get("created_at"), limit=80)
    updated_at = _normalized_record_text(payload.get("updated_at"), limit=80) or created_at
    return {
        "kind": _normalized_record_text(payload.get("scope"), limit=160) or "world_feed_subscription",
        "summary": _normalized_record_text(payload.get("label"), limit=220),
        "slot_key": _normalized_record_text(payload.get("region"), limit=160),
        "value_key": _normalized_record_text(payload.get("subscription_id"), limit=160),
        "created_at": created_at,
        "updated_at": updated_at,
        "status": "active" if bool(payload.get("active", True)) else "inactive",
    }


def _state_metadata(payload: Mapping[str, object]) -> Mapping[str, object]:
    last_recalibrated_at = _normalized_record_text(payload.get("last_recalibrated_at"), limit=80)
    last_refreshed_at = _normalized_record_text(payload.get("last_refreshed_at"), limit=80)
    last_discovered_at = _normalized_record_text(payload.get("last_discovered_at"), limit=80)
    summary = _normalized_record_text(payload.get("last_discovery_query"), limit=220) or "world intelligence state"
    updated_at = last_recalibrated_at or last_refreshed_at or last_discovered_at
    return {
        "kind": "world_intelligence_state",
        "summary": summary,
        "updated_at": updated_at,
        "created_at": last_discovered_at or updated_at,
    }


@dataclass(slots=True)
class RemoteStateWorldIntelligenceStore:
    """Load and save world-intelligence snapshots via remote-primary state.

    Loader compatibility:
      * subscriptions: legacy schema_version=1 and current schema_version=2
      * state: legacy bare payloads and future envelope payloads using {"schema_version": 2, "data": ...}

    The write path intentionally preserves the legacy bare state payload shape to remain
    drop-in compatible with older deployments that call WorldIntelligenceState.from_payload
    directly on the stored state snapshot.
    """

    subscriptions_snapshot_kind: str = DEFAULT_WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND
    state_snapshot_kind: str = DEFAULT_WORLD_INTELLIGENCE_STATE_KIND
    max_subscriptions: int = _DEFAULT_MAX_SUBSCRIPTIONS
    max_container_items: int = _DEFAULT_MAX_CONTAINER_ITEMS
    max_string_chars: int = _DEFAULT_MAX_STRING_CHARS
    max_nesting_depth: int = _DEFAULT_MAX_NESTING_DEPTH
    _save_lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.subscriptions_snapshot_kind, str) or not self.subscriptions_snapshot_kind.strip():
            raise ValueError("subscriptions_snapshot_kind must be a non-empty string.")
        if not isinstance(self.state_snapshot_kind, str) or not self.state_snapshot_kind.strip():
            raise ValueError("state_snapshot_kind must be a non-empty string.")
        if self.max_subscriptions < 1:
            raise ValueError("max_subscriptions must be >= 1.")
        if self.max_container_items < 1:
            raise ValueError("max_container_items must be >= 1.")
        if self.max_string_chars < 16:
            raise ValueError("max_string_chars must be >= 16.")
        if self.max_nesting_depth < 1:
            raise ValueError("max_nesting_depth must be >= 1.")

    def load_subscriptions(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[WorldFeedSubscription, ...]:
        """Load persisted feed subscriptions when present."""
        resolved = _resolve_remote_state(config=config, remote_state=remote_state)
        if resolved is None:
            return ()
        current_records = LongTermRemoteCurrentRecordStore(resolved)
        head = current_records.load_current_head(snapshot_kind=self.subscriptions_snapshot_kind)
        if isinstance(head, Mapping):
            raw_items = list(current_records.load_collection_payloads(snapshot_kind=self.subscriptions_snapshot_kind))
            total_items = len(raw_items)
        else:
            payload = resolved.load_snapshot(snapshot_kind=self.subscriptions_snapshot_kind)
            if payload is None:
                return ()
            try:
                envelope = _ensure_mapping(payload, path=self.subscriptions_snapshot_kind)
            except ValueError as exc:
                _LOGGER.warning(
                    "Ignoring corrupt %s snapshot: %s",
                    self.subscriptions_snapshot_kind,
                    exc,
                )
                return ()

            schema_version = envelope.get("schema_version")
            if schema_version is not None and not isinstance(schema_version, int):
                _LOGGER.warning(
                    "Ignoring corrupt %s snapshot: schema_version must be an int.",
                    self.subscriptions_snapshot_kind,
                )
                return ()
            if isinstance(schema_version, int) and schema_version not in _SUPPORTED_SUBSCRIPTIONS_SCHEMA_VERSIONS:
                _LOGGER.warning(
                    "%s uses unknown schema_version=%s; attempting best-effort decode.",
                    self.subscriptions_snapshot_kind,
                    schema_version,
                )

            if "items" not in envelope:
                _LOGGER.warning(
                    "Ignoring corrupt %s snapshot: missing required 'items' key.",
                    self.subscriptions_snapshot_kind,
                )
                return ()

            try:
                raw_items = _ensure_sequence(
                    envelope["items"],
                    path=f"{self.subscriptions_snapshot_kind}.items",
                )
            except ValueError as exc:
                _LOGGER.warning(
                    "Ignoring corrupt %s snapshot: %s",
                    self.subscriptions_snapshot_kind,
                    exc,
                )
                return ()

            total_items = len(raw_items)
            if total_items > self.max_subscriptions:
                _LOGGER.warning(
                    "%s snapshot contains %d subscriptions; truncating to max_subscriptions=%d.",
                    self.subscriptions_snapshot_kind,
                    total_items,
                    self.max_subscriptions,
                )
                raw_items = list(raw_items[: self.max_subscriptions])
            else:
                raw_items = list(raw_items)

            declared_item_count = envelope.get("item_count")
            if declared_item_count is not None and (
                not isinstance(declared_item_count, int) or declared_item_count < 0
            ):
                _LOGGER.warning(
                    "%s snapshot has invalid item_count=%r; ignoring metadata.",
                    self.subscriptions_snapshot_kind,
                    declared_item_count,
                )
            elif isinstance(declared_item_count, int) and declared_item_count != total_items:
                _LOGGER.warning(
                    "%s snapshot item_count=%d but payload contains %d items.",
                    self.subscriptions_snapshot_kind,
                    declared_item_count,
                    total_items,
                )

            expected_digest = envelope.get("payload_sha256")
            if isinstance(expected_digest, str) and len(raw_items) == total_items:
                actual_digest = _canonical_json_sha256(raw_items)
                if actual_digest is not None and actual_digest != expected_digest:
                    _LOGGER.warning(
                        "Ignoring corrupt %s snapshot: payload_sha256 mismatch.",
                        self.subscriptions_snapshot_kind,
                    )
                    return ()

        loaded: list[WorldFeedSubscription] = []
        invalid_items = 0

        for index, item in enumerate(raw_items):
            if not isinstance(item, Mapping):
                invalid_items += 1
                _LOGGER.warning(
                    "Skipping invalid %s.items[%d]: expected a mapping, got %s.",
                    self.subscriptions_snapshot_kind,
                    index,
                    type(item).__name__,
                )
                continue

            try:
                normalized_item = _bounded_json_clone(
                    item,
                    path=f"{self.subscriptions_snapshot_kind}.items[{index}]",
                    max_container_items=self.max_container_items,
                    max_string_chars=self.max_string_chars,
                    max_nesting_depth=self.max_nesting_depth,
                )
                loaded.append(WorldFeedSubscription.from_payload(normalized_item))
            except Exception as exc:
                invalid_items += 1
                _LOGGER.warning(
                    "Skipping invalid %s.items[%d]: %s",
                    self.subscriptions_snapshot_kind,
                    index,
                    exc,
                )

        if invalid_items:
            _LOGGER.warning(
                "%s restored %d/%d subscriptions; skipped %d invalid item(s).",
                self.subscriptions_snapshot_kind,
                len(loaded),
                len(raw_items),
                invalid_items,
            )

        if not isinstance(head, Mapping) and loaded:
            try:
                serialized_items = [item.to_payload() for item in loaded]
                current_records.save_collection(
                    snapshot_kind=self.subscriptions_snapshot_kind,
                    item_payloads=serialized_items,
                    item_id_getter=lambda item: item.get("subscription_id"),
                    metadata_builder=_subscription_metadata,
                    content_builder=_record_json_content,
                    written_at=_subscriptions_written_at(serialized_items),
                )
            except Exception:
                _LOGGER.warning(
                    "Failed to promote legacy %s snapshot into the current-head record contract.",
                    self.subscriptions_snapshot_kind,
                    exc_info=True,
                )

        return tuple(loaded)

    def save_subscriptions(
        self,
        *,
        config: TwinrConfig,
        subscriptions: Sequence[WorldFeedSubscription],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist feed subscriptions through remote-primary state."""
        resolved = _resolve_remote_state(config=config, remote_state=remote_state)
        if resolved is None:
            return
        current_records = LongTermRemoteCurrentRecordStore(resolved)

        serialized_items: list[Mapping[str, object]] = []
        for index, item in enumerate(subscriptions):
            to_payload = getattr(item, "to_payload", None)
            if not callable(to_payload):
                raise TypeError(
                    f"subscriptions[{index}] must provide a callable to_payload(); "
                    f"got {type(item).__name__}."
                )

            raw_payload = to_payload()
            normalized_payload = _bounded_json_clone(
                _ensure_mapping(
                    raw_payload,
                    path=f"{self.subscriptions_snapshot_kind}.items[{index}]",
                ),
                path=f"{self.subscriptions_snapshot_kind}.items[{index}]",
                max_container_items=self.max_container_items,
                max_string_chars=self.max_string_chars,
                max_nesting_depth=self.max_nesting_depth,
            )
            serialized_items.append(normalized_payload)

        if len(serialized_items) > self.max_subscriptions:
            raise ValueError(
                f"{self.subscriptions_snapshot_kind} exceeds max_subscriptions="
                f"{self.max_subscriptions} ({len(serialized_items)} provided)."
            )

        with self._save_lock:
            current_records.save_collection(
                snapshot_kind=self.subscriptions_snapshot_kind,
                item_payloads=serialized_items,
                item_id_getter=lambda item: item.get("subscription_id"),
                metadata_builder=_subscription_metadata,
                content_builder=_record_json_content,
                written_at=_subscriptions_written_at(serialized_items) or _utc_now_iso8601(),
            )

    def load_state(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> WorldIntelligenceState:
        """Load the global refresh/discovery timing snapshot."""
        resolved = _resolve_remote_state(config=config, remote_state=remote_state)
        if resolved is None:
            return WorldIntelligenceState()
        current_records = LongTermRemoteCurrentRecordStore(resolved)
        payload = current_records.load_single_payload(snapshot_kind=self.state_snapshot_kind)
        if payload is None:
            payload = resolved.load_snapshot(snapshot_kind=self.state_snapshot_kind)
        if payload is None:
            return WorldIntelligenceState()

        try:
            raw_payload = _ensure_mapping(payload, path=self.state_snapshot_kind)
        except ValueError as exc:
            _LOGGER.warning(
                "Ignoring corrupt %s snapshot and falling back to defaults: %s",
                self.state_snapshot_kind,
                exc,
            )
            return WorldIntelligenceState()

        # Forward-compatible read path for a future enveloped state payload.
        if _FUTURE_STATE_ENVELOPE_KEY in raw_payload:
            schema_version = raw_payload.get("schema_version")
            if schema_version is not None and not isinstance(schema_version, int):
                _LOGGER.warning(
                    "Ignoring corrupt %s snapshot and falling back to defaults: "
                    "schema_version must be an int.",
                    self.state_snapshot_kind,
                )
                return WorldIntelligenceState()

            try:
                candidate_state_payload = _ensure_mapping(
                    raw_payload[_FUTURE_STATE_ENVELOPE_KEY],
                    path=f"{self.state_snapshot_kind}.{_FUTURE_STATE_ENVELOPE_KEY}",
                )
            except ValueError as exc:
                _LOGGER.warning(
                    "Ignoring corrupt %s snapshot and falling back to defaults: %s",
                    self.state_snapshot_kind,
                    exc,
                )
                return WorldIntelligenceState()

            expected_digest = raw_payload.get("payload_sha256")
            if isinstance(expected_digest, str):
                actual_digest = _canonical_json_sha256(candidate_state_payload)
                if actual_digest is not None and actual_digest != expected_digest:
                    _LOGGER.warning(
                        "Ignoring corrupt %s snapshot and falling back to defaults: "
                        "payload_sha256 mismatch.",
                        self.state_snapshot_kind,
                    )
                    return WorldIntelligenceState()

            raw_payload = candidate_state_payload

        try:
            normalized_payload = _bounded_json_clone(
                raw_payload,
                path=self.state_snapshot_kind,
                max_container_items=self.max_container_items,
                max_string_chars=self.max_string_chars,
                max_nesting_depth=self.max_nesting_depth,
            )
            state = WorldIntelligenceState.from_payload(normalized_payload)
            # Keep prompt-time intelligence reads read-only so first-turn prompt
            # assembly cannot stall on legacy-head promotion writes.
            return state
        except Exception as exc:
            _LOGGER.warning(
                "Ignoring invalid %s snapshot and falling back to defaults: %s",
                self.state_snapshot_kind,
                exc,
            )
            return WorldIntelligenceState()

    def save_state(
        self,
        *,
        config: TwinrConfig,
        state: WorldIntelligenceState,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist the global refresh/discovery timing snapshot.

        NOTE:
            This intentionally preserves the legacy write format (bare state payload)
            for drop-in compatibility. The loader already accepts a future envelope
            format with {"schema_version": 2, "data": ...} if the surrounding storage
            layer is upgraded later to support versioned metadata or conditional writes.
        """
        resolved = _resolve_remote_state(config=config, remote_state=remote_state)
        if resolved is None:
            return
        current_records = LongTermRemoteCurrentRecordStore(resolved)

        normalized_payload = _bounded_json_clone(
            _ensure_mapping(state.to_payload(), path=self.state_snapshot_kind),
            path=self.state_snapshot_kind,
            max_container_items=self.max_container_items,
            max_string_chars=self.max_string_chars,
            max_nesting_depth=self.max_nesting_depth,
        )

        with self._save_lock:
            current_records.save_single_payload(
                snapshot_kind=self.state_snapshot_kind,
                payload=normalized_payload,
                metadata_builder=_state_metadata,
                content_builder=_record_json_content,
                written_at=_normalized_record_text(
                    normalized_payload.get("last_recalibrated_at")
                    or normalized_payload.get("last_refreshed_at")
                    or normalized_payload.get("last_discovered_at"),
                    limit=80,
                )
                or None,
            )

    def probe_remote_current_subscriptions(self, *, config: TwinrConfig, remote_state: LongTermRemoteStateStore | None = None) -> dict[str, object] | None:
        """Expose the subscriptions current head for readiness or telemetry."""

        resolved = _resolve_remote_state(config=config, remote_state=remote_state)
        if resolved is None:
            return None
        return LongTermRemoteCurrentRecordStore(resolved).probe_current_head(snapshot_kind=self.subscriptions_snapshot_kind)

    def probe_remote_current_state(self, *, config: TwinrConfig, remote_state: LongTermRemoteStateStore | None = None) -> dict[str, object] | None:
        """Expose the world-intelligence state current head for readiness or telemetry."""

        resolved = _resolve_remote_state(config=config, remote_state=remote_state)
        if resolved is None:
            return None
        return LongTermRemoteCurrentRecordStore(resolved).probe_current_head(snapshot_kind=self.state_snapshot_kind)

    def subscriptions_snapshot_json_schema(self) -> Mapping[str, object]:
        """Expose a self-describing JSON Schema for ops, CI, and migration tooling."""
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "World Intelligence Subscriptions Snapshot",
            "type": "object",
            "properties": {
                "schema_version": {"type": "integer", "minimum": 1},
                "saved_at": {"type": "string", "format": "date-time"},
                "item_count": {"type": "integer", "minimum": 0},
                "payload_sha256": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{64}$",
                },
                "items": {
                    "type": "array",
                    "maxItems": self.max_subscriptions,
                    "items": {"type": "object"},
                },
            },
            "required": ["schema_version", "items"],
            "additionalProperties": True,
        }

    @staticmethod
    def state_snapshot_envelope_json_schema() -> Mapping[str, object]:
        """Expose the future state-envelope schema accepted by load_state."""
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "World Intelligence State Snapshot Envelope",
            "type": "object",
            "properties": {
                "schema_version": {"type": "integer", "minimum": 2},
                "payload_sha256": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{64}$",
                },
                "data": {"type": "object"},
            },
            "required": ["schema_version", "data"],
            "additionalProperties": True,
        }
