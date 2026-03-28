"""Load and save structured personality state via versioned remote-backed snapshots."""

from __future__ import annotations

# CHANGELOG: 2026-03-27
# BUG-1: Fixed silent data loss where malformed list snapshots with a missing/null
#        "items" field were treated as an empty snapshot instead of a corrupted one.
# BUG-2: Fixed whole-snapshot load failures caused by a single malformed list item by
#        adding item-level salvage with warnings and strict-mode support.
# BUG-3: Fixed redundant remote writes of unchanged snapshots via canonical payload hashing
#        and same-process save deduplication.
# SEC-1: Added versioned, tamper-evident snapshot envelopes with SHA-256 integrity checks,
#        optional HMAC authentication, and size guards suitable for Raspberry Pi 4 deployments.
# SEC-2: Added per-snapshot locks so concurrent same-process load/save calls cannot interleave
#        envelope reads and writes for the same snapshot kind.
# IMP-1: Upgraded persistence to a versioned envelope that supports legacy reads, audit metadata,
#        bounded payloads, and future schema migration.
# IMP-2: Added optional zstd/gzip compression for larger payloads to reduce remote IO and storage.
# IMP-3: Added configurable list compaction so append-only signal histories stay bounded on edge devices.

import base64
import binascii
import gzip
import hashlib
import hmac
import json
import logging
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, TypeVar, cast

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality._remote_state_utils import (
    resolve_remote_state as _resolve_remote_state,
)
from twinr.agent.personality.models import (
    DEFAULT_PERSONALITY_SNAPSHOT_KIND,
    INTERACTION_SIGNAL_SNAPSHOT_KIND,
    PERSONALITY_DELTA_SNAPSHOT_KIND,
    PLACE_SIGNAL_SNAPSHOT_KIND,
    WORLD_SIGNAL_SNAPSHOT_KIND,
    InteractionSignal,
    PersonalityDelta,
    PersonalitySnapshot,
    PlaceSignal,
    WorldSignal,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore

_ItemT = TypeVar("_ItemT")

logger = logging.getLogger(__name__)

_ENVELOPE_FORMAT = "twinr.remote_snapshot.v2"
_ENVELOPE_SCHEMA_VERSION = 2
_DEFAULT_MAX_PAYLOAD_BYTES = 2 * 1024 * 1024
_DEFAULT_MAX_COMPRESSED_BYTES = 2 * 1024 * 1024
_DEFAULT_COMPRESSION_THRESHOLD_BYTES = 16 * 1024
_DEFAULT_MAX_LIST_ITEMS = 4096
_DEFAULT_REVISION_ID_LENGTH = 16
_HMAC_CONFIG_ATTRS = (
    "remote_state_hmac_key",
    "snapshot_hmac_key",
    "personality_state_hmac_key",
)
_HMAC_ENV_VARS = (
    "TWINR_REMOTE_STATE_HMAC_KEY",
    "TWINR_SNAPSHOT_HMAC_KEY",
    "TWINR_PERSONALITY_STATE_HMAC_KEY",
)
_MISSING = object()

_SNAPSHOT_LOCKS: dict[str, threading.RLock] = {}
_SNAPSHOT_LOCKS_GUARD = threading.Lock()
_LAST_SAVED_FINGERPRINTS: dict[tuple[int, str], str] = {}
_LAST_SAVED_FINGERPRINTS_GUARD = threading.Lock()


class SnapshotIntegrityError(ValueError):
    """Raised when a snapshot envelope fails integrity or authenticity checks."""


class SnapshotSizeError(ValueError):
    """Raised when a snapshot exceeds configured safety limits."""


def _snapshot_lock(snapshot_kind: str) -> threading.RLock:
    with _SNAPSHOT_LOCKS_GUARD:
        lock = _SNAPSHOT_LOCKS.get(snapshot_kind)
        if lock is None:
            lock = threading.RLock()
            _SNAPSHOT_LOCKS[snapshot_kind] = lock
        return lock


def _cache_key(
    resolved_remote_state: LongTermRemoteStateStore,
    snapshot_kind: str,
) -> tuple[int, str]:
    return (id(resolved_remote_state), snapshot_kind)


def _get_cached_snapshot_fingerprint(
    *,
    resolved_remote_state: LongTermRemoteStateStore,
    snapshot_kind: str,
) -> str | None:
    with _LAST_SAVED_FINGERPRINTS_GUARD:
        return _LAST_SAVED_FINGERPRINTS.get(_cache_key(resolved_remote_state, snapshot_kind))


def _set_cached_snapshot_fingerprint(
    *,
    resolved_remote_state: LongTermRemoteStateStore,
    snapshot_kind: str,
    snapshot_fingerprint: str,
) -> None:
    with _LAST_SAVED_FINGERPRINTS_GUARD:
        _LAST_SAVED_FINGERPRINTS[_cache_key(resolved_remote_state, snapshot_kind)] = snapshot_fingerprint


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Snapshot payload must be JSON-serializable and contain only finite values."
        ) from exc
    return text.encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_hmac_key(config: TwinrConfig) -> bytes | None:
    for attr_name in _HMAC_CONFIG_ATTRS:
        maybe_value = getattr(config, attr_name, None)
        if isinstance(maybe_value, bytes) and maybe_value:
            return maybe_value
        if isinstance(maybe_value, str) and maybe_value:
            return maybe_value.encode("utf-8")
    for env_name in _HMAC_ENV_VARS:
        maybe_value = os.getenv(env_name)
        if maybe_value:
            return maybe_value.encode("utf-8")
    return None


def _hmac_sha256_hex(*, key: bytes, data: bytes) -> str:
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def _compress_payload_bytes(
    data: bytes,
    *,
    compression_threshold_bytes: int,
) -> tuple[str, str] | None:
    if len(data) < compression_threshold_bytes:
        return None

    try:
        import compression.zstd as _compression_zstd  # type: ignore[attr-defined]  # pylint: disable=import-error
    except Exception:
        _compression_zstd = None

    if _compression_zstd is not None:
        compressed = _compression_zstd.compress(data)
        if len(compressed) + 64 < len(data):
            return ("zstd-base64-json", base64.b64encode(compressed).decode("ascii"))

    try:
        import zstandard as _zstandard  # type: ignore[import-not-found]  # pylint: disable=import-error
    except Exception:
        _zstandard = None

    if _zstandard is not None:
        compressed = _zstandard.ZstdCompressor(level=3).compress(data)
        if len(compressed) + 64 < len(data):
            return ("zstd-base64-json", base64.b64encode(compressed).decode("ascii"))

    compressed = gzip.compress(data, compresslevel=6)
    if len(compressed) + 64 < len(data):
        return ("gzip-base64-json", base64.b64encode(compressed).decode("ascii"))
    return None


def _decompress_payload_bytes(
    *,
    encoding: str,
    payload_b64: str,
    max_payload_bytes: int,
    max_compressed_bytes: int,
) -> bytes:
    try:
        compressed = base64.b64decode(payload_b64.encode("ascii"), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise SnapshotIntegrityError("Snapshot payload_b64 is not valid base64.") from exc

    if len(compressed) > max_compressed_bytes:
        raise SnapshotSizeError(
            f"Compressed snapshot payload is {len(compressed)} bytes, exceeding the configured limit "
            f"of {max_compressed_bytes} bytes."
        )

    if encoding == "zstd-base64-json":
        try:
            import compression.zstd as _compression_zstd  # type: ignore[attr-defined]  # pylint: disable=import-error
        except Exception:
            _compression_zstd = None

        if _compression_zstd is not None:
            data = _compression_zstd.decompress(compressed)
        else:
            try:
                import zstandard as _zstandard  # type: ignore[import-not-found]  # pylint: disable=import-error
            except Exception as exc:
                raise SnapshotIntegrityError(
                    "Snapshot uses zstd compression but no zstd decoder is available."
                ) from exc
            decompressor = _zstandard.ZstdDecompressor()
            try:
                data = decompressor.decompress(compressed, max_output_size=max_payload_bytes)
            except TypeError:
                data = decompressor.decompress(compressed)
    elif encoding == "gzip-base64-json":
        data = gzip.decompress(compressed)
    else:
        raise SnapshotIntegrityError(f"Unsupported snapshot payload encoding: {encoding!r}.")

    if len(data) > max_payload_bytes:
        raise SnapshotSizeError(
            f"Decoded snapshot payload is {len(data)} bytes, exceeding the configured limit "
            f"of {max_payload_bytes} bytes."
        )
    return data


def _snapshot_item_count(payload: Mapping[str, object]) -> int | None:
    items = payload.get("items")
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
        return len(items)
    return None


def _snapshot_fingerprint(
    snapshot_payload: Mapping[str, object],
) -> str:
    if snapshot_payload.get("__twinr_format__") == _ENVELOPE_FORMAT:
        stable_envelope = dict(snapshot_payload)
        stable_envelope.pop("created_at", None)
        return _sha256_hex(_canonical_json_bytes(stable_envelope))
    return _sha256_hex(_canonical_json_bytes(snapshot_payload))


def _build_envelope(
    *,
    config: TwinrConfig,
    snapshot_kind: str,
    payload: Mapping[str, object],
    max_payload_bytes: int,
    max_compressed_bytes: int,
    compression_threshold_bytes: int,
) -> Mapping[str, object]:
    payload_bytes = _canonical_json_bytes(payload)
    if len(payload_bytes) > max_payload_bytes:
        raise SnapshotSizeError(
            f"{snapshot_kind} is {len(payload_bytes)} bytes after serialization, exceeding the configured "
            f"limit of {max_payload_bytes} bytes."
        )

    payload_hash = _sha256_hex(payload_bytes)
    hmac_key = _resolve_hmac_key(config)
    payload_hmac = (
        _hmac_sha256_hex(key=hmac_key, data=payload_bytes)
        if hmac_key is not None
        else None
    )

    compressed = _compress_payload_bytes(
        payload_bytes,
        compression_threshold_bytes=compression_threshold_bytes,
    )
    if compressed is None:
        payload_encoding = "inline-json"
        payload_field: Mapping[str, object] | None = dict(payload)
        payload_b64 = None
        compressed_size = len(payload_bytes)
    else:
        payload_encoding, payload_b64 = compressed
        payload_field = None
        compressed_size = len(payload_b64.encode("ascii"))

    if compressed_size > max_compressed_bytes:
        raise SnapshotSizeError(
            f"{snapshot_kind} compressed payload is {compressed_size} bytes, exceeding the configured "
            f"limit of {max_compressed_bytes} bytes."
        )

    return {
        "__twinr_format__": _ENVELOPE_FORMAT,
        "schema_version": _ENVELOPE_SCHEMA_VERSION,
        "snapshot_kind": snapshot_kind,
        "revision_id": payload_hash[:_DEFAULT_REVISION_ID_LENGTH],
        "created_at": _utc_now_iso(),
        "payload_encoding": payload_encoding,
        "payload_byte_length": len(payload_bytes),
        "payload_sha256": payload_hash,
        "payload_hmac_sha256": payload_hmac,
        "item_count": _snapshot_item_count(payload),
        "payload": payload_field,
        "payload_b64": payload_b64,
    }


def _unwrap_snapshot_payload(
    *,
    config: TwinrConfig,
    snapshot_kind: str,
    raw_payload: Mapping[str, object],
    max_payload_bytes: int,
    max_compressed_bytes: int,
) -> tuple[Mapping[str, object], str]:
    format_marker = raw_payload.get("__twinr_format__")
    if format_marker != _ENVELOPE_FORMAT:
        return raw_payload, _sha256_hex(_canonical_json_bytes(raw_payload))

    declared_snapshot_kind = raw_payload.get("snapshot_kind")
    if declared_snapshot_kind != snapshot_kind:
        raise SnapshotIntegrityError(
            f"Envelope snapshot kind mismatch for {snapshot_kind!r}: got {declared_snapshot_kind!r}."
        )

    schema_version = raw_payload.get("schema_version")
    if schema_version != _ENVELOPE_SCHEMA_VERSION:
        raise SnapshotIntegrityError(
            f"Unsupported envelope schema version for {snapshot_kind!r}: {schema_version!r}."
        )

    payload_encoding = raw_payload.get("payload_encoding")
    if not isinstance(payload_encoding, str):
        raise SnapshotIntegrityError(
            f"{snapshot_kind} envelope is missing a valid payload_encoding."
        )

    payload_byte_length = raw_payload.get("payload_byte_length")
    if not isinstance(payload_byte_length, int) or payload_byte_length < 0:
        raise SnapshotIntegrityError(
            f"{snapshot_kind} envelope is missing a valid payload_byte_length."
        )
    if payload_byte_length > max_payload_bytes:
        raise SnapshotSizeError(
            f"{snapshot_kind} declares {payload_byte_length} payload bytes, exceeding the configured "
            f"limit of {max_payload_bytes} bytes."
        )

    if payload_encoding == "inline-json":
        payload_obj = raw_payload.get("payload")
        if not isinstance(payload_obj, Mapping):
            raise SnapshotIntegrityError(
                f"{snapshot_kind} inline snapshot payload must be a mapping."
            )
        payload = cast(Mapping[str, object], payload_obj)
        payload_bytes = _canonical_json_bytes(payload)
    else:
        payload_b64 = raw_payload.get("payload_b64")
        if not isinstance(payload_b64, str):
            raise SnapshotIntegrityError(
                f"{snapshot_kind} compressed snapshot payload_b64 must be a string."
            )
        payload_bytes = _decompress_payload_bytes(
            encoding=payload_encoding,
            payload_b64=payload_b64,
            max_payload_bytes=max_payload_bytes,
            max_compressed_bytes=max_compressed_bytes,
        )
        try:
            payload_obj = json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SnapshotIntegrityError(
                f"{snapshot_kind} compressed payload is not valid JSON."
            ) from exc
        if not isinstance(payload_obj, dict):
            raise SnapshotIntegrityError(
                f"{snapshot_kind} compressed payload must decode to a mapping."
            )
        payload = cast(Mapping[str, object], payload_obj)

    if len(payload_bytes) != payload_byte_length:
        raise SnapshotIntegrityError(
            f"{snapshot_kind} payload length mismatch: expected {payload_byte_length}, got {len(payload_bytes)}."
        )

    declared_hash = raw_payload.get("payload_sha256")
    if not isinstance(declared_hash, str):
        raise SnapshotIntegrityError(f"{snapshot_kind} envelope is missing payload_sha256.")
    actual_hash = _sha256_hex(payload_bytes)
    if actual_hash != declared_hash:
        raise SnapshotIntegrityError(
            f"{snapshot_kind} payload SHA-256 mismatch. Snapshot may be corrupted or tampered with."
        )

    declared_hmac = raw_payload.get("payload_hmac_sha256")
    if declared_hmac is not None:
        if not isinstance(declared_hmac, str):
            raise SnapshotIntegrityError(
                f"{snapshot_kind} envelope has an invalid payload_hmac_sha256 field."
            )
        hmac_key = _resolve_hmac_key(config)
        if hmac_key is None:
            raise SnapshotIntegrityError(
                f"{snapshot_kind} snapshot requires HMAC verification but no HMAC key is configured."
            )
        actual_hmac = _hmac_sha256_hex(key=hmac_key, data=payload_bytes)
        if not hmac.compare_digest(actual_hmac, declared_hmac):
            raise SnapshotIntegrityError(
                f"{snapshot_kind} payload HMAC mismatch. Snapshot authenticity verification failed."
            )

    return payload, actual_hash


def _load_raw_snapshot_payload(
    *,
    config: TwinrConfig,
    remote_state: LongTermRemoteStateStore | None,
    snapshot_kind: str,
    max_payload_bytes: int,
    max_compressed_bytes: int,
) -> tuple[LongTermRemoteStateStore | None, Mapping[str, object] | None, str | None]:
    resolved_remote_state = _resolve_remote_state(config=config, remote_state=remote_state)
    if resolved_remote_state is None:
        return None, None, None

    with _snapshot_lock(snapshot_kind):
        raw_payload = resolved_remote_state.load_snapshot(snapshot_kind=snapshot_kind)
        if raw_payload is None:
            return resolved_remote_state, None, None
        if not isinstance(raw_payload, Mapping):
            raise ValueError(f"{snapshot_kind} must decode to a mapping payload.")

        payload, payload_hash = _unwrap_snapshot_payload(
            config=config,
            snapshot_kind=snapshot_kind,
            raw_payload=raw_payload,
            max_payload_bytes=max_payload_bytes,
            max_compressed_bytes=max_compressed_bytes,
        )
        snapshot_fingerprint = _snapshot_fingerprint(raw_payload)
        _set_cached_snapshot_fingerprint(
            resolved_remote_state=resolved_remote_state,
            snapshot_kind=snapshot_kind,
            snapshot_fingerprint=snapshot_fingerprint,
        )
        return resolved_remote_state, payload, payload_hash


def _save_raw_snapshot_payload(
    *,
    config: TwinrConfig,
    remote_state: LongTermRemoteStateStore | None,
    snapshot_kind: str,
    payload: Mapping[str, object],
    max_payload_bytes: int,
    max_compressed_bytes: int,
    compression_threshold_bytes: int,
) -> None:
    resolved_remote_state = _resolve_remote_state(config=config, remote_state=remote_state)
    if resolved_remote_state is None:
        return

    envelope = _build_envelope(
        config=config,
        snapshot_kind=snapshot_kind,
        payload=payload,
        max_payload_bytes=max_payload_bytes,
        max_compressed_bytes=max_compressed_bytes,
        compression_threshold_bytes=compression_threshold_bytes,
    )
    snapshot_fingerprint = _snapshot_fingerprint(envelope)

    with _snapshot_lock(snapshot_kind):
        cached_fingerprint = _get_cached_snapshot_fingerprint(
            resolved_remote_state=resolved_remote_state,
            snapshot_kind=snapshot_kind,
        )
        if cached_fingerprint == snapshot_fingerprint:
            logger.debug("Skipping unchanged snapshot save for %s.", snapshot_kind)
            return

        # BREAKING: remote snapshots are now written as v2 envelopes. Legacy snapshots still load,
        # but any code that bypasses this store and reads raw remote payloads directly must unwrap the envelope.
        resolved_remote_state.save_snapshot(snapshot_kind=snapshot_kind, payload=envelope)
        _set_cached_snapshot_fingerprint(
            resolved_remote_state=resolved_remote_state,
            snapshot_kind=snapshot_kind,
            snapshot_fingerprint=snapshot_fingerprint,
        )


def _validated_list_items(
    *,
    snapshot_kind: str,
    payload: Mapping[str, object],
    max_items: int,
) -> Sequence[object]:
    items = payload.get("items", _MISSING)
    if items is _MISSING or items is None:
        raise ValueError(
            f"{snapshot_kind}.items is missing or null. Refusing to treat a corrupted snapshot as an empty list."
        )
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        raise ValueError(f"{snapshot_kind}.items must be a sequence.")

    if len(items) > max_items:
        logger.warning(
            "Snapshot %s contains %s items, exceeding the configured limit of %s. "
            "Keeping the newest %s items only.",
            snapshot_kind,
            len(items),
            max_items,
            max_items,
        )
        return list(items[-max_items:])
    return items


def _load_list_snapshot(
    *,
    config: TwinrConfig,
    remote_state: LongTermRemoteStateStore | None,
    snapshot_kind: str,
    item_factory: Callable[[Mapping[str, object]], _ItemT],
    max_items: int = _DEFAULT_MAX_LIST_ITEMS,
    max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
    max_compressed_bytes: int = _DEFAULT_MAX_COMPRESSED_BYTES,
    strict_load: bool = False,
) -> tuple[_ItemT, ...]:
    """Load a typed list snapshot from remote state when present."""

    _, payload, _ = _load_raw_snapshot_payload(
        config=config,
        remote_state=remote_state,
        snapshot_kind=snapshot_kind,
        max_payload_bytes=max_payload_bytes,
        max_compressed_bytes=max_compressed_bytes,
    )
    if payload is None:
        return ()

    items = _validated_list_items(
        snapshot_kind=snapshot_kind,
        payload=payload,
        max_items=max_items,
    )

    loaded: list[_ItemT] = []
    dropped_count = 0
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            if strict_load:
                raise ValueError(f"{snapshot_kind}.items[{index}] must be a mapping.")
            dropped_count += 1
            logger.warning(
                "Skipping malformed item in %s at index %s because it is not a mapping.",
                snapshot_kind,
                index,
            )
            continue
        try:
            loaded.append(item_factory(item))
        except Exception as exc:
            if strict_load:
                raise ValueError(f"{snapshot_kind}.items[{index}] failed validation.") from exc
            dropped_count += 1
            logger.warning(
                "Skipping malformed item in %s at index %s during payload decoding: %s",
                snapshot_kind,
                index,
                exc,
            )

    if dropped_count:
        logger.warning(
            "Recovered %s valid item(s) from %s after dropping %s malformed item(s).",
            len(loaded),
            snapshot_kind,
            dropped_count,
        )
    return tuple(loaded)


def _compact_list_payload(
    *,
    snapshot_kind: str,
    items: Sequence[object],
    item_serializer: Callable[[object], Mapping[str, object]],
    max_items: int,
    max_payload_bytes: int,
) -> Mapping[str, object]:
    serialized_items = [dict(item_serializer(item)) for item in items]
    if len(serialized_items) > max_items:
        # BREAKING: append-only list snapshots are now capped to keep remote state bounded on edge devices.
        logger.warning(
            "Snapshot %s contains %s item(s), exceeding the configured limit of %s. "
            "Dropping the oldest %s item(s).",
            snapshot_kind,
            len(serialized_items),
            max_items,
            len(serialized_items) - max_items,
        )
        serialized_items = serialized_items[-max_items:]

    payload: Mapping[str, object] = {
        "schema_version": 1,
        "items": serialized_items,
    }

    payload_bytes = _canonical_json_bytes(payload)
    while len(payload_bytes) > max_payload_bytes and serialized_items:
        if len(serialized_items) == 1:
            raise SnapshotSizeError(
                f"{snapshot_kind} cannot fit within the configured payload limit of {max_payload_bytes} bytes."
            )
        trim_count = max(1, len(serialized_items) // 8)
        logger.warning(
            "Snapshot %s is %s bytes and exceeds the configured limit of %s bytes. "
            "Dropping the oldest %s item(s) and retrying compaction.",
            snapshot_kind,
            len(payload_bytes),
            max_payload_bytes,
            trim_count,
        )
        serialized_items = serialized_items[trim_count:]
        payload = {
            "schema_version": 1,
            "items": serialized_items,
        }
        payload_bytes = _canonical_json_bytes(payload)

    return payload


def _save_list_snapshot(
    *,
    config: TwinrConfig,
    remote_state: LongTermRemoteStateStore | None,
    snapshot_kind: str,
    items: Sequence[object],
    item_serializer: Callable[[object], Mapping[str, object]],
    max_items: int = _DEFAULT_MAX_LIST_ITEMS,
    max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
    max_compressed_bytes: int = _DEFAULT_MAX_COMPRESSED_BYTES,
    compression_threshold_bytes: int = _DEFAULT_COMPRESSION_THRESHOLD_BYTES,
) -> None:
    """Save a typed list snapshot through the remote snapshot adapter."""

    payload = _compact_list_payload(
        snapshot_kind=snapshot_kind,
        items=items,
        item_serializer=item_serializer,
        max_items=max_items,
        max_payload_bytes=max_payload_bytes,
    )
    _save_raw_snapshot_payload(
        config=config,
        remote_state=remote_state,
        snapshot_kind=snapshot_kind,
        payload=payload,
        max_payload_bytes=max_payload_bytes,
        max_compressed_bytes=max_compressed_bytes,
        compression_threshold_bytes=compression_threshold_bytes,
    )


class PersonalitySnapshotStore(Protocol):
    """Describe a loader/saver for the promptable personality snapshot."""

    def load_snapshot(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> PersonalitySnapshot | None:
        """Load the latest personality snapshot for prompt assembly."""

    def save_snapshot(
        self,
        *,
        config: TwinrConfig,
        snapshot: PersonalitySnapshot,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist the latest promptable personality snapshot."""


@dataclass(slots=True)
class RemoteStatePersonalitySnapshotStore:
    """Load and save structured personality state via remote snapshots."""

    snapshot_kind: str = DEFAULT_PERSONALITY_SNAPSHOT_KIND
    max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES
    max_compressed_bytes: int = _DEFAULT_MAX_COMPRESSED_BYTES
    compression_threshold_bytes: int = _DEFAULT_COMPRESSION_THRESHOLD_BYTES

    def load_snapshot(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> PersonalitySnapshot | None:
        """Load a typed snapshot from remote state when one exists."""

        _, payload, _ = _load_raw_snapshot_payload(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.snapshot_kind,
            max_payload_bytes=self.max_payload_bytes,
            max_compressed_bytes=self.max_compressed_bytes,
        )
        if payload is None:
            return None
        return PersonalitySnapshot.from_payload(payload)

    def save_snapshot(
        self,
        *,
        config: TwinrConfig,
        snapshot: PersonalitySnapshot,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist a typed snapshot through remote-primary state."""

        _save_raw_snapshot_payload(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.snapshot_kind,
            payload=snapshot.to_payload(),
            max_payload_bytes=self.max_payload_bytes,
            max_compressed_bytes=self.max_compressed_bytes,
            compression_threshold_bytes=self.compression_threshold_bytes,
        )


class PersonalityEvolutionStore(Protocol):
    """Describe persistence for learning signals and accepted deltas."""

    def load_interaction_signals(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[InteractionSignal, ...]:
        """Load persisted interaction signals."""

    def save_interaction_signals(
        self,
        *,
        config: TwinrConfig,
        signals: Sequence[InteractionSignal],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist interaction signals."""

    def load_place_signals(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[PlaceSignal, ...]:
        """Load persisted place signals."""

    def save_place_signals(
        self,
        *,
        config: TwinrConfig,
        signals: Sequence[PlaceSignal],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist place signals."""

    def load_world_signals(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[WorldSignal, ...]:
        """Load persisted world signals."""

    def save_world_signals(
        self,
        *,
        config: TwinrConfig,
        signals: Sequence[WorldSignal],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist world signals."""

    def load_personality_deltas(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[PersonalityDelta, ...]:
        """Load accepted or rejected personality deltas."""

    def save_personality_deltas(
        self,
        *,
        config: TwinrConfig,
        deltas: Sequence[PersonalityDelta],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist personality deltas."""


@dataclass(slots=True)
class RemoteStatePersonalityEvolutionStore:
    """Persist learning signals and deltas through remote-primary snapshots."""

    interaction_snapshot_kind: str = INTERACTION_SIGNAL_SNAPSHOT_KIND
    place_snapshot_kind: str = PLACE_SIGNAL_SNAPSHOT_KIND
    world_snapshot_kind: str = WORLD_SIGNAL_SNAPSHOT_KIND
    delta_snapshot_kind: str = PERSONALITY_DELTA_SNAPSHOT_KIND
    max_items_per_snapshot: int = _DEFAULT_MAX_LIST_ITEMS
    max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES
    max_compressed_bytes: int = _DEFAULT_MAX_COMPRESSED_BYTES
    compression_threshold_bytes: int = _DEFAULT_COMPRESSION_THRESHOLD_BYTES
    strict_load: bool = False

    def load_interaction_signals(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[InteractionSignal, ...]:
        """Load persisted interaction signals."""

        return _load_list_snapshot(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.interaction_snapshot_kind,
            item_factory=InteractionSignal.from_payload,
            max_items=self.max_items_per_snapshot,
            max_payload_bytes=self.max_payload_bytes,
            max_compressed_bytes=self.max_compressed_bytes,
            strict_load=self.strict_load,
        )

    def save_interaction_signals(
        self,
        *,
        config: TwinrConfig,
        signals: Sequence[InteractionSignal],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist interaction signals."""

        _save_list_snapshot(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.interaction_snapshot_kind,
            items=signals,
            item_serializer=lambda item: item.to_payload(),
            max_items=self.max_items_per_snapshot,
            max_payload_bytes=self.max_payload_bytes,
            max_compressed_bytes=self.max_compressed_bytes,
            compression_threshold_bytes=self.compression_threshold_bytes,
        )

    def load_place_signals(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[PlaceSignal, ...]:
        """Load persisted place signals."""

        return _load_list_snapshot(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.place_snapshot_kind,
            item_factory=PlaceSignal.from_payload,
            max_items=self.max_items_per_snapshot,
            max_payload_bytes=self.max_payload_bytes,
            max_compressed_bytes=self.max_compressed_bytes,
            strict_load=self.strict_load,
        )

    def save_place_signals(
        self,
        *,
        config: TwinrConfig,
        signals: Sequence[PlaceSignal],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist place signals."""

        _save_list_snapshot(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.place_snapshot_kind,
            items=signals,
            item_serializer=lambda item: item.to_payload(),
            max_items=self.max_items_per_snapshot,
            max_payload_bytes=self.max_payload_bytes,
            max_compressed_bytes=self.max_compressed_bytes,
            compression_threshold_bytes=self.compression_threshold_bytes,
        )

    def load_world_signals(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[WorldSignal, ...]:
        """Load persisted world signals."""

        return _load_list_snapshot(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.world_snapshot_kind,
            item_factory=WorldSignal.from_payload,
            max_items=self.max_items_per_snapshot,
            max_payload_bytes=self.max_payload_bytes,
            max_compressed_bytes=self.max_compressed_bytes,
            strict_load=self.strict_load,
        )

    def save_world_signals(
        self,
        *,
        config: TwinrConfig,
        signals: Sequence[WorldSignal],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist world signals."""

        _save_list_snapshot(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.world_snapshot_kind,
            items=signals,
            item_serializer=lambda item: item.to_payload(),
            max_items=self.max_items_per_snapshot,
            max_payload_bytes=self.max_payload_bytes,
            max_compressed_bytes=self.max_compressed_bytes,
            compression_threshold_bytes=self.compression_threshold_bytes,
        )

    def load_personality_deltas(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[PersonalityDelta, ...]:
        """Load persisted personality deltas."""

        return _load_list_snapshot(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.delta_snapshot_kind,
            item_factory=PersonalityDelta.from_payload,
            max_items=self.max_items_per_snapshot,
            max_payload_bytes=self.max_payload_bytes,
            max_compressed_bytes=self.max_compressed_bytes,
            strict_load=self.strict_load,
        )

    def save_personality_deltas(
        self,
        *,
        config: TwinrConfig,
        deltas: Sequence[PersonalityDelta],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist personality deltas."""

        _save_list_snapshot(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.delta_snapshot_kind,
            items=deltas,
            item_serializer=lambda item: item.to_payload(),
            max_items=self.max_items_per_snapshot,
            max_payload_bytes=self.max_payload_bytes,
            max_compressed_bytes=self.max_compressed_bytes,
            compression_threshold_bytes=self.compression_threshold_bytes,
        )
