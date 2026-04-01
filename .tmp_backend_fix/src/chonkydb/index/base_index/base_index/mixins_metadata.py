##REFACTOR: 2026-02-13##
##REVIEWED 2025-10-14
# -*- coding: utf-8 -*-

from __future__ import annotations

# Changelog (Audit Fixes)
#
# - AUDIT-FIX [P0/Correctness]: Prevent unsafe docid-mapping restore by adding plausibility checks for recovered metadata.
# - AUDIT-FIX [P0/Correctness]: Fix array-like `.tolist()` handling in `_prepare_data_for_serialization` for objects without `__dict__` (e.g., numpy.ndarray).
# - AUDIT-FIX [P0/Concurrency]: Snapshot mutable dict metadata before recursive serialization so live writers cannot trip `dictionary changed size during iteration` during commit.
# - AUDIT-FIX [P0/Reliability]: Align close-path persist lock-order with `_persist_metadata_now` by moving `store_index_data` outside INDEX lock.
# - AUDIT-FIX [P1/Correctness]: Parse `CHONK_STRICT_METADATA` using explicit truthy values (matches other env flags).
# - AUDIT-FIX [P1/Reliability]: Defer in-memory teardown on transaction-bridged close until after successful commit; ensure finalize closes cache.
# - AUDIT-FIX [P1/Concurrency]: Prevent parallel `open_index` restore/finalize races by holding the shared open lock through restore+finalize and re-checking `_is_open`.
# - AUDIT-FIX [P2/Performance]: Reduce `get_metadata` lock-hold time for large pickleable metadata via pickle round-trip; fallback to deepcopy.
# - AUDIT-FIX [P2/Reliability]: Prevent unbounded growth of shared per-index locks/guards by using weak registries (GC-driven cleanup).
# - AUDIT-FIX [P2/Observability]: Emit a warning when metadata load falls through to empty state in non-strict mode (except new indices).
import hashlib
import os
import threading
import time
import weakref  # AUDIT-FIX [P2/Reliability]: avoid unbounded growth of shared lock registries via weak references
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from chonkydb.errors_observability.errors import *  # noqa: F403,F401  # chonk_errors
from chonkydb.file_operations.router import (
    FileType as _FT,
)
from chonkydb.serialization.serialization import (
    deserialize_data,
)
from chonkydb.transaction_management.transaction_context import (
    TransactionScope,
)
from chonkydb.transaction_management.transaction_manager import (
    get_global_transaction_manager,
)

from .constants import BASE_INDEX_OPEN_GAUGE, DEFAULT_OFFSETS
from .utils import NoopCtx, decode_dict_keys_and_values

_READ_ONLY_TRUTHY_VALUES = ("1", "true", "yes", "on")


def _read_only_env_enabled() -> bool:
    for env_name in ("CHONK_READ_ONLY", "CHONKY_READ_ONLY"):
        if (
            str(os.environ.get(env_name, "") or "").strip().lower()
            in _READ_ONLY_TRUTHY_VALUES
        ):
            return True
    return False


class BaseIndexMetadataMixin:
    """
    API-stabile Mixin-Basis für Index-Metadaten.
    - Keine öffentliche Signatur geändert.
    - Interne Robustheit, Defensive Programming und Beobachtbarkeit erhöht.
    - Hot-paths bevorzugt, Fallbacks bleiben funktionsgleich, aber präziser.
    """

    _META_OFFSET_REGISTERED = False
    _META_OFFSET_LOCK = threading.RLock()

    # Shared per-index locks (prevent warmup/request open races).
    _OPEN_IO_LOCKS_GUARD = threading.Lock()
    _OPEN_IO_LOCKS: "weakref.WeakValueDictionary[Tuple[str, str], threading.Lock]" = (
        weakref.WeakValueDictionary()
    )  # type: ignore[type-arg]

    def _determine_file_type(self) -> None:
        """
        Bestimmt robust den FileType auf Basis deklarierter index_type bzw. des Namens.
        """
        # Respect explicitly configured file type set by subclasses/facades
        try:
            from chonkydb.file_operations.router import (
                FileType as _FT_LOCAL,
            )  # local import for safety

            if hasattr(self, "_file_type") and isinstance(
                getattr(self, "_file_type"), _FT_LOCAL
            ):
                # Already set to a valid FileType (e.g., RELATIONAL by RelationalIndexManager)
                return
        except Exception:
            # Fall through to best-effort detection
            pass

        if hasattr(self, "index_type") and self.index_type:
            type_lower = str(self.index_type).lower()
            if type_lower == "btree":
                self._file_type = _FT.BTREE
            elif type_lower in ("temporal", "temporal_btree"):
                self._file_type = _FT.TEMPORAL
            elif type_lower == "fulltext":
                self._file_type = _FT.FULLTEXT
            elif type_lower == "graph":
                self._file_type = _FT.GRAPH
            elif type_lower in ("octree", "spatial"):
                self._file_type = _FT.OCTREE
            elif type_lower in ("vector", "hnsw"):
                self._file_type = _FT.HNSW
            elif type_lower == "relational":
                self._file_type = _FT.RELATIONAL
            elif type_lower == "column":
                self._file_type = _FT.COLUMN
            else:
                self._determine_file_type_by_name()
        else:
            self._determine_file_type_by_name()

        chonk_errors.debug(  # noqa: F405
            f"[BaseIndex._determine_file_type] Index '{getattr(self, 'index_name', '?')}' "
            f"(type={getattr(self, 'index_type', 'unknown')}) mapped to file type: {getattr(self, '_file_type', '?')}"
        )

    def _determine_file_type_by_name(self) -> None:
        name_lower = str(getattr(self, "index_name", "")).lower()
        if "temporal" in name_lower or "time" in name_lower:
            self._file_type = _FT.TEMPORAL
        elif "fulltext" in name_lower or "text" in name_lower:
            self._file_type = _FT.FULLTEXT
        elif "graph" in name_lower:
            self._file_type = _FT.GRAPH
        elif "octree" in name_lower or "spatial" in name_lower:
            self._file_type = _FT.OCTREE
        elif "hnsw" in name_lower or "vector" in name_lower:
            self._file_type = _FT.HNSW
        elif "column" in name_lower:
            self._file_type = _FT.COLUMN
        elif "btree" in name_lower:
            self._file_type = _FT.BTREE
        elif "relational" in name_lower:
            self._file_type = _FT.RELATIONAL
        else:
            self._file_type = _FT.BTREE

    def _init_utility_modules(self) -> None:
        from chonkydb.locking.lock_hierarchy import (
            LockHierarchyManager,
        )
        from chonkydb.serialization.serialization import (
            CompressionType,
            FloatPrecision,
            SerializationFormat,
            SerializationOptions,
        )

        self._lock_manager = LockHierarchyManager()
        self._serialization_options = SerializationOptions(
            format=SerializationFormat.CHONKBIN,
            compression=CompressionType.NONE,
            float_precision=FloatPrecision.DOUBLE,
            use_bin_type=True,  # msgpack: bytes bleiben bytes
            strict_map_key=True,  # msgpack: Keys stabil (str/bytes)
            avoid_double_serialization=True,
            avoid_double_compression=True,
        )

        if getattr(self, "_enable_cache", False):
            self._init_cache()
        if getattr(self, "_enable_bloom_filter", False):
            self._init_bloom_filter()

        self._cache_closed = False

    def _init_cache(self) -> None:
        from chonkydb.caching_filtering.caching import (
            UniversalCacheSOTA as cache_class,
        )

        def weight_fn(v: Any) -> int:
            # O(1) konservative Heuristik; vermeidet Tiefenrekursion (siehe sys.getsizeof-Limits).
            import sys

            try:
                if isinstance(v, tuple) and len(v) == 2:
                    payload, meta = v
                    size = 0
                    if isinstance(payload, (bytes, bytearray, memoryview)):
                        size += len(payload)
                    else:
                        size += max(64, min(1_048_576, sys.getsizeof(payload)))
                    if isinstance(meta, (bytes, bytearray, memoryview)):
                        size += len(meta)
                    else:
                        size += max(32, min(65_536, sys.getsizeof(meta)))
                    return max(96, size)

                if isinstance(v, (bytes, bytearray, memoryview)):
                    # Direktgrößen-Pfad für Binärdaten
                    return max(32, len(v))

                # Fallback: shallow size; konservativ clampen
                return max(64, min(1_048_576, sys.getsizeof(v)))
            except Exception:
                return 256  # sichere Untergrenze

        self._cache = cache_class(
            capacity=getattr(self, "_cache_capacity", 0),
            weight_fn=weight_fn,
            eviction_policy="ARC",  # ARC ≫ LRU bei Misch-Workloads
            segments=getattr(self, "_cache_segments", 1),
            default_ttl=getattr(self, "_cache_ttl", 0.0),
            grace_period=getattr(self, "_cache_grace_period", 0.0),
        )
        self._cache_closed = False
        chonk_errors.info(  # noqa: F405
            f"[BaseIndex._init_cache] Initialized cache for {getattr(self, 'index_name', '?')} "
            f"with capacity={getattr(self, '_cache_capacity', None)}, TTL={getattr(self, '_cache_ttl', None)}"
        )

    def _init_bloom_filter(self) -> None:
        from chonkydb.caching_filtering.filtering import (
            StableBloomFilter,
        )

        self._bloom = StableBloomFilter(
            bits=getattr(self, "_bloom_size", 0),
            k=getattr(self, "_bloom_hash_count", 0),
            decay_rate=getattr(self, "_bloom_decay_rate", 0.0),
        )
        chonk_errors.info(  # noqa: F405
            f"[BaseIndex._init_bloom_filter] Initialized bloom filter for {getattr(self, 'index_name', '?')} "
            f"size={getattr(self, '_bloom_size', None)}, k={getattr(self, '_bloom_hash_count', None)}"
        )

    @classmethod
    def _ensure_meta_offset_registered(cls) -> None:
        if not cls._META_OFFSET_REGISTERED:
            with cls._META_OFFSET_LOCK:
                if not cls._META_OFFSET_REGISTERED:
                    try:
                        from chonkydb.management_functions.id_facade import (
                            OffsetRegistry,
                        )

                        OffsetRegistry.register(
                            name="base_index_meta", base=100_000_000, size=100_000_000
                        )
                        cls._META_OFFSET_REGISTERED = True
                        chonk_errors.info(  # noqa: F405
                            "[BaseIndex] Registered base_index_meta offset range: 100_000_000 - 199_999_999"
                        )
                    except Exception as e:
                        chonk_errors.debug(f"[BaseIndex] Offset registration note: {e}")  # noqa: F405
                        cls._META_OFFSET_REGISTERED = True

    def _calc_baseindex_docid(self) -> int:
        """
        Stabiler, rückwärtskompatibler 32-Bit BLAKE2s-Hash in 50M-Space mit festem Basisoffset.
        """
        base_offset = DEFAULT_OFFSETS.get("base_index_meta", 100_000_000)
        base_str = f"BaseIndex::{getattr(self, 'index_name', '')}"
        stable_hash_hex = hashlib.blake2s(
            base_str.encode("utf-8"), digest_size=4
        ).hexdigest()
        hash_value = int(stable_hash_hex, 16) % 50_000_000
        return base_offset + hash_value

    def get_metadata(self) -> Dict[str, Any]:
        """
        Thread-sicherer, defensive Zugriff auf Struktur-Snapshot.
        """
        try:
            # AUDIT-FIX [P2/Performance]: For large, pickleable metadata, prefer a C-accelerated pickle round-trip
            # to reduce time holding the index lock; fall back to deepcopy for full compatibility.
            # CAUTION: For exotic objects with custom deepcopy/pickle semantics, fallback may be exercised.
            blob: Optional[bytes] = None
            data_ref: Any = None

            with self.acquire_lock(
                "get_metadata", getattr(self, "lock_timeout", 0.0), is_write=False
            ):
                if (
                    not hasattr(self, "_index_structure_data")
                    or self._index_structure_data is None
                ):
                    return {}
                data_ref = self._index_structure_data
                if isinstance(data_ref, dict) and len(data_ref) >= 256:
                    try:
                        import pickle as _pickle

                        blob = _pickle.dumps(
                            data_ref, protocol=getattr(_pickle, "HIGHEST_PROTOCOL", 5)
                        )
                    except Exception:
                        blob = None
                if blob is None:
                    return deepcopy(data_ref)

            if blob is not None:
                try:
                    import pickle as _pickle

                    return _pickle.loads(blob)
                except Exception:
                    # Fallback: preserve legacy semantics by copying under lock if unpickling fails.
                    with self.acquire_lock(
                        "get_metadata_fallback",
                        getattr(self, "lock_timeout", 0.0),
                        is_write=False,
                    ):
                        if (
                            not hasattr(self, "_index_structure_data")
                            or self._index_structure_data is None
                        ):
                            return {}
                        return deepcopy(self._index_structure_data)

            return {}
        except Exception as e:
            chonk_errors.error(  # noqa: F405
                f"[get_metadata] index={getattr(self, 'index_name', '?')} Error while fetching metadata: {e}",
                original_exception=e,
            )
            raise RuntimeError(
                f"Error retrieving metadata for index '{getattr(self, 'index_name', '?')}'."
            ) from e

    def _metadata_plausibility_check(self, data: Dict[str, Any]) -> bool:
        """
        Best-effort Plausibility check for recovered metadata.
        Returns False only on clear mismatch signals; otherwise True to preserve legacy behavior.

        NOTE: This is intentionally permissive to avoid breaking legacy metadata layouts.
        """
        try:
            idx_name = str(getattr(self, "index_name", "") or "")
        except Exception:
            idx_name = ""

        try:
            # AUDIT-FIX [P0/Correctness]: Reduce risk of docid-mapping collisions / wrong restores by rejecting obvious mismatches.
            for k in ("index_name", "index", "name"):
                try:
                    v = data.get(k)
                except Exception:
                    continue
                if v is None:
                    continue
                try:
                    if idx_name and str(v) != idx_name:
                        return False
                except Exception:
                    continue
        except Exception:
            pass

        # Tighten for relational indices where we know required shape.
        try:
            if getattr(self, "_file_type", None) == _FT.RELATIONAL:
                if "relational_header" not in data:
                    return False
        except Exception:
            pass

        return True

    def open_index(self, tx_id: Optional[str] = None) -> None:
        """
        Lädt Metadaten robust; schnelle Pfade nutzen Memoization, ohne Kompatibilität zu brechen.
        """
        read_only = _read_only_env_enabled()
        self._opened_read_only = bool(read_only)
        is_new_index = bool(getattr(self, "_is_new_index", False))
        open_use_tx_env = str(
            os.environ.get("CHONKY_BASE_OPEN_USE_TX", "") or ""
        ).strip().lower() in ("1", "true", "yes", "on")
        open_requires_tx_scope = bool(is_new_index or open_use_tx_env)

        # Existing index opens are restore/read paths and must not enqueue a commit by default.
        # Only new-index open flows or explicit operator opt-in may use a transaction scope here.
        if (
            tx_id is None
            and hasattr(self, "_transaction_bridge")
            and self._transaction_bridge
            and (not read_only)
            and bool(open_requires_tx_scope)
        ):
            with self._transaction_bridge.transaction_scope(
                f"open_index_{getattr(self, 'index_name', '?')}",
                auto_prepare=True,
                auto_commit=True,
            ) as bridge_tx_id:
                return self.open_index(tx_id=bridge_tx_id)

        if tx_id:
            if hasattr(self, "_transaction_bridge") and self._transaction_bridge:
                self._transaction_bridge.ensure_lazy_registration_for_tx(tx_id)
            else:
                self._ensure_lazy_bridge_registration(tx_id)
            router = getattr(self, "router", None)
            if router is not None and hasattr(
                router, "_ensure_registered_for_transaction"
            ):
                try:
                    router._ensure_registered_for_transaction(tx_id)
                except Exception as e:
                    raise TransactionError(  # noqa: F405
                        f"Router pre-registration failed for tx={tx_id}"
                    ) from e

        if getattr(self, "_is_open", False):
            chonk_errors.debug(  # noqa: F405
                f"[BaseIndex.open_index] index={getattr(self, 'index_name', '?')} already open; skipping load"
            )
            return

        # Typ und Cache vorbereiten
        self._determine_file_type()
        if getattr(self, "_enable_cache", False) and (
            (not hasattr(self, "_cache"))
            or (self._cache is None)
            or getattr(self, "_cache_closed", True)
            or getattr(self._cache, "_closed", False)
        ):
            self._init_cache()

        if (
            not hasattr(self, "_metadata_key_lookup_cache")
            or self._metadata_key_lookup_cache is None
        ):
            self._metadata_key_lookup_cache = {}

        chonk_errors.info(
            f"[BaseIndex.open_index] index={getattr(self, 'index_name', '?')} loading metadata"
        )  # noqa: F405

        # Avoid expensive/unsafe header reloads while ANY transaction is active (process-wide).
        active_tx = False
        try:
            tm = get_global_transaction_manager()
            active_tx = bool(
                tm is not None
                and hasattr(tm, "is_active_transaction")
                and tm.is_active_transaction()
            )
        except Exception:
            active_tx = False
        if tx_id is not None:
            active_tx = True

        # Shared lock per (file_type,index_name) so concurrent instances don't race open/finalize.
        try:
            _ft = (
                getattr(getattr(self, "_file_type", None), "value", None)
                or getattr(self, "_file_type", None)
                or "?"
            )
            lock_key = (str(_ft), str(getattr(self, "index_name", "") or ""))
        except Exception:
            lock_key = ("?", str(getattr(self, "index_name", "") or ""))

        with self.__class__._OPEN_IO_LOCKS_GUARD:
            shared_lock = self.__class__._OPEN_IO_LOCKS.get(lock_key)
            if shared_lock is None:
                shared_lock = threading.Lock()
                self.__class__._OPEN_IO_LOCKS[lock_key] = shared_lock
        self._open_io_lock = shared_lock

        with self._open_io_lock:
            # AUDIT-FIX [P1/Concurrency]: Re-check open state under the shared open lock to avoid parallel restore/finalize races.
            if getattr(self, "_is_open", False):
                chonk_errors.debug(  # noqa: F405
                    f"[BaseIndex.open_index] index={getattr(self, 'index_name', '?')} became open while waiting; skipping load"
                )
                return

            # Keep index.router aligned with engine.chonk_router on every open path.
            # Root cause: singleton/lazy init order can leave self.router stale/None while
            # engine router is available, causing false "router missing" startup failures.
            router_obj = getattr(self, "router", None)
            engine_obj = getattr(self, "engine", None)
            engine_router = (
                getattr(engine_obj, "chonk_router", None)
                if engine_obj is not None
                else None
            )

            if (
                engine_obj is not None
                and engine_router is None
                and hasattr(engine_obj, "_init_router")
            ):
                try:
                    engine_obj._init_router()
                    engine_router = getattr(engine_obj, "chonk_router", None)
                except Exception:
                    engine_router = None

            if engine_router is not None and router_obj is not engine_router:
                self.router = engine_router
                router_obj = engine_router

            if router_obj is not None:
                try:
                    pm = getattr(self, "_persistence_manager", None)
                    if pm is not None and getattr(pm, "router", None) is not router_obj:
                        setattr(pm, "router", router_obj)
                except Exception:
                    pass

            if router_obj is None:
                try:
                    is_init_attr = getattr(engine_obj, "is_initialized", None)
                    if callable(is_init_attr):
                        engine_is_initialized = bool(is_init_attr())
                    elif is_init_attr is None:
                        engine_is_initialized = None
                    else:
                        engine_is_initialized = bool(is_init_attr)
                except Exception:
                    engine_is_initialized = None
                raise chonk_errors.create_error(  # noqa: F405
                    error_type=ErrorType.CONFIGURATION,  # noqa: F405
                    message=(
                        f"[BaseIndex.open_index] index={getattr(self, 'index_name', '?')}: "
                        "router is missing; cannot load metadata"
                    ),
                    severity=ErrorSeverity.CRITICAL,  # noqa: F405
                    metadata={
                        "index_name": getattr(self, "index_name", "?"),
                        "engine_has_chonk_router": bool(engine_router is not None),
                        "engine_initialized": engine_is_initialized,
                    },
                )

            self._opening_in_progress = True
            try:
                scope_tx_id = tx_id if tx_id is not None else None
                doc_id = self._calc_baseindex_docid()
                index_type_name = self.__class__.__name__
                metadata_key = f"index_metadata_{index_type_name}_{getattr(self, 'index_name', '')}_{doc_id}"
                chonk_errors.info(  # noqa: F405
                    f"[BaseIndex.open_index] Loading with key: {metadata_key}, doc_id: {doc_id}"
                )

                loaded_data: Optional[Dict[str, Any]] = None

                # Strict metadata load for integrity-critical indices.
                # FULLTEXT/GRAPH must never silently degrade to empty state; that produces 0-hits bugs.
                ft = getattr(self, "_file_type", None)
                _strict_env = (
                    str(os.environ.get("CHONK_STRICT_METADATA", "") or "")
                    .strip()
                    .lower()
                )
                strict_meta = (_strict_env in ("1", "true", "yes", "on")) or (
                    ft in (_FT.FULLTEXT, _FT.GRAPH)
                )
                # -1) Subclass-specific early recovery hook for indices that can reconstruct
                # metadata from sidecars without opening the main index file first.
                if loaded_data is None and not is_new_index:
                    recover_before_router = getattr(
                        self, "_recover_missing_metadata_before_router_read_on_open", None
                    )
                    if callable(recover_before_router):
                        try:
                            recovered = recover_before_router(
                                metadata_key=str(metadata_key),
                                doc_id=int(doc_id),
                                strict_meta=bool(strict_meta),
                            )
                        except Exception as recovery_err:
                            chonk_errors.info(  # noqa: F405
                                f"[BaseIndex.open_index] Early metadata recovery hook failed for "
                                f"index='{getattr(self, 'index_name', '')}': {recovery_err}"
                            )
                        else:
                            if isinstance(
                                recovered, dict
                            ) and self._metadata_plausibility_check(recovered):
                                loaded_data = recovered
                                chonk_errors.info(  # noqa: F405
                                    f"[BaseIndex.open_index] Recovered metadata for "
                                    f"index='{getattr(self, 'index_name', '')}' via pre-router subclass hook"
                                )
                            elif recovered is not None:
                                chonk_errors.warning(  # noqa: F405
                                    f"[BaseIndex.open_index] Early metadata recovery hook returned implausible "
                                    f"data for index='{getattr(self, 'index_name', '')}'; ignoring"
                                )

                # 0) Memoized Key Lookup
                try:
                    memo_key = self._metadata_key_lookup_cache.get(
                        (
                            getattr(self._file_type, "value", None),
                            getattr(self, "index_name", ""),
                        )
                    )
                except Exception:
                    memo_key = None
                if memo_key and memo_key != metadata_key:
                    try:
                        raw_data = self.router.read_index_data(
                            file_type=self._file_type, key=memo_key
                        )
                        if isinstance(raw_data, dict):
                            loaded_data = raw_data
                    except KeyError:
                        pass
                    except Exception as exc:
                        if strict_meta:
                            raise PersistencyError(  # noqa: F405
                                "Failed to load index metadata (memoized key)",
                                severity=ErrorSeverity.CRITICAL,  # noqa: F405
                                original_exception=exc,
                                metadata={
                                    "index_name": getattr(self, "index_name", "?"),
                                    "file_type": getattr(
                                        getattr(self, "_file_type", None),
                                        "value",
                                        str(getattr(self, "_file_type", None)),
                                    ),
                                    "metadata_key": str(metadata_key),
                                    "memo_key": str(memo_key),
                                },
                            ) from exc

                # 1) Direkt-Read
                if loaded_data is None:
                    try:
                        raw_data = self.router.read_index_data(
                            file_type=self._file_type, key=metadata_key
                        )
                        if isinstance(raw_data, dict):
                            loaded_data = raw_data
                        try:
                            self._metadata_key_lookup_cache[
                                (
                                    getattr(self._file_type, "value", None),
                                    getattr(self, "index_name", ""),
                                )
                            ] = metadata_key
                        except Exception:
                            pass
                    except KeyError:
                        pass
                    except Exception as exc:
                        if strict_meta:
                            raise PersistencyError(  # noqa: F405
                                "Failed to load index metadata (direct read)",
                                severity=ErrorSeverity.CRITICAL,  # noqa: F405
                                original_exception=exc,
                                metadata={
                                    "index_name": getattr(self, "index_name", "?"),
                                    "file_type": getattr(
                                        getattr(self, "_file_type", None),
                                        "value",
                                        str(getattr(self, "_file_type", None)),
                                    ),
                                    "metadata_key": str(metadata_key),
                                },
                            ) from exc

                # 2) Relational-Sentinel-Fallback
                if (
                    loaded_data is None
                    and getattr(self, "_file_type", None) == _FT.RELATIONAL
                ):
                    try:
                        sentinel_key = (
                            f"relation_header_{getattr(self, 'index_name', '')}"
                        )
                        hdr_data = self.router.read_index_data(
                            file_type=self._file_type, key=sentinel_key
                        )
                        if isinstance(hdr_data, dict) and hdr_data:
                            loaded_data = {"relational_header": hdr_data}
                            chonk_errors.info(  # noqa: F405
                                f"[BaseIndex.open_index] Loaded relational header via sentinel key '{sentinel_key}'"
                            )
                    except Exception:
                        pass

                # 3) Header-Scan
                if loaded_data is None and not strict_meta:
                    try:
                        core = self.router.get_core(self._file_type)
                        if (not active_tx) and hasattr(core, "reload_header_state"):
                            try:
                                core.reload_header_state()
                            except Exception:
                                pass
                        hdr = getattr(core, "header", None)
                        if hdr and hasattr(hdr, "offset_table"):
                            ns_key = None
                            try:
                                if hasattr(core, "_get_namespaced_key"):
                                    ns_key = core._get_namespaced_key(metadata_key)
                            except Exception:
                                ns_key = None

                            candidate_keys = []
                            if ns_key:
                                candidate_keys.append(str(ns_key))
                            candidate_keys.append(metadata_key)

                            index_type_name = self.__class__.__name__
                            prefix_raw = f"index_metadata_{index_type_name}_{getattr(self, 'index_name', '')}_"
                            prefix_ns = f"{getattr(self._file_type, 'value', str(self._file_type))}:{prefix_raw}"
                            generic_fragment = "index_metadata_"
                            index_name_fragment = f"_{getattr(self, 'index_name', '')}_"

                            target = None
                            _scan_budget = 5000
                            _scanned = 0
                            for k in hdr.offset_table.keys():
                                ks = str(k)
                                if (
                                    ks in candidate_keys
                                    or ks.startswith(prefix_raw)
                                    or ks.startswith(prefix_ns)
                                    or (
                                        generic_fragment in ks
                                        and index_name_fragment in ks
                                        and (
                                            ks.startswith(
                                                f"{getattr(self._file_type, 'value', str(self._file_type))}:"
                                            )
                                            or ":" not in ks
                                        )
                                    )
                                ):
                                    target = ks
                                    break
                                _scanned += 1
                                if _scanned >= _scan_budget:
                                    chonk_errors.debug(  # noqa: F405
                                        f"[BaseIndex.open_index] Header scan budget reached ({_scan_budget}) while seeking metadata key"
                                    )
                                    break

                            if target is not None:
                                pos, length, _ = hdr.offset_table[target]
                                block = core._read_block(pos, length)
                                if isinstance(block, dict):
                                    loaded_data = block
                                else:
                                    loaded_data = deserialize_data(
                                        block,
                                        getattr(
                                            core, "_default_serialization_options", None
                                        )
                                        or self._serialization_options,
                                        is_temporal_data=False,
                                    )
                                chonk_errors.info(  # noqa: F405
                                    f"[BaseIndex.open_index] Header-scan loaded metadata for key '{target}' (pos={pos}, len={length})"
                                )
                                try:
                                    self._metadata_key_lookup_cache[
                                        (
                                            getattr(self._file_type, "value", None),
                                            getattr(self, "index_name", ""),
                                        )
                                    ] = target
                                except Exception:
                                    pass
                    except Exception:
                        pass

                # 4) Cross-File-Scan
                if loaded_data is None and not strict_meta:
                    try:
                        from chonkydb.file_operations.router import (
                            FileType as _FT_ALL,
                        )

                        candidate_types = [
                            _FT_ALL.RELATIONAL,
                            _FT_ALL.BTREE,
                            _FT_ALL.GRAPH,
                            _FT_ALL.FULLTEXT,
                            _FT_ALL.OCTREE,
                            _FT_ALL.HNSW,
                            _FT_ALL.TEMPORAL,
                            _FT_ALL.COLUMN,
                        ]
                        index_type_name2 = self.__class__.__name__
                        raw_prefix = f"index_metadata_{index_type_name2}_{getattr(self, 'index_name', '')}_"
                        anyclass_prefix_fragment = "index_metadata_"
                        index_name_fragment = f"_{getattr(self, 'index_name', '')}_"

                        for ftype in candidate_types:
                            try:
                                # IMPORTANT: avoid creating empty .chonk files during recovery probing.
                                # `router.get_core()` will create an empty file if it doesn't exist.
                                # Cross-file scan is a best-effort recovery for legacy/misrouted metadata,
                                # so we only inspect file types that already exist on disk.
                                cand_path = None
                                try:
                                    cfg = getattr(self.router, "file_configs", {}).get(
                                        ftype
                                    )
                                    if cfg is not None and hasattr(cfg, "file_name"):
                                        cand_path = (
                                            getattr(self.router, "base_path", None)
                                            / cfg.file_name
                                        )  # type: ignore[operator]
                                except Exception:
                                    cand_path = None
                                if cand_path is None or not Path(cand_path).exists():
                                    continue

                                c = self.router.get_core(ftype)
                                if (not active_tx) and hasattr(
                                    c, "reload_header_state"
                                ):
                                    try:
                                        c.reload_header_state()
                                    except Exception:
                                        pass
                                hdr2 = getattr(c, "header", None)
                                if not hdr2 or not hasattr(hdr2, "offset_table"):
                                    continue

                                try:
                                    ns_prefix = (
                                        c._get_namespaced_key(raw_prefix)
                                        if hasattr(c, "_get_namespaced_key")
                                        else f"{ftype.value}:{raw_prefix}"
                                    )
                                except Exception:
                                    ns_prefix = f"{ftype.value}:{raw_prefix}"

                                found_key = None
                                _scan_budget = 5000
                                _scanned = 0
                                for k in hdr2.offset_table.keys():
                                    ks = str(k)
                                    if (
                                        ks.startswith(str(ns_prefix))
                                        or ks.startswith(raw_prefix)
                                        or (
                                            anyclass_prefix_fragment in ks
                                            and index_name_fragment in ks
                                            and (
                                                ks.startswith(f"{ftype.value}:")
                                                or ":" not in ks
                                            )
                                        )
                                    ):
                                        found_key = ks
                                        break
                                    _scanned += 1
                                    if _scanned >= _scan_budget:
                                        chonk_errors.debug(  # noqa: F405
                                            f"[BaseIndex.open_index] Cross-file scan budget reached ({_scan_budget}) while seeking metadata key"
                                        )
                                        break

                                if found_key:
                                    pos, length, _ = hdr2.offset_table[found_key]
                                    block = c._read_block(pos, length)
                                    if isinstance(block, dict):
                                        loaded_data = block
                                    else:
                                        loaded_data = deserialize_data(
                                            block,
                                            getattr(
                                                c,
                                                "_default_serialization_options",
                                                None,
                                            )
                                            or self._serialization_options,
                                            is_temporal_data=False,
                                        )
                                    chonk_errors.info(  # noqa: F405
                                        f"[BaseIndex.open_index] Cross-file scan loaded metadata for '{found_key}' from file_type={ftype.value}"
                                    )
                                    try:
                                        # memoize unter Ziel-Datei-Typ
                                        self._metadata_key_lookup_cache[
                                            (
                                                getattr(self._file_type, "value", None),
                                                getattr(self, "index_name", ""),
                                            )
                                        ] = found_key
                                    except Exception:
                                        pass
                                    break
                            except Exception:
                                continue
                    except Exception as _xfile_err:
                        chonk_errors.debug(
                            f"[BaseIndex.open_index] Cross-file scan skipped due to error: {_xfile_err}"
                        )  # noqa: F405

                # 5) DocID-Mapping-Fallback
                if loaded_data is None and not strict_meta:
                    try:
                        doc_id2 = self._calc_baseindex_docid()
                        mapping_mgr = getattr(self, "engine", None)
                        mapping_mgr = getattr(
                            mapping_mgr, "docid_mapping_manager", None
                        )
                        if mapping_mgr is not None:
                            record = mapping_mgr.get_mapping(doc_id2)
                            if record and hasattr(record, "subindexes"):
                                loc = record.subindexes.get(
                                    getattr(self, "index_name", "")
                                )
                                if isinstance(loc, (tuple, list)) and len(loc) >= 2:
                                    pos, length = int(loc[0]), int(loc[1])
                                    core = self.router.get_core(self._file_type)
                                    if (not active_tx) and hasattr(
                                        core, "reload_header_state"
                                    ):
                                        try:
                                            core.reload_header_state()
                                        except Exception:
                                            pass
                                    block = core._read_block(pos, length)
                                    if isinstance(block, dict):
                                        cand = block
                                    else:
                                        cand = deserialize_data(
                                            block,
                                            getattr(
                                                core,
                                                "_default_serialization_options",
                                                None,
                                            )
                                            or self._serialization_options,
                                            is_temporal_data=False,
                                        )
                                    # AUDIT-FIX [P0/Correctness]: Guard against docid collisions / wrong mapping restores.
                                    if isinstance(
                                        cand, dict
                                    ) and self._metadata_plausibility_check(cand):
                                        loaded_data = cand
                                        chonk_errors.info(  # noqa: F405
                                            f"[BaseIndex.open_index] DocID mapping fallback restored metadata for '{getattr(self, 'index_name', '')}' "
                                            f"at pos={pos}, len={length}"
                                        )
                                    else:
                                        chonk_errors.warning(  # noqa: F405
                                            f"[BaseIndex.open_index] DocID mapping fallback returned implausible metadata for "
                                            f"index='{getattr(self, 'index_name', '')}'; ignoring"
                                        )
                    except Exception as map_err:
                        chonk_errors.info(
                            f"[BaseIndex.open_index] DocID mapping fallback failed: {map_err}"
                        )  # noqa: F405

                # 6) Subclass-specific recovery hook for indices that can reconstruct
                # minimal metadata from durable sidecars or per-record state.
                if loaded_data is None and not is_new_index:
                    recover_missing = getattr(
                        self, "_recover_missing_metadata_on_open", None
                    )
                    if callable(recover_missing):
                        try:
                            recovered = recover_missing(
                                metadata_key=str(metadata_key),
                                doc_id=int(doc_id),
                                strict_meta=bool(strict_meta),
                            )
                        except Exception as recovery_err:
                            chonk_errors.info(  # noqa: F405
                                f"[BaseIndex.open_index] Missing-metadata recovery hook failed for "
                                f"index='{getattr(self, 'index_name', '')}': {recovery_err}"
                            )
                        else:
                            if isinstance(
                                recovered, dict
                            ) and self._metadata_plausibility_check(recovered):
                                loaded_data = recovered
                                chonk_errors.info(  # noqa: F405
                                    f"[BaseIndex.open_index] Recovered missing metadata for "
                                    f"index='{getattr(self, 'index_name', '')}' via subclass hook"
                                )
                            elif recovered is not None:
                                chonk_errors.warning(  # noqa: F405
                                    f"[BaseIndex.open_index] Missing-metadata recovery hook returned implausible "
                                    f"data for index='{getattr(self, 'index_name', '')}'; ignoring"
                                )

                # Daten verarbeiten
                if loaded_data is None:
                    if strict_meta and not is_new_index:
                        # Allow truly empty/new indices to open without metadata.
                        # Fail-fast if any non-metadata keys already exist (inconsistent/corrupt state).
                        try:
                            core = self.router.get_core(self._file_type)
                            ot = (
                                getattr(
                                    getattr(core, "header", None), "offset_table", {}
                                )
                                or {}
                            )
                            idx_name = str(getattr(self, "index_name", "") or "")
                            if idx_name:
                                non_meta = [
                                    k
                                    for k in ot.keys()
                                    if (
                                        idx_name in str(k)
                                        and "index_metadata" not in str(k)
                                    )
                                ]
                            else:
                                non_meta = [
                                    k
                                    for k in ot.keys()
                                    if "index_metadata" not in str(k)
                                ]
                        except Exception as exc:
                            raise PersistencyError(  # noqa: F405
                                "Missing index metadata (strict mode) preflight failed",
                                severity=ErrorSeverity.CRITICAL,  # noqa: F405
                                original_exception=exc,
                                metadata={
                                    "index_name": getattr(self, "index_name", "?"),
                                    "file_type": getattr(
                                        getattr(self, "_file_type", None),
                                        "value",
                                        str(getattr(self, "_file_type", None)),
                                    ),
                                    "metadata_key": str(metadata_key),
                                    "doc_id": int(doc_id),
                                },
                            ) from exc

                        if non_meta:
                            raise PersistencyError(  # noqa: F405
                                "Missing index metadata (strict mode)",
                                severity=ErrorSeverity.CRITICAL,  # noqa: F405
                                metadata={
                                    "index_name": getattr(self, "index_name", "?"),
                                    "file_type": getattr(
                                        getattr(self, "_file_type", None),
                                        "value",
                                        str(getattr(self, "_file_type", None)),
                                    ),
                                    "metadata_key": str(metadata_key),
                                    "doc_id": int(doc_id),
                                    "non_meta_key_sample": [
                                        str(k) for k in list(non_meta)[:10]
                                    ],
                                    "non_meta_key_count": int(len(non_meta)),
                                },
                            )

                    self._index_structure_data = {}
                    # AUDIT-FIX [P2/Observability]: In non-strict mode, warn on missing metadata (except for new indices).
                    if (not strict_meta) and (not is_new_index):
                        read_only_open = bool(getattr(self, "_opened_read_only", False))
                        log_fn = (
                            chonk_errors.info
                            if read_only_open
                            else chonk_errors.warning
                        )  # noqa: F405
                        log_fn(
                            f"[BaseIndex.open_index] index={getattr(self, 'index_name', '?')} opened with EMPTY metadata (non-strict); key={metadata_key}"
                        )
                elif isinstance(loaded_data, dict):
                    try:
                        if any(
                            isinstance(k, (bytes, bytearray, memoryview))
                            for k in loaded_data.keys()
                        ):
                            self._index_structure_data = decode_dict_keys_and_values(
                                loaded_data
                            )
                        else:
                            self._index_structure_data = loaded_data
                    except Exception:
                        self._index_structure_data = decode_dict_keys_and_values(
                            loaded_data
                        )
                else:
                    if strict_meta and not is_new_index:
                        raise PersistencyError(  # noqa: F405
                            "Corrupt index metadata: expected dict",
                            severity=ErrorSeverity.CRITICAL,  # noqa: F405
                            metadata={
                                "index_name": getattr(self, "index_name", "?"),
                                "file_type": getattr(
                                    getattr(self, "_file_type", None),
                                    "value",
                                    str(getattr(self, "_file_type", None)),
                                ),
                                "metadata_key": str(metadata_key),
                                "loaded_type": type(loaded_data).__name__,
                                "doc_id": int(doc_id),
                            },
                        )
                    self._index_structure_data = {}
            finally:
                self._opening_in_progress = False

            # AUDIT-FIX [P1/Concurrency]: Keep restore+finalize under the same shared open lock to prevent concurrent restores.
            try:
                self._restore_all_metadata_from_baseindex()
            except Exception as restore_err:
                raise PersistencyError(  # noqa: F405
                    f"Failed to restore internal state for index '{getattr(self, 'index_name', '?')}' from loaded metadata"
                ) from restore_err

            # Serialize finalize across instances (warmup vs first request) to avoid spurious LockErrors.
            finalize_timeout_raw = str(
                os.environ.get("CHONK_OPEN_INDEX_FINALIZE_LOCK_TIMEOUT_S", "") or ""
            ).strip()
            finalize_timeout_s = 60.0
            if finalize_timeout_raw:
                try:
                    finalize_timeout_s = float(finalize_timeout_raw)
                except Exception:
                    finalize_timeout_s = 60.0
            if finalize_timeout_s <= 0.0:
                finalize_timeout_s = 60.0

            with self.acquire_lock(
                "open_index_finalize", finalize_timeout_s, is_write=False
            ):
                self._is_open = True
                chonk_errors.info(
                    "[open_index] {} is now open".format(
                        getattr(self, "index_name", "?")
                    )
                )  # noqa: F405
                try:
                    BASE_INDEX_OPEN_GAUGE.inc()
                except Exception:
                    pass

            # Nicht-blockierende Nebenaktivitäten best-effort
            try:
                self.run_hpc_aggregator()
            except Exception:
                pass

            emit_open_lifecycle = bool(
                (not read_only)
                and (
                    tx_id is not None
                    or bool(open_requires_tx_scope)
                )
            )
            if emit_open_lifecycle:
                self._replicate_if_needed(
                    "OPEN_INDEX",
                    {"index_name": getattr(self, "index_name", "?"), "doc_id": doc_id},
                )
                self._write_wal_record(
                    "OPEN_INDEX",
                    {
                        "index": getattr(self, "index_name", "?"),
                        "doc_id": doc_id,
                        "status": "opened",
                    },
                )

    def close_index(self, tx_id: Optional[str] = None) -> None:
        if hasattr(self, "_closing_in_progress") and self._closing_in_progress:
            chonk_errors.debug(  # noqa: F405
                f"[BaseIndex.close_index] Close already in progress for {getattr(self, 'index_name', '?')}, skipping recursive call"
            )
            return

        read_only = _read_only_env_enabled()

        if (
            tx_id is None
            and hasattr(self, "_transaction_bridge")
            and self._transaction_bridge
            and (not read_only)
        ):
            current_tx = TransactionScope.current_tx_id()
            if current_tx:
                return self._close_index_impl(tx_id=current_tx)

            bridge_tx_id = None
            result = None
            try:
                # AUDIT-FIX [P1/Reliability]: Defer in-memory teardown until after successful commit in bridge-managed close.
                self._defer_close_teardown = True
                with self._transaction_bridge.transaction_scope(
                    f"close_index_{getattr(self, 'index_name', '?')}",
                    auto_prepare=True,
                    auto_commit=False,
                ) as bridge_tx_id:
                    result = self._close_index_impl(tx_id=bridge_tx_id)
                get_global_transaction_manager().commit_transaction(bridge_tx_id)
                self._finalize_index_closure()
                return result
            except Exception as e:
                # Avoid spurious 'TX not found' by checking active status before rollback
                try:
                    if bridge_tx_id is not None:
                        _tm = get_global_transaction_manager()
                        if _tm.is_active(bridge_tx_id):
                            _tm.rollback_transaction(bridge_tx_id)
                        else:
                            chonk_errors.debug(  # noqa: F405
                                f"[BaseIndex.close_index] TX already completed; skip rollback for {bridge_tx_id}"
                            )
                except Exception as rollback_err:
                    chonk_errors.warning(  # noqa: F405
                        f"[BaseIndex.close_index] Rollback failed for {getattr(self, 'index_name', '?')}: {rollback_err}",
                        exc_info=True,
                    )
                raise e
            finally:
                try:
                    self._defer_close_teardown = False
                except Exception:
                    pass

        return self._close_index_impl(tx_id=tx_id)

    def _close_index_impl(self, tx_id: Optional[str] = None) -> None:
        if tx_id:
            if hasattr(self, "_transaction_bridge") and self._transaction_bridge:
                self._transaction_bridge.ensure_lazy_registration_for_tx(tx_id)
            else:
                self._ensure_lazy_bridge_registration(tx_id)

        if not getattr(self, "_is_open", False):
            chonk_errors.warning(  # noqa: F405
                f"[BaseIndex.close_index] index={getattr(self, 'index_name', '?')} not open; skipping close"
            )
            return

        read_only = _read_only_env_enabled()
        if read_only:
            # Read-only mode: never persist metadata or touch WAL/replication.
            self._closing_in_progress = True
            try:
                if getattr(self, "_cache", None) is not None and not getattr(
                    self, "_cache_closed", True
                ):
                    try:
                        self._cache.close()
                        chonk_errors.info(  # noqa: F405
                            f"[BaseIndex.close_index] Closed cache for {getattr(self, 'index_name', '?')}"
                        )
                    finally:
                        self._cache_closed = True
                self._cache = None
            finally:
                self._closing_in_progress = False
            self._finalize_index_closure()
            return

        self._closing_in_progress = True
        try:
            chonk_errors.info(
                f"[BaseIndex.close_index] storing metadata for index={getattr(self, 'index_name', '?')}"
            )  # noqa: F405

            # AUDIT-FIX [P0/Reliability]: Snapshot/prepare under INDEX lock, then persist outside to avoid lock-order violations.
            with self.acquire_lock(
                "close_index", getattr(self, "lock_timeout", 0.0), is_write=True
            ):
                if not getattr(self, "router", None):
                    raise chonk_errors.create_error(  # noqa: F405
                        message=f"[BaseIndex.close_index] Router missing for index '{getattr(self, 'index_name', '?')}'",
                        error_type=ErrorType.CONFIGURATION,  # noqa: F405
                        severity=ErrorSeverity.CRITICAL,  # noqa: F405
                    )
                scope_tx_id = tx_id if tx_id is not None else None
                self._flush_pending_writes(tx_id=scope_tx_id)
                data_to_serialize = deepcopy(getattr(self, "_index_structure_data", {}))
                serializable_data = self._prepare_data_for_serialization(
                    data_to_serialize
                )
                doc_id = self._calc_baseindex_docid()
                index_type_name = self.__class__.__name__
                metadata_key = f"index_metadata_{index_type_name}_{getattr(self, 'index_name', '')}_{doc_id}"

                # Relational-Validierung (zusätzlich)
                try:
                    if getattr(self, "_file_type", None) == _FT.RELATIONAL:
                        if (not isinstance(self._index_structure_data, dict)) or (
                            "relational_header" not in self._index_structure_data
                        ):
                            raise PersistencyError(  # noqa: F405
                                f"Relational index '{getattr(self, 'index_name', '?')}' snapshot missing 'relational_header' (close_index)."
                            )
                except Exception as _rel_check_err:
                    if not isinstance(_rel_check_err, PersistencyError):  # noqa: F405
                        raise PersistencyError(  # noqa: F405
                            f"Relational metadata validation failed during close for '{getattr(self, 'index_name', '?')}': {_rel_check_err}"
                        ) from _rel_check_err

            # Persist outside INDEX lock (may allocate FREESPACE/STORAGE inside router).
            try:
                pos, length = self.router.store_index_data(
                    file_type=self._file_type,
                    key=metadata_key,
                    data=serializable_data,
                    tx_id=scope_tx_id,
                )
            except Exception as store_err:
                try:
                    from chonkydb.locking.lock_hierarchy import (
                        thread_lock_stack_snapshot,
                    )

                    held = thread_lock_stack_snapshot()
                except Exception:
                    held = []
                chonk_errors.error(  # noqa: F405
                    "[BaseIndex._persist_metadata_now] store_index_data failed",
                    metadata={
                        "index": getattr(self, "index_name", "?"),
                        "tx_id": tx_id or "",
                        "metadata_key": metadata_key,
                        "held_locks": held,
                    },
                    original_exception=store_err,
                )
                raise

            chonk_errors.info(
                f"[BaseIndex.close_index] Stored metadata at pos={pos}, len={length}"
            )  # noqa: F405

            # Best-effort Read-back Verification (akzeptiert memoryview)
            try:
                core = self.router.get_core(self._file_type)
                raw = core._read_block(pos, length)
                if isinstance(raw, memoryview):
                    raw = raw.tobytes()
                if not isinstance(raw, (bytes, bytearray)) or len(raw) != int(length):
                    raise PersistencyError(  # noqa: F405
                        f"Read-back verification failed: expected {length} bytes, got "
                        f"{len(raw) if isinstance(raw, (bytes, bytearray)) else 'non-bytes'}"
                    )
            except Exception as verify_err:
                raise PersistencyError(
                    f"Read-back verification failed for '{metadata_key}'"
                ) from verify_err  # noqa: F405

            # Defer teardown when bridge-managed close will commit after this call.
            defer_teardown = bool(
                getattr(self, "_defer_close_teardown", False) and (tx_id is not None)
            )
            if not defer_teardown:
                if getattr(self, "_cache", None) is not None and not getattr(
                    self, "_cache_closed", True
                ):
                    try:
                        self._cache.close()
                        chonk_errors.info(  # noqa: F405
                            f"[BaseIndex.close_index] Closed cache for {getattr(self, 'index_name', '?')}"
                        )
                    except Exception as cache_err:
                        chonk_errors.warning(
                            f"[BaseIndex.close_index] Error closing cache: {cache_err}"
                        )  # noqa: F405
                    finally:
                        self._cache_closed = True
                self._cache = None
            else:
                # AUDIT-FIX [P1/Reliability]: Keep cache/resources intact until commit succeeds; closed in _finalize_index_closure().
                pass
        finally:
            self._closing_in_progress = False

        if tx_id is None:
            self._finalize_index_closure()

    def _finalize_index_closure(self) -> None:
        # AUDIT-FIX [P1/Reliability]: Ensure cache is closed during finalize (needed for deferred teardown paths).
        try:
            if getattr(self, "_cache", None) is not None and not getattr(
                self, "_cache_closed", True
            ):
                try:
                    self._cache.close()
                    chonk_errors.info(  # noqa: F405
                        f"[BaseIndex.close_index] Closed cache for {getattr(self, 'index_name', '?')} (finalize)"
                    )
                finally:
                    self._cache_closed = True
            self._cache = None
        except Exception:
            pass

        # Mirror vector-index bridge teardown if present to avoid cross-engine leakage
        try:
            if hasattr(self, "_tx_coordinator") and self._tx_coordinator is not None:
                teardown = getattr(
                    self._tx_coordinator, "teardown_transaction_bridge", None
                )
                if callable(teardown):
                    teardown()
        except Exception:
            pass

        self._is_open = False
        chonk_errors.info(
            f"[BaseIndex.close_index] index={getattr(self, 'index_name', '?')} now closed"
        )  # noqa: F405
        try:
            BASE_INDEX_OPEN_GAUGE.dec()
        except Exception:
            pass

        try:
            self.run_hpc_aggregator()
        except Exception:
            pass

        read_only = _read_only_env_enabled()
        if not read_only:
            self._replicate_if_needed(
                "CLOSE_INDEX", {"index_name": getattr(self, "index_name", "?")}
            )
            self._write_wal_record(
                "CLOSE_INDEX",
                {"index": getattr(self, "index_name", "?"), "status": "closed"},
            )

    def _log_skip_metadata_persist_read_only(self, *, index_name: str) -> None:
        chonk_errors.info(  # noqa: F405
            f"[BaseIndex._persist_metadata_now] CHONK_READ_ONLY=1 -> skip metadata persist for index={index_name}"
        )

    def _persist_metadata_now(
        self, tx_id: Optional[str], note: str = "tx_commit", skip_fsync: bool = False
    ) -> None:
        if not getattr(self, "router", None):
            raise chonk_errors.create_error(  # noqa: F405
                message=f"[BaseIndex._persist_metadata_now] Router missing for index '{getattr(self, 'index_name', '?')}'",
                error_type=ErrorType.CONFIGURATION,  # noqa: F405
                severity=ErrorSeverity.CRITICAL,  # noqa: F405
                metadata={"index_name": getattr(self, "index_name", "?")},
            )
        if not getattr(self, "_is_open", False):
            return

        # Never persist in read-only mode.
        #
        # Root cause:
        # - Some long-running read-only services create/open indices under temporary
        #   CHONK_READ_ONLY env overrides and then reuse the same API/engine objects
        #   after the ambient env is restored.
        # - In that situation the index may still be backed by read-only file handles
        #   even though process env no longer advertises read-only.
        #
        # Contract:
        # - `_opened_read_only` is sticky for the lifetime of the opened index and must
        #   block metadata persistence even if env drifted later.
        if _read_only_env_enabled() or bool(getattr(self, "_opened_read_only", False)):
            idx = getattr(self, "index_name", "?")
            self._log_skip_metadata_persist_read_only(index_name=str(idx))
            return

        import time as _t_prof

        _prof_enabled = os.getenv("CHONKY_PROFILING", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        try:
            if os.getenv("CHONKY_COALESCE_FSYNC", "0").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                skip_fsync = True
        except Exception:
            pass

        _t_start = _t_prof.perf_counter()
        try:
            sw_guard = None
            use_sw = (
                os.getenv("CHONKY_RELATIONAL_SINGLE_WRITER", "1") or "1"
            ).strip().lower() not in ("0", "false", "no")
            if use_sw:
                try:
                    is_rel = getattr(self, "_file_type", None) == _FT.RELATIONAL
                except Exception:
                    is_rel = False

                if is_rel or str(getattr(self, "index_name", "")) == "code_rel":
                    if not hasattr(type(self), "_PERSIST_GUARDS"):
                        # AUDIT-FIX [P2/Reliability]: Use weak registry to prevent unbounded growth of persist guards.
                        type(self)._PERSIST_GUARDS = weakref.WeakValueDictionary()
                        type(self)._PERSIST_GUARDS_LOCK = threading.RLock()

                    with type(self)._PERSIST_GUARDS_LOCK:  # type: ignore[attr-defined]
                        try:
                            guards = getattr(type(self), "_PERSIST_GUARDS", None)  # type: ignore[attr-defined]
                        except Exception:
                            guards = None

                        if guards is None:
                            type(self)._PERSIST_GUARDS = weakref.WeakValueDictionary()  # type: ignore[attr-defined]
                            guards = type(self)._PERSIST_GUARDS  # type: ignore[attr-defined]

                        sw_guard = getattr(type(self), "_PERSIST_GUARDS", {}).get(
                            getattr(self, "index_name", "")
                        )  # type: ignore[attr-defined]
                        if sw_guard is None:
                            sw_guard = threading.RLock()
                            type(self)._PERSIST_GUARDS[
                                getattr(self, "index_name", "")
                            ] = sw_guard  # type: ignore[attr-defined]
        except Exception:
            sw_guard = None

        with sw_guard if sw_guard is not None else NoopCtx():
            # NOTE: router.store_index_data may allocate (FREESPACE/STORAGE); never call it while holding INDEX lock.
            with self.acquire_lock(
                "persist_metadata", getattr(self, "lock_timeout", 0.0), is_write=True
            ):
                self._validate_structure_data()
                try:
                    if getattr(self, "_file_type", None) == _FT.RELATIONAL:
                        if (not isinstance(self._index_structure_data, dict)) or (
                            "relational_header" not in self._index_structure_data
                        ):
                            raise PersistencyError(  # noqa: F405
                                f"Relational index '{getattr(self, 'index_name', '?')}' snapshot missing 'relational_header' ({note})."
                            )
                except Exception as _rel_check_err:
                    if not isinstance(_rel_check_err, PersistencyError):  # noqa: F405
                        raise PersistencyError(  # noqa: F405
                            f"Relational metadata validation failed for '{getattr(self, 'index_name', '?')}' ({note}): {_rel_check_err}"
                        ) from _rel_check_err

                doc_id = self._calc_baseindex_docid()
                index_type_name = self.__class__.__name__
                metadata_key = f"index_metadata_{index_type_name}_{getattr(self, 'index_name', '')}_{doc_id}"
                data_to_serialize = getattr(
                    self, "_index_structure_data", {}
                )  # avoid deepcopy: large metadata snapshots must not double-copy
                serializable_data = self._prepare_data_for_serialization(
                    data_to_serialize
                )

            # Force a true non-transactional metadata write when callers pass `tx_id=None`.
            # Passing `None` into the router would otherwise let legacy ambient/global
            # transaction fallback reattach this write to an unrelated active tx.
            router_tx_id = tx_id if tx_id is not None else ""
            pos, length = self.router.store_index_data(
                file_type=self._file_type,
                key=metadata_key,
                data=serializable_data,
                tx_id=router_tx_id,
            )

            # Optional fsync
            if not skip_fsync and hasattr(self.router, "sync_files"):
                try:
                    files = set()
                    try:
                        files.update(set(self.get_files_to_sync(tx_id or "")))  # type: ignore[arg-type]
                    except Exception:
                        pass
                    try:
                        core = self.router.get_core(self._file_type)
                        if core and hasattr(core, "file_path"):
                            files.add(str(core.file_path))
                    except Exception:
                        pass
                    if files:
                        self.router.sync_files(list(files))
                except Exception as sync_err:
                    raise PersistencyError(  # noqa: F405
                        f"Failed to fsync metadata for index '{getattr(self, 'index_name', '?')}': {sync_err}"
                    ) from sync_err

        if _prof_enabled:
            try:
                total_ms = round((time.perf_counter() - _t_start) * 1000.0, 3)
            except Exception:
                total_ms = None
            chonk_errors.success(  # noqa: F405
                "PERSIST_PROF",
                metadata={
                    "index": getattr(self, "index_name", "?"),
                    "note": note,
                    "tx_id": tx_id or "",
                    "total_ms": total_ms,
                },
            )

    def _restore_all_metadata_from_baseindex(self) -> None:
        # Default-Implementation (subclasses override)
        return None

    def _recover_missing_metadata_before_router_read_on_open(
        self, *, metadata_key: str, doc_id: int, strict_meta: bool
    ) -> Optional[Dict[str, Any]]:
        # Default-Implementation (subclasses override)
        _ = (metadata_key, doc_id, strict_meta)
        return None

    def _prepare_data_for_serialization(
        self, data: Any, stack=None, depth: int = 0, max_depth: int = 100
    ) -> Any:
        """
        Zyklensichere, deterministische Normalisierung:
        - Primitive: Rückgabe direkt
        - dict/list/tuple/set: rekursiv (Sets deterministisch nach str sortiert)
        - Dataclasses: asdict() + Rekursion
        - Objekte: öffentliche Felder; Numpy-Arrays via .tolist()
        - Fallback: str()
        """
        if depth > max_depth:
            return "<max depth reached>"
        if stack is None:
            stack = set()

        if isinstance(data, (int, float, bool, type(None), str)):
            return data

        obj_id = id(data)
        if obj_id in stack:
            return "<circular reference>"
        stack.add(obj_id)
        try:
            # AUDIT-FIX [P0/Correctness]: Handle array-like objects with `.tolist()` even if they don't expose `__dict__` (e.g., numpy.ndarray).
            # CAUTION: For array-like objects, output may change from legacy `str(obj)` to a normalized Python structure for better fidelity.
            try:
                tolist = getattr(data, "tolist", None)
                if callable(tolist) and not isinstance(
                    data, (dict, list, tuple, set, bytes, bytearray, memoryview)
                ):
                    mod = getattr(type(data), "__module__", "") or ""
                    is_arrayish = mod.startswith(
                        ("numpy", "jax", "jaxlib", "array")
                    ) or (hasattr(data, "shape") and hasattr(data, "dtype"))
                    if is_arrayish:
                        # Avoid catastrophic expansions on extremely large arrays; preserve legacy compact `str()` in that edge case.
                        try:
                            size_attr = getattr(data, "size", None)
                            size = int(size_attr) if size_attr is not None else None
                        except Exception:
                            size = None
                        if size is not None and size > 1_000_000:
                            return str(
                                data
                            )  # preserves legacy behavior to avoid massive allocations
                        out = tolist()
                        return self._prepare_data_for_serialization(
                            out, stack, depth + 1, max_depth
                        )
            except Exception:
                pass

            if isinstance(data, dict):
                # Snapshot mutable metadata dicts before the recursive walk. Live
                # index writers can mutate these structures while the fulltext
                # commit thread is persisting metadata, and iterating the live
                # dict view can raise `RuntimeError: dictionary changed size
                # during iteration`.
                snapshot = data.copy()
                result: Dict[str, Any] = {}
                for k, v in snapshot.items():
                    if not isinstance(k, str):
                        k = str(k)
                    result[k] = self._prepare_data_for_serialization(
                        v, stack, depth + 1, max_depth
                    )
                return result
            if isinstance(data, list):
                return [
                    self._prepare_data_for_serialization(
                        elem, stack, depth + 1, max_depth
                    )
                    for elem in data
                ]
            if isinstance(data, tuple):
                return tuple(
                    self._prepare_data_for_serialization(
                        elem, stack, depth + 1, max_depth
                    )
                    for elem in data
                )
            if isinstance(data, set):
                try:
                    sorted_elems = sorted(list(data), key=lambda e: str(e))
                except Exception:
                    sorted_elems = list(data)
                return [
                    self._prepare_data_for_serialization(
                        elem, stack, depth + 1, max_depth
                    )
                    for elem in sorted_elems
                ]
            if is_dataclass(data):
                return self._prepare_data_for_serialization(
                    asdict(data), stack, depth + 1, max_depth
                )
            if hasattr(data, "__dict__"):
                if hasattr(data, "tolist"):
                    out = data.tolist()
                    return self._prepare_data_for_serialization(
                        out, stack, depth + 1, max_depth
                    )
                out_obj: Dict[str, Any] = {}
                for k, v in data.__dict__.items():
                    if str(k).startswith("_"):
                        continue
                    out_obj[k] = self._prepare_data_for_serialization(
                        v, stack, depth + 1, max_depth
                    )
                return out_obj

            try:
                return str(data)
            except Exception:
                return "<non-serializable>"
        finally:
            # garantiertes Pop aus dem Zyklen-Wächter
            try:
                stack.remove(obj_id)
            except Exception:
                pass
