from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import stat
from threading import Lock
import tempfile
from contextlib import suppress

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.client import chonkydb_data_path
from twinr.memory.fulltext import FullTextDocument, FullTextSelector
from twinr.memory.longterm.models import LongTermMidtermPacketV1, LongTermReflectionResultV1
from twinr.memory.longterm.remote_state import LongTermRemoteUnavailableError
from twinr.memory.longterm.remote_state import LongTermRemoteStateStore
from twinr.text_utils import retrieval_terms


_MIDTERM_STORE_SCHEMA = "twinr_memory_midterm_store"
_MIDTERM_STORE_VERSION = 1
_DEFAULT_RETRIEVAL_LIMIT = 3

# AUDIT-FIX(#8): Add module-level logging so storage recovery paths are diagnosable in production.
LOGGER = logging.getLogger(__name__)


def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_limit(limit: int) -> int:
    # AUDIT-FIX(#7): Honor explicit zero/negative limits as "return nothing" and log invalid caller input.
    if not isinstance(limit, int):
        LOGGER.warning(
            "Invalid midterm packet limit %r; falling back to default=%d",
            limit,
            _DEFAULT_RETRIEVAL_LIMIT,
        )
        return _DEFAULT_RETRIEVAL_LIMIT
    return max(0, limit)


def _safe_retrieval_terms(value: str) -> set[str]:
    # AUDIT-FIX(#5): Retrieval tokenization must fail closed to an empty token-set instead of aborting memory lookup.
    try:
        return {term for term in retrieval_terms(value) if isinstance(term, str) and not term.isdigit()}
    except Exception:
        LOGGER.warning("Failed to tokenize midterm retrieval text", exc_info=True)
        return set()


def _read_json_object(path: Path) -> dict[str, object] | None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError:
        # AUDIT-FIX(#4): Treat file-open failures as recoverable load errors instead of crashing the caller.
        LOGGER.warning("Failed to open midterm store file %s", path, exc_info=True)
        return None

    try:
        # AUDIT-FIX(#4): Refuse symlinked/non-regular files to reduce local file-system redirection risk.
        stat_result = os.fstat(fd)
        if not stat.S_ISREG(stat_result.st_mode):
            raise OSError(f"Refusing to read non-regular midterm store file: {path}")

        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    except OSError:
        LOGGER.warning("Failed to read midterm store file %s", path, exc_info=True)
        return None
    finally:
        os.close(fd)

    try:
        loaded = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        LOGGER.warning("Invalid JSON in midterm store file %s", path, exc_info=True)
        return None

    if not isinstance(loaded, dict):
        LOGGER.warning("Ignoring non-object midterm store payload from %s", path)
        return None

    return dict(loaded)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    temp_name: str | None = None

    try:
        # AUDIT-FIX(#1): Use a unique temp file in the target directory to avoid .tmp collisions across writers.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(serialized)
            handle.flush()
            # AUDIT-FIX(#1): Flush file contents before replace so sudden power loss is less likely to drop state.
            os.fsync(handle.fileno())

        os.replace(temp_name, path)
        temp_name = None
        # AUDIT-FIX(#1): Flush the directory entry so the rename itself is durable on Linux-class filesystems.
        _fsync_directory(path.parent)
    finally:
        if temp_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temp_name)


def _validate_payload(payload: object, *, source: str) -> dict[str, object] | None:
    # AUDIT-FIX(#3): Validate container type and schema metadata before handing data to model loaders.
    if not isinstance(payload, dict):
        LOGGER.warning("Ignoring invalid midterm payload from %s because it is not a JSON object", source)
        return None

    schema = payload.get("schema")
    if schema not in (None, _MIDTERM_STORE_SCHEMA):
        LOGGER.warning(
            "Ignoring midterm payload from %s due to schema mismatch: %r != %r",
            source,
            schema,
            _MIDTERM_STORE_SCHEMA,
        )
        return None

    version = payload.get("version")
    if version not in (None, _MIDTERM_STORE_VERSION):
        LOGGER.warning(
            "Ignoring midterm payload from %s due to version mismatch: %r != %r",
            source,
            version,
            _MIDTERM_STORE_VERSION,
        )
        return None

    items = payload.get("packets", [])
    if not isinstance(items, list):
        LOGGER.warning("Ignoring midterm payload from %s because packets is not a list", source)
        return None

    return dict(payload)


@dataclass(slots=True)
class LongTermMidtermStore:
    base_path: Path
    remote_state: LongTermRemoteStateStore | None = None
    _lock: Lock = field(default_factory=Lock, repr=False)

    packet_type = LongTermMidtermPacketV1

    def __post_init__(self) -> None:
        # AUDIT-FIX(#6): Normalize and validate the configured base path early so misconfiguration fails deterministically.
        normalized_base_path = Path(self.base_path).expanduser().resolve(strict=False)
        if normalized_base_path.exists() and not normalized_base_path.is_dir():
            raise ValueError(f"LongTermMidtermStore base_path must be a directory: {normalized_base_path}")
        self.base_path = normalized_base_path

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LongTermMidtermStore":
        return cls(
            base_path=chonkydb_data_path(config),
            remote_state=LongTermRemoteStateStore.from_config(config),
        )

    @property
    def packets_path(self) -> Path:
        return self.base_path / "twinr_memory_midterm_v1.json"

    def ensure_remote_snapshot(self) -> bool:
        if self.remote_state is None or not self.remote_state.enabled:
            return False
        payload = self.remote_state.load_snapshot(snapshot_kind="midterm", local_path=self.packets_path)
        if _validate_payload(payload, source="remote:midterm") is not None:
            return False
        local_payload = _validate_payload(_read_json_object(self.packets_path), source=str(self.packets_path))
        self.remote_state.save_snapshot(
            snapshot_kind="midterm",
            payload=local_payload or self._empty_payload(),
        )
        return True

    def load_packets(self) -> tuple[LongTermMidtermPacketV1, ...]:
        payload = self._load_payload()
        if payload is None:
            return ()

        items = payload.get("packets", [])
        if not isinstance(items, list):
            LOGGER.warning("Midterm store payload has non-list packets field at %s", self.packets_path)
            return ()

        packets: list[LongTermMidtermPacketV1] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                LOGGER.warning("Skipping non-object midterm packet at index %d", index)
                continue
            try:
                # AUDIT-FIX(#3): Skip only the malformed packet instead of aborting the whole store load.
                packets.append(LongTermMidtermPacketV1.from_payload(item))
            except Exception:
                LOGGER.warning("Skipping invalid midterm packet at index %d", index, exc_info=True)

        return tuple(packets)

    def save_packets(self, *, packets: tuple[LongTermMidtermPacketV1, ...]) -> None:
        with self._lock:
            payload = {
                "schema": _MIDTERM_STORE_SCHEMA,
                "version": _MIDTERM_STORE_VERSION,
                "packets": [item.to_payload() for item in sorted(packets, key=lambda row: row.packet_id)],
            }
            # AUDIT-FIX(#2): Always persist locally first because the device is file-backed and the network is intermittent.
            _write_json_atomic(self.packets_path, payload)

            if self.remote_state is not None and self.remote_state.enabled:
                try:
                    # AUDIT-FIX(#2): Remote persistence is best-effort replication; local durability must survive remote outages.
                    self.remote_state.save_snapshot(snapshot_kind="midterm", payload=payload)
                except LongTermRemoteUnavailableError:
                    if self.remote_state.required:
                        raise
                    LOGGER.warning(
                        "Failed to replicate midterm snapshot remotely; local snapshot at %s was kept",
                        self.packets_path,
                        exc_info=True,
                    )
                except Exception:
                    LOGGER.warning(
                        "Failed to replicate midterm snapshot remotely; local snapshot at %s was kept",
                        self.packets_path,
                        exc_info=True,
                    )

    def _empty_payload(self) -> dict[str, object]:
        return {
            "schema": _MIDTERM_STORE_SCHEMA,
            "version": _MIDTERM_STORE_VERSION,
            "packets": [],
        }

    def apply_reflection(self, result: LongTermReflectionResultV1) -> None:
        self.save_packets(packets=result.midterm_packets)

    def _load_payload(self) -> dict[str, object] | None:
        local_payload = _validate_payload(_read_json_object(self.packets_path), source=str(self.packets_path))
        if local_payload is not None:
            return local_payload

        if self.remote_state is None or not self.remote_state.enabled:
            return None

        try:
            # AUDIT-FIX(#4): Remote snapshot failure must degrade to "no remote data" instead of crashing memory retrieval.
            remote_payload = self.remote_state.load_snapshot(
                snapshot_kind="midterm",
                local_path=self.packets_path,
            )
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            LOGGER.warning("Failed to load remote midterm snapshot", exc_info=True)
            return None

        validated_remote_payload = _validate_payload(remote_payload, source="remote:midterm")
        if validated_remote_payload is None:
            return None

        try:
            # AUDIT-FIX(#2): Refresh the local cache from a valid remote snapshot so later offline loads still work.
            _write_json_atomic(self.packets_path, validated_remote_payload)
        except OSError:
            LOGGER.warning(
                "Loaded remote midterm snapshot but failed to refresh local cache at %s",
                self.packets_path,
                exc_info=True,
            )

        return validated_remote_payload

    def select_relevant_packets(
        self,
        query_text: str | None,
        *,
        limit: int = 3,
    ) -> tuple[LongTermMidtermPacketV1, ...]:
        normalized_limit = _normalize_limit(limit)
        if normalized_limit == 0:
            return ()

        packets = self.load_packets()
        if not packets:
            return ()

        clean_query = _normalize_text(query_text)
        if not clean_query:
            return packets[:normalized_limit]

        try:
            selector = FullTextSelector(
                tuple(
                    FullTextDocument(
                        doc_id=item.packet_id,
                        category="midterm",
                        content=self._packet_search_text(item),
                    )
                    for item in packets
                )
            )
            selected_ids = selector.search(
                clean_query,
                limit=normalized_limit,
                category="midterm",
            )
        except Exception:
            # AUDIT-FIX(#5): Retrieval ranking failure must fall back to deterministic packet ordering, not crash the caller.
            LOGGER.warning("Midterm full-text selection failed; falling back to in-order packets", exc_info=True)
            selected_ids = [item.packet_id for item in packets]

        by_id = {item.packet_id: item for item in packets}
        selected = [by_id[packet_id] for packet_id in selected_ids if packet_id in by_id]
        if not selected:
            selected = list(packets)

        return tuple(
            self._filter_query_relevant_packets(
                clean_query,
                selected=selected,
                limit=normalized_limit,
            )
        )

    def _packet_search_text(self, item: LongTermMidtermPacketV1) -> str:
        parts: list[str] = [
            item.kind,
            item.summary,
            item.details or "",
        ]

        if isinstance(item.query_hints, (list, tuple)):
            parts.extend(hint for hint in item.query_hints if isinstance(hint, str))

        attributes = item.attributes or {}
        if isinstance(attributes, dict):
            for key, value in attributes.items():
                parts.append(str(key))
                if isinstance(value, str):
                    parts.append(value)
                elif isinstance(value, (list, tuple)):
                    parts.extend(str(entry) for entry in value if isinstance(entry, str))

        # AUDIT-FIX(#3): Coerce unexpected scalar fields to strings so partially corrupt packets stay searchable.
        return _normalize_text(" ".join(str(part) for part in parts if part))

    def _filter_query_relevant_packets(
        self,
        query_text: str,
        *,
        selected: list[LongTermMidtermPacketV1],
        limit: int,
    ) -> tuple[LongTermMidtermPacketV1, ...]:
        normalized_limit = _normalize_limit(limit)
        if normalized_limit == 0:
            return ()

        query_terms = _safe_retrieval_terms(query_text)
        if not query_terms:
            return tuple(selected[:normalized_limit])

        filtered = [
            item
            for item in selected
            if query_terms.intersection(_safe_retrieval_terms(self._packet_search_text(item)))
        ]
        if not filtered:
            # AUDIT-FIX(#5): If token filtering is stricter than the ranker, keep the ranked candidates instead of returning nothing.
            return tuple(selected[:normalized_limit])

        return tuple(filtered[:normalized_limit])


__all__ = ["LongTermMidtermStore"]
