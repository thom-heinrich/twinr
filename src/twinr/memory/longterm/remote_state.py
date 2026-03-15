from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb import ChonkyDBClient, ChonkyDBConnectionConfig, ChonkyDBError
from twinr.memory.chonkydb.models import ChonkyDBRecordRequest


_REMOTE_NAMESPACE_PREFIX = "twinr_longterm_v1"
_SNAPSHOT_SCHEMA = "twinr_remote_snapshot_v1"
def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _mapping_dict(value: Mapping[str, object] | None) -> dict[str, object] | None:
    if value is None:
        return None
    return dict(value)


def _safe_json_text(payload: Mapping[str, object]) -> str:
    return json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class LongTermRemoteUnavailableError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class LongTermRemoteStatus:
    mode: str
    ready: bool
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class _RemoteSnapshotFetchResult:
    status: str
    payload: dict[str, object] | None = None
    detail: str | None = None


@dataclass(slots=True)
class LongTermRemoteStateStore:
    config: TwinrConfig
    read_client: ChonkyDBClient | None = None
    write_client: ChonkyDBClient | None = None
    namespace: str | None = None

    def __post_init__(self) -> None:
        if not self.namespace:
            self.namespace = _remote_namespace_for_config(self.config)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LongTermRemoteStateStore":
        namespace = _remote_namespace_for_config(config)
        if not (config.chonkydb_base_url and config.chonkydb_api_key):
            return cls(config=config, namespace=namespace)
        read_client = ChonkyDBClient(
            ChonkyDBConnectionConfig(
                base_url=config.chonkydb_base_url,
                api_key=config.chonkydb_api_key,
                api_key_header=config.chonkydb_api_key_header,
                allow_bearer_auth=config.chonkydb_allow_bearer_auth,
                timeout_s=config.long_term_memory_remote_read_timeout_s,
            )
        )
        write_client = ChonkyDBClient(
            ChonkyDBConnectionConfig(
                base_url=config.chonkydb_base_url,
                api_key=config.chonkydb_api_key,
                api_key_header=config.chonkydb_api_key_header,
                allow_bearer_auth=config.chonkydb_allow_bearer_auth,
                timeout_s=config.long_term_memory_remote_write_timeout_s,
            )
        )
        return cls(
            config=config,
            read_client=read_client,
            write_client=write_client,
            namespace=namespace,
        )

    @property
    def enabled(self) -> bool:
        return self.config.long_term_memory_enabled and self.config.long_term_memory_mode == "remote_primary"

    @property
    def required(self) -> bool:
        return self.enabled and self.config.long_term_memory_remote_required

    def status(self) -> LongTermRemoteStatus:
        if not self.enabled:
            return LongTermRemoteStatus(mode="disabled", ready=False)
        if self.read_client is None or self.write_client is None:
            return LongTermRemoteStatus(mode="remote_primary", ready=False, detail="ChonkyDB is not configured.")
        try:
            instance = self.read_client.instance()
        except (ChonkyDBError, ValueError) as exc:
            return LongTermRemoteStatus(mode="remote_primary", ready=False, detail=str(exc))
        if not instance.ready:
            return LongTermRemoteStatus(
                mode="remote_primary",
                ready=False,
                detail="ChonkyDB instance responded but is not ready.",
            )
        return LongTermRemoteStatus(mode="remote_primary", ready=True)

    def load_snapshot(self, *, snapshot_kind: str, local_path: Path | None = None) -> dict[str, object] | None:
        if not self.enabled:
            return None
        read_client = self._require_client(self.read_client, operation="read")
        result = self._load_snapshot_via_uri(read_client, snapshot_kind=snapshot_kind)
        if result.payload is not None:
            return result.payload
        if result.status == "unavailable":
            if self.required:
                raise LongTermRemoteUnavailableError(
                    result.detail
                    or f"Failed to read remote long-term snapshot {snapshot_kind!r}."
                )
            return None
        if (
            result.status == "not_found"
            and self.config.long_term_memory_migration_enabled
            and local_path is not None
            and local_path.exists()
        ):
            payload = json.loads(local_path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                try:
                    self.save_snapshot(snapshot_kind=snapshot_kind, payload=payload)
                except LongTermRemoteUnavailableError:
                    if self.required:
                        raise
                    return None
                return dict(payload)
        return None

    def save_snapshot(self, *, snapshot_kind: str, payload: Mapping[str, object]) -> None:
        if not self.enabled:
            return
        write_client = self._require_client(self.write_client, operation="write")
        updated_at = _utcnow_iso()
        record = ChonkyDBRecordRequest(
            payload={
                "schema": _SNAPSHOT_SCHEMA,
                "namespace": self.namespace or _REMOTE_NAMESPACE_PREFIX,
                "snapshot_kind": snapshot_kind,
                "updated_at": updated_at,
                "body": dict(payload),
            },
            metadata={
                "twinr_namespace": self.namespace or _REMOTE_NAMESPACE_PREFIX,
                "twinr_snapshot_kind": snapshot_kind,
                "twinr_snapshot_updated_at": updated_at,
                "twinr_snapshot_schema": _SNAPSHOT_SCHEMA,
            },
            uri=self._snapshot_uri(snapshot_kind),
            content=_safe_json_text(
                {
                    "schema": _SNAPSHOT_SCHEMA,
                    "namespace": self.namespace or _REMOTE_NAMESPACE_PREFIX,
                    "snapshot_kind": snapshot_kind,
                    "updated_at": updated_at,
                    "body": dict(payload),
                }
            ),
            enable_chunking=False,
            include_insights_in_response=False,
        )
        try:
            write_client.store_record(record)
        except ChonkyDBError as exc:
            raise LongTermRemoteUnavailableError(
                f"Failed to write remote long-term snapshot {snapshot_kind!r}: {exc}"
            ) from exc

    def _require_client(self, client: ChonkyDBClient | None, *, operation: str) -> ChonkyDBClient:
        if client is not None:
            return client
        raise LongTermRemoteUnavailableError(
            f"Remote-primary long-term memory is enabled but ChonkyDB is not configured for {operation} operations."
        )

    def _extract_snapshot_body(
        self,
        payload: Mapping[str, object],
        *,
        snapshot_kind: str,
    ) -> dict[str, object] | None:
        for candidate in self._iter_snapshot_candidates(payload):
            if candidate.get("namespace") != (self.namespace or _REMOTE_NAMESPACE_PREFIX):
                continue
            if candidate.get("snapshot_kind") != snapshot_kind:
                continue
            body = candidate.get("body")
            if isinstance(body, Mapping):
                return dict(body)
        return None

    def _iter_snapshot_candidates(self, payload: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
        direct = payload.get("payload")
        if isinstance(direct, Mapping):
            yield direct

        nested = payload.get("record")
        if isinstance(nested, Mapping):
            nested_payload = nested.get("payload")
            if isinstance(nested_payload, Mapping):
                yield nested_payload

        content = payload.get("content")
        if isinstance(content, str):
            parsed = self._parse_snapshot_content(content)
            if parsed is not None:
                yield parsed

        chunks = payload.get("chunks")
        if isinstance(chunks, list):
            for chunk in chunks:
                if not isinstance(chunk, Mapping):
                    continue
                chunk_payload = chunk.get("payload")
                if isinstance(chunk_payload, Mapping):
                    yield chunk_payload
                chunk_content = chunk.get("content")
                if isinstance(chunk_content, str):
                    parsed = self._parse_snapshot_content(chunk_content)
                    if parsed is not None:
                        yield parsed

    def _parse_snapshot_content(self, content: str) -> Mapping[str, object] | None:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, Mapping):
            return parsed
        return None

    def _load_snapshot_via_uri(
        self,
        client: ChonkyDBClient,
        *,
        snapshot_kind: str,
    ) -> _RemoteSnapshotFetchResult:
        try:
            payload = client.fetch_full_document(
                origin_uri=self._snapshot_uri(snapshot_kind),
                include_content=True,
                max_content_chars=2_000_000,
            )
        except ChonkyDBError as exc:
            if exc.status_code == 404:
                return _RemoteSnapshotFetchResult(status="not_found")
            return _RemoteSnapshotFetchResult(
                status="unavailable",
                detail=f"Failed to read remote long-term snapshot {snapshot_kind!r}: {exc}",
            )
        direct = self._extract_snapshot_body(payload, snapshot_kind=snapshot_kind)
        if direct is not None:
            return _RemoteSnapshotFetchResult(status="found", payload=direct)
        return _RemoteSnapshotFetchResult(
            status="unavailable",
            detail=(
                f"Remote long-term snapshot {snapshot_kind!r} returned malformed content "
                "that Twinr could not parse."
            ),
        )

    def _snapshot_uri(self, snapshot_kind: str) -> str:
        return f"twinr://longterm/{self.namespace or _REMOTE_NAMESPACE_PREFIX}/{snapshot_kind}"


def _remote_namespace_for_config(config: TwinrConfig) -> str:
    override = _normalize_text(config.long_term_memory_remote_namespace)
    if override:
        return override
    root = Path(config.project_root).resolve()
    memory_path = Path(config.long_term_memory_path)
    resolved_memory_path = memory_path if memory_path.is_absolute() else (root / memory_path)
    digest = hashlib.sha1(str(resolved_memory_path.resolve()).encode("utf-8")).hexdigest()[:12]
    stem = root.name or "twinr"
    return f"{_REMOTE_NAMESPACE_PREFIX}:{stem}:{digest}"


__all__ = [
    "LongTermRemoteStateStore",
    "LongTermRemoteStatus",
    "LongTermRemoteUnavailableError",
]
