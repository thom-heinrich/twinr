
"""Persist core Adaptive Skill Engine records under Twinr's state directory.

The store is intentionally narrow: it owns dialogue, compile-job, activation,
status, and artifact persistence. Higher-level orchestration stays outside this
module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from numbers import Integral
from pathlib import Path
import json
import logging
import os
import re
import stat
import tempfile
import threading
from typing import TYPE_CHECKING, Any, Iterable
from uuid import uuid4

from twinr.text_utils import is_valid_stable_identifier, truncate_text

from .contracts import (
    ActivationRecord,
    CompileArtifactRecord,
    CompileJobRecord,
    CompileRunStatusRecord,
    ExecutionRunStatusRecord,
    LiveE2EStatusRecord,
    RequirementsDialogueSession,
    SkillHealthRecord,
)
from .status import ArtifactKind

if TYPE_CHECKING:
    from twinr.agent.base_agent.config import TwinrConfig


logger = logging.getLogger(__name__)

_SAFE_SUFFIX_PATTERN = re.compile(r"^\.[a-z0-9][a-z0-9._-]{0,15}$")
_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_INTEGER_TEXT_PATTERN = re.compile(r"^[+-]?\d+$")


def self_coding_store_root(project_root: str | Path) -> Path:
    """Return the canonical on-disk root for self-coding runtime state."""

    return Path(project_root).expanduser().resolve(strict=False) / "state" / "self_coding"


def _generate_identifier(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _open_directory_fd(path: Path, *, create: bool) -> int:
    """Open a directory without following a symlink on POSIX."""

    if os.name != "posix":
        raise OSError("_open_directory_fd is only supported on POSIX")
    if create:
        # AUDIT-FIX(#1): Create store directories with private permissions before anchoring file ops to a directory fd.
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = getattr(os, "O_PATH", os.O_RDONLY)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise ValueError(f"Expected directory at {path}")
        return fd
    except Exception:
        os.close(fd)
        raise


def _fsync_directory_fd(fd: int) -> None:
    """Force directory metadata to disk on POSIX for crash-durable updates."""

    if os.name != "posix":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    sync_fd = os.open(".", flags, dir_fd=fd)
    try:
        os.fsync(sync_fd)
    finally:
        os.close(sync_fd)


def _fsync_directory(path: Path) -> None:
    """Force directory metadata to disk on POSIX for crash-durable renames."""

    if os.name != "posix":
        return
    fd = _open_directory_fd(path, create=False)
    try:
        _fsync_directory_fd(fd)
    finally:
        os.close(fd)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if os.name != "posix":
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=str(path.parent),
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            _fsync_directory(path.parent)
        except Exception:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to remove temporary file %s", temp_path, exc_info=True)
            raise
        return

    parent_fd = _open_directory_fd(path.parent, create=True)
    temp_name = f".{path.name}.{uuid4().hex}.tmp"
    temp_fd = -1
    try:
        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            create_flags |= os.O_NOFOLLOW
        # AUDIT-FIX(#1): Use dir_fd-based create/replace so writes stay anchored to the validated store directory.
        temp_fd = os.open(temp_name, create_flags, 0o600, dir_fd=parent_fd)
        try:
            with os.fdopen(temp_fd, "wb", closefd=True) as handle:
                temp_fd = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            _fsync_directory_fd(parent_fd)
        except Exception:
            if temp_fd != -1:
                os.close(temp_fd)
                temp_fd = -1
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning("Failed to remove temporary file %s/%s", path.parent, temp_name, exc_info=True)
            raise
    finally:
        os.close(parent_fd)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    _write_bytes_atomic(path, encoded)


def _read_bytes_file_safe(path: Path) -> bytes:
    if os.name != "posix":
        if path.is_symlink():
            raise ValueError(f"Refusing to read symlink at {path}")
        if not path.is_file():
            raise ValueError(f"Expected regular file at {path}")
        with path.open("rb") as handle:
            return handle.read()

    parent_fd = _open_directory_fd(path.parent, create=False)
    fd = -1
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        # AUDIT-FIX(#1): Read through a validated parent dir_fd so path re-binding cannot escape the store.
        fd = os.open(path.name, flags, dir_fd=parent_fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"Expected regular file at {path}")
        with os.fdopen(fd, "rb", closefd=True) as handle:
            fd = -1
            return handle.read()
    finally:
        if fd != -1:
            os.close(fd)
        os.close(parent_fd)


def _read_text_file_safe(path: Path, *, encoding: str = "utf-8") -> str:
    return _read_bytes_file_safe(path).decode(encoding)


def _unlink_file_safe(path: Path, *, missing_ok: bool = False) -> bool:
    if os.name != "posix":
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
                return True
            if path.exists():
                raise ValueError(f"Expected file at {path}")
            return False
        except FileNotFoundError:
            if missing_ok:
                return False
            raise

    parent_fd = _open_directory_fd(path.parent, create=False)
    try:
        try:
            file_stat = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            if missing_ok:
                return False
            raise
        if stat.S_ISDIR(file_stat.st_mode):
            raise ValueError(f"Expected file at {path}")
        # AUDIT-FIX(#1): Unlink via dir_fd and fsync the directory so deletes survive power loss and symlink swaps.
        os.unlink(path.name, dir_fd=parent_fd)
        _fsync_directory_fd(parent_fd)
        return True
    finally:
        os.close(parent_fd)


def _read_json_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(_read_text_file_safe(path, encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object payload in {path}")
    return payload


class SelfCodingStore:
    """Store self-coding job and artifact records under one project root."""

    _locks_guard = threading.Lock()
    _locks: dict[str, threading.RLock] = {}

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        if self.root.exists() and not self.root.is_dir():
            raise ValueError("root must point to a directory")
        self.dialogues_dir = self.root / "dialogues"
        self.jobs_dir = self.root / "jobs"
        self.activations_dir = self.root / "activations"
        self.artifacts_dir = self.root / "artifacts"
        self.status_dir = self.root / "status"
        self.execution_runs_dir = self.root / "execution_runs"
        self.health_dir = self.root / "health"
        self.live_e2e_dir = self.root / "live_e2e"
        self.contents_dir = self.root / "contents"
        self._lock = self._lock_for_root(self.root)

    @classmethod
    def _lock_for_root(cls, root: Path) -> threading.RLock:
        key = os.fspath(root)
        with cls._locks_guard:
            lock = cls._locks.get(key)
            if lock is None:
                lock = threading.RLock()
                cls._locks[key] = lock
            return lock

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "SelfCodingStore":
        return cls(self_coding_store_root(project_root))

    @classmethod
    def from_config(cls, config: "TwinrConfig") -> "SelfCodingStore":
        return cls.from_project_root(getattr(config, "project_root", "."))

    def save_job(self, record: CompileJobRecord) -> CompileJobRecord:
        """Persist one compile-job record."""

        normalized = self._normalize_job_record(record)
        path = self._record_json_path(self.jobs_dir, normalized.job_id)
        with self._lock:
            _write_json_atomic(path, normalized.to_payload())
        return normalized

    def load_job(self, job_id: str) -> CompileJobRecord:
        """Load one compile-job record by identifier."""

        normalized_id = self._require_identifier(job_id, field_name="job_id")
        return self._normalize_job_record(
            CompileJobRecord.from_payload(_read_json_payload(self._record_json_path(self.jobs_dir, normalized_id)))
        )

    def list_jobs(self) -> tuple[CompileJobRecord, ...]:
        """Return all persisted compile-job records sorted by update time."""

        records: list[CompileJobRecord] = []
        for path in self._iter_record_paths(self.jobs_dir):
            try:
                records.append(self._normalize_job_record(CompileJobRecord.from_payload(_read_json_payload(path))))
            except Exception:
                logger.warning("Skipping invalid compile-job record at %s", path, exc_info=True)
        return tuple(
            sorted(
                records,
                key=lambda item: (
                    self._sortable_timestamp(item.updated_at),
                    self._sortable_timestamp(item.created_at),
                    item.job_id,
                ),
                reverse=True,
            )
        )

    def find_job_for_session(self, session_id: str) -> CompileJobRecord | None:
        """Return the newest compile job linked to one dialogue session, if any."""

        normalized_session_id = self._require_identifier(session_id, field_name="session_id")
        for record in self.list_jobs():
            if record.metadata.get("session_id") == normalized_session_id:
                return record
        return None

    def save_dialogue_session(self, record: RequirementsDialogueSession) -> RequirementsDialogueSession:
        """Persist one requirements-dialogue session record."""

        normalized = self._normalize_dialogue_session_record(record)
        path = self._record_json_path(self.dialogues_dir, normalized.session_id)
        with self._lock:
            _write_json_atomic(path, normalized.to_payload())
        return normalized

    def load_dialogue_session(self, session_id: str) -> RequirementsDialogueSession:
        """Load one requirements-dialogue session by identifier."""

        normalized_id = self._require_identifier(session_id, field_name="session_id")
        return self._normalize_dialogue_session_record(
            RequirementsDialogueSession.from_payload(
                _read_json_payload(self._record_json_path(self.dialogues_dir, normalized_id))
            )
        )

    def list_dialogue_sessions(self) -> tuple[RequirementsDialogueSession, ...]:
        """Return all persisted dialogue sessions sorted by update time."""

        records: list[RequirementsDialogueSession] = []
        for path in self._iter_record_paths(self.dialogues_dir):
            try:
                records.append(
                    self._normalize_dialogue_session_record(
                        RequirementsDialogueSession.from_payload(_read_json_payload(path))
                    )
                )
            except Exception:
                logger.warning("Skipping invalid dialogue-session record at %s", path, exc_info=True)
        return tuple(
            sorted(
                records,
                key=lambda item: (
                    self._sortable_timestamp(item.updated_at),
                    self._sortable_timestamp(item.created_at),
                    item.session_id,
                ),
                reverse=True,
            )
        )

    def save_artifact(self, record: CompileArtifactRecord) -> CompileArtifactRecord:
        """Persist one compile-artifact record."""

        normalized = self._normalize_artifact_record(record)
        path = self._record_json_path(self.artifacts_dir, normalized.artifact_id)
        with self._lock:
            _write_json_atomic(path, normalized.to_payload())
        return normalized

    def save_compile_status(self, record: CompileRunStatusRecord) -> CompileRunStatusRecord:
        """Persist the current runtime status for one compile job."""

        normalized = self._normalize_compile_status_record(record)
        path = self._record_json_path(self.status_dir, normalized.job_id)
        with self._lock:
            _write_json_atomic(path, normalized.to_payload())
        return normalized

    def load_compile_status(self, job_id: str) -> CompileRunStatusRecord:
        """Load the runtime status for one compile job."""

        normalized_id = self._require_identifier(job_id, field_name="job_id")
        return self._normalize_compile_status_record(
            CompileRunStatusRecord.from_payload(_read_json_payload(self._record_json_path(self.status_dir, normalized_id)))
        )

    def list_compile_statuses(self) -> tuple[CompileRunStatusRecord, ...]:
        """Return all persisted compile runtime statuses sorted by update time."""

        records: list[CompileRunStatusRecord] = []
        for path in self._iter_record_paths(self.status_dir):
            try:
                records.append(
                    self._normalize_compile_status_record(
                        CompileRunStatusRecord.from_payload(_read_json_payload(path))
                    )
                )
            except Exception:
                logger.warning("Skipping invalid compile-status record at %s", path, exc_info=True)
        return tuple(
            sorted(
                records,
                key=lambda item: (
                    self._sortable_timestamp(item.updated_at),
                    item.job_id,
                ),
                reverse=True,
            )
        )

    def save_activation(self, record: ActivationRecord) -> ActivationRecord:
        """Persist one learned-skill activation record."""

        normalized = self._normalize_activation_record(record)
        path = self._activation_path(normalized.skill_id, normalized.version)
        with self._lock:
            _write_json_atomic(path, normalized.to_payload())
        return normalized

    def load_activation(self, skill_id: str, *, version: int) -> ActivationRecord:
        """Load one learned-skill activation record by skill id and version."""

        normalized_skill_id = self._require_identifier(skill_id, field_name="skill_id")
        normalized_version = self._require_version(version)
        return self._normalize_activation_record(
            ActivationRecord.from_payload(_read_json_payload(self._activation_path(normalized_skill_id, normalized_version)))
        )

    def list_activations(self, *, skill_id: str | None = None) -> tuple[ActivationRecord, ...]:
        """Return all persisted activation records, optionally filtered by skill id."""

        normalized_skill_id = None if skill_id is None else self._require_identifier(skill_id, field_name="skill_id")
        records: list[ActivationRecord] = []
        for path in self._iter_record_paths(self.activations_dir):
            try:
                record = self._normalize_activation_record(ActivationRecord.from_payload(_read_json_payload(path)))
            except Exception:
                logger.warning("Skipping invalid activation record at %s", path, exc_info=True)
                continue
            if normalized_skill_id is not None and record.skill_id != normalized_skill_id:
                continue
            records.append(record)
        return tuple(
            sorted(
                records,
                key=lambda item: (
                    self._sortable_timestamp(item.updated_at),
                    item.version,
                    item.skill_id,
                ),
                reverse=True,
            )
        )

    def find_activation_for_job(self, job_id: str) -> ActivationRecord | None:
        """Return the newest activation linked to one compile job, if present."""

        normalized_job_id = self._require_identifier(job_id, field_name="job_id")
        for record in self.list_activations():
            if record.job_id == normalized_job_id:
                return record
        return None

    def save_skill_health(self, record: SkillHealthRecord) -> SkillHealthRecord:
        """Persist one learned-skill health snapshot."""

        normalized = self._normalize_skill_health_record(record)
        path = self._health_path(normalized.skill_id, normalized.version)
        with self._lock:
            _write_json_atomic(path, normalized.to_payload())
        return normalized

    def load_skill_health(self, skill_id: str, *, version: int) -> SkillHealthRecord:
        """Load one learned-skill health snapshot."""

        normalized_skill_id = self._require_identifier(skill_id, field_name="skill_id")
        normalized_version = self._require_version(version)
        return self._normalize_skill_health_record(
            SkillHealthRecord.from_payload(_read_json_payload(self._health_path(normalized_skill_id, normalized_version)))
        )

    def list_skill_health(self, *, skill_id: str | None = None) -> tuple[SkillHealthRecord, ...]:
        """Return all persisted skill-health snapshots, optionally filtered by skill id."""

        normalized_skill_id = None if skill_id is None else self._require_identifier(skill_id, field_name="skill_id")
        records: list[SkillHealthRecord] = []
        for path in self._iter_record_paths(self.health_dir):
            try:
                record = self._normalize_skill_health_record(SkillHealthRecord.from_payload(_read_json_payload(path)))
            except Exception:
                logger.warning("Skipping invalid skill-health record at %s", path, exc_info=True)
                continue
            if normalized_skill_id is not None and record.skill_id != normalized_skill_id:
                continue
            records.append(record)
        return tuple(
            sorted(
                records,
                key=lambda item: (
                    self._sortable_timestamp(item.updated_at),
                    item.version,
                    item.skill_id,
                ),
                reverse=True,
            )
        )

    def delete_skill_health(self, skill_id: str, *, version: int) -> bool:
        """Remove one persisted learned-skill health snapshot, if present."""

        normalized_skill_id = self._require_identifier(skill_id, field_name="skill_id")
        normalized_version = self._require_version(version)
        path = self._health_path(normalized_skill_id, normalized_version)
        with self._lock:
            # AUDIT-FIX(#1): Delete through the same hardened file primitive used by writes and fsync the directory.
            return _unlink_file_safe(path, missing_ok=True)

    def save_execution_run(
        self,
        record: ExecutionRunStatusRecord | None = None,
        /,
        **kwargs: Any,
    ) -> ExecutionRunStatusRecord:
        """Persist one sandbox execution or retest run snapshot."""

        # AUDIT-FIX(#4): Reject ambiguous call shapes so callers cannot silently lose kwargs or get raw constructor tracebacks.
        if record is not None and kwargs:
            raise ValueError("save_execution_run accepts either a record or keyword fields, not both")
        if record is None and not kwargs:
            raise ValueError("save_execution_run requires a record or keyword fields")
        if record is None:
            try:
                candidate = ExecutionRunStatusRecord(**kwargs)
            except TypeError as exc:
                raise ValueError("save_execution_run received invalid keyword fields") from exc
        else:
            candidate = record
        normalized = self._normalize_execution_run_record(candidate)
        path = self._execution_run_path(normalized.run_id)
        with self._lock:
            _write_json_atomic(path, normalized.to_payload())
        return normalized

    def load_execution_run(self, run_id: str) -> ExecutionRunStatusRecord:
        """Load one sandbox execution or retest run snapshot."""

        normalized_run_id = self._require_identifier(run_id, field_name="run_id")
        return self._normalize_execution_run_record(
            ExecutionRunStatusRecord.from_payload(
                _read_json_payload(self._execution_run_path(normalized_run_id))
            )
        )

    def list_execution_runs(
        self,
        *,
        skill_id: str | None = None,
        run_kind: str | None = None,
        status: str | None = None,
    ) -> tuple[ExecutionRunStatusRecord, ...]:
        """Return persisted sandbox execution runs with optional filters."""

        normalized_skill_id = None if skill_id is None else self._require_identifier(skill_id, field_name="skill_id")
        normalized_run_kind = None if run_kind is None else self._require_identifier(run_kind, field_name="run_kind")
        # AUDIT-FIX(#5): Normalize status with the same helper used at persistence time so filtering is deterministic.
        normalized_status = None if status is None else self._normalize_execution_status(status, field_name="status")
        records: list[ExecutionRunStatusRecord] = []
        for path in self._iter_record_paths(self.execution_runs_dir):
            try:
                record = self._normalize_execution_run_record(
                    ExecutionRunStatusRecord.from_payload(_read_json_payload(path))
                )
            except Exception:
                logger.warning("Skipping invalid execution-run record at %s", path, exc_info=True)
                continue
            if normalized_skill_id is not None and record.skill_id != normalized_skill_id:
                continue
            if normalized_run_kind is not None and record.run_kind != normalized_run_kind:
                continue
            if normalized_status is not None and record.status != normalized_status:
                continue
            records.append(record)
        return tuple(
            sorted(
                records,
                key=lambda item: (
                    self._sortable_timestamp(item.updated_at),
                    item.run_id,
                ),
                reverse=True,
            )
        )

    def save_live_e2e_status(self, record: LiveE2EStatusRecord) -> LiveE2EStatusRecord:
        """Persist one latest live end-to-end proof snapshot."""

        normalized = self._normalize_live_e2e_status_record(record)
        path = self._live_e2e_path(normalized.suite_id, normalized.environment)
        with self._lock:
            _write_json_atomic(path, normalized.to_payload())
        return normalized

    def load_live_e2e_status(self, suite_id: str, *, environment: str) -> LiveE2EStatusRecord:
        """Load one persisted live end-to-end proof snapshot."""

        normalized_suite_id = self._require_identifier(suite_id, field_name="suite_id")
        normalized_environment = self._require_identifier(environment, field_name="environment")
        return self._normalize_live_e2e_status_record(
            LiveE2EStatusRecord.from_payload(
                _read_json_payload(self._live_e2e_path(normalized_suite_id, normalized_environment))
            )
        )

    def list_live_e2e_statuses(
        self,
        *,
        suite_id: str | None = None,
        environment: str | None = None,
    ) -> tuple[LiveE2EStatusRecord, ...]:
        """Return all persisted live proof snapshots with optional filters."""

        normalized_suite_id = None if suite_id is None else self._require_identifier(suite_id, field_name="suite_id")
        normalized_environment = (
            None if environment is None else self._require_identifier(environment, field_name="environment")
        )
        records: list[LiveE2EStatusRecord] = []
        for path in self._iter_record_paths(self.live_e2e_dir):
            try:
                record = self._normalize_live_e2e_status_record(LiveE2EStatusRecord.from_payload(_read_json_payload(path)))
            except Exception:
                logger.warning("Skipping invalid live-e2e status record at %s", path, exc_info=True)
                continue
            if normalized_suite_id is not None and record.suite_id != normalized_suite_id:
                continue
            if normalized_environment is not None and record.environment != normalized_environment:
                continue
            records.append(record)
        return tuple(
            sorted(
                records,
                key=lambda item: (
                    self._sortable_timestamp(item.updated_at),
                    item.environment,
                    item.suite_id,
                ),
                reverse=True,
            )
        )

    def load_artifact(self, artifact_id: str) -> CompileArtifactRecord:
        """Load one compile-artifact record by identifier."""

        normalized_id = self._require_identifier(artifact_id, field_name="artifact_id")
        return self._normalize_artifact_record(
            CompileArtifactRecord.from_payload(_read_json_payload(self._record_json_path(self.artifacts_dir, normalized_id)))
        )

    def list_artifacts(self, *, job_id: str | None = None) -> tuple[CompileArtifactRecord, ...]:
        """Return all persisted artifacts, optionally filtered by compile job."""

        normalized_job_id = None if job_id is None else self._require_identifier(job_id, field_name="job_id")
        records: list[CompileArtifactRecord] = []
        for path in self._iter_record_paths(self.artifacts_dir):
            try:
                record = self._normalize_artifact_record(CompileArtifactRecord.from_payload(_read_json_payload(path)))
            except Exception:
                logger.warning("Skipping invalid artifact record at %s", path, exc_info=True)
                continue
            if normalized_job_id is not None and record.job_id != normalized_job_id:
                continue
            records.append(record)
        return tuple(
            sorted(
                records,
                key=lambda item: (
                    self._sortable_timestamp(item.created_at),
                    item.artifact_id,
                ),
                reverse=True,
            )
        )

    def write_text_artifact(
        self,
        *,
        job_id: str,
        kind: ArtifactKind | str,
        text: str,
        media_type: str = "text/plain",
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
        artifact_id: str | None = None,
        suffix: str = ".txt",
    ) -> CompileArtifactRecord:
        """Persist a text artifact and its metadata record."""

        normalized_job_id = self._require_identifier(job_id, field_name="job_id")
        normalized_artifact_id = (
            _generate_identifier("artifact")
            if artifact_id is None
            else self._require_identifier(artifact_id, field_name="artifact_id")
        )
        normalized_suffix = self._normalize_suffix(suffix)
        content_path = self._content_file_path(normalized_artifact_id, normalized_suffix)
        # AUDIT-FIX(#6): Fail fast on non-string payloads instead of persisting repr() garbage into senior-facing artifacts.
        content_text = self._require_text_payload(text, field_name="text")
        encoded = content_text.encode("utf-8")
        record = CompileArtifactRecord(
            artifact_id=normalized_artifact_id,
            job_id=normalized_job_id,
            kind=kind,
            media_type=media_type,
            content_path=str(content_path.relative_to(self.root)),
            sha256=sha256(encoded).hexdigest(),
            size_bytes=len(encoded),
            summary=summary,
            metadata=metadata or {},
        )
        normalized_record = self._normalize_artifact_record(record)
        previous_content_bytes: bytes | None = None
        with self._lock:
            existing_record = self._load_artifact_if_present(normalized_record.artifact_id)
            if existing_record is not None and existing_record.content_path == normalized_record.content_path:
                try:
                    previous_content_bytes = _read_bytes_file_safe(
                        self._resolve_relative_content_path(existing_record.content_path)
                    )
                except FileNotFoundError:
                    previous_content_bytes = None
            _write_bytes_atomic(content_path, encoded)
            try:
                _write_json_atomic(
                    self._record_json_path(self.artifacts_dir, normalized_record.artifact_id),
                    normalized_record.to_payload(),
                )
            except Exception:
                # AUDIT-FIX(#2): Restore the prior content (or remove the new file) if metadata publication fails.
                if previous_content_bytes is not None:
                    try:
                        _write_bytes_atomic(content_path, previous_content_bytes)
                    except Exception:
                        logger.error(
                            "Failed to restore previous artifact content for %s",
                            normalized_record.artifact_id,
                            exc_info=True,
                        )
                else:
                    try:
                        _unlink_file_safe(content_path, missing_ok=True)
                    except Exception:
                        logger.warning("Failed to clean up orphaned content file %s", content_path, exc_info=True)
                raise
        return normalized_record

    def read_text_artifact(self, artifact_id: str) -> str:
        """Read the text content for one stored artifact."""

        record = self.load_artifact(artifact_id)
        if not record.content_path:
            raise ValueError(f"Artifact {artifact_id!r} has no content_path")
        resolved = self._resolve_relative_content_path(record.content_path)
        content_text = _read_text_file_safe(resolved, encoding="utf-8")
        encoded = content_text.encode("utf-8")
        expected_size = self._require_non_negative_int(record.size_bytes, field_name="size_bytes")
        if len(encoded) != expected_size:
            raise ValueError(f"Artifact {artifact_id!r} content size mismatch")
        expected_sha256 = str(record.sha256 or "").strip().lower()
        if not _SHA256_HEX_PATTERN.fullmatch(expected_sha256):
            raise ValueError(f"Artifact {artifact_id!r} has an invalid sha256")
        if sha256(encoded).hexdigest() != expected_sha256:
            raise ValueError(f"Artifact {artifact_id!r} content hash mismatch")
        return content_text

    def append_artifact_to_job(self, job_id: str, artifact_id: str) -> CompileJobRecord:
        """Attach a persisted artifact id to one compile job."""

        normalized_job_id = self._require_identifier(job_id, field_name="job_id")
        normalized_artifact_id = self._require_identifier(artifact_id, field_name="artifact_id")
        with self._lock:
            record = self.load_job(normalized_job_id)
            self.load_artifact(normalized_artifact_id)
            if normalized_artifact_id in record.artifact_ids:
                return record
            updated = replace(
                record,
                artifact_ids=tuple((*record.artifact_ids, normalized_artifact_id)),
            )
            _write_json_atomic(self._record_json_path(self.jobs_dir, updated.job_id), updated.to_payload())
            return updated

    def _resolve_relative_content_path(self, relative_path: str) -> Path:
        normalized_relative_path = self._normalize_content_path(relative_path)
        return self.root / normalized_relative_path

    def _resolve_store_directory(self, directory: Path) -> Path:
        resolved_directory = directory.expanduser().resolve(strict=False)
        try:
            resolved_directory.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Store directory escapes root: {directory}") from exc
        if resolved_directory.exists() and not resolved_directory.is_dir():
            raise ValueError(f"Store directory is not a directory: {resolved_directory}")
        return resolved_directory

    def _record_json_path(self, directory: Path, identifier: str) -> Path:
        resolved_directory = self._resolve_store_directory(directory)
        return resolved_directory / f"{identifier}.json"

    def _activation_path(self, skill_id: str, version: int) -> Path:
        filename = self._activation_filename(skill_id, version)
        return self._resolve_store_directory(self.activations_dir) / filename

    def _health_path(self, skill_id: str, version: int) -> Path:
        filename = self._activation_filename(skill_id, version)
        return self._resolve_store_directory(self.health_dir) / filename

    def _live_e2e_path(self, suite_id: str, environment: str) -> Path:
        normalized_suite_id = self._require_identifier(suite_id, field_name="suite_id")
        normalized_environment = self._require_identifier(environment, field_name="environment")
        return self._resolve_store_directory(self.live_e2e_dir) / f"{normalized_suite_id}__{normalized_environment}.json"

    def _execution_run_path(self, run_id: str) -> Path:
        normalized_run_id = self._require_identifier(run_id, field_name="run_id")
        return self._resolve_store_directory(self.execution_runs_dir) / f"{normalized_run_id}.json"

    def _content_file_path(self, artifact_id: str, suffix: str) -> Path:
        resolved_contents_dir = self._resolve_store_directory(self.contents_dir)
        return resolved_contents_dir / f"{artifact_id}{suffix}"

    def _iter_record_paths(self, directory: Path) -> tuple[Path, ...]:
        resolved_directory = self._resolve_store_directory(directory)
        if not resolved_directory.exists():
            return ()
        paths: list[Path] = []
        for path in sorted(resolved_directory.glob("*.json")):
            try:
                resolved_target = path.resolve(strict=False)
                resolved_target.relative_to(resolved_directory)
                paths.append(path)
            except Exception:
                logger.warning("Skipping unsafe record path at %s", path, exc_info=True)
        return tuple(paths)

    # AUDIT-FIX(#2): Probe the existing artifact record before overwriting so rollback decisions are based on real prior state.
    def _load_artifact_if_present(self, artifact_id: str) -> CompileArtifactRecord | None:
        try:
            return self.load_artifact(artifact_id)
        except FileNotFoundError:
            return None

    def _normalize_job_record(self, record: CompileJobRecord) -> CompileJobRecord:
        metadata = self._coerce_mapping(record.metadata, field_name="metadata")
        session_id = metadata.get("session_id")
        if session_id is not None and str(session_id).strip():
            metadata["session_id"] = self._require_identifier(session_id, field_name="metadata.session_id")
        return replace(
            record,
            job_id=self._require_identifier(record.job_id, field_name="job_id"),
            artifact_ids=self._normalize_identifier_sequence(record.artifact_ids, field_name="artifact_id"),
            metadata=metadata,
        )

    def _normalize_dialogue_session_record(
        self,
        record: RequirementsDialogueSession,
    ) -> RequirementsDialogueSession:
        return replace(record, session_id=self._require_identifier(record.session_id, field_name="session_id"))

    def _normalize_artifact_record(self, record: CompileArtifactRecord) -> CompileArtifactRecord:
        content_path = None
        if record.content_path is not None and str(record.content_path).strip():
            content_path = self._normalize_content_path(record.content_path)
        return replace(
            record,
            artifact_id=self._require_identifier(record.artifact_id, field_name="artifact_id"),
            job_id=self._require_identifier(record.job_id, field_name="job_id"),
            content_path=content_path,
            metadata=self._coerce_mapping(record.metadata, field_name="metadata"),
        )

    def _normalize_compile_status_record(self, record: CompileRunStatusRecord) -> CompileRunStatusRecord:
        return replace(record, job_id=self._require_identifier(record.job_id, field_name="job_id"))

    def _normalize_activation_record(self, record: ActivationRecord) -> ActivationRecord:
        return replace(
            record,
            skill_id=self._require_identifier(record.skill_id, field_name="skill_id"),
            version=self._require_version(record.version),
            # AUDIT-FIX(#7): Treat blank optional identifiers as absent so legacy payloads do not get dropped during reads.
            job_id=self._normalize_optional_identifier(record.job_id, field_name="job_id"),
        )

    def _normalize_skill_health_record(self, record: SkillHealthRecord) -> SkillHealthRecord:
        return replace(
            record,
            skill_id=self._require_identifier(record.skill_id, field_name="skill_id"),
            version=self._require_version(record.version),
        )

    def _normalize_live_e2e_status_record(self, record: LiveE2EStatusRecord) -> LiveE2EStatusRecord:
        return replace(
            record,
            suite_id=self._require_identifier(record.suite_id, field_name="suite_id"),
            environment=self._require_identifier(record.environment, field_name="environment"),
        )

    def _normalize_execution_run_record(self, record: ExecutionRunStatusRecord) -> ExecutionRunStatusRecord:
        return replace(
            record,
            run_id=self._require_identifier(record.run_id, field_name="run_id"),
            run_kind=self._require_identifier(record.run_kind, field_name="run_kind"),
            skill_id=self._require_identifier(record.skill_id, field_name="skill_id"),
            # AUDIT-FIX(#5): Normalize execution status on save/load so filtering is stable and case-insensitive.
            status=self._normalize_execution_status(record.status, field_name="status"),
            version=self._require_version(record.version),
        )

    def _normalize_content_path(self, relative_path: object) -> str:
        raw_path = str(relative_path or "").strip().replace("\\", "/")
        if not raw_path:
            raise ValueError("content_path must not be empty")
        candidate = Path(raw_path)
        if candidate.is_absolute():
            raise ValueError("content_path must be relative to the store root")
        if any(part in ("", ".", "..") for part in candidate.parts):
            raise ValueError("content_path must not contain traversal segments")
        resolved = (self.root / candidate).resolve(strict=False)
        try:
            resolved.relative_to(self._resolve_store_directory(self.contents_dir))
        except ValueError as exc:
            raise ValueError("content_path must stay within the contents directory") from exc
        return str(resolved.relative_to(self.root))

    @staticmethod
    def _coerce_mapping(value: object, *, field_name: str) -> dict[str, Any]:
        if value is None:
            return {}
        # AUDIT-FIX(#7): Accept generic Mapping inputs instead of rejecting harmless mapping implementations from old payloads/tests.
        if not isinstance(value, Mapping):
            raise ValueError(f"{field_name} must be an object")
        return dict(value)

    @classmethod
    def _normalize_identifier_sequence(
        cls,
        values: Iterable[object] | None,
        *,
        field_name: str,
    ) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        # AUDIT-FIX(#3): Reject string/bytes/mapping inputs so we do not silently explode one identifier into characters or keys.
        if values is None:
            return ()
        if isinstance(values, (str, bytes, bytearray, Mapping)):
            raise ValueError(f"{field_name} values must be an iterable of identifiers")
        for value in values:
            normalized_value = cls._require_identifier(value, field_name=field_name)
            if normalized_value in seen:
                continue
            seen.add(normalized_value)
            normalized.append(normalized_value)
        return tuple(normalized)

    @classmethod
    def _normalize_optional_identifier(cls, value: object | None, *, field_name: str) -> str | None:
        # AUDIT-FIX(#7): Blank optional identifiers from legacy payloads should normalize to None, not poison the whole record.
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return cls._require_identifier(value, field_name=field_name)

    @staticmethod
    def _sortable_timestamp(value: object) -> tuple[int, str]:
        if value is None:
            return (0, "")
        if isinstance(value, datetime):
            if value.tzinfo is not None:
                return (1, value.astimezone(timezone.utc).isoformat())
            return (1, value.isoformat())
        text = str(value).strip()
        if not text:
            return (0, "")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return (1, text)
        if parsed.tzinfo is not None:
            return (1, parsed.astimezone(timezone.utc).isoformat())
        return (1, parsed.isoformat())

    @staticmethod
    def _require_identifier(value: object, *, field_name: str) -> str:
        raw = str(value or "").strip().lower()
        if len(raw) > 128:
            raise ValueError(f"{field_name} must be <= 128 characters")
        text = truncate_text(raw, limit=128)
        if not is_valid_stable_identifier(text):
            raise ValueError(f"{field_name} must be a stable identifier")
        return text

    @staticmethod
    def _normalize_execution_status(value: object, *, field_name: str) -> str:
        raw = str(value or "").strip().lower()
        if len(raw) > 128:
            raise ValueError(f"{field_name} must be <= 128 characters")
        text = truncate_text(raw, limit=128)
        if not text:
            raise ValueError(f"{field_name} must not be empty")
        return text

    @staticmethod
    def _require_text_payload(value: object | None, *, field_name: str) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")
        return value

    @staticmethod
    def _parse_strict_int(value: object, *, field_name: str) -> int:
        # AUDIT-FIX(#8): Reject float/Decimal truncation and accept only real integers or digit-only strings.
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be an integer")
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if not _INTEGER_TEXT_PATTERN.fullmatch(text):
                raise ValueError(f"{field_name} must be an integer")
            return int(text)
        raise ValueError(f"{field_name} must be an integer")

    @classmethod
    def _require_version(cls, value: object) -> int:
        normalized = cls._parse_strict_int(value, field_name="version")
        if normalized < 1:
            raise ValueError("version must be >= 1")
        return normalized

    @staticmethod
    def _normalize_suffix(value: object) -> str:
        raw = str(value or ".txt").strip().lower()
        if not raw.startswith("."):
            raw = f".{raw}"
        if len(raw) > 16:
            raise ValueError("suffix must be <= 16 characters")
        suffix = truncate_text(raw, limit=16)
        if "/" in suffix or "\\" in suffix or "\x00" in suffix:
            raise ValueError("suffix must not contain path separators")
        if not _SAFE_SUFFIX_PATTERN.fullmatch(suffix):
            raise ValueError("suffix must be a short safe file extension")
        return suffix

    @classmethod
    def _require_non_negative_int(cls, value: object, *, field_name: str) -> int:
        normalized = cls._parse_strict_int(value, field_name=field_name)
        if normalized < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
        return normalized

    @classmethod
    def _activation_filename(cls, skill_id: str, version: int) -> str:
        normalized_skill_id = cls._require_identifier(skill_id, field_name="skill_id")
        normalized_version = cls._require_version(version)
        return f"{normalized_skill_id}__v{normalized_version}.json"