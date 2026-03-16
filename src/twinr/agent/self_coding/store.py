"""Persist core Adaptive Skill Engine records under Twinr's state directory.

The store is intentionally narrow: it owns compile-job metadata, compile-
artifact metadata, and optional text artifacts written by later compile
workers. Higher-level orchestration stays outside this module.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
import os
import tempfile
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from twinr.text_utils import is_valid_stable_identifier, truncate_text

from .contracts import CompileArtifactRecord, CompileJobRecord, RequirementsDialogueSession
from .status import ArtifactKind

if TYPE_CHECKING:
    from twinr.agent.base_agent.config import TwinrConfig


def self_coding_store_root(project_root: str | Path) -> Path:
    """Return the canonical on-disk root for self-coding runtime state."""

    return Path(project_root).expanduser().resolve(strict=False) / "state" / "self_coding"


def _generate_identifier(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


def _read_json_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object payload in {path}")
    return payload


class SelfCodingStore:
    """Store self-coding job and artifact records under one project root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self.dialogues_dir = self.root / "dialogues"
        self.jobs_dir = self.root / "jobs"
        self.artifacts_dir = self.root / "artifacts"
        self.contents_dir = self.root / "contents"

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "SelfCodingStore":
        return cls(self_coding_store_root(project_root))

    @classmethod
    def from_config(cls, config: "TwinrConfig") -> "SelfCodingStore":
        return cls.from_project_root(getattr(config, "project_root", "."))

    def save_job(self, record: CompileJobRecord) -> CompileJobRecord:
        """Persist one compile-job record."""

        path = self.jobs_dir / f"{record.job_id}.json"
        _write_json_atomic(path, record.to_payload())
        return record

    def load_job(self, job_id: str) -> CompileJobRecord:
        """Load one compile-job record by identifier."""

        normalized_id = self._require_identifier(job_id, field_name="job_id")
        return CompileJobRecord.from_payload(_read_json_payload(self.jobs_dir / f"{normalized_id}.json"))

    def list_jobs(self) -> tuple[CompileJobRecord, ...]:
        """Return all persisted compile-job records sorted by update time."""

        records: list[CompileJobRecord] = []
        if not self.jobs_dir.exists():
            return ()
        for path in sorted(self.jobs_dir.glob("*.json")):
            records.append(CompileJobRecord.from_payload(_read_json_payload(path)))
        return tuple(sorted(records, key=lambda item: (item.updated_at, item.created_at, item.job_id), reverse=True))

    def save_dialogue_session(self, record: RequirementsDialogueSession) -> RequirementsDialogueSession:
        """Persist one requirements-dialogue session record."""

        path = self.dialogues_dir / f"{record.session_id}.json"
        _write_json_atomic(path, record.to_payload())
        return record

    def load_dialogue_session(self, session_id: str) -> RequirementsDialogueSession:
        """Load one requirements-dialogue session by identifier."""

        normalized_id = self._require_identifier(session_id, field_name="session_id")
        return RequirementsDialogueSession.from_payload(_read_json_payload(self.dialogues_dir / f"{normalized_id}.json"))

    def list_dialogue_sessions(self) -> tuple[RequirementsDialogueSession, ...]:
        """Return all persisted dialogue sessions sorted by update time."""

        records: list[RequirementsDialogueSession] = []
        if not self.dialogues_dir.exists():
            return ()
        for path in sorted(self.dialogues_dir.glob("*.json")):
            records.append(RequirementsDialogueSession.from_payload(_read_json_payload(path)))
        return tuple(sorted(records, key=lambda item: (item.updated_at, item.created_at, item.session_id), reverse=True))

    def save_artifact(self, record: CompileArtifactRecord) -> CompileArtifactRecord:
        """Persist one compile-artifact record."""

        path = self.artifacts_dir / f"{record.artifact_id}.json"
        _write_json_atomic(path, record.to_payload())
        return record

    def load_artifact(self, artifact_id: str) -> CompileArtifactRecord:
        """Load one compile-artifact record by identifier."""

        normalized_id = self._require_identifier(artifact_id, field_name="artifact_id")
        return CompileArtifactRecord.from_payload(_read_json_payload(self.artifacts_dir / f"{normalized_id}.json"))

    def list_artifacts(self, *, job_id: str | None = None) -> tuple[CompileArtifactRecord, ...]:
        """Return all persisted artifacts, optionally filtered by compile job."""

        normalized_job_id = None if job_id is None else self._require_identifier(job_id, field_name="job_id")
        records: list[CompileArtifactRecord] = []
        if not self.artifacts_dir.exists():
            return ()
        for path in sorted(self.artifacts_dir.glob("*.json")):
            record = CompileArtifactRecord.from_payload(_read_json_payload(path))
            if normalized_job_id is not None and record.job_id != normalized_job_id:
                continue
            records.append(record)
        return tuple(sorted(records, key=lambda item: (item.created_at, item.artifact_id), reverse=True))

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
        normalized_suffix = truncate_text(str(suffix or ".txt"), limit=16) or ".txt"
        if not normalized_suffix.startswith("."):
            normalized_suffix = f".{normalized_suffix}"
        content_path = self.contents_dir / f"{normalized_artifact_id}{normalized_suffix}"
        content_path.parent.mkdir(parents=True, exist_ok=True)
        content_text = "" if text is None else str(text)
        content_path.write_text(content_text, encoding="utf-8")
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
        self.save_artifact(record)
        return record

    def read_text_artifact(self, artifact_id: str) -> str:
        """Read the text content for one stored artifact."""

        record = self.load_artifact(artifact_id)
        if not record.content_path:
            raise ValueError(f"Artifact {artifact_id!r} has no content_path")
        resolved = self._resolve_relative_content_path(record.content_path)
        return resolved.read_text(encoding="utf-8")

    def append_artifact_to_job(self, job_id: str, artifact_id: str) -> CompileJobRecord:
        """Attach a persisted artifact id to one compile job."""

        record = self.load_job(job_id)
        normalized_artifact_id = self._require_identifier(artifact_id, field_name="artifact_id")
        if normalized_artifact_id in record.artifact_ids:
            return record
        updated = replace(
            record,
            artifact_ids=tuple((*record.artifact_ids, normalized_artifact_id)),
        )
        return self.save_job(updated)

    def _resolve_relative_content_path(self, relative_path: str) -> Path:
        candidate = (self.root / Path(relative_path)).resolve(strict=False)
        candidate.relative_to(self.root)
        return candidate

    @staticmethod
    def _require_identifier(value: object, *, field_name: str) -> str:
        text = truncate_text(str(value or "").strip().lower(), limit=128)
        if not is_valid_stable_identifier(text):
            raise ValueError(f"{field_name} must be a stable identifier")
        return text
