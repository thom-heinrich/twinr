"""Persist and load live midterm-memory attestation artifacts.

This module keeps the lightweight result contract and artifact I/O used by the
live midterm acceptance script and the web operator debug surface. It does not
run provider or ChonkyDB calls itself; execution lives in
``live_midterm_acceptance.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import json
import os


_SCHEMA_VERSION = 1
_ATTEST_KIND = "live_midterm_e2e"
_OPS_ARTIFACT_NAME = "memory_attest.json"
_REPORT_DIR_NAME = "memory_attest"


def _coerce_text(value: object | None) -> str:
    """Normalize arbitrary values into a stable single-line string."""

    return " ".join(str(value or "").split()).strip()


def _coerce_optional_text(value: object | None) -> str | None:
    """Normalize optional text values and collapse blanks to ``None``."""

    text = _coerce_text(value)
    return text or None


def _coerce_str_tuple(values: object | None) -> tuple[str, ...]:
    """Normalize optional string iterables into a compact tuple."""

    if values is None:
        return ()
    if isinstance(values, str):
        text = _coerce_text(values)
        return (text,) if text else ()
    if not isinstance(values, (list, tuple)):
        return ()
    normalized: list[str] = []
    for item in values:
        text = _coerce_text(item)
        if text:
            normalized.append(text)
    return tuple(normalized)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON payload atomically to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, path)


@dataclass(frozen=True, slots=True)
class LiveMidtermSeedTurn:
    """Capture one live seed turn used during memory attestation."""

    prompt: str
    response_text: str
    model: str | None = None
    request_id: str | None = None
    response_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _coerce_text(self.prompt))
        object.__setattr__(self, "response_text", _coerce_text(self.response_text))
        object.__setattr__(self, "model", _coerce_optional_text(self.model))
        object.__setattr__(self, "request_id", _coerce_optional_text(self.request_id))
        object.__setattr__(self, "response_id", _coerce_optional_text(self.response_id))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LiveMidtermSeedTurn":
        """Hydrate one stored seed turn payload."""

        return cls(
            prompt=str(payload.get("prompt", "") or ""),
            response_text=str(payload.get("response_text", "") or ""),
            model=_coerce_optional_text(payload.get("model")),
            request_id=_coerce_optional_text(payload.get("request_id")),
            response_id=_coerce_optional_text(payload.get("response_id")),
        )


@dataclass(frozen=True, slots=True)
class LiveMidtermAttestResult:
    """Describe one live midterm write/read/usage attestation run."""

    probe_id: str
    status: str
    started_at: str
    finished_at: str
    env_path: str
    base_project_root: str
    runtime_namespace: str
    writer_root: str | None = None
    fresh_reader_root: str | None = None
    flush_ok: bool = False
    midterm_context_present: bool = False
    follow_up_query: str | None = None
    follow_up_answer_text: str | None = None
    follow_up_model: str | None = None
    follow_up_request_id: str | None = None
    follow_up_response_id: str | None = None
    expected_answer_terms: tuple[str, ...] = ()
    matched_answer_terms: tuple[str, ...] = ()
    writer_packet_ids: tuple[str, ...] = ()
    remote_packet_ids: tuple[str, ...] = ()
    fresh_reader_packet_ids: tuple[str, ...] = ()
    seed_turns: tuple[LiveMidtermSeedTurn, ...] = ()
    midterm_warning_logs: str | None = None
    outside_remote_read_logs: str | None = None
    outside_fallback_logs: str | None = None
    last_path_warning_class: str | None = None
    last_path_warning_message: str | None = None
    error_message: str | None = None
    artifact_path: str | None = None
    report_path: str | None = None
    attest_kind: str = field(default=_ATTEST_KIND, init=False)
    schema_version: int = field(default=_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "probe_id", _coerce_text(self.probe_id))
        object.__setattr__(self, "status", _coerce_text(self.status).lower() or "unknown")
        object.__setattr__(self, "started_at", _coerce_text(self.started_at))
        object.__setattr__(self, "finished_at", _coerce_text(self.finished_at))
        object.__setattr__(self, "env_path", _coerce_text(self.env_path))
        object.__setattr__(self, "base_project_root", _coerce_text(self.base_project_root))
        object.__setattr__(self, "runtime_namespace", _coerce_text(self.runtime_namespace))
        object.__setattr__(self, "writer_root", _coerce_optional_text(self.writer_root))
        object.__setattr__(self, "fresh_reader_root", _coerce_optional_text(self.fresh_reader_root))
        object.__setattr__(self, "follow_up_query", _coerce_optional_text(self.follow_up_query))
        object.__setattr__(self, "follow_up_answer_text", _coerce_optional_text(self.follow_up_answer_text))
        object.__setattr__(self, "follow_up_model", _coerce_optional_text(self.follow_up_model))
        object.__setattr__(self, "follow_up_request_id", _coerce_optional_text(self.follow_up_request_id))
        object.__setattr__(self, "follow_up_response_id", _coerce_optional_text(self.follow_up_response_id))
        object.__setattr__(self, "expected_answer_terms", _coerce_str_tuple(self.expected_answer_terms))
        object.__setattr__(self, "matched_answer_terms", _coerce_str_tuple(self.matched_answer_terms))
        object.__setattr__(self, "writer_packet_ids", _coerce_str_tuple(self.writer_packet_ids))
        object.__setattr__(self, "remote_packet_ids", _coerce_str_tuple(self.remote_packet_ids))
        object.__setattr__(self, "fresh_reader_packet_ids", _coerce_str_tuple(self.fresh_reader_packet_ids))
        object.__setattr__(
            self,
            "seed_turns",
            tuple(
                item if isinstance(item, LiveMidtermSeedTurn) else LiveMidtermSeedTurn.from_dict(dict(item))
                for item in (self.seed_turns or ())
            ),
        )
        object.__setattr__(self, "midterm_warning_logs", _coerce_optional_text(self.midterm_warning_logs))
        object.__setattr__(self, "outside_remote_read_logs", _coerce_optional_text(self.outside_remote_read_logs))
        object.__setattr__(self, "outside_fallback_logs", _coerce_optional_text(self.outside_fallback_logs))
        object.__setattr__(self, "last_path_warning_class", _coerce_optional_text(self.last_path_warning_class))
        object.__setattr__(self, "last_path_warning_message", _coerce_optional_text(self.last_path_warning_message))
        object.__setattr__(self, "error_message", _coerce_optional_text(self.error_message))
        object.__setattr__(self, "artifact_path", _coerce_optional_text(self.artifact_path))
        object.__setattr__(self, "report_path", _coerce_optional_text(self.report_path))

    @property
    def ready(self) -> bool:
        """Return whether this attestation run proved the full chain."""

        return self.status == "ok"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["seed_turns"] = [item.to_dict() for item in self.seed_turns]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LiveMidtermAttestResult":
        """Hydrate one stored attestation payload."""

        raw_seed_turns = payload.get("seed_turns")
        seed_turns: tuple[LiveMidtermSeedTurn, ...] = ()
        if isinstance(raw_seed_turns, list):
            seed_turns = tuple(
                LiveMidtermSeedTurn.from_dict(dict(item))
                for item in raw_seed_turns
                if isinstance(item, dict)
            )
        return cls(
            probe_id=str(payload.get("probe_id", "") or ""),
            status=str(payload.get("status", "unknown") or "unknown"),
            started_at=str(payload.get("started_at", "") or ""),
            finished_at=str(payload.get("finished_at", "") or ""),
            env_path=str(payload.get("env_path", "") or ""),
            base_project_root=str(payload.get("base_project_root", "") or ""),
            runtime_namespace=str(payload.get("runtime_namespace", "") or ""),
            writer_root=_coerce_optional_text(payload.get("writer_root")),
            fresh_reader_root=_coerce_optional_text(payload.get("fresh_reader_root")),
            flush_ok=bool(payload.get("flush_ok", False)),
            midterm_context_present=bool(payload.get("midterm_context_present", False)),
            follow_up_query=_coerce_optional_text(payload.get("follow_up_query")),
            follow_up_answer_text=_coerce_optional_text(payload.get("follow_up_answer_text")),
            follow_up_model=_coerce_optional_text(payload.get("follow_up_model")),
            follow_up_request_id=_coerce_optional_text(payload.get("follow_up_request_id")),
            follow_up_response_id=_coerce_optional_text(payload.get("follow_up_response_id")),
            expected_answer_terms=_coerce_str_tuple(payload.get("expected_answer_terms")),
            matched_answer_terms=_coerce_str_tuple(payload.get("matched_answer_terms")),
            writer_packet_ids=_coerce_str_tuple(payload.get("writer_packet_ids")),
            remote_packet_ids=_coerce_str_tuple(payload.get("remote_packet_ids")),
            fresh_reader_packet_ids=_coerce_str_tuple(payload.get("fresh_reader_packet_ids")),
            seed_turns=seed_turns,
            midterm_warning_logs=_coerce_optional_text(payload.get("midterm_warning_logs")),
            outside_remote_read_logs=_coerce_optional_text(payload.get("outside_remote_read_logs")),
            outside_fallback_logs=_coerce_optional_text(payload.get("outside_fallback_logs")),
            last_path_warning_class=_coerce_optional_text(payload.get("last_path_warning_class")),
            last_path_warning_message=_coerce_optional_text(payload.get("last_path_warning_message")),
            error_message=_coerce_optional_text(payload.get("error_message")),
            artifact_path=_coerce_optional_text(payload.get("artifact_path")),
            report_path=_coerce_optional_text(payload.get("report_path")),
        )


def default_live_midterm_attest_path(project_root: str | Path) -> Path:
    """Return the rolling ops artifact path for the latest live memory attest."""

    return Path(project_root).expanduser().resolve() / "artifacts" / "stores" / "ops" / _OPS_ARTIFACT_NAME


def default_live_midterm_attest_report_dir(project_root: str | Path) -> Path:
    """Return the report directory used for per-run attestation snapshots."""

    return Path(project_root).expanduser().resolve() / "artifacts" / "reports" / _REPORT_DIR_NAME


def load_live_midterm_attest(path: str | Path) -> LiveMidtermAttestResult | None:
    """Load one attestation artifact from disk if it exists and is valid."""

    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.exists():
        return None
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Memory attest artifact must be a JSON object: {artifact_path}")
    return LiveMidtermAttestResult.from_dict(payload)


def write_live_midterm_attest_artifacts(
    result: LiveMidtermAttestResult,
    *,
    project_root: str | Path,
) -> LiveMidtermAttestResult:
    """Persist the latest ops artifact plus a per-run report snapshot."""

    artifact_path = default_live_midterm_attest_path(project_root)
    report_dir = default_live_midterm_attest_report_dir(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{result.probe_id}.json"
    persisted = replace(
        result,
        artifact_path=str(artifact_path),
        report_path=str(report_path),
    )
    payload = persisted.to_dict()
    _atomic_write_json(report_path, payload)
    _atomic_write_json(artifact_path, payload)
    return persisted


def load_latest_live_midterm_attest(project_root: str | Path) -> LiveMidtermAttestResult | None:
    """Load the latest rolling live-memory attestation artifact."""

    return load_live_midterm_attest(default_live_midterm_attest_path(project_root))
