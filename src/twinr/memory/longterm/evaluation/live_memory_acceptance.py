"""Run a live synthetic-memory acceptance matrix against OpenAI and ChonkyDB.

This module seeds one isolated remote namespace with adversarial long-term
memory fixtures, asks the live LLM several recall/control questions before and
after conflict confirmation, then repeats the critical cases from a fresh
runtime root to prove restart persistence. The goal is not benchmark breadth;
it is a sharp acceptance proof for earlier-memory recall, conflict handling,
confirmed-memory recall, restart persistence, and control-query containment.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from twinr.agent.base_agent.conversation.language import memory_and_response_contract
from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.core.models import (
    LongTermConflictQueueItemV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.providers.openai import OpenAIBackend
from twinr.text_utils import folded_lookup_text
from twinr.memory.longterm.evaluation.live_midterm_acceptance import (
    _build_isolated_config,
    _close_openai_backend,
    _configure_openai_backend_client,
    _normalize_base_project_root,
    _shutdown_service,
)


_SCHEMA_VERSION = 1
_ACCEPT_KIND = "live_memory_acceptance"
_OPS_ARTIFACT_NAME = "memory_live_acceptance.json"
_REPORT_DIR_NAME = "memory_live_acceptance"
_MODEL_TIMEOUT_S = 45.0
_MODEL_MAX_RETRIES = 1
_REMOTE_READ_DIAGNOSTICS_ROOT_ENV = "TWINR_REMOTE_READ_DIAGNOSTICS_PROJECT_ROOT"


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


def _push_project_root_override(project_root: Path) -> tuple[str, str | None]:
    """Point remote-read diagnostics at the durable base project root."""

    previous = os.environ.get(_REMOTE_READ_DIAGNOSTICS_ROOT_ENV)
    os.environ[_REMOTE_READ_DIAGNOSTICS_ROOT_ENV] = str(project_root)
    return _REMOTE_READ_DIAGNOSTICS_ROOT_ENV, previous


def _restore_project_root_override(state: tuple[str, str | None]) -> None:
    """Restore the previous remote-read diagnostics project-root override."""

    key, previous = state
    if previous is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = previous


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in stable ISO-8601 form."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_namespace_suffix(value: str) -> str:
    """Normalize a probe id into a ChonkyDB-safe namespace suffix."""

    safe_chars: list[str] = []
    for char in str(value or "").lower():
        if char.isalnum():
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    normalized = "".join(safe_chars).strip("_")
    return normalized or "memory_live"


def _normalize_lookup_text(text: object | None) -> str:
    """Fold and whitespace-normalize text for deterministic membership checks."""

    return " ".join(folded_lookup_text(str(text or "")).split())


def _matched_terms(text: str | None, expected_terms: tuple[str, ...]) -> tuple[str, ...]:
    """Return the expected lookup terms present in the supplied text."""

    normalized = _normalize_lookup_text(text)
    matches: list[str] = []
    for item in expected_terms:
        normalized_item = _normalize_lookup_text(item)
        if normalized_item and normalized_item in normalized:
            matches.append(item)
    return tuple(matches)


def _missing_terms(text: str | None, expected_terms: tuple[str, ...]) -> tuple[str, ...]:
    """Return the expected terms missing from one normalized text."""

    matched = set(_matched_terms(text, expected_terms))
    return tuple(term for term in expected_terms if term not in matched)


def _source(event_id: str) -> LongTermSourceRefV1:
    """Build one canonical source reference for seeded synthetic objects."""

    return LongTermSourceRefV1(
        source_type="synthetic_acceptance",
        event_ids=(event_id,),
        speaker="user",
        modality="text",
    )


@dataclass(frozen=True, slots=True)
class _AcceptanceCase:
    """Define one live query case in the synthetic-memory acceptance matrix."""

    case_id: str
    phase: str
    query_text: str
    required_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    required_context_terms: tuple[str, ...] = ()
    forbidden_context_terms: tuple[str, ...] = ()
    include_conflict_queue_context: bool = False


@dataclass(frozen=True, slots=True)
class LiveMemoryAcceptanceCaseResult:
    """Capture the result of one live memory acceptance query."""

    case_id: str
    phase: str
    query_text: str
    answer_text: str
    matched_terms: tuple[str, ...] = ()
    missing_terms: tuple[str, ...] = ()
    leaked_terms: tuple[str, ...] = ()
    context_matched_terms: tuple[str, ...] = ()
    context_missing_terms: tuple[str, ...] = ()
    context_leaked_terms: tuple[str, ...] = ()
    passed: bool = False
    model: str | None = None
    request_id: str | None = None
    response_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _coerce_text(self.case_id))
        object.__setattr__(self, "phase", _coerce_text(self.phase))
        object.__setattr__(self, "query_text", _coerce_text(self.query_text))
        object.__setattr__(self, "answer_text", _coerce_text(self.answer_text))
        object.__setattr__(self, "matched_terms", _coerce_str_tuple(self.matched_terms))
        object.__setattr__(self, "missing_terms", _coerce_str_tuple(self.missing_terms))
        object.__setattr__(self, "leaked_terms", _coerce_str_tuple(self.leaked_terms))
        object.__setattr__(self, "context_matched_terms", _coerce_str_tuple(self.context_matched_terms))
        object.__setattr__(self, "context_missing_terms", _coerce_str_tuple(self.context_missing_terms))
        object.__setattr__(self, "context_leaked_terms", _coerce_str_tuple(self.context_leaked_terms))
        object.__setattr__(self, "model", _coerce_optional_text(self.model))
        object.__setattr__(self, "request_id", _coerce_optional_text(self.request_id))
        object.__setattr__(self, "response_id", _coerce_optional_text(self.response_id))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LiveMemoryAcceptanceResult:
    """Describe one complete live synthetic-memory acceptance run."""

    probe_id: str
    status: str
    started_at: str
    finished_at: str
    env_path: str
    base_project_root: str
    runtime_namespace: str
    writer_root: str | None = None
    fresh_reader_root: str | None = None
    selected_memory_id: str | None = None
    queue_before_count: int = 0
    queue_after_count: int = 0
    restart_queue_count: int = 0
    case_results: tuple[LiveMemoryAcceptanceCaseResult, ...] = ()
    error_message: str | None = None
    artifact_path: str | None = None
    report_path: str | None = None
    accept_kind: str = field(default=_ACCEPT_KIND, init=False)
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
        object.__setattr__(self, "selected_memory_id", _coerce_optional_text(self.selected_memory_id))
        object.__setattr__(self, "error_message", _coerce_optional_text(self.error_message))
        object.__setattr__(self, "artifact_path", _coerce_optional_text(self.artifact_path))
        object.__setattr__(self, "report_path", _coerce_optional_text(self.report_path))
        object.__setattr__(
            self,
            "case_results",
            tuple(
                item if isinstance(item, LiveMemoryAcceptanceCaseResult) else LiveMemoryAcceptanceCaseResult(**dict(item))
                for item in (self.case_results or ())
            ),
        )

    @property
    def passed_cases(self) -> int:
        """Return the number of passing query cases."""

        return sum(1 for item in self.case_results if item.passed)

    @property
    def total_cases(self) -> int:
        """Return the total number of query cases."""

        return len(self.case_results)

    @property
    def ready(self) -> bool:
        """Return whether the full live acceptance run passed."""

        return (
            self.status == "ok"
            and self.total_cases > 0
            and self.passed_cases == self.total_cases
            and self.queue_before_count > 0
            and self.queue_after_count == 0
            and self.restart_queue_count == 0
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["case_results"] = [item.to_dict() for item in self.case_results]
        payload["passed_cases"] = self.passed_cases
        payload["total_cases"] = self.total_cases
        payload["ready"] = self.ready
        return payload


def default_live_memory_acceptance_path(project_root: str | Path) -> Path:
    """Return the rolling ops artifact path for the latest live acceptance."""

    return Path(project_root).expanduser().resolve() / "artifacts" / "stores" / "ops" / _OPS_ARTIFACT_NAME


def default_live_memory_acceptance_report_dir(project_root: str | Path) -> Path:
    """Return the report directory used for per-run acceptance snapshots."""

    return Path(project_root).expanduser().resolve() / "artifacts" / "reports" / _REPORT_DIR_NAME


def write_live_memory_acceptance_artifacts(
    result: LiveMemoryAcceptanceResult,
    *,
    project_root: str | Path,
) -> LiveMemoryAcceptanceResult:
    """Persist the latest ops artifact plus a per-run report snapshot."""

    artifact_path = default_live_memory_acceptance_path(project_root)
    report_dir = default_live_memory_acceptance_report_dir(project_root)
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


def _seed_synthetic_memory(service: LongTermMemoryService) -> str:
    """Seed the isolated namespace with earlier, generic, and conflicting facts."""

    occurred_at = datetime(2026, 3, 18, 9, 0, tzinfo=timezone.utc)
    thermos = LongTermMemoryObjectV1(
        memory_id="fact:thermos_location_old",
        kind="fact",
        summary="Früher stand die rote Thermoskanne im Flurschrank.",
        details="Historische Ortsangabe zur roten Thermoskanne.",
        source=_source("thermos:old"),
        status="active",
        confidence=0.99,
        confirmed_by_user=True,
        slot_key="object:red_thermos:location",
        value_key="hallway_cupboard",
        updated_at=occurred_at,
        created_at=occurred_at,
    )
    jam_generic = LongTermMemoryObjectV1(
        memory_id="fact:jam_generic",
        kind="fact",
        summary="User usually likes some jam on bread at breakfast.",
        details='User said: "Ich mag beim Frühstück meistens etwas Marmelade auf dem Brot."',
        source=_source("jam:generic"),
        status="active",
        confidence=0.84,
        slot_key="fact:user:breakfast:jam",
        value_key="jam_on_bread_at_breakfast",
        attributes={
            "fact_type": "general",
            "memory_domain": "general",
            "value_text": "jam on bread at breakfast",
        },
        updated_at=occurred_at,
        created_at=occurred_at,
    )
    jam_old = LongTermMemoryObjectV1(
        memory_id="fact:jam_preference_old",
        kind="fact",
        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
        details="Aeltere Vorliebe fuer das Fruehstueck.",
        source=_source("jam:old"),
        status="active",
        confidence=0.94,
        slot_key="preference:breakfast:jam",
        value_key="strawberry",
        updated_at=occurred_at,
        created_at=occurred_at,
    )
    jam_new = LongTermMemoryObjectV1(
        memory_id="fact:jam_preference_new",
        kind="fact",
        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
        details="Neuere Vorliebe fuer das Fruehstueck.",
        source=_source("jam:new"),
        status="uncertain",
        confidence=0.95,
        slot_key="preference:breakfast:jam",
        value_key="apricot",
        updated_at=occurred_at,
        created_at=occurred_at,
    )
    conflict = LongTermMemoryConflictV1(
        slot_key="preference:breakfast:jam",
        candidate_memory_id=jam_new.memory_id,
        existing_memory_ids=(jam_old.memory_id,),
        question="Welche Marmelade stimmt gerade?",
        reason="Widerspruechliche Marmeladenpraeferenzen liegen vor.",
    )
    service.object_store.commit_active_delta(
        object_upserts=(thermos, jam_generic, jam_old, jam_new),
        conflict_upserts=(conflict,),
    )
    return jam_new.memory_id


def _default_cases() -> tuple[_AcceptanceCase, ...]:
    """Return the fixed adversarial case set for the live acceptance run."""

    return (
        _AcceptanceCase(
            case_id="earlier_before",
            phase="before_resolution",
            query_text="Wo stand früher meine rote Thermoskanne?",
            required_terms=("Flurschrank",),
            required_context_terms=("Flurschrank",),
        ),
        _AcceptanceCase(
            case_id="conflict_before",
            phase="before_resolution",
            query_text="Welche Marmeladen stehen gerade im Widerspruch?",
            required_terms=("Erdbeermarmelade", "Aprikosenmarmelade"),
            required_context_terms=("Erdbeermarmelade", "Aprikosenmarmelade"),
            include_conflict_queue_context=True,
        ),
        _AcceptanceCase(
            case_id="control_before",
            phase="before_resolution",
            query_text="Was ist ein Regenbogen?",
            forbidden_terms=("Flurschrank", "Erdbeermarmelade", "Aprikosenmarmelade"),
            forbidden_context_terms=("Erdbeermarmelade", "Aprikosenmarmelade"),
        ),
        _AcceptanceCase(
            case_id="resolved_meta_writer",
            phase="after_resolution",
            query_text="Welche Marmelade ist jetzt als bestaetigt gespeichert?",
            required_terms=("Aprikosenmarmelade",),
            forbidden_terms=("Erdbeermarmelade",),
            required_context_terms=("Aprikosenmarmelade", "confirmed_by_user"),
        ),
        _AcceptanceCase(
            case_id="resolved_current_writer",
            phase="after_resolution",
            query_text="Welche Marmeladensorte ist aktuell gespeichert?",
            required_terms=("Aprikosenmarmelade",),
            forbidden_terms=("Erdbeermarmelade",),
            required_context_terms=("Aprikosenmarmelade",),
        ),
        _AcceptanceCase(
            case_id="resolved_meta_restart",
            phase="after_restart",
            query_text="Welche Marmelade ist jetzt als bestaetigt gespeichert?",
            required_terms=("Aprikosenmarmelade",),
            forbidden_terms=("Erdbeermarmelade",),
            required_context_terms=("Aprikosenmarmelade", "confirmed_by_user"),
        ),
        _AcceptanceCase(
            case_id="earlier_restart",
            phase="after_restart",
            query_text="Wo stand früher meine rote Thermoskanne?",
            required_terms=("Flurschrank",),
            required_context_terms=("Flurschrank",),
        ),
        _AcceptanceCase(
            case_id="control_restart",
            phase="after_restart",
            query_text="Was ist ein Regenbogen?",
            forbidden_terms=("Flurschrank", "Erdbeermarmelade", "Aprikosenmarmelade"),
            forbidden_context_terms=("Erdbeermarmelade", "Aprikosenmarmelade"),
        ),
    )


def _run_case(
    *,
    service: LongTermMemoryService,
    config: TwinrConfig,
    backend: OpenAIBackend,
    case: _AcceptanceCase,
) -> LiveMemoryAcceptanceCaseResult:
    """Execute one live query case against the current isolated runtime."""

    conversation = _acceptance_provider_conversation(
        service=service,
        config=config,
        case=case,
    )
    context_text = "\n".join(content for role, content in conversation if str(role) == "system")
    response = backend.respond_with_metadata(
        case.query_text,
        conversation=conversation,
        allow_web_search=False,
    )
    answer_text = str(getattr(response, "text", "") or "").strip()
    missing_terms = _missing_terms(answer_text, case.required_terms)
    leaked_terms = _matched_terms(answer_text, case.forbidden_terms)
    context_missing_terms = _missing_terms(context_text, case.required_context_terms)
    context_leaked_terms = _matched_terms(context_text, case.forbidden_context_terms)
    passed = not missing_terms and not leaked_terms and not context_missing_terms and not context_leaked_terms
    return LiveMemoryAcceptanceCaseResult(
        case_id=case.case_id,
        phase=case.phase,
        query_text=case.query_text,
        answer_text=answer_text,
        matched_terms=_matched_terms(answer_text, case.required_terms),
        missing_terms=missing_terms,
        leaked_terms=leaked_terms,
        context_matched_terms=_matched_terms(context_text, case.required_context_terms),
        context_missing_terms=context_missing_terms,
        context_leaked_terms=context_leaked_terms,
        passed=passed,
        model=getattr(response, "model", None),
        request_id=getattr(response, "request_id", None),
        response_id=getattr(response, "response_id", None),
    )


def _acceptance_provider_conversation(
    *,
    service: LongTermMemoryService,
    config: TwinrConfig,
    case: _AcceptanceCase,
) -> tuple[tuple[str, str], ...]:
    """Build a bounded provider conversation focused on memory recall proof.

    The live synthetic-memory acceptance must prove earlier-memory recall and
    conflict resolution without dragging in unrelated graph/world-state
    full-document reads. Compose the low-latency fast-topic lane with the
    tool-safe durable context without graph fallback, and only add an explicit
    conflict-queue block for the cases that actually need it.
    """

    messages: list[tuple[str, str]] = []
    try:
        contract = memory_and_response_contract(config.openai_realtime_language)
    except Exception:
        contract = None
    if contract:
        messages.append(("system", contract))

    seen_system_messages: set[str] = set()
    fast_context = service.build_fast_provider_context(case.query_text)
    for item in fast_context.system_messages():
        text = str(item or "").strip()
        if text and text not in seen_system_messages:
            messages.append(("system", text))
            seen_system_messages.add(text)

    tool_context = service.build_tool_provider_context(
        case.query_text,
        include_graph_fallback=False,
    )
    for item in tool_context.system_messages():
        text = str(item or "").strip()
        if text and text not in seen_system_messages:
            messages.append(("system", text))
            seen_system_messages.add(text)

    if case.include_conflict_queue_context:
        conflict_context = _render_conflict_queue_context(
            service.select_conflict_queue(case.query_text),
        )
        if conflict_context and conflict_context not in seen_system_messages:
            messages.append(("system", conflict_context))
    return tuple(messages)


def _render_conflict_queue_context(queue: tuple[LongTermConflictQueueItemV1, ...]) -> str | None:
    """Render one compact conflict-queue system block for acceptance prompts."""

    if not queue:
        return None
    lines = [
        "twinr_acceptance_conflict_queue_v1",
        "Open memory conflicts relevant to this turn. Use them only when the user is explicitly asking about conflicting memories.",
    ]
    for item in queue:
        lines.append(f"- question: {item.question}")
        lines.append(f"  reason: {item.reason}")
        for option in item.options:
            lines.append(
                "  option: "
                + str(option.summary)
                + f" [memory_id={option.memory_id}; status={option.status}]"
            )
    return "\n".join(lines)


def _assert_ready_result(result: LiveMemoryAcceptanceResult) -> None:
    """Raise when the acceptance result does not prove the required behavior."""

    if result.queue_before_count <= 0:
        raise RuntimeError("No open conflict was visible before confirmation.")
    if result.queue_after_count != 0:
        raise RuntimeError("Conflict queue was not cleared after confirmation.")
    if result.restart_queue_count != 0:
        raise RuntimeError("Conflict queue reappeared after restart.")
    failing = [item.case_id for item in result.case_results if not item.passed]
    if failing:
        raise RuntimeError("Live memory acceptance failed cases: " + ", ".join(failing))


def run_live_memory_acceptance(
    *,
    env_path: str | Path = ".env",
    probe_id: str | None = None,
    write_artifacts: bool = True,
) -> LiveMemoryAcceptanceResult:
    """Run the live synthetic-memory acceptance matrix and persist its evidence."""

    resolved_env_path = Path(env_path).expanduser().resolve(strict=False)
    started_at = _utc_now_iso()
    effective_probe_id = " ".join(
        str(probe_id or f"memory_live_{started_at.replace(':', '').replace('-', '')}").split()
    ).strip()
    base_config = TwinrConfig.from_env(resolved_env_path)
    base_project_root = _normalize_base_project_root(resolved_env_path, base_config)
    runtime_namespace = f"twinr_memory_live_{_safe_namespace_suffix(effective_probe_id)}"

    writer_service: LongTermMemoryService | None = None
    fresh_reader_service: LongTermMemoryService | None = None
    writer_backend: OpenAIBackend | None = None
    fresh_reader_backend: OpenAIBackend | None = None

    result = LiveMemoryAcceptanceResult(
        probe_id=effective_probe_id,
        status="running",
        started_at=started_at,
        finished_at=started_at,
        env_path=str(resolved_env_path),
        base_project_root=str(base_project_root),
        runtime_namespace=runtime_namespace,
    )

    try:
        if not base_config.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for live memory acceptance.")
        if not base_config.chonkydb_base_url or not base_config.chonkydb_api_key:
            raise RuntimeError("ChonkyDB credentials are required for live memory acceptance.")

        with ExitStack() as stack:
            stack.callback(_restore_project_root_override, _push_project_root_override(base_project_root))
            writer_temp_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix=f"{effective_probe_id}_writer_"))
            fresh_reader_temp_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix=f"{effective_probe_id}_reader_"))
            writer_root = Path(writer_temp_dir).resolve(strict=False)
            fresh_reader_root = Path(fresh_reader_temp_dir).resolve(strict=False)
            result = replace(
                result,
                writer_root=str(writer_root),
                fresh_reader_root=str(fresh_reader_root),
            )

            writer_config = _build_isolated_config(
                base_config=base_config,
                base_project_root=base_project_root,
                runtime_root=writer_root,
                remote_namespace=runtime_namespace,
                background_store_turns=False,
            )
            fresh_reader_config = _build_isolated_config(
                base_config=base_config,
                base_project_root=base_project_root,
                runtime_root=fresh_reader_root,
                remote_namespace=runtime_namespace,
                background_store_turns=False,
            )

            writer_service = LongTermMemoryService.from_config(writer_config)
            fresh_reader_service = LongTermMemoryService.from_config(fresh_reader_config)
            writer_service.ensure_remote_ready()
            fresh_reader_service.ensure_remote_ready()
            writer_backend = OpenAIBackend(writer_config)
            fresh_reader_backend = OpenAIBackend(fresh_reader_config)
            _configure_openai_backend_client(writer_backend, timeout_s=_MODEL_TIMEOUT_S, max_retries=_MODEL_MAX_RETRIES)
            _configure_openai_backend_client(fresh_reader_backend, timeout_s=_MODEL_TIMEOUT_S, max_retries=_MODEL_MAX_RETRIES)

            selected_memory_id = _seed_synthetic_memory(writer_service)
            queue_before = writer_service.select_conflict_queue("Welche Marmeladen stehen gerade im Widerspruch?")
            before_cases = tuple(
                _run_case(service=writer_service, config=writer_config, backend=writer_backend, case=case)
                for case in _default_cases()
                if case.phase == "before_resolution"
            )

            writer_service.confirm_memory(memory_id=selected_memory_id)
            queue_after = writer_service.select_conflict_queue("Welche Marmelade ist jetzt bestaetigt?")
            after_resolution_cases = tuple(
                _run_case(service=writer_service, config=writer_config, backend=writer_backend, case=case)
                for case in _default_cases()
                if case.phase == "after_resolution"
            )

            restart_queue = fresh_reader_service.select_conflict_queue("Welche Marmelade ist jetzt bestaetigt?")
            after_restart_cases = tuple(
                _run_case(service=fresh_reader_service, config=fresh_reader_config, backend=fresh_reader_backend, case=case)
                for case in _default_cases()
                if case.phase == "after_restart"
            )

            result = replace(
                result,
                status="ok",
                finished_at=_utc_now_iso(),
                selected_memory_id=selected_memory_id,
                queue_before_count=len(queue_before),
                queue_after_count=len(queue_after),
                restart_queue_count=len(restart_queue),
                case_results=tuple((*before_cases, *after_resolution_cases, *after_restart_cases)),
            )
            _assert_ready_result(result)
    except Exception as exc:
        result = replace(
            result,
            status="failed",
            finished_at=_utc_now_iso(),
            error_message=f"{type(exc).__name__}: {exc}",
        )
    finally:
        _close_openai_backend(writer_backend)
        _close_openai_backend(fresh_reader_backend)
        _shutdown_service(writer_service)
        _shutdown_service(fresh_reader_service)

    if write_artifacts:
        result = write_live_memory_acceptance_artifacts(result, project_root=base_project_root)
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the live memory acceptance runner."""

    parser = argparse.ArgumentParser(description="Run the live synthetic-memory acceptance suite.")
    parser.add_argument("--env-file", default=".env", help="Path to the Twinr env file.")
    parser.add_argument("--probe-id", default=None, help="Optional stable probe id / namespace suffix.")
    parser.add_argument(
        "--no-write-artifacts",
        action="store_true",
        help="Skip writing the rolling ops artifact and per-run report snapshot.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entrypoint and print the structured result JSON."""

    args = _build_arg_parser().parse_args(argv)
    result = run_live_memory_acceptance(
        env_path=args.env_file,
        probe_id=args.probe_id,
        write_artifacts=not args.no_write_artifacts,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.ready else 1


if __name__ == "__main__":  # pragma: no cover - manual CLI execution
    raise SystemExit(main())


__all__ = [
    "LiveMemoryAcceptanceCaseResult",
    "LiveMemoryAcceptanceResult",
    "default_live_memory_acceptance_path",
    "default_live_memory_acceptance_report_dir",
    "run_live_memory_acceptance",
    "write_live_memory_acceptance_artifacts",
]
