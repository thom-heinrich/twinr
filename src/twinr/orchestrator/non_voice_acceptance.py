"""Run a deterministic non-voice E2E acceptance matrix for Twinr.

This module proves the text-only runtime through three authoritative paths:
one short direct question, one live web/tool question, and the long-term
memory acceptance matrix. It persists a compact rolling ops artifact plus a
per-run report so operators can see exactly which stage failed and whether the
process terminated cleanly instead of inferring health from ad-hoc shell logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Callable

from twinr.agent.base_agent import TwinrRuntime
from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.evaluation.live_memory_acceptance import (
    LiveMemoryAcceptanceResult,
    run_live_memory_acceptance,
)
from twinr.memory.longterm.evaluation.live_midterm_acceptance import _normalize_base_project_root
from twinr.orchestrator.probe_turn import (
    OrchestratorProbeStageResult,
    run_orchestrator_probe_turn,
)
from twinr.providers.openai import OpenAIBackend


_SCHEMA_VERSION = 1
_ACCEPT_KIND = "non_voice_e2e_acceptance"
_OPS_ARTIFACT_NAME = "non_voice_e2e_acceptance.json"
_REPORT_DIR_NAME = "non_voice_e2e_acceptance"
_DEFAULT_DIRECT_PROMPT = "Wie geht's dir?"
_DEFAULT_TOOL_PROMPT = (
    "Was sind heute im Web die wichtigsten Nachrichten für Deutschland? "
    "Bitte nenne drei kurze Punkte und nutze Websuche."
)


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in stable ISO-8601 form."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_text(value: object | None) -> str:
    """Normalize arbitrary values into a stable single-line string."""

    return " ".join(str(value or "").split()).strip()


def _coerce_optional_text(value: object | None) -> str | None:
    """Normalize optional text values and collapse blanks to ``None``."""

    text = _coerce_text(value)
    return text or None


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


def default_non_voice_acceptance_path(project_root: str | Path) -> Path:
    """Return the rolling ops artifact path for non-voice E2E acceptance."""

    return Path(project_root).resolve(strict=False) / "artifacts" / "stores" / "ops" / _OPS_ARTIFACT_NAME


def default_non_voice_acceptance_report_dir(project_root: str | Path) -> Path:
    """Return the per-run report directory for non-voice E2E acceptance."""

    return Path(project_root).resolve(strict=False) / "artifacts" / "reports" / _REPORT_DIR_NAME


@dataclass(frozen=True, slots=True)
class NonVoiceAcceptanceCaseResult:
    """Describe one direct or tool text-turn acceptance case."""

    case_id: str
    mode: str
    prompt: str
    status: str
    response_text: str | None = None
    error_message: str | None = None
    used_web_search: bool | None = None
    rounds: int | None = None
    model: str | None = None
    request_id: str | None = None
    response_id: str | None = None
    tool_handler_count: int = 0
    stage_results: tuple[OrchestratorProbeStageResult, ...] = ()
    probe_lines: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _coerce_text(self.case_id))
        object.__setattr__(self, "mode", _coerce_text(self.mode))
        object.__setattr__(self, "prompt", _coerce_text(self.prompt))
        object.__setattr__(self, "status", _coerce_text(self.status).lower() or "unknown")
        object.__setattr__(self, "response_text", _coerce_optional_text(self.response_text))
        object.__setattr__(self, "error_message", _coerce_optional_text(self.error_message))
        object.__setattr__(self, "model", _coerce_optional_text(self.model))
        object.__setattr__(self, "request_id", _coerce_optional_text(self.request_id))
        object.__setattr__(self, "response_id", _coerce_optional_text(self.response_id))
        object.__setattr__(self, "tool_handler_count", max(0, int(self.tool_handler_count)))
        object.__setattr__(
            self,
            "stage_results",
            tuple(
                item if isinstance(item, OrchestratorProbeStageResult) else OrchestratorProbeStageResult(**dict(item))
                for item in self.stage_results
            ),
        )
        object.__setattr__(self, "probe_lines", tuple(_coerce_text(item) for item in self.probe_lines if _coerce_text(item)))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "case_id": self.case_id,
            "mode": self.mode,
            "prompt": self.prompt,
            "status": self.status,
            "response_text": self.response_text,
            "error_message": self.error_message,
            "used_web_search": self.used_web_search,
            "rounds": self.rounds,
            "model": self.model,
            "request_id": self.request_id,
            "response_id": self.response_id,
            "tool_handler_count": self.tool_handler_count,
            "stage_results": [asdict(item) for item in self.stage_results],
            "probe_lines": list(self.probe_lines),
        }


@dataclass(frozen=True, slots=True)
class NonVoiceAcceptanceResult:
    """Describe one complete non-voice direct/tool/memory acceptance run."""

    run_id: str
    status: str
    started_at: str
    finished_at: str
    env_path: str
    base_project_root: str
    direct_case: NonVoiceAcceptanceCaseResult
    tool_case: NonVoiceAcceptanceCaseResult
    memory_result: LiveMemoryAcceptanceResult
    artifact_path: str | None = None
    report_path: str | None = None
    accept_kind: str = field(default=_ACCEPT_KIND, init=False)
    schema_version: int = field(default=_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _coerce_text(self.run_id))
        object.__setattr__(self, "status", _coerce_text(self.status).lower() or "unknown")
        object.__setattr__(self, "started_at", _coerce_text(self.started_at))
        object.__setattr__(self, "finished_at", _coerce_text(self.finished_at))
        object.__setattr__(self, "env_path", _coerce_text(self.env_path))
        object.__setattr__(self, "base_project_root", _coerce_text(self.base_project_root))
        object.__setattr__(self, "artifact_path", _coerce_optional_text(self.artifact_path))
        object.__setattr__(self, "report_path", _coerce_optional_text(self.report_path))
        if not isinstance(self.direct_case, NonVoiceAcceptanceCaseResult):
            object.__setattr__(self, "direct_case", NonVoiceAcceptanceCaseResult(**dict(self.direct_case)))
        if not isinstance(self.tool_case, NonVoiceAcceptanceCaseResult):
            object.__setattr__(self, "tool_case", NonVoiceAcceptanceCaseResult(**dict(self.tool_case)))
        if not isinstance(self.memory_result, LiveMemoryAcceptanceResult):
            object.__setattr__(self, "memory_result", LiveMemoryAcceptanceResult(**dict(self.memory_result)))

    @property
    def ready(self) -> bool:
        """Return whether all three acceptance paths completed successfully."""

        return (
            self.status == "ok"
            and self.direct_case.status == "ok"
            and self.tool_case.status == "ok"
            and self.memory_result.ready
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "env_path": self.env_path,
            "base_project_root": self.base_project_root,
            "artifact_path": self.artifact_path,
            "report_path": self.report_path,
            "accept_kind": self.accept_kind,
            "schema_version": self.schema_version,
            "ready": self.ready,
            "direct_case": self.direct_case.to_dict(),
            "tool_case": self.tool_case.to_dict(),
            "memory_result": self.memory_result.to_dict(),
        }


def write_non_voice_acceptance_artifacts(
    result: NonVoiceAcceptanceResult,
    *,
    project_root: str | Path,
) -> NonVoiceAcceptanceResult:
    """Persist the rolling ops artifact and a per-run report snapshot."""

    artifact_path = default_non_voice_acceptance_path(project_root)
    report_dir = default_non_voice_acceptance_report_dir(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{result.run_id}.json"
    payload = result.to_dict()
    _atomic_write_json(artifact_path, payload)
    _atomic_write_json(report_path, payload)
    return NonVoiceAcceptanceResult(
        run_id=result.run_id,
        status=result.status,
        started_at=result.started_at,
        finished_at=result.finished_at,
        env_path=result.env_path,
        base_project_root=result.base_project_root,
        direct_case=result.direct_case,
        tool_case=result.tool_case,
        memory_result=result.memory_result,
        artifact_path=str(artifact_path),
        report_path=str(report_path),
    )


def _probe_case(
    *,
    case_id: str,
    mode: str,
    prompt: str,
    config: TwinrConfig,
    runtime: TwinrRuntime,
    backend: OpenAIBackend,
    emit_line: Callable[[str], None],
) -> NonVoiceAcceptanceCaseResult:
    """Run one text probe case and capture its streamed diagnostic evidence."""

    lines: list[str] = []

    def _emit_case_line(line: str) -> None:
        lines.append(str(line))
        emit_line(f"{case_id}:{line}")

    try:
        outcome = run_orchestrator_probe_turn(
            config=config,
            runtime=runtime,
            backend=backend,
            prompt=prompt,
            emit_line=_emit_case_line,
        )
        return NonVoiceAcceptanceCaseResult(
            case_id=case_id,
            mode=mode,
            prompt=prompt,
            status="ok",
            response_text=outcome.result.text,
            used_web_search=outcome.result.used_web_search,
            rounds=outcome.result.rounds,
            model=outcome.result.model,
            request_id=outcome.result.request_id,
            response_id=outcome.result.response_id,
            tool_handler_count=outcome.tool_handler_count,
            stage_results=outcome.stage_results,
            probe_lines=tuple(lines),
        )
    except Exception as exc:
        return NonVoiceAcceptanceCaseResult(
            case_id=case_id,
            mode=mode,
            prompt=prompt,
            status="failed",
            error_message=f"{type(exc).__name__}: {exc}",
            probe_lines=tuple(lines),
        )


def run_non_voice_acceptance(
    *,
    env_path: str | Path = ".env",
    direct_prompt: str = _DEFAULT_DIRECT_PROMPT,
    tool_prompt: str = _DEFAULT_TOOL_PROMPT,
    write_artifacts: bool = True,
    emit_line: Callable[[str], None] = print,
) -> NonVoiceAcceptanceResult:
    """Run the authoritative non-voice direct/tool/memory acceptance matrix."""

    resolved_env_path = Path(env_path).expanduser().resolve(strict=False)
    started_at = _utc_now_iso()
    run_id = f"non_voice_{started_at.replace(':', '').replace('-', '')}"
    config = TwinrConfig.from_env(resolved_env_path)
    base_project_root = _normalize_base_project_root(resolved_env_path, config)
    runtime: TwinrRuntime | None = None
    backend: OpenAIBackend | None = None

    direct_case = NonVoiceAcceptanceCaseResult(
        case_id="direct_short",
        mode="direct",
        prompt=direct_prompt,
        status="failed",
        error_message="Not run",
    )
    tool_case = NonVoiceAcceptanceCaseResult(
        case_id="tool_web_news",
        mode="tool",
        prompt=tool_prompt,
        status="failed",
        error_message="Not run",
    )
    memory_result = LiveMemoryAcceptanceResult(
        probe_id=f"{run_id}_memory",
        status="failed",
        started_at=started_at,
        finished_at=started_at,
        env_path=str(resolved_env_path),
        base_project_root=str(base_project_root),
        runtime_namespace=f"{run_id}_memory_namespace",
        error_message="Not run",
    )
    runner_error_message: str | None = None

    try:
        runtime = TwinrRuntime(config=config)
        backend = OpenAIBackend(config)
        direct_case = _probe_case(
            case_id="direct_short",
            mode="direct",
            prompt=direct_prompt,
            config=config,
            runtime=runtime,
            backend=backend,
            emit_line=emit_line,
        )
        tool_case = _probe_case(
            case_id="tool_web_news",
            mode="tool",
            prompt=tool_prompt,
            config=config,
            runtime=runtime,
            backend=backend,
            emit_line=emit_line,
        )
        memory_result = run_live_memory_acceptance(env_path=resolved_env_path, write_artifacts=write_artifacts)
    except Exception as exc:
        runner_error_message = f"{type(exc).__name__}: {exc}"
        if direct_case.error_message == "Not run":
            direct_case = NonVoiceAcceptanceCaseResult(
                case_id=direct_case.case_id,
                mode=direct_case.mode,
                prompt=direct_case.prompt,
                status="failed",
                error_message=runner_error_message,
            )
        if tool_case.error_message == "Not run":
            tool_case = NonVoiceAcceptanceCaseResult(
                case_id=tool_case.case_id,
                mode=tool_case.mode,
                prompt=tool_case.prompt,
                status="failed",
                error_message=runner_error_message,
            )
        if memory_result.error_message == "Not run":
            memory_result = LiveMemoryAcceptanceResult(
                probe_id=memory_result.probe_id,
                status="failed",
                started_at=memory_result.started_at,
                finished_at=_utc_now_iso(),
                env_path=memory_result.env_path,
                base_project_root=memory_result.base_project_root,
                runtime_namespace=memory_result.runtime_namespace,
                error_message=runner_error_message,
            )
    finally:
        if backend is not None:
            close_backend = getattr(backend, "close", None)
            if callable(close_backend):
                try:
                    close_backend()  # pylint: disable=not-callable
                except Exception:
                    emit_line("non_voice_acceptance_cleanup=backend_close_failed")
        if runtime is not None:
            try:
                runtime.shutdown(timeout_s=1.0)
            except Exception:
                emit_line("non_voice_acceptance_cleanup=runtime_shutdown_failed")

    result = NonVoiceAcceptanceResult(
        run_id=run_id,
        status=(
            "ok"
            if direct_case.status == "ok" and tool_case.status == "ok" and memory_result.ready
            else "failed"
        ),
        started_at=started_at,
        finished_at=_utc_now_iso(),
        env_path=str(resolved_env_path),
        base_project_root=str(base_project_root),
        direct_case=direct_case,
        tool_case=tool_case,
        memory_result=memory_result,
    )
    if write_artifacts:
        result = write_non_voice_acceptance_artifacts(result, project_root=base_project_root)
    return result


__all__ = [
    "NonVoiceAcceptanceCaseResult",
    "NonVoiceAcceptanceResult",
    "default_non_voice_acceptance_path",
    "default_non_voice_acceptance_report_dir",
    "run_non_voice_acceptance",
    "write_non_voice_acceptance_artifacts",
]
