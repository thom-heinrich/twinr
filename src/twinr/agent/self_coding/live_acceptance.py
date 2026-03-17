"""Execute the real end-to-end morning-briefing self_coding acceptance flow."""

from __future__ import annotations

import fcntl
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.self_coding.activation import SelfCodingActivationService
from twinr.agent.self_coding.health import SelfCodingHealthService
from twinr.agent.self_coding.runtime import SelfCodingSkillExecutionService, skill_package_from_document
from twinr.agent.self_coding.store import SelfCodingStore
from twinr.agent.self_coding.worker import SelfCodingCompileWorker
from twinr.automations import AutomationStore
from twinr.hardware.audio import WaveAudioPlayer
from twinr.providers.factory import build_streaming_provider_bundle
from twinr.providers.openai import OpenAIBackend

from .contracts import FeasibilityResult, RequirementsDialogueSession
from .status import CompileTarget, FeasibilityOutcome, LearnedSkillStatus, RequirementsDialogueStatus

_FAILURE_STATUSES = frozenset({"error", "failed", "failure", "exception", "crashed", "rejected", "cancelled"})


class SpeechOutput(Protocol):
    """Abstract the final speech delivery used by the acceptance runner."""

    def speak(self, text: str) -> None:
        """Deliver one spoken text."""


@dataclass(slots=True)
class MemorySpeechOutput:
    """Capture spoken acceptance output in memory for tests and dry runs."""

    spoken_texts: tuple[str, ...] = ()
    _buffer: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._buffer = list(self.spoken_texts)

    def speak(self, text: str) -> None:
        self._buffer.append(str(text))
        object.__setattr__(self, "spoken_texts", tuple(self._buffer))


@dataclass(slots=True)
class WaveAudioSpeechOutput:
    """Play acceptance speech through the configured TTS provider and player."""

    tts_provider: Any
    player: WaveAudioPlayer

    def speak(self, text: str) -> None:
        spoken = str(text).strip()
        if not spoken:
            raise RuntimeError("morning briefing acceptance produced empty speech")
        try:
            payload = self.tts_provider.synthesize(spoken)
        except Exception as exc:  # AUDIT-FIX(#11): wrap provider errors with stage-specific context.
            raise RuntimeError("morning briefing acceptance failed during speech synthesis") from exc
        if not payload:
            raise RuntimeError("morning briefing acceptance produced no synthesized speech")
        try:
            self.player.play_wav_bytes(payload)
        except Exception as exc:  # AUDIT-FIX(#11): wrap playback errors with stage-specific context.
            raise RuntimeError("morning briefing acceptance failed during audio playback") from exc


@dataclass(slots=True)
class CountingBackendProxy:
    """Count the backend calls made by the acceptance flow."""

    backend: Any
    search_call_count: int = 0
    summary_call_count: int = 0
    last_summary_text: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)  # AUDIT-FIX(#10): preserve backend compatibility for methods not explicitly counted.

    def search_live_info_with_metadata(self, question: str, *, conversation=None, location_hint=None, date_context=None):
        self.search_call_count += 1
        return self.backend.search_live_info_with_metadata(
            question,
            conversation=conversation,
            location_hint=location_hint,
            date_context=date_context,
        )

    def respond_with_metadata(self, prompt: str, *, instructions=None, allow_web_search=None):
        self.summary_call_count += 1
        result = self.backend.respond_with_metadata(
            prompt,
            instructions=instructions,
            allow_web_search=allow_web_search,
        )
        self.last_summary_text = str(getattr(result, "text", "") or "").strip() or None
        return result


class MorningBriefingAcceptanceOwner:
    """Expose the runtime surface expected by materialized self_coding skills."""

    def __init__(
        self,
        *,
        backend: CountingBackendProxy,
        speech_output: SpeechOutput,
        timezone_name: str,
        presence_session_id: int | None = None,
    ) -> None:
        self.config = SimpleNamespace(local_timezone_name=str(timezone_name).strip() or "Europe/Berlin")
        self.print_backend = backend
        self.agent_provider = backend
        self.runtime = SimpleNamespace(search_provider_conversation_context=lambda: ())
        self._speech_output = speech_output
        self._latest_sensor_observation_facts = {"camera": {"person_visible": True, "count_persons": 1}}
        self._night_mode = False
        self._presence_session_id = presence_session_id if presence_session_id is not None else _generate_presence_session_id()  # AUDIT-FIX(#6): use a unique presence session id per owner/run.
        self.spoken_texts: list[str] = []

    @classmethod
    def with_backend(
        cls,
        backend: CountingBackendProxy,
        *,
        speech_output: SpeechOutput,
        timezone_name: str = "Europe/Berlin",
        presence_session_id: int | None = None,
    ) -> "MorningBriefingAcceptanceOwner":
        """Build a standard owner for the morning-briefing acceptance flow."""

        return cls(
            backend=backend,
            speech_output=speech_output,
            timezone_name=timezone_name,
            presence_session_id=presence_session_id,
        )

    def _current_presence_session_id(self) -> int | None:
        return self._presence_session_id

    def _speak_automation_text(self, entry: object, text: str) -> None:
        del entry
        spoken = str(text).strip()
        if not spoken:
            raise RuntimeError("morning briefing acceptance produced empty speech")
        self._speech_output.speak(spoken)  # AUDIT-FIX(#9): only count speech after the output backend actually succeeds.
        self.spoken_texts.append(spoken)


@dataclass(frozen=True, slots=True)
class MorningBriefingAcceptanceResult:
    """Summarize one compiled and executed morning-briefing acceptance run."""

    job_id: str
    job_status: str
    skill_id: str
    version: int
    activation_status: str
    refresh_status: str
    delivery_status: str
    delivery_delivered: bool
    search_call_count: int
    summary_call_count: int
    spoken_count: int
    last_summary_text: str | None


def build_morning_briefing_ready_session(
    *,
    session_id: str = "dialogue_live_morning_briefing",
) -> RequirementsDialogueSession:
    """Return the canonical ready-for-compile minimum morning-briefing skill."""

    return RequirementsDialogueSession(
        session_id=session_id,
        request_summary="Every day at 08:00 research three morning topics, write one short German abstract, and when I enter the room in the morning read that abstract aloud once.",
        skill_name="Morning Briefing",
        action="Research three morning topics at 08:00, write one short German abstract, and read it aloud when I enter the room in the morning.",
        capabilities=("web_search", "llm_call", "memory", "speaker", "camera", "scheduler", "safety"),
        feasibility=FeasibilityResult(
            outcome=FeasibilityOutcome.YELLOW,
            summary="This request needs the skill-package path.",
            suggested_target=CompileTarget.SKILL_PACKAGE,
        ),
        status=RequirementsDialogueStatus.READY_FOR_COMPILE,
        trigger_mode="push",
        trigger_conditions=("camera_person_visible", "daily_0800"),
        scope={
            "channel": "voice",
            "time_of_day": "08:00",
            "query_count": 3,
        },
        constraints=("read_once_per_morning", "quiet_at_night"),
    )


def run_morning_briefing_acceptance(
    *,
    store: SelfCodingStore,
    automation_store: AutomationStore,
    compile_worker: SelfCodingCompileWorker,
    activation_service: SelfCodingActivationService,
    runtime_service: SelfCodingSkillExecutionService,
    owner: MorningBriefingAcceptanceOwner,
    session: RequirementsDialogueSession | None = None,
    refresh_now: datetime | None = None,
    delivery_now: datetime | None = None,
    live_e2e_environment: str | None = None,
    live_e2e_model: str = "gpt-5-codex",
    live_e2e_reasoning_effort: str = "high",
) -> MorningBriefingAcceptanceResult:
    """Compile, activate, execute, and optionally record the morning briefing."""

    del automation_store
    selected_session = build_morning_briefing_ready_session() if session is None else session
    health = SelfCodingHealthService(store=store)
    started_at = datetime.now(UTC)
    initial_search_call_count = owner.print_backend.search_call_count
    initial_summary_call_count = owner.print_backend.summary_call_count
    initial_spoken_count = len(owner.spoken_texts)
    try:
        job = compile_worker.ensure_job_for_session(selected_session)  # AUDIT-FIX(#1): acceptance must materialize the requested session instead of reusing any existing active skill.
        completed = compile_worker.run_job(job.job_id)
        completed_job_id = str(getattr(completed, "job_id", "") or "").strip()
        if not completed_job_id:  # AUDIT-FIX(#3): require a concrete compile job identifier before activation.
            raise RuntimeError("morning briefing compile job returned no job_id")
        job_status = _status_value(completed, "compile job")
        if _is_failure_status(job_status):  # AUDIT-FIX(#3): fail fast on an explicit compile failure before activation.
            raise RuntimeError(f"morning briefing compile job failed with status {job_status!r}")

        active = activation_service.confirm_activation(job_id=completed_job_id, confirmed=True)
        if active is None:  # AUDIT-FIX(#3): guard against activation confirmation returning no activation document.
            raise RuntimeError("morning briefing activation confirmation returned no activation document")

        activation_status = _status_value(active, "activation")
        expected_active_status = str(getattr(LearnedSkillStatus.ACTIVE, "value", LearnedSkillStatus.ACTIVE)).lower()
        if activation_status.lower() != expected_active_status:  # AUDIT-FIX(#3): verify the activation really became active before runtime execution.
            raise RuntimeError(f"morning briefing activation is not active: {activation_status!r}")

        active_skill_id = str(getattr(active, "skill_id", "") or "").strip()
        if not active_skill_id:  # AUDIT-FIX(#3): require a concrete skill id before runtime execution.
            raise RuntimeError("morning briefing activation returned no skill_id")
        active_version = getattr(active, "version", None)
        if not isinstance(active_version, int):  # AUDIT-FIX(#3): require an integer activation version before runtime execution.
            raise RuntimeError(f"morning briefing activation returned invalid version: {active_version!r}")
        artifact_id = str(getattr(active, "artifact_id", "") or "").strip()
        if not artifact_id:  # AUDIT-FIX(#3): require a concrete artifact id before artifact loading.
            raise RuntimeError("morning briefing activation returned no artifact_id")

        artifact_text = store.read_text_artifact(artifact_id)
        if not str(artifact_text).strip():  # AUDIT-FIX(#3): reject empty artifacts before package parsing produces opaque errors.
            raise RuntimeError(f"morning briefing artifact {artifact_id!r} is empty")

        package = skill_package_from_document(artifact_text)
        scheduled_trigger = _first_trigger(package, "scheduled_triggers", "scheduled")  # AUDIT-FIX(#2): validate required triggers before indexing them.
        sensor_trigger = _first_trigger(package, "sensor_triggers", "sensor")  # AUDIT-FIX(#2): validate required triggers before indexing them.

        effective_refresh_now = _normalize_utc_datetime(refresh_now, field_name="refresh_now")
        effective_delivery_now = _normalize_utc_datetime(
            delivery_now,
            field_name="delivery_now",
            default=effective_refresh_now,
        )
        if effective_delivery_now < effective_refresh_now:  # AUDIT-FIX(#7): keep delivery ordering deterministic even for caller-supplied timestamps.
            raise ValueError("delivery_now must be greater than or equal to refresh_now")

        refresh = runtime_service.execute_scheduled(
            owner,
            skill_id=active_skill_id,
            version=active_version,
            trigger_id=scheduled_trigger.trigger_id,
            now=effective_refresh_now,
        )
        refresh_status = _mapping_status(refresh, stage_name="scheduled refresh")
        if _is_failure_status(refresh_status):  # AUDIT-FIX(#4): never proceed to delivery after a failed refresh stage.
            raise RuntimeError(f"morning briefing refresh failed with status {refresh_status!r}")

        delivery = runtime_service.execute_sensor_event(
            owner,
            skill_id=active_skill_id,
            version=active_version,
            trigger_id=sensor_trigger.trigger_id,
            event_name="camera.person_visible",
            now=effective_delivery_now,
        )
        delivery_status = _mapping_status(delivery, stage_name="sensor delivery")

        search_call_count = max(0, owner.print_backend.search_call_count - initial_search_call_count)  # AUDIT-FIX(#8): report per-run metrics instead of cumulative lifetime counters.
        summary_call_count = max(0, owner.print_backend.summary_call_count - initial_summary_call_count)  # AUDIT-FIX(#8): report per-run metrics instead of cumulative lifetime counters.
        spoken_count = max(0, len(owner.spoken_texts) - initial_spoken_count)  # AUDIT-FIX(#8): report per-run spoken output only.
        delivery_delivered = bool(delivery.get("delivered", False))
        last_summary_text = owner.print_backend.last_summary_text if summary_call_count > 0 else None  # AUDIT-FIX(#8): avoid leaking stale summary text from earlier runs.

        if _is_failure_status(delivery_status):  # AUDIT-FIX(#5): reject explicit delivery failures instead of reporting a passed acceptance.
            raise RuntimeError(f"morning briefing delivery failed with status {delivery_status!r}")
        if not delivery_delivered:  # AUDIT-FIX(#5): acceptance must confirm that delivery actually happened.
            raise RuntimeError("morning briefing delivery did not deliver speech")
        if search_call_count < 1:  # AUDIT-FIX(#5): require at least one live-info call for a real end-to-end acceptance run.
            raise RuntimeError("morning briefing acceptance observed no live-search backend calls")
        if summary_call_count < 1:  # AUDIT-FIX(#5): require at least one summarization call for a real end-to-end acceptance run.
            raise RuntimeError("morning briefing acceptance observed no summary backend calls")
        if spoken_count < 1:  # AUDIT-FIX(#5): require concrete spoken output instead of a silent logical success.
            raise RuntimeError("morning briefing acceptance observed no spoken output")

        duration_seconds = max(0.0, (datetime.now(UTC) - started_at).total_seconds())
        if live_e2e_environment:
            _safe_record_live_e2e_status(  # AUDIT-FIX(#12): health telemetry is best-effort and must never change the acceptance outcome.
                health,
                suite_id="morning_briefing",
                environment=live_e2e_environment,
                status="passed",
                duration_seconds=duration_seconds,
                model=live_e2e_model,
                reasoning_effort=live_e2e_reasoning_effort,
                details=(
                    "Morning briefing acceptance passed "
                    f"(job_status={job_status}, activation_status={activation_status}, "
                    f"refresh_status={refresh_status}, delivery_status={delivery_status})."
                ),
            )
        return MorningBriefingAcceptanceResult(
            job_id=completed_job_id,
            job_status=job_status,
            skill_id=active_skill_id,
            version=active_version,
            activation_status=activation_status,
            refresh_status=refresh_status,
            delivery_status=delivery_status,
            delivery_delivered=delivery_delivered,
            search_call_count=search_call_count,
            summary_call_count=summary_call_count,
            spoken_count=spoken_count,
            last_summary_text=last_summary_text,
        )
    except Exception as exc:
        if live_e2e_environment:
            duration_seconds = max(0.0, (datetime.now(UTC) - started_at).total_seconds())
            _safe_record_live_e2e_status(  # AUDIT-FIX(#12): preserve the original failure instead of masking it with telemetry write errors.
                health,
                suite_id="morning_briefing",
                environment=live_e2e_environment,
                status="failed",
                duration_seconds=duration_seconds,
                model=live_e2e_model,
                reasoning_effort=live_e2e_reasoning_effort,
                details=f"Morning briefing acceptance failed: {type(exc).__name__}: {exc}",
            )
        raise


def _load_active_activation(*, store: SelfCodingStore, skill_id: str):
    active_activations = [activation for activation in store.list_activations(skill_id=skill_id) if activation.status == LearnedSkillStatus.ACTIVE]
    if not active_activations:
        return None
    if len(active_activations) > 1:  # AUDIT-FIX(#1): detect ambiguous active state instead of returning whichever activation happens to come first.
        versions = ", ".join(str(getattr(activation, "version", "?")) for activation in active_activations)
        raise RuntimeError(f"multiple active activations found for skill {skill_id!r}: {versions}")
    return active_activations[0]


def run_live_morning_briefing_acceptance(
    *,
    project_root: str | Path,
    env_file: str | Path,
    speak_out_loud: bool,
    live_e2e_environment: str,
) -> MorningBriefingAcceptanceResult:
    """Run the full morning-briefing acceptance path on a real configured machine."""

    root = _resolve_existing_path(project_root, field_name="project_root", expect_directory=True)  # AUDIT-FIX(#13): resolve and validate filesystem inputs before they reach file-backed services.
    env_path = _resolve_existing_path(env_file, field_name="env_file", expect_directory=False)  # AUDIT-FIX(#13): resolve and validate filesystem inputs before they reach file-backed services.
    state_dir = root / "state"
    if not state_dir.is_dir():  # AUDIT-FIX(#13): fail with a clear message when the required state directory is missing.
        raise FileNotFoundError(f"project_root state directory does not exist: {state_dir}")
    lock_path = state_dir / "morning_briefing_acceptance.lock"

    with _exclusive_file_lock(lock_path):  # AUDIT-FIX(#14): serialize live acceptance runs to reduce file-backed state races.
        config = TwinrConfig.from_env(env_path)
        store = SelfCodingStore.from_project_root(root)
        automation_store = AutomationStore(state_dir / "automations.json", timezone_name=config.local_timezone_name)
        activation = SelfCodingActivationService(store=store, automation_store=automation_store)
        health = SelfCodingHealthService(store=store)
        runtime = SelfCodingSkillExecutionService(store=store, health_service=health)
        worker = SelfCodingCompileWorker(store=store)

        backend: OpenAIBackend | None = None
        provider_bundle: Any | None = None
        player: WaveAudioPlayer | None = None
        try:
            backend = OpenAIBackend(config=config)
            provider_bundle = build_streaming_provider_bundle(config, support_backend=backend)
            speech_output: SpeechOutput
            if speak_out_loud:
                player = WaveAudioPlayer.from_config(config)
                speech_output = WaveAudioSpeechOutput(
                    tts_provider=provider_bundle.tts,
                    player=player,
                )
            else:
                speech_output = MemorySpeechOutput()
            run_now = datetime.now(UTC)  # AUDIT-FIX(#7): capture one timestamp and reuse it for refresh and delivery.
            owner = MorningBriefingAcceptanceOwner.with_backend(
                CountingBackendProxy(backend),
                speech_output=speech_output,
                timezone_name=config.local_timezone_name,
                presence_session_id=_generate_presence_session_id(run_now),  # AUDIT-FIX(#6): unique per-run presence identity prevents read-once logic from reusing session id 1.
            )
            return run_morning_briefing_acceptance(
                store=store,
                automation_store=automation_store,
                compile_worker=worker,
                activation_service=activation,
                runtime_service=runtime,
                owner=owner,
                session=build_morning_briefing_ready_session(session_id="dialogue_live_pi_morning_briefing"),
                refresh_now=run_now,
                delivery_now=run_now,
                live_e2e_environment=live_e2e_environment,
            )
        finally:
            _best_effort_close(getattr(provider_bundle, "tts", None))  # AUDIT-FIX(#15): release provider-side TTS resources after live runs when supported.
            _best_effort_close(player)  # AUDIT-FIX(#15): release hardware/audio resources after live runs when the implementation exposes close().
            _best_effort_close(provider_bundle)  # AUDIT-FIX(#15): release provider resources after live runs when supported.
            _best_effort_close(backend)  # AUDIT-FIX(#15): release backend resources after live runs when supported.


def _best_effort_close(resource: object | None) -> None:
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if callable(close):
        with suppress(Exception):
            close()


@contextmanager
def _exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            with suppress(OSError):
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _resolve_existing_path(path_value: str | Path, *, field_name: str, expect_directory: bool) -> Path:
    resolved = Path(path_value).expanduser().resolve(strict=True)
    if expect_directory and not resolved.is_dir():
        raise NotADirectoryError(f"{field_name} must point to an existing directory: {resolved}")
    if not expect_directory and not resolved.is_file():
        raise FileNotFoundError(f"{field_name} must point to an existing file: {resolved}")
    return resolved


def _generate_presence_session_id(now: datetime | None = None) -> int:
    reference_now = _normalize_utc_datetime(now, field_name="presence_session_now") if now is not None else datetime.now(UTC)
    return int(reference_now.timestamp() * 1_000_000)


def _normalize_utc_datetime(value: datetime | None, *, field_name: str, default: datetime | None = None) -> datetime:
    if value is None:
        value = default if default is not None else datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _status_value(document: object, label: str) -> str:
    status = getattr(document, "status", None)
    raw_value = getattr(status, "value", status)
    normalized = str(raw_value or "").strip()
    if not normalized:
        raise RuntimeError(f"{label} produced no status")
    return normalized


def _mapping_status(result: object, *, stage_name: str) -> str:
    if not isinstance(result, Mapping):
        raise RuntimeError(f"{stage_name} returned an unsupported result type: {type(result).__name__}")
    status = str(result.get("status", "") or "").strip()
    if not status:
        raise RuntimeError(f"{stage_name} returned no status")
    return status


def _is_failure_status(status: str) -> bool:
    return status.strip().lower() in _FAILURE_STATUSES


def _first_trigger(package: object, attribute_name: str, label: str) -> Any:
    triggers = getattr(package, attribute_name, None)
    if not triggers:
        raise RuntimeError(f"compiled morning briefing package has no {label} triggers")
    trigger = triggers[0]
    trigger_id = str(getattr(trigger, "trigger_id", "") or "").strip()
    if not trigger_id:
        raise RuntimeError(f"compiled morning briefing {label} trigger is missing a trigger_id")
    return trigger


def _safe_record_live_e2e_status(health: SelfCodingHealthService, **kwargs: Any) -> None:
    with suppress(Exception):
        health.record_live_e2e_status(**kwargs)


__all__ = [
    "CountingBackendProxy",
    "MemorySpeechOutput",
    "MorningBriefingAcceptanceOwner",
    "MorningBriefingAcceptanceResult",
    "WaveAudioSpeechOutput",
    "build_morning_briefing_ready_session",
    "run_live_morning_briefing_acceptance",
    "run_morning_briefing_acceptance",
]