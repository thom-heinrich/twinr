from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory, mkdtemp
from typing import Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.context_store import (
    ManagedContextFileStore,
    PersistentMemoryMarkdownStore,
    PromptContextStore,
)
from twinr.memory.longterm.core.models import (
    LongTermConversationTurn,
    LongTermMemoryContext,
    LongTermMultimodalEvidence,
)
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.memory.query_normalization import LongTermQueryProfile


_FIXED_SEED_TARGET = 500
_FIXED_CASE_TARGET = 50
_FALLBACK_TIMEZONE_NAME = "Europe/Berlin"


@dataclass(frozen=True, slots=True)
class MultimodalEvalCase:
    case_id: str
    category: str
    query_text: str
    canonical_query_text: str
    expected_durable_contains: tuple[str, ...] = ()
    expected_episodic_contains: tuple[str, ...] = ()
    expected_durable_absent: tuple[str, ...] = ()
    expected_episodic_absent: tuple[str, ...] = ()
    expect_durable_context: bool | None = None
    expect_episodic_context: bool | None = None


@dataclass(frozen=True, slots=True)
class MultimodalEvalCaseResult:
    case_id: str
    category: str
    passed: bool
    durable_context_present: bool
    episodic_context_present: bool
    missing_durable: tuple[str, ...]
    missing_episodic: tuple[str, ...]
    present_forbidden_durable: tuple[str, ...]
    present_forbidden_episodic: tuple[str, ...]
    error: str | None = None  # AUDIT-FIX(#3): Preserve per-case retrieval failures without aborting the full evaluation run.


@dataclass(frozen=True, slots=True)
class MultimodalEvalSeedStats:
    multimodal_events: int
    episodic_turns: int
    consolidated_object_count: int
    episodic_entry_count: int

    @property
    def total_seed_entries(self) -> int:
        return self.multimodal_events + self.episodic_turns


@dataclass(frozen=True, slots=True)
class MultimodalEvalSummary:
    total_cases: int
    passed_cases: int
    category_case_counts: dict[str, int]
    category_pass_counts: dict[str, int]

    @property
    def accuracy(self) -> float:
        if self.total_cases <= 0:
            return 0.0
        return self.passed_cases / self.total_cases


@dataclass(frozen=True, slots=True)
class MultimodalEvalResult:
    seed_stats: MultimodalEvalSeedStats
    summary: MultimodalEvalSummary
    cases: tuple[MultimodalEvalCaseResult, ...]
    temp_root: str
    object_store_path: str
    memory_path: str


@dataclass(frozen=True, slots=True)
class _TargetEpisode:
    transcript: str
    response: str
    occurred_at: datetime


@dataclass(slots=True)
class _StaticQueryRewriter:
    canonical_queries: Mapping[str, str]

    def profile(self, query_text: str | None) -> LongTermQueryProfile:
        original = " ".join(str(query_text or "").split()).strip()
        return LongTermQueryProfile.from_text(
            original,
            canonical_english_text=self.canonical_queries.get(original),
        )


def run_multimodal_longterm_eval(
    *,
    seed_target: int = _FIXED_SEED_TARGET,
    case_target: int = _FIXED_CASE_TARGET,
    project_root: str | Path | None = None,
) -> MultimodalEvalResult:
    if seed_target != _FIXED_SEED_TARGET:
        raise ValueError(
            f"seed_target is fixed to {_FIXED_SEED_TARGET} for the current multimodal eval."
        )
    if case_target != _FIXED_CASE_TARGET:
        raise ValueError(
            f"case_target is fixed to {_FIXED_CASE_TARGET} for the current multimodal eval."
        )

    base_dir = _resolve_project_root(project_root)
    with TemporaryDirectory(  # AUDIT-FIX(#1,#4): Use a unique workspace under a validated base directory; snapshot results before cleanup.
        dir=str(base_dir) if base_dir is not None else None,
        prefix="twinr-multimodal-eval-work-",
    ) as temp_dir:
        workspace_root = Path(temp_dir).resolve(strict=True)
        result = _run_multimodal_longterm_eval_in_root(
            workspace_root=workspace_root,
            seed_target=seed_target,
            case_target=case_target,
        )
        return _snapshot_eval_result(
            workspace_root=workspace_root,
            result=result,
            base_dir=base_dir,
        )


def _resolve_project_root(project_root: str | Path | None) -> Path | None:
    if project_root is None:
        return None

    candidate = Path(project_root).expanduser()
    if _path_has_existing_symlink_component(candidate):  # AUDIT-FIX(#4): Reject symlinked base paths to prevent writes outside the intended tree.
        raise ValueError("project_root and its existing parent directories must not be symlinks.")
    if candidate.exists() and not candidate.is_dir():
        raise ValueError("project_root must point to a directory.")
    candidate.mkdir(parents=True, exist_ok=True)  # AUDIT-FIX(#4): Create a missing base directory instead of failing later inside tempfile.
    return candidate.resolve(strict=True)


def _path_has_existing_symlink_component(path: Path) -> bool:
    current = path
    while True:
        if current.is_symlink():
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _run_multimodal_longterm_eval_in_root(
    *,
    workspace_root: Path,
    seed_target: int,
    case_target: int,
) -> MultimodalEvalResult:
    state_dir = workspace_root / "state"
    personality_dir = workspace_root / "personality"
    state_dir.mkdir(parents=True, exist_ok=True)  # AUDIT-FIX(#2): Ensure file-backed state parents exist before store/service initialization.
    personality_dir.mkdir(parents=True, exist_ok=True)

    config = TwinrConfig(
        project_root=str(workspace_root),
        personality_dir="personality",
        memory_markdown_path=str(state_dir / "MEMORY.md"),
        long_term_memory_enabled=True,
        long_term_memory_background_store_turns=False,
        long_term_memory_recall_limit=6,
        long_term_memory_path=str(state_dir / "chonkydb"),
        openai_realtime_language="de",
        openai_web_search_timezone="Europe/Berlin",
    )
    prompt_context_store = PromptContextStore(
        memory_store=PersistentMemoryMarkdownStore(config.memory_markdown_path, max_entries=2048),
        user_store=ManagedContextFileStore(
            personality_dir / "USER.md",
            section_title="Twinr managed user updates",
        ),
        personality_store=ManagedContextFileStore(
            personality_dir / "PERSONALITY.md",
            section_title="Twinr managed personality updates",
        ),
    )

    service: LongTermMemoryService | None = None
    active_exception: BaseException | None = None
    try:
        service = LongTermMemoryService.from_config(
            config,
            prompt_context_store=prompt_context_store,
        )
        seed_stats = _seed_multimodal_store(service)
        if seed_stats.total_seed_entries != seed_target:
            raise AssertionError(  # AUDIT-FIX(#7): Fail fast when fixture drift changes the fixed 500-entry seed contract.
                f"Expected {seed_target} total seed entries, got {seed_stats.total_seed_entries}."
            )

        cases = _build_multimodal_eval_cases()
        if len(cases) != case_target:
            raise AssertionError(
                f"Expected {case_target} multimodal eval cases, got {len(cases)}."
            )

        service.query_rewriter = _StaticQueryRewriter(
            {case.query_text: case.canonical_query_text for case in cases}
        )
        case_results = tuple(_run_case(service, case) for case in cases)
        summary = _summarize(case_results)
        object_store_path = str(service.object_store.objects_path)
        memory_path = str(Path(config.memory_markdown_path))
        return MultimodalEvalResult(
            seed_stats=seed_stats,
            summary=summary,
            cases=case_results,
            temp_root=str(workspace_root),
            object_store_path=object_store_path,
            memory_path=memory_path,
        )
    except BaseException as exc:
        active_exception = exc
        raise
    finally:
        if service is not None:
            try:
                service.shutdown(timeout_s=30.0)  # AUDIT-FIX(#5): Do not let shutdown errors mask the real failure that triggered cleanup.
            except Exception:
                if active_exception is None:
                    raise


def _snapshot_eval_result(
    *,
    workspace_root: Path,
    result: MultimodalEvalResult,
    base_dir: Path | None,
) -> MultimodalEvalResult:
    snapshot_root = Path(
        mkdtemp(
            dir=str(base_dir) if base_dir is not None else None,
            prefix="twinr-multimodal-eval-artifacts-",
        )
    ).resolve(strict=True)

    try:
        shutil.copytree(workspace_root, snapshot_root, dirs_exist_ok=True)  # AUDIT-FIX(#1): Persist artifacts referenced by the returned paths beyond the workspace lifetime.
    except Exception:
        shutil.rmtree(snapshot_root, ignore_errors=True)
        raise

    return MultimodalEvalResult(
        seed_stats=result.seed_stats,
        summary=result.summary,
        cases=result.cases,
        temp_root=str(snapshot_root),
        object_store_path=_remap_snapshot_path(
            original_path=result.object_store_path,
            workspace_root=workspace_root,
            snapshot_root=snapshot_root,
        ),
        memory_path=_remap_snapshot_path(
            original_path=result.memory_path,
            workspace_root=workspace_root,
            snapshot_root=snapshot_root,
        ),
    )


def _remap_snapshot_path(
    *,
    original_path: str,
    workspace_root: Path,
    snapshot_root: Path,
) -> str:
    original = Path(original_path)
    try:
        relative_path = original.resolve(strict=False).relative_to(workspace_root)
    except ValueError:
        return original_path
    return str((snapshot_root / relative_path).resolve(strict=False))


def _seed_multimodal_store(service: LongTermMemoryService) -> MultimodalEvalSeedStats:
    timezone = _resolve_eval_timezone(service)
    total_multimodal = 0
    total_turns = 0

    morning_presence = {
        "facts": {
            "pir": {
                "motion_detected": True,
                "low_motion": False,
                "no_motion_for_s": 0.0,
            },
            "camera": {
                "person_visible": True,
                "person_visible_for_s": 18.0,
                "looking_toward_device": True,
                "body_pose": "standing",
                "smiling": False,
                "hand_or_object_near_camera": True,
                "hand_or_object_near_camera_for_s": 5.0,
            },
        },
        "event_names": ["pir.motion_detected", "camera.person_visible", "camera.hand_or_object_near_camera"],
    }
    afternoon_presence = {
        "facts": {
            "pir": {
                "motion_detected": True,
                "low_motion": False,
                "no_motion_for_s": 0.0,
            },
            "camera": {
                "person_visible": True,
                "person_visible_for_s": 10.0,
                "looking_toward_device": False,
                "body_pose": "sitting",
                "smiling": False,
                "hand_or_object_near_camera": False,
                "hand_or_object_near_camera_for_s": 0.0,
            },
        },
        "event_names": ["pir.motion_detected", "camera.person_visible"],
    }

    for index in range(70):
        created_at = datetime(2026, 3, 14, 8, 0 + (index % 55), 0, tzinfo=timezone)
        _persist_multimodal(
            service,
            LongTermMultimodalEvidence(
                event_name="sensor_observation",
                modality="sensor",
                source="proactive_monitor",
                message="Morning presence observation.",
                data=morning_presence,
                created_at=created_at,
            ),
        )
        total_multimodal += 1

    for index in range(40):
        created_at = datetime(2026, 3, 14, 14, 0 + (index % 50), 0, tzinfo=timezone)
        _persist_multimodal(
            service,
            LongTermMultimodalEvidence(
                event_name="sensor_observation",
                modality="sensor",
                source="proactive_monitor",
                message="Afternoon presence observation.",
                data=afternoon_presence,
                created_at=created_at,
            ),
        )
        total_multimodal += 1

    for index in range(25):
        created_at = datetime(2026, 3, 14, 8, 5 + (index % 30), 0, tzinfo=timezone)
        _persist_multimodal(
            service,
            LongTermMultimodalEvidence(
                event_name="button_interaction",
                modality="button",
                source="runtime_flow",
                message="Green button started listening.",
                data={"button": "green", "action": "start_listening"},
                created_at=created_at,
            ),
        )
        total_multimodal += 1

    for index in range(25):
        created_at = datetime(2026, 3, 14, 15, 5 + (index % 30), 0, tzinfo=timezone)
        _persist_multimodal(
            service,
            LongTermMultimodalEvidence(
                event_name="button_interaction",
                modality="button",
                source="runtime_flow",
                message="Yellow button requested a print.",
                data={"button": "yellow", "action": "print_request"},
                created_at=created_at,
            ),
        )
        total_multimodal += 1

    for index in range(20):
        created_at = datetime(2026, 3, 14, 15, 10 + (index % 25), 0, tzinfo=timezone)
        _persist_multimodal(
            service,
            LongTermMultimodalEvidence(
                event_name="print_completed",
                modality="printer",
                source="realtime_print",
                message="Printed Twinr output delivered.",
                data={"request_source": "button", "queue": "Thermal_GP58"},
                created_at=created_at,
            ),
        )
        total_multimodal += 1

    for index in range(20):
        created_at = datetime(2026, 3, 14, 13, 0 + (index % 40), 0, tzinfo=timezone)
        _persist_multimodal(
            service,
            LongTermMultimodalEvidence(
                event_name="camera_capture",
                modality="camera",
                source="camera_tool",
                message="Camera inspection capture.",
                data={"purpose": "vision_inspection"},
                created_at=created_at,
            ),
        )
        total_multimodal += 1

    targeted_episodes = _targeted_episodes(timezone)
    if len(targeted_episodes) > 300:
        raise AssertionError("Targeted episodes exceed the fixed 300-turn episodic budget.")  # AUDIT-FIX(#7): Guard against silent fixture drift.
    for episode in targeted_episodes:
        _persist_turn(
            service,
            LongTermConversationTurn(
                transcript=episode.transcript,
                response=episode.response,
                created_at=episode.occurred_at,
            ),
        )
        total_turns += 1

    for index in range(300 - len(targeted_episodes)):
        created_at = datetime(2026, 3, 14, 9 + (index % 8), index % 60, 0, tzinfo=timezone)
        _persist_turn(
            service,
            LongTermConversationTurn(
                transcript=f"I talked about distractor topic {index} and a routine unrelated to buttons.",
                response=f"Twinr answered distractor topic {index} in a calm way.",
                created_at=created_at,
            ),
        )
        total_turns += 1

    object_count = _count_loaded_items(service.object_store.load_objects())
    episodic_count = _count_loaded_items(service.prompt_context_store.memory_store.load_entries())
    seed_stats = MultimodalEvalSeedStats(
        multimodal_events=total_multimodal,
        episodic_turns=total_turns,
        consolidated_object_count=object_count,
        episodic_entry_count=episodic_count,
    )
    if seed_stats.total_seed_entries != _FIXED_SEED_TARGET:
        raise AssertionError(
            f"Expected {_FIXED_SEED_TARGET} total seeded entries, got {seed_stats.total_seed_entries}."
        )
    return seed_stats


def _resolve_eval_timezone(service: LongTermMemoryService) -> ZoneInfo:
    timezone_name = getattr(service.config, "local_timezone_name", None) or _FALLBACK_TIMEZONE_NAME
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        return ZoneInfo(_FALLBACK_TIMEZONE_NAME)  # AUDIT-FIX(#6): Keep the fixed eval dataset usable even when local_timezone_name is absent or invalid.


def _count_loaded_items(items: object) -> int:
    try:
        return len(items)  # type: ignore[arg-type]
    except TypeError:
        try:
            iterator = iter(items)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("Loaded store items must be sized or iterable.") from exc
        return sum(1 for _ in iterator)


def _targeted_episodes(timezone: ZoneInfo) -> tuple[_TargetEpisode, ...]:
    return (
        _TargetEpisode(
            transcript="After breakfast I usually ask Twinr about the weather before I start my day.",
            response="I can keep that morning weather routine in mind.",
            occurred_at=datetime(2026, 3, 14, 8, 12, 0, tzinfo=timezone),
        ),
        _TargetEpisode(
            transcript="In the morning I often start by asking Twinr for my appointments.",
            response="Then morning schedule checks are a useful routine for you.",
            occurred_at=datetime(2026, 3, 14, 8, 18, 0, tzinfo=timezone),
        ),
        _TargetEpisode(
            transcript="After lunch I usually print my shopping list with Twinr.",
            response="I can keep that afternoon shopping-list print routine in mind.",
            occurred_at=datetime(2026, 3, 14, 15, 22, 0, tzinfo=timezone),
        ),
        _TargetEpisode(
            transcript="When I ask about recipes, I often print the answer later in the afternoon.",
            response="That sounds like an afternoon print routine for recipes.",
            occurred_at=datetime(2026, 3, 14, 15, 28, 0, tzinfo=timezone),
        ),
        _TargetEpisode(
            transcript="If something looks odd, I use the camera to inspect it with Twinr in the afternoon.",
            response="Then camera inspection is part of your afternoon routine.",
            occurred_at=datetime(2026, 3, 14, 13, 18, 0, tzinfo=timezone),
        ),
        _TargetEpisode(
            transcript="When I stand close to the device in the morning, I usually ask my first question right away.",
            response="That morning presence cue can help start the day calmly.",
            occurred_at=datetime(2026, 3, 14, 8, 25, 0, tzinfo=timezone),
        ),
    )


def _build_multimodal_eval_cases() -> tuple[MultimodalEvalCase, ...]:
    cases: list[MultimodalEvalCase] = []

    for index in range(5):
        cases.append(
            MultimodalEvalCase(
                case_id=f"presence_morning_{index + 1}",
                category="presence_routine",
                query_text=f"Wann bin ich morgens meistens bei Twinr? Variante {index + 1}",
                canonical_query_text="when am i usually near the device in the morning",
                expected_durable_contains=("Presence near the device was observed in the morning.",),
                expect_durable_context=True,
            )
        )
        cases.append(
            MultimodalEvalCase(
                case_id=f"presence_afternoon_{index + 1}",
                category="presence_routine",
                query_text=f"Wann bin ich nachmittags meistens bei Twinr? Variante {index + 1}",
                canonical_query_text="when am i usually near the device in the afternoon",
                expected_durable_contains=("Presence near the device was observed in the afternoon.",),
                expect_durable_context=True,
            )
        )

    for index in range(5):
        cases.append(
            MultimodalEvalCase(
                case_id=f"green_button_{index + 1}",
                category="button_print_routine",
                query_text=f"Wie starte ich morgens meistens ein Gespräch mit Twinr? Variante {index + 1}",
                canonical_query_text="how do i usually start a conversation with twinr in the morning green button",
                expected_durable_contains=("The green button was used to start a conversation in the morning.",),
                expect_durable_context=True,
            )
        )
        cases.append(
            MultimodalEvalCase(
                case_id=f"print_button_{index + 1}",
                category="button_print_routine",
                query_text=f"Wie drucke ich nachmittags meistens Antworten mit Twinr? Variante {index + 1}",
                canonical_query_text="how do i usually print answers with twinr in the afternoon yellow button print",
                expected_durable_contains=(
                    "The yellow button was used to request a printed answer in the afternoon.",
                    "Printed Twinr output was used in the afternoon.",
                ),
                expect_durable_context=True,
            )
        )

    for index in range(5):
        cases.append(
            MultimodalEvalCase(
                case_id=f"camera_use_{index + 1}",
                category="camera_interaction",
                query_text=f"Wann nutze ich mit Twinr die Kamera zur Inspektion? Variante {index + 1}",
                canonical_query_text="when do i use the device camera for inspection in the afternoon",
                expected_durable_contains=("The device camera was used in the afternoon.",),
                expect_durable_context=True,
            )
        )
        cases.append(
            MultimodalEvalCase(
                case_id=f"camera_side_{index + 1}",
                category="camera_interaction",
                query_text=f"Wann gibt es morgens Bewegung direkt an der Kamera? Variante {index + 1}",
                canonical_query_text="when is there camera side interaction in the morning",
                expected_durable_contains=("Camera-side interaction was observed in the morning.",),
                expect_durable_context=True,
            )
        )

    for index in range(5):
        cases.append(
            MultimodalEvalCase(
                case_id=f"combined_weather_{index + 1}",
                category="combined_context",
                query_text=f"Was passt morgens gut zu meiner Wetterroutine mit Twinr? Variante {index + 1}",
                canonical_query_text="weather routine after breakfast morning twinr",
                expected_durable_contains=("The green button was used to start a conversation in the morning.",),
                expected_episodic_contains=("After breakfast I usually ask Twinr about the weather",),
                expect_durable_context=True,
                expect_episodic_context=True,
            )
        )
        cases.append(
            MultimodalEvalCase(
                case_id=f"combined_print_{index + 1}",
                category="combined_context",
                query_text=f"Was passt nachmittags gut zu meiner Druckroutine für Einkaufslisten? Variante {index + 1}",
                canonical_query_text="after lunch usually print shopping list twinr afternoon",
                expected_durable_contains=("Printed Twinr output was used in the afternoon.",),
                expected_episodic_contains=("After lunch I usually print my shopping list with Twinr",),
                expect_durable_context=True,
                expect_episodic_context=True,
            )
        )

    controls = (
        ("math", "Was ist 27 mal 14?", "27 14 multiplication"),
        ("tokyo", "Wie spät ist es in Tokio, wenn es hier 10 Uhr ist?", "tokyo time conversion"),
        ("rainbow", "Erklär mir kurz, was ein Regenbogen ist.", "rainbow explanation"),
        ("tea", "Wie kocht man schwarzen Tee richtig?", "black tea brewing"),
        ("chair", "Worauf sollte ich bei einem guten Stuhl achten?", "chair buying advice"),
        ("history", "Wer war Marie Curie?", "marie curie biography"),
        ("temperature", "Wie rechnet man 20 Grad Celsius in Fahrenheit um?", "celsius fahrenheit conversion"),
        ("calendar", "Wie viele Tage hat der Februar 2028?", "february 2028 leap year"),
        ("music", "Was ist der Unterschied zwischen Walzer und Tango?", "waltz tango difference"),
        ("plants", "Wie oft sollte man Basilikum gießen?", "basil watering frequency"),
    )
    for case_id, query_text, canonical in controls:
        cases.append(
            MultimodalEvalCase(
                case_id=f"control_{case_id}",
                category="control_irrelevant",
                query_text=query_text,
                canonical_query_text=canonical,
                expect_durable_context=False,
                expect_episodic_context=False,
            )
        )

    if len(cases) != 50:
        raise AssertionError(f"Expected 50 multimodal eval cases, got {len(cases)}.")
    return tuple(cases)


def _persist_turn(service: LongTermMemoryService, item: LongTermConversationTurn) -> None:
    LongTermMemoryService._persist_longterm_turn(
        config=service.config,
        store=service.prompt_context_store,
        graph_store=service.graph_store,
        object_store=service.object_store,
        midterm_store=service.midterm_store,
        extractor=service.extractor,
        consolidator=service.consolidator,
        reflector=service.reflector,
        sensor_memory=service.sensor_memory,
        retention_policy=service.retention_policy,
        item=item,
    )


def _persist_multimodal(service: LongTermMemoryService, item: LongTermMultimodalEvidence) -> None:
    LongTermMemoryService._persist_multimodal_evidence(
        object_store=service.object_store,
        midterm_store=service.midterm_store,
        multimodal_extractor=service.multimodal_extractor,
        consolidator=service.consolidator,
        reflector=service.reflector,
        sensor_memory=service.sensor_memory,
        retention_policy=service.retention_policy,
        item=item,
    )


def _run_case(service: LongTermMemoryService, case: MultimodalEvalCase) -> MultimodalEvalCaseResult:
    try:
        context = service.build_provider_context(case.query_text)
        if not isinstance(context, LongTermMemoryContext):
            raise TypeError(
                f"build_provider_context returned {type(context).__name__}, expected LongTermMemoryContext."
            )
    except Exception as exc:
        return MultimodalEvalCaseResult(  # AUDIT-FIX(#3): Convert single-case failures into explicit failed results so the suite still completes.
            case_id=case.case_id,
            category=case.category,
            passed=False,
            durable_context_present=False,
            episodic_context_present=False,
            missing_durable=case.expected_durable_contains,
            missing_episodic=case.expected_episodic_contains,
            present_forbidden_durable=(),
            present_forbidden_episodic=(),
            error=f"{type(exc).__name__}: {exc}",
        )

    durable_text = _normalized_context_text(context, "durable_context")
    episodic_text = _normalized_context_text(context, "episodic_context")
    missing_durable = tuple(
        needle for needle in case.expected_durable_contains if needle not in durable_text
    )
    missing_episodic = tuple(
        needle for needle in case.expected_episodic_contains if needle not in episodic_text
    )
    forbidden_durable = tuple(
        needle for needle in case.expected_durable_absent if needle in durable_text
    )
    forbidden_episodic = tuple(
        needle for needle in case.expected_episodic_absent if needle in episodic_text
    )

    durable_present = _context_text_present(context, "durable_context")
    episodic_present = _context_text_present(context, "episodic_context")
    passed = (
        not missing_durable
        and not missing_episodic
        and not forbidden_durable
        and not forbidden_episodic
    )
    if case.expect_durable_context is not None:
        passed = passed and (durable_present == case.expect_durable_context)  # AUDIT-FIX(#8): Treat empty/whitespace-only context as absent to avoid false positives.
    if case.expect_episodic_context is not None:
        passed = passed and (episodic_present == case.expect_episodic_context)

    return MultimodalEvalCaseResult(
        case_id=case.case_id,
        category=case.category,
        passed=passed,
        durable_context_present=durable_present,
        episodic_context_present=episodic_present,
        missing_durable=missing_durable,
        missing_episodic=missing_episodic,
        present_forbidden_durable=forbidden_durable,
        present_forbidden_episodic=forbidden_episodic,
    )


def _normalized_context_text(context: LongTermMemoryContext, attribute_name: str) -> str:
    raw_value = getattr(context, attribute_name, None)
    if raw_value is None:
        return ""
    return str(raw_value).strip()


def _context_text_present(context: LongTermMemoryContext, attribute_name: str) -> bool:
    return bool(_normalized_context_text(context, attribute_name))


def _summarize(results: tuple[MultimodalEvalCaseResult, ...]) -> MultimodalEvalSummary:
    category_case_counts: dict[str, int] = {}
    category_pass_counts: dict[str, int] = {}
    passed = 0
    for result in results:
        category_case_counts[result.category] = category_case_counts.get(result.category, 0) + 1
        if result.passed:
            passed += 1
            category_pass_counts[result.category] = category_pass_counts.get(result.category, 0) + 1
    return MultimodalEvalSummary(
        total_cases=len(results),
        passed_cases=passed,
        category_case_counts=category_case_counts,
        category_pass_counts=category_pass_counts,
    )


def _result_to_payload(result: MultimodalEvalResult) -> dict[str, object]:
    return {
        "seed_stats": {
            "multimodal_events": result.seed_stats.multimodal_events,
            "episodic_turns": result.seed_stats.episodic_turns,
            "total_seed_entries": result.seed_stats.total_seed_entries,
            "consolidated_object_count": result.seed_stats.consolidated_object_count,
            "episodic_entry_count": result.seed_stats.episodic_entry_count,
        },
        "summary": {
            "total_cases": result.summary.total_cases,
            "passed_cases": result.summary.passed_cases,
            "accuracy": result.summary.accuracy,
            "category_case_counts": result.summary.category_case_counts,
            "category_pass_counts": result.summary.category_pass_counts,
        },
        "cases": [
            {
                "case_id": case.case_id,
                "category": case.category,
                "passed": case.passed,
                "durable_context_present": case.durable_context_present,
                "episodic_context_present": case.episodic_context_present,
                "missing_durable": list(case.missing_durable),
                "missing_episodic": list(case.missing_episodic),
                "present_forbidden_durable": list(case.present_forbidden_durable),
                "present_forbidden_episodic": list(case.present_forbidden_episodic),
                "error": case.error,
            }
            for case in result.cases
        ],
        "temp_root": result.temp_root,
        "object_store_path": result.object_store_path,
        "memory_path": result.memory_path,
    }


def _error_payload(exc: BaseException) -> dict[str, object]:
    return {
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    }


def main() -> int:
    try:
        result = run_multimodal_longterm_eval()
    except Exception as exc:
        print(
            json.dumps(_error_payload(exc), ensure_ascii=False, indent=2),
            flush=True,
        )  # AUDIT-FIX(#9): Emit machine-readable failure output instead of an opaque traceback-only crash.
        return 1

    print(json.dumps(_result_to_payload(result), ensure_ascii=False, indent=2), flush=True)
    return 0


__all__ = [
    "MultimodalEvalCase",
    "MultimodalEvalCaseResult",
    "MultimodalEvalResult",
    "MultimodalEvalSeedStats",
    "MultimodalEvalSummary",
    "run_multimodal_longterm_eval",
]


if __name__ == "__main__":
    raise SystemExit(main())