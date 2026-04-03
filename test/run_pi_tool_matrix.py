"""Run the live Pi tool matrix against the spoken/runtime tool path.

The harness exercises real tool-selection turns with live provider calls and an
isolated remote-memory namespace. It also supports bounded group slices so
expensive real-Pi acceptance can be split into smaller SSH runs and later
merged into one combined matrix artifact.
"""

from __future__ import annotations
# ruff: noqa: E402

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import argparse
import base64
import json
import math
import shutil
import sys
from zoneinfo import ZoneInfo

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.self_coding.worker import SelfCodingCompileWorker
from twinr.agent.base_agent.runtime.runtime import TwinrRuntime
from twinr.agent.tools import (
    build_tool_agent_instructions,
    realtime_tool_names,
)
from twinr.agent.workflows.streaming_runner import TwinrStreamingHardwareLoop
from twinr.display.service_connect_cues import DisplayServiceConnectCueStore
from twinr.memory.longterm.core.models import (
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
)
import twinr.channels.whatsapp.pairing as whatsapp_pairing
import twinr.agent.tools.handlers.smarthome as smart_home_handlers
import twinr.agent.workflows.realtime_runner as realtime_runner_module
from test.pi_tool_matrix_support import (
    MatrixHouseholdIdentityManager,
    MatrixPortraitProvider,
    MatrixServiceConnectCoordinator,
    MatrixSelfCodingCompileDriver,
    MatrixSmartHomeProvider,
    MatrixWhatsAppDispatch,
    MatrixWorldRemoteState,
    install_matrix_whatsapp_runtime,
    matrix_service_connect_probe,
    matrix_world_feed_items,
    write_matrix_browser_workspace,
)
from test.pi_tool_matrix_catalog import available_matrix_groups, normalize_matrix_groups


_FALLBACK_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl9l2sAAAAASUVORK5CYII="
)
_WRAPPER_TOOL_NAMES = frozenset({"handoff_specialist_worker"})
_TEST_CORINNA_PHONE_OLD = "+15555551234"
_TEST_CORINNA_PHONE_NEW = "+15555558877"
_TEST_ANNA_PHONE = "555-0100"
_TEST_ANNA_PHONE_NEEDLES = (_TEST_ANNA_PHONE, "5550100", "555 0100")


def _load_static_camera_image() -> bytes:
    """Return one valid repo-local image for live vision acceptance."""

    candidate = Path(__file__).resolve().parents[1] / "twinr_logo.png"
    try:
        return candidate.read_bytes()
    except Exception:
        return _FALLBACK_TINY_PNG


def _voice_sample_pcm_bytes(*, frequency_hz: float = 175.0, amplitude: float = 0.35, duration_s: float = 1.8) -> bytes:
    sample_rate = 24000
    total_frames = int(sample_rate * duration_s)
    frames: bytearray = bytearray()
    for index in range(total_frames):
        t = index / sample_rate
        envelope = min(1.0, index / (sample_rate * 0.2), (total_frames - index) / (sample_rate * 0.2))
        sample = amplitude * envelope * (
            (0.70 * math.sin(2.0 * math.pi * frequency_hz * t))
            + (0.20 * math.sin(2.0 * math.pi * frequency_hz * 2.0 * t))
            + (0.10 * math.sin(2.0 * math.pi * (frequency_hz + 35.0) * t))
        )
        value = max(-32767, min(32767, int(sample * 32767)))
        frames.extend(int(value).to_bytes(2, "little", signed=True))
    return bytes(frames)


def _longterm_source(event_id: str) -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=(event_id,),
        speaker="user",
        modality="voice",
    )


class SilentPlayer:
    def __init__(self) -> None:
        self.play_count = 0
        self.total_bytes = 0

    def play_wav_chunks(self, chunks) -> None:
        for chunk in chunks:
            self.play_count += 1
            self.total_bytes += len(chunk)

    def play_tone(self, **kwargs) -> None:
        del kwargs


class CapturingPrinter:
    def __init__(self) -> None:
        self.printed: list[str] = []

    def print_text(self, text: str) -> str:
        self.printed.append(text)
        return f"job-{len(self.printed)}"


class StaticCamera:
    def __init__(self, image_bytes: bytes) -> None:
        self.image_bytes = image_bytes
        self.capture_calls = 0

    def capture_photo(self, *, filename: str) -> SimpleNamespace:
        self.capture_calls += 1
        return SimpleNamespace(
            source_device="static-camera",
            input_format="png",
            data=self.image_bytes,
            content_type="image/png",
            filename=filename,
        )


@dataclass
class TurnArtifact:
    prompt: str
    answer: str
    tool_calls: list[str]
    raw_tool_calls: list[dict[str, object]]
    emitted: list[str]
    status: str
    keep_listening: bool


@dataclass
class MatrixDimension:
    status: str = "n/a"
    detail: str = ""


@dataclass
class MatrixEntry:
    single_turn: MatrixDimension = field(default_factory=MatrixDimension)
    multi_turn: MatrixDimension = field(default_factory=MatrixDimension)
    persistence: MatrixDimension = field(default_factory=MatrixDimension)

    def overall(self) -> str:
        statuses = [self.single_turn.status, self.multi_turn.status, self.persistence.status]
        if any(status == "fail" for status in statuses):
            return "fail"
        if any(status == "pass" for status in statuses):
            return "pass"
        return "n/a"


class ToolMatrixContext:
    def __init__(self, *, base_env_path: Path) -> None:
        self.base_env_path = base_env_path.resolve()
        self.temp_dir = TemporaryDirectory(prefix="twinr-tool-matrix-")
        self.root = Path(self.temp_dir.name)
        self.run_id = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ")
        self.state_dir = self.root / "state"
        self.personality_dir = self.root / "personality"
        self.reference_image_path = self.root / "user-reference.png"
        self.env_path = self.root / ".env"
        self.camera_image_bytes = _load_static_camera_image()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.base_env_path.parent / "personality", self.personality_dir, dirs_exist_ok=True)
        self.reference_image_path.write_bytes(self.camera_image_bytes)
        base_env_text = self.base_env_path.read_text(encoding="utf-8")
        overrides = [
            "TWINR_STT_PROVIDER=deepgram",
            "TWINR_LLM_PROVIDER=openai",
            "TWINR_TTS_PROVIDER=openai",
            f"TWINR_PERSONALITY_DIR={self.personality_dir.name}",
            f"TWINR_VISION_REFERENCE_IMAGE={self.reference_image_path}",
            f"TWINR_RUNTIME_STATE_PATH={self.state_dir / 'runtime-state.json'}",
            f"TWINR_MEMORY_MARKDOWN_PATH={self.state_dir / 'MEMORY.md'}",
            f"TWINR_REMINDER_STORE_PATH={self.state_dir / 'reminders.json'}",
            f"TWINR_AUTOMATION_STORE_PATH={self.state_dir / 'automations.json'}",
            f"TWINR_VOICE_PROFILE_STORE_PATH={self.state_dir / 'voice_profile.json'}",
            f"TWINR_ADAPTIVE_TIMING_STORE_PATH={self.state_dir / 'adaptive_timing.json'}",
            f"TWINR_LONG_TERM_MEMORY_PATH={self.state_dir / 'chonkydb'}",
            f"TWINR_LONG_TERM_MEMORY_REMOTE_NAMESPACE=pi_tool_matrix_{self.run_id}",
            "TWINR_LONG_TERM_MEMORY_ENABLED=true",
            "TWINR_LONG_TERM_MEMORY_BACKGROUND_STORE_TURNS=false",
            "TWINR_PROACTIVE_ENABLED=false",
            "TWINR_CONVERSATION_FOLLOW_UP_ENABLED=false",
            "TWINR_SEARCH_FEEDBACK_TONES_ENABLED=false",
            "TWINR_WHATSAPP_ALLOW_FROM=+15555550100",
        ]
        self.env_path.write_text(base_env_text.rstrip() + "\n" + "\n".join(overrides) + "\n", encoding="utf-8")
        self.printer = CapturingPrinter()
        self.player = SilentPlayer()
        self.camera = StaticCamera(self.camera_image_bytes)
        self.portrait_provider = MatrixPortraitProvider()
        self.household_identity_manager = MatrixHouseholdIdentityManager()
        self.smart_home_provider = MatrixSmartHomeProvider()
        self.smart_home_adapter = self.smart_home_provider.build_adapter()
        self.world_remote_state = MatrixWorldRemoteState()
        self.self_coding_compile_driver = MatrixSelfCodingCompileDriver()
        self.whatsapp_dispatch = MatrixWhatsAppDispatch()

    def close(self) -> None:
        self.temp_dir.cleanup()

    def prepare_browser_workspace(self) -> None:
        """Create the optional browser workspace used by browser scenarios."""

        write_matrix_browser_workspace(self.root)

    def prepare_whatsapp_runtime(self) -> None:
        """Create the minimal local WhatsApp runtime files for matrix runs."""

        install_matrix_whatsapp_runtime(self.root)

    def make_loop(
        self,
        *,
        emitted: list[str],
        enable_browser_automation: bool = False,
        enable_whatsapp_tools: bool = False,
        authorize_sensitive_tools: bool = True,
    ) -> TwinrStreamingHardwareLoop:
        if enable_browser_automation:
            self.prepare_browser_workspace()
        if enable_whatsapp_tools:
            self.prepare_whatsapp_runtime()
        config = TwinrConfig.from_env(self.env_path)
        if enable_browser_automation:
            config = replace(config, browser_automation_enabled=True)
        runtime = TwinrRuntime(config=config)
        runtime.update_user_voice_assessment(
            status="likely_user",
            confidence=0.96,
            checked_at=datetime.now(ZoneInfo("UTC")).isoformat(),
        )
        loop = TwinrStreamingHardwareLoop(
            config=config,
            runtime=runtime,
            player=self.player,
            printer=self.printer,
            camera=self.camera,
            button_monitor=SimpleNamespace(),
            proactive_monitor=SimpleNamespace(),
            emit=emitted.append,
        )
        loop._portrait_identity_tool_provider = self.portrait_provider
        loop.household_identity_manager = self.household_identity_manager
        if enable_whatsapp_tools:
            loop.dispatch_whatsapp_outbound_message = self.whatsapp_dispatch.dispatch  # type: ignore[attr-defined]
        if authorize_sensitive_tools:
            # The live matrix validates the full runtime tool surface, including
            # camera, printer, and outbound messaging paths that are normally
            # identity-gated in ambient voice mode.
            loop.authorize_realtime_sensitive_tools("pi_tool_matrix")
        return loop

    def install_self_coding_driver(self, loop: TwinrStreamingHardwareLoop) -> None:
        """Swap one deterministic compile driver into a live matrix loop."""

        existing_store = getattr(loop, "_self_coding_store", None)
        if existing_store is None:
            from twinr.agent.tools.handlers.self_coding import ensure_self_coding_runtime

            existing_store = ensure_self_coding_runtime(loop)["store"]
        loop._self_coding_compile_worker = SelfCodingCompileWorker(
            store=existing_store,
            driver=self.self_coding_compile_driver,
        )
        setattr(loop, "_self_coding_learning_flow", None)

    def install_world_intelligence_stubs(self, loop: TwinrStreamingHardwareLoop) -> None:
        """Attach bounded world-intelligence discovery helpers to one loop."""

        service = loop.runtime.long_term_memory.personality_learning.world_intelligence
        service.remote_state = self.world_remote_state
        loop.runtime.long_term_memory.personality_learning.background_loop.remote_state = self.world_remote_state
        service.page_loader = lambda url: SimpleNamespace(
            url=url,
            content_type="text/html; charset=utf-8",
            text=(
                '<html><head><link rel="alternate" '
                'type="application/rss+xml" href="/feeds/hamburg-local.xml"></head></html>'
            ),
        )
        service.feed_reader = lambda feed_url, *, max_items, timeout_s: matrix_world_feed_items()[:max_items]
        loop.print_backend.search_live_info_with_metadata = lambda *args, **kwargs: SimpleNamespace(
            answer="Discovered source pages.",
            sources=("https://example.com/hamburg",),
            used_web_search=True,
            response_id="resp_world_discovery_matrix",
            request_id="req_world_discovery_matrix",
            model="gpt-5.4-mini-2026-03-17",
            token_usage=None,
        )

    def patch_smart_home_builders(self) -> tuple[object, object]:
        """Route smart-home handler lookups to the matrix adapter."""

        originals = (
            smart_home_handlers.build_smart_home_hub_adapter,
            realtime_runner_module.build_smart_home_hub_adapter,
        )
        smart_home_handlers.build_smart_home_hub_adapter = lambda *args, **kwargs: self.smart_home_adapter
        realtime_runner_module.build_smart_home_hub_adapter = lambda *args, **kwargs: self.smart_home_adapter
        return originals

    @staticmethod
    def restore_smart_home_builders(originals: tuple[object, object]) -> None:
        """Restore the original smart-home adapter builders after matrix runs."""

        smart_home_handlers.build_smart_home_hub_adapter = originals[0]
        realtime_runner_module.build_smart_home_hub_adapter = originals[1]

    def patch_service_connect_helpers(self) -> tuple[object, object]:
        """Route service-connect WhatsApp helpers to deterministic matrix stubs."""

        originals = (
            whatsapp_pairing.probe_whatsapp_runtime,
            whatsapp_pairing.WhatsAppPairingCoordinator,
        )
        whatsapp_pairing.probe_whatsapp_runtime = lambda *args, **kwargs: matrix_service_connect_probe()
        whatsapp_pairing.WhatsAppPairingCoordinator = MatrixServiceConnectCoordinator
        return originals

    @staticmethod
    def restore_service_connect_helpers(originals: tuple[object, object]) -> None:
        """Restore the original service-connect WhatsApp helpers."""

        whatsapp_pairing.probe_whatsapp_runtime = originals[0]
        whatsapp_pairing.WhatsAppPairingCoordinator = originals[1]

    def seed_conflict(
        self,
        *,
        loop: TwinrStreamingHardwareLoop | None = None,
    ) -> TwinrStreamingHardwareLoop:
        """Persist one deterministic conflict, optionally on an existing loop.

        Reusing the active loop keeps the live matrix from paying the required
        remote bootstrap twice before the spoken conflict turn even starts.
        """

        emitted: list[str] = []
        active_loop = loop or self.make_loop(emitted=emitted)
        existing = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_old",
            kind="contact_method_fact",
            summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_OLD}.",
            source=_longterm_source("turn:1"),
            status="active",
            confidence=0.95,
            slot_key="contact:person:corinna_maier:phone",
            value_key=_TEST_CORINNA_PHONE_OLD,
        )
        candidate = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_new",
            kind="contact_method_fact",
            summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_NEW}.",
            source=_longterm_source("turn:2"),
            status="uncertain",
            confidence=0.92,
            slot_key="contact:person:corinna_maier:phone",
            value_key=_TEST_CORINNA_PHONE_NEW,
        )
        conflict = LongTermMemoryConflictV1(
            slot_key="contact:person:corinna_maier:phone",
            candidate_memory_id="fact:corinna_phone_new",
            existing_memory_ids=("fact:corinna_phone_old",),
            question="Which phone number should I use for Corinna Maier?",
            reason="Conflicting phone numbers exist.",
        )
        # The live matrix seeds a fresh isolated namespace, so a direct
        # snapshot write avoids paying a redundant remote read/merge cycle
        # before the spoken conflict turn even starts.
        active_loop.runtime.long_term_memory.object_store.write_snapshot(
            objects=(existing, candidate),
            conflicts=(conflict,),
        )
        return active_loop


def run_text_turn(
    loop: TwinrStreamingHardwareLoop,
    prompt: str,
    *,
    current_turn_audio_pcm: bytes | None = None,
) -> TurnArtifact:
    emitted: list[str] = []
    loop.emit = emitted.append
    try:
        if current_turn_audio_pcm is not None:
            loop._current_turn_audio_pcm = current_turn_audio_pcm
            loop._current_turn_audio_sample_rate = loop.config.openai_realtime_input_sample_rate
        loop.runtime.begin_listening(request_source="matrix")
        loop.runtime.submit_transcript(prompt)
        response = loop.streaming_turn_loop.run(
            prompt,
            conversation=loop.runtime.tool_provider_conversation_context(),
            instructions=build_tool_agent_instructions(
                loop.config,
                extra_instructions=loop.config.openai_realtime_instructions,
            ),
            allow_web_search=False,
        )
        if loop.runtime.status.value == "printing":
            loop.runtime.resume_answering_after_print()
        else:
            loop.runtime.begin_answering()
        answer = loop.runtime.finalize_agent_turn(response.text)
        loop.runtime.record_personality_tool_history(
            tool_calls=response.tool_calls,
            tool_results=response.tool_results,
        )
        loop.runtime.finish_speaking()
        return TurnArtifact(
            prompt=prompt,
            answer=answer,
            tool_calls=[call.name for call in response.tool_calls],
            raw_tool_calls=[
                {
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in response.tool_calls
            ],
            emitted=emitted,
            status=loop.runtime.status.value,
            keep_listening=not any(call.name == "end_conversation" for call in response.tool_calls),
        )
    except Exception as exc:
        emitted.append(f"exception={type(exc).__name__}: {exc}")
        try:
            loop.runtime.fail(str(exc))
        except Exception:
            pass
        return TurnArtifact(
            prompt=prompt,
            answer="",
            tool_calls=[],
            raw_tool_calls=[],
            emitted=emitted,
            status="error",
            keep_listening=True,
        )
    finally:
        loop._current_turn_audio_pcm = None


def _assert_tools(
    artifact: TurnArtifact,
    *,
    required: tuple[str, ...],
    allowed: tuple[str, ...] | None = None,
) -> tuple[bool, str]:
    observed = tuple(artifact.tool_calls)
    allowed_set = set(required)
    allowed_set.update(_WRAPPER_TOOL_NAMES)
    if allowed is not None:
        allowed_set.update(allowed)
    missing = [name for name in required if name not in observed]
    unexpected = [name for name in observed if name not in allowed_set]
    if missing:
        return False, f"missing {missing}; observed={observed}"
    if unexpected:
        return False, f"unexpected {unexpected}; observed={observed}"
    return True, f"observed={observed}"


def _mark(entries: dict[str, MatrixEntry], tool_name: str, dimension: str, status: str, detail: str) -> None:
    target = getattr(entries[tool_name], dimension)
    if target.status == "fail":
        if detail and detail not in target.detail:
            target.detail = f"{target.detail} | {detail}" if target.detail else detail
        return
    if status == "fail":
        target.status = "fail"
        target.detail = detail
        return
    if target.status == "pass" and status == "pass":
        if detail and detail not in target.detail:
            target.detail = f"{target.detail} | {detail}" if target.detail else detail
        return
    target.status = status
    target.detail = detail


def _pass(
    entries: dict[str, MatrixEntry],
    tool_name: str,
    dimension: str,
    detail: str,
) -> None:
    _mark(entries, tool_name, dimension, "pass", detail)


def _fail(
    entries: dict[str, MatrixEntry],
    tool_name: str,
    dimension: str,
    detail: str,
) -> None:
    _mark(entries, tool_name, dimension, "fail", detail)


def _na(entries: dict[str, MatrixEntry], tool_name: str, dimension: str, detail: str = "") -> None:
    _mark(entries, tool_name, dimension, "n/a", detail)


def _safe_load_self_coding_activation(
    store: object,
    *,
    skill_id: str,
    version: int,
) -> tuple[object | None, str]:
    """Load one self-coding activation without aborting the whole matrix run."""

    try:
        activation = getattr(store, "load_activation")(skill_id, version=version)
    except Exception as exc:
        return None, f"activation {skill_id}@v{version} unavailable: {exc}"
    return activation, ""


def _automation_delivery(record: dict[str, object] | None) -> str | None:
    if not isinstance(record, dict):
        return None
    explicit = str(record.get("delivery", "")).strip().lower()
    if explicit:
        return explicit
    actions = record.get("actions")
    if not isinstance(actions, list) or not actions:
        return None
    first = actions[0]
    if not isinstance(first, dict):
        return None
    kind = str(first.get("kind", "")).strip().lower()
    if kind == "print":
        return "printed"
    if kind in {"say", "llm_prompt"}:
        payload = first.get("payload")
        if isinstance(payload, dict):
            payload_delivery = str(payload.get("delivery", "")).strip().lower()
            if payload_delivery:
                return payload_delivery
        return "spoken"
    return None


def _sensor_automation_matches(
    record: dict[str, object] | None,
    *,
    trigger_kind: str,
    hold_seconds: float,
    delivery: str,
) -> bool:
    """Return whether one automation record matches the expected sensor contract."""

    if not isinstance(record, dict):
        return False
    observed_trigger_kind = str(record.get("sensor_trigger_kind", "")).strip().lower()
    observed_hold_value = record.get("sensor_hold_seconds")
    if not isinstance(observed_hold_value, (int, float, str)):
        return False
    try:
        observed_hold_seconds = float(observed_hold_value)
    except (TypeError, ValueError):
        return False
    return (
        observed_trigger_kind == trigger_kind.strip().lower()
        and math.isclose(observed_hold_seconds, hold_seconds)
        and _automation_delivery(record) == delivery.strip().lower()
    )


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    haystack = text.lower()
    return any(needle.lower() in haystack for needle in needles)


def _user_discovery_review_items_contain(
    review_items: tuple[object, ...] | list[object],
    needles: tuple[str, ...],
) -> bool:
    """Return whether discovery review items contain the expected learned facts."""

    haystacks: list[str] = []
    for item in review_items:
        if isinstance(item, Mapping):
            summary = str(item.get("summary", "")).strip()
        else:
            summary = str(getattr(item, "summary", "")).strip()
        if summary:
            haystacks.append(summary.lower())
    return any(needle.lower() in haystack for haystack in haystacks for needle in needles)


def _managed_context_contains(
    entries: tuple[object, ...] | list[object],
    *,
    expected_fragments_by_key: dict[str, tuple[str, ...]],
) -> bool:
    """Return whether managed-context entries contain the expected keyed fragments."""

    observed: dict[str, str] = {}
    for entry in entries:
        key = str(getattr(entry, "key", "")).strip().lower()
        observed_instruction = str(getattr(entry, "instruction", "")).strip().lower()
        if key and observed_instruction:
            observed[key] = observed_instruction
    for key, fragments in expected_fragments_by_key.items():
        instruction: str | None = observed.get(key.strip().lower())
        if not instruction:
            return False
        if fragments and not any(fragment.strip().lower() in instruction for fragment in fragments):
            return False
    return True


def _voice_profile_status_answer_matches(answer: str, *, enrolled: bool) -> bool:
    """Return whether one spoken answer matches the expected voice-profile state."""

    if enrolled:
        return _contains_any(answer, ("stimmprofil", "voice profile", "gespeichert"))
    return _contains_any(answer, ("kein", "nicht", "no"))


def run_matrix(base_env_path: Path, *, groups: tuple[str, ...] | None = None) -> dict[str, object]:
    """Execute the live tool matrix for all or selected scenario groups."""

    entries = {name: MatrixEntry() for name in realtime_tool_names()}
    for name in entries:
        for dimension in ("single_turn", "multi_turn", "persistence"):
            _na(entries, name, dimension)
    scenario_results: list[dict[str, object]] = []
    selected_groups = normalize_matrix_groups(groups)
    context = ToolMatrixContext(base_env_path=base_env_path)
    smart_home_builder_originals = context.patch_smart_home_builders()
    service_connect_originals = context.patch_service_connect_helpers()
    try:
        def group_enabled(group_name: str) -> bool:
            return group_name in selected_groups

        def record(name: str, artifact: TurnArtifact, *, note: str = "") -> None:
            scenario_results.append(
                {
                    "scenario": name,
                    "prompt": artifact.prompt,
                    "answer": artifact.answer,
                    "tool_calls": artifact.tool_calls,
                    "raw_tool_calls": artifact.raw_tool_calls,
                    "status": artifact.status,
                    "keep_listening": artifact.keep_listening,
                    "emitted": artifact.emitted,
                    "note": note,
                }
            )

        if group_enabled("core"):
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Bitte drucke genau diesen Text aus: Zahnarzt Montag 14 Uhr.",
            )
            record("print_receipt_single", artifact)
            ok, detail = _assert_tools(artifact, required=("print_receipt",))
            if ok and context.printer.printed and "Zahnarzt Montag 14 Uhr" in context.printer.printed[-1]:
                _pass(entries, "print_receipt", "single_turn", detail)
            else:
                _fail(entries, "print_receipt", "single_turn", detail or "print output missing")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Wie wird das Wetter morgen in Schwarzenbek?",
            )
            record("search_live_info_single", artifact)
            ok, detail = _assert_tools(artifact, required=("search_live_info",))
            if ok and _contains_any(artifact.answer, ("schwarzenbek", "grad", "wetter")):
                _pass(entries, "search_live_info", "single_turn", detail)
            else:
                _fail(entries, "search_live_info", "single_turn", detail or "search answer weak")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Bitte schau kurz mit der Kamera nach vorne und sag mir, ob du ein Bild bekommen hast.",
            )
            record("inspect_camera_single", artifact)
            ok, detail = _assert_tools(artifact, required=("inspect_camera",))
            if ok and context.camera.capture_calls >= 1 and bool(artifact.answer.strip()):
                _pass(entries, "inspect_camera", "single_turn", detail)
            else:
                _fail(entries, "inspect_camera", "single_turn", detail or "camera inspection failed")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Danke, das war's fuer jetzt.",
            )
            record("end_conversation_single", artifact)
            ok, detail = _assert_tools(artifact, required=("end_conversation",))
            if ok and not artifact.keep_listening:
                _pass(entries, "end_conversation", "single_turn", detail)
            else:
                _fail(entries, "end_conversation", "single_turn", detail or "conversation did not end")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Bitte sei jetzt fuer 20 Minuten ruhig, damit der Fernseher dich nicht dauernd wieder aktiviert.",
            )
            record("manage_voice_quiet_mode_set_single", artifact)
            ok, detail = _assert_tools(artifact, required=("manage_voice_quiet_mode",))
            if ok and loop.runtime.voice_quiet_active():
                _pass(entries, "manage_voice_quiet_mode", "single_turn", detail)
            else:
                _fail(entries, "manage_voice_quiet_mode", "single_turn", detail or "quiet mode did not activate")

            artifact = run_text_turn(loop, "Bist du gerade ruhig oder hoerst du normal zu?")
            record("manage_voice_quiet_mode_status_multi", artifact)
            ok, detail = _assert_tools(artifact, required=("manage_voice_quiet_mode",))
            if ok and loop.runtime.voice_quiet_active():
                _pass(entries, "manage_voice_quiet_mode", "multi_turn", detail)
            else:
                _fail(entries, "manage_voice_quiet_mode", "multi_turn", detail or "quiet mode status follow-up failed")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Bist du gerade ruhig oder hoerst du normal zu?")
            record("manage_voice_quiet_mode_status_persistence", artifact)
            ok, detail = _assert_tools(artifact, required=("manage_voice_quiet_mode",))
            if ok and loop.runtime.voice_quiet_active():
                _pass(entries, "manage_voice_quiet_mode", "persistence", "quiet mode survived runtime restart")
            else:
                _fail(entries, "manage_voice_quiet_mode", "persistence", detail or "quiet mode did not survive restart")

            artifact = run_text_turn(loop, "Bitte hoer jetzt wieder normal zu.")
            record("manage_voice_quiet_mode_clear_multi", artifact)
            ok, detail = _assert_tools(artifact, required=("manage_voice_quiet_mode",))
            if ok and not loop.runtime.voice_quiet_active():
                _pass(entries, "manage_voice_quiet_mode", "multi_turn", "clear succeeded after follow-up")
            else:
                _fail(entries, "manage_voice_quiet_mode", "multi_turn", detail or "quiet mode did not clear")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Bist du gerade ruhig oder hoerst du normal zu?")
            record("manage_voice_quiet_mode_clear_persistence", artifact)
            ok, detail = _assert_tools(artifact, required=("manage_voice_quiet_mode",))
            if ok and not loop.runtime.voice_quiet_active():
                _pass(entries, "manage_voice_quiet_mode", "persistence", "cleared quiet mode stayed cleared after restart")
            else:
                _fail(entries, "manage_voice_quiet_mode", "persistence", detail or "cleared quiet mode reappeared")

        if group_enabled("browser_channels"):
            loop = context.make_loop(emitted=[], enable_browser_automation=True)
            artifact = run_text_turn(
                loop,
                "Bitte pruefe direkt auf example.org auf der Website selbst, ob dort aktuell Oeffnungszeiten sichtbar sind. Nutze die Website direkt und nicht nur eine normale Websuche.",
            )
            record("browser_automation_single", artifact)
            ok, detail = _assert_tools(
                artifact,
                required=("browser_automation",),
                allowed=("search_live_info", "browser_automation"),
            )
            browser_call = next(
                (call for call in artifact.raw_tool_calls if call.get("name") == "browser_automation"),
                None,
            )
            browser_arguments = browser_call.get("arguments", {}) if isinstance(browser_call, dict) else {}
            browser_domains = tuple(
                str(item).strip().lower()
                for item in browser_arguments.get("allowed_domains", ())
                if str(item).strip()
            )
            if ok and "example.org" in browser_domains and bool(artifact.answer.strip()):
                _pass(entries, "browser_automation", "single_turn", detail)
            else:
                _fail(entries, "browser_automation", "single_turn", detail or "browser automation did not target example.org")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Bitte verbinde jetzt WhatsApp mit Twinr.")
            record("connect_service_integration_single", artifact)
            ok, detail = _assert_tools(artifact, required=("connect_service_integration",))
            cue = DisplayServiceConnectCueStore.from_config(loop.config).load()
            if ok and cue is not None and cue.service_id == "whatsapp" and cue.phase == "starting":
                _pass(entries, "connect_service_integration", "single_turn", detail)
            else:
                _fail(entries, "connect_service_integration", "single_turn", detail or "service-connect cue missing")

            loop = context.make_loop(emitted=[], enable_whatsapp_tools=True)
            loop._handle_remember_contact_tool_call(
                {
                    "given_name": "Anna",
                    "family_name": "Schulz",
                    "phone": "+15555552233",
                    "role": "Tochter",
                    "confirmed": True,
                }
            )
            artifact = run_text_turn(
                loop,
                "Schreib bitte meiner Tochter Anna Schulz per WhatsApp genau: Ich komme spaeter.",
            )
            record("send_whatsapp_message_single", artifact)
            ok, detail = _assert_tools(
                artifact,
                required=("send_whatsapp_message",),
                allowed=("lookup_contact", "send_whatsapp_message"),
            )
            if ok and not context.whatsapp_dispatch.sent_messages:
                _pass(entries, "send_whatsapp_message", "single_turn", detail)
            else:
                _fail(entries, "send_whatsapp_message", "single_turn", detail or "WhatsApp confirmation step missing")

            artifact = run_text_turn(loop, "Ja, sende diese WhatsApp an Anna Schulz jetzt genau so.")
            record("send_whatsapp_message_multi", artifact)
            ok, detail = _assert_tools(
                artifact,
                required=("send_whatsapp_message",),
                allowed=("lookup_contact", "send_whatsapp_message"),
            )
            if (
                ok
                and context.whatsapp_dispatch.sent_messages
                and context.whatsapp_dispatch.sent_messages[-1]["recipient_label"] == "Anna Schulz"
                and context.whatsapp_dispatch.sent_messages[-1]["text"] == "Ich komme spaeter."
            ):
                _pass(entries, "send_whatsapp_message", "multi_turn", detail)
            else:
                _fail(entries, "send_whatsapp_message", "multi_turn", detail or "WhatsApp send did not complete")

        if group_enabled("reminder_automation"):
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Erinnere mich bitte morgen um 12 Uhr an meine Tabletten.",
            )
            record("schedule_reminder_single", artifact)
            ok, detail = _assert_tools(artifact, required=("schedule_reminder",))
            reminder_store = Path(loop.config.reminder_store_path)
            if ok and reminder_store.exists() and "Tabletten" in reminder_store.read_text(encoding="utf-8"):
                _pass(entries, "schedule_reminder", "single_turn", detail)
            else:
                _fail(entries, "schedule_reminder", "single_turn", detail or "reminder store missing")
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Woran willst du mich morgen um 12 Uhr erinnern?",
            )
            record("schedule_reminder_persistence", artifact)
            if _contains_any(artifact.answer, ("tabletten", "nehmen")):
                _pass(entries, "schedule_reminder", "multi_turn", "follow-up answer references stored reminder")
                if reminder_store.exists() and "Tabletten" in reminder_store.read_text(encoding="utf-8"):
                    _pass(entries, "schedule_reminder", "persistence", "new runtime kept stored reminder")
                else:
                    _fail(entries, "schedule_reminder", "persistence", "reminder store missing after restart")
            else:
                _fail(entries, "schedule_reminder", "multi_turn", "follow-up answer did not recall stored reminder")
                _fail(entries, "schedule_reminder", "persistence", "new runtime did not recall stored reminder")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Erstelle bitte eine taegliche Automation namens Morgenwetter fuer 8 Uhr morgens. Sie soll mir den Wetterbericht vorlesen und dafuer im Web suchen.",
            )
            record("create_time_automation_single", artifact)
            ok, detail = _assert_tools(artifact, required=("create_time_automation",))
            automation_records = list(loop.runtime.list_automation_records())
            if ok and any(record["name"] == "Morgenwetter" for record in automation_records):
                _pass(entries, "create_time_automation", "single_turn", detail)
            else:
                _fail(entries, "create_time_automation", "single_turn", detail or "automation not created")

            artifact = run_text_turn(
                loop,
                "Welche Automatisierungen hast du aktuell?",
            )
            record("list_automations_after_time_create", artifact)
            ok, detail = _assert_tools(artifact, required=("list_automations",))
            if ok and bool(artifact.answer.strip()):
                _pass(entries, "list_automations", "single_turn", detail)
            else:
                _fail(entries, "list_automations", "single_turn", detail or "list answer missing created automation")

            artifact = run_text_turn(
                loop,
                "Aendere die Automation Morgenwetter bitte auf 9 Uhr 15 und lass sie statt sprechen drucken.",
            )
            record("update_time_automation_multi", artifact)
            ok, detail = _assert_tools(
                artifact,
                required=("update_time_automation",),
                allowed=("list_automations", "update_time_automation"),
            )
            updated_records = list(loop.runtime.list_automation_records())
            updated_record = next((record for record in updated_records if record["name"] == "Morgenwetter"), None)
            if ok and updated_record is not None and updated_record.get("time_of_day") == "09:15" and _automation_delivery(updated_record) == "printed":
                _pass(entries, "update_time_automation", "single_turn", detail)
                _pass(entries, "create_time_automation", "multi_turn", "created automation could be updated in a follow-up turn")
                _pass(entries, "update_time_automation", "multi_turn", detail)
            else:
                if ok and _contains_any(artifact.answer, ("09:15", "druckt", "gedruckt")):
                    _pass(entries, "update_time_automation", "single_turn", detail)
                    _pass(entries, "update_time_automation", "multi_turn", detail)
                else:
                    _fail(entries, "update_time_automation", "single_turn", detail or "time automation not updated")
                    _fail(entries, "update_time_automation", "multi_turn", detail or "time automation follow-up update failed")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Welche Automatisierungen hast du aktuell?",
            )
            record("list_automations_time_persistence", artifact)
            ok, detail = _assert_tools(artifact, required=("list_automations",))
            if ok and _contains_any(artifact.answer, ("morgenwetter", "09:15", "09.15", "gedruckt", "druckt")):
                _pass(entries, "create_time_automation", "persistence", "created automation survived runtime restart")
                _pass(entries, "list_automations", "multi_turn", "list tool worked after create/update sequence")
                _pass(entries, "list_automations", "persistence", "list tool worked after runtime restart")
                _pass(entries, "update_time_automation", "persistence", "updated time automation survived runtime restart")
            else:
                _fail(entries, "create_time_automation", "persistence", "created automation missing after restart")
                _fail(entries, "list_automations", "persistence", "list did not show automation after restart")
                _fail(entries, "update_time_automation", "persistence", "updated time automation missing after restart")

            artifact = run_text_turn(
                loop,
                "Loesch die Automation Morgenwetter bitte wieder.",
            )
            record("delete_time_automation_multi", artifact)
            ok, detail = _assert_tools(
                artifact,
                required=("delete_automation",),
                allowed=("list_automations", "delete_automation"),
            )
            if ok and not any(record["name"] == "Morgenwetter" for record in loop.runtime.list_automation_records()):
                _pass(entries, "delete_automation", "single_turn", detail)
                _pass(entries, "delete_automation", "multi_turn", detail)
            else:
                _fail(entries, "delete_automation", "single_turn", detail or "delete failed")
                _fail(entries, "delete_automation", "multi_turn", detail or "delete follow-up failed")
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Welche Automatisierungen hast du aktuell?")
            record("delete_time_automation_persistence", artifact)
            if not any(record["name"] == "Morgenwetter" for record in loop.runtime.list_automation_records()):
                _pass(entries, "delete_automation", "persistence", "deleted automation stayed deleted after restart")
            else:
                _fail(entries, "delete_automation", "persistence", "deleted automation reappeared after restart")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Erstelle bitte eine Sensor-Automation namens Besuchergruss. Wenn die Kamera eine Person sieht, soll Twinr freundlich Hallo sagen.",
            )
            record("create_sensor_automation_single", artifact)
            ok, detail = _assert_tools(artifact, required=("create_sensor_automation",))
            sensor_records = list(loop.runtime.list_automation_records())
            created_sensor = next((record for record in sensor_records if record["name"] == "Besuchergruss"), None)
            if ok and created_sensor is not None and created_sensor.get("sensor_trigger_kind") == "camera_person_visible":
                _pass(entries, "create_sensor_automation", "single_turn", detail)
            else:
                _fail(entries, "create_sensor_automation", "single_turn", detail or "sensor automation not created")

            artifact = run_text_turn(loop, "Welche Automatisierungen hast du aktuell?")
            record("list_automations_after_sensor_create", artifact)
            ok, detail = _assert_tools(artifact, required=("list_automations",))
            if ok and bool(artifact.answer.strip()):
                _pass(entries, "list_automations", "single_turn", "listed sensor automation after creation")
            else:
                _fail(entries, "list_automations", "single_turn", "list tool did not describe sensor automation")

            artifact = run_text_turn(
                loop,
                "Aendere Besuchergruss bitte so, dass er erst nach 45 Sekunden Ruhe ausloest und den Hinweis dann druckt.",
            )
            record("update_sensor_automation_multi", artifact)
            ok, detail = _assert_tools(
                artifact,
                required=("update_sensor_automation",),
                allowed=("list_automations", "update_sensor_automation"),
            )
            updated_sensor_records = list(loop.runtime.list_automation_records())
            updated_sensor = next((record for record in updated_sensor_records if record["name"] == "Besuchergruss"), None)
            if ok and _sensor_automation_matches(
                updated_sensor,
                trigger_kind="pir_no_motion",
                hold_seconds=45.0,
                delivery="printed",
            ):
                _pass(entries, "update_sensor_automation", "single_turn", detail)
                _pass(entries, "create_sensor_automation", "multi_turn", "sensor automation could be updated in a follow-up turn")
                _pass(entries, "update_sensor_automation", "multi_turn", detail)
            else:
                _fail(entries, "update_sensor_automation", "single_turn", detail or "sensor automation not updated")
                _fail(entries, "update_sensor_automation", "multi_turn", detail or "sensor automation follow-up update failed")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Welche Automatisierungen hast du aktuell?")
            record("sensor_automation_persistence", artifact)
            ok, detail = _assert_tools(artifact, required=("list_automations",))
            if ok and _contains_any(artifact.answer, ("besuchergruss", "kamera", "person")):
                _pass(entries, "create_sensor_automation", "persistence", "sensor automation survived runtime restart")
                if _contains_any(artifact.answer, ("45", "ruhe", "druck")):
                    _pass(entries, "update_sensor_automation", "persistence", "updated sensor automation survived runtime restart")
                else:
                    _fail(entries, "update_sensor_automation", "persistence", "updated sensor automation missing after restart")
            else:
                _fail(entries, "create_sensor_automation", "persistence", "sensor automation missing after restart")
                _fail(entries, "update_sensor_automation", "persistence", "updated sensor automation missing after restart")

            artifact = run_text_turn(loop, "Loesch Besuchergruss bitte wieder.")
            record("delete_sensor_automation_multi", artifact)
            ok, detail = _assert_tools(
                artifact,
                required=("delete_automation",),
                allowed=("list_automations", "delete_automation"),
            )
            if ok and not any(record["name"] == "Besuchergruss" for record in loop.runtime.list_automation_records()):
                _pass(entries, "delete_automation", "single_turn", "sensor automation delete succeeded")
                _pass(entries, "delete_automation", "multi_turn", "sensor automation delete succeeded in follow-up turn")
            else:
                _fail(entries, "delete_automation", "single_turn", "sensor automation delete failed")
                _fail(entries, "delete_automation", "multi_turn", "sensor automation delete failed in follow-up turn")

        if group_enabled("memory_profile"):
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Bitte merk dir als wichtige Information fuer spaeter: Mein Hausschluessel liegt im blauen Kasten im Flur.",
            )
            record("remember_memory_single", artifact)
            ok, detail = _assert_tools(artifact, required=("remember_memory",))
            if ok:
                _pass(entries, "remember_memory", "single_turn", detail)
            else:
                _fail(entries, "remember_memory", "single_turn", detail or "remember_memory tool missing")
            artifact = run_text_turn(loop, "Wo liegt mein Hausschluessel?")
            record("remember_memory_multi", artifact)
            if _contains_any(artifact.answer, ("blauen kasten", "flur")):
                _pass(entries, "remember_memory", "multi_turn", "same-session recall succeeded")
            else:
                _fail(entries, "remember_memory", "multi_turn", "same-session recall failed")
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Wo liegt mein Hausschluessel?")
            record("remember_memory_persistence", artifact)
            if _contains_any(artifact.answer, ("blauen kasten", "flur")):
                _pass(entries, "remember_memory", "persistence", "restart recall succeeded")
            else:
                _fail(entries, "remember_memory", "persistence", "restart recall failed")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                f"Bitte merk dir als Kontakt: Meine Tochter Anna Schulz hat die Telefonnummer {_TEST_ANNA_PHONE}.",
            )
            record("remember_contact_single", artifact)
            ok, detail = _assert_tools(artifact, required=("remember_contact",))
            if ok:
                _pass(entries, "remember_contact", "single_turn", detail)
            else:
                _fail(entries, "remember_contact", "single_turn", detail)
            artifact = run_text_turn(loop, "Wie ist die Telefonnummer von Anna Schulz?")
            record("lookup_contact_multi", artifact)
            ok, detail = _assert_tools(artifact, required=("lookup_contact",))
            if ok and _contains_any(artifact.answer, _TEST_ANNA_PHONE_NEEDLES):
                _pass(entries, "lookup_contact", "single_turn", detail)
                _pass(entries, "remember_contact", "multi_turn", "contact lookup succeeded in same session")
                _pass(entries, "lookup_contact", "multi_turn", detail)
            else:
                _fail(entries, "lookup_contact", "single_turn", detail or "lookup failed")
                _fail(entries, "lookup_contact", "multi_turn", detail or "lookup follow-up failed")
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Wie ist die Telefonnummer von Anna Schulz?")
            record("lookup_contact_persistence", artifact)
            ok, detail = _assert_tools(artifact, required=("lookup_contact",))
            if ok and _contains_any(artifact.answer, _TEST_ANNA_PHONE_NEEDLES):
                _pass(entries, "remember_contact", "persistence", "contact survived restart")
                _pass(entries, "lookup_contact", "persistence", detail)
            else:
                _fail(entries, "remember_contact", "persistence", "contact missing after restart")
                _fail(entries, "lookup_contact", "persistence", detail or "lookup failed after restart")

            loop = context.make_loop(emitted=[])
            context.seed_conflict(loop=loop)
            artifact = run_text_turn(loop, "Gibt es bei Corinna Maier offene Erinnerungskonflikte?")
            record("get_memory_conflicts_single", artifact)
            ok, detail = _assert_tools(artifact, required=("get_memory_conflicts",))
            if ok and _contains_any(artifact.answer, ("corinna", "nummer", "konflikt")):
                _pass(entries, "get_memory_conflicts", "single_turn", detail)
                _pass(entries, "get_memory_conflicts", "persistence", "tool read seeded persisted conflicts")
            else:
                _fail(entries, "get_memory_conflicts", "single_turn", detail or "conflicts not surfaced")
                _fail(entries, "get_memory_conflicts", "persistence", "persisted conflicts not surfaced")
            artifact = run_text_turn(loop, "Bitte nutze fuer Corinna Maier die Nummer 040 998877.")
            record("resolve_memory_conflict_multi", artifact)
            ok, detail = _assert_tools(
                artifact,
                required=("resolve_memory_conflict",),
                allowed=("get_memory_conflicts", "resolve_memory_conflict"),
            )
            if ok:
                _pass(entries, "resolve_memory_conflict", "single_turn", detail)
                _pass(entries, "get_memory_conflicts", "multi_turn", "conflict listing enabled follow-up resolution")
                _pass(entries, "resolve_memory_conflict", "multi_turn", detail)
            else:
                _fail(entries, "resolve_memory_conflict", "single_turn", detail)
                _fail(entries, "resolve_memory_conflict", "multi_turn", detail)
            loop = context.make_loop(emitted=[])
            conflicts_after = loop.runtime.long_term_memory.object_store.load_conflicts()
            objects_after = {
                item.memory_id: item
                for item in loop.runtime.long_term_memory.object_store.load_objects()
            }
            if not conflicts_after and objects_after["fact:corinna_phone_new"].status == "active":
                _pass(entries, "resolve_memory_conflict", "persistence", "resolution survived restart")
            else:
                _fail(entries, "resolve_memory_conflict", "persistence", "resolution did not persist")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Bitte merk dir als dauerhafte Vorliebe: Ich mag Melitta Kaffee.",
            )
            record("remember_preference_single", artifact)
            ok, detail = _assert_tools(artifact, required=("remember_preference",))
            if ok:
                _pass(entries, "remember_preference", "single_turn", detail)
            else:
                _fail(entries, "remember_preference", "single_turn", detail)
            artifact = run_text_turn(loop, "Welche Kaffeemarke mag ich?")
            record("remember_preference_multi", artifact)
            if _contains_any(artifact.answer, ("melitta",)):
                _pass(entries, "remember_preference", "multi_turn", "same-session preference recall succeeded")
            else:
                _fail(entries, "remember_preference", "multi_turn", "same-session preference recall failed")
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Welche Kaffeemarke mag ich?")
            record("remember_preference_persistence", artifact)
            if _contains_any(artifact.answer, ("melitta",)):
                _pass(entries, "remember_preference", "persistence", "preference survived restart")
            else:
                _fail(entries, "remember_preference", "persistence", "preference missing after restart")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Bitte merk dir fuer spaeter: Ich will heute noch spazieren gehen.",
            )
            record("remember_plan_single", artifact)
            ok, detail = _assert_tools(artifact, required=("remember_plan",))
            if ok:
                _pass(entries, "remember_plan", "single_turn", detail)
            else:
                _fail(entries, "remember_plan", "single_turn", detail)
            artifact = run_text_turn(loop, "Was habe ich heute noch vor?")
            record("remember_plan_multi", artifact)
            if _contains_any(artifact.answer, ("spazieren", "spaziergang", "walk")):
                _pass(entries, "remember_plan", "multi_turn", "same-session plan recall succeeded")
            else:
                _fail(entries, "remember_plan", "multi_turn", "same-session plan recall failed")
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Was habe ich heute noch vor?")
            record("remember_plan_persistence", artifact)
            if _contains_any(artifact.answer, ("spazieren", "spaziergang", "walk")):
                _pass(entries, "remember_plan", "persistence", "plan survived restart")
            else:
                _fail(entries, "remember_plan", "persistence", "plan missing after restart")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Bitte aktualisiere mein Benutzerprofil fuer kuenftige Gespraeche: Ich wohne in Schwarzenbek bei Hamburg.",
            )
            record("update_user_profile_single", artifact)
            ok, detail = _assert_tools(artifact, required=("update_user_profile",))
            if ok:
                _pass(entries, "update_user_profile", "single_turn", detail)
            else:
                _fail(entries, "update_user_profile", "single_turn", detail)
            artifact = run_text_turn(loop, "Wo wohne ich?")
            record("update_user_profile_multi", artifact)
            if _contains_any(artifact.answer, ("schwarzenbek", "hamburg")):
                _pass(entries, "update_user_profile", "multi_turn", "same-session profile recall succeeded")
            else:
                _fail(entries, "update_user_profile", "multi_turn", "same-session profile recall failed")
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Wo wohne ich?")
            record("update_user_profile_persistence", artifact)
            if _contains_any(artifact.answer, ("schwarzenbek", "hamburg")):
                _pass(entries, "update_user_profile", "persistence", "profile survived restart")
            else:
                _fail(entries, "update_user_profile", "persistence", "profile missing after restart")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Bitte aendere dein zukuenftiges Verhalten: Antworte kuenftig sehr kurz und ruhig.",
            )
            record("update_personality_single", artifact)
            ok, detail = _assert_tools(artifact, required=("update_personality",))
            personality_entries = loop.runtime.long_term_memory.prompt_context_store.personality_store.load_entries()
            if ok and _managed_context_contains(
                personality_entries,
                expected_fragments_by_key={
                    "response_style": ("very brief", "very short", "short and calm", "kurz und ruhig"),
                },
            ):
                _pass(entries, "update_personality", "single_turn", detail)
            else:
                _fail(entries, "update_personality", "single_turn", detail or "managed personality context not updated")
            loop = context.make_loop(emitted=[])
            personality_entries = loop.runtime.long_term_memory.prompt_context_store.personality_store.load_entries()
            if _managed_context_contains(
                personality_entries,
                expected_fragments_by_key={
                    "response_style": ("very brief", "very short", "short and calm", "kurz und ruhig"),
                },
            ):
                _pass(entries, "update_personality", "persistence", "managed personality context survived restart")
            else:
                _fail(entries, "update_personality", "persistence", "managed personality context missing after restart")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Du bist vergesslich. Bitte merk dir kuenftig mehr vom Gespraech.",
            )
            record("update_simple_setting_single", artifact)
            ok, detail = _assert_tools(artifact, required=("update_simple_setting",))
            env_text = context.env_path.read_text(encoding="utf-8")
            if ok and "TWINR_MEMORY_MAX_TURNS=28" in env_text and "TWINR_MEMORY_KEEP_RECENT=12" in env_text:
                _pass(entries, "update_simple_setting", "single_turn", detail)
                _pass(entries, "update_simple_setting", "persistence", "bounded setting persisted to .env")
            else:
                _fail(entries, "update_simple_setting", "single_turn", detail or ".env not updated")
                _fail(entries, "update_simple_setting", "persistence", "updated simple setting missing after restart")

        voice_pcm = _voice_sample_pcm_bytes()
        if group_enabled("voice_profile"):
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Bitte lege jetzt ein lokales Stimmprofil von mir an.",
                current_turn_audio_pcm=voice_pcm,
            )
            record("enroll_voice_profile_single", artifact)
            ok, detail = _assert_tools(artifact, required=("enroll_voice_profile",))
            if ok:
                _pass(entries, "enroll_voice_profile", "single_turn", detail)
            else:
                _fail(entries, "enroll_voice_profile", "single_turn", detail)
            artifact = run_text_turn(loop, "Hast du aktuell ein lokales Stimmprofil von mir gespeichert?")
            record("get_voice_profile_status_multi", artifact)
            ok, detail = _assert_tools(artifact, required=("get_voice_profile_status",))
            if ok and _contains_any(artifact.answer, ("stimmprofil", "voice profile", "gespeichert")):
                _pass(entries, "get_voice_profile_status", "single_turn", detail)
                _pass(entries, "enroll_voice_profile", "multi_turn", "status confirmed after enrollment")
                _pass(entries, "get_voice_profile_status", "multi_turn", detail)
            else:
                _fail(entries, "get_voice_profile_status", "single_turn", detail or "status lookup failed")
                _fail(entries, "get_voice_profile_status", "multi_turn", detail or "status lookup failed after enrollment")
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Hast du aktuell ein lokales Stimmprofil von mir gespeichert?")
            record("get_voice_profile_status_persistence", artifact)
            ok, detail = _assert_tools(artifact, required=("get_voice_profile_status",))
            enrolled_after_restart = loop.voice_profile_monitor.summary().enrolled
            if enrolled_after_restart and _voice_profile_status_answer_matches(artifact.answer, enrolled=True):
                _pass(entries, "enroll_voice_profile", "persistence", "voice profile survived restart")
                _pass(
                    entries,
                    "get_voice_profile_status",
                    "persistence",
                    detail if ok else "status answered correctly from persisted voice-profile state without explicit tool call",
                )
            else:
                _fail(entries, "enroll_voice_profile", "persistence", "voice profile missing after restart")
                _fail(entries, "get_voice_profile_status", "persistence", detail or "status missing after restart")
            artifact = run_text_turn(loop, "Bitte loesche mein gespeichertes Stimmprofil wieder.")
            record("reset_voice_profile_single", artifact)
            ok, detail = _assert_tools(artifact, required=("reset_voice_profile",))
            if ok:
                _pass(entries, "reset_voice_profile", "single_turn", detail)
                _pass(entries, "reset_voice_profile", "multi_turn", detail)
            else:
                _fail(entries, "reset_voice_profile", "single_turn", detail)
                _fail(entries, "reset_voice_profile", "multi_turn", detail)
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Hast du aktuell ein lokales Stimmprofil von mir gespeichert?")
            record("reset_voice_profile_persistence", artifact)
            ok, detail = _assert_tools(artifact, required=("get_voice_profile_status",))
            if not loop.voice_profile_monitor.summary().enrolled and _voice_profile_status_answer_matches(artifact.answer, enrolled=False):
                _pass(entries, "reset_voice_profile", "persistence", "voice profile stayed deleted after restart")
            else:
                _fail(entries, "reset_voice_profile", "persistence", "voice profile reappeared after restart")

        if group_enabled("discovery_world"):
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Ich moechte, dass du mich besser kennenlernst und deinen Kennenlernmodus startest.",
            )
            record("manage_user_discovery_single", artifact)
            ok, detail = _assert_tools(artifact, required=("manage_user_discovery",))
            if ok:
                _pass(entries, "manage_user_discovery", "single_turn", detail)
            else:
                _fail(entries, "manage_user_discovery", "single_turn", detail or "discovery flow did not start")
            artifact = run_text_turn(
                loop,
                "Auf deine Kennenlernfrage: Ich trinke morgens gern schwarzen Kaffee und starte lieber ruhig.",
            )
            record("manage_user_discovery_multi", artifact)
            ok, detail = _assert_tools(artifact, required=("manage_user_discovery",))
            review_result = loop.runtime.manage_user_discovery(action="review_profile")
            if ok and _user_discovery_review_items_contain(
                review_result.review_items,
                ("coffee", "kaffee", "quiet", "ruhig", "morning", "morgens"),
            ):
                _pass(entries, "manage_user_discovery", "multi_turn", detail)
            else:
                _fail(entries, "manage_user_discovery", "multi_turn", detail or "discovery answer was not committed")
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Was hast du bisher im Kennenlernmodus ueber mich gelernt?")
            record("manage_user_discovery_persistence", artifact)
            ok, detail = _assert_tools(artifact, required=("manage_user_discovery",))
            if ok and _contains_any(artifact.answer, ("kaffee", "ruhig", "morgens")):
                _pass(entries, "manage_user_discovery", "persistence", detail)
            else:
                _fail(entries, "manage_user_discovery", "persistence", detail or "discovery profile missing after restart")

            loop = context.make_loop(emitted=[])
            context.install_world_intelligence_stubs(loop)
            artifact = run_text_turn(
                loop,
                "Bitte richte dauerhaft RSS-Quellen fuer Hamburger Lokalpolitik ein und aktualisiere sie sofort.",
            )
            record("configure_world_intelligence_single", artifact)
            ok, detail = _assert_tools(artifact, required=("configure_world_intelligence",))
            if ok and bool(context.world_remote_state.snapshots):
                _pass(entries, "configure_world_intelligence", "single_turn", detail)
            else:
                _fail(entries, "configure_world_intelligence", "single_turn", detail or "world intelligence tool did not persist subscriptions")
            artifact = run_text_turn(
                loop,
                "Welche gespeicherten Weltquellen fuer Hamburger Lokalpolitik hast du jetzt aktuell?",
            )
            record("configure_world_intelligence_multi", artifact)
            ok, detail = _assert_tools(artifact, required=("configure_world_intelligence",))
            if ok and _contains_any(artifact.answer, ("hamburg", "rss", "quelle", "feed")):
                _pass(entries, "configure_world_intelligence", "multi_turn", detail)
            else:
                _fail(entries, "configure_world_intelligence", "multi_turn", detail or "world intelligence follow-up did not describe saved feeds")
            loop = context.make_loop(emitted=[])
            context.install_world_intelligence_stubs(loop)
            artifact = run_text_turn(
                loop,
                "Welche gespeicherten Weltquellen fuer Hamburger Lokalpolitik hast du weiterhin gespeichert?",
            )
            record("configure_world_intelligence_persistence", artifact)
            ok, detail = _assert_tools(artifact, required=("configure_world_intelligence",))
            if ok and bool(context.world_remote_state.snapshots):
                _pass(entries, "configure_world_intelligence", "persistence", detail)
            else:
                _fail(entries, "configure_world_intelligence", "persistence", detail or "world intelligence subscriptions missing after restart")

        if group_enabled("local_identity"):
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Bitte aktualisiere dein lokales Gesichtsprofil von mir.")
            record("enroll_portrait_identity_single", artifact)
            ok, detail = _assert_tools(artifact, required=("enroll_portrait_identity",))
            if ok and context.portrait_provider.reference_image_count >= 1:
                _pass(entries, "enroll_portrait_identity", "single_turn", detail)
            else:
                _fail(entries, "enroll_portrait_identity", "single_turn", detail or "portrait enrollment failed")
            artifact = run_text_turn(loop, "Hast du aktuell ein lokales Gesichtsprofil von mir gespeichert?")
            record("get_portrait_identity_status_multi", artifact)
            ok, detail = _assert_tools(artifact, required=("get_portrait_identity_status",))
            if ok and context.portrait_provider.reference_image_count >= 1:
                _pass(entries, "get_portrait_identity_status", "single_turn", detail)
                _pass(entries, "enroll_portrait_identity", "multi_turn", "portrait enrollment was readable immediately afterwards")
                _pass(entries, "get_portrait_identity_status", "multi_turn", detail)
            else:
                _fail(entries, "get_portrait_identity_status", "single_turn", detail or "portrait status failed")
                _fail(entries, "get_portrait_identity_status", "multi_turn", detail or "portrait status follow-up failed")
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Hast du aktuell ein lokales Gesichtsprofil von mir gespeichert?")
            record("get_portrait_identity_status_persistence", artifact)
            ok, detail = _assert_tools(artifact, required=("get_portrait_identity_status",))
            if ok and context.portrait_provider.reference_image_count >= 1:
                _pass(entries, "enroll_portrait_identity", "persistence", "portrait profile survived restart")
                _pass(entries, "get_portrait_identity_status", "persistence", detail)
            else:
                _fail(entries, "enroll_portrait_identity", "persistence", "portrait profile missing after restart")
                _fail(entries, "get_portrait_identity_status", "persistence", detail or "portrait status missing after restart")
            artifact = run_text_turn(loop, "Bitte loesche dein lokales Gesichtsprofil von mir wieder.")
            record("reset_portrait_identity_single", artifact)
            ok, detail = _assert_tools(artifact, required=("reset_portrait_identity",))
            if ok and context.portrait_provider.reference_image_count == 0:
                _pass(entries, "reset_portrait_identity", "single_turn", detail)
            else:
                _fail(entries, "reset_portrait_identity", "single_turn", detail or "portrait reset failed")
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Hast du aktuell ein lokales Gesichtsprofil von mir gespeichert?")
            record("reset_portrait_identity_persistence", artifact)
            ok, detail = _assert_tools(artifact, required=("get_portrait_identity_status",))
            if ok and context.portrait_provider.reference_image_count == 0 and _contains_any(artifact.answer, ("kein", "nicht", "no")):
                _pass(entries, "reset_portrait_identity", "persistence", "portrait profile stayed deleted after restart")
            else:
                _fail(entries, "reset_portrait_identity", "persistence", detail or "portrait profile reappeared after restart")

            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(
                loop,
                "Bitte speichere mich in deiner lokalen Haushalts-Identitaet mit meinem Gesicht als Thom.",
            )
            record("manage_household_identity_face_single", artifact)
            ok, detail = _assert_tools(artifact, required=("manage_household_identity",))
            if ok:
                _pass(entries, "manage_household_identity", "single_turn", detail)
            else:
                _fail(entries, "manage_household_identity", "single_turn", detail or "household identity face enrollment failed")
            artifact = run_text_turn(
                loop,
                "Bitte speichere mich in deiner lokalen Haushalts-Identitaet jetzt auch mit meiner Stimme.",
                current_turn_audio_pcm=voice_pcm,
            )
            record("manage_household_identity_voice_multi", artifact)
            ok, detail = _assert_tools(artifact, required=("manage_household_identity",))
            if ok:
                _pass(entries, "manage_household_identity", "multi_turn", detail)
            else:
                _fail(entries, "manage_household_identity", "multi_turn", detail or "household identity voice enrollment failed")
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Wen erkennst du aktuell in deiner lokalen Haushalts-Identitaet?")
            record("manage_household_identity_persistence", artifact)
            ok, detail = _assert_tools(artifact, required=("manage_household_identity",))
            if ok and _contains_any(artifact.answer, ("thom", "haushalt", "identitaet", "stimme", "gesicht")):
                _pass(entries, "manage_household_identity", "persistence", detail)
            else:
                _fail(entries, "manage_household_identity", "persistence", detail or "household identity status missing after restart")

        if group_enabled("smart_home"):
            loop = context.make_loop(emitted=[])
            artifact = run_text_turn(loop, "Welche Smart-Home-Geraete kennst du aktuell?")
            record("list_smart_home_entities_single", artifact)
            ok, detail = _assert_tools(artifact, required=("list_smart_home_entities",))
            if ok:
                _pass(entries, "list_smart_home_entities", "single_turn", detail)
            else:
                _fail(entries, "list_smart_home_entities", "single_turn", detail or "smart-home list failed")
            artifact = run_text_turn(
                loop,
                "Lies bitte den exakten Smart-Home-Zustand von light.living_room und light.hallway.",
            )
            record("read_smart_home_state_single", artifact)
            ok, detail = _assert_tools(artifact, required=("read_smart_home_state",))
            if ok and context.smart_home_provider.last_read_entity_ids == ("light.living_room", "light.hallway"):
                _pass(entries, "read_smart_home_state", "single_turn", detail)
            else:
                _fail(entries, "read_smart_home_state", "single_turn", detail or "smart-home exact state read failed")
            artifact = run_text_turn(loop, "Bitte schalte das Smart-Home-Geraet light.hallway jetzt ein.")
            record("control_smart_home_entities_single", artifact)
            ok, detail = _assert_tools(artifact, required=("control_smart_home_entities",))
            if ok and context.smart_home_provider.state_for("light.hallway").get("power") == "on":
                _pass(entries, "control_smart_home_entities", "single_turn", detail)
            else:
                _fail(entries, "control_smart_home_entities", "single_turn", detail or "smart-home control did not update state")
            artifact = run_text_turn(loop, "Welche Smart-Home-Ereignisse gab es zuletzt im Flur?")
            record("read_smart_home_sensor_stream_single", artifact)
            ok, detail = _assert_tools(
                artifact,
                required=("read_smart_home_sensor_stream",),
                allowed=("list_smart_home_entities",),
            )
            if ok and context.smart_home_provider.last_sensor_limit >= 1:
                _pass(entries, "read_smart_home_sensor_stream", "single_turn", detail)
            else:
                _fail(entries, "read_smart_home_sensor_stream", "single_turn", detail or "smart-home event stream failed")

        if group_enabled("self_coding"):
            loop = context.make_loop(emitted=[])
            context.install_self_coding_driver(loop)
            artifact = run_text_turn(
                loop,
                "Twinr, lerne bitte einen neuen wiederholbaren Skill namens Announce Family Updates: Wenn neue Familiennachrichten eingehen, lies sie mir vor.",
            )
            record("propose_skill_learning_single", artifact)
            ok, detail = _assert_tools(artifact, required=("propose_skill_learning",))
            if ok:
                _pass(entries, "propose_skill_learning", "single_turn", detail)
            else:
                _fail(entries, "propose_skill_learning", "single_turn", detail or "self-coding proposal failed")
            artifact = run_text_turn(loop, "Nur wenn ich sichtbar bin.")
            record("answer_skill_question_trigger", artifact)
            ok, detail = _assert_tools(artifact, required=("answer_skill_question",))
            if ok:
                _pass(entries, "answer_skill_question", "single_turn", detail)
            else:
                _fail(entries, "answer_skill_question", "single_turn", detail or "self-coding dialogue did not accept the first answer")
            artifact = run_text_turn(loop, "Nur fuer Familienkontakte.")
            record("answer_skill_question_scope", artifact)
            ok, detail = _assert_tools(artifact, required=("answer_skill_question",))
            if ok:
                _pass(entries, "answer_skill_question", "multi_turn", detail)
            else:
                _fail(entries, "answer_skill_question", "multi_turn", detail or "self-coding dialogue did not advance on the second answer")
            artifact = run_text_turn(loop, "Frag vorher kurz nach.")
            record("answer_skill_question_constraints", artifact)
            artifact = run_text_turn(loop, "Ja, genau so soll der neue Skill fuer mich sein.")
            record("answer_skill_question_confirm", artifact)
            self_coding_store = getattr(loop, "_self_coding_store", None)
            compile_worker = getattr(loop, "_self_coding_compile_worker", None)
            if self_coding_store is None or compile_worker is None:
                raise RuntimeError("self-coding runtime helpers are unavailable on the matrix loop")
            compile_jobs = self_coding_store.list_jobs()
            latest_job = compile_jobs[-1] if compile_jobs else None
            if latest_job is not None:
                compile_worker.run_job(latest_job.job_id)
            artifact = run_text_turn(loop, "Bitte aktiviere den gelernten Skill Announce Family Updates jetzt.")
            record("confirm_skill_activation_single", artifact)
            ok, detail = _assert_tools(artifact, required=("confirm_skill_activation",))
            activations = self_coding_store.list_activations(skill_id="announce_family_updates")
            active_versions = [item for item in activations if str(getattr(item, "status", "")) == "active"]
            if ok and active_versions:
                _pass(entries, "confirm_skill_activation", "single_turn", detail)
            else:
                _fail(entries, "confirm_skill_activation", "single_turn", detail or "self-coding activation failed")
            artifact = run_text_turn(loop, "Pausiere den gelernten Skill announce_family_updates in Version 1 bitte.")
            record("pause_skill_activation_single", artifact)
            ok, detail = _assert_tools(artifact, required=("pause_skill_activation",))
            paused, paused_detail = _safe_load_self_coding_activation(
                self_coding_store,
                skill_id="announce_family_updates",
                version=1,
            )
            if ok and paused is not None and str(getattr(paused, "status", "")) == "paused":
                _pass(entries, "pause_skill_activation", "single_turn", detail)
            else:
                _fail(
                    entries,
                    "pause_skill_activation",
                    "single_turn",
                    detail or paused_detail or "self-coding pause failed",
                )
            artifact = run_text_turn(loop, "Aktiviere den pausierten Skill announce_family_updates in Version 1 bitte wieder.")
            record("reactivate_skill_activation_single", artifact)
            ok, detail = _assert_tools(artifact, required=("reactivate_skill_activation",))
            reactivated, reactivated_detail = _safe_load_self_coding_activation(
                self_coding_store,
                skill_id="announce_family_updates",
                version=1,
            )
            if ok and reactivated is not None and str(getattr(reactivated, "status", "")) == "active":
                _pass(entries, "reactivate_skill_activation", "single_turn", detail)
                _pass(entries, "confirm_skill_activation", "multi_turn", "activated skill stayed usable for later lifecycle turns")
            else:
                _fail(
                    entries,
                    "reactivate_skill_activation",
                    "single_turn",
                    detail or reactivated_detail or "self-coding reactivate failed",
                )
            seeded_v2 = loop._handle_propose_skill_learning_tool_call(
                {
                    "name": "Announce Family Updates",
                    "action": "Read new family updates aloud",
                    "request_summary": "Read out new family updates.",
                    "capabilities": ["speaker", "safety", "rules"],
                    "trigger_mode": "push",
                    "trigger_conditions": ["new_message"],
                    "scope": {"contacts": ["family"]},
                    "constraints": ["ask_first"],
                }
            )
            session_id = seeded_v2["session_id"]
            loop._handle_answer_skill_question_tool_call({"session_id": session_id, "trigger_conditions": ["user_visible"]})
            loop._handle_answer_skill_question_tool_call({"session_id": session_id, "scope": {"contacts": ["family"]}})
            loop._handle_answer_skill_question_tool_call({"session_id": session_id, "constraints": ["ask_first"]})
            seeded_final = loop._handle_answer_skill_question_tool_call({"session_id": session_id, "confirmed": True})
            seeded_job_id = seeded_final["compile_job_id"]
            compile_worker.run_job(seeded_job_id)
            loop._handle_confirm_skill_activation_tool_call({"job_id": seeded_job_id, "confirmed": True})
            artifact = run_text_turn(loop, "Stelle den gelernten Skill announce_family_updates bitte wieder auf Version 1 zurueck.")
            record("rollback_skill_activation_single", artifact)
            ok, detail = _assert_tools(artifact, required=("rollback_skill_activation",))
            rollback_activation, rollback_detail = _safe_load_self_coding_activation(
                self_coding_store,
                skill_id="announce_family_updates",
                version=1,
            )
            if ok and rollback_activation is not None and str(getattr(rollback_activation, "status", "")) == "active":
                _pass(entries, "rollback_skill_activation", "single_turn", detail)
            else:
                _fail(
                    entries,
                    "rollback_skill_activation",
                    "single_turn",
                    detail or rollback_detail or "self-coding rollback failed",
                )
            loop = context.make_loop(emitted=[])
            context.install_self_coding_driver(loop)
            restart_self_coding_store = getattr(loop, "_self_coding_store", None)
            if restart_self_coding_store is None:
                raise RuntimeError("self-coding store is unavailable after restart")
            activation_records = restart_self_coding_store.list_activations(skill_id="announce_family_updates")
            if any(str(getattr(item, "status", "")) == "active" for item in activation_records):
                _pass(entries, "confirm_skill_activation", "persistence", "activated self-coding skill survived restart")
                _pass(entries, "reactivate_skill_activation", "persistence", "reactivated self-coding skill survived restart")
                _pass(entries, "rollback_skill_activation", "persistence", "rolled-back active version survived restart")
            else:
                _fail(entries, "confirm_skill_activation", "persistence", "activated self-coding skill missing after restart")
                _fail(entries, "reactivate_skill_activation", "persistence", "reactivated self-coding skill missing after restart")
                _fail(entries, "rollback_skill_activation", "persistence", "rolled-back active version missing after restart")

    finally:
        ToolMatrixContext.restore_smart_home_builders(smart_home_builder_originals)
        ToolMatrixContext.restore_service_connect_helpers(service_connect_originals)
        project_root = str(context.root)
        memory_markdown_path = str(context.state_dir / "MEMORY.md")
        reminder_store_path = str(context.state_dir / "reminders.json")
        automation_store_path = str(context.state_dir / "automations.json")
        voice_profile_store_path = str(context.state_dir / "voice_profile.json")

    failed_tools = sorted(name for name, entry in entries.items() if entry.overall() == "fail")
    passed_tools = sorted(name for name, entry in entries.items() if entry.overall() == "pass")
    return {
        "base_env_path": str(base_env_path.resolve()),
        "selected_groups": list(selected_groups),
        "planned_stack": {
            "stt_provider": "deepgram",
            "llm_provider": "openai",
            "tts_provider": "openai",
        },
        "tool_names": list(realtime_tool_names()),
        "tool_count": len(realtime_tool_names()),
        "project_root": project_root,
        "artifacts": {
            "memory_markdown_path": memory_markdown_path,
            "reminder_store_path": reminder_store_path,
            "automation_store_path": automation_store_path,
            "voice_profile_store_path": voice_profile_store_path,
        },
        "tools": {
            name: {
                "single_turn": entries[name].single_turn.__dict__,
                "multi_turn": entries[name].multi_turn.__dict__,
                "persistence": entries[name].persistence.__dict__,
                "overall": entries[name].overall(),
            }
            for name in sorted(entries)
        },
        "summary": {
            "passed_tool_count": len(passed_tools),
            "failed_tool_count": len(failed_tools),
            "passed_tools": passed_tools,
            "failed_tools": failed_tools,
        },
        "scenarios": scenario_results,
        "printer_outputs": context.printer.printed if hasattr(context, "printer") else [],
    }


def main() -> int:
    """Run the live matrix from CLI and optionally write one JSON artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-env", type=Path, default=Path(".env"))
    parser.add_argument("--group", action="append", choices=available_matrix_groups(), default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = run_matrix(args.base_env, groups=tuple(args.group) if args.group else None)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
