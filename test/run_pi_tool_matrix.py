from __future__ import annotations

from dataclasses import dataclass, field
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import AgentToolCall
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.agent.tools import (
    build_tool_agent_instructions,
    realtime_tool_names,
)
from twinr.agent.workflows.streaming_runner import TwinrStreamingHardwareLoop
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
)


ONE_BY_ONE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl9l2sAAAAASUVORK5CYII="
)


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
        self.state_dir = self.root / "state"
        self.personality_dir = self.root / "personality"
        self.reference_image_path = self.root / "user-reference.png"
        self.env_path = self.root / ".env"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.base_env_path.parent / "personality", self.personality_dir, dirs_exist_ok=True)
        self.reference_image_path.write_bytes(ONE_BY_ONE_PNG)
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
            "TWINR_LONG_TERM_MEMORY_ENABLED=true",
            "TWINR_LONG_TERM_MEMORY_BACKGROUND_STORE_TURNS=false",
            "TWINR_PROACTIVE_ENABLED=false",
            "TWINR_WAKEWORD_ENABLED=false",
            "TWINR_CONVERSATION_FOLLOW_UP_ENABLED=false",
            "TWINR_SEARCH_FEEDBACK_TONES_ENABLED=false",
        ]
        self.env_path.write_text(base_env_text.rstrip() + "\n" + "\n".join(overrides) + "\n", encoding="utf-8")
        self.printer = CapturingPrinter()
        self.player = SilentPlayer()
        self.camera = StaticCamera(ONE_BY_ONE_PNG)

    def close(self) -> None:
        self.temp_dir.cleanup()

    def make_loop(self, *, emitted: list[str]) -> TwinrStreamingHardwareLoop:
        config = TwinrConfig.from_env(self.env_path)
        runtime = TwinrRuntime(config=config)
        return TwinrStreamingHardwareLoop(
            config=config,
            runtime=runtime,
            player=self.player,
            printer=self.printer,
            camera=self.camera,
            button_monitor=SimpleNamespace(),
            proactive_monitor=SimpleNamespace(),
            emit=emitted.append,
        )

    def seed_conflict(self) -> None:
        emitted: list[str] = []
        loop = self.make_loop(emitted=emitted)
        existing = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_old",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +491761234.",
            source=_longterm_source("turn:1"),
            status="active",
            confidence=0.95,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+491761234",
        )
        candidate = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_new",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +4940998877.",
            source=_longterm_source("turn:2"),
            status="uncertain",
            confidence=0.92,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+4940998877",
        )
        conflict = LongTermMemoryConflictV1(
            slot_key="contact:person:corinna_maier:phone",
            candidate_memory_id="fact:corinna_phone_new",
            existing_memory_ids=("fact:corinna_phone_old",),
            question="Which phone number should I use for Corinna Maier?",
            reason="Conflicting phone numbers exist.",
        )
        loop.runtime.long_term_memory.object_store.apply_consolidation(
            LongTermConsolidationResultV1(
                turn_id="turn:2",
                occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                episodic_objects=(),
                durable_objects=(existing,),
                deferred_objects=(candidate,),
                conflicts=(conflict,),
                graph_edges=(),
            )
        )


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
    allowed_set = set(allowed or required)
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


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    haystack = text.lower()
    return any(needle.lower() in haystack for needle in needles)


def run_matrix(base_env_path: Path) -> dict[str, object]:
    entries = {name: MatrixEntry() for name in realtime_tool_names()}
    for name in entries:
        for dimension in ("single_turn", "multi_turn", "persistence"):
            _na(entries, name, dimension)
    scenario_results: list[dict[str, object]] = []
    context = ToolMatrixContext(base_env_path=base_env_path)
    try:
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
        automations_path = Path(loop.config.automation_store_path)
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
        if ok and updated_sensor is not None and updated_sensor.get("sensor_trigger_kind") == "vad_quiet" and updated_sensor.get("sensor_hold_seconds") == 45.0 and _automation_delivery(updated_sensor) == "printed":
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

        loop = context.make_loop(emitted=[])
        artifact = run_text_turn(
            loop,
            "Bitte merk dir als wichtige Information fuer spaeter: Mein Hausschluessel liegt im blauen Kasten im Flur.",
        )
        record("remember_memory_single", artifact)
        ok, detail = _assert_tools(artifact, required=("remember_memory",))
        memory_path = Path(loop.config.memory_markdown_path)
        if ok and memory_path.exists():
            _pass(entries, "remember_memory", "single_turn", detail)
        else:
            _fail(entries, "remember_memory", "single_turn", detail or "MEMORY.md missing")
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
            "Bitte merk dir als Kontakt: Meine Tochter Anna Schulz hat die Telefonnummer 040 1234567.",
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
        if ok and _contains_any(artifact.answer, ("040", "1234567")):
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
        if ok and _contains_any(artifact.answer, ("040", "1234567")):
            _pass(entries, "remember_contact", "persistence", "contact survived restart")
            _pass(entries, "lookup_contact", "persistence", detail)
        else:
            _fail(entries, "remember_contact", "persistence", "contact missing after restart")
            _fail(entries, "lookup_contact", "persistence", detail or "lookup failed after restart")

        context.seed_conflict()
        loop = context.make_loop(emitted=[])
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
        if _contains_any(artifact.answer, ("spazieren", "walk")):
            _pass(entries, "remember_plan", "multi_turn", "same-session plan recall succeeded")
        else:
            _fail(entries, "remember_plan", "multi_turn", "same-session plan recall failed")
        loop = context.make_loop(emitted=[])
        artifact = run_text_turn(loop, "Was habe ich heute noch vor?")
        record("remember_plan_persistence", artifact)
        if _contains_any(artifact.answer, ("spazieren", "walk")):
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
        personality_text = (context.personality_dir / "PERSONALITY.md").read_text(encoding="utf-8")
        if ok and _contains_any(personality_text, ("response_style:", "very short", "short and calm", "kurz und ruhig")):
            _pass(entries, "update_personality", "single_turn", detail)
            _pass(entries, "update_personality", "persistence", "personality file persisted updated behavior")
        else:
            _fail(entries, "update_personality", "single_turn", detail or "PERSONALITY.md not updated")
            _fail(entries, "update_personality", "persistence", "PERSONALITY.md missing update after restart")

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
        if ok and loop.voice_profile_monitor.summary().enrolled:
            _pass(entries, "enroll_voice_profile", "persistence", "voice profile survived restart")
            _pass(entries, "get_voice_profile_status", "persistence", detail)
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
        if ok and not loop.voice_profile_monitor.summary().enrolled:
            _pass(entries, "reset_voice_profile", "persistence", "voice profile stayed deleted after restart")
        else:
            _fail(entries, "reset_voice_profile", "persistence", "voice profile reappeared after restart")

    finally:
        project_root = str(context.root)
        memory_markdown_path = str(context.state_dir / "MEMORY.md")
        reminder_store_path = str(context.state_dir / "reminders.json")
        automation_store_path = str(context.state_dir / "automations.json")
        voice_profile_store_path = str(context.state_dir / "voice_profile.json")

    failed_tools = sorted(name for name, entry in entries.items() if entry.overall() == "fail")
    passed_tools = sorted(name for name, entry in entries.items() if entry.overall() == "pass")
    return {
        "base_env_path": str(base_env_path.resolve()),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-env", type=Path, default=Path(".env"))
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = run_matrix(args.base_env)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
