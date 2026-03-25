"""Run a bounded live Pi tool matrix through Twinr's realtime tool path.

The matrix exercises the productive realtime tool delegates with a real
OpenAI-backed runtime on the Raspberry Pi while keeping side effects isolated:
it writes state into a temporary project root, forces a unique remote-memory
namespace for the run, and replaces physical camera/printer peripherals with
safe stubs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import argparse
import json
import shutil
import sys
import time
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshot
from twinr.agent.workflows.realtime_runner import TwinrRealtimeHardwareLoop
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.display.debug_log import TwinrDisplayDebugLogBuilder
from twinr.ops.events import TwinrOpsEventStore
from twinr.ops.usage import TwinrUsageStore

_FALLBACK_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc````\x00"
    b"\x00\x00\x05\x00\x01\xa5\xf6E@\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _load_static_camera_image() -> bytes:
    """Return one valid repo-local image for live vision acceptance."""

    candidate = Path(__file__).resolve().parents[1] / "twinr_logo.png"
    try:
        return candidate.read_bytes()
    except Exception:
        return _FALLBACK_TINY_PNG


class NoopRecorder:
    """Provide the recorder surface required by the realtime loop constructor."""

    def capture_pcm_until_pause_with_options(self, **kwargs):
        del kwargs
        raise RuntimeError("Recorder should not be used in the live tool matrix.")

    def record_pcm_until_pause_with_options(self, **kwargs) -> bytes:
        del kwargs
        raise RuntimeError("Recorder should not be used in the live tool matrix.")


class SilentPlayer:
    """Suppress all playback side effects while keeping the runtime contract."""

    def __init__(self) -> None:
        self.played_chunks = 0
        self.stop_calls = 0

    def play_tone(self, **kwargs) -> None:
        del kwargs

    def play_pcm16_chunks(self, chunks, *, sample_rate: int, channels: int = 1, should_stop=None) -> None:
        del sample_rate, channels
        for chunk in chunks:
            if should_stop is not None and should_stop():
                break
            self.played_chunks += len(chunk)

    def play_wav_chunks(self, chunks, *, should_stop=None) -> None:
        for chunk in chunks:
            if should_stop is not None and should_stop():
                break
            self.played_chunks += len(chunk)

    def play_wav_bytes(self, audio_bytes: bytes) -> None:
        self.played_chunks += len(audio_bytes)

    def stop_playback(self) -> None:
        self.stop_calls += 1


class CapturingPrinter:
    """Capture printed output instead of talking to the physical printer."""

    def __init__(self) -> None:
        self.jobs: list[str] = []

    def print_text(self, text: str) -> str:
        self.jobs.append(text)
        return f"job-{len(self.jobs)}"


class StaticCamera:
    """Return one deterministic image for live vision tool calls."""

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


class NoopContext:
    """Stand in for hardware/proactive context managers not used by the matrix."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb


@dataclass(frozen=True, slots=True)
class ToolMatrixResult:
    """Capture one live tool invocation and its operator-visible telemetry."""

    tool_name: str
    event_name: str
    ok: bool
    status: str
    latency_ms: int
    result: dict[str, object]
    generic_event: dict[str, object]
    usage_record: dict[str, object] | None
    llm_log_lines: tuple[str, ...]
    new_event_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Describe one bounded live tool invocation for the matrix."""

    tool_name: str
    method_name: str
    arguments: dict[str, object]


class ToolMatrixContext:
    """Build an isolated temp Twinr project rooted in a copied Pi env."""

    def __init__(self, *, env_path: Path, run_id: str) -> None:
        self.env_path = env_path.resolve()
        self.run_id = run_id
        self.temp_dir = TemporaryDirectory(prefix=f"twinr-tool-matrix-{run_id}-")
        self.root = Path(self.temp_dir.name)
        self.state_dir = self.root / "state"
        self.personality_dir = self.root / "personality"
        self.reference_image_path = self.root / "user-reference.png"
        self.temp_env_path = self.root / ".env"
        self.camera_image_bytes = _load_static_camera_image()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        source_personality_dir = self.env_path.parent / "personality"
        if source_personality_dir.exists():
            shutil.copytree(source_personality_dir, self.personality_dir, dirs_exist_ok=True)
        else:
            self.personality_dir.mkdir(parents=True, exist_ok=True)
        self.reference_image_path.write_bytes(self.camera_image_bytes)
        self._write_temp_env()

    def close(self) -> None:
        self.temp_dir.cleanup()

    def build_loop(self, *, emit_lines: list[str]) -> TwinrRealtimeHardwareLoop:
        config = TwinrConfig.from_env(self.temp_env_path)
        runtime = TwinrRuntime(config=config)
        return TwinrRealtimeHardwareLoop(
            config=config,
            runtime=runtime,
            button_monitor=NoopContext(),
            recorder=NoopRecorder(),
            player=SilentPlayer(),
            printer=CapturingPrinter(),
            camera=StaticCamera(self.camera_image_bytes),
            proactive_monitor=NoopContext(),
            emit=emit_lines.append,
            sleep=lambda _seconds: None,
            error_reset_seconds=0.0,
        )

    def _write_temp_env(self) -> None:
        base_env_text = self.env_path.read_text(encoding="utf-8")
        overrides = [
            "TWINR_PERSONALITY_DIR=personality",
            f"TWINR_VISION_REFERENCE_IMAGE={self.reference_image_path}",
            f"TWINR_RUNTIME_STATE_PATH={self.state_dir / 'runtime-state.json'}",
            f"TWINR_MEMORY_MARKDOWN_PATH={self.state_dir / 'MEMORY.md'}",
            f"TWINR_REMINDER_STORE_PATH={self.state_dir / 'reminders.json'}",
            f"TWINR_AUTOMATION_STORE_PATH={self.state_dir / 'automations.json'}",
            f"TWINR_ADAPTIVE_TIMING_STORE_PATH={self.state_dir / 'adaptive_timing.json'}",
            f"TWINR_LONG_TERM_MEMORY_PATH={self.state_dir / 'chonkydb'}",
            f"TWINR_LONG_TERM_MEMORY_REMOTE_NAMESPACE=pi_tool_matrix_{self.run_id}",
            "TWINR_PROACTIVE_ENABLED=false",
            "TWINR_CONVERSATION_FOLLOW_UP_ENABLED=false",
            "TWINR_TURN_CONTROLLER_ENABLED=false",
            "TWINR_CONVERSATION_CLOSURE_GUARD_ENABLED=false",
            "TWINR_SEARCH_FEEDBACK_TONES_ENABLED=false",
        ]
        self.temp_env_path.write_text(
            base_env_text.rstrip() + "\n" + "\n".join(overrides) + "\n",
            encoding="utf-8",
        )


def _default_output_path(project_root: Path, *, run_id: str) -> Path:
    return project_root / "artifacts" / "reports" / f"pi_realtime_tool_matrix_{run_id}.json"


def _build_tool_specs(config: TwinrConfig) -> tuple[ToolSpec, ...]:
    now = datetime.now(ZoneInfo(config.local_timezone_name)).replace(second=0, microsecond=0)
    due_at = (now + timedelta(minutes=15)).isoformat()
    return (
        ToolSpec(
            tool_name="search_live_info",
            method_name="_handle_search_tool_call",
            arguments={
                "question": "Was ist aktuell in der Hamburger Lokalpolitik los?",
                "location_hint": "Hamburg",
            },
        ),
        ToolSpec(
            tool_name="inspect_camera",
            method_name="_handle_inspect_camera_tool_call",
            arguments={
                "question": "Was siehst du auf dem Bild?",
            },
        ),
        ToolSpec(
            tool_name="remember_memory",
            method_name="_handle_remember_memory_tool_call",
            arguments={
                "kind": "appointment",
                "summary": "Arzttermin morgen um 15 Uhr.",
                "details": "Bei Dr. Meyer in Hamburg.",
                "confirmed": True,
            },
        ),
        ToolSpec(
            tool_name="schedule_reminder",
            method_name="_handle_schedule_reminder_tool_call",
            arguments={
                "due_at": due_at,
                "summary": "Arzttermin",
                "details": "Bei Dr. Meyer",
                "kind": "appointment",
            },
        ),
        ToolSpec(
            tool_name="update_simple_setting",
            method_name="_handle_update_simple_setting_tool_call",
            arguments={
                "setting": "memory_capacity",
                "action": "increase",
                "confirmed": True,
            },
        ),
        ToolSpec(
            tool_name="print_receipt",
            method_name="_handle_print_tool_call",
            arguments={
                "text": "Bitte erinnere mich kurz an den Arzttermin.",
            },
        ),
    )


def _event_payload(entry: object) -> dict[str, object]:
    if not isinstance(entry, dict):
        return {}
    data = entry.get("data")
    return dict(data) if isinstance(data, dict) else {}


def _build_snapshot(loop: TwinrRealtimeHardwareLoop) -> RuntimeSnapshot:
    return RuntimeSnapshot(
        status=getattr(loop.runtime.status, "value", "waiting"),
        last_transcript=getattr(loop.runtime, "last_transcript", ""),
        last_response=getattr(loop.runtime, "last_response", ""),
        error_message=getattr(loop.runtime, "error_message", None),
    )


def _poll_new_usage(
    usage_store: TwinrUsageStore,
    *,
    previous_count: int,
    timeout_s: float,
) -> dict[str, object] | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        records = usage_store.tail(limit=256)
        if len(records) > previous_count:
            return records[-1].to_dict()
        time.sleep(0.05)
    records = usage_store.tail(limit=256)
    if len(records) > previous_count:
        return records[-1].to_dict()
    return None


def _run_spec(loop: TwinrRealtimeHardwareLoop, spec: ToolSpec) -> ToolMatrixResult:
    event_store = TwinrOpsEventStore.from_config(loop.config)
    usage_store = TwinrUsageStore.from_config(loop.config)
    before_events = event_store.tail(limit=256)
    before_usage = usage_store.tail(limit=256)
    handler = getattr(loop, spec.method_name)
    started = time.monotonic()
    result = handler(dict(spec.arguments))
    handler_latency_ms = max(0, int(round((time.monotonic() - started) * 1000.0)))

    deadline = time.monotonic() + 3.0
    generic_event_name = ""
    generic_event_payload: dict[str, object] = {}
    new_events: list[dict[str, object]] = []
    while time.monotonic() < deadline:
        current_events = event_store.tail(limit=256)
        new_events = current_events[len(before_events):]
        matches = [
            entry
            for entry in new_events
            if entry.get("event") in {"tool_call_finished", "tool_call_failed"}
            and _event_payload(entry).get("tool_name") == spec.tool_name
        ]
        if matches:
            generic_event_name = str(matches[-1].get("event") or "")
            generic_event_payload = _event_payload(matches[-1])
            break
        time.sleep(0.05)
    if not generic_event_payload:
        raise RuntimeError(f"No generic tool event captured for {spec.tool_name}.")

    usage_record = _poll_new_usage(
        usage_store,
        previous_count=len(before_usage),
        timeout_s=1.5 if spec.tool_name == "search_live_info" else 0.3,
    )
    sections = TwinrDisplayDebugLogBuilder.from_config(loop.config).build_sections(
        snapshot=_build_snapshot(loop),
        runtime_status=str(getattr(loop.runtime.status, "value", "waiting")),
        internet_state="ok",
        ai_state="ok",
        system_state="ok",
        clock_text=datetime.now(ZoneInfo(loop.config.local_timezone_name)).strftime("%H:%M"),
    )
    llm_lines = next(lines for title, lines in sections if title == "LLM Log")
    status = str(generic_event_payload.get("status") or result.get("status") or "unknown")
    ok = generic_event_name == "tool_call_finished"
    return ToolMatrixResult(
        tool_name=spec.tool_name,
        event_name=generic_event_name,
        ok=ok,
        status=status,
        latency_ms=int(generic_event_payload.get("latency_ms") or handler_latency_ms),
        result=dict(result),
        generic_event=generic_event_payload,
        usage_record=usage_record,
        llm_log_lines=tuple(llm_lines),
        new_event_names=tuple(str(entry.get("event") or "") for entry in new_events),
    )


def run_matrix(*, env_file: Path, output_path: Path | None, run_id: str | None) -> dict[str, object]:
    resolved_run_id = run_id or datetime.now(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ")
    context = ToolMatrixContext(env_path=env_file, run_id=resolved_run_id)
    try:
        emit_lines: list[str] = []
        loop = context.build_loop(emit_lines=emit_lines)
        specs = _build_tool_specs(loop.config)
        results = [asdict(_run_spec(loop, spec)) for spec in specs]
        ok = all(bool(item["ok"]) for item in results)
        payload = {
            "ok": ok,
            "run_id": resolved_run_id,
            "env_file": str(env_file.resolve()),
            "project_root": str(Path(loop.config.project_root).resolve()),
            "remote_namespace": loop.config.long_term_memory_remote_namespace,
            "tool_count": len(results),
            "results": results,
            "emit_line_count": len(emit_lines),
        }
        resolved_output = output_path or _default_output_path(Path.cwd(), run_id=resolved_run_id)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        payload["output_path"] = str(resolved_output.resolve())
        return payload
    finally:
        context.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default="/twinr/.env", help="Authoritative Twinr env file on the Pi.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument("--run-id", default=None, help="Optional stable run id for artifacts and remote namespace.")
    args = parser.parse_args()

    payload = run_matrix(
        env_file=Path(args.env_file),
        output_path=Path(args.output).expanduser() if args.output else None,
        run_id=args.run_id,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
