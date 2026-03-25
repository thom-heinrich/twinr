"""Bootstrap the Twinr command-line entrypoint.

This module keeps argument parsing and high-level command dispatch at the
package root so operators can launch Twinr via ``python -m twinr`` or the
installed ``twinr`` script. Runtime behavior stays in the focused subsystem
packages that this bootstrap imports on demand.
"""

# pylint: disable=no-name-in-module

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from threading import Lock
from typing import Any

from twinr.runtime_paths import prime_raspberry_pi_system_site_packages

prime_raspberry_pi_system_site_packages()

from twinr.agent.base_agent import TwinrConfig  # noqa: E402
from twinr.ops.locks import TwinrInstanceAlreadyRunningError  # noqa: E402
from twinr.ops.runtime_env import prime_user_session_audio_env  # noqa: E402

_RUNTIME_SUPERVISOR_ENV_KEY = "TWINR_RUNTIME_SUPERVISOR_ACTIVE"


def build_parser() -> argparse.ArgumentParser:
    """Build the authoritative CLI parser for Twinr runtime commands."""

    parser = argparse.ArgumentParser(description="Twinr bootstrap runtime")
    parser.add_argument("--env-file", default=".env", help="Path to the .env file")
    parser.add_argument(
        "--demo-transcript",
        help="Run a simple end-to-end placeholder flow for a transcript",
    )
    parser.add_argument(
        "--openai-prompt",
        help="Send a prompt through the OpenAI backend",
    )
    parser.add_argument(
        "--vision-prompt",
        help="Send a prompt plus one or more images through the OpenAI backend",
    )
    parser.add_argument(
        "--vision-image",
        action="append",
        type=Path,
        default=[],
        help="Attach a local image file; repeat for multiple images",
    )
    parser.add_argument(
        "--vision-camera-capture",
        action="store_true",
        help="Capture a fresh camera image and attach it as the first vision input",
    )
    parser.add_argument(
        "--vision-save-capture",
        type=Path,
        help="Optionally persist the camera image used for --vision-prompt",
    )
    parser.add_argument(
        "--camera-capture-output",
        type=Path,
        help="Capture a still image from the configured camera and write it to this path",
    )
    parser.add_argument(
        "--proactive-observe-once",
        action="store_true",
        help="Capture one proactive vision observation and print the parsed fields",
    )
    parser.add_argument(
        "--proactive-audio-observe-once",
        action="store_true",
        help="Capture one proactive ambient-audio observation and print the parsed fields",
    )
    parser.add_argument(
        "--openai-web-search",
        action="store_true",
        default=None,
        help="Enable the OpenAI web search tool for --openai-prompt",
    )
    parser.add_argument(
        "--audio-file",
        type=Path,
        help="Transcribe an audio file with OpenAI speech-to-text",
    )
    parser.add_argument(
        "--tts-text",
        help="Synthesize standalone text with OpenAI text-to-speech",
    )
    parser.add_argument(
        "--tts-output",
        type=Path,
        default=Path("twinr-tts.wav"),
        help="Write synthesized speech to this file",
    )
    parser.add_argument(
        "--format-for-print",
        help="Rewrite text into a short thermal-printer format using OpenAI",
    )
    parser.add_argument(
        "--run-streaming-loop",
        action="store_true",
        help="Run the GPIO -> STT -> tool-calling LLM -> TTS -> speaker/printer loop",
    )
    parser.add_argument(
        "--run-whatsapp-channel",
        action="store_true",
        help="Run the consumer WhatsApp text channel via the Baileys worker",
    )
    parser.add_argument(
        "--run-runtime-supervisor",
        action="store_true",
        help="Run the authoritative Pi runtime supervisor for the streaming loop and remote watchdog.",
    )
    parser.add_argument(
        "--self-coding-codex-self-test",
        action="store_true",
        help="Run the bounded self_coding Codex SDK preflight on this machine.",
    )
    parser.add_argument(
        "--self-coding-live-auth-check",
        action="store_true",
        help="When used with --self-coding-codex-self-test, also run a tiny live Codex auth probe.",
    )
    parser.add_argument(
        "--self-coding-morning-briefing-acceptance",
        action="store_true",
        help="Compile and execute the minimum self_coding morning-briefing acceptance flow.",
    )
    parser.add_argument(
        "--long-term-memory-live-acceptance",
        action="store_true",
        help="Run the live synthetic-memory acceptance matrix against the real OpenAI and ChonkyDB path.",
    )
    parser.add_argument(
        "--self-coding-acceptance-capture-only",
        action="store_true",
        help="Capture morning-briefing speech in memory instead of playing it aloud during acceptance.",
    )
    parser.add_argument(
        "--run-orchestrator-server",
        action="store_true",
        help="Run the Twinr websocket orchestrator service",
    )
    parser.add_argument(
        "--orchestrator-probe-turn",
        help="Send one text turn through the websocket orchestrator and print the streamed result",
    )
    parser.add_argument(
        "--loop-duration",
        type=float,
        help="Optional max runtime in seconds for the loop commands",
    )
    parser.add_argument(
        "--run-web",
        action="store_true",
        help="Run the local Twinr settings dashboard",
    )
    parser.add_argument(
        "--watch-remote-memory",
        action="store_true",
        help="Run the rolling remote-memory watchdog and print one sample line per probe.",
    )
    parser.add_argument(
        "--display-test",
        action="store_true",
        help="Render a test card on the configured Twinr display backend",
    )
    parser.add_argument(
        "--run-display-loop",
        action="store_true",
        help="Run the status display loop for the configured Twinr display backend",
    )
    return parser


def _uses_pi_runtime_root(env_file: str | Path) -> bool:
    """Return whether the provided env file targets the Pi acceptance checkout."""

    env_path = Path(env_file).resolve()
    pi_root = Path("/twinr").resolve()
    return pi_root in env_path.parents or env_path == pi_root / ".env"


def _is_raspberry_pi_host() -> bool:
    """Detect whether the current machine reports itself as a Raspberry Pi."""

    model_path = Path("/proc/device-tree/model")
    try:
        return "Raspberry Pi" in model_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def _should_enable_display_companion(config: TwinrConfig, env_file: str | Path) -> bool:
    """Enable the display companion only for intended authoritative runtime hosts."""

    if not _uses_pi_runtime_root(env_file):
        return False
    explicit_setting = getattr(config, "display_companion_enabled", None)
    if explicit_setting is not None:
        return bool(explicit_setting)
    return _is_raspberry_pi_host()


def _should_enable_respeaker_led_companion(config: TwinrConfig, env_file: str | Path) -> bool:
    """Enable the ReSpeaker LED companion on authoritative Pi runtime hosts."""

    if not _uses_pi_runtime_root(env_file):
        return False
    explicit_setting = getattr(config, "respeaker_led_enabled", None)
    if explicit_setting is not None:
        return bool(explicit_setting)
    if not _is_raspberry_pi_host():
        return False
    from twinr.hardware.respeaker import config_targets_respeaker, probe_respeaker_xvf3800

    if config_targets_respeaker(
        getattr(config, "voice_orchestrator_audio_device", None),
        getattr(config, "proactive_audio_input_device", None),
        getattr(config, "audio_input_device", None),
    ):
        return True
    # Field Pi deployments often route the XVF3800 through ALSA/PipeWire
    # "default", so string-only config matching can miss the real hardware.
    try:
        probe = probe_respeaker_xvf3800()
    except Exception:
        return False
    return getattr(probe, "usb_device", None) is not None


def _should_ensure_remote_watchdog_companion(config: TwinrConfig, env_file: str | Path) -> bool:
    """Return whether the remote-memory watchdog companion must be started."""

    if str(os.environ.get(_RUNTIME_SUPERVISOR_ENV_KEY, "")).strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if not _uses_pi_runtime_root(env_file):
        return False
    return (
        config.long_term_memory_enabled
        and str(config.long_term_memory_mode or "").strip().lower() == "remote_primary"
        and config.long_term_memory_remote_required
        and str(config.long_term_memory_remote_runtime_check_mode or "").strip().lower()
        == "watchdog_artifact"
    )


def _assert_pi_runtime_root(env_file: str | Path, *, command_name: str) -> None:
    """Reject Pi-state commands launched from the development checkout."""

    if not _uses_pi_runtime_root(env_file):
        return
    if not _is_raspberry_pi_host():
        raise RuntimeError(
            f"{command_name} with /twinr runtime state is only allowed on a Raspberry Pi host, "
            f"not {os.uname().nodename}"
        )
    pi_root = Path("/twinr").resolve()
    cwd = Path.cwd().resolve()
    if cwd != pi_root:
        raise RuntimeError(
            f"{command_name} with /twinr runtime state must be launched from /twinr, not {cwd}"
        )
    import twinr as twinr_package

    package_file = getattr(twinr_package, "__file__", None)
    package_root = Path(package_file).resolve().parent.parent if package_file else None
    expected_source_root = pi_root / "src"
    if package_root != expected_source_root:
        raise RuntimeError(
            f"{command_name} with /twinr runtime state must import from {expected_source_root}, not {package_root}"
        )


def _build_runtime(config: TwinrConfig) -> Any:
    """Create the live runtime lazily so loop locks can be acquired first."""

    from twinr.agent.base_agent import TwinrRuntime

    return TwinrRuntime(config=config)


def _print_runtime_banner(runtime: Any, config: TwinrConfig, env_path: Path) -> None:
    """Emit the standard runtime bootstrap facts once the runtime exists."""

    print(f"status={runtime.status.value}")
    print(f"web_port={config.web_port}")
    print(f"model={config.default_model}")
    print(f"openai_reasoning_effort={config.openai_reasoning_effort}")
    print(f"runtime_cwd={Path.cwd().resolve()}")
    print(f"runtime_source_root={Path(sys.path[0] or '.').resolve()}")
    try:
        import twinr as twinr_package

        print(f"runtime_package_root={Path(getattr(twinr_package, '__file__', '.')).resolve().parent.parent}")
    except Exception:
        print("runtime_package_root=unknown")
    print(f"runtime_env_file={env_path}")


def _print_self_coding_codex_report(report: Any) -> None:
    """Emit the bounded self_coding Codex preflight result as key/value lines."""

    print(f"self_coding_codex_status={getattr(report, 'status', 'unknown')}")
    print(f"self_coding_codex_ready={str(bool(getattr(report, 'ready', False))).lower()}")
    print(f"self_coding_codex_detail={getattr(report, 'detail', '')}")
    if getattr(report, "node_version", None):
        print(f"self_coding_codex_node_version={report.node_version}")
    if getattr(report, "npm_version", None):
        print(f"self_coding_codex_npm_version={report.npm_version}")
    if getattr(report, "codex_version", None):
        print(f"self_coding_codex_version={report.codex_version}")
    print(f"self_coding_codex_auth_present={str(bool(getattr(report, 'auth_present', False))).lower()}")
    if getattr(report, "local_self_test_ok", None) is not None:
        print(f"self_coding_codex_bridge_self_test={str(bool(report.local_self_test_ok)).lower()}")
    if getattr(report, "live_auth_check_ok", None) is not None:
        print(f"self_coding_codex_live_auth_check={str(bool(report.live_auth_check_ok)).lower()}")


def _print_morning_briefing_acceptance_result(result: Any) -> None:
    """Emit the morning-briefing acceptance result as key/value lines."""

    print("self_coding_acceptance_case=morning_briefing")
    print(f"self_coding_acceptance_job_id={getattr(result, 'job_id', '')}")
    print(f"self_coding_acceptance_job_status={getattr(result, 'job_status', '')}")
    print(f"self_coding_acceptance_skill_id={getattr(result, 'skill_id', '')}")
    print(f"self_coding_acceptance_version={getattr(result, 'version', '')}")
    print(f"self_coding_acceptance_activation_status={getattr(result, 'activation_status', '')}")
    print(f"self_coding_acceptance_refresh_status={getattr(result, 'refresh_status', '')}")
    print(f"self_coding_acceptance_delivery_status={getattr(result, 'delivery_status', '')}")
    print(f"self_coding_acceptance_delivery_delivered={str(bool(getattr(result, 'delivery_delivered', False))).lower()}")
    print(f"self_coding_acceptance_search_calls={getattr(result, 'search_call_count', 0)}")
    print(f"self_coding_acceptance_summary_calls={getattr(result, 'summary_call_count', 0)}")
    print(f"self_coding_acceptance_spoken_count={getattr(result, 'spoken_count', 0)}")
    if getattr(result, "last_summary_text", None):
        print(f"self_coding_acceptance_last_summary={result.last_summary_text}")


def _print_long_term_memory_live_acceptance_result(result: Any) -> None:
    """Emit the live synthetic-memory acceptance result as key/value lines."""

    passed_cases = getattr(result, "passed_cases", 0)
    total_cases = getattr(result, "total_cases", 0)
    print("long_term_memory_live_acceptance_case=synthetic_memory")
    print(f"long_term_memory_live_acceptance_probe_id={getattr(result, 'probe_id', '')}")
    print(f"long_term_memory_live_acceptance_status={getattr(result, 'status', 'unknown')}")
    print(f"long_term_memory_live_acceptance_ready={str(bool(getattr(result, 'ready', False))).lower()}")
    print(f"long_term_memory_live_acceptance_passed_cases={passed_cases}")
    print(f"long_term_memory_live_acceptance_total_cases={total_cases}")
    print(f"long_term_memory_live_acceptance_queue_before={getattr(result, 'queue_before_count', 0)}")
    print(f"long_term_memory_live_acceptance_queue_after={getattr(result, 'queue_after_count', 0)}")
    print(f"long_term_memory_live_acceptance_restart_queue={getattr(result, 'restart_queue_count', 0)}")
    if getattr(result, "artifact_path", None):
        print(f"long_term_memory_live_acceptance_artifact_path={result.artifact_path}")
    if getattr(result, "report_path", None):
        print(f"long_term_memory_live_acceptance_report_path={result.report_path}")
    if getattr(result, "error_message", None):
        print(f"long_term_memory_live_acceptance_error={result.error_message}")


def _run_web_server(config: TwinrConfig, env_file: str | Path) -> int:
    """Start the local web control plane without bootstrapping the runtime."""

    _assert_pi_runtime_root(env_file, command_name="run-web")
    from twinr.web import create_app

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The web dashboard dependencies are not installed. Run `pip install -e .` in /twinr."
        ) from exc

    app = create_app(Path(env_file))
    uvicorn.run(app, host=config.web_host, port=config.web_port)
    return 0


def _run_orchestrator_server(config: TwinrConfig, env_file: str | Path) -> int:
    """Start the websocket orchestrator service without bootstrapping the runtime."""

    _assert_pi_runtime_root(env_file, command_name="run-orchestrator-server")
    from twinr.orchestrator import create_app

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The web/orchestrator dependencies are not installed. Run `pip install -e .` in /twinr."
        ) from exc

    app = create_app(Path(env_file))
    uvicorn.run(app, host=config.orchestrator_host, port=config.orchestrator_port)
    return 0


def _run_whatsapp_channel(
    config: TwinrConfig,
    runtime: Any,
    backend: Any,
    *,
    env_file: str | Path,
    duration_s: float | None,
    lock_config: TwinrConfig | None = None,
) -> int:
    """Start the Baileys-backed WhatsApp text channel listener."""

    _assert_pi_runtime_root(env_file, command_name="run-whatsapp-channel")
    from twinr.channels.whatsapp import TwinrWhatsAppChannelLoop
    from twinr.ops import loop_instance_lock

    loop = TwinrWhatsAppChannelLoop(config=config, runtime=runtime, backend=backend)
    with loop_instance_lock(lock_config or config, "whatsapp-channel"):
        return loop.run(duration_s=duration_s)


def _ensure_remote_watchdog_for_runtime_boot(config: TwinrConfig, env_file: str | Path) -> None:
    """Start the external remote-memory watchdog before runtime bootstrap when required."""

    if not _should_ensure_remote_watchdog_companion(config, env_file):
        return
    from twinr.ops.remote_memory_watchdog_companion import ensure_remote_memory_watchdog_process

    ensure_remote_memory_watchdog_process(config, env_file=env_file)


def main() -> int:
    """Dispatch the requested Twinr CLI command and return its exit code."""

    prime_user_session_audio_env()
    args = build_parser().parse_args()
    config = TwinrConfig.from_env(Path(args.env_file))
    env_path = Path(args.env_file).resolve()

    if args.watch_remote_memory:
        _assert_pi_runtime_root(args.env_file, command_name="watch-remote-memory")
        from twinr.ops import RemoteMemoryWatchdog, loop_instance_lock

        watchdog = RemoteMemoryWatchdog.from_config(config)
        print(f"remote_memory_watchdog_artifact={watchdog.artifact_path}")
        print(f"remote_memory_watchdog_interval_s={watchdog.interval_s}")
        print(f"remote_memory_watchdog_history_limit={watchdog.history_limit}")
        with loop_instance_lock(config, "remote-memory-watchdog"):
            return watchdog.run(duration_s=args.loop_duration)

    if args.run_runtime_supervisor:
        _assert_pi_runtime_root(args.env_file, command_name="run-runtime-supervisor")
        from twinr.ops import loop_instance_lock
        from twinr.ops.runtime_supervisor import TwinrRuntimeSupervisor

        supervisor = TwinrRuntimeSupervisor(config=config, env_file=args.env_file)
        with loop_instance_lock(config, "runtime-supervisor"):
            return supervisor.run(duration_s=args.loop_duration)

    if args.self_coding_codex_self_test:
        _assert_pi_runtime_root(args.env_file, command_name="self-coding-codex-self-test")
        from twinr.agent.self_coding.codex_driver.environment import collect_codex_sdk_environment_report

        report = collect_codex_sdk_environment_report(
            run_local_self_test=True,
            run_live_auth_check=args.self_coding_live_auth_check,
        )
        _print_self_coding_codex_report(report)
        return 0 if report.ready else 1

    if args.self_coding_morning_briefing_acceptance:
        _assert_pi_runtime_root(args.env_file, command_name="self-coding-morning-briefing-acceptance")
        from twinr.agent.self_coding.live_acceptance import run_live_morning_briefing_acceptance

        result = run_live_morning_briefing_acceptance(
            project_root=config.project_root,
            env_file=args.env_file,
            speak_out_loud=not args.self_coding_acceptance_capture_only,
            live_e2e_environment="pi" if _uses_pi_runtime_root(args.env_file) else "local",
        )
        _print_morning_briefing_acceptance_result(result)
        return 0 if getattr(result, "delivery_delivered", False) else 1

    if args.long_term_memory_live_acceptance:
        _assert_pi_runtime_root(args.env_file, command_name="long-term-memory-live-acceptance")
        from twinr.memory.longterm.evaluation.live_memory_acceptance import run_live_memory_acceptance

        result = run_live_memory_acceptance(env_path=args.env_file)
        _print_long_term_memory_live_acceptance_result(result)
        return 0 if getattr(result, "ready", False) else 1

    if args.run_web:
        return _run_web_server(config, args.env_file)

    if args.run_orchestrator_server:
        return _run_orchestrator_server(config, args.env_file)

    uses_openai = any(
        [
            args.openai_prompt,
            args.vision_prompt,
            args.audio_file,
            args.tts_text,
            args.format_for_print,
            args.proactive_observe_once,
            args.run_streaming_loop,
            args.run_whatsapp_channel,
            args.run_orchestrator_server,
            args.orchestrator_probe_turn,
        ]
    )
    uses_camera = bool(args.vision_camera_capture or args.camera_capture_output or args.proactive_observe_once)

    runtime = None
    runtime_config = config
    try:
        if args.run_streaming_loop:
            _assert_pi_runtime_root(args.env_file, command_name="run-streaming-loop")
            from twinr.agent.workflows.runtime_error_hold import hold_runtime_error_state
            from twinr.agent.workflows.streaming_runner import TwinrStreamingHardwareLoop
            from twinr.display.companion import optional_display_companion
            from twinr.hardware.respeaker.companion import optional_respeaker_led_companion
            from twinr.ops import loop_instance_lock
            from twinr.providers import build_streaming_provider_bundle
            from twinr.providers.openai import OpenAIBackend

            _ensure_remote_watchdog_for_runtime_boot(config, args.env_file)

            with loop_instance_lock(config, "streaming-loop"):
                with optional_display_companion(
                    config,
                    enabled=_should_enable_display_companion(config, args.env_file),
                ):
                    with optional_respeaker_led_companion(
                        config,
                        enabled=_should_enable_respeaker_led_companion(config, args.env_file),
                    ):
                        runtime = _build_runtime(config)
                        _print_runtime_banner(runtime, config, env_path)
                        try:
                            backend = OpenAIBackend(config=config)
                            provider_bundle = build_streaming_provider_bundle(config, support_backend=backend)
                            loop = TwinrStreamingHardwareLoop(
                                config=config,
                                runtime=runtime,
                                print_backend=provider_bundle.print_backend,
                                stt_provider=provider_bundle.stt,
                                verification_stt_provider=getattr(provider_bundle, "verification_stt", None),
                                agent_provider=provider_bundle.agent,
                                tts_provider=provider_bundle.tts,
                                tool_agent_provider=provider_bundle.tool_agent,
                            )
                            return loop.run(duration_s=args.loop_duration)
                        except Exception as exc:
                            return hold_runtime_error_state(
                                runtime=runtime,
                                error=exc,
                                duration_s=args.loop_duration,
                            )

        if args.run_whatsapp_channel:
            from twinr.ops.runtime_scope import build_scoped_runtime_config

            runtime_config = build_scoped_runtime_config(
                config,
                scope_name="whatsapp-channel",
                restore_runtime_state_on_startup=False,
            )

        _ensure_remote_watchdog_for_runtime_boot(config, args.env_file)
        runtime = _build_runtime(runtime_config)
        _print_runtime_banner(runtime, runtime_config, env_path)

        if args.demo_transcript:
            runtime.press_green_button()
            print(f"status={runtime.status.value}")
            runtime.submit_transcript(args.demo_transcript)
            print(f"status={runtime.status.value}")
            answer = runtime.complete_agent_turn(
                f"Placeholder answer for: {args.demo_transcript.strip()}"
            )
            print(f"status={runtime.status.value}")
            print(f"response={answer}")
            runtime.finish_speaking()
            print(f"status={runtime.status.value}")

        backend = None
        if uses_openai:
            from twinr.providers.openai import OpenAIBackend

            backend = OpenAIBackend(config=runtime_config)

        camera = None
        if uses_camera:
            from twinr.hardware.camera import V4L2StillCamera

            camera = V4L2StillCamera.from_config(runtime_config)

        if args.display_test:
            from twinr.display import create_display_adapter

            display = create_display_adapter(config)
            try:
                display.show_test_pattern()
                print("display_test=ok")
            finally:
                display.close()
            return 0

        if args.run_display_loop:
            _assert_pi_runtime_root(args.env_file, command_name="run-display-loop")
            from twinr.display import TwinrStatusDisplayLoop
            from twinr.ops import loop_instance_lock

            loop = TwinrStatusDisplayLoop.from_config(config)
            with loop_instance_lock(config, "display-loop"):
                return loop.run(duration_s=args.loop_duration)

        if args.run_whatsapp_channel:
            if backend is None:
                raise RuntimeError("WhatsApp channel requires configured providers")
            return _run_whatsapp_channel(
                runtime_config,
                runtime,
                backend,
                env_file=args.env_file,
                duration_s=args.loop_duration,
                lock_config=config,
            )

        if args.orchestrator_probe_turn:
            _assert_pi_runtime_root(args.env_file, command_name="orchestrator-probe-turn")
            from twinr.agent.tools import bind_realtime_tool_handlers
            from twinr.agent.workflows.streaming_runner import TwinrStreamingHardwareLoop
            from twinr.orchestrator import OrchestratorTurnRequest, OrchestratorWebSocketClient
            from twinr.providers import build_streaming_provider_bundle

            if backend is None:
                raise RuntimeError("Orchestrator probe requires configured providers")
            provider_bundle = build_streaming_provider_bundle(config, support_backend=backend)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                print_backend=provider_bundle.print_backend,
                stt_provider=provider_bundle.stt,
                verification_stt_provider=getattr(provider_bundle, "verification_stt", None),
                agent_provider=provider_bundle.agent,
                tts_provider=provider_bundle.tts,
                tool_agent_provider=provider_bundle.tool_agent,
            )
            client = OrchestratorWebSocketClient(
                config.orchestrator_ws_url,
                shared_secret=config.orchestrator_shared_secret,
            )
            deltas: list[str] = []
            result = client.run_turn(
                OrchestratorTurnRequest(
                    prompt=args.orchestrator_probe_turn,
                    conversation=runtime.tool_provider_conversation_context(),
                    supervisor_conversation=runtime.supervisor_provider_conversation_context(),
                ),
                tool_handlers=bind_realtime_tool_handlers(loop.tool_executor),
                on_ack=lambda event: print(f"ack={event.ack_id}:{event.text}"),
                on_text_delta=lambda delta: deltas.append(delta),
            )
            if deltas:
                print(f"streamed={''.join(deltas)}")
            print(f"response={result.text}")
            print(f"rounds={result.rounds}")
            print(f"used_web_search={str(result.used_web_search).lower()}")
            if result.response_id:
                print(f"response_id={result.response_id}")
            if result.request_id:
                print(f"request_id={result.request_id}")
            if result.model:
                print(f"model={result.model}")
            return 0

        if args.audio_file and backend is not None:
            transcript = backend.transcribe_path(args.audio_file)
            print(f"transcript={transcript}")

        if args.camera_capture_output and camera is not None:
            capture = camera.capture_photo(output_path=args.camera_capture_output, filename=args.camera_capture_output.name)
            print(f"camera_capture={args.camera_capture_output}")
            print(f"camera_capture_bytes={len(capture.data)}")
            print(f"camera_device={capture.source_device}")
            print(f"camera_input_format={capture.input_format or 'default'}")

        if args.proactive_observe_once:
            provider_name = (getattr(config, "proactive_vision_provider", "local_first") or "local_first").strip().lower()
            if provider_name == "openai":
                if backend is None or camera is None:
                    raise RuntimeError("OpenAI proactive observation requires configured OpenAI and camera access")
                from twinr.proactive import OpenAIVisionObservationProvider

                observer = OpenAIVisionObservationProvider(
                    backend=backend,
                    camera=camera,
                    camera_lock=Lock(),
                )
            else:
                from twinr.proactive.social.local_camera_provider import LocalAICameraObservationProvider

                observer = LocalAICameraObservationProvider.from_config(config)
            snapshot = observer.observe()
            print(f"proactive_person_visible={str(snapshot.observation.person_visible).lower()}")
            print(f"proactive_camera_ready={str(snapshot.observation.camera_ready).lower()}")
            print(f"proactive_camera_ai_ready={str(snapshot.observation.camera_ai_ready).lower()}")
            print(f"proactive_looking_toward_device={str(snapshot.observation.looking_toward_device).lower()}")
            print(f"proactive_body_pose={snapshot.observation.body_pose.value}")
            print(f"proactive_motion_state={snapshot.observation.motion_state.value}")
            print(f"proactive_smiling={str(snapshot.observation.smiling).lower()}")
            print(
                "proactive_hand_or_object_near_camera="
                f"{str(snapshot.observation.hand_or_object_near_camera).lower()}"
            )
            print(f"proactive_coarse_arm_gesture={snapshot.observation.coarse_arm_gesture.value}")
            print(f"proactive_fine_hand_gesture={snapshot.observation.fine_hand_gesture.value}")
            print(f"proactive_gesture_event={snapshot.observation.gesture_event.value}")
            print(f"proactive_person_count={snapshot.observation.person_count}")
            if snapshot.response_id:
                print(f"proactive_response_id={snapshot.response_id}")
            if snapshot.request_id:
                print(f"proactive_request_id={snapshot.request_id}")
            if snapshot.model:
                print(f"proactive_model={snapshot.model}")

        if args.proactive_audio_observe_once:
            from twinr.proactive.runtime.audio_perception import (
                observe_audio_perception_once,
                render_audio_perception_snapshot_lines,
            )

            snapshot = observe_audio_perception_once(config)
            for line in render_audio_perception_snapshot_lines(snapshot):
                print(line)

        if args.vision_prompt and backend is not None:
            from twinr.providers.openai import OpenAIImageInput

            images: list[OpenAIImageInput] = []
            if args.vision_camera_capture:
                if camera is None:
                    raise RuntimeError("Camera access is not configured for --vision-camera-capture")
                capture_filename = args.vision_save_capture.name if args.vision_save_capture else "camera-capture.png"
                capture = camera.capture_photo(
                    output_path=args.vision_save_capture,
                    filename=capture_filename,
                )
                images.append(
                    OpenAIImageInput(
                        data=capture.data,
                        content_type=capture.content_type,
                        filename=capture.filename,
                        label="Image 1: live camera capture from the device.",
                    )
                )
                print(f"vision_camera_device={capture.source_device}")
                print(f"vision_camera_input_format={capture.input_format or 'default'}")
                if args.vision_save_capture is not None:
                    print(f"vision_saved_capture={args.vision_save_capture}")
            for index, image_path in enumerate(args.vision_image, start=1):
                label = f"Reference image {index}: user-provided comparison image."
                images.append(OpenAIImageInput.from_path(image_path, label=label))
            if not images:
                raise RuntimeError(
                    "Vision requests need at least one image. Use --vision-camera-capture and/or --vision-image."
                )

            runtime.press_green_button()
            print(f"status={runtime.status.value}")
            runtime.submit_transcript(args.vision_prompt)
            print(f"status={runtime.status.value}")
            response = backend.respond_to_images_with_metadata(
                args.vision_prompt,
                images=images,
                conversation=runtime.conversation_context(),
                allow_web_search=args.openai_web_search,
            )
            answer = runtime.complete_agent_turn(response.text)
            print(f"status={runtime.status.value}")
            print(f"vision_image_count={len(images)}")
            print(f"response={answer}")
            if response.response_id:
                print(f"openai_response_id={response.response_id}")
            if response.request_id:
                print(f"openai_request_id={response.request_id}")
            print(f"openai_used_web_search={str(response.used_web_search).lower()}")
            runtime.finish_speaking()
            print(f"status={runtime.status.value}")

        if args.openai_prompt and backend is not None:
            runtime.press_green_button()
            print(f"status={runtime.status.value}")
            runtime.submit_transcript(args.openai_prompt)
            print(f"status={runtime.status.value}")
            response = backend.respond_with_metadata(
                args.openai_prompt,
                conversation=runtime.conversation_context(),
                allow_web_search=args.openai_web_search,
            )
            answer = runtime.complete_agent_turn(response.text)
            print(f"status={runtime.status.value}")
            print(f"response={answer}")
            if response.response_id:
                print(f"openai_response_id={response.response_id}")
            if response.request_id:
                print(f"openai_request_id={response.request_id}")
            print(f"openai_used_web_search={str(response.used_web_search).lower()}")
            runtime.finish_speaking()
            print(f"status={runtime.status.value}")

        if args.tts_text and backend is not None:
            audio_bytes = backend.synthesize(args.tts_text)
            args.tts_output.write_bytes(audio_bytes)
            print(f"tts_output={args.tts_output}")
            print(f"tts_bytes={len(audio_bytes)}")

        if args.format_for_print and backend is not None:
            formatted = backend.format_for_print_with_metadata(args.format_for_print)
            print(f"print_text={formatted.text}")
            if formatted.response_id:
                print(f"print_response_id={formatted.response_id}")
    except Exception as exc:
        if isinstance(exc, TwinrInstanceAlreadyRunningError):
            if runtime is not None:
                print(f"status={runtime.status.value}")
            print(f"error={exc}")
            return 1
        if runtime is not None:
            runtime.fail(str(exc))
            print(f"status={runtime.status.value}")
        print(f"error={exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
