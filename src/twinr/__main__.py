from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from threading import Lock
from typing import Any

from twinr.agent.base_agent import TwinrConfig
from twinr.ops.locks import TwinrInstanceAlreadyRunningError
from twinr.ops.runtime_env import prime_user_session_audio_env

_RUNTIME_SUPERVISOR_ENV_KEY = "TWINR_RUNTIME_SUPERVISOR_ACTIVE"


def build_parser() -> argparse.ArgumentParser:
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
        "--wakeword-eval",
        action="store_true",
        help="Evaluate the current wakeword detector against labeled captures or a JSONL manifest.",
    )
    parser.add_argument(
        "--wakeword-autotune",
        action="store_true",
        help="Search for a better wakeword calibration profile from labeled captures or a JSONL manifest.",
    )
    parser.add_argument(
        "--wakeword-manifest",
        type=Path,
        help="Optional JSONL manifest of labeled wakeword captures for eval/autotune.",
    )
    parser.add_argument(
        "--wakeword-label-capture",
        type=Path,
        help="Record an operator label for one stored wakeword capture.",
    )
    parser.add_argument(
        "--wakeword-label",
        help="Operator label to save for --wakeword-label-capture, for example correct or false_positive.",
    )
    parser.add_argument(
        "--wakeword-label-notes",
        help="Optional operator note stored with --wakeword-label-capture.",
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
        "--run-hardware-loop",
        action="store_true",
        help="Run the GPIO -> mic -> OpenAI -> speaker/printer loop",
    )
    parser.add_argument(
        "--run-realtime-loop",
        action="store_true",
        help="Run the GPIO -> mic -> OpenAI Realtime -> speaker/printer loop",
    )
    parser.add_argument(
        "--run-streaming-loop",
        action="store_true",
        help="Run the GPIO -> STT -> tool-calling LLM -> TTS -> speaker/printer loop",
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
        help="Render a black/white test card on the configured Waveshare display",
    )
    parser.add_argument(
        "--run-display-loop",
        action="store_true",
        help="Run the e-paper status display loop",
    )
    return parser


def _uses_pi_runtime_root(env_file: str | Path) -> bool:
    env_path = Path(env_file).resolve()
    pi_root = Path("/twinr").resolve()
    return pi_root in env_path.parents or env_path == pi_root / ".env"


def _is_raspberry_pi_host() -> bool:
    model_path = Path("/proc/device-tree/model")
    try:
        return "Raspberry Pi" in model_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def _should_enable_display_companion(env_file: str | Path) -> bool:
    return _uses_pi_runtime_root(env_file) and _is_raspberry_pi_host()


def _should_ensure_remote_watchdog_companion(config: TwinrConfig, env_file: str | Path) -> bool:
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
    if not _uses_pi_runtime_root(env_file):
        return
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


def main() -> int:
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

    uses_openai = any(
        [
            args.openai_prompt,
            args.vision_prompt,
            args.audio_file,
            args.tts_text,
            args.format_for_print,
            args.proactive_observe_once,
            args.run_hardware_loop,
            args.run_realtime_loop,
            args.run_streaming_loop,
            args.run_orchestrator_server,
            args.orchestrator_probe_turn,
        ]
    )
    uses_camera = bool(args.vision_camera_capture or args.camera_capture_output or args.proactive_observe_once)

    try:
        if args.run_streaming_loop:
            _assert_pi_runtime_root(args.env_file, command_name="run-streaming-loop")
            from twinr.agent.workflows.streaming_runner import TwinrStreamingHardwareLoop
            from twinr.display.companion import optional_display_companion
            from twinr.ops import loop_instance_lock
            from twinr.providers import build_streaming_provider_bundle
            from twinr.providers.openai import OpenAIBackend

            if _should_ensure_remote_watchdog_companion(config, args.env_file):
                from twinr.ops.remote_memory_watchdog_companion import ensure_remote_memory_watchdog_process

                ensure_remote_memory_watchdog_process(config, env_file=args.env_file)

            with loop_instance_lock(config, "streaming-loop"):
                with optional_display_companion(
                    config,
                    enabled=_should_enable_display_companion(args.env_file),
                ):
                    runtime = _build_runtime(config)
                    _print_runtime_banner(runtime, config, env_path)
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

        runtime = _build_runtime(config)
        _print_runtime_banner(runtime, config, env_path)

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

            backend = OpenAIBackend(config=config)

        camera = None
        if uses_camera:
            from twinr.hardware.camera import V4L2StillCamera

            camera = V4L2StillCamera.from_config(config)

        if args.run_web:
            _assert_pi_runtime_root(args.env_file, command_name="run-web")
            from twinr.web import create_app

            try:
                import uvicorn
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "The web dashboard dependencies are not installed. Run `pip install -e .` in /twinr."
                ) from exc

            app = create_app(Path(args.env_file))
            uvicorn.run(app, host=config.web_host, port=config.web_port)
            return 0

        if args.run_orchestrator_server:
            _assert_pi_runtime_root(args.env_file, command_name="run-orchestrator-server")
            from twinr.orchestrator import create_app

            try:
                import uvicorn
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "The web/orchestrator dependencies are not installed. Run `pip install -e .` in /twinr."
                ) from exc

            app = create_app(Path(args.env_file))
            uvicorn.run(app, host=config.orchestrator_host, port=config.orchestrator_port)
            return 0

        if args.display_test:
            from twinr.display import WaveshareEPD4In2V2

            display = WaveshareEPD4In2V2.from_config(config)
            display.show_test_pattern()
            print("display_test=ok")
            return 0

        if args.run_display_loop:
            _assert_pi_runtime_root(args.env_file, command_name="run-display-loop")
            from twinr.display import TwinrStatusDisplayLoop
            from twinr.ops import loop_instance_lock

            loop = TwinrStatusDisplayLoop.from_config(config)
            with loop_instance_lock(config, "display-loop"):
                return loop.run(duration_s=args.loop_duration)

        if args.run_realtime_loop:
            _assert_pi_runtime_root(args.env_file, command_name="run-realtime-loop")
            from twinr.agent.workflows.realtime_runner import TwinrRealtimeHardwareLoop
            from twinr.display.companion import optional_display_companion
            from twinr.ops import loop_instance_lock

            if _should_ensure_remote_watchdog_companion(config, args.env_file):
                from twinr.ops.remote_memory_watchdog_companion import ensure_remote_memory_watchdog_process

                ensure_remote_memory_watchdog_process(config, env_file=args.env_file)
            loop = TwinrRealtimeHardwareLoop(
                config=config,
                runtime=runtime,
                print_backend=backend,
            )
            with loop_instance_lock(config, "realtime-loop"):
                with optional_display_companion(config, enabled=_should_enable_display_companion(args.env_file)):
                    return loop.run(duration_s=args.loop_duration)

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

        if args.run_hardware_loop and backend is not None:
            _assert_pi_runtime_root(args.env_file, command_name="run-hardware-loop")
            from twinr.agent.legacy.classic_hardware_loop import TwinrHardwareLoop
            from twinr.display.companion import optional_display_companion
            from twinr.ops import loop_instance_lock

            if _should_ensure_remote_watchdog_companion(config, args.env_file):
                from twinr.ops.remote_memory_watchdog_companion import ensure_remote_memory_watchdog_process

                ensure_remote_memory_watchdog_process(config, env_file=args.env_file)
            loop = TwinrHardwareLoop(
                config=config,
                runtime=runtime,
                backend=backend,
            )
            with loop_instance_lock(config, "hardware-loop"):
                with optional_display_companion(config, enabled=_should_enable_display_companion(args.env_file)):
                    return loop.run(duration_s=args.loop_duration)

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
            if backend is None or camera is None:
                raise RuntimeError("Proactive observation requires configured OpenAI and camera access")
            from twinr.proactive import OpenAIVisionObservationProvider

            observer = OpenAIVisionObservationProvider(
                backend=backend,
                camera=camera,
                camera_lock=Lock(),
            )
            snapshot = observer.observe()
            print(f"proactive_person_visible={str(snapshot.observation.person_visible).lower()}")
            print(f"proactive_looking_toward_device={str(snapshot.observation.looking_toward_device).lower()}")
            print(f"proactive_body_pose={snapshot.observation.body_pose.value}")
            print(f"proactive_smiling={str(snapshot.observation.smiling).lower()}")
            print(
                "proactive_hand_or_object_near_camera="
                f"{str(snapshot.observation.hand_or_object_near_camera).lower()}"
            )
            if snapshot.response_id:
                print(f"proactive_response_id={snapshot.response_id}")
            if snapshot.request_id:
                print(f"proactive_request_id={snapshot.request_id}")
            if snapshot.model:
                print(f"proactive_model={snapshot.model}")

        if args.proactive_audio_observe_once:
            from twinr.hardware.audio import AmbientAudioSampler
            from twinr.proactive import AmbientAudioObservationProvider

            observer = AmbientAudioObservationProvider(
                sampler=AmbientAudioSampler.from_config(config),
                audio_lock=Lock(),
                sample_ms=config.proactive_audio_sample_ms,
                distress_enabled=config.proactive_audio_distress_enabled,
            )
            snapshot = observer.observe()
            print(f"proactive_speech_detected={str(snapshot.observation.speech_detected).lower()}")
            if snapshot.observation.distress_detected is not None:
                print(f"proactive_distress_detected={str(snapshot.observation.distress_detected).lower()}")
            if snapshot.sample is not None:
                print(f"proactive_audio_peak_rms={snapshot.sample.peak_rms}")
                print(f"proactive_audio_average_rms={snapshot.sample.average_rms}")
                print(f"proactive_audio_active_ratio={snapshot.sample.active_ratio:.2f}")

        if args.wakeword_label_capture:
            if not args.wakeword_label:
                raise RuntimeError("--wakeword-label is required with --wakeword-label-capture")
            from twinr.proactive import append_wakeword_capture_label

            entry = append_wakeword_capture_label(
                config,
                capture_path=args.wakeword_label_capture,
                label=args.wakeword_label,
                notes=args.wakeword_label_notes,
            )
            print(f"wakeword_label_capture={args.wakeword_label_capture}")
            print(f"wakeword_label={entry['data']['label']}")
            return 0

        if args.wakeword_eval:
            from twinr.proactive import run_wakeword_eval

            verifier_backend = backend
            if verifier_backend is None and config.wakeword_verifier_mode != "disabled" and config.openai_api_key:
                from twinr.providers.openai import OpenAIBackend

                verifier_backend = OpenAIBackend(config=config)
            report = run_wakeword_eval(
                config=config,
                manifest_path=args.wakeword_manifest,
                backend=verifier_backend,
            )
            print(f"wakeword_eval_entries={report.evaluated_entries}")
            print(f"wakeword_eval_precision={report.metrics.precision:.4f}")
            print(f"wakeword_eval_recall={report.metrics.recall:.4f}")
            print(f"wakeword_eval_fpr={report.metrics.false_positive_rate:.4f}")
            print(f"wakeword_eval_fnr={report.metrics.false_negative_rate:.4f}")
            if report.report_path is not None:
                print(f"wakeword_eval_report={report.report_path}")
            return 0

        if args.wakeword_autotune:
            from twinr.proactive import autotune_wakeword_profile

            verifier_backend = backend
            if verifier_backend is None and config.wakeword_verifier_mode != "disabled" and config.openai_api_key:
                from twinr.providers.openai import OpenAIBackend

                verifier_backend = OpenAIBackend(config=config)
            recommendation = autotune_wakeword_profile(
                config=config,
                manifest_path=args.wakeword_manifest,
                backend=verifier_backend,
            )
            print(f"wakeword_autotune_precision={recommendation.metrics.precision:.4f}")
            print(f"wakeword_autotune_recall={recommendation.metrics.recall:.4f}")
            print(f"wakeword_autotune_fpr={recommendation.metrics.false_positive_rate:.4f}")
            print(f"wakeword_autotune_score={recommendation.score:.4f}")
            if recommendation.profile_path is not None:
                print(f"wakeword_autotune_profile={recommendation.profile_path}")
            print(f"wakeword_autotune_threshold={recommendation.profile.threshold}")
            print(f"wakeword_autotune_patience_frames={recommendation.profile.patience_frames}")
            print(f"wakeword_autotune_activation_samples={recommendation.profile.activation_samples}")
            return 0

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
            print(f"status={runtime.status.value}")
            print(f"error={exc}")
            return 1
        runtime.fail(str(exc))
        print(f"status={runtime.status.value}")
        print(f"error={exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
