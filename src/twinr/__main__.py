from __future__ import annotations

import argparse
from pathlib import Path
from threading import Lock

from twinr.agent.base_agent import TwinrConfig, TwinrRuntime


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


def main() -> int:
    args = build_parser().parse_args()
    config = TwinrConfig.from_env(Path(args.env_file))
    runtime = TwinrRuntime(config=config)

    print(f"status={runtime.status.value}")
    print(f"web_port={config.web_port}")
    print(f"model={config.default_model}")
    print(f"openai_reasoning_effort={config.openai_reasoning_effort}")

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
        ]
    )
    uses_camera = bool(args.vision_camera_capture or args.camera_capture_output or args.proactive_observe_once)

    try:
        backend = None
        if uses_openai:
            from twinr.providers.openai import OpenAIBackend

            backend = OpenAIBackend(config=config)

        camera = None
        if uses_camera:
            from twinr.hardware.camera import V4L2StillCamera

            camera = V4L2StillCamera.from_config(config)

        if args.run_web:
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

        if args.display_test:
            from twinr.display import WaveshareEPD4In2V2

            display = WaveshareEPD4In2V2.from_config(config)
            display.show_test_pattern()
            print("display_test=ok")
            return 0

        if args.run_display_loop:
            from twinr.display import TwinrStatusDisplayLoop
            from twinr.ops import loop_instance_lock

            loop = TwinrStatusDisplayLoop.from_config(config)
            with loop_instance_lock(config, "display-loop"):
                return loop.run(duration_s=args.loop_duration)

        if args.run_realtime_loop:
            from twinr.agent.workflows.realtime_runner import TwinrRealtimeHardwareLoop
            from twinr.ops import loop_instance_lock

            loop = TwinrRealtimeHardwareLoop(
                config=config,
                runtime=runtime,
                print_backend=backend,
            )
            with loop_instance_lock(config, "realtime-loop"):
                return loop.run(duration_s=args.loop_duration)

        if args.run_streaming_loop:
            from twinr.agent.workflows.streaming_runner import TwinrStreamingHardwareLoop
            from twinr.ops import loop_instance_lock
            from twinr.providers.openai import OpenAIProviderBundle

            if backend is None:
                raise RuntimeError("Streaming loop requires configured providers")
            provider_bundle = OpenAIProviderBundle.from_backend(backend)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                print_backend=provider_bundle.combined,
                stt_provider=provider_bundle.stt,
                agent_provider=provider_bundle.agent,
                tts_provider=provider_bundle.tts,
                tool_agent_provider=provider_bundle.tool_agent,
            )
            with loop_instance_lock(config, "streaming-loop"):
                return loop.run(duration_s=args.loop_duration)

        if args.run_hardware_loop and backend is not None:
            from twinr.agent.workflows.runner import TwinrHardwareLoop
            from twinr.ops import loop_instance_lock

            loop = TwinrHardwareLoop(
                config=config,
                runtime=runtime,
                backend=backend,
            )
            with loop_instance_lock(config, "hardware-loop"):
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
        runtime.fail(str(exc))
        print(f"status={runtime.status.value}")
        print(f"error={exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
