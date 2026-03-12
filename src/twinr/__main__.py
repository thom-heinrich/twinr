from __future__ import annotations

import argparse
from pathlib import Path

from twinr.config import TwinrConfig
from twinr.providers.openai_backend import OpenAIBackend
from twinr.realtime_runner import TwinrRealtimeHardwareLoop
from twinr.runner import TwinrHardwareLoop
from twinr.runtime import TwinrRuntime


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
        "--loop-duration",
        type=float,
        help="Optional max runtime in seconds for --run-hardware-loop",
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
            args.audio_file,
            args.tts_text,
            args.format_for_print,
            args.run_hardware_loop,
            args.run_realtime_loop,
        ]
    )

    try:
        backend = OpenAIBackend(config=config) if uses_openai else None

        if args.run_realtime_loop:
            loop = TwinrRealtimeHardwareLoop(
                config=config,
                runtime=runtime,
                print_backend=backend,
            )
            return loop.run(duration_s=args.loop_duration)

        if args.run_hardware_loop and backend is not None:
            loop = TwinrHardwareLoop(
                config=config,
                runtime=runtime,
                backend=backend,
            )
            return loop.run(duration_s=args.loop_duration)

        if args.audio_file and backend is not None:
            transcript = backend.transcribe_path(args.audio_file)
            print(f"transcript={transcript}")

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
