"""Bootstrap the Twinr command-line entrypoint.

This module keeps argument parsing and high-level command dispatch at the
package root so operators can launch Twinr via ``python -m twinr`` or the
installed ``twinr`` script. Runtime behavior stays in the focused subsystem
packages that this bootstrap imports on demand.
"""

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
        "--wakeword-eval",
        action="store_true",
        help="Evaluate the current wakeword detector against labeled captures or a JSONL manifest.",
    )
    parser.add_argument(
        "--wakeword-stream-eval",
        action="store_true",
        help="Replay labeled captures through Twinr's runtime-faithful wakeword streaming path.",
    )
    parser.add_argument(
        "--wakeword-autotune",
        action="store_true",
        help="Search for a better wakeword calibration profile from labeled captures or a JSONL manifest.",
    )
    parser.add_argument(
        "--wakeword-manifest",
        type=Path,
        help="Optional JSONL or JSON-array manifest of labeled wakeword captures for eval/autotune/verifier training or post-training acceptance tuning.",
    )
    parser.add_argument(
        "--wakeword-training-plan",
        action="store_true",
        help="Render the canonical Twinr wakeword training and Pi-acceptance plan.",
    )
    parser.add_argument(
        "--wakeword-training-plan-output",
        type=Path,
        help="Optional Markdown output path for --wakeword-training-plan.",
    )
    parser.add_argument(
        "--wakeword-export-wekws",
        action="store_true",
        help="Export labeled Twinr wakeword manifests into WeKws/Kaldi-style split directories.",
    )
    parser.add_argument(
        "--wakeword-wekws-output-dir",
        type=Path,
        help="Output directory for --wakeword-export-wekws.",
    )
    parser.add_argument(
        "--wakeword-wekws-train-manifest",
        type=Path,
        help="Required train manifest for --wakeword-export-wekws.",
    )
    parser.add_argument(
        "--wakeword-wekws-dev-manifest",
        type=Path,
        help="Optional dev manifest for --wakeword-export-wekws.",
    )
    parser.add_argument(
        "--wakeword-wekws-test-manifest",
        type=Path,
        help="Optional test manifest for --wakeword-export-wekws.",
    )
    parser.add_argument(
        "--wakeword-wekws-positive-token",
        default="TWINR_FAMILY",
        help="Positive class token for --wakeword-export-wekws; angle brackets are added automatically.",
    )
    parser.add_argument(
        "--wakeword-wekws-filler-token",
        default="FILLER",
        help="Negative filler token for --wakeword-export-wekws; angle brackets are added automatically.",
    )
    parser.add_argument(
        "--wakeword-prepare-wekws-experiment",
        action="store_true",
        help="Prepare a reproducible WeKws training workspace from an exported Twinr WeKws dataset.",
    )
    parser.add_argument(
        "--wakeword-wekws-dataset-dir",
        type=Path,
        help="Input dataset directory produced by --wakeword-export-wekws.",
    )
    parser.add_argument(
        "--wakeword-wekws-experiment-dir",
        type=Path,
        help="Output directory for --wakeword-prepare-wekws-experiment.",
    )
    parser.add_argument(
        "--wakeword-wekws-recipe",
        default="mdtc_fbank_stream",
        help="Built-in WeKws experiment recipe id for --wakeword-prepare-wekws-experiment.",
    )
    parser.add_argument(
        "--wakeword-wekws-gpus",
        default="0",
        help="Comma-separated GPU ids to embed into the generated WeKws runner script.",
    )
    parser.add_argument(
        "--wakeword-wekws-num-workers",
        type=int,
        default=8,
        help="Data-loader worker count to embed into the generated WeKws runner script.",
    )
    parser.add_argument(
        "--wakeword-wekws-cmvn-num-workers",
        type=int,
        default=16,
        help="CMVN worker count to embed into the generated WeKws runner script.",
    )
    parser.add_argument(
        "--wakeword-wekws-min-duration-frames",
        type=int,
        default=50,
        help="Minimum keyword duration in frames for the generated WeKws training command.",
    )
    parser.add_argument(
        "--wakeword-wekws-seed",
        type=int,
        default=666,
        help="Random seed to embed into the generated WeKws training command.",
    )
    parser.add_argument(
        "--wakeword-wekws-base-checkpoint",
        type=Path,
        help="Optional upstream WeKws checkpoint to resume or fine-tune from.",
    )
    parser.add_argument(
        "--wakeword-kws-provision",
        action="store_true",
        help="Download and prepare one official sherpa-onnx KWS bundle plus Twinr keyword files.",
    )
    parser.add_argument(
        "--wakeword-kws-bundle",
        default="gigaspeech_3_3m_bpe_int8",
        help="Built-in sherpa-onnx KWS bundle id for --wakeword-kws-provision.",
    )
    parser.add_argument(
        "--wakeword-kws-output-dir",
        type=Path,
        help="Where to write the prepared sherpa-onnx KWS bundle; defaults to src/twinr/proactive/wakeword/models/kws.",
    )
    parser.add_argument(
        "--wakeword-kws-keyword",
        action="append",
        default=[],
        help="Explicit KWS keyword phrase; repeat to override derivation from the configured wakeword phrases.",
    )
    parser.add_argument(
        "--wakeword-kws-lexicon-entry",
        action="append",
        default=[],
        help=(
            "Optional custom phone-lexicon entry in the form WORD=PHONE PHONE ...; "
            "repeat to add multiple pronunciations or words for phone-based bundles."
        ),
    )
    parser.add_argument(
        "--wakeword-kws-force",
        action="store_true",
        help="Overwrite an existing KWS output directory for --wakeword-kws-provision.",
    )
    parser.add_argument(
        "--wakeword-promotion-eval",
        action="store_true",
        help="Run runtime-faithful suite and ambient promotion guards from one JSON spec.",
    )
    parser.add_argument(
        "--wakeword-promotion-spec",
        type=Path,
        help="JSON spec for --wakeword-promotion-eval with labeled suites and ambient guards.",
    )
    parser.add_argument(
        "--wakeword-train-model",
        action="store_true",
        help="Train a local openWakeWord-compatible Twinr base model from a generated dataset root.",
    )
    parser.add_argument(
        "--wakeword-dataset-root",
        type=Path,
        help="Dataset root produced by scripts/generate_multivoice_dataset.py for base-model training.",
    )
    parser.add_argument(
        "--wakeword-model-output",
        type=Path,
        help="Where to write the trained wakeword .onnx model.",
    )
    parser.add_argument(
        "--wakeword-model-metadata-output",
        type=Path,
        help="Optional metadata .json output path for --wakeword-train-model.",
    )
    parser.add_argument(
        "--wakeword-training-workdir",
        type=Path,
        help="Optional working directory for intermediate wakeword training features.",
    )
    parser.add_argument(
        "--wakeword-training-rounds",
        type=int,
        default=2,
        help="How many fixed-length jitter variants to build per training clip.",
    )
    parser.add_argument(
        "--wakeword-training-steps",
        type=int,
        default=20000,
        help="Maximum training steps for the openWakeWord base-model loop.",
    )
    parser.add_argument(
        "--wakeword-training-layer-dim",
        type=int,
        default=128,
        help="Hidden layer width for the trained wakeword detector.",
    )
    parser.add_argument(
        "--wakeword-training-model-type",
        default="mlp",
        help="Wakeword base-model training backend: mlp, dnn, or rnn.",
    )
    parser.add_argument(
        "--wakeword-training-feature-device",
        default="cpu",
        help="Feature extraction device for wakeword training: cpu or gpu.",
    )
    parser.add_argument(
        "--wakeword-training-difficulty-model",
        type=Path,
        help="Optional reference .onnx detector used to upweight deployment-proximate hard examples during MLP training.",
    )
    parser.add_argument(
        "--wakeword-training-difficulty-positive-scale",
        type=float,
        default=0.0,
        help="Additional positive hard-example emphasis derived from the reference detector scores.",
    )
    parser.add_argument(
        "--wakeword-training-difficulty-negative-scale",
        type=float,
        default=0.0,
        help="Additional negative hard-example emphasis derived from the reference detector scores.",
    )
    parser.add_argument(
        "--wakeword-training-difficulty-power",
        type=float,
        default=2.0,
        help="Exponent applied to difficulty-derived sample weighting during MLP training.",
    )
    parser.add_argument(
        "--wakeword-train-verifier",
        action="store_true",
        help="Train a local openWakeWord custom verifier from a labeled wakeword manifest.",
    )
    parser.add_argument(
        "--wakeword-train-sequence-verifier",
        action="store_true",
        help="Train a Twinr clip-level sequence verifier for the openWakeWord cascade from a labeled wakeword manifest.",
    )
    parser.add_argument(
        "--wakeword-verifier-output",
        type=Path,
        help="Where to write the trained wakeword verifier .pkl file.",
    )
    parser.add_argument(
        "--wakeword-verifier-model",
        help="Optional wakeword model name or local model path used to train the verifier.",
    )
    parser.add_argument(
        "--wakeword-sequence-verifier-output",
        type=Path,
        help="Where to write the trained Twinr sequence verifier .pkl file.",
    )
    parser.add_argument(
        "--wakeword-sequence-verifier-model",
        help="Wakeword model name or local model path used as the stage-1 detector for sequence-verifier training.",
    )
    parser.add_argument(
        "--wakeword-sequence-verifier-aux-model",
        action="append",
        default=[],
        help="Optional auxiliary wakeword model path or name used to add extra score tracks to the sequence verifier; repeat for multiple models.",
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


def _default_wakeword_verifier_output_path(model_name: str | None) -> Path | None:
    """Infer one sibling verifier path for a local wakeword model file."""

    if model_name is None:
        return None
    normalized_model_name = str(model_name).strip()
    if not normalized_model_name:
        return None
    candidate = Path(normalized_model_name).expanduser()
    if not candidate.exists():
        return None
    return candidate.resolve(strict=False).with_suffix(".verifier.pkl")


def _default_wakeword_sequence_verifier_output_path(model_name: str | None) -> Path | None:
    """Infer one sibling sequence-verifier path for a local wakeword model file."""

    if model_name is None:
        return None
    normalized_model_name = str(model_name).strip()
    if not normalized_model_name:
        return None
    candidate = Path(normalized_model_name).expanduser()
    if not candidate.exists():
        return None
    return candidate.resolve(strict=False).with_suffix(".sequence_verifier.pkl")


def _default_wakeword_kws_output_dir(config: TwinrConfig) -> Path:
    """Return the canonical repo-local directory for provisioned KWS assets."""

    return Path(config.project_root).resolve(strict=False) / "src/twinr/proactive/wakeword/models/kws"


def _parse_kws_lexicon_entries(values: list[str] | tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Parse repeated WORD=PHONE PHONE CLI flags into one lexicon-entry mapping."""

    parsed: dict[str, list[str]] = {}
    for raw_value in values:
        text = str(raw_value or "").strip()
        if not text:
            continue
        word, separator, pronunciation = text.partition("=")
        word = word.strip()
        pronunciation = " ".join(pronunciation.split())
        if separator != "=" or not word or not pronunciation:
            raise ValueError(
                "Each --wakeword-kws-lexicon-entry must use WORD=PHONE PHONE ... syntax."
            )
        parsed.setdefault(word, []).append(pronunciation)
    return {word: tuple(pronunciations) for word, pronunciations in parsed.items()}


def _build_wakeword_verifier_backend(config: TwinrConfig, backend) -> Any:
    """Build one verifier backend when wakeword verification needs STT."""

    if backend is not None:
        return backend
    if config.wakeword_verifier_mode == "disabled" or not config.openai_api_key:
        return None
    from twinr.providers.openai import OpenAIBackend

    return OpenAIBackend(config=config)


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

    if args.wakeword_training_plan:
        from twinr.proactive.wakeword import (
            build_default_wakeword_training_plan,
            render_wakeword_training_plan_markdown,
        )

        plan = build_default_wakeword_training_plan(project_root=config.project_root)
        markdown = render_wakeword_training_plan_markdown(plan)
        if args.wakeword_training_plan_output is not None:
            output_path = args.wakeword_training_plan_output.expanduser().resolve(strict=False)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown, encoding="utf-8")
            print(f"wakeword_training_plan_output={output_path}")
            print(f"wakeword_training_plan_model_name={plan.stage1_model_name}")
            print(f"wakeword_training_plan_phrase_profile={plan.stage1_phrase_profile}")
            return 0
        print(markdown.rstrip())
        return 0

    if args.wakeword_export_wekws:
        if args.wakeword_wekws_output_dir is None:
            raise RuntimeError("--wakeword-wekws-output-dir is required with --wakeword-export-wekws")
        if args.wakeword_wekws_train_manifest is None:
            raise RuntimeError("--wakeword-wekws-train-manifest is required with --wakeword-export-wekws")
        from twinr.proactive.wakeword import export_wakeword_manifests_to_wekws

        report = export_wakeword_manifests_to_wekws(
            output_dir=args.wakeword_wekws_output_dir,
            train_manifest=args.wakeword_wekws_train_manifest,
            dev_manifest=args.wakeword_wekws_dev_manifest,
            test_manifest=args.wakeword_wekws_test_manifest,
            positive_token=args.wakeword_wekws_positive_token,
            filler_token=args.wakeword_wekws_filler_token,
        )
        print(f"wakeword_wekws_output_dir={report.output_dir}")
        print(f"wakeword_wekws_dict={report.dict_path}")
        print(f"wakeword_wekws_words={report.words_path}")
        print(f"wakeword_wekws_metadata={report.metadata_path}")
        for split_report in report.split_reports:
            prefix = f"wakeword_wekws_{split_report.split_name}"
            print(f"{prefix}_manifest={split_report.manifest_path}")
            print(f"{prefix}_output_dir={split_report.output_dir}")
            print(f"{prefix}_entries={split_report.entry_count}")
            print(f"{prefix}_positive={split_report.positive_count}")
            print(f"{prefix}_negative={split_report.negative_count}")
            print(f"{prefix}_ignored={split_report.ignored_count}")
        return 0

    if args.wakeword_prepare_wekws_experiment:
        if args.wakeword_wekws_dataset_dir is None:
            raise RuntimeError("--wakeword-wekws-dataset-dir is required with --wakeword-prepare-wekws-experiment")
        if args.wakeword_wekws_experiment_dir is None:
            raise RuntimeError(
                "--wakeword-wekws-experiment-dir is required with --wakeword-prepare-wekws-experiment"
            )
        from twinr.proactive.wakeword import prepare_wekws_experiment

        report = prepare_wekws_experiment(
            output_dir=args.wakeword_wekws_experiment_dir,
            exported_dataset_dir=args.wakeword_wekws_dataset_dir,
            recipe_id=str(args.wakeword_wekws_recipe or "").strip() or "mdtc_fbank_stream",
            seed=int(args.wakeword_wekws_seed),
            gpus=str(args.wakeword_wekws_gpus or "0"),
            num_workers=int(args.wakeword_wekws_num_workers),
            cmvn_num_workers=int(args.wakeword_wekws_cmvn_num_workers),
            min_duration_frames=int(args.wakeword_wekws_min_duration_frames),
            base_checkpoint=args.wakeword_wekws_base_checkpoint,
        )
        print(f"wakeword_wekws_experiment_dir={report.output_dir}")
        print(f"wakeword_wekws_recipe={report.recipe.recipe_id}")
        print(f"wakeword_wekws_config={report.config_path}")
        print(f"wakeword_wekws_dict={report.dict_dir}")
        print(f"wakeword_wekws_model_dir={report.model_dir}")
        print(f"wakeword_wekws_script={report.script_path}")
        print(f"wakeword_wekws_metadata={report.metadata_path}")
        for split_report in report.split_reports:
            prefix = f"wakeword_wekws_{split_report.split_name}"
            print(f"{prefix}_source_dir={split_report.source_dir}")
            print(f"{prefix}_output_dir={split_report.output_dir}")
            print(f"{prefix}_entries={split_report.utterance_count}")
            print(f"{prefix}_data_list={split_report.data_list_path}")
        return 0

    if args.wakeword_kws_provision:
        from twinr.proactive.wakeword import provision_builtin_kws_bundle

        report = provision_builtin_kws_bundle(
            output_dir=args.wakeword_kws_output_dir or _default_wakeword_kws_output_dir(config),
            bundle_id=str(args.wakeword_kws_bundle or "").strip() or "gigaspeech_3_3m_bpe_int8",
            phrases=config.wakeword_phrases,
            explicit_keywords=tuple(str(item).strip() for item in (args.wakeword_kws_keyword or []) if str(item).strip()),
            lexicon_entries=_parse_kws_lexicon_entries(args.wakeword_kws_lexicon_entry or []),
            force=args.wakeword_kws_force,
        )
        print(f"wakeword_kws_bundle_id={report.bundle_id}")
        print(f"wakeword_kws_output_dir={report.output_dir}")
        print("wakeword_kws_keywords=" + ",".join(report.keyword_names))
        print(f"wakeword_kws_tokens_path={report.tokens_path}")
        print(f"wakeword_kws_encoder_path={report.encoder_path}")
        print(f"wakeword_kws_decoder_path={report.decoder_path}")
        print(f"wakeword_kws_joiner_path={report.joiner_path}")
        print(f"wakeword_kws_keywords_file_path={report.keywords_path}")
        if report.lexicon_path is not None:
            print(f"wakeword_kws_lexicon_path={report.lexicon_path}")
        return 0

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
            from twinr.ops import loop_instance_lock
            from twinr.providers import build_streaming_provider_bundle
            from twinr.providers.openai import OpenAIBackend

            _ensure_remote_watchdog_for_runtime_boot(config, args.env_file)

            with loop_instance_lock(config, "streaming-loop"):
                with optional_display_companion(
                    config,
                    enabled=_should_enable_display_companion(config, args.env_file),
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

        if args.run_realtime_loop:
            _assert_pi_runtime_root(args.env_file, command_name="run-realtime-loop")
            from twinr.agent.workflows.runtime_error_hold import hold_runtime_error_state
            from twinr.agent.workflows.realtime_runner import TwinrRealtimeHardwareLoop
            from twinr.display.companion import optional_display_companion
            from twinr.ops import loop_instance_lock

            with loop_instance_lock(config, "realtime-loop"):
                with optional_display_companion(
                    config,
                    enabled=_should_enable_display_companion(config, args.env_file),
                ):
                    try:
                        loop = TwinrRealtimeHardwareLoop(
                            config=config,
                            runtime=runtime,
                            print_backend=backend,
                        )
                        return loop.run(duration_s=args.loop_duration)
                    except Exception as exc:
                        return hold_runtime_error_state(
                            runtime=runtime,
                            error=exc,
                            duration_s=args.loop_duration,
                        )

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

        if args.run_hardware_loop and backend is not None:
            _assert_pi_runtime_root(args.env_file, command_name="run-hardware-loop")
            from twinr.agent.legacy.classic_hardware_loop import TwinrHardwareLoop
            from twinr.agent.workflows.runtime_error_hold import hold_runtime_error_state
            from twinr.display.companion import optional_display_companion
            from twinr.ops import loop_instance_lock

            with loop_instance_lock(config, "hardware-loop"):
                with optional_display_companion(
                    config,
                    enabled=_should_enable_display_companion(config, args.env_file),
                ):
                    try:
                        loop = TwinrHardwareLoop(
                            config=config,
                            runtime=runtime,
                            backend=backend,
                        )
                        return loop.run(duration_s=args.loop_duration)
                    except Exception as exc:
                        return hold_runtime_error_state(
                            runtime=runtime,
                            error=exc,
                            duration_s=args.loop_duration,
                        )

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

        if args.wakeword_train_model:
            if args.wakeword_dataset_root is None:
                raise RuntimeError("--wakeword-dataset-root is required with --wakeword-train-model")
            if args.wakeword_model_output is None:
                raise RuntimeError("--wakeword-model-output is required with --wakeword-train-model")
            from twinr.proactive.wakeword import train_wakeword_base_model_from_dataset_root

            report = train_wakeword_base_model_from_dataset_root(
                dataset_root=args.wakeword_dataset_root,
                output_model_path=args.wakeword_model_output,
                metadata_path=args.wakeword_model_metadata_output,
                acceptance_manifest=args.wakeword_manifest,
                workdir=args.wakeword_training_workdir,
                training_rounds=args.wakeword_training_rounds,
                model_type=str(args.wakeword_training_model_type or "mlp").strip().lower() or "mlp",
                layer_dim=args.wakeword_training_layer_dim,
                steps=args.wakeword_training_steps,
                feature_device=str(args.wakeword_training_feature_device or "cpu").strip().lower() or "cpu",
                difficulty_reference_model_path=args.wakeword_training_difficulty_model,
                difficulty_positive_scale=args.wakeword_training_difficulty_positive_scale,
                difficulty_negative_scale=args.wakeword_training_difficulty_negative_scale,
                difficulty_power=args.wakeword_training_difficulty_power,
                evaluation_config=config,
            )
            print(f"wakeword_model_dataset_root={report.dataset_root}")
            print(f"wakeword_model_output={report.output_model_path}")
            print(f"wakeword_model_metadata={report.metadata_path}")
            print(f"wakeword_model_total_length_samples={report.total_length_samples}")
            print(f"wakeword_model_train_positive_clips={report.train_positive_clips}")
            print(f"wakeword_model_train_negative_clips={report.train_negative_clips}")
            print(f"wakeword_model_validation_positive_clips={report.validation_positive_clips}")
            print(f"wakeword_model_validation_negative_clips={report.validation_negative_clips}")
            print(f"wakeword_model_selected_threshold={report.selected_threshold:.6f}")
            if report.acceptance_metrics is not None:
                print(f"wakeword_model_acceptance_precision={report.acceptance_metrics.precision:.4f}")
                print(f"wakeword_model_acceptance_recall={report.acceptance_metrics.recall:.4f}")
                print(f"wakeword_model_acceptance_fpr={report.acceptance_metrics.false_positive_rate:.4f}")
                print(f"wakeword_model_acceptance_fnr={report.acceptance_metrics.false_negative_rate:.4f}")
            return 0

        if args.wakeword_train_verifier:
            if args.wakeword_manifest is None:
                raise RuntimeError("--wakeword-manifest is required with --wakeword-train-verifier")
            from twinr.proactive.wakeword import train_wakeword_custom_verifier_from_manifest

            configured_models = tuple(str(item).strip() for item in config.wakeword_openwakeword_models if str(item).strip())
            model_name = str(args.wakeword_verifier_model or (configured_models[0] if configured_models else "")).strip()
            if not model_name:
                raise RuntimeError(
                    "Wakeword verifier training requires --wakeword-verifier-model or one configured openWakeWord model."
                )
            output_path = args.wakeword_verifier_output or _default_wakeword_verifier_output_path(model_name)
            if output_path is None:
                raise RuntimeError(
                    "--wakeword-verifier-output is required when the verifier model is not a local file path."
                )
            report = train_wakeword_custom_verifier_from_manifest(
                manifest_path=args.wakeword_manifest,
                output_path=output_path,
                model_name=model_name,
                inference_framework=config.wakeword_openwakeword_inference_framework,
            )
            print(f"wakeword_verifier_manifest={report.manifest_path}")
            print(f"wakeword_verifier_model={report.model_name}")
            print(f"wakeword_verifier_positive_clips={report.positive_clips}")
            print(f"wakeword_verifier_negative_clips={report.negative_clips}")
            print(f"wakeword_verifier_negative_seconds={report.negative_seconds:.3f}")
            print(f"wakeword_verifier_output={report.output_path}")
            return 0

        if args.wakeword_train_sequence_verifier:
            if args.wakeword_manifest is None:
                raise RuntimeError("--wakeword-manifest is required with --wakeword-train-sequence-verifier")
            from twinr.proactive.wakeword import train_wakeword_sequence_verifier_from_manifest

            configured_models = tuple(
                str(item).strip() for item in config.wakeword_openwakeword_models if str(item).strip()
            )
            model_name = str(
                args.wakeword_sequence_verifier_model or (configured_models[0] if configured_models else "")
            ).strip()
            if not model_name:
                raise RuntimeError(
                    "Sequence verifier training requires --wakeword-sequence-verifier-model or one configured openWakeWord model."
                )
            output_path = (
                args.wakeword_sequence_verifier_output
                or _default_wakeword_sequence_verifier_output_path(model_name)
            )
            if output_path is None:
                raise RuntimeError(
                    "--wakeword-sequence-verifier-output is required when the sequence-verifier model is not a local file path."
                )
            auxiliary_models = tuple(
                str(item).strip() for item in (args.wakeword_sequence_verifier_aux_model or []) if str(item).strip()
            )
            report = train_wakeword_sequence_verifier_from_manifest(
                manifest_path=args.wakeword_manifest,
                output_path=output_path,
                model_name=model_name,
                auxiliary_models=auxiliary_models,
                inference_framework=config.wakeword_openwakeword_inference_framework,
            )
            print(f"wakeword_sequence_verifier_manifest={report.manifest_path}")
            print(f"wakeword_sequence_verifier_model={report.model_name}")
            print(
                "wakeword_sequence_verifier_auxiliary_models="
                + ",".join(report.auxiliary_models)
            )
            print(f"wakeword_sequence_verifier_positive_clips={report.positive_clips}")
            print(f"wakeword_sequence_verifier_negative_clips={report.negative_clips}")
            print(f"wakeword_sequence_verifier_negative_seconds={report.negative_seconds:.3f}")
            print(f"wakeword_sequence_verifier_total_length_samples={report.total_length_samples}")
            print(f"wakeword_sequence_verifier_embedding_frames={report.embedding_frames}")
            print(f"wakeword_sequence_verifier_feature_dimensions={report.feature_dimensions}")
            print(f"wakeword_sequence_verifier_output={report.output_path}")
            return 0

        if args.wakeword_eval:
            from twinr.proactive import run_wakeword_eval

            verifier_backend = _build_wakeword_verifier_backend(config, backend)
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

        if args.wakeword_stream_eval:
            if args.wakeword_manifest is None:
                raise RuntimeError("--wakeword-manifest is required with --wakeword-stream-eval")
            from twinr.proactive import run_wakeword_stream_eval

            verifier_backend = _build_wakeword_verifier_backend(config, backend)
            report = run_wakeword_stream_eval(
                config=config,
                manifest_path=args.wakeword_manifest,
                backend=verifier_backend,
            )
            print(f"wakeword_stream_eval_entries={report.evaluated_entries}")
            print(f"wakeword_stream_eval_precision={report.metrics.precision:.4f}")
            print(f"wakeword_stream_eval_recall={report.metrics.recall:.4f}")
            print(f"wakeword_stream_eval_fpr={report.metrics.false_positive_rate:.4f}")
            print(f"wakeword_stream_eval_fnr={report.metrics.false_negative_rate:.4f}")
            print(f"wakeword_stream_eval_accepted_detections={report.accepted_detection_count}")
            print(f"wakeword_stream_eval_total_audio_seconds={report.total_audio_seconds:.3f}")
            if report.report_path is not None:
                print(f"wakeword_stream_eval_report={report.report_path}")
            return 0

        if args.wakeword_promotion_eval:
            if args.wakeword_promotion_spec is None:
                raise RuntimeError("--wakeword-promotion-spec is required with --wakeword-promotion-eval")
            from twinr.proactive import run_wakeword_promotion_eval

            verifier_backend = _build_wakeword_verifier_backend(config, backend)
            report = run_wakeword_promotion_eval(
                config=config,
                spec_path=args.wakeword_promotion_spec,
                backend=verifier_backend,
            )
            print(f"wakeword_promotion_passed={str(report.passed).lower()}")
            print(f"wakeword_promotion_blocker_count={len(report.blockers)}")
            if report.blockers:
                print("wakeword_promotion_blockers=" + " | ".join(report.blockers))
            for index, suite_result in enumerate(report.suite_results):
                prefix = f"wakeword_promotion_suite_{index}"
                print(f"{prefix}_name={suite_result.spec.name}")
                print(f"{prefix}_precision={suite_result.report.metrics.precision:.4f}")
                print(f"{prefix}_recall={suite_result.report.metrics.recall:.4f}")
                print(f"{prefix}_fp={suite_result.report.metrics.false_positive}")
                print(f"{prefix}_fn={suite_result.report.metrics.false_negative}")
            for index, ambient_result in enumerate(report.ambient_results):
                prefix = f"wakeword_promotion_ambient_{index}"
                print(f"{prefix}_name={ambient_result.spec.name}")
                print(f"{prefix}_false_accepts_per_hour={ambient_result.false_accepts_per_hour:.6f}")
                print(f"{prefix}_accepted_detections={ambient_result.report.accepted_detection_count}")
            if report.report_path is not None:
                print(f"wakeword_promotion_report={report.report_path}")
            return 0

        if args.wakeword_autotune:
            from twinr.proactive import autotune_wakeword_profile

            verifier_backend = _build_wakeword_verifier_backend(config, backend)
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
