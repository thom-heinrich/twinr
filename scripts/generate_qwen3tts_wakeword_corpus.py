#!/usr/bin/env python3
"""Generate a large synthetic Twinr wakeword corpus with Qwen3TTS.

Purpose
-------
Build a deterministic, shardable synthetic corpus for Twinr's professional
WeKws training path. The script loads Qwen3TTS once on one GPU, synthesizes a
broad wakeword-family plus hard-negative corpus across many preset speakers and
style conditions, applies lightweight channel degradations, and writes
Twinr-compatible train/dev/test manifests.

Usage
-----
Command-line invocation examples::

    PYTHONPATH=src python3 scripts/generate_qwen3tts_wakeword_corpus.py \
      --output-root /tmp/twinr_qwen3tts_corpus_v1 \
      --device cuda:0 \
      --speaker-shard-index 0 \
      --speaker-shard-count 2

    PYTHONPATH=src python3 scripts/generate_qwen3tts_wakeword_corpus.py \
      --output-root /tmp/twinr_qwen3tts_corpus_v1 \
      --device cuda:1 \
      --speaker-shard-index 1 \
      --speaker-shard-count 2 \
      --overwrite

Inputs
------
- Qwen3TTS Python package in the active environment (required)
- CUDA-capable PyTorch environment when ``--device`` targets a GPU
- ``--output-root`` target directory for manifests and audio

Outputs
-------
- ``<output-root>/audio/<split>/*.wav`` synthetic clips
- ``<output-root>/manifests/<split>.jsonl`` Twinr-compatible manifest files
- ``<output-root>/generation_summary.json`` deterministic run summary

Notes
-----
The script is resume-friendly by default: existing WAVs are skipped and missing
manifest entries are reconstructed from the deterministic plan when needed.
Use ``--overwrite`` to rebuild the selected shard from scratch.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    from twinr.proactive.wakeword.synthetic_corpus import (
        DEFAULT_AUGMENTATION_PROFILES,
        DEFAULT_FOLLOW_UPS,
        DEFAULT_GENERATION_PROFILES,
        DEFAULT_PREFIXES,
        DEFAULT_SEEDS,
        DEFAULT_STYLE_PROFILES,
        QWEN3TTS_SUPPORTED_SPEAKERS,
        SyntheticAugmentationProfile,
        SyntheticGenerationProfile,
        SyntheticStyleProfile,
        apply_augmentation,
        build_qwen3tts_synthetic_corpus_plan,
        manifest_row_for_request,
        write_pcm16_wav,
    )
except Exception:
    _MODULE_PATH = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "twinr"
        / "proactive"
        / "wakeword"
        / "synthetic_corpus.py"
    )
    _SPEC = importlib.util.spec_from_file_location("twinr_synthetic_corpus", _MODULE_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise RuntimeError(f"Unable to load synthetic corpus module from {_MODULE_PATH}")
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules.setdefault("twinr_synthetic_corpus", _MODULE)
    _SPEC.loader.exec_module(_MODULE)
    DEFAULT_AUGMENTATION_PROFILES = _MODULE.DEFAULT_AUGMENTATION_PROFILES
    DEFAULT_FOLLOW_UPS = _MODULE.DEFAULT_FOLLOW_UPS
    DEFAULT_GENERATION_PROFILES = _MODULE.DEFAULT_GENERATION_PROFILES
    DEFAULT_PREFIXES = _MODULE.DEFAULT_PREFIXES
    DEFAULT_SEEDS = _MODULE.DEFAULT_SEEDS
    DEFAULT_STYLE_PROFILES = _MODULE.DEFAULT_STYLE_PROFILES
    QWEN3TTS_SUPPORTED_SPEAKERS = _MODULE.QWEN3TTS_SUPPORTED_SPEAKERS
    SyntheticAugmentationProfile = _MODULE.SyntheticAugmentationProfile
    SyntheticGenerationProfile = _MODULE.SyntheticGenerationProfile
    SyntheticStyleProfile = _MODULE.SyntheticStyleProfile
    apply_augmentation = _MODULE.apply_augmentation
    build_qwen3tts_synthetic_corpus_plan = _MODULE.build_qwen3tts_synthetic_corpus_plan
    manifest_row_for_request = _MODULE.manifest_row_for_request
    write_pcm16_wav = _MODULE.write_pcm16_wav


def _parse_csv_strings(raw_values: list[str], *, default: tuple[str, ...]) -> tuple[str, ...]:
    values: list[str] = []
    for raw in raw_values:
        for item in str(raw).split(","):
            normalized_item = item.strip()
            if normalized_item.lower() in {"__empty__", "<empty>", "empty"}:
                values.append("")
            else:
                values.append(normalized_item)
    if not values:
        return default
    normalized = tuple(values)
    return normalized or default


def _select_style_profiles(keys: tuple[str, ...]) -> tuple[SyntheticStyleProfile, ...]:
    allowed = {key.strip().lower() for key in keys if key.strip()}
    if not allowed:
        return DEFAULT_STYLE_PROFILES
    selected = tuple(profile for profile in DEFAULT_STYLE_PROFILES if profile.key in allowed)
    if not selected:
        raise ValueError(f"No style profiles matched: {sorted(allowed)}")
    return selected


def _select_generation_profiles(keys: tuple[str, ...]) -> tuple[SyntheticGenerationProfile, ...]:
    allowed = {key.strip().lower() for key in keys if key.strip()}
    if not allowed:
        return DEFAULT_GENERATION_PROFILES
    selected = tuple(profile for profile in DEFAULT_GENERATION_PROFILES if profile.key in allowed)
    if not selected:
        raise ValueError(f"No generation profiles matched: {sorted(allowed)}")
    return selected


def _select_augmentation_profiles(keys: tuple[str, ...]) -> tuple[SyntheticAugmentationProfile, ...]:
    allowed = {key.strip().lower() for key in keys if key.strip()}
    if not allowed:
        return DEFAULT_AUGMENTATION_PROFILES
    selected = tuple(profile for profile in DEFAULT_AUGMENTATION_PROFILES if profile.key in allowed)
    if not selected:
        raise ValueError(f"No augmentation profiles matched: {sorted(allowed)}")
    return selected


def _load_existing_manifest_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        utterance_id = str(payload.get("utterance_id") or "").strip()
        if utterance_id:
            rows[utterance_id] = payload
    return rows


def _rewrite_manifests(
    output_root: Path,
    rows_by_split: dict[str, dict[str, dict[str, Any]]],
) -> None:
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in rows_by_split.items():
        manifest_path = manifests_dir / f"{split_name}.jsonl"
        ordered_rows = [rows[key] for key in sorted(rows)]
        manifest_path.write_text(
            "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in ordered_rows),
            encoding="utf-8",
        )


def _summary_from_rows(rows_by_split: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"splits": {}}
    for split_name, rows in rows_by_split.items():
        positives = sum(1 for row in rows.values() if str(row.get("label")) == "positive")
        negatives = sum(1 for row in rows.values() if str(row.get("label")) == "negative")
        summary["splits"][split_name] = {
            "count": len(rows),
            "positive_count": positives,
            "negative_count": negatives,
        }
    return summary


def _base_request_key(request) -> tuple[str, str, str, str, str, int]:
    """Return the synthesis key before augmentation fan-out.

    Requests that differ only by augmentation should share one expensive TTS
    forward pass and branch into lightweight local DSP afterwards.
    """

    return (
        str(request.text),
        str(request.speaker),
        str(request.style_key),
        str(request.generation_key),
        str(request.label),
        int(request.seed),
    )


def _load_qwen3tts_model(*, device: str, dtype: str):
    from qwen_tts import Qwen3TTSModel

    resolved_dtype: Any
    normalized_dtype = str(dtype or "auto").strip().lower()
    if normalized_dtype == "auto":
        resolved_dtype = "auto"
    else:
        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        try:
            resolved_dtype = dtype_map[normalized_dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {dtype}") from exc
    return Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=str(device),
        dtype=resolved_dtype,
    )


def _synthesize_one(model, *, request) -> tuple[np.ndarray, int]:
    import torch

    torch.manual_seed(int(request.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(request.seed))
    wavs, sample_rate = model.generate_custom_voice(
        text=request.text,
        speaker=request.speaker,
        language="german",
        instruct=request.style_instruction,
        top_p=request.top_p,
        temperature=request.temperature,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
    )
    samples = np.asarray(wavs[0], dtype=np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        samples = (samples / peak).astype(np.float32, copy=False)
    return samples, int(sample_rate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, help="Synthetic corpus output root.")
    parser.add_argument(
        "--speaker",
        action="append",
        default=[],
        help="Optional speaker id(s) or CSV values. Defaults to all known Qwen3TTS preset speakers.",
    )
    parser.add_argument("--speaker-shard-index", type=int, default=0)
    parser.add_argument("--speaker-shard-count", type=int, default=1)
    parser.add_argument(
        "--seed",
        action="append",
        default=[],
        help="Optional integer seed(s) or CSV values. Defaults to the built-in seed set.",
    )
    parser.add_argument(
        "--style-profile",
        action="append",
        default=[],
        help="Optional style profile key(s): plain,warm,fast,slow,soft,urgent.",
    )
    parser.add_argument(
        "--generation-profile",
        action="append",
        default=[],
        help="Optional generation profile key(s): stable,diverse.",
    )
    parser.add_argument(
        "--augmentation-profile",
        action="append",
        default=[],
        help="Optional augmentation profile key(s): clean,far_field,room_noise,phone_band,clipped.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="Optional spoken prefix variants as repeated values or CSV. Defaults to the built-in prefix set.",
    )
    parser.add_argument(
        "--follow-up",
        action="append",
        default=[],
        help="Optional spoken follow-up variants as repeated values or CSV. Defaults to the built-in follow-up set.",
    )
    parser.add_argument("--max-items", type=int, default=0, help="Optional cap for dry or smoke runs.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    if args.overwrite and output_root.exists():
        for path in sorted(output_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
    output_root.mkdir(parents=True, exist_ok=True)

    speakers = _parse_csv_strings(args.speaker, default=QWEN3TTS_SUPPORTED_SPEAKERS)
    seed_strings = _parse_csv_strings(args.seed, default=tuple(str(item) for item in DEFAULT_SEEDS))
    seeds = tuple(int(item) for item in seed_strings)
    style_profiles = _select_style_profiles(_parse_csv_strings(args.style_profile, default=()))
    generation_profiles = _select_generation_profiles(_parse_csv_strings(args.generation_profile, default=()))
    augmentation_profiles = _select_augmentation_profiles(_parse_csv_strings(args.augmentation_profile, default=()))
    prefixes = _parse_csv_strings(args.prefix, default=DEFAULT_PREFIXES)
    follow_ups = _parse_csv_strings(args.follow_up, default=DEFAULT_FOLLOW_UPS)

    requests = build_qwen3tts_synthetic_corpus_plan(
        speakers=speakers,
        seeds=seeds,
        style_profiles=style_profiles,
        generation_profiles=generation_profiles,
        augmentation_profiles=augmentation_profiles,
        prefixes=prefixes,
        follow_ups=follow_ups,
        shard_index=int(args.speaker_shard_index),
        shard_count=int(args.speaker_shard_count),
    )
    if args.max_items > 0:
        requests = requests[: int(args.max_items)]

    rows_by_split = {
        "train": _load_existing_manifest_rows(output_root / "manifests" / "train.jsonl"),
        "dev": _load_existing_manifest_rows(output_root / "manifests" / "dev.jsonl"),
        "test": _load_existing_manifest_rows(output_root / "manifests" / "test.jsonl"),
    }

    model = _load_qwen3tts_model(device=str(args.device), dtype=str(args.dtype))
    generated_count = 0
    skipped_count = 0
    grouped_requests: dict[tuple[str, str, str, str, str, int], list[Any]] = defaultdict(list)
    for request in requests:
        grouped_requests[_base_request_key(request)].append(request)

    total_groups = len(grouped_requests)
    generated_groups = 0
    for group_index, group in enumerate(grouped_requests.values(), start=1):
        pending_requests: list[Any] = []
        for request in group:
            output_path = output_root / request.output_rel_path
            existing_row = rows_by_split[request.split].get(request.utterance_id)
            if output_path.is_file() and existing_row is not None:
                skipped_count += 1
                continue
            pending_requests.append(request)
        if not pending_requests:
            continue
        samples, sample_rate = _synthesize_one(model, request=pending_requests[0])
        for request in pending_requests:
            output_path = output_root / request.output_rel_path
            augmentation_profile = next(
                profile for profile in augmentation_profiles if profile.key == request.augmentation_key
            )
            degraded = apply_augmentation(
                samples,
                sample_rate=sample_rate,
                profile=augmentation_profile,
                seed=request.seed,
            )
            write_pcm16_wav(output_path, samples=degraded, sample_rate=sample_rate)
            rows_by_split[request.split][request.utterance_id] = manifest_row_for_request(
                request,
                audio_path=output_path,
            )
            generated_count += 1
        generated_groups += 1
        if group_index == 1 or group_index % 25 == 0 or group_index == total_groups:
            print(
                json.dumps(
                    {
                        "event": "qwen3tts_generation_progress",
                        "group_index": group_index,
                        "total_groups": total_groups,
                        "generated_count": generated_count,
                        "skipped_count": skipped_count,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                flush=True,
            )

    _rewrite_manifests(output_root, rows_by_split)
    summary = {
        "schema": "twinr_qwen3tts_synthetic_corpus_v1",
        "output_root": str(output_root),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "speaker_shard_index": int(args.speaker_shard_index),
        "speaker_shard_count": int(args.speaker_shard_count),
        "speakers": list(speakers),
        "seeds": list(seeds),
        "style_profiles": [profile.key for profile in style_profiles],
        "generation_profiles": [profile.key for profile in generation_profiles],
        "augmentation_profiles": [profile.key for profile in augmentation_profiles],
        "prefixes": list(prefixes),
        "follow_ups": list(follow_ups),
        "planned_count": len(requests),
        "planned_group_count": total_groups,
        "generated_group_count": generated_groups,
        "generated_count": generated_count,
        "skipped_count": skipped_count,
        **_summary_from_rows(rows_by_split),
    }
    (output_root / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
