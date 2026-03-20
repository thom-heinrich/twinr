"""Generate a multivoice wakeword dataset for Twinr openWakeWord training.

Purpose
-------
Create a reproducible synthetic dataset root with positive and negative wakeword
splits that can later be consumed by Twinr's offline base-model training flow.
The script supports both the broad Twinr wakeword family and a stricter
``Twinr``-only profile for local detector training when false positives on
nearby variants must be reduced.

Usage
-----
Command-line invocation examples::

    python scripts/generate_multivoice_dataset.py --model-name twinr_family_v3
    python scripts/generate_multivoice_dataset.py --model-name twinr_family_v3 \
      --phrase-profile strict_twinr \
      --extra-positive-dir /tmp/twinr_oww_capture_room_20260319a/positive \
      --extra-negative-dir /tmp/twinr_oww_capture_room_20260319a/negative
    python scripts/generate_multivoice_dataset.py --model-name twinr_strict_fpmined_v1 \
      --phrase-profile strict_twinr \
      --extra-positive-dir /tmp/twinr_oww_capture_room_20260319a/positive \
      --exclude-manifest /tmp/twinr_critical16_v2.json \
      --hard-negative-manifest /tmp/twinr_oww_capture_room_20260319a/captured_manifest.json \
      --hard-negative-model /tmp/twinr/src/twinr/proactive/wakeword/models/twinr_v1.onnx

Outputs
-------
- ``<base-dir>/runs/<model-name>/positive_train/*.wav``
- ``<base-dir>/runs/<model-name>/negative_train/*.wav``
- ``<base-dir>/runs/<model-name>/positive_test/*.wav``
- ``<base-dir>/runs/<model-name>/negative_test/*.wav``

Notes
-----
The phrase lists here are intentionally aligned with Twinr's current wakeword
family and known hard negatives. Choose the stricter profile when the base
detector should learn only the literal ``Twinr`` wakeword instead of alias
spellings like ``Twinna`` or ``Twinner``. Use hard-negative mining when a
current detector already exposes recurring false activations on real room
captures and those clips should be folded back into the next training run
without leaking a held-out acceptance subset.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shutil
import urllib.request
import uuid
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]


VOICE_SPECS = {
    "en_US-lessac-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true",
    },
    "en_US-ljspeech-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx?download=true",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json?download=true",
    },
    "de_DE-thorsten-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx?download=true",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json?download=true",
    },
    "de_DE-thorsten_emotional-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten_emotional/medium/de_DE-thorsten_emotional-medium.onnx?download=true",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten_emotional/medium/de_DE-thorsten_emotional-medium.onnx.json?download=true",
    },
}

FAMILY_POSITIVE_PHRASES = [
    "hey twinr",
    "hallo twinr",
    "twinr",
    "twinr bitte hilf mir",
    "twinr wie ist das wetter heute",
    "hey twinna",
    "hallo twinna",
    "twinna",
    "twinna bitte hilf mir",
    "twinna wie ist das wetter heute",
    "hey twina",
    "hallo twina",
    "twina",
    "twina bitte hilf mir",
    "twina wie ist das wetter heute",
    "hey twinner",
    "hallo twinner",
    "twinner",
    "twinner bitte hilf mir",
    "twinner wie spaet ist es",
]

STRICT_TWINR_POSITIVE_PHRASES = [
    "hey twinr",
    "hallo twinr",
    "twinr",
    "twinr bitte hilf mir",
    "twinr wie ist das wetter heute",
]

BASE_NEGATIVE_PHRASES = [
    "hallo twin",
    "hey twin",
    "twin",
    "hallo tina",
    "hey tina",
    "tina",
    "hallo winter",
    "hey winter",
    "winter",
    "hallo winner",
    "hey winner",
    "winner",
    "hallo twitter",
    "hey twitter",
    "twitter",
    "hallo timer",
    "hey timer",
    "timer",
    "hallo trainer",
    "hey trainer",
    "trainer",
    "hallo tinder",
    "hey tinder",
    "tinder",
    "guten morgen",
    "wie geht es dir",
    "alles gut",
    "danke schoen",
    "danke",
    "brauchst du hilfe",
    "kann ich dir helfen",
    "ist alles in ordnung",
    "wie spaet ist es",
    "wie ist das wetter heute",
    "bitte erinnere mich an meine tabletten",
    "wann kommt meine tochter",
    "spiel musik",
    "ja bitte",
    "nein danke",
]

STRICT_TWINR_ALIAS_NEGATIVE_PHRASES = [
    "hallo twinna",
    "hey twinna",
    "twinna",
    "hallo twina",
    "hey twina",
    "twina",
    "hallo twinner",
    "hey twinner",
    "twinner",
]

PHRASE_PROFILES = {
    "family": {
        "positive": tuple(FAMILY_POSITIVE_PHRASES),
        "negative": tuple(BASE_NEGATIVE_PHRASES),
    },
    "strict_twinr": {
        "positive": tuple(STRICT_TWINR_POSITIVE_PHRASES),
        "negative": tuple(BASE_NEGATIVE_PHRASES + STRICT_TWINR_ALIAS_NEGATIVE_PHRASES),
    },
}

_NEGATIVE_LABELS = {
    "false_positive",
    "background_tv",
    "cross_talk",
    "far_field",
    "noise",
    "negative",
}

def _download(url: str, target: Path) -> None:
    """Download one Piper asset only when the local target is missing."""

    if target.exists() and target.stat().st_size > 0:
        print(f"exists={target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"download={url} -> {target}")
    urllib.request.urlretrieve(url, target)


def _default_base_dir() -> Path:
    """Resolve the default dataset workspace root for generated samples."""

    configured = os.environ.get("TWINR_MULTIVOICE_BASE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return REPO_ROOT / "models" / "openwakeword_dataset"


def _load_generators(generator_root: Path):
    """Import the Piper sample-generation helpers from one checkout root."""

    import sys

    sys.path.insert(0, str(generator_root))
    try:
        generate_pt = importlib.import_module("generate_samples").generate_samples
    except ModuleNotFoundError:
        generator_module = importlib.import_module("piper_sample_generator.__main__")
        generate_pt = generator_module.generate_samples
        generate_samples_onnx = generator_module.generate_samples_onnx
        return generate_pt, generate_samples_onnx
    generate_samples_onnx = importlib.import_module("piper_sample_generator.__main__").generate_samples_onnx
    return generate_pt, generate_samples_onnx


def ensure_voices(models_dir: Path) -> list[Path]:
    """Ensure the configured Piper voice assets exist locally."""

    paths: list[Path] = []
    for name, spec in VOICE_SPECS.items():
        onnx_path = models_dir / f"{name}.onnx"
        json_path = models_dir / f"{name}.onnx.json"
        _download(spec["onnx"], onnx_path)
        _download(spec["json"], json_path)
        paths.append(onnx_path)
    return paths


def _fresh_dir(path: Path) -> None:
    """Recreate one output directory from scratch."""

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _uuid_names(count: int) -> list[str]:
    """Create deterministic-free output filenames for generated WAVs."""

    return [f"{uuid.uuid4().hex}.wav" for _ in range(count)]


def _resolve_phrase_profile(profile_name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the positive/negative phrase sets for one configured profile."""

    normalized = profile_name.strip().lower()
    try:
        profile = PHRASE_PROFILES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported phrase profile: {profile_name}") from exc
    return tuple(profile["positive"]), tuple(profile["negative"])


def generate_split(
    output_dir: Path,
    *,
    phrases: tuple[str, ...],
    generator_count: int,
    onnx_count: int,
    onnx_models: list[Path],
    generator_model: Path,
    generate_pt,
    generate_samples_onnx,
) -> None:
    """Generate one positive or negative dataset split under ``output_dir``."""

    phrase_list = list(phrases)
    _fresh_dir(output_dir)
    if generator_count > 0:
        print(f"generate_pt={output_dir} count={generator_count}")
        generate_pt(
            text=phrase_list,
            output_dir=output_dir,
            model=generator_model,
            max_samples=generator_count,
            file_names=_uuid_names(generator_count),
            batch_size=60,
            slerp_weights=(0.15, 0.35, 0.5, 0.65, 0.85),
            length_scales=(0.7, 0.85, 1.0, 1.15, 1.3),
            noise_scales=(0.55, 0.7, 0.9, 1.05),
            noise_scale_ws=(0.7, 0.85, 1.0),
            max_speakers=300,
        )
    if onnx_count > 0:
        print(f"generate_onnx={output_dir} count={onnx_count}")
        generate_samples_onnx(
            text=phrase_list,
            output_dir=output_dir,
            model=[str(path) for path in onnx_models],
            max_samples=onnx_count,
            file_names=_uuid_names(onnx_count),
            length_scales=(0.72, 0.88, 1.0, 1.12, 1.25),
            noise_scales=(0.6, 0.8, 1.0),
            noise_scale_ws=(0.7, 0.85, 1.0),
            max_speakers=8,
        )
    print(f"generated={output_dir} wavs={len(list(output_dir.glob('*.wav')))}")


def _copy_extra_samples(
    *,
    source_dirs: list[Path],
    train_dir: Path,
    test_dir: Path,
    prefix: str,
    exclude_paths: set[Path] | None = None,
) -> None:
    """Merge extra real-environment samples into the generated splits."""

    extra_files: list[Path] = []
    excluded = exclude_paths or set()
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        extra_files.extend(
            sorted(
                path
                for path in source_dir.rglob("*.wav")
                if path.is_file() and path.expanduser().resolve(strict=False) not in excluded
            )
        )
    if not extra_files:
        return
    split_index = max(1, math.floor(len(extra_files) * 0.8))
    train_files = extra_files[:split_index]
    test_files = extra_files[split_index:] or extra_files[-1:]
    for source in train_files:
        shutil.copy2(source, train_dir / f"{prefix}_{source.stem}.wav")
    for source in test_files:
        shutil.copy2(source, test_dir / f"{prefix}_{source.stem}.wav")
    print(
        f"{prefix}_samples="
        f"{len(extra_files)} train={len(train_files)} test={len(test_files)}"
    )


def _load_manifest_payloads(manifest_path: Path) -> list[dict[str, object]]:
    """Load one JSONL or JSON-array wakeword manifest into plain dictionaries."""

    raw_text = manifest_path.read_text(encoding="utf-8")
    stripped = raw_text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        payload = json.loads(stripped)
        if not isinstance(payload, list):
            raise ValueError(f"{manifest_path} must contain a JSON array or JSONL objects.")
        items = payload
    else:
        items = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    entries: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"{manifest_path} contained a non-object manifest item.")
        entries.append(item)
    return entries


def _resolve_manifest_audio_path(payload: dict[str, object], *, manifest_path: Path) -> Path:
    """Resolve the best available audio path for one manifest item."""

    for key in ("captured_audio_path", "audio_path"):
        raw_value = str(payload.get(key) or "").strip()
        if not raw_value:
            continue
        audio_path = Path(raw_value).expanduser()
        if not audio_path.is_absolute():
            audio_path = (manifest_path.parent / audio_path).resolve(strict=False)
        return audio_path.resolve(strict=False)
    raise ValueError(f"{manifest_path} contains an item without audio_path or captured_audio_path.")


def _load_excluded_audio_paths(manifest_paths: list[Path]) -> set[Path]:
    """Resolve a stable exclusion set from one or more manifest files."""

    excluded: set[Path] = set()
    for manifest_path in manifest_paths:
        for payload in _load_manifest_payloads(manifest_path):
            excluded.add(_resolve_manifest_audio_path(payload, manifest_path=manifest_path))
    return excluded


def _default_openwakeword_model_factory(*, wakeword_models: list[str]):
    """Instantiate one openWakeWord model for offline hard-negative mining."""

    from openwakeword.model import Model

    return Model(wakeword_models=wakeword_models, inference_framework="onnx")


def _peak_model_score(model, *, model_key: str, audio_path: Path) -> float:
    """Return the highest per-frame detector score for one WAV clip."""

    model.reset()
    predictions = model.predict_clip(str(audio_path), padding=1, chunk_size=1280)
    if not predictions:
        return 0.0
    return max(float(frame.get(model_key, 0.0)) for frame in predictions)


def _mine_hard_negative_records(
    *,
    manifest_path: Path,
    model_path: Path,
    threshold: float,
    exclude_paths: set[Path],
    max_per_text: int,
    model_factory: Callable[..., object] | None = None,
) -> list[dict[str, object]]:
    """Replay one manifest and return the negative clips that still activate.

    The output is sorted by text family, then by descending peak score, so a
    later train/test split stays deterministic across hosts.
    """

    resolved_manifest = manifest_path.expanduser().resolve(strict=True)
    resolved_model = model_path.expanduser().resolve(strict=True)
    detector_key = resolved_model.stem
    factory = model_factory or _default_openwakeword_model_factory
    model = factory(wakeword_models=[str(resolved_model)])

    rows: list[dict[str, object]] = []
    for payload in _load_manifest_payloads(resolved_manifest):
        label = str(payload.get("label") or "").strip().lower().replace(" ", "_")
        if label not in _NEGATIVE_LABELS:
            continue
        audio_path = _resolve_manifest_audio_path(payload, manifest_path=resolved_manifest)
        if audio_path in exclude_paths:
            continue
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        peak_score = _peak_model_score(model, model_key=detector_key, audio_path=audio_path)
        if peak_score < threshold:
            continue
        rows.append(
            {
                "audio_path": str(audio_path),
                "label": label,
                "text": str(payload.get("text") or "").strip(),
                "peak_score": round(peak_score, 6),
            }
        )

    if max_per_text > 0:
        grouped_rows: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            grouped_rows.setdefault(str(row["text"]), []).append(row)
        rows = []
        for text in sorted(grouped_rows):
            ranked_rows = sorted(
                grouped_rows[text],
                key=lambda item: (-float(item["peak_score"]), str(item["audio_path"])),
            )
            rows.extend(ranked_rows[:max_per_text])

    return sorted(
        rows,
        key=lambda item: (
            str(item["text"]),
            -float(item["peak_score"]),
            str(item["audio_path"]),
        ),
    )


def _copy_mined_hard_negatives(
    *,
    rows: list[dict[str, object]],
    train_dir: Path,
    test_dir: Path,
    prefix: str,
) -> list[dict[str, object]]:
    """Copy mined hard negatives into the generated dataset splits."""

    if not rows:
        return []
    grouped_rows: dict[str, list[dict[str, object]]] = {}
    copied_rows: list[dict[str, object]] = []
    for row in rows:
        grouped_rows.setdefault(str(row["text"]), []).append(row)
    for text in sorted(grouped_rows):
        ranked_rows = grouped_rows[text]
        if len(ranked_rows) == 1:
            train_rows = ranked_rows
            test_rows: list[dict[str, object]] = []
        else:
            split_index = max(1, math.floor(len(ranked_rows) * 0.8))
            train_rows = ranked_rows[:split_index]
            test_rows = ranked_rows[split_index:]
        for split_name, target_dir, split_rows in (
            ("train", train_dir, train_rows),
            ("test", test_dir, test_rows),
        ):
            for row in split_rows:
                source = Path(str(row["audio_path"]))
                target_name = f"{prefix}_{source.stem}.wav"
                shutil.copy2(source, target_dir / target_name)
                copied_rows.append(
                    {
                        **row,
                        "split": split_name,
                        "copied_path": str((target_dir / target_name).resolve(strict=False)),
                    }
                )
    print(
        "mined_neg_samples="
        f"{len(copied_rows)} train={sum(1 for row in copied_rows if row['split'] == 'train')} "
        f"test={sum(1 for row in copied_rows if row['split'] == 'test')}"
    )
    return copied_rows


def main() -> int:
    """Run the Twinr multivoice wakeword dataset generator."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--base-dir",
        default=str(_default_base_dir()),
        help="Dataset workspace root. Defaults to $TWINR_MULTIVOICE_BASE_DIR or <repo>/models/openwakeword_dataset.",
    )
    parser.add_argument(
        "--generator-root",
        default="",
        help="Path to the piper-sample-generator checkout. Defaults to <base-dir>/piper-sample-generator or $TWINR_PIPER_SAMPLE_GENERATOR_ROOT.",
    )
    parser.add_argument(
        "--models-dir",
        default="",
        help="Directory for Piper voice models. Defaults to <generator-root>/voices or $TWINR_PIPER_VOICES_DIR.",
    )
    parser.add_argument(
        "--generator-model",
        default="",
        help="Path to the generator .pt model. Defaults to <generator-root>/models/en_US-libritts_r-medium.pt or $TWINR_PIPER_GENERATOR_MODEL.",
    )
    parser.add_argument(
        "--phrase-profile",
        choices=tuple(sorted(PHRASE_PROFILES)),
        default="family",
        help="Phrase profile for synthetic positives/negatives. Use strict_twinr to train a narrower local detector on the literal Twinr wakeword.",
    )
    parser.add_argument("--positive-train-generator", type=int, default=900)
    parser.add_argument("--positive-train-onnx", type=int, default=1800)
    parser.add_argument("--positive-test-generator", type=int, default=300)
    parser.add_argument("--positive-test-onnx", type=int, default=600)
    parser.add_argument("--negative-train-generator", type=int, default=900)
    parser.add_argument("--negative-train-onnx", type=int, default=1800)
    parser.add_argument("--negative-test-generator", type=int, default=300)
    parser.add_argument("--negative-test-onnx", type=int, default=600)
    parser.add_argument(
        "--extra-negative-dir",
        action="append",
        default=[],
        help="Directory containing real-environment negative .wav clips to merge into the generated negative splits.",
    )
    parser.add_argument(
        "--extra-positive-dir",
        action="append",
        default=[],
        help="Directory containing real-environment positive .wav clips to merge into the generated positive splits.",
    )
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="Optional manifest(s) whose audio clips must stay out of copied room captures and mined hard negatives, for example a held-out acceptance subset.",
    )
    parser.add_argument(
        "--hard-negative-manifest",
        default="",
        help="Optional labeled wakeword manifest used to mine false-positive negatives from an existing local model.",
    )
    parser.add_argument(
        "--hard-negative-model",
        default="",
        help="Optional local openWakeWord .onnx model path used for hard-negative mining.",
    )
    parser.add_argument(
        "--hard-negative-threshold",
        type=float,
        default=0.5,
        help="Minimum detector score required for one negative clip to be copied into the mined hard-negative splits.",
    )
    parser.add_argument(
        "--hard-negative-max-per-text",
        type=int,
        default=0,
        help="Optional cap per transcript text family for mined hard negatives. Zero disables capping.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    generator_root = Path(
        args.generator_root or os.environ.get("TWINR_PIPER_SAMPLE_GENERATOR_ROOT", "") or (base_dir / "piper-sample-generator")
    ).expanduser().resolve()
    models_dir = Path(
        args.models_dir or os.environ.get("TWINR_PIPER_VOICES_DIR", "") or (generator_root / "voices")
    ).expanduser().resolve()
    generator_model = Path(
        args.generator_model
        or os.environ.get("TWINR_PIPER_GENERATOR_MODEL", "")
        or (generator_root / "models" / "en_US-libritts_r-medium.pt")
    ).expanduser().resolve()

    if not generator_root.exists():
        raise SystemExit(f"generator root not found: {generator_root}")
    if not generator_model.exists():
        raise SystemExit(f"generator model not found: {generator_model}")

    generate_pt, generate_samples_onnx = _load_generators(generator_root)
    onnx_models = ensure_voices(models_dir)
    positive_phrases, negative_phrases = _resolve_phrase_profile(args.phrase_profile)
    target_root = base_dir / "runs" / args.model_name
    target_root.mkdir(parents=True, exist_ok=True)
    excluded_audio_paths = _load_excluded_audio_paths(
        [Path(path).expanduser().resolve(strict=True) for path in args.exclude_manifest]
    )

    hard_negative_manifest = Path(str(args.hard_negative_manifest or "")).expanduser()
    hard_negative_model = Path(str(args.hard_negative_model or "")).expanduser()
    if bool(str(args.hard_negative_manifest or "").strip()) != bool(str(args.hard_negative_model or "").strip()):
        raise SystemExit("--hard-negative-manifest and --hard-negative-model must be provided together.")

    generate_split(
        target_root / "positive_train",
        phrases=positive_phrases,
        generator_count=args.positive_train_generator,
        onnx_count=args.positive_train_onnx,
        onnx_models=onnx_models,
        generator_model=generator_model,
        generate_pt=generate_pt,
        generate_samples_onnx=generate_samples_onnx,
    )
    generate_split(
        target_root / "positive_test",
        phrases=positive_phrases,
        generator_count=args.positive_test_generator,
        onnx_count=args.positive_test_onnx,
        onnx_models=onnx_models,
        generator_model=generator_model,
        generate_pt=generate_pt,
        generate_samples_onnx=generate_samples_onnx,
    )
    generate_split(
        target_root / "negative_train",
        phrases=negative_phrases,
        generator_count=args.negative_train_generator,
        onnx_count=args.negative_train_onnx,
        onnx_models=onnx_models,
        generator_model=generator_model,
        generate_pt=generate_pt,
        generate_samples_onnx=generate_samples_onnx,
    )
    generate_split(
        target_root / "negative_test",
        phrases=negative_phrases,
        generator_count=args.negative_test_generator,
        onnx_count=args.negative_test_onnx,
        onnx_models=onnx_models,
        generator_model=generator_model,
        generate_pt=generate_pt,
        generate_samples_onnx=generate_samples_onnx,
    )
    _copy_extra_samples(
        source_dirs=[Path(path) for path in args.extra_positive_dir],
        train_dir=target_root / "positive_train",
        test_dir=target_root / "positive_test",
        prefix="extra_pos",
        exclude_paths=excluded_audio_paths,
    )
    _copy_extra_samples(
        source_dirs=[Path(path) for path in args.extra_negative_dir],
        train_dir=target_root / "negative_train",
        test_dir=target_root / "negative_test",
        prefix="extra_neg",
        exclude_paths=excluded_audio_paths,
    )
    if str(args.hard_negative_manifest or "").strip():
        mined_rows = _mine_hard_negative_records(
            manifest_path=hard_negative_manifest.resolve(strict=True),
            model_path=hard_negative_model.resolve(strict=True),
            threshold=float(args.hard_negative_threshold),
            exclude_paths=excluded_audio_paths,
            max_per_text=max(0, int(args.hard_negative_max_per_text)),
        )
        copied_rows = _copy_mined_hard_negatives(
            rows=mined_rows,
            train_dir=target_root / "negative_train",
            test_dir=target_root / "negative_test",
            prefix="mined_neg",
        )
        (target_root / "hard_negative_mining.json").write_text(
            json.dumps(
                {
                    "manifest_path": str(hard_negative_manifest.resolve(strict=True)),
                    "model_path": str(hard_negative_model.resolve(strict=True)),
                    "threshold": float(args.hard_negative_threshold),
                    "max_per_text": max(0, int(args.hard_negative_max_per_text)),
                    "excluded_audio_paths": sorted(str(path) for path in excluded_audio_paths),
                    "copied_rows": copied_rows,
                },
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
