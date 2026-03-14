from __future__ import annotations

import argparse
import importlib
import math
import os
import shutil
import urllib.request
import uuid
from pathlib import Path

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

POSITIVE_PHRASES = [
    "hey twinna",
    "hey twina",
    "hey twinner",
    "hey tweena",
    "hallo twinna",
    "hallo twina",
    "hallo twinner",
    "twinna",
    "twina",
    "twinner",
    "tweena",
    "twinna bist du da",
    "twina bist du da",
    "twinner bist du da",
    "tweena bist du da",
]

NEGATIVE_PHRASES = [
    "hallo",
    "hey",
    "ja hallo",
    "guten morgen",
    "wie geht es dir",
    "alles gut",
    "danke schoen",
    "danke",
    "brauchst du hilfe",
    "kann ich dir helfen",
    "ist alles in ordnung",
    "wie spaet ist es",
    "hallo tina",
    "hey tina",
    "winner",
    "dinner",
    "twitter",
    "tinder",
    "trainer",
    "thin air",
    "thomas",
    "hallo thom",
    "ja bitte",
    "nein danke",
]

def _download(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"exists={target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"download={url} -> {target}")
    urllib.request.urlretrieve(url, target)


def _default_base_dir() -> Path:
    configured = os.environ.get("TWINR_MULTIVOICE_BASE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return REPO_ROOT / "models" / "openwakeword_dataset"


def _load_generators(generator_root: Path):
    import sys

    sys.path.insert(0, str(generator_root))
    generate_pt = importlib.import_module("generate_samples").generate_samples
    generate_samples_onnx = importlib.import_module(
        "piper_sample_generator.__main__"
    ).generate_samples_onnx
    return generate_pt, generate_samples_onnx


def ensure_voices(models_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for name, spec in VOICE_SPECS.items():
        onnx_path = models_dir / f"{name}.onnx"
        json_path = models_dir / f"{name}.onnx.json"
        _download(spec["onnx"], onnx_path)
        _download(spec["json"], json_path)
        paths.append(onnx_path)
    return paths


def _fresh_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _uuid_names(count: int) -> list[str]:
    return [f"{uuid.uuid4().hex}.wav" for _ in range(count)]


def generate_split(
    output_dir: Path,
    *,
    positive: bool,
    generator_count: int,
    onnx_count: int,
    onnx_models: list[Path],
    generator_model: Path,
    generate_pt,
    generate_samples_onnx,
) -> None:
    phrases = POSITIVE_PHRASES if positive else NEGATIVE_PHRASES
    _fresh_dir(output_dir)
    if generator_count > 0:
        print(f"generate_pt={output_dir} count={generator_count}")
        generate_pt(
            text=phrases,
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
            text=phrases,
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


def _copy_extra_negative_samples(
    *,
    source_dirs: list[Path],
    train_dir: Path,
    test_dir: Path,
) -> None:
    extra_files: list[Path] = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        extra_files.extend(sorted(path for path in source_dir.rglob("*.wav") if path.is_file()))
    if not extra_files:
        return
    split_index = max(1, math.floor(len(extra_files) * 0.8))
    train_files = extra_files[:split_index]
    test_files = extra_files[split_index:] or extra_files[-1:]
    for source in train_files:
        shutil.copy2(source, train_dir / f"extra_neg_{source.stem}.wav")
    for source in test_files:
        shutil.copy2(source, test_dir / f"extra_neg_{source.stem}.wav")
    print(
        "extra_negative_samples="
        f"{len(extra_files)} train={len(train_files)} test={len(test_files)}"
    )


def main() -> int:
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
    target_root = base_dir / "runs" / args.model_name
    target_root.mkdir(parents=True, exist_ok=True)

    generate_split(
        target_root / "positive_train",
        positive=True,
        generator_count=args.positive_train_generator,
        onnx_count=args.positive_train_onnx,
        onnx_models=onnx_models,
        generator_model=generator_model,
        generate_pt=generate_pt,
        generate_samples_onnx=generate_samples_onnx,
    )
    generate_split(
        target_root / "positive_test",
        positive=True,
        generator_count=args.positive_test_generator,
        onnx_count=args.positive_test_onnx,
        onnx_models=onnx_models,
        generator_model=generator_model,
        generate_pt=generate_pt,
        generate_samples_onnx=generate_samples_onnx,
    )
    generate_split(
        target_root / "negative_train",
        positive=False,
        generator_count=args.negative_train_generator,
        onnx_count=args.negative_train_onnx,
        onnx_models=onnx_models,
        generator_model=generator_model,
        generate_pt=generate_pt,
        generate_samples_onnx=generate_samples_onnx,
    )
    generate_split(
        target_root / "negative_test",
        positive=False,
        generator_count=args.negative_test_generator,
        onnx_count=args.negative_test_onnx,
        onnx_models=onnx_models,
        generator_model=generator_model,
        generate_pt=generate_pt,
        generate_samples_onnx=generate_samples_onnx,
    )
    _copy_extra_negative_samples(
        source_dirs=[Path(path) for path in args.extra_negative_dir],
        train_dir=target_root / "negative_train",
        test_dir=target_root / "negative_test",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
