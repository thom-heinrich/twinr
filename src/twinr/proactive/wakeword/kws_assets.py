"""Provision reproducible sherpa-onnx KWS bundles for Twinr.

This module downloads or reuses one official sherpa-onnx keyword-spotting
archive, selects a known-good runtime asset set, derives Twinr keyword labels
from the configured wakeword phrases, and emits a ready-to-use bundle
directory for the local ``kws`` backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Callable
from urllib.request import Request, urlopen

from twinr.proactive.wakeword.matching import DEFAULT_WAKEWORD_PHRASES
from twinr.text_utils import folded_lookup_text

_GENERIC_WAKEWORD_WORDS = frozenset({"hey", "hallo", "he", "hi", "ok", "okay"})
_DEFAULT_BUNDLE_ID = "gigaspeech_3_3m_bpe_int8"


@dataclass(frozen=True, slots=True)
class WakewordKwsProvisionSpec:
    """Describe one supported upstream sherpa-onnx bundle recipe."""

    bundle_id: str
    source_url: str
    archive_root: str
    tokens_type: str
    tokens_filename: str
    tokenizer_filename: str
    encoder_filename: str
    decoder_filename: str
    joiner_filename: str
    readme_filename: str = "README.md"


@dataclass(frozen=True, slots=True)
class ProvisionedWakewordKwsBundle:
    """Describe one prepared local KWS asset directory."""

    bundle_id: str
    output_dir: Path
    keyword_names: tuple[str, ...]
    tokens_path: Path
    encoder_path: Path
    decoder_path: Path
    joiner_path: Path
    keywords_raw_path: Path
    keywords_path: Path
    tokenizer_path: Path
    metadata_path: Path


def available_builtin_kws_bundle_specs() -> dict[str, WakewordKwsProvisionSpec]:
    """Return the built-in official sherpa-onnx bundles Twinr can provision."""

    return {
        _DEFAULT_BUNDLE_ID: WakewordKwsProvisionSpec(
            bundle_id=_DEFAULT_BUNDLE_ID,
            source_url=(
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                "kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2"
            ),
            archive_root="sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
            tokens_type="bpe",
            tokens_filename="tokens.txt",
            tokenizer_filename="bpe.model",
            encoder_filename="encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
            decoder_filename="decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
            joiner_filename="joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        )
    }


def derive_kws_keyword_names(
    *,
    phrases: tuple[str, ...] | list[str] | None = None,
    explicit_keywords: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    """Return deduplicated keyword labels suitable for KWS bundle generation."""

    if explicit_keywords:
        source_values = explicit_keywords
    else:
        source_values = phrases or DEFAULT_WAKEWORD_PHRASES
    keyword_names: list[str] = []
    seen: set[str] = set()
    for value in source_values:
        normalized = folded_lookup_text(str(value or ""))
        if not normalized:
            continue
        filtered_words = [word for word in normalized.split() if word not in _GENERIC_WAKEWORD_WORDS]
        candidate = " ".join(filtered_words)
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        keyword_names.append(candidate)
    return tuple(keyword_names)


def provision_builtin_kws_bundle(
    *,
    output_dir: str | Path,
    bundle_id: str = _DEFAULT_BUNDLE_ID,
    phrases: tuple[str, ...] | list[str] | None = None,
    explicit_keywords: tuple[str, ...] | list[str] | None = None,
    force: bool = False,
    archive_path: str | Path | None = None,
    text2token_fn: Callable[..., list[list[str | int]]] | None = None,
) -> ProvisionedWakewordKwsBundle:
    """Download or reuse one official sherpa-onnx bundle and emit Twinr assets."""

    specs = available_builtin_kws_bundle_specs()
    try:
        spec = specs[str(bundle_id).strip()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported KWS bundle id {bundle_id!r}. Available: {', '.join(sorted(specs))}"
        ) from exc
    keyword_names = derive_kws_keyword_names(phrases=phrases, explicit_keywords=explicit_keywords)
    if not keyword_names:
        raise ValueError("At least one KWS keyword is required to provision a bundle.")
    output_root = Path(output_dir).expanduser().resolve(strict=False)
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(f"KWS output path is not a directory: {output_root}")
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"KWS output directory already exists and is not empty: {output_root}")

    with tempfile.TemporaryDirectory(prefix="twinr-kws-provision-") as temp_dir:
        temp_root = Path(temp_dir)
        archive = _resolve_bundle_archive(spec=spec, temp_root=temp_root, archive_path=archive_path)
        staged = temp_root / "bundle"
        staged.mkdir(parents=True, exist_ok=True)
        _stage_bundle_assets(spec=spec, archive_path=archive, staged_root=staged)
        keywords_raw_path = staged / "keywords_raw.txt"
        keywords_path = staged / "keywords.txt"
        raw_lines = tuple(_keyword_raw_line(name) for name in keyword_names)
        keywords_raw_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
        if text2token_fn is not None:
            token_lines = text2token_fn(
                list(raw_lines),
                tokens=str(staged / "tokens.txt"),
                tokens_type=spec.tokens_type,
                bpe_model=str(staged / spec.tokenizer_filename),
            )
            keywords_path.write_text(
                "\n".join(" ".join(str(item) for item in line) for line in token_lines) + "\n",
                encoding="utf-8",
            )
        else:
            _run_text2token_cli(
                input_path=keywords_raw_path,
                output_path=keywords_path,
                tokens_path=staged / "tokens.txt",
                tokens_type=spec.tokens_type,
                tokenizer_path=staged / spec.tokenizer_filename,
            )
        metadata_path = staged / "bundle_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "bundle_id": spec.bundle_id,
                    "source_url": spec.source_url,
                    "archive_root": spec.archive_root,
                    "tokens_type": spec.tokens_type,
                    "keyword_names": list(keyword_names),
                    "spec": asdict(spec),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        output_root.mkdir(parents=True, exist_ok=True)
        for filename in (
            "tokens.txt",
            "encoder.onnx",
            "decoder.onnx",
            "joiner.onnx",
            spec.tokenizer_filename,
            "keywords_raw.txt",
            "keywords.txt",
            "bundle_metadata.json",
            "upstream.README.md",
        ):
            shutil.copy2(staged / filename, output_root / filename)

    return ProvisionedWakewordKwsBundle(
        bundle_id=spec.bundle_id,
        output_dir=output_root,
        keyword_names=keyword_names,
        tokens_path=output_root / "tokens.txt",
        encoder_path=output_root / "encoder.onnx",
        decoder_path=output_root / "decoder.onnx",
        joiner_path=output_root / "joiner.onnx",
        keywords_raw_path=output_root / "keywords_raw.txt",
        keywords_path=output_root / "keywords.txt",
        tokenizer_path=output_root / spec.tokenizer_filename,
        metadata_path=output_root / "bundle_metadata.json",
    )


def _resolve_bundle_archive(
    *,
    spec: WakewordKwsProvisionSpec,
    temp_root: Path,
    archive_path: str | Path | None,
) -> Path:
    if archive_path is not None:
        archive = Path(archive_path).expanduser().resolve(strict=False)
        if not archive.is_file():
            raise FileNotFoundError(f"KWS archive does not exist: {archive}")
        return archive
    archive = temp_root / f"{spec.bundle_id}.tar.bz2"
    request = Request(spec.source_url, headers={"User-Agent": "TwinrWakewordKwsProvisioner/1.0"})
    with urlopen(request, timeout=60.0) as response, archive.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return archive


def _stage_bundle_assets(
    *,
    spec: WakewordKwsProvisionSpec,
    archive_path: Path,
    staged_root: Path,
) -> None:
    required_members = {
        f"{spec.archive_root}/{spec.tokens_filename}": staged_root / "tokens.txt",
        f"{spec.archive_root}/{spec.tokenizer_filename}": staged_root / spec.tokenizer_filename,
        f"{spec.archive_root}/{spec.encoder_filename}": staged_root / "encoder.onnx",
        f"{spec.archive_root}/{spec.decoder_filename}": staged_root / "decoder.onnx",
        f"{spec.archive_root}/{spec.joiner_filename}": staged_root / "joiner.onnx",
        f"{spec.archive_root}/{spec.readme_filename}": staged_root / "upstream.README.md",
    }
    with tarfile.open(archive_path, "r:*") as archive:
        members = {member.name: member for member in archive.getmembers()}
        missing = [name for name in required_members if name not in members]
        if missing:
            raise FileNotFoundError(
                "KWS archive is missing required members: " + ", ".join(sorted(missing))
            )
        for member_name, target_path in required_members.items():
            extracted = archive.extractfile(members[member_name])
            if extracted is None:
                raise FileNotFoundError(f"KWS archive member could not be read: {member_name}")
            with extracted, target_path.open("wb") as handle:
                shutil.copyfileobj(extracted, handle)


def _keyword_raw_line(keyword_name: str) -> str:
    normalized = folded_lookup_text(keyword_name)
    if not normalized:
        raise ValueError("KWS keywords must not be empty.")
    display_phrase = " ".join(word.upper() for word in normalized.split())
    detector_label = normalized.replace(" ", "_")
    return f"{display_phrase} @{detector_label}"


def _run_text2token_cli(
    *,
    input_path: Path,
    output_path: Path,
    tokens_path: Path,
    tokens_type: str,
    tokenizer_path: Path,
) -> None:
    cli_path = Path(sys.executable).absolute().with_name("sherpa-onnx-cli")
    command = [
        str(cli_path if cli_path.is_file() else "sherpa-onnx-cli"),
        "text2token",
        "--tokens",
        str(tokens_path),
        "--tokens-type",
        tokens_type,
        "--bpe-model",
        str(tokenizer_path),
        str(input_path),
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "sherpa-onnx-cli is not available. Install sherpa-onnx plus its CLI tools before provisioning."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            "sherpa-onnx text2token failed. Install sentencepiece and pypinyin before provisioning KWS assets."
            + (f" Details: {detail}" if detail else "")
        ) from exc
    if result.returncode != 0 or not output_path.is_file():
        raise RuntimeError("sherpa-onnx text2token did not produce a keywords.txt output file.")


__all__ = [
    "ProvisionedWakewordKwsBundle",
    "WakewordKwsProvisionSpec",
    "available_builtin_kws_bundle_specs",
    "derive_kws_keyword_names",
    "provision_builtin_kws_bundle",
]
