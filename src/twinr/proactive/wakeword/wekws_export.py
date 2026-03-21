"""Export Twinr wakeword manifests into WeKws/Kaldi-style training data.

Professional wakeword stacks do not ship one generic open-vocabulary bundle as
the final product path. They train a conventional keyword detector for the
target wakeword family and pair it with stricter verification and deployment
metrics. WeKws expects Kaldi-style split directories such as ``train/wav.scp``
and ``train/text``. This module bridges Twinr's labeled room-capture manifests
into that format so the repo can train a real custom keyword detector instead
of staying locked to generic demo bundles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import wave

_POSITIVE_LABELS = {"correct", "false_negative", "positive"}
_NEGATIVE_LABELS = {
    "false_positive",
    "background_tv",
    "cross_talk",
    "far_field",
    "noise",
    "negative",
}
_IGNORED_LABELS = {"unclear"}
_DEFAULT_POSITIVE_TOKEN = "<TWINR_FAMILY>"
_DEFAULT_FILLER_TOKEN = "<FILLER>"
_MANIFEST_AUDIO_PATH_KEYS = ("captured_audio_path", "audio_path")


@dataclass(frozen=True, slots=True)
class _ManifestEvalEntry:
    """Describe one labeled audio clip exported into WeKws splits."""

    audio_path: Path
    label: str
    source: str = "manifest"
    notes: str | None = None


def _normalize_label(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace(" ", "_")
    return normalized or None


def _token_for_entry(
    entry: _ManifestEvalEntry,
    *,
    positive_token: str,
    filler_token: str,
) -> str | None:
    """Return the WeKws training token for one labeled Twinr manifest entry."""

    normalized = _normalize_label(entry.label)
    if normalized in _POSITIVE_LABELS:
        return positive_token
    if normalized in _NEGATIVE_LABELS:
        return filler_token
    if normalized in _IGNORED_LABELS:
        return None
    return filler_token


def _wav_duration_seconds(path: Path) -> float:
    """Return the duration of one WAV clip in seconds."""

    with wave.open(str(path), "rb") as wav_file:
        frame_count = int(wav_file.getnframes())
        sample_rate = int(wav_file.getframerate())
    return frame_count / max(1, sample_rate)


def _sanitize_token(token: str) -> str:
    normalized = str(token or "").strip().upper().replace(" ", "_")
    if not normalized:
        raise ValueError("WeKws export tokens must be non-empty.")
    if not normalized.startswith("<"):
        normalized = f"<{normalized}"
    if not normalized.endswith(">"):
        normalized = f"{normalized}>"
    return normalized


def _build_utterance_id(split_name: str, entry: _ManifestEvalEntry, index: int) -> str:
    stem = entry.audio_path.stem.strip().replace(" ", "_")
    safe_stem = "".join(ch for ch in stem if ch.isalnum() or ch in {"_", "-"}) or "clip"
    return f"{split_name}_{index:05d}_{safe_stem}"


def _load_manifest_payloads(manifest_path: str | Path) -> tuple[Path, list[dict[str, object]]]:
    """Load one wakeword manifest as JSONL or a single JSON array."""

    path = Path(manifest_path).expanduser().resolve(strict=False)
    if not path.exists():
        raise FileNotFoundError(path)
    raw_text = path.read_text(encoding="utf-8")
    stripped = raw_text.strip()
    if not stripped:
        return path, []
    if stripped.startswith("["):
        payload = json.loads(stripped)
        if not isinstance(payload, list):
            raise ValueError("Wakeword eval manifest must be a JSON array or JSONL file.")
        raw_entries = payload
    else:
        raw_entries = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    entries: list[dict[str, object]] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            raise ValueError("Wakeword eval manifest items must be JSON objects.")
        entries.append(item)
    return path, entries


def _resolve_manifest_audio_path(payload: dict[str, object], *, manifest_path: Path) -> Path:
    """Resolve the canonical audio path for one export manifest entry."""

    for key in _MANIFEST_AUDIO_PATH_KEYS:
        raw_value = str(payload.get(key) or "").strip()
        if not raw_value:
            continue
        audio_path = Path(raw_value).expanduser()
        if not audio_path.is_absolute():
            audio_path = (manifest_path.parent / audio_path).resolve(strict=False)
        return audio_path
    raise ValueError("Wakeword eval manifest items require audio_path or captured_audio_path.")


def _load_eval_manifest(manifest_path: str | Path) -> list[_ManifestEvalEntry]:
    """Load one lightweight wakeword manifest for WeKws export."""

    resolved_path, payloads = _load_manifest_payloads(manifest_path)
    entries: list[_ManifestEvalEntry] = []
    for payload in payloads:
        audio_path = _resolve_manifest_audio_path(payload, manifest_path=resolved_path)
        label = str(payload.get("label") or "").strip()
        if not label:
            raise ValueError(f"{resolved_path} entries require a non-empty label.")
        entries.append(
            _ManifestEvalEntry(
                audio_path=audio_path,
                label=label,
                source=str(payload.get("source") or "manifest"),
                notes=None if payload.get("notes") is None else str(payload.get("notes")),
            )
        )
    return entries


@dataclass(frozen=True, slots=True)
class WekwsExportSplitReport:
    """Summarize one exported WeKws split."""

    split_name: str
    manifest_path: Path
    output_dir: Path
    entry_count: int
    positive_count: int
    negative_count: int
    ignored_count: int


@dataclass(frozen=True, slots=True)
class WekwsExportReport:
    """Describe one completed Twinr-to-WeKws export run."""

    output_dir: Path
    dict_path: Path
    words_path: Path
    positive_token: str
    filler_token: str
    split_reports: tuple[WekwsExportSplitReport, ...]
    metadata_path: Path


def export_wakeword_manifests_to_wekws(
    *,
    output_dir: str | Path,
    train_manifest: str | Path,
    dev_manifest: str | Path | None = None,
    test_manifest: str | Path | None = None,
    positive_token: str = _DEFAULT_POSITIVE_TOKEN,
    filler_token: str = _DEFAULT_FILLER_TOKEN,
) -> WekwsExportReport:
    """Export Twinr labeled manifests into WeKws/Kaldi-style split directories."""

    resolved_output_dir = Path(output_dir).expanduser().resolve(strict=False)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    normalized_positive_token = _sanitize_token(positive_token)
    normalized_filler_token = _sanitize_token(filler_token)
    split_specs = [("train", Path(train_manifest).expanduser().resolve(strict=False))]
    if dev_manifest is not None:
        split_specs.append(("dev", Path(dev_manifest).expanduser().resolve(strict=False)))
    if test_manifest is not None:
        split_specs.append(("test", Path(test_manifest).expanduser().resolve(strict=False)))

    split_reports: list[WekwsExportSplitReport] = []
    for split_name, manifest_path in split_specs:
        entries = _load_eval_manifest(manifest_path)
        split_dir = resolved_output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        wav_scp_lines: list[str] = []
        text_lines: list[str] = []
        utt2spk_lines: list[str] = []
        wav_dur_lines: list[str] = []
        positive_count = 0
        negative_count = 0
        ignored_count = 0
        written_count = 0
        for index, entry in enumerate(entries):
            token = _token_for_entry(
                entry,
                positive_token=normalized_positive_token,
                filler_token=normalized_filler_token,
            )
            if token is None:
                continue
            if not entry.audio_path.is_file():
                raise FileNotFoundError(entry.audio_path)
            utt_id = _build_utterance_id(split_name, entry, index)
            wav_scp_lines.append(f"{utt_id} {entry.audio_path}")
            text_lines.append(f"{utt_id} {token}")
            utt2spk_lines.append(f"{utt_id} {split_name}")
            duration_seconds = _wav_duration_seconds(entry.audio_path)
            wav_dur_lines.append(f"{utt_id} {duration_seconds:.6f}")
            written_count += 1
            if token == normalized_positive_token:
                positive_count += 1
            else:
                negative_count += 1
        if written_count == 0:
            raise ValueError(f"{manifest_path} did not produce any WeKws-exportable entries.")
        (split_dir / "wav.scp").write_text("\n".join(wav_scp_lines) + "\n", encoding="utf-8")
        (split_dir / "text").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
        (split_dir / "utt2spk").write_text("\n".join(utt2spk_lines) + "\n", encoding="utf-8")
        (split_dir / "wav.dur").write_text("\n".join(wav_dur_lines) + "\n", encoding="utf-8")
        split_reports.append(
            WekwsExportSplitReport(
                split_name=split_name,
                manifest_path=manifest_path,
                output_dir=split_dir,
                entry_count=written_count,
                positive_count=positive_count,
                negative_count=negative_count,
                ignored_count=ignored_count,
            )
        )

    dict_dir = resolved_output_dir / "dict"
    dict_dir.mkdir(parents=True, exist_ok=True)
    dict_path = dict_dir / "dict.txt"
    words_path = dict_dir / "words.txt"
    dict_path.write_text(
        f"{normalized_filler_token} -1\n{normalized_positive_token} 0\n",
        encoding="utf-8",
    )
    words_path.write_text(
        f"{normalized_filler_token}\n{normalized_positive_token}\n",
        encoding="utf-8",
    )
    metadata = {
        "schema": "twinr_wekws_export_v1",
        "output_dir": str(resolved_output_dir),
        "positive_token": normalized_positive_token,
        "filler_token": normalized_filler_token,
        "splits": [
            {
                "name": report.split_name,
                "manifest_path": str(report.manifest_path),
                "output_dir": str(report.output_dir),
                "entry_count": report.entry_count,
                "positive_count": report.positive_count,
                "negative_count": report.negative_count,
                "ignored_count": report.ignored_count,
            }
            for report in split_reports
        ],
    }
    metadata_path = resolved_output_dir / "wekws_export_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return WekwsExportReport(
        output_dir=resolved_output_dir,
        dict_path=dict_path,
        words_path=words_path,
        positive_token=normalized_positive_token,
        filler_token=normalized_filler_token,
        split_reports=tuple(split_reports),
        metadata_path=metadata_path,
    )


__all__ = [
    "WekwsExportReport",
    "WekwsExportSplitReport",
    "export_wakeword_manifests_to_wekws",
]
