"""Tests for Twinr's synthetic Qwen3TTS wakeword corpus planner."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from twinr.proactive.wakeword.synthetic_corpus import (
    DEFAULT_AUGMENTATION_PROFILES,
    DEFAULT_FOLLOW_UPS,
    DEFAULT_GENERATION_PROFILES,
    DEFAULT_PREFIXES,
    DEFAULT_STYLE_PROFILES,
    SyntheticAugmentationProfile,
    apply_augmentation,
    build_qwen3tts_synthetic_corpus_plan,
    float_audio_to_pcm16,
    hash_to_split,
    manifest_row_for_request,
    normalize_synthetic_labels,
    render_wakeword_phrase,
    shard_speakers,
    synthesize_phrase_inventory,
    write_pcm16_wav,
)


class SyntheticCorpusTests(unittest.TestCase):
    def test_render_wakeword_phrase_normalizes_spacing(self) -> None:
        self.assertEqual(
            render_wakeword_phrase("twinna", prefix="  hallo  ", suffix="   ich brauche hilfe "),
            "hallo twinna ich brauche hilfe",
        )

    def test_hash_to_split_is_deterministic(self) -> None:
        self.assertEqual(hash_to_split("abc"), hash_to_split("abc"))
        self.assertIn(hash_to_split("abc"), {"train", "dev", "test"})

    def test_shard_speakers_selects_deterministic_subset(self) -> None:
        speakers = ("a", "b", "c", "d", "e")
        self.assertEqual(shard_speakers(speakers, shard_index=0, shard_count=2), ("a", "c", "e"))
        self.assertEqual(shard_speakers(speakers, shard_index=1, shard_count=2), ("b", "d"))

    def test_synthesize_phrase_inventory_expands_aliases(self) -> None:
        rows = synthesize_phrase_inventory(
            aliases=("twinna",),
            prefixes=("", "hallo "),
            follow_ups=("", " ich habe eine frage"),
        )
        self.assertEqual(
            rows,
            (
                ("twinna", "twinna"),
                ("twinna", "twinna ich habe eine frage"),
                ("twinna", "hallo twinna"),
                ("twinna", "hallo twinna ich habe eine frage"),
            ),
        )

    def test_build_plan_contains_positive_and_negative_rows(self) -> None:
        requests = build_qwen3tts_synthetic_corpus_plan(
            speakers=("vivian", "serena"),
            seeds=(11,),
            style_profiles=DEFAULT_STYLE_PROFILES[:1],
            generation_profiles=DEFAULT_GENERATION_PROFILES[:1],
            augmentation_profiles=DEFAULT_AUGMENTATION_PROFILES[:1],
            shard_index=0,
            shard_count=1,
        )
        self.assertGreater(len(requests), 0)
        labels = {request.label for request in requests}
        self.assertEqual(labels, {"negative", "positive"})
        splits = {request.split for request in requests}
        self.assertTrue(splits.issubset({"train", "dev", "test"}))

    def test_build_plan_honors_custom_prefixes_and_follow_ups(self) -> None:
        requests = build_qwen3tts_synthetic_corpus_plan(
            speakers=("vivian",),
            seeds=(11,),
            style_profiles=DEFAULT_STYLE_PROFILES[:1],
            generation_profiles=DEFAULT_GENERATION_PROFILES[:1],
            augmentation_profiles=DEFAULT_AUGMENTATION_PROFILES[:1],
            prefixes=(DEFAULT_PREFIXES[0],),
            follow_ups=(DEFAULT_FOLLOW_UPS[0],),
            shard_index=0,
            shard_count=1,
        )
        self.assertEqual(len(requests), 26)

    def test_normalize_synthetic_labels_accepts_all_and_deduplicates(self) -> None:
        self.assertEqual(normalize_synthetic_labels(("positive", "negative")), ("positive", "negative"))
        self.assertEqual(normalize_synthetic_labels(("negative", "negative")), ("negative",))
        self.assertEqual(normalize_synthetic_labels(("all",)), ("positive", "negative"))

    def test_build_plan_can_filter_negative_only(self) -> None:
        requests = build_qwen3tts_synthetic_corpus_plan(
            speakers=("vivian",),
            seeds=(11,),
            style_profiles=DEFAULT_STYLE_PROFILES[:1],
            generation_profiles=DEFAULT_GENERATION_PROFILES[:1],
            augmentation_profiles=DEFAULT_AUGMENTATION_PROFILES[:1],
            labels=("negative",),
            shard_index=0,
            shard_count=1,
        )
        self.assertGreater(len(requests), 0)
        self.assertEqual({request.label for request in requests}, {"negative"})

    def test_apply_augmentation_changes_signal_and_keeps_bounds(self) -> None:
        t = np.linspace(0.0, 1.0, 1600, endpoint=False, dtype=np.float32)
        base = np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
        profile = SyntheticAugmentationProfile(
            key="test",
            gain_db=-6.0,
            additive_noise_level=0.01,
            impulse_decay=0.8,
            impulse_length=5,
            lowpass_alpha=0.2,
            clip_ratio=0.8,
        )
        out = apply_augmentation(base, sample_rate=16000, profile=profile, seed=11)
        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(out.size, 0)
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)
        self.assertFalse(np.allclose(base, out[: base.size], atol=1e-4))

    def test_manifest_row_carries_twinr_eval_fields(self) -> None:
        request = build_qwen3tts_synthetic_corpus_plan(
            speakers=("vivian",),
            seeds=(11,),
            style_profiles=DEFAULT_STYLE_PROFILES[:1],
            generation_profiles=DEFAULT_GENERATION_PROFILES[:1],
            augmentation_profiles=DEFAULT_AUGMENTATION_PROFILES[:1],
        )[0]
        row = manifest_row_for_request(request, audio_path=Path("/tmp/example.wav"))
        self.assertEqual(row["label"], request.label)
        self.assertEqual(row["audio_path"], "/tmp/example.wav")
        self.assertEqual(row["speaker"], request.speaker)
        self.assertEqual(row["augmentation_key"], request.augmentation_key)

    def test_write_pcm16_wav_persists_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.wav"
            write_pcm16_wav(
                path,
                samples=np.linspace(-0.5, 0.5, 800, dtype=np.float32),
                sample_rate=16000,
            )
            self.assertTrue(path.is_file())
            pcm = float_audio_to_pcm16(np.array([0.0, 0.5, -0.5], dtype=np.float32))
            self.assertEqual(pcm.dtype.str, "<i2")


if __name__ == "__main__":
    unittest.main()
