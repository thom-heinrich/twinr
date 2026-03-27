# CHANGELOG: 2026-03-27
# BUG-1: Fixed non-deterministic slot sampling when recipe slot pools are unordered collections (e.g. set/frozenset).
# BUG-2: Fixed train/dev/test leakage by replacing per-sample hash splits with deterministic group-aware splits.
# BUG-3: Fixed silent label ambiguity where identical normalized text across labels kept the first label and discarded the rest.
# BUG-4: Fixed parameter coercions that silently turned invalid requests (e.g. <=0 samples-per-label) into different outputs.
# SEC-1: Hardened output writes with atomic temp-file replacement and destination symlink refusal to reduce file-clobber/torn-write risk.
# SEC-2: Added explicit parameter validation and raw-sample caps to reduce practical self-DoS risk on Raspberry Pi deployments.
# IMP-1: Upgraded near-duplicate curation from family-local O(n^2) scanning to label-wide MinHash-style candidate indexing plus exact verification.
# IMP-2: Added adaptive generation rounds so strict curation can recover requested per-label counts without manual oversampling guesswork.
# IMP-3: Tightened generation-leakage detection to phrase-based markers, reducing false rejections of legitimate utterances containing words like "route" or "router".
# BREAKING: Default split assignment is now "group_stratified" instead of legacy per-sample hashing to prevent evaluation leakage across paraphrase families.

"""Generate and curate synthetic transcript corpora for the local router.

This module owns Twinr's deterministic bootstrap corpora for the local
two-stage router. It produces broad German utterance families, attaches both
backend-route labels and user-centered labels, applies light transcript-noise
variants, removes duplicates and generation leakage, and exports JSONL that the
bundle builders can consume directly.

Example:

```bash
PYTHONPATH=src python3 -m twinr.agent.routing.synthetic_corpus \
  --output-path artifacts/router/synthetic/router_samples.jsonl \
  --samples-per-label 1024
```
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import unicodedata
from typing import Iterable, Iterator, Mapping, Sequence

from .contracts import ROUTE_LABEL_VALUES, normalize_route_label
from .synthetic_recipe_bank import (
    SyntheticRouteRecipe,
    SyntheticRouteTemplate as _SyntheticRouteTemplate,
    _ROUTE_RECIPES,
)
from .user_intent import (
    USER_INTENT_LABEL_VALUES,
    default_user_intent_for_route_label,
    normalize_user_intent_label,
)

_REFERENCE_DATE = "2026-03-27"
_SCHEMA_VERSION = "synthetic_router_v3"
_SPLIT_TRAIN_RATIO = 0.82
_SPLIT_DEV_RATIO = 0.91
_DEFAULT_SOURCE = "synthetic_router_v3_frontier_curated"
_LABEL_NAMESPACE_BACKEND = "backend"
_LABEL_NAMESPACE_USER = "user"
_NEAR_DUP_SCOPE_LABEL = "label"
_NEAR_DUP_SCOPE_GLOBAL = "global"
_SPLIT_STRATEGY_GROUP = "group_stratified"
_SPLIT_STRATEGY_LEGACY = "legacy_hash"
_MAX_GENERATION_ROUNDS = 6
_MAX_TOTAL_RAW_SAMPLES = 250_000
_MINHASH_PERMUTATIONS = 24
_MINHASH_BAND_SIZE = 4
_HASH_MASK_64 = (1 << 64) - 1

_GENERATION_LEAKAGE_PHRASES: tuple[tuple[str, ...], ...] = (
    ("backend", "label"),
    ("user", "label"),
    ("route", "label"),
    ("intent", "label"),
    ("memory", "class"),
    ("tool", "call"),
    ("web", "research"),
    ("parametric",),
    ("routing",),
)

SyntheticRouteTemplate = _SyntheticRouteTemplate

_MINHASH_SEEDS: tuple[int, ...] = tuple(
    ((0x9E3779B185EBCA87 * (index + 1)) ^ 0xC2B2AE3D27D4EB4F) & _HASH_MASK_64
    for index in range(_MINHASH_PERMUTATIONS)
)


@dataclass(frozen=True, slots=True)
class SyntheticRouteSample:
    """Store one generated router transcript sample plus provenance metadata."""

    text: str
    label: str
    sample_id: str
    split: str
    user_label: str | None = None
    source: str = _DEFAULT_SOURCE
    family_key: str | None = None
    template_key: str | None = None
    difficulty: str = "standard"
    noise_key: str = "clean"
    reference_date: str = _REFERENCE_DATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", " ".join(str(self.text or "").strip().split()))
        object.__setattr__(self, "label", normalize_route_label(self.label))
        object.__setattr__(
            self,
            "user_label",
            normalize_user_intent_label(
                self.user_label or default_user_intent_for_route_label(self.label)
            ),
        )
        object.__setattr__(self, "sample_id", str(self.sample_id or "").strip())
        object.__setattr__(self, "split", str(self.split or "").strip().lower())
        object.__setattr__(self, "source", str(self.source or _DEFAULT_SOURCE).strip())
        object.__setattr__(self, "family_key", str(self.family_key or "").strip().lower() or None)
        object.__setattr__(self, "template_key", str(self.template_key or "").strip().lower() or None)
        object.__setattr__(self, "difficulty", str(self.difficulty or "standard").strip().lower())
        object.__setattr__(self, "noise_key", str(self.noise_key or "clean").strip().lower())
        object.__setattr__(self, "reference_date", str(self.reference_date or _REFERENCE_DATE).strip())
        if not self.text:
            raise ValueError("SyntheticRouteSample.text must not be empty.")
        if not self.sample_id:
            raise ValueError("SyntheticRouteSample.sample_id must not be empty.")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError("SyntheticRouteSample.split must be train/dev/test.")

    def label_for_namespace(self, namespace: str) -> str:
        """Return the label used for one export namespace."""

        normalized_namespace = _normalize_label_namespace(namespace)
        if normalized_namespace == _LABEL_NAMESPACE_USER:
            return str(self.user_label)
        return self.label

    def to_json_dict(self, *, label_namespace: str = _LABEL_NAMESPACE_BACKEND) -> dict[str, object]:
        """Return one JSON-serializable payload for JSONL export."""

        payload: dict[str, object] = {
            "id": self.sample_id,
            "text": self.text,
            "label": self.label_for_namespace(label_namespace),
            "split": self.split,
            "source": self.source,
            "reference_date": self.reference_date,
            "schema_version": _SCHEMA_VERSION,
        }
        payload["backend_label"] = self.label
        payload["user_label"] = str(self.user_label)
        if self.family_key:
            payload["family"] = self.family_key
        if self.template_key:
            payload["template"] = self.template_key
        if self.difficulty:
            payload["difficulty"] = self.difficulty
        if self.noise_key:
            payload["noise"] = self.noise_key
        return payload


@dataclass(frozen=True, slots=True)
class SyntheticRouteCurationReport:
    """Summarize one synthetic-router generation and curation run."""

    generated_count: int
    kept_count: int
    rejected_exact_duplicates: int
    rejected_near_duplicates: int
    rejected_style_collapses: int
    rejected_generation_leakage: int
    per_label: Mapping[str, int]
    per_split: Mapping[str, int]
    rejected_label_conflicts: int = 0
    rejected_quota_overflow: int = 0
    generation_rounds: int = 1
    split_strategy: str = _SPLIT_STRATEGY_GROUP
    schema_version: str = _SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-friendly report payload."""

        return {
            "generated_count": int(self.generated_count),
            "kept_count": int(self.kept_count),
            "rejected_exact_duplicates": int(self.rejected_exact_duplicates),
            "rejected_near_duplicates": int(self.rejected_near_duplicates),
            "rejected_style_collapses": int(self.rejected_style_collapses),
            "rejected_generation_leakage": int(self.rejected_generation_leakage),
            "rejected_label_conflicts": int(self.rejected_label_conflicts),
            "rejected_quota_overflow": int(self.rejected_quota_overflow),
            "generation_rounds": int(self.generation_rounds),
            "split_strategy": self.split_strategy,
            "schema_version": self.schema_version,
            "per_label": dict(self.per_label),
            "per_split": dict(self.per_split),
        }


@dataclass(frozen=True, slots=True)
class _AcceptedSignature:
    token_signature: frozenset[str]
    trigram_signature: frozenset[str]
    combined_minhash: tuple[int, ...]


def generate_synthetic_route_samples(
    *,
    samples_per_label: int = 1024,
    seed: int = 20260322,
    oversample_factor: float = 1.7,
    max_near_duplicate_similarity: float = 0.92,
    max_samples_per_template_bucket: int = 40,
    label_namespace: str = _LABEL_NAMESPACE_BACKEND,
    split_strategy: str = _SPLIT_STRATEGY_GROUP,
    near_duplicate_scope: str = _NEAR_DUP_SCOPE_LABEL,
    max_generation_rounds: int = _MAX_GENERATION_ROUNDS,
    max_total_raw_samples: int = _MAX_TOTAL_RAW_SAMPLES,
) -> tuple[tuple[SyntheticRouteSample, ...], SyntheticRouteCurationReport]:
    """Generate one curated synthetic dataset for the local semantic router."""

    _validate_generation_parameters(
        samples_per_label=samples_per_label,
        oversample_factor=oversample_factor,
        max_near_duplicate_similarity=max_near_duplicate_similarity,
        max_samples_per_template_bucket=max_samples_per_template_bucket,
        max_generation_rounds=max_generation_rounds,
        max_total_raw_samples=max_total_raw_samples,
    )
    normalized_namespace = _normalize_label_namespace(label_namespace)
    normalized_split_strategy = _normalize_split_strategy(split_strategy)
    normalized_near_duplicate_scope = _normalize_near_duplicate_scope(near_duplicate_scope)
    label_values = _label_values_for_namespace(normalized_namespace)
    target_per_label = int(samples_per_label)
    rng = random.Random(seed)

    recipes_by_label: dict[str, list[SyntheticRouteRecipe]] = defaultdict(list)
    for recipe in _ROUTE_RECIPES:
        recipes_by_label[_recipe_label_for_namespace(recipe, normalized_namespace)].append(recipe)

    missing_recipe_labels = [label for label in label_values if not recipes_by_label.get(label)]
    if missing_recipe_labels:
        raise RuntimeError(
            "Synthetic route generation has no recipes for some labels in the requested namespace: "
            f"{missing_recipe_labels}"
        )

    recipe_cycles = {
        label: _shuffled_cycle(recipes_by_label[label], rng=rng)
        for label in label_values
    }

    raw_samples: list[SyntheticRouteSample] = []
    raw_counts: Counter[str] = Counter()
    current_oversample_factor = float(oversample_factor)
    last_report: SyntheticRouteCurationReport | None = None

    for generation_round in range(1, max_generation_rounds + 1):
        desired_raw_per_label = max(
            target_per_label,
            int(math.ceil(target_per_label * current_oversample_factor)),
        )
        _extend_raw_samples_to_target(
            raw_samples=raw_samples,
            raw_counts=raw_counts,
            desired_raw_per_label=desired_raw_per_label,
            label_values=label_values,
            recipe_cycles=recipe_cycles,
            rng=rng,
            max_total_raw_samples=max_total_raw_samples,
        )
        curated, report = curate_synthetic_route_samples(
            raw_samples,
            target_per_label=target_per_label,
            max_near_duplicate_similarity=max_near_duplicate_similarity,
            max_samples_per_template_bucket=max_samples_per_template_bucket,
            label_namespace=normalized_namespace,
            split_strategy=normalized_split_strategy,
            near_duplicate_scope=normalized_near_duplicate_scope,
            strict_target_counts=False,
        )
        report = replace(report, generation_rounds=generation_round)
        last_report = report
        if all(report.per_label.get(label, 0) >= target_per_label for label in label_values):
            return curated, report
        current_oversample_factor = max(current_oversample_factor * 1.35, current_oversample_factor + 0.25)

    assert last_report is not None
    raise RuntimeError(
        "Synthetic route generation could not keep enough samples for every label after adaptive generation. "
        f"Counts={dict(last_report.per_label)} requested_per_label={target_per_label} "
        f"rounds={max_generation_rounds}"
    )


def generate_synthetic_user_intent_samples(
    *,
    samples_per_label: int = 1024,
    seed: int = 20260322,
    oversample_factor: float = 1.7,
    max_near_duplicate_similarity: float = 0.92,
    max_samples_per_template_bucket: int = 40,
    split_strategy: str = _SPLIT_STRATEGY_GROUP,
    near_duplicate_scope: str = _NEAR_DUP_SCOPE_LABEL,
    max_generation_rounds: int = _MAX_GENERATION_ROUNDS,
    max_total_raw_samples: int = _MAX_TOTAL_RAW_SAMPLES,
) -> tuple[tuple[SyntheticRouteSample, ...], SyntheticRouteCurationReport]:
    """Generate one curated synthetic dataset labeled by user-centered intent."""

    return generate_synthetic_route_samples(
        samples_per_label=samples_per_label,
        seed=seed,
        oversample_factor=oversample_factor,
        max_near_duplicate_similarity=max_near_duplicate_similarity,
        max_samples_per_template_bucket=max_samples_per_template_bucket,
        label_namespace=_LABEL_NAMESPACE_USER,
        split_strategy=split_strategy,
        near_duplicate_scope=near_duplicate_scope,
        max_generation_rounds=max_generation_rounds,
        max_total_raw_samples=max_total_raw_samples,
    )


def curate_synthetic_route_samples(
    samples: Sequence[SyntheticRouteSample],
    *,
    target_per_label: int | None = None,
    max_near_duplicate_similarity: float = 0.92,
    max_samples_per_template_bucket: int = 40,
    label_namespace: str = _LABEL_NAMESPACE_BACKEND,
    split_strategy: str = _SPLIT_STRATEGY_GROUP,
    near_duplicate_scope: str = _NEAR_DUP_SCOPE_LABEL,
    strict_target_counts: bool = True,
) -> tuple[tuple[SyntheticRouteSample, ...], SyntheticRouteCurationReport]:
    """Deduplicate and sanitize synthetic router samples."""

    _validate_curation_parameters(
        target_per_label=target_per_label,
        max_near_duplicate_similarity=max_near_duplicate_similarity,
        max_samples_per_template_bucket=max_samples_per_template_bucket,
    )
    normalized_namespace = _normalize_label_namespace(label_namespace)
    normalized_split_strategy = _normalize_split_strategy(split_strategy)
    normalized_near_duplicate_scope = _normalize_near_duplicate_scope(near_duplicate_scope)
    label_values = _label_values_for_namespace(normalized_namespace)
    requested_per_label = None if target_per_label is None else int(target_per_label)

    ambiguous_texts = _ambiguous_normalized_texts(samples, label_namespace=normalized_namespace)

    kept: list[SyntheticRouteSample] = []
    exact_seen: set[str] = set()
    shape_counts: Counter[tuple[str, str | None, str | None]] = Counter()
    per_label_counts: Counter[str] = Counter()
    accepted_records: list[_AcceptedSignature] = []
    band_index: dict[tuple[str, int, tuple[int, ...]], list[int]] = defaultdict(list)

    rejected_exact_duplicates = 0
    rejected_near_duplicates = 0
    rejected_style_collapses = 0
    rejected_generation_leakage = 0
    rejected_label_conflicts = 0
    rejected_quota_overflow = 0

    for sample in samples:
        sample_label = sample.label_for_namespace(normalized_namespace)
        if requested_per_label is not None and per_label_counts[sample_label] >= requested_per_label:
            rejected_quota_overflow += 1
            continue

        normalized_text = _normalize_for_dedupe(sample.text)
        if normalized_text in ambiguous_texts:
            rejected_label_conflicts += 1
            continue
        if normalized_text in exact_seen:
            rejected_exact_duplicates += 1
            continue
        if _looks_like_generation_leakage(normalized_text):
            rejected_generation_leakage += 1
            continue

        bucket_key = (sample_label, sample.family_key, sample.template_key)
        if shape_counts[bucket_key] >= max_samples_per_template_bucket:
            rejected_style_collapses += 1
            continue

        token_signature = frozenset(_tokenize(normalized_text))
        trigram_signature = frozenset(_char_trigrams(normalized_text))
        combined_shingles = frozenset(
            [f"w:{token}" for token in token_signature]
            + [f"c:{trigram}" for trigram in trigram_signature]
        )
        combined_minhash = _minhash_signature(combined_shingles)
        candidate_namespace = sample_label if normalized_near_duplicate_scope == _NEAR_DUP_SCOPE_LABEL else "__global__"
        candidate_indices = _lookup_candidate_indices(
            candidate_namespace=candidate_namespace,
            combined_minhash=combined_minhash,
            band_index=band_index,
        )

        duplicate_found = False
        for index in candidate_indices:
            accepted = accepted_records[index]
            if max(
                _jaccard_similarity(token_signature, accepted.token_signature),
                _jaccard_similarity(trigram_signature, accepted.trigram_signature),
            ) >= max_near_duplicate_similarity:
                duplicate_found = True
                break
        if duplicate_found:
            rejected_near_duplicates += 1
            continue

        exact_seen.add(normalized_text)
        shape_counts[bucket_key] += 1
        per_label_counts[sample_label] += 1
        accepted_index = len(accepted_records)
        accepted_records.append(
            _AcceptedSignature(
                token_signature=token_signature,
                trigram_signature=trigram_signature,
                combined_minhash=combined_minhash,
            )
        )
        _add_candidate_index(
            accepted_index=accepted_index,
            candidate_namespace=candidate_namespace,
            combined_minhash=combined_minhash,
            band_index=band_index,
        )
        kept.append(sample)
        if requested_per_label is not None and all(
            per_label_counts[label] >= requested_per_label for label in label_values
        ):
            break

    if strict_target_counts and requested_per_label is not None and any(
        per_label_counts[label] < requested_per_label for label in label_values
    ):
        raise RuntimeError(
            "Synthetic route curation could not keep enough samples for every label. "
            f"Counts={dict(per_label_counts)} requested_per_label={requested_per_label}"
        )

    reassigned = _assign_splits(
        kept,
        label_namespace=normalized_namespace,
        split_strategy=normalized_split_strategy,
    )
    split_counts = Counter(sample.split for sample in reassigned)

    report = SyntheticRouteCurationReport(
        generated_count=len(samples),
        kept_count=len(reassigned),
        rejected_exact_duplicates=rejected_exact_duplicates,
        rejected_near_duplicates=rejected_near_duplicates,
        rejected_style_collapses=rejected_style_collapses,
        rejected_generation_leakage=rejected_generation_leakage,
        rejected_label_conflicts=rejected_label_conflicts,
        rejected_quota_overflow=rejected_quota_overflow,
        per_label={label: int(per_label_counts.get(label, 0)) for label in label_values},
        per_split={split: int(split_counts.get(split, 0)) for split in ("train", "dev", "test")},
        split_strategy=normalized_split_strategy,
    )
    return tuple(reassigned), report


def write_synthetic_route_samples_jsonl(
    samples: Iterable[SyntheticRouteSample],
    path: str | Path,
    *,
    label_namespace: str = _LABEL_NAMESPACE_BACKEND,
) -> Path:
    """Write synthetic route samples to JSONL on disk."""

    normalized_namespace = _normalize_label_namespace(label_namespace)
    output_path = _prepare_output_path(path)
    _atomic_write_lines(
        output_path,
        (
            json.dumps(
                sample.to_json_dict(label_namespace=normalized_namespace),
                ensure_ascii=True,
            )
            for sample in samples
        ),
    )
    return output_path


def write_synthetic_user_intent_samples_jsonl(
    samples: Iterable[SyntheticRouteSample],
    path: str | Path,
) -> Path:
    """Write synthetic samples to JSONL with user-centered labels."""

    return write_synthetic_route_samples_jsonl(
        samples,
        path,
        label_namespace=_LABEL_NAMESPACE_USER,
    )


def _extend_raw_samples_to_target(
    *,
    raw_samples: list[SyntheticRouteSample],
    raw_counts: Counter[str],
    desired_raw_per_label: int,
    label_values: Sequence[str],
    recipe_cycles: Mapping[str, Iterator[SyntheticRouteRecipe]],
    rng: random.Random,
    max_total_raw_samples: int,
) -> None:
    while any(raw_counts[label] < desired_raw_per_label for label in label_values):
        for label in label_values:
            if raw_counts[label] >= desired_raw_per_label:
                continue
            if len(raw_samples) >= max_total_raw_samples:
                raise RuntimeError(
                    "Synthetic route generation exceeded the configured raw-sample cap before "
                    "reaching the requested size. Increase max_total_raw_samples only if you "
                    "intend to allow larger generation jobs."
                )
            recipe = next(recipe_cycles[label])
            raw_samples.append(_render_recipe_sample(recipe, rng=rng))
            raw_counts[label] += 1


def _render_recipe_sample(
    recipe: SyntheticRouteRecipe,
    *,
    rng: random.Random,
) -> SyntheticRouteSample:
    template = _stable_random_choice(rng, recipe.templates)
    slot_payload = {
        name: _stable_random_choice(rng, values)
        for name, values in dict(template.resolved_slot_values(recipe.slot_values)).items()
    }
    rendered = " ".join(template.render(slot_payload).split())
    noise_key = _stable_random_choice(rng, recipe.noise_pool)
    noisy_text = _apply_noise_profile(rendered, noise_key=str(noise_key), rng=rng)
    normalized_text = " ".join(noisy_text.split())
    sample_key = "|".join(
        (
            normalize_route_label(recipe.label),
            str(recipe.family_key or ""),
            str(template.key or ""),
            str(recipe.difficulty or "standard"),
            str(noise_key),
            normalized_text,
        )
    )
    sample_id = hashlib.sha256(sample_key.encode("utf-8")).hexdigest()[:16]
    legacy_split = _hash_to_split(sample_id)
    return SyntheticRouteSample(
        text=normalized_text,
        label=recipe.label,
        sample_id=f"{normalize_route_label(recipe.label)}_{sample_id}",
        split=legacy_split,
        user_label=recipe.user_label,
        family_key=recipe.family_key,
        template_key=template.key,
        difficulty=recipe.difficulty,
        noise_key=str(noise_key),
    )


def _normalize_label_namespace(value: object) -> str:
    normalized = str(value or _LABEL_NAMESPACE_BACKEND).strip().lower()
    if normalized not in {_LABEL_NAMESPACE_BACKEND, _LABEL_NAMESPACE_USER}:
        raise ValueError(f"Unsupported synthetic label namespace: {value!r}")
    return normalized


def _normalize_near_duplicate_scope(value: object) -> str:
    normalized = str(value or _NEAR_DUP_SCOPE_LABEL).strip().lower()
    if normalized not in {_NEAR_DUP_SCOPE_LABEL, _NEAR_DUP_SCOPE_GLOBAL}:
        raise ValueError(f"Unsupported near-duplicate scope: {value!r}")
    return normalized


def _normalize_split_strategy(value: object) -> str:
    normalized = str(value or _SPLIT_STRATEGY_GROUP).strip().lower()
    if normalized not in {_SPLIT_STRATEGY_GROUP, _SPLIT_STRATEGY_LEGACY}:
        raise ValueError(f"Unsupported split strategy: {value!r}")
    return normalized


def _label_values_for_namespace(namespace: str) -> tuple[str, ...]:
    normalized_namespace = _normalize_label_namespace(namespace)
    if normalized_namespace == _LABEL_NAMESPACE_USER:
        return USER_INTENT_LABEL_VALUES
    return ROUTE_LABEL_VALUES


def _recipe_label_for_namespace(recipe: SyntheticRouteRecipe, namespace: str) -> str:
    normalized_namespace = _normalize_label_namespace(namespace)
    if normalized_namespace == _LABEL_NAMESPACE_USER:
        return normalize_user_intent_label(
            recipe.user_label or default_user_intent_for_route_label(recipe.label)
        )
    return normalize_route_label(recipe.label)


def _shuffled_cycle(
    items: Sequence[SyntheticRouteRecipe],
    *,
    rng: random.Random,
) -> Iterator[SyntheticRouteRecipe]:
    pool = list(items)
    if not pool:
        raise ValueError("Synthetic recipe pools must not be empty.")
    shuffled: list[SyntheticRouteRecipe] = []
    while True:
        if not shuffled:
            shuffled = list(pool)
            rng.shuffle(shuffled)
        yield shuffled.pop()


def _stable_random_choice(rng: random.Random, values: object):
    sequence = _coerce_choice_sequence(values)
    return sequence[rng.randrange(len(sequence))]


def _coerce_choice_sequence(values: object) -> tuple[object, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        sequence: list[object] = [values]
    else:
        try:
            sequence = list(values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(f"Expected an iterable choice pool, got {type(values)!r}") from exc
    if not sequence:
        raise ValueError("Choice pools must not be empty.")
    if isinstance(values, (set, frozenset)):
        sequence.sort(key=_stable_sort_key)
    return tuple(sequence)


def _stable_sort_key(value: object) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _apply_noise_profile(
    text: str,
    *,
    noise_key: str,
    rng: random.Random,
) -> str:
    normalized_key = str(noise_key or "clean").strip().lower()
    if normalized_key == "clean":
        return text
    if normalized_key == "lowercase":
        return text.lower()
    if normalized_key == "no_punct":
        return _strip_transcript_punctuation(text).lower()
    if normalized_key == "umlaut_flat":
        flattened = (
            text.replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("Ä", "Ae")
            .replace("Ö", "Oe")
            .replace("Ü", "Ue")
            .replace("ß", "ss")
        )
        return _strip_transcript_punctuation(flattened).lower()
    if normalized_key == "filler":
        prefix = _stable_random_choice(rng, ("aeh ", "hm ", "also ", "sag mal "))
        return _strip_transcript_punctuation(f"{prefix}{text}").lower()
    raise ValueError(f"Unsupported synthetic transcript noise profile: {noise_key!r}")


def _hash_to_split(sample_id: str) -> str:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()[0]
    ratio = float(digest) / 255.0
    if ratio < _SPLIT_TRAIN_RATIO:
        return "train"
    if ratio < _SPLIT_DEV_RATIO:
        return "dev"
    return "test"


def _assign_splits(
    samples: Sequence[SyntheticRouteSample],
    *,
    label_namespace: str,
    split_strategy: str,
) -> list[SyntheticRouteSample]:
    normalized_split_strategy = _normalize_split_strategy(split_strategy)
    if normalized_split_strategy == _SPLIT_STRATEGY_LEGACY:
        return [replace(sample, split=_hash_to_split(sample.sample_id)) for sample in samples]
    return _assign_group_stratified_splits(samples, label_namespace=label_namespace)


def _assign_group_stratified_splits(
    samples: Sequence[SyntheticRouteSample],
    *,
    label_namespace: str,
) -> list[SyntheticRouteSample]:
    if not samples:
        return []
    normalized_namespace = _normalize_label_namespace(label_namespace)
    assigned_splits: dict[int, str] = {}
    samples_by_label: dict[str, list[tuple[int, SyntheticRouteSample]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        label = sample.label_for_namespace(normalized_namespace)
        samples_by_label[label].append((index, sample))

    for label, labeled_samples in samples_by_label.items():
        group_mode = _group_mode_for_label(tuple(sample for _, sample in labeled_samples))
        groups: dict[str, list[int]] = defaultdict(list)
        for index, sample in labeled_samples:
            group_key = _sample_group_key(sample, label_namespace=normalized_namespace, group_mode=group_mode)
            groups[group_key].append(index)

        total = sum(len(indices) for indices in groups.values())
        targets = _split_targets(total)
        counts: Counter[str] = Counter()
        group_items = list(groups.items())
        group_items.sort(
            key=lambda item: (
                -len(item[1]),
                _stable_u64_hash(f"{label}|{item[0]}"),
            )
        )
        for group_key, indices in group_items:
            size = len(indices)
            split = _choose_split_for_group(
                label=label,
                group_key=group_key,
                size=size,
                counts=counts,
                targets=targets,
            )
            counts[split] += size
            for sample_index in indices:
                assigned_splits[sample_index] = split

    reassigned: list[SyntheticRouteSample] = []
    for index, sample in enumerate(samples):
        reassigned.append(replace(sample, split=assigned_splits[index]))
    return reassigned


def _split_targets(total: int) -> dict[str, int]:
    if total < 0:
        raise ValueError("Split target total must be non-negative.")
    if total == 0:
        return {"train": 0, "dev": 0, "test": 0}
    weights = {
        "train": _SPLIT_TRAIN_RATIO,
        "dev": _SPLIT_DEV_RATIO - _SPLIT_TRAIN_RATIO,
        "test": 1.0 - _SPLIT_DEV_RATIO,
    }
    exact = {split: total * weight for split, weight in weights.items()}
    floors = {split: int(math.floor(value)) for split, value in exact.items()}
    remainder = total - sum(floors.values())
    if remainder > 0:
        ranked = sorted(
            exact,
            key=lambda split: (-(exact[split] - floors[split]), split),
        )
        for split in ranked[:remainder]:
            floors[split] += 1
    return floors


def _choose_split_for_group(
    *,
    label: str,
    group_key: str,
    size: int,
    counts: Mapping[str, int],
    targets: Mapping[str, int],
) -> str:
    if size <= 0:
        raise ValueError("Group size must be positive.")
    stable_tie = _stable_u64_hash(f"{label}|{group_key}")
    split_order = ("train", "dev", "test")
    rotated_order = split_order[stable_tie % len(split_order):] + split_order[:stable_tie % len(split_order)]

    def score(split: str) -> tuple[int, int, int, int]:
        current = int(counts.get(split, 0))
        target = int(targets.get(split, 0))
        overflow = max(0, current + size - target)
        distance = abs((current + size) - target)
        fill_rank = current
        tie_break = rotated_order.index(split)
        return (overflow, distance, fill_rank, tie_break)

    return min(split_order, key=score)


def _group_mode_for_label(samples: Sequence[SyntheticRouteSample]) -> str:
    family_keys = {sample.family_key for sample in samples if sample.family_key}
    template_keys = {sample.template_key for sample in samples if sample.template_key}
    if len(family_keys) >= 3:
        return "family"
    if len(template_keys) >= 3:
        return "template"
    return "text"


def _sample_group_key(
    sample: SyntheticRouteSample,
    *,
    label_namespace: str,
    group_mode: str,
) -> str:
    normalized_namespace = _normalize_label_namespace(label_namespace)
    label = sample.label_for_namespace(normalized_namespace)
    if group_mode == "family" and sample.family_key:
        return f"{label}|family:{sample.family_key}"
    if group_mode == "template" and sample.template_key:
        return f"{label}|template:{sample.template_key}"
    return f"{label}|text:{_normalize_for_dedupe(sample.text)}"


def _ambiguous_normalized_texts(
    samples: Sequence[SyntheticRouteSample],
    *,
    label_namespace: str,
) -> set[str]:
    label_sets_by_text: dict[str, set[str]] = defaultdict(set)
    normalized_namespace = _normalize_label_namespace(label_namespace)
    for sample in samples:
        normalized_text = _normalize_for_dedupe(sample.text)
        label_sets_by_text[normalized_text].add(sample.label_for_namespace(normalized_namespace))
    return {text for text, labels in label_sets_by_text.items() if len(labels) > 1}


def _looks_like_generation_leakage(normalized_text: str) -> bool:
    tokens = _tokenize(normalized_text)
    if not tokens:
        return False
    for phrase in _GENERATION_LEAKAGE_PHRASES:
        phrase_length = len(phrase)
        for index in range(len(tokens) - phrase_length + 1):
            if tokens[index:index + phrase_length] == phrase:
                return True
    return False


def _normalize_for_dedupe(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    cleaned = []
    for character in normalized:
        if character.isalnum() or character.isspace():
            cleaned.append(character)
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.split() if token)


def _char_trigrams(text: str) -> tuple[str, ...]:
    padded = f"  {text}  "
    if len(padded) < 3:
        return (padded,)
    return tuple(padded[index:index + 3] for index in range(len(padded) - 2))


def _jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right)) / float(len(union))


def _stable_u64_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def _minhash_signature(shingles: frozenset[str]) -> tuple[int, ...]:
    if not shingles:
        return tuple(0 for _ in range(_MINHASH_PERMUTATIONS))
    base_hashes = [_stable_u64_hash(shingle) for shingle in shingles]
    signature: list[int] = []
    for seed in _MINHASH_SEEDS:
        minimum = _HASH_MASK_64
        for base_hash in base_hashes:
            mixed = ((base_hash ^ seed) * 0x9E3779B97F4A7C15) & _HASH_MASK_64
            if mixed < minimum:
                minimum = mixed
        signature.append(minimum)
    return tuple(signature)


def _minhash_bands(signature: Sequence[int]) -> Iterator[tuple[int, tuple[int, ...]]]:
    if len(signature) != _MINHASH_PERMUTATIONS:
        raise ValueError("Unexpected MinHash signature length.")
    for offset in range(0, len(signature), _MINHASH_BAND_SIZE):
        band_index = offset // _MINHASH_BAND_SIZE
        yield band_index, tuple(signature[offset:offset + _MINHASH_BAND_SIZE])


def _lookup_candidate_indices(
    *,
    candidate_namespace: str,
    combined_minhash: Sequence[int],
    band_index: Mapping[tuple[str, int, tuple[int, ...]], list[int]],
) -> tuple[int, ...]:
    candidate_ids: set[int] = set()
    for band_number, band_value in _minhash_bands(combined_minhash):
        candidate_ids.update(band_index.get((candidate_namespace, band_number, band_value), ()))
    return tuple(sorted(candidate_ids))


def _add_candidate_index(
    *,
    accepted_index: int,
    candidate_namespace: str,
    combined_minhash: Sequence[int],
    band_index: dict[tuple[str, int, tuple[int, ...]], list[int]],
) -> None:
    for band_number, band_value in _minhash_bands(combined_minhash):
        band_index[(candidate_namespace, band_number, band_value)].append(accepted_index)


def _strip_transcript_punctuation(text: str) -> str:
    characters: list[str] = []
    for character in str(text or ""):
        category = unicodedata.category(character)
        if category.startswith("P") or category.startswith("S"):
            characters.append(" ")
        else:
            characters.append(character)
    return "".join(characters)


def _prepare_output_path(path: str | Path) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.is_symlink():
        raise ValueError(f"Refusing to overwrite symlinked output path: {output_path}")
    return output_path.resolve(strict=False)


def _atomic_write_lines(path: Path, lines: Iterable[str]) -> None:
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            for line in lines:
                handle.write(line)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _validate_generation_parameters(
    *,
    samples_per_label: int,
    oversample_factor: float,
    max_near_duplicate_similarity: float,
    max_samples_per_template_bucket: int,
    max_generation_rounds: int,
    max_total_raw_samples: int,
) -> None:
    if int(samples_per_label) <= 0:
        raise ValueError(f"samples_per_label must be > 0, got {samples_per_label!r}")
    if float(oversample_factor) < 1.0:
        raise ValueError(f"oversample_factor must be >= 1.0, got {oversample_factor!r}")
    if not (0.0 < float(max_near_duplicate_similarity) <= 1.0):
        raise ValueError(
            "max_near_duplicate_similarity must be in the interval (0.0, 1.0], "
            f"got {max_near_duplicate_similarity!r}"
        )
    if int(max_samples_per_template_bucket) <= 0:
        raise ValueError(
            "max_samples_per_template_bucket must be > 0, "
            f"got {max_samples_per_template_bucket!r}"
        )
    if int(max_generation_rounds) <= 0:
        raise ValueError(f"max_generation_rounds must be > 0, got {max_generation_rounds!r}")
    if int(max_total_raw_samples) <= 0:
        raise ValueError(f"max_total_raw_samples must be > 0, got {max_total_raw_samples!r}")


def _validate_curation_parameters(
    *,
    target_per_label: int | None,
    max_near_duplicate_similarity: float,
    max_samples_per_template_bucket: int,
) -> None:
    if target_per_label is not None and int(target_per_label) <= 0:
        raise ValueError(f"target_per_label must be > 0, got {target_per_label!r}")
    if not (0.0 < float(max_near_duplicate_similarity) <= 1.0):
        raise ValueError(
            "max_near_duplicate_similarity must be in the interval (0.0, 1.0], "
            f"got {max_near_duplicate_similarity!r}"
        )
    if int(max_samples_per_template_bucket) <= 0:
        raise ValueError(
            "max_samples_per_template_bucket must be > 0, "
            f"got {max_samples_per_template_bucket!r}"
        )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic JSONL corpus for the Twinr semantic router."
    )
    parser.add_argument("--output-path", required=True, help="Destination JSONL path")
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=1024,
        help="Number of kept samples per target label",
    )
    parser.add_argument("--seed", type=int, default=20260322, help="Deterministic generation seed")
    parser.add_argument(
        "--label-namespace",
        default=_LABEL_NAMESPACE_BACKEND,
        choices=(_LABEL_NAMESPACE_BACKEND, _LABEL_NAMESPACE_USER),
        help="Whether JSONL labels should target backend routes or user-centered intents",
    )
    parser.add_argument(
        "--oversample-factor",
        type=float,
        default=1.7,
        help="Raw generation multiplier before curation",
    )
    parser.add_argument(
        "--max-near-duplicate-similarity",
        type=float,
        default=0.92,
        help="Jaccard similarity threshold for near-duplicate rejection",
    )
    parser.add_argument(
        "--max-samples-per-template-bucket",
        type=int,
        default=40,
        help="Cap for one label/family/template bucket before style-collapse rejection",
    )
    parser.add_argument(
        "--split-strategy",
        default=_SPLIT_STRATEGY_GROUP,
        choices=(_SPLIT_STRATEGY_GROUP, _SPLIT_STRATEGY_LEGACY),
        help="How train/dev/test splits are assigned",
    )
    parser.add_argument(
        "--near-duplicate-scope",
        default=_NEAR_DUP_SCOPE_LABEL,
        choices=(_NEAR_DUP_SCOPE_LABEL, _NEAR_DUP_SCOPE_GLOBAL),
        help="Whether near-duplicate suppression is label-local or global",
    )
    parser.add_argument(
        "--max-generation-rounds",
        type=int,
        default=_MAX_GENERATION_ROUNDS,
        help="Maximum adaptive generation rounds before failing",
    )
    parser.add_argument(
        "--max-total-raw-samples",
        type=int,
        default=_MAX_TOTAL_RAW_SAMPLES,
        help="Hard cap for generated raw samples across all rounds",
    )
    parser.add_argument("--report-path", default=None, help="Optional JSON summary path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for synthetic router-dataset generation."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    samples, report = generate_synthetic_route_samples(
        samples_per_label=args.samples_per_label,
        seed=args.seed,
        oversample_factor=args.oversample_factor,
        max_near_duplicate_similarity=args.max_near_duplicate_similarity,
        max_samples_per_template_bucket=args.max_samples_per_template_bucket,
        label_namespace=args.label_namespace,
        split_strategy=args.split_strategy,
        near_duplicate_scope=args.near_duplicate_scope,
        max_generation_rounds=args.max_generation_rounds,
        max_total_raw_samples=args.max_total_raw_samples,
    )
    output_path = write_synthetic_route_samples_jsonl(
        samples,
        args.output_path,
        label_namespace=args.label_namespace,
    )
    if args.report_path:
        report_path = _prepare_output_path(args.report_path)
        _atomic_write_lines(
            report_path,
            (json.dumps(report.to_dict(), indent=2, sort_keys=True),),
        )
    print(json.dumps({"output_path": str(output_path), **report.to_dict()}, ensure_ascii=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via module CLI.
    raise SystemExit(main())
