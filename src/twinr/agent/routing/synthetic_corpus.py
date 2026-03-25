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
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import random
import unicodedata
from typing import Iterable, Mapping, Sequence

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


_REFERENCE_DATE = "2026-03-22"
_SPLIT_TRAIN_RATIO = 0.82
_SPLIT_DEV_RATIO = 0.91
_DEFAULT_SOURCE = "synthetic_router_v2_user_centered"
_LABEL_NAMESPACE_BACKEND = "backend"
_LABEL_NAMESPACE_USER = "user"
_GENERATION_LEAKAGE_MARKERS: tuple[str, ...] = (
    "label ",
    "klasse ",
    "route ",
    "routing",
    "router",
    "parametric",
    "web research",
    "memory class",
    "tool call",
    "intent ",
)

SyntheticRouteTemplate = _SyntheticRouteTemplate


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

    def to_json_dict(self, *, label_namespace: str = _LABEL_NAMESPACE_BACKEND) -> dict[str, str]:
        """Return one JSON-serializable payload for JSONL export."""

        payload = {
            "id": self.sample_id,
            "text": self.text,
            "label": self.label_for_namespace(label_namespace),
            "split": self.split,
            "source": self.source,
            "reference_date": self.reference_date,
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

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-friendly report payload."""

        return {
            "generated_count": int(self.generated_count),
            "kept_count": int(self.kept_count),
            "rejected_exact_duplicates": int(self.rejected_exact_duplicates),
            "rejected_near_duplicates": int(self.rejected_near_duplicates),
            "rejected_style_collapses": int(self.rejected_style_collapses),
            "rejected_generation_leakage": int(self.rejected_generation_leakage),
            "per_label": dict(self.per_label),
            "per_split": dict(self.per_split),
        }


def generate_synthetic_route_samples(
    *,
    samples_per_label: int = 1024,
    seed: int = 20260322,
    oversample_factor: float = 1.7,
    max_near_duplicate_similarity: float = 0.92,
    max_samples_per_template_bucket: int = 40,
    label_namespace: str = _LABEL_NAMESPACE_BACKEND,
) -> tuple[tuple[SyntheticRouteSample, ...], SyntheticRouteCurationReport]:
    """Generate one curated synthetic dataset for the local semantic router."""

    normalized_namespace = _normalize_label_namespace(label_namespace)
    label_values = _label_values_for_namespace(normalized_namespace)
    target_per_label = max(1, int(samples_per_label))
    raw_target_per_label = max(target_per_label, int(round(target_per_label * float(oversample_factor))))
    rng = random.Random(seed)
    recipes_by_label: dict[str, list[SyntheticRouteRecipe]] = defaultdict(list)
    for recipe in _ROUTE_RECIPES:
        recipes_by_label[_recipe_label_for_namespace(recipe, normalized_namespace)].append(recipe)
    raw_samples: list[SyntheticRouteSample] = []
    raw_counts: Counter[str] = Counter()
    recipe_cycles = {
        label: _shuffled_cycle(recipes_by_label[label], rng=rng)
        for label in label_values
    }
    max_attempts = raw_target_per_label * len(label_values) * 50
    attempts = 0
    while any(raw_counts[label] < raw_target_per_label for label in label_values):
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "Synthetic route generation exhausted its attempt budget before reaching the requested size."
            )
        for label in label_values:
            if raw_counts[label] >= raw_target_per_label:
                continue
            recipe = next(recipe_cycles[label])
            raw_samples.append(_render_recipe_sample(recipe, rng=rng))
            raw_counts[label] += 1
    curated, report = curate_synthetic_route_samples(
        raw_samples,
        target_per_label=target_per_label,
        max_near_duplicate_similarity=max_near_duplicate_similarity,
        max_samples_per_template_bucket=max_samples_per_template_bucket,
        label_namespace=normalized_namespace,
    )
    return curated, report


def generate_synthetic_user_intent_samples(
    *,
    samples_per_label: int = 1024,
    seed: int = 20260322,
    oversample_factor: float = 1.7,
    max_near_duplicate_similarity: float = 0.92,
    max_samples_per_template_bucket: int = 40,
) -> tuple[tuple[SyntheticRouteSample, ...], SyntheticRouteCurationReport]:
    """Generate one curated synthetic dataset labeled by user-centered intent."""

    return generate_synthetic_route_samples(
        samples_per_label=samples_per_label,
        seed=seed,
        oversample_factor=oversample_factor,
        max_near_duplicate_similarity=max_near_duplicate_similarity,
        max_samples_per_template_bucket=max_samples_per_template_bucket,
        label_namespace=_LABEL_NAMESPACE_USER,
    )


def curate_synthetic_route_samples(
    samples: Sequence[SyntheticRouteSample],
    *,
    target_per_label: int | None = None,
    max_near_duplicate_similarity: float = 0.92,
    max_samples_per_template_bucket: int = 40,
    label_namespace: str = _LABEL_NAMESPACE_BACKEND,
) -> tuple[tuple[SyntheticRouteSample, ...], SyntheticRouteCurationReport]:
    """Deduplicate and sanitize synthetic router samples."""

    normalized_namespace = _normalize_label_namespace(label_namespace)
    label_values = _label_values_for_namespace(normalized_namespace)
    requested_per_label = None if target_per_label is None else max(1, int(target_per_label))
    kept: list[SyntheticRouteSample] = []
    exact_seen: set[str] = set()
    shape_counts: Counter[tuple[str, str | None, str | None]] = Counter()
    per_label_counts: Counter[str] = Counter()
    accepted_signatures: dict[tuple[str, str | None], list[tuple[frozenset[str], frozenset[str]]]] = defaultdict(list)
    rejected_exact_duplicates = 0
    rejected_near_duplicates = 0
    rejected_style_collapses = 0
    rejected_generation_leakage = 0
    for sample in samples:
        sample_label = sample.label_for_namespace(normalized_namespace)
        if requested_per_label is not None and per_label_counts[sample_label] >= requested_per_label:
            continue
        normalized_text = _normalize_for_dedupe(sample.text)
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
        family_signatures = accepted_signatures[(sample_label, sample.family_key)]
        if any(
            max(
                _jaccard_similarity(token_signature, existing_tokens),
                _jaccard_similarity(trigram_signature, existing_trigrams),
            ) >= max_near_duplicate_similarity
            for existing_tokens, existing_trigrams in family_signatures
        ):
            rejected_near_duplicates += 1
            continue
        exact_seen.add(normalized_text)
        shape_counts[bucket_key] += 1
        per_label_counts[sample_label] += 1
        family_signatures.append((token_signature, trigram_signature))
        kept.append(sample)
        if requested_per_label is not None and all(per_label_counts[label] >= requested_per_label for label in label_values):
            break
    if requested_per_label is not None and any(per_label_counts[label] < requested_per_label for label in label_values):
        raise RuntimeError(
            "Synthetic route curation could not keep enough samples for every label. "
            f"Counts={dict(per_label_counts)} requested_per_label={requested_per_label}"
        )
    split_counts = Counter(sample.split for sample in kept)
    report = SyntheticRouteCurationReport(
        generated_count=len(samples),
        kept_count=len(kept),
        rejected_exact_duplicates=rejected_exact_duplicates,
        rejected_near_duplicates=rejected_near_duplicates,
        rejected_style_collapses=rejected_style_collapses,
        rejected_generation_leakage=rejected_generation_leakage,
        per_label={label: int(per_label_counts.get(label, 0)) for label in label_values},
        per_split={split: int(split_counts.get(split, 0)) for split in ("train", "dev", "test")},
    )
    return tuple(kept), report


def write_synthetic_route_samples_jsonl(
    samples: Iterable[SyntheticRouteSample],
    path: str | Path,
    *,
    label_namespace: str = _LABEL_NAMESPACE_BACKEND,
) -> Path:
    """Write synthetic route samples to JSONL on disk."""

    output_path = Path(path).expanduser().resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(sample.to_json_dict(label_namespace=label_namespace), ensure_ascii=True)
        for sample in samples
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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


def _render_recipe_sample(
    recipe: SyntheticRouteRecipe,
    *,
    rng: random.Random,
) -> SyntheticRouteSample:
    template = rng.choice(recipe.templates)
    slot_payload = {
        name: rng.choice(tuple(values))
        for name, values in dict(recipe.slot_values).items()
    }
    rendered = " ".join(template.text.format(**slot_payload).split())
    noise_key = rng.choice(recipe.noise_pool)
    noisy_text = _apply_noise_profile(rendered, noise_key=noise_key, rng=rng)
    normalized_text = " ".join(noisy_text.split())
    sample_key = "|".join(
        (
            recipe.label,
            recipe.family_key,
            template.key,
            recipe.difficulty,
            noise_key,
            normalized_text,
        )
    )
    sample_id = hashlib.sha1(sample_key.encode("utf-8")).hexdigest()[:16]
    split = _hash_to_split(sample_id)
    return SyntheticRouteSample(
        text=normalized_text,
        label=recipe.label,
        sample_id=f"{recipe.label}_{sample_id}",
        split=split,
        user_label=recipe.user_label,
        family_key=recipe.family_key,
        template_key=template.key,
        difficulty=recipe.difficulty,
        noise_key=noise_key,
    )


def _normalize_label_namespace(value: object) -> str:
    """Return one validated synthetic-label namespace identifier."""

    normalized = str(value or _LABEL_NAMESPACE_BACKEND).strip().lower()
    if normalized not in {_LABEL_NAMESPACE_BACKEND, _LABEL_NAMESPACE_USER}:
        raise ValueError(f"Unsupported synthetic label namespace: {value!r}")
    return normalized


def _label_values_for_namespace(namespace: str) -> tuple[str, ...]:
    """Return the target label set for one synthetic export namespace."""

    normalized_namespace = _normalize_label_namespace(namespace)
    if normalized_namespace == _LABEL_NAMESPACE_USER:
        return USER_INTENT_LABEL_VALUES
    return ROUTE_LABEL_VALUES


def _recipe_label_for_namespace(recipe: SyntheticRouteRecipe, namespace: str) -> str:
    """Return the recipe label for one synthetic export namespace."""

    normalized_namespace = _normalize_label_namespace(namespace)
    if normalized_namespace == _LABEL_NAMESPACE_USER:
        return str(recipe.user_label)
    return recipe.label


def _shuffled_cycle(
    items: Sequence[SyntheticRouteRecipe],
    *,
    rng: random.Random,
):
    """Yield items in repeatedly shuffled order to avoid family collapse."""

    pool = list(items)
    if not pool:
        raise ValueError("Synthetic recipe pools must not be empty.")
    shuffled: list[SyntheticRouteRecipe] = []
    while True:
        if not shuffled:
            shuffled = list(pool)
            rng.shuffle(shuffled)
        yield shuffled.pop()


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
        return _strip_simple_punctuation(text).lower()
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
        return _strip_simple_punctuation(flattened).lower()
    if normalized_key == "filler":
        prefix = rng.choice(("aeh ", "hm ", "also ", "sag mal "))
        return _strip_simple_punctuation(f"{prefix}{text}").lower()
    raise ValueError(f"Unsupported synthetic transcript noise profile: {noise_key!r}")


def _hash_to_split(sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).digest()[0]
    ratio = float(digest) / 255.0
    if ratio < _SPLIT_TRAIN_RATIO:
        return "train"
    if ratio < _SPLIT_DEV_RATIO:
        return "dev"
    return "test"


def _looks_like_generation_leakage(normalized_text: str) -> bool:
    return any(marker in normalized_text for marker in _GENERATION_LEAKAGE_MARKERS)


def _normalize_for_dedupe(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    cleaned = "".join(character if character.isalnum() or character.isspace() else " " for character in normalized)
    return " ".join(cleaned.split())


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


def _strip_simple_punctuation(text: str) -> str:
    translation_table: dict[int, str | int | None] = {
        ord(character): " "
        for character in ".,!?;:\"'"
    }
    return text.translate(str.maketrans(translation_table))


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic JSONL corpus for the Twinr semantic router.")
    parser.add_argument("--output-path", required=True, help="Destination JSONL path")
    parser.add_argument("--samples-per-label", type=int, default=1024, help="Number of kept samples per route label")
    parser.add_argument("--seed", type=int, default=20260322, help="Deterministic generation seed")
    parser.add_argument(
        "--label-namespace",
        default=_LABEL_NAMESPACE_BACKEND,
        choices=(_LABEL_NAMESPACE_BACKEND, _LABEL_NAMESPACE_USER),
        help="Whether JSONL labels should target backend routes or user-centered intents",
    )
    parser.add_argument("--oversample-factor", type=float, default=1.7, help="Raw generation multiplier before curation")
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
    )
    output_path = write_synthetic_route_samples_jsonl(
        samples,
        args.output_path,
        label_namespace=args.label_namespace,
    )
    if args.report_path:
        report_path = Path(args.report_path).expanduser().resolve(strict=False)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), **report.to_dict()}, ensure_ascii=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via module CLI.
    raise SystemExit(main())
