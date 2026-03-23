from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import twinr.agent.routing.synthetic_corpus as synthetic_corpus_module

from twinr.agent.routing import (
    LabeledRouteSample,
    LocalSemanticRouter,
    LocalUserIntentRouter,
    OnnxSentenceEncoder,
    ScoredRouteRecord,
    ScoredUserIntentRecord,
    SemanticRouteDecision,
    SyntheticRouteSample,
    TwoStageLocalSemanticRouter,
    UserIntentDecision,
    evaluate_user_intent_records,
    curate_synthetic_route_samples,
    evaluate_route_records,
    generate_synthetic_route_samples,
    generate_synthetic_user_intent_samples,
    load_labeled_route_samples,
    load_semantic_router_bundle,
    load_user_intent_bundle,
    train_router_bundle_from_jsonl,
    train_user_intent_bundle_from_jsonl,
    write_synthetic_route_samples_jsonl,
    write_synthetic_user_intent_samples_jsonl,
)


class FakeEncoder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = {
            text: np.asarray(vector, dtype=np.float32)
            for text, vector in vectors.items()
        }

    def encode(self, texts):
        return np.vstack([self._vectors[text] for text in texts]).astype(np.float32)


def _write_bundle(
    root_dir: Path,
    *,
    metadata_overrides: dict[str, object] | None = None,
    centroids: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    bias: np.ndarray | None = None,
) -> Path:
    bundle_dir = root_dir / "router_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "model.onnx").write_bytes(b"fake-onnx")
    (bundle_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "classifier_type": "embedding_centroid_v1",
        "labels": ["parametric", "web", "memory", "tool"],
        "model_id": "test-router",
        "max_length": 64,
        "pooling": "mean",
        "temperature": 0.2,
        "thresholds": {
            "parametric": 1.0,
            "web": 0.45,
            "memory": 0.45,
            "tool": 0.45,
        },
        "authoritative_labels": ["web", "memory", "tool"],
        "min_margin": 0.05,
        "normalize_embeddings": True,
        "normalize_centroids": True,
        "reference_date": "2026-03-22",
    }
    metadata.update(metadata_overrides or {})
    if metadata["classifier_type"] == "embedding_linear_softmax_v1":
        np.save(bundle_dir / "weights.npy", weights if weights is not None else np.eye(4, dtype=np.float32))
        np.save(bundle_dir / "bias.npy", bias if bias is not None else np.zeros(4, dtype=np.float32))
    else:
        np.save(bundle_dir / "centroids.npy", centroids if centroids is not None else np.eye(4, dtype=np.float32))
    (bundle_dir / "router_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return bundle_dir


def _write_user_intent_bundle(
    root_dir: Path,
    *,
    metadata_overrides: dict[str, object] | None = None,
    centroids: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    bias: np.ndarray | None = None,
) -> Path:
    bundle_dir = root_dir / "user_intent_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "model.onnx").write_bytes(b"fake-onnx")
    (bundle_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "classifier_type": "embedding_centroid_v1",
        "labels": ["wissen", "nachschauen", "persoenlich", "machen_oder_pruefen"],
        "model_id": "test-user-intent-router",
        "max_length": 64,
        "pooling": "mean",
        "temperature": 0.2,
        "normalize_embeddings": True,
        "normalize_centroids": True,
        "reference_date": "2026-03-22",
    }
    metadata.update(metadata_overrides or {})
    if metadata["classifier_type"] == "embedding_linear_softmax_v1":
        np.save(bundle_dir / "weights.npy", weights if weights is not None else np.eye(4, dtype=np.float32))
        np.save(bundle_dir / "bias.npy", bias if bias is not None else np.zeros(4, dtype=np.float32))
    else:
        np.save(bundle_dir / "centroids.npy", centroids if centroids is not None else np.eye(4, dtype=np.float32))
    (bundle_dir / "user_intent_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return bundle_dir


class SemanticRouterTests(unittest.TestCase):
    def test_load_bundle_rejects_missing_route_label(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bundle_dir = _write_bundle(
                Path(temp_dir),
                metadata_overrides={"labels": ["parametric", "web", "memory"]},
            )

            with self.assertRaises(ValueError):
                load_semantic_router_bundle(bundle_dir)

    def test_local_semantic_router_applies_authority_policy(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bundle = load_semantic_router_bundle(_write_bundle(Path(temp_dir)))
            router = LocalSemanticRouter(
                bundle,
                encoder=FakeEncoder(
                    {
                        "fresh": [0.1, 5.0, 0.0, 0.0],
                        "ambiguous": [0.99, 1.0, 0.0, 0.0],
                        "stable": [5.0, 0.1, 0.0, 0.0],
                    }
                ),
            )

            fresh = router.classify("fresh")
            ambiguous = router.classify("ambiguous")
            stable = router.classify("stable")

        self.assertEqual(fresh.label, "web")
        self.assertTrue(fresh.authoritative)
        self.assertIsNone(fresh.fallback_reason)
        self.assertGreater(fresh.confidence, 0.9)

        self.assertEqual(ambiguous.label, "web")
        self.assertFalse(ambiguous.authoritative)
        self.assertEqual(ambiguous.fallback_reason, "below_margin_threshold")

        self.assertEqual(stable.label, "parametric")
        self.assertFalse(stable.authoritative)
        self.assertEqual(stable.fallback_reason, "label_not_authoritative")

    def test_local_semantic_router_respects_allowed_labels_without_inflating_confidence(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bundle = load_semantic_router_bundle(_write_bundle(Path(temp_dir)))
            router = LocalSemanticRouter(
                bundle,
                encoder=FakeEncoder(
                    {
                        "personal_calendar": [0.0, 4.4, 4.2, 4.0],
                    }
                ),
            )

            constrained = router.classify(
                "personal_calendar",
                allowed_labels=("memory", "tool"),
            )

        self.assertEqual(constrained.label, "memory")
        self.assertLess(constrained.confidence, 0.6)
        self.assertLess(constrained.margin, 0.0)
        self.assertEqual(constrained.scores["web"], 0.0)
        self.assertGreater(constrained.scores["memory"], 0.0)
        self.assertGreater(constrained.scores["tool"], 0.0)
        self.assertFalse(constrained.authoritative)
        self.assertEqual(constrained.fallback_reason, "below_confidence_threshold")

    def test_local_semantic_router_supports_linear_head_bundles(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bundle = load_semantic_router_bundle(
                _write_bundle(
                    Path(temp_dir),
                    metadata_overrides={
                        "classifier_type": "embedding_linear_softmax_v1",
                        "temperature": 1.0,
                    },
                    weights=np.asarray(
                        [
                            [3.0, 0.0, 0.0, 0.0],
                            [0.0, 3.0, 0.0, 0.0],
                            [0.0, 0.0, 3.0, 0.0],
                            [0.0, 0.0, 0.0, 3.0],
                        ],
                        dtype=np.float32,
                    ),
                    bias=np.zeros(4, dtype=np.float32),
                )
            )
            router = LocalSemanticRouter(
                bundle,
                encoder=FakeEncoder(
                    {
                        "calendar": [0.0, 0.0, 5.0, 0.0],
                    }
                ),
            )

            decision = router.classify("calendar")

        self.assertEqual(decision.label, "memory")
        self.assertGreater(decision.scores["memory"], decision.scores["tool"])

    def test_evaluate_route_records_reports_authority_metrics(self) -> None:
        records = (
            ScoredRouteRecord(
                sample=LabeledRouteSample(text="A", label="web"),
                decision=SemanticRouteDecision(
                    label="web",
                    confidence=0.91,
                    margin=0.4,
                    scores={"parametric": 0.03, "web": 0.91, "memory": 0.03, "tool": 0.03},
                    model_id="router",
                    latency_ms=4.0,
                    authoritative=True,
                ),
            ),
            ScoredRouteRecord(
                sample=LabeledRouteSample(text="B", label="memory"),
                decision=SemanticRouteDecision(
                    label="memory",
                    confidence=0.44,
                    margin=0.03,
                    scores={"parametric": 0.08, "web": 0.2, "memory": 0.44, "tool": 0.28},
                    model_id="router",
                    latency_ms=4.0,
                    authoritative=False,
                    fallback_reason="below_confidence_threshold",
                ),
            ),
            ScoredRouteRecord(
                sample=LabeledRouteSample(text="C", label="tool"),
                decision=SemanticRouteDecision(
                    label="memory",
                    confidence=0.73,
                    margin=0.2,
                    scores={"parametric": 0.06, "web": 0.08, "memory": 0.73, "tool": 0.13},
                    model_id="router",
                    latency_ms=4.0,
                    authoritative=True,
                ),
            ),
        )

        evaluation = evaluate_route_records(records)

        self.assertEqual(evaluation.total, 3)
        self.assertAlmostEqual(evaluation.accuracy, 2.0 / 3.0)
        self.assertAlmostEqual(evaluation.fallback_rate, 1.0 / 3.0)
        self.assertAlmostEqual(evaluation.authoritative_rate, 2.0 / 3.0)
        self.assertAlmostEqual(evaluation.unsafe_authoritative_error_rate, 1.0 / 3.0)
        self.assertEqual(evaluation.confusion_matrix["memory"]["memory"], 1)
        self.assertEqual(evaluation.confusion_matrix["tool"]["memory"], 1)

    def test_onnx_sentence_encoder_initializes_lazy_state_with_slots(self) -> None:
        encoder = OnnxSentenceEncoder(
            model_path=Path("model.onnx"),
            tokenizer_path=Path("tokenizer.json"),
            max_length=64,
        )

        self.assertIsNone(encoder._tokenizer)
        self.assertIsNone(encoder._session)
        self.assertIsNone(encoder._session_input_names)

    def test_local_user_intent_router_supports_linear_head_bundles(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bundle = load_user_intent_bundle(
                _write_user_intent_bundle(
                    Path(temp_dir),
                    metadata_overrides={
                        "classifier_type": "embedding_linear_softmax_v1",
                        "temperature": 1.0,
                    },
                    weights=np.asarray(
                        [
                            [3.0, 0.0, 0.0, 0.0],
                            [0.0, 3.0, 0.0, 0.0],
                            [0.0, 0.0, 3.0, 0.0],
                            [0.0, 0.0, 0.0, 3.0],
                        ],
                        dtype=np.float32,
                    ),
                    bias=np.zeros(4, dtype=np.float32),
                )
            )
            router = LocalUserIntentRouter(
                bundle,
                encoder=FakeEncoder(
                    {
                        "private": [0.0, 0.0, 5.0, 0.0],
                    }
                ),
            )

            decision = router.classify("private")

        self.assertEqual(decision.label, "persoenlich")
        self.assertGreater(decision.scores["persoenlich"], decision.scores["wissen"])

    def test_generate_synthetic_route_samples_balanced_and_exportable(self) -> None:
        samples, report = generate_synthetic_route_samples(samples_per_label=6, seed=9)

        self.assertEqual(len(samples), 24)
        self.assertEqual(
            Counter(sample.label for sample in samples),
            Counter({"parametric": 6, "web": 6, "memory": 6, "tool": 6}),
        )
        self.assertEqual(report.kept_count, 24)
        self.assertGreater(report.per_split["train"], 0)

        with TemporaryDirectory() as temp_dir:
            dataset_path = write_synthetic_route_samples_jsonl(samples, Path(temp_dir) / "synthetic.jsonl")
            loaded = load_labeled_route_samples(dataset_path)

        self.assertEqual(len(loaded), len(samples))
        self.assertEqual(Counter(sample.label for sample in loaded), Counter(sample.label for sample in samples))

    def test_generate_synthetic_user_intent_samples_balanced_and_exportable(self) -> None:
        samples, report = generate_synthetic_user_intent_samples(samples_per_label=6, seed=9)

        self.assertEqual(len(samples), 24)
        self.assertEqual(
            Counter(str(sample.user_label) for sample in samples),
            Counter(
                {
                    "wissen": 6,
                    "nachschauen": 6,
                    "persoenlich": 6,
                    "machen_oder_pruefen": 6,
                }
            ),
        )
        self.assertEqual(report.kept_count, 24)

        with TemporaryDirectory() as temp_dir:
            dataset_path = write_synthetic_user_intent_samples_jsonl(
                samples,
                Path(temp_dir) / "synthetic_user_intent.jsonl",
            )
            payloads = [
                json.loads(line)
                for line in dataset_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(Counter(payload["label"] for payload in payloads)["persoenlich"], 6)
        self.assertIn("backend_label", payloads[0])
        self.assertIn("user_label", payloads[0])

    def test_boundary_recipe_registry_keeps_user_centered_confusion_families(self) -> None:
        recipes = {
            recipe.family_key: (recipe.label, str(recipe.user_label))
            for recipe in synthetic_corpus_module._ROUTE_RECIPES
        }

        self.assertEqual(recipes["stable_short_explanation"], ("parametric", "wissen"))
        self.assertEqual(recipes["stable_why_questions"], ("parametric", "wissen"))
        self.assertEqual(recipes["stable_usage_questions"], ("parametric", "wissen"))
        self.assertEqual(recipes["current_role_holder"], ("web", "nachschauen"))
        self.assertEqual(recipes["mutable_role_plain_question"], ("web", "nachschauen"))
        self.assertEqual(recipes["short_known_people_preferences"], ("memory", "persoenlich"))
        self.assertEqual(recipes["person_reason_memory"], ("memory", "persoenlich"))
        self.assertEqual(recipes["personal_day_overview"], ("tool", "persoenlich"))
        self.assertEqual(recipes["personal_short_schedule_status"], ("tool", "persoenlich"))
        self.assertEqual(recipes["care_support_visits"], ("tool", "persoenlich"))
        self.assertEqual(recipes["home_environment_state"], ("tool", "machen_oder_pruefen"))
        self.assertEqual(recipes["doorbell_and_presence_check"], ("tool", "machen_oder_pruefen"))

    def test_dual_label_corpus_keeps_tool_backed_personal_examples(self) -> None:
        samples, _report = generate_synthetic_route_samples(samples_per_label=64, seed=20260322)
        family_pairs = {
            (sample.family_key, sample.label, str(sample.user_label))
            for sample in samples
        }

        self.assertIn(("personal_day_overview", "tool", "persoenlich"), family_pairs)
        self.assertIn(("care_support_visits", "tool", "persoenlich"), family_pairs)
        self.assertIn(("personal_short_schedule_status", "tool", "persoenlich"), family_pairs)

    def test_generate_synthetic_route_samples_spreads_across_boundary_families(self) -> None:
        samples, _report = generate_synthetic_route_samples(samples_per_label=96, seed=20260322)
        families_by_label: dict[str, set[str]] = {}
        for sample in samples:
            families_by_label.setdefault(sample.label, set()).add(str(sample.family_key))

        self.assertGreaterEqual(len(families_by_label["parametric"]), 8)
        self.assertGreaterEqual(len(families_by_label["web"]), 8)
        self.assertGreaterEqual(len(families_by_label["memory"]), 10)
        self.assertGreaterEqual(len(families_by_label["tool"]), 10)

    def test_curate_synthetic_route_samples_rejects_duplicates_leakage_and_style_collapse(self) -> None:
        samples = (
            SyntheticRouteSample(
                text="Wie funktioniert Photosynthese?",
                label="parametric",
                sample_id="p1",
                split="train",
                family_key="stable_explanation",
                template_key="plain",
            ),
            SyntheticRouteSample(
                text="Wie funktioniert Photosynthese?",
                label="parametric",
                sample_id="p2",
                split="train",
                family_key="stable_explanation",
                template_key="plain",
            ),
            SyntheticRouteSample(
                text="Wie funktioniert Photosynthese bitte",
                label="parametric",
                sample_id="p3",
                split="train",
                family_key="stable_explanation",
                template_key="spoken",
            ),
            SyntheticRouteSample(
                text="label: web heute berlin",
                label="web",
                sample_id="w1",
                split="train",
                family_key="current_news",
                template_key="plain",
            ),
            SyntheticRouteSample(
                text="Druck meine Einkaufsliste.",
                label="tool",
                sample_id="t1",
                split="train",
                family_key="print_and_write",
                template_key="print",
            ),
            SyntheticRouteSample(
                text="Druck den Arztbrief.",
                label="tool",
                sample_id="t2",
                split="train",
                family_key="print_and_write",
                template_key="print",
            ),
        )

        curated, report = curate_synthetic_route_samples(
            samples,
            max_near_duplicate_similarity=0.7,
            max_samples_per_template_bucket=1,
        )

        self.assertEqual([sample.sample_id for sample in curated], ["p1", "t1"])
        self.assertEqual(report.rejected_exact_duplicates, 1)
        self.assertEqual(report.rejected_near_duplicates, 1)
        self.assertEqual(report.rejected_generation_leakage, 1)
        self.assertEqual(report.rejected_style_collapses, 1)

    def test_train_router_bundle_from_jsonl_supports_injected_training_dependencies(self) -> None:
        samples = (
            SyntheticRouteSample(
                text="Erklaer mir Photosynthese.",
                label="parametric",
                sample_id="parametric_train",
                split="train",
                family_key="stable_explanation",
                template_key="plain",
            ),
            SyntheticRouteSample(
                text="Wie wird das Wetter heute in Berlin?",
                label="web",
                sample_id="web_train",
                split="train",
                family_key="current_weather",
                template_key="forecast",
            ),
            SyntheticRouteSample(
                text="Welche Allergien habe ich?",
                label="memory",
                sample_id="memory_dev",
                split="dev",
                family_key="personal_health_notes",
                template_key="allergies",
            ),
            SyntheticRouteSample(
                text="Stell einen Timer auf 10 Minuten.",
                label="tool",
                sample_id="tool_test",
                split="test",
                family_key="timers_and_reminders",
                template_key="timer",
            ),
        )

        def fake_builder(**kwargs) -> Path:
            output_dir = Path(kwargs["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir

        def fake_scorer(_router, scored_samples):
            records = []
            for sample in scored_samples:
                scores = {label: 0.02 for label in ("parametric", "web", "memory", "tool")}
                scores[sample.label] = 0.94
                records.append(
                    ScoredRouteRecord(
                        sample=sample,
                        decision=SemanticRouteDecision(
                            label=sample.label,
                            confidence=0.94,
                            margin=0.9,
                            scores=scores,
                            model_id="fake-router",
                            latency_ms=1.0,
                            authoritative=sample.label != "parametric",
                            fallback_reason=None if sample.label != "parametric" else "label_not_authoritative",
                        ),
                    )
                )
            return tuple(records)

        with TemporaryDirectory() as temp_dir:
            dataset_path = write_synthetic_route_samples_jsonl(samples, Path(temp_dir) / "dataset.jsonl")
            report_path = Path(temp_dir) / "training_report.json"
            report = train_router_bundle_from_jsonl(
                source_dir=Path(temp_dir) / "source_model",
                dataset_path=dataset_path,
                output_dir=Path(temp_dir) / "bundle",
                report_path=report_path,
                model_id="synthetic-test-router",
                bundle_builder=fake_builder,
                bundle_loader=lambda _path: object(),
                router_factory=lambda _bundle: object(),
                router_scorer=fake_scorer,
                record_evaluator=evaluate_route_records,
                dataset_summary={"generated_count": 4, "kept_count": 4},
            )
            persisted = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report.total_samples, 4)
        self.assertEqual(report.split_counts, {"train": 2, "dev": 1, "test": 1})
        self.assertEqual(report.model_id, "synthetic-test-router")
        self.assertEqual(report.evaluations["train"]["accuracy"], 1.0)
        self.assertEqual(report.evaluations["dev"]["accuracy"], 1.0)
        self.assertEqual(report.evaluations["test"]["accuracy"], 1.0)
        self.assertEqual(persisted["dataset_summary"]["kept_count"], 4)

    def test_train_user_intent_bundle_from_jsonl_supports_injected_training_dependencies(self) -> None:
        samples = (
            SyntheticRouteSample(
                text="Erklaer mir Photosynthese.",
                label="parametric",
                user_label="wissen",
                sample_id="wissen_train",
                split="train",
                family_key="stable_explanation",
                template_key="plain",
            ),
            SyntheticRouteSample(
                text="Wie wird das Wetter heute in Berlin?",
                label="web",
                user_label="nachschauen",
                sample_id="nachschauen_train",
                split="train",
                family_key="current_weather",
                template_key="forecast",
            ),
            SyntheticRouteSample(
                text="Was mag Anna besonders gern?",
                label="memory",
                user_label="persoenlich",
                sample_id="persoenlich_dev",
                split="dev",
                family_key="known_people_preferences",
                template_key="likes",
            ),
            SyntheticRouteSample(
                text="Schalte bitte das Licht ein.",
                label="tool",
                user_label="machen_oder_pruefen",
                sample_id="machen_test",
                split="test",
                family_key="device_control",
                template_key="play",
            ),
        )

        def fake_builder(**kwargs) -> Path:
            output_dir = Path(kwargs["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir

        def fake_scorer(_router, scored_samples):
            label_to_scores = {
                "wissen": {
                    "wissen": 0.94,
                    "nachschauen": 0.02,
                    "persoenlich": 0.02,
                    "machen_oder_pruefen": 0.02,
                },
                "nachschauen": {
                    "wissen": 0.02,
                    "nachschauen": 0.94,
                    "persoenlich": 0.02,
                    "machen_oder_pruefen": 0.02,
                },
                "persoenlich": {
                    "wissen": 0.02,
                    "nachschauen": 0.02,
                    "persoenlich": 0.94,
                    "machen_oder_pruefen": 0.02,
                },
                "machen_oder_pruefen": {
                    "wissen": 0.02,
                    "nachschauen": 0.02,
                    "persoenlich": 0.02,
                    "machen_oder_pruefen": 0.94,
                },
            }
            return tuple(
                ScoredUserIntentRecord(
                    sample=sample,
                    decision=UserIntentDecision(
                        label=sample.label,
                        confidence=0.94,
                        margin=0.92,
                        scores=label_to_scores[sample.label],
                        model_id="fake-user-router",
                        latency_ms=1.0,
                    ),
                )
                for sample in scored_samples
            )

        with TemporaryDirectory() as temp_dir:
            dataset_path = write_synthetic_user_intent_samples_jsonl(
                samples,
                Path(temp_dir) / "dataset.jsonl",
            )
            report_path = Path(temp_dir) / "training_report.json"
            report = train_user_intent_bundle_from_jsonl(
                source_dir=Path(temp_dir) / "source_model",
                dataset_path=dataset_path,
                output_dir=Path(temp_dir) / "bundle",
                report_path=report_path,
                model_id="synthetic-user-intent-router",
                bundle_builder=fake_builder,
                bundle_loader=lambda _path: object(),
                router_factory=lambda _bundle: object(),
                router_scorer=fake_scorer,
                record_evaluator=evaluate_user_intent_records,
                dataset_summary={"generated_count": 4, "kept_count": 4},
            )
            persisted = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report.total_samples, 4)
        self.assertEqual(report.split_counts, {"train": 2, "dev": 1, "test": 1})
        self.assertEqual(report.model_id, "synthetic-user-intent-router")
        self.assertEqual(report.evaluations["train"]["accuracy"], 1.0)
        self.assertEqual(report.evaluations["dev"]["accuracy"], 1.0)
        self.assertEqual(report.evaluations["test"]["accuracy"], 1.0)
        self.assertEqual(persisted["dataset_summary"]["kept_count"], 4)

    def test_two_stage_router_returns_user_intent_and_constrained_backend_route(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            user_bundle = load_user_intent_bundle(
                _write_user_intent_bundle(
                    root,
                    centroids=np.asarray(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    ),
                )
            )
            route_bundle = load_semantic_router_bundle(
                _write_bundle(
                    root,
                    centroids=np.asarray(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    ),
                )
            )
            shared_encoder = FakeEncoder(
                {
                    "appointments": [0.0, 0.1, 4.4, 4.1],
                }
            )
            router = TwoStageLocalSemanticRouter(
                user_bundle,
                route_bundle,
                user_intent_router=LocalUserIntentRouter(user_bundle, encoder=shared_encoder),
                route_router=LocalSemanticRouter(route_bundle, encoder=shared_encoder),
            )

            decision = router.classify("appointments")

        self.assertEqual(decision.user_intent.label, "persoenlich")
        self.assertEqual(decision.allowed_route_labels, ("memory", "tool"))
        self.assertEqual(decision.route_decision.label, "memory")
        self.assertEqual(decision.route_decision.scores["web"], 0.0)
        self.assertGreater(
            decision.route_decision.scores["memory"],
            decision.route_decision.scores["tool"],
        )


if __name__ == "__main__":
    unittest.main()
