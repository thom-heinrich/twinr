from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import twinr.agent.routing.bootstrap as router_bootstrap_module
import twinr.agent.routing.inference as inference_module
import twinr.agent.routing.synthetic_corpus as synthetic_corpus_module
import twinr.agent.routing.user_intent_bootstrap as user_intent_bootstrap_module

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


class WarmupEncoder(FakeEncoder):
    def __init__(self) -> None:
        super().__init__({})
        self.calls: list[str] = []

    def warmup(self, probe_text: str = "warmup") -> None:
        self.calls.append(str(probe_text))


def _write_test_onnx_model(
    path: Path,
    *,
    max_length: int = 64,
    embedding_dim: int = 4,
) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    input_ids = helper.make_tensor_value_info(
        "input_ids",
        TensorProto.INT64,
        [None, max_length],
    )
    sentence_embedding = helper.make_tensor_value_info(
        "sentence_embedding",
        TensorProto.FLOAT,
        [None, embedding_dim],
    )
    projection = np.zeros((max_length, embedding_dim), dtype=np.float32)
    for index in range(min(max_length, embedding_dim)):
        projection[index, index] = 1.0
    graph = helper.make_graph(
        [
            helper.make_node(
                "Cast",
                inputs=["input_ids"],
                outputs=["input_ids_float"],
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "MatMul",
                inputs=["input_ids_float", "projection"],
                outputs=["sentence_embedding"],
            ),
        ],
        "twinr_test_sentence_encoder",
        [input_ids],
        [sentence_embedding],
        initializer=[numpy_helper.from_array(projection, name="projection")],
    )
    model = helper.make_model(
        graph,
        producer_name="twinr-tests",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _write_test_tokenizer(path: Path, *, max_length: int = 64) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(
        WordLevel(
            vocab={"[UNK]": 0, "[PAD]": 1, "hallo": 2, "termine": 3},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(direction="right", pad_id=1, pad_token="[PAD]")
    tokenizer.save(str(path))


def _write_test_model_artifacts(root_dir: Path, *, max_length: int = 64) -> None:
    _write_test_onnx_model(root_dir / "model.onnx", max_length=max_length)
    _write_test_tokenizer(root_dir / "tokenizer.json", max_length=max_length)


def _write_test_source_model(root_dir: Path, *, max_length: int = 64) -> Path:
    source_dir = root_dir / "source_model"
    source_dir.mkdir(parents=True, exist_ok=True)
    _write_test_model_artifacts(source_dir, max_length=max_length)
    return source_dir


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
    _write_test_model_artifacts(bundle_dir, max_length=int(metadata["max_length"]))
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
    _write_test_model_artifacts(bundle_dir, max_length=int(metadata["max_length"]))
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

    def test_load_bundle_accepts_onnxruntime_compatible_model_when_checker_rejects(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bundle_dir = _write_bundle(Path(temp_dir))
            with mock.patch(
                "onnx.checker.check_model",
                side_effect=Exception("No Op registered for LayerNormalization"),
            ):
                bundle = load_semantic_router_bundle(bundle_dir)

        self.assertEqual(bundle.metadata.model_id, "test-router")

    def test_load_user_intent_bundle_accepts_onnxruntime_compatible_model_when_checker_rejects(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bundle_dir = _write_user_intent_bundle(Path(temp_dir))
            with mock.patch(
                "onnx.checker.check_model",
                side_effect=Exception("No Op registered for LayerNormalization"),
            ):
                bundle = load_user_intent_bundle(bundle_dir)

        self.assertEqual(bundle.metadata.model_id, "test-user-intent-router")

    def test_router_bootstrap_materializes_ort_sidecar_for_runtime_bundle(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = _write_test_source_model(root)
            bundle_dir = root / "bundle"
            bundle_dir.mkdir(parents=True, exist_ok=True)

            report = router_bootstrap_module._materialize_model_artifact(
                source_model_path=source_dir / "model.onnx",
                source_tokenizer_path=source_dir / "tokenizer.json",
                destination_dir=bundle_dir,
                max_length=64,
                pooling="mean",
                output_name=None,
                probe_texts=("hallo",),
                optimize_onnx=False,
                quantize_onnx="none",
                min_probe_cosine=0.0,
                encode_batch_size=1,
            )
            self.assertTrue((bundle_dir / "model.ort").is_file())
            self.assertTrue(report["ort_sidecar_created"])

    def test_user_intent_bootstrap_materializes_ort_sidecar_for_runtime_bundle(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = _write_test_source_model(root)
            stage_dir = root / "bundle"
            stage_dir.mkdir(parents=True, exist_ok=True)

            prepared = user_intent_bootstrap_module._prepare_model_artifacts(
                source_dir=source_dir,
                stage_dir=stage_dir,
                quantize_mode="off",
                optimize_model=False,
            )
            self.assertTrue((stage_dir / "model.ort").is_file())
            self.assertIn(stage_dir / "model.ort", prepared.files)

    def test_user_intent_bundle_reuses_validation_session_for_first_encoder_load(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bundle_dir = _write_user_intent_bundle(Path(temp_dir))
            (bundle_dir / "model.ort").write_bytes(b"ort-placeholder")
            inference_module._PRELOADED_ORT_SESSIONS.clear()
            call_count = 0

            class FakeNodeArg:
                def __init__(self, name: str, shape: list[object], type_name: str) -> None:
                    self.name = name
                    self.shape = shape
                    self.type = type_name

            class FakeSessionOptions:
                def __init__(self) -> None:
                    self.graph_optimization_level = None
                    self.intra_op_num_threads = None
                    self.inter_op_num_threads = None

                def add_session_config_entry(self, _key: str, _value: str) -> None:
                    return None

            class FakeSession:
                def get_inputs(self):
                    return [FakeNodeArg("input_ids", [None, 64], "tensor(int64)")]

                def get_outputs(self):
                    return [FakeNodeArg("sentence_embedding", [None, 4], "tensor(float)")]

            def fake_inference_session(*_args, **_kwargs):
                nonlocal call_count
                call_count += 1
                return FakeSession()

            fake_ort = types.ModuleType("onnxruntime")
            fake_ort.GraphOptimizationLevel = types.SimpleNamespace(
                ORT_DISABLE_ALL="disable",
                ORT_ENABLE_BASIC="basic",
                ORT_ENABLE_EXTENDED="extended",
                ORT_ENABLE_ALL="all",
            )
            fake_ort.ExecutionMode = types.SimpleNamespace(
                ORT_SEQUENTIAL="sequential",
                ORT_PARALLEL="parallel",
            )
            fake_ort.SessionOptions = FakeSessionOptions
            fake_ort.InferenceSession = fake_inference_session

            with mock.patch(
                "twinr.agent.routing.user_intent_bundle._get_onnxruntime",
                return_value=fake_ort,
            ), mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                bundle = load_user_intent_bundle(bundle_dir)
                encoder = OnnxSentenceEncoder(
                    model_path=bundle.model_path,
                    tokenizer_path=bundle.tokenizer_path,
                    max_length=bundle.metadata.max_length,
                    pooling=bundle.metadata.pooling,
                    output_name=bundle.resolved_output_name,
                    normalize_embeddings=bundle.metadata.normalize_embeddings,
                )
                encoder._load_session()

        self.assertEqual(call_count, 1)

    def test_user_intent_bundle_can_defer_ort_runtime_validation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bundle_dir = _write_user_intent_bundle(Path(temp_dir))
            (bundle_dir / "model.ort").write_bytes(b"ort-placeholder")
            inference_module._PRELOADED_ORT_SESSIONS.clear()
            call_count = 0

            def fake_inference_session(*_args, **_kwargs):
                nonlocal call_count
                call_count += 1
                raise AssertionError("Deferred runtime validation must not build an ORT session.")

            fake_ort = types.ModuleType("onnxruntime")
            fake_ort.GraphOptimizationLevel = types.SimpleNamespace(
                ORT_DISABLE_ALL="disable",
                ORT_ENABLE_BASIC="basic",
                ORT_ENABLE_EXTENDED="extended",
                ORT_ENABLE_ALL="all",
            )
            fake_ort.ExecutionMode = types.SimpleNamespace(
                ORT_SEQUENTIAL="sequential",
                ORT_PARALLEL="parallel",
            )
            fake_ort.SessionOptions = object
            fake_ort.InferenceSession = fake_inference_session

            with mock.patch(
                "twinr.agent.routing.user_intent_bundle._get_onnxruntime",
                return_value=fake_ort,
            ), mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                bundle = load_user_intent_bundle(
                    bundle_dir,
                    eager_runtime_validation=False,
                )

        self.assertEqual(call_count, 0)
        self.assertEqual(bundle.model_format, "ort")
        self.assertEqual(bundle.model_input_names, ())
        self.assertEqual(bundle.model_output_names, ())
        self.assertIsNone(bundle.resolved_output_name)
        self.assertEqual(bundle.embedding_dim, 4)
        self.assertEqual(inference_module._PRELOADED_ORT_SESSIONS, {})

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
        self.assertGreater(constrained.margin, 0.0)
        self.assertLess(constrained.margin, 0.1)
        self.assertEqual(constrained.scores["web"], 0.0)
        self.assertGreater(constrained.scores["memory"], 0.0)
        self.assertGreater(constrained.scores["tool"], 0.0)
        self.assertTrue(constrained.authoritative)
        self.assertIsNone(constrained.fallback_reason)

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
        self.assertAlmostEqual(evaluation.unsafe_authoritative_error_rate, 0.5)
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
        self.assertEqual(encoder._session_input_specs, ())

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
                text="backend label web heute berlin",
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
                source_dir=_write_test_source_model(Path(temp_dir)),
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
                source_dir=_write_test_source_model(Path(temp_dir)),
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

    def test_local_semantic_router_warmup_delegates_to_encoder(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bundle = load_semantic_router_bundle(_write_bundle(Path(temp_dir)))
            encoder = WarmupEncoder()
            router = LocalSemanticRouter(bundle, encoder=encoder)

            router.warmup("bereit")

        self.assertEqual(encoder.calls, ["bereit"])

    def test_two_stage_router_warmup_reuses_shared_encoder_once(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            user_bundle = load_user_intent_bundle(_write_user_intent_bundle(root))
            route_bundle = load_semantic_router_bundle(_write_bundle(root))
            shared_encoder = WarmupEncoder()
            router = TwoStageLocalSemanticRouter(
                user_bundle,
                route_bundle,
                user_intent_router=LocalUserIntentRouter(user_bundle, encoder=shared_encoder),
                route_router=LocalSemanticRouter(route_bundle, encoder=shared_encoder),
            )

            router.warmup("vorheizen")

        self.assertEqual(shared_encoder.calls, ["vorheizen"])

    def test_two_stage_router_infers_shared_encoder_from_identical_artifacts_without_metadata_fingerprint(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            user_bundle = load_user_intent_bundle(_write_user_intent_bundle(root))
            route_bundle = load_semantic_router_bundle(_write_bundle(root))

            router = TwoStageLocalSemanticRouter(user_bundle, route_bundle)

        shared_encoder = router.shared_encoder
        self.assertIsNotNone(router.shared_encoder)
        self.assertIs(shared_encoder, router.user_intent_router.encoder)
        self.assertIs(shared_encoder, router.route_router.encoder)
        assert shared_encoder is not None
        self.assertFalse(shared_encoder.prefer_ort_model)
        self.assertEqual(shared_encoder.model_path.name, "model.onnx")

    def test_two_stage_router_prefers_shared_ort_encoder_when_sidecars_exist_for_identical_source_model(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            user_bundle_dir = _write_user_intent_bundle(root)
            route_bundle_dir = _write_bundle(root)
            user_bundle = load_user_intent_bundle(user_bundle_dir)
            route_bundle = load_semantic_router_bundle(route_bundle_dir)
            (user_bundle_dir / "model.ort").write_bytes(b"user-intent-ort-export")
            (route_bundle_dir / "model.ort").write_bytes(b"route-ort-export")
            user_bundle = replace(
                user_bundle,
                model_path=(user_bundle_dir / "model.ort").resolve(),
                model_format="ort",
            )
            route_bundle = replace(
                route_bundle,
                model_path=(route_bundle_dir / "model.ort").resolve(),
                runtime_format="ort",
            )

            router = TwoStageLocalSemanticRouter(user_bundle, route_bundle)

        shared_encoder = router.shared_encoder
        self.assertIsNotNone(shared_encoder)
        self.assertIs(shared_encoder, router.user_intent_router.encoder)
        self.assertIs(shared_encoder, router.route_router.encoder)
        assert shared_encoder is not None
        self.assertTrue(shared_encoder.prefer_ort_model)
        self.assertEqual(shared_encoder.model_path.suffix, ".ort")
        self.assertEqual(shared_encoder.model_path.name, "model.ort")
        self.assertEqual(shared_encoder.model_path, (user_bundle_dir / "model.ort").resolve())


if __name__ == "__main__":
    unittest.main()
