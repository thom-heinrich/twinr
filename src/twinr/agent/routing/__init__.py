"""Expose Twinr's local semantic-routing contracts and runtime helpers.

The routing package keeps local transcript classification isolated from the
streaming workflow loops. Runtime callers should load a versioned bundle,
classify transcripts through ``LocalSemanticRouter``, and let workflow-local
adapters decide how to apply the result.
"""

from .bootstrap import (
    build_centroid_router_bundle,
    build_centroid_router_bundle_from_jsonl,
    build_linear_router_bundle,
    build_linear_router_bundle_from_jsonl,
    compute_label_centroids,
)
from .bundle import (
    SemanticRouterBundle,
    SemanticRouterBundleMetadata,
    load_semantic_router_bundle,
)
from .contracts import (
    ROUTE_LABEL_VALUES,
    RouteLabel,
    SemanticRouteDecision,
    normalize_route_label,
)
from .evaluation import (
    LabeledRouteSample,
    ScoredRouteRecord,
    SemanticRouterEvaluation,
    evaluate_route_records,
    load_labeled_route_samples,
    score_semantic_router,
    split_labeled_route_samples,
    tune_policy_thresholds,
)
from .policy import SemanticRouterPolicy
from .service import LocalSemanticRouter, OnnxSentenceEncoder
from .synthetic_corpus import (
    SyntheticRouteCurationReport,
    SyntheticRouteSample,
    curate_synthetic_route_samples,
    generate_synthetic_route_samples,
    generate_synthetic_user_intent_samples,
    write_synthetic_route_samples_jsonl,
    write_synthetic_user_intent_samples_jsonl,
)
from .linear_head import LinearHeadTrainingResult, fit_multiclass_linear_head
from .two_stage import LocalUserIntentRouter, TwoStageLocalSemanticRouter
from .training import (
    SemanticRouterTrainingReport,
    TwoStageSemanticRouterTrainingReport,
    UserIntentTrainingReport,
    bootstrap_two_stage_synthetic_router_bundle,
    bootstrap_synthetic_router_bundle,
    train_router_bundle_from_jsonl,
    train_user_intent_bundle_from_jsonl,
)
from .user_intent import (
    ALLOWED_ROUTE_LABELS_BY_USER_INTENT,
    DEFAULT_USER_INTENT_BY_ROUTE_LABEL,
    USER_INTENT_LABEL_VALUES,
    TwoStageSemanticRouteDecision,
    UserIntentDecision,
    UserIntentLabel,
    allowed_route_labels_for_user_intent,
    default_user_intent_for_route_label,
    normalize_user_intent_label,
)
from .user_intent_bootstrap import (
    build_centroid_user_intent_bundle,
    build_centroid_user_intent_bundle_from_jsonl,
    build_linear_user_intent_bundle,
    build_linear_user_intent_bundle_from_jsonl,
)
from .user_intent_bundle import (
    UserIntentBundle,
    UserIntentBundleMetadata,
    load_user_intent_bundle,
)
from .user_intent_evaluation import (
    LabeledUserIntentSample,
    ScoredUserIntentRecord,
    UserIntentRouterEvaluation,
    evaluate_user_intent_records,
    load_labeled_user_intent_samples,
    score_user_intent_router,
    split_labeled_user_intent_samples,
)

__all__ = [
    "LabeledRouteSample",
    "LabeledUserIntentSample",
    "LinearHeadTrainingResult",
    "LocalSemanticRouter",
    "LocalUserIntentRouter",
    "OnnxSentenceEncoder",
    "ALLOWED_ROUTE_LABELS_BY_USER_INTENT",
    "DEFAULT_USER_INTENT_BY_ROUTE_LABEL",
    "ROUTE_LABEL_VALUES",
    "RouteLabel",
    "ScoredRouteRecord",
    "ScoredUserIntentRecord",
    "SemanticRouteDecision",
    "SemanticRouterBundle",
    "SemanticRouterBundleMetadata",
    "SemanticRouterEvaluation",
    "SemanticRouterPolicy",
    "SemanticRouterTrainingReport",
    "SyntheticRouteCurationReport",
    "SyntheticRouteSample",
    "TwoStageLocalSemanticRouter",
    "TwoStageSemanticRouteDecision",
    "TwoStageSemanticRouterTrainingReport",
    "USER_INTENT_LABEL_VALUES",
    "UserIntentBundle",
    "UserIntentBundleMetadata",
    "UserIntentDecision",
    "UserIntentLabel",
    "UserIntentRouterEvaluation",
    "UserIntentTrainingReport",
    "allowed_route_labels_for_user_intent",
    "bootstrap_synthetic_router_bundle",
    "bootstrap_two_stage_synthetic_router_bundle",
    "build_centroid_router_bundle",
    "build_centroid_router_bundle_from_jsonl",
    "build_linear_router_bundle",
    "build_linear_router_bundle_from_jsonl",
    "build_centroid_user_intent_bundle",
    "build_centroid_user_intent_bundle_from_jsonl",
    "build_linear_user_intent_bundle",
    "build_linear_user_intent_bundle_from_jsonl",
    "compute_label_centroids",
    "curate_synthetic_route_samples",
    "default_user_intent_for_route_label",
    "evaluate_route_records",
    "evaluate_user_intent_records",
    "fit_multiclass_linear_head",
    "generate_synthetic_route_samples",
    "generate_synthetic_user_intent_samples",
    "load_labeled_route_samples",
    "load_labeled_user_intent_samples",
    "load_semantic_router_bundle",
    "load_user_intent_bundle",
    "normalize_route_label",
    "normalize_user_intent_label",
    "score_semantic_router",
    "score_user_intent_router",
    "split_labeled_route_samples",
    "split_labeled_user_intent_samples",
    "train_router_bundle_from_jsonl",
    "train_user_intent_bundle_from_jsonl",
    "tune_policy_thresholds",
    "write_synthetic_route_samples_jsonl",
    "write_synthetic_user_intent_samples_jsonl",
]
