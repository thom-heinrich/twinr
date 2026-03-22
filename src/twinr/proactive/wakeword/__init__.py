"""Expose the wakeword package surface for proactive runtime integrations.

Import from ``twinr.proactive.wakeword`` when callers need transcript
matching, local wakeword spotters, decision policy helpers, calibration
stores, or the streaming monitor as one coherent package boundary.

Optional evaluation, training, and WeKws-specific helpers must not break the
normal runtime import surface when their heavier dependencies are unavailable
on the Pi.
"""

from __future__ import annotations

from twinr.proactive.wakeword.calibration import (
    WakewordCalibrationProfile,
    WakewordCalibrationStore,
    apply_wakeword_calibration,
)
from twinr.proactive.wakeword.kws import (
    WakewordKwsAssetBundle,
    WakewordSherpaOnnxFrameSpotter,
    WakewordSherpaOnnxSpotter,
)
from twinr.proactive.wakeword.kws_assets import (
    ProvisionedWakewordKwsBundle,
    WakewordKwsProvisionSpec,
    available_builtin_kws_bundle_specs,
    derive_kws_keyword_names,
    provision_builtin_kws_bundle,
)
from twinr.proactive.wakeword.matching import (
    DEFAULT_WAKEWORD_PHRASES,
    WakewordMatch,
    WakewordPhraseSpotter,
    match_wakeword_transcript,
    normalize_detector_label,
    phrase_from_detector_label,
    wakeword_primary_prompt,
)
from twinr.proactive.wakeword.policy import (
    SttWakewordVerifier,
    WakewordDecision,
    WakewordDecisionPolicy,
    WakewordVerification,
    normalize_wakeword_backend,
    normalize_wakeword_verifier_mode,
)
from twinr.proactive.wakeword.spotter import (
    OpenWakeWordPrediction,
    WakewordOpenWakeWordFrameSpotter,
    WakewordOpenWakeWordSpotter,
)
from twinr.proactive.wakeword.stream import OpenWakeWordStreamingMonitor, WakewordStreamDetection
from twinr.proactive.wakeword.training_plan import (
    WakewordAcceptanceMetricPlan,
    WakewordTrainingCommand,
    WakewordTrainingPlan,
    build_default_wakeword_training_plan,
    render_wakeword_training_plan_markdown,
)


_OPTIONAL_WAKEWORD_IMPORT_ERROR: ModuleNotFoundError | Exception | None = None


def _raise_missing_optional_wakeword_dependency(*args, **kwargs):
    """Raise the deferred optional wakeword dependency failure on first use."""

    raise ModuleNotFoundError(
        "Optional wakeword dependency unavailable for this helper on the current machine."
    ) from _OPTIONAL_WAKEWORD_IMPORT_ERROR


try:
    from twinr.proactive.wakeword.cascade import (
        WakewordSequenceCaptureVerifier,
        WakewordSequenceVerifier,
        WakewordSequenceVerifierTrainingReport,
        train_wakeword_sequence_verifier_from_manifest,
    )
    from twinr.proactive.wakeword.evaluation import (
        WakewordAutotuneRecommendation,
        WakewordEvalEntry,
        WakewordEvalMetrics,
        WakewordEvalReport,
        WakewordScoreSanityReport,
        WakewordVerifierTrainingReport,
        append_wakeword_capture_label,
        autotune_wakeword_profile,
        evaluate_wekws_score_sanity_entries,
        load_eval_manifest,
        load_labeled_ops_captures,
        run_wakeword_eval,
        train_wakeword_custom_verifier_from_manifest,
    )
    from twinr.proactive.wakeword.promotion import (
        WakewordAmbientGuardResult,
        WakewordAmbientGuardSpec,
        WakewordPromotionReport,
        WakewordPromotionSpec,
        WakewordPromotionSuiteResult,
        WakewordPromotionSuiteSpec,
        WakewordStreamEvalReport,
        evaluate_wakeword_stream_entries,
        load_wakeword_promotion_spec,
        run_wakeword_promotion_eval,
        run_wakeword_stream_eval,
    )
    from twinr.proactive.wakeword.training import (
        WakewordBaseModelTrainingReport,
        train_wakeword_base_model_from_dataset_root,
    )
    from twinr.proactive.wakeword.wekws_export import (
        WekwsExportReport,
        WekwsExportSplitReport,
        export_wakeword_manifests_to_wekws,
    )
    from twinr.proactive.wakeword.wekws_experiment import (
        PreparedWekwsExperiment,
        PreparedWekwsSplit,
        WekwsRecipeSpec,
        available_wekws_recipe_specs,
        prepare_wekws_experiment,
    )
    from twinr.proactive.wakeword.wekws import (
        WakewordWekwsAssetBundle,
        WakewordWekwsFrameSpotter,
        WakewordWekwsModelConfig,
        WakewordWekwsSpotter,
    )
except Exception as exc:
    _OPTIONAL_WAKEWORD_IMPORT_ERROR = exc
    for _name in (
        "WakewordSequenceCaptureVerifier",
        "WakewordSequenceVerifier",
        "WakewordSequenceVerifierTrainingReport",
        "train_wakeword_sequence_verifier_from_manifest",
        "WakewordAutotuneRecommendation",
        "WakewordEvalEntry",
        "WakewordEvalMetrics",
        "WakewordEvalReport",
        "WakewordScoreSanityReport",
        "WakewordVerifierTrainingReport",
        "append_wakeword_capture_label",
        "autotune_wakeword_profile",
        "evaluate_wekws_score_sanity_entries",
        "load_eval_manifest",
        "load_labeled_ops_captures",
        "run_wakeword_eval",
        "train_wakeword_custom_verifier_from_manifest",
        "WakewordAmbientGuardResult",
        "WakewordAmbientGuardSpec",
        "WakewordPromotionReport",
        "WakewordPromotionSpec",
        "WakewordPromotionSuiteResult",
        "WakewordPromotionSuiteSpec",
        "WakewordStreamEvalReport",
        "evaluate_wakeword_stream_entries",
        "load_wakeword_promotion_spec",
        "run_wakeword_promotion_eval",
        "run_wakeword_stream_eval",
        "WakewordBaseModelTrainingReport",
        "train_wakeword_base_model_from_dataset_root",
        "WekwsExportReport",
        "WekwsExportSplitReport",
        "export_wakeword_manifests_to_wekws",
        "PreparedWekwsExperiment",
        "PreparedWekwsSplit",
        "WekwsRecipeSpec",
        "available_wekws_recipe_specs",
        "prepare_wekws_experiment",
        "WakewordWekwsAssetBundle",
        "WakewordWekwsFrameSpotter",
        "WakewordWekwsModelConfig",
        "WakewordWekwsSpotter",
    ):
        globals()[_name] = _raise_missing_optional_wakeword_dependency


__all__ = [
    "DEFAULT_WAKEWORD_PHRASES",
    "OpenWakeWordPrediction",
    "OpenWakeWordStreamingMonitor",
    "ProvisionedWakewordKwsBundle",
    "SttWakewordVerifier",
    "WakewordAmbientGuardResult",
    "WakewordAmbientGuardSpec",
    "WakewordAutotuneRecommendation",
    "WakewordBaseModelTrainingReport",
    "WakewordCalibrationProfile",
    "WakewordCalibrationStore",
    "WakewordDecision",
    "WakewordDecisionPolicy",
    "WakewordEvalEntry",
    "WakewordEvalMetrics",
    "WakewordEvalReport",
    "WakewordScoreSanityReport",
    "WakewordKwsAssetBundle",
    "WakewordKwsProvisionSpec",
    "WakewordVerifierTrainingReport",
    "WakewordMatch",
    "WakewordOpenWakeWordFrameSpotter",
    "WakewordOpenWakeWordSpotter",
    "WakewordPhraseSpotter",
    "WakewordPromotionReport",
    "WakewordPromotionSpec",
    "WakewordPromotionSuiteResult",
    "WakewordPromotionSuiteSpec",
    "WakewordSequenceCaptureVerifier",
    "WakewordSequenceVerifier",
    "WakewordSequenceVerifierTrainingReport",
    "WakewordSherpaOnnxFrameSpotter",
    "WakewordSherpaOnnxSpotter",
    "WakewordStreamEvalReport",
    "WakewordStreamDetection",
    "WakewordTrainingCommand",
    "WakewordTrainingPlan",
    "WakewordVerification",
    "WekwsExportReport",
    "WekwsExportSplitReport",
    "WakewordWekwsAssetBundle",
    "WakewordWekwsFrameSpotter",
    "WakewordWekwsModelConfig",
    "WakewordWekwsSpotter",
    "PreparedWekwsExperiment",
    "PreparedWekwsSplit",
    "WekwsRecipeSpec",
    "WakewordAcceptanceMetricPlan",
    "append_wakeword_capture_label",
    "apply_wakeword_calibration",
    "autotune_wakeword_profile",
    "available_builtin_kws_bundle_specs",
    "available_wekws_recipe_specs",
    "build_default_wakeword_training_plan",
    "derive_kws_keyword_names",
    "evaluate_wekws_score_sanity_entries",
    "evaluate_wakeword_stream_entries",
    "export_wakeword_manifests_to_wekws",
    "load_eval_manifest",
    "load_wakeword_promotion_spec",
    "load_labeled_ops_captures",
    "match_wakeword_transcript",
    "normalize_detector_label",
    "normalize_wakeword_backend",
    "normalize_wakeword_verifier_mode",
    "phrase_from_detector_label",
    "prepare_wekws_experiment",
    "provision_builtin_kws_bundle",
    "render_wakeword_training_plan_markdown",
    "run_wakeword_promotion_eval",
    "run_wakeword_eval",
    "run_wakeword_stream_eval",
    "train_wakeword_base_model_from_dataset_root",
    "train_wakeword_custom_verifier_from_manifest",
    "train_wakeword_sequence_verifier_from_manifest",
    "wakeword_primary_prompt",
]
