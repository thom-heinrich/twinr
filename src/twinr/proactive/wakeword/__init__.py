"""Expose the wakeword package surface for proactive runtime integrations.

Import from ``twinr.proactive.wakeword`` when callers need transcript
matching, openWakeWord spotters, decision policy helpers, calibration stores,
evaluation/promotion tools, or the streaming monitor as one coherent package
boundary.
"""

from twinr.proactive.wakeword.calibration import (
    WakewordCalibrationProfile,
    WakewordCalibrationStore,
    apply_wakeword_calibration,
)
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
    WakewordVerifierTrainingReport,
    append_wakeword_capture_label,
    autotune_wakeword_profile,
    load_eval_manifest,
    load_labeled_ops_captures,
    run_wakeword_eval,
    train_wakeword_custom_verifier_from_manifest,
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
from twinr.proactive.wakeword.training import (
    WakewordBaseModelTrainingReport,
    train_wakeword_base_model_from_dataset_root,
)

__all__ = [
    "DEFAULT_WAKEWORD_PHRASES",
    "OpenWakeWordPrediction",
    "OpenWakeWordStreamingMonitor",
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
    "WakewordStreamEvalReport",
    "WakewordStreamDetection",
    "WakewordTrainingCommand",
    "WakewordTrainingPlan",
    "WakewordVerification",
    "WakewordAcceptanceMetricPlan",
    "append_wakeword_capture_label",
    "apply_wakeword_calibration",
    "autotune_wakeword_profile",
    "build_default_wakeword_training_plan",
    "evaluate_wakeword_stream_entries",
    "load_wakeword_promotion_spec",
    "load_eval_manifest",
    "load_labeled_ops_captures",
    "match_wakeword_transcript",
    "normalize_detector_label",
    "normalize_wakeword_backend",
    "normalize_wakeword_verifier_mode",
    "phrase_from_detector_label",
    "render_wakeword_training_plan_markdown",
    "run_wakeword_promotion_eval",
    "run_wakeword_eval",
    "run_wakeword_stream_eval",
    "train_wakeword_custom_verifier_from_manifest",
    "train_wakeword_sequence_verifier_from_manifest",
    "train_wakeword_base_model_from_dataset_root",
    "wakeword_primary_prompt",
]
