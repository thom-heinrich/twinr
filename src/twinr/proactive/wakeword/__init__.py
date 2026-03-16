"""Expose the wakeword package surface for proactive runtime integrations.

Import from ``twinr.proactive.wakeword`` when callers need transcript
matching, openWakeWord spotters, decision policy helpers, calibration stores,
evaluation tools, or the streaming monitor as one coherent package boundary.
"""

from twinr.proactive.wakeword.calibration import (
    WakewordCalibrationProfile,
    WakewordCalibrationStore,
    apply_wakeword_calibration,
)
from twinr.proactive.wakeword.evaluation import (
    WakewordAutotuneRecommendation,
    WakewordEvalEntry,
    WakewordEvalMetrics,
    WakewordEvalReport,
    append_wakeword_capture_label,
    autotune_wakeword_profile,
    load_eval_manifest,
    load_labeled_ops_captures,
    run_wakeword_eval,
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

__all__ = [
    "DEFAULT_WAKEWORD_PHRASES",
    "OpenWakeWordPrediction",
    "OpenWakeWordStreamingMonitor",
    "SttWakewordVerifier",
    "WakewordAutotuneRecommendation",
    "WakewordCalibrationProfile",
    "WakewordCalibrationStore",
    "WakewordDecision",
    "WakewordDecisionPolicy",
    "WakewordEvalEntry",
    "WakewordEvalMetrics",
    "WakewordEvalReport",
    "WakewordMatch",
    "WakewordOpenWakeWordFrameSpotter",
    "WakewordOpenWakeWordSpotter",
    "WakewordPhraseSpotter",
    "WakewordStreamDetection",
    "WakewordVerification",
    "append_wakeword_capture_label",
    "apply_wakeword_calibration",
    "autotune_wakeword_profile",
    "load_eval_manifest",
    "load_labeled_ops_captures",
    "match_wakeword_transcript",
    "normalize_detector_label",
    "normalize_wakeword_backend",
    "normalize_wakeword_verifier_mode",
    "phrase_from_detector_label",
    "run_wakeword_eval",
    "wakeword_primary_prompt",
]
