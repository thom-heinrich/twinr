"""Expose Twinr's compact multimodal event-fusion building blocks.

Import from ``twinr.proactive.event_fusion`` when you need rolling buffers,
audio/vision sequence derivation, or conservative fused event claims. The
package is intentionally separate from proactive runtime orchestration so the
fusion layer can stay inspectable and easy to test in isolation.
"""

from twinr.proactive.event_fusion.audio_events import (
    AudioClassifierHints,
    AudioEventConfig,
    AudioEventKind,
    AudioMicroEvent,
    derive_audio_micro_events,
)
from twinr.proactive.event_fusion.buffers import RollingWindowBuffer, TimedSample
from twinr.proactive.event_fusion.claims import (
    EventFusionPolicyContext,
    FusedEventClaim,
    FusionActionLevel,
    build_fused_claim,
)
from twinr.proactive.event_fusion.fusion import (
    MultimodalEventFusionConfig,
    MultimodalEventFusionTracker,
    derive_fused_event_claims,
)
from twinr.proactive.event_fusion.review import (
    KeyframeReviewConfig,
    KeyframeReviewPlan,
    ReviewKeyframeCandidate,
    build_keyframe_review_plan,
)
from twinr.proactive.event_fusion.vision_sequences import (
    VisionSequenceConfig,
    VisionSequenceEvent,
    VisionSequenceKind,
    derive_vision_sequences,
)

__all__ = [
    "AudioClassifierHints",
    "AudioEventConfig",
    "AudioEventKind",
    "AudioMicroEvent",
    "EventFusionPolicyContext",
    "FusedEventClaim",
    "FusionActionLevel",
    "KeyframeReviewConfig",
    "KeyframeReviewPlan",
    "MultimodalEventFusionConfig",
    "MultimodalEventFusionTracker",
    "ReviewKeyframeCandidate",
    "RollingWindowBuffer",
    "TimedSample",
    "VisionSequenceConfig",
    "VisionSequenceEvent",
    "VisionSequenceKind",
    "build_fused_claim",
    "build_keyframe_review_plan",
    "derive_audio_micro_events",
    "derive_fused_event_claims",
    "derive_vision_sequences",
]
