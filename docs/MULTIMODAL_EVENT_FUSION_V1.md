# Multimodal Event Fusion V1

## Purpose

This document defines the next Twinr sensing layer beyond single-frame and
single-window classification.

The goal is not "more labels". The goal is to derive a small set of stable
multimodal event claims from short rolling sensor history so Twinr reacts to
real sequences instead of noisy instants.

This is the correct layer for:

- `possible_fall`
- `floor_stillness_after_drop`
- `shout_like_audio`
- `laugh_like_positive_contact`
- `cry_like_distress_possible`
- `slumped_quiet`

This is not the layer for:

- hard identity
- rich emotion inference
- medical diagnosis
- permanent raw audio/video recording

## Why Rolling Buffers Are Required

A large part of useful Twinr sensing is temporal:

- a fall is a transition, not a frame
- shouting is a short acoustic event, not a boolean sample
- crying is usually a repeated audio pattern, not one loud chunk
- positive contact is smile/laughter/attention over a short span, not a frame

Without short rolling buffers, the system is forced to over-trust one image or
one audio window. That is brittle and produces exactly the wrong kind of
"smartness".

The correct design is:

1. keep a short RAM-only rolling history per modality
2. derive small per-modality micro-events
3. fuse those micro-events over `2-8s`
4. emit conservative event claims with confidence and action limits

## Core Design Rules

- Buffers stay in RAM by default.
- Raw audio/video is not persisted by default.
- Video buffering must stay keyframe- and metadata-oriented, not full-motion raw video.
- Fusion outputs event-like claims, not psychological truths.
- Every claim carries `confidence`, `source`, `requires_confirmation`, and a bounded time window.
- Runtime policy remains fail-closed under ambiguity, background media, or multi-person uncertainty.
- `service.py` stays orchestration-only. Fusion logic belongs in a dedicated package.

## Buffer Architecture

### 1. Audio Rolling Buffer

Keep a short audio ring buffer in RAM.

Recommended V1:

- raw mono/downmixed PCM ring: `8s`
- feature frame hop: `100-250ms`
- event-classification windows: `0.5-2.0s`

The raw ring is only for recent context and optional bounded review. The main
online path should run on derived feature frames and audio micro-events.

Recommended audio-derived micro-events:

- `speech_activity`
- `voice_energy_spike`
- `laugh_like_audio`
- `cry_like_audio`
- `shout_like_audio`
- `cough_like_audio`
- `alarm_like_audio`
- `water_like_audio`
- `background_media_likely`
- `speech_overlap_likely`

### 2. Vision Rolling Buffer

For video, Twinr should not keep a heavyweight raw video loop by default.

Recommended V1:

- structured observation ring: `8s`
- observation cadence: `2-4 Hz`
- optional keyframe ring: last `4-8` JPEG keyframes only

The structured observation ring is the main path. Keyframes exist only for
bounded local review, operator debugging, or optional second opinion.

Recommended vision-derived micro-events:

- `person_visible`
- `person_count`
- `primary_person_zone`
- `looking_toward_device`
- `engaged_with_device`
- `showing_intent_likely`
- `body_pose = upright | seated | slumped | lying_low | floor`
- `downward_transition`
- `floor_pose_entered`
- `smile_visible`
- `hand_or_object_near_camera`
- `motion_state = still | walking | approaching | leaving`

### 3. Fusion Window

On top of modality-local buffers, Twinr needs one short aligned fusion window.

Recommended V1:

- fusion horizon: `2-8s`
- alignment resolution: `100-250ms`
- inputs: audio micro-events, vision observations, PIR motion, room-state facts

This layer should produce only small typed event claims, not free-form text.

## Recommended Event Vocabulary

### Audio-first events

These should be classified primarily from audio, then vetoed or supported by
vision and room context:

- `shout_like_audio`
- `cough_like_audio`
- `laugh_like_audio`
- `cry_like_audio`
- `alarm_like_audio`
- `water_like_audio`

### Vision-first events

These should be classified primarily from camera observations and transitions:

- `person_returned`
- `attention_window`
- `showing_intent`
- `slumped_quiet`
- `downward_transition`
- `floor_pose_entered`
- `floor_stillness`

### Fused events

These should be the actual runtime-facing Twinr event claims:

- `possible_fall`
  - example sequence: upright or seated -> downward transition or visibility loss -> floor/lying_low or prolonged disappearance -> low motion -> quiet

- `floor_stillness_after_drop`
  - example sequence: floor pose -> stillness hold -> quiet hold

- `distress_possible`
  - example sequence: distress-like audio -> visible or slumped person -> no media-like room context

- `cry_like_distress_possible`
  - example sequence: repeated cry/sob-like audio -> visible person -> slumped or low-engagement posture -> quiet room -> no TV

- `laugh_like_positive_contact`
  - example sequence: laugh-like audio -> smile or positive attention -> looking toward device -> no background media

- `slumped_quiet`
  - example sequence: slumped pose -> low motion -> quiet hold

## What Should Stay Out Of Scope

This layer should not claim:

- `sad`
- `lonely`
- `confused`
- `in pain`
- `medical event`
- `panic attack`
- `same person with certainty`

If Twinr needs those kinds of behaviors later, they must remain downstream
prompting or escalation policy, not sensor-fusion facts.

## Claim Contract

Each fused event should expose:

```json
{
  "state": "possible_fall",
  "active": true,
  "confidence": 0.83,
  "source": "audio_event_plus_pose_sequence",
  "requires_confirmation": true,
  "window_start_s": 1234.1,
  "window_end_s": 1240.8,
  "action_level": "prompt_only"
}
```

Optional additional fields:

- `supporting_audio_events`
- `supporting_vision_events`
- `blocked_by`
- `review_recommended`

## Action Levels

- `direct`
  - internal gates only, never user-facing facts

- `prompt_only`
  - safe for a gentle spoken or visual check-in

- `review_only`
  - safe only after a buffered second opinion

- `never`
  - must not be presented as fact

Recommended V1:

- `possible_fall` -> `prompt_only` or `review_only`
- `floor_stillness_after_drop` -> `prompt_only` or `review_only`
- `shout_like_audio` -> `prompt_only`
- `cry_like_distress_possible` -> `review_only` or very conservative `prompt_only`
- `laugh_like_positive_contact` -> `prompt_only`

## Pi Budget And Privacy

### RAM budget

The short-buffer approach is cheap if designed correctly:

## Research-backed V2 Notes

The next Twinr step should copy three patterns that are consistent across
recent primary-source work:

1. Multi-scale temporal modeling instead of single-window scoring.
   Recent AVEL systems consistently improve by aggregating evidence over
   multiple temporal scales rather than trusting one instant or one global
   clip. See UniAV (multi-scale unified encoder), AVE-CLIP (multi-window
   temporal transformer), and the dense AVEL benchmark/baseline work on
   multi-scale cross-modal dependency modeling.

2. Explicit background suppression and temporal dynamics.
   Cross-Modal Background Suppression shows that asynchronous or single-modal
   background noise must be suppressed explicitly. ICCV 2025 video-centric AVL
   work likewise shows that global audio plus frame-level mapping is not
   enough; temporal dynamics are a first-class requirement.

3. Keyframe review should optimize relevance plus coverage, not just sample
   the newest frame or uniform spacing. CVPR 2025 keyframe work shows that
   useful frame selection depends on both relevance and coverage, while
   temporal-search benchmarks show that naive frame search remains weak even in
   strong systems.

Twinr's V2 implementation should therefore use:

- exponential time decay on evidence confidence
- multi-scale scoring over recent 2s / 4s / 8s views
- compact onset / peak / latest keyframe review plans
- the existing fail-closed room/media gates before any spoken action

Primary sources:

- UniAV (arXiv 2024): https://arxiv.org/abs/2404.03179
- AVE-CLIP (WACV 2023): https://openaccess.thecvf.com/content/WACV2023/html/Mahmud_AVE-CLIP_AudioCLIP-Based_Multi-Window_Temporal_Transformer_for_Audio_Visual_Event_Localization_WACV_2023_paper.html
- Dense-Localizing Audio-Visual Events in Untrimmed Videos (CVPR 2023): https://openaccess.thecvf.com/content/CVPR2023/html/Geng_Dense-Localizing_Audio-Visual_Events_in_Untrimmed_Videos_A_Large-Scale_Benchmark_and_CVPR_2023_paper.html
- Cross-Modal Background Suppression for AVEL (CVPR 2022): https://openaccess.thecvf.com/content/CVPR2022/html/Xia_Cross-Modal_Background_Suppression_for_Audio-Visual_Event_Localization_CVPR_2022_paper.html
- Towards Open-Vocabulary AVEL (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_Towards_Open-Vocabulary_Audio-Visual_Event_Localization_CVPR_2025_paper.html
- What's Making That Sound Right Now? (ICCV 2025): https://openaccess.thecvf.com/content/ICCV2025/html/Choi_Whats_Making_That_Sound_Right_Now_Video-centric_Audio-Visual_Localization_ICCV_2025_paper.html
- Adaptive Keyframe Sampling (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Adaptive_Keyframe_Sampling_for_Long_Video_Understanding_CVPR_2025_paper.html
- Re-thinking Temporal Search (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Ye_Re-thinking_Temporal_Search_for_Long-Form_Video_Understanding_CVPR_2025_paper.html
- EventVAD (arXiv 2025): https://arxiv.org/abs/2504.13092

- `8s` mono PCM16 at `16 kHz` is about `256 KB`
- structured vision observations are tiny
- `4-8` bounded JPEG keyframes are acceptable
- full raw video buffering is unnecessary and should be avoided

### Privacy

V1 should keep:

- raw audio ring in RAM only
- keyframes in RAM or temp review scope only
- structured event claims in normal runtime telemetry

V1 should not keep:

- permanent rolling audio archives
- permanent rolling video archives
- implicit long-term emotional profiles

## Recommended Package Layout

To preserve strict separation of concerns, add a dedicated package:

- `src/twinr/proactive/event_fusion/__init__.py`
- `src/twinr/proactive/event_fusion/buffers.py`
- `src/twinr/proactive/event_fusion/audio_events.py`
- `src/twinr/proactive/event_fusion/vision_sequences.py`
- `src/twinr/proactive/event_fusion/fusion.py`
- `src/twinr/proactive/event_fusion/claims.py`

Suggested ownership:

- `buffers.py`
  - RAM-only rolling audio, keyframe, and observation buffers

- `audio_events.py`
  - coarse audio event proposals from short windows

- `vision_sequences.py`
  - temporal pose and visibility transitions

- `fusion.py`
  - aligned multimodal `2-8s` event fusion

- `claims.py`
  - normalized runtime-facing claim dataclasses and serialization

`src/twinr/proactive/runtime/service.py` should only:

- feed the buffers
- request the latest fused event claims
- pass those claims into runtime policy and ops telemetry

## Implementation Sequence

### Step 1

Add RAM-only rolling buffers and structured sequence dataclasses.

Do not add user-facing labels yet.

### Step 2

Add coarse audio event proposals:

- `shout_like_audio`
- `cough_like_audio`
- `laugh_like_audio`
- `cry_like_audio`
- `background_media_likely`

### Step 3

Add temporal vision sequence proposals:

- `downward_transition`
- `floor_pose_entered`
- `slumped_hold`
- `smile_hold`
- `showing_intent_hold`

### Step 4

Add fused runtime claims:

- `possible_fall`
- `floor_stillness_after_drop`
- `cry_like_distress_possible`
- `laugh_like_positive_contact`

### Step 5

Route those claims into:

- proactive prompting
- suppression policy
- bounded review path
- ops/debug surfaces

## Recommended First V1 Slice

The highest-value first slice is:

1. audio ring buffer
2. observation ring buffer
3. `shout_like_audio`
4. `cough_like_audio`
5. `possible_fall`
6. `floor_stillness_after_drop`
7. `laugh_like_positive_contact`

That gives Twinr real temporal sensor fusion quickly, without drifting into
emotion theater.
