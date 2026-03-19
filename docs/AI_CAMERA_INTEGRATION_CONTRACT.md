# AI Camera Integration Contract

## Purpose

This document defines what the Raspberry Pi AI Camera path must deliver so Twinr
can integrate it as a calm, bounded, trustworthy runtime signal source.

The goal is not "use the camera somehow". The goal is to define the stable
outputs, quality bar, and safety boundaries that let the rest of Twinr consume
camera signals without knowing model-specific details.

This document is the camera-side contract. It complements
[`SENSOR_FUSION.md`](./SENSOR_FUSION.md), which describes the broader sensor
target field across PIR, audio, and camera.

## Core Rule

The camera path must deliver structured, bounded, policy-safe signals.

It must **not** deliver:
- product decisions such as "speak now"
- hidden prompt-shaped interpretations
- raw-frame dependence in the main runtime contract
- emotion claims, medical claims, or identity claims as facts

The camera subsystem proposes observations. Twinr policy decides what to do.

## Classification Separation Rule

The camera contract must not collapse all human-state signals into one generic
`gesture_event`.

Twinr should reason about four separate classes:
- `body_pose`
- `motion`
- `coarse_arm_gesture`
- `fine_hand_gesture`

These classes serve different consumers, have different confidence limits, and
must be allowed to evolve independently.

Examples:
- `standing`, `seated` -> `body_pose`
- `walking`, `approaching`, `leaving` -> `motion`
- `wave`, `stop`, `dismiss`, `two_hand_dismiss`, `timeout_t`, `arms_crossed`
  -> `coarse_arm_gesture`
- `thumbs_up`, `thumbs_down`, `ok_sign`, `pointing`, `middle_finger`
  -> `fine_hand_gesture`

On the current hardware and pipeline, Twinr V1 should treat `body_pose`,
`motion`, and `coarse_arm_gesture` as realistic first-class requirements.
`fine_hand_gesture` is a separate extension surface that should only become
policy-relevant when a dedicated hand/finger provider proves it can support it
defensibly.

## What The Camera Must Deliver

### 1. Health And Readiness

Twinr must be able to tell whether the camera path is:
- physically present
- enumerated and ready
- currently streaming
- currently running AI post-processing
- degraded, unavailable, or recently failed

Minimum required fields:
- `camera_online: bool`
- `camera_ready: bool`
- `camera_ai_ready: bool`
- `camera_error: str | null`
- `last_camera_frame_at: timestamp | null`
- `last_camera_health_change_at: timestamp | null`

This is required for the Devices page, runtime gating, and clean operator
debugging.

### 2. Stable Presence Confirmation

The camera must deliver a stable answer to:
- is a person visible now?
- how many people are plausibly visible?
- did a person just appear?
- did the visible person disappear?
- did a person return after absence?

Minimum required fields:
- `person_visible: bool`
- `person_count: int`
- `person_recently_visible: bool`
- `person_appeared_at: timestamp | null`
- `person_disappeared_at: timestamp | null`
- `person_returned_after_absence: bool`

This is the camera-side backbone for the Presence Ladder and Resume Engine.

### 3. Spatial Anchor For Voice Matching

The camera must provide a coarse spatial anchor for the main visible person so
DOA from the microphone array can be matched against vision.

Minimum required fields:
- `primary_person_box`: normalized bounding box
- `primary_person_zone: left | center | right | unknown`
- `primary_person_center_x: float | null`
- `primary_person_center_y: float | null`

This does not need identity. It needs a stable left/center/right style mapping
good enough for two-person household turns.

### 4. Attention And Engagement Signals

The camera must provide coarse interaction-quality signals, not mind-reading.

Safe required signals:
- `looking_toward_device: bool | unknown`
- `person_near_device: bool | unknown`
- `engaged_with_device: bool | unknown`
- `visual_attention_score: float | null`

Unsafe/non-required signals:
- emotion labels
- loneliness, confusion, pain, sadness
- fine gaze tracking
- blink-rate driven policy

These signals are for pacing, initiative gating, and display-state transitions.

### 5. Body Pose Class

The camera must provide only coarse posture states that are defensible on the
current hardware.

Minimum required field:
- `body_pose: upright | seated | slumped | lying_low | floor | unknown`

Optional supporting fields:
- `pose_confidence: float | null`
- `body_state_changed_at: timestamp | null`

This is for calm prompts such as concern follow-up, not diagnosis.

### 6. Motion Class

The camera must provide a temporal motion view separate from posture and
separate from gestures.

Minimum required field:
- `motion_state: still | walking | approaching | leaving | unknown`

Optional supporting fields:
- `motion_confidence: float | null`
- `motion_state_changed_at: timestamp | null`

This signal must come from bounded multi-frame evidence. A single-frame pose
guess is not enough to claim `walking`.

### 7. Showing-Intent And Near-Camera Intent

Twinr needs to know when the user is intentionally presenting something to the
device.

Minimum required fields:
- `hand_or_object_near_camera: bool`
- `showing_intent_likely: bool | unknown`
- `showing_intent_started_at: timestamp | null`

This supports "show me this" flows, document/object assistance, and gentle
context prompts.

### 8. Coarse Arm Gesture Class

The camera path should expose a small, high-value set of coarse arm/upper-body
gestures that the current pose stack can plausibly defend.

Realistic V1 target vocabulary:
- `wave`
- `stop`
- `dismiss`
- `confirm`
- `arms_crossed`
- `two_hand_dismiss`
- `timeout_t`

Minimum required field:
- `coarse_arm_gesture: none | wave | stop | dismiss | confirm | arms_crossed | two_hand_dismiss | timeout_t | unknown`

Contract rules:
- this surface must stay small and inspectable
- classifications must be confidence-gated and debounced
- policy must tolerate `unknown`
- the camera contract should not grow into an open-ended gesture language

### 9. Fine Hand Gesture Class

Fine hand/finger gestures are a separate class and must not be faked from a
coarse body-pose model.

Target vocabulary:
- `thumbs_up`
- `thumbs_down`
- `ok_sign`
- `pointing`
- `middle_finger`

Output field:
- `fine_hand_gesture: none | thumbs_up | thumbs_down | ok_sign | pointing | middle_finger | unknown`

Contract rules:
- this class is not a required V1 policy dependency on the current pipeline
- if no dedicated hand/finger provider is active, `unknown` is the honest value
- product logic must not silently reinterpret coarse-arm output as finger-level
  intent
- this class should only graduate into policy use after dedicated validation on
  the Pi

### 10. On-Device Object Detection

The camera path should expose bounded object detections that can support local
automation and contextual help.

Minimum required output shape:
- `objects`: list of `{label, confidence, zone, stable}`

Contract rules:
- labels must be explicit and inspectable
- detections must be confidence-gated
- only stable detections should reach policy consumers by default
- object detection must not silently imply intent

The rest of Twinr may later map these detections into routines or prompts, but
the camera path must only deliver the observation itself.

### 11. Snapshot Surface And Event Surface

Twinr should not consume raw frame-by-frame jitter. The camera path must expose:

1. a current snapshot for synchronous consumers such as display and policy
2. debounced events for transitions that matter

Minimum snapshot examples:
- `person_visible`
- `person_count`
- `primary_person_zone`
- `looking_toward_device`
- `body_pose`
- `motion_state`
- `hand_or_object_near_camera`
- `coarse_arm_gesture`
- `fine_hand_gesture`
- `objects`
- `camera_ready`

Minimum event examples:
- `camera_person_visible_changed`
- `camera_person_returned`
- `camera_attention_window_opened`
- `camera_showing_intent_started`
- `camera_motion_changed`
- `camera_coarse_arm_gesture_detected`
- `camera_fine_hand_gesture_detected`
- `camera_object_detected_stable`

## Realistic Scope On Current Hardware

Twinr should explicitly separate what the current AI-camera stack can defend
today from what needs a stronger provider.

Realistic V1 scope on the current pipeline:
- `body_pose`
- `motion`
- `showing_intent`
- `coarse_arm_gesture`
- stable `objects`

Only promote to V2 after dedicated Pi validation:
- `fine_hand_gesture`

This means a user-level wish such as "recognize standing, sitting, going,
waving, thumbs up, thumbs down, ok sign, pointing, middle finger, crossed arms,
two-hand dismiss, timeout T" should be split as follows:
- `standing`, `sitting` -> V1 `body_pose`
- `going` -> V1 `motion`
- `waving`, `crossed arms`, `two-hand dismiss`, `timeout T` -> V1 `coarse_arm_gesture`
- `thumbs up`, `thumbs down`, `ok sign`, `pointing`, `middle finger` -> V2 `fine_hand_gesture`

## Quality Bar

The camera path must be:
- local-first for core gating signals
- bounded in latency
- debounced and hysteresis-aware
- explicit about confidence
- explicit about unknown state

Twinr should be able to trust:
- `ready` / `not ready`
- `person visible` / `not visible`
- coarse spatial zone
- coarse posture
- coarse motion
- stable coarse-arm/object events

Twinr should **not** be asked to trust:
- emotional interpretation
- identity recognition
- medical conclusions
- fragile single-frame guesses
- finger-level gesture claims without a dedicated hand provider

## Runtime Integration Targets

These are the main consumers the camera contract must support:

| Consumer | Camera deliverables it needs |
|---|---|
| Presence Ladder | `camera_ready`, `person_visible`, `person_recently_visible`, `person_returned_after_absence` |
| Resume Engine | `person_visible`, `person_returned_after_absence`, `engaged_with_device` |
| DOA person matching | `primary_person_box`, `primary_person_zone`, `person_count` |
| Awareness display state machine | `camera_ready`, `person_visible`, `looking_toward_device`, `showing_intent_likely` |
| Initiative score policy | presence, attention, showing-intent, stable motion/coarse-arm/object events |
| Object-led proactive flows | stable `objects` output only |
| Gesture control | `coarse_arm_gesture` first, `fine_hand_gesture` only after dedicated validation |
| Sensor memory | summary events and counts, not raw frames |

## Memory And Privacy Requirements

The camera path must support memory and learning without turning Twinr into a raw
video archive.

Required rules:
- no raw frame persistence by default
- no long-lived frame storage in the main runtime path
- only structured observations may enter sensor memory by default
- event/history output must be inspectable and redactable

Safe memory examples:
- `person_visible morning true`
- `showing_intent event at 10:14`
- `object_detected label=medication_box stable=true`

Unsafe default behavior:
- storing raw frames for routine learning
- storing prompt summaries of private scenes without a bounded reason

## Explicit Non-Requirements

The camera does **not** need to deliver:
- who the person is
- what they feel
- whether they are lonely, sad, confused, or in pain
- whether a medical event definitely happened
- why an object is present
- whether Twinr should speak
- finger-level gesture semantics from a coarse pose-only model

Those are either policy decisions, consent decisions, or claims the hardware
cannot defend.

## Definition Of Done For Seamless Integration

The camera path is ready for seamless Twinr integration when it can provide:

1. a stable health/readiness signal
2. a stable person-visible / person-returned signal
3. a coarse spatial anchor for the visible person
4. a bounded engagement signal
5. a coarse `body_pose` signal
6. a bounded `motion` signal
7. a showing-intent signal
8. a small `coarse_arm_gesture` surface
9. stable object detections
10. both snapshot and event interfaces
11. memory-safe structured outputs with no raw-frame dependence

`fine_hand_gesture` is part of the broader camera target surface, but it is not
required for V1 seamless integration unless a dedicated hand/finger provider is
present and validated on the Pi.

If those V1 pieces are present, Twinr can wire the camera into presence,
initiative, resume, display, gesture, and proactive behavior without coupling
the product to camera-model internals.
