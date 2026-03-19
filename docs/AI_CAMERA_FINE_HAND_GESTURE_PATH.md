# AI Camera Fine Hand Gesture Path

## Purpose

This note evaluates a realistic local path for Twinr to add
`fine_hand_gesture` support on top of the existing Raspberry Pi AI Camera path.

The target class is:
- `thumbs_up`
- `thumbs_down`
- `ok_sign`
- `pointing`
- `middle_finger`

This document is deliberately separate from the main camera contract. The
current V1 camera path must stay defensible without pretending that coarse
pose-only output can already support finger-level semantics.

## Current Reality In Twinr

The current Twinr camera stack already has:
- local-first IMX500 person detection
- coarse body pose
- coarse motion
- coarse arm gestures
- stable social-camera snapshot and event surfaces

The current stack does **not** have:
- finger landmarks
- a hand crop/ROI path
- a fine-hand gesture classifier
- task models for hand landmarks or gesture recognition checked into the repo

Environment observations from the current workspace:
- repo-local `.venv` on the leading checkout does **not** currently have `mediapipe`
- the Pi runtime `.venv` on `/twinr` **does** have `mediapipe 0.10.18`
- the Pi runtime has both `HandLandmarker` and `GestureRecognizer` APIs available
- no `*.task` hand/gesture model assets were found under `/twinr`

## Official Source Signals

MediaPipe official sources indicate:
- the Hand Landmarker path is an on-device vision task intended to return hand landmarks
- the Gesture Recognizer task has live-stream support and task APIs suitable for on-device use
- Google provides an official customization path for gesture recognizers based on hand-landmark extraction

Raspberry Pi official sources indicate:
- the AI Camera on IMX500 supports deploying custom neural network models
- Raspberry Pi officially supports pre-packaged and custom IMX500 models through the documented packaging/conversion toolchain

## Candidate Paths

### Option A: MediaPipe Hand Landmarker On The Pi CPU

Use the existing AI-camera person/pose path as the cheap always-on gate. When
Twinr sees a plausible near-device hand or coarse-arm intent, crop one or two
hand candidate ROIs locally and run MediaPipe Hand Landmarker on the Pi CPU.

Pros:
- strongest near-term path for actual finger-level landmarks
- already partially de-risked because `mediapipe` is present on the Pi runtime
- keeps the current IMX500 path for cheap always-on gating
- clean separation: coarse always-on vision stays on IMX500, fine hand analysis
  runs only on demand

Cons:
- adds a second local vision runtime path
- needs ROI extraction and bounded scheduling
- needs model assets to be added and versioned deliberately

Assessment:
- best practical V2 path

### Option B: MediaPipe Gesture Recognizer On The Pi CPU

Use the higher-level Gesture Recognizer task instead of only landmarks.

Pros:
- official on-device task
- faster path to a working canned gesture surface

Cons:
- canned gestures will not necessarily match Twinr's exact product labels
- `pointing` and `middle_finger` will likely still need custom logic or custom
  training
- less transparent than working directly from landmarks

Assessment:
- useful as a bootstrap, but likely not enough by itself for Twinr's desired
  label set

### Option C: Custom IMX500 Fine-Hand Model

Train and package a custom model for the AI Camera itself.

Pros:
- best long-term efficiency if fine-hand gestures become a first-class,
  always-on requirement
- keeps more inference work on the camera sensor

Cons:
- highest implementation cost
- requires dataset, training, quantization, conversion, packaging, and Pi-side
  post-processing
- much slower iteration loop than CPU-side landmarks

Assessment:
- attractive later, but wrong first step for Twinr right now

## Recommendation

Recommended path:
1. Keep IMX500 as the always-on coarse gate.
2. Add a bounded local `hand_landmarks` worker using MediaPipe on the Pi CPU.
3. Trigger it only when the current camera path sees:
   - `person_visible`
   - `person_near_device` or `showing_intent`
   - a likely upper-body/hand gesture window
4. Build Twinr's `fine_hand_gesture` surface from landmarks, not from coarse
   pose heuristics.

Why this is the right cut:
- it is local-first
- it matches the current Pi environment
- it keeps the always-on runtime cheap
- it avoids lying about finger semantics on top of a shoulder/elbow/wrist model

## Suggested V2 Architecture

- `src/twinr/hardware/hand_landmarks.py`
  - own the local MediaPipe runtime, task loading, ROI execution, and bounded
    inference API
- `src/twinr/perception/hand_gesture/`
  - own landmark-to-gesture classification and confidence rules
- `src/twinr/proactive/social/local_camera_provider.py`
  - stay thin; only map outputs onto the social contract
- `src/twinr/proactive/social/camera_surface.py`
  - stabilize `fine_hand_gesture` and expose bounded events

## Immediate Follow-Up

The next concrete experiment should be:
- add task-model assets explicitly
- prove one bounded Pi script that runs Hand Landmarker on still frames or ROI
  crops
- validate whether `thumbs_up`, `thumbs_down`, and `pointing` are already
  separable with landmarks before committing to product wiring

Do **not** start with `middle_finger` as the engineering anchor. It is a poor
first benchmark and not product-critical.

## Sources

- Google AI Edge, MediaPipe Gesture Recognizer API:
  https://ai.google.dev/edge/mediapipe/api/solutions/python/mp/tasks/vision/GestureRecognizerOptions
- Google AI Edge, gesture recognizer customization guide:
  https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer
- Raspberry Pi official AI Camera documentation:
  https://www.raspberrypi.com/documentation/accessories/ai-camera.html
