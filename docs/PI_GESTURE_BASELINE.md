# Pi Gesture Baseline

Frozen Raspberry Pi gesture-recognition baseline accepted on `2026-03-23`.

This document records the currently working gesture stack exactly as it is
configured on `/twinr`. Treat it as the reference point to preserve until a new
Pi-backed retune is explicitly accepted.

## Scope

This baseline covers:

- IMX500 camera low-light adaptation
- MediaPipe hand and gesture thresholds
- ROI and whole-frame fallback geometry
- HDMI gesture-ack publish thresholds
- Candidate-capture debugging settings

This baseline does **not** cover:

- voice activation tuning
- eye-follow tuning
- servo-follow tuning
- general social-vision policy outside HDMI gesture acknowledgement

## Source Of Truth

- [`src/twinr/hardware/camera_ai/config.py`](../src/twinr/hardware/camera_ai/config.py)
- [`src/twinr/hardware/camera_ai/imx500_runtime.py`](../src/twinr/hardware/camera_ai/imx500_runtime.py)
- [`src/twinr/hardware/camera_ai/live_gesture_pipeline.py`](../src/twinr/hardware/camera_ai/live_gesture_pipeline.py)
- [`src/twinr/hardware/hand_landmarks.py`](../src/twinr/hardware/hand_landmarks.py)
- [`src/twinr/proactive/runtime/gesture_ack_lane.py`](../src/twinr/proactive/runtime/gesture_ack_lane.py)

The values below were re-read from the effective Pi runtime config on
`/twinr/.env` via `TwinrConfig.from_env(".env")` on `2026-03-23`.

## Effective Config

### Camera And Low Light

| Setting | Value |
|---|---|
| Normal frame rate | `15 fps` |
| Low-light frame rate | `6 fps` |
| Enter low-light mode at | `<= 1.2 lux` |
| Recover to normal mode at | `>= 2.2 lux` |
| Manual low-light exposure ratio | `0.90` |
| Manual low-light analogue gain | `8.0` |
| AE capped detection ratio | `0.60` |

Low-light behavior is adaptive, not fixed-brightness:

1. Lower frame rate from `15 fps` to `6 fps`.
2. Keep auto exposure if it uses the larger frame budget.
3. Force bounded manual long exposure only if AE still stays capped.
4. Restore normal auto exposure and frame rate once lux recovers.

### MediaPipe Models And Thresholds

| Setting | Value |
|---|---|
| Pose model | `state/mediapipe/models/pose_landmarker_full.task` |
| Hand landmarker model | `state/mediapipe/models/hand_landmarker.task` |
| Built-in gesture model | `state/mediapipe/models/gesture_recognizer.task` |
| Custom gesture model | `state/mediapipe/models/custom_gesture.task` |
| Built-in gesture min score | `0.35` |
| Custom gesture min score | `0.45` |
| Hand detection confidence | `0.35` |
| Hand presence confidence | `0.35` |
| Hand tracking confidence | `0.35` |
| MediaPipe sequence window | `0.55 s` |
| MediaPipe sequence min frames | `3` |
| Configured MediaPipe hand count | `2` |
| Dedicated live hot-path hand count | `1` |

The dedicated HDMI live lane intentionally keeps the fast recognizer at one hand
while the broader MediaPipe config still allows two hands for ROI and fallback work.

### ROI And Rescue Geometry

| Setting | Value |
|---|---|
| Max ROI candidates | `4` |
| Primary person ROI padding | `0.18` |
| Primary person upper-body ratio | `0.78` |
| Wrist ROI scale | `0.34` |
| Recent primary-person TTL | `0.45 s` |
| Recent visible-person TTL | `0.45 s` |
| Recent live-hand-box TTL | `0.45 s` |

The current resolution order is:

1. `live_stream`
2. `person_roi` / `visible_person_roi` / recent visible-person ROI reuse
3. `live_hand_roi` / recent live-hand ROI reuse
4. `full_frame_hand_roi`
5. `mediapipe_pose_fallback`

The current freeze point includes the extra whole-frame rescue when a visible
person ROI exists but yields `0` hand detections.

### HDMI Ack Lane Floors

All current HDMI ack gestures use `confirm_samples=1`.

| Gesture | Effective min confidence | Hold |
|---|---|---|
| `thumbs_up` | `0.56` | `0.35 s` |
| `thumbs_down` | `0.44` | `0.35 s` |
| `pointing` | `0.66` | `0.32 s` |
| `peace_sign` | `0.60` | `0.40 s` |
| `ok_sign` | `0.86` | `0.46 s` |
| `middle_finger` | `0.90` | `0.28 s` |
| `wave` coarse event | `0.68` | `0.35 s` |

These floors are Pi-tuned acknowledgement floors for the HDMI emoji lane only.
They are intentionally separate from broader social/wakeup policy.
`thumbs_down` is intentionally lower than the earlier freeze because fresh Pi
runtime rows on 2026-03-23 still showed valid live ROI detections around
`0.446` and `0.489`, while `0.360` remained safely below the floor.

### Candidate Capture Debugging

| Setting | Value |
|---|---|
| Capture directory | `artifacts/tmp/gesture_candidate_captures` |
| Capture cooldown | `1.5 s` |
| Max retained images | `24` |

This stays enabled so later regressions can be checked visually instead of being
judged only from counters.

## Acceptance Snapshot

Accepted on `2026-03-23` with:

- Local targeted slice: `59 passed`
  `test/test_live_gesture_pipeline.py`
  `test/test_ai_camera.py`
  `test/test_hand_landmarks.py`
- Pi targeted slice: `59 passed`
  same test set on `/twinr`
- Pi runtime services active:
  `twinr-runtime-supervisor.service`
  `twinr-remote-memory-watchdog.service`
  `twinr-web.service`

## Regression Guard

The frozen baseline is now pinned by explicit regression tests for:

- live gesture resolution order and rescue boundaries in [`test/test_live_gesture_pipeline.py`](../test/test_live_gesture_pipeline.py)
- HDMI ack-lane floor boundaries for every supported frozen gesture in [`test/test_gesture_ack_lane.py`](../test/test_gesture_ack_lane.py)
- end-to-end dedicated gesture refresh publication in [`test/test_proactive_monitor.py`](../test/test_proactive_monitor.py)

## Change Rule

If gesture behavior is retuned later, do not silently overwrite this file.
Update this baseline only after:

1. leading-repo code changes are deployed to `/twinr`
2. Pi validation passes
3. the new accepted values are copied here
