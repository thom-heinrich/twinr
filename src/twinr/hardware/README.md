# hardware

`hardware` owns Twinr's low-level audio, GPIO, camera, printer, local
voice-profile adapters, and the bounded local household-identity building
blocks that combine portrait, voice, and explicit feedback on device.

## Responsibility

`hardware` owns:
- capture and play bounded audio for conversation and ambient sensing
- lift under-driven PCM16 wakeword-ack playback to an audible level before it reaches the Pi speaker path when provider TTS arrives too quiet
- monitor GPIO buttons and PIR motion across supported Pi backends
- capture still photos from V4L2 cameras, with a bounded `rpicam-still` fallback for Pi `unicam-image` devices when the V4L2 node is busy
- persist local portrait identities with multiple reference images per user
- perform bounded local portrait matching from enrolled identities plus still-camera captures
- persist multi-user local household voice identities and assess current-turn audio against enrolled household members
- coordinate one bounded local household identity manager across portrait matching, multi-user voice matching, explicit confirm or deny feedback, and short session history
- run bounded hybrid IMX500 + MediaPipe camera inference for proactive sensing, with IMX500 always-on gating, MediaPipe pose enrichment, dedicated hand-landmark ROI work, stable full-frame gesture recognition, coarse motion, coarse-arm gestures, and bounded fine-hand gesture output
- preserve bounded multi-person IMX500 person anchors in the camera contract so downstream runtime layers can track visible people continuously instead of collapsing every frame to a single primary box
- supplement the IMX500 person path with bounded local YuNet face anchors when SSD only sees zero or one person, so table-side multi-person attention can still recover more than one visible target without cloud vision
- when a YuNet face falls inside an existing person box, retarget that visible-person anchor to the face box for downstream eye-follow, so Twinr looks toward the head instead of the torso centroid while keeping the primary body box for body-pose and presence semantics
- surface IMX500 detection/runtime failures explicitly instead of collapsing them into healthy-looking empty no-person frames, so downstream gaze and gesture consumers can distinguish camera faults from real absence
- expose bounded Pi-side tuning for live hand-gesture responsiveness, including camera FPS, hand-detection/tracking thresholds, gesture score floors, and hand ROI sizing
- keep local AI-camera metadata waits and pose-refresh reuse windows short enough for interactive face-follow and gesture acknowledgement; multi-second waits or stale 10s+ pose reuse are not acceptable for the Pi HCI path
- auto-enable the staged local custom gesture model at `state/mediapipe/models/custom_gesture.task` when it exists, so fine-hand symbols such as `OK_SIGN` do not require a separate env override on deployed Pi images
- treat negative custom labels such as `none` as authoritative suppression, so a weak `ok_sign`/other symbol does not win when the classifier itself prefers "no gesture"
- normalize MediaPipe-bound image buffers centrally so full frames and ROI crops reach the Pi runtime with supported dtype and contiguous layout
- keep multi-ROI MediaPipe tasks monotonic by advancing and reserving timestamps across candidate inference within a frame
- keep MediaPipe gesture recognition latency bounded by preferring ROI gesture recognition when concrete wrist-localized hand crops already exist, and only falling back to the stable full-frame recognizer when ROI crops did not yield a real symbol
- run the dedicated ROI hand-landmark worker in MediaPipe IMAGE mode, because upper-body and wrist crops are independent images rather than one temporally stable video stream; broad upper-body ROI search is a fallback for missing wrist cues, not a mandatory hot-path crop on every frame
- expose a cheap attention-only AI-camera observation that reuses IMX500 person detection plus optional face-anchor supplementation but skips the heavy MediaPipe gesture stack, so HDMI eye-follow can stay reactive while explicit gesture acknowledgement uses the slower full path separately
- expose a dedicated live-stream gesture observation that reuses the RGB camera session but keeps the hot path close to the official Pi-style pattern: one thin built-in live recognizer for immediate symbols, plus bounded primary-person/hand-ROI fallback when that fast path still returns `none`
- allow only short-lived primary-person and hand-box hint reuse inside that gesture lane so brief detector flickers do not make the same gesture randomly disappear, while still keeping user-facing HDMI symbol acknowledgement independent from the broader pose/social observation stack
- format and submit bounded receipt print jobs
- probe ReSpeaker XVF3800 runtime state and expose typed host-control signals
- derive conservative XVF3800 runtime facts such as direction confidence and busy-state interruption hints
- schedule heavy XVF3800 host-control polling more slowly than the cheap ambient-audio path so Pi runtime loops stay bounded while idle
- prove bounded readable-frame capture before higher runtime layers trust a targeted microphone path as voice-ready
- resolve one calm XVF3800 indicator contract so future ring/LED control can mirror listening and mute state without twitching on weak audio evidence
- attach a per-claim ReSpeaker confidence/source contract that later runtime and memory layers can inspect without guessing
- derive conservative XVF3800 non-speech and background-media suppression facts from the same capture path
- scrub borrowed Wayland/user-session env from root-owned ALSA subprocesses so Pi audio probes do not inherit non-root Pulse/PipeWire runtime sockets from the HDMI surface
- persist and score local voice profiles on device

`hardware` does **not** own:
- hardware-loop orchestration or user interaction policy
- display rendering or e-paper control
- OS/bootstrap setup scripts under `hardware/`
- provider STT/TTS/model logic

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Narrow package export surface |
| [audio.py](./audio.py) | Bounded audio capture, readable-frame probing, and playback |
| [audio_env.py](./audio_env.py) | Sanitize child-process env for ALSA helpers when the Pi runtime borrows a different user's Wayland session |
| [camera_ai/](./camera_ai/) | Internal SoC package for camera contracts, IMX500 runtime, MediaPipe runtime, pose, motion, and gesture modules |
| [camera_ai/live_gesture_pipeline.py](./camera_ai/live_gesture_pipeline.py) | Dedicated low-latency MediaPipe live-stream gesture lane for HDMI symbol acknowledgement |
| [ai_camera.py](./ai_camera.py) | Stable compatibility facade for the public local AI-camera adapter surface |
| [ai_camera_diagnostics.py](./ai_camera_diagnostics.py) | Bounded Pi-facing diagnostics for pose candidate selection and keypoint support |
| [hand_landmarks.py](./hand_landmarks.py) | Bounded MediaPipe hand-landmark ROI worker for full-frame-to-hand crop resolution |
| [mediapipe_vision.py](./mediapipe_vision.py) | Stable compatibility facade for the public MediaPipe camera pipeline surface |
| [buttons.py](./buttons.py) | GPIO button monitoring backends |
| [camera.py](./camera.py) | bounded still capture with ffmpeg/V4L2 plus Pi `rpicam-still` fallback for `unicam-image` devices |
| [portrait_identity.py](./portrait_identity.py) | Atomic local portrait-identity store with enrolled reference-image management |
| [portrait_match.py](./portrait_match.py) | Local YuNet/SFace portrait-match adapter with enrollment flow, multi-reference scoring, and temporal fusion |
| [household_voice_identity.py](./household_voice_identity.py) | Multi-user local voice-identity store and matcher for enrolled household members |
| [household_identity.py](./household_identity.py) | Shared local household identity manager for portrait, voice, feedback, and bounded session fusion |
| [pir.py](./pir.py) | PIR motion wrapper on buttons |
| [printer.py](./printer.py) | Receipt formatting and CUPS submission |
| [respeaker/](./respeaker/) | XVF3800 probe, host-control transport, typed primitive snapshots, conservative derived signals, indicator semantics, claim-contract builders, scheduled polling wrapper, and runtime signal provider |
| [respeaker/ambient_classification.py](./respeaker/ambient_classification.py) | Conservative non-speech/background-media classification from bounded ambient capture plus XVF3800 speech facts |
| [respeaker/indicator_policy.py](./respeaker/indicator_policy.py) | Shared calm ring/indicator semantics for listening, mute, and unavailable states |
| [respeaker/scheduled_provider.py](./respeaker/scheduled_provider.py) | Activity-aware wrapper that keeps cheap audio sensing continuous while throttling expensive XVF3800 host-control refreshes on idle ticks |
| [voice_profile.py](./voice_profile.py) | Local voice profile store and scoring |
| [component.yaml](./component.yaml) | Structured package metadata |
| [AGENTS.md](./AGENTS.md) | Local editing rules |

## Usage

```python
from twinr.hardware.audio import SilenceDetectedRecorder
from twinr.hardware.camera import V4L2StillCamera

recorder = SilenceDetectedRecorder.from_config(config)
camera = V4L2StillCamera.from_config(config)
pcm_bytes = recorder.record_pcm_until_pause(pause_ms=900)
```

```python
from twinr.hardware import VoiceProfileMonitor, configured_pir_monitor

voice_monitor = VoiceProfileMonitor.from_config(config)
pir_monitor = configured_pir_monitor(config)
```

```python
from twinr.hardware.household_identity import HouseholdIdentityManager

manager = HouseholdIdentityManager.from_config(config, camera=camera, camera_lock=camera_lock)
status = manager.status(audio_pcm=current_turn_audio, sample_rate=16000, channels=1)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [camera_ai/](./camera_ai/)
- [workflow orchestration](../agent/workflows/README.md)
- [ops self-test entry points](../ops/self_test.py)
