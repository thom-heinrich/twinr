# hardware

`hardware` owns Twinr's low-level audio, GPIO, camera, printer, local
voice-profile adapters, and the bounded local household-identity building
blocks that combine portrait, voice, and explicit feedback on device.

## Responsibility

`hardware` owns:
- capture and play bounded audio for conversation and ambient sensing
- monitor GPIO buttons and PIR motion across supported Pi backends
- capture still photos from V4L2 cameras, with a bounded `rpicam-still` fallback for Pi `unicam-image` devices when the V4L2 node is busy
- persist local portrait identities with multiple reference images per user
- perform bounded local portrait matching from enrolled identities plus still-camera captures
- persist multi-user local household voice identities and assess current-turn audio against enrolled household members
- coordinate one bounded local household identity manager across portrait matching, multi-user voice matching, explicit confirm or deny feedback, and short session history
- run bounded hybrid IMX500 + MediaPipe camera inference for proactive sensing, with IMX500 always-on gating, MediaPipe pose enrichment, dedicated hand-landmark ROI work, ROI-guided gesture recognition, coarse motion, coarse-arm gestures, and bounded fine-hand gesture output
- normalize MediaPipe-bound image buffers centrally so full frames and ROI crops reach the Pi runtime with supported dtype and contiguous layout
- keep multi-ROI MediaPipe tasks monotonic by advancing and reserving timestamps across candidate inference within a frame
- format and submit bounded receipt print jobs
- probe ReSpeaker XVF3800 runtime state and expose typed host-control signals
- derive conservative XVF3800 runtime facts such as direction confidence and busy-state interruption hints
- schedule heavy XVF3800 host-control polling more slowly than the cheap ambient-audio path so Pi runtime loops stay bounded while idle
- prove bounded readable-frame capture before higher runtime layers trust a targeted microphone path as voice-ready
- resolve one calm XVF3800 indicator contract so future ring/LED control can mirror listening and mute state without twitching on weak audio evidence
- attach a per-claim ReSpeaker confidence/source contract that later runtime and memory layers can inspect without guessing
- derive conservative XVF3800 non-speech and background-media suppression facts from the same capture path
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
| [camera_ai/](./camera_ai/) | Internal SoC package for camera contracts, IMX500 runtime, MediaPipe runtime, pose, motion, and gesture modules |
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
