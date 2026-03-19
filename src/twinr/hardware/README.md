# hardware

`hardware` owns Twinr's low-level audio, GPIO, camera, printer, and local
voice-profile adapters. It exposes bounded device I/O helpers that workflows,
ops checks, and proactive services compose into higher-level behavior.

## Responsibility

`hardware` owns:
- capture and play bounded audio for conversation and ambient sensing
- monitor GPIO buttons and PIR motion across supported Pi backends
- capture still photos from V4L2 cameras
- run bounded hybrid IMX500 + MediaPipe camera inference for proactive sensing, with IMX500 always-on gating, MediaPipe pose/gesture enrichment, coarse motion, coarse-arm gestures, and bounded fine-hand gesture output
- format and submit bounded receipt print jobs
- probe ReSpeaker XVF3800 runtime state and expose typed host-control signals
- derive conservative XVF3800 runtime facts such as direction confidence and busy-state interruption hints
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
| [audio.py](./audio.py) | Bounded audio capture and playback |
| [ai_camera.py](./ai_camera.py) | Hybrid IMX500 gate plus MediaPipe-backed health, detection, motion, coarse-arm, and fine-hand adapter |
| [ai_camera_diagnostics.py](./ai_camera_diagnostics.py) | Bounded Pi-facing diagnostics for pose candidate selection and keypoint support |
| [mediapipe_vision.py](./mediapipe_vision.py) | Pi-side MediaPipe pose, temporal coarse-arm classification, and fine-hand gesture bridge |
| [buttons.py](./buttons.py) | GPIO button monitoring backends |
| [camera.py](./camera.py) | ffmpeg-backed still capture |
| [pir.py](./pir.py) | PIR motion wrapper on buttons |
| [printer.py](./printer.py) | Receipt formatting and CUPS submission |
| [respeaker/](./respeaker/) | XVF3800 probe, host-control transport, typed primitive snapshots, conservative derived signals, and runtime signal provider |
| [respeaker/ambient_classification.py](./respeaker/ambient_classification.py) | Conservative non-speech/background-media classification from bounded ambient capture plus XVF3800 speech facts |
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

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [workflow orchestration](../agent/workflows/README.md)
- [ops self-test entry points](../ops/self_test.py)
