# hardware

`hardware` owns Twinr's low-level audio, GPIO, camera, printer, and local
voice-profile adapters. It exposes bounded device I/O helpers that workflows,
ops checks, and proactive services compose into higher-level behavior.

## Responsibility

`hardware` owns:
- capture and play bounded audio for conversation and ambient sensing
- monitor GPIO buttons and PIR motion across supported Pi backends
- capture still photos from V4L2 cameras
- format and submit bounded receipt print jobs
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
| [buttons.py](./buttons.py) | GPIO button monitoring backends |
| [camera.py](./camera.py) | ffmpeg-backed still capture |
| [pir.py](./pir.py) | PIR motion wrapper on buttons |
| [printer.py](./printer.py) | Receipt formatting and CUPS submission |
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
