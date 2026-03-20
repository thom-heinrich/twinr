# mic

Pi-side audio setup script for Twinr playback, capture, and proactive listening defaults.

## Responsibility

`mic` owns:
- select ALSA and PipeWire/Pulse default devices for Twinr
- support split playback/capture defaults when output and microphone hardware differ
- persist proactive-audio env keys used by the runtime
- install the XVF3800 USB-access rule needed for non-root host-control reads
- run bounded playback and capture smoke checks

`mic` does **not** own:
- runtime audio capture logic in `src/twinr/hardware/audio.py`
- wakeword or proactive policy
- printer, display, or GPIO setup

## Key files

| File | Purpose |
|---|---|
| [setup_audio.sh](./setup_audio.sh) | Configure audio defaults, normalize playback loudness, and smoke test |
| [setup_respeaker_access.sh](./setup_respeaker_access.sh) | Install the XVF3800 udev rule for non-root host-control access |

## Usage

```bash
sudo ./hardware/mic/setup_audio.sh --env-file .env --test
sudo ./hardware/mic/setup_audio.sh --env-file .env --proactive-device-match reSpeaker
sudo ./hardware/mic/setup_audio.sh --env-file .env --proactive-device-match PlayStation
sudo ./hardware/mic/setup_respeaker_access.sh
```

`setup_audio.sh` now also resets the selected playback sink plus the target
card's playback mixer controls to audible defaults so a newly selected
ReSpeaker output does not stay effectively muted by inherited low volume.
Pass `--skip-playback-volume` only when you intentionally want to keep the
current sink/card loudness unchanged.

## See also

- [Top-level hardware README](../README.md)
- [Runtime audio adapter](../../src/twinr/hardware/audio.py)
