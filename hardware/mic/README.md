# mic

Pi-side audio setup script for Twinr playback, capture, and proactive listening defaults.

## Responsibility

`mic` owns:
- select ALSA and PipeWire/Pulse default devices for Twinr
- persist proactive-audio env keys used by the runtime
- run bounded playback and capture smoke checks

`mic` does **not** own:
- runtime audio capture logic in `src/twinr/hardware/audio.py`
- wakeword or proactive policy
- printer, display, or GPIO setup

## Key files

| File | Purpose |
|---|---|
| [setup_audio.sh](./setup_audio.sh) | Configure audio defaults and smoke test |

## Usage

```bash
sudo ./hardware/mic/setup_audio.sh --env-file .env --device-match Jabra --test
sudo ./hardware/mic/setup_audio.sh --env-file .env --device-match Jabra --proactive-device-match PlayStation
```

## See also

- [Top-level hardware README](../README.md)
- [Runtime audio adapter](../../src/twinr/hardware/audio.py)
