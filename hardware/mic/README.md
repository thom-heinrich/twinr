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
sudo ./hardware/mic/setup_audio.sh --env-file .env --device-match reSpeaker --capture-device-match reSpeaker --softvol-max-db 18
sudo ./hardware/mic/setup_respeaker_access.sh
```

`setup_audio.sh` now also resets the selected playback sink plus the target
card's playback mixer controls to audible defaults so a newly selected
ReSpeaker output does not stay effectively muted by inherited low volume.
Playback matching can be forced independently from capture via
`TWINR_AUDIO_OUTPUT_DEVICE_MATCH` or `--device-match`, so Twinr can use HDMI
speakers while keeping the ReSpeaker as the capture path.
If the selected playback hardware is already at `100%` mixer volume but still
too quiet, `--softvol-max-db` (or `TWINR_AUDIO_OUTPUT_SOFTVOL_MAX_DB`) adds an
ALSA `softvol` stage so Twinr can push louder software gain without depending
on a desktop PipeWire session. The script now primes that virtual playback PCM
with one silent block so ALSA materializes the `Twinr Playback` control
immediately and then pins it to `100%`, instead of leaving the new gain stage
at its quiet default.
After normalizing those ALSA playback controls it also stores the card state,
so a later re-enumeration or service restart does not restore the old muted
profile.
When it can resolve the target capture card by name, it also persists
`TWINR_AUDIO_INPUT_DEVICE` as an explicit `plughw:CARD=...,DEV=...` path so
the normal conversation recorder does not drift back onto a stale generic
`default` alias while wakeword/proactive capture already uses the real
ReSpeaker device.
The same setup flow now also persists `TWINR_AUDIO_OUTPUT_DEVICE` as an
explicit Twinr-owned playback PCM, preferring `twinr_playback_softvol` when
the script materializes that ALSA stage and otherwise pinning Twinr to the
resolved playback card directly instead of leaving speech output on the
desktop-facing `default` alias.
Pass `--skip-playback-volume` only when you intentionally want to keep the
current sink/card loudness unchanged.

## See also

- [Top-level hardware README](../README.md)
- [Runtime audio adapter](../../src/twinr/hardware/audio.py)
