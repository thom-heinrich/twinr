# Twinr Audio

Twinr currently uses the `Jabra SPEAK 510 USB` attached to the Raspberry Pi as its default microphone and speaker.
For proactive background listening, Twinr can additionally use the `PlayStation Eye` USB microphone without replacing the main default input.

## Configure audio defaults

```bash
cd /twinr
sudo hardware/mic/setup_audio.sh --device-match Jabra
```

This script:

- writes `/etc/asound.conf` so ALSA `default` uses the selected USB audio device
- sets the default PipeWire/Pulse sink and source for the current user session
- can run a short playback/capture smoke test with `--test`
- can optionally persist `TWINR_PROACTIVE_AUDIO_*` into `.env` for the background-audio path

## Useful options

- `--card-index N` when auto-detection is not sufficient
- `--skip-alsa` when only PipeWire/Pulse defaults should change
- `--skip-pulse` when only ALSA should change
- `--env-file /twinr/.env` to persist proactive-audio env updates
- `--proactive-device plughw:CARD=CameraB409241,DEV=0` to set the proactive PS-Eye mic explicitly
- `--proactive-device-match PlayStation` to auto-detect a PS-Eye style capture card and store it as `plughw:CARD=...,DEV=...`
- `--proactive-sample-ms 900` to persist the proactive background-audio sample window

## Example: Jabra main audio plus PS-Eye proactive mic

```bash
cd /twinr
sudo hardware/mic/setup_audio.sh \
  --device-match Jabra \
  --proactive-device-match Camera-B4.09.24.1 \
  --proactive-sample-ms 900 \
  --test
```

After that, `/twinr/.env` should contain at least:

```dotenv
TWINR_PROACTIVE_AUDIO_ENABLED=true
TWINR_PROACTIVE_AUDIO_DEVICE=plughw:CARD=CameraB409241,DEV=0
TWINR_PROACTIVE_AUDIO_SAMPLE_MS=900
```
