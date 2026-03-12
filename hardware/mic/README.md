# Twinr Audio

Twinr currently uses the `Jabra SPEAK 510 USB` attached to the Raspberry Pi as its default microphone and speaker.

## Configure audio defaults

```bash
cd /twinr
sudo hardware/mic/setup_audio.sh --device-match Jabra
```

This script:

- writes `/etc/asound.conf` so ALSA `default` uses the selected USB audio device
- sets the default PipeWire/Pulse sink and source for the current user session
- can run a short playback/capture smoke test with `--test`

## Useful options

- `--card-index N` when auto-detection is not sufficient
- `--skip-alsa` when only PipeWire/Pulse defaults should change
- `--skip-pulse` when only ALSA should change
