#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEVICE_MATCH="${TWINR_AUDIO_DEVICE_MATCH:-Jabra}"
CARD_INDEX=""
DEVICE_INDEX=0
SKIP_ALSA=0
SKIP_PULSE=0
RUN_TEST=0

usage() {
  cat <<'HELP'
Usage: setup_audio.sh [options]

Configure Twinr audio defaults for the Raspberry Pi.

Options:
  --device-match TEXT  Match substring for the USB audio device (default: Jabra)
  --card-index N       Explicit ALSA card index to use
  --device-index N     ALSA device index (default: 0)
  --skip-alsa          Do not write /etc/asound.conf
  --skip-pulse         Do not set PipeWire/Pulse default sink/source
  --test               Run a short playback and capture smoke test
  -h, --help           Show this help text
HELP
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device-match)
      [[ $# -ge 2 ]] || fail "Missing value for --device-match"
      DEVICE_MATCH="$2"
      shift 2
      ;;
    --card-index)
      [[ $# -ge 2 ]] || fail "Missing value for --card-index"
      CARD_INDEX="$2"
      shift 2
      ;;
    --device-index)
      [[ $# -ge 2 ]] || fail "Missing value for --device-index"
      DEVICE_INDEX="$2"
      shift 2
      ;;
    --skip-alsa)
      SKIP_ALSA=1
      shift
      ;;
    --skip-pulse)
      SKIP_PULSE=1
      shift
      ;;
    --test)
      RUN_TEST=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

command -v aplay >/dev/null 2>&1 || fail "aplay not found"
command -v arecord >/dev/null 2>&1 || fail "arecord not found"
[[ "$DEVICE_INDEX" =~ ^[0-9]+$ ]] || fail "--device-index must be an integer"

if [[ -z "$CARD_INDEX" ]]; then
  CARD_INDEX="$(aplay -l | awk -v needle="$DEVICE_MATCH" 'BEGIN { IGNORECASE=1 } $0 ~ needle && $1 == "card" { gsub(":", "", $2); print $2; exit }')"
fi

[[ -n "$CARD_INDEX" ]] || fail "Could not detect an ALSA playback card matching: $DEVICE_MATCH"
[[ "$CARD_INDEX" =~ ^[0-9]+$ ]] || fail "Resolved card index is not numeric: $CARD_INDEX"

CAPTURE_CARD_INDEX="$(arecord -l | awk -v needle="$DEVICE_MATCH" 'BEGIN { IGNORECASE=1 } $0 ~ needle && $1 == "card" { gsub(":", "", $2); print $2; exit }')"
CAPTURE_CARD_INDEX="${CAPTURE_CARD_INDEX:-$CARD_INDEX}"

if [[ "$SKIP_ALSA" -eq 0 ]]; then
  if sudo test -f /etc/asound.conf; then
    sudo cp /etc/asound.conf "/etc/asound.conf.bak.$(date +%Y%m%d%H%M%S)"
  fi
  sudo tee /etc/asound.conf >/dev/null <<ASOUND
pcm.!default {
  type asym
  playback.pcm {
    type plug
    slave.pcm "hw:${CARD_INDEX},${DEVICE_INDEX}"
  }
  capture.pcm {
    type plug
    slave.pcm "hw:${CAPTURE_CARD_INDEX},${DEVICE_INDEX}"
  }
}

ctl.!default {
  type hw
  card ${CARD_INDEX}
}
ASOUND
fi

SINK_NAME=""
SOURCE_NAME=""
if command -v pactl >/dev/null 2>&1; then
  SINK_NAME="$(pactl list short sinks 2>/dev/null | awk -F '\t' -v needle="$DEVICE_MATCH" 'BEGIN { IGNORECASE=1 } $0 ~ needle { print $2; exit }')"
  SOURCE_NAME="$(pactl list short sources 2>/dev/null | awk -F '\t' -v needle="$DEVICE_MATCH" 'BEGIN { IGNORECASE=1 } $0 ~ needle && $2 !~ /monitor/ { print $2; exit }')"
fi

if [[ "$SKIP_PULSE" -eq 0 ]]; then
  command -v pactl >/dev/null 2>&1 || fail "pactl not found"
  [[ -n "$SINK_NAME" ]] || fail "Could not detect a Pulse/PipeWire sink matching: $DEVICE_MATCH"
  [[ -n "$SOURCE_NAME" ]] || fail "Could not detect a Pulse/PipeWire source matching: $DEVICE_MATCH"
  pactl set-default-sink "$SINK_NAME"
  pactl set-default-source "$SOURCE_NAME"
fi

printf 'playback_card=%s\n' "$CARD_INDEX"
printf 'capture_card=%s\n' "$CAPTURE_CARD_INDEX"
printf 'alsa_device=%s\n' "$DEVICE_INDEX"
if [[ -n "$SINK_NAME" ]]; then
  printf 'pulse_sink=%s\n' "$SINK_NAME"
fi
if [[ -n "$SOURCE_NAME" ]]; then
  printf 'pulse_source=%s\n' "$SOURCE_NAME"
fi

if [[ "$RUN_TEST" -eq 1 ]]; then
  python3 - <<'PY' >/tmp/twinr_audio_test.wav
import math
import struct
import sys
import wave

with wave.open(sys.stdout.buffer, 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(16000)
    for idx in range(1600):
        sample = int(12000 * math.sin(2 * math.pi * 440 * (idx / 16000)))
        wav.writeframesraw(struct.pack('<h', sample))
PY
  aplay -D default /tmp/twinr_audio_test.wav >/dev/null 2>&1 || fail "Playback smoke test failed"
  timeout 3 arecord -D default -f S16_LE -r 16000 -c 1 /tmp/twinr_audio_capture.wav >/dev/null 2>&1 || true
  rm -f /tmp/twinr_audio_test.wav /tmp/twinr_audio_capture.wav
  printf 'audio_smoke_test=ok\n'
fi
