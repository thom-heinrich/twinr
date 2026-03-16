#!/usr/bin/env bash
# Configure Twinr audio defaults for playback, capture, and proactive sensing.
#
# Selects the target ALSA/PipeWire devices, optionally persists proactive-audio
# env keys, and can run a short playback/capture smoke test on the Pi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${TWINR_ENV_FILE:-$REPO_ROOT/.env}"
PYTHON_BIN=""
DEVICE_MATCH="${TWINR_AUDIO_DEVICE_MATCH:-Jabra}"
CARD_INDEX=""
DEVICE_INDEX=0
PROACTIVE_DEVICE="${TWINR_PROACTIVE_AUDIO_DEVICE:-}"
PROACTIVE_DEVICE_MATCH="${TWINR_PROACTIVE_AUDIO_DEVICE_MATCH:-}"
PROACTIVE_DEVICE_INDEX=0
PROACTIVE_SAMPLE_MS=""
SKIP_ALSA=0
SKIP_PULSE=0
RUN_TEST=0

usage() {
  cat <<'HELP'
Usage: setup_audio.sh [options]

Configure Twinr audio defaults for the Raspberry Pi.

Options:
  --env-file PATH      Path to .env file for proactive audio updates (default: /twinr/.env)
  --device-match TEXT  Match substring for the USB audio device (default: Jabra)
  --card-index N       Explicit ALSA card index to use
  --device-index N     ALSA device index (default: 0)
  --proactive-device ALSA  Explicit ALSA capture device for proactive background audio
  --proactive-device-match TEXT
                      Match substring for the proactive capture device and store it as plughw:CARD=...,DEV=...
  --proactive-device-index N
                      ALSA device index for proactive capture auto-detection (default: 0)
  --proactive-sample-ms N
                      Persist TWINR_PROACTIVE_AUDIO_SAMPLE_MS in the env file
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

resolve_python_bin() {
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    fail "Python 3.11+ not found. Create $REPO_ROOT/.venv or install python3.11."
  fi
  "$PYTHON_BIN" - <<'PY' || fail "Twinr requires Python 3.11+ for setup_audio.sh"
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      [[ $# -ge 2 ]] || fail "Missing value for --env-file"
      ENV_FILE="$2"
      shift 2
      ;;
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
    --proactive-device)
      [[ $# -ge 2 ]] || fail "Missing value for --proactive-device"
      PROACTIVE_DEVICE="$2"
      shift 2
      ;;
    --proactive-device-match)
      [[ $# -ge 2 ]] || fail "Missing value for --proactive-device-match"
      PROACTIVE_DEVICE_MATCH="$2"
      shift 2
      ;;
    --proactive-device-index)
      [[ $# -ge 2 ]] || fail "Missing value for --proactive-device-index"
      PROACTIVE_DEVICE_INDEX="$2"
      shift 2
      ;;
    --proactive-sample-ms)
      [[ $# -ge 2 ]] || fail "Missing value for --proactive-sample-ms"
      PROACTIVE_SAMPLE_MS="$2"
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
resolve_python_bin
[[ "$DEVICE_INDEX" =~ ^[0-9]+$ ]] || fail "--device-index must be an integer"
[[ "$PROACTIVE_DEVICE_INDEX" =~ ^[0-9]+$ ]] || fail "--proactive-device-index must be an integer"
if [[ -n "$PROACTIVE_SAMPLE_MS" ]]; then
  [[ "$PROACTIVE_SAMPLE_MS" =~ ^[0-9]+$ ]] || fail "--proactive-sample-ms must be an integer"
fi

if [[ -z "$CARD_INDEX" ]]; then
  CARD_INDEX="$(aplay -l | awk -v needle="$DEVICE_MATCH" 'BEGIN { IGNORECASE=1 } $0 ~ needle && $1 == "card" { gsub(":", "", $2); print $2; exit }')"
fi

[[ -n "$CARD_INDEX" ]] || fail "Could not detect an ALSA playback card matching: $DEVICE_MATCH"
[[ "$CARD_INDEX" =~ ^[0-9]+$ ]] || fail "Resolved card index is not numeric: $CARD_INDEX"

CAPTURE_CARD_INDEX="$(arecord -l | awk -v needle="$DEVICE_MATCH" 'BEGIN { IGNORECASE=1 } $0 ~ needle && $1 == "card" { gsub(":", "", $2); print $2; exit }')"
CAPTURE_CARD_INDEX="${CAPTURE_CARD_INDEX:-$CARD_INDEX}"

detect_proactive_device() {
  local match="$1"
  local device_index="$2"
  arecord -l | awk -v needle="$match" -v wanted_idx="$device_index" '
    BEGIN { IGNORECASE=1 }
    $0 ~ needle && $1 == "card" {
      card_name = $3
      device_idx = ""
      for (i = 1; i <= NF; i++) {
        if ($i == "device") {
          device_idx = $(i + 1)
          gsub(":", "", device_idx)
          break
        }
      }
      if (device_idx == wanted_idx) {
        printf("plughw:CARD=%s,DEV=%s\n", card_name, device_idx)
        exit
      }
    }
  '
}

if [[ -z "$PROACTIVE_DEVICE" && -n "$PROACTIVE_DEVICE_MATCH" ]]; then
  PROACTIVE_DEVICE="$(detect_proactive_device "$PROACTIVE_DEVICE_MATCH" "$PROACTIVE_DEVICE_INDEX")"
fi

if [[ -n "$PROACTIVE_DEVICE" ]]; then
  mkdir -p "$(dirname "$ENV_FILE")"
  touch "$ENV_FILE"
  "$PYTHON_BIN" - <<PY
from pathlib import Path

path = Path(r'''$ENV_FILE''')
updates = {
    "TWINR_PROACTIVE_AUDIO_ENABLED": "true",
    "TWINR_PROACTIVE_AUDIO_DEVICE": r'''$PROACTIVE_DEVICE''',
}
sample_ms = r'''$PROACTIVE_SAMPLE_MS'''.strip()
if sample_ms:
    updates["TWINR_PROACTIVE_AUDIO_SAMPLE_MS"] = sample_ms

lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
result = []
seen = set()
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        result.append(line)
        continue
    key = stripped.split("=", 1)[0].strip()
    if key in updates:
        if key not in seen:
            result.append(f"{key}={updates[key]}")
            seen.add(key)
        continue
    result.append(line)
for key, value in updates.items():
    if key not in seen:
        result.append(f"{key}={value}")
path.write_text("\n".join(result) + "\n", encoding="utf-8")
PY
fi

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
if [[ -n "$PROACTIVE_DEVICE" ]]; then
  printf 'proactive_audio_device=%s\n' "$PROACTIVE_DEVICE"
  printf 'proactive_env_file=%s\n' "$ENV_FILE"
  if [[ -n "$PROACTIVE_SAMPLE_MS" ]]; then
    printf 'proactive_audio_sample_ms=%s\n' "$PROACTIVE_SAMPLE_MS"
  fi
fi

if [[ "$RUN_TEST" -eq 1 ]]; then
  "$PYTHON_BIN" - <<'PY' >/tmp/twinr_audio_test.wav
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
  if [[ -n "$PROACTIVE_DEVICE" ]]; then
    arecord -D "$PROACTIVE_DEVICE" -f S16_LE -r 16000 -c 1 -d 2 /tmp/twinr_proactive_audio_capture.wav >/dev/null 2>&1 \
      || fail "Proactive audio smoke test failed for $PROACTIVE_DEVICE"
  fi
  rm -f /tmp/twinr_audio_test.wav /tmp/twinr_audio_capture.wav /tmp/twinr_proactive_audio_capture.wav
  printf 'audio_smoke_test=ok\n'
  if [[ -n "$PROACTIVE_DEVICE" ]]; then
    printf 'proactive_audio_smoke_test=ok\n'
  fi
fi
