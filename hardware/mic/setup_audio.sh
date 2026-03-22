#!/usr/bin/env bash
# Configure Twinr audio defaults for playback, capture, and proactive sensing.
#
# Selects the target ALSA/PipeWire devices, defaults playback/capture matching
# to the ReSpeaker path used on the Pi, optionally persists proactive-audio env
# keys, and can run a short playback/capture smoke test on the Pi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${TWINR_ENV_FILE:-$REPO_ROOT/.env}"
PYTHON_BIN=""
DEVICE_MATCH="${TWINR_AUDIO_OUTPUT_DEVICE_MATCH:-${TWINR_AUDIO_DEVICE_MATCH:-reSpeaker}}"
CARD_INDEX=""
DEVICE_INDEX=0
CAPTURE_DEVICE_MATCH="${TWINR_AUDIO_CAPTURE_DEVICE_MATCH:-}"
CAPTURE_MATCH="${TWINR_AUDIO_CAPTURE_DEVICE_MATCH:-${TWINR_AUDIO_DEVICE_MATCH:-reSpeaker}}"
CAPTURE_CARD_INDEX=""
CAPTURE_DEVICE_INDEX=""
PROACTIVE_DEVICE="${TWINR_PROACTIVE_AUDIO_DEVICE:-}"
PROACTIVE_DEVICE_MATCH="${TWINR_PROACTIVE_AUDIO_DEVICE_MATCH:-}"
PROACTIVE_DEVICE_INDEX=0
PROACTIVE_SAMPLE_MS=""
CAPTURE_DEVICE=""
SINK_VOLUME_PERCENT="${TWINR_AUDIO_OUTPUT_VOLUME_PERCENT:-100}"
CARD_PLAYBACK_VOLUME_PERCENT="${TWINR_AUDIO_CARD_PLAYBACK_VOLUME_PERCENT:-100}"
SOFTVOL_MAX_DB="${TWINR_AUDIO_OUTPUT_SOFTVOL_MAX_DB:-0}"
SOFTVOL_MAX_DB_RENDERED="0.0"
SOFTVOL_CONTROL_NAME="Twinr Playback"
SKIP_ALSA=0
SKIP_PULSE=0
SKIP_PLAYBACK_VOLUME=0
RUN_TEST=0

usage() {
  cat <<'HELP'
Usage: setup_audio.sh [options]

Configure Twinr audio defaults for the Raspberry Pi.

Options:
  --env-file PATH      Path to .env file for proactive audio updates (default: /twinr/.env)
  --device-match TEXT  Match substring for the playback device (default: TWINR_AUDIO_OUTPUT_DEVICE_MATCH or TWINR_AUDIO_DEVICE_MATCH, else reSpeaker)
  --card-index N       Explicit ALSA playback card index to use
  --device-index N     ALSA playback device index (default: 0)
  --capture-device-match TEXT
                      Match substring for the capture device (defaults to --device-match)
  --capture-card-index N
                      Explicit ALSA capture card index to use
  --capture-device-index N
                      ALSA capture device index (defaults to --device-index)
  --proactive-device ALSA  Explicit ALSA capture device for proactive background audio
  --proactive-device-match TEXT
                      Match substring for the proactive capture device and store it as plughw:CARD=...,DEV=...
  --proactive-device-index N
                      ALSA device index for proactive capture auto-detection (default: 0)
  --proactive-sample-ms N
                      Persist TWINR_PROACTIVE_AUDIO_SAMPLE_MS in the env file
  --sink-volume-percent N
                      Set the selected PipeWire/Pulse sink volume to N percent (default: 100)
  --card-playback-volume-percent N
                      Set playback mixer controls on the selected ALSA card to N percent (default: 100)
  --softvol-max-db DB
                      Add an ALSA softvol playback stage with up to +DB software gain (default: 0 = disabled)
  --skip-alsa          Do not write /etc/asound.conf
  --skip-pulse         Do not set PipeWire/Pulse default sink/source
  --skip-playback-volume
                      Do not normalize sink or ALSA playback volume for the selected output card
  --test               Run a short playback and capture smoke test
  -h, --help           Show this help text
HELP
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

detect_card_index() {
  local command_name="$1"
  local match="$2"

  [[ -n "$match" ]] || return 0
  "$command_name" -l | awk -v needle="$match" '
    BEGIN { IGNORECASE=1 }
    $0 ~ needle && $1 == "card" {
      gsub(":", "", $2)
      print $2
      exit
    }
  '
}

detect_pactl_name() {
  local table="$1"
  local match="$2"
  local skip_monitors="${3:-0}"

  [[ -n "$match" ]] || return 0
  run_pactl list short "$table" 2>/dev/null | awk -F '\t' -v needle="$match" -v skip_monitors="$skip_monitors" '
    function normalize(value) {
      value = tolower(value)
      gsub(/[_-]+/, " ", value)
      gsub(/[[:space:]]+/, " ", value)
      return value
    }

    {
      if (skip_monitors == "1" && $2 ~ /monitor/) {
        next
      }
      if (normalize($0) ~ normalize(needle)) {
        print $2
        exit
      }
    }
  '
}

run_pactl() {
  if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
    local runtime_dir
    runtime_dir="/run/user/$(id -u "$SUDO_USER")"
    sudo -u "$SUDO_USER" env XDG_RUNTIME_DIR="$runtime_dir" pactl "$@"
    return
  fi
  pactl "$@"
}

run_wpctl() {
  if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
    local runtime_dir
    runtime_dir="/run/user/$(id -u "$SUDO_USER")"
    sudo -u "$SUDO_USER" env XDG_RUNTIME_DIR="$runtime_dir" wpctl "$@"
    return
  fi
  wpctl "$@"
}

# Apply one bounded output volume to the selected PipeWire/Pulse sink so the
# active playback path does not inherit a stale near-muted level.
set_sink_volume_percent() {
  local sink_name="$1"
  local sink_id="$2"
  local percent="$3"

  if [[ -n "$sink_name" ]]; then
    run_pactl set-sink-volume "$sink_name" "${percent}%"
  fi
  if command -v wpctl >/dev/null 2>&1 && [[ -n "$sink_id" ]]; then
    run_wpctl set-volume "$sink_id" "${percent}%"
  fi
}

# Raise all playback-capable mixer controls on the selected card to one
# operator-visible level without touching capture-only controls.
set_card_playback_controls_percent() {
  local card_index="$1"
  local percent="$2"
  local line=""
  local control_name=""
  local control_index=""
  local control_ref=""
  local control_dump=""

  command -v amixer >/dev/null 2>&1 || fail "amixer not found"

  while IFS= read -r line; do
    [[ "$line" == "Simple mixer control "* ]] || continue
    control_name="${line#Simple mixer control \'}"
    control_name="${control_name%%\'*}"
    control_index="${line##*,}"
    control_ref="${control_name},${control_index}"
    control_dump="$(amixer -c "$card_index" sget "$control_ref" 2>/dev/null || true)"
    case "$control_dump" in
      *"Playback channels:"*|*"Capabilities: pvolume"*|*"Capabilities: pswitch"*|*"Capabilities: pvolume pswitch"*|*"Capabilities: pvolume pvolume-joined pswitch pswitch-joined"*)
        amixer -c "$card_index" sset "$control_ref" "${percent}%" unmute >/dev/null 2>&1 || true
        ;;
    esac
  done < <(amixer -c "$card_index" scontrols)
}

# Persist the selected ALSA card state after playback controls were normalized
# so later device re-enumeration or service restarts do not restore a stale,
# effectively muted mixer profile.
store_card_playback_state() {
  local card_index="$1"

  command -v alsactl >/dev/null 2>&1 || return 0
  alsactl store "$card_index" >/dev/null 2>&1 || true
}

set_softvol_control_percent() {
  local card_index="$1"
  local percent="$2"

  command -v amixer >/dev/null 2>&1 || fail "amixer not found"
  amixer -c "$card_index" sset "$SOFTVOL_CONTROL_NAME" "${percent}%" >/dev/null 2>&1 || true
}

prime_softvol_control() {
  # ALSA only exposes the new softvol mixer control after the routed playback
  # PCM is opened once. Feed one tiny block of silence through `default` so the
  # control materializes without producing an audible setup tone.
  head -c 3200 /dev/zero | aplay -q -D default -t raw -f S16_LE -c 1 -r 16000 - >/dev/null 2>&1 || true
}

detect_wpctl_id() {
  local table="$1"
  local match="$2"

  [[ -n "$match" ]] || return 0
  run_wpctl status 2>/dev/null | awk -v wanted_table="$table" -v needle="$match" '
    function normalize(value) {
      value = tolower(value)
      gsub(/[_-]+/, " ", value)
      gsub(/[[:space:]]+/, " ", value)
      return value
    }

    /^Audio$/ {
      in_audio = 1
      next
    }

    !in_audio {
      next
    }

    /Sinks:/ {
      current = "sinks"
      next
    }

    /Sources:/ {
      current = "sources"
      next
    }

    /Sink endpoints:/ || /Source endpoints:/ || /^Video$/ || /^Settings$/ {
      current = ""
      next
    }

    current == wanted_table && normalize($0) ~ normalize(needle) {
      for (i = 1; i <= NF; i++) {
        token = $i
        if (token == "*") {
          continue
        }
        if (token ~ /^[0-9]+\.$/) {
          sub(/\.$/, "", token)
          print token
          exit
        }
      }
    }
  '
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
    --capture-device-match)
      [[ $# -ge 2 ]] || fail "Missing value for --capture-device-match"
      CAPTURE_DEVICE_MATCH="$2"
      shift 2
      ;;
    --capture-card-index)
      [[ $# -ge 2 ]] || fail "Missing value for --capture-card-index"
      CAPTURE_CARD_INDEX="$2"
      shift 2
      ;;
    --capture-device-index)
      [[ $# -ge 2 ]] || fail "Missing value for --capture-device-index"
      CAPTURE_DEVICE_INDEX="$2"
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
    --sink-volume-percent)
      [[ $# -ge 2 ]] || fail "Missing value for --sink-volume-percent"
      SINK_VOLUME_PERCENT="$2"
      shift 2
      ;;
    --card-playback-volume-percent)
      [[ $# -ge 2 ]] || fail "Missing value for --card-playback-volume-percent"
      CARD_PLAYBACK_VOLUME_PERCENT="$2"
      shift 2
      ;;
    --softvol-max-db)
      [[ $# -ge 2 ]] || fail "Missing value for --softvol-max-db"
      SOFTVOL_MAX_DB="$2"
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
    --skip-playback-volume)
      SKIP_PLAYBACK_VOLUME=1
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
if [[ -z "$CAPTURE_DEVICE_INDEX" ]]; then
  CAPTURE_DEVICE_INDEX="$DEVICE_INDEX"
fi
[[ "$CAPTURE_DEVICE_INDEX" =~ ^[0-9]+$ ]] || fail "--capture-device-index must be an integer"
[[ "$PROACTIVE_DEVICE_INDEX" =~ ^[0-9]+$ ]] || fail "--proactive-device-index must be an integer"
if [[ -n "$PROACTIVE_SAMPLE_MS" ]]; then
  [[ "$PROACTIVE_SAMPLE_MS" =~ ^[0-9]+$ ]] || fail "--proactive-sample-ms must be an integer"
fi
[[ "$SINK_VOLUME_PERCENT" =~ ^[0-9]+$ ]] || fail "--sink-volume-percent must be an integer"
[[ "$CARD_PLAYBACK_VOLUME_PERCENT" =~ ^[0-9]+$ ]] || fail "--card-playback-volume-percent must be an integer"
python3 - <<PY || fail "--softvol-max-db must be a finite float between 0 and 30"
import math
value = float(r'''$SOFTVOL_MAX_DB''')
raise SystemExit(0 if math.isfinite(value) and 0.0 <= value <= 30.0 else 1)
PY
SOFTVOL_MAX_DB_RENDERED="$(python3 - <<PY
value = float(r'''$SOFTVOL_MAX_DB''')
print(f"{value:.1f}")
PY
)"
(( SINK_VOLUME_PERCENT >= 0 && SINK_VOLUME_PERCENT <= 150 )) || fail "--sink-volume-percent must be between 0 and 150"
(( CARD_PLAYBACK_VOLUME_PERCENT >= 0 && CARD_PLAYBACK_VOLUME_PERCENT <= 100 )) || fail "--card-playback-volume-percent must be between 0 and 100"

if [[ -z "$CARD_INDEX" ]]; then
  CARD_INDEX="$(detect_card_index aplay "$DEVICE_MATCH")"
fi

[[ -n "$CARD_INDEX" ]] || fail "Could not detect an ALSA playback card matching: $DEVICE_MATCH"
[[ "$CARD_INDEX" =~ ^[0-9]+$ ]] || fail "Resolved card index is not numeric: $CARD_INDEX"

if [[ -z "$CAPTURE_CARD_INDEX" ]]; then
  CAPTURE_CARD_INDEX="$(detect_card_index arecord "$CAPTURE_MATCH")"
fi
CAPTURE_CARD_INDEX="${CAPTURE_CARD_INDEX:-$CARD_INDEX}"
[[ "$CAPTURE_CARD_INDEX" =~ ^[0-9]+$ ]] || fail "Resolved capture card index is not numeric: $CAPTURE_CARD_INDEX"

detect_plughw_capture_device() {
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
  PROACTIVE_DEVICE="$(detect_plughw_capture_device "$PROACTIVE_DEVICE_MATCH" "$PROACTIVE_DEVICE_INDEX")"
fi

if [[ -n "$CAPTURE_MATCH" ]]; then
  CAPTURE_DEVICE="$(detect_plughw_capture_device "$CAPTURE_MATCH" "$CAPTURE_DEVICE_INDEX")"
fi

if [[ -n "$CAPTURE_DEVICE" || -n "$PROACTIVE_DEVICE" ]]; then
  mkdir -p "$(dirname "$ENV_FILE")"
  touch "$ENV_FILE"
  "$PYTHON_BIN" - <<PY
from pathlib import Path

path = Path(r'''$ENV_FILE''')
updates = {}
capture_device = r'''$CAPTURE_DEVICE'''.strip()
if capture_device:
    updates["TWINR_AUDIO_INPUT_DEVICE"] = capture_device
proactive_device = r'''$PROACTIVE_DEVICE'''.strip()
if proactive_device:
    updates["TWINR_PROACTIVE_AUDIO_ENABLED"] = "true"
    updates["TWINR_PROACTIVE_AUDIO_DEVICE"] = proactive_device
sample_ms = r'''$PROACTIVE_SAMPLE_MS'''.strip()
if proactive_device and sample_ms:
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
    backup_path="/etc/asound.conf.bak.$(date +%Y%m%d%H%M%S).$$"
    sudo cp /etc/asound.conf "$backup_path"
  fi
  if python3 - <<PY
import math
value = float(r'''$SOFTVOL_MAX_DB''')
raise SystemExit(0 if value > 0.0 else 1)
PY
  then
    sudo tee /etc/asound.conf >/dev/null <<ASOUND
pcm.twinr_playback_hw {
  type plug
  slave.pcm "hw:${CARD_INDEX},${DEVICE_INDEX}"
}

pcm.twinr_playback_softvol {
  type softvol
  slave.pcm "twinr_playback_hw"
  control {
    name "${SOFTVOL_CONTROL_NAME}"
    card ${CARD_INDEX}
  }
  min_dB -20.0
  max_dB ${SOFTVOL_MAX_DB_RENDERED}
  resolution 256
}

pcm.!default {
  type asym
  playback.pcm "twinr_playback_softvol"
  capture.pcm {
    type plug
    slave.pcm "hw:${CAPTURE_CARD_INDEX},${CAPTURE_DEVICE_INDEX}"
  }
}

ctl.!default {
  type hw
  card ${CARD_INDEX}
}
ASOUND
  else
    sudo tee /etc/asound.conf >/dev/null <<ASOUND
pcm.!default {
  type asym
  playback.pcm {
    type plug
    slave.pcm "hw:${CARD_INDEX},${DEVICE_INDEX}"
  }
  capture.pcm {
    type plug
    slave.pcm "hw:${CAPTURE_CARD_INDEX},${CAPTURE_DEVICE_INDEX}"
  }
}

ctl.!default {
  type hw
  card ${CARD_INDEX}
}
ASOUND
  fi
fi

SINK_NAME=""
SOURCE_NAME=""
WPCTL_SINK_ID=""
WPCTL_SOURCE_ID=""
SOURCE_MATCH="${CAPTURE_DEVICE_MATCH:-$DEVICE_MATCH}"
if command -v pactl >/dev/null 2>&1; then
  SINK_NAME="$(detect_pactl_name sinks "$DEVICE_MATCH")"
  SOURCE_NAME="$(detect_pactl_name sources "$SOURCE_MATCH" 1)"
fi
if command -v wpctl >/dev/null 2>&1; then
  WPCTL_SINK_ID="$(detect_wpctl_id sinks "$DEVICE_MATCH")"
  WPCTL_SOURCE_ID="$(detect_wpctl_id sources "$SOURCE_MATCH")"
fi

if [[ "$SKIP_PULSE" -eq 0 ]]; then
  command -v pactl >/dev/null 2>&1 || fail "pactl not found"
  [[ -n "$SINK_NAME" ]] || fail "Could not detect a Pulse/PipeWire sink matching: $DEVICE_MATCH"
  run_pactl set-default-sink "$SINK_NAME"
  if [[ -n "$SOURCE_NAME" ]]; then
    run_pactl set-default-source "$SOURCE_NAME"
  fi
  if command -v wpctl >/dev/null 2>&1; then
    [[ -n "$WPCTL_SINK_ID" ]] || fail "Could not detect a WirePlumber sink matching: $DEVICE_MATCH"
    run_wpctl set-default "$WPCTL_SINK_ID"
    if [[ -n "$WPCTL_SOURCE_ID" ]]; then
      run_wpctl set-default "$WPCTL_SOURCE_ID"
    fi
  fi
fi

if [[ "$SKIP_PLAYBACK_VOLUME" -eq 0 ]]; then
  if [[ "$SKIP_PULSE" -eq 0 ]]; then
    set_sink_volume_percent "$SINK_NAME" "$WPCTL_SINK_ID" "$SINK_VOLUME_PERCENT"
  fi
  set_card_playback_controls_percent "$CARD_INDEX" "$CARD_PLAYBACK_VOLUME_PERCENT"
  if python3 - <<PY
import math
value = float(r'''$SOFTVOL_MAX_DB''')
raise SystemExit(0 if value > 0.0 else 1)
PY
  then
    prime_softvol_control
    set_softvol_control_percent "$CARD_INDEX" 100
  fi
  store_card_playback_state "$CARD_INDEX"
fi

printf 'playback_card=%s\n' "$CARD_INDEX"
printf 'capture_card=%s\n' "$CAPTURE_CARD_INDEX"
printf 'playback_device=%s\n' "$DEVICE_INDEX"
printf 'capture_device=%s\n' "$CAPTURE_DEVICE_INDEX"
printf 'sink_volume_percent=%s\n' "$SINK_VOLUME_PERCENT"
printf 'card_playback_volume_percent=%s\n' "$CARD_PLAYBACK_VOLUME_PERCENT"
printf 'softvol_max_db=%s\n' "$SOFTVOL_MAX_DB"
if [[ -n "$SINK_NAME" ]]; then
  printf 'pulse_sink=%s\n' "$SINK_NAME"
fi
if [[ -n "$SOURCE_NAME" ]]; then
  printf 'pulse_source=%s\n' "$SOURCE_NAME"
else
  printf 'pulse_source=unavailable\n'
fi
if [[ -n "$WPCTL_SINK_ID" ]]; then
  printf 'wpctl_sink_id=%s\n' "$WPCTL_SINK_ID"
fi
if [[ -n "$WPCTL_SOURCE_ID" ]]; then
  printf 'wpctl_source_id=%s\n' "$WPCTL_SOURCE_ID"
else
  printf 'wpctl_source_id=unavailable\n'
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
    # Re-run the repo's runtime-faithful audio-perception path so recovery
    # checks prove semantic ReSpeaker facts, not only bare readable frames.
    PROACTIVE_AUDIO_PERCEPTION_OUTPUT="$(
      cd "$REPO_ROOT" && PYTHONPATH=src "$PYTHON_BIN" -m twinr --env-file "$ENV_FILE" --proactive-audio-observe-once
    )" || fail "Proactive audio perception sanity check failed for $PROACTIVE_DEVICE"
    printf '%s\n' "$PROACTIVE_AUDIO_PERCEPTION_OUTPUT"
    for required_key in \
      proactive_audio_room_context \
      proactive_non_speech_audio_likely \
      proactive_background_media_likely \
      proactive_device_directed_speech_candidate \
      proactive_audio_policy_runtime_alert_code
    do
      printf '%s\n' "$PROACTIVE_AUDIO_PERCEPTION_OUTPUT" | grep -q "^${required_key}=" \
        || fail "Proactive audio perception sanity output is missing ${required_key}"
    done
  fi
  rm -f /tmp/twinr_audio_test.wav /tmp/twinr_audio_capture.wav /tmp/twinr_proactive_audio_capture.wav
  printf 'audio_smoke_test=ok\n'
  if [[ -n "$PROACTIVE_DEVICE" ]]; then
    printf 'proactive_audio_smoke_test=ok\n'
    printf 'proactive_audio_perception_test=ok\n'
  fi
fi
