#!/usr/bin/env bash
# Configure the active Twinr display backend and optional smoke validation.
#
# HDMI deployments persist fullscreen Wayland or framebuffer output settings
# and remove the old Waveshare GPIO/SPI wiring from the runtime env.
# Waveshare deployments keep the legacy vendor-driver install path for the
# e-paper panel.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
DRIVER="hdmi_wayland"
FB_PATH="/dev/fb0"
WAYLAND_DISPLAY="wayland-0"
WAYLAND_RUNTIME_DIR=""
DISPLAY_LAYOUT=""
DISPLAY_RUNTIME_TRACE_ENABLED=""
VENDOR_DIR="${ROOT_DIR}/state/display/vendor"
PYTHON_BIN=""
SPI_BUS=0
SPI_DEVICE=0
CS_GPIO=8
DC_GPIO=25
RESET_GPIO=17
BUSY_GPIO=24
DISPLAY_WIDTH=""
DISPLAY_HEIGHT=""
ROTATION_DEGREES=""
SKIP_TEST=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --env-file PATH          Path to the Twinr .env file (default: ${ENV_FILE})
  --driver NAME            Display driver id: hdmi_wayland, hdmi_fbdev or waveshare_4in2_v2 (default: ${DRIVER})
  --fb-path PATH           Framebuffer path for hdmi_fbdev (default: ${FB_PATH})
  --wayland-display NAME   Wayland socket name for hdmi_wayland (default: ${WAYLAND_DISPLAY})
  --wayland-runtime-dir P  Runtime dir that contains the Wayland socket (auto-detected when omitted)
  --layout NAME            Display layout: default or debug_log (default: default for HDMI, keep current env otherwise)
  --runtime-trace BOOL     Enable verbose display runtime tracing: true or false (default: false for HDMI, keep current env otherwise)
  --width N                Optional logical display width; HDMI drivers auto-probe when omitted
  --height N               Optional logical display height; HDMI drivers auto-probe when omitted
  --rotation-degrees N     Optional display rotation; defaults to 0 for HDMI and 270 for Waveshare
  --vendor-dir PATH        Where to store the Waveshare vendor driver (default: ${VENDOR_DIR})
  --spi-bus N              SPI bus number for Waveshare (default: ${SPI_BUS})
  --spi-device N           SPI device number for Waveshare (default: ${SPI_DEVICE})
  --cs-gpio N              GPIO for Waveshare CS/CE (default: ${CS_GPIO})
  --dc-gpio N              GPIO for Waveshare DC (default: ${DC_GPIO})
  --reset-gpio N           GPIO for Waveshare RST (default: ${RESET_GPIO})
  --busy-gpio N            GPIO for Waveshare BUSY (default: ${BUSY_GPIO})
  --skip-test              Skip the final display smoke test
  --help                   Show this help text
EOF
}

resolve_python_bin() {
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Python 3.11+ not found. Create ${ROOT_DIR}/.venv or install python3.11." >&2
    exit 1
  fi
  "$PYTHON_BIN" - <<'PY' || { echo "Twinr requires Python 3.11+ for setup_display.sh" >&2; exit 1; }
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
}

probe_framebuffer_geometry() {
  local geometry
  geometry="$(
    FB_PATH="${FB_PATH}" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import fcntl
import os
import struct

FBIOGET_VSCREENINFO = 0x4600
path = Path(os.environ["FB_PATH"])
with path.open("rb", buffering=0) as framebuffer:
    vinfo = bytearray(160)
    fcntl.ioctl(framebuffer, FBIOGET_VSCREENINFO, vinfo, True)
    width, height = struct.unpack_from("2I", vinfo, 0)
    print(f"{int(width)} {int(height)}")
PY
  )"
  DISPLAY_WIDTH="${DISPLAY_WIDTH:-${geometry%% *}}"
  DISPLAY_HEIGHT="${DISPLAY_HEIGHT:-${geometry##* }}"
}

probe_wayland_runtime_dir() {
  if [[ -n "${WAYLAND_RUNTIME_DIR}" ]]; then
    return
  fi
  if [[ -n "${XDG_RUNTIME_DIR:-}" && -S "${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY}" ]]; then
    WAYLAND_RUNTIME_DIR="${XDG_RUNTIME_DIR}"
    return
  fi
  local detected_socket=""
  detected_socket="$(find /run/user -maxdepth 2 -type s -name "${WAYLAND_DISPLAY}" 2>/dev/null | sort | head -n 1 || true)"
  if [[ -n "${detected_socket}" ]]; then
    WAYLAND_RUNTIME_DIR="$(dirname "${detected_socket}")"
    return
  fi
  echo "Unable to locate Wayland socket ${WAYLAND_DISPLAY}. Pass --wayland-runtime-dir PATH." >&2
  exit 1
}

install_waveshare_driver() {
  mkdir -p "${VENDOR_DIR}/waveshare_epd"
  export ROOT_DIR VENDOR_DIR SPI_BUS SPI_DEVICE CS_GPIO DC_GPIO RESET_GPIO BUSY_GPIO
  apt-get update
  apt-get install -y python3-pil python3-spidev python3-gpiozero python3-lgpio python3-libgpiod
  "$PYTHON_BIN" hardware/display/vendor_patch.py \
    --vendor-dir "${VENDOR_DIR}" \
    --spi-bus "${SPI_BUS}" \
    --spi-device "${SPI_DEVICE}" \
    --cs-gpio "${CS_GPIO}" \
    --dc-gpio "${DC_GPIO}" \
    --reset-gpio "${RESET_GPIO}" \
    --busy-gpio "${BUSY_GPIO}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --driver)
      DRIVER="$2"
      shift 2
      ;;
    --fb-path)
      FB_PATH="$2"
      shift 2
      ;;
    --wayland-display)
      WAYLAND_DISPLAY="$2"
      shift 2
      ;;
    --wayland-runtime-dir)
      WAYLAND_RUNTIME_DIR="$2"
      shift 2
      ;;
    --layout)
      DISPLAY_LAYOUT="$2"
      shift 2
      ;;
    --runtime-trace)
      DISPLAY_RUNTIME_TRACE_ENABLED="$2"
      shift 2
      ;;
    --width)
      DISPLAY_WIDTH="$2"
      shift 2
      ;;
    --height)
      DISPLAY_HEIGHT="$2"
      shift 2
      ;;
    --rotation-degrees)
      ROTATION_DEGREES="$2"
      shift 2
      ;;
    --vendor-dir)
      VENDOR_DIR="$2"
      shift 2
      ;;
    --spi-bus)
      SPI_BUS="$2"
      shift 2
      ;;
    --spi-device)
      SPI_DEVICE="$2"
      shift 2
      ;;
    --cs-gpio)
      CS_GPIO="$2"
      shift 2
      ;;
    --dc-gpio)
      DC_GPIO="$2"
      shift 2
      ;;
    --reset-gpio)
      RESET_GPIO="$2"
      shift 2
      ;;
    --busy-gpio)
      BUSY_GPIO="$2"
      shift 2
      ;;
    --skip-test)
      SKIP_TEST=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

resolve_python_bin

case "${DRIVER}" in
  hdmi_wayland|hdmi_fbdev|waveshare_4in2_v2)
    ;;
  *)
    echo "Unsupported display driver: ${DRIVER}" >&2
    exit 1
    ;;
esac

if [[ -z "${ROTATION_DEGREES}" ]]; then
  if [[ "${DRIVER}" == hdmi_* ]]; then
    ROTATION_DEGREES=0
  else
    ROTATION_DEGREES=270
  fi
fi

if [[ -n "${DISPLAY_LAYOUT}" ]]; then
  case "${DISPLAY_LAYOUT}" in
    default|debug_log)
      ;;
    *)
      echo "Unsupported display layout: ${DISPLAY_LAYOUT}" >&2
      exit 1
      ;;
  esac
fi

if [[ -n "${DISPLAY_RUNTIME_TRACE_ENABLED}" ]]; then
  case "${DISPLAY_RUNTIME_TRACE_ENABLED}" in
    true|false)
      ;;
    *)
      echo "Unsupported display runtime trace flag: ${DISPLAY_RUNTIME_TRACE_ENABLED}" >&2
      exit 1
      ;;
  esac
fi

if [[ "${DRIVER}" == hdmi_* ]]; then
  probe_framebuffer_geometry
  DISPLAY_LAYOUT="${DISPLAY_LAYOUT:-default}"
  DISPLAY_RUNTIME_TRACE_ENABLED="${DISPLAY_RUNTIME_TRACE_ENABLED:-false}"
  if [[ "${DRIVER}" == "hdmi_wayland" ]]; then
    probe_wayland_runtime_dir
  fi
else
  DISPLAY_WIDTH="${DISPLAY_WIDTH:-400}"
  DISPLAY_HEIGHT="${DISPLAY_HEIGHT:-300}"
  install_waveshare_driver
fi

touch "${ENV_FILE}"

upsert_env() {
  local key="$1"
  local value="$2"
  KEY="$key" VALUE="$value" ENV_FILE="$ENV_FILE" "$PYTHON_BIN" <<'PY'
from pathlib import Path
import os

env_file = Path(os.environ["ENV_FILE"])
key = os.environ["KEY"]
value = os.environ["VALUE"]
lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
updated = False
new_lines = []
for line in lines:
    current_key, separator, _rest = line.partition("=")
    if separator and current_key == key:
        new_lines.append(f"{key}={value}")
        updated = True
    else:
        new_lines.append(line)
if not updated:
    new_lines.append(f"{key}={value}")
env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
PY
}

remove_env() {
  local key="$1"
  KEY="$key" ENV_FILE="$ENV_FILE" "$PYTHON_BIN" <<'PY'
from pathlib import Path
import os

env_file = Path(os.environ["ENV_FILE"])
key = os.environ["KEY"]
if not env_file.exists():
    raise SystemExit(0)
lines = env_file.read_text(encoding="utf-8").splitlines()
kept = [line for line in lines if not line.partition("=")[0] == key]
env_file.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
PY
}

upsert_env "TWINR_DISPLAY_DRIVER" "${DRIVER}"
upsert_env "TWINR_DISPLAY_WIDTH" "${DISPLAY_WIDTH}"
upsert_env "TWINR_DISPLAY_HEIGHT" "${DISPLAY_HEIGHT}"
upsert_env "TWINR_DISPLAY_ROTATION_DEGREES" "${ROTATION_DEGREES}"
if [[ -n "${DISPLAY_LAYOUT}" ]]; then
  upsert_env "TWINR_DISPLAY_LAYOUT" "${DISPLAY_LAYOUT}"
fi
if [[ -n "${DISPLAY_RUNTIME_TRACE_ENABLED}" ]]; then
  upsert_env "TWINR_DISPLAY_RUNTIME_TRACE_ENABLED" "${DISPLAY_RUNTIME_TRACE_ENABLED}"
fi

if [[ "${DRIVER}" == "hdmi_fbdev" ]]; then
  upsert_env "TWINR_DISPLAY_FB_PATH" "${FB_PATH}"
  remove_env "TWINR_DISPLAY_WAYLAND_DISPLAY"
  remove_env "TWINR_DISPLAY_WAYLAND_RUNTIME_DIR"
  remove_env "TWINR_DISPLAY_VENDOR_DIR"
  remove_env "TWINR_DISPLAY_SPI_BUS"
  remove_env "TWINR_DISPLAY_SPI_DEVICE"
  remove_env "TWINR_DISPLAY_CS_GPIO"
  remove_env "TWINR_DISPLAY_DC_GPIO"
  remove_env "TWINR_DISPLAY_RESET_GPIO"
  remove_env "TWINR_DISPLAY_BUSY_GPIO"
elif [[ "${DRIVER}" == "hdmi_wayland" ]]; then
  upsert_env "TWINR_DISPLAY_WAYLAND_DISPLAY" "${WAYLAND_DISPLAY}"
  upsert_env "TWINR_DISPLAY_WAYLAND_RUNTIME_DIR" "${WAYLAND_RUNTIME_DIR}"
  remove_env "TWINR_DISPLAY_FB_PATH"
  remove_env "TWINR_DISPLAY_VENDOR_DIR"
  remove_env "TWINR_DISPLAY_SPI_BUS"
  remove_env "TWINR_DISPLAY_SPI_DEVICE"
  remove_env "TWINR_DISPLAY_CS_GPIO"
  remove_env "TWINR_DISPLAY_DC_GPIO"
  remove_env "TWINR_DISPLAY_RESET_GPIO"
  remove_env "TWINR_DISPLAY_BUSY_GPIO"
else
  upsert_env "TWINR_DISPLAY_VENDOR_DIR" "$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os
root = Path(os.environ["ROOT_DIR"]).resolve()
vendor = Path(os.environ["VENDOR_DIR"]).resolve()
try:
    print(vendor.relative_to(root))
except ValueError:
    print(vendor)
PY
)"
  upsert_env "TWINR_DISPLAY_SPI_BUS" "${SPI_BUS}"
  upsert_env "TWINR_DISPLAY_SPI_DEVICE" "${SPI_DEVICE}"
  upsert_env "TWINR_DISPLAY_CS_GPIO" "${CS_GPIO}"
  upsert_env "TWINR_DISPLAY_DC_GPIO" "${DC_GPIO}"
  upsert_env "TWINR_DISPLAY_RESET_GPIO" "${RESET_GPIO}"
  upsert_env "TWINR_DISPLAY_BUSY_GPIO" "${BUSY_GPIO}"
  remove_env "TWINR_DISPLAY_FB_PATH"
  remove_env "TWINR_DISPLAY_WAYLAND_DISPLAY"
  remove_env "TWINR_DISPLAY_WAYLAND_RUNTIME_DIR"
fi

cd "${ROOT_DIR}"
if [[ -x "${ROOT_DIR}/.venv/bin/pip" ]]; then
  "${ROOT_DIR}/.venv/bin/pip" install -e .
else
  "$PYTHON_BIN" -m pip install -e .
fi

if [[ "${SKIP_TEST}" -eq 0 ]]; then
  timeout 120 "$PYTHON_BIN" hardware/display/display_test.py --env-file "${ENV_FILE}"
fi

echo "Display config written to ${ENV_FILE}"
if [[ "${DRIVER}" == "hdmi_fbdev" ]]; then
  echo "HDMI framebuffer configured at ${FB_PATH} (${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}, rotation ${ROTATION_DEGREES})"
elif [[ "${DRIVER}" == "hdmi_wayland" ]]; then
  echo "HDMI Wayland configured at ${WAYLAND_RUNTIME_DIR}/${WAYLAND_DISPLAY} (${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}, rotation ${ROTATION_DEGREES})"
else
  echo "Waveshare display driver installed in ${VENDOR_DIR}/waveshare_epd"
fi
