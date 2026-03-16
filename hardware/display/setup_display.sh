#!/usr/bin/env bash
# Configure the Waveshare display driver files and Twinr env wiring.
#
# Installs the vendor driver into a runtime-owned directory, persists display
# GPIO/SPI settings into the selected `.env`, and can run a bounded smoke test.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
VENDOR_DIR="${ROOT_DIR}/state/display/vendor"
PYTHON_BIN=""
SPI_BUS=0
SPI_DEVICE=0
CS_GPIO=8
DC_GPIO=25
RESET_GPIO=17
BUSY_GPIO=24
SKIP_TEST=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --env-file PATH       Path to the Twinr .env file (default: ${ENV_FILE})
  --vendor-dir PATH     Where to store the Waveshare vendor driver (default: ${VENDOR_DIR})
  --spi-bus N           SPI bus number (default: ${SPI_BUS})
  --spi-device N        SPI device number (default: ${SPI_DEVICE})
  --cs-gpio N           GPIO for CS/CE (default: ${CS_GPIO})
  --dc-gpio N           GPIO for DC (default: ${DC_GPIO})
  --reset-gpio N        GPIO for RST (default: ${RESET_GPIO})
  --busy-gpio N         GPIO for BUSY (default: ${BUSY_GPIO})
  --skip-test           Skip the final display smoke test
  --help                Show this help text
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
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

mkdir -p "${VENDOR_DIR}/waveshare_epd"
resolve_python_bin

export ROOT_DIR ENV_FILE VENDOR_DIR SPI_BUS SPI_DEVICE CS_GPIO DC_GPIO RESET_GPIO BUSY_GPIO

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

touch "${ENV_FILE}"

upsert_env() {
  local key="$1"
  local value="$2"
  KEY="$key" VALUE="$value" "$PYTHON_BIN" <<'PY'
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

upsert_env "TWINR_DISPLAY_DRIVER" "waveshare_4in2_v2"
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
upsert_env "TWINR_DISPLAY_WIDTH" "400"
upsert_env "TWINR_DISPLAY_HEIGHT" "300"

cd "${ROOT_DIR}"
if [[ -x "${ROOT_DIR}/.venv/bin/pip" ]]; then
  "${ROOT_DIR}/.venv/bin/pip" install -e .
else
  "$PYTHON_BIN" -m pip install -e .
fi

if [[ "${SKIP_TEST}" -eq 0 ]]; then
  timeout 120 "$PYTHON_BIN" hardware/display/display_test.py --env-file "${ENV_FILE}"
fi

echo "Display driver installed in ${VENDOR_DIR}/waveshare_epd"
echo "Display config written to ${ENV_FILE}"
