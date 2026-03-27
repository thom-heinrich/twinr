#!/usr/bin/env bash
set -euo pipefail

# Build the Twinr on-device Crazyflie failsafe app as an out-of-tree firmware
# module. The script prefers a local ARM toolchain when available and otherwise
# falls back to the official Bitcraze Toolbelt Docker image.

usage() {
  cat <<'EOF'
Usage:
  bash hardware/bitcraze/build_on_device_failsafe.sh [--firmware-root PATH] [--app-root PATH] [--use-docker]

Options:
  --firmware-root PATH   Crazyflie firmware checkout to build against.
                         Default: /tmp/crazyflie-firmware
  --app-root PATH        Twinr OOT app folder.
                         Default: hardware/bitcraze/twinr_on_device_failsafe
  --use-docker           Force the Dockerized Bitcraze Toolbelt build path.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FIRMWARE_ROOT="/tmp/crazyflie-firmware"
APP_ROOT="${REPO_ROOT}/hardware/bitcraze/twinr_on_device_failsafe"
USE_DOCKER=0

while (($# > 0)); do
  case "$1" in
    --firmware-root)
      FIRMWARE_ROOT="$2"
      shift 2
      ;;
    --app-root)
      APP_ROOT="$2"
      shift 2
      ;;
    --use-docker)
      USE_DOCKER=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "${FIRMWARE_ROOT}" ]]; then
  echo "firmware root not found: ${FIRMWARE_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${APP_ROOT}/Makefile" ]]; then
  echo "app root not found or incomplete: ${APP_ROOT}" >&2
  exit 1
fi

if [[ ${USE_DOCKER} -eq 0 ]] && command -v arm-none-eabi-gcc >/dev/null 2>&1; then
  make -C "${APP_ROOT}" CRAZYFLIE_BASE="${FIRMWARE_ROOT}" clean
  make -C "${APP_ROOT}" CRAZYFLIE_BASE="${FIRMWARE_ROOT}" cf2_defconfig
  make -C "${APP_ROOT}" CRAZYFLIE_BASE="${FIRMWARE_ROOT}" -j"$(nproc)"
else
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required when arm-none-eabi-gcc is unavailable" >&2
    exit 1
  fi
  docker run --rm \
    -v "${FIRMWARE_ROOT}:/workspace/crazyflie-firmware" \
    -v "${APP_ROOT}:/workspace/app" \
    -w /workspace/app \
    --entrypoint /bin/sh \
    bitcraze/toolbelt:latest \
    -lc "make CRAZYFLIE_BASE=/workspace/crazyflie-firmware clean && make CRAZYFLIE_BASE=/workspace/crazyflie-firmware cf2_defconfig && make CRAZYFLIE_BASE=/workspace/crazyflie-firmware -j\$(nproc)"
fi

if [[ ! -f "${APP_ROOT}/build/cf2.bin" ]]; then
  echo "build finished without build/cf2.bin under ${APP_ROOT}" >&2
  exit 1
fi

cp "${APP_ROOT}/build/cf2.bin" "${APP_ROOT}/build/twinr_on_device_failsafe.bin"
echo "built=${APP_ROOT}/build/twinr_on_device_failsafe.bin"
