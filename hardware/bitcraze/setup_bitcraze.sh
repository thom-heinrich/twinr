#!/usr/bin/env bash
#
# Provision a clean Crazyradio/Crazyflie workspace under /twinr/bitcraze.
# The setup stays separate from Twinr runtime logic on purpose: USB access,
# firmware staging, and vendor Python tooling are operational concerns.

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORKSPACE="/twinr/bitcraze"
RUNTIME_USER="thh"
USB_GROUP="plugdev"
TTY_GROUP="dialout"
USB_MODE="0660"
TTY_MODE="0660"
RULE_PATH="/etc/udev/rules.d/99-bitcraze-crazyradio.rules"
PYTHON_BIN="python3"
FLASH_MODE="emulation"
TRIGGER_RULES=1
PROBE_AFTER_SETUP=1
INSTALL_CFCLIENT=1

CFLIB_VERSION="0.1.31"
CFCLIENT_VERSION="2025.12.1"
CRPA_EMULATION_URL="https://github.com/bitcraze/crazyradio2-firmware/releases/download/1.1/crazyradio2-CRPA-emulation-1.1.uf2"
CRPA_EMULATION_SHA256="02b0a496c93800b691fda6694e7dd480e7aed0a0420313899ce3e7c7d2f2d6f6"
CRAZYRADIO2_NATIVE_URL="https://github.com/bitcraze/crazyradio2-firmware/releases/download/5.4/crazyradio2-5.4.uf2"
CRAZYRADIO2_NATIVE_SHA256="32dd735996f744a616a513da7a180c42187f2056de16eee050c19933510aaa03"

MOUNTED_BY_SCRIPT=""
TEMP_MOUNT=""

usage() {
    cat <<'EOF'
Usage: setup_bitcraze.sh [options]

Install the Crazyradio access rules, stage pinned Bitcraze firmware, create a
workspace under /twinr/bitcraze, and validate cflib/cfclient access.

Options:
  --workspace PATH         Workspace root to create (default: /twinr/bitcraze)
  --runtime-user USER      User that should own and use the workspace
                           (default: thh)
  --usb-group GROUP        Group for Crazyradio USB device access (default: plugdev)
  --tty-group GROUP        Group for Crazyradio ACM serial access (default: dialout)
  --rule-path PATH         Destination path for the udev rule
  --python PATH            Python interpreter for the workspace venv
                           (default: python3)
  --flash-mode MODE        Firmware target: emulation, native, or none
                           (default: emulation)
  --skip-trigger           Reload udev rules but do not trigger matching devices
  --skip-probe             Skip the final probe run
  --skip-cfclient          Install cflib only
  --help                   Show this help text
EOF
}

cleanup() {
    if [[ -n "${MOUNTED_BY_SCRIPT}" ]] && mountpoint -q "${MOUNTED_BY_SCRIPT}" 2>/dev/null; then
        umount "${MOUNTED_BY_SCRIPT}" >/dev/null 2>&1 || true
    fi
}

trap cleanup EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace)
            WORKSPACE="${2:-}"
            shift 2
            ;;
        --runtime-user)
            RUNTIME_USER="${2:-}"
            shift 2
            ;;
        --usb-group)
            USB_GROUP="${2:-}"
            shift 2
            ;;
        --tty-group)
            TTY_GROUP="${2:-}"
            shift 2
            ;;
        --rule-path)
            RULE_PATH="${2:-}"
            shift 2
            ;;
        --python)
            PYTHON_BIN="${2:-}"
            shift 2
            ;;
        --flash-mode)
            FLASH_MODE="${2:-}"
            shift 2
            ;;
        --skip-trigger)
            TRIGGER_RULES=0
            shift
            ;;
        --skip-probe)
            PROBE_AFTER_SETUP=0
            shift
            ;;
        --skip-cfclient)
            INSTALL_CFCLIENT=0
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ${EUID} -ne 0 ]]; then
    echo "This script must run as root, e.g. via sudo." >&2
    exit 1
fi

if [[ "${FLASH_MODE}" != "emulation" && "${FLASH_MODE}" != "native" && "${FLASH_MODE}" != "none" ]]; then
    echo "Flash mode must be one of: emulation, native, none." >&2
    exit 2
fi

for required_binary in curl install "${PYTHON_BIN}" sha256sum udevadm lsusb runuser; do
    if ! command -v "${required_binary}" >/dev/null 2>&1; then
        echo "Required binary not found: ${required_binary}" >&2
        exit 1
    fi
done

if ! id -u "${RUNTIME_USER}" >/dev/null 2>&1; then
    echo "Runtime user not found: ${RUNTIME_USER}" >&2
    exit 1
fi

ensure_group_exists() {
    local group_name="$1"
    if ! getent group "${group_name}" >/dev/null 2>&1; then
        groupadd "${group_name}"
    fi
}

ensure_user_in_group() {
    local user_name="$1"
    local group_name="$2"

    if id -nG "${user_name}" | tr ' ' '\n' | grep -Fxq "${group_name}"; then
        return 0
    fi

    usermod -a -G "${group_name}" "${user_name}"
}

ensure_group_exists "${USB_GROUP}"
ensure_group_exists "${TTY_GROUP}"
ensure_user_in_group "${RUNTIME_USER}" "${USB_GROUP}"
ensure_user_in_group "${RUNTIME_USER}" "${TTY_GROUP}"

TMP_RULE="$(mktemp)"
trap 'rm -f "${TMP_RULE}"; cleanup' EXIT

cat >"${TMP_RULE}" <<EOF
# Twinr Bitcraze Crazyradio access for both PA emulation and Crazyradio 2.0 UF2/native states.
SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="1915", ATTR{idProduct}=="7777", GROUP="${USB_GROUP}", MODE="${USB_MODE}"
SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="35f0", ATTR{idProduct}=="bad2", GROUP="${USB_GROUP}", MODE="${USB_MODE}"
KERNEL=="ttyACM*", ATTRS{idVendor}=="35f0", ATTRS{idProduct}=="bad2", GROUP="${TTY_GROUP}", MODE="${TTY_MODE}"
EOF

install -D -m 0644 "${TMP_RULE}" "${RULE_PATH}"
udevadm control --reload-rules

if [[ ${TRIGGER_RULES} -eq 1 ]]; then
    udevadm trigger --subsystem-match=usb --attr-match=idVendor=1915 --attr-match=idProduct=7777 >/dev/null 2>&1 || true
    udevadm trigger --subsystem-match=usb --attr-match=idVendor=35f0 --attr-match=idProduct=bad2 >/dev/null 2>&1 || true
    udevadm trigger --subsystem-match=tty --attr-match=idVendor=35f0 --attr-match=idProduct=bad2 >/dev/null 2>&1 || true
    udevadm settle || true
fi

install -d -m 0775 -o "${RUNTIME_USER}" -g "${RUNTIME_USER}" \
    "${WORKSPACE}" \
    "${WORKSPACE}/firmware" \
    "${WORKSPACE}/firmware/backups"

download_and_verify() {
    local url="$1"
    local destination="$2"
    local expected_sha="$3"

    if [[ -f "${destination}" ]]; then
        local existing_sha
        existing_sha="$(sha256sum "${destination}" | awk '{print $1}')"
        if [[ "${existing_sha}" == "${expected_sha}" ]]; then
            return 0
        fi
    fi

    curl -fsSL "${url}" -o "${destination}"
    local actual_sha
    actual_sha="$(sha256sum "${destination}" | awk '{print $1}')"
    if [[ "${actual_sha}" != "${expected_sha}" ]]; then
        echo "Downloaded file checksum mismatch for ${destination}" >&2
        echo "Expected: ${expected_sha}" >&2
        echo "Actual:   ${actual_sha}" >&2
        exit 1
    fi
}

download_and_verify \
    "${CRPA_EMULATION_URL}" \
    "${WORKSPACE}/firmware/crazyradio2-CRPA-emulation-1.1.uf2" \
    "${CRPA_EMULATION_SHA256}"
download_and_verify \
    "${CRAZYRADIO2_NATIVE_URL}" \
    "${WORKSPACE}/firmware/crazyradio2-5.4.uf2" \
    "${CRAZYRADIO2_NATIVE_SHA256}"

detect_mode() {
    if lsusb -d 1915:7777 >/dev/null 2>&1; then
        echo "pa_emulation"
        return 0
    fi
    if lsusb -d 35f0:bad2 >/dev/null 2>&1; then
        if [[ -e /dev/disk/by-label/Crazyradio2 ]]; then
            echo "uf2_bootloader"
        else
            echo "crazyradio2_native"
        fi
        return 0
    fi
    echo "not_found"
}

ensure_crazyradio_mount() {
    if [[ ! -e /dev/disk/by-label/Crazyradio2 ]]; then
        return 1
    fi

    local block_device
    block_device="$(readlink -f /dev/disk/by-label/Crazyradio2)"
    local mountpoint
    mountpoint="$(lsblk -nrpo MOUNTPOINT "${block_device}" | sed -n '1p')"

    if [[ -n "${mountpoint}" ]]; then
        echo "${mountpoint}"
        return 0
    fi

    TEMP_MOUNT="${WORKSPACE}/firmware/mounted-uf2"
    install -d -m 0775 -o "${RUNTIME_USER}" -g "${RUNTIME_USER}" "${TEMP_MOUNT}"
    mount "${block_device}" "${TEMP_MOUNT}"
    MOUNTED_BY_SCRIPT="${TEMP_MOUNT}"
    echo "${TEMP_MOUNT}"
    return 0
}

backup_current_firmware() {
    local mountpoint="$1"
    local timestamp
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    local backup_dir="${WORKSPACE}/firmware/backups/${timestamp}"
    install -d -m 0775 -o "${RUNTIME_USER}" -g "${RUNTIME_USER}" "${backup_dir}"
    for candidate in CURRENT.UF2 INFO_UF2.TXT README.TXT README.HTM; do
        if [[ -f "${mountpoint}/${candidate}" ]]; then
            cp "${mountpoint}/${candidate}" "${backup_dir}/${candidate}"
        fi
    done
    chown -R "${RUNTIME_USER}:${RUNTIME_USER}" "${backup_dir}"
}

wait_for_usb_id() {
    local usb_id="$1"
    local timeout_s="$2"

    local elapsed=0
    while [[ ${elapsed} -lt ${timeout_s} ]]; do
        if lsusb -d "${usb_id}" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

flash_firmware_if_needed() {
    local current_mode="$1"

    if [[ "${FLASH_MODE}" == "none" ]]; then
        echo "Skipping firmware flash by request."
        return 0
    fi

    if [[ "${FLASH_MODE}" == "emulation" && "${current_mode}" == "pa_emulation" ]]; then
        echo "Crazyradio is already in PA emulation mode."
        return 0
    fi

    local mountpoint
    if ! mountpoint="$(ensure_crazyradio_mount)"; then
        echo "Crazyradio UF2 volume not available. Reinsert the dongle in UF2 mode if you want to flash firmware." >&2
        exit 1
    fi

    backup_current_firmware "${mountpoint}"

    local firmware_path expected_usb_id
    if [[ "${FLASH_MODE}" == "emulation" ]]; then
        firmware_path="${WORKSPACE}/firmware/crazyradio2-CRPA-emulation-1.1.uf2"
        expected_usb_id="1915:7777"
    else
        firmware_path="${WORKSPACE}/firmware/crazyradio2-5.4.uf2"
        expected_usb_id="35f0:bad2"
    fi

    cp "${firmware_path}" "${mountpoint}/$(basename "${firmware_path}")"
    sync
    sleep 2

    if [[ -n "${MOUNTED_BY_SCRIPT}" ]] && mountpoint -q "${MOUNTED_BY_SCRIPT}" 2>/dev/null; then
        umount "${MOUNTED_BY_SCRIPT}" >/dev/null 2>&1 || true
        MOUNTED_BY_SCRIPT=""
    fi

    if ! wait_for_usb_id "${expected_usb_id}" 20; then
        echo "Expected Crazyradio USB id ${expected_usb_id} did not appear after flashing." >&2
        exit 1
    fi

    echo "Flashed firmware: $(basename "${firmware_path}")"
}

ensure_workspace_venv() {
    if [[ ! -x "${WORKSPACE}/.venv/bin/python" || ! -x "${WORKSPACE}/.venv/bin/pip" ]]; then
        runuser -u "${RUNTIME_USER}" -- "${PYTHON_BIN}" -m venv --clear "${WORKSPACE}/.venv"
    fi

    runuser -u "${RUNTIME_USER}" -- "${WORKSPACE}/.venv/bin/pip" install --upgrade pip setuptools wheel

    local pip_packages=("cflib==${CFLIB_VERSION}")
    if [[ ${INSTALL_CFCLIENT} -eq 1 ]]; then
        pip_packages+=("cfclient==${CFCLIENT_VERSION}")
    fi

    runuser -u "${RUNTIME_USER}" -- "${WORKSPACE}/.venv/bin/pip" install "${pip_packages[@]}"
    printf "%s\n" "${pip_packages[@]}" >"${WORKSPACE}/requirements.lock.txt"
    chown "${RUNTIME_USER}:${RUNTIME_USER}" "${WORKSPACE}/requirements.lock.txt"
}

write_workspace_readme() {
    cat >"${WORKSPACE}/README.md" <<EOF
# bitcraze workspace

This workspace was provisioned by \`hardware/bitcraze/setup_bitcraze.sh\`.

## Pinned Python packages

- \`cflib==${CFLIB_VERSION}\`
EOF

    if [[ ${INSTALL_CFCLIENT} -eq 1 ]]; then
        cat >>"${WORKSPACE}/README.md" <<EOF
- \`cfclient==${CFCLIENT_VERSION}\`
EOF
    fi

    cat >>"${WORKSPACE}/README.md" <<EOF

## Firmware assets

- \`firmware/crazyradio2-CRPA-emulation-1.1.uf2\`
- \`firmware/crazyradio2-5.4.uf2\`

The default Twinr-side Python setup expects PA emulation because the current
official \`cflib\` path still opens the classic Crazyradio PA USB identity.

## Useful commands

\`\`\`bash
${WORKSPACE}/.venv/bin/python ${SELF_DIR}/probe_crazyradio.py --workspace ${WORKSPACE}
\`\`\`
EOF

    if [[ ${INSTALL_CFCLIENT} -eq 1 ]]; then
        cat >>"${WORKSPACE}/README.md" <<EOF

\`\`\`bash
${WORKSPACE}/.venv/bin/cfclient
\`\`\`
EOF
    fi

    chown "${RUNTIME_USER}:${RUNTIME_USER}" "${WORKSPACE}/README.md"
}

CURRENT_MODE="$(detect_mode)"
echo "Detected Crazyradio mode before setup: ${CURRENT_MODE}"

flash_firmware_if_needed "${CURRENT_MODE}"
ensure_workspace_venv
write_workspace_readme

if [[ ${PROBE_AFTER_SETUP} -eq 1 ]]; then
    probe_args=("${PYTHON_BIN}" "${SELF_DIR}/probe_crazyradio.py" "--workspace" "${WORKSPACE}")
    if [[ "${FLASH_MODE}" == "emulation" ]]; then
        probe_args+=("--expect-mode" "pa_emulation" "--require-cflib-access")
    fi
    runuser -u "${RUNTIME_USER}" -- "${probe_args[@]}"
fi

echo "Bitcraze workspace ready at ${WORKSPACE}"
echo "Installed udev rule: ${RULE_PATH}"
echo "Runtime user group access: ${RUNTIME_USER} -> ${USB_GROUP}, ${TTY_GROUP}"
