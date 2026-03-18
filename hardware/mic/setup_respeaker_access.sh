#!/usr/bin/env bash
#
# Configure a udev rule so the Twinr runtime user can read XVF3800 host-control
# over libusb without sudo. This is separate from ALSA/PipeWire routing on
# purpose: USB device-node permissions and audio default-device setup are
# different operational concerns.

set -euo pipefail

RULE_PATH="/etc/udev/rules.d/99-twinr-respeaker-xvf3800.rules"
USB_GROUP="audio"
USB_MODE="0660"
TRIGGER_RULES=1

usage() {
    cat <<'EOF'
Usage: setup_respeaker_access.sh [options]

Install or refresh the Twinr XVF3800 udev rule so the runtime user can read
ReSpeaker host-control primitives without sudo.

Options:
  --group NAME          Unix group that should own the USB node (default: audio)
  --mode MODE           Octal device mode to set on the USB node (default: 0660)
  --rule-path PATH      udev rule destination path
  --skip-trigger        Reload rules but do not trigger matching USB devices
  --help                Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --group)
            USB_GROUP="${2:-}"
            shift 2
            ;;
        --mode)
            USB_MODE="${2:-}"
            shift 2
            ;;
        --rule-path)
            RULE_PATH="${2:-}"
            shift 2
            ;;
        --skip-trigger)
            TRIGGER_RULES=0
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

if [[ -z "${USB_GROUP}" ]]; then
    echo "USB group must not be empty." >&2
    exit 2
fi

if ! [[ "${USB_MODE}" =~ ^0[0-7]{3}$ ]]; then
    echo "USB mode must look like an octal permission, e.g. 0660." >&2
    exit 2
fi

TMP_RULE="$(mktemp)"
trap 'rm -f "${TMP_RULE}"' EXIT

cat >"${TMP_RULE}" <<EOF
# Twinr XVF3800 USB access for libusb host-control reads.
SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="2886", ATTR{idProduct}=="001a", GROUP="${USB_GROUP}", MODE="${USB_MODE}"
EOF

install -D -m 0644 "${TMP_RULE}" "${RULE_PATH}"
udevadm control --reload-rules

if [[ ${TRIGGER_RULES} -eq 1 ]]; then
    udevadm trigger --subsystem-match=usb --attr-match=idVendor=2886 --attr-match=idProduct=001a >/dev/null 2>&1 || true
    udevadm settle || true
fi

echo "Installed XVF3800 access rule at ${RULE_PATH}"
echo "Configured USB ownership: group=${USB_GROUP} mode=${USB_MODE}"

if command -v lsusb >/dev/null 2>&1; then
    MATCHED_USB_ROWS="$(lsusb -d 2886:001a || true)"
    if [[ -n "${MATCHED_USB_ROWS}" ]]; then
        echo "Visible XVF3800 devices:"
        echo "${MATCHED_USB_ROWS}"
        while IFS= read -r row; do
            [[ -n "${row}" ]] || continue
            bus="$(awk '{print $2}' <<<"${row}")"
            device="$(awk '{gsub(/:/, "", $4); print $4}' <<<"${row}")"
            device_node="/dev/bus/usb/${bus}/${device}"
            if [[ -e "${device_node}" ]]; then
                ls -l "${device_node}"
            fi
        done <<<"${MATCHED_USB_ROWS}"
    else
        echo "No XVF3800 USB device is currently visible."
    fi
fi
