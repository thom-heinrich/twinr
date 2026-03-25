#!/usr/bin/env bash
#
# Install the official Pololu udev rule and configure one connected Maestro to
# USB_DUAL_PORT so Twinr's USB serial runtime path talks to the command
# processor instead of the factory-default UART loopback path.

set -euo pipefail

RULE_PATH="/etc/udev/rules.d/99-pololu.rules"
DOWNLOAD_URL="https://www.pololu.com/file/0J315/maestro-linux-241004.tar.gz"
DEVICE_SERIAL=""
TRIGGER_RULES=1

usage() {
    cat <<'EOF'
Usage: setup_pololu_maestro.sh [options]

Install the official Pololu USB udev rule and configure one connected Maestro
to SerialMode=USB_DUAL_PORT so Twinr's command-port writer can control it.

Options:
  --device SERIAL        Target one Maestro serial number when multiple are attached
  --rule-path PATH       Destination path for the Pololu udev rule
  --download-url URL     Override the official Pololu Linux package URL
  --skip-trigger         Reload rules but do not trigger matching USB devices
  --help                 Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE_SERIAL="${2:-}"
            shift 2
            ;;
        --rule-path)
            RULE_PATH="${2:-}"
            shift 2
            ;;
        --download-url)
            DOWNLOAD_URL="${2:-}"
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

for required_binary in curl install python3 tar udevadm; do
    if ! command -v "${required_binary}" >/dev/null 2>&1; then
        echo "Required binary not found: ${required_binary}" >&2
        exit 1
    fi
done

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

cat >"${TMP_DIR}/99-pololu.rules" <<'EOF'
# Official Pololu USB access rule for native Maestro diagnostics and configuration.
SUBSYSTEM=="usb", ATTRS{idVendor}=="1ffb", MODE="0666"
EOF

install -D -m 0644 "${TMP_DIR}/99-pololu.rules" "${RULE_PATH}"
udevadm control --reload-rules
if [[ ${TRIGGER_RULES} -eq 1 ]]; then
    udevadm trigger --subsystem-match=usb --attr-match=idVendor=1ffb >/dev/null 2>&1 || true
    udevadm settle || true
fi

curl -fsSL "${DOWNLOAD_URL}" -o "${TMP_DIR}/maestro-linux.tar.gz"
tar -xzf "${TMP_DIR}/maestro-linux.tar.gz" -C "${TMP_DIR}"

USC_PATH="${TMP_DIR}/maestro-linux/UscCmd"
if [[ ! -e "${USC_PATH}" ]]; then
    echo "Pololu Linux package did not contain UscCmd at ${USC_PATH}" >&2
    exit 1
fi

if "${USC_PATH}" --help >/dev/null 2>&1; then
    USC_RUN=("${USC_PATH}")
elif command -v mono >/dev/null 2>&1; then
    USC_RUN=(mono "${USC_PATH}")
else
    echo "UscCmd requires mono on this system, but no mono binary was found." >&2
    exit 1
fi

DEVICE_ARGS=()
if [[ -n "${DEVICE_SERIAL}" ]]; then
    DEVICE_ARGS=(--device "${DEVICE_SERIAL}")
fi

CONFIG_PATH="${TMP_DIR}/maestro.cfg"
VERIFY_PATH="${TMP_DIR}/maestro-verify.cfg"

"${USC_RUN[@]}" "${DEVICE_ARGS[@]}" --getconf "${CONFIG_PATH}"

CURRENT_MODE="$(python3 - "${CONFIG_PATH}" <<'PY'
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

config_path = Path(sys.argv[1])
root = ET.parse(config_path).getroot()
serial_mode = root.findtext("SerialMode", default="UNKNOWN").strip()
print(serial_mode)
PY
)"

echo "Current Pololu Maestro serial mode: ${CURRENT_MODE}"

if [[ "${CURRENT_MODE}" != "USB_DUAL_PORT" ]]; then
    python3 - "${CONFIG_PATH}" <<'PY'
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

config_path = Path(sys.argv[1])
tree = ET.parse(config_path)
root = tree.getroot()
serial_mode = root.find("SerialMode")
if serial_mode is None:
    raise SystemExit("Pololu Maestro config file is missing the SerialMode field.")
serial_mode.text = "USB_DUAL_PORT"
tree.write(config_path, encoding="utf-8", xml_declaration=False)
PY
    "${USC_RUN[@]}" "${DEVICE_ARGS[@]}" --configure "${CONFIG_PATH}"
fi

"${USC_RUN[@]}" "${DEVICE_ARGS[@]}" --getconf "${VERIFY_PATH}"

VERIFIED_MODE="$(python3 - "${VERIFY_PATH}" <<'PY'
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

config_path = Path(sys.argv[1])
root = ET.parse(config_path).getroot()
serial_mode = root.findtext("SerialMode", default="UNKNOWN").strip()
print(serial_mode)
PY
)"

if [[ "${VERIFIED_MODE}" != "USB_DUAL_PORT" ]]; then
    echo "Failed to switch Pololu Maestro into USB_DUAL_PORT; current mode is ${VERIFIED_MODE}." >&2
    exit 1
fi

echo "Configured Pololu Maestro serial mode: ${VERIFIED_MODE}"
"${USC_RUN[@]}" "${DEVICE_ARGS[@]}" --status
