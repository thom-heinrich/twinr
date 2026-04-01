#!/usr/bin/env bash
#
# Install targeted snd-usb-audio overrides for the Seeed ReSpeaker XVF3800 so
# ALSA registration waits for the last audio interface before the card becomes
# visible to the rest of the audio stack.

set -euo pipefail

CONF_PATH="/etc/modprobe.d/99-twinr-respeaker-xvf3800-audio.conf"
DELAYED_REGISTER_SPEC="2886001a:2"

usage() {
    cat <<'EOF'
Usage: setup_respeaker_stability.sh [options]

Install or refresh Twinr's targeted snd-usb-audio overrides for the Seeed
ReSpeaker XVF3800 on the Raspberry Pi.

Options:
  --conf-path PATH            Destination modprobe file
  --delayed-register SPEC     snd-usb-audio delayed_register selector (default: 2886001a:2)
  --help                      Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conf-path)
            CONF_PATH="${2:-}"
            shift 2
            ;;
        --delayed-register)
            DELAYED_REGISTER_SPEC="${2:-}"
            shift 2
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

if [[ -z "${DELAYED_REGISTER_SPEC}" ]]; then
    echo "Delayed-register spec must be non-empty." >&2
    exit 2
fi

TMP_CONF="$(mktemp)"
trap 'rm -f "${TMP_CONF}"' EXIT

cat >"${TMP_CONF}" <<EOF
# Twinr XVF3800 stability overrides:
# - delayed_register waits until the last audio interface is present before
#   ALSA registers the card, avoiding partial multi-interface enumeration.
# - the current Pi kernel accepts delayed_register but rejects the documented
#   per-device quirk_flags syntax at module-load time, so this helper keeps the
#   safe registration workaround only.
options snd-usb-audio delayed_register=${DELAYED_REGISTER_SPEC}
EOF

install -D -m 0644 "${TMP_CONF}" "${CONF_PATH}"

echo "Installed XVF3800 stability overrides at ${CONF_PATH}"
echo "Configured delayed_register=${DELAYED_REGISTER_SPEC}"
echo "Reload snd-usb-audio or reboot the Pi for the new module options to take effect."
