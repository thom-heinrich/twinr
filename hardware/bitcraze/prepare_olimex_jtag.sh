#!/usr/bin/env bash
#
# Prepare this host for AI-Deck GAP8 JTAG recovery with an Olimex ARM-USB-TINY-H.
# The script is intentionally conservative: by default it only checks/stages the
# workspace and reports missing prerequisites. Package installation and group
# changes are opt-in.

set -euo pipefail

WORKSPACE="/twinr/bitcraze"
RUNTIME_USER="thh"
INSTALL_APT=0
ENSURE_USER_GROUPS=0
PREPARE_DOCKER=1

usage() {
    cat <<'EOF'
Usage: prepare_olimex_jtag.sh [options]

Check or prepare the local host for AI-Deck GAP8 JTAG recovery with an
Olimex ARM-USB-TINY-H bundle.

Options:
  --workspace PATH         Workspace root to stage recovery notes under
                           (default: /twinr/bitcraze)
  --runtime-user USER      User that should be able to access prepared tools
                           (default: thh)
  --install-apt            Install missing Debian/Ubuntu packages
                           (`openocd` and optionally `docker.io`)
  --ensure-user-groups     Add the runtime user to `plugdev`, `dialout`,
                           and `docker` when available
  --skip-docker            Do not require or install Docker
  --help                   Show this help text
EOF
}

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
        --install-apt)
            INSTALL_APT=1
            shift
            ;;
        --ensure-user-groups)
            ENSURE_USER_GROUPS=1
            shift
            ;;
        --skip-docker)
            PREPARE_DOCKER=0
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

if [[ ${INSTALL_APT} -eq 1 || ${ENSURE_USER_GROUPS} -eq 1 ]]; then
    if [[ ${EUID} -ne 0 ]]; then
        echo "This operation needs root. Re-run via sudo to install packages or modify groups." >&2
        exit 1
    fi
fi

command_status() {
    local binary="$1"
    if command -v "${binary}" >/dev/null 2>&1; then
        echo "present"
    else
        echo "missing"
    fi
}

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

install_packages_if_requested() {
    if [[ ${INSTALL_APT} -ne 1 ]]; then
        return 0
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "apt-get is required for --install-apt on this host." >&2
        exit 1
    fi
    local packages=(openocd)
    if [[ ${PREPARE_DOCKER} -eq 1 ]]; then
        packages+=(docker.io)
    fi
    apt-get update
    apt-get install -y "${packages[@]}"
    if [[ ${PREPARE_DOCKER} -eq 1 ]] && command -v systemctl >/dev/null 2>&1; then
        systemctl enable --now docker >/dev/null 2>&1 || true
    fi
}

install -d -m 0775 "${WORKSPACE}" "${WORKSPACE}/recovery" "${WORKSPACE}/recovery/olimex"

install_packages_if_requested

if [[ ${ENSURE_USER_GROUPS} -eq 1 ]]; then
    if ! id -u "${RUNTIME_USER}" >/dev/null 2>&1; then
        echo "Runtime user not found: ${RUNTIME_USER}" >&2
        exit 1
    fi
    ensure_group_exists "plugdev"
    ensure_group_exists "dialout"
    ensure_user_in_group "${RUNTIME_USER}" "plugdev"
    ensure_user_in_group "${RUNTIME_USER}" "dialout"
    if [[ ${PREPARE_DOCKER} -eq 1 ]] && getent group docker >/dev/null 2>&1; then
        ensure_user_in_group "${RUNTIME_USER}" "docker"
    fi
fi

OPENOCD_STATUS="$(command_status openocd)"
DOCKER_STATUS="skipped"
if [[ ${PREPARE_DOCKER} -eq 1 ]]; then
    DOCKER_STATUS="$(command_status docker)"
fi
LSUSB_STATUS="$(command_status lsusb)"
GIT_STATUS="$(command_status git)"
OLIMEX_MATCHES="$(lsusb | grep -i 'olimex\|arm-usb-tiny' || true)"

SUMMARY_PATH="${WORKSPACE}/recovery/olimex/host_prep_status.txt"
cat >"${SUMMARY_PATH}" <<EOF
workspace=${WORKSPACE}
runtime_user=${RUNTIME_USER}
openocd=${OPENOCD_STATUS}
docker=${DOCKER_STATUS}
lsusb=${LSUSB_STATUS}
git=${GIT_STATUS}
olimex_connected=$(if [[ -n "${OLIMEX_MATCHES}" ]]; then echo yes; else echo no; fi)
EOF

echo "Bitcraze Olimex JTAG host prep"
echo "workspace=${WORKSPACE}"
echo "openocd=${OPENOCD_STATUS}"
echo "docker=${DOCKER_STATUS}"
echo "lsusb=${LSUSB_STATUS}"
echo "git=${GIT_STATUS}"
if [[ -n "${OLIMEX_MATCHES}" ]]; then
    echo "olimex_connected=yes"
    echo "${OLIMEX_MATCHES}"
else
    echo "olimex_connected=no"
fi
echo "summary_path=${SUMMARY_PATH}"
echo
echo "Recommended next steps:"
echo "1. Attach the Olimex bundle and rerun this script to confirm USB visibility."
echo "2. If openocd or docker is still missing, rerun with --install-apt."
echo "3. Use the official Bitcraze AI-Deck flashing/JTAG guide once the hardware is connected."
