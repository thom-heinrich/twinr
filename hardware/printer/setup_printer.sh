#!/usr/bin/env bash
set -euo pipefail

QUEUE_NAME="${TWINR_PRINTER_QUEUE:-Thermal_GP58}"
DEVICE_URI="${TWINR_PRINTER_DEVICE_URI:-}"
MAKE_DEFAULT=0
RUN_TEST=0
LPADMIN_BIN="${LPADMIN_BIN:-/usr/sbin/lpadmin}"
CUPSENABLE_BIN="${CUPSENABLE_BIN:-/usr/sbin/cupsenable}"
CUPSACCEPT_BIN="${CUPSACCEPT_BIN:-/usr/sbin/cupsaccept}"

usage() {
  cat <<'HELP'
Usage: setup_printer.sh [--queue NAME] [--device-uri URI] [--default] [--test]

Configure the DFRobot DFR0503-EN thermal printer as a raw CUPS queue.

Options:
  --queue NAME        Queue name to create or replace (default: Thermal_GP58)
  --device-uri URI    Explicit CUPS device URI
  --default           Set the queue as the system default printer
  --test              Submit a small raw test print after setup
  -h, --help          Show this help text

Environment:
  TWINR_PRINTER_QUEUE
  TWINR_PRINTER_DEVICE_URI
HELP
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --queue)
      [[ $# -ge 2 ]] || fail "Missing value for --queue"
      QUEUE_NAME="$2"
      shift 2
      ;;
    --device-uri)
      [[ $# -ge 2 ]] || fail "Missing value for --device-uri"
      DEVICE_URI="$2"
      shift 2
      ;;
    --default)
      MAKE_DEFAULT=1
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

[[ -x "$LPADMIN_BIN" ]] || fail "lpadmin not found at $LPADMIN_BIN"
[[ -x "$CUPSENABLE_BIN" ]] || fail "cupsenable not found at $CUPSENABLE_BIN"
[[ -x "$CUPSACCEPT_BIN" ]] || fail "cupsaccept not found at $CUPSACCEPT_BIN"
command -v lpstat >/dev/null 2>&1 || fail "lpstat not found"
command -v lp >/dev/null 2>&1 || fail "lp not found"
command -v systemctl >/dev/null 2>&1 || fail "systemctl not found"

if ! systemctl is-active --quiet cups; then
  printf 'Starting cups service\n'
  sudo systemctl start cups
fi

discover_uri() {
  if [[ -n "$DEVICE_URI" ]]; then
    printf '%s\n' "$DEVICE_URI"
    return 0
  fi

  local existing_uri
  existing_uri="$(lpstat -v 2>/dev/null | awk -F': ' '/usb:\/\/Gprinter\/GP-58/ {print $2; exit}')"
  if [[ -n "$existing_uri" ]]; then
    printf '%s\n' "$existing_uri"
    return 0
  fi

  if command -v lpinfo >/dev/null 2>&1; then
    local detected_uri
    detected_uri="$(lpinfo -v 2>/dev/null | awk '/usb:\/\/Gprinter\/GP-58/ {print $2; exit}')"
    if [[ -n "$detected_uri" ]]; then
      printf '%s\n' "$detected_uri"
      return 0
    fi
  fi

  if command -v lsusb >/dev/null 2>&1 && lsusb -d 0525:a4a7 >/dev/null 2>&1; then
    printf '%s\n' 'usb://Gprinter/GP-58?serial=WTTING%20'
    return 0
  fi

  return 1
}

DEVICE_URI="$(discover_uri)" || fail "Could not detect the thermal printer URI. Pass --device-uri explicitly."

printf 'Configuring queue %s\n' "$QUEUE_NAME"
printf 'Using device URI %s\n' "$DEVICE_URI"

sudo "$LPADMIN_BIN" -x "$QUEUE_NAME" >/dev/null 2>&1 || true
sudo "$LPADMIN_BIN" -p "$QUEUE_NAME" -E -v "$DEVICE_URI" -m raw
sudo "$CUPSENABLE_BIN" "$QUEUE_NAME"
sudo "$CUPSACCEPT_BIN" "$QUEUE_NAME"

if [[ "$MAKE_DEFAULT" -eq 1 ]]; then
  sudo "$LPADMIN_BIN" -d "$QUEUE_NAME"
fi

lpstat -p "$QUEUE_NAME"
lpstat -v | awk -v queue="$QUEUE_NAME" '$3 == queue":" {print}'

if [[ "$RUN_TEST" -eq 1 ]]; then
  printf 'Twinr printer setup OK\nThermal path is ready.\n\n\n' | lp -d "$QUEUE_NAME" -o raw
  printf 'Queued raw test print on %s\n' "$QUEUE_NAME"
fi
