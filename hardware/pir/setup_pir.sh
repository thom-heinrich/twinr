#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${TWINR_ENV_FILE:-$REPO_ROOT/.env}"
GPIO_CHIP=""
MOTION_GPIO=""
ACTIVE_HIGH=""
BIAS=""
DEBOUNCE_MS=""
RUN_PROBE=0
PROBE_DURATION=20

usage() {
  cat <<'HELP'
Usage: setup_pir.sh [options]

Validate and persist the Twinr PIR GPIO mapping.

Options:
  --env-file PATH      Path to .env file (default: /twinr/.env)
  --chip NAME          GPIO chip name (default: keep current or gpiochip0)
  --motion GPIO        GPIO line for the PIR motion output
  --active-high BOOL   true/false (default: keep current or true)
  --bias MODE          pull-up, pull-down, disable, as-is (default: keep current or pull-down)
  --debounce-ms INT    debounce window in milliseconds (default: keep current or 120)
  --probe              Run a short PIR probe after saving the mapping
  --duration SEC       Probe duration in seconds when --probe is used (default: 20)
  -h, --help           Show this help text
HELP
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      [[ $# -ge 2 ]] || fail "Missing value for --env-file"
      ENV_FILE="$2"
      shift 2
      ;;
    --chip)
      [[ $# -ge 2 ]] || fail "Missing value for --chip"
      GPIO_CHIP="$2"
      shift 2
      ;;
    --motion)
      [[ $# -ge 2 ]] || fail "Missing value for --motion"
      MOTION_GPIO="$2"
      shift 2
      ;;
    --active-high)
      [[ $# -ge 2 ]] || fail "Missing value for --active-high"
      ACTIVE_HIGH="$2"
      shift 2
      ;;
    --bias)
      [[ $# -ge 2 ]] || fail "Missing value for --bias"
      BIAS="$2"
      shift 2
      ;;
    --debounce-ms)
      [[ $# -ge 2 ]] || fail "Missing value for --debounce-ms"
      DEBOUNCE_MS="$2"
      shift 2
      ;;
    --probe)
      RUN_PROBE=1
      shift
      ;;
    --duration)
      [[ $# -ge 2 ]] || fail "Missing value for --duration"
      PROBE_DURATION="$2"
      shift 2
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

[[ -f "$ENV_FILE" ]] || fail "Env file not found: $ENV_FILE"
command -v python3 >/dev/null 2>&1 || fail "python3 not found"

read_current() {
  local key="$1"
  python3 - <<PY
from pathlib import Path
path = Path(r'''$ENV_FILE''')
value = None
for raw_line in path.read_text(encoding='utf-8').splitlines():
    line = raw_line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    name, raw_value = line.split('=', 1)
    if name.strip() == r'''$key''':
        value = raw_value.strip().strip('"').strip("'")
        break
print(value or "")
PY
}

MOTION_GPIO="${MOTION_GPIO:-$(read_current TWINR_PIR_MOTION_GPIO)}"
GPIO_CHIP="${GPIO_CHIP:-$(read_current TWINR_GPIO_CHIP)}"
ACTIVE_HIGH="${ACTIVE_HIGH:-$(read_current TWINR_PIR_ACTIVE_HIGH)}"
BIAS="${BIAS:-$(read_current TWINR_PIR_BIAS)}"
DEBOUNCE_MS="${DEBOUNCE_MS:-$(read_current TWINR_PIR_DEBOUNCE_MS)}"

GPIO_CHIP="${GPIO_CHIP:-gpiochip0}"
ACTIVE_HIGH="${ACTIVE_HIGH:-true}"
BIAS="${BIAS:-pull-down}"
DEBOUNCE_MS="${DEBOUNCE_MS:-120}"

[[ "$MOTION_GPIO" =~ ^[0-9]+$ ]] || fail "Motion GPIO must be set to an integer"
[[ "$DEBOUNCE_MS" =~ ^[0-9]+$ ]] || fail "Debounce must be set to an integer"
[[ "$PROBE_DURATION" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "Probe duration must be numeric"

case "${ACTIVE_HIGH,,}" in
  true|false|1|0|yes|no|on|off) ;;
  *) fail "--active-high must be a boolean value" ;;
esac

case "${BIAS,,}" in
  pull-up|pull-down|disable|disabled|as-is|none) ;;
  *) fail "--bias must be one of: pull-up, pull-down, disable, as-is" ;;
esac

python3 - <<PY
from pathlib import Path
path = Path(r'''$ENV_FILE''')
updates = {
    'TWINR_GPIO_CHIP': r'''$GPIO_CHIP''',
    'TWINR_PIR_MOTION_GPIO': r'''$MOTION_GPIO''',
    'TWINR_PIR_ACTIVE_HIGH': r'''$ACTIVE_HIGH''',
    'TWINR_PIR_BIAS': r'''$BIAS''',
    'TWINR_PIR_DEBOUNCE_MS': r'''$DEBOUNCE_MS''',
}
lines = path.read_text(encoding='utf-8').splitlines()
result = []
seen = set()
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith('#') or '=' not in stripped:
        result.append(line)
        continue
    key = stripped.split('=', 1)[0].strip()
    if key in updates:
        if key not in seen:
            result.append(f"{key}={updates[key]}")
            seen.add(key)
        continue
    result.append(line)
for key, value in updates.items():
    if key not in seen:
        result.append(f"{key}={value}")
path.write_text("\n".join(result) + "\n", encoding='utf-8')
PY

cd "$REPO_ROOT"
PYTHONPATH=src python3 - <<PY
from twinr.config import TwinrConfig
config = TwinrConfig.from_env(r'''$ENV_FILE''')
print(f"chip={config.gpio_chip}")
print(f"pir=GPIO{config.pir_motion_gpio}")
print(f"active_high={config.pir_active_high}")
print(f"bias={config.pir_bias}")
print(f"debounce_ms={config.pir_debounce_ms}")
PY

if [[ "$RUN_PROBE" -eq 1 ]]; then
  exec python3 hardware/pir/probe_pir.py --env-file "$ENV_FILE" --duration "$PROBE_DURATION"
fi
