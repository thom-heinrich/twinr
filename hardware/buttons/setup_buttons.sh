#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${TWINR_ENV_FILE:-$REPO_ROOT/.env}"
PYTHON_BIN=""
GPIO_CHIP=""
GREEN_GPIO=""
YELLOW_GPIO=""
ACTIVE_LOW=""
BIAS=""
RUN_PROBE=0
PROBE_DURATION=10

usage() {
  cat <<'HELP'
Usage: setup_buttons.sh [options]

Validate and persist the Twinr button GPIO mapping.

Options:
  --env-file PATH      Path to .env file (default: /twinr/.env)
  --chip NAME          GPIO chip name (default: keep current or gpiochip0)
  --green GPIO         GPIO line for the green button
  --yellow GPIO        GPIO line for the yellow button
  --active-low BOOL    true/false (default: keep current or true)
  --bias MODE          pull-up, pull-down, disable, as-is (default: keep current or pull-up)
  --probe              Run a short configured probe after saving the mapping
  --duration SEC       Probe duration in seconds when --probe is used (default: 10)
  -h, --help           Show this help text
HELP
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

resolve_python_bin() {
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    fail "Python 3.11+ not found. Create $REPO_ROOT/.venv or install python3.11."
  fi
  "$PYTHON_BIN" - <<'PY' || fail "Twinr requires Python 3.11+ for setup_buttons.sh"
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
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
    --green)
      [[ $# -ge 2 ]] || fail "Missing value for --green"
      GREEN_GPIO="$2"
      shift 2
      ;;
    --yellow)
      [[ $# -ge 2 ]] || fail "Missing value for --yellow"
      YELLOW_GPIO="$2"
      shift 2
      ;;
    --active-low)
      [[ $# -ge 2 ]] || fail "Missing value for --active-low"
      ACTIVE_LOW="$2"
      shift 2
      ;;
    --bias)
      [[ $# -ge 2 ]] || fail "Missing value for --bias"
      BIAS="$2"
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
resolve_python_bin

read_current() {
  local key="$1"
  "$PYTHON_BIN" - <<PY
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

GREEN_GPIO="${GREEN_GPIO:-$(read_current TWINR_GREEN_BUTTON_GPIO)}"
YELLOW_GPIO="${YELLOW_GPIO:-$(read_current TWINR_YELLOW_BUTTON_GPIO)}"
GPIO_CHIP="${GPIO_CHIP:-$(read_current TWINR_GPIO_CHIP)}"
ACTIVE_LOW="${ACTIVE_LOW:-$(read_current TWINR_BUTTON_ACTIVE_LOW)}"
BIAS="${BIAS:-$(read_current TWINR_BUTTON_BIAS)}"

GPIO_CHIP="${GPIO_CHIP:-gpiochip0}"
ACTIVE_LOW="${ACTIVE_LOW:-true}"
BIAS="${BIAS:-pull-up}"

[[ "$GREEN_GPIO" =~ ^[0-9]+$ ]] || fail "Green button GPIO must be set to an integer"
[[ "$YELLOW_GPIO" =~ ^[0-9]+$ ]] || fail "Yellow button GPIO must be set to an integer"
[[ "$GREEN_GPIO" != "$YELLOW_GPIO" ]] || fail "Green and yellow buttons must use different GPIO lines"
[[ "$PROBE_DURATION" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "Probe duration must be numeric"

case "${ACTIVE_LOW,,}" in
  true|false|1|0|yes|no|on|off) ;;
  *) fail "--active-low must be a boolean value" ;;
esac

case "${BIAS,,}" in
  pull-up|pull-down|disable|disabled|as-is|none) ;;
  *) fail "--bias must be one of: pull-up, pull-down, disable, as-is" ;;
esac

"$PYTHON_BIN" - <<PY
from pathlib import Path
path = Path(r'''$ENV_FILE''')
updates = {
    'TWINR_GPIO_CHIP': r'''$GPIO_CHIP''',
    'TWINR_GREEN_BUTTON_GPIO': r'''$GREEN_GPIO''',
    'TWINR_YELLOW_BUTTON_GPIO': r'''$YELLOW_GPIO''',
    'TWINR_BUTTON_ACTIVE_LOW': r'''$ACTIVE_LOW''',
    'TWINR_BUTTON_BIAS': r'''$BIAS''',
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
PYTHONPATH=src "$PYTHON_BIN" - <<PY
from twinr.config import TwinrConfig
from twinr.hardware.buttons import build_button_bindings
config = TwinrConfig.from_env(r'''$ENV_FILE''')
bindings = build_button_bindings(config)
print(f"chip={config.gpio_chip}")
for binding in bindings:
    print(f"{binding.name}=GPIO{binding.line_offset}")
print(f"active_low={config.button_active_low}")
print(f"bias={config.button_bias}")
PY

if [[ "$RUN_PROBE" -eq 1 ]]; then
  exec "$PYTHON_BIN" hardware/buttons/probe_buttons.py --env-file "$ENV_FILE" --configured --duration "$PROBE_DURATION"
fi
