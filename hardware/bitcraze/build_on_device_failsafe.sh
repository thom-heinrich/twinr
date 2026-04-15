#!/usr/bin/env bash
set -euo pipefail

# Build the Twinr on-device Crazyflie failsafe app as an out-of-tree firmware
# module from pinned sources only. The script fails closed unless the caller
# names the expected Bitcraze firmware revision and the build can produce a
# matching build attestation for the resulting STM32 artifact.

usage() {
  cat <<'EOF'
Usage:
  bash hardware/bitcraze/build_on_device_failsafe.sh --expected-firmware-revision REV [--firmware-root PATH] [--app-root PATH] [--attestation-out PATH] [--patch PATH]... [--keep-worktree]

Options:
  --firmware-root PATH   Crazyflie firmware checkout to build against.
                         Default: /tmp/crazyflie-firmware
  --app-root PATH        Twinr OOT app folder.
                         Default: hardware/bitcraze/twinr_on_device_failsafe
  --expected-firmware-revision REV
                         Required. Exact git revision or tag that the firmware
                         checkout must resolve to before building.
  --attestation-out PATH Where to write the build attestation JSON.
                         Default: <app-root>/build/twinr_on_device_failsafe.build-attestation.json
  --patch PATH           Repo-local Crazyflie firmware patch to apply inside a
                         temporary worktree before building the OOT app.
                         Repeat as needed.
  --keep-worktree        Keep the temporary patched worktree for inspection.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FIRMWARE_ROOT="/tmp/crazyflie-firmware"
APP_ROOT="${REPO_ROOT}/hardware/bitcraze/twinr_on_device_failsafe"
ATTESTATION_OUT=""
EXPECTED_FIRMWARE_REVISION=""
PATCH_PATHS=()
KEEP_WORKTREE=0
WORKTREE_DIR=""
WORKTREE_ADDED=0
BUILD_FIRMWARE_ROOT=""

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
    --expected-firmware-revision)
      EXPECTED_FIRMWARE_REVISION="$2"
      shift 2
      ;;
    --attestation-out)
      ATTESTATION_OUT="$2"
      shift 2
      ;;
    --patch)
      PATCH_PATHS+=("$2")
      shift 2
      ;;
    --keep-worktree)
      KEEP_WORKTREE=1
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

cleanup() {
  if [[ ${KEEP_WORKTREE} -eq 0 && ${WORKTREE_ADDED} -eq 1 && -n "${WORKTREE_DIR}" ]]; then
    git -C "${FIRMWARE_ROOT}" worktree remove --force "${WORKTREE_DIR}" >/dev/null 2>&1 || true
    rm -rf "${WORKTREE_DIR}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ -z "${EXPECTED_FIRMWARE_REVISION}" ]]; then
  echo "--expected-firmware-revision is required" >&2
  usage >&2
  exit 1
fi

if [[ ! -d "${FIRMWARE_ROOT}" ]]; then
  echo "firmware root not found: ${FIRMWARE_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${APP_ROOT}/Makefile" ]]; then
  echo "app root not found or incomplete: ${APP_ROOT}" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

if ! command -v arm-none-eabi-gcc >/dev/null 2>&1; then
  echo "arm-none-eabi-gcc is required; Docker fallback is intentionally disabled" >&2
  exit 1
fi

if ! git -C "${FIRMWARE_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "firmware root is not a git checkout: ${FIRMWARE_ROOT}" >&2
  exit 1
fi

if ! git -C "${FIRMWARE_ROOT}" diff --quiet --ignore-submodules --exit-code; then
  echo "firmware root has uncommitted changes: ${FIRMWARE_ROOT}" >&2
  exit 1
fi

EXPECTED_FIRMWARE_COMMIT="$(git -C "${FIRMWARE_ROOT}" rev-parse "${EXPECTED_FIRMWARE_REVISION}^{commit}" 2>/dev/null || true)"
if [[ -z "${EXPECTED_FIRMWARE_COMMIT}" ]]; then
  echo "expected firmware revision does not resolve to a commit: ${EXPECTED_FIRMWARE_REVISION}" >&2
  exit 1
fi

FIRMWARE_GIT_COMMIT="$(git -C "${FIRMWARE_ROOT}" rev-parse HEAD)"
if [[ "${FIRMWARE_GIT_COMMIT}" != "${EXPECTED_FIRMWARE_COMMIT}" ]]; then
  echo "firmware root HEAD ${FIRMWARE_GIT_COMMIT} does not match expected revision ${EXPECTED_FIRMWARE_REVISION} (${EXPECTED_FIRMWARE_COMMIT})" >&2
  exit 1
fi

FIRMWARE_GIT_TAG="$(git -C "${FIRMWARE_ROOT}" describe --tags --exact-match 2>/dev/null || true)"

for patch_path in "${PATCH_PATHS[@]}"; do
  if [[ ! -f "${patch_path}" ]]; then
    echo "patch not found: ${patch_path}" >&2
    exit 1
  fi
done

if [[ ${#PATCH_PATHS[@]} -gt 0 ]]; then
  NORMALIZED_PATCH_PATHS=()
  for patch_path in "${PATCH_PATHS[@]}"; do
    NORMALIZED_PATCH_PATHS+=("$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${patch_path}")")
  done
  PATCH_PATHS=("${NORMALIZED_PATCH_PATHS[@]}")
fi

BUILD_FIRMWARE_ROOT="${FIRMWARE_ROOT}"

if [[ ${#PATCH_PATHS[@]} -gt 0 ]]; then
  WORKTREE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/crazyflie-oot-patched.XXXXXX")"
  git -C "${FIRMWARE_ROOT}" worktree add --detach "${WORKTREE_DIR}" "${EXPECTED_FIRMWARE_COMMIT}" >/dev/null
  WORKTREE_ADDED=1
  git -C "${WORKTREE_DIR}" submodule update --init --recursive
  for patch_path in "${PATCH_PATHS[@]}"; do
    git -C "${WORKTREE_DIR}" apply --check "${patch_path}"
    git -C "${WORKTREE_DIR}" apply "${patch_path}"
  done
  BUILD_FIRMWARE_ROOT="${WORKTREE_DIR}"
fi

if [[ -z "${ATTESTATION_OUT}" ]]; then
  ATTESTATION_OUT="${APP_ROOT}/build/twinr_on_device_failsafe.build-attestation.json"
fi

make -C "${APP_ROOT}" CRAZYFLIE_BASE="${BUILD_FIRMWARE_ROOT}" clean
make -C "${APP_ROOT}" CRAZYFLIE_BASE="${BUILD_FIRMWARE_ROOT}" cf2_defconfig
make -C "${APP_ROOT}" CRAZYFLIE_BASE="${BUILD_FIRMWARE_ROOT}" -j"$(nproc)"

if [[ ! -f "${APP_ROOT}/build/cf2.bin" ]]; then
  echo "build finished without build/cf2.bin under ${APP_ROOT}" >&2
  exit 1
fi

cp "${APP_ROOT}/build/cf2.bin" "${APP_ROOT}/build/twinr_on_device_failsafe.bin"

python3 - "${REPO_ROOT}" "${APP_ROOT}/build/twinr_on_device_failsafe.bin" "${FIRMWARE_ROOT}" "${FIRMWARE_GIT_COMMIT}" "${FIRMWARE_GIT_TAG}" "${EXPECTED_FIRMWARE_REVISION}" "${APP_ROOT}" "${ATTESTATION_OUT}" "${PATCH_PATHS[@]}" <<'PY'
from pathlib import Path
import sys

repo_root = Path(sys.argv[1])
artifact_path = Path(sys.argv[2])
firmware_root = Path(sys.argv[3])
firmware_git_commit = sys.argv[4]
firmware_git_tag = sys.argv[5] or None
expected_firmware_revision = sys.argv[6]
app_root = Path(sys.argv[7])
attestation_out = Path(sys.argv[8])
patch_paths = tuple(Path(value) for value in sys.argv[9:])

sys.path.insert(0, str(repo_root / "hardware" / "bitcraze"))
from firmware_release_policy import compute_tree_sha256, create_build_attestation, write_attestation

attestation = create_build_attestation(
    artifact_path=artifact_path,
    firmware_root=firmware_root,
    firmware_git_commit=firmware_git_commit,
    firmware_git_tag=firmware_git_tag,
    expected_firmware_revision=expected_firmware_revision,
    app_root=app_root,
    app_tree_sha256=compute_tree_sha256(app_root),
    toolchain="arm-none-eabi-gcc",
    firmware_patch_paths=patch_paths,
)
write_attestation(attestation_out, attestation)
PY

echo "built=${APP_ROOT}/build/twinr_on_device_failsafe.bin"
echo "attestation=${ATTESTATION_OUT}"
