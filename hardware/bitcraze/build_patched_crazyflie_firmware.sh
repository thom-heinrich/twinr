#!/usr/bin/env bash
set -euo pipefail

# Build one patched Crazyflie STM32 firmware image from pinned Bitcraze sources
# plus explicit repo-local patch files. The workflow is fail-closed:
# - exact firmware revision required
# - source checkout must be clean
# - every patch must apply cleanly
# - resulting cf2.bin/cf2.elf plus one manifest are emitted outside the
#   temporary worktree so the build remains reproducible after cleanup

usage() {
  cat <<'EOF'
Usage:
  bash hardware/bitcraze/build_patched_crazyflie_firmware.sh \
    --expected-firmware-revision REV \
    [--firmware-root PATH] \
    [--patch PATH]... \
    [--artifact-out PATH] \
    [--elf-out PATH] \
    [--manifest-out PATH] \
    [--keep-worktree]

Options:
  --firmware-root PATH   Crazyflie firmware checkout to build against.
                         Default: /tmp/crazyflie-firmware
  --expected-firmware-revision REV
                         Required. Exact git revision or tag the checkout must
                         resolve to before a patched build is allowed.
  --patch PATH           Repo-local patch to apply inside the temporary worktree.
                         Repeat as needed. Default:
                         hardware/bitcraze/patches/crazyflie_firmware_i2c1_boot_hang_fail_closed.patch
  --artifact-out PATH    Where to copy the patched cf2.bin.
                         Default: hardware/bitcraze/build/crazyflie_i2c1_fail_closed_stm32.bin
  --elf-out PATH         Where to copy the patched cf2.elf.
                         Default: hardware/bitcraze/build/crazyflie_i2c1_fail_closed_stm32.elf
  --manifest-out PATH    Where to write the build manifest JSON.
                         Default: hardware/bitcraze/build/crazyflie_i2c1_fail_closed_stm32.build-manifest.json
  --keep-worktree        Keep the temporary patched worktree for post-build inspection.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FIRMWARE_ROOT="/tmp/crazyflie-firmware"
DEFAULT_PATCH="${REPO_ROOT}/hardware/bitcraze/patches/crazyflie_firmware_i2c1_boot_hang_fail_closed.patch"
ARTIFACT_OUT="${REPO_ROOT}/hardware/bitcraze/build/crazyflie_i2c1_fail_closed_stm32.bin"
ELF_OUT="${REPO_ROOT}/hardware/bitcraze/build/crazyflie_i2c1_fail_closed_stm32.elf"
MANIFEST_OUT="${REPO_ROOT}/hardware/bitcraze/build/crazyflie_i2c1_fail_closed_stm32.build-manifest.json"
EXPECTED_FIRMWARE_REVISION=""
KEEP_WORKTREE=0
PATCH_PATHS=()
WORKTREE_DIR=""
WORKTREE_ADDED=0

while (($# > 0)); do
  case "$1" in
    --firmware-root)
      FIRMWARE_ROOT="$2"
      shift 2
      ;;
    --expected-firmware-revision)
      EXPECTED_FIRMWARE_REVISION="$2"
      shift 2
      ;;
    --patch)
      PATCH_PATHS+=("$2")
      shift 2
      ;;
    --artifact-out)
      ARTIFACT_OUT="$2"
      shift 2
      ;;
    --elf-out)
      ELF_OUT="$2"
      shift 2
      ;;
    --manifest-out)
      MANIFEST_OUT="$2"
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

if [[ ${#PATCH_PATHS[@]} -eq 0 ]]; then
  PATCH_PATHS=("${DEFAULT_PATCH}")
fi

if [[ ! -d "${FIRMWARE_ROOT}" ]]; then
  echo "firmware root not found: ${FIRMWARE_ROOT}" >&2
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

if ! command -v make >/dev/null 2>&1; then
  echo "make is required" >&2
  exit 1
fi

if ! command -v arm-none-eabi-gcc >/dev/null 2>&1; then
  echo "arm-none-eabi-gcc is required" >&2
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

mkdir -p "$(dirname "${ARTIFACT_OUT}")" "$(dirname "${ELF_OUT}")" "$(dirname "${MANIFEST_OUT}")"

WORKTREE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/crazyflie-fw-patched.XXXXXX")"
git -C "${FIRMWARE_ROOT}" worktree add --detach "${WORKTREE_DIR}" "${EXPECTED_FIRMWARE_COMMIT}" >/dev/null
WORKTREE_ADDED=1
git -C "${WORKTREE_DIR}" submodule update --init --recursive

for patch_path in "${PATCH_PATHS[@]}"; do
  git -C "${WORKTREE_DIR}" apply --check "${patch_path}"
  git -C "${WORKTREE_DIR}" apply "${patch_path}"
done

make -C "${WORKTREE_DIR}" clean
make -C "${WORKTREE_DIR}" cf2_defconfig
make -C "${WORKTREE_DIR}" -j"$(nproc)"

if [[ ! -f "${WORKTREE_DIR}/build/cf2.bin" || ! -f "${WORKTREE_DIR}/build/cf2.elf" ]]; then
  echo "patched build finished without build/cf2.bin and build/cf2.elf under ${WORKTREE_DIR}" >&2
  exit 1
fi

cp "${WORKTREE_DIR}/build/cf2.bin" "${ARTIFACT_OUT}"
cp "${WORKTREE_DIR}/build/cf2.elf" "${ELF_OUT}"

python3 - "${ARTIFACT_OUT}" "${ELF_OUT}" "${FIRMWARE_ROOT}" "${FIRMWARE_GIT_COMMIT}" "${FIRMWARE_GIT_TAG}" "${EXPECTED_FIRMWARE_REVISION}" "${MANIFEST_OUT}" "${WORKTREE_DIR}" "${KEEP_WORKTREE}" "${PATCH_PATHS[@]}" <<'PY'
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


artifact_out = Path(sys.argv[1])
elf_out = Path(sys.argv[2])
firmware_root = Path(sys.argv[3])
firmware_git_commit = sys.argv[4]
firmware_git_tag = sys.argv[5] or None
expected_revision = sys.argv[6]
manifest_out = Path(sys.argv[7])
worktree_dir = Path(sys.argv[8])
keep_worktree = bool(int(sys.argv[9]))
patch_paths = [Path(value) for value in sys.argv[10:]]

toolchain = subprocess.run(
    ["arm-none-eabi-gcc", "--version"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()[0]

manifest = {
    "report_version": 1,
    "built_at_epoch_s": time.time(),
    "firmware_root": str(firmware_root),
    "firmware_git_commit": firmware_git_commit,
    "firmware_git_tag": firmware_git_tag,
    "expected_firmware_revision": expected_revision,
    "artifact_path": str(artifact_out),
    "artifact_sha256": sha256(artifact_out),
    "elf_path": str(elf_out),
    "elf_sha256": sha256(elf_out),
    "toolchain": toolchain,
    "keep_worktree": keep_worktree,
    "worktree_path": str(worktree_dir) if keep_worktree else None,
    "patches": [
        {
            "path": str(path),
            "sha256": sha256(path),
        }
        for path in patch_paths
    ],
}

manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "built=${ARTIFACT_OUT}"
echo "elf=${ELF_OUT}"
echo "manifest=${MANIFEST_OUT}"
if [[ ${KEEP_WORKTREE} -eq 1 ]]; then
  echo "worktree=${WORKTREE_DIR}"
fi
