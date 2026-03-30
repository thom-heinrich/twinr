# CHANGELOG: 2026-03-30
# BUG-1: Made Codex version parsing resilient to current output variants (`codex-cli X`, `codex version X`, `vX`).
# BUG-2: Fixed remote home handling by using `$HOME` instead of hardcoding `/home/{user}`.
# BUG-3: Fixed self-test output truncation by returning combined stdout/stderr.
# BUG-4: Fixed repeated Pi slowdowns by skipping bridge `node_modules` during rsync and caching `npm ci` by lock hash.
# SEC-1: Removed `StrictHostKeyChecking=no`; the bootstrap now requires a trusted host key source
#        (`PI_SSH_HOST_KEY`, `PI_SSH_HOST_FINGERPRINT_SHA256`, `PI_SSH_KNOWN_HOSTS`, or pre-populated `~/.ssh/known_hosts`).
# SEC-2: Removed `/tmp` secret staging; Codex auth/config are streamed over SSH and committed atomically with mode 600.
# SEC-3: Replaced `SSHPASS` environment exposure with `sshpass -d` pipe-based password passing.
# IMP-1: Switched to user-local, pinned installs for Node.js 22.22.2 and Codex on the Pi to avoid `sudo npm install -g`.
# IMP-2: Prefer official Codex release binaries on supported Linux architectures, with npm fallback on unsupported ones.
# IMP-3: Added SSH connection multiplexing and explicit remote version verification for faster, more reliable repeated runs.
# BREAKING: First-time bootstrap now requires trusted SSH host material unless the Pi is already present in `~/.ssh/known_hosts`.
# BREAKING: The bootstrap now expects `sdk_bridge/package-lock.json` for reproducible installs because it uses `npm ci`.

"""Bootstrap the Pi-side self_coding Codex runtime from the leading repo.

This module owns the reproducible operator workflow that prepares `/twinr` for
SDK-backed self_coding compiles: install system prerequisites, sync the pinned
bridge files, copy the local Codex auth/config into the Pi user's codex home,
and run the explicit Twinr self-test remotely.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from typing import Any

_SubprocessRunner = Any
_CODEX_VERSION_PATTERNS = (
    re.compile(r"\bcodex-cli\s+v?([0-9][0-9A-Za-z.\-]+)\b", re.IGNORECASE),
    re.compile(r"\bcodex(?:\s+version)?\s+v?([0-9][0-9A-Za-z.\-]+)\b", re.IGNORECASE),
    re.compile(r"\bv?([0-9]+\.[0-9]+\.[0-9]+(?:[-.][0-9A-Za-z]+)*)\b"),
)
_FINGERPRINT_PATTERN = re.compile(r"\bSHA256:[A-Za-z0-9+/=]+\b")
_DEFAULT_NODEJS_VERSION = "22.22.2"
_MINIMUM_NODEJS_MAJOR = 22


@dataclass(frozen=True, slots=True)
class PiConnectionSettings:
    """Hold the SSH credentials used for the Pi acceptance instance."""

    host: str
    user: str
    password: str | None = None
    port: int = 22
    ssh_key_path: str | None = None
    known_hosts_path: str | None = None
    host_key: str | None = None
    host_fingerprint_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class PiBootstrapResult:
    """Summarize one Pi bootstrap run for operator logging and scripts."""

    ready: bool
    host: str
    remote_root: str
    codex_cli_version: str
    self_test_output: str


@dataclass(frozen=True, slots=True)
class _PreparedSshSession:
    remote_spec: str
    ssh_args: tuple[str, ...]
    rsync_ssh_command: str
    password: str | None = None


def load_pi_connection_settings(env_path: str | Path) -> PiConnectionSettings:
    """Load SSH settings from a dotenv file."""

    values = _read_env_values(Path(env_path))
    host = str(values.get("PI_HOST", "")).strip()
    user = str(values.get("PI_SSH_USER", "")).strip()
    password = _clean_optional_value(values.get("PI_SSH_PW"))
    ssh_key_path = _clean_optional_value(values.get("PI_SSH_KEY_PATH"))
    known_hosts_path = _clean_optional_value(values.get("PI_SSH_KNOWN_HOSTS"))
    host_key = _clean_optional_value(values.get("PI_SSH_HOST_KEY"))
    host_fingerprint_sha256 = _normalize_fingerprint(values.get("PI_SSH_HOST_FINGERPRINT_SHA256"))
    port_text = str(values.get("PI_SSH_PORT", "22")).strip() or "22"

    if not host:
        raise ValueError("PI_HOST is missing from the Pi env file")
    if not user:
        raise ValueError("PI_SSH_USER is missing from the Pi env file")
    if not password and not ssh_key_path:
        raise ValueError("either PI_SSH_KEY_PATH or PI_SSH_PW must be set in the Pi env file")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"PI_SSH_PORT must be an integer, got: {port_text!r}") from exc
    if port <= 0 or port > 65535:
        raise ValueError(f"PI_SSH_PORT must be between 1 and 65535, got: {port}")

    if ssh_key_path:
        ssh_key = Path(ssh_key_path).expanduser()
        if not ssh_key.exists() or not ssh_key.is_file():
            raise ValueError(f"PI_SSH_KEY_PATH does not exist or is not a file: {ssh_key}")
        ssh_key_path = str(ssh_key)

    if known_hosts_path:
        known_hosts = Path(known_hosts_path).expanduser()
        if not known_hosts.exists() or not known_hosts.is_file():
            raise ValueError(f"PI_SSH_KNOWN_HOSTS does not exist or is not a file: {known_hosts}")
        known_hosts_path = str(known_hosts)

    return PiConnectionSettings(
        host=host,
        user=user,
        password=password,
        port=port,
        ssh_key_path=ssh_key_path,
        known_hosts_path=known_hosts_path,
        host_key=host_key,
        host_fingerprint_sha256=host_fingerprint_sha256,
    )


def bootstrap_self_coding_pi(
    *,
    project_root: str | Path,
    pi_env_path: str | Path,
    remote_root: str = "/twinr",
    local_codex_home: str | Path | None = None,
    subprocess_runner: _SubprocessRunner = subprocess.run,
) -> PiBootstrapResult:
    """Install Pi prerequisites and prove the remote self_coding Codex path.

    Args:
        project_root: Leading-repo root on the development machine.
        pi_env_path: Path to `.env.pi`.
        remote_root: Runtime checkout root on the Pi.
        local_codex_home: Optional override for the source Codex home.
        subprocess_runner: Injectable subprocess runner for tests.

    Returns:
        A normalized bootstrap result with the installed CLI version and the
        remote self-test output.
    """

    root = Path(project_root).resolve()
    remote_root_normalized = _normalize_remote_root(remote_root)
    settings = load_pi_connection_settings(pi_env_path)
    codex_home = _default_codex_home() if local_codex_home is None else Path(local_codex_home).expanduser()
    auth_file = codex_home / "auth.json"
    config_file = codex_home / "config.toml"
    if not auth_file.exists() or not auth_file.is_file():
        raise ValueError(f"local Codex auth file is missing: {auth_file}")
    if not config_file.exists() or not config_file.is_file():
        raise ValueError(f"local Codex config file is missing: {config_file}")

    bridge_root = root / "src" / "twinr" / "agent" / "self_coding" / "codex_driver" / "sdk_bridge"
    if not bridge_root.exists() or not bridge_root.is_dir():
        raise ValueError(f"local SDK bridge directory is missing: {bridge_root}")
    if not (bridge_root / "package.json").exists():
        raise ValueError(f"local SDK bridge package.json is missing: {bridge_root / 'package.json'}")
    if not (bridge_root / "package-lock.json").exists():
        raise ValueError(
            f"local SDK bridge package-lock.json is missing: {bridge_root / 'package-lock.json'}"
        )

    codex_version_output = _run_local(
        ["codex", "--version"],
        subprocess_runner=subprocess_runner,
    )
    codex_cli_version = _parse_codex_cli_version(_combined_output(codex_version_output))

    remote_spec = f"{settings.user}@{settings.host}"
    remote_bridge_root = _remote_path_join(
        remote_root_normalized,
        "src",
        "twinr",
        "agent",
        "self_coding",
        "codex_driver",
        "sdk_bridge",
    )
    remote_env_file = _remote_path_join(remote_root_normalized, ".env")
    remote_python = _remote_path_join(remote_root_normalized, ".venv", "bin", "python")

    with ExitStack() as stack:
        session = _prepare_ssh_session(
            settings=settings,
            remote_spec=remote_spec,
            subprocess_runner=subprocess_runner,
            stack=stack,
        )
        stack.callback(_best_effort_close_master, session, subprocess_runner)

        _run_remote(
            session,
            _remote_prepare_dirs_command(remote_bridge_root=remote_bridge_root),
            subprocess_runner=subprocess_runner,
        )

        _run_local_command(
            [
                "rsync",
                "-az",
                "--delete",
                "--exclude",
                "node_modules/",
                "--exclude",
                ".git/",
                "--exclude",
                ".DS_Store",
                "-e",
                session.rsync_ssh_command,
                f"{bridge_root}/",
                f"{remote_spec}:{remote_bridge_root}/",
            ],
            subprocess_runner=subprocess_runner,
            password=session.password,
        )

        _stream_file_to_remote(
            session=session,
            local_file=auth_file,
            remote_command=_remote_secret_write_command(filename="auth.json"),
            subprocess_runner=subprocess_runner,
        )
        _stream_file_to_remote(
            session=session,
            local_file=config_file,
            remote_command=_remote_secret_write_command(filename="config.toml"),
            subprocess_runner=subprocess_runner,
        )

        _run_remote(
            session,
            _remote_install_command(
                codex_cli_version=codex_cli_version,
                remote_root=remote_root_normalized,
                remote_bridge_root=remote_bridge_root,
                nodejs_version=_DEFAULT_NODEJS_VERSION,
            ),
            subprocess_runner=subprocess_runner,
        )

        self_test = _run_remote(
            session,
            _remote_self_test_command(
                remote_root=remote_root_normalized,
                remote_python=remote_python,
                remote_env_file=remote_env_file,
            ),
            subprocess_runner=subprocess_runner,
        )
        output = _combined_output(self_test).strip()
        return PiBootstrapResult(
            ready=True,
            host=settings.host,
            remote_root=remote_root_normalized,
            codex_cli_version=codex_cli_version,
            self_test_output=output,
        )


def _run_local(
    args: list[str],
    *,
    subprocess_runner: _SubprocessRunner,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    pass_fds: tuple[int, ...] = (),
) -> subprocess.CompletedProcess[str]:
    completed = subprocess_runner(
        args,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=env,
        input=input_text,
        pass_fds=pass_fds,
    )
    if completed.returncode != 0:
        message = _combined_output(completed).strip() or f"command failed: {' '.join(args)}"
        raise RuntimeError(message)
    return completed


def _run_local_command(
    args: list[str],
    *,
    subprocess_runner: _SubprocessRunner,
    password: str | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if password:
        return _run_local_with_sshpass(
            args,
            password=password,
            subprocess_runner=subprocess_runner,
            input_text=input_text,
        )
    return _run_local(
        args,
        subprocess_runner=subprocess_runner,
        input_text=input_text,
    )


def _run_local_with_sshpass(
    args: list[str],
    *,
    password: str,
    subprocess_runner: _SubprocessRunner,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, (password + "\n").encode("utf-8"))
        os.close(write_fd)
        write_fd = -1
        return _run_local(
            ["sshpass", "-d", str(read_fd), *args],
            subprocess_runner=subprocess_runner,
            input_text=input_text,
            pass_fds=(read_fd,),
        )
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        os.close(read_fd)


def _run_remote(
    session: _PreparedSshSession,
    remote_command: str,
    *,
    subprocess_runner: _SubprocessRunner,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return _run_local_command(
        ["ssh", *session.ssh_args, session.remote_spec, remote_command],
        subprocess_runner=subprocess_runner,
        password=session.password,
        input_text=input_text,
    )


def _stream_file_to_remote(
    *,
    session: _PreparedSshSession,
    local_file: Path,
    remote_command: str,
    subprocess_runner: _SubprocessRunner,
) -> None:
    content = local_file.read_text(encoding="utf-8")
    _run_remote(
        session,
        remote_command,
        subprocess_runner=subprocess_runner,
        input_text=content,
    )


def _prepare_ssh_session(
    *,
    settings: PiConnectionSettings,
    remote_spec: str,
    subprocess_runner: _SubprocessRunner,
    stack: ExitStack,
) -> _PreparedSshSession:
    control_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="twinr-ssh-")))
    control_path = control_dir / "%C"
    known_hosts_path = _resolve_known_hosts_path(
        settings=settings,
        subprocess_runner=subprocess_runner,
        stack=stack,
    )

    ssh_args: list[str] = [
        "-p",
        str(settings.port),
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts_path}",
        "-o",
        "UpdateHostKeys=yes",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=60",
        "-o",
        f"ControlPath={control_path}",
    ]
    if settings.ssh_key_path:
        ssh_args.extend(
            [
                "-i",
                settings.ssh_key_path,
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "BatchMode=yes",
            ]
        )

    rsync_ssh_command = " ".join(shlex.quote(part) for part in ["ssh", *ssh_args])
    return _PreparedSshSession(
        remote_spec=remote_spec,
        ssh_args=tuple(ssh_args),
        rsync_ssh_command=rsync_ssh_command,
        password=settings.password if not settings.ssh_key_path else None,
    )


def _resolve_known_hosts_path(
    *,
    settings: PiConnectionSettings,
    subprocess_runner: _SubprocessRunner,
    stack: ExitStack,
) -> str:
    if settings.host_key:
        return _write_temp_known_hosts(settings.host_key, stack)
    if settings.host_fingerprint_sha256:
        return _scan_and_pin_host_key(
            settings=settings,
            subprocess_runner=subprocess_runner,
            stack=stack,
        )
    if settings.known_hosts_path:
        return settings.known_hosts_path

    default_known_hosts = Path.home() / ".ssh" / "known_hosts"
    if default_known_hosts.exists() and default_known_hosts.is_file():
        return str(default_known_hosts)

    raise ValueError(
        "trusted SSH host material is missing; provide PI_SSH_HOST_KEY, "
        "PI_SSH_HOST_FINGERPRINT_SHA256, PI_SSH_KNOWN_HOSTS, or add the Pi to ~/.ssh/known_hosts"
    )


def _write_temp_known_hosts(content: str, stack: ExitStack) -> str:
    temp = stack.enter_context(tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, prefix="twinr-known-hosts-"))
    temp.write(content.strip() + "\n")
    temp.flush()
    temp.close()
    os.chmod(temp.name, 0o600)
    stack.callback(_safe_unlink, temp.name)
    return temp.name


def _scan_and_pin_host_key(
    *,
    settings: PiConnectionSettings,
    subprocess_runner: _SubprocessRunner,
    stack: ExitStack,
) -> str:
    scan = _run_local(
        [
            "ssh-keyscan",
            "-T",
            "5",
            "-p",
            str(settings.port),
            settings.host,
        ],
        subprocess_runner=subprocess_runner,
    )
    scanned_keys = (scan.stdout or "").strip()
    if not scanned_keys:
        raise RuntimeError(f"ssh-keyscan returned no host keys for {settings.host}:{settings.port}")

    fingerprints_output = _run_local(
        ["ssh-keygen", "-lf", "-", "-E", "sha256"],
        subprocess_runner=subprocess_runner,
        input_text=scanned_keys,
    )
    fingerprints = set(_FINGERPRINT_PATTERN.findall(_combined_output(fingerprints_output)))
    expected = settings.host_fingerprint_sha256
    if expected is None or expected not in fingerprints:
        found = ", ".join(sorted(fingerprints)) or "<none>"
        raise RuntimeError(
            f"host fingerprint mismatch for {settings.host}:{settings.port}; "
            f"expected {expected!r}, found {found}"
        )
    return _write_temp_known_hosts(scanned_keys, stack)


def _best_effort_close_master(
    session: _PreparedSshSession,
    subprocess_runner: _SubprocessRunner,
) -> None:
    try:
        _run_local_command(
            ["ssh", *session.ssh_args, "-O", "exit", session.remote_spec],
            subprocess_runner=subprocess_runner,
            password=session.password,
        )
    except Exception:
        return

def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        return


def _default_codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".codex"


def _parse_codex_cli_version(text: str) -> str:
    text = str(text or "").strip()
    for pattern in _CODEX_VERSION_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return match.group(1)
    raise RuntimeError(f"could not parse codex version from: {text!r}")


def _normalize_remote_root(remote_root: str | Path) -> str:
    root = str(remote_root).strip()
    if not root:
        raise ValueError("remote_root must not be empty")
    if root == "/":
        return "/"
    return root.rstrip("/") or "/"


def _remote_path_join(root: str, *parts: str) -> str:
    clean_parts = [part.strip("/") for part in parts if part and part.strip("/")]
    if root == "/":
        return "/" + "/".join(clean_parts)
    if not clean_parts:
        return root
    return root.rstrip("/") + "/" + "/".join(clean_parts)


def _remote_prepare_dirs_command(*, remote_bridge_root: str) -> str:
    remote_bridge_q = shlex.quote(remote_bridge_root)
    return (
        "set -euo pipefail; "
        "install -d -m 700 \"$HOME/.codex\"; "
        f"mkdir -p {remote_bridge_q}"
    )


def _remote_secret_write_command(*, filename: str) -> str:
    filename_q = shlex.quote(filename)
    return (
        "set -euo pipefail; "
        "umask 077; "
        "install -d -m 700 \"$HOME/.codex\"; "
        f"target=\"$HOME/.codex/{filename_q}\"; "
        "tmp=\"$(mktemp \"$HOME/.codex/.incoming.XXXXXX\")\"; "
        "cat > \"$tmp\"; "
        "mv \"$tmp\" \"$target\"; "
        "chmod 600 \"$target\""
    )


def _remote_install_command(
    *,
    codex_cli_version: str,
    remote_root: str,
    remote_bridge_root: str,
    nodejs_version: str,
) -> str:
    remote_root_q = shlex.quote(remote_root)
    remote_bridge_q = shlex.quote(remote_bridge_root)
    codex_version_q = shlex.quote(codex_cli_version)
    nodejs_version_q = shlex.quote(nodejs_version)
    minimum_node_major_q = shlex.quote(str(_MINIMUM_NODEJS_MAJOR))
    return f"""
set -euo pipefail

RUNTIME_ROOT="$HOME/.local/share/twinr/self_coding"
BIN_DIR="$HOME/.local/bin"
NODE_VERSION={nodejs_version_q}
CODEX_VERSION={codex_version_q}
MIN_NODE_MAJOR={minimum_node_major_q}
REMOTE_ROOT={remote_root_q}
REMOTE_BRIDGE={remote_bridge_q}

mkdir -p "$RUNTIME_ROOT" "$BIN_DIR"
PATH="$BIN_DIR:$RUNTIME_ROOT/node-current/bin:$PATH"
export PATH

download_url() {{
  local url="$1"
  local destination="$2"
  python3 - "$url" "$destination" <<'PY'
import pathlib
import sys
import time
import urllib.request

url = sys.argv[1]
destination = pathlib.Path(sys.argv[2])

last_error = None
for attempt in range(3):
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        destination.write_bytes(data)
        raise SystemExit(0)
    except Exception as exc:  # pragma: no cover - remote shell helper
        last_error = exc
        time.sleep(1.5 * (attempt + 1))
raise SystemExit(f"download failed for {{url}}: {{last_error}}")
PY
}}

detect_arch() {{
  case "$(uname -m)" in
    aarch64|arm64) printf '%s' 'arm64' ;;
    armv7l|armv7) printf '%s' 'armv7l' ;;
    x86_64|amd64) printf '%s' 'x64' ;;
    *) printf '%s' "$(uname -m)" ;;
  esac
}}

install_node_if_needed() {{
  local detected_arch node_major tmp_dir node_url extracted_dir target_dir
  detected_arch="$(detect_arch)"
  node_major=""
  if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    node_major="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || true)"
  fi
  case "$node_major" in
    ''|*[!0-9]*) ;;
    *)
      if [ "$node_major" -ge "$MIN_NODE_MAJOR" ]; then
        return 0
      fi
      ;;
  esac

  case "$detected_arch" in
    arm64|armv7l|x64) ;;
    *)
      printf 'unsupported architecture for pinned Node.js install: %s\\n' "$detected_arch" >&2
      return 1
      ;;
  esac

  tmp_dir="$(mktemp -d)"
  node_url="https://nodejs.org/download/release/v${{NODE_VERSION}}/node-v${{NODE_VERSION}}-linux-${{detected_arch}}.tar.gz"
  download_url "$node_url" "$tmp_dir/node.tar.gz"
  tar -xzf "$tmp_dir/node.tar.gz" -C "$tmp_dir"
  extracted_dir="$(find "$tmp_dir" -maxdepth 1 -mindepth 1 -type d -name 'node-v*' | head -n 1)"
  if [ -z "$extracted_dir" ]; then
    printf 'failed to extract Node.js archive from %s\\n' "$node_url" >&2
    rm -rf "$tmp_dir"
    return 1
  fi
  target_dir="$RUNTIME_ROOT/node-v${{NODE_VERSION}}-${{detected_arch}}"
  rm -rf "$target_dir"
  mv "$extracted_dir" "$target_dir"
  ln -sfn "$target_dir" "$RUNTIME_ROOT/node-current"
  rm -rf "$tmp_dir"
  PATH="$BIN_DIR:$RUNTIME_ROOT/node-current/bin:$PATH"
  export PATH
}}

codex_version_matches() {{
  local actual
  actual="$(codex --version 2>&1 || true)"
  case "$actual" in
    *"$CODEX_VERSION"*) return 0 ;;
    *) return 1 ;;
  esac
}}

install_codex() {{
  local detected_arch asset_name asset_url tmp_dir binary_path target_dir
  if codex_version_matches; then
    return 0
  fi

  detected_arch="$(detect_arch)"
  case "$detected_arch" in
    arm64) asset_name="codex-aarch64-unknown-linux-musl.tar.gz" ;;
    x64) asset_name="codex-x86_64-unknown-linux-musl.tar.gz" ;;
    *) asset_name="" ;;
  esac

  if [ -n "$asset_name" ]; then
    tmp_dir="$(mktemp -d)"
    asset_url="https://github.com/openai/codex/releases/download/rust-v${{CODEX_VERSION}}/${{asset_name}}"
    if download_url "$asset_url" "$tmp_dir/codex.tar.gz" && tar -xzf "$tmp_dir/codex.tar.gz" -C "$tmp_dir"; then
      binary_path="$(find "$tmp_dir" -maxdepth 1 -type f -name 'codex-*' | head -n 1)"
      if [ -n "$binary_path" ]; then
        target_dir="$RUNTIME_ROOT/codex/${{CODEX_VERSION}}"
        mkdir -p "$target_dir"
        install -m 755 "$binary_path" "$target_dir/codex"
        ln -sfn "$target_dir/codex" "$BIN_DIR/codex"
        rm -rf "$tmp_dir"
        if codex_version_matches; then
          return 0
        fi
      fi
    fi
    rm -rf "$tmp_dir"
  fi

  npm install --global --prefix "$HOME/.local" "@openai/codex@$CODEX_VERSION"
  codex_version_matches
}}

install_bridge_dependencies() {{
  local lock_hash node_signature stamp_dir stamp_file current_stamp
  if [ ! -f "$REMOTE_BRIDGE/package-lock.json" ]; then
    printf 'sdk_bridge/package-lock.json is required for reproducible npm ci installs: %s\\n' "$REMOTE_BRIDGE/package-lock.json" >&2
    return 1
  fi

  lock_hash="$(
    python3 - "$REMOTE_BRIDGE/package.json" "$REMOTE_BRIDGE/package-lock.json" <<'PY'
import hashlib
import pathlib
import sys

digest = hashlib.sha256()
for item in sys.argv[1:]:
    path = pathlib.Path(item)
    digest.update(path.name.encode("utf-8"))
    digest.update(b"\\0")
    digest.update(path.read_bytes())
    digest.update(b"\\0")
print(digest.hexdigest())
PY
  )"
  node_signature="$(node --version 2>/dev/null || printf '%s' 'missing-node')"
  stamp_dir="$REMOTE_BRIDGE/node_modules/.twinr-bootstrap"
  stamp_file="$stamp_dir/install-stamp"
  current_stamp=""
  if [ -f "$stamp_file" ]; then
    current_stamp="$(cat "$stamp_file")"
  fi
  if [ -d "$REMOTE_BRIDGE/node_modules" ] && [ "$current_stamp" = "${{lock_hash}}:${{node_signature}}" ]; then
    return 0
  fi

  cd "$REMOTE_BRIDGE"
  npm ci --no-audit --no-fund
  mkdir -p "$stamp_dir"
  printf '%s' "${{lock_hash}}:${{node_signature}}" > "$stamp_file"
}}

install_node_if_needed
install_codex
if ! codex_version_matches; then
  printf 'installed Codex version does not match expected version %s; got: %s\\n' "$CODEX_VERSION" "$(codex --version 2>&1 || true)" >&2
  exit 1
fi
install_bridge_dependencies
cd "$REMOTE_ROOT"
""".strip()


def _remote_self_test_command(
    *,
    remote_root: str,
    remote_python: str,
    remote_env_file: str,
) -> str:
    remote_root_q = shlex.quote(remote_root)
    remote_python_q = shlex.quote(remote_python)
    remote_env_file_q = shlex.quote(remote_env_file)
    return (
        "set -euo pipefail; "
        "PATH=\"$HOME/.local/bin:$HOME/.local/share/twinr/self_coding/node-current/bin:$PATH\"; "
        f"cd {remote_root_q}; "
        f"PYTHONPATH=src {remote_python_q} -m twinr --env-file {remote_env_file_q} "
        "--self-coding-codex-self-test --self-coding-live-auth-check"
    )


def _combined_output(completed: subprocess.CompletedProcess[str]) -> str:
    parts = []
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(stderr)
    return "\n".join(parts)


def _read_env_values(path: Path) -> dict[str, str]:
    if not path.exists() or not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _parse_env_value(raw_value)
    return values


def _parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""

    if value[0] in {"'", '"'}:
        quote = value[0]
        if len(value) >= 2 and value[-1] == quote:
            inner = value[1:-1]
            if quote == '"':
                return bytes(inner, "utf-8").decode("unicode_escape")
            return inner

    stripped = _strip_unquoted_inline_comment(value).strip()
    if stripped.startswith('"') and stripped.endswith('"') and len(stripped) >= 2:
        return bytes(stripped[1:-1], "utf-8").decode("unicode_escape")
    if stripped.startswith("'") and stripped.endswith("'") and len(stripped) >= 2:
        return stripped[1:-1]
    return stripped


def _strip_unquoted_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    escape = False
    for index, char in enumerate(value):
        if escape:
            escape = False
            continue
        if char == "\\" and in_double:
            escape = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if char == "#" and not in_single and not in_double:
            if index == 0 or value[index - 1].isspace():
                return value[:index]
    return value


def _clean_optional_value(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalize_fingerprint(value: object) -> str | None:
    text = _clean_optional_value(value)
    if text is None:
        return None
    return text if text.startswith("SHA256:") else f"SHA256:{text}"


__all__ = [
    "PiBootstrapResult",
    "PiConnectionSettings",
    "bootstrap_self_coding_pi",
    "load_pi_connection_settings",
]
