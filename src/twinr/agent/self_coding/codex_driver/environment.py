"""Inspect and validate the local self_coding Codex runtime environment.

This module centralizes the non-LLM preflight that the self_coding compile
path, ops checks, and Pi bootstrap workflows all need to answer the same
question consistently: is the local Codex SDK bridge actually runnable on this
machine right now?
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re  # AUDIT-FIX(#5): Parse live auth probe output instead of trusting exit code 0 alone.
import shutil
import subprocess
import tomllib  # AUDIT-FIX(#3): Validate config.toml contents before reporting the runtime as ready.

from twinr.agent.self_coding.codex_driver.types import CodexDriverUnavailableError

_SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]
_WhichResolver = Callable[[str], str | None]
_DEFAULT_SELF_TEST_TIMEOUT_SECONDS = 15.0
_DEFAULT_LIVE_AUTH_TIMEOUT_SECONDS = 60.0
_DEFAULT_AUTH_PROBE_PROMPT = "Reply with the single word READY."
_MAX_VALIDATED_FILE_BYTES = 262_144  # AUDIT-FIX(#3): Bound local validation reads to avoid pathological files.
_READY_LINE_PATTERN = re.compile(r"^\s*READY\s*$", re.IGNORECASE)  # AUDIT-FIX(#5): Require an explicit READY acknowledgement.


@dataclass(frozen=True, slots=True)
class CodexSdkEnvironmentReport:
    """Summarize whether the local self_coding Codex runtime is usable."""

    status: str = "fail"
    ready: bool = False
    detail: str = "Codex SDK runtime is not ready."
    issues: tuple[str, ...] = ()
    node_path: str | None = None
    npm_path: str | None = None
    codex_path: str | None = None
    bridge_script_path: str | None = None
    codex_home_path: str | None = None
    auth_file_path: str | None = None
    config_file_path: str | None = None
    auth_present: bool = False
    config_present: bool = False
    bridge_dependencies_ready: bool = False
    node_version: str | None = None
    npm_version: str | None = None
    codex_version: str | None = None
    local_self_test_ok: bool | None = None
    live_auth_check_ok: bool | None = None


def default_bridge_install_root() -> Path:
    """Return the pinned self_coding SDK bridge directory."""

    return Path(__file__).with_name("sdk_bridge")


def default_bridge_script_path() -> Path:
    """Return the pinned Node bridge script used by the SDK-backed driver."""

    return default_bridge_install_root() / "run_compile.mjs"


def default_codex_home() -> Path:
    """Return the effective Codex home directory for the current process."""

    configured = os.environ.get("CODEX_HOME", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".codex"


def collect_codex_sdk_environment_report(
    *,
    bridge_script: str | Path | None = None,
    bridge_command: Sequence[str] | None = None,
    codex_home: str | Path | None = None,
    which_resolver: _WhichResolver = shutil.which,
    subprocess_runner: _SubprocessRunner = subprocess.run,
    run_local_self_test: bool = True,
    run_live_auth_check: bool = False,
    self_test_timeout_seconds: float = _DEFAULT_SELF_TEST_TIMEOUT_SECONDS,
    live_auth_timeout_seconds: float = _DEFAULT_LIVE_AUTH_TIMEOUT_SECONDS,
    auth_probe_model: str = "gpt-5-codex",
    auth_probe_reasoning_effort: str = "low",
    require_bridge_dependencies: bool = True,
    require_codex_auth: bool = True,
) -> CodexSdkEnvironmentReport:
    """Inspect local Codex prerequisites and optional live-auth readiness.

    Args:
        bridge_script: Optional override for the pinned bridge script.
        codex_home: Optional override for the local Codex home directory.
        which_resolver: Injectable PATH lookup for tests.
        subprocess_runner: Injectable subprocess runner for tests.
        run_local_self_test: Whether to execute the bounded bridge startup
            self-test after static file checks pass.
        run_live_auth_check: Whether to execute a real `codex exec --json`
            probe that proves local credentials and upstream access work.
        self_test_timeout_seconds: Timeout for the local bridge self-test.
        live_auth_timeout_seconds: Timeout for the live auth probe.
        auth_probe_model: Model name used for the live auth probe.
        auth_probe_reasoning_effort: Reasoning effort used for the live probe.

    Returns:
        A normalized readiness report suitable for CLI output, ops checks, and
        driver preflight failures.
    """

    issues: list[str] = []

    # AUDIT-FIX(#2): Derive defaults defensively so broken HOME/cwd state yields a report instead of a hard crash.
    bridge_script_candidate, bridge_candidate_issue = _bridge_script_candidate(bridge_script)
    if bridge_candidate_issue:
        issues.append(bridge_candidate_issue)
    # AUDIT-FIX(#2): Derive CODEX_HOME defensively for the same reason.
    codex_home_candidate, codex_home_candidate_issue = _codex_home_candidate(codex_home)
    if codex_home_candidate_issue:
        issues.append(codex_home_candidate_issue)

    # AUDIT-FIX(#2): Normalize candidate paths without Path.resolve(strict=False), which can raise on bad symlink trees.
    resolved_bridge_script, bridge_path_issue = _normalize_candidate_path(
        bridge_script_candidate,
        label="codex-sdk bridge script path",
    )
    if bridge_path_issue:
        issues.append(bridge_path_issue)
    # AUDIT-FIX(#2): Normalize CODEX_HOME safely so a broken cwd or malformed path becomes a fail report instead of an exception.
    resolved_codex_home, codex_home_issue = _normalize_candidate_path(
        codex_home_candidate,
        label="codex home path",
    )
    if codex_home_issue:
        issues.append(codex_home_issue)

    auth_file = resolved_codex_home / "auth.json"
    config_file = resolved_codex_home / "config.toml"

    # AUDIT-FIX(#6): Reject empty or NUL-bearing custom bridge commands before they hit subprocess.run().
    raw_bridge_command = tuple(bridge_command or ())
    resolved_bridge_command, bridge_command_issue = _normalize_command_parts(raw_bridge_command)
    uses_custom_bridge_command = bool(raw_bridge_command)
    if uses_custom_bridge_command and bridge_command_issue:
        issues.append(bridge_command_issue)

    node_path = (
        resolved_bridge_command[0]
        if uses_custom_bridge_command and bridge_command_issue is None
        else None
        if uses_custom_bridge_command
        else _resolved_command_path("node", which_resolver)
    )
    npm_path = _resolved_command_path("npm", which_resolver)
    codex_path = _resolved_command_path("codex", which_resolver) if require_codex_auth else None

    if node_path is None and not (uses_custom_bridge_command and bridge_command_issue is not None):
        issues.append("node is not available on PATH.")
    if npm_path is None:
        issues.append("npm is not available on PATH.")
    if require_codex_auth and codex_path is None:
        issues.append("codex is not available on PATH.")

    bridge_install_root = resolved_bridge_script.parent
    bridge_dependencies_ready = _bridge_dependencies_ready(
        resolved_bridge_script,
        require_bridge_dependencies=require_bridge_dependencies,
    )
    # AUDIT-FIX(#1): Reject symlinked or unreadable bridge scripts before they can ever be executed.
    bridge_script_issue = _validate_bridge_script_file(resolved_bridge_script)
    if bridge_script_issue:
        issues.append(bridge_script_issue)
    elif require_bridge_dependencies and not bridge_dependencies_ready:
        issues.append(
            f"codex-sdk bridge dependencies are missing under {bridge_install_root}; run `npm ci` there first."
        )

    auth_present = _path_is_file(auth_file)
    config_present = _path_is_file(config_file)
    if require_codex_auth and not auth_present:
        issues.append(f"codex auth file is missing: {auth_file}")
    elif require_codex_auth:
        # AUDIT-FIX(#3): Presence alone is insufficient; validate that auth.json is readable, bounded, and valid JSON.
        auth_file_issue = _validate_auth_file(auth_file)
        if auth_file_issue:
            issues.append(auth_file_issue)

    if config_present:
        # AUDIT-FIX(#3): A malformed config.toml can poison the runtime even when the file merely “exists”.
        config_file_issue = _validate_config_file(config_file)
        if config_file_issue:
            issues.append(config_file_issue)

    node_version = (
        None
        if node_path is None
        else None
        if uses_custom_bridge_command and Path(node_path).name != "node"
        else _read_command_version(node_path, subprocess_runner=subprocess_runner)
    )
    npm_version = _read_command_version(npm_path, subprocess_runner=subprocess_runner)
    codex_version = _read_command_version(codex_path, subprocess_runner=subprocess_runner)

    local_self_test_ok: bool | None = None
    if run_local_self_test and not issues and node_path is not None:
        local_self_test_ok, local_self_test_issue, detected_codex_version = _run_local_self_test(
            bridge_command=resolved_bridge_command if uses_custom_bridge_command else (node_path,),
            bridge_script=resolved_bridge_script,
            subprocess_runner=subprocess_runner,
            timeout_seconds=self_test_timeout_seconds,
        )
        if detected_codex_version and not codex_version:
            codex_version = detected_codex_version
        if not local_self_test_ok and local_self_test_issue:
            issues.append(local_self_test_issue)

    live_auth_check_ok: bool | None = None
    if run_live_auth_check and require_codex_auth and not issues and codex_path is not None:
        live_auth_check_ok, live_auth_issue = _run_live_auth_check(
            codex_path=codex_path,
            subprocess_runner=subprocess_runner,
            timeout_seconds=live_auth_timeout_seconds,
            model=auth_probe_model,
            reasoning_effort=auth_probe_reasoning_effort,
        )
        if not live_auth_check_ok and live_auth_issue:
            issues.append(live_auth_issue)

    ready = not issues
    detail = _build_detail(
        ready=ready,
        issues=tuple(issues),
        node_version=node_version,
        npm_version=npm_version,
        codex_version=codex_version,
        auth_present=auth_present,
        local_self_test_ok=local_self_test_ok,
        live_auth_check_ok=live_auth_check_ok,
    )
    return CodexSdkEnvironmentReport(
        status="ok" if ready else "fail",
        ready=ready,
        detail=detail,
        issues=tuple(issues),
        node_path=node_path,
        npm_path=npm_path,
        codex_path=codex_path,
        bridge_script_path=str(resolved_bridge_script),
        codex_home_path=str(resolved_codex_home),
        auth_file_path=str(auth_file),
        config_file_path=str(config_file),
        auth_present=auth_present,
        config_present=config_present,
        bridge_dependencies_ready=bridge_dependencies_ready,
        node_version=node_version,
        npm_version=npm_version,
        codex_version=codex_version,
        local_self_test_ok=local_self_test_ok,
        live_auth_check_ok=live_auth_check_ok,
    )


def assert_codex_sdk_environment_ready(report: CodexSdkEnvironmentReport) -> CodexSdkEnvironmentReport:
    """Raise a stable driver error when a preflight report is unhealthy."""

    if report.ready:
        return report
    raise CodexDriverUnavailableError(report.detail)


def _bridge_script_candidate(bridge_script: str | Path | None) -> tuple[str | Path, str | None]:
    if bridge_script is not None:
        return bridge_script, None
    try:
        return default_bridge_script_path(), None
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return Path("sdk_bridge") / "run_compile.mjs", f"codex-sdk bridge script path could not be derived: {exc}"


def _codex_home_candidate(codex_home: str | Path | None) -> tuple[str | Path, str | None]:
    if codex_home is not None:
        return codex_home, None
    try:
        return default_codex_home(), None
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return Path(".codex"), f"codex home path could not be derived: {exc}"


def _normalize_candidate_path(candidate: str | Path, *, label: str) -> tuple[Path, str | None]:
    # AUDIT-FIX(#2): Build an absolute path without resolve() so malformed symlink trees cannot crash preflight.
    try:
        if isinstance(candidate, str):
            normalized_candidate = candidate.strip()
            if not normalized_candidate:
                fallback = _fallback_absolute_path(Path("."))
                return fallback, f"{label} is empty."
            expanded = Path(normalized_candidate).expanduser()
        else:
            expanded = Path(candidate).expanduser()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return Path(str(candidate)), f"{label} could not be normalized: {exc}"

    if expanded.is_absolute():
        return expanded, None

    try:
        return Path.cwd() / expanded, None
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return expanded, f"{label} could not be made absolute: {exc}"


def _fallback_absolute_path(path: Path) -> Path:
    try:
        return Path.cwd() / path
    except (OSError, RuntimeError, TypeError, ValueError):
        return path


def _normalize_command_parts(parts: Sequence[str]) -> tuple[tuple[str, ...], str | None]:
    normalized_parts: list[str] = []
    for index, part in enumerate(parts):
        text = str(part).strip()
        if not text:
            return (), f"codex-sdk bridge command contains an empty argument at position {index}"
        if "\x00" in text:
            return (), f"codex-sdk bridge command contains a NUL byte at position {index}"
        normalized_parts.append(text)
    return tuple(normalized_parts), None


def _resolved_command_path(command: str, which_resolver: _WhichResolver) -> str | None:
    # AUDIT-FIX(#2): Injectable resolvers must not be allowed to crash the report path.
    try:
        resolved = which_resolver(command)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if not resolved:
        return None
    normalized = str(resolved).strip()
    if not normalized or "\x00" in normalized:
        return None
    return normalized


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _path_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _path_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _bridge_dependencies_ready(bridge_script: Path, *, require_bridge_dependencies: bool) -> bool:
    if not require_bridge_dependencies:
        return _path_is_file(bridge_script)
    install_root = bridge_script.parent
    sdk_package = install_root / "node_modules" / "@openai" / "codex-sdk"
    package_json = install_root / "package.json"
    return _path_is_file(package_json) and _path_is_dir(sdk_package)


def _validate_bridge_script_file(bridge_script: Path) -> str | None:
    try:
        if bridge_script.is_symlink():
            return f"codex-sdk bridge script must not be a symlink: {bridge_script}"
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return f"codex-sdk bridge script could not be inspected: {bridge_script} ({_normalize_error_message(str(exc), fallback='inspection failed')})"

    if not _path_exists(bridge_script):
        return f"codex-sdk bridge script is missing: {bridge_script}"
    if not _path_is_file(bridge_script):
        return f"codex-sdk bridge script is not a regular file: {bridge_script}"

    try:
        with bridge_script.open("rb") as handle:
            handle.read(1)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        detail = _normalize_error_message(str(exc), fallback="unreadable")
        return f"codex-sdk bridge script is not readable: {bridge_script} ({detail})"

    return None


def _read_text_file_bounded(
    path: Path,
    *,
    label: str,
    limit: int = _MAX_VALIDATED_FILE_BYTES,
) -> tuple[str | None, str | None]:
    if not _path_is_file(path):
        return None, f"{label} is not a regular file: {path}"

    try:
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        detail = _normalize_error_message(str(exc), fallback="unreadable")
        return None, f"{label} is not readable: {path} ({detail})"

    if len(raw) > limit:
        return None, f"{label} is too large to validate safely: {path}"

    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError as exc:
        return None, f"{label} is not valid UTF-8: {path} ({exc.reason})"


def _validate_auth_file(auth_file: Path) -> str | None:
    text, issue = _read_text_file_bounded(auth_file, label="codex auth file")
    if issue:
        return issue
    if text is None:
        return f"codex auth file is not readable: {auth_file}"

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return f"codex auth file is invalid JSON: {auth_file} ({exc.msg})"

    if not isinstance(payload, dict):
        return f"codex auth file must contain a JSON object: {auth_file}"
    return None


def _validate_config_file(config_file: Path) -> str | None:
    text, issue = _read_text_file_bounded(config_file, label="codex config file")
    if issue:
        return issue
    if text is None:
        return f"codex config file is not readable: {config_file}"

    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        detail = _normalize_error_message(str(exc), fallback="invalid TOML")
        return f"codex config file is invalid TOML: {config_file} ({detail})"
    return None


def _read_command_version(
    command_path: str | None,
    *,
    subprocess_runner: _SubprocessRunner,
) -> str | None:
    if not command_path:
        return None
    try:
        completed = subprocess_runner(
            [command_path, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            check=False,
            timeout=10.0,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, subprocess.SubprocessError, ValueError):
        return None

    # AUDIT-FIX(#7): Only trust version text from successful commands, otherwise stderr can masquerade as a version.
    if completed.returncode != 0:
        return None

    text = _first_non_empty_line(_coerce_output_text(completed.stdout))
    if text:
        return text
    return _first_non_empty_line(_coerce_output_text(completed.stderr))


def _run_local_self_test(
    *,
    bridge_command: Sequence[str],
    bridge_script: Path,
    subprocess_runner: _SubprocessRunner,
    timeout_seconds: float,
) -> tuple[bool, str | None, str | None]:
    # AUDIT-FIX(#1): Revalidate the bridge script immediately before spawn to reduce path-swap and symlink risk.
    bridge_script_issue = _validate_bridge_script_file(bridge_script)
    if bridge_script_issue:
        return False, bridge_script_issue, None

    try:
        completed = subprocess_runner(
            [*bridge_command, str(bridge_script), "--self-test"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            check=False,
            timeout=timeout_seconds,
            cwd=str(bridge_script.parent),
        )
    except FileNotFoundError:
        return False, "the configured codex-sdk bridge command is unavailable on this machine", None
    except subprocess.TimeoutExpired:
        return False, "the codex-sdk bridge startup self-test timed out", None
    except (OSError, RuntimeError, TypeError, subprocess.SubprocessError, ValueError) as exc:
        return False, f"failed to start the codex-sdk bridge startup self-test: {exc}", None

    # AUDIT-FIX(#4): Parse raw stdout so verbose logs cannot truncate away the final JSON health payload.
    raw_stdout = _coerce_output_text(completed.stdout)
    raw_stderr = _coerce_output_text(completed.stderr)
    stdout_text = _bounded_text(raw_stdout)
    stderr_text = _bounded_text(raw_stderr)
    if completed.returncode != 0:
        detail = _normalize_error_message(
            stderr_text or stdout_text,
            fallback="the codex-sdk bridge startup self-test failed",
        )
        return False, f"codex-sdk bridge startup self-test failed: {detail}", None
    if not raw_stdout.strip():
        return False, "codex-sdk bridge startup self-test returned no result payload", None

    payload = _find_last_self_test_payload(raw_stdout)
    if payload is None:
        return False, "codex-sdk bridge startup self-test returned invalid JSON", None
    if payload.get("ok") is not True:
        return False, "codex-sdk bridge startup self-test did not confirm a healthy runtime", None
    codex_version = _bounded_text(payload.get("codexVersion"))
    return True, None, codex_version or None


def _run_live_auth_check(
    *,
    codex_path: str,
    subprocess_runner: _SubprocessRunner,
    timeout_seconds: float,
    model: str,
    reasoning_effort: str,
) -> tuple[bool, str | None]:
    try:
        completed = subprocess_runner(
            [
                codex_path,
                "exec",
                "--json",
                "--skip-git-repo-check",
                "-m",
                model,
                "-c",
                f'model_reasoning_effort="{reasoning_effort}"',
                _DEFAULT_AUTH_PROBE_PROMPT,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return False, "the configured codex CLI is unavailable for the live auth probe"
    except subprocess.TimeoutExpired:
        return False, "the live codex auth probe timed out"
    except (OSError, RuntimeError, TypeError, subprocess.SubprocessError, ValueError) as exc:
        return False, f"failed to run the live codex auth probe: {exc}"

    raw_stdout = _coerce_output_text(completed.stdout)
    raw_stderr = _coerce_output_text(completed.stderr)
    if completed.returncode != 0:
        detail = _normalize_error_message(
            _bounded_text(raw_stderr) or _bounded_text(raw_stdout),
            fallback="the live codex auth probe failed",
        )
        return False, detail

    # AUDIT-FIX(#5): Exit code 0 is not enough; require the probe to actually confirm READY.
    if _live_auth_probe_confirmed(raw_stdout):
        return True, None

    detail = _normalize_error_message(
        _bounded_text(raw_stdout) or _bounded_text(raw_stderr),
        fallback="the live codex auth probe did not confirm READY",
    )
    return False, f"the live codex auth probe did not confirm READY: {detail}"


def _coerce_output_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value or "")


def _first_non_empty_line(value: str) -> str | None:
    for line in value.splitlines():
        normalized = line.strip()
        if normalized:
            return _bounded_text(normalized)
    return None


def _iter_json_lines(value: str) -> Iterator[object]:
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line or line[0] not in '[{\"':
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _find_last_self_test_payload(value: str) -> dict[str, object] | None:
    for raw_line in reversed(value.splitlines()):
        line = raw_line.strip()
        if not line or line[0] not in '[{\"':
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "ok" in payload:
            return payload
    return None


def _live_auth_probe_confirmed(stdout_text: str) -> bool:
    saw_json_payload = False
    for payload in _iter_json_lines(stdout_text):
        saw_json_payload = True
        if _payload_contains_ready(payload):
            return True

    if saw_json_payload:
        return False

    for raw_line in stdout_text.splitlines():
        if _READY_LINE_PATTERN.fullmatch(raw_line.strip()):
            return True
    return False


def _payload_contains_ready(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized == _DEFAULT_AUTH_PROBE_PROMPT:
            return False
        return bool(_READY_LINE_PATTERN.fullmatch(normalized))
    if isinstance(value, dict):
        return any(_payload_contains_ready(item) for item in value.values())
    if isinstance(value, list):
        return any(_payload_contains_ready(item) for item in value)
    return False


def _bounded_text(value: object, *, limit: int = 2048) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _normalize_error_message(value: str, *, fallback: str) -> str:
    normalized = " ".join(str(value or "").split())
    return normalized or fallback


def _build_detail(
    *,
    ready: bool,
    issues: Sequence[str],
    node_version: str | None,
    npm_version: str | None,
    codex_version: str | None,
    auth_present: bool,
    local_self_test_ok: bool | None,
    live_auth_check_ok: bool | None,
) -> str:
    if not ready:
        return issues[0] if issues else "Codex SDK runtime is not ready."
    parts: list[str] = []
    if node_version:
        parts.append(f"node {node_version}")
    if npm_version:
        parts.append(f"npm {npm_version}")
    if codex_version:
        parts.append(codex_version)
    parts.append("auth present" if auth_present else "auth missing")
    if local_self_test_ok is True:
        parts.append("bridge self-test ok")
    if live_auth_check_ok is True:
        parts.append("live auth probe ok")
    return " · ".join(parts) or "Codex SDK runtime is ready."


__all__ = [
    "CodexSdkEnvironmentReport",
    "assert_codex_sdk_environment_ready",
    "collect_codex_sdk_environment_report",
    "default_bridge_install_root",
    "default_bridge_script_path",
    "default_codex_home",
]
