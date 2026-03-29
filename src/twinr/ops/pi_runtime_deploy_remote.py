"""Remote execution and verification helpers for Pi runtime deploys.

This module owns the SSH/SCP-side primitives used by the operator-facing Pi
deploy flow. Besides core repo/env/systemd steps, it also installs optional
runtime support files that live inside mirrored local workspaces such as the
ignored ``browser_automation/`` tree.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Sequence

from twinr.ops.self_coding_pi import PiConnectionSettings
from twinr.ops.venv_system_site_bridge import PiVenvSystemSiteBridgeResult
from twinr.ops.venv_wrapper_repair import PiVenvScriptRepairResult, repair_venv_python_shebangs


_SubprocessRunner = Any


@dataclass(frozen=True, slots=True)
class PiSyncedFileResult:
    """Describe one authoritative file sync from the leading repo to the Pi."""

    local_path: str
    remote_path: str
    sha256: str
    changed: bool
    backup_path: str | None


@dataclass(frozen=True, slots=True)
class PiSystemdServiceState:
    """Summarize one productive service state on the Pi."""

    name: str
    active_state: str
    sub_state: str
    unit_file_state: str
    main_pid: int | None
    exec_main_status: int | None
    healthy: bool


@dataclass(frozen=True, slots=True)
class PiPythonImportContractResult:
    """Summarize one remote Python import attestation."""

    python_path: str
    checked_modules: tuple[str, ...]
    imported_modules: tuple[str, ...]
    checked_attribute_contracts: tuple[str, ...]
    validated_attribute_contracts: tuple[str, ...]
    elapsed_s: float


class PiRemoteExecutor:
    """Run bounded SSH/SCP commands against the Pi acceptance host."""

    def __init__(
        self,
        *,
        settings: PiConnectionSettings,
        subprocess_runner: _SubprocessRunner,
        timeout_s: float,
    ) -> None:
        self.settings = settings
        self._subprocess_runner = subprocess_runner
        self.timeout_s = timeout_s

    @property
    def remote_spec(self) -> str:
        """Return the canonical ``user@host`` SSH target string."""

        return f"{self.settings.user}@{self.settings.host}"

    def run_ssh(self, script: str) -> subprocess.CompletedProcess[str]:
        """Run one remote bash script over SSH and capture UTF-8 text output."""

        return self._run_local(
            [
                "sshpass",
                "-e",
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=10",
                self.remote_spec,
                "bash -lc " + shlex.quote("set -euo pipefail; " + script),
            ]
        )

    def run_scp(self, local_path: Path, remote_path: str) -> subprocess.CompletedProcess[str]:
        """Copy one local file to the Pi over SCP."""

        return self._run_local(
            [
                "sshpass",
                "-e",
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                str(local_path),
                f"{self.remote_spec}:{remote_path}",
            ]
        )

    def _run_local(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        completed = self._subprocess_runner(
            args,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=sshpass_env(self.settings.password),
            timeout=self.timeout_s,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            if not message:
                message = f"command failed: {' '.join(args)}"
            raise RuntimeError(message)
        return completed


def sync_authoritative_file(
    *,
    remote: PiRemoteExecutor,
    local_path: Path,
    remote_path: str,
    mode: str,
) -> PiSyncedFileResult:
    """Copy one authoritative local file onto the Pi with checksum verification."""

    local_bytes = local_path.read_bytes()
    local_sha = hashlib.sha256(local_bytes).hexdigest()
    remote_sha = read_remote_sha256(remote=remote, remote_path=remote_path)
    if remote_sha == local_sha:
        return PiSyncedFileResult(
            local_path=str(local_path),
            remote_path=remote_path,
            sha256=local_sha,
            changed=False,
            backup_path=None,
        )

    remote_temp = f"/tmp/twinr-deploy-{time.time_ns()}-{local_path.name}"
    backup_suffix = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    remote.run_scp(local_path, remote_temp)
    completed = remote.run_ssh(
        "\n".join(
            (
                f"target={shlex.quote(remote_path)}",
                f"tmp={shlex.quote(remote_temp)}",
                f"expected_sha={shlex.quote(local_sha)}",
                f"mode={shlex.quote(mode)}",
                f"backup=\"$target.deploy-backup-{backup_suffix}\"",
                "if [ -e \"$target\" ]; then cp \"$target\" \"$backup\"; else backup=\"\"; fi",
                "install -D -m \"$mode\" \"$tmp\" \"$target\"",
                "rm -f \"$tmp\"",
                "actual_sha=$(sha256sum \"$target\" | awk '{print $1}')",
                "if [ \"$actual_sha\" != \"$expected_sha\" ]; then",
                "  echo \"remote checksum mismatch after sync\" >&2",
                "  exit 1",
                "fi",
                "printf '%s' \"$backup\"",
            )
        )
    )
    backup_path = (completed.stdout or "").strip() or None
    return PiSyncedFileResult(
        local_path=str(local_path),
        remote_path=remote_path,
        sha256=local_sha,
        changed=True,
        backup_path=backup_path,
    )


def read_remote_sha256(*, remote: PiRemoteExecutor, remote_path: str) -> str | None:
    """Return the remote file checksum if the file exists, otherwise ``None``."""

    completed = remote.run_ssh(
        f"if [ -f {shlex.quote(remote_path)} ]; then sha256sum {shlex.quote(remote_path)} | awk '{{print $1}}'; fi"
    )
    checksum = (completed.stdout or "").strip()
    return checksum or None


def install_editable_package(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    install_with_deps: bool,
) -> str:
    """Ensure the Pi venv exists and refresh the editable Twinr install.

    The default deploy path keeps ``pip install -e`` in ``--no-deps`` mode so
    stable Pi-host packages such as ``PyQt5`` are not rebuilt on every rollout.
    To keep that fast path integration-safe when the project adds new runtime
    dependencies, the deploy also compares the mirrored ``pyproject.toml``
    dependency list against the Pi venv and installs only missing or
    out-of-spec requirements individually.
    """

    remote_python = f"{remote_root}/.venv/bin/python"
    pip_args = "-e \"$remote_root\"" if install_with_deps else "--no-deps -e \"$remote_root\""
    completed = remote.run_ssh(
        "\n".join(
            (
                f"remote_root={shlex.quote(remote_root)}",
                f"remote_python={shlex.quote(remote_python)}",
                "if [ ! -x \"$remote_python\" ]; then python3 -m venv \"$remote_root/.venv\"; fi",
                f"\"$remote_python\" -m pip install {pip_args}",
            )
        )
    )
    dependency_sync_summary = ""
    if not install_with_deps:
        dependency_sync_summary = sync_project_runtime_dependencies(
            remote=remote,
            remote_root=remote_root,
            remote_python=remote_python,
        )
    repair_result = _repair_remote_venv_python_shebangs(
        remote=remote,
        remote_root=remote_root,
        remote_python=remote_python,
    )
    site_bridge_result = _bridge_remote_venv_system_site_packages(
        remote=remote,
        remote_root=remote_root,
        remote_python=remote_python,
    )
    summary_parts = [summarize_output(completed), dependency_sync_summary]
    if repair_result.rewritten_files:
        repaired = ", ".join(repair_result.sample_paths) or "stale wrappers"
        summary_parts.append(
            "normalized "
            f"{repair_result.rewritten_files} stale venv wrapper shebang(s): {repaired}"
        )
    elif repair_result.checked_files:
        summary_parts.append(
            "verified "
            f"{repair_result.checked_files} venv wrapper shebang(s); no stale paths found"
        )
    if site_bridge_result.active_paths:
        path_count = len(site_bridge_result.active_paths)
        noun = "path" if path_count == 1 else "paths"
        if site_bridge_result.changed:
            summary_parts.append(
                "bridged "
                f"{path_count} Pi system site-package {noun} into the venv"
            )
        else:
            summary_parts.append(
                "verified Pi system site-package bridge for "
                f"{path_count} {noun}"
            )
    return summarize_text("\n".join(part for part in summary_parts if part))


def sync_project_runtime_dependencies(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    remote_python: str,
) -> str:
    """Install only mirrored project requirements that the Pi venv still needs."""

    pending_requirements = load_pending_project_runtime_requirements(
        remote=remote,
        remote_root=remote_root,
        remote_python=remote_python,
    )
    if not pending_requirements:
        return "verified mirrored project dependencies; no changes needed"

    requirement_args = " ".join(shlex.quote(requirement) for requirement in pending_requirements)
    completed = remote.run_ssh(
        "\n".join(
            (
                f"remote_python={shlex.quote(remote_python)}",
                f"\"$remote_python\" -m pip install {requirement_args}",
            )
        )
    )
    dependency_count = len(pending_requirements)
    noun = "dependency" if dependency_count == 1 else "dependencies"
    installed = ", ".join(pending_requirements)
    prefix = f"installed {dependency_count} mirrored project {noun}: {installed}"
    details = summarize_output(completed)
    return prefix if not details else f"{prefix}\n{details}"


def load_pending_project_runtime_requirements(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    remote_python: str,
) -> tuple[str, ...]:
    """Return mirrored project requirements missing or out of spec on the Pi."""

    project_pyproject = f"{remote_root}/pyproject.toml"
    script = "\n".join(
        (
            f"{shlex.quote(remote_python)} - <<'PY'",
            "from __future__ import annotations",
            "import importlib.metadata as importlib_metadata",
            "import json",
            "from pathlib import Path",
            "import tomllib",
            "try:",
            "    from packaging.markers import default_environment",
            "    from packaging.requirements import Requirement",
            "except Exception:",
            "    from pip._vendor.packaging.markers import default_environment",
            "    from pip._vendor.packaging.requirements import Requirement",
            f"project_pyproject = Path({json.dumps(project_pyproject)})",
            "project_payload = tomllib.loads(project_pyproject.read_text(encoding='utf-8'))",
            "dependencies = project_payload.get('project', {}).get('dependencies', [])",
            "environment = default_environment()",
            "pending_requirements = []",
            "for raw_dependency in dependencies:",
            "    requirement = Requirement(raw_dependency)",
            "    if requirement.marker is not None and not requirement.marker.evaluate(environment):",
            "        continue",
            "    try:",
            "        installed_version = importlib_metadata.version(requirement.name)",
            "    except importlib_metadata.PackageNotFoundError:",
            "        pending_requirements.append(raw_dependency)",
            "        continue",
            "    try:",
            "        if requirement.specifier and not requirement.specifier.contains(installed_version, prereleases=True):",
            "            pending_requirements.append(raw_dependency)",
            "    except Exception:",
            "        pending_requirements.append(raw_dependency)",
            "print(json.dumps({'requirements': pending_requirements}, ensure_ascii=False))",
            "PY",
        )
    )
    completed = remote.run_ssh(script)
    payload = json.loads((completed.stdout or "{}").strip() or "{}")
    raw_requirements = payload.get("requirements", ())
    if not isinstance(raw_requirements, list):
        return ()
    pending_requirements = tuple(
        str(requirement).strip()
        for requirement in raw_requirements
        if str(requirement).strip()
    )
    return pending_requirements


def _repair_remote_venv_python_shebangs(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    remote_python: str,
) -> PiVenvScriptRepairResult:
    """Run the venv wrapper-shebang repair in the Pi runtime environment."""

    remote_bin_dir = f"{remote_root}/.venv/bin"
    helper_path = f"{remote_root}/src/twinr/ops/venv_wrapper_repair.py"
    script = "\n".join(
        (
            f"{shlex.quote(remote_python)} - <<'PY'",
            "import importlib.util",
            "import sys",
            "from pathlib import Path",
            "import json",
            f"helper_path = Path({json.dumps(helper_path)})",
            "spec = importlib.util.spec_from_file_location('twinr_ops_venv_wrapper_repair', helper_path)",
            "if spec is None or spec.loader is None:",
            "    raise RuntimeError(f'Could not load venv repair helper from {helper_path}')",
            "module = importlib.util.module_from_spec(spec)",
            "sys.modules[spec.name] = module",
            "spec.loader.exec_module(module)",
            (
                "result = module.repair_venv_python_shebangs("
                f"bin_dir=Path({json.dumps(remote_bin_dir)}), "
                f"expected_interpreter={json.dumps(remote_python)}"
                ")"
            ),
            "print(json.dumps({",
            "    'checked_files': result.checked_files,",
            "    'rewritten_files': result.rewritten_files,",
            "    'sample_paths': list(result.sample_paths),",
            "}, ensure_ascii=False))",
            "PY",
        )
    )
    completed = remote.run_ssh(script)
    payload = json.loads((completed.stdout or "{}").strip() or "{}")
    sample_paths_raw = payload.get("sample_paths", ())
    sample_paths = (
        tuple(str(item).strip() for item in sample_paths_raw if str(item).strip())
        if isinstance(sample_paths_raw, list)
        else ()
    )
    return PiVenvScriptRepairResult(
        checked_files=max(0, int(payload.get("checked_files", 0) or 0)),
        rewritten_files=max(0, int(payload.get("rewritten_files", 0) or 0)),
        sample_paths=sample_paths,
    )


def _bridge_remote_venv_system_site_packages(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    remote_python: str,
) -> PiVenvSystemSiteBridgeResult:
    """Expose Pi OS-managed `dist-packages` inside the preserved venv."""

    helper_path = f"{remote_root}/src/twinr/ops/venv_system_site_bridge.py"
    script = "\n".join(
        (
            f"{shlex.quote(remote_python)} - <<'PY'",
            "import importlib.util",
            "import json",
            "import sys",
            "import sysconfig",
            "from pathlib import Path",
            f"helper_path = Path({json.dumps(helper_path)})",
            "spec = importlib.util.spec_from_file_location('twinr_ops_venv_system_site_bridge', helper_path)",
            "if spec is None or spec.loader is None:",
            "    raise RuntimeError(f'Could not load venv bridge helper from {helper_path}')",
            "module = importlib.util.module_from_spec(spec)",
            "sys.modules[spec.name] = module",
            "spec.loader.exec_module(module)",
            (
                "result = module.ensure_pi_system_site_packages_bridge("
                "site_packages_dir=Path(sysconfig.get_path('purelib'))"
                ")"
            ),
            "print(json.dumps({",
            "    'bridge_path': result.bridge_path,",
            "    'active_paths': list(result.active_paths),",
            "    'changed': result.changed,",
            "}, ensure_ascii=False))",
            "PY",
        )
    )
    completed = remote.run_ssh(script)
    payload = json.loads((completed.stdout or "{}").strip() or "{}")
    active_paths_raw = payload.get("active_paths", ())
    active_paths = (
        tuple(str(item).strip() for item in active_paths_raw if str(item).strip())
        if isinstance(active_paths_raw, list)
        else ()
    )
    return PiVenvSystemSiteBridgeResult(
        bridge_path=str(payload.get("bridge_path", "")).strip(),
        active_paths=active_paths,
        changed=bool(payload.get("changed", False)),
    )


def install_python_requirements_manifest(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    manifest_relpath: str,
    label: str,
) -> str:
    """Install one mirrored Python requirements manifest into the Pi venv."""

    remote_python = f"{remote_root}/.venv/bin/python"
    requirements_path = f"{remote_root}/{manifest_relpath.lstrip('/')}"
    completed = remote.run_ssh(
        "\n".join(
            (
                f"remote_root={shlex.quote(remote_root)}",
                f"remote_python={shlex.quote(remote_python)}",
                f"requirements_path={shlex.quote(requirements_path)}",
                "if [ ! -x \"$remote_python\" ]; then python3 -m venv \"$remote_root/.venv\"; fi",
                'test -s "$requirements_path"',
                f'echo "[{label}] installing python requirements"',
                '"$remote_python" -m pip install -r "$requirements_path"',
            )
        )
    )
    return summarize_output(completed)


def verify_python_import_contract(
    *,
    remote: PiRemoteExecutor,
    remote_python: str,
    modules: Sequence[str],
    attribute_contracts: dict[str, Sequence[str]] | None = None,
) -> PiPythonImportContractResult:
    """Import and attest a fixed Python surface inside the remote Pi venv.

    ``attribute_contracts`` maps one import target spec to the attributes that
    must exist on that target after import. A target spec is either one module
    path (for module-level attributes) or ``module.path:Symbol.path`` for a
    nested object inside that module.
    """

    normalized_modules = tuple(str(module).strip() for module in modules if str(module).strip())
    normalized_attribute_contracts: dict[str, tuple[str, ...]] = {}
    for raw_target, raw_attributes in (attribute_contracts or {}).items():
        target = str(raw_target).strip()
        if not target:
            continue
        attributes = tuple(
            dict.fromkeys(str(attribute).strip() for attribute in raw_attributes if str(attribute).strip())
        )
        if not attributes:
            continue
        normalized_attribute_contracts[target] = attributes

    expected_attribute_checks = tuple(
        f"{target}.{attribute}"
        for target, attributes in normalized_attribute_contracts.items()
        for attribute in attributes
    )

    if not normalized_modules and not normalized_attribute_contracts:
        return PiPythonImportContractResult(
            python_path=remote_python,
            checked_modules=(),
            imported_modules=(),
            checked_attribute_contracts=(),
            validated_attribute_contracts=(),
            elapsed_s=0.0,
        )

    script = "\n".join(
        (
            f"{shlex.quote(remote_python)} - <<'PY'",
            "import importlib",
            "import json",
            "import time",
            f"modules = {json.dumps(list(normalized_modules), ensure_ascii=False)}",
            "attribute_contracts = "
            + json.dumps(
                {target: list(attributes) for target, attributes in normalized_attribute_contracts.items()},
                ensure_ascii=False,
            ),
            "started = time.perf_counter()",
            "imported = []",
            "failed = {}",
            "checked_attribute_contracts = []",
            "validated_attribute_contracts = []",
            "failed_attribute_contracts = {}",
            "missing_attributes = {}",
            "module_cache = {}",
            "def load_module(name):",
            "    module = module_cache.get(name)",
            "    if module is not None:",
            "        return module",
            "    module = importlib.import_module(name)",
            "    module_cache[name] = module",
            "    return module",
            "def resolve_target(spec):",
            "    module_name, separator, symbol_path = spec.partition(':')",
            "    module = load_module(module_name if separator else spec)",
            "    target = module",
            "    if separator and symbol_path:",
            "        for segment in symbol_path.split('.'):",
            "            target = getattr(target, segment)",
            "    return target",
            "for name in modules:",
            "    try:",
            "        load_module(name)",
            "    except Exception as exc:",
            "        failed[name] = f'{type(exc).__name__}: {exc}'",
            "    else:",
            "        imported.append(name)",
            "for target_spec, required_attributes in attribute_contracts.items():",
            "    checked_attribute_contracts.extend(f'{target_spec}.{attribute}' for attribute in required_attributes)",
            "    try:",
            "        target = resolve_target(target_spec)",
            "    except Exception as exc:",
            "        failed_attribute_contracts[target_spec] = f'{type(exc).__name__}: {exc}'",
            "        continue",
            "    missing = []",
            "    for attribute in required_attributes:",
            "        if hasattr(target, attribute):",
            "            validated_attribute_contracts.append(f'{target_spec}.{attribute}')",
            "            continue",
            "        missing.append(attribute)",
            "    if missing:",
            "        missing_attributes[target_spec] = missing",
            "payload = {",
            "    'python_path': " + json.dumps(remote_python) + ",",
            "    'checked_modules': modules,",
            "    'imported_modules': imported,",
            "    'failed_imports': failed,",
            "    'checked_attribute_contracts': checked_attribute_contracts,",
            "    'validated_attribute_contracts': validated_attribute_contracts,",
            "    'failed_attribute_contracts': failed_attribute_contracts,",
            "    'missing_attributes': missing_attributes,",
            "    'elapsed_s': round(time.perf_counter() - started, 3),",
            "}",
            "print(json.dumps(payload, ensure_ascii=False))",
            "PY",
        )
    )
    completed = remote.run_ssh(script)
    payload = json.loads((completed.stdout or "{}").strip() or "{}")

    checked_modules_raw = payload.get("checked_modules", ())
    imported_modules_raw = payload.get("imported_modules", ())
    failed_imports_raw = payload.get("failed_imports", {})
    checked_attribute_contracts_raw = payload.get("checked_attribute_contracts", ())
    validated_attribute_contracts_raw = payload.get("validated_attribute_contracts", ())
    failed_attribute_contracts_raw = payload.get("failed_attribute_contracts", {})
    missing_attributes_raw = payload.get("missing_attributes", {})

    checked_modules = (
        tuple(str(item).strip() for item in checked_modules_raw if str(item).strip())
        if isinstance(checked_modules_raw, list)
        else normalized_modules
    )
    imported_modules = (
        tuple(str(item).strip() for item in imported_modules_raw if str(item).strip())
        if isinstance(imported_modules_raw, list)
        else ()
    )
    failed_imports = (
        {
            str(module).strip(): str(error).strip()
            for module, error in failed_imports_raw.items()
            if str(module).strip()
        }
        if isinstance(failed_imports_raw, dict)
        else {}
    )
    if failed_imports:
        details = "; ".join(f"{module} -> {error}" for module, error in sorted(failed_imports.items()))
        raise RuntimeError(f"remote python import contract failed for {remote_python}: {details}")
    missing_modules = tuple(module for module in checked_modules if module not in set(imported_modules))
    if missing_modules:
        missing_list = ", ".join(missing_modules)
        raise RuntimeError(
            "remote python import contract returned an incomplete result for "
            f"{remote_python}: missing {missing_list}"
        )

    checked_attribute_contracts = (
        tuple(str(item).strip() for item in checked_attribute_contracts_raw if str(item).strip())
        if isinstance(checked_attribute_contracts_raw, list)
        else ()
    )
    validated_attribute_contracts = (
        tuple(str(item).strip() for item in validated_attribute_contracts_raw if str(item).strip())
        if isinstance(validated_attribute_contracts_raw, list)
        else ()
    )
    failed_attribute_contracts = (
        {
            str(target).strip(): str(error).strip()
            for target, error in failed_attribute_contracts_raw.items()
            if str(target).strip()
        }
        if isinstance(failed_attribute_contracts_raw, dict)
        else {}
    )
    missing_attributes = (
        {
            str(target).strip(): tuple(
                str(attribute).strip()
                for attribute in attributes
                if str(attribute).strip()
            )
            for target, attributes in missing_attributes_raw.items()
            if str(target).strip() and isinstance(attributes, list)
        }
        if isinstance(missing_attributes_raw, dict)
        else {}
    )
    if failed_attribute_contracts:
        details = "; ".join(
            f"{target} -> {error}" for target, error in sorted(failed_attribute_contracts.items())
        )
        raise RuntimeError(f"remote python attribute contract failed for {remote_python}: {details}")
    if expected_attribute_checks:
        missing_checked_contracts = tuple(
            contract for contract in expected_attribute_checks if contract not in set(checked_attribute_contracts)
        )
        if missing_checked_contracts:
            missing_list = ", ".join(missing_checked_contracts)
            raise RuntimeError(
                "remote python import contract returned an incomplete attribute result for "
                f"{remote_python}: missing {missing_list}"
            )
        if missing_attributes:
            details = "; ".join(
                f"{target} -> {', '.join(attributes)}"
                for target, attributes in sorted(missing_attributes.items())
            )
            raise RuntimeError(f"remote python attribute contract failed for {remote_python}: {details}")
        missing_validated_contracts = tuple(
            contract for contract in expected_attribute_checks if contract not in set(validated_attribute_contracts)
        )
        if missing_validated_contracts:
            missing_list = ", ".join(missing_validated_contracts)
            raise RuntimeError(
                "remote python import contract returned an incomplete validation result for "
                f"{remote_python}: missing {missing_list}"
            )

    return PiPythonImportContractResult(
        python_path=str(payload.get("python_path", remote_python)).strip() or remote_python,
        checked_modules=checked_modules,
        imported_modules=imported_modules,
        checked_attribute_contracts=checked_attribute_contracts,
        validated_attribute_contracts=validated_attribute_contracts,
        elapsed_s=max(0.0, float(payload.get("elapsed_s", 0.0) or 0.0)),
    )


def refresh_python_bytecode(
    *,
    remote: PiRemoteExecutor,
    remote_python: str,
    roots: Sequence[str],
) -> str:
    """Rebuild checked-hash bytecode for mirrored Python sources on the Pi.

    The repo mirror intentionally excludes ``__pycache__`` and ``*.pyc`` from
    authoritative sync, so the deploy must refresh mirrored bytecode explicitly
    to prevent stale timestamp-based caches from shadowing newer source files.
    Older Pi imports can also leave those cache directories owned by ``root``,
    so the refresh first hands any existing mirrored ``__pycache__`` trees back
    to the runtime user before `compileall` rebuilds checked-hash bytecode.
    """

    normalized_roots = tuple(dict.fromkeys(str(root).strip() for root in roots if str(root).strip()))
    if not normalized_roots:
        return ""
    quoted_roots = " ".join(shlex.quote(root) for root in normalized_roots)
    completed = remote.run_ssh(
        "\n".join(
            (
                f"remote_python={shlex.quote(remote_python)}",
                f"find {quoted_roots} -type d -name __pycache__ -exec sudo chown -R \"$(id -u):$(id -g)\" {{}} +",
                f"\"$remote_python\" -m compileall -q -f --invalidation-mode checked-hash {quoted_roots}",
            )
        )
    )
    summary = summarize_output(completed)
    if summary:
        return summary
    roots_summary = ", ".join(normalized_roots)
    return f"refreshed checked-hash bytecode for {roots_summary}"


def install_browser_automation_runtime_support(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    install_python_requirements: bool,
    install_playwright_browsers: bool,
) -> str:
    """Install mirrored browser-automation runtime requirements on the Pi.

    Args:
        remote: Remote executor targeting the Pi acceptance host.
        remote_root: Mirrored Twinr checkout root on the Pi.
        install_python_requirements: Whether to install Python packages from
            ``browser_automation/runtime_requirements.txt``.
        install_playwright_browsers: Whether to install Playwright browser
            binaries listed in ``browser_automation/playwright_browsers.txt``.

    Returns:
        One compact summary of the remote install output.
    """

    remote_python = f"{remote_root}/.venv/bin/python"
    browsers_path = f"{remote_root}/browser_automation/playwright_browsers.txt"
    summary_parts: list[str] = []
    if install_python_requirements:
        summary_parts.append(
            install_python_requirements_manifest(
                remote=remote,
                remote_root=remote_root,
                manifest_relpath="browser_automation/runtime_requirements.txt",
                label="browser_automation",
            )
        )
    lines = [
        f"remote_root={shlex.quote(remote_root)}",
        f"remote_python={shlex.quote(remote_python)}",
        f"browsers_path={shlex.quote(browsers_path)}",
        "if [ ! -x \"$remote_python\" ]; then python3 -m venv \"$remote_root/.venv\"; fi",
    ]
    if install_playwright_browsers:
        lines.extend(
            (
                'test -s "$browsers_path"',
                "browser_names=()",
                'while IFS= read -r raw_line; do',
                '  line="${raw_line%%#*}"',
                '  set -- $line',
                '  if [ "$#" -eq 0 ]; then',
                "    continue",
                "  fi",
                '  browser_names+=("$1")',
                'done < "$browsers_path"',
                'if [ "${#browser_names[@]}" -eq 0 ]; then',
                '  echo "playwright browser manifest did not contain any browser names" >&2',
                "  exit 1",
                "fi",
                'echo "[browser_automation] installing Playwright browsers: ${browser_names[*]}"',
                '"$remote_python" -m playwright install "${browser_names[@]}"',
            )
        )
    if install_playwright_browsers:
        completed = remote.run_ssh("\n".join(lines))
        summary_parts.append(summarize_output(completed))
    return summarize_text("\n".join(part for part in summary_parts if part))


def install_service_units(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    services: Sequence[str],
) -> None:
    """Install the productive service units from the mirrored repo onto the Pi."""

    lines = [f"remote_root={shlex.quote(remote_root)}"]
    for service_name in services:
        source_path = shlex.quote(f"{remote_root}/hardware/ops/{service_name}")
        target_path = shlex.quote(f"/etc/systemd/system/{service_name}")
        lines.extend(
            (
                f"test -f {source_path}",
                f"sudo install -m 644 {source_path} {target_path}",
            )
        )
    services_arg = " ".join(shlex.quote(name) for name in services)
    lines.extend(
        (
            "sudo systemctl daemon-reload",
            f"sudo systemctl enable {services_arg}",
        )
    )
    remote.run_ssh("\n".join(lines))


def restart_services(*, remote: PiRemoteExecutor, services: Sequence[str]) -> None:
    """Restart the selected productive services on the Pi."""

    services_arg = " ".join(shlex.quote(name) for name in services)
    remote.run_ssh(f"sudo systemctl restart {services_arg}")


def wait_for_services(
    *,
    remote: PiRemoteExecutor,
    services: Sequence[str],
    wait_timeout_s: float,
) -> tuple[PiSystemdServiceState, ...]:
    """Poll the productive services until they report healthy or time out."""

    deadline = time.monotonic() + wait_timeout_s
    latest_states: tuple[PiSystemdServiceState, ...] = ()
    while True:
        latest_states = load_service_states(remote=remote, services=services)
        if latest_states and all(state.healthy for state in latest_states):
            return latest_states
        if time.monotonic() >= deadline:
            break
        time.sleep(1.0)
    failing = [state for state in latest_states if not state.healthy]
    failing_names = ", ".join(state.name for state in failing) or ", ".join(services)
    journal_excerpt = load_journal_excerpt(remote=remote, service_name=failing[0].name if failing else services[0])
    raise RuntimeError(
        f"services did not become healthy within {wait_timeout_s:.1f}s: {failing_names}\n{journal_excerpt}"
    )


def load_service_states(
    *,
    remote: PiRemoteExecutor,
    services: Sequence[str],
) -> tuple[PiSystemdServiceState, ...]:
    """Load the current ``systemctl show`` snapshot for the requested services."""

    script = "\n".join(
        (
            "python3 - <<'PY'",
            "import json",
            "import subprocess",
            f"services = {json.dumps(list(services), ensure_ascii=False)}",
            "payload = []",
            "for name in services:",
            "    completed = subprocess.run(",
            "        ['systemctl', 'show', name, '--property=ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus'],",
            "        check=False,",
            "        capture_output=True,",
            "        text=True,",
            "        encoding='utf-8',",
            "        errors='replace',",
            "    )",
            "    values = {}",
            "    for raw_line in completed.stdout.splitlines():",
            "        if '=' not in raw_line:",
            "            continue",
            "        key, value = raw_line.split('=', 1)",
            "        values[key] = value",
            "    payload.append(",
            "        {",
            "            'name': name,",
            "            'active_state': values.get('ActiveState', ''),",
            "            'sub_state': values.get('SubState', ''),",
            "            'unit_file_state': values.get('UnitFileState', ''),",
            "            'main_pid': values.get('MainPID', ''),",
            "            'exec_main_status': values.get('ExecMainStatus', ''),",
            "        }",
            "    )",
            "print(json.dumps(payload, ensure_ascii=False))",
            "PY",
        )
    )
    completed = remote.run_ssh(script)
    raw_payload = json.loads((completed.stdout or "[]").strip() or "[]")
    states: list[PiSystemdServiceState] = []
    for item in raw_payload:
        active_state = str(item.get("active_state", "")).strip()
        sub_state = str(item.get("sub_state", "")).strip()
        states.append(
            PiSystemdServiceState(
                name=str(item.get("name", "")).strip(),
                active_state=active_state,
                sub_state=sub_state,
                unit_file_state=str(item.get("unit_file_state", "")).strip(),
                main_pid=parse_optional_int(item.get("main_pid")),
                exec_main_status=parse_optional_int(item.get("exec_main_status")),
                healthy=active_state == "active" and sub_state == "running",
            )
        )
    return tuple(states)


def load_journal_excerpt(
    *,
    remote: PiRemoteExecutor,
    service_name: str,
    lines: int = 40,
) -> str:
    """Return a short recent journal excerpt for one failing service."""

    completed = remote.run_ssh(
        f"journalctl -u {shlex.quote(service_name)} -n {int(lines)} --no-pager --output cat || true"
    )
    text = (completed.stdout or completed.stderr or "").strip()
    return summarize_text(text)


def run_env_contract_probe(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    env_path: str,
    live_text: str | None,
    live_search: str | None,
) -> dict[str, object]:
    """Run the bounded Pi env-contract probe and parse its JSON result."""

    remote_python = f"{remote_root}/.venv/bin/python"
    args = [
        shlex.quote(remote_python),
        shlex.quote(f"{remote_root}/hardware/ops/check_pi_openai_env_contract.py"),
        "--env-file",
        shlex.quote(env_path),
    ]
    if live_text is not None:
        args.extend(("--live-text", shlex.quote(live_text)))
    if live_search is not None:
        args.extend(("--live-search", shlex.quote(live_search)))
    completed = remote.run_ssh(" ".join(args))
    return json.loads((completed.stdout or "").strip() or "{}")


def parse_optional_int(value: object) -> int | None:
    """Normalize systemd numeric fields to ``int | None``."""

    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = int(text)
    except ValueError:
        return None
    return number if number > 0 else None


def summarize_output(completed: subprocess.CompletedProcess[str]) -> str:
    """Return one compact summary string for a completed subprocess."""

    text = "\n".join(
        part.strip()
        for part in (completed.stdout or "", completed.stderr or "")
        if part and part.strip()
    )
    return summarize_text(text)


def summarize_text(text: str, *, max_lines: int = 12, max_chars: int = 1200) -> str:
    """Collapse multi-line command output into a bounded summary string."""

    normalized = "\n".join(line.rstrip() for line in str(text or "").splitlines() if line.strip())
    if not normalized:
        return ""
    lines = normalized.splitlines()
    if len(lines) > max_lines:
        normalized = "\n".join(lines[-max_lines:])
    if len(normalized) > max_chars:
        normalized = normalized[-max_chars:]
    return normalized


def sshpass_env(password: str) -> dict[str, str]:
    """Return the local environment with ``SSHPASS`` injected."""

    import os

    env = dict(os.environ)
    env["SSHPASS"] = password
    return env


__all__ = [
    "PiPythonImportContractResult",
    "PiRemoteExecutor",
    "PiSyncedFileResult",
    "PiSystemdServiceState",
    "PiVenvSystemSiteBridgeResult",
    "PiVenvScriptRepairResult",
    "install_browser_automation_runtime_support",
    "install_editable_package",
    "install_python_requirements_manifest",
    "install_service_units",
    "load_journal_excerpt",
    "load_service_states",
    "parse_optional_int",
    "read_remote_sha256",
    "refresh_python_bytecode",
    "restart_services",
    "run_env_contract_probe",
    "repair_venv_python_shebangs",
    "sshpass_env",
    "summarize_output",
    "summarize_text",
    "sync_authoritative_file",
    "verify_python_import_contract",
    "wait_for_services",
]
