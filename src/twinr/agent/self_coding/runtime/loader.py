"""Load materialized self-coding skill entry modules from disk."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import stat
import sys
import threading
from pathlib import Path
from types import ModuleType

_MODULE_NAME_SANITIZER = re.compile(r"[^0-9A-Za-z_]+")  # AUDIT-FIX(#6): Normalize untrusted skill identifiers before using them in module names.
_LOAD_LOCK = threading.RLock()  # AUDIT-FIX(#4): Serialize dynamic module registration so callers never observe a half-initialized module.


class SelfCodingSkillLoadError(RuntimeError):
    """Raised when a materialized self-coding skill package cannot be loaded."""


def _normalize_version(version: int) -> int:
    # AUDIT-FIX(#5): Reject ambiguous version inputs early instead of letting int(...) fail deep inside module-name creation.
    if isinstance(version, bool):
        raise SelfCodingSkillLoadError("Version must be a non-negative integer")
    try:
        version_int = int(version)
    except (TypeError, ValueError) as exc:
        raise SelfCodingSkillLoadError("Version must be a non-negative integer") from exc
    if version_int < 0:
        raise SelfCodingSkillLoadError("Version must be a non-negative integer")
    return version_int


def _sanitize_skill_id(skill_id: str) -> str:
    # AUDIT-FIX(#5): Fail fast on empty or malformed identifiers so the loader does not produce confusing low-level import errors.
    if not isinstance(skill_id, str) or not skill_id.strip():
        raise SelfCodingSkillLoadError("skill_id must be a non-empty string")
    sanitized = _MODULE_NAME_SANITIZER.sub("_", skill_id.strip()).strip("_")
    return (sanitized[:64] or "skill")


def _resolve_entry_path(*, materialized_root: Path, entry_module: str) -> tuple[Path, Path]:
    # AUDIT-FIX(#5): Validate the configured root explicitly so misconfiguration is distinguishable from a missing entry file.
    try:
        root_path = Path(materialized_root).resolve(strict=True)
    except FileNotFoundError as exc:
        raise SelfCodingSkillLoadError(f"Materialized root does not exist: {materialized_root}") from exc
    except OSError as exc:
        raise SelfCodingSkillLoadError(f"Could not resolve materialized root: {materialized_root}") from exc
    if not root_path.is_dir():
        raise SelfCodingSkillLoadError(f"Materialized root is not a directory: {materialized_root}")

    # AUDIT-FIX(#5): Reject empty or malformed entry-module values before they reach pathlib/importlib internals.
    if not isinstance(entry_module, str) or not entry_module.strip():
        raise SelfCodingSkillLoadError("entry_module must be a non-empty relative path")
    try:
        entry_rel = Path(entry_module)
    except (TypeError, ValueError) as exc:
        raise SelfCodingSkillLoadError("entry_module must be a valid relative path") from exc

    # AUDIT-FIX(#1): Block absolute-path and parent-directory traversal before path resolution can escape the materialized root.
    if entry_rel.is_absolute():
        raise SelfCodingSkillLoadError(f"Entry module must be a relative path: {entry_module}")
    if any(part == ".." for part in entry_rel.parts):
        raise SelfCodingSkillLoadError(f"Entry module must stay within the materialized root: {entry_module}")

    candidate_path = root_path / entry_rel
    try:
        entry_path = candidate_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise SelfCodingSkillLoadError(f"Missing materialized entry module: {entry_module}") from exc
    except OSError as exc:
        raise SelfCodingSkillLoadError(f"Could not resolve materialized entry module: {entry_module}") from exc

    # AUDIT-FIX(#1): Enforce post-resolution confinement so symlinks cannot redirect the loader outside the trusted root.
    try:
        entry_path.relative_to(root_path)
    except ValueError as exc:
        raise SelfCodingSkillLoadError(f"Entry module escapes materialized root: {entry_module}") from exc

    # AUDIT-FIX(#2): Restrict the loader to regular Python source files, not directories, devices, or opaque compiled artifacts.
    if entry_path.suffix != ".py":
        raise SelfCodingSkillLoadError(f"Entry module must be a .py file: {entry_module}")
    if not entry_path.is_file():
        raise SelfCodingSkillLoadError(f"Entry module is not a regular file: {entry_module}")

    return root_path, entry_path


def _open_entry_module_fd(*, root_path: Path, entry_path: Path) -> int:
    relative_parts = entry_path.relative_to(root_path).parts
    if not relative_parts:
        raise SelfCodingSkillLoadError(f"Entry module is not a regular file: {entry_path.name}")

    # AUDIT-FIX(#2): Walk the path from the trusted root with openat/O_NOFOLLOW to block symlink swaps and classic TOCTOU races.
    dir_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    dir_fds: list[int] = []

    try:
        current_fd = os.open(root_path, dir_flags)
        dir_fds.append(current_fd)

        for part in relative_parts[:-1]:
            next_fd = os.open(part, dir_flags, dir_fd=current_fd)
            dir_fds.append(next_fd)
            current_fd = next_fd

        file_fd = os.open(relative_parts[-1], file_flags, dir_fd=current_fd)
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode):
            os.close(file_fd)
            raise SelfCodingSkillLoadError(f"Entry module is not a regular file: {entry_path.name}")
        return file_fd
    except SelfCodingSkillLoadError:
        raise
    except OSError as exc:
        raise SelfCodingSkillLoadError(f"Could not securely open materialized entry module: {entry_path.name}") from exc
    finally:
        for dir_fd in reversed(dir_fds):
            os.close(dir_fd)


def _read_entry_source(*, root_path: Path, entry_path: Path) -> str:
    # AUDIT-FIX(#2): Read from the securely opened file descriptor so the executed bytes match the validated file on disk.
    file_fd = _open_entry_module_fd(root_path=root_path, entry_path=entry_path)
    try:
        with os.fdopen(file_fd, "rb", closefd=False) as handle:
            source_bytes = handle.read()
    finally:
        os.close(file_fd)

    return importlib.util.decode_source(source_bytes)


def _build_module_name(*, skill_id: str, version: int, entry_path: Path) -> str:
    safe_skill_id = _sanitize_skill_id(skill_id)
    # AUDIT-FIX(#6): Add a stable digest so sanitized names cannot collide across different skills or entry paths.
    digest_source = f"{skill_id}\0{version}\0{entry_path}".encode("utf-8", "surrogatepass")
    digest = hashlib.blake2s(digest_source, digest_size=8).hexdigest()
    return f"twinr_self_coding_runtime_{safe_skill_id}_v{version}_{digest}"


def load_skill_module(*, skill_id: str, version: int, materialized_root: Path, entry_module: str) -> ModuleType:
    """Import one materialized entry module under a unique module name."""

    root_path, entry_path = _resolve_entry_path(materialized_root=materialized_root, entry_module=entry_module)
    version_int = _normalize_version(version)
    module_name = _build_module_name(skill_id=skill_id, version=version_int, entry_path=entry_path)

    spec = importlib.util.spec_from_file_location(module_name, entry_path)
    if spec is None or spec.loader is None:
        raise SelfCodingSkillLoadError(f"Could not build import spec for {entry_module}")

    try:
        module = importlib.util.module_from_spec(spec)
    except Exception as exc:
        # AUDIT-FIX(#3): Normalize module-construction failures under the loader's public exception contract.
        raise SelfCodingSkillLoadError(f"Could not initialize module object for {entry_module}") from exc

    with _LOAD_LOCK:
        previous_module = sys.modules.get(module_name)
        # AUDIT-FIX(#4): Register the module before execution so dataclasses, self-imports, and package-relative imports resolve correctly.
        sys.modules[module_name] = module
        try:
            source_text = _read_entry_source(root_path=root_path, entry_path=entry_path)
            code = compile(source_text, str(entry_path), "exec", dont_inherit=True)
            exec(code, module.__dict__)
        except Exception as exc:
            # AUDIT-FIX(#3): Roll back partially initialized state and raise the domain-specific loader error instead of leaking raw exceptions.
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module
            raise SelfCodingSkillLoadError(f"Failed to load materialized entry module: {entry_module}") from exc

    return module


__all__ = [
    "SelfCodingSkillLoadError",
    "load_skill_module",
]