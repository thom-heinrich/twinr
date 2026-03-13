"""
Shared store layout helpers for agentic tools.

We want durable, machine-readable "stores" (tasks/chat/rules/ssot/...) to live in
one canonical folder. Legacy locations are treated as errors to avoid split-brain
or silent fallbacks.

Canonical root (default):
  <repo>/artifacts/stores/

Override (optional):
  CAIA_STORES_ROOT=/abs/or/relative/to/repo
  TESSAIRACT_STORES_ROOT=/abs/or/relative/to/repo   (alias)
"""

##REFACTOR: 2026-01-16##

from __future__ import annotations  # Deprecated in Python 3.14+, kept for backward-compatible runtime behavior.

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent


def _truthy_env(key: str) -> bool:
    return (os.getenv(key) or "").strip().lower() in ("1", "true", "yes", "y", "on")


def _expand_path_vars(s: str) -> str:
    """
    Expand environment variables in a path string.

    Python does not expand variables automatically; this makes env-based overrides behave as operators expect.
    """
    # os.path.expandvars is a no-op if no variables are present.
    return os.path.expandvars(s)


def resolve_repo_path(raw: str, *, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a path allowing repo-root-relative strings for determinism."""
    if _truthy_env("CAIA_STORES_CACHE"):
        return _resolve_repo_path_cached(str(raw or ""), repo_root=repo_root)
    return _resolve_repo_path_uncached(raw, repo_root=repo_root)


def _resolve_repo_path_uncached(raw: str, *, repo_root: Path) -> Path:
    s = str(raw or "").strip()
    if not s:
        raise ValueError("empty path")
    s = _expand_path_vars(s)
    p = Path(s).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (repo_root / p).resolve()


@lru_cache(maxsize=256)
def _resolve_repo_path_cached(raw_s: str, *, repo_root: Path) -> Path:
    # Note: raw_s is pre-coerced to str for stable caching. We still preserve the "empty path" behavior.
    return _resolve_repo_path_uncached(raw_s, repo_root=repo_root)


def _legacy_stores_root(repo_root: Path) -> Path:
    return (repo_root / "state" / "stores").resolve()


def _canonical_stores_root(repo_root: Path) -> Path:
    return (repo_root / "artifacts" / "stores").resolve()


def _is_under(path: Path, root: Path) -> bool:
    """
    Return True if 'path' is within 'root' (lexically), False otherwise.
    """
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _path_exists(p: Path) -> bool:
    """
    Existence check with optional strict FS semantics.

    Default: defer to Path.exists() to preserve runtime/version behavior.
    Optional strict mode: CAIA_STORES_STRICT_FS=1 will treat inaccessible paths as errors (via stat()).
    """
    if not _truthy_env("CAIA_STORES_STRICT_FS"):
        return p.exists()
    try:
        p.stat()
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        raise RuntimeError(f"Failed to stat path {p}: {e}") from e


def stores_root(*, repo_root: Path = REPO_ROOT) -> Path:
    """Return canonical stores root, with optional env override."""
    raw = (os.getenv("CAIA_STORES_ROOT") or os.getenv("TESSAIRACT_STORES_ROOT") or "").strip()
    if raw:
        override = resolve_repo_path(raw, repo_root=repo_root)
        legacy_root = _legacy_stores_root(repo_root)
        if _is_under(override, legacy_root) and not _truthy_env("CAIA_STORES_ALLOW_LEGACY"):
            raise RuntimeError(
                f"Legacy stores root is at {legacy_root} but override points into it at {override}. "
                "Refusing to use legacy via override. "
                "Migrate to <repo>/artifacts/stores or set CAIA_STORES_ALLOW_LEGACY=1 temporarily."
            )
        return override

    artifacts = _canonical_stores_root(repo_root)
    legacy_state_stores = _legacy_stores_root(repo_root)
    if _path_exists(legacy_state_stores) and not _path_exists(artifacts):
        if _truthy_env("CAIA_STORES_ALLOW_LEGACY"):
            return legacy_state_stores
        raise RuntimeError(
            f"Legacy stores root exists at {legacy_state_stores} but canonical {artifacts} is missing. "
            "Run scripts/ops/migrate_artifact_roots_to_artifacts_root.py --mode move --apply."
        )
    return artifacts


def _canonical_rel_path(canonical_rel: str) -> Path:
    """
    Validate canonical_rel as a relative, non-traversing path fragment.

    We only forbid absolute paths and '..' traversal components.
    """
    p = Path(str(canonical_rel))
    if p.is_absolute() or ".." in p.parts:
        raise RuntimeError(
            f"Invalid canonical_rel {canonical_rel!r}: must be relative to stores_root and must not contain '..'."
        )
    return p


def resolve_store_file(
    *,
    env_keys: Sequence[str],
    legacy_rel: Optional[str],
    canonical_rel: str,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """
    Resolve a store file path.

    Precedence:
      1) First non-empty env key in env_keys (repo-relative allowed)
      2) Canonical path (new install)

    Legacy paths are rejected when they exist to avoid split-brain. Migrate first.
    """
    legacy_root = _legacy_stores_root(repo_root)

    for key in env_keys:
        val = (os.getenv(key) or "").strip()
        if val:
            override = resolve_repo_path(val, repo_root=repo_root)
            if _is_under(override, legacy_root) and not _truthy_env("CAIA_STORES_ALLOW_LEGACY"):
                raise RuntimeError(
                    f"Legacy store path is under {legacy_root} but env override {key} points to {override}. "
                    "Refusing to use legacy via override. "
                    "Migrate to <repo>/artifacts/stores or set CAIA_STORES_ALLOW_LEGACY=1 temporarily."
                )
            return override

    # Compute canonical and legacy paths similarly to the original implementation.
    canonical = (stores_root(repo_root=repo_root) / Path(str(canonical_rel))).resolve()
    legacy = (resolve_repo_path(legacy_rel, repo_root=repo_root) if legacy_rel else None)

    legacy_exists = bool(legacy and _path_exists(legacy))
    if legacy_exists and not _truthy_env("CAIA_STORES_ALLOW_LEGACY"):
        raise RuntimeError(
            f"Legacy store path exists at {legacy}. Refusing to use it. "
            f"Migrate it to {canonical} (scripts/ops/migrate_artifact_roots_to_artifacts_root.py --mode move --apply) "
            "or set CAIA_STORES_ALLOW_LEGACY=1 temporarily."
        )

    # Legacy exists but is temporarily allowed: still prefer the canonical path. This avoids
    # split-brain reads/writes while letting operators keep legacy files around during migration.
    _canonical_rel_path(canonical_rel)
    return canonical


def resolve_store_dir(
    *,
    legacy_rel: Optional[str],
    canonical_rel: str,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """
    Resolve a store directory path (no per-tool env override).

    Precedence:
      1) Canonical dir (new install)

    Legacy paths are rejected when they exist to avoid split-brain. Migrate first.
    """
    canonical = (stores_root(repo_root=repo_root) / Path(str(canonical_rel))).resolve()
    legacy = (resolve_repo_path(legacy_rel, repo_root=repo_root) if legacy_rel else None)

    legacy_exists = bool(legacy and _path_exists(legacy))
    if legacy_exists and not _truthy_env("CAIA_STORES_ALLOW_LEGACY"):
        raise RuntimeError(
            f"Legacy store dir exists at {legacy}. Refusing to use it. "
            f"Migrate it to {canonical} (scripts/ops/migrate_artifact_roots_to_artifacts_root.py --mode move --apply) "
            "or set CAIA_STORES_ALLOW_LEGACY=1 temporarily."
        )

    # Legacy exists but is temporarily allowed: still prefer the canonical dir.
    _canonical_rel_path(canonical_rel)
    return canonical


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
