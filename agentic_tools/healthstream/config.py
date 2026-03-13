"""
Contract
- Purpose:
  - Central configuration defaults for the healthstream tool (store path, timeouts).
- Inputs (types, units):
  - Env vars: HEALTHSTREAM_FILE, HEALTHSTREAM_LOCK_TIMEOUT_SEC.
- Outputs (types, units):
  - Resolved store path, numeric defaults.
- Invariants:
  - Default store location is repo-local `artifacts/stores/healthstream/healthstream.json`
    and legacy locations are rejected to avoid split-brain (migrate first).
- Error semantics:
  - Raises `ValueError` for invalid env var values.
- External boundaries:
  - Reads env vars only.
- Telemetry:
  - None.
"""

##REFACTOR: 2026-01-16##

import math
import os
from pathlib import Path


def _env_truthy(key: str) -> bool:
    return (os.environ.get(key) or "").strip().lower() in ("1", "true", "yes", "y", "on")


def _path_exists_strict(p: Path) -> bool:
    """
    Strict existence check:
    - True if stat() succeeds
    - False if FileNotFoundError
    - Re-raises all other OSError (e.g., PermissionError), preserving pre-Python-3.14 semantics
      where Path.exists() could raise.
    """
    try:
        p.stat()
        return True
    except FileNotFoundError:
        return False


def _looks_like_repo_root(p: Path) -> bool:
    # Best-effort marker check (avoid raising on weird filesystems).
    try:
        return (
            (p / "pyproject.toml").is_file()
            or (p / ".git").exists()
            or (p / "artifacts").exists()
        )
    except OSError:
        return False


def _find_repo_root(start: Path) -> Path | None:
    # Walk upwards looking for common repo markers.
    for cand in (start, *start.parents):
        if _looks_like_repo_root(cand):
            return cand
    return None


def _determine_repo_root() -> Path:
    # Optional explicit override (kept narrow to avoid unintended changes in environments).
    override = (os.environ.get("HEALTHSTREAM_REPO_ROOT") or "").strip()
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()

    # Preserve original behavior as primary candidate.
    try:
        candidate = here.parents[2]
    except IndexError:
        candidate = here.parent

    # If candidate doesn't look like a repo root, fall back to marker-based search.
    if _looks_like_repo_root(candidate):
        return candidate

    found = _find_repo_root(here.parent)
    return found if found is not None else candidate


REPO_ROOT = _determine_repo_root()

from agentic_tools._store_layout import resolve_repo_path, resolve_store_file


def store_path_from_env() -> Path:
    # Env override always wins (repo-relative allowed).
    raw = (os.environ.get("HEALTHSTREAM_FILE") or "").strip()
    allow_legacy = _env_truthy("CAIA_STORES_ALLOW_LEGACY")

    legacy_candidates = [
        (REPO_ROOT / "state" / "healthstream" / ".healthstream.json").resolve(),
        (REPO_ROOT / ".healthstream.json").resolve(),
    ]

    def _raise_legacy(legacy: Path) -> None:
        raise RuntimeError(
            f"Legacy healthstream store exists at {legacy}. "
            "Migrate to artifacts/stores/healthstream/healthstream.json first "
            "(scripts/ops/migrate_artifact_roots_to_artifacts_root.py --mode move --apply) "
            "or set CAIA_STORES_ALLOW_LEGACY=1 temporarily."
        )

    # Enforce split-brain protection consistently (including when HEALTHSTREAM_FILE is set).
    if not allow_legacy:
        for legacy in legacy_candidates:
            if _path_exists_strict(legacy):
                _raise_legacy(legacy)

    if raw:
        return resolve_repo_path(raw, repo_root=REPO_ROOT)

    # If legacy is explicitly allowed, prefer existing legacy stores deterministically.
    if allow_legacy:
        for legacy in legacy_candidates:
            if _path_exists_strict(legacy):
                return legacy

    return resolve_store_file(
        env_keys=(),
        legacy_rel="state/healthstream/.healthstream.json",
        canonical_rel="healthstream/healthstream.json",
        repo_root=REPO_ROOT,
    )


def lock_timeout_sec_from_env() -> float:
    raw = os.environ.get("HEALTHSTREAM_LOCK_TIMEOUT_SEC", "10").strip()
    try:
        val = float(raw)
    except ValueError as exc:
        raise ValueError(f"invalid HEALTHSTREAM_LOCK_TIMEOUT_SEC={raw!r}") from exc

    if not math.isfinite(val) or val <= 0:
        if math.isfinite(val):
            raise ValueError("HEALTHSTREAM_LOCK_TIMEOUT_SEC must be > 0")
        raise ValueError("HEALTHSTREAM_LOCK_TIMEOUT_SEC must be a finite number > 0")

    return val
