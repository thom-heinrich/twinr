"""
Contract
- Purpose: Provide a shared, validated failure-modes checklist for FixReports.
- Inputs (types, units): YAML catalog + a user-provided failure_modes block.
- Outputs (types, units): (ok, errors) validation; JSON-friendly catalog payload.
- Invariants: Only known FM_* ids; status in {pass,fail,unknown,na}; catalog_version matches.
- Error semantics: Fail-fast, human-actionable error strings.
- External boundaries: Reads agentic_tools/fixreport/failure_modes_catalog.yaml.
"""

##REFACTOR: 2026-01-16##

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
from types import MappingProxyType

import yaml


_ALLOWED_STATUSES = {"pass", "fail", "unknown", "na"}


@dataclass(frozen=True)
class FailureModeCheck:
    fm_id: str
    title: str
    description: str


@dataclass(frozen=True)
class FailureModesCatalog:
    version: int
    checks: Mapping[str, FailureModeCheck]

    def to_json(self) -> Dict[str, Any]:
        return {
            "catalog_version": int(self.version),
            "allowed_status": sorted(_ALLOWED_STATUSES),
            "checks": [
                {"id": c.fm_id, "title": c.title, "description": c.description}
                for c in sorted(self.checks.values(), key=lambda x: x.fm_id)
            ],
        }


def _yaml_safe_load_fast(text: str) -> Any:
    """
    Safe YAML parsing with an optional fast path when LibYAML-backed loaders exist.
    Behavior for valid YAML should match yaml.safe_load() semantics (safe constructors).
    """
    # PyYAML exposes CSafeLoader when built with LibYAML, which is significantly faster.
    loader = getattr(yaml, "CSafeLoader", None)
    if loader is None:
        return yaml.safe_load(text)
    # Explicit Loader avoids yaml.load default loader risks; CSafeLoader is "safe".
    return yaml.load(text, Loader=loader)


def _raise_wrapped(exc: BaseException, *, context: str) -> None:
    # Keep wrapping opt-in to preserve drop-in exception semantics by default.
    raise ValueError(context) from exc


def _parse_failure_modes_catalog_raw(
    raw: Any,
    *,
    strict: bool,
    id_prefix: Optional[str],
) -> Tuple[int, Dict[str, FailureModeCheck], List[str]]:
    if not isinstance(raw, Mapping):
        raise ValueError("failure_modes_catalog.yaml is not a mapping")

    ver = int(raw.get("catalog_version", 0) or 0)
    checks_raw = raw.get("checks")
    if not isinstance(checks_raw, Mapping):
        raise ValueError("failure_modes_catalog.yaml missing checks mapping")

    checks: Dict[str, FailureModeCheck] = {}
    parse_errors: List[str] = []

    for k, v in checks_raw.items():
        if not isinstance(k, str) or not k:
            if strict:
                parse_errors.append(f"invalid check id (must be non-empty string): {k!r}")
            continue
        if id_prefix is not None and not k.startswith(id_prefix):
            if strict:
                parse_errors.append(f"invalid check id (expected prefix {id_prefix!r}): {k}")
            continue
        if not isinstance(v, Mapping):
            if strict:
                parse_errors.append(f"check {k} must be a mapping/object")
            continue

        title = str(v.get("title") or "").strip()
        desc = str(v.get("description") or "").strip()
        if not title:
            title = k
        checks[k] = FailureModeCheck(fm_id=k, title=title, description=desc)

    if strict and parse_errors:
        raise ValueError("failure_modes_catalog.yaml invalid checks:\n- " + "\n- ".join(parse_errors))

    if not checks:
        raise ValueError("failure_modes_catalog.yaml has no checks")

    return ver, checks, parse_errors


@lru_cache(maxsize=32)
def _load_failure_modes_catalog_cached(
    path_str: str,
    mtime_ns: int,
    size: int,
    strict: bool,
    id_prefix: Optional[str],
) -> Tuple[int, Tuple[Tuple[str, str, str], ...]]:
    """
    Cache the parsed catalog keyed by resolved path + mtime/size + parsing options.
    Returns an immutable representation to avoid shared-mutable-state hazards.
    """
    p = Path(path_str)
    text = p.read_text(encoding="utf-8")
    raw = _yaml_safe_load_fast(text) or {}
    ver, checks, _ = _parse_failure_modes_catalog_raw(raw, strict=strict, id_prefix=id_prefix)

    # Stable sorted tuple for immutability + deterministic caching.
    items = tuple(sorted(((k, c.title, c.description) for k, c in checks.items()), key=lambda x: x[0]))
    return ver, items


def load_failure_modes_catalog(
    path: Path,
    *,
    strict: bool = False,
    id_prefix: Optional[str] = None,
    wrap_exceptions: bool = False,
    freeze: bool = False,
    use_cache: bool = False,
) -> FailureModesCatalog:
    """
    Load and parse the failure-modes catalog YAML.

    Backwards-compatible defaults:
    - strict=False preserves "skip invalid entries" behavior.
    - wrap_exceptions=False preserves original exception types from I/O/YAML parsing.
    - freeze=False preserves mutability of catalog.checks mapping.
    - use_cache=False preserves always-read-and-parse behavior.
    """
    if use_cache:
        try:
            st = path.stat()
            ver, items = _load_failure_modes_catalog_cached(
                str(path.resolve()),
                int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))),
                int(st.st_size),
                bool(strict),
                id_prefix,
            )
        except (OSError, yaml.YAMLError, ValueError) as e:
            if wrap_exceptions:
                _raise_wrapped(e, context=f"failed to load failure_modes catalog from {path}")
            raise

        checks_dict: Dict[str, FailureModeCheck] = {
            k: FailureModeCheck(fm_id=k, title=title, description=desc) for (k, title, desc) in items
        }
        checks_map: Mapping[str, FailureModeCheck] = MappingProxyType(checks_dict) if freeze else checks_dict
        return FailureModesCatalog(version=ver, checks=checks_map)

    try:
        text = path.read_text(encoding="utf-8")
        raw = _yaml_safe_load_fast(text) or {}
        ver, checks_dict, _ = _parse_failure_modes_catalog_raw(raw, strict=strict, id_prefix=id_prefix)
        checks_map = MappingProxyType(checks_dict) if freeze else checks_dict
        return FailureModesCatalog(version=ver, checks=checks_map)
    except (OSError, yaml.YAMLError, ValueError) as e:
        if wrap_exceptions:
            _raise_wrapped(e, context=f"failed to load failure_modes catalog from {path}")
        raise


def load_failure_modes_catalog_from_text(
    text: str,
    *,
    strict: bool = False,
    id_prefix: Optional[str] = None,
    freeze: bool = False,
    wrap_exceptions: bool = False,
) -> FailureModesCatalog:
    """
    Convenience loader for callers that already have the YAML content in-memory.
    Does not touch the filesystem.
    """
    try:
        raw = _yaml_safe_load_fast(text) or {}
        ver, checks_dict, _ = _parse_failure_modes_catalog_raw(raw, strict=strict, id_prefix=id_prefix)
        checks_map = MappingProxyType(checks_dict) if freeze else checks_dict
        return FailureModesCatalog(version=ver, checks=checks_map)
    except (yaml.YAMLError, ValueError) as e:
        if wrap_exceptions:
            _raise_wrapped(e, context="failed to load failure_modes catalog from text")
        raise


def load_default_failure_modes_catalog(
    *,
    strict: bool = False,
    id_prefix: Optional[str] = None,
    freeze: bool = False,
    wrap_exceptions: bool = False,
) -> FailureModesCatalog:
    """
    Load the default catalog from package resources if available.
    Falls back to a conventional on-disk location relative to this module if needed.

    This is additive (does not change existing APIs) and improves packaging robustness.
    """
    resource_text: Optional[str] = None

    # Preferred: importlib.resources (works for wheels/zipimport where files may not exist on disk).
    try:
        import importlib.resources as resources  # stdlib
    except Exception:
        resources = None  # type: ignore[assignment]

    if resources is not None:
        try:
            # Contract boundary: agentic_tools/fixreport/failure_modes_catalog.yaml
            resource_text = (
                resources.files("agentic_tools.fixreport")
                .joinpath("failure_modes_catalog.yaml")
                .read_text(encoding="utf-8")
            )
        except Exception:
            resource_text = None

    # Fallback: try relative file next to this module (best-effort, preserves old deployment patterns).
    if resource_text is None:
        try:
            candidate = Path(__file__).resolve().parent / "failure_modes_catalog.yaml"
            resource_text = candidate.read_text(encoding="utf-8")
        except Exception as e:
            if wrap_exceptions:
                _raise_wrapped(
                    e, context="failed to load default failure_modes catalog (resource and filesystem fallback)"
                )
            raise

    return load_failure_modes_catalog_from_text(
        resource_text,
        strict=strict,
        id_prefix=id_prefix,
        freeze=freeze,
        wrap_exceptions=wrap_exceptions,
    )


def default_failure_modes_block(*, catalog: FailureModesCatalog) -> Dict[str, Any]:
    return {
        "catalog_version": int(catalog.version),
        "checks": {k: "unknown" for k in sorted(catalog.checks.keys())},
        "notes": "",
    }


def validate_failure_modes_block(
    *,
    catalog: FailureModesCatalog,
    block: Any,
    require_block: bool = False,
    require_all_checks: bool = False,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if block is None:
        if require_block:
            return False, ["failure_modes is required"]
        return True, []
    if not isinstance(block, Mapping):
        return False, ["failure_modes must be a mapping/object"]

    ver = block.get("catalog_version")
    if ver is None:
        errors.append("failure_modes.catalog_version is required")
    else:
        try:
            ver_i = int(ver)
        except Exception:
            ver_i = -1
        if ver_i != int(catalog.version):
            errors.append(f"failure_modes.catalog_version mismatch: {ver_i} != {catalog.version}")

    checks = block.get("checks")
    if checks is None:
        errors.append("failure_modes.checks is required")
        return False, errors
    if not isinstance(checks, Mapping):
        errors.append("failure_modes.checks must be a mapping")
        return False, errors

    for k, v in checks.items():
        if not isinstance(k, str) or not k:
            errors.append("failure_modes.checks has non-string key")
            continue
        if k not in catalog.checks:
            errors.append(f"failure_modes.checks contains unknown id: {k}")
            continue
        if not isinstance(v, str) or not v:
            errors.append(f"failure_modes.checks[{k}] must be a status token")
            continue
        if v not in _ALLOWED_STATUSES:
            errors.append(f"failure_modes.checks[{k}] invalid status: {v}")

    if require_all_checks:
        provided_ids = {k for k in checks.keys() if isinstance(k, str) and k}
        missing = set(catalog.checks.keys()) - provided_ids
        if missing:
            errors.append("failure_modes.checks missing ids: " + ", ".join(sorted(missing)))

    notes = block.get("notes")
    if notes is not None and not isinstance(notes, str):
        errors.append("failure_modes.notes must be a string")

    return (len(errors) == 0), errors


def parse_fm_assignments(*, assignments: List[str]) -> Tuple[Optional[Dict[str, str]], List[str]]:
    """
    Parse CLI assignments of the form FM_ID=status.
    Returns (mapping or None, errors).
    """
    out: Dict[str, str] = {}
    errors: List[str] = []
    for raw in assignments:
        s = str(raw or "").strip()
        if not s:
            continue
        if "=" not in s:
            errors.append(f"invalid --fm value (expected FM_ID=status): {s}")
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            errors.append(f"invalid --fm value (empty key/value): {s}")
            continue
        out[k] = v
    return (out or None), errors
