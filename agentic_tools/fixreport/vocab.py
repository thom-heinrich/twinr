"""
Contract
- Purpose: Load and validate FixReport entries against the controlled vocab YAML.
- Inputs (types, units): Python dict candidate with fixreport fields.
- Outputs (types, units): (ok, errors). Some safe normalizations may be applied in-place
  when `fields` is a mutable mapping (e.g., YAML 1.1 boolean scalars for enum fields).
- Invariants: Required fields are present; enum values are from allowed sets.
- Error semantics: Returns a list of human-actionable validation errors.
- Time/Horizon: Validates ts_utc ISO-8601 UTC "Z" format.
- External boundaries: Reads agentic_tools/fixreport/vocab.yaml.
"""

##REFACTOR: 2026-01-16##

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from collections.abc import MutableMapping

import yaml


_SUPPORTED_TYPES = {"timestamp", "enum", "token", "path", "list_token", "list_path"}

_BF_ID_RE = re.compile(r"^BF[0-9]{6}$")
_TOKEN_RE = re.compile(r"^[^\s]+$")  # allow dots/colons; forbid whitespace
_ISO8601_UTC_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

# YAML 1.1 historically treats multiple plain scalars as booleans (e.g., yes/no/on/off/y/n/true/false).
# When a YAML loader coerces such tokens to Python bools, we normalize deterministically to "yes"/"no"
# and also include common synonyms to reduce surprise across producers.
_YAML_BOOL_TRUE_SYNONYMS = ("yes", "true", "on", "y")
_YAML_BOOL_FALSE_SYNONYMS = ("no", "false", "off", "n")


def _is_iso8601_utc_z(ts: str) -> bool:
    # Strictly validate the existing contract format: YYYY-MM-DDTHH:MM:SSZ
    if not isinstance(ts, str) or not _ISO8601_UTC_Z_RE.fullmatch(ts):
        return False
    try:
        # Also validates day/month ranges.
        datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return False
    return True


def _coerce_bool(val: Any, default: bool = False) -> bool:
    """Best-effort coercion for YAML-ish booleans without changing behavior for real bools."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        # Preserve legacy truthiness but keep common 0/1 behavior.
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"true", "yes", "y", "on", "1"}:
            return True
        if s in {"false", "no", "n", "off", "0"}:
            return False
    return bool(val)


def _normalize_allowed_item(item: Any) -> List[str]:
    # Normalize YAML-coerced booleans into deterministic enum tokens + common synonyms.
    if item is True:
        return list(_YAML_BOOL_TRUE_SYNONYMS)
    if item is False:
        return list(_YAML_BOOL_FALSE_SYNONYMS)
    return [str(item)]


def _normalize_enum_value_for_validation(val: Any) -> Tuple[Optional[str], bool]:
    """
    Returns (normalized_value, mutated).
    - normalized_value is a string if it can be safely treated as an enum token, else None.
    - mutated indicates whether the caller should consider writing back normalization (handled elsewhere).
    """
    if isinstance(val, str):
        return val, False
    if val is True:
        return "yes", True
    if val is False:
        return "no", True
    return None, False


@lru_cache(maxsize=4096)
def _compile_pattern_cached(pattern: str) -> Optional[re.Pattern]:
    try:
        return re.compile(pattern)
    except re.error:
        return None


def _get_compiled_pattern(vocab: "Vocab", field_name: str, pattern: Optional[str]) -> Optional[re.Pattern]:
    if not pattern:
        return None
    compiled_map = getattr(vocab, "_compiled_patterns", None)
    if isinstance(compiled_map, dict):
        compiled = compiled_map.get(field_name)
        if compiled is not None:
            return compiled
        # If we explicitly stored a None (invalid regex), return None.
        if field_name in compiled_map:
            return None
    return _compile_pattern_cached(pattern)


def _should_enforce_safe_paths(vocab: "Vocab") -> bool:
    rules = getattr(vocab, "rules", None)
    if not isinstance(rules, Mapping):
        return False
    # Optional opt-in knobs; default preserves legacy behavior.
    if _coerce_bool(rules.get("enforce_safe_paths"), default=False):
        return True
    policy = rules.get("path_policy")
    if isinstance(policy, str) and policy.strip().lower() in {"safe_relative", "relative_safe", "safe"}:
        return True
    return False


def _is_safe_relative_path_token(p: str) -> bool:
    """
    Minimal safe-path check. Only used when explicitly enabled by vocab.rules.
    Keeps legacy behavior by default.
    """
    if "\x00" in p:
        return False
    # Disallow absolute POSIX paths
    if p.startswith("/"):
        return False
    # Disallow Windows drive-absolute and UNC-style
    if re.match(r"^[A-Za-z]:[\\/]", p):
        return False
    if p.startswith("\\\\"):
        return False
    # Disallow traversal segments
    parts = re.split(r"[\\/]+", p)
    if any(part == ".." for part in parts):
        return False
    return True


@dataclass(frozen=True)
class VocabField:
    name: str
    type: str
    required: bool
    free: bool
    allowed: Optional[List[str]] = None
    pattern: Optional[str] = None


@dataclass(frozen=True)
class Vocab:
    version: int
    rules: Mapping[str, Any]
    fields: Mapping[str, VocabField]


def load_vocab(vocab_path: Path) -> Vocab:
    raw = yaml.safe_load(vocab_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("vocab.yaml is not a mapping")
    schema = raw.get("schema")
    if not isinstance(schema, dict):
        raise ValueError("vocab.yaml missing schema mapping")

    vocab_errors: List[str] = []
    compiled_patterns: Dict[str, Optional[re.Pattern]] = {}
    fields: Dict[str, VocabField] = {}

    for name, spec in schema.items():
        if not isinstance(spec, dict):
            # Historically this was silently skipped; now we fail-closed via a vocab error,
            # but keep the module API stable (no new exception thrown here).
            vocab_errors.append(f"vocab error: schema field {name} must be a mapping")
            fields[name] = VocabField(
                name=name,
                type="",
                required=False,
                free=False,
                allowed=None,
                pattern=None,
            )
            compiled_patterns[name] = None
            continue

        allowed_raw = spec.get("allowed")
        allowed_norm: Optional[List[str]] = None
        if isinstance(allowed_raw, list):
            allowed_norm = []
            seen: set[str] = set()
            for item in allowed_raw:
                for tok in _normalize_allowed_item(item):
                    if tok not in seen:
                        allowed_norm.append(tok)
                        seen.add(tok)

        # Preserve prior behavior for missing keys, but with stronger type/shape checking.
        type_raw = spec.get("type", "")
        ftype = ""
        if type_raw is None:
            ftype = ""
        else:
            ftype = str(type_raw).strip()

        if not ftype:
            vocab_errors.append(f"vocab error: field {name} missing type")
        elif ftype not in _SUPPORTED_TYPES:
            vocab_errors.append(f"vocab error: field {name} has unsupported type '{ftype}'")

        pattern_val = spec.get("pattern")
        pattern = str(pattern_val) if pattern_val else None
        if pattern:
            try:
                compiled_patterns[name] = re.compile(pattern)
            except re.error as e:
                vocab_errors.append(f"vocab error: field {name} has invalid pattern {pattern}: {e}")
                compiled_patterns[name] = None
        else:
            compiled_patterns[name] = None

        fields[name] = VocabField(
            name=name,
            type=ftype,
            required=_coerce_bool(spec.get("required", False), default=False),
            free=_coerce_bool(spec.get("free", False), default=False),
            allowed=allowed_norm,
            pattern=pattern,
        )

    vocab = Vocab(
        version=int(raw.get("vocab_version", 0) or 0),
        rules=raw.get("rules") if isinstance(raw.get("rules"), dict) else {},
        fields=fields,
    )

    # Attach internal, backward-compatible metadata without changing dataclass shape.
    try:
        object.__setattr__(vocab, "_vocab_errors", tuple(vocab_errors))
        object.__setattr__(vocab, "_compiled_patterns", compiled_patterns)
    except Exception:
        # If the environment disallows this, validation still works with runtime compilation.
        pass

    return vocab


def validate_fixreport_fields(vocab: Vocab, fields: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if not isinstance(fields, Mapping):
        return False, ["fixreport must be a mapping/object"]

    # Fail-closed on vocab integrity issues (recorded during load_vocab or via runtime checks).
    vocab_errs = getattr(vocab, "_vocab_errors", None)
    if isinstance(vocab_errs, (list, tuple)):
        errors.extend([str(e) for e in vocab_errs if e])

    for fname, f in vocab.fields.items():
        if f.required and fname not in fields:
            errors.append(f"missing required field: {fname}")

    enforce_safe_paths = _should_enforce_safe_paths(vocab)

    # Validate only fields defined in vocab; allow extra keys (narrative/evidence, etc.)
    for key, val in fields.items():
        if key not in vocab.fields:
            continue
        spec = vocab.fields[key]
        ftype = spec.type

        if ftype == "timestamp":
            if not isinstance(val, str) or not _is_iso8601_utc_z(val):
                errors.append(f"invalid {key}: expected ISO-8601 UTC '...Z' timestamp")
            continue

        if ftype == "enum":
            norm, mutated = _normalize_enum_value_for_validation(val)
            if norm is None:
                errors.append(f"invalid {key}: expected string enum")
                continue

            # Optional safe normalization in-place (for YAML 1.1 boolean scalars).
            if mutated and isinstance(fields, MutableMapping):
                try:
                    fields[key] = norm  # type: ignore[index]
                except Exception:
                    pass

            if spec.allowed is not None and norm not in spec.allowed:
                errors.append(f"invalid {key}: '{norm}' not in allowed set")
            continue

        if ftype in {"token", "path"}:
            if not isinstance(val, str) or not val:
                errors.append(f"invalid {key}: expected non-empty string")
                continue
            if "\x00" in val:
                errors.append(f"invalid {key}: token contains NUL byte")
                continue
            if not _TOKEN_RE.fullmatch(val):
                errors.append(f"invalid {key}: token contains whitespace")
                continue

            if enforce_safe_paths and ftype == "path":
                if not _is_safe_relative_path_token(val):
                    errors.append(f"invalid {key}: unsafe path")
                    continue

            if spec.pattern:
                compiled = _get_compiled_pattern(vocab, key, spec.pattern)
                if compiled is None:
                    # Vocab error; fail-closed without throwing.
                    errors.append(f"vocab error: field {key} has invalid pattern {spec.pattern}")
                elif compiled.fullmatch(val) is None:
                    errors.append(f"invalid {key}: does not match pattern {spec.pattern}")

            if key == "bf_id" and not _BF_ID_RE.fullmatch(val):
                errors.append("invalid bf_id: must match ^BF[0-9]{6}$")
            continue

        if ftype in {"list_token", "list_path"}:
            if val is None:
                if spec.required:
                    errors.append(f"invalid {key}: expected list")
                continue
            if not isinstance(val, list):
                errors.append(f"invalid {key}: expected list")
                continue
            if spec.required and len(val) == 0:
                errors.append(f"invalid {key}: expected non-empty list")
                continue

            compiled = None
            if spec.pattern:
                compiled = _get_compiled_pattern(vocab, key, spec.pattern)
                if compiled is None:
                    errors.append(f"vocab error: field {key} has invalid pattern {spec.pattern}")

            for i, item in enumerate(val):
                if not isinstance(item, str) or not item or "\x00" in item or not _TOKEN_RE.fullmatch(item):
                    errors.append(f"invalid {key}[{i}]: expected non-empty token string without whitespace")
                    continue

                if enforce_safe_paths and ftype == "list_path":
                    if not _is_safe_relative_path_token(item):
                        errors.append(f"invalid {key}[{i}]: unsafe path")
                        continue

                if compiled is not None and compiled.fullmatch(item) is None:
                    errors.append(f"invalid {key}[{i}]: does not match pattern {spec.pattern}")

                if key == "bf_id" and not _BF_ID_RE.fullmatch(item):
                    errors.append(f"invalid {key}[{i}]: must match ^BF[0-9]{{6}}$")
                if spec.allowed is not None and item not in spec.allowed:
                    errors.append(f"invalid {key}[{i}]: '{item}' not in allowed set")
            continue

        if not ftype:
            # Unknown/missing type: be permissive but surface.
            errors.append(f"vocab error: field {key} missing type")
            continue

        # Unsupported non-empty type: surface as vocab error (previously would silently skip validation).
        errors.append(f"vocab error: field {key} has unsupported type '{ftype}'")

    return (len(errors) == 0), errors
