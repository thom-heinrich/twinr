"""
Contract
- Purpose: Define typed structures for FixReport storage and CLI I/O.
- Inputs (types, units): Python dicts decoded from YAML/CLI.
- Outputs (types, units): Dataclasses/TypedDicts for internal correctness.
- Invariants: `bf_id` matches ^BF[0-9]{6}$; timestamps are UTC ISO strings.
- Error semantics: Validation is handled in vocab/validation layer, not here.
- External boundaries: None.
"""

##REFACTOR: 2026-01-16##

from __future__ import annotations

from dataclasses import dataclass, field
import copy as _copy
import inspect as _inspect
import os as _os
import sys as _sys
from typing import Any, Dict, List, Mapping, Optional, TypedDict


# Optional performance knob (opt-in to avoid breaking drop-in expectations around __dict__).
# When enabled and supported by the runtime (Python >= 3.10), dataclass slots are used.
_USE_SLOTS: bool = (
    _os.getenv("FIXREPORT_ENABLE_SLOTS", "").strip() == "1" and _sys.version_info >= (3, 10)
)
_DATACLASS_KWARGS = {"frozen": True}
if _USE_SLOTS:
    # `slots` is available on @dataclass() starting with Python 3.10.
    _DATACLASS_KWARGS["slots"] = True


def _safe_shallow_copy_list(value: Optional[List[Any]]) -> Optional[List[Any]]:
    """
    Best-effort shallow copy for list-like values.

    Goal: reduce external aliasing (mutations to the input list after construction)
    without introducing new exceptions or enforcing deep immutability (which could be breaking).
    """
    if value is None:
        return None
    if type(value) is list:
        # Fast path, doesn't invoke copy protocol hooks.
        return value.copy()
    if isinstance(value, list):
        try:
            return _copy.copy(value)
        except Exception:
            # Preserve original behavior if copying fails for exotic subclasses.
            return value
    return value


def _safe_shallow_copy_dict(value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Best-effort shallow copy for dict-like values.

    Same rationale as _safe_shallow_copy_list().
    """
    if value is None:
        return None
    if type(value) is dict:
        return value.copy()
    if isinstance(value, dict):
        try:
            return _copy.copy(value)
        except Exception:
            return value
    return value


def _safe_shallow_copy_evidence(
    value: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Best-effort shallow copy for evidence: list[dict[str, Any]].

    Copies:
      - the top-level list (to avoid input aliasing),
      - each dict item (to avoid shared dict references).
    """
    if value is None:
        return None
    if not isinstance(value, list):
        return value

    copied_list = _safe_shallow_copy_list(value)
    if copied_list is None:
        return None

    # If list copying failed and we returned the original, keep behavior and avoid extra work.
    if copied_list is value:
        return value

    for idx, item in enumerate(copied_list):
        if isinstance(item, dict):
            copied_list[idx] = _safe_shallow_copy_dict(item) or item
    return copied_list


class FixReportRequired(TypedDict):
    bf_id: str
    ts_utc: str
    commit: str
    repo_area: str
    target_kind: str
    target_path: str

    scope: str
    mode: str
    bug_type: str
    symptom: str
    root_cause: str
    failure_mode: str
    fix_type: str
    impact_area: str
    severity: str
    verification: str


class FixReportTypedDict(FixReportRequired, total=False):
    # Optional fields (mirrors FixReport)
    paths_touched: Optional[List[str]]
    systemd_unit: Optional[str]
    contracts: Optional[List[str]]
    topics: Optional[List[str]]
    tables_views: Optional[List[str]]
    asset: Optional[str]
    venue: Optional[str]
    instrument: Optional[str]
    horizon: Optional[str]

    tests_added: Optional[str]
    breaking: Optional[str]
    backfill_needed: Optional[str]

    hypothesis_ids: Optional[List[str]]
    task_ids: Optional[List[str]]
    ops_msg_ids: Optional[List[str]]
    audit_ids: Optional[List[str]]

    links: Optional[List[str]]

    signature: Optional[str]
    tags: Optional[List[str]]

    created_by: Optional[str]

    narrative: Optional[Dict[str, Any]]
    evidence: Optional[List[Dict[str, Any]]]


@dataclass(**_DATACLASS_KWARGS)
class FixReport:
    bf_id: str
    ts_utc: str
    commit: str
    repo_area: str
    target_kind: str
    target_path: str

    scope: str
    mode: str
    bug_type: str
    symptom: str
    root_cause: str
    failure_mode: str
    fix_type: str
    impact_area: str
    severity: str
    verification: str

    # Optional fields
    paths_touched: Optional[List[str]] = field(default=None, hash=False)
    systemd_unit: Optional[str] = None
    contracts: Optional[List[str]] = field(default=None, hash=False)
    topics: Optional[List[str]] = field(default=None, hash=False)
    tables_views: Optional[List[str]] = field(default=None, hash=False)
    asset: Optional[str] = None
    venue: Optional[str] = None
    instrument: Optional[str] = None
    horizon: Optional[str] = None

    tests_added: Optional[str] = None
    breaking: Optional[str] = None
    backfill_needed: Optional[str] = None

    hypothesis_ids: Optional[List[str]] = field(default=None, hash=False)
    task_ids: Optional[List[str]] = field(default=None, hash=False)
    ops_msg_ids: Optional[List[str]] = field(default=None, hash=False)
    audit_ids: Optional[List[str]] = field(default=None, hash=False)

    # Generic cross-tool outbound links (kind:id). Optional and additive to structured IDs above.
    links: Optional[List[str]] = field(default=None, hash=False)

    signature: Optional[str] = None
    tags: Optional[List[str]] = field(default=None, hash=False)

    created_by: Optional[str] = None

    narrative: Optional[Dict[str, Any]] = field(default=None, hash=False, repr=False)
    evidence: Optional[List[Dict[str, Any]]] = field(default=None, hash=False, repr=False)

    def __post_init__(self) -> None:
        # Defensive copies: reduce accidental aliasing with caller-owned mutable inputs.
        object.__setattr__(self, "paths_touched", _safe_shallow_copy_list(self.paths_touched))
        object.__setattr__(self, "contracts", _safe_shallow_copy_list(self.contracts))
        object.__setattr__(self, "topics", _safe_shallow_copy_list(self.topics))
        object.__setattr__(self, "tables_views", _safe_shallow_copy_list(self.tables_views))

        object.__setattr__(self, "hypothesis_ids", _safe_shallow_copy_list(self.hypothesis_ids))
        object.__setattr__(self, "task_ids", _safe_shallow_copy_list(self.task_ids))
        object.__setattr__(self, "ops_msg_ids", _safe_shallow_copy_list(self.ops_msg_ids))
        object.__setattr__(self, "audit_ids", _safe_shallow_copy_list(self.audit_ids))

        object.__setattr__(self, "links", _safe_shallow_copy_list(self.links))
        object.__setattr__(self, "tags", _safe_shallow_copy_list(self.tags))

        object.__setattr__(self, "narrative", _safe_shallow_copy_dict(self.narrative))
        object.__setattr__(self, "evidence", _safe_shallow_copy_evidence(self.evidence))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FixReport":
        """
        Keyword-only friendly constructor (mitigates positional-argument footguns)
        without changing the existing __init__ behavior/signature.
        """
        return cls(**data)


def get_fixreport_annotations(*, eval_str: bool = False) -> Dict[str, Any]:
    """
    Helper for robust annotation introspection across Python versions/runtimes.

    - eval_str=False: returns raw stored annotations (may be strings under PEP 563/future import).
    - eval_str=True: best-effort evaluation of string annotations.
    """
    try:
        return dict(_inspect.get_annotations(FixReport, eval_str=eval_str))
    except Exception:
        # Fallback to raw annotations if introspection fails for any reason.
        return dict(getattr(FixReport, "__annotations__", {}))


FixReportDict = Dict[str, Any]
