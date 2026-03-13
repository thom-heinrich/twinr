from __future__ import annotations

import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


LINK_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]{0,32}:[A-Za-z0-9._:/@+-]{1,256}$")
REPORT_ID_RE = re.compile(r"^RPT[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")

REPORT_KINDS: Tuple[str, ...] = ("research", "benchmark", "deployment", "ops", "release", "other")
REPORT_STATUSES: Tuple[str, ...] = ("draft", "final")
BENCHMARK_HINT_TAGS: Tuple[str, ...] = ("benchmark", "bench", "test", "tests", "evaluation", "eval", "ablation")

REPORT_QUALITY_POLICY: Dict[str, Any] = {
    "min_summary_items": 2,
    "min_insight_items": 3,
    "min_evidence_items": 2,
    "min_limitations_items": 1,
    "min_risks_items": 1,
    "min_next_actions_items": 3,
    "min_repro_steps": 1,
    "min_explanatory_chars": 24,
    "context_required_keys": ["objective", "scope", "methodology", "data_window"],
    "min_kpis_generic": 1,
    "benchmark_min_kpis": 3,
    "benchmark_min_tests": 1,
    "benchmark_required_kpi_fields": ["name", "value", "baseline", "delta", "explanation"],
}


def utc_now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_report_id(*, now_utc: Optional[str] = None) -> str:
    ts = (now_utc or "").strip()
    if not ts:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = secrets.token_hex(4)
    return f"RPT{ts}_{suffix}"


def is_safe_report_id(report_id: str) -> bool:
    return bool(REPORT_ID_RE.fullmatch((report_id or "").strip()))


def _safe_str(v: Any) -> str:
    return "" if v is None else str(v)


def _safe_text(v: Any) -> str:
    return _safe_str(v).strip()


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for raw in items:
        s = str(raw or "").strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def normalize_tags(value: Any) -> List[str]:
    items = [_safe_str(x).strip() for x in _as_list(value)]
    items = [x for x in items if x]
    # Keep tags human-friendly: no forced lowercasing, but trim whitespace and dedupe.
    return _dedupe_keep_order(items)


def normalize_lines(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # Allow a single string with newlines; split to list.
        lines = [x.strip() for x in value.splitlines() if x.strip()]
        return _dedupe_keep_order(lines)
    if isinstance(value, list):
        lines = [_safe_str(x).strip() for x in value if _safe_str(x).strip()]
        return _dedupe_keep_order(lines)
    s = _safe_str(value).strip()
    return [s] if s else []


def normalize_links(value: Any) -> Tuple[List[str], List[str]]:
    raw = [_safe_str(x).strip() for x in _as_list(value)]
    raw = [x for x in raw if x]
    deduped = _dedupe_keep_order(raw)
    bad = [x for x in deduped if not LINK_TOKEN_RE.fullmatch(x)]
    ok = [x for x in deduped if x not in bad]
    return ok, bad


def normalize_kind(value: Any) -> str:
    k = _safe_str(value).strip().lower()
    return k


def normalize_status(value: Any) -> str:
    s = _safe_str(value).strip().lower()
    return s


def _normalize_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


@dataclass(frozen=True)
class EvidenceItem:
    kind: str
    ref: str
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind, "ref": self.ref}
        if self.note:
            out["note"] = self.note
        return out


def normalize_evidence(value: Any) -> Tuple[List[EvidenceItem], List[str]]:
    errs: List[str] = []
    out: List[EvidenceItem] = []
    if value is None:
        return [], []
    items = value if isinstance(value, list) else [value]
    for i, it in enumerate(items):
        if not isinstance(it, Mapping):
            errs.append(f"evidence[{i}] must be a mapping")
            continue
        kind = _safe_str(it.get("kind")).strip()
        ref = _safe_str(it.get("ref")).strip()
        note = _safe_str(it.get("note")).strip()
        if not kind:
            errs.append(f"evidence[{i}].kind is required")
        if not ref:
            errs.append(f"evidence[{i}].ref is required")
        if kind and ref:
            out.append(EvidenceItem(kind=kind, ref=ref, note=note))
    return out, errs


def safe_asset_name(raw: str) -> str:
    """
    Convert a user-provided attachment name to a safe filename.

    - Strips directory segments.
    - Replaces whitespace with underscores.
    - Removes characters outside a conservative allowlist.
    """
    base = Path(str(raw or "").strip()).name
    base = base.replace(" ", "_")
    # Conservative: allow alnum + a few separators.
    base2 = re.sub(r"[^A-Za-z0-9._+-]", "_", base)
    base2 = re.sub(r"_+", "_", base2).strip("._")
    return base2 or "asset"


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    raw = raw.strip()
    if not raw:
        return int(default)
    try:
        return int(raw, 10)
    except Exception:
        return int(default)


def _extract_kpi_items(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = results.get("kpis")
    if not isinstance(raw, list) or not raw:
        raw = results.get("metrics")
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw[:500]:
        if isinstance(item, Mapping):
            out.append(dict(item))
    return out


def _extract_test_items(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = results.get("tests")
    if not isinstance(raw, list) or not raw:
        raw = results.get("benchmarks")
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw[:500]:
        if isinstance(item, Mapping):
            out.append(dict(item))
    return out


def is_benchmark_like_report(*, kind: str, tags: Sequence[str], results: Mapping[str, Any]) -> bool:
    k = _safe_text(kind).lower()
    if k == "benchmark":
        return True
    tag_set = {str(t).strip().lower() for t in (tags or []) if str(t).strip()}
    if any(tag in tag_set for tag in BENCHMARK_HINT_TAGS):
        return True
    if _extract_test_items(results):
        return True
    return False


def _validate_line_section(
    *,
    section: str,
    lines: Sequence[str],
    min_items: int,
    min_chars: int,
) -> List[str]:
    errs: List[str] = []
    if len(lines) < int(min_items):
        errs.append(f"{section} must contain at least {int(min_items)} items")
    short_positions = [idx + 1 for idx, value in enumerate(lines) if len(str(value).strip()) < int(min_chars)]
    if short_positions:
        errs.append(f"{section} items must be explanatory (>= {int(min_chars)} chars); short: {short_positions[:8]}")
    return errs


def validate_report_payload_quality(report: Mapping[str, Any]) -> List[str]:
    """
    Validate that a report payload is detailed enough for durable operator reports.

    This gate intentionally rejects minimalist reports. It is used by create/lint/finalize.
    """
    errs: List[str] = []
    min_chars = int(REPORT_QUALITY_POLICY["min_explanatory_chars"])

    summary = normalize_lines(report.get("summary"))
    errs.extend(
        _validate_line_section(
            section="summary",
            lines=summary,
            min_items=int(REPORT_QUALITY_POLICY["min_summary_items"]),
            min_chars=min_chars,
        )
    )

    insights = normalize_lines(report.get("insights"))
    errs.extend(
        _validate_line_section(
            section="insights",
            lines=insights,
            min_items=int(REPORT_QUALITY_POLICY["min_insight_items"]),
            min_chars=min_chars,
        )
    )

    limitations = normalize_lines(report.get("limitations"))
    errs.extend(
        _validate_line_section(
            section="limitations",
            lines=limitations,
            min_items=int(REPORT_QUALITY_POLICY["min_limitations_items"]),
            min_chars=min_chars,
        )
    )

    risks = normalize_lines(report.get("risks"))
    errs.extend(
        _validate_line_section(
            section="risks",
            lines=risks,
            min_items=int(REPORT_QUALITY_POLICY["min_risks_items"]),
            min_chars=min_chars,
        )
    )

    next_actions = normalize_lines(report.get("next_actions"))
    errs.extend(
        _validate_line_section(
            section="next_actions",
            lines=next_actions,
            min_items=int(REPORT_QUALITY_POLICY["min_next_actions_items"]),
            min_chars=min_chars,
        )
    )

    repro = normalize_lines(report.get("repro"))
    if len(repro) < int(REPORT_QUALITY_POLICY["min_repro_steps"]):
        errs.append(f"repro must contain at least {int(REPORT_QUALITY_POLICY['min_repro_steps'])} command")

    evidence_items, evidence_errs = normalize_evidence(report.get("evidence"))
    errs.extend(evidence_errs)
    if len(evidence_items) < int(REPORT_QUALITY_POLICY["min_evidence_items"]):
        errs.append(f"evidence must contain at least {int(REPORT_QUALITY_POLICY['min_evidence_items'])} items")
    for idx, item in enumerate(evidence_items):
        if len(item.note.strip()) < min_chars:
            errs.append(f"evidence[{idx}].note must explain relevance (>= {min_chars} chars)")

    context = _normalize_mapping(report.get("context"))
    if not context:
        errs.append("context mapping is required")
    else:
        for key in REPORT_QUALITY_POLICY["context_required_keys"]:
            value = _safe_text(context.get(key))
            if len(value) < min_chars:
                errs.append(f"context.{key} is required and must be >= {min_chars} chars")

    results = _normalize_mapping(report.get("results"))
    if not results:
        errs.append("results mapping is required")
        return _dedupe_keep_order(errs)

    narrative = _safe_text(results.get("narrative") or results.get("explanation") or results.get("summary"))
    if len(narrative) < min_chars:
        errs.append(f"results.narrative (or results.explanation) must be >= {min_chars} chars")

    kind = _safe_text(report.get("kind")).lower()
    tags = normalize_tags(report.get("tags"))
    benchmark_like = is_benchmark_like_report(kind=kind, tags=tags, results=results)

    kpi_items = _extract_kpi_items(results)
    min_kpis = int(REPORT_QUALITY_POLICY["benchmark_min_kpis"] if benchmark_like else REPORT_QUALITY_POLICY["min_kpis_generic"])
    if len(kpi_items) < min_kpis:
        errs.append(f"results.kpis (or results.metrics) must contain at least {min_kpis} item(s)")
    for idx, item in enumerate(kpi_items):
        name = _safe_text(item.get("name"))
        value = item.get("value")
        explanation = _safe_text(item.get("explanation") or item.get("notes") or item.get("interpretation"))
        if not name:
            errs.append(f"results.kpis[{idx}].name is required")
        if value is None or _safe_text(value) == "":
            errs.append(f"results.kpis[{idx}].value is required")
        if len(explanation) < min_chars:
            errs.append(f"results.kpis[{idx}].explanation (or notes/interpretation) must be >= {min_chars} chars")
        if benchmark_like:
            if not _safe_text(item.get("baseline")):
                errs.append(f"results.kpis[{idx}].baseline is required for benchmark/test reports")
            if not _safe_text(item.get("delta")):
                errs.append(f"results.kpis[{idx}].delta is required for benchmark/test reports")

    if benchmark_like:
        tests = _extract_test_items(results)
        min_tests = int(REPORT_QUALITY_POLICY["benchmark_min_tests"])
        if len(tests) < min_tests:
            errs.append(f"results.tests (or results.benchmarks) must contain at least {min_tests} item(s) for benchmark/test reports")
        for idx, test in enumerate(tests):
            name = _safe_text(test.get("name"))
            dataset = _safe_text(test.get("dataset_or_suite") or test.get("dataset"))
            status = _safe_text(test.get("status") or test.get("pass_fail"))
            kpi_refs = [str(x).strip() for x in _as_list(test.get("kpi_refs")) if str(x).strip()]
            if not name:
                errs.append(f"results.tests[{idx}].name is required")
            if not dataset:
                errs.append(f"results.tests[{idx}].dataset_or_suite (or dataset) is required")
            if not status:
                errs.append(f"results.tests[{idx}].status (or pass_fail) is required")
            if not kpi_refs:
                errs.append(f"results.tests[{idx}].kpi_refs must be a non-empty list")
            sample_size = test.get("sample_size_n")
            if sample_size is None or _safe_text(sample_size) == "":
                errs.append(f"results.tests[{idx}].sample_size_n is required")

    return _dedupe_keep_order(errs)


def format_report_markdown(doc: Mapping[str, Any]) -> str:
    """
    Deterministic Markdown renderer for a report doc.

    The intent is "human-first but machine-derived" so that an agent can generate
    a complete report without hand-writing Markdown, while still allowing an
    operator to overwrite the body via --body-md-*.
    """
    report = doc.get("report") if isinstance(doc, Mapping) else None
    if not isinstance(report, Mapping):
        return "# Report\n"

    title = _safe_str(report.get("title")).strip() or "Report"
    rid = _safe_str(report.get("report_id")).strip()
    kind = _safe_str(report.get("kind")).strip()
    status = _safe_str(report.get("status")).strip()
    created_at = _safe_str(report.get("created_at_utc")).strip()
    created_by = _safe_str(report.get("created_by")).strip()

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    meta_parts = [
        f"- **Report ID:** `{rid}`" if rid else "",
        f"- **Kind:** `{kind}`" if kind else "",
        f"- **Status:** `{status}`" if status else "",
        f"- **Created:** `{created_at}`" if created_at else "",
        f"- **Author:** `{created_by}`" if created_by else "",
    ]
    lines.extend([p for p in meta_parts if p])

    tags = report.get("tags")
    if isinstance(tags, list) and [str(x).strip() for x in tags if str(x).strip()]:
        tag_txt = ", ".join(str(x).strip() for x in tags if str(x).strip())
        lines.append(f"- **Tags:** {tag_txt}")
    lines.append("")

    summary = report.get("summary")
    summary_lines = normalize_lines(summary)
    if summary_lines:
        lines.append("## Summary")
        lines.append("")
        for s in summary_lines:
            lines.append(f"- {s}")
        lines.append("")

    insights = report.get("insights")
    insights_lines = normalize_lines(insights)
    if insights_lines:
        lines.append("## Insights")
        lines.append("")
        for s in insights_lines:
            lines.append(f"- {s}")
        lines.append("")

    context = _normalize_mapping(report.get("context"))
    if context:
        lines.append("## Context")
        lines.append("")
        preferred = [
            ("objective", "Objective"),
            ("scope", "Scope"),
            ("methodology", "Methodology"),
            ("data_window", "Data Window"),
        ]
        emitted: set[str] = set()
        for key, label in preferred:
            value = _safe_text(context.get(key))
            if value:
                lines.append(f"- **{label}:** {value}")
                emitted.add(key)
        for key in sorted(context.keys()):
            if key in emitted:
                continue
            value = _safe_text(context.get(key))
            if value:
                lines.append(f"- **{key}:** {value}")
        lines.append("")

    results = _normalize_mapping(report.get("results"))
    if results:
        lines.append("## Results")
        lines.append("")
        narrative = _safe_text(results.get("narrative") or results.get("explanation") or results.get("summary"))
        if narrative:
            lines.append(narrative)
            lines.append("")

        kpis = _extract_kpi_items(results)
        if kpis:
            lines.append("### KPIs")
            lines.append("")
            for m in kpis[:200]:
                name = _safe_text(m.get("name"))
                value = _safe_text(m.get("value"))
                unit = _safe_text(m.get("unit"))
                split = _safe_text(m.get("split"))
                baseline = _safe_text(m.get("baseline"))
                delta = _safe_text(m.get("delta"))
                explanation = _safe_text(m.get("explanation") or m.get("notes") or m.get("interpretation"))
                if not name:
                    continue
                parts = [f"**{name}**"]
                if split:
                    parts.append(f"({split})")
                if value:
                    parts.append(f"= `{value}{(' ' + unit) if unit else ''}`".strip())
                if baseline:
                    parts.append(f"(baseline `{baseline}`)")
                if delta:
                    parts.append(f"(Δ `{delta}`)")
                lines.append(f"- {' '.join(parts)}")
                if explanation:
                    lines.append(f"  - explanation: {explanation}")
            lines.append("")

        tests = _extract_test_items(results)
        if tests:
            lines.append("### Benchmarks & Tests")
            lines.append("")
            for test in tests[:200]:
                name = _safe_text(test.get("name")) or "test"
                dataset = _safe_text(test.get("dataset_or_suite") or test.get("dataset"))
                status = _safe_text(test.get("status") or test.get("pass_fail"))
                sample_size = _safe_text(test.get("sample_size_n"))
                kpi_refs = [str(x).strip() for x in _as_list(test.get("kpi_refs")) if str(x).strip()]
                lines.append(f"- **{name}**")
                if dataset:
                    lines.append(f"  - dataset/suite: {dataset}")
                if status:
                    lines.append(f"  - status: {status}")
                if sample_size:
                    lines.append(f"  - sample_size_n: {sample_size}")
                if kpi_refs:
                    lines.append(f"  - kpi_refs: {', '.join(kpi_refs)}")
                note = _safe_text(test.get("notes") or test.get("explanation"))
                if note:
                    lines.append(f"  - explanation: {note}")
            lines.append("")

    evidence = report.get("evidence")
    if isinstance(evidence, list) and evidence:
        lines.append("## Evidence")
        lines.append("")
        for e in evidence[:200]:
            if not isinstance(e, Mapping):
                continue
            kind_e = _safe_str(e.get("kind")).strip()
            ref_e = _safe_str(e.get("ref")).strip()
            note_e = _safe_str(e.get("note")).strip()
            if not (kind_e and ref_e):
                continue
            lines.append(f"- **{kind_e}:** `{ref_e}`")
            if note_e:
                lines.append(f"  - {note_e}")
        lines.append("")

    repro = report.get("repro")
    if isinstance(repro, list) and [str(x).strip() for x in repro if str(x).strip()]:
        lines.append("## Repro")
        lines.append("")
        for cmd in [str(x).strip() for x in repro if str(x).strip()][:80]:
            lines.append(f"- `{cmd}`")
        lines.append("")

    limitations = normalize_lines(report.get("limitations"))
    if limitations:
        lines.append("## Limitations")
        lines.append("")
        for s in limitations:
            lines.append(f"- {s}")
        lines.append("")

    risks = normalize_lines(report.get("risks"))
    if risks:
        lines.append("## Risks")
        lines.append("")
        for s in risks:
            lines.append(f"- {s}")
        lines.append("")

    next_actions = normalize_lines(report.get("next_actions"))
    if next_actions:
        lines.append("## Next Actions")
        lines.append("")
        for s in next_actions:
            lines.append(f"- {s}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
