"""
Deterministic hint-envelope builder for agentic CLIs.

This module converts legacy `next_steps` entries into a normalized, machine-
readable hint envelope:

    {
      "hints_version": "1.0",
      "hints": [...],
      "suppressed_hints": [...],
      "next_tools": [...],
      "hint_stats": {...},
      "hint_engine": {...}
    }

Design goals:
- Deterministic output ordering and IDs
- Guardrail-oriented hint structure (what/why/how + safety)
- Best-effort conflict suppression to avoid contradictory next steps
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = REPO_ROOT / "agentic_tools" / "hints" / "policies.yaml"
HINTS_VERSION = "1.0"


def _s(value: Any) -> str:
    return str(value or "").strip()


def _strings(values: Any) -> List[str]:
    if isinstance(values, list):
        out: List[str] = []
        for value in values:
            text = _s(value)
            if text:
                out.append(text)
        return out
    if isinstance(values, tuple):
        return _strings(list(values))
    if isinstance(values, str):
        text = _s(values)
        return [text] if text else []
    return []


def _first_command_tool(commands: Sequence[str]) -> Optional[str]:
    for command in commands:
        text = _s(command)
        if not text or text.startswith("#"):
            continue
        token = text.split()[0].strip()
        if token:
            return token
    return None


def _tokenize(text: str) -> str:
    return " ".join(_s(text).lower().split())


def _humanize_command(command: str) -> str:
    """
    Convert a CLI command into a conservative, agent-facing hint text.

    This must stay deterministic and avoid "clever" interpretations. When we
    don't recognize a (tool, subcommand) pair, we fall back to a minimal
    restatement.
    """
    cmd = _s(command)
    if not cmd:
        return "Run the suggested command."
    parts = cmd.split()
    tool = parts[0] if parts else ""
    subcmd = ""
    if len(parts) >= 2 and not parts[1].startswith("-"):
        subcmd = parts[1]

    # A small allowlist of conservative descriptions for common workflows.
    desc_map: Dict[Tuple[str, str], str] = {
        ("findings", "overview"): "Review Findings overview (counts, recents, triage).",
        ("findings", "list"): "List Findings (filter by status/severity) to pick next work.",
        ("findings", "address"): "Mark a Finding as addressed (work item created).",
        ("findings", "update"): "Update a Finding (status/links/fields) after progress.",
        ("findings", "explain"): "Explain a Finding into what/why/how and actionable next steps.",
        ("findings", "get"): "Open a Finding document (links, anchors, next steps).",
        ("tasks", "overview"): "Review task board overview (WIP, ready, blocked).",
        ("tasks", "list"): "List tasks (filter by column/status) to select work.",
        ("tasks", "plan"): "Show per-agent execution plan (next tasks).",
        ("tasks", "read"): "Open a task by id.",
        ("tasks", "explain"): "Explain a task (what/why/how + linked context).",
        ("tasks", "add"): "Create a new task (link it to scripts/findings).",
        ("tasks", "update"): "Update an existing task (fields/links/status).",
        ("tasks", "log"): "Append a structured note/log to a task (progress/evidence).",
        ("fixreport", "overview"): "Review FixReport corpus (recurring bugs + mitigations).",
        ("fixreport", "search"): "Search FixReports for similar failure modes/root causes.",
        ("fixreport", "explain"): "Explain a FixReport into what/why/how + prevention.",
        ("fixreport", "create"): "Create a FixReport (root cause + verification + links).",
        ("rules", "overview"): "Review Rules overview (governance inventory).",
        ("rules", "search"): "Search Rules for constraints/guardrails relevant to your change.",
        ("rules", "explain"): "Explain a Rule (what/why/how + guardrails/evidence).",
        ("ssot", "list"): "List SSOT entries (canonical paths/subsystems).",
        ("ssot", "get"): "Open an SSOT entry (authoritative references).",
        ("scripthistory", "explain"): "Explain a script's evolution (what changed/why/risks).",
        ("scripthistory", "get"): "Open script history events for one path.",
        ("scriptinfo", ""): "Inspect a script's context (tasks/findings/reviews/timeline).",
        ("hypothesis", "overview"): "Review hypotheses overview (status counts, top items).",
        ("hypothesis", "list"): "List hypotheses (filter by status/scope) to pick one.",
        ("hypothesis", "get"): "Open a hypothesis (optionally with evidence).",
        ("hypothesis", "explain"): "Explain a hypothesis into what/why/how (with evidence).",
        ("checkreview", "check"): "Acknowledge the CCR review (mark checked).",
        ("tools", "list"): "List available agentic tools (discovery).",
        ("tools", "help"): "Show a tool's usage and guidance.",
    }
    if tool and (tool, subcmd) in desc_map:
        return desc_map[(tool, subcmd)]
    if tool and subcmd:
        return f"Run: {tool} {subcmd}"
    return f"Run: {cmd}"


def _hash_id(*parts: str, size: int = 10) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:size]


def _default_policy() -> Dict[str, Any]:
    return {
        "hints_version": HINTS_VERSION,
        "policy_revision": "2026-02-06",
        "limits": {"max_hints": 6, "max_commands_per_hint": 3},
        "category_rules": [
            {"category": "governance", "keywords": ["finding", "fixreport", "resolved", "links", "link token", "status"]},
            {"category": "safety", "keywords": ["checkreview", "rollback", "verify", "lint", "pytest", "guardrail"]},
            {"category": "workflow", "keywords": ["next", "workflow", "overview", "explain", "task"]},
        ],
        "priority_rules": [
            {"priority": 95, "keywords": ["status\":\"resolved", "status=resolved", "close", "closure"]},
            {"priority": 90, "keywords": ["fixreport", "finding"]},
            {"priority": 85, "keywords": ["checkreview", "verify", "rollback", "pytest", "lint"]},
            {"priority": 70, "keywords": ["overview", "explain", "search"]},
            {"priority": 55, "keywords": ["log", "note"]},
        ],
        "conflict_rules": [
            {
                "id": "finding_status_resolved_vs_open",
                "left_keywords": ["status\":\"resolved", "status=resolved"],
                "right_keywords": ["status\":\"open", "status=open"],
            },
            {
                "id": "hypothesis_supported_vs_refuted",
                "left_keywords": ["--status supported", "status=supported"],
                "right_keywords": ["--status refuted", "status=refuted"],
            },
            {
                "id": "fixreport_create_vs_missing_links",
                "left_keywords": ["fixreport create", "--link finding:"],
                "right_keywords": ["missing link", "no fixreport link"],
            },
        ],
        "global_guardrails": [
            "Verify linked IDs/paths exist before mutating stores.",
            "Use smallest safe scope; avoid broad edits without evidence.",
            "After risky updates, verify outcome and note rollback path.",
        ],
        "link_kind_tool_map": {
            "task": "tasks",
            "finding": "findings",
            "fixreport": "fixreport",
            "hypothesis": "hypothesis",
            "script": "scripthistory",
            "code_review": "checkreview",
            "rule": "rules",
            "ssot": "ssot",
        },
    }


def _merge_policy(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**dict(merged[key]), **value}
            continue
        merged[key] = value
    return merged


@lru_cache(maxsize=1)
def load_policy() -> Dict[str, Any]:
    base = _default_policy()
    if not POLICY_PATH.exists():
        return base
    try:
        raw = POLICY_PATH.read_text(encoding="utf-8")
        parsed = yaml.safe_load(raw) or {}
        if not isinstance(parsed, dict):
            return base
        return _merge_policy(base, parsed)
    except Exception:
        return base


def _pick_category(text_blob: str, policy: Mapping[str, Any]) -> str:
    category_rules = policy.get("category_rules")
    if isinstance(category_rules, list):
        for rule in category_rules:
            if not isinstance(rule, Mapping):
                continue
            category = _s(rule.get("category")) or "workflow"
            keywords = _strings(rule.get("keywords"))
            if any(keyword in text_blob for keyword in keywords):
                return category
    return "workflow"


def _pick_priority(text_blob: str, policy: Mapping[str, Any]) -> int:
    best = 50
    priority_rules = policy.get("priority_rules")
    if not isinstance(priority_rules, list):
        return best
    for rule in priority_rules:
        if not isinstance(rule, Mapping):
            continue
        keywords = _strings(rule.get("keywords"))
        try:
            priority = int(rule.get("priority"))
        except Exception:
            continue
        if any(keyword in text_blob for keyword in keywords):
            best = max(best, priority)
    return max(1, min(best, 100))


def _to_hint_item(
    *,
    tool: str,
    action: str,
    index: int,
    step: Mapping[str, Any],
    policy: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    text = _s(step.get("text"))
    commands = _strings(step.get("commands"))
    links = _strings(step.get("links"))
    if not text and not commands:
        return None
    if not text and commands:
        text = f"Run: {commands[0]}"

    max_commands = int(((policy.get("limits") or {}).get("max_commands_per_hint")) or 3)
    if max_commands > 0:
        commands = commands[:max_commands]

    # If a step uses a generic placeholder text, replace it with a conservative
    # description derived from the first command. This improves agent UX while
    # staying deterministic and non-speculative.
    if commands:
        text_norm = _tokenize(text)
        if text_norm in {
            "follow the recommended workflow step.",
            "run recommended follow up command.",
            "run recommended follow-up command.",
            "suggested action.",
            "suggested action",
        }:
            text = _humanize_command(commands[0])

    text_blob = _tokenize(" ".join([text] + commands + links))

    category = _pick_category(text_blob, policy)
    priority = _pick_priority(text_blob, policy)
    guardrails = _strings(policy.get("global_guardrails"))
    hint_id = f"hint_{_hash_id(tool, action, str(index), text_blob)}"

    confidence = 0.65
    if commands:
        confidence += 0.15
    if links:
        confidence += 0.1
    if context.get("strict_mode"):
        confidence += 0.05
    confidence = max(0.0, min(confidence, 0.99))

    return {
        "id": hint_id,
        "priority": int(priority),
        "category": category,
        "when": "if preconditions are met",
        "why": text,
        "what": text,
        "how": commands,
        "guardrails": guardrails,
        "confidence": round(confidence, 2),
        "blocking": False,
        "conflicts_with": [],
        "links": links,
    }


def _dedupe_hints(hints: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for hint in hints:
        text = _s(hint.get("what"))
        how = "|".join(_strings(hint.get("how")))
        key = _tokenize(f"{text}|{how}")
        if not key:
            continue
        if key in seen_keys:
            suppressed.append({"id": _s(hint.get("id")), "reason": "duplicate"})
            continue
        seen_keys.add(key)
        kept.append(dict(hint))
    return kept, suppressed


def _contains_any(blob: str, keywords: Sequence[str]) -> bool:
    if not blob:
        return False
    return any(_s(keyword).lower() in blob for keyword in keywords if _s(keyword))


def _conflict_suppress(
    hints: List[Dict[str, Any]], policy: Mapping[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    rules = policy.get("conflict_rules")
    if not isinstance(rules, list) or len(hints) <= 1:
        return hints, [], 0

    keep = [True] * len(hints)
    suppressed: List[Dict[str, Any]] = []
    conflicts_detected = 0

    blobs = [
        _tokenize(" ".join([_s(h.get("what")), *(_strings(h.get("how")))]))
        for h in hints
    ]

    for rule in rules:
        if not isinstance(rule, Mapping):
            continue
        rule_id = _s(rule.get("id")) or "conflict"
        left = _strings(rule.get("left_keywords"))
        right = _strings(rule.get("right_keywords"))
        if not left or not right:
            continue
        for i in range(len(hints)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(hints)):
                if not keep[j]:
                    continue
                bi = blobs[i]
                bj = blobs[j]
                match = (_contains_any(bi, left) and _contains_any(bj, right)) or (
                    _contains_any(bi, right) and _contains_any(bj, left)
                )
                if not match:
                    continue
                conflicts_detected += 1
                pi = int(hints[i].get("priority") or 0)
                pj = int(hints[j].get("priority") or 0)
                drop_i = pi < pj
                if pi == pj:
                    drop_i = _s(hints[i].get("id")) > _s(hints[j].get("id"))
                loser = i if drop_i else j
                winner = j if drop_i else i
                keep[loser] = False
                hints[winner]["conflicts_with"] = sorted(
                    set(_strings(hints[winner].get("conflicts_with")) + [_s(hints[loser].get("id"))])
                )
                suppressed.append({"id": _s(hints[loser].get("id")), "reason": f"conflict:{rule_id}"})

    out = [hint for idx, hint in enumerate(hints) if keep[idx]]
    return out, suppressed, conflicts_detected


def _derive_next_tools(
    *, hints: Sequence[Mapping[str, Any]], policy: Mapping[str, Any], related_tools: Sequence[str]
) -> List[str]:
    out: List[str] = []
    link_map = policy.get("link_kind_tool_map")
    link_kind_tool_map = dict(link_map) if isinstance(link_map, Mapping) else {}

    for hint in hints:
        commands = _strings(hint.get("how"))
        tool_name = _first_command_tool(commands)
        if tool_name:
            out.append(tool_name)
        for link in _strings(hint.get("links")):
            if ":" not in link:
                continue
            kind = link.split(":", 1)[0].strip().lower()
            mapped = _s(link_kind_tool_map.get(kind))
            if mapped:
                out.append(mapped)

    out.extend([_s(x) for x in related_tools if _s(x)])
    uniq: List[str] = []
    seen: set[str] = set()
    for tool_name in out:
        if not tool_name or tool_name in seen:
            continue
        seen.add(tool_name)
        uniq.append(tool_name)
    return sorted(uniq)


def build_hint_envelope(
    *,
    tool: str,
    action: str,
    next_steps: Sequence[Mapping[str, Any]],
    context: Optional[Mapping[str, Any]] = None,
    related_tools: Sequence[str] = (),
    max_hints: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convert legacy `next_steps` entries into a normalized deterministic envelope.
    """
    policy = load_policy()
    ctx: Dict[str, Any] = dict(context or {})
    raw_hints: List[Dict[str, Any]] = []
    for idx, step in enumerate(next_steps or ()):
        if not isinstance(step, Mapping):
            continue
        item = _to_hint_item(
            tool=_s(tool),
            action=_s(action),
            index=idx,
            step=step,
            policy=policy,
            context=ctx,
        )
        if item is not None:
            raw_hints.append(item)

    deduped, dedupe_suppressed = _dedupe_hints(raw_hints)
    no_conflict, conflict_suppressed, conflicts_detected = _conflict_suppress(deduped, policy)

    no_conflict.sort(key=lambda item: (-int(item.get("priority") or 0), _s(item.get("id"))))
    cap = max_hints
    if cap is None:
        try:
            cap = int(((policy.get("limits") or {}).get("max_hints")) or 6)
        except Exception:
            cap = 6
    cap = max(1, int(cap))

    emitted = no_conflict[:cap]
    overflow = no_conflict[cap:]
    suppressed_hints = dedupe_suppressed + conflict_suppressed + [
        {"id": _s(item.get("id")), "reason": "max_hints_cap"} for item in overflow
    ]

    next_tools = _derive_next_tools(hints=emitted, policy=policy, related_tools=related_tools)
    return {
        "hints_version": _s(policy.get("hints_version")) or HINTS_VERSION,
        "hints": emitted,
        "suppressed_hints": suppressed_hints,
        "next_tools": next_tools,
        "hint_stats": {
            "input_count": len(raw_hints),
            "emitted_count": len(emitted),
            "suppressed_count": len(suppressed_hints),
            "conflicts_detected": int(conflicts_detected),
        },
        "hint_engine": {
            "policy_revision": _s(policy.get("policy_revision")) or "unknown",
            "policy_source": str(POLICY_PATH),
            "deterministic": True,
        },
    }


def hint_envelope_schema() -> Dict[str, Any]:
    """Machine-readable schema snippet for CLI `schema` payloads."""
    return {
        "hints_version": "string",
        "hints": [
            {
                "id": "string",
                "priority": "int(1..100)",
                "category": "string",
                "when": "string",
                "why": "string",
                "what": "string",
                "how": ["string"],
                "guardrails": ["string"],
                "confidence": "float(0..1)",
                "blocking": "bool",
                "conflicts_with": ["string"],
                "links": ["kind:id"],
            }
        ],
        "suppressed_hints": [{"id": "string", "reason": "string"}],
        "next_tools": ["tool_name"],
        "hint_stats": {
            "input_count": "int",
            "emitted_count": "int",
            "suppressed_count": "int",
            "conflicts_detected": "int",
        },
        "hint_engine": {
            "policy_revision": "string",
            "policy_source": "path",
            "deterministic": "bool",
        },
    }
