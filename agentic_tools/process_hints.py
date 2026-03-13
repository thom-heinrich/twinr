"""
Process guidance ("next steps") for agentic tools.

Goal
----
Provide lightweight, machine-readable suggestions that help agents/humans
follow the intended repo workflow (Findings -> Tasks -> FixReport -> Closure)
without baking this logic into the LLM prompt only.

Contract
--------
- Returned values are JSON-serializable.
- Suggestions are best-effort and MUST NOT be treated as required semantics.
- Keep outputs short and deterministic (avoid timestamps, randomness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from agentic_tools.hint_engine import build_hint_envelope
from agentic_tools.hint_engine import hint_envelope_schema


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _lower_set(items: Iterable[str]) -> set[str]:
    return {str(x or "").strip().lower() for x in items if str(x or "").strip()}


def _parse_links(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x or "").strip()]
    return []


def _split_token(tok: str) -> Tuple[str, str]:
    s = str(tok or "").strip()
    if ":" not in s:
        return ("", "")
    k, r = s.split(":", 1)
    return (k.strip().lower(), r.strip())


def _extract_script_paths(links: Sequence[str]) -> List[str]:
    out: List[str] = []
    for tok in links or ():
        k, r = _split_token(tok)
        if k != "script":
            continue
        if not r:
            continue
        out.append(r)
    # De-dupe while preserving order.
    seen: set[str] = set()
    deduped: List[str] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)
    return deduped


@dataclass(frozen=True)
class NextStep:
    """
    Machine-readable guidance entry.

    Keep `text` short; `commands` are suggestions, not required.
    """

    text: str
    commands: Tuple[str, ...] = ()
    links: Tuple[str, ...] = ()

    def as_json(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "commands": list(self.commands),
            "links": list(self.links),
        }


def _related_tool_names(value: Any) -> List[str]:
    if isinstance(value, Mapping):
        out: List[str] = []
        for key in value.keys():
            key_s = str(key or "").strip()
            if key_s:
                out.append(key_s)
        return out
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x or "").strip()]
    return []


def _derive_recommendation_steps(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Build conservative fallback steps from recommendation fields when explicit
    `next_steps` are missing.
    """
    steps: List[Dict[str, Any]] = []
    seen: set[str] = set()

    recommended_calls = payload.get("recommended_next_calls")
    for command in recommended_calls if isinstance(recommended_calls, list) else []:
        cmd = str(command or "").strip()
        if not cmd or cmd in seen:
            continue
        seen.add(cmd)
        steps.append(
            {
                "text": "Run recommended follow-up command.",
                "commands": [cmd],
                "links": [],
            }
        )

    recommended_workflow = payload.get("recommended_workflow")
    for command in recommended_workflow if isinstance(recommended_workflow, list) else []:
        cmd = str(command or "").strip()
        if not cmd or cmd in seen:
            continue
        seen.add(cmd)
        steps.append(
            {
                "text": "Follow the recommended workflow step.",
                "commands": [cmd],
                "links": [],
            }
        )

    overview = payload.get("overview")
    actions = overview.get("actions") if isinstance(overview, Mapping) else None
    for action in actions if isinstance(actions, list) else []:
        if not isinstance(action, Mapping):
            continue
        cmd = str(action.get("command") or "").strip()
        if not cmd or cmd in seen:
            continue
        seen.add(cmd)
        steps.append(
            {
                "text": str(action.get("title") or "Suggested action."),
                "commands": [cmd],
                "links": [],
            }
        )

    return steps[:8]


def enrich_payload_with_hints(
    payload: Mapping[str, Any],
    *,
    tool: str,
    default_action: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach normalized hint-envelope fields to a CLI payload.

    Behavior:
    - For error payloads (`ok=false`), return payload unchanged.
    - Prefer explicit `next_steps`; fallback to recommendation fields.
    - Keep legacy `next_steps` for backwards compatibility.
    """
    out: Dict[str, Any] = dict(payload or {})
    if out.get("ok") is False:
        return out
    if "hints_version" in out and "hints" in out:
        return out

    action = str(out.get("action") or default_action or "").strip()
    next_steps = out.get("next_steps")
    if not isinstance(next_steps, list):
        next_steps = _derive_recommendation_steps(out)
        if next_steps:
            out["next_steps"] = next_steps
    related_tools = _related_tool_names(out.get("related_tools"))

    envelope = build_hint_envelope(
        tool=str(tool or "").strip(),
        action=action,
        next_steps=next_steps if isinstance(next_steps, list) else [],
        context=dict(context or {}),
        related_tools=related_tools,
    )
    out.update(envelope)
    return out


def hints_schema_fragment() -> Dict[str, Any]:
    """Schema fragment for normalized hint-envelope fields."""
    return hint_envelope_schema()


def next_steps_for_checkreview(
    *,
    script: str,
    found: bool,
    no_mark: bool,
    mark_checked: bool,
    latest_checked: bool,
    related_open_findings: Optional[int],
) -> List[Dict[str, Any]]:
    steps: List[NextStep] = []
    if not found:
        steps.append(
            NextStep(
                text="No CCR review found for this script yet; wait for the code-change review agent or open /reviews in the Portal.",
                commands=(),
                links=(f"script:{script}",),
            )
        )
        return [s.as_json() for s in steps]

    if no_mark:
        steps.append(
            NextStep(
                text="You inspected the review read-only. If you have read it, acknowledge it so it shows as checked in /reviews.",
                commands=(f"checkreview check {script}",),
                links=(f"script:{script}",),
            )
        )
    elif mark_checked:
        steps.append(
            NextStep(
                text="Review acknowledged (checked=true). Closure in /reviews is driven by checked + no open related Findings.",
                commands=(),
                links=(f"script:{script}",),
            )
        )
    elif latest_checked:
        steps.append(
            NextStep(
                text="Review already acknowledged earlier (checked=true).",
                commands=(),
                links=(f"script:{script}",),
            )
        )

    if related_open_findings is not None and int(related_open_findings) > 0:
        steps.append(
            NextStep(
                text="There are open related Findings for this script. Close them (or mark wont_fix/duplicate) to get the /reviews 'done' badge.",
                commands=(
                    f"findings list --q {script}",
                    "# then: create/advance task, fix, fixreport create, findings update status=resolved",
                ),
                links=(f"script:{script}",),
            )
        )
    return [s.as_json() for s in steps]


def next_steps_for_finding_doc(*, action: str, finding_id: str, doc: Mapping[str, Any]) -> List[Dict[str, Any]]:
    steps: List[NextStep] = []
    f = doc.get("finding") if isinstance(doc.get("finding"), Mapping) else {}
    links = _parse_links(doc.get("links"))
    links_l = _lower_set(links)

    status = _safe_str(f.get("status")).strip().lower()
    addressed = bool(f.get("addressed") is True)

    target = f.get("target") if isinstance(f.get("target"), Mapping) else {}
    target_path = _safe_str(target.get("path")).strip()

    script_paths = _extract_script_paths(links)
    script_path = script_paths[0] if script_paths else (target_path or "")

    has_task_link = any(tok.startswith("task:") for tok in links_l)
    has_fixreport_link = any(tok.startswith("fixreport:") for tok in links_l) or bool(
        isinstance(f.get("related"), Mapping) and list((f.get("related") or {}).get("fixreport_ids") or [])
    )
    has_code_review_link = any(tok.startswith("code_review:") for tok in links_l)

    # Addressed vs. fixed semantics are easy to confuse: guide aggressively.
    if status in {"open", "triage", ""}:
        if not addressed:
            steps.append(
                NextStep(
                    text="This Finding is open and not addressed. Turn it into work: create/attach a task, then mark it addressed.",
                    commands=(
                        f"tasks add todo --title \"{finding_id}: <short title>\" --script {script_path or '<path>'} --link finding:{finding_id}",
                        f"findings address --id {finding_id} --actor <you>",
                    ),
                    links=(f"finding:{finding_id}",) + ((f"script:{script_path}",) if script_path else ()),
                )
            )
        elif not has_task_link:
            steps.append(
                NextStep(
                    text="This Finding is addressed but has no task: link the work item so Meta/backlinks stay truthful.",
                    commands=(
                        f"tasks add todo --title \"{finding_id}: <short title>\" --script {script_path or '<path>'} --link finding:{finding_id}",
                        f"findings update --id {finding_id} --add-link task:<TASK_ID> --actor <you>",
                    ),
                    links=(f"finding:{finding_id}",) + ((f"script:{script_path}",) if script_path else ()),
                )
            )

        steps.append(
            NextStep(
                text="When fixed: create a FixReport and then set finding.status=resolved (addressed != fixed).",
                commands=(
                    f"fixreport create --title \"{finding_id}: <short fix title>\" --repo-area <area> --target-kind script --target-path {script_path or '<path>'} --scope <scope> --mode <mode> --bug-type <...> --symptom <...> --root-cause <...> --failure-mode <...> --fix-type <...> --impact-area <...> --severity <p?> --verification <...> --link finding:{finding_id} --link script:{script_path or '<path>'}",
                    f"findings update --id {finding_id} --set-finding-json '{{\"status\":\"resolved\"}}' --add-link fixreport:<BF_ID> --actor <you>",
                ),
                links=(f"finding:{finding_id}",) + ((f"script:{script_path}",) if script_path else ()),
            )
        )

    if status in {"resolved", "wont_fix", "duplicate"}:
        if status == "resolved" and not has_fixreport_link:
            steps.append(
                NextStep(
                    text="Finding is resolved but no FixReport is linked. Prefer linking a FixReport for postmortem + prevention memory.",
                    commands=(
                        f"fixreport create --title \"{finding_id}: <short fix title>\" --link finding:{finding_id} --link script:{script_path or '<path>'}",
                        f"findings update --id {finding_id} --add-link fixreport:<BF_ID> --actor <you>",
                    ),
                    links=(f"finding:{finding_id}",) + ((f"script:{script_path}",) if script_path else ()),
                )
            )
        if has_code_review_link and script_path:
            steps.append(
                NextStep(
                    text="This Finding links a CCR review; acknowledge the latest CCR so /reviews can show 'done' (checked + no open findings).",
                    commands=(f"checkreview check {script_path}",),
                    links=(f"finding:{finding_id}", f"script:{script_path}"),
                )
            )

    # For "get", also nudge about "disposition" if there is a CCR link.
    if action == "get" and has_code_review_link and status not in {"resolved", "wont_fix", "duplicate"}:
        steps.append(
            NextStep(
                text="If this Finding exists only as a CCR disposition, consider closing it via Portal /reviews/<CCR>/disposition to keep the CCR workflow consistent.",
                commands=(),
                links=(f"finding:{finding_id}",) + ((f"script:{script_path}",) if script_path else ()),
            )
        )

    return [s.as_json() for s in steps]


def next_steps_for_fixreport_create(*, bf_id: str, target_path: str, links: Sequence[str]) -> List[Dict[str, Any]]:
    steps: List[NextStep] = []
    links_l = _lower_set(links)
    finding_links = [tok for tok in links if tok.strip().lower().startswith("finding:")]
    has_finding_link = bool(finding_links)
    script_path = (target_path or "").strip()

    if not has_finding_link:
        actor = "<you>"
        steps.append(
            NextStep(
                text="If this fix corresponds to a Finding, link it (finding:FND...) so Meta/backlinks can join fix -> outcome.",
                commands=(f"fixreport update --bf-id {bf_id} --link-add finding:<FND_ID> --actor {actor}",),
                links=(f"fixreport:{bf_id}",) + ((f"script:{script_path}",) if script_path else ()),
            )
        )
    else:
        finding_ids = [tok.split(":", 1)[1].strip() for tok in finding_links if ":" in tok and tok.split(":", 1)[1].strip()]
        f0 = finding_ids[0] if finding_ids else "<FND_ID>"
        steps.append(
            NextStep(
                text="Close the linked Finding(s): set finding.status=resolved and add fixreport:<BF...> link token.",
                commands=(
                    f"findings update --id {f0} --set-finding-json '{{\"status\":\"resolved\"}}' --add-link fixreport:{bf_id} --actor <you>",
                ),
                links=(f"fixreport:{bf_id}",) + tuple(str(x) for x in finding_links[:5]),
            )
        )

    if script_path.endswith(".py") and "code_review:" in " ".join(sorted(links_l)):
        steps.append(
            NextStep(
                text="If there is a CCR review for this script, acknowledge it so /reviews reflects closure.",
                commands=(f"checkreview check {script_path}",),
                links=(f"fixreport:{bf_id}", f"script:{script_path}"),
            )
        )
    return [s.as_json() for s in steps]


def next_steps_for_fixreport_doc(*, action: str, bf_id: str, doc: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Guidance for fixreport get/update outputs based on the stored report.

    Shape note: FixReport YAML stores most fields under top-level `fixreport`.
    """
    fr = doc.get("fixreport") if isinstance(doc.get("fixreport"), Mapping) else {}
    links = fr.get("links") if isinstance(fr, Mapping) else None
    links_list = _parse_links(links)
    target_path = _safe_str((fr or {}).get("target_path")).strip() if isinstance(fr, Mapping) else ""
    # Reuse the create guidance as baseline.
    out = next_steps_for_fixreport_create(bf_id=str(bf_id), target_path=target_path, links=links_list)

    act = str(action or "").strip().lower()
    if act in {"get", "update"}:
        # Encourage linking back to tasks/finding if missing.
        links_l = _lower_set(links_list)
        if not any(t.startswith("task:") for t in links_l):
            out.append(
                NextStep(
                    text="If this fix was done under a Task, link task:<id> into the FixReport links so Meta/backlinks can join the work thread.",
                    commands=(f"fixreport update --bf-id {bf_id} --link-add task:<TASK_ID> --actor <you>",),
                    links=(f"fixreport:{bf_id}",) + ((f"script:{target_path}",) if target_path else ()),
                ).as_json()
            )
    return out[:12]


def _extract_ids(links: Sequence[str], *, kind: str) -> List[str]:
    want = str(kind or "").strip().lower()
    if not want:
        return []
    out: List[str] = []
    for tok in links or ():
        k, r = _split_token(tok)
        if k != want:
            continue
        if not r:
            continue
        out.append(r)
    # De-dupe while preserving order.
    seen: set[str] = set()
    deduped: List[str] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)
    return deduped


def next_steps_for_task_action(
    *,
    action: str,
    task_id: str,
    title: str,
    column: str,
    status: str,
    links: Sequence[str],
    agent: str = "",
    moved_from: str = "",
    moved_to: str = "",
    leased: bool = False,
    deleted: bool = False,
) -> List[Dict[str, Any]]:
    """
    Provide workflow guidance for tasks tool actions.

    This intentionally does not read other stores (Findings/FixReport/...) because tasks
    is latency-sensitive and should remain dependency-light. Guidance is derived from
    task state + link tokens only.
    """
    steps: List[NextStep] = []
    act = str(action or "").strip().lower()
    tid = str(task_id or "").strip()

    links_list = [str(x).strip() for x in (links or []) if str(x or "").strip()]
    scripts = _extract_script_paths(links_list)
    findings = _extract_ids(links_list, kind="finding")
    fixreports = _extract_ids(links_list, kind="fixreport")
    code_reviews = _extract_ids(links_list, kind="code_review")
    hypotheses = _extract_ids(links_list, kind="hypothesis")

    # 0) Positioning: where in the process are we?
    stage = (column or status or "").strip().lower()
    if deleted:
        stage = "deleted"
    elif stage in {"backlog", "open"}:
        stage = "intake"
    elif stage in {"todo", "ready"}:
        stage = "ready"
    elif stage in {"doing", "in_progress"}:
        stage = "in_progress"
    elif stage in {"review"}:
        stage = "review"
    elif stage in {"done", "archived"}:
        stage = "closure"

    deps = f"deps: scripts={len(scripts)} findings={len(findings)} fixreports={len(fixreports)} reviews={len(code_reviews)} hypotheses={len(hypotheses)}"
    steps.append(
        NextStep(
            text=f"Task stage: {stage} ({deps}).",
            commands=(),
            links=(f"task:{tid}",),
        )
    )

    # 1) Keep Meta graph truthful: script anchors are the primary join key.
    if not scripts and not deleted:
        steps.append(
            NextStep(
                text="Task has no script anchor. Add at least one script:<repo_path> link so Meta/backlinks can join work across tools.",
                commands=(f"tasks update --id {tid} --script-add <repo_relative_path>",),
                links=(f"task:{tid}",),
            )
        )

    # 2) Findings <-> Tasks: once task exists, finding should be 'addressed' (checkmark).
    if findings and not deleted:
        fnd0 = findings[0]
        actor = agent or "<you>"
        steps.append(
            NextStep(
                text="This task references a Finding. Ensure the Finding is marked addressed (addressed=true means taskified, not fixed).",
                commands=(f"findings address --id {fnd0} --actor {actor}",),
                links=(f"task:{tid}", f"finding:{fnd0}"),
            )
        )

    # 3) Movement semantics: what to do when changing columns.
    if act == "move" and (moved_to or moved_from):
        mf = moved_from or ""
        mt = moved_to or ""
        steps.append(
            NextStep(
                text=f"Task moved: {mf} -> {mt}. Validate next actions match the new stage.",
                commands=(),
                links=(f"task:{tid}",),
            )
        )

    if stage == "review" and not deleted:
        cmd = "conda run -n tessairact python -m pytest -q"
        steps.append(
            NextStep(
                text="In review: run tests/lint and collect evidence; then log results onto the task before marking done.",
                commands=(cmd, f"tasks log --id {tid} --agent {agent or '<you>'} --log '{{\"kind\":\"note\",\"text\":\"tests: <result>\"}}'"),
                links=(f"task:{tid}",),
            )
        )
        if scripts:
            steps.append(
                NextStep(
                    text="If you changed code, check/ack the latest CCR review for the script(s) (Portal /reviews).",
                    commands=(f"checkreview check --no-mark {scripts[0]}", f"checkreview check {scripts[0]}"),
                    links=(f"task:{tid}", f"script:{scripts[0]}"),
                )
            )

    if stage == "closure" and not deleted:
        if findings and not fixreports:
            steps.append(
                NextStep(
                    text="Task is done and references Finding(s) but no FixReport link is present. Prefer creating a FixReport and linking it for prevention memory.",
                    commands=(
                        f"fixreport create --title \"{tid}: {title or '<short title>'}\" --link task:{tid} --link finding:{findings[0]} --link script:{scripts[0] if scripts else '<path>'}",
                        f"tasks update --id {tid} --links-add fixreport:<BF_ID>",
                    ),
                    links=(f"task:{tid}", f"finding:{findings[0]}") + ((f"script:{scripts[0]}",) if scripts else ()),
                )
            )
        if findings:
            bf_tok = fixreports[0] if fixreports else "<BF_ID>"
            steps.append(
                NextStep(
                    text="After a fix: close the Finding(s) (status=resolved or wont_fix/duplicate) and link fixreport:<BF...> so /reviews and Meta can converge.",
                    commands=(
                        f"findings update --id {findings[0]} --set-finding-json '{{\"status\":\"resolved\"}}' --add-link task:{tid} --add-link fixreport:{bf_tok} --actor {agent or '<you>'}",
                    ),
                    links=(f"task:{tid}", f"finding:{findings[0]}"),
                )
            )
        if scripts and scripts[0].endswith(".py"):
            steps.append(
                NextStep(
                    text="If this task changed code, acknowledge the latest CCR review so /reviews can reflect closure (checked + no open Findings).",
                    commands=(f"checkreview check --no-mark {scripts[0]}", f"checkreview check {scripts[0]}"),
                    links=(f"task:{tid}", f"script:{scripts[0]}"),
                )
            )

    # 4) Leases: after pull, move to doing and start logging progress.
    if leased and not deleted:
        steps.append(
            NextStep(
                text="Task is leased to you. Move it to 'doing' and start logging progress so others can see current state.",
                commands=(
                    f"tasks move --id {tid} --to-column doing --agent {agent or '<you>'}",
                    f"tasks log --id {tid} --agent {agent or '<you>'} --log '{{\"kind\":\"note\",\"text\":\"starting\"}}'",
                ),
                links=(f"task:{tid}",),
            )
        )

    # 5) Cross-edges: encourage linking hypotheses to tasks when relevant.
    if hypotheses and not deleted:
        steps.append(
            NextStep(
                text="Task links a hypothesis; make sure evidence/results are attached as hypothesis evidence entries when you progress.",
                commands=(f"hypothesis add-evidence --hypothesis-id {hypotheses[0]} --kind observation --polarity inconclusive --strength weak --summary \"linked from task:{tid}\" --sources-json '[]' --actor {agent or '<you>'}",),
                links=(f"task:{tid}", f"hypothesis:{hypotheses[0]}"),
            )
        )

    # 6) CCR review cross-edge: keep /reviews 'done' badge aligned.
    if code_reviews and scripts and not deleted:
        steps.append(
            NextStep(
                text="Task links a CCR review; acknowledge the CCR and ensure no open related Findings remain so /reviews can show done.",
                commands=(f"checkreview check {scripts[0]}",),
                links=(f"task:{tid}", f"code_review:{code_reviews[0]}", f"script:{scripts[0]}"),
            )
        )

    # Bound output.
    return [s.as_json() for s in steps[:12]]


def next_steps_for_metric_eval_action(
    *,
    action: str,
    yaml_path: str,
    audit_id: str = "",
    target: str = "",
    links: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """
    Guidance for the metric-eval tool.

    We keep this conservative: metric audits can be used for many purposes.
    Suggestions are phrased as conditionals ("if this is evidence for ...").
    """
    steps: List[NextStep] = []
    act = str(action or "").strip().lower()
    yaml_s = str(yaml_path or "").strip()
    audit_s = str(audit_id or "").strip()
    target_s = str(target or "").strip()

    links_list = [str(x).strip() for x in (links or []) if str(x or "").strip()]
    scripts = _extract_script_paths(links_list)
    tasks = _extract_ids(links_list, kind="task")
    findings = _extract_ids(links_list, kind="finding")
    hypotheses = _extract_ids(links_list, kind="hypothesis")
    script_anchor = scripts[0] if scripts else (target_s if target_s and " " not in target_s and ":" not in target_s else "<path>")

    if act == "new":
        steps.append(
            NextStep(
                text="A draft metric audit version was created. Next: run your evaluator/backtest and merge results via `metric-eval apply`.",
                commands=(
                    f"metric-eval apply --yaml {yaml_s} --audit-id {audit_s or '<audit_id>'} --data @payload.json",
                    f"metric-eval lint --yaml {yaml_s} --strict-metrics",
                ),
                links=tuple(links_list[:6]),
            )
        )
    elif act == "apply":
        steps.append(
            NextStep(
                text="Audit updated. Validate consistency with lint (and keep strict-metrics on tool-generated versions).",
                commands=(f"metric-eval lint --yaml {yaml_s} --strict-metrics",),
                links=tuple(links_list[:6]),
            )
        )
    elif act in {"build_payload", "build-payload"}:
        steps.append(
            NextStep(
                text="Payload built. Next: apply it to the intended audit YAML + run lint.",
                commands=(
                    "metric-eval apply --yaml <audit_yaml> --audit-id <audit_id> --data @payload.json",
                    "metric-eval lint --yaml <audit_yaml> --strict-metrics",
                ),
                links=tuple(links_list[:6]),
            )
        )

    if hypotheses:
        h0 = hypotheses[0]
        steps.append(
            NextStep(
                text="If this audit is evidence for the linked hypothesis: add an evidence item referencing this YAML/audit_id.",
                commands=(
                    f"hypothesis add-evidence --hypothesis-id {h0} --kind experiment --polarity supporting --strength medium --summary \"metric-eval: {audit_s or '<audit_id>'}\" --sources-json '[{{\"kind\":\"file\",\"path\":\"{yaml_s}\",\"note\":\"audit_id {audit_s or '<audit_id>'}\"}}]' --actor <you> --link script:{script_anchor} --link task:{tasks[0] if tasks else '<TASK_ID>'}",
                ),
                links=(f"hypothesis:{h0}",) + ((f"script:{scripts[0]}",) if scripts else ()),
            )
        )
    else:
        steps.append(
            NextStep(
                text="If this audit should drive decisions, link it to the relevant hypothesis/task/finding via link tokens so Meta can join evidence -> action.",
                commands=(
                    f"metric-eval apply --yaml {yaml_s} --audit-id {audit_s or '<audit_id>'} --data '{{\"links\":[\"hypothesis:<H_ID>\",\"task:<TASK_ID>\",\"finding:<FND_ID>\"]}}'",
                ),
                links=tuple(links_list[:4]),
            )
        )

    if findings and tasks:
        steps.append(
            NextStep(
                text="If the audit confirms a Finding is fixed or a regression is resolved, remember to close the Finding and link the FixReport (not just the task).",
                commands=(
                    f"findings update --id {findings[0]} --set-finding-json '{{\"status\":\"resolved\"}}' --add-link task:{tasks[0]} --add-link fixreport:<BF_ID> --actor <you>",
                ),
                links=(f"finding:{findings[0]}", f"task:{tasks[0]}"),
            )
        )

    return [s.as_json() for s in steps[:10]]


def next_steps_for_hypothesis_action(
    *,
    action: str,
    hypothesis_id: str,
    status: str,
    links: Sequence[str] = (),
    evidence_stats: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Guidance for hypothesis tool outputs.

    Keep strictly non-binding and avoid store-spanning heuristics to prevent
    false positives.
    """
    steps: List[NextStep] = []
    act = str(action or "").strip().lower()
    hid = str(hypothesis_id or "").strip()
    st = str(status or "").strip().lower()

    links_list = [str(x).strip() for x in (links or []) if str(x or "").strip()]
    scripts = _extract_script_paths(links_list)
    tasks = _extract_ids(links_list, kind="task")
    findings = _extract_ids(links_list, kind="finding")

    # Evidence stats are best-effort; only use if present and well-shaped.
    ev = dict(evidence_stats or {}) if isinstance(evidence_stats, Mapping) else {}

    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return 0

    ev_supporting = _safe_int(ev.get("supporting"))
    ev_contradicting = _safe_int(ev.get("contradicting"))
    ev_inconclusive = _safe_int(ev.get("inconclusive"))
    try:
        ev_total = int(ev.get("total") or 0)
    except Exception:
        ev_total = 0
    if ev_total <= 0:
        ev_total = max(0, ev_supporting) + max(0, ev_contradicting) + max(0, ev_inconclusive)

    if act == "create":
        steps.append(
            NextStep(
                text="Hypothesis created. Next: add at least one evidence item (even weak/inconclusive) so it can move from opinion -> tracked evidence.",
                commands=(
                    f"hypothesis add-evidence --hypothesis-id {hid} --kind observation --polarity inconclusive --strength weak --summary \"initial evidence\" --sources-json '[]' --actor <you>",
                ),
                links=(f"hypothesis:{hid}",) + ((f"script:{scripts[0]}",) if scripts else ()),
            )
        )
        if scripts:
            steps.append(
                NextStep(
                    text="If you need a quantitative audit as evidence, create one and link it to this hypothesis.",
                    commands=(
                        f"metric-eval new --target {scripts[0]} --preset llm.agent.service --link hypothesis:{hid} --link script:{scripts[0]}",
                    ),
                    links=(f"hypothesis:{hid}", f"script:{scripts[0]}"),
                )
            )

    if act == "get":
        if st in {"open", "in_review"} and ev_total == 0:
            steps.append(
                NextStep(
                    text="This hypothesis has no evidence yet. Add at least one evidence item before treating it as guidance.",
                    commands=(f"hypothesis add-evidence --hypothesis-id {hid} --kind observation --polarity inconclusive --strength weak --summary \"evidence\" --sources-json '[]' --actor <you>",),
                    links=(f"hypothesis:{hid}",),
                )
            )
        steps.append(
            NextStep(
                text="If this hypothesis should drive work, ensure it is linked to the relevant Task/Finding so plans stay consistent.",
                commands=(
                    f"hypothesis add-links --hypothesis-id {hid} --link-add task:<TASK_ID> --link-add finding:<FND_ID> --actor <you>",
                ),
                links=(f"hypothesis:{hid}",),
            )
        )

    if act == "add-evidence":
        steps.append(
            NextStep(
                text="Evidence added. If this changes your confidence, update status and record a short verdict rationale.",
                commands=(
                    f"hypothesis set-status --hypothesis-id {hid} --status in_review --confidence 0.6 --reason \"evidence added\" --actor <you>",
                ),
                links=(f"hypothesis:{hid}",),
            )
        )

    if act in {"set-status", "add-links"}:
        if tasks:
            steps.append(
                NextStep(
                    text="Hypothesis updated and linked to a task. Log the current verdict/evidence into the task so execution stays aligned.",
                    commands=(f"tasks log --id {tasks[0]} --agent <you> --log '{{\"kind\":\"note\",\"text\":\"hypothesis {hid} ({st or '<status>'})\"}}'",),
                    links=(f"hypothesis:{hid}", f"task:{tasks[0]}"),
                )
            )
        if findings and st in {"supported", "refuted"}:
            steps.append(
                NextStep(
                    text="If this hypothesis resolves (or invalidates) an actionable Finding, update the Finding status accordingly.",
                    commands=(f"findings update --id {findings[0]} --set-finding-json '{{\"status\":\"resolved\"}}' --add-link hypothesis:{hid} --actor <you>",),
                    links=(f"hypothesis:{hid}", f"finding:{findings[0]}"),
                )
            )

    return [s.as_json() for s in steps[:10]]


def next_steps_for_chat_action(
    *,
    action: str,
    message_id: str,
    channel: str,
    links: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """
    Guidance for chat tool outputs (send/get/thread/list).

    Chat is coordination; only suggest follow-ups when explicit link tokens exist
    to avoid confusing "do X" prompts on unrelated chatter.
    """
    steps: List[NextStep] = []
    act = str(action or "").strip().lower()
    mid = str(message_id or "").strip()
    chan = str(channel or "").strip()

    links_list = [str(x).strip() for x in (links or []) if str(x or "").strip()]
    scripts = _extract_script_paths(links_list)
    tasks = _extract_ids(links_list, kind="task")
    findings = _extract_ids(links_list, kind="finding")
    fixreports = _extract_ids(links_list, kind="fixreport")
    hypotheses = _extract_ids(links_list, kind="hypothesis")
    code_reviews = _extract_ids(links_list, kind="code_review")

    if act == "send":
        steps.append(
            NextStep(
                text="Message sent. If this is coordinating work, keep the truth graph updated via explicit links on the underlying artifacts (task/finding/fixreport).",
                commands=(),
                links=(f"chat:{mid}",) if mid else (),
            )
        )

    if tasks:
        steps.append(
            NextStep(
                text="This message links a task. Consider logging the same update onto the task so progress is visible on the board.",
                commands=(f"tasks log --id {tasks[0]} --agent <you> --log '{{\"kind\":\"note\",\"text\":\"see chat {mid or '<msg>'} in #{chan or 'channel'}\"}}'",),
                links=(f"task:{tasks[0]}",) + ((f"chat:{mid}",) if mid else ()),
            )
        )
    if findings:
        steps.append(
            NextStep(
                text="This message links a Finding. Ensure it is addressed (taskified) or closed with a reason so it doesn't linger.",
                commands=(
                    f"findings get --id {findings[0]}",
                    f"findings address --id {findings[0]} --actor <you>",
                ),
                links=(f"finding:{findings[0]}",) + ((f"chat:{mid}",) if mid else ()),
            )
        )
    if fixreports:
        steps.append(
            NextStep(
                text="This message links a FixReport. Ensure linked Findings are closed and CCR is acknowledged if applicable.",
                commands=(f"fixreport get --bf-id {fixreports[0]}",),
                links=(f"fixreport:{fixreports[0]}",) + ((f"chat:{mid}",) if mid else ()),
            )
        )
    if hypotheses:
        steps.append(
            NextStep(
                text="This message links a hypothesis. If it contains new evidence, add it as an evidence entry (not just chat text).",
                commands=(f"hypothesis add-evidence --hypothesis-id {hypotheses[0]} --kind observation --polarity inconclusive --strength weak --summary \"from chat {mid or '<msg>'}\" --sources-json '[]' --actor <you>",),
                links=(f"hypothesis:{hypotheses[0]}",) + ((f"chat:{mid}",) if mid else ()),
            )
        )
    if code_reviews and scripts:
        steps.append(
            NextStep(
                text="This message links a CCR review. Inspect/ack the latest CCR so /reviews can converge to done (checked + no open Findings).",
                commands=(f"checkreview check --no-mark {scripts[0]}", f"checkreview check {scripts[0]}"),
                links=(f"code_review:{code_reviews[0]}", f"script:{scripts[0]}"),
            )
        )

    return [s.as_json() for s in steps[:10]]
