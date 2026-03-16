"""Build deterministic compile prompts for local self-coding jobs.

This module keeps the human-readable compile prompt separate from the worker
orchestration code. The goal is to anchor Codex on Twinr's actual
``automation_manifest`` contract instead of letting the model invent adjacent
manifest formats that the deterministic compiler cannot activate safely.
"""

from __future__ import annotations

import json

from twinr.agent.self_coding.contracts import CompileJobRecord, RequirementsDialogueSession
from twinr.agent.self_coding.status import CompileTarget

_DEFAULT_TIME_TRIGGER_EXAMPLE = {
    "kind": "time",
    "schedule": "daily",
    "time_of_day": "08:00",
    "weekdays": [0, 1, 2, 3, 4, 5, 6],
    "timezone_name": "Europe/Berlin",
}


def build_compile_prompt(job: CompileJobRecord, session: RequirementsDialogueSession) -> str:
    """Build one deterministic prompt for a local compile request.

    Args:
        job: Compile job metadata describing the requested target.
        session: Confirmed requirements dialogue session for the skill.

    Returns:
        A plain-text prompt that instructs Codex to return only JSON matching
        the strict compile output schema and to embed one valid Twinr artifact.
    """

    target = job.requested_target.value
    prompt = [
        "You are compiling a Twinr self-coding request into reviewable artifacts.",
        "",
        "Read the workspace files `REQUEST.md`, `skill_spec.json`, `dialogue_session.json`, and `compile_job.json`.",
        "Return only JSON that matches the provided output schema.",
        "Do not ask follow-up questions.",
        f"Requested target: {target}",
        f"Skill name: {session.skill_name}",
        f"Request summary: {session.request_summary}",
        "",
        "Rules:",
        "- If the request fits the target, set `status` to `ok` and include at least one artifact of the requested kind.",
        "- If the request does not fit the target, set `status` to `unsupported`, explain why in `summary`, and keep artifacts empty.",
        "- Put all artifact contents directly into the JSON response; do not describe patches or external files.",
    ]
    if job.requested_target == CompileTarget.AUTOMATION_MANIFEST:
        prompt.extend(_automation_manifest_prompt_lines(session))
    return "\n".join(prompt) + "\n"


def _automation_manifest_prompt_lines(session: RequirementsDialogueSession) -> list[str]:
    """Build the target-specific prompt block for `automation_manifest` output."""

    event_trigger_example = _event_trigger_example(session)
    action_example = _action_example(session)
    compile_context = {
        "trigger_mode": session.trigger_mode,
        "trigger_conditions": list(session.trigger_conditions),
        "scope": dict(session.scope),
        "constraints": list(session.constraints),
        "capabilities": list(session.capabilities),
        "preferred_event_trigger": event_trigger_example,
        "preferred_action": action_example,
    }
    automation_example = {
        "automation": {
            "name": session.skill_name,
            "description": session.request_summary,
            "trigger": event_trigger_example,
            "actions": [action_example],
        }
    }
    time_example = {
        "automation": {
            "name": session.skill_name,
            "description": session.request_summary,
            "trigger": _DEFAULT_TIME_TRIGGER_EXAMPLE,
            "actions": [action_example],
        }
    }
    return [
        "- For `automation_manifest`, emit a complete JSON object with a top-level `automation` object only.",
        "- The `automation` object must contain `name`, `description`, `trigger`, and `actions`.",
        "- The `trigger` object must always include `kind` and the required fields for that kind.",
        "- Each action object must use Twinr automation fields such as `kind`, `text`, `tool_name`, `payload`, and `enabled`.",
        "- Do not invent alternate manifest schemas or extra wrapper fields.",
        "- Do not include `schema`, `version`, `skill_id`, `skill_name`, `capabilities`, `constraints`, `mode`, `conditions`, `type`, `channel`, `device`, or `ask_first` inside the artifact content unless they appear in the exact examples below.",
        "- If the request sounds schedule-based but the files do not provide enough data for a full time trigger, prefer the event-style trigger example below instead of guessing a clock time or emitting an incomplete trigger.",
        "- If the requested behavior is spoken output and the capabilities include `speaker`, prefer a concrete `say` action unless the files clearly require another supported action.",
        "Compile context:",
        _json_block(compile_context),
        "Valid event-style automation example:",
        _json_block(automation_example),
        "Valid time-style automation example:",
        _json_block(time_example),
    ]


def _event_trigger_example(session: RequirementsDialogueSession) -> dict[str, object]:
    """Return a minimal valid event trigger example for the current session."""

    event_name = session.trigger_conditions[0] if session.trigger_conditions else "user_visible"
    return {
        "kind": "if_then",
        "event_name": event_name,
        "all_conditions": [],
        "any_conditions": [],
        "cooldown_seconds": 60,
    }


def _action_example(session: RequirementsDialogueSession) -> dict[str, object]:
    """Return a minimal valid action example for the current session."""

    spoken_text = session.action.strip() or session.request_summary.strip() or session.skill_name.strip()
    if "speaker" in session.capabilities:
        return {
            "kind": "say",
            "text": spoken_text,
            "enabled": True,
        }
    return {
        "kind": "tool_call",
        "tool_name": "todo",
        "payload": {},
        "enabled": True,
    }


def _json_block(payload: object) -> str:
    """Render one JSON snippet for prompt embedding."""

    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
