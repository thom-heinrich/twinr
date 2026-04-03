"""Self-coding tool schema families."""

from __future__ import annotations

from typing import Any

from .context import SchemaBuildContext
from .shared import array_property, boolean_property, string_property


def build_propose_skill_learning_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "propose_skill_learning",
        "description": (
            "Start Twinr's self-coding learning flow for a new persistent capability the current tool surface cannot already satisfy. "
            "Use this only for new repeatable behaviors, not for one-off answers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": string_property(
                    "Short human-readable name for the skill Twinr should learn.",
                    min_length=1,
                ),
                "action": string_property(
                    "Plain-language summary of what the learned skill should do.",
                    min_length=1,
                ),
                "request_summary": string_property(
                    "Optional short paraphrase of the user's request in user-facing language.",
                    min_length=1,
                ),
                "skill_id": string_property(
                    "Optional stable identifier if one is already known.",
                    min_length=1,
                ),
                "trigger_mode": string_property(
                    "Optional preliminary trigger mode.",
                    enum=["push", "pull"],
                ),
                "trigger_conditions": array_property(
                    "Optional preliminary trigger conditions as stable identifiers.",
                    string_property("Trigger condition identifier.", min_length=1),
                    unique_items=True,
                ),
                # OpenAI strict tool schemas only support explicitly enumerated object keys,
                # so arbitrary skill scopes travel as compact JSON object strings.
                "scope": string_property(
                    'Optional preliminary structured scope encoded as a compact JSON object string, for example {"channel":"voice","contacts":["family"]}.',
                    min_length=2,
                ),
                "constraints": array_property(
                    "Optional preliminary constraints in plain language.",
                    string_property("Constraint.", min_length=1),
                    unique_items=True,
                ),
                "capabilities": array_property(
                    "Required ASE capabilities such as camera, pir, speaker, llm_call, memory, scheduler, rules, safety, email, or calendar.",
                    string_property("Capability identifier.", min_length=1),
                    min_items=1,
                    unique_items=True,
                ),
            },
            "required": ["name", "action", "capabilities"],
            "additionalProperties": False,
        },
    }


def build_answer_skill_question_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "answer_skill_question",
        "description": (
            "Continue an active self-coding requirements dialogue after Twinr has already asked one of its short follow-up questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": string_property(
                    "Dialogue session identifier previously returned by propose_skill_learning.",
                    min_length=1,
                ),
                "use_default": boolean_property(
                    "Set true when the user explicitly says to do whatever makes sense or to use the default."
                ),
                "trigger_mode": string_property(
                    "Optional updated trigger mode for the current answer.",
                    enum=["push", "pull"],
                ),
                "trigger_conditions": array_property(
                    "Optional extra trigger conditions to merge into the draft skill.",
                    string_property("Trigger condition identifier.", min_length=1),
                    unique_items=True,
                ),
                "scope": string_property(
                    'Optional shallow scope patch encoded as a compact JSON object string, for example {"contacts":["family"]}.',
                    min_length=2,
                ),
                "constraints": array_property(
                    "Optional extra constraints to merge into the draft skill.",
                    string_property("Constraint.", min_length=1),
                    unique_items=True,
                ),
                "action": string_property(
                    "Optional refined action wording for how the skill should behave.",
                    min_length=1,
                ),
                "answer_summary": string_property(
                    "Optional short summary of the user's answer for auditability.",
                    min_length=1,
                ),
                "confirmed": boolean_property(
                    "Set true or false only when Twinr is at the final confirmation step."
                ),
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    }


def build_confirm_skill_activation_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "confirm_skill_activation",
        "description": (
            "Enable a compiled self-coding skill only after the user explicitly agrees that the soft-launch version should be turned on."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": string_property(
                    "Compile job identifier that is already soft-launch ready.",
                    min_length=1,
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly approved enabling this learned behavior."
                ),
            },
            "required": ["job_id", "confirmed"],
            "additionalProperties": False,
        },
    }


def build_rollback_skill_activation_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "rollback_skill_activation",
        "description": (
            "Roll a learned self-coding skill back to an earlier version when the user wants the new behavior undone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_id": string_property(
                    "Stable identifier of the learned skill family to roll back.",
                    min_length=1,
                ),
                "target_version": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional explicit older version to restore. If omitted, restore the newest earlier stable version.",
                },
            },
            "required": ["skill_id"],
            "additionalProperties": False,
        },
    }


def build_pause_skill_activation_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "pause_skill_activation",
        "description": (
            "Pause one active learned self-coding skill version when the user or operator wants it temporarily disabled."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_id": string_property(
                    "Stable identifier of the learned skill family to pause.",
                    min_length=1,
                ),
                "version": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Active learned skill version to pause.",
                },
                "reason": string_property(
                    "Optional short pause reason such as operator_pause.",
                    min_length=1,
                ),
            },
            "required": ["skill_id", "version"],
            "additionalProperties": False,
        },
    }


def build_reactivate_skill_activation_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "reactivate_skill_activation",
        "description": (
            "Re-enable one paused learned self-coding skill version after the user or operator wants it active again."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_id": string_property(
                    "Stable identifier of the learned skill family to reactivate.",
                    min_length=1,
                ),
                "version": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Paused learned skill version to reactivate.",
                },
            },
            "required": ["skill_id", "version"],
            "additionalProperties": False,
        },
    }


TOOL_BUILDERS = (
    ("propose_skill_learning", build_propose_skill_learning_schema),
    ("answer_skill_question", build_answer_skill_question_schema),
    ("confirm_skill_activation", build_confirm_skill_activation_schema),
    ("rollback_skill_activation", build_rollback_skill_activation_schema),
    ("pause_skill_activation", build_pause_skill_activation_schema),
    ("reactivate_skill_activation", build_reactivate_skill_activation_schema),
)
