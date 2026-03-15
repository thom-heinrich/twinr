from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Mapping, Protocol

from twinr.agent.base_agent.config import TwinrConfig
from twinr.llm_json import request_structured_json_object
from twinr.memory.longterm.models import LongTermMemoryObjectV1

if TYPE_CHECKING:
    from twinr.providers.openai import OpenAIBackend


class LongTermStructuredReflectionProgram(Protocol):
    def compile_reflection(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        timezone_name: str,
        packet_limit: int,
    ) -> Mapping[str, object]:
        ...


@dataclass(frozen=True, slots=True)
class OpenAIStructuredReflectionProgram:
    backend: "OpenAIBackend"
    model: str | None = None
    max_output_tokens: int = 900

    def compile_reflection(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        timezone_name: str,
        packet_limit: int,
    ) -> Mapping[str, object]:
        object_payload = [
            {
                "memory_id": item.memory_id,
                "kind": item.kind,
                "summary": item.summary,
                "details": item.details,
                "status": item.status,
                "confidence": item.confidence,
                "sensitivity": item.sensitivity,
                "valid_from": item.valid_from,
                "valid_to": item.valid_to,
                "attributes": dict(item.attributes or {}),
            }
            for item in objects
        ]
        return request_structured_json_object(
            self.backend,
            prompt="\n".join(
                (
                    "Compile a bounded mid-term memory layer from the current long-term memory window.",
                    "Internal memory must stay in canonical English.",
                    "Focus on near-term continuity that should influence the next few turns: ongoing life threads, today's situations, active plans, practical follow-ups, and fresh multi-turn context.",
                    "Do not invent unstated diagnoses, motives, or preferences.",
                    "Prefer broad packet kinds and put semantic detail into attributes.",
                    f"Timezone: {timezone_name}",
                    f"Packet limit: {packet_limit}",
                    "Window objects JSON:",
                    json.dumps(object_payload, ensure_ascii=False),
                )
            ),
            instructions="\n".join(
                (
                    "Return one strict JSON object only.",
                    "Create zero to packet_limit midterm packets.",
                    "Each packet must summarize a recent or currently relevant continuity thread.",
                    "Use source_memory_ids to cite the memory objects the packet is grounded in.",
                    "Use query_hints for words that should help retrieval later.",
                    "Use canonical English in summary, details, and query_hints.",
                    "Do not output duplicate packets for the same continuity thread.",
                )
            ),
            schema_name="twinr_long_term_midterm_reflection_v1",
            schema=_midterm_reflection_schema(),
            model=self.model or self.backend.config.default_model,
            reasoning_effort="low",
            max_output_tokens=max(300, self.max_output_tokens),
        )


def structured_reflection_program_from_config(
    config: TwinrConfig,
) -> LongTermStructuredReflectionProgram | None:
    if not config.long_term_memory_reflection_compiler_enabled:
        return None
    if not config.openai_api_key:
        return None
    from twinr.providers.openai import OpenAIBackend

    return OpenAIStructuredReflectionProgram(
        backend=OpenAIBackend(
            config,
            base_instructions="",
        ),
        model=config.long_term_memory_reflection_compiler_model,
        max_output_tokens=config.long_term_memory_reflection_compiler_max_output_tokens,
    )


def _midterm_reflection_schema() -> dict[str, object]:
    nullable_string = {
        "anyOf": [
            {"type": "string"},
            {"type": "null"},
        ]
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "midterm_packets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "packet_id": {"type": "string"},
                        "kind": {"type": "string"},
                        "summary": {"type": "string"},
                        "details": nullable_string,
                        "source_memory_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "query_hints": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "sensitivity": {"type": "string"},
                        "valid_from": nullable_string,
                        "valid_to": nullable_string,
                    },
                    "required": [
                        "packet_id",
                        "kind",
                        "summary",
                        "source_memory_ids",
                        "query_hints",
                        "sensitivity",
                        "details",
                        "valid_from",
                        "valid_to",
                    ],
                },
            }
        },
        "required": ["midterm_packets"],
    }


__all__ = [
    "LongTermStructuredReflectionProgram",
    "OpenAIStructuredReflectionProgram",
    "structured_reflection_program_from_config",
]
