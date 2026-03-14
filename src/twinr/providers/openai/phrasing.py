from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from twinr.agent.base_agent.personality import merge_instructions
from twinr.memory.reminders import ReminderEntry, format_due_label
from twinr.ops.usage import extract_model_name, extract_token_usage

from .instructions import (
    AUTOMATION_EXECUTION_INSTRUCTIONS,
    PROACTIVE_PROMPT_INSTRUCTIONS,
    REMINDER_DELIVERY_INSTRUCTIONS,
)
from .types import ConversationLike, OpenAITextResponse


class OpenAIMessagePhrasingMixin:
    def phrase_due_reminder_with_metadata(
        self,
        reminder: ReminderEntry,
        *,
        now: datetime | None = None,
    ) -> OpenAITextResponse:
        current_time = now or datetime.now(ZoneInfo(self.config.local_timezone_name))
        prompt_parts = [
            "A stored Twinr reminder is due now.",
            f"Current local time: {format_due_label(current_time, timezone_name=self.config.local_timezone_name)}",
            f"Scheduled reminder time: {format_due_label(reminder.due_at, timezone_name=self.config.local_timezone_name)}",
            f"Reminder kind: {reminder.kind}",
            f"Reminder summary: {reminder.summary}",
        ]
        if reminder.details:
            prompt_parts.append(f"Reminder details: {reminder.details}")
        if reminder.original_request:
            prompt_parts.append(f"Original user request: {reminder.original_request}")
        prompt_parts.append("Speak the reminder now.")
        request = self._build_response_request(
            "\n".join(prompt_parts),
            instructions=merge_instructions(self._resolve_base_instructions(), REMINDER_DELIVERY_INSTRUCTIONS),
            allow_web_search=False,
            model=self.config.default_model,
            reasoning_effort="low",
            max_output_tokens=140,
        )
        response = self._client.responses.create(**request)
        return OpenAITextResponse(
            text=self._extract_output_text(response),
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, request["model"]),
            token_usage=extract_token_usage(response),
            used_web_search=False,
        )

    def phrase_proactive_prompt_with_metadata(
        self,
        *,
        trigger_id: str,
        reason: str,
        default_prompt: str,
        priority: int,
        conversation: ConversationLike | None = None,
        recent_prompts: tuple[str, ...] = (),
        observation_facts: tuple[str, ...] = (),
    ) -> OpenAITextResponse:
        prompt_parts = [
            "Twinr is about to speak a short proactive social prompt.",
            f"Trigger id: {trigger_id}",
            f"Priority: {priority}",
            f"Observation summary: {reason.strip() or '[none]'}",
            f"Default fallback wording: {default_prompt.strip() or '[none]'}",
        ]
        fact_lines = [item.strip() for item in observation_facts if item.strip()]
        if fact_lines:
            prompt_parts.append("Observed evidence:")
            prompt_parts.extend(f"- {line}" for line in fact_lines[:5])
        recent_lines = [item.strip() for item in recent_prompts if item.strip()]
        if recent_lines:
            prompt_parts.append("Recent proactive wording to avoid repeating too closely:")
            prompt_parts.extend(f"- {line}" for line in recent_lines[:3])
        prompt_parts.append("Write the proactive spoken line now.")
        request = self._build_response_request(
            "\n".join(prompt_parts),
            conversation=self._limit_recent_conversation(conversation, max_turns=4),
            instructions=merge_instructions(
                self._resolve_base_instructions(),
                PROACTIVE_PROMPT_INSTRUCTIONS,
            ),
            allow_web_search=False,
            model=self.config.default_model,
            reasoning_effort="low",
            max_output_tokens=80,
        )
        response = self._client.responses.create(**request)
        return OpenAITextResponse(
            text=self._extract_output_text(response),
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, request["model"]),
            token_usage=extract_token_usage(response),
            used_web_search=False,
        )

    def fulfill_automation_prompt_with_metadata(
        self,
        prompt: str,
        *,
        allow_web_search: bool,
        delivery: str = "spoken",
    ) -> OpenAITextResponse:
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise RuntimeError("Automation prompt must not be empty")
        delivery_mode = delivery.strip().lower() or "spoken"
        request = self._build_response_request(
            "\n".join(
                [
                    f"Scheduled automation request: {normalized_prompt}",
                    f"Delivery mode: {delivery_mode}",
                    "Fulfill the automation request now.",
                ]
            ),
            instructions=merge_instructions(self._resolve_base_instructions(), AUTOMATION_EXECUTION_INSTRUCTIONS),
            allow_web_search=allow_web_search,
            model=self.config.default_model,
            reasoning_effort="medium" if allow_web_search else "low",
            max_output_tokens=220 if delivery_mode == "printed" else 160,
        )
        response = self._client.responses.create(**request)
        return OpenAITextResponse(
            text=self._extract_output_text(response),
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, request["model"]),
            token_usage=extract_token_usage(response),
            used_web_search=self._used_web_search(response),
        )
