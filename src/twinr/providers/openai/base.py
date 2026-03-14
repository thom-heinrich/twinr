from __future__ import annotations

import base64
from typing import Any, Callable, Sequence

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.language import user_response_language_instruction
from twinr.agent.base_agent.personality import load_personality_instructions, merge_instructions

from .client import _default_client_factory
from .types import ConversationLike, OpenAIImageInput


class OpenAIBackendBase:
    def __init__(
        self,
        config: TwinrConfig,
        *,
        client: Any | None = None,
        client_factory: Callable[[TwinrConfig], Any] | None = None,
        base_instructions: str | None = None,
    ) -> None:
        self.config = config
        factory = client_factory or _default_client_factory
        self._client = client or factory(config)
        self._base_instructions_override = base_instructions

    def _resolve_base_instructions(self) -> str | None:
        if self._base_instructions_override is not None:
            return self._base_instructions_override
        return load_personality_instructions(self.config)

    def _call_with_model_fallback(
        self,
        preferred_model: str,
        fallback_models: Sequence[str],
        call: Callable[[str], Any],
    ) -> tuple[Any, str]:
        attempted_models: list[str] = []
        last_error: Exception | None = None
        for model in (preferred_model, *fallback_models):
            if not model or model in attempted_models:
                continue
            attempted_models.append(model)
            try:
                return call(model), model
            except Exception as exc:
                if not self._is_model_access_error(exc):
                    raise
                last_error = exc
        if last_error is not None:
            candidate_list = ", ".join(attempted_models)
            raise RuntimeError(
                f"OpenAI project does not have access to any candidate models for this request: {candidate_list}"
            ) from last_error
        raise RuntimeError("No model candidates were available for the OpenAI request")

    def _is_model_access_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        body = getattr(exc, "body", None)
        error_code = None
        if isinstance(body, dict):
            error_code = body.get("error", {}).get("code")
        message = str(exc).lower()
        return (
            error_code == "model_not_found"
            or "does not have access to model" in message
            or "model_not_found" in message
            or status_code in {403, 404}
        )

    def _build_response_request(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int | None = None,
        extra_user_content: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model,
            "input": self._build_input(prompt, conversation, extra_user_content=extra_user_content),
            "reasoning": {"effort": reasoning_effort},
            "store": False,
        }
        merged_instructions = merge_instructions(
            instructions,
            user_response_language_instruction(self.config.openai_realtime_language),
        )
        if merged_instructions:
            request["instructions"] = merged_instructions
        if max_output_tokens is not None:
            request["max_output_tokens"] = max_output_tokens

        use_web_search = self.config.openai_enable_web_search if allow_web_search is None else allow_web_search
        tools = self._build_tools(use_web_search)
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        return request

    def _build_input(
        self,
        prompt: str,
        conversation: ConversationLike | None = None,
        *,
        extra_user_content: Sequence[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if conversation:
            for item in conversation:
                role, content = self._coerce_message(item)
                if not content:
                    continue
                messages.append(
                    {
                        "role": role,
                        "content": [{"type": self._content_type_for_role(role), "text": content}],
                    }
                )
        user_content: list[dict[str, Any]] = []
        prompt_text = prompt.strip()
        if prompt_text:
            user_content.append({"type": "input_text", "text": prompt_text})
        if extra_user_content:
            user_content.extend(extra_user_content)
        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_image_content(self, images: Sequence[OpenAIImageInput]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for image in images:
            if image.label:
                content.append({"type": "input_text", "text": image.label})
            image_item: dict[str, Any] = {
                "type": "input_image",
                "image_url": self._image_data_url(image),
            }
            detail = (image.detail or self.config.openai_vision_detail or "").strip()
            if detail:
                image_item["detail"] = detail
            content.append(image_item)
        return content

    def _image_data_url(self, image: OpenAIImageInput) -> str:
        if not image.content_type.startswith("image/"):
            raise ValueError(f"Unsupported image content type: {image.content_type}")
        encoded = base64.b64encode(image.data).decode("ascii")
        return f"data:{image.content_type};base64,{encoded}"

    def _coerce_message(self, item: object) -> tuple[str, str]:
        if isinstance(item, tuple) and len(item) == 2:
            role, content = item
            return str(role), str(content).strip()
        role = str(getattr(item, "role"))
        content = str(getattr(item, "content")).strip()
        return role, content

    def _content_type_for_role(self, role: str) -> str:
        if role.strip().lower() == "assistant":
            return "output_text"
        return "input_text"

    def _build_tools(self, use_web_search: bool) -> list[dict[str, Any]]:
        if not use_web_search:
            return []

        tool: dict[str, Any] = {
            "type": "web_search",
            "search_context_size": self.config.openai_web_search_context_size,
        }
        user_location = self._build_user_location()
        if user_location:
            tool["user_location"] = user_location
        return [tool]

    def _build_user_location(self) -> dict[str, str] | None:
        fields = {
            "type": "approximate",
            "country": self.config.openai_web_search_country,
            "region": self.config.openai_web_search_region,
            "city": self.config.openai_web_search_city,
            "timezone": self.config.openai_web_search_timezone,
        }
        if not any(value for key, value in fields.items() if key != "type"):
            return None
        return {key: value for key, value in fields.items() if value}

    def _extract_output_text(self, response: Any) -> str:
        text = str(getattr(response, "output_text", "")).strip()
        if text:
            return text

        output_items = getattr(response, "output", None) or []
        fragments: list[str] = []
        for item in output_items:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) in {"output_text", "text"} and getattr(content, "text", None):
                    fragments.append(str(content.text).strip())
        return "\n".join(fragment for fragment in fragments if fragment).strip()

    def _used_web_search(self, response: Any) -> bool:
        output_items = getattr(response, "output", None) or []
        return any(getattr(item, "type", None) == "web_search_call" for item in output_items)

    def _limit_recent_conversation(
        self,
        conversation: ConversationLike | None,
        *,
        max_turns: int,
    ) -> ConversationLike | None:
        if not conversation:
            return conversation
        turns = list(conversation)
        if max_turns <= 0 or len(turns) <= max_turns:
            return turns
        return turns[-max_turns:]
