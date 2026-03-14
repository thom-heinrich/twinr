from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from twinr.agent.base_agent.personality import merge_instructions
from twinr.ops.usage import extract_model_name, extract_token_usage

from .types import ConversationLike, OpenAIImageInput, OpenAITextResponse


class OpenAIResponseMixin:
    def respond(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> str:
        return self.respond_with_metadata(
            prompt,
            conversation=conversation,
            instructions=instructions,
            allow_web_search=allow_web_search,
        ).text

    def respond_with_metadata(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> OpenAITextResponse:
        request = self._build_response_request(
            prompt,
            conversation=conversation,
            instructions=merge_instructions(self._resolve_base_instructions(), instructions),
            allow_web_search=allow_web_search,
            model=self.config.default_model,
            reasoning_effort=self.config.openai_reasoning_effort,
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

    def respond_to_images(
        self,
        prompt: str,
        *,
        images: Sequence[OpenAIImageInput],
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> str:
        return self.respond_to_images_with_metadata(
            prompt,
            images=images,
            conversation=conversation,
            instructions=instructions,
            allow_web_search=allow_web_search,
        ).text

    def respond_to_images_with_metadata(
        self,
        prompt: str,
        *,
        images: Sequence[OpenAIImageInput],
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> OpenAITextResponse:
        if not images:
            raise ValueError("At least one image is required for a vision request")

        request = self._build_response_request(
            prompt,
            conversation=conversation,
            instructions=merge_instructions(self._resolve_base_instructions(), instructions),
            allow_web_search=allow_web_search,
            model=self.config.default_model,
            reasoning_effort=self.config.openai_reasoning_effort,
            extra_user_content=self._build_image_content(images),
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

    def respond_streaming(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> OpenAITextResponse:
        request = self._build_response_request(
            prompt,
            conversation=conversation,
            instructions=merge_instructions(self._resolve_base_instructions(), instructions),
            allow_web_search=allow_web_search,
            model=self.config.default_model,
            reasoning_effort=self.config.openai_reasoning_effort,
        )
        with self._client.responses.stream(**request) as stream:
            for event in stream:
                if getattr(event, "type", None) == "response.output_text.delta" and on_text_delta is not None:
                    delta = str(getattr(event, "delta", ""))
                    if delta:
                        on_text_delta(delta)
            response = stream.get_final_response()
        return OpenAITextResponse(
            text=self._extract_output_text(response),
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, request["model"]),
            token_usage=extract_token_usage(response),
            used_web_search=self._used_web_search(response),
        )
