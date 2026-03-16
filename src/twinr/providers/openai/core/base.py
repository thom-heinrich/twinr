"""Provide shared OpenAI request-building helpers for Twinr backends.

``OpenAIBackendBase`` owns client injection, base-instruction resolution, and
Responses API payload normalization that the higher OpenAI capability mixins
reuse.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.conversation.language import user_response_language_instruction
from twinr.agent.base_agent.prompting.personality import (
    load_personality_instructions,
    load_tool_loop_instructions,
    merge_instructions,
)

from .client import _default_client_factory
from .types import ConversationLike, OpenAIImageInput


class OpenAIBackendBase:
    """Provide shared request-building and normalization primitives.

    Capability mixins inherit from this base class to reuse canonical handling
    for conversation replay, prompt caching, web-search tools, image payloads,
    and common OpenAI Responses API request fields.

    Attributes:
        config: Active Twinr configuration used for request defaults.
    """

    _ALLOWED_MESSAGE_ROLES = frozenset({"assistant", "developer", "system", "user"})
    _ALLOWED_ASSISTANT_PHASES = frozenset({"commentary", "final_answer"})
    _ALLOWED_IMAGE_DETAILS = frozenset({"auto", "high", "low", "original"})
    _ALLOWED_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
    _ALLOWED_WEB_SEARCH_CONTEXT_SIZES = frozenset({"low", "medium", "high"})
    _MAX_IMAGE_URL_LENGTH = 20_971_520
    _MAX_PROMPT_CACHE_KEY_LENGTH = 64

    def __init__(
        self,
        config: TwinrConfig,
        *,
        client: Any | None = None,
        client_factory: Callable[[TwinrConfig], Any] | None = None,
        base_instructions: str | None = None,
    ) -> None:
        """Store config and construct or accept the underlying OpenAI client.

        Args:
            config: Active Twinr runtime configuration.
            client: Optional prebuilt client instance for tests or custom wiring.
            client_factory: Optional factory used when ``client`` is absent.
            base_instructions: Optional override for the shared base prompt.
        """

        self.config = config
        factory = client_factory or _default_client_factory
        self._client = client if client is not None else factory(config)  # AUDIT-FIX(#10): honor explicitly injected false-y client objects.
        self._base_instructions_override = base_instructions

    def _resolve_base_instructions(self) -> str | None:
        """Return the base personality instructions for normal turns."""

        if self._base_instructions_override is not None:
            return self._base_instructions_override
        return load_personality_instructions(self.config)

    def _resolve_tool_loop_base_instructions(self) -> str | None:
        """Return the base instructions used for tool-loop requests."""

        if self._base_instructions_override is not None:
            return self._base_instructions_override
        return load_tool_loop_instructions(self.config)

    def _call_with_model_fallback(
        self,
        preferred_model: str,
        fallback_models: Sequence[str],
        call: Callable[[str], Any],
    ) -> tuple[Any, str]:
        """Call a model-dependent function with deduplicated fallback models.

        Args:
            preferred_model: First model candidate to try.
            fallback_models: Additional model identifiers to try in order.
            call: Callback that performs the actual request for one model.

        Returns:
            A tuple of the callback result and the model identifier that
            succeeded.

        Raises:
            RuntimeError: If no candidate model is available to the project.
            Exception: Re-raises non-model-access failures from ``call``.
        """

        attempted_models: list[str] = []
        attempted_model_keys: set[str] = set()
        last_error: Exception | None = None
        for raw_model in (preferred_model, *fallback_models):
            model = self._clean_text(raw_model)
            normalized_model = model.lower()
            if not model or normalized_model in attempted_model_keys:
                continue
            attempted_model_keys.add(normalized_model)
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
        """Detect whether an exception represents model-access failure."""

        status_code = self._get_field(exc, "status_code")
        body = self._get_field(exc, "body")
        error = body.get("error") if isinstance(body, Mapping) else None
        error_code = self._clean_text(self._get_field(error, "code")).lower()
        error_message = self._clean_text(self._get_field(error, "message")).lower()
        message = " ".join(part for part in (error_message, self._clean_text(exc).lower()) if part)
        if error_code in {"model_not_found", "unsupported_model"}:
            return True
        if "does not have access to model" in message or "model_not_found" in message:
            return True
        if status_code in {403, 404} and "model" in message and any(
            needle in message for needle in ("access", "not found", "unsupported")
        ):
            return True
        return False  # AUDIT-FIX(#6): avoid masking unrelated auth/request/tool failures as model-access problems.

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
        prompt_cache_scope: str | None = None,
    ) -> dict[str, Any]:
        """Build a validated Responses API request payload.

        Args:
            prompt: Current user prompt text.
            conversation: Optional replayable prior conversation turns.
            instructions: Optional system-style instructions to attach.
            allow_web_search: Optional override for web-search tool usage.
            model: Model identifier to send to OpenAI.
            reasoning_effort: Requested reasoning-effort level.
            max_output_tokens: Optional output-token cap.
            extra_user_content: Optional non-text user content blocks.
            prompt_cache_scope: Optional logical scope for prompt caching.

        Returns:
            A request mapping ready for ``client.responses.create``.

        Raises:
            ValueError: If the model, output token limit, or prompt content is
                invalid.
        """

        normalized_model = self._clean_text(model)
        if not normalized_model:
            raise ValueError("model must be a non-empty string")
        request: dict[str, Any] = {
            "model": normalized_model,
            "input": self._build_input(prompt, conversation, extra_user_content=extra_user_content),
            "store": False,
        }
        self._apply_reasoning_effort(request, model=normalized_model, reasoning_effort=reasoning_effort)
        merged_instructions = merge_instructions(
            instructions,
            user_response_language_instruction(self.config.openai_realtime_language),
        )
        if merged_instructions:
            request["instructions"] = merged_instructions
        if max_output_tokens is not None:
            if max_output_tokens <= 0:
                raise ValueError("max_output_tokens must be greater than zero")
            request["max_output_tokens"] = max_output_tokens

        use_web_search = self.config.openai_enable_web_search if allow_web_search is None else allow_web_search
        tools = self._build_tools(use_web_search, model=normalized_model)
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        self._apply_prompt_cache(
            request,
            scope=prompt_cache_scope,
            model=normalized_model,
        )
        return request

    def _apply_prompt_cache(
        self,
        request: dict[str, Any],
        *,
        scope: str | None,
        model: str,
    ) -> None:
        """Attach prompt-cache settings when caching is enabled and scoped."""

        if not self.config.openai_prompt_cache_enabled:
            return
        normalized_scope = self._clean_text(scope)
        if not normalized_scope:
            return
        language = self._clean_text(self.config.openai_realtime_language).lower() or "default"
        key_source = f"{normalized_scope}:{self._clean_text(model).lower()}:{language}"
        key_digest = hashlib.sha256(key_source.encode("utf-8")).hexdigest()[:16]
        scope_hint = self._cache_key_component(normalized_scope, limit=20)
        request["prompt_cache_key"] = (f"twinr:{scope_hint}:{key_digest}")[: self._MAX_PROMPT_CACHE_KEY_LENGTH]  # AUDIT-FIX(#1): keep prompt_cache_key within the API's 64-char limit and avoid leaking raw scope strings.
        retention = self._normalize_prompt_cache_retention(self.config.openai_prompt_cache_retention)
        if retention:
            request["prompt_cache_retention"] = retention  # AUDIT-FIX(#1): only send retention values accepted by the API.

    def _apply_reasoning_effort(
        self,
        request: dict[str, Any],
        *,
        model: str,
        reasoning_effort: str | None,
    ) -> None:
        """Attach reasoning-effort settings when the target model supports it."""

        normalized_effort = self._clean_text(reasoning_effort).lower()
        if not normalized_effort:
            return
        if normalized_effort not in self._ALLOWED_REASONING_EFFORTS:
            raise ValueError(
                "reasoning_effort must be one of: none, minimal, low, medium, high, xhigh"
            )  # AUDIT-FIX(#7): fail fast on invalid effort values instead of sending an avoidable 400 to OpenAI.
        if not self._model_supports_reasoning_effort(model):
            return
        request["reasoning"] = {"effort": normalized_effort}

    def _model_supports_reasoning_effort(self, model: str) -> bool:
        """Return whether a model identifier supports reasoning controls."""

        normalized = self._clean_text(model).lower()
        if not normalized:
            return False
        return normalized.startswith(("gpt-5", "o"))  # AUDIT-FIX(#7): match the Responses API's gpt-5 and o-series reasoning families.

    def _build_input(
        self,
        prompt: str,
        conversation: ConversationLike | None = None,
        *,
        extra_user_content: Sequence[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build the outbound message list for a prompt and optional history.

        Args:
            prompt: Current prompt text for the user turn.
            conversation: Optional prior conversation turns to replay.
            extra_user_content: Optional non-text user content items.

        Returns:
            A list of Responses API message objects.

        Raises:
            ValueError: If the final payload would be empty.
            TypeError: If ``extra_user_content`` contains invalid items.
        """

        messages: list[dict[str, Any]] = []
        if conversation:
            for item in conversation:
                role, content, phase = self._coerce_message(item)
                if not role or not content:
                    continue
                message: dict[str, Any] = {
                    "role": role,
                    "content": [{"type": self._content_type_for_role(role), "text": content}],
                }
                if phase:
                    message["phase"] = phase  # AUDIT-FIX(#8): preserve assistant phase when replaying history so multi-step follow-ups keep their original semantics.
                messages.append(message)
        user_content: list[dict[str, Any]] = []
        prompt_text = self._clean_text(prompt)
        if prompt_text:
            user_content.append({"type": "input_text", "text": prompt_text})
        if extra_user_content:
            user_content.extend(self._normalize_extra_user_content(extra_user_content))
        if user_content:
            messages.append({"role": "user", "content": user_content})
        if not messages:
            raise ValueError(
                "At least one non-empty prompt, message, or extra_user_content item is required"
            )  # AUDIT-FIX(#2): avoid emitting an invalid empty user message/request payload.
        return messages

    def _build_image_content(self, images: Sequence[OpenAIImageInput]) -> list[dict[str, Any]]:
        """Convert validated image inputs into Responses API content blocks."""

        content: list[dict[str, Any]] = []
        for image in images:
            label = self._clean_text(getattr(image, "label", None))
            if label:
                content.append({"type": "input_text", "text": label})
            image_item: dict[str, Any] = {
                "type": "input_image",
                "image_url": self._image_data_url(image),
            }
            detail = self._normalize_image_detail(getattr(image, "detail", None) or self.config.openai_vision_detail)
            if detail:
                image_item["detail"] = detail
            content.append(image_item)
        return content

    def _image_data_url(self, image: OpenAIImageInput) -> str:
        """Encode an image input as a bounded data URL."""

        content_type = self._clean_text(getattr(image, "content_type", None)).lower()
        if not content_type.startswith("image/"):
            raise ValueError(f"Unsupported image content type: {content_type or '<empty>'}")
        data = getattr(image, "data", None)
        if not isinstance(data, (bytes, bytearray)) or not data:
            raise ValueError("Image data must be non-empty bytes")  # AUDIT-FIX(#4): reject empty or non-bytes image payloads before they become malformed data URLs.
        encoded = base64.b64encode(bytes(data)).decode("ascii")
        data_url = f"data:{content_type};base64,{encoded}"
        if len(data_url) > self._MAX_IMAGE_URL_LENGTH:
            raise ValueError(
                "Image payload exceeds the OpenAI Responses API image_url size limit"
            )  # AUDIT-FIX(#4): prevent oversized image requests that would 400 and can pressure RPi memory.
        return data_url

    def _coerce_message(self, item: object) -> tuple[str | None, str, str | None]:
        """Normalize one stored conversation item into role, text, and phase."""

        if isinstance(item, tuple) and len(item) == 2:
            role, content = item
            normalized_role = self._normalize_message_role(role)
            normalized_content = self._normalize_message_content(content)
            return normalized_role, normalized_content, None
        role: object | None
        content: object | None
        phase: object | None
        if isinstance(item, Mapping):
            item_type = self._clean_text(item.get("type"))
            if item_type and item_type != "message":
                return None, "", None
            role = item.get("role")
            content = item.get("content")
            phase = item.get("phase")
        else:
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
            phase = getattr(item, "phase", None)
        normalized_role = self._normalize_message_role(role)
        normalized_content = self._normalize_message_content(content)
        normalized_phase = self._normalize_assistant_phase(phase, normalized_role)
        return normalized_role, normalized_content, normalized_phase  # AUDIT-FIX(#2): support mapping/object messages, skip non-message items, and never stringify None into prompt text.

    def _content_type_for_role(self, role: str) -> str:
        """Return the Responses API text content type for replayed messages."""

        if role == "assistant":
            return "output_text"
        return "input_text"

    def _build_tools(self, use_web_search: bool, *, model: str) -> list[dict[str, Any]]:
        """Build the optional tool list for an outbound request."""

        if not use_web_search:
            return []

        if self._clean_text(model).lower() == "gpt-4.1-nano":
            return []

        tool: dict[str, Any] = {
            "type": "web_search",
            "search_context_size": self._normalize_web_search_context_size(
                self.config.openai_web_search_context_size
            ),  # AUDIT-FIX(#5): constrain search_context_size to the values accepted by the API.
        }
        user_location = self._build_user_location()
        if user_location:
            tool["user_location"] = user_location
        return [tool]

    def _build_user_location(self) -> dict[str, str] | None:
        """Build the approximate ``user_location`` block for web search."""

        country = self._normalize_country_code(self.config.openai_web_search_country)
        region = self._normalize_free_text(self.config.openai_web_search_region)
        city = self._normalize_free_text(self.config.openai_web_search_city)
        timezone = self._normalize_timezone(self.config.openai_web_search_timezone)
        if not any((country, region, city, timezone)):
            return None
        user_location: dict[str, str] = {"type": "approximate"}
        if country:
            user_location["country"] = country
        if region:
            user_location["region"] = region
        if city:
            user_location["city"] = city
        if timezone:
            user_location["timezone"] = timezone
        return user_location  # AUDIT-FIX(#5): only emit user_location fields that conform to the web_search schema.

    def _extract_output_text(self, response: Any) -> str:
        """Extract plain text from a Responses API result object or mapping."""

        text = self._clean_text(self._get_field(response, "output_text"))
        if text:
            return text

        output_items = self._get_field(response, "output") or []
        fragments: list[str] = []
        for item in output_items:
            for content in self._get_field(item, "content") or []:
                content_type = self._clean_text(self._get_field(content, "type"))
                content_text = self._clean_text(self._get_field(content, "text"))
                if content_type in {"output_text", "text"} and content_text:
                    fragments.append(content_text)
        return "\n".join(fragment for fragment in fragments if fragment).strip()  # AUDIT-FIX(#9): support both SDK objects and dict-shaped response payloads.

    def _used_web_search(self, response: Any) -> bool:
        """Return whether a response emitted a web-search tool call."""

        output_items = self._get_field(response, "output") or []
        return any(self._clean_text(self._get_field(item, "type")) == "web_search_call" for item in output_items)  # AUDIT-FIX(#9): support both SDK objects and dict-shaped response payloads.

    def _limit_recent_conversation(
        self,
        conversation: ConversationLike | None,
        *,
        max_turns: int,
    ) -> ConversationLike | None:
        """Trim conversation history to the newest ``max_turns`` entries."""

        if not conversation:
            return conversation
        turns = list(conversation)
        if max_turns <= 0 or len(turns) <= max_turns:
            return turns
        return turns[-max_turns:]

    def _get_field(self, item: object, field: str, default: Any = None) -> Any:
        """Read a named field from either a mapping or an attribute object."""

        if item is None:
            return default
        if isinstance(item, Mapping):
            return item.get(field, default)
        return getattr(item, field, default)

    def _clean_text(self, value: object) -> str:
        """Convert a value into stripped text without propagating ``None``."""

        if value is None:
            return ""
        return str(value).strip()

    def _normalize_message_role(self, role: object) -> str | None:
        """Normalize a conversation role into an allowed OpenAI role."""

        normalized = self._clean_text(role).lower()
        if normalized in self._ALLOWED_MESSAGE_ROLES:
            return normalized
        return None

    def _normalize_assistant_phase(self, phase: object, role: str | None) -> str | None:
        """Normalize assistant phase metadata for replayed history items."""

        if role != "assistant":
            return None
        normalized = self._clean_text(phase).lower()
        if normalized in self._ALLOWED_ASSISTANT_PHASES:
            return normalized
        return None

    def _normalize_message_content(self, content: object) -> str:
        """Flatten nested message content into trimmed plain text."""

        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, Mapping):
            nested_content = self._get_field(content, "content")
            if nested_content is not None:
                nested_text = self._normalize_message_content(nested_content)
                if nested_text:
                    return nested_text
            return self._clean_text(self._get_field(content, "text"))
        if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray, str)):
            parts: list[str] = []
            for part in content:
                part_text = self._normalize_message_content(part)
                if part_text:
                    parts.append(part_text)
            return "\n".join(parts).strip()
        text = self._clean_text(self._get_field(content, "text"))
        if text:
            return text
        nested_content = self._get_field(content, "content")
        if nested_content is not None:
            nested_text = self._normalize_message_content(nested_content)
            if nested_text:
                return nested_text
        return self._clean_text(content)

    def _normalize_extra_user_content(
        self,
        extra_user_content: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Copy and validate extra user content items for request assembly."""

        normalized: list[dict[str, Any]] = []
        for item in extra_user_content:
            if not isinstance(item, Mapping):
                raise TypeError("extra_user_content items must be mapping-like objects")
            normalized.append(dict(item))
        return normalized

    def _normalize_image_detail(self, detail: object) -> str | None:
        """Normalize an image detail hint into an allowed API enum value."""

        normalized = self._clean_text(detail).lower()
        if normalized in self._ALLOWED_IMAGE_DETAILS:
            return normalized
        return None

    def _normalize_prompt_cache_retention(self, retention: object) -> str | None:
        """Normalize prompt-cache retention into a supported API value."""

        normalized = self._clean_text(retention).lower().replace("-", "_")
        if not normalized:
            return None
        if normalized in {"24h", "in_memory"}:
            return normalized
        return None

    def _cache_key_component(self, value: str, *, limit: int) -> str:
        """Collapse a free-form scope string into a safe cache-key fragment."""

        cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")
        if not cleaned:
            return "default"
        return cleaned[:limit]

    def _normalize_web_search_context_size(self, value: object) -> str:
        """Normalize web-search context size to the supported API enum."""

        normalized = self._clean_text(value).lower()
        if normalized in self._ALLOWED_WEB_SEARCH_CONTEXT_SIZES:
            return normalized
        return "medium"

    def _normalize_free_text(self, value: object) -> str | None:
        """Normalize an optional free-text location field."""

        normalized = " ".join(self._clean_text(value).split())
        if not normalized:
            return None
        return normalized

    def _normalize_country_code(self, value: object) -> str | None:
        """Normalize an optional two-letter country code."""

        normalized = self._clean_text(value).upper()
        if len(normalized) == 2 and normalized.isalpha():
            return normalized
        return None

    def _normalize_timezone(self, value: object) -> str | None:
        """Validate and normalize an optional IANA timezone name."""

        normalized = self._clean_text(value)
        if not normalized:
            return None
        try:
            ZoneInfo(normalized)
        except ZoneInfoNotFoundError:
            return None
        return normalized
