"""Provide shared OpenAI request-building helpers for Twinr backends.

``OpenAIBackendBase`` owns client injection, instruction resolution, 2026-era
Responses API request assembly, conversation-state handling, and payload
normalization for higher-level OpenAI capability mixins.
"""

from __future__ import annotations

import base64
import copy
import hashlib
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
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


# CHANGELOG: 2026-03-30
# BUG-1: Canonicalize prompt_cache_retention safely across current API-spec drift; accept both "in_memory" and "in-memory", emit the Responses API reference spelling.
# BUG-2: Fix unsupported web-search combinations that would 400 in production (e.g. gpt-5 + reasoning=minimal, deep-research + user_location).
# BUG-3: Preserve full Responses API items across turns instead of flattening everything to plain text, so tool/reasoning state can survive multi-turn agent loops.
# BUG-4: Validate known model-specific reasoning.effort constraints locally to fail fast instead of shipping avoidable upstream 400s.
# SEC-1: Prevent practical Raspberry-Pi memory amplification / OOM risk by preflighting image data-url size before base64 encoding large payloads.
# IMP-1: Add 2026 Responses API controls: previous_response_id, conversation, include, safety_identifier, metadata, truncation, context_management, text verbosity, max_tool_calls, parallel_tool_calls.
# IMP-2: Add input-token counting helper, web-search domain filters, cache-only web search, and model-aware request shaping for long-running Physical AI agents.


class OpenAIBackendBase:
    """Provide shared request-building and normalization primitives.

    Capability mixins inherit from this base class to reuse canonical handling
    for conversation replay, conversation state, prompt caching, web-search
    tools, image payloads, reasoning settings, and common OpenAI Responses API
    request fields.

    Attributes:
        config: Active Twinr configuration used for request defaults.
    """

    _ALLOWED_MESSAGE_ROLES = frozenset({"assistant", "developer", "system", "user"})
    _ALLOWED_ASSISTANT_PHASES = frozenset({"commentary", "final_answer"})
    _ALLOWED_IMAGE_DETAILS = frozenset({"auto", "high", "low", "original"})
    _ALLOWED_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
    _ALLOWED_REASONING_SUMMARIES = frozenset({"auto", "concise", "detailed"})
    _ALLOWED_WEB_SEARCH_CONTEXT_SIZES = frozenset({"low", "medium", "high"})
    _ALLOWED_TEXT_VERBOSITIES = frozenset({"low", "medium", "high"})
    _ALLOWED_TRUNCATIONS = frozenset({"auto", "disabled"})
    _MAX_IMAGE_URL_LENGTH = 20_971_520
    _MAX_PROMPT_CACHE_KEY_LENGTH = 64
    _MAX_SAFETY_IDENTIFIER_LENGTH = 64
    _MAX_METADATA_ITEMS = 16
    _MAX_METADATA_KEY_LENGTH = 64
    _MAX_METADATA_VALUE_LENGTH = 512
    _DEFAULT_STATELESS_INCLUDE = ("reasoning.encrypted_content",)
    _DEFAULT_WEB_SEARCH_SOURCES_INCLUDE = "web_search_call.action.sources"
    _NON_WEB_SEARCH_MODELS = (
        "gpt-4.1-nano",
        "gpt-4.1-nano-",
        "gpt-4o-search-preview",
        "gpt-4o-mini-search-preview",
        "gpt-5-search-api",
    )
    _DEEP_RESEARCH_MODELS = (
        "o3-deep-research",
        "o3-deep-research-",
        "o4-mini-deep-research",
        "o4-mini-deep-research-",
    )
    _NON_REASONING_MODEL_PREFIXES = (
        "gpt-4.1",
        "gpt-4o",
        "gpt-image",
        "gpt-audio",
        "gpt-realtime",
        "computer-use-preview",
        "omni-moderation",
        "text-moderation",
        "text-embedding",
        "whisper",
        "tts-",
    )

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
        self._client = client if client is not None else factory(config)
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
        """Call a model-dependent function with deduplicated fallback models."""

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
        error = body.get("error") if isinstance(body, MappingABC) else None
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
        return False

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
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        include: Sequence[str] | None = None,
        parallel_tool_calls: bool | None = None,
        max_tool_calls: int | None = None,
        safety_identifier: str | None = None,
        reasoning_summary: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        truncation: str | None = None,
        context_management: Sequence[Mapping[str, Any]] | None = None,
        response_text_verbosity: str | None = None,
        store: bool | None = None,
    ) -> dict[str, Any]:
        """Build a validated Responses API request payload."""

        normalized_model = self._clean_text(model)
        if not normalized_model:
            raise ValueError("model must be a non-empty string")

        normalized_previous_response_id = self._clean_text(previous_response_id)
        normalized_conversation_id = self._clean_text(conversation_id)
        if normalized_previous_response_id and normalized_conversation_id:
            # BREAKING: ambiguous mixed state modes now fail fast instead of producing server-defined behavior.
            raise ValueError("previous_response_id and conversation_id cannot be used together")
        if conversation and (normalized_previous_response_id or normalized_conversation_id):
            # BREAKING: manual replay and server-managed conversation state are now treated as mutually exclusive.
            raise ValueError(
                "conversation replay cannot be combined with previous_response_id or conversation_id"
            )

        normalized_store = self._normalize_optional_bool(
            store,
            default=self._normalize_optional_bool(
                self._get_config_value("openai_store_responses"),
                default=False,
            ),
        )
        request: dict[str, Any] = {
            "model": normalized_model,
            "store": normalized_store,
        }
        if normalized_previous_response_id:
            request["previous_response_id"] = normalized_previous_response_id
        elif normalized_conversation_id:
            request["conversation"] = {"id": normalized_conversation_id}

        request["input"] = self._build_input(
            prompt,
            conversation,
            extra_user_content=extra_user_content,
        )

        self._apply_reasoning_settings(
            request,
            model=normalized_model,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
        )

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
        tools = self._build_tools(
            use_web_search,
            model=normalized_model,
            reasoning_effort=reasoning_effort,
        )
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"

        normalized_parallel_tool_calls = self._normalize_optional_bool(parallel_tool_calls)
        if normalized_parallel_tool_calls is not None:
            request["parallel_tool_calls"] = normalized_parallel_tool_calls

        if max_tool_calls is not None:
            if max_tool_calls <= 0:
                raise ValueError("max_tool_calls must be greater than zero")
            request["max_tool_calls"] = int(max_tool_calls)

        normalized_include = self._resolve_include(
            explicit_include=include,
            model=normalized_model,
            store=normalized_store,
            tools=tools,
        )
        if normalized_include:
            request["include"] = normalized_include

        normalized_safety_identifier = self._normalize_safety_identifier(safety_identifier)
        if normalized_safety_identifier:
            request["safety_identifier"] = normalized_safety_identifier

        normalized_metadata = self._normalize_metadata(metadata)
        if normalized_metadata:
            request["metadata"] = normalized_metadata

        normalized_truncation = self._resolve_truncation(truncation)
        if normalized_truncation:
            request["truncation"] = normalized_truncation

        normalized_context_management = self._resolve_context_management(context_management)
        if normalized_context_management:
            request["context_management"] = normalized_context_management

        normalized_text_verbosity = self._resolve_text_verbosity(response_text_verbosity)
        if normalized_text_verbosity:
            request["text"] = {"verbosity": normalized_text_verbosity}

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
        request["prompt_cache_key"] = (
            f"twinr:{scope_hint}:{key_digest}"
        )[: self._MAX_PROMPT_CACHE_KEY_LENGTH]
        retention = self._normalize_prompt_cache_retention(self.config.openai_prompt_cache_retention)
        if retention:
            request["prompt_cache_retention"] = retention

    def _apply_reasoning_settings(
        self,
        request: dict[str, Any],
        *,
        model: str,
        reasoning_effort: str | None,
        reasoning_summary: str | None,
    ) -> None:
        """Attach validated reasoning settings when the target model supports them."""

        normalized_effort = self._clean_text(reasoning_effort).lower()
        normalized_summary = self._resolve_reasoning_summary(reasoning_summary)

        if not normalized_effort and not normalized_summary:
            return
        if not self._model_supports_reasoning(model):
            return

        if normalized_effort:
            if normalized_effort not in self._ALLOWED_REASONING_EFFORTS:
                raise ValueError(
                    "reasoning_effort must be one of: none, minimal, low, medium, high, xhigh"
                )
            supported_efforts = self._supported_reasoning_efforts(model)
            if supported_efforts is not None and normalized_effort not in supported_efforts:
                # BREAKING: known-invalid model/effort combinations now fail locally instead of producing opaque OpenAI 400s.
                raise ValueError(
                    f"reasoning_effort={normalized_effort!r} is not supported for model {self._clean_text(model)!r}; "
                    f"supported values are: {', '.join(sorted(supported_efforts))}"
                )

        reasoning: dict[str, Any] = {}
        if normalized_effort:
            reasoning["effort"] = normalized_effort
        if normalized_summary:
            reasoning["summary"] = normalized_summary
        if reasoning:
            request["reasoning"] = reasoning

    def _model_supports_reasoning(self, model: str) -> bool:
        """Return whether a model identifier supports reasoning controls."""

        normalized = self._capability_model_name(model)
        if not normalized:
            return False
        if any(normalized.startswith(prefix) for prefix in self._NON_REASONING_MODEL_PREFIXES):
            return False
        return normalized.startswith("gpt-5") or normalized.startswith(("o1", "o3", "o4"))

    def _supported_reasoning_efforts(self, model: str) -> frozenset[str] | None:
        """Return a known supported reasoning-effort set for the model, if known."""

        normalized = self._capability_model_name(model)
        if not normalized or not normalized.startswith(("gpt-5", "o1", "o3", "o4")):
            return None

        if normalized.startswith(("gpt-5.4-pro", "gpt-5.2-pro")):
            return frozenset({"medium", "high", "xhigh"})
        if normalized.startswith("gpt-5-pro"):
            return frozenset({"high"})
        if normalized.startswith(("gpt-5.4", "gpt-5.2")):
            return frozenset({"none", "low", "medium", "high", "xhigh"})
        if normalized.startswith("gpt-5.1"):
            return frozenset({"none", "low", "medium", "high"})
        if normalized == "gpt-5" or normalized.startswith("gpt-5-"):
            return frozenset({"minimal", "low", "medium", "high"})
        if normalized.startswith(("o3-mini", "o4-mini", "o3", "o1")):
            return frozenset({"low", "medium", "high"})
        return None

    def _build_input(
        self,
        prompt: str,
        conversation: ConversationLike | None = None,
        *,
        extra_user_content: Sequence[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build the outbound message/item list for a prompt and optional history."""

        items: list[dict[str, Any]] = []
        if conversation:
            for item in conversation:
                normalized_item = self._normalize_conversation_item(item)
                if normalized_item:
                    items.append(normalized_item)

        user_content: list[dict[str, Any]] = []
        prompt_text = self._clean_text(prompt)
        if prompt_text:
            user_content.append({"type": "input_text", "text": prompt_text})
        if extra_user_content:
            user_content.extend(self._normalize_extra_user_content(extra_user_content))

        if user_content:
            items.append({"role": "user", "content": user_content})
        if not items:
            raise ValueError(
                "At least one non-empty prompt, message, or extra_user_content item is required"
            )
        return items

    def _normalize_conversation_item(self, item: object) -> dict[str, Any] | None:
        """Normalize one replay item for the Responses API."""

        mapping = self._as_mapping(item)
        item_type = self._clean_text(mapping.get("type")).lower() if mapping else ""
        if item_type and item_type != "message":
            return self._sanitize_passthrough_item(mapping)

        role: object | None
        content: object | None
        phase: object | None
        status: object | None
        if isinstance(item, tuple):
            if len(item) == 2:
                role, content = item
                phase = None
                status = None
            elif len(item) == 3:
                role, content, phase = item
                status = None
            else:
                role = None
                content = None
                phase = None
                status = None
        elif mapping is not None:
            role = mapping.get("role")
            content = mapping.get("content")
            phase = mapping.get("phase")
            status = mapping.get("status")
        else:
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
            phase = getattr(item, "phase", None)
            status = getattr(item, "status", None)

        normalized_role = self._normalize_message_role(role)
        if not normalized_role:
            return None
        content_items = self._normalize_message_content_items(normalized_role, content)
        if not content_items:
            return None

        message: dict[str, Any] = {
            "role": normalized_role,
            "content": content_items,
        }
        normalized_phase = self._normalize_assistant_phase(phase, normalized_role)
        if normalized_phase:
            message["phase"] = normalized_phase
        normalized_status = self._normalize_item_status(status)
        if normalized_status:
            message["status"] = normalized_status
        return message

    def _build_image_content(self, images: Sequence[OpenAIImageInput]) -> list[dict[str, Any]]:
        """Convert validated image inputs into Responses API content blocks."""

        content: list[dict[str, Any]] = []
        for image in images:
            label = self._clean_text(getattr(image, "label", None))
            if label:
                content.append({"type": "input_text", "text": label})
            content.append(self._image_input_content(image))
        return content

    def _image_input_content(self, image: OpenAIImageInput) -> dict[str, Any]:
        """Build one Responses API image content block from an image input."""

        image_item: dict[str, Any] = {"type": "input_image"}

        file_id = self._clean_text(getattr(image, "file_id", None))
        if file_id:
            image_item["file_id"] = file_id
        else:
            image_url = self._clean_text(getattr(image, "image_url", None))
            if image_url:
                if len(image_url) > self._MAX_IMAGE_URL_LENGTH:
                    raise ValueError("Image URL exceeds the OpenAI Responses API image_url size limit")
                image_item["image_url"] = image_url
            else:
                image_item["image_url"] = self._image_data_url(image)

        detail = self._normalize_image_detail(
            getattr(image, "detail", None) or self.config.openai_vision_detail
        )
        if detail:
            image_item["detail"] = detail
        return image_item

    def _image_data_url(self, image: OpenAIImageInput) -> str:
        """Encode an image input as a bounded data URL."""

        content_type = self._clean_text(getattr(image, "content_type", None)).lower()
        if not content_type.startswith("image/"):
            raise ValueError(f"Unsupported image content type: {content_type or '<empty>'}")
        data = getattr(image, "data", None)
        if not isinstance(data, (bytes, bytearray)) or not data:
            raise ValueError("Image data must be non-empty bytes")

        raw_bytes = bytes(data)
        data_url_prefix = f"data:{content_type};base64,"
        encoded_length = ((len(raw_bytes) + 2) // 3) * 4
        if len(data_url_prefix) + encoded_length > self._MAX_IMAGE_URL_LENGTH:
            raise ValueError(
                "Image payload exceeds the OpenAI Responses API image_url size limit"
            )
        encoded = base64.b64encode(raw_bytes).decode("ascii")
        return f"{data_url_prefix}{encoded}"

    def _coerce_message(self, item: object) -> tuple[str | None, str, str | None]:
        """Normalize one stored conversation item into role, text, and phase."""

        normalized_item = self._normalize_conversation_item(item)
        if not normalized_item:
            return None, "", None
        role = self._clean_text(normalized_item.get("role")).lower() or None
        phase = self._clean_text(normalized_item.get("phase")).lower() or None
        text_parts: list[str] = []
        for content_item in normalized_item.get("content", []):
            if not isinstance(content_item, MappingABC):
                continue
            item_type = self._clean_text(content_item.get("type")).lower()
            if item_type in {"input_text", "output_text"}:
                text = self._clean_text(content_item.get("text"))
                if text:
                    text_parts.append(text)
            elif item_type == "refusal":
                refusal = self._clean_text(content_item.get("refusal"))
                if refusal:
                    text_parts.append(refusal)
        return role, "\n".join(text_parts).strip(), phase

    def _content_type_for_role(self, role: str) -> str:
        """Return the canonical text content type for replayed messages."""

        return "output_text" if role == "assistant" else "input_text"

    def _build_tools(
        self,
        use_web_search: bool,
        *,
        model: str,
        reasoning_effort: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the optional tool list for an outbound request."""

        if not use_web_search:
            return []
        if not self._model_supports_web_search(model=model, reasoning_effort=reasoning_effort):
            return []

        tool: dict[str, Any] = {
            "type": "web_search",
            "search_context_size": self._normalize_web_search_context_size(
                self.config.openai_web_search_context_size
            ),
        }

        external_web_access = self._normalize_optional_bool(
            self._get_config_value("openai_web_search_external_web_access")
        )
        if external_web_access is not None:
            tool["external_web_access"] = external_web_access

        allowed_domains = self._normalize_allowed_domains(
            self._get_config_value("openai_web_search_allowed_domains")
        )
        if allowed_domains:
            tool["filters"] = {"allowed_domains": allowed_domains}

        user_location = self._build_user_location(model=model)
        if user_location:
            tool["user_location"] = user_location
        return [tool]

    def _model_supports_web_search(self, *, model: str, reasoning_effort: str | None) -> bool:
        """Return whether built-in web_search is supported for the request."""

        normalized_model = self._capability_model_name(model)
        if not normalized_model:
            return False
        if any(normalized_model == prefix or normalized_model.startswith(prefix) for prefix in self._NON_WEB_SEARCH_MODELS):
            return False
        if normalized_model == "gpt-5" and self._clean_text(reasoning_effort).lower() == "minimal":
            return False
        return True

    def _build_user_location(self, *, model: str) -> dict[str, str] | None:
        """Build the approximate ``user_location`` block for web search."""

        normalized_model = self._capability_model_name(model)
        if any(normalized_model == prefix or normalized_model.startswith(prefix) for prefix in self._DEEP_RESEARCH_MODELS):
            return None

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
        return user_location

    def _extract_output_text(self, response: Any) -> str:
        """Extract plain text from a Responses API result object or mapping."""

        text = self._clean_text(self._get_field(response, "output_text"))
        if text:
            return text

        output_items = self._get_field(response, "output") or []
        fragments: list[str] = []
        for item in output_items:
            for content in self._get_field(item, "content") or []:
                content_type = self._clean_text(self._get_field(content, "type")).lower()
                if content_type in {"output_text", "text"}:
                    content_text = self._clean_text(self._get_field(content, "text"))
                    if content_text:
                        fragments.append(content_text)
                elif content_type == "refusal":
                    refusal = self._clean_text(self._get_field(content, "refusal"))
                    if refusal:
                        fragments.append(refusal)
        return "\n".join(fragment for fragment in fragments if fragment).strip()

    def _used_web_search(self, response: Any) -> bool:
        """Return whether a response emitted a web-search tool call."""

        output_items = self._get_field(response, "output") or []
        return any(
            self._clean_text(self._get_field(item, "type")) == "web_search_call"
            for item in output_items
        )

    def _count_request_input_tokens(self, request: Mapping[str, Any]) -> int | None:
        """Return the exact input token count for a would-be Responses request if supported."""

        responses = getattr(self._client, "responses", None)
        input_tokens_api = getattr(responses, "input_tokens", None) if responses is not None else None
        count_method = getattr(input_tokens_api, "count", None)
        if not callable(count_method):
            return None

        payload: dict[str, Any] = {"model": self._clean_text(request.get("model"))}
        for key in (
            "conversation",
            "context_management",
            "include",
            "input",
            "instructions",
            "metadata",
            "previous_response_id",
            "reasoning",
            "safety_identifier",
            "text",
            "tools",
            "truncation",
        ):
            if key in request:
                payload[key] = copy.deepcopy(request[key])
        try:
            result = count_method(**payload)
        except Exception:
            return None
        input_tokens = self._get_field(result, "input_tokens")
        try:
            return int(input_tokens)
        except (TypeError, ValueError):
            return None

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
        if isinstance(item, MappingABC):
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

    def _normalize_item_status(self, status: object) -> str | None:
        """Normalize an optional item status."""

        normalized = self._clean_text(status).lower()
        if normalized in {"in_progress", "completed", "incomplete"}:
            return normalized
        return None

    def _normalize_message_content(self, content: object) -> str:
        """Flatten nested message content into trimmed plain text."""

        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, MappingABC):
            nested_content = self._get_field(content, "content")
            if nested_content is not None:
                nested_text = self._normalize_message_content(nested_content)
                if nested_text:
                    return nested_text
            return self._clean_text(self._get_field(content, "text"))
        if isinstance(content, SequenceABC) and not isinstance(content, (bytes, bytearray, str)):
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

    def _normalize_message_content_items(
        self,
        role: str,
        content: object,
    ) -> list[dict[str, Any]]:
        """Normalize message content while preserving structured Responses items."""

        if content is None:
            return []
        if isinstance(content, str):
            text = content.strip()
            if not text:
                return []
            return [{"type": self._content_type_for_role(role), "text": text}]

        mapping = self._as_mapping(content)
        if mapping is not None:
            explicit_type = self._clean_text(mapping.get("type")).lower()
            if explicit_type:
                content_item = self._normalize_message_content_item(role, mapping)
                if content_item:
                    return [content_item]

            nested_content = mapping.get("content")
            if nested_content is not None:
                nested_items = self._normalize_message_content_items(role, nested_content)
                if nested_items:
                    return nested_items

            if role == "assistant":
                refusal = self._clean_text(mapping.get("refusal"))
                if refusal:
                    return [{"type": "refusal", "refusal": refusal}]

            text = self._clean_text(mapping.get("text"))
            if text:
                return [{"type": self._content_type_for_role(role), "text": text}]
            return []

        if isinstance(content, SequenceABC) and not isinstance(content, (bytes, bytearray, str)):
            items: list[dict[str, Any]] = []
            for part in content:
                items.extend(self._normalize_message_content_items(role, part))
            return items

        text = self._clean_text(self._get_field(content, "text"))
        if text:
            return [{"type": self._content_type_for_role(role), "text": text}]
        nested_content = self._get_field(content, "content")
        if nested_content is not None:
            return self._normalize_message_content_items(role, nested_content)
        plain = self._clean_text(content)
        if plain:
            return [{"type": self._content_type_for_role(role), "text": plain}]
        return []

    def _normalize_message_content_item(
        self,
        role: str,
        item: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Normalize one structured message content item."""

        item_type = self._clean_text(item.get("type")).lower()
        if item_type in {"input_text", "output_text", "text"}:
            text = self._clean_text(item.get("text"))
            if not text:
                return None
            normalized_type = self._content_type_for_role(role)
            normalized_item: dict[str, Any] = {
                "type": normalized_type,
                "text": text,
            }
            if normalized_type == "output_text":
                annotations = item.get("annotations")
                if isinstance(annotations, SequenceABC) and not isinstance(annotations, (str, bytes, bytearray)):
                    normalized_item["annotations"] = copy.deepcopy(list(annotations))
                logprobs = item.get("logprobs")
                if isinstance(logprobs, SequenceABC) and not isinstance(logprobs, (str, bytes, bytearray)):
                    normalized_item["logprobs"] = copy.deepcopy(list(logprobs))
            return normalized_item

        if item_type == "refusal" and role == "assistant":
            refusal = self._clean_text(item.get("refusal"))
            if refusal:
                return {"type": "refusal", "refusal": refusal}
            return None

        if item_type == "input_image" and role != "assistant":
            normalized_item = {"type": "input_image"}
            file_id = self._clean_text(item.get("file_id"))
            image_url = self._clean_text(item.get("image_url"))
            if file_id:
                normalized_item["file_id"] = file_id
            elif image_url:
                if len(image_url) > self._MAX_IMAGE_URL_LENGTH:
                    raise ValueError("Image URL exceeds the OpenAI Responses API image_url size limit")
                normalized_item["image_url"] = image_url
            else:
                return None
            detail = self._normalize_image_detail(item.get("detail"))
            if detail:
                normalized_item["detail"] = detail
            return normalized_item

        if item_type == "input_file" and role != "assistant":
            normalized_item: dict[str, Any] = {"type": "input_file"}
            file_id = self._clean_text(item.get("file_id"))
            file_url = self._clean_text(item.get("file_url"))
            file_data = self._clean_text(item.get("file_data"))
            filename = self._clean_text(item.get("filename"))
            if file_id:
                normalized_item["file_id"] = file_id
            elif file_url:
                normalized_item["file_url"] = file_url
            elif file_data:
                normalized_item["file_data"] = file_data
            else:
                return None
            if filename:
                normalized_item["filename"] = filename
            return normalized_item

        return None

    def _normalize_extra_user_content(
        self,
        extra_user_content: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Copy and lightly validate extra user content items for request assembly."""

        normalized: list[dict[str, Any]] = []
        for item in extra_user_content:
            mapping = self._as_mapping(item)
            if mapping is None:
                raise TypeError("extra_user_content items must be mapping-like objects")
            normalized_item = self._prune_nones(copy.deepcopy(dict(mapping)))
            if not isinstance(normalized_item, MappingABC):
                raise TypeError("extra_user_content items must normalize to mapping-like objects")
            normalized.append(dict(normalized_item))
        return normalized

    def _normalize_image_detail(self, detail: object) -> str | None:
        """Normalize an image detail hint into an allowed API enum value."""

        normalized = self._clean_text(detail).lower()
        if normalized in self._ALLOWED_IMAGE_DETAILS:
            return normalized
        return None

    def _normalize_prompt_cache_retention(self, retention: object) -> str | None:
        """Normalize prompt-cache retention into a supported API value."""

        normalized = self._clean_text(retention).lower()
        if not normalized:
            return None
        if normalized == "24h":
            return "24h"
        if normalized in {"in_memory", "in-memory"}:
            return "in-memory"
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

    def _resolve_include(
        self,
        *,
        explicit_include: Sequence[str] | None,
        model: str,
        store: bool,
        tools: Sequence[Mapping[str, Any]],
    ) -> list[str]:
        """Resolve the include list with sane stateless and web-search defaults."""

        includes: list[str] = []
        seen: set[str] = set()

        def add(value: object) -> None:
            normalized = self._clean_text(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                includes.append(normalized)

        if explicit_include:
            for value in explicit_include:
                add(value)

        configured_include = self._get_config_value("openai_response_include")
        if isinstance(configured_include, str):
            for part in configured_include.split(","):
                add(part)
        elif isinstance(configured_include, SequenceABC) and not isinstance(
            configured_include, (str, bytes, bytearray)
        ):
            for value in configured_include:
                add(value)

        if not store and self._model_supports_reasoning(model):
            for value in self._DEFAULT_STATELESS_INCLUDE:
                add(value)

        if tools and self._normalize_optional_bool(
            self._get_config_value("openai_include_web_search_sources"),
            default=True,
        ):
            if any(self._clean_text(tool.get("type")).lower() == "web_search" for tool in tools):
                add(self._DEFAULT_WEB_SEARCH_SOURCES_INCLUDE)

        return includes

    def _resolve_reasoning_summary(self, value: object) -> str | None:
        """Resolve an explicit or configured reasoning summary mode."""

        configured = value if value is not None else self._get_config_value("openai_reasoning_summary")
        normalized = self._clean_text(configured).lower()
        if normalized in self._ALLOWED_REASONING_SUMMARIES:
            return normalized
        return None

    def _resolve_truncation(self, value: object) -> str | None:
        """Resolve truncation mode from the explicit value or config."""

        configured = value if value is not None else self._get_config_value("openai_truncation")
        normalized = self._clean_text(configured).lower()
        if normalized in self._ALLOWED_TRUNCATIONS:
            return normalized
        return None

    def _resolve_text_verbosity(self, value: object) -> str | None:
        """Resolve text verbosity from the explicit value or config."""

        configured = value if value is not None else self._get_config_value("openai_text_verbosity")
        normalized = self._clean_text(configured).lower()
        if normalized in self._ALLOWED_TEXT_VERBOSITIES:
            return normalized
        return None

    def _resolve_context_management(
        self,
        value: Sequence[Mapping[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """Resolve server-side context management configuration."""

        configured = value if value is not None else self._get_config_value("openai_context_management")
        if configured is None:
            auto_compaction_enabled = self._normalize_optional_bool(
                self._get_config_value("openai_enable_server_side_compaction"),
                default=False,
            )
            compact_threshold = self._get_config_value("openai_compaction_threshold")
            if auto_compaction_enabled and compact_threshold is not None:
                configured = [{"type": "compaction", "compact_threshold": compact_threshold}]
            else:
                return None

        if isinstance(configured, MappingABC):
            configured_items: Sequence[Any] = [configured]
        elif isinstance(configured, SequenceABC) and not isinstance(configured, (str, bytes, bytearray)):
            configured_items = list(configured)
        else:
            raise TypeError("context_management must be a mapping or a sequence of mappings")

        normalized_items: list[dict[str, Any]] = []
        for item in configured_items:
            mapping = self._as_mapping(item)
            if mapping is None:
                raise TypeError("context_management entries must be mapping-like objects")
            item_type = self._clean_text(mapping.get("type")).lower()
            if item_type != "compaction":
                raise ValueError("context_management entries currently only support type='compaction'")
            normalized_item: dict[str, Any] = {"type": "compaction"}
            if "compact_threshold" in mapping and mapping.get("compact_threshold") is not None:
                try:
                    compact_threshold = int(mapping["compact_threshold"])
                except (TypeError, ValueError) as exc:
                    raise ValueError("compact_threshold must be an integer") from exc
                if compact_threshold < 1000:
                    raise ValueError("compact_threshold must be at least 1000")
                normalized_item["compact_threshold"] = compact_threshold
            normalized_items.append(normalized_item)
        return normalized_items or None

    def _normalize_allowed_domains(self, value: object) -> list[str]:
        """Normalize web-search allowed domains to the API format."""

        domains: list[str] = []
        seen: set[str] = set()

        def add(raw_value: object) -> None:
            normalized = self._clean_text(raw_value).lower()
            if not normalized:
                return
            for prefix in ("http://", "https://"):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
                    break
            normalized = normalized.split("/", 1)[0].strip().strip(".")
            if not normalized:
                return
            if ":" in normalized:
                host, port = normalized.rsplit(":", 1)
                if port.isdigit():
                    normalized = host
            if "." not in normalized or any(ch.isspace() for ch in normalized):
                return
            if normalized not in seen:
                seen.add(normalized)
                domains.append(normalized)

        if value is None:
            return domains
        if isinstance(value, str):
            for part in value.split(","):
                add(part)
            return domains[:100]
        if isinstance(value, SequenceABC) and not isinstance(value, (bytes, bytearray)):
            for part in value:
                add(part)
            return domains[:100]
        add(value)
        return domains[:100]

    def _normalize_safety_identifier(self, value: object) -> str | None:
        """Normalize and bound the Responses API safety identifier."""

        normalized = self._clean_text(value)
        if not normalized:
            return None
        if len(normalized) > self._MAX_SAFETY_IDENTIFIER_LENGTH:
            return normalized[: self._MAX_SAFETY_IDENTIFIER_LENGTH]
        return normalized

    def _normalize_metadata(self, value: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """Normalize metadata to the constraints used by the Responses API."""

        if value is None:
            return None
        metadata_mapping = self._as_mapping(value)
        if metadata_mapping is None:
            raise TypeError("metadata must be mapping-like")
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in metadata_mapping.items():
            key = self._clean_text(raw_key)
            if not key:
                continue
            key = key[: self._MAX_METADATA_KEY_LENGTH]
            if isinstance(raw_value, bool):
                normalized_value: Any = raw_value
            elif isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
                normalized_value = raw_value
            else:
                normalized_text = self._clean_text(raw_value)
                if not normalized_text:
                    continue
                normalized_value = normalized_text[: self._MAX_METADATA_VALUE_LENGTH]
            normalized[key] = normalized_value
            if len(normalized) >= self._MAX_METADATA_ITEMS:
                break
        return normalized or None

    def _normalize_optional_bool(self, value: object, *, default: bool | None = None) -> bool | None:
        """Normalize a bool-like config value."""

        if value is None:
            return default
        if isinstance(value, bool):
            return value
        normalized = self._clean_text(value).lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    def _capability_model_name(self, model: object) -> str:
        """Normalize a model identifier for capability checks."""

        normalized = self._clean_text(model).lower()
        if "/" in normalized:
            normalized = normalized.rsplit("/", 1)[-1]
        return normalized

    def _sanitize_passthrough_item(self, item: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """Deep-copy a non-message Responses item for replay."""

        if item is None:
            return None
        sanitized = self._prune_nones(copy.deepcopy(dict(item)))
        if not isinstance(sanitized, MappingABC):
            return None
        item_type = self._clean_text(sanitized.get("type"))
        if not item_type:
            return None
        return dict(sanitized)

    def _prune_nones(self, value: Any) -> Any:
        """Recursively remove ``None`` values while preserving falsey primitives."""

        if isinstance(value, MappingABC):
            return {
                key: self._prune_nones(item)
                for key, item in value.items()
                if item is not None
            }
        if isinstance(value, list):
            return [self._prune_nones(item) for item in value if item is not None]
        return value

    def _as_mapping(self, item: object) -> Mapping[str, Any] | None:
        """Best-effort conversion of mapping-like SDK objects into mappings."""

        if isinstance(item, MappingABC):
            return item
        for attribute_name in ("model_dump", "to_dict", "dict"):
            converter = getattr(item, attribute_name, None)
            if callable(converter):
                try:
                    converted = converter()
                except TypeError:
                    try:
                        converted = converter(exclude_none=False)
                    except Exception:
                        continue
                except Exception:
                    continue
                if isinstance(converted, MappingABC):
                    return converted
        return None

    def _get_config_value(self, name: str, default: Any = None) -> Any:
        """Read an optional config attribute without assuming it exists."""

        return getattr(self.config, name, default)
