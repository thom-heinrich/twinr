# CHANGELOG: 2026-03-30
# BUG-1: Completed Responses API calls that returned only tool/function output were silently mapped to empty text; the mixin now enforces a text/refusal contract for text helpers.
# BUG-2: failed/cancelled/incomplete/background responses could slip through as "successful" replies; the mixin now validates response status and raises contextual errors.
# BUG-3: Sync OpenAI I/O on a running asyncio loop only emitted a warning and could freeze the assistant's async voice stack; the mixin now fails fast by default, with an explicit opt-out.
# BUG-4: Streaming ignored failure/error lifecycle events and lost refusal text in fallback mode; the stream path now handles errors and captures both text and refusals.
# SEC-1: Responses are stored by default for at least 30 days; for senior-care prompts/images this is an avoidable privacy risk. The mixin now defaults to store=False unless explicitly overridden.
# IMP-1: Added request_overrides plus config-driven raw request defaults so new Responses API capabilities can be adopted without changing this mixin every SDK cycle.
# IMP-2: Added server-side compaction, service-tier, prompt-cache-retention, and stream obfuscation hooks aligned with 2026 Responses API guidance.
# IMP-3: Added fine-grained timeout parsing (including HTTPX timeout mappings) and stronger workflow spans for vision and streaming paths.

"""Handle synchronous OpenAI Responses API calls for Twinr backends.

This module keeps the generic text, vision, and streaming response helpers that
sit above the shared request-building logic in ``twinr.providers.openai.core``.
Import these mixins into composed backend classes rather than calling them as
standalone utilities.

The mixin intentionally remains a synchronous HTTP helper. For long-running or
latency-critical voice interactions, 2026 best practice is to use the OpenAI
Realtime API or Responses WebSocket mode in a dedicated async path.
"""

from __future__ import annotations

import asyncio
import io
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from twinr.agent.base_agent import merge_instructions
from twinr.agent.workflows.forensics import workflow_span
from twinr.ops.usage import extract_model_name, extract_token_usage

from ..core.types import ConversationLike, OpenAIImageInput, OpenAITextResponse


class OpenAIResponseError(RuntimeError):
    """Base error raised by the synchronous Responses mixin."""


class OpenAIResponseContractError(OpenAIResponseError):
    """Raised when a text helper receives a non-textual response shape."""


class OpenAIResponseStatusError(OpenAIResponseError):
    """Raised when the OpenAI response finished in a non-success state."""


class OpenAIResponseUsageError(OpenAIResponseError):
    """Raised when the sync mixin is used in an unsafe or unsupported way."""


class OpenAIResponseMixin:
    """Provide text, vision, and streaming response helpers for OpenAI backends.

    The mixin expects the host backend to supply configuration, a synchronous
    OpenAI client, request-building helpers, and response-parsing helpers from
    the adjacent ``core`` package.
    """

    def _config_value(self, *names: str, default: Any = None) -> Any:
        """Return the first non-None config attribute among ``names``."""

        for name in names:
            if hasattr(self.config, name):
                value = getattr(self.config, name)
                if value is not None:
                    return value
        return default

    def _coerce_bool(self, value: Any, *, field_name: str) -> bool:
        """Parse bool-ish config values with strict error messages."""

        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        raise TypeError(f"{field_name} must be a boolean-compatible value, got {value!r}")

    def _coerce_int(self, value: Any, *, field_name: str) -> int:
        """Parse integer config values without silently truncating floats."""

        if isinstance(value, bool):
            raise TypeError(f"{field_name} must be an integer, got boolean {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value.strip())
        raise TypeError(f"{field_name} must be an integer-compatible value, got {value!r}")

    def _coerce_timeout(self, value: Any, *, field_name: str) -> Any:
        """Parse OpenAI/httpx timeout config into an SDK-compatible object."""

        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError(f"{field_name} must be a timeout value, got boolean {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.strip())
        if isinstance(value, Mapping):
            import httpx

            timeout_kwargs: dict[str, float | None] = {}
            default_timeout = value.get("timeout", value.get("default"))
            for key in ("connect", "read", "write", "pool"):
                part = value.get(key)
                if part is not None:
                    timeout_kwargs[key] = float(part)

            if default_timeout is None and not timeout_kwargs:
                raise ValueError(
                    f"{field_name} timeout mapping must define 'timeout'/'default' "
                    f"or at least one of connect/read/write/pool"
                )

            if default_timeout is None:
                default_timeout = next(iter(timeout_kwargs.values()))

            return httpx.Timeout(float(default_timeout), **timeout_kwargs)
        return value

    def _conversation_message_count(self, conversation: ConversationLike | None) -> int:
        """Best-effort count for tracing details without assuming a concrete type."""

        if conversation is None:
            return 0
        if isinstance(conversation, (str, bytes)):
            return 1
        try:
            return len(conversation)  # type: ignore[arg-type]
        except TypeError:
            return 1

    def _response_request_config_overrides(self) -> dict[str, Any]:
        """Read config-level raw request overrides for future API features."""

        overrides = self._config_value("openai_response_request_overrides", default=None)
        if overrides is None:
            return {}
        if not isinstance(overrides, Mapping):
            raise TypeError(
                "config.openai_response_request_overrides must be a mapping, "
                f"got {type(overrides)!r}"
            )
        return dict(overrides)

    def _merge_request_dict(self, base: dict[str, Any], extra: Mapping[str, Any] | None) -> dict[str, Any]:
        """Shallow-merge request overrides with targeted nested-map support."""

        if extra is None:
            return base

        merged = dict(base)
        for key, value in extra.items():
            if (
                key in {"metadata", "prompt", "reasoning", "stream_options", "text"}
                and isinstance(merged.get(key), Mapping)
                and isinstance(value, Mapping)
            ):
                merged[key] = {**dict(merged[key]), **dict(value)}
            else:
                merged[key] = value
        return merged

    def _default_store_responses(self) -> bool:
        """Default to non-persistent responses unless explicitly configured otherwise."""

        raw = self._config_value("openai_store_responses", default=False)
        return self._coerce_bool(raw, field_name="openai_store_responses")

    def _allow_sync_on_running_loop(self) -> bool:
        """Whether callers explicitly accept blocking the current event loop."""

        raw = self._config_value("openai_allow_sync_on_running_loop", default=False)
        return self._coerce_bool(raw, field_name="openai_allow_sync_on_running_loop")

    def _allow_incomplete_responses(self) -> bool:
        """Whether partial/incomplete responses should be returned instead of raised."""

        raw = self._config_value("openai_allow_incomplete_responses", default=False)
        return self._coerce_bool(raw, field_name="openai_allow_incomplete_responses")

    def _allow_tool_only_responses(self) -> bool:
        """Whether a text helper may accept a response with no assistant text/refusal."""

        raw = self._config_value("openai_allow_tool_only_responses", default=False)
        return self._coerce_bool(raw, field_name="openai_allow_tool_only_responses")

    def _context_management_default(self) -> list[dict[str, Any]] | None:
        """Return optional server-side compaction config from the backend config."""

        raw = self._config_value("openai_context_compact_threshold", default=None)
        if raw is None:
            return None

        threshold = self._coerce_int(raw, field_name="openai_context_compact_threshold")
        if threshold <= 0:
            raise ValueError("openai_context_compact_threshold must be > 0")
        return [{"type": "compaction", "compact_threshold": threshold}]

    def _service_tier_default(self) -> str | None:
        """Return an optional service tier default for latency/cost routing."""

        raw = self._config_value("openai_service_tier", default=None)
        if raw is None:
            return None
        if not isinstance(raw, str):
            raise TypeError(f"openai_service_tier must be a string, got {type(raw)!r}")
        normalized = raw.strip()
        if not normalized:
            raise ValueError("openai_service_tier cannot be empty")
        return normalized

    def _prompt_cache_retention_default(self) -> str | None:
        """Return an optional prompt-cache-retention policy."""

        raw = self._config_value("openai_prompt_cache_retention", default=None)
        if raw is None:
            return None
        if not isinstance(raw, str):
            raise TypeError(
                f"openai_prompt_cache_retention must be a string, got {type(raw)!r}"
            )
        normalized = raw.strip()
        if not normalized:
            raise ValueError("openai_prompt_cache_retention cannot be empty")
        return normalized

    def _stream_include_obfuscation_default(self) -> bool | None:
        """Return an optional stream obfuscation override for trusted links."""

        raw = self._config_value("openai_stream_include_obfuscation", default=None)
        if raw is None:
            return None
        return self._coerce_bool(raw, field_name="openai_stream_include_obfuscation")

    def _timeout_default(self) -> Any:
        """Return a request timeout from config, supporting granular HTTPX mappings."""

        raw = self._config_value("openai_timeout", "openai_timeout_seconds", default=None)
        return self._coerce_timeout(raw, field_name="openai_timeout")

    def _max_retries_default(self) -> int | None:
        """Return optional retry configuration."""

        raw = self._config_value("openai_max_retries", default=None)
        if raw is None:
            return None
        retries = self._coerce_int(raw, field_name="openai_max_retries")
        if retries < 0:
            raise ValueError("openai_max_retries must be >= 0")
        return retries

    def _ensure_sync_request_contract(self, request: dict[str, Any], *, operation: str, for_stream: bool) -> None:
        """Reject request shapes this synchronous mixin cannot safely fulfill."""

        background = request.get("background", False)
        if background is not None and self._coerce_bool(background, field_name="background"):
            raise OpenAIResponseUsageError(
                f"{self.__class__.__name__}.{operation} does not support background=True. "
                "Use an async poll/webhook path or a dedicated background-response helper instead."
            )

        if not for_stream:
            stream = request.get("stream", False)
            if stream is not None and self._coerce_bool(stream, field_name="stream"):
                raise OpenAIResponseUsageError(
                    f"{self.__class__.__name__}.{operation} received stream=True on a non-streaming helper. "
                    "Use respond_streaming() instead."
                )

    def _prepare_request(
        self,
        request: dict[str, Any],
        *,
        operation: str,
        for_stream: bool,
        request_overrides: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Apply config defaults and per-call overrides to a Responses request."""

        prepared = dict(request)
        prepared = self._merge_request_dict(prepared, self._response_request_config_overrides())

        if "store" not in prepared:
            # BREAKING: default storage is now opt-in to avoid retaining senior-care prompts/images.
            prepared["store"] = self._default_store_responses()

        service_tier = self._service_tier_default()
        if service_tier is not None and "service_tier" not in prepared:
            prepared["service_tier"] = service_tier

        prompt_cache_retention = self._prompt_cache_retention_default()
        if prompt_cache_retention is not None and "prompt_cache_retention" not in prepared:
            prepared["prompt_cache_retention"] = prompt_cache_retention

        context_management = self._context_management_default()
        if context_management is not None and "context_management" not in prepared:
            prepared["context_management"] = context_management

        if for_stream:
            include_obfuscation = self._stream_include_obfuscation_default()
            if include_obfuscation is not None:
                stream_options = prepared.get("stream_options")
                if stream_options is None:
                    stream_options = {}
                if not isinstance(stream_options, Mapping):
                    raise TypeError(
                        f"stream_options must be a mapping, got {type(stream_options)!r}"
                    )
                merged_stream_options = dict(stream_options)
                merged_stream_options.setdefault("include_obfuscation", include_obfuscation)
                prepared["stream_options"] = merged_stream_options
        else:
            prepared.pop("stream", None)
            prepared.pop("stream_options", None)

        prepared = self._merge_request_dict(prepared, request_overrides)
        self._ensure_sync_request_contract(prepared, operation=operation, for_stream=for_stream)

        if for_stream:
            prepared.pop("stream", None)
        return prepared

    def _build_text_request(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        prompt_cache_scope: str,
        extra_user_content: Any = None,
        request_overrides: Mapping[str, Any] | None = None,
        for_stream: bool = False,
    ) -> dict[str, Any]:
        """Build a normalized Responses API request for text or vision prompts.

        Args:
            prompt: User-visible prompt text.
            conversation: Optional recent conversation context.
            instructions: Optional extra instruction block to merge.
            allow_web_search: Override for web search enablement.
            prompt_cache_scope: Cache namespace used by the request builder.
            extra_user_content: Optional extra user content such as images.
            request_overrides: Optional raw Responses API fields to merge into the request.
            for_stream: Whether the request is being prepared for streaming.

        Returns:
            A mutable mapping ready to pass to the OpenAI Responses client.

        Raises:
            TypeError: If the underlying request builder does not return a mapping.
            Exception: Propagates request-builder failures with added context.
        """

        try:
            merged_instructions = merge_instructions(self._resolve_base_instructions(), instructions)
            request = self._build_response_request(
                prompt,
                conversation=conversation,
                instructions=merged_instructions,
                allow_web_search=allow_web_search,
                model=self.config.default_model,
                reasoning_effort=self.config.openai_reasoning_effort,
                prompt_cache_scope=prompt_cache_scope,
                extra_user_content=extra_user_content,
            )
        except Exception as exc:
            self._add_exception_note(exc, operation="_build_text_request")
            raise

        if not isinstance(request, Mapping):
            raise TypeError(f"_build_response_request() must return a mapping, got {type(request)!r}")

        prepared_request = self._prepare_request(
            dict(request),
            operation="_build_text_request",
            for_stream=for_stream,
            request_overrides=request_overrides,
        )
        return prepared_request

    def _responses_client(self, *, operation: str) -> Any:
        """Return a client view with configured timeout and retry overrides."""

        self._warn_if_running_event_loop(operation=operation)

        client = self._client
        with_options = getattr(client, "with_options", None)
        if not callable(with_options):
            return client

        request_options: dict[str, Any] = {}
        timeout = self._timeout_default()
        if timeout is not None:
            request_options["timeout"] = timeout

        max_retries = self._max_retries_default()
        if max_retries is not None:
            request_options["max_retries"] = max_retries

        if not request_options:
            return client

        try:
            return with_options(**request_options)
        except Exception as exc:
            self._add_exception_note(exc, operation=f"{operation}.with_options")
            raise

    def _warn_if_running_event_loop(self, *, operation: str) -> None:
        """Warn or fail when synchronous provider I/O would block an active event loop."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return

        if self._allow_sync_on_running_loop():
            warnings.warn(
                (
                    f"{self.__class__.__name__}.{operation} is performing synchronous OpenAI I/O "
                    "on a running event loop; this can block the entire async process. "
                    "Use AsyncOpenAI or move the call behind an explicit threadpool boundary."
                ),
                RuntimeWarning,
                stacklevel=3,
            )
            return

        # BREAKING: fail fast by default instead of merely warning, because blocking the loop
        # can freeze the entire assistant process (wake-word, TTS, timers, watchdogs).
        raise OpenAIResponseUsageError(
            f"{self.__class__.__name__}.{operation} attempted synchronous OpenAI I/O on a running "
            "asyncio event loop. Configure openai_allow_sync_on_running_loop=true to opt back into "
            "the old warning-only behavior, or use the async backend."
        )

    def _add_exception_note(self, exc: BaseException, *, operation: str) -> None:
        """Attach operation context to an exception when Python supports notes."""

        add_note = getattr(exc, "add_note", None)
        if callable(add_note):
            add_note(f"{self.__class__.__name__}.{operation} failed")

    def _create_response(self, request: dict[str, Any], *, operation: str) -> Any:
        """Create a non-streaming Responses API response with contextual errors."""

        try:
            response = self._responses_client(operation=operation).responses.create(**request)
        except Exception as exc:
            self._add_exception_note(exc, operation=operation)
            raise

        self._raise_for_response_status(response, operation=operation)
        return response

    def _open_response_stream(self, request: dict[str, Any], *, operation: str) -> Any:
        """Open a streaming Responses API handle for the given request."""

        return self._responses_client(operation=operation).responses.stream(**request)

    def _response_error_message(self, error: Any) -> str:
        """Extract a stable human-readable error message from an SDK error payload."""

        if error is None:
            return "unknown response error"

        if isinstance(error, Mapping):
            code = error.get("code")
            message = error.get("message")
        else:
            code = getattr(error, "code", None)
            message = getattr(error, "message", None)

        if code and message:
            return f"{code}: {message}"
        if message:
            return str(message)
        if code:
            return str(code)
        return repr(error)

    def _incomplete_reason(self, incomplete_details: Any) -> str:
        """Return the most useful incomplete reason available."""

        if incomplete_details is None:
            return "unknown"
        if isinstance(incomplete_details, Mapping):
            reason = incomplete_details.get("reason")
        else:
            reason = getattr(incomplete_details, "reason", None)
        return str(reason) if reason else repr(incomplete_details)

    def _response_output_summary(self, response: Any) -> str:
        """Summarize output item types/statuses for debugging and exception messages."""

        summary_parts: list[str] = []
        for item in getattr(response, "output", ()) or ():
            item_type = getattr(item, "type", None) or type(item).__name__
            item_status = getattr(item, "status", None)
            if item_status:
                summary_parts.append(f"{item_type}:{item_status}")
            else:
                summary_parts.append(str(item_type))
        return ", ".join(summary_parts) if summary_parts else "<no output items>"

    def _response_has_message_text_or_refusal(self, response: Any) -> bool:
        """Whether the response contains an assistant message with text or refusal content."""

        for item in getattr(response, "output", ()) or ():
            if getattr(item, "type", None) != "message":
                continue
            for part in getattr(item, "content", ()) or ():
                if getattr(part, "type", None) in {"output_text", "refusal"}:
                    return True
        return False

    def _extract_text_or_refusal_from_output_items(self, response: Any) -> str:
        """Read assistant text/refusal parts directly from response.output when needed."""

        parts: list[str] = []
        for item in getattr(response, "output", ()) or ():
            if getattr(item, "type", None) != "message":
                continue
            for part in getattr(item, "content", ()) or ():
                part_type = getattr(part, "type", None)
                if part_type == "output_text":
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text:
                        parts.append(text)
                elif part_type == "refusal":
                    refusal = getattr(part, "refusal", None)
                    if isinstance(refusal, str) and refusal:
                        parts.append(refusal)
        return "".join(parts)

    def _raise_for_response_status(
        self,
        response: Any,
        *,
        operation: str,
        fallback_text: str = "",
    ) -> None:
        """Validate final response state before Twinr treats it as successful."""

        error = getattr(response, "error", None)
        if error is not None:
            raise OpenAIResponseStatusError(
                f"{self.__class__.__name__}.{operation} received a response error: "
                f"{self._response_error_message(error)}"
            )

        status = getattr(response, "status", None)
        if status in {"failed", "cancelled", "queued", "in_progress"}:
            raise OpenAIResponseStatusError(
                f"{self.__class__.__name__}.{operation} expected a completed response but got "
                f"status={status!r}. output={self._response_output_summary(response)}"
            )

        incomplete_details = getattr(response, "incomplete_details", None)
        if status == "incomplete" or incomplete_details is not None:
            if not self._allow_incomplete_responses():
                raise OpenAIResponseStatusError(
                    f"{self.__class__.__name__}.{operation} received an incomplete response "
                    f"(reason={self._incomplete_reason(incomplete_details)}; "
                    f"partial_text_len={len(fallback_text)})."
                )

    def _extract_output_text_with_fallback(self, response: Any, *, fallback_text: str = "") -> str:
        """Extract final response text and fall back to refusal/items/stream deltas if needed."""

        try:
            text = self._extract_output_text(response)
        except Exception as exc:
            item_text = self._extract_text_or_refusal_from_output_items(response)
            if item_text:
                warnings.warn(
                    (
                        f"Falling back to response.output item parsing in "
                        f"{self.__class__.__name__}: {exc!r}"
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                return item_text
            if fallback_text:
                warnings.warn(
                    (
                        f"Falling back to accumulated streamed text in "
                        f"{self.__class__.__name__}: {exc!r}"
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                return fallback_text
            self._add_exception_note(exc, operation="_extract_output_text_with_fallback")
            raise

        if isinstance(text, str) and text:
            return text

        item_text = self._extract_text_or_refusal_from_output_items(response)
        if item_text:
            return item_text

        if text is None:
            return fallback_text
        if isinstance(text, str):
            return text or fallback_text
        return str(text)

    def _ensure_text_response_contract(self, response: Any, text: str, *, operation: str) -> None:
        """Enforce that the text helper actually produced assistant text/refusal."""

        if text:
            return
        if self._response_has_message_text_or_refusal(response):
            return
        if self._allow_tool_only_responses():
            return

        raise OpenAIResponseContractError(
            f"{self.__class__.__name__}.{operation} expected assistant text/refusal, but the "
            f"response only contained non-text output items: {self._response_output_summary(response)}"
        )

    def _build_text_response(
        self,
        response: Any,
        request: dict[str, Any],
        *,
        fallback_text: str = "",
        operation: str = "_build_text_response",
    ) -> OpenAITextResponse:
        """Map an OpenAI response object into Twinr's text-response contract."""

        try:
            self._raise_for_response_status(response, operation=operation, fallback_text=fallback_text)
            text = self._extract_output_text_with_fallback(response, fallback_text=fallback_text)
            self._ensure_text_response_contract(response, text, operation=operation)
            requested_model = request.get("model", self.config.default_model)
            return OpenAITextResponse(
                text=text,
                response_id=getattr(response, "id", None),
                request_id=getattr(response, "_request_id", None),
                model=extract_model_name(response, str(requested_model)),
                token_usage=extract_token_usage(response),
                used_web_search=self._used_web_search(response),
            )
        except Exception as exc:
            self._add_exception_note(exc, operation=operation)
            raise

    def respond(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        request_overrides: Mapping[str, Any] | None = None,
    ) -> str:
        """Return plain response text for a standard OpenAI prompt.

        Args:
            prompt: User prompt to answer.
            conversation: Optional recent conversation context.
            instructions: Optional extra system-style instructions.
            allow_web_search: Override for live web search usage.
            request_overrides: Optional raw Responses API fields such as
                ``previous_response_id``, ``conversation``, ``text``, or
                ``context_management``.

        Returns:
            The assistant text extracted from ``respond_with_metadata()``.
        """

        return self.respond_with_metadata(
            prompt,
            conversation=conversation,
            instructions=instructions,
            allow_web_search=allow_web_search,
            request_overrides=request_overrides,
        ).text

    def respond_with_metadata(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        request_overrides: Mapping[str, Any] | None = None,
    ) -> OpenAITextResponse:
        """Return a text response plus OpenAI metadata for a prompt.

        Args:
            prompt: User prompt to answer.
            conversation: Optional recent conversation context.
            instructions: Optional extra system-style instructions.
            allow_web_search: Override for live web search usage.
            request_overrides: Optional raw Responses API fields.

        Returns:
            An ``OpenAITextResponse`` containing text, IDs, model, and usage.
        """

        with workflow_span(
            name="openai_response_build_text_request",
            kind="llm_call",
            details={
                "conversation_messages": self._conversation_message_count(conversation),
                "allow_web_search": allow_web_search,
            },
        ):
            request = self._build_text_request(
                prompt,
                conversation=conversation,
                instructions=instructions,
                allow_web_search=allow_web_search,
                prompt_cache_scope="response",
                request_overrides=request_overrides,
            )
        with workflow_span(
            name="openai_response_create",
            kind="llm_call",
            details={
                "model": request.get("model"),
                "allow_web_search": allow_web_search,
                "store": request.get("store"),
                "service_tier": request.get("service_tier"),
            },
        ):
            response = self._create_response(request, operation="respond_with_metadata")
        with workflow_span(
            name="openai_response_build_text_response",
            kind="llm_call",
            details={"model": request.get("model")},
        ):
            return self._build_text_response(
                response,
                request,
                operation="respond_with_metadata",
            )

    def respond_to_images(
        self,
        prompt: str,
        *,
        images: Sequence[OpenAIImageInput],
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        request_overrides: Mapping[str, Any] | None = None,
    ) -> str:
        """Return plain response text for a multimodal prompt with images.

        Args:
            prompt: User prompt to answer.
            images: One or more image inputs to send alongside the prompt.
            conversation: Optional recent conversation context.
            instructions: Optional extra system-style instructions.
            allow_web_search: Override for live web search usage.
            request_overrides: Optional raw Responses API fields.

        Returns:
            The assistant text extracted from ``respond_to_images_with_metadata()``.
        """

        return self.respond_to_images_with_metadata(
            prompt,
            images=images,
            conversation=conversation,
            instructions=instructions,
            allow_web_search=allow_web_search,
            request_overrides=request_overrides,
        ).text

    def respond_to_images_with_metadata(
        self,
        prompt: str,
        *,
        images: Sequence[OpenAIImageInput],
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        request_overrides: Mapping[str, Any] | None = None,
    ) -> OpenAITextResponse:
        """Return a metadata-rich multimodal response for a prompt and images.

        Args:
            prompt: User prompt to answer.
            images: One or more image inputs to send alongside the prompt.
            conversation: Optional recent conversation context.
            instructions: Optional extra system-style instructions.
            allow_web_search: Override for live web search usage.
            request_overrides: Optional raw Responses API fields.

        Returns:
            An ``OpenAITextResponse`` for the multimodal request.

        Raises:
            ValueError: If no images are provided.
            Exception: Propagates image-content or request execution failures.
        """

        if not images:
            raise ValueError("At least one image is required for a vision request")

        with workflow_span(
            name="openai_response_build_image_content",
            kind="llm_call",
            details={"image_count": len(images)},
        ):
            try:
                image_content = self._build_image_content(images)
            except Exception as exc:
                self._add_exception_note(exc, operation="respond_to_images_with_metadata")
                raise

        with workflow_span(
            name="openai_response_build_vision_request",
            kind="llm_call",
            details={
                "conversation_messages": self._conversation_message_count(conversation),
                "allow_web_search": allow_web_search,
                "image_count": len(images),
            },
        ):
            request = self._build_text_request(
                prompt,
                conversation=conversation,
                instructions=instructions,
                allow_web_search=allow_web_search,
                extra_user_content=image_content,
                prompt_cache_scope="vision_response",
                request_overrides=request_overrides,
            )

        with workflow_span(
            name="openai_response_create_vision",
            kind="llm_call",
            details={
                "model": request.get("model"),
                "store": request.get("store"),
                "service_tier": request.get("service_tier"),
            },
        ):
            response = self._create_response(
                request,
                operation="respond_to_images_with_metadata",
            )
        return self._build_text_response(
            response,
            request,
            operation="respond_to_images_with_metadata",
        )

    def _stream_error_message(self, event: Any) -> str:
        """Extract a readable streaming error message from an SSE event."""

        event_error = getattr(event, "error", None)
        if event_error is not None:
            return self._response_error_message(event_error)

        message = getattr(event, "message", None)
        if isinstance(message, str) and message:
            return message

        data = getattr(event, "data", None)
        if data is not None:
            return repr(data)
        return repr(event)

    def _extract_stream_delta_text(self, event: Any) -> str:
        """Extract text-like deltas from output-text and refusal stream events."""

        event_type = getattr(event, "type", None)
        if not isinstance(event_type, str) or not event_type.endswith(".delta"):
            return ""

        raw_delta = None
        if event_type == "response.output_text.delta":
            raw_delta = getattr(event, "delta", None)
        elif "refusal" in event_type:
            raw_delta = getattr(event, "delta", None)
            if raw_delta is None:
                raw_delta = getattr(event, "refusal", None)

        if raw_delta is None:
            return ""
        if isinstance(raw_delta, str):
            return raw_delta
        return str(raw_delta)

    def _handle_stream_event(
        self,
        *,
        event: Any,
        on_text_delta: Callable[[str], None] | None,
        fallback_buffer: io.StringIO,
        callback_warning_emitted: list[bool],
    ) -> None:
        """Process one streamed event, forwarding safe text/refusal deltas."""

        event_type = getattr(event, "type", None)
        if event_type in {"error", "response.failed"}:
            raise OpenAIResponseStatusError(
                f"{self.__class__.__name__}.respond_streaming received a stream error: "
                f"{self._stream_error_message(event)}"
            )

        delta_text = self._extract_stream_delta_text(event)
        if not delta_text:
            return

        fallback_buffer.write(delta_text)

        if on_text_delta is None:
            return

        try:
            on_text_delta(delta_text)
        except Exception as exc:
            if not callback_warning_emitted[0]:
                warnings.warn(
                    (
                        f"Non-fatal on_text_delta callback failure in "
                        f"{self.__class__.__name__}.respond_streaming: {exc!r}"
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                callback_warning_emitted[0] = True

    def respond_streaming(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
        request_overrides: Mapping[str, Any] | None = None,
    ) -> OpenAITextResponse:
        """Stream a response while returning the final Twinr text-response object.

        Args:
            prompt: User prompt to answer.
            conversation: Optional recent conversation context.
            instructions: Optional extra system-style instructions.
            allow_web_search: Override for live web search usage.
            on_text_delta: Optional callback invoked for each streamed text delta.
            request_overrides: Optional raw Responses API fields.

        Returns:
            The final ``OpenAITextResponse`` assembled from the completed stream.

        Raises:
            Exception: Propagates stream setup, iteration, or finalization errors.
        """

        with workflow_span(
            name="openai_response_build_stream_request",
            kind="llm_call",
            details={
                "conversation_messages": self._conversation_message_count(conversation),
                "allow_web_search": allow_web_search,
            },
        ):
            request = self._build_text_request(
                prompt,
                conversation=conversation,
                instructions=instructions,
                allow_web_search=allow_web_search,
                prompt_cache_scope="response_stream",
                request_overrides=request_overrides,
                for_stream=True,
            )

        fallback_buffer = io.StringIO()
        callback_warning_emitted = [False]
        try:
            with workflow_span(
                name="openai_response_stream",
                kind="llm_call",
                details={
                    "model": request.get("model"),
                    "store": request.get("store"),
                    "service_tier": request.get("service_tier"),
                },
            ):
                with self._open_response_stream(request, operation="respond_streaming") as stream:
                    for event in stream:
                        self._handle_stream_event(
                            event=event,
                            on_text_delta=on_text_delta,
                            fallback_buffer=fallback_buffer,
                            callback_warning_emitted=callback_warning_emitted,
                        )
                    response = stream.get_final_response()
        except Exception as exc:
            self._add_exception_note(exc, operation="respond_streaming")
            raise

        with workflow_span(
            name="openai_response_build_stream_response",
            kind="llm_call",
            details={"model": request.get("model")},
        ):
            return self._build_text_response(
                response,
                request,
                fallback_text=fallback_buffer.getvalue(),
                operation="respond_streaming",
            )