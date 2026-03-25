"""Handle synchronous OpenAI Responses API calls for Twinr backends.

This module keeps the generic text, vision, and streaming response helpers that
sit above the shared request-building logic in ``twinr.providers.openai.core``.
Import these mixins into composed backend classes rather than calling them as
standalone utilities.
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from twinr.agent.base_agent import merge_instructions
from twinr.agent.workflows.forensics import workflow_span
from twinr.ops.usage import extract_model_name, extract_token_usage

from ..core.types import ConversationLike, OpenAIImageInput, OpenAITextResponse


class OpenAIResponseMixin:
    """Provide text, vision, and streaming response helpers for OpenAI backends.

    The mixin expects the host backend to supply configuration, a synchronous
    OpenAI client, request-building helpers, and response-parsing helpers from
    the adjacent ``core`` package.
    """

    # AUDIT-FIX(#6): Zentralisiert Request-Bau, Fehlerkontext und Response-Hydration,
    # um Drift zwischen Text-/Vision-/Streaming-Pfaden zu vermeiden.
    def _build_text_request(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        prompt_cache_scope: str,
        extra_user_content: Any = None,
    ) -> dict[str, Any]:
        """Build a normalized Responses API request for text or vision prompts.

        Args:
            prompt: User-visible prompt text.
            conversation: Optional recent conversation context.
            instructions: Optional extra instruction block to merge.
            allow_web_search: Override for web search enablement.
            prompt_cache_scope: Cache namespace used by the request builder.
            extra_user_content: Optional extra user content such as images.

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
        return dict(request)

    # AUDIT-FIX(#4): Unterstützt optionale per-request Timeout-/Retry-Overrides,
    # ohne das bestehende .env-Schema zu brechen.
    def _responses_client(self, *, operation: str) -> Any:
        """Return a client view with configured timeout and retry overrides."""

        self._warn_if_running_event_loop(operation=operation)

        client = self._client
        with_options = getattr(client, "with_options", None)
        if not callable(with_options):
            return client

        request_options: dict[str, Any] = {}
        timeout = getattr(self.config, "openai_timeout_seconds", None)
        if timeout is not None:
            if isinstance(timeout, str):
                timeout = float(timeout)
            request_options["timeout"] = timeout

        max_retries = getattr(self.config, "openai_max_retries", None)
        if max_retries is not None:
            if isinstance(max_retries, str):
                max_retries = int(max_retries)
            request_options["max_retries"] = max_retries

        if not request_options:
            return client

        try:
            return with_options(**request_options)
        except Exception as exc:
            self._add_exception_note(exc, operation=f"{operation}.with_options")
            raise

    # AUDIT-FIX(#1): Macht das Architekturproblem sichtbar, wenn dieser synchrone Mixin
    # versehentlich auf einem laufenden Event-Loop verwendet wird.
    def _warn_if_running_event_loop(self, *, operation: str) -> None:
        """Warn when synchronous provider I/O would block an active event loop."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return

        warnings.warn(
            (
                f"{self.__class__.__name__}.{operation} is performing synchronous OpenAI I/O "
                "on a running event loop; this can block the entire async process. "
                "Use AsyncOpenAI or move the call behind an explicit threadpool boundary."
            ),
            RuntimeWarning,
            stacklevel=3,
        )

    # AUDIT-FIX(#6): Nutzt Python-3.11-Exception-Notes, damit Fehler in Logs/Tracebacks
    # den konkreten Mixin-Operationnamen enthalten.
    def _add_exception_note(self, exc: BaseException, *, operation: str) -> None:
        """Attach operation context to an exception when Python supports notes."""

        add_note = getattr(exc, "add_note", None)
        if callable(add_note):
            add_note(f"{self.__class__.__name__}.{operation} failed")

    # AUDIT-FIX(#6): Kapselt SDK-Aufrufe, damit alle Exceptions denselben Kontext tragen.
    def _create_response(self, request: dict[str, Any], *, operation: str) -> Any:
        """Create a non-streaming Responses API response with contextual errors."""

        try:
            return self._responses_client(operation=operation).responses.create(**request)
        except Exception as exc:
            self._add_exception_note(exc, operation=operation)
            raise

    # AUDIT-FIX(#6): Zentralisiert den Stream-Handle-Aufbau; die vollständige
    # Fehleranreicherung erfolgt im aufrufenden Kontextmanager-Pfad.
    def _open_response_stream(self, request: dict[str, Any], *, operation: str) -> Any:
        """Open a streaming Responses API handle for the given request."""

        return self._responses_client(operation=operation).responses.stream(**request)

    # AUDIT-FIX(#5): Normalisiert extrahierten Text und verhindert Artefakte wie "None".
    # AUDIT-FIX(#2): Nutzt gestreamte Deltas als Fallback, falls Final-Response-Extraktion fehlschlägt.
    def _extract_output_text_with_fallback(self, response: Any, *, fallback_text: str = "") -> str:
        """Extract final response text and fall back to streamed deltas if needed."""

        try:
            text = self._extract_output_text(response)
        except Exception as exc:
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

        if text is None:
            return fallback_text
        if isinstance(text, str):
            return text or fallback_text
        return str(text)

    # AUDIT-FIX(#6): Zentralisiert Response-Mapping, damit alle Pfade identische Metadaten liefern.
    def _build_text_response(
        self,
        response: Any,
        request: dict[str, Any],
        *,
        fallback_text: str = "",
    ) -> OpenAITextResponse:
        """Map an OpenAI response object into Twinr's text-response contract."""

        try:
            requested_model = request.get("model", self.config.default_model)
            return OpenAITextResponse(
                text=self._extract_output_text_with_fallback(response, fallback_text=fallback_text),
                response_id=getattr(response, "id", None),
                request_id=getattr(response, "_request_id", None),
                model=extract_model_name(response, str(requested_model)),
                token_usage=extract_token_usage(response),
                used_web_search=self._used_web_search(response),
            )
        except Exception as exc:
            self._add_exception_note(exc, operation="_build_text_response")
            raise

    def respond(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> str:
        """Return plain response text for a standard OpenAI prompt.

        Args:
            prompt: User prompt to answer.
            conversation: Optional recent conversation context.
            instructions: Optional extra system-style instructions.
            allow_web_search: Override for live web search usage.

        Returns:
            The assistant text extracted from ``respond_with_metadata()``.
        """

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
        """Return a text response plus OpenAI metadata for a prompt.

        Args:
            prompt: User prompt to answer.
            conversation: Optional recent conversation context.
            instructions: Optional extra system-style instructions.
            allow_web_search: Override for live web search usage.

        Returns:
            An ``OpenAITextResponse`` containing text, IDs, model, and usage.
        """

        with workflow_span(
            name="openai_response_build_text_request",
            kind="llm_call",
            details={
                "conversation_messages": len(conversation or ()),
                "allow_web_search": allow_web_search,
            },
        ):
            request = self._build_text_request(
                prompt,
                conversation=conversation,
                instructions=instructions,
                allow_web_search=allow_web_search,
                prompt_cache_scope="response",
            )
        with workflow_span(
            name="openai_response_create",
            kind="llm_call",
            details={
                "model": request.get("model"),
                "allow_web_search": allow_web_search,
            },
        ):
            response = self._create_response(request, operation="respond_with_metadata")
        with workflow_span(
            name="openai_response_build_text_response",
            kind="llm_call",
            details={"model": request.get("model")},
        ):
            return self._build_text_response(response, request)

    def respond_to_images(
        self,
        prompt: str,
        *,
        images: Sequence[OpenAIImageInput],
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> str:
        """Return plain response text for a multimodal prompt with images.

        Args:
            prompt: User prompt to answer.
            images: One or more image inputs to send alongside the prompt.
            conversation: Optional recent conversation context.
            instructions: Optional extra system-style instructions.
            allow_web_search: Override for live web search usage.

        Returns:
            The assistant text extracted from ``respond_to_images_with_metadata()``.
        """

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
        """Return a metadata-rich multimodal response for a prompt and images.

        Args:
            prompt: User prompt to answer.
            images: One or more image inputs to send alongside the prompt.
            conversation: Optional recent conversation context.
            instructions: Optional extra system-style instructions.
            allow_web_search: Override for live web search usage.

        Returns:
            An ``OpenAITextResponse`` for the multimodal request.

        Raises:
            ValueError: If no images are provided.
            Exception: Propagates image-content or request execution failures.
        """

        if not images:
            raise ValueError("At least one image is required for a vision request")

        try:
            image_content = self._build_image_content(images)
        except Exception as exc:
            self._add_exception_note(exc, operation="respond_to_images_with_metadata")
            raise

        request = self._build_text_request(
            prompt,
            conversation=conversation,
            instructions=instructions,
            allow_web_search=allow_web_search,
            extra_user_content=image_content,
            prompt_cache_scope="vision_response",
        )
        response = self._create_response(request, operation="respond_to_images_with_metadata")
        return self._build_text_response(response, request)

    # AUDIT-FIX(#3): Kapselt Streaming-Callbacks, damit UI-/TTS-Fehler die Modellantwort
    # nicht verwerfen und nur einmal als non-fatal Warnung auftauchen.
    def _emit_text_delta(
        self,
        *,
        event: Any,
        on_text_delta: Callable[[str], None] | None,
        streamed_text_parts: list[str],
        callback_warning_emitted: list[bool],
    ) -> None:
        """Forward one streaming text delta to state and optional callback."""

        if getattr(event, "type", None) != "response.output_text.delta":
            return

        raw_delta = getattr(event, "delta", "")
        if not isinstance(raw_delta, str) or not raw_delta:
            return

        streamed_text_parts.append(raw_delta)

        if on_text_delta is None:
            return

        try:
            on_text_delta(raw_delta)
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
    ) -> OpenAITextResponse:
        """Stream a response while returning the final Twinr text-response object.

        Args:
            prompt: User prompt to answer.
            conversation: Optional recent conversation context.
            instructions: Optional extra system-style instructions.
            allow_web_search: Override for live web search usage.
            on_text_delta: Optional callback invoked for each streamed text delta.

        Returns:
            The final ``OpenAITextResponse`` assembled from the completed stream.

        Raises:
            Exception: Propagates stream setup, iteration, or finalization errors.
        """

        request = self._build_text_request(
            prompt,
            conversation=conversation,
            instructions=instructions,
            allow_web_search=allow_web_search,
            prompt_cache_scope="response_stream",
        )

        streamed_text_parts: list[str] = []
        callback_warning_emitted = [False]
        try:
            with self._open_response_stream(request, operation="respond_streaming") as stream:
                for event in stream:
                    self._emit_text_delta(
                        event=event,
                        on_text_delta=on_text_delta,
                        streamed_text_parts=streamed_text_parts,
                        callback_warning_emitted=callback_warning_emitted,
                    )
                response = stream.get_final_response()
        except Exception as exc:
            self._add_exception_note(exc, operation="respond_streaming")
            raise

        return self._build_text_response(response, request, fallback_text="".join(streamed_text_parts))
