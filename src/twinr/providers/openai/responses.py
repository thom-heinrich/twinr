from __future__ import annotations

import asyncio
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from twinr.agent.base_agent.personality import merge_instructions
from twinr.ops.usage import extract_model_name, extract_token_usage

from .types import ConversationLike, OpenAIImageInput, OpenAITextResponse


class OpenAIResponseMixin:
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
        add_note = getattr(exc, "add_note", None)
        if callable(add_note):
            add_note(f"{self.__class__.__name__}.{operation} failed")

    # AUDIT-FIX(#6): Kapselt SDK-Aufrufe, damit alle Exceptions denselben Kontext tragen.
    def _create_response(self, request: dict[str, Any], *, operation: str) -> Any:
        try:
            return self._responses_client(operation=operation).responses.create(**request)
        except Exception as exc:
            self._add_exception_note(exc, operation=operation)
            raise

    # AUDIT-FIX(#6): Zentralisiert den Stream-Handle-Aufbau; die vollständige
    # Fehleranreicherung erfolgt im aufrufenden Kontextmanager-Pfad.
    def _open_response_stream(self, request: dict[str, Any], *, operation: str) -> Any:
        return self._responses_client(operation=operation).responses.stream(**request)

    # AUDIT-FIX(#5): Normalisiert extrahierten Text und verhindert Artefakte wie "None".
    # AUDIT-FIX(#2): Nutzt gestreamte Deltas als Fallback, falls Final-Response-Extraktion fehlschlägt.
    def _extract_output_text_with_fallback(self, response: Any, *, fallback_text: str = "") -> str:
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
        request = self._build_text_request(
            prompt,
            conversation=conversation,
            instructions=instructions,
            allow_web_search=allow_web_search,
            prompt_cache_scope="response",
        )
        response = self._create_response(request, operation="respond_with_metadata")
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