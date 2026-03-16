from __future__ import annotations

import logging
import unicodedata
from typing import Any

from twinr.ops.usage import extract_model_name, extract_token_usage

from .instructions import PRINT_COMPOSER_INSTRUCTIONS, PRINT_FORMATTER_INSTRUCTIONS
from .types import ConversationLike, OpenAITextResponse


_LOG = logging.getLogger(__name__)


class OpenAIPrintMixin:
    # AUDIT-FIX(#6): Clamp invalid or missing print-related config values to safe defaults for Python 3.11 / .env resilience.
    _DEFAULT_PRINT_MAX_LINES = 8
    _DEFAULT_PRINT_MAX_CHARS = 240
    _DEFAULT_PRINT_CONTEXT_TURNS = 6
    _DEFAULT_PRINT_MODEL_TIMEOUT_SECONDS = 12.0

    # AUDIT-FIX(#8): Use senior-safe low-jargon errors when nothing printable is available or preparation fails.
    _EMPTY_PRINT_ERROR = "There is nothing ready to print yet."
    _PREPARE_PRINT_ERROR = "I couldn't prepare that for printing."

    def format_for_print(self, text: str) -> str:
        return self.format_for_print_with_metadata(text).text

    def format_for_print_with_metadata(self, text: str) -> OpenAITextResponse:
        # AUDIT-FIX(#1): Always sanitize printable content and keep a deterministic local fallback so the public formatter cannot emit raw/unsafe printer text.
        safe_input_text = self._coerce_optional_text(text)
        fallback_text, fallback_error = self._try_sanitize_print_text(safe_input_text)
        request: dict[str, Any] | None = None
        try:
            request = self._build_response_request(
                safe_input_text,
                instructions=PRINT_FORMATTER_INSTRUCTIONS,
                allow_web_search=False,
                model=self.config.default_model,
                reasoning_effort="low",
                max_output_tokens=140,
            )
            # AUDIT-FIX(#2): Route model calls through a best-effort timeout wrapper instead of making an unbounded blocking network call directly.
            response = self._responses_create(request)
            final_text = self._sanitize_print_text(self._extract_output_text(response))
            return self._build_text_response(
                text=final_text,
                response=response,
                request=request,
            )
        except Exception as exc:
            # AUDIT-FIX(#2): Degrade gracefully to deterministic formatting when the model stack or metadata extraction fails.
            self._log_print_failure("format_for_print_with_metadata", exc)
            if fallback_text:
                return self._build_text_response(
                    text=fallback_text,
                    response=None,
                    request=request,
                )
            if fallback_error is not None:
                raise fallback_error from exc
            raise RuntimeError(self._PREPARE_PRINT_ERROR) from exc

    def compose_print_job(
        self,
        *,
        conversation: ConversationLike | None = None,
        focus_hint: str | None = None,
        direct_text: str | None = None,
        request_source: str = "button",
    ) -> str:
        return self.compose_print_job_with_metadata(
            conversation=conversation,
            focus_hint=focus_hint,
            direct_text=direct_text,
            request_source=request_source,
        ).text

    def compose_print_job_with_metadata(
        self,
        *,
        conversation: ConversationLike | None = None,
        focus_hint: str | None = None,
        direct_text: str | None = None,
        request_source: str = "button",
    ) -> OpenAITextResponse:
        # AUDIT-FIX(#5): Materialize conversation once so generators are not exhausted and malformed state cannot desynchronize prompt/fallback/context handling.
        materialized_conversation = self._materialize_conversation(conversation)
        # AUDIT-FIX(#5): Normalize request_source defensively because callers may pass None/non-str values from external state or endpoints.
        normalized_request_source = self._normalize_request_source(request_source)

        literal_tool_text = self._literal_tool_print_text(
            direct_text=direct_text,
            request_source=normalized_request_source,
        )
        if literal_tool_text is not None:
            return self._build_text_response(
                text=literal_tool_text,
                response=None,
                request=None,
            )

        prompt = self._build_print_composer_prompt(
            conversation=materialized_conversation,
            focus_hint=focus_hint,
            direct_text=direct_text,
            request_source=normalized_request_source,
        )
        fallback_source = self._fallback_print_source(materialized_conversation, direct_text)

        request: dict[str, Any] | None = None
        response: Any | None = None
        candidate_text = ""
        candidate_error: RuntimeError | None = None

        try:
            request = self._build_response_request(
                prompt,
                conversation=self._limit_print_conversation(materialized_conversation),
                instructions=self._print_composer_instructions(),
                allow_web_search=False,
                model=self.config.default_model,
                reasoning_effort="medium",
                max_output_tokens=180,
            )
            # AUDIT-FIX(#2): Apply the same bounded/best-effort model call wrapper to the composer path.
            response = self._responses_create(request)
            candidate_text, candidate_error = self._try_sanitize_print_text(self._extract_output_text(response))
        except Exception as exc:
            # AUDIT-FIX(#2): Convert transport/parsing failures into graceful fallback instead of crashing the print flow.
            self._log_print_failure("compose_print_job_with_metadata", exc)
            candidate_error = RuntimeError(self._PREPARE_PRINT_ERROR)

        if candidate_text and not self._should_use_print_fallback(candidate_text, fallback_source):
            return self._build_text_response(
                text=candidate_text,
                response=response,
                request=request,
            )

        if fallback_source:
            try:
                fallback_response = self.format_for_print_with_metadata(fallback_source)
                # AUDIT-FIX(#3): When fallback text is used, return metadata for the response that actually produced the final print text.
                return fallback_response
            except RuntimeError:
                if candidate_text:
                    return self._build_text_response(
                        text=candidate_text,
                        response=response,
                        request=request,
                    )

        if candidate_text:
            return self._build_text_response(
                text=candidate_text,
                response=response,
                request=request,
            )

        if candidate_error is not None:
            raise candidate_error
        raise RuntimeError(self._EMPTY_PRINT_ERROR)

    def _print_composer_instructions(self) -> str:
        return (
            f"{PRINT_COMPOSER_INSTRUCTIONS} "
            f"Never exceed {self._print_max_lines()} lines or {self._print_max_chars()} characters. "
            "If there is no clear printable content, return exactly: NO_PRINT_CONTENT"
        )

    def _build_print_composer_prompt(
        self,
        *,
        conversation: ConversationLike | None,
        focus_hint: str | None,
        direct_text: str | None,
        request_source: str,
    ) -> str:
        # AUDIT-FIX(#4): Bound all prompt fields, including the latest exchange, so long replies cannot blow up latency, cost, or context limits.
        safe_focus_hint = self._clip_prompt_field(focus_hint, 240)
        safe_direct_text = self._clip_prompt_field(
            direct_text,
            min(max(self._print_max_chars() * 2, 240), 4000),
        )
        latest_user, latest_assistant = self._latest_print_exchange(conversation)
        safe_latest_user = self._clip_prompt_field(latest_user, 1200)
        safe_latest_assistant = self._clip_prompt_field(latest_assistant, 1600)
        parts = [
            f"Request source: {request_source or 'button'}",
            "Interpretation rule: if the request came from the print button, assume the user wants a compact print of the latest exchange.",
            f"Focus hint: {safe_focus_hint or '[none]'}",
            f"Latest user message: {safe_latest_user or '[none]'}",
            f"Latest assistant message: {safe_latest_assistant or '[none]'}",
            f"Direct print text: {safe_direct_text or '[none]'}",
            "Create the final thermal receipt text now.",
        ]
        return "\n".join(parts)

    def _limit_print_conversation(self, conversation: ConversationLike | None) -> ConversationLike | None:
        return self._limit_recent_conversation(
            conversation,
            max_turns=self._print_context_turns(),
        )

    def _sanitize_print_text(self, text: str) -> str:
        # AUDIT-FIX(#1): Strip unsafe control / format characters so model text or tool text cannot inject ESC/POS commands or bidi spoofing into printer output.
        normalized = self._strip_unsafe_print_text(text, preserve_newlines=True).strip()
        # AUDIT-FIX(#9): Treat common sentinel variants as empty content so the literal marker never reaches the printer.
        if self._is_no_print_content_marker(normalized):
            raise RuntimeError(self._EMPTY_PRINT_ERROR)

        cleaned_lines: list[str] = []
        for raw_line in normalized.split("\n"):
            compact = " ".join(raw_line.strip().split())
            if compact:
                cleaned_lines.append(compact)

        if not cleaned_lines:
            raise RuntimeError(self._EMPTY_PRINT_ERROR)

        max_lines = self._print_max_lines()
        max_chars = self._print_max_chars()

        truncated = len(cleaned_lines) > max_lines
        cleaned_lines = cleaned_lines[:max_lines]
        result = "\n".join(cleaned_lines)
        if len(result) > max_chars:
            result = result[:max_chars].rstrip()
            truncated = True
        if truncated and result and not result.endswith("…"):
            if len(result) >= max_chars:
                result = result[:-1].rstrip()
            result = result.rstrip(" .") + "…"
        result = result.strip()
        if not result:
            raise RuntimeError(self._EMPTY_PRINT_ERROR)
        return result

    def _try_sanitize_print_text(self, text: str) -> tuple[str, RuntimeError | None]:
        try:
            return self._sanitize_print_text(text), None
        except RuntimeError as exc:
            return "", exc

    def _latest_print_exchange(self, conversation: ConversationLike | None) -> tuple[str | None, str | None]:
        latest_user: str | None = None
        latest_assistant: str | None = None
        if not conversation:
            return None, None
        for item in conversation:
            try:
                # AUDIT-FIX(#5): Skip malformed conversation items instead of letting one corrupt record break the entire print path.
                role, content = self._coerce_message(item)
            except Exception as exc:
                self._log_print_failure("_latest_print_exchange", exc)
                continue
            normalized_role = self._coerce_optional_text(role).strip().lower()
            normalized_content = self._coerce_optional_text(content).strip()
            if normalized_role == "user" and normalized_content:
                latest_user = normalized_content
            elif normalized_role == "assistant" and normalized_content:
                latest_assistant = normalized_content
        return latest_user, latest_assistant

    def _fallback_print_source(
        self,
        conversation: ConversationLike | None,
        direct_text: str | None,
    ) -> str:
        if self._coerce_optional_text(direct_text).strip():
            return self._coerce_optional_text(direct_text).strip()
        latest_user, latest_assistant = self._latest_print_exchange(conversation)
        # AUDIT-FIX(#4): Fall back to the latest user text when no assistant reply exists yet, instead of failing with an empty print source.
        return (latest_assistant or latest_user or "").strip()

    def _literal_tool_print_text(
        self,
        *,
        direct_text: str | None,
        request_source: str,
    ) -> str | None:
        if self._normalize_request_source(request_source) != "tool":
            return None
        clean_direct_text = self._coerce_optional_text(direct_text).strip()
        if not clean_direct_text:
            return None
        sanitized, error = self._try_sanitize_print_text(clean_direct_text)
        if sanitized:
            return sanitized
        if error is not None:
            raise error
        return None

    def _should_use_print_fallback(self, candidate_text: str, fallback_source: str) -> bool:
        if not fallback_source:
            return False
        line_count = len([line for line in candidate_text.splitlines() if line.strip()])
        candidate_length = len(candidate_text.strip())
        fallback_length = len(fallback_source.strip())
        if candidate_length == 0:
            return True
        # AUDIT-FIX(#7): Keep concise but valid one-line receipts instead of aggressively replacing them with a longer fallback.
        if candidate_length <= 8 and fallback_length >= 48:
            return True
        if line_count == 0:
            return True
        return False

    # AUDIT-FIX(#2): Centralize OpenAI response creation so timeout handling and compatibility fallback are consistent across composer/formatter paths.
    def _responses_create(self, request: dict[str, Any]) -> Any:
        timeout_seconds = self._print_model_timeout_seconds()
        try:
            return self._client.responses.create(timeout=timeout_seconds, **request)
        except TypeError as exc:
            message = str(exc)
            if "timeout" not in message or not any(
                token in message for token in ("unexpected keyword", "multiple values")
            ):
                raise
            return self._client.responses.create(**request)

    # AUDIT-FIX(#3): Build metadata defensively so instrumentation failures do not block printing and returned metadata matches the actual text source.
    def _build_text_response(
        self,
        *,
        text: str,
        response: Any | None = None,
        request: dict[str, Any] | None = None,
    ) -> OpenAITextResponse:
        request_model = None
        if isinstance(request, dict):
            request_model = request.get("model")

        model: str | None = None
        token_usage: Any | None = None
        if response is not None:
            try:
                model = extract_model_name(response, request_model)
            except Exception as exc:
                self._log_print_failure("_build_text_response:model", exc)
            try:
                token_usage = extract_token_usage(response)
            except Exception as exc:
                self._log_print_failure("_build_text_response:token_usage", exc)

        return OpenAITextResponse(
            text=text,
            response_id=getattr(response, "id", None) if response is not None else None,
            request_id=getattr(response, "_request_id", None) if response is not None else None,
            model=model,
            token_usage=token_usage,
            used_web_search=False,
        )

    # AUDIT-FIX(#5): Materialize iterables once so multi-pass prompt/fallback logic remains stable even if ConversationLike is a generator.
    def _materialize_conversation(self, conversation: ConversationLike | None) -> ConversationLike | None:
        if conversation is None:
            return None
        if isinstance(conversation, tuple):
            return conversation
        try:
            return tuple(conversation)
        except TypeError:
            return conversation

    # AUDIT-FIX(#5): Normalize potentially non-string inputs from state or web handlers before prompt construction.
    def _coerce_optional_text(self, value: object | None) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (bytes, bytearray)):
            return bytes(value).decode("utf-8", "replace")
        try:
            return str(value)
        except Exception:
            return ""

    # AUDIT-FIX(#5): Restrict request_source to a short safe token so prompt fields stay predictable and do not throw on None/non-str values.
    def _normalize_request_source(self, request_source: object | None) -> str:
        normalized = self._coerce_optional_text(request_source).strip().lower()
        if not normalized:
            return "button"
        safe_chars = "".join(ch for ch in normalized if ch.isalnum() or ch in {"_", "-"})
        return safe_chars[:32] or "button"

    # AUDIT-FIX(#4): Clip prompt text deterministically and remove unsafe control characters before embedding model-visible fields.
    def _clip_prompt_field(self, value: object | None, limit: int) -> str:
        clean_text = self._strip_unsafe_print_text(self._coerce_optional_text(value), preserve_newlines=False).strip()
        if limit <= 0:
            return ""
        if len(clean_text) <= limit:
            return clean_text
        if limit == 1:
            return "…"
        return clean_text[: limit - 1].rstrip() + "…"

    # AUDIT-FIX(#1): Remove printer-dangerous control chars and invisible format chars while preserving normal line breaks.
    def _strip_unsafe_print_text(self, text: object | None, *, preserve_newlines: bool) -> str:
        raw_text = self._coerce_optional_text(text).replace("\r\n", "\n").replace("\r", "\n")
        cleaned_characters: list[str] = []
        for character in raw_text:
            if character == "\n" and preserve_newlines:
                cleaned_characters.append("\n")
                continue
            if character == "\t":
                cleaned_characters.append(" ")
                continue
            category = unicodedata.category(character)
            if category.startswith("C"):
                continue
            cleaned_characters.append(character)
        return "".join(cleaned_characters)

    # AUDIT-FIX(#9): Detect common model deviations around the NO_PRINT_CONTENT sentinel instead of printing the sentinel literally.
    def _is_no_print_content_marker(self, text: str) -> bool:
        marker = "".join(character for character in text.upper() if character.isalnum() or character == "_")
        return marker == "NO_PRINT_CONTENT"

    # AUDIT-FIX(#6): Validate numeric config sourced from .env so bad values do not produce empty receipts or negative slicing behavior.
    def _positive_int_config(self, name: str, default: int, *, minimum: int = 1) -> int:
        value = getattr(self.config, name, default)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(parsed, minimum)

    # AUDIT-FIX(#2): Validate timeout config so model calls use a sane bounded default when the .env value is absent or invalid.
    def _positive_float_config(self, name: str, default: float, *, minimum: float = 0.1) -> float:
        value = getattr(self.config, name, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return max(parsed, minimum)

    def _print_max_lines(self) -> int:
        return self._positive_int_config("print_max_lines", self._DEFAULT_PRINT_MAX_LINES)

    def _print_max_chars(self) -> int:
        return self._positive_int_config("print_max_chars", self._DEFAULT_PRINT_MAX_CHARS)

    def _print_context_turns(self) -> int:
        return self._positive_int_config("print_context_turns", self._DEFAULT_PRINT_CONTEXT_TURNS)

    def _print_model_timeout_seconds(self) -> float:
        return self._positive_float_config(
            "print_model_timeout_seconds",
            self._DEFAULT_PRINT_MODEL_TIMEOUT_SECONDS,
        )

    # AUDIT-FIX(#2): Log failures without printing user content, preserving operator visibility without leaking sensitive text.
    def _log_print_failure(self, stage: str, exc: BaseException) -> None:
        _LOG.warning("Print pipeline issue at %s: %s", stage, exc.__class__.__name__)