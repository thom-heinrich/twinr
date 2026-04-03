# CHANGELOG: 2026-03-30
# BUG-1: Fixed stale/latest-exchange selection so print fallback no longer grabs an older assistant reply when the newest turn is user-only or assistant commentary.
# BUG-2: Fixed prompt-field sanitization so stripped newlines become spaces instead of silently merging words and corrupting prompts.
# BUG-3: Fixed "bounded timeout" behavior by disabling implicit SDK retries for print calls; print latency now respects the configured timeout budget much more closely.
# BUG-4: Removed the second network round-trip from the fallback path; when model output is unusable, the mixin now falls back locally and deterministically.
# SEC-1: Disabled server-side response storage for print requests (`store=False`) so sensitive senior-assistance content is not persisted unnecessarily.
# SEC-2: Added hard caps for scanned conversation items and raw input size to reduce practical CPU/RAM denial-of-service risk on Raspberry Pi deployments.
# IMP-1: Upgraded print generation to Structured Outputs (`text.format` with strict JSON Schema) plus low verbosity, matching current Responses API best practice.
# IMP-2: Preserved 2026 assistant `phase` semantics by ignoring commentary-only assistant updates when selecting printable content.
# IMP-3: Switched to per-request `with_options(timeout=..., max_retries=0)` where available, with compatibility fallbacks for older clients.
# IMP-4: Added deterministic local compaction, sentence-aware segmentation, Unicode normalization, and bounded config values for safer receipt formatting.
# IMP-5: Added optional dedicated `config.print_model` support, while remaining drop-in compatible with `config.default_model`.
# BUG-5: Incomplete/max-output print responses now fail closed instead of leaking truncated JSON into receipt text.
# BUG-6: Structured print composition now retries once on the plain-text request when the structured request hits a transient provider/server error.

"""Compose and sanitize Twinr print output through OpenAI.

This module keeps thermal-printer-specific prompt construction, text
sanitization, and fallback handling separate from the generic response helpers
used by the rest of the OpenAI backend package.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import deque
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any

import httpx

from twinr.ops.usage import extract_model_name, extract_token_usage

from ..core.instructions import PRINT_COMPOSER_INSTRUCTIONS, PRINT_FORMATTER_INSTRUCTIONS
from ..core.types import ConversationLike, OpenAITextResponse


_LOG = logging.getLogger(__name__)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[^\s])")


class _NoPrintableContentError(RuntimeError):
    """Raised when no safe printable content is available."""


class OpenAIPrintMixin:
    """Provide bounded, printer-safe receipt composition helpers.

    The mixin keeps final print text short, sanitized, and deterministic so the
    runtime can always fall back to a local path when model output is unusable.
    """

    _DEFAULT_PRINT_MAX_LINES = 8
    _DEFAULT_PRINT_MAX_CHARS = 240
    _DEFAULT_PRINT_CONTEXT_TURNS = 6
    _DEFAULT_PRINT_MODEL_TIMEOUT_SECONDS = 12.0
    _DEFAULT_PRINT_MODEL_MAX_RETRIES = 0
    _DEFAULT_PRINT_MAX_OUTPUT_TOKENS = 320
    _DEFAULT_PRINT_INPUT_CHAR_LIMIT = 16_000
    _DEFAULT_PRINT_SCAN_ITEMS = 128
    _DEFAULT_PRINT_MODEL = "gpt-5.4-mini"

    _EMPTY_PRINT_ERROR = "There is nothing ready to print yet."
    _PREPARE_PRINT_ERROR = "I couldn't prepare that for printing."

    _LOW_INFORMATION_PRINT_MARKERS = frozenset(
        {
            "ok",
            "okay",
            "done",
            "ready",
            "printed",
            "print",
            "yes",
            "no",
            "sure",
            "thanks",
            "thank you",
            "youre welcome",
            "you're welcome",
            "working on it",
        }
    )

    def format_for_print(self, text: str) -> str:
        """Return printer-ready text without OpenAI metadata."""

        return self.format_for_print_with_metadata(text).text

    def format_for_print_with_metadata(self, text: str) -> OpenAITextResponse:
        """Format free text into printer-safe receipt copy.

        Args:
            text: Source text to compact and sanitize for receipt output.

        Returns:
            An ``OpenAITextResponse`` containing final printer-safe text.

        Raises:
            RuntimeError: If no safe printable output can be prepared.
        """

        safe_input_text = self._bounded_runtime_text(text, self._print_input_char_limit())
        fallback_text, fallback_error = self._try_sanitize_print_text(safe_input_text)

        request: dict[str, Any] | None = None
        response: Any | None = None
        try:
            request = self._build_response_request(
                safe_input_text,
                instructions=PRINT_FORMATTER_INSTRUCTIONS,
                allow_web_search=False,
                model=self._print_model_name(),
                reasoning_effort="low",
                max_output_tokens=140,
            )
            request = self._prepare_print_request(request)
            response, request, structured_used = self._create_print_response(
                request,
                stage="format_for_print_with_metadata",
            )
            final_text = self._sanitize_print_text(
                self._extract_print_text_from_response(
                    response,
                    structured_expected=structured_used,
                )
            )
            return self._build_print_text_response(
                text=final_text,
                response=response,
                request=request,
            )
        except Exception as exc:
            self._log_print_failure("format_for_print_with_metadata", exc)
            if fallback_text:
                return self._build_print_text_response(
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
        """Return printer-ready text for the current print request."""

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
        """Compose a print job from conversation context or explicit text.

        Args:
            conversation: Optional recent conversation context to summarize.
            focus_hint: Optional hint about what to print.
            direct_text: Optional explicit print content.
            request_source: Source label such as ``button`` or ``tool``.

        Returns:
            An ``OpenAITextResponse`` containing final receipt text and metadata.

        Raises:
            RuntimeError: If no printable content can be prepared.
        """

        materialized_conversation = self._materialize_conversation(conversation)
        normalized_request_source = self._normalize_request_source(request_source)

        literal_tool_text = self._literal_tool_print_text(
            direct_text=direct_text,
            request_source=normalized_request_source,
        )
        if literal_tool_text is not None:
            return self._build_print_text_response(
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
        fallback_text, fallback_error = self._try_sanitize_print_text(fallback_source)

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
                model=self._print_model_name(),
                reasoning_effort="low",
                max_output_tokens=self._print_max_output_tokens(),
            )
            request = self._prepare_print_request(request)
            response, request, structured_used = self._create_print_response(
                request,
                stage="compose_print_job_with_metadata",
            )
            candidate_text, candidate_error = self._try_extract_and_sanitize_print_text(
                response,
                structured_expected=structured_used,
            )
        except Exception as exc:
            self._log_print_failure("compose_print_job_with_metadata", exc)
            candidate_error = RuntimeError(self._PREPARE_PRINT_ERROR)

        if candidate_text and not self._should_use_print_fallback(candidate_text, fallback_source):
            return self._build_print_text_response(
                text=candidate_text,
                response=response,
                request=request,
            )

        if fallback_text:
            return self._build_print_text_response(
                text=fallback_text,
                response=None,
                request=request,
            )

        if candidate_text:
            return self._build_print_text_response(
                text=candidate_text,
                response=response,
                request=request,
            )

        if candidate_error is not None:
            raise candidate_error
        if fallback_error is not None:
            raise fallback_error
        raise RuntimeError(self._EMPTY_PRINT_ERROR)

    def _print_composer_instructions(self) -> str:
        """Return the instruction block used for print composition."""

        return (
            f"{PRINT_COMPOSER_INSTRUCTIONS} "
            "Ignore assistant commentary updates and prefer the latest completed assistant answer "
            "for the latest user turn. "
            f"Never exceed {self._print_max_lines()} lines or {self._print_max_chars()} characters. "
            "If there is no clear printable content, return status=no_print in the requested schema."
        )

    def _build_print_composer_prompt(
        self,
        *,
        conversation: ConversationLike | None,
        focus_hint: str | None,
        direct_text: str | None,
        request_source: str,
    ) -> str:
        """Build the model prompt used to compose receipt text."""

        safe_focus_hint = self._clip_prompt_field(focus_hint, 240)
        safe_direct_text = self._clip_prompt_field(
            self._bounded_runtime_text(direct_text, self._print_input_char_limit()),
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
        """Trim conversation history to the configured print context window."""

        return self._limit_recent_conversation(
            conversation,
            max_turns=self._print_context_turns(),
        )

    def _sanitize_print_text(self, text: str) -> str:
        """Normalize model or tool text into receipt-safe output."""

        normalized = self._strip_unsafe_print_text(text, preserve_newlines=True).strip()
        if self._is_no_print_content_marker(normalized):
            raise _NoPrintableContentError(self._EMPTY_PRINT_ERROR)

        segments = self._normalize_print_segments(normalized)
        if not segments:
            raise _NoPrintableContentError(self._EMPTY_PRINT_ERROR)

        max_lines = self._print_max_lines()
        max_chars = self._print_max_chars()

        lines: list[str] = []
        truncated = False

        for segment in segments:
            if len(lines) >= max_lines:
                truncated = True
                break

            proposed = "\n".join(lines + [segment]).strip()
            if proposed and len(proposed) <= max_chars:
                lines.append(segment)
                continue

            remaining_chars = max_chars - len("\n".join(lines))
            if lines:
                remaining_chars -= 1
            fitted = self._fit_segment_to_budget(segment, remaining_chars)
            if fitted:
                lines.append(fitted)
            truncated = True
            break

        result = "\n".join(lines).strip()
        if not result:
            raise _NoPrintableContentError(self._EMPTY_PRINT_ERROR)

        if truncated:
            result = self._append_print_ellipsis(result, max_chars)

        result = result.strip()
        if not result:
            raise _NoPrintableContentError(self._EMPTY_PRINT_ERROR)
        return result

    def _try_sanitize_print_text(self, text: str) -> tuple[str, RuntimeError | None]:
        """Return sanitized print text or the raised sanitization error."""

        try:
            return self._sanitize_print_text(text), None
        except RuntimeError as exc:
            return "", exc

    def _latest_print_exchange(self, conversation: ConversationLike | None) -> tuple[str | None, str | None]:
        """Return the latest user text and its latest completed assistant answer."""

        if not conversation:
            return None, None

        parsed_messages: list[tuple[str, str, str, int]] = []
        for index, item in enumerate(conversation):
            try:
                role, content, phase = self._coerce_message(item)
            except Exception as exc:
                self._log_print_failure("_latest_print_exchange", exc)
                continue

            normalized_role = self._coerce_optional_text(role).strip().lower()
            normalized_content = self._bounded_runtime_text(content, self._print_input_char_limit()).strip()
            normalized_phase = self._coerce_optional_text(phase).strip().lower()

            if normalized_role not in {"user", "assistant"} or not normalized_content:
                continue
            parsed_messages.append((normalized_role, normalized_content, normalized_phase, index))

        if not parsed_messages:
            return None, None

        latest_user: tuple[str, str, str, int] | None = None
        for message in reversed(parsed_messages):
            if message[0] == "user":
                latest_user = message
                break

        if latest_user is not None:
            latest_user_index = latest_user[3]
            assistant_after_user = [
                message
                for message in parsed_messages
                if message[0] == "assistant" and message[3] > latest_user_index
            ]
            final_assistant_after_user = [
                message for message in assistant_after_user if message[2] == "final_answer"
            ]
            if final_assistant_after_user:
                return latest_user[1], final_assistant_after_user[-1][1]
            if assistant_after_user:
                return latest_user[1], assistant_after_user[-1][1]
            return latest_user[1], None

        assistant_messages = [message for message in parsed_messages if message[0] == "assistant"]
        final_assistants = [message for message in assistant_messages if message[2] == "final_answer"]
        assistant = final_assistants[-1] if final_assistants else assistant_messages[-1]
        return None, assistant[1]

    def _fallback_print_source(
        self,
        conversation: ConversationLike | None,
        direct_text: str | None,
    ) -> str:
        """Select the best local fallback source for print formatting."""

        clean_direct_text = self._bounded_runtime_text(direct_text, self._print_input_char_limit()).strip()
        if clean_direct_text:
            return clean_direct_text
        latest_user, latest_assistant = self._latest_print_exchange(conversation)
        return (latest_assistant or latest_user or "").strip()

    def _literal_tool_print_text(
        self,
        *,
        direct_text: str | None,
        request_source: str,
    ) -> str | None:
        """Return direct tool-supplied print text when it is already printable."""

        if self._normalize_request_source(request_source) != "tool":
            return None
        clean_direct_text = self._bounded_runtime_text(direct_text, self._print_input_char_limit()).strip()
        if not clean_direct_text:
            return None
        sanitized, error = self._try_sanitize_print_text(clean_direct_text)
        if sanitized:
            return sanitized
        if error is not None:
            raise error
        return None

    def _should_use_print_fallback(self, candidate_text: str, fallback_source: str) -> bool:
        """Decide whether local fallback text is safer than model output."""

        if not fallback_source:
            return False

        candidate_clean = candidate_text.strip()
        if not candidate_clean:
            return True

        normalized_candidate = self._normalize_low_information_text(candidate_clean)
        if normalized_candidate in self._LOW_INFORMATION_PRINT_MARKERS:
            return True

        line_count = len([line for line in candidate_clean.splitlines() if line.strip()])
        return line_count == 0

    def _responses_create(self, request: dict[str, Any]) -> Any:
        """Create a response for print flows with bounded timeout handling."""

        timeout = self._httpx_print_timeout()
        max_retries = self._print_model_max_retries()
        client = self._client

        with_options = getattr(client, "with_options", None)
        if callable(with_options):
            try:
                return with_options(timeout=timeout, max_retries=max_retries).responses.create(**request)
            except TypeError as exc:
                if not self._looks_like_option_signature_issue(exc):
                    raise

        try:
            return client.responses.create(timeout=timeout, **request)
        except TypeError as exc:
            if not self._looks_like_option_signature_issue(exc):
                raise
            return client.responses.create(**request)

    def _build_print_text_response(
        self,
        *,
        text: str,
        response: Any | None = None,
        request: dict[str, Any] | None = None,
    ) -> OpenAITextResponse:
        """Build the Twinr print-response object for final receipt text."""

        request_model = None
        if isinstance(request, dict):
            request_model = request.get("model")

        model: str | None = None
        token_usage: Any | None = None
        if response is not None:
            try:
                model = extract_model_name(response, request_model)
            except Exception as exc:
                self._log_print_failure("_build_print_text_response:model", exc)
            try:
                token_usage = extract_token_usage(response)
            except Exception as exc:
                self._log_print_failure("_build_print_text_response:token_usage", exc)

        return OpenAITextResponse(
            text=text,
            response_id=getattr(response, "id", None) if response is not None else None,
            request_id=getattr(response, "_request_id", None) if response is not None else None,
            model=model,
            token_usage=token_usage,
            used_web_search=False,
        )

    def _materialize_conversation(self, conversation: ConversationLike | None) -> ConversationLike | None:
        """Materialize recent conversation items while bounding scan cost."""

        if conversation is None:
            return None
        if isinstance(conversation, (str, bytes, bytearray, Mapping)):
            return None

        scan_limit = self._print_scan_items()

        if isinstance(conversation, tuple):
            return conversation[-scan_limit:]
        if isinstance(conversation, list):
            return tuple(conversation[-scan_limit:])
        if isinstance(conversation, Sequence):
            try:
                return tuple(conversation[-scan_limit:])
            except Exception:
                pass

        try:
            iterator = iter(conversation)
        except TypeError:
            return None

        hard_limit = max(scan_limit, min(scan_limit * 4, 512))
        recent_items: deque[Any] = deque(maxlen=scan_limit)
        for item in islice(iterator, hard_limit):
            recent_items.append(item)
        return tuple(recent_items)

    def _coerce_optional_text(self, value: object | None) -> str:
        """Convert optional runtime text inputs into safe strings."""

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

    def _normalize_request_source(self, request_source: object | None) -> str:
        """Normalize the print request source to a short safe token."""

        normalized = self._coerce_optional_text(request_source).strip().lower()
        if not normalized:
            return "button"
        safe_chars = "".join(ch for ch in normalized if ch.isalnum() or ch in {"_", "-"})
        return safe_chars[:32] or "button"

    def _clip_prompt_field(self, value: object | None, limit: int) -> str:
        """Clip prompt text deterministically after sanitization."""

        clean_text = self._strip_unsafe_print_text(
            self._bounded_runtime_text(value, self._print_input_char_limit()),
            preserve_newlines=False,
        ).strip()
        if limit <= 0:
            return ""
        if len(clean_text) <= limit:
            return clean_text
        if limit == 1:
            return "…"
        return clean_text[: limit - 1].rstrip() + "…"

    def _strip_unsafe_print_text(self, text: object | None, *, preserve_newlines: bool) -> str:
        """Remove control and format characters from printer-facing text."""

        raw_text = self._coerce_optional_text(text)
        raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        raw_text = raw_text.replace("\u2028", "\n").replace("\u2029", "\n").replace("\u0085", "\n")
        raw_text = unicodedata.normalize("NFC", raw_text)

        cleaned_characters: list[str] = []
        for character in raw_text:
            if character == "\n":
                cleaned_characters.append("\n" if preserve_newlines else " ")
                continue
            if character == "\t":
                cleaned_characters.append(" ")
                continue
            category = unicodedata.category(character)
            if category.startswith("C"):
                continue
            cleaned_characters.append(character)
        return "".join(cleaned_characters)

    def _is_no_print_content_marker(self, text: str) -> bool:
        """Return whether text is a normalized ``NO_PRINT_CONTENT`` sentinel."""

        marker = "".join(character for character in text.upper() if character.isalnum() or character == "_")
        return marker == "NO_PRINT_CONTENT"

    def _bounded_int_config(
        self,
        name: str,
        default: int,
        *,
        minimum: int = 0,
        maximum: int | None = None,
    ) -> int:
        """Return a bounded integer config value or the supplied default."""

        value = getattr(self.config, name, default)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        parsed = max(parsed, minimum)
        if maximum is not None:
            parsed = min(parsed, maximum)
        return parsed

    def _bounded_float_config(
        self,
        name: str,
        default: float,
        *,
        minimum: float = 0.0,
        maximum: float | None = None,
    ) -> float:
        """Return a bounded float config value or the supplied default."""

        value = getattr(self.config, name, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = default
        parsed = max(parsed, minimum)
        if maximum is not None:
            parsed = min(parsed, maximum)
        return parsed

    def _print_max_lines(self) -> int:
        """Return the configured maximum number of receipt lines."""

        return self._bounded_int_config(
            "print_max_lines",
            self._DEFAULT_PRINT_MAX_LINES,
            minimum=1,
            maximum=32,
        )

    def _print_max_chars(self) -> int:
        """Return the configured maximum number of receipt characters."""

        return self._bounded_int_config(
            "print_max_chars",
            self._DEFAULT_PRINT_MAX_CHARS,
            minimum=8,
            maximum=2_048,
        )

    def _print_context_turns(self) -> int:
        """Return the configured number of conversation turns kept for print."""

        return self._bounded_int_config(
            "print_context_turns",
            self._DEFAULT_PRINT_CONTEXT_TURNS,
            minimum=1,
            maximum=32,
        )

    def _print_model_timeout_seconds(self) -> float:
        """Return the timeout used for print-model requests."""

        return self._bounded_float_config(
            "print_model_timeout_seconds",
            self._DEFAULT_PRINT_MODEL_TIMEOUT_SECONDS,
            minimum=0.5,
            maximum=30.0,
        )

    def _print_model_max_retries(self) -> int:
        """Return the retry budget for print-model requests."""

        return self._bounded_int_config(
            "print_model_max_retries",
            self._DEFAULT_PRINT_MODEL_MAX_RETRIES,
            minimum=0,
            maximum=2,
        )

    def _print_max_output_tokens(self) -> int:
        """Return the output-token budget used for print composition."""

        return self._bounded_int_config(
            "print_max_output_tokens",
            self._DEFAULT_PRINT_MAX_OUTPUT_TOKENS,
            minimum=140,
            maximum=1024,
        )

    def _print_input_char_limit(self) -> int:
        """Return the maximum raw input characters processed by the print path."""

        return self._bounded_int_config(
            "print_input_char_limit",
            self._DEFAULT_PRINT_INPUT_CHAR_LIMIT,
            minimum=self._print_max_chars(),
            maximum=64_000,
        )

    def _print_scan_items(self) -> int:
        """Return the maximum number of conversation items scanned for printing."""

        return self._bounded_int_config(
            "print_scan_items",
            self._DEFAULT_PRINT_SCAN_ITEMS,
            minimum=max(self._print_context_turns() * 2, 8),
            maximum=512,
        )

    def _print_model_name(self) -> str:
        """Return the model used for print generation."""

        configured = self._coerce_optional_text(getattr(self.config, "print_model", None)).strip()
        if configured:
            return configured
        default_model = self._coerce_optional_text(getattr(self.config, "default_model", None)).strip()
        return default_model or self._DEFAULT_PRINT_MODEL

    def _log_print_failure(self, stage: str, exc: BaseException) -> None:
        """Log bounded print pipeline failures without leaking receipt content."""

        _LOG.warning("Print pipeline issue at %s: %s", stage, exc.__class__.__name__)

    def _bounded_runtime_text(self, value: object | None, limit: int) -> str:
        """Coerce runtime text and bound its size before further processing."""

        text = self._coerce_optional_text(value)
        if limit <= 0 or len(text) <= limit:
            return text
        return text[:limit]

    def _normalize_print_segments(self, text: str) -> list[str]:
        """Split sanitized text into compact receipt-sized segments."""

        segments: list[str] = []
        for raw_line in text.split("\n"):
            compact = " ".join(raw_line.strip().split())
            if not compact:
                continue
            if compact[:1] in {"-", "*", "•", "·"}:
                segments.append(compact)
                continue
            split_segments = [part.strip() for part in _SENTENCE_SPLIT_RE.split(compact) if part.strip()]
            segments.extend(split_segments or [compact])
        return segments

    def _fit_segment_to_budget(self, segment: str, budget: int) -> str:
        """Fit a segment into the remaining character budget."""

        if budget <= 0:
            return ""
        if len(segment) <= budget:
            return segment
        if budget == 1:
            return segment[:1]

        clipped = segment[:budget].rstrip()
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0].rstrip(" ,;:-")
        return clipped or segment[:budget].rstrip()

    def _append_print_ellipsis(self, text: str, max_chars: int) -> str:
        """Append a single ellipsis while preserving the global char limit."""

        result = text.rstrip()
        if not result:
            raise _NoPrintableContentError(self._EMPTY_PRINT_ERROR)
        if result.endswith("…") and len(result) <= max_chars:
            return result

        while result and len(result) >= max_chars:
            result = result[:-1].rstrip()
        result = result.rstrip(" .")
        if not result:
            raise _NoPrintableContentError(self._EMPTY_PRINT_ERROR)
        return result + "…"

    def _normalize_low_information_text(self, text: str) -> str:
        """Normalize a short candidate for low-information checks."""

        collapsed = " ".join(text.strip().split()).lower()
        return "".join(ch for ch in collapsed if ch.isalnum() or ch.isspace()).strip()

    def _prepare_print_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Apply print-safe Responses API defaults."""

        prepared = dict(request)
        prepared["model"] = self._print_model_name()
        prepared["store"] = False
        prepared["truncation"] = "auto"

        text_config = prepared.get("text")
        if not isinstance(text_config, dict):
            text_config = {}
        else:
            text_config = dict(text_config)
        text_config["verbosity"] = "low"
        prepared["text"] = text_config
        return prepared

    def _create_print_response(
        self,
        request: dict[str, Any],
        *,
        stage: str,
    ) -> tuple[Any, dict[str, Any], bool]:
        """Create a print response, preferring structured outputs with plain-text fallback."""

        structured_request = self._with_structured_print_format(request)
        try:
            return self._responses_create(structured_request), structured_request, True
        except Exception as exc:
            if not (
                self._looks_like_structured_output_request_issue(exc)
                or self._looks_like_transient_print_response_error(exc)
            ):
                raise
            self._log_print_failure(f"{stage}:structured_output", exc)
            plain_request = self._without_structured_print_format(request)
            return self._responses_create(plain_request), plain_request, False

    def _with_structured_print_format(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return a request variant that enforces a strict print JSON schema."""

        prepared = self._without_structured_print_format(request)
        text_config = dict(prepared.get("text") or {})
        text_config["format"] = {
            "type": "json_schema",
            "name": "twinr_print_receipt",
            "description": "Compact printer-safe receipt content for a thermal printer.",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["ready", "no_print"],
                    },
                    "text": {
                        "type": "string",
                    },
                },
                "required": ["status", "text"],
                "additionalProperties": False,
            },
        }
        prepared["text"] = text_config
        return prepared

    def _without_structured_print_format(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return a request variant without JSON-schema formatting."""

        prepared = dict(request)
        text_config = prepared.get("text")
        if isinstance(text_config, dict):
            text_config = dict(text_config)
            text_config.pop("format", None)
            prepared["text"] = text_config
        return prepared

    def _extract_print_text_from_response(self, response: Any, *, structured_expected: bool) -> str:
        """Extract text from a print response, parsing structured outputs when present."""

        incomplete_reason = self._print_response_incomplete_reason(response)
        if incomplete_reason is not None:
            raise RuntimeError(f"print response incomplete: {incomplete_reason}")
        output_text = self._response_output_text(response)
        if structured_expected:
            try:
                payload = json.loads(output_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError("print structured output was not valid JSON") from exc
            if not isinstance(payload, dict):
                raise RuntimeError("print structured output must be a JSON object")

            status = self._coerce_optional_text(payload.get("status")).strip().lower()
            text = self._coerce_optional_text(payload.get("text"))
            if status == "no_print":
                raise _NoPrintableContentError(self._EMPTY_PRINT_ERROR)
            if status not in {"", "ready"}:
                raise RuntimeError(self._PREPARE_PRINT_ERROR)
            if not text:
                raise RuntimeError("print structured output is missing text")
            return text

        return output_text

    def _try_extract_and_sanitize_print_text(
        self,
        response: Any,
        *,
        structured_expected: bool,
    ) -> tuple[str, RuntimeError | None]:
        """Extract response text and sanitize it for printing."""

        try:
            raw_text = self._extract_print_text_from_response(
                response,
                structured_expected=structured_expected,
            )
            return self._sanitize_print_text(raw_text), None
        except RuntimeError as exc:
            return "", exc

    def _response_output_text(self, response: Any) -> str:
        """Extract the aggregated output text from a Responses API object."""

        direct_output_text = getattr(response, "output_text", None)
        if isinstance(direct_output_text, str) and direct_output_text.strip():
            return direct_output_text
        try:
            extracted = self._extract_output_text(response)
        except Exception as exc:
            self._log_print_failure("_response_output_text", exc)
            extracted = direct_output_text
        return self._coerce_optional_text(extracted)

    def _httpx_print_timeout(self) -> httpx.Timeout:
        """Return a granular timeout object for print requests."""

        total = self._print_model_timeout_seconds()
        connect = min(2.0, total)
        pool = min(2.0, total)
        write = min(5.0, total)
        return httpx.Timeout(total, connect=connect, read=total, write=write, pool=pool)

    def _looks_like_option_signature_issue(self, exc: TypeError) -> bool:
        """Return whether a TypeError likely comes from unsupported request options."""

        message = str(exc).lower()
        return "timeout" in message or "max_retries" in message or "with_options" in message

    def _looks_like_structured_output_request_issue(self, exc: BaseException) -> bool:
        """Return whether an exception indicates unsupported structured-output parameters."""

        message = self._coerce_optional_text(exc).lower()
        hints = (
            "json_schema",
            "structured",
            "text.format",
            "response_format",
            "unsupported parameter",
            "unknown parameter",
            "invalid schema",
            "strict",
        )
        return any(hint in message for hint in hints)

    def _looks_like_transient_print_response_error(self, exc: BaseException) -> bool:
        """Return whether a print request failed with a transient server/runtime error."""

        message = self._coerce_optional_text(exc).lower()
        error_name = exc.__class__.__name__.lower()
        if error_name in {"internalservererror", "apitimeouterror", "apiconnectionerror"}:
            return True
        return any(
            hint in message
            for hint in (
                "internalservererror",
                "server error",
                "status code: 500",
                "status code: 502",
                "status code: 503",
                "status code: 504",
                "timed out",
                "connection error",
            )
        )

    def _print_response_incomplete_reason(self, response: Any) -> str | None:
        """Return the best available incomplete-response reason for one print request."""

        status = self._coerce_optional_text(getattr(response, "status", None)).strip().lower()
        if status and status != "incomplete":
            return None
        top_level = getattr(response, "incomplete_details", None)
        if top_level is not None:
            detail = self._coerce_optional_text(getattr(top_level, "reason", None) or top_level).strip()
            if detail:
                return detail
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) != "message" or getattr(item, "status", None) != "incomplete":
                continue
            detail = getattr(item, "incomplete_details", None)
            normalized = self._coerce_optional_text(getattr(detail, "reason", None) or detail).strip()
            if normalized:
                return normalized
        return "incomplete" if status == "incomplete" else None
