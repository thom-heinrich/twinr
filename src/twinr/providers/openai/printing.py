from __future__ import annotations

from twinr.ops.usage import extract_model_name, extract_token_usage

from .instructions import PRINT_COMPOSER_INSTRUCTIONS, PRINT_FORMATTER_INSTRUCTIONS
from .types import ConversationLike, OpenAITextResponse


class OpenAIPrintMixin:
    def format_for_print(self, text: str) -> str:
        return self.format_for_print_with_metadata(text).text

    def format_for_print_with_metadata(self, text: str) -> OpenAITextResponse:
        request = self._build_response_request(
            text,
            instructions=PRINT_FORMATTER_INSTRUCTIONS,
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
        prompt = self._build_print_composer_prompt(
            conversation=conversation,
            focus_hint=focus_hint,
            direct_text=direct_text,
            request_source=request_source,
        )
        request = self._build_response_request(
            prompt,
            conversation=self._limit_print_conversation(conversation),
            instructions=self._print_composer_instructions(),
            allow_web_search=False,
            model=self.config.default_model,
            reasoning_effort="medium",
            max_output_tokens=180,
        )
        response = self._client.responses.create(**request)
        candidate_text = self._sanitize_print_text(self._extract_output_text(response))
        fallback_source = self._fallback_print_source(conversation, direct_text)
        final_text = candidate_text
        if self._should_use_print_fallback(candidate_text, fallback_source):
            fallback = self.format_for_print_with_metadata(fallback_source)
            final_text = self._sanitize_print_text(fallback.text)
        return OpenAITextResponse(
            text=final_text,
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, request["model"]),
            token_usage=extract_token_usage(response),
            used_web_search=False,
        )

    def _print_composer_instructions(self) -> str:
        return (
            f"{PRINT_COMPOSER_INSTRUCTIONS} "
            f"Never exceed {self.config.print_max_lines} lines or {self.config.print_max_chars} characters. "
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
        safe_focus_hint = (focus_hint or "").strip()[:240]
        safe_direct_text = (direct_text or "").strip()[: max(self.config.print_max_chars * 2, 240)]
        latest_user, latest_assistant = self._latest_print_exchange(conversation)
        parts = [
            f"Request source: {request_source.strip() or 'button'}",
            "Interpretation rule: if the request came from the print button, assume the user wants a compact print of the latest exchange.",
            f"Focus hint: {safe_focus_hint or '[none]'}",
            f"Latest user message: {latest_user or '[none]'}",
            f"Latest assistant message: {latest_assistant or '[none]'}",
            f"Direct print text: {safe_direct_text or '[none]'}",
            "Create the final thermal receipt text now.",
        ]
        return "\n".join(parts)

    def _limit_print_conversation(self, conversation: ConversationLike | None) -> ConversationLike | None:
        return self._limit_recent_conversation(
            conversation,
            max_turns=self.config.print_context_turns,
        )

    def _sanitize_print_text(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if normalized == "NO_PRINT_CONTENT":
            raise RuntimeError("No clear printable content is available")

        cleaned_lines: list[str] = []
        for raw_line in normalized.split("\n"):
            compact = " ".join(raw_line.strip().split())
            if compact:
                cleaned_lines.append(compact)

        if not cleaned_lines:
            raise RuntimeError("Print composer returned empty output")

        truncated = len(cleaned_lines) > self.config.print_max_lines
        cleaned_lines = cleaned_lines[: self.config.print_max_lines]
        result = "\n".join(cleaned_lines)
        if len(result) > self.config.print_max_chars:
            result = result[: self.config.print_max_chars].rstrip()
            truncated = True
        if truncated and result and not result.endswith("…"):
            if len(result) >= self.config.print_max_chars:
                result = result[:-1].rstrip()
            result = result.rstrip(" .") + "…"
        return result.strip()

    def _latest_print_exchange(self, conversation: ConversationLike | None) -> tuple[str | None, str | None]:
        latest_user: str | None = None
        latest_assistant: str | None = None
        if not conversation:
            return None, None
        for item in conversation:
            role, content = self._coerce_message(item)
            normalized_role = role.strip().lower()
            if normalized_role == "user" and content:
                latest_user = content
            elif normalized_role == "assistant" and content:
                latest_assistant = content
        return latest_user, latest_assistant

    def _fallback_print_source(
        self,
        conversation: ConversationLike | None,
        direct_text: str | None,
    ) -> str:
        if direct_text and direct_text.strip():
            return direct_text.strip()
        _latest_user, latest_assistant = self._latest_print_exchange(conversation)
        return (latest_assistant or "").strip()

    def _should_use_print_fallback(self, candidate_text: str, fallback_source: str) -> bool:
        if not fallback_source:
            return False
        line_count = len([line for line in candidate_text.splitlines() if line.strip()])
        candidate_length = len(candidate_text.strip())
        fallback_length = len(fallback_source.strip())
        if candidate_length == 0:
            return True
        if candidate_length < min(24, max(12, fallback_length // 3)):
            return True
        if line_count <= 1 and fallback_length > 40:
            return True
        return False
