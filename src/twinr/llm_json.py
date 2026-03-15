from __future__ import annotations

from typing import Any, Mapping

from twinr.text_utils import extract_json_object


def _response_status(response: Any) -> str:
    return str(getattr(response, "status", "") or "").strip().lower()


def _incomplete_reason(response: Any) -> str:
    details = getattr(response, "incomplete_details", None)
    if isinstance(details, Mapping):
        return str(details.get("reason", "") or "").strip().lower()
    return str(getattr(details, "reason", "") or "").strip().lower()


def _next_retry_max_output_tokens(current: int) -> int:
    return min(4096, max(current * 2, current + 400))


def _structured_response_error_message(
    *,
    schema_name: str,
    response: Any,
    output_text: str,
    max_output_tokens: int,
) -> str:
    status = _response_status(response) or "unknown"
    reason = _incomplete_reason(response) or "unknown"
    suffix = output_text[-220:].replace("\n", "\\n")
    return (
        f"Structured JSON response for {schema_name!r} was not parseable "
        f"(status={status}, incomplete_reason={reason}, "
        f"max_output_tokens={max_output_tokens}, output_text_len={len(output_text)}, "
        f"output_text_tail={suffix!r})"
    )


def request_structured_json_object(
    backend: Any,
    *,
    prompt: str,
    instructions: str,
    schema_name: str,
    schema: Mapping[str, object],
    model: str | None = None,
    reasoning_effort: str = "low",
    max_output_tokens: int = 512,
) -> dict[str, object]:
    current_max_output_tokens = max(1, int(max_output_tokens))
    response: Any | None = None
    output_text = ""
    last_error: Exception | None = None
    for _ in range(3):
        request = backend._build_response_request(
            prompt,
            instructions=instructions,
            allow_web_search=False,
            model=model or backend.config.default_model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=current_max_output_tokens,
        )
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": dict(schema),
                "strict": True,
            }
        }
        response = backend._client.responses.create(**request)
        parsed = getattr(response, "output_parsed", None)
        if isinstance(parsed, dict):
            return parsed
        output_text = backend._extract_output_text(response)
        if _response_status(response) == "incomplete" and _incomplete_reason(response) == "max_output_tokens":
            next_limit = _next_retry_max_output_tokens(current_max_output_tokens)
            if next_limit <= current_max_output_tokens:
                break
            current_max_output_tokens = next_limit
            continue
        try:
            return extract_json_object(output_text)
        except ValueError as exc:
            last_error = exc
            break
    raise ValueError(
        _structured_response_error_message(
            schema_name=schema_name,
            response=response,
            output_text=output_text,
            max_output_tokens=current_max_output_tokens,
        )
    ) from last_error


__all__ = ["request_structured_json_object"]
