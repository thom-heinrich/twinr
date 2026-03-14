from __future__ import annotations

from typing import Any, Mapping

from twinr.text_utils import extract_json_object


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
    request = backend._build_response_request(
        prompt,
        instructions=instructions,
        allow_web_search=False,
        model=model or backend.config.default_model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
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
    return extract_json_object(backend._extract_output_text(response))


__all__ = ["request_structured_json_object"]
