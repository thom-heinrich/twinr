# CHANGELOG: 2026-03-30
# BUG-1: Replaced the partial hand-rolled JSON Schema validator with a cached standards-based validator
# BUG-2: Failed/refused/empty backend responses now fail closed (or retry when retryable) instead of degrading into misleading parse errors
# SEC-1: Removed the unbounded timeoutless fallback path; backends must honor a bounded timeout kwarg or expose with_options(timeout=...)
# SEC-2: Raw-text JSON fallback is now disabled for failed/refused/incomplete responses so unconstrained text cannot be mistaken for trusted structured output
# IMP-1: Added $schema-aware validator selection, canonical deep-copy of schemas, and validator caching for correctness/perf on Raspberry Pi 4
# IMP-2: Added explicit response error/refusal extraction, retry handling for retryable failed responses, and optional format enforcement via backend config
# BREAKING: Requires jsonschema>=4.26 to be installed
# BREAKING: Backends that cannot enforce a bounded request timeout must be updated instead of silently running without a timeout

"""Request structured JSON outputs from model backends with local validation.

This module wraps backend structured-response calls with bounded retries,
timeout handling, privacy-safe diagnostics, and a cautious fallback path that
extracts and validates JSON objects from raw model text.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import re
import time
from functools import lru_cache
from typing import Any, Mapping, Sequence

# BREAKING: Standard-compliant local validation now depends on jsonschema>=4.26.
try:
    from jsonschema import Draft202012Validator, exceptions as jsonschema_exceptions
    from jsonschema.validators import validator_for
except ImportError as exc:  # pragma: no cover - dependency/import failure path
    raise RuntimeError(
        "request_structured_json_object requires jsonschema>=4.26. "
        "Install it with: pip install 'jsonschema>=4.26'"
    ) from exc

from twinr.text_utils import extract_json_object

_MAX_STRUCTURED_RESPONSE_OUTPUT_TOKENS = 4096
_DEFAULT_STRUCTURED_RESPONSE_TIMEOUT_SECONDS = 30.0
_TRANSPORT_RETRY_BACKOFF_SECONDS = (0.25, 0.5)
_SIMPLE_PATH_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RETRYABLE_ERROR_NAME_MARKERS = (
    "apiconnectionerror",
    "apitimeouterror",
    "ratelimiterror",
    "internalservererror",
    "serviceunavailableerror",
    "timeout",
    "connection",
)
_RETRYABLE_ERROR_MESSAGE_MARKERS = (
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "connection refused",
    "temporarily unavailable",
    "temporary failure",
    "rate limit",
    "too many requests",
    "server error",
    "bad gateway",
    "gateway timeout",
    "service unavailable",
    "502",
    "503",
    "504",
    "429",
)


def _mapping_get_or_attr(container: Any, name: str, default: Any = None) -> Any:
    """Read a named field from a mapping-like or attribute-bearing object."""

    if isinstance(container, Mapping):
        return container.get(name, default)
    return getattr(container, name, default)


def _response_status(response: Any) -> str:
    """Normalize a response status string for control-flow checks."""

    return str(_mapping_get_or_attr(response, "status", "") or "").strip().lower()


def _response_id(response: Any) -> str:
    """Return the response id when available."""

    return str(_mapping_get_or_attr(response, "id", "") or "").strip()


def _incomplete_reason(response: Any) -> str:
    """Normalize the response's incomplete reason for retry decisions."""

    details = _mapping_get_or_attr(response, "incomplete_details", None)
    if isinstance(details, Mapping):
        return str(details.get("reason", "") or "").strip().lower()
    return str(_mapping_get_or_attr(details, "reason", "") or "").strip().lower()


def _response_error_code_and_message(response: Any) -> tuple[str, str]:
    """Extract backend error code/message pairs from a response object."""

    error = _mapping_get_or_attr(response, "error", None)
    if error is None:
        return "", ""
    if isinstance(error, Mapping):
        code = error.get("code", "")
        message = error.get("message", "")
    else:
        code = _mapping_get_or_attr(error, "code", "")
        message = _mapping_get_or_attr(error, "message", "")
    return str(code or "").strip().lower(), _normalized_output_text(message).strip()


def _iter_response_output_parts(response: Any) -> Sequence[Any]:
    """Return a flat list of message/content parts from a response."""

    parts: list[Any] = []
    output_items = _mapping_get_or_attr(response, "output", None)
    if not isinstance(output_items, Sequence) or isinstance(output_items, (str, bytes, bytearray)):
        return parts
    for item in output_items:
        direct_refusal = _mapping_get_or_attr(item, "refusal", None)
        if direct_refusal not in (None, ""):
            parts.append({"type": "refusal", "refusal": direct_refusal})
        content = _mapping_get_or_attr(item, "content", None)
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
            parts.extend(content)
    return parts


def _response_refusal_text(response: Any) -> str:
    """Extract a programmatically detectable refusal text when present."""

    direct_refusal = _mapping_get_or_attr(response, "refusal", None)
    if direct_refusal not in (None, ""):
        return _normalized_output_text(direct_refusal).strip()

    refusal_chunks: list[str] = []
    for part in _iter_response_output_parts(response):
        part_type = str(_mapping_get_or_attr(part, "type", "") or "").strip().lower()
        if part_type != "refusal":
            continue
        refusal_text = _normalized_output_text(
            _mapping_get_or_attr(part, "refusal", None) or _mapping_get_or_attr(part, "text", None)
        ).strip()
        if refusal_text:
            refusal_chunks.append(refusal_text)
    return "\n".join(refusal_chunks).strip()


def _response_failed_retryable(response: Any) -> bool:
    """Return whether a failed response should be retried."""

    code, message = _response_error_code_and_message(response)
    combined = f"{code} {message}".lower().replace("_", " ").replace("-", " ")
    return any(marker in combined for marker in _RETRYABLE_ERROR_MESSAGE_MARKERS)


def _next_retry_max_output_tokens(current: int) -> int:
    """Grow the retry token budget without exceeding the hard ceiling."""

    return max(current, min(_MAX_STRUCTURED_RESPONSE_OUTPUT_TOKENS, max(current * 2, current + 400)))


def _normalized_output_text(output_text: Any) -> str:
    """Coerce extracted model output into a text string."""

    if output_text is None:
        return ""
    if isinstance(output_text, str):
        return output_text
    if isinstance(output_text, bytes):
        return output_text.decode("utf-8", errors="replace")
    return str(output_text)


def _output_text_digest(output_text: str) -> str:
    """Return a short stable digest for privacy-safe diagnostics."""

    return hashlib.sha256(output_text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _structured_response_error_message(
    *,
    schema_name: str,
    response: Any,
    output_text: str,
    max_output_tokens: int,
    attempted_output_token_limits: Sequence[int],
    error_detail: str | None = None,
) -> str:
    """Format a redacted error message for structured-response failures."""

    status = _response_status(response) or "unknown"
    reason = _incomplete_reason(response) or "unknown"
    response_id = _response_id(response)
    error_code, error_message = _response_error_code_and_message(response)
    parts = [
        f"Structured JSON response for {schema_name!r} was not parseable",
        f"status={status}",
        f"incomplete_reason={reason}",
        f"max_output_tokens={max_output_tokens}",
        f"attempted_output_token_limits={list(attempted_output_token_limits)!r}",
        f"output_text_len={len(output_text)}",
        f"output_text_sha256_16={_output_text_digest(output_text)}",
    ]
    if response_id:
        parts.append(f"response_id={response_id}")
    if error_code:
        parts.append(f"response_error_code={error_code}")
    if error_message:
        parts.append(f"response_error_len={len(error_message)}")
        parts.append(f"response_error_sha256_16={_output_text_digest(error_message)}")
    if error_detail:
        parts.append(f"error={error_detail}")
    return " (".join((parts[0], ", ".join(parts[1:]))) + ")"


def _validated_schema_name(schema_name: str) -> str:
    """Validate and normalize the logical schema name used in requests."""

    if not isinstance(schema_name, str):
        raise TypeError("schema_name must be a string")
    normalized_schema_name = schema_name.strip()
    if not normalized_schema_name:
        raise ValueError("schema_name must not be empty")
    return normalized_schema_name


def _validated_max_output_tokens(value: object) -> int:
    """Validate the configured ``max_output_tokens`` value."""

    if isinstance(value, bool):
        raise TypeError("max_output_tokens must be an integer >= 1, not bool")
    try:
        normalized_value = operator.index(value)
    except TypeError as exc:
        raise TypeError("max_output_tokens must be an integer >= 1") from exc
    if normalized_value < 1:
        raise ValueError("max_output_tokens must be >= 1")
    return normalized_value


def _schema_path(path: str, key: str) -> str:
    """Format a nested schema path for diagnostics."""

    if _SIMPLE_PATH_KEY_RE.match(key):
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def _jsonschema_error_path(error: Any) -> str:
    """Format a ValidationError path using the module's historical style."""

    path = "$"
    for part in getattr(error, "absolute_path", ()):
        if isinstance(part, int):
            path = f"{path}[{part}]"
            continue
        path = _schema_path(path, str(part))
    return path


def _structured_response_validate_formats_enabled(backend: Any) -> bool:
    """Return whether local validation should assert JSON Schema format keywords."""

    config = getattr(backend, "config", None)
    raw_value = getattr(config, "structured_response_validate_formats", False)
    return bool(raw_value)


@lru_cache(maxsize=64)
def _cached_jsonschema_validator(schema_json: str, enforce_formats: bool) -> Any:
    """Build and cache a jsonschema validator for a canonicalized schema."""

    schema_payload = json.loads(schema_json)
    validator_cls = validator_for(schema_payload, default=Draft202012Validator)
    validator_cls.check_schema(schema_payload)
    format_checker = validator_cls.FORMAT_CHECKER if enforce_formats else None
    return validator_cls(schema_payload, format_checker=format_checker)


def _validated_schema_object(
    schema: Mapping[str, object], *, enforce_formats: bool
) -> tuple[dict[str, object], str, Any]:
    """Validate, deep-copy, canonicalize, and compile the schema mapping."""

    if not isinstance(schema, Mapping):
        raise TypeError("schema must be a mapping")
    try:
        schema_json = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        schema_payload = json.loads(schema_json)
    except (TypeError, ValueError) as exc:
        raise TypeError("schema must be JSON-serializable") from exc
    try:
        validator = _cached_jsonschema_validator(schema_json, enforce_formats)
    except jsonschema_exceptions.SchemaError as exc:
        raise ValueError(f"schema is invalid: {exc.message}") from exc
    return dict(schema_payload), schema_json, validator


def _request_timeout_seconds(backend: Any) -> float:
    """Resolve the structured-response timeout from backend config."""

    config = getattr(backend, "config", None)
    raw_timeout = None
    for attr_name in (
        "structured_response_timeout_seconds",
        "response_timeout_seconds",
        "request_timeout_seconds",
        "openai_timeout_seconds",
    ):
        candidate = getattr(config, attr_name, None)
        if candidate is not None:
            raw_timeout = candidate
            break
    if raw_timeout is None:
        return _DEFAULT_STRUCTURED_RESPONSE_TIMEOUT_SECONDS
    if isinstance(raw_timeout, bool):
        return _DEFAULT_STRUCTURED_RESPONSE_TIMEOUT_SECONDS
    try:
        timeout_seconds = float(raw_timeout)
    except (TypeError, ValueError):
        return _DEFAULT_STRUCTURED_RESPONSE_TIMEOUT_SECONDS
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        return _DEFAULT_STRUCTURED_RESPONSE_TIMEOUT_SECONDS
    return timeout_seconds


def _merge_structured_text_format(
    request: Mapping[str, object],
    *,
    schema_name: str,
    schema_payload: Mapping[str, object],
) -> dict[str, object]:
    """Merge JSON-schema text formatting into a response request payload."""

    merged_request = dict(request)
    existing_text = merged_request.get("text")
    text_config: dict[str, object]
    if isinstance(existing_text, Mapping):
        text_config = dict(existing_text)
    else:
        text_config = {}
    text_config["format"] = {
        "type": "json_schema",
        "name": schema_name,
        "schema": dict(schema_payload),
        "strict": True,
    }
    merged_request["text"] = text_config
    return merged_request


def _is_timeout_keyword_error(exc: TypeError) -> bool:
    """Return whether a ``TypeError`` came from an unsupported timeout kwarg."""

    message = str(exc).lower()
    return "timeout" in message and ("unexpected keyword" in message or "got an unexpected keyword argument" in message)


def _is_retryable_response_exception(exc: Exception) -> bool:
    """Return whether a transport exception is worth retrying."""

    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    if any(marker in name for marker in _RETRYABLE_ERROR_NAME_MARKERS):
        return True
    return any(marker in message for marker in _RETRYABLE_ERROR_MESSAGE_MARKERS)


def _call_with_timeout_via_client_options(
    client: Any, request_payload: Mapping[str, object], timeout_seconds: float
) -> Any:
    """Try to apply a timeout via SDK client options when create(timeout=...) is unsupported."""

    with_options = getattr(client, "with_options", None)
    if not callable(with_options):
        raise TypeError("client has no with_options(timeout=...) support")
    timeout_client = with_options(timeout=timeout_seconds)
    responses_api = getattr(timeout_client, "responses", None)
    create_method = getattr(responses_api, "create", None)
    if not callable(create_method):
        raise TypeError("client.with_options(timeout=...) did not yield responses.create")
    return create_method(**dict(request_payload))


def _call_structured_response_create(
    backend: Any, request: Mapping[str, object], *, timeout_seconds: float
) -> Any:
    """Call the backend structured-response API with bounded retry handling."""

    request_payload = dict(request)
    client = getattr(backend, "_client", None)
    responses_api = getattr(client, "responses", None)
    create_method = getattr(responses_api, "create", None)
    if not callable(create_method):
        raise AttributeError("backend._client.responses.create is not callable")

    retry_index = 0
    while True:
        try:
            if "timeout" in request_payload:
                return create_method(**request_payload)
            return create_method(**dict(request_payload, timeout=timeout_seconds))
        except TypeError as exc:
            if not _is_timeout_keyword_error(exc):
                raise
            try:
                request_payload_without_timeout = dict(request_payload)
                request_payload_without_timeout.pop("timeout", None)
                return _call_with_timeout_via_client_options(
                    client,
                    request_payload_without_timeout,
                    timeout_seconds,
                )
            except Exception as timeout_exc:
                # BREAKING: Fail closed instead of silently removing the timeout in a safety-critical agent.
                raise RuntimeError(
                    "Structured-response backend does not support a bounded timeout. "
                    "Expose create(timeout=...) or client.with_options(timeout=...)."
                ) from timeout_exc
        except Exception as exc:
            if retry_index >= len(_TRANSPORT_RETRY_BACKOFF_SECONDS) or not _is_retryable_response_exception(exc):
                raise
            time.sleep(_TRANSPORT_RETRY_BACKOFF_SECONDS[retry_index])
            retry_index += 1


def _coerce_parsed_object(value: Any) -> dict[str, object] | None:
    """Coerce SDK-parsed output into a plain dict when possible."""

    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped_value = model_dump()
        if isinstance(dumped_value, Mapping):
            return dict(dumped_value)
    return None


def _validated_candidate_object(candidate: object, validator: Any) -> dict[str, object]:
    """Validate a candidate payload with jsonschema and require a top-level object."""

    errors = list(validator.iter_errors(candidate))
    if errors:
        best_error = jsonschema_exceptions.best_match(errors)
        raise ValueError(f"{_jsonschema_error_path(best_error)}: {best_error.message}")
    if not isinstance(candidate, Mapping):
        raise ValueError(f"$: expected top-level object, got {type(candidate).__name__}")
    return dict(candidate)


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
    """Request one structured JSON object from a backend and validate it.

    Args:
        backend: Backend object that exposes `_build_response_request`,
            `_client.responses.create`, and `_extract_output_text`.
        prompt: User-facing prompt content sent to the backend.
        instructions: System or developer instructions paired with the prompt.
        schema_name: Logical name reported to the model for the JSON schema.
        schema: JSON Schema object that the response must satisfy.
        model: Optional explicit model override.
        reasoning_effort: Backend reasoning-effort hint. Defaults to ``"low"``.
        max_output_tokens: Initial response token budget before bounded retries.

    Returns:
        A validated JSON object that satisfies ``schema``.

    Raises:
        TypeError: If ``schema`` or ``max_output_tokens`` has an invalid type.
        ValueError: If the backend response cannot be produced, parsed, or
            validated against the schema.
    """

    normalized_schema_name = _validated_schema_name(schema_name)
    schema_payload, _schema_json, validator = _validated_schema_object(
        schema,
        enforce_formats=_structured_response_validate_formats_enabled(backend),
    )
    current_max_output_tokens = _validated_max_output_tokens(max_output_tokens)
    timeout_seconds = _request_timeout_seconds(backend)

    response: Any | None = None
    output_text = ""
    last_error: Exception | None = None
    attempted_output_token_limits: list[int] = []
    failed_response_retry_index = 0

    while True:
        attempted_output_token_limits.append(current_max_output_tokens)
        try:
            request = backend._build_response_request(
                prompt,
                instructions=instructions,
                allow_web_search=False,
                model=model or backend.config.default_model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=current_max_output_tokens,
            )
        except Exception as exc:
            raise ValueError(
                _structured_response_error_message(
                    schema_name=normalized_schema_name,
                    response=response,
                    output_text=output_text,
                    max_output_tokens=current_max_output_tokens,
                    attempted_output_token_limits=attempted_output_token_limits,
                    error_detail=f"request_build_failed:{exc.__class__.__name__}",
                )
            ) from exc

        request = _merge_structured_text_format(
            request,
            schema_name=normalized_schema_name,
            schema_payload=schema_payload,
        )

        try:
            response = _call_structured_response_create(
                backend,
                request,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            raise ValueError(
                _structured_response_error_message(
                    schema_name=normalized_schema_name,
                    response=response,
                    output_text=output_text,
                    max_output_tokens=current_max_output_tokens,
                    attempted_output_token_limits=attempted_output_token_limits,
                    error_detail=f"response_create_failed:{exc.__class__.__name__}",
                )
            ) from exc

        status = _response_status(response)
        reason = _incomplete_reason(response)

        if status == "incomplete":
            if reason == "max_output_tokens":
                next_limit = _next_retry_max_output_tokens(current_max_output_tokens)
                if next_limit <= current_max_output_tokens:
                    last_error = ValueError("response incomplete: max_output_tokens")
                    break
                current_max_output_tokens = next_limit
                continue
            last_error = ValueError(f"response incomplete: {reason or 'unknown'}")
            break

        if status == "failed":
            if failed_response_retry_index < len(_TRANSPORT_RETRY_BACKOFF_SECONDS) and _response_failed_retryable(response):
                time.sleep(_TRANSPORT_RETRY_BACKOFF_SECONDS[failed_response_retry_index])
                failed_response_retry_index += 1
                continue
            error_code, error_message = _response_error_code_and_message(response)
            if error_code or error_message:
                redacted = []
                if error_code:
                    redacted.append(f"code={error_code}")
                if error_message:
                    redacted.append(
                        f"message_len={len(error_message)} message_sha256_16={_output_text_digest(error_message)}"
                    )
                last_error = ValueError(f"response failed: {' '.join(redacted)}")
            else:
                last_error = ValueError("response failed")
            break

        refusal_text = _response_refusal_text(response)
        if refusal_text:
            last_error = ValueError(
                f"model refusal: refusal_len={len(refusal_text)} refusal_sha256_16={_output_text_digest(refusal_text)}"
            )
            break

        parsed = _coerce_parsed_object(_mapping_get_or_attr(response, "output_parsed", None))
        if parsed is not None:
            try:
                return _validated_candidate_object(parsed, validator)
            except ValueError as exc:
                last_error = exc

        try:
            output_text = _normalized_output_text(backend._extract_output_text(response))
        except Exception as exc:
            raise ValueError(
                _structured_response_error_message(
                    schema_name=normalized_schema_name,
                    response=response,
                    output_text=output_text,
                    max_output_tokens=current_max_output_tokens,
                    attempted_output_token_limits=attempted_output_token_limits,
                    error_detail=f"output_text_extraction_failed:{exc.__class__.__name__}",
                )
            ) from exc

        if status not in ("", "completed"):
            last_error = ValueError(f"unexpected response status: {status}")
            break

        if not output_text.strip():
            if last_error is None:
                last_error = ValueError("response produced no parsed object and no output text")
            break

        try:
            candidate = extract_json_object(output_text)
            return _validated_candidate_object(candidate, validator)
        except ValueError as exc:
            if last_error is None:
                last_error = exc
            else:
                last_error = ValueError(f"{last_error}; fallback_parse_failed:{exc}")
            break

    raise ValueError(
        _structured_response_error_message(
            schema_name=normalized_schema_name,
            response=response,
            output_text=output_text,
            max_output_tokens=current_max_output_tokens,
            attempted_output_token_limits=attempted_output_token_limits,
            error_detail=None if last_error is None else f"{last_error.__class__.__name__}:{last_error}",
        )
    ) from last_error


__all__ = ["request_structured_json_object"]