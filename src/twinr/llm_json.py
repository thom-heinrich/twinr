"""Request structured JSON outputs from model backends with local validation.

This module wraps backend structured-response calls with bounded retries,
timeout handling, privacy-safe diagnostics, and a fallback path that extracts
and validates JSON objects from raw model text.
"""

from __future__ import annotations

import hashlib
import math
import operator
import re
import time
from typing import Any, Mapping, Sequence

from twinr.text_utils import extract_json_object

_MAX_STRUCTURED_RESPONSE_OUTPUT_TOKENS = 4096  # AUDIT-FIX(#6): Centralize the retry ceiling so the final expanded token budget is actually attempted.
_DEFAULT_STRUCTURED_RESPONSE_TIMEOUT_SECONDS = 30.0  # AUDIT-FIX(#4): Add a bounded default timeout for network calls in the single-process agent.
_TRANSPORT_RETRY_BACKOFF_SECONDS = (0.25, 0.5)  # AUDIT-FIX(#3): Use short bounded backoff for transient upstream failures.
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


def _incomplete_reason(response: Any) -> str:
    """Normalize the response's incomplete reason for retry decisions."""

    details = _mapping_get_or_attr(response, "incomplete_details", None)
    if isinstance(details, Mapping):
        return str(details.get("reason", "") or "").strip().lower()
    return str(_mapping_get_or_attr(details, "reason", "") or "").strip().lower()


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
    parts = [
        f"Structured JSON response for {schema_name!r} was not parseable",
        f"status={status}",
        f"incomplete_reason={reason}",
        f"max_output_tokens={max_output_tokens}",
        f"attempted_output_token_limits={list(attempted_output_token_limits)!r}",
        f"output_text_len={len(output_text)}",
        f"output_text_sha256_16={_output_text_digest(output_text)}",
    ]
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


def _validated_schema_object(schema: Mapping[str, object]) -> dict[str, object]:
    """Validate and copy the schema mapping for downstream use."""

    if not isinstance(schema, Mapping):
        raise TypeError("schema must be a mapping")
    return dict(schema)


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


def _call_structured_response_create(backend: Any, request: Mapping[str, object], *, timeout_seconds: float) -> Any:
    """Call the backend structured-response API with bounded retry handling."""

    request_payload = dict(request)
    added_timeout = False
    if "timeout" not in request_payload:
        request_payload["timeout"] = timeout_seconds
        added_timeout = True

    retry_index = 0
    while True:
        try:
            return backend._client.responses.create(**request_payload)
        except TypeError as exc:
            if added_timeout and _is_timeout_keyword_error(exc):
                request_payload.pop("timeout", None)
                added_timeout = False
                continue
            raise
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


def _resolve_local_json_pointer(root_schema: Mapping[str, object], ref: str) -> Mapping[str, object]:
    """Resolve a local JSON Schema reference against the root schema."""

    if not ref.startswith("#/"):
        raise ValueError(f"Unsupported JSON schema reference: {ref!r}")
    current: Any = root_schema
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"Unresolvable JSON schema reference: {ref!r}")
        current = current[part]
    if not isinstance(current, Mapping):
        raise ValueError(f"JSON schema reference does not resolve to an object schema: {ref!r}")
    return current


def _normalized_schema_types(raw_types: object) -> tuple[str, ...] | None:
    """Normalize a schema ``type`` declaration into a tuple of names."""

    if isinstance(raw_types, str):
        return (raw_types,)
    if isinstance(raw_types, list) and all(isinstance(item, str) for item in raw_types):
        return tuple(raw_types)
    return None


def _json_type_matches(value: object, expected_type: str) -> bool:
    """Return whether a JSON-compatible value matches a schema type name."""

    if expected_type == "object":
        return isinstance(value, Mapping)
    if expected_type == "array":
        return isinstance(value, (list, tuple))
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return ((isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)) and math.isfinite(float(value))
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return False


def _schema_path(path: str, key: str) -> str:
    """Format a nested schema path for diagnostics."""

    if _SIMPLE_PATH_KEY_RE.match(key):
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def _optional_non_negative_int(value: object) -> int | None:
    """Coerce optional non-negative integer schema constraints."""

    if isinstance(value, bool):
        return None
    try:
        normalized_value = operator.index(value)
    except TypeError:
        return None
    if normalized_value < 0:
        return None
    return normalized_value


def _optional_json_number(value: object) -> float | int | None:
    """Coerce optional finite JSON numeric schema constraints."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _validate_embedded_schema(
    value: object,
    schema: object,
    *,
    root_schema: Mapping[str, object],
    path: str,
    depth: int,
) -> None:
    """Validate a nested schema fragment or boolean schema against a value."""

    if schema is True:
        return
    if schema is False:
        raise ValueError(f"{path}: schema forbids this value")
    if not isinstance(schema, Mapping):
        raise ValueError(f"{path}: embedded schema must be a mapping or bool, got {type(schema).__name__}")
    _validate_json_against_schema(value, schema, root_schema=root_schema, path=path, depth=depth)


def _validate_string_constraints(value: str, schema: Mapping[str, object], *, path: str) -> None:
    """Validate string-specific JSON Schema constraints."""

    min_length = _optional_non_negative_int(schema.get("minLength"))
    max_length = _optional_non_negative_int(schema.get("maxLength"))
    if min_length is not None and len(value) < min_length:
        raise ValueError(f"{path}: string shorter than minLength={min_length}")
    if max_length is not None and len(value) > max_length:
        raise ValueError(f"{path}: string longer than maxLength={max_length}")
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and re.search(pattern, value) is None:
        raise ValueError(f"{path}: string does not match pattern {pattern!r}")


def _validate_numeric_constraints(value: int | float, schema: Mapping[str, object], *, path: str) -> None:
    """Validate numeric JSON Schema constraints."""

    minimum = _optional_json_number(schema.get("minimum"))
    maximum = _optional_json_number(schema.get("maximum"))
    exclusive_minimum = _optional_json_number(schema.get("exclusiveMinimum"))
    exclusive_maximum = _optional_json_number(schema.get("exclusiveMaximum"))
    multiple_of = _optional_json_number(schema.get("multipleOf"))

    if minimum is not None and value < minimum:
        raise ValueError(f"{path}: value smaller than minimum={minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{path}: value larger than maximum={maximum}")
    if exclusive_minimum is not None and value <= exclusive_minimum:
        raise ValueError(f"{path}: value must be > {exclusive_minimum}")
    if exclusive_maximum is not None and value >= exclusive_maximum:
        raise ValueError(f"{path}: value must be < {exclusive_maximum}")
    if multiple_of is not None:
        if multiple_of <= 0:
            raise ValueError(f"{path}: schema multipleOf must be > 0")
        quotient = float(value) / float(multiple_of)
        if not math.isclose(quotient, round(quotient), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{path}: value is not a multiple of {multiple_of}")


def _validate_object_constraints(
    value: Mapping[str, object],
    schema: Mapping[str, object],
    *,
    root_schema: Mapping[str, object],
    path: str,
    depth: int,
) -> None:
    """Validate object-specific JSON Schema constraints."""

    min_properties = _optional_non_negative_int(schema.get("minProperties"))
    max_properties = _optional_non_negative_int(schema.get("maxProperties"))
    if min_properties is not None and len(value) < min_properties:
        raise ValueError(f"{path}: object has fewer than minProperties={min_properties}")
    if max_properties is not None and len(value) > max_properties:
        raise ValueError(f"{path}: object has more than maxProperties={max_properties}")

    properties = schema.get("properties")
    property_schemas = properties if isinstance(properties, Mapping) else {}
    required = schema.get("required")
    if isinstance(required, list):
        for required_key in required:
            if isinstance(required_key, str) and required_key not in value:
                raise ValueError(f"{_schema_path(path, required_key)}: missing required property")

    additional_properties = schema.get("additionalProperties", True)
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{path}: object keys must be strings, got {type(key).__name__}")
        item_path = _schema_path(path, key)
        item_schema = property_schemas.get(key)
        if item_schema is not None:
            _validate_embedded_schema(item, item_schema, root_schema=root_schema, path=item_path, depth=depth + 1)
            continue
        if additional_properties is False:
            raise ValueError(f"{item_path}: additional properties are not allowed")
        if additional_properties is True:
            continue
        _validate_embedded_schema(item, additional_properties, root_schema=root_schema, path=item_path, depth=depth + 1)


def _validate_array_constraints(
    value: Sequence[object],
    schema: Mapping[str, object],
    *,
    root_schema: Mapping[str, object],
    path: str,
    depth: int,
) -> None:
    """Validate array-specific JSON Schema constraints."""

    min_items = _optional_non_negative_int(schema.get("minItems"))
    max_items = _optional_non_negative_int(schema.get("maxItems"))
    if min_items is not None and len(value) < min_items:
        raise ValueError(f"{path}: array has fewer than minItems={min_items}")
    if max_items is not None and len(value) > max_items:
        raise ValueError(f"{path}: array has more than maxItems={max_items}")

    items = schema.get("items")
    if items is False:
        if value:
            raise ValueError(f"{path}: array items are not allowed")
        return
    if items is True:
        return
    if isinstance(items, Mapping):
        for index, item in enumerate(value):
            _validate_json_against_schema(item, items, root_schema=root_schema, path=f"{path}[{index}]", depth=depth + 1)
        return
    if isinstance(items, list):
        for index, item in enumerate(value):
            if index < len(items):
                _validate_embedded_schema(item, items[index], root_schema=root_schema, path=f"{path}[{index}]", depth=depth + 1)
                continue
            additional_items = schema.get("additionalItems", True)
            if additional_items is False:
                raise ValueError(f"{path}[{index}]: additional array items are not allowed")
            if additional_items is True:
                continue
            _validate_embedded_schema(item, additional_items, root_schema=root_schema, path=f"{path}[{index}]", depth=depth + 1)


def _validate_json_against_schema(
    value: object,
    schema: Mapping[str, object],
    *,
    root_schema: Mapping[str, object],
    path: str,
    depth: int = 0,
) -> None:
    """Validate a JSON-compatible value against a local JSON Schema subset."""

    if depth > 64:
        raise ValueError(f"{path}: schema validation exceeded maximum recursion depth")

    ref = schema.get("$ref")
    if isinstance(ref, str):
        resolved_schema = _resolve_local_json_pointer(root_schema, ref)
        _validate_json_against_schema(value, resolved_schema, root_schema=root_schema, path=path, depth=depth + 1)
        return

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for index, sub_schema in enumerate(all_of):
            _validate_embedded_schema(value, sub_schema, root_schema=root_schema, path=path, depth=depth + 1)

    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        any_of_errors: list[str] = []
        for sub_schema in any_of:
            try:
                _validate_embedded_schema(value, sub_schema, root_schema=root_schema, path=path, depth=depth + 1)
                break
            except ValueError as exc:
                any_of_errors.append(str(exc))
        else:
            raise ValueError(f"{path}: value does not satisfy anyOf ({'; '.join(any_of_errors)})")

    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        match_count = 0
        one_of_errors: list[str] = []
        for sub_schema in one_of:
            try:
                _validate_embedded_schema(value, sub_schema, root_schema=root_schema, path=path, depth=depth + 1)
            except ValueError as exc:
                one_of_errors.append(str(exc))
            else:
                match_count += 1
        if match_count != 1:
            raise ValueError(f"{path}: value must satisfy exactly one schema in oneOf (matches={match_count}, errors={one_of_errors})")

    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: value does not match const={schema['const']!r}")

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        raise ValueError(f"{path}: value is not in enum={enum_values!r}")

    expected_types = _normalized_schema_types(schema.get("type"))
    if expected_types is not None and not any(_json_type_matches(value, expected_type) for expected_type in expected_types):
        raise ValueError(f"{path}: expected type {expected_types!r}, got {type(value).__name__}")

    if value is None:
        return

    if isinstance(value, Mapping):
        if expected_types is None or "object" in expected_types or any(
            key in schema for key in ("properties", "required", "additionalProperties", "minProperties", "maxProperties")
        ):
            _validate_object_constraints(value, schema, root_schema=root_schema, path=path, depth=depth)
        return

    if isinstance(value, (list, tuple)):
        if expected_types is None or "array" in expected_types or any(key in schema for key in ("items", "minItems", "maxItems")):
            _validate_array_constraints(value, schema, root_schema=root_schema, path=path, depth=depth)
        return

    if isinstance(value, str):
        if expected_types is None or "string" in expected_types or any(key in schema for key in ("minLength", "maxLength", "pattern")):
            _validate_string_constraints(value, schema, path=path)
        return

    if isinstance(value, bool):
        return

    if isinstance(value, int):
        if expected_types is None or "integer" in expected_types or "number" in expected_types or any(
            key in schema for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf")
        ):
            _validate_numeric_constraints(value, schema, path=path)
        return

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: numeric value must be finite")
        if expected_types is None or "number" in expected_types or any(
            key in schema for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf")
        ):
            _validate_numeric_constraints(value, schema, path=path)
        return


def _validated_candidate_object(candidate: object, schema: Mapping[str, object]) -> dict[str, object]:
    """Validate a candidate payload and require a top-level object."""

    _validate_json_against_schema(candidate, schema, root_schema=schema, path="$")
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

    normalized_schema_name = _validated_schema_name(schema_name)  # AUDIT-FIX(#5): Fail fast on empty/invalid schema names instead of shipping malformed requests downstream.
    schema_payload = _validated_schema_object(schema)  # AUDIT-FIX(#1): Validate the schema container once and reuse it for request construction and fallback validation.
    current_max_output_tokens = _validated_max_output_tokens(max_output_tokens)  # AUDIT-FIX(#5): Reject silent coercions from bool/str/float config mistakes.
    timeout_seconds = _request_timeout_seconds(backend)  # AUDIT-FIX(#4): Enforce bounded network waits with backward-compatible config overrides.

    response: Any | None = None
    output_text = ""
    last_error: Exception | None = None
    attempted_output_token_limits: list[int] = []  # AUDIT-FIX(#6): Track each token budget that was actually attempted for accurate diagnostics.

    while True:
        attempted_output_token_limits.append(current_max_output_tokens)  # AUDIT-FIX(#6): Keep retry accounting aligned with the real request sequence.
        try:
            request = backend._build_response_request(
                prompt,
                instructions=instructions,
                allow_web_search=False,
                model=model or backend.config.default_model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=current_max_output_tokens,
            )
        except Exception as exc:  # AUDIT-FIX(#3): Preserve backend request-build failures with stable redacted context instead of crashing naked.
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
        )  # AUDIT-FIX(#7): Preserve existing text configuration instead of clobbering backend defaults/future SDK fields.

        try:
            response = _call_structured_response_create(
                backend,
                request,
                timeout_seconds=timeout_seconds,
            )  # AUDIT-FIX(#3,#4): Add bounded timeout-aware API retries for transient upstream failures.
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
        if status == "incomplete":  # AUDIT-FIX(#1): Never trust or salvage incomplete model outputs; retry only for token exhaustion, otherwise fail closed.
            if reason == "max_output_tokens":
                next_limit = _next_retry_max_output_tokens(current_max_output_tokens)
                if next_limit <= current_max_output_tokens:
                    break
                current_max_output_tokens = next_limit
                continue
            last_error = ValueError(f"response incomplete: {reason or 'unknown'}")
            break

        parsed = _coerce_parsed_object(_mapping_get_or_attr(response, "output_parsed", None))  # AUDIT-FIX(#8): Accept Mapping/model_dump structured outputs, not just plain dict instances.
        if parsed is not None:
            try:
                return _validated_candidate_object(parsed, schema_payload)  # AUDIT-FIX(#1): Enforce schema compliance even on SDK-parsed outputs.
            except ValueError as exc:
                last_error = exc

        try:
            output_text = _normalized_output_text(backend._extract_output_text(response))  # AUDIT-FIX(#8): Normalize non-string/bytes output safely before parsing and diagnostics.
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

        try:
            candidate = extract_json_object(output_text)
            return _validated_candidate_object(candidate, schema_payload)  # AUDIT-FIX(#1): Validate extracted fallback JSON against the declared schema before returning it.
        except ValueError as exc:
            if last_error is None:
                last_error = exc
            else:
                last_error = ValueError(f"{last_error}; fallback_parse_failed:{exc}")  # AUDIT-FIX(#1): Preserve the stronger schema-validation error when raw-text fallback adds no better signal.
            break

    raise ValueError(
        _structured_response_error_message(
            schema_name=normalized_schema_name,
            response=response,
            output_text=output_text,
            max_output_tokens=current_max_output_tokens,
            attempted_output_token_limits=attempted_output_token_limits,
            error_detail=None if last_error is None else f"{last_error.__class__.__name__}:{last_error}",
        )  # AUDIT-FIX(#2): Emit privacy-safe diagnostics using metadata and hashes, not raw model output tails.
    ) from last_error


__all__ = ["request_structured_json_object"]
