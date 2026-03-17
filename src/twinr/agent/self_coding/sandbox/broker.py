"""Bridge sandboxed child code to the bounded parent-side skill context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import inspect
import math
import threading
from typing import Any


_DEFAULT_RPC_TIMEOUT_SECONDS = 60.0
_TRUE_TEXT_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_TEXT_VALUES = frozenset({"0", "false", "no", "off"})


class BrokerProtocolError(RuntimeError):
    """Raised when the broker payload or transport violates the protocol."""


@dataclass(frozen=True, slots=True)
class SkillBrokerPolicy:
    """Describe the ctx-method surface exposed to sandboxed child code."""

    allowed_methods: frozenset[str] = frozenset(
        {
            "current_presence_session_id",
            "delete_json",
            "is_night_mode",
            "is_private_for_speech",
            "load_json",
            "log_event",
            "now_iso",
            "say",
            "search_web",
            "store_json",
            "summarize_text",
            "today_local_date",
        }
    )

    def require_allowed(self, method_name: object) -> str:
        """Return the normalized method name or raise when the policy blocks it."""

        normalized = str(method_name or "").strip()
        if normalized not in self.allowed_methods:
            raise BrokerProtocolError(f"sandbox broker blocks ctx.{normalized or 'unknown'}")  # AUDIT-FIX(#1): Return a safe protocol error instead of crashing the broker loop.
        return normalized


@dataclass(frozen=True, slots=True)
class BrokeredSkillSearchResult:
    """Return one safe, JSON-serializable view of a live search result."""

    answer: str
    sources: tuple[str, ...]
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BrokeredSkillSearchResult":
        """Build one result object from a broker payload."""

        if not isinstance(payload, Mapping):
            raise BrokerProtocolError("sandbox search returned an invalid payload")  # AUDIT-FIX(#4): Reject malformed search payloads instead of iterating arbitrary objects.
        return cls(
            answer=str(payload.get("answer") or "").strip(),
            sources=_normalized_string_tuple(payload.get("sources")),
            response_id=_optional_text(payload.get("response_id")),
            request_id=_optional_text(payload.get("request_id")),
            model=_optional_text(payload.get("model")),
        )

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable broker payload."""

        return {
            "answer": self.answer,
            "sources": list(self.sources),
            "response_id": self.response_id,
            "request_id": self.request_id,
            "model": self.model,
        }


class ParentSkillContextBroker:
    """Dispatch broker requests onto the real parent-side runtime context."""

    def __init__(self, *, context: Any, policy: SkillBrokerPolicy | None = None) -> None:
        self.context = context
        self.policy = policy if policy is not None else SkillBrokerPolicy()

    def dispatch(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one brokered ctx-method call and return a response payload."""

        call_id = _normalize_call_id(payload) if isinstance(payload, Mapping) else ""
        if not isinstance(payload, Mapping):
            return _rpc_error_payload(call_id, "sandbox broker received an invalid payload")  # AUDIT-FIX(#1): Always return structured broker errors for malformed requests.

        if str(payload.get("kind") or "").strip() != "rpc_call":
            return _rpc_error_payload(call_id, "sandbox broker received an unexpected payload kind")  # AUDIT-FIX(#1): Guard the protocol surface explicitly.

        try:
            method_name = self.policy.require_allowed(payload.get("method"))
            args = _normalized_rpc_args(payload.get("args"))
            kwargs = _normalized_rpc_kwargs(payload.get("kwargs"))
            method = getattr(self.context, method_name, None)
            if not callable(method):
                raise BrokerProtocolError(f"sandbox context method is unavailable: {method_name}")  # AUDIT-FIX(#1): Fail closed when the broker surface is misconfigured.

            result = method(*args, **kwargs)
            if inspect.isawaitable(result):
                _close_awaitable_result(result)  # AUDIT-FIX(#5): Close accidental coroutine results so the broker does not leak un-awaited tasks or warnings.
                raise BrokerProtocolError(f"sandbox context method must be synchronous: {method_name}")  # AUDIT-FIX(#5): Reject accidental async ctx methods before they leak coroutines or blank results.

            if method_name == "search_web":
                result_payload = _search_result_to_payload(result)  # AUDIT-FIX(#5): Normalize mapping- or attribute-based search results and reject malformed awaitable outputs.
            else:
                result_payload = _ensure_json_compatible(result, path="result")  # AUDIT-FIX(#6): Keep the broker boundary JSON-safe instead of returning arbitrary Python objects.

            return {"kind": "rpc_result", "call_id": call_id, "result": result_payload}
        except BrokerProtocolError as exc:
            return _rpc_error_payload(call_id, str(exc))  # AUDIT-FIX(#1): Convert safe broker faults into rpc_error responses instead of propagating exceptions.
        except Exception:
            return _rpc_error_payload(call_id, "sandbox context call failed")  # AUDIT-FIX(#1): Avoid leaking parent-side exception details into the sandbox.


class SandboxSkillContextProxy:
    """Expose the same small ctx surface inside the sandboxed child process."""

    def __init__(self, *, connection: Any, rpc_timeout_seconds: float | None = _DEFAULT_RPC_TIMEOUT_SECONDS) -> None:
        send = getattr(connection, "send", None)
        recv = getattr(connection, "recv", None)
        poll = getattr(connection, "poll", None)
        normalized_timeout = _normalized_timeout_seconds(rpc_timeout_seconds)  # AUDIT-FIX(#3): Validate timeout configuration before enabling bounded IPC waits.
        if not callable(send) or not callable(recv):
            raise TypeError("sandbox broker connection must provide send() and recv()")  # AUDIT-FIX(#3): Validate the transport surface before first use.
        if normalized_timeout is not None and not callable(poll):
            raise TypeError("sandbox broker connection must provide poll() when rpc_timeout_seconds is enabled")  # AUDIT-FIX(#3): Require bounded waiting support for fail-fast IPC.

        self._connection = connection
        self._next_call_id = 0
        self._rpc_lock = threading.RLock()  # AUDIT-FIX(#7): Serialize access to the shared connection so send/recv pairs cannot interleave across callers.
        self._rpc_timeout_seconds = normalized_timeout

    def search_web(self, question: str, *, location_hint: str | None = None, date_context: str | None = None) -> BrokeredSkillSearchResult:
        payload = self._rpc(
            "search_web",
            question,
            location_hint=location_hint,
            date_context=date_context,
        )
        return BrokeredSkillSearchResult.from_payload(payload)

    def summarize_text(self, text: str, instructions: str | None = None) -> str:
        return str(self._rpc("summarize_text", text, instructions=instructions) or "").strip()

    def say(self, text: str) -> None:
        self._rpc("say", text)

    def store_json(self, key: str, value: Any) -> None:
        self._rpc("store_json", key, value)

    def load_json(self, key: str, default: Any | None = None) -> Any:
        return self._rpc("load_json", key, default)

    def delete_json(self, key: str) -> None:
        self._rpc("delete_json", key)

    def today_local_date(self) -> str:
        return str(self._rpc("today_local_date") or "").strip()

    def now_iso(self) -> str:
        return str(self._rpc("now_iso") or "").strip()

    def current_presence_session_id(self) -> int | None:
        result = self._rpc("current_presence_session_id")
        if result is None:
            return None
        if isinstance(result, bool):
            raise RuntimeError("sandbox broker returned an invalid presence session id")  # AUDIT-FIX(#8): Reject bool-as-int coercion at the trust boundary.
        if isinstance(result, int):
            return result
        if isinstance(result, str):
            text = result.strip()
            if text and (text.isdigit() or (text.startswith("-") and text[1:].isdigit())):
                return int(text)
        raise RuntimeError("sandbox broker returned an invalid presence session id")  # AUDIT-FIX(#8): Fail clearly on malformed session identifiers.

    def is_night_mode(self) -> bool:
        return _coerce_bool_result(self._rpc("is_night_mode"), method_name="is_night_mode")  # AUDIT-FIX(#2): Avoid silent truthiness bugs that can invert night-mode behavior.

    def is_private_for_speech(self) -> bool:
        return _coerce_bool_result(
            self._rpc("is_private_for_speech"),
            method_name="is_private_for_speech",
        )  # AUDIT-FIX(#2): Avoid privacy leaks caused by bool("False") == True.

    def log_event(self, event: str, *, severity: str = "info", **data: object) -> None:
        self._rpc("log_event", event, severity=severity, **data)

    def _rpc(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        normalized_method_name = str(method_name or "").strip()
        if not normalized_method_name:
            raise RuntimeError("sandbox broker method name is required")  # AUDIT-FIX(#1): Reject malformed child-side broker calls early.

        safe_args = [_ensure_json_compatible(arg, path=f"args[{index}]") for index, arg in enumerate(args)]  # AUDIT-FIX(#6): Keep outbound broker payloads JSON-compatible.
        safe_kwargs = _normalized_rpc_kwargs(kwargs)  # AUDIT-FIX(#4): Normalize kwargs deterministically before transport.

        with self._rpc_lock:
            self._next_call_id += 1
            call_id = str(self._next_call_id)

            try:
                self._connection.send(
                    {
                        "kind": "rpc_call",
                        "call_id": call_id,
                        "method": normalized_method_name,
                        "args": safe_args,
                        "kwargs": safe_kwargs,
                    }
                )
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise RuntimeError("sandbox broker connection failed while sending the request") from exc  # AUDIT-FIX(#3): Bound transport failures with a clear runtime error.

            if self._rpc_timeout_seconds is not None:
                try:
                    ready = bool(self._connection.poll(self._rpc_timeout_seconds))
                except (BrokenPipeError, EOFError, OSError, ValueError) as exc:
                    raise RuntimeError("sandbox broker connection failed while waiting for a response") from exc  # AUDIT-FIX(#3): Surface liveness failures instead of hanging forever.
                if not ready:
                    raise RuntimeError("sandbox broker call timed out")  # AUDIT-FIX(#3): Prevent indefinite child-process stalls on lost parent replies.

            try:
                response = self._connection.recv()
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise RuntimeError("sandbox broker connection failed while receiving the response") from exc  # AUDIT-FIX(#3): Handle parent disconnects cleanly.

        if not isinstance(response, Mapping):
            raise RuntimeError("sandbox broker returned an invalid response")  # AUDIT-FIX(#1): Validate the broker response envelope before use.
        if response.get("kind") == "rpc_error":
            raise RuntimeError(str(response.get("error") or "sandbox broker call failed"))  # AUDIT-FIX(#1): Preserve structured broker errors as child-side exceptions.
        if response.get("kind") != "rpc_result" or str(response.get("call_id") or "").strip() != call_id:
            raise RuntimeError("sandbox broker response did not match the request")  # AUDIT-FIX(#7): Detect response mismatches explicitly if a transport invariant is violated.
        return _ensure_json_compatible(response.get("result"), path="result")  # AUDIT-FIX(#6): Validate inbound result payloads even if the parent-side broker is bypassed or buggy.


def _coerce_bool_result(value: object, *, method_name: str) -> bool:
    # AUDIT-FIX(#2): Parse booleans strictly so privacy-sensitive flags cannot flip via Python truthiness rules.
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_TEXT_VALUES:
            return True
        if normalized in _FALSE_TEXT_VALUES:
            return False
    raise RuntimeError(f"sandbox broker returned an invalid boolean for {method_name}")


def _normalize_call_id(payload: Mapping[str, Any]) -> str:
    return str(payload.get("call_id") or "").strip()


def _normalized_rpc_args(value: object) -> tuple[Any, ...]:
    # AUDIT-FIX(#4): Reject malformed arg containers instead of iterating arbitrary iterables like strings one character at a time.
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise BrokerProtocolError("sandbox broker args must be a list or tuple")
    return tuple(_ensure_json_compatible(item, path=f"args[{index}]") for index, item in enumerate(value))


def _normalized_rpc_kwargs(value: object) -> dict[str, Any]:
    # AUDIT-FIX(#4): Enforce mapping-only kwargs with string keys for predictable method dispatch.
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BrokerProtocolError("sandbox broker kwargs must be a mapping")

    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise BrokerProtocolError("sandbox broker kwargs keys must be strings")
        key_text = key.strip()
        if not key_text:
            raise BrokerProtocolError("sandbox broker kwargs keys must be non-empty strings")
        normalized[key_text] = _ensure_json_compatible(item, path=f"kwargs[{key_text!r}]")
    return normalized


def _normalized_string_tuple(value: object) -> tuple[str, ...]:
    # AUDIT-FIX(#4): Normalize source collections without splitting bare strings into per-character entries.
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        text = str(value).strip()
        return (text,) if text else ()
    if not isinstance(value, (list, tuple)):
        raise BrokerProtocolError("sandbox broker string collection must be a list or tuple")

    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _normalized_timeout_seconds(value: float | None) -> float | None:
    # AUDIT-FIX(#3): Validate timeout configuration so the proxy never enters undefined wait states.
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("rpc_timeout_seconds must be a positive number or None")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("rpc_timeout_seconds must be a positive finite number")
    return timeout


def _rpc_error_payload(call_id: str, error: str) -> dict[str, str]:
    normalized_error = str(error or "sandbox broker call failed").strip() or "sandbox broker call failed"
    return {"kind": "rpc_error", "call_id": call_id, "error": normalized_error}


def _ensure_json_compatible(value: Any, *, path: str = "value", _seen: set[int] | None = None) -> Any:
    # AUDIT-FIX(#6): Constrain broker traffic to JSON-compatible primitives to avoid arbitrary object crossing at the sandbox boundary.
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BrokerProtocolError(f"{path} must be a finite float")
        return value

    if _seen is None:
        _seen = set()

    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in _seen:
            raise BrokerProtocolError(f"{path} must not contain reference cycles")
        _seen.add(value_id)
        try:
            normalized_mapping: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise BrokerProtocolError(f"{path} keys must be strings")
                normalized_mapping[key] = _ensure_json_compatible(item, path=f"{path}.{key}", _seen=_seen)
            return normalized_mapping
        finally:
            _seen.remove(value_id)

    if isinstance(value, (list, tuple)):
        value_id = id(value)
        if value_id in _seen:
            raise BrokerProtocolError(f"{path} must not contain reference cycles")
        _seen.add(value_id)
        try:
            return [_ensure_json_compatible(item, path=f"{path}[{index}]", _seen=_seen) for index, item in enumerate(value)]
        finally:
            _seen.remove(value_id)

    raise BrokerProtocolError(f"{path} must be JSON-compatible")


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _search_result_to_payload(result: object) -> dict[str, Any]:
    # AUDIT-FIX(#5): Support both mapping and attribute search results so serialization does not silently erase valid answers.
    if isinstance(result, Mapping):
        answer = result.get("answer")
        sources = result.get("sources")
        response_id = result.get("response_id")
        request_id = result.get("request_id")
        model = result.get("model")
    else:
        answer = getattr(result, "answer", "")
        sources = getattr(result, "sources", ())
        response_id = getattr(result, "response_id", None)
        request_id = getattr(result, "request_id", None)
        model = getattr(result, "model", None)

    return BrokeredSkillSearchResult(
        answer=str(answer or "").strip(),
        sources=_normalized_string_tuple(sources),
        response_id=_optional_text(response_id),
        request_id=_optional_text(request_id),
        model=_optional_text(model),
    ).to_payload()


def _close_awaitable_result(result: object) -> None:
    # AUDIT-FIX(#5): Best-effort cleanup for accidental coroutine returns from sync broker calls.
    close = getattr(result, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass