# CHANGELOG: 2026-03-30
# BUG-1: Fixed SSE parsing that incorrectly dispatched the last event on EOF even
#        when the stream did not end with the mandatory blank line.
# BUG-2: Fixed event-stream resume gaps by persisting the last Hue SSE event id
#        and sending Last-Event-ID on subsequent connections.
# BUG-3: Fixed silent resource loss in list_resources(); invalid list entries now
#        raise instead of being silently dropped.
# BUG-4: Fixed opaque error handling for non-2xx responses; structured Hue error
#        payloads and status codes are now surfaced instead of being discarded by
#        urlopen()-style exceptions.
# SEC-1: Fixed proxy-leak risk by disabling environment-derived proxies and other
#        trust_env behavior for local bridge traffic.
# SEC-2: Fixed path-segment injection risk by validating resource_type and
#        resource_id before constructing bridge paths.
# SEC-3: Fixed accidental cleartext transport risk by enforcing HTTPS base URLs.
# IMP-1: Replaced per-request urllib transport with a pooled persistent HTTPX
#        client, optional HTTP/2, bounded resource limits, and explicit close().
# IMP-2: Added response content-type validation and stricter SSE parsing aligned
#        with the WHATWG event-stream processing model.
# IMP-3: Added optional custom CA loading via config.ca_cert_path /
#        config.ca_cert_pem / config.ca_cert_data for deployments that want TLS
#        verification without globally disabling verify_tls.
# BREAKING: The default transport now requires the `httpx` package at runtime.
#           Install `httpx`, and optionally `httpx[http2]` for HTTP/2 support.
# BREAKING: config.base_url must now use HTTPS. Hue V2 no longer supports HTTP.

"""Speak the local Hue bridge JSON and event-stream APIs.

The client keeps raw HTTPS/SSE transport separate from Hue resource
normalization so the provider package can stay testable and future bridge
changes stay isolated to one place.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import codecs
import importlib.util
import json
import ssl
import threading
from typing import Any
from urllib.parse import quote, urlsplit

from twinr.integrations.smarthome.hue.models import HueBridgeConfig

JsonRequester = Callable[[str, str, Mapping[str, object] | None], dict[str, object]]
EventReader = Callable[[str, float, int], list[dict[str, object]]]

_SENTINEL = object()
_HTTP2_AVAILABLE = importlib.util.find_spec("h2") is not None


def _require_response_mapping(payload: object, *, field_name: str) -> dict[str, object]:
    """Require a mapping-shaped Hue response payload."""

    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{field_name} must be a mapping.")
    return {str(key): value for key, value in payload.items()}


def _require_httpx() -> Any:
    """Import httpx lazily so injected test doubles stay dependency-light."""

    try:
        import httpx  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "HueBridgeClient default transport requires the 'httpx' package. "
            "Install 'httpx' and optionally 'httpx[http2]' for HTTP/2 support."
        ) from exc
    return httpx


def _validate_base_url(value: object) -> str:
    """Validate the configured bridge base URL."""

    base_url = str(value).strip()
    if not base_url:
        raise ValueError("Hue bridge base_url must not be empty.")

    parts = urlsplit(base_url)
    if parts.scheme.lower() != "https":
        raise ValueError("Hue bridge base_url must use HTTPS.")
    if not parts.netloc:
        raise ValueError("Hue bridge base_url must include a host.")
    if parts.query or parts.fragment:
        raise ValueError("Hue bridge base_url must not include a query or fragment.")
    return base_url.rstrip("/")


def _validate_application_key(value: object) -> str:
    """Validate the configured Hue application key."""

    application_key = str(value).strip()
    if not application_key:
        raise ValueError("Hue bridge application_key must not be empty.")
    if any(ord(char) < 33 or ord(char) == 127 for char in application_key):
        raise ValueError(
            "Hue bridge application_key must not contain whitespace or control characters."
        )
    return application_key


def _validate_path_segment(value: str, *, field_name: str) -> str:
    """Validate and encode one bridge path segment."""

    segment = value.strip()
    if not segment:
        raise ValueError(f"{field_name} must not be empty.")
    if segment in {".", ".."}:
        raise ValueError(f"{field_name} must not be '.' or '..'.")
    if "/" in segment or "\\" in segment:
        raise ValueError(f"{field_name} must not contain path separators.")
    if any(ord(char) < 32 or ord(char) == 127 for char in segment):
        raise ValueError(f"{field_name} must not contain control characters.")
    return quote(segment, safe="._~-")


def _single_optional_space(value: str) -> str:
    """Remove exactly one leading space after an SSE field separator."""

    return value[1:] if value.startswith(" ") else value


class HueBridgeClient:
    """Perform bounded HTTPS requests against one local Hue bridge."""

    def __init__(
        self,
        config: HueBridgeConfig,
        *,
        json_requester: JsonRequester | None = None,
        event_reader: EventReader | None = None,
    ) -> None:
        self.config = config
        self._base_url = _validate_base_url(config.base_url)
        self._application_key = _validate_application_key(config.application_key)
        self._json_requester = json_requester or self._request_json
        self._event_reader = event_reader or self._read_event_stream
        self._http_client: Any | None = None
        self._client_lock = threading.RLock()
        self._event_stream_lock = threading.Lock()
        self._last_event_id = ""

    def close(self) -> None:
        """Close pooled transport resources."""

        with self._client_lock:
            client = self._http_client
            self._http_client = None
        if client is not None:
            client.close()

    def __enter__(self) -> HueBridgeClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def list_resources(self) -> list[dict[str, object]]:
        """Return all bridge resources from the generic CLIP v2 resource endpoint."""

        payload = _require_response_mapping(
            self._json_requester("GET", "/clip/v2/resource", None),
            field_name="list_resources",
        )
        data = payload.get("data", ())
        if not isinstance(data, list):
            raise RuntimeError("Hue bridge resource listing returned an invalid payload.")

        resources: list[dict[str, object]] = []
        for index, item in enumerate(data):
            if not isinstance(item, Mapping):
                raise RuntimeError(
                    f"Hue bridge resource listing returned a non-mapping item at index {index}."
                )
            resources.append(dict(item))
        return resources

    def put_resource(
        self,
        resource_type: str,
        resource_id: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        """Update one Hue resource via the CLIP v2 resource endpoint."""

        safe_resource_type = _validate_path_segment(resource_type, field_name="resource_type")
        safe_resource_id = _validate_path_segment(resource_id, field_name="resource_id")
        return _require_response_mapping(
            self._json_requester(
                "PUT",
                f"/clip/v2/resource/{safe_resource_type}/{safe_resource_id}",
                payload,
            ),
            field_name="put_resource",
        )

    def read_event_stream(
        self,
        *,
        timeout_s: float | None = None,
        max_events: int = 20,
    ) -> list[dict[str, object]]:
        """Read a bounded batch of Hue event-stream messages."""

        timeout = self.config.timeout_s if timeout_s is None else float(timeout_s)
        if max_events < 1:
            raise ValueError("max_events must be >= 1")
        try:
            return self._event_reader("/eventstream/clip/v2", timeout, max_events)
        except TimeoutError:
            # Treat an idle SSE timeout as a bounded no-event read, not as a transport failure.
            return []

    def _ensure_http_client(self) -> Any:
        """Create the pooled HTTP client on first use."""

        with self._client_lock:
            if self._http_client is None:
                httpx = _require_httpx()
                limits = httpx.Limits(
                    max_keepalive_connections=max(
                        2, int(getattr(self.config, "max_keepalive_connections", 2))
                    ),
                    max_connections=max(2, int(getattr(self.config, "max_connections", 4))),
                    keepalive_expiry=float(getattr(self.config, "keepalive_expiry_s", 30.0)),
                )
                timeout = httpx.Timeout(
                    connect=float(self.config.timeout_s),
                    read=float(self.config.timeout_s),
                    write=float(self.config.timeout_s),
                    pool=float(self.config.timeout_s),
                )
                self._http_client = httpx.Client(
                    base_url=self._base_url,
                    headers={
                        "Hue-Application-Key": self._application_key,
                        "User-Agent": "TWINR-HueBridgeClient/2026.03",
                        "Accept-Encoding": "identity",
                    },
                    timeout=timeout,
                    limits=limits,
                    verify=self._ssl_context(),
                    follow_redirects=False,
                    trust_env=False,
                    http2=self._http2_enabled(),
                    default_encoding="utf-8",
                )
            return self._http_client

    def _http2_enabled(self) -> bool:
        """Return whether HTTP/2 should be attempted."""

        return bool(getattr(self.config, "enable_http2", True)) and _HTTP2_AVAILABLE

    def _ssl_context(self) -> ssl.SSLContext:
        """Build the TLS context for local bridge requests."""

        context = ssl.create_default_context()
        ca_cert_path = getattr(self.config, "ca_cert_path", None)
        ca_cert_data = getattr(self.config, "ca_cert_pem", None) or getattr(
            self.config, "ca_cert_data", None
        )
        if ca_cert_path is not None or ca_cert_data is not None:
            context.load_verify_locations(cafile=ca_cert_path, cadata=ca_cert_data)

        if self._http2_enabled():
            context.set_alpn_protocols(["h2", "http/1.1"])
        else:
            context.set_alpn_protocols(["http/1.1"])

        if not self.config.verify_tls:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        return context

    def _serialize_json_body(self, payload: Mapping[str, object] | None) -> bytes | None:
        """Serialize a request body using the same JSON constraints as before."""

        if payload is None:
            return None
        try:
            return json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Hue bridge request payload is not valid JSON.") from exc

    def _bounded_read(self, chunks: Iterable[bytes], *, max_bytes: int, what: str) -> bytes:
        """Read response bytes while enforcing a hard size limit."""

        buffer = bytearray()
        for chunk in chunks:
            if not chunk:
                continue
            buffer.extend(chunk)
            if len(buffer) > max_bytes:
                raise RuntimeError(f"{what} exceeded the configured size limit.")
        return bytes(buffer)

    def _summarize_error_payload(self, raw_body: bytes, *, fallback: str) -> str:
        """Extract a compact error summary from JSON or text payloads."""

        if not raw_body:
            return fallback

        try:
            decoded = raw_body.decode("utf-8")
        except UnicodeDecodeError:
            return fallback

        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError:
            snippet = decoded.strip().replace("\n", " ")
            return snippet[:200] or fallback

        if isinstance(payload, Mapping):
            errors = payload.get("errors")
            if isinstance(errors, list):
                descriptions: list[str] = []
                for item in errors:
                    if isinstance(item, Mapping):
                        description = item.get("description")
                        if isinstance(description, str) and description.strip():
                            descriptions.append(description.strip())
                if descriptions:
                    return "; ".join(descriptions)

            error = payload.get("error")
            if isinstance(error, Mapping):
                description = error.get("description")
                if isinstance(description, str) and description.strip():
                    return description.strip()

        snippet = decoded.strip().replace("\n", " ")
        return snippet[:200] or fallback

    def _check_content_type(
        self,
        headers: Mapping[str, str],
        *,
        expected_prefix: str,
        what: str,
    ) -> None:
        """Validate response content-type when the server provided one."""

        content_type = headers.get("content-type") or headers.get("Content-Type") or ""
        if not content_type:
            return
        if not content_type.lower().startswith(expected_prefix):
            raise RuntimeError(
                f"Hue bridge returned unexpected content-type for {what}: {content_type!r}."
            )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None,
    ) -> dict[str, object]:
        """Perform one JSON request against the bridge."""

        httpx = _require_httpx()
        client = self._ensure_http_client()
        body = self._serialize_json_body(payload)
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"

        try:
            with client.stream(method, path, headers=headers, content=body) as response:
                raw_body = self._bounded_read(
                    response.iter_bytes(),
                    max_bytes=int(self.config.max_response_bytes),
                    what="Hue bridge response",
                )
                self._check_content_type(
                    response.headers,
                    expected_prefix="application/json",
                    what="JSON request",
                )
                status_code = int(response.status_code)
                reason = response.reason_phrase or "request failed"
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"Hue bridge {method} {path} timed out.") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Hue bridge {method} {path} failed: {exc}.") from exc

        if status_code >= 400:
            details = self._summarize_error_payload(raw_body, fallback=reason)
            raise RuntimeError(
                f"Hue bridge {method} {path} failed with HTTP {status_code}: {details}."
            )

        try:
            parsed = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Hue bridge returned invalid JSON.") from exc
        return _require_response_mapping(parsed, field_name="json_response")

    def _iter_decoded_lines(self, byte_chunks: Iterable[bytes]) -> Iterable[str]:
        """Yield UTF-8 decoded transport lines without their line terminators."""

        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        pending = ""

        def pop_one_line(buffer: str) -> tuple[str | None, str]:
            for index, char in enumerate(buffer):
                if char == "\n":
                    return buffer[:index], buffer[index + 1 :]
                if char == "\r":
                    next_index = index + 1
                    if next_index < len(buffer) and buffer[next_index] == "\n":
                        return buffer[:index], buffer[next_index + 1 :]
                    return buffer[:index], buffer[next_index:]
            return None, buffer

        for chunk in byte_chunks:
            if not chunk:
                continue
            pending += decoder.decode(chunk, final=False)
            while True:
                line, pending = pop_one_line(pending)
                if line is None:
                    break
                yield line

        pending += decoder.decode(b"", final=True)
        if pending:
            yield pending

    def _read_event_stream(
        self,
        path: str,
        timeout_s: float,
        max_events: int,
    ) -> list[dict[str, object]]:
        """Read a bounded number of SSE messages from the bridge."""

        httpx = _require_httpx()
        client = self._ensure_http_client()
        with self._client_lock:
            resume_from = self._last_event_id

        headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
        }
        if resume_from:
            headers["Last-Event-ID"] = resume_from

        events: list[dict[str, object]] = []
        current_id: object = _SENTINEL
        current_event: str | None = None
        data_lines: list[str] = []
        data_bytes = 0
        stream_timeout = httpx.Timeout(
            connect=float(self.config.timeout_s),
            read=float(timeout_s),
            write=float(self.config.timeout_s),
            pool=float(self.config.timeout_s),
        )

        with self._event_stream_lock:
            try:
                with client.stream("GET", path, headers=headers, timeout=stream_timeout) as response:
                    status_code = int(response.status_code)
                    reason = response.reason_phrase or "request failed"
                    if status_code >= 400:
                        raw_body = self._bounded_read(
                            response.iter_bytes(),
                            max_bytes=int(self.config.max_response_bytes),
                            what="Hue bridge event-stream error response",
                        )
                        details = self._summarize_error_payload(raw_body, fallback=reason)
                        raise RuntimeError(
                            f"Hue bridge GET {path} failed with HTTP {status_code}: {details}."
                        )

                    self._check_content_type(
                        response.headers,
                        expected_prefix="text/event-stream",
                        what="event stream",
                    )

                    last_event_id = resume_from
                    for line in self._iter_decoded_lines(response.iter_bytes()):
                        if line == "":
                            parsed = self._finalize_event(
                                current_id=current_id,
                                event_name=current_event,
                                data_lines=data_lines,
                                last_event_id=last_event_id,
                            )
                            if parsed is not None:
                                events.append(parsed)
                                last_event_id = str(parsed["id"])
                                with self._client_lock:
                                    self._last_event_id = last_event_id
                                if len(events) >= max_events:
                                    break
                            current_id = _SENTINEL
                            current_event = None
                            data_lines = []
                            data_bytes = 0
                            continue

                        if line.startswith(":"):
                            continue

                        field, separator, value = line.partition(":")
                        payload_value = _single_optional_space(value) if separator else ""

                        if field == "id":
                            if "\x00" not in payload_value:
                                current_id = payload_value
                            continue
                        if field == "event":
                            current_event = payload_value
                            continue
                        if field == "data":
                            data_bytes += len(payload_value.encode("utf-8"))
                            if data_bytes > int(self.config.max_event_bytes):
                                raise RuntimeError(
                                    "Hue event payload exceeded the configured size limit."
                                )
                            data_lines.append(payload_value)
                            continue
                        if field == "retry":
                            continue
                    # Important: do not dispatch a trailing unterminated event at EOF.
            except httpx.ReadTimeout as exc:
                raise TimeoutError("Hue bridge event stream timed out while idle.") from exc
            except httpx.TimeoutException as exc:
                raise RuntimeError("Hue bridge event stream timed out.") from exc
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Hue bridge event stream failed: {exc}.") from exc

        return events

    @staticmethod
    def _finalize_event(
        *,
        current_id: object,
        event_name: str | None,
        data_lines: list[str],
        last_event_id: str,
    ) -> dict[str, object] | None:
        """Parse one accumulated SSE event into a JSON-safe record."""

        if not data_lines:
            return None

        payload_text = "\n".join(data_lines)
        try:
            parsed_payload: Any = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Hue event stream returned invalid JSON.") from exc

        event_id = last_event_id if current_id is _SENTINEL else str(current_id)
        return {
            "id": event_id,
            "event": event_name or "message",
            "data": parsed_payload,
        }


__all__ = ["HueBridgeClient"]