"""Speak the local Hue bridge JSON and event-stream APIs.

The client keeps raw HTTPS/SSE transport separate from Hue resource
normalization so the provider package can stay testable and future bridge
changes stay isolated to one place.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import ssl
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from twinr.integrations.smarthome.hue.models import HueBridgeConfig

JsonRequester = Callable[[str, str, Mapping[str, object] | None], dict[str, object]]
EventReader = Callable[[str, float, int], list[dict[str, object]]]


def _require_response_mapping(payload: object, *, field_name: str) -> dict[str, object]:
    """Require a mapping-shaped Hue response payload."""

    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{field_name} must be a mapping.")
    return {str(key): value for key, value in payload.items()}


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
        self._json_requester = json_requester or self._request_json
        self._event_reader = event_reader or self._read_event_stream

    def list_resources(self) -> list[dict[str, object]]:
        """Return all bridge resources from the generic CLIP v2 resource endpoint."""

        payload = _require_response_mapping(
            self._json_requester("GET", "/clip/v2/resource", None),
            field_name="list_resources",
        )
        data = payload.get("data", ())
        if not isinstance(data, list):
            raise RuntimeError("Hue bridge resource listing returned an invalid payload.")
        return [dict(item) for item in data if isinstance(item, Mapping)]

    def put_resource(
        self,
        resource_type: str,
        resource_id: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        """Update one Hue resource via the CLIP v2 resource endpoint."""

        return _require_response_mapping(
            self._json_requester(
                "PUT",
                f"/clip/v2/resource/{resource_type.strip()}/{resource_id.strip()}",
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
        return self._event_reader("/eventstream/clip/v2", timeout, max_events)

    def _ssl_context(self) -> ssl.SSLContext:
        """Build the TLS context for local bridge requests."""

        context = ssl.create_default_context()
        if not self.config.verify_tls:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        return context

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None,
    ) -> dict[str, object]:
        """Perform one JSON request against the bridge."""

        body = None
        headers = {
            "Accept": "application/json",
            "Hue-Application-Key": self.config.application_key,
        }
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            urljoin(self.config.base_url, path),
            data=body,
            headers=headers,
            method=method,
        )
        with urlopen(request, timeout=self.config.timeout_s, context=self._ssl_context()) as response:
            raw_body = response.read(self.config.max_response_bytes + 1)
        if len(raw_body) > self.config.max_response_bytes:
            raise RuntimeError("Hue bridge response exceeded the configured size limit.")
        try:
            parsed = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Hue bridge returned invalid JSON.") from exc
        return _require_response_mapping(parsed, field_name="json_response")

    def _read_event_stream(self, path: str, timeout_s: float, max_events: int) -> list[dict[str, object]]:
        """Read a bounded number of SSE messages from the bridge."""

        request = Request(
            urljoin(self.config.base_url, path),
            headers={
                "Accept": "text/event-stream",
                "Hue-Application-Key": self.config.application_key,
            },
            method="GET",
        )
        events: list[dict[str, object]] = []
        current_id: str | None = None
        current_event: str | None = None
        data_lines: list[str] = []
        data_bytes = 0

        with urlopen(request, timeout=timeout_s, context=self._ssl_context()) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="strict").rstrip("\r\n")
                if not line:
                    parsed = self._finalize_event(current_id, current_event, data_lines)
                    if parsed is not None:
                        events.append(parsed)
                        if len(events) >= max_events:
                            break
                    current_id = None
                    current_event = None
                    data_lines = []
                    data_bytes = 0
                    continue
                if line.startswith(":"):
                    continue
                field, _, value = line.partition(":")
                payload_value = value.lstrip()
                if field == "id":
                    current_id = payload_value
                    continue
                if field == "event":
                    current_event = payload_value
                    continue
                if field == "data":
                    data_bytes += len(payload_value.encode("utf-8"))
                    if data_bytes > self.config.max_event_bytes:
                        raise RuntimeError("Hue event payload exceeded the configured size limit.")
                    data_lines.append(payload_value)
            else:
                parsed = self._finalize_event(current_id, current_event, data_lines)
                if parsed is not None and len(events) < max_events:
                    events.append(parsed)
        return events

    @staticmethod
    def _finalize_event(
        event_id: str | None,
        event_name: str | None,
        data_lines: list[str],
    ) -> dict[str, object] | None:
        """Parse one accumulated SSE event into a JSON-safe record."""

        if not data_lines:
            return None
        payload_text = "\n".join(data_lines)
        try:
            parsed_payload: Any = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Hue event stream returned invalid JSON.") from exc
        return {
            "id": event_id or "",
            "event": event_name or "message",
            "data": parsed_payload,
        }


__all__ = ["HueBridgeClient"]
