"""Reach a Pololu Maestro attached to a peer Pi over bounded HTTP.

This module keeps the main Pi's attention-servo runtime independent from the
physical USB location of the Maestro. The main runtime still issues the same
channel/pulse commands, but this writer forwards them to a dedicated helper-Pi
proxy service on the direct Ethernet link.
"""

from __future__ import annotations

import json
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


_DEFAULT_TIMEOUT_S = 1.5


def _bounded_timeout_s(timeout_s: float) -> float:
    return max(0.1, float(timeout_s))


class PeerPololuMaestroServoPulseWriter:
    """Drive one Maestro channel through the peer-Pi servo proxy service."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        normalized_base_url = str(base_url or "").strip().rstrip("/")
        if not normalized_base_url:
            raise RuntimeError("peer_pololu_maestro requires TWINR_ATTENTION_SERVO_PEER_BASE_URL")
        self._base_url = normalized_base_url
        self._timeout_s = _bounded_timeout_s(timeout_s)

    def probe(self, gpio: int) -> None:
        self._request_json("POST", "/servo/probe", payload={"channel": int(gpio)})

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        del gpio_chip
        self._request_json(
            "POST",
            "/servo/write",
            payload={
                "channel": int(gpio),
                "pulse_width_us": int(pulse_width_us),
            },
        )

    def disable(self, *, gpio_chip: str, gpio: int) -> None:
        del gpio_chip
        self._request_json("POST", "/servo/disable", payload={"channel": int(gpio)})

    def close(self) -> None:
        return

    def current_pulse_width_us(self, *, gpio_chip: str, gpio: int) -> int | None:
        del gpio_chip
        payload = self._request_json(
            "GET",
            "/servo/position",
            query={"channel": int(gpio)},
        )
        pulse_width_us = payload.get("pulse_width_us")
        if pulse_width_us is None:
            return None
        return _coerce_int(pulse_width_us, field_name="pulse_width_us")

    def _request_json(
        self,
        method: str,
        route: str,
        *,
        payload: dict[str, object] | None = None,
        query: dict[str, object] | None = None,
    ) -> dict[str, object]:
        normalized_route = route.strip().lstrip("/")
        if query:
            encoded_query = urlencode({key: str(value) for key, value in query.items() if value is not None})
            if encoded_query:
                normalized_route = f"{normalized_route}?{encoded_query}"
        body: bytes | None = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self._base_url}/{normalized_route}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self._timeout_s) as response:
                charset = response.headers.get_content_charset("utf-8")
                response_payload = json.loads(response.read().decode(charset))
        except HTTPError as exc:
            raise RuntimeError(_render_http_error(exc)) from exc
        except (URLError, TimeoutError, socket.timeout) as exc:
            raise RuntimeError(f"Peer Pololu Maestro request failed: {_remote_error_code(exc)}") from exc
        if not isinstance(response_payload, dict):
            raise RuntimeError("Peer Pololu Maestro proxy returned a non-object JSON payload")
        if not bool(response_payload.get("ok", True)):
            message = str(response_payload.get("error") or "peer_pololu_maestro_request_failed")
            raise RuntimeError(message)
        return response_payload


def _render_http_error(exc: HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    if body:
        return f"Peer Pololu Maestro request failed: HTTP {exc.code} {body}"
    return f"Peer Pololu Maestro request failed: HTTP {exc.code}"


def _remote_error_code(exc: BaseException) -> str:
    reason = getattr(exc, "reason", None)
    if isinstance(reason, BaseException):
        return reason.__class__.__name__
    if reason is not None:
        normalized_reason = str(reason).strip()
        if normalized_reason:
            return normalized_reason
    return exc.__class__.__name__


def _coerce_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, (int, float, str)):
        raise RuntimeError(f"Peer Pololu Maestro proxy returned invalid {field_name}")
    return int(value)
