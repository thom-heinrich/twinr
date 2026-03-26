"""Fetch live AI-camera observations from the dedicated proxy Pi.

This module supports two peer-LAN camera modes:

- ``remote_proxy`` keeps the existing behavior where the helper Pi computes the
  full camera observation and the main Pi consumes JSON only.
- ``remote_frame`` keeps the helper Pi as the physical sensor host, but moves
  the hot attention/gesture MediaPipe lifting back onto the main Pi by fetching
  one coherent helper-side detection-plus-frame bundle over the direct link.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, replace
from io import BytesIO
import json
import logging
import socket
import time
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.ai_camera import AICameraObservation, LocalAICameraAdapter

from .local_camera_provider import LocalAICameraObservationProvider, LocalAICameraProviderConfig
from .observers import ProactiveVisionSnapshot


logger = logging.getLogger(__name__)

_MIN_REMOTE_FRAME_WIDTH = 64
_MIN_REMOTE_FRAME_HEIGHT = 64
_MAX_REMOTE_FRAME_WIDTH = 1920
_MAX_REMOTE_FRAME_HEIGHT = 1080


@dataclass(frozen=True, slots=True)
class RemoteAICameraProviderConfig:
    """Store provider-level wiring for peer-LAN helper-Pi camera transport."""

    base_url: str
    timeout_s: float = 4.0
    source_device: str = "imx500"
    input_format: str | None = "remote-ai-proxy"
    snapshot_width: int = 640
    snapshot_height: int = 480

    def __post_init__(self) -> None:
        """Normalize the remote-base URL, bounded timeout, and frame size."""

        normalized_base_url = str(self.base_url or "").strip().rstrip("/")
        if not normalized_base_url:
            raise ValueError("proactive_remote_camera_base_url is required for remote camera vision mode")
        object.__setattr__(self, "base_url", normalized_base_url)
        object.__setattr__(self, "timeout_s", max(0.1, float(self.timeout_s)))
        object.__setattr__(self, "source_device", str(self.source_device or "imx500").strip() or "imx500")
        if self.input_format is not None:
            normalized_input_format = str(self.input_format).strip() or None
            object.__setattr__(self, "input_format", normalized_input_format)
        object.__setattr__(
            self,
            "snapshot_width",
            _bounded_dimension(
                self.snapshot_width,
                minimum=_MIN_REMOTE_FRAME_WIDTH,
                maximum=_MAX_REMOTE_FRAME_WIDTH,
                fallback=640,
            ),
        )
        object.__setattr__(
            self,
            "snapshot_height",
            _bounded_dimension(
                self.snapshot_height,
                minimum=_MIN_REMOTE_FRAME_HEIGHT,
                maximum=_MAX_REMOTE_FRAME_HEIGHT,
                fallback=480,
            ),
        )

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "RemoteAICameraProviderConfig":
        """Build one remote-provider config from ``TwinrConfig``."""

        return cls(
            base_url=str(getattr(config, "proactive_remote_camera_base_url", "") or ""),
            timeout_s=float(getattr(config, "proactive_remote_camera_timeout_s", 4.0) or 4.0),
            source_device=str(getattr(config, "proactive_local_camera_source_device", "imx500") or "imx500"),
            input_format="remote-ai-proxy",
            snapshot_width=int(getattr(config, "camera_width", 640) or 640),
            snapshot_height=int(getattr(config, "camera_height", 480) or 480),
        )


class _RemoteAICameraTransport:
    """Share bounded HTTP transport helpers across remote camera providers."""

    def __init__(self, *, config: RemoteAICameraProviderConfig) -> None:
        self.config = config

    def _request_json(
        self,
        route: str,
        *,
        query: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Perform one bounded JSON request against the proxy service."""

        normalized_route = route.strip().lstrip("/")
        if query:
            encoded_query = urlencode(
                {
                    key: str(value)
                    for key, value in query.items()
                    if value is not None
                }
            )
            if encoded_query:
                normalized_route = f"{normalized_route}?{encoded_query}"
        request = Request(
            f"{self.config.base_url}/{normalized_route}",
            headers={"Accept": "application/json"},
            method="GET",
        )
        with urlopen(request, timeout=self.config.timeout_s) as response:
            charset = response.headers.get_content_charset("utf-8")
            payload = json.loads(response.read().decode(charset))
        if not isinstance(payload, dict):
            raise ValueError("remote_ai_camera_invalid_payload")
        return payload

    def _request_bytes(
        self,
        route: str,
        *,
        accept: str,
        query: dict[str, object] | None = None,
    ) -> bytes:
        """Perform one bounded byte request against the proxy service."""

        normalized_route = route.strip().lstrip("/")
        if query:
            encoded_query = urlencode(
                {
                    key: str(value)
                    for key, value in query.items()
                    if value is not None
                }
            )
            if encoded_query:
                normalized_route = f"{normalized_route}?{encoded_query}"
        request = Request(
            f"{self.config.base_url}/{normalized_route}",
            headers={"Accept": accept},
            method="GET",
        )
        with urlopen(request, timeout=self.config.timeout_s) as response:
            return response.read()


class RemoteAICameraObservationProvider(_RemoteAICameraTransport):
    """Return live AI-camera observations fetched from the helper Pi."""

    supports_attention_refresh = True
    supports_gesture_refresh = True

    def __init__(self, *, config: RemoteAICameraProviderConfig) -> None:
        """Initialize one remote JSON-only provider around the proxy endpoint."""

        super().__init__(config=config)
        self._mapper = LocalAICameraObservationProvider(
            adapter=cast(Any, object()),
            config=LocalAICameraProviderConfig(
                source_device=config.source_device,
                input_format=config.input_format,
            ),
        )
        self._last_attention_debug_details: dict[str, object] | None = None
        self._last_gesture_debug_details: dict[str, object] | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "RemoteAICameraObservationProvider":
        """Build one remote provider directly from ``TwinrConfig``."""

        return cls(config=RemoteAICameraProviderConfig.from_config(config))

    def observe(self) -> ProactiveVisionSnapshot:
        """Fetch one general social observation from the proxy Pi."""

        snapshot, _debug_details = self._observe_remote("observe")
        self._last_attention_debug_details = None
        self._last_gesture_debug_details = None
        return snapshot

    def observe_attention(self) -> ProactiveVisionSnapshot:
        """Fetch one low-latency HDMI attention snapshot from the proxy Pi."""

        snapshot, debug_details = self._observe_remote("observe_attention")
        self._last_attention_debug_details = debug_details
        return snapshot

    def observe_gesture(self) -> ProactiveVisionSnapshot:
        """Fetch one low-latency HDMI gesture snapshot from the proxy Pi."""

        snapshot, debug_details = self._observe_remote("observe_gesture")
        self._last_gesture_debug_details = debug_details
        return snapshot

    def gesture_debug_details(self) -> dict[str, object] | None:
        """Return the newest proxy-side gesture debug payload when present."""

        if self._last_gesture_debug_details is None:
            return None
        return dict(self._last_gesture_debug_details)

    def attention_debug_details(self) -> dict[str, object] | None:
        """Return the newest proxy-side attention debug payload when present."""

        if self._last_attention_debug_details is None:
            return None
        return dict(self._last_attention_debug_details)

    def _observe_remote(
        self,
        route: str,
    ) -> tuple[ProactiveVisionSnapshot, dict[str, object] | None]:
        """Fetch one remote observation and optional debug payload."""

        observation, captured_at, model, debug_details = self._fetch_remote_observation(route)
        return self._snapshot_from_observation(
            observation,
            captured_at=captured_at,
            model=model,
        ), debug_details

    def _fetch_remote_observation(
        self,
        route: str,
    ) -> tuple[AICameraObservation, float, str | None, dict[str, object] | None]:
        """Fetch one remote observation and normalize its payload contract."""

        payload: dict[str, object] | None = None
        try:
            payload = self._request_json(route)
            observation_payload = payload.get("observation")
            if not isinstance(observation_payload, dict):
                raise ValueError("remote_ai_camera_missing_observation")
            observation = AICameraObservation(**observation_payload)
        except Exception as exc:
            error_code = _remote_error_code(exc)
            logger.warning("Remote AI camera request %s failed with %s.", route, error_code)
            logger.debug("Remote AI camera request failed.", exc_info=True)
            observation = AICameraObservation(
                observed_at=0.0,
                camera_online=False,
                camera_ready=False,
                camera_ai_ready=False,
                camera_error=error_code,
            )
            payload = None

        captured_at = observation.last_camera_frame_at or observation.observed_at
        if payload is not None:
            captured_at = _coerce_captured_at(
                payload.get("captured_at"),
                fallback=captured_at,
            )
        debug_details = _coerce_debug_details(None if payload is None else payload.get("debug_details"))
        if payload is not None and payload.get("cache_state") is not None:
            debug_details = dict(debug_details or {})
            debug_details["cache_state"] = str(payload.get("cache_state") or "")
        model = observation.model
        if payload is not None and payload.get("model") is not None:
            model = str(payload.get("model")).strip() or observation.model
        return observation, captured_at, model, debug_details

    def _snapshot_from_observation(
        self,
        observation: AICameraObservation,
        *,
        captured_at: float,
        model: str | None,
    ) -> ProactiveVisionSnapshot:
        """Map one normalized camera observation onto the social provider contract."""

        social = self._mapper._to_social_observation(observation)
        return ProactiveVisionSnapshot(
            observation=social,
            response_text=self._mapper._response_text(observation),
            captured_at=captured_at,
            image=None,
            source_device=self.config.source_device,
            input_format=self.config.input_format,
            response_id=None,
            request_id=None,
            model=model or observation.model,
        )


class RemoteFrameAICameraObservationProvider(_RemoteAICameraTransport):
    """Run the hot attention/gesture lifting on the main Pi from remote frames."""

    supports_attention_refresh = True
    supports_gesture_refresh = True

    def __init__(
        self,
        *,
        config: RemoteAICameraProviderConfig,
        processor: LocalAICameraAdapter,
    ) -> None:
        """Initialize one remote-frame provider around the helper transport."""

        super().__init__(config=config)
        self._processor = processor
        self._mapper = LocalAICameraObservationProvider(
            adapter=processor,
            config=LocalAICameraProviderConfig(
                source_device=config.source_device,
                input_format=config.input_format,
            ),
        )
        self._proxy_provider = RemoteAICameraObservationProvider(
            config=replace(config, input_format="remote-ai-proxy")
        )
        self._last_attention_debug_details: dict[str, object] | None = None
        self._last_gesture_debug_details: dict[str, object] | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "RemoteFrameAICameraObservationProvider":
        """Build one remote-frame provider directly from ``TwinrConfig``."""

        provider_config = RemoteAICameraProviderConfig(
            base_url=str(getattr(config, "proactive_remote_camera_base_url", "") or ""),
            timeout_s=float(getattr(config, "proactive_remote_camera_timeout_s", 4.0) or 4.0),
            source_device=str(getattr(config, "proactive_local_camera_source_device", "imx500") or "imx500"),
            input_format="remote-ai-frame",
            snapshot_width=int(getattr(config, "camera_width", 640) or 640),
            snapshot_height=int(getattr(config, "camera_height", 480) or 480),
        )
        return cls(
            config=provider_config,
            processor=LocalAICameraAdapter.from_config(config),
        )

    def close(self) -> None:
        """Close the nested local processor when the runtime exposes a hook."""

        self._processor.close()

    def observe(self) -> ProactiveVisionSnapshot:
        """Keep the non-hot full observation path on the existing proxy contract."""

        self._last_attention_debug_details = None
        self._last_gesture_debug_details = None
        return self._proxy_provider.observe()

    def observe_attention(self) -> ProactiveVisionSnapshot:
        """Fetch one coherent helper bundle and run the fast attention path locally."""

        provider_started_ns = time.monotonic_ns()
        remote_fetch_started_ns = provider_started_ns
        observation, captured_at, _model, remote_debug, frame_rgb = self._fetch_remote_frame_bundle()
        remote_fetch_ms = _elapsed_ms(remote_fetch_started_ns)
        if not _camera_observation_ready(observation):
            self._last_attention_debug_details = _append_provider_stage_ms(
                _merge_debug_details(
                    None,
                    remote_debug=remote_debug,
                    transport_mode="remote_frame_passthrough_fault",
                    remote_route="observe_frame_bundle",
                ),
                remote_fetch_ms=remote_fetch_ms,
                provider_total_ms=_elapsed_ms(provider_started_ns),
            )
            return self._snapshot_from_observation(
                observation,
                captured_at=captured_at,
                model=observation.model,
            )

        process_started_ns = time.monotonic_ns()
        detection = self._processor._coerce_detection_result(observation)
        if not self._processor._needs_rgb_frame_for_attention(detection=detection):
            frame_rgb = None
        processed_observation = self._processor.observe_attention_from_frame(
            detection=detection,
            frame_rgb=frame_rgb,
            observed_at=time.time(),
            frame_at=captured_at,
        )
        local_process_ms = _elapsed_ms(process_started_ns)
        processed_observation = replace(
            processed_observation,
            model="remote-imx500-detection+local-attention-fast",
        )
        self._last_attention_debug_details = _append_provider_stage_ms(
            _merge_debug_details(
                self._processor.get_last_attention_debug_details(),
                remote_debug=remote_debug,
                transport_mode="remote_frame_local_attention",
                remote_route="observe_frame_bundle",
            ),
            remote_fetch_ms=remote_fetch_ms,
            local_process_ms=local_process_ms,
            provider_total_ms=_elapsed_ms(provider_started_ns),
        )
        return self._snapshot_from_observation(
            processed_observation,
            captured_at=captured_at,
            model=processed_observation.model,
        )

    def observe_gesture(self) -> ProactiveVisionSnapshot:
        """Fetch one coherent helper bundle and run the heavy gesture path locally."""

        provider_started_ns = time.monotonic_ns()
        remote_fetch_started_ns = provider_started_ns
        observation, captured_at, _model, remote_debug, frame_rgb = self._fetch_remote_frame_bundle()
        remote_fetch_ms = _elapsed_ms(remote_fetch_started_ns)
        if not _camera_observation_ready(observation):
            self._last_gesture_debug_details = _append_provider_stage_ms(
                _merge_debug_details(
                    None,
                    remote_debug=remote_debug,
                    transport_mode="remote_frame_passthrough_fault",
                    remote_route="observe_frame_bundle",
                ),
                remote_fetch_ms=remote_fetch_ms,
                provider_total_ms=_elapsed_ms(provider_started_ns),
            )
            return self._snapshot_from_observation(
                observation,
                captured_at=captured_at,
                model=observation.model,
            )

        process_started_ns = time.monotonic_ns()
        processed_observation = self._processor.observe_gesture_from_frame(
            detection=self._processor._coerce_detection_result(observation),
            frame_rgb=frame_rgb,
            observed_at=time.time(),
            frame_at=captured_at,
            # The dedicated remote-frame refresh only feeds the fine-hand
            # HDMI/wakeup lane, so the coarse-only pose fallback adds latency
            # without improving the symbols this path can emit.
            allow_pose_fallback=False,
        )
        local_process_ms = _elapsed_ms(process_started_ns)
        processed_observation = replace(
            processed_observation,
            model=_remote_frame_gesture_model_name(processed_observation.model),
        )
        self._last_gesture_debug_details = _append_provider_stage_ms(
            _merge_debug_details(
                self._processor.get_last_gesture_debug_details(),
                remote_debug=remote_debug,
                transport_mode="remote_frame_local_gesture",
                remote_route="observe_frame_bundle",
            ),
            remote_fetch_ms=remote_fetch_ms,
            local_process_ms=local_process_ms,
            provider_total_ms=_elapsed_ms(provider_started_ns),
        )
        return self._snapshot_from_observation(
            processed_observation,
            captured_at=captured_at,
            model=processed_observation.model,
        )

    def gesture_debug_details(self) -> dict[str, object] | None:
        """Return the newest main-Pi-side gesture debug payload when present."""

        if self._last_gesture_debug_details is None:
            return None
        return dict(self._last_gesture_debug_details)

    def attention_debug_details(self) -> dict[str, object] | None:
        """Return the newest main-Pi-side attention debug payload when present."""

        if self._last_attention_debug_details is None:
            return None
        return dict(self._last_attention_debug_details)

    def _fetch_remote_frame_bundle(
        self,
    ) -> tuple[AICameraObservation, float, str | None, dict[str, object] | None, Any]:
        """Fetch one coherent helper bundle containing detection facts plus PNG frame."""

        payload: dict[str, object] | None = None
        try:
            payload = self._request_json(
                "observe_frame_bundle",
                query={
                    "width": self.config.snapshot_width,
                    "height": self.config.snapshot_height,
                },
            )
            observation_payload = payload.get("observation")
            if not isinstance(observation_payload, dict):
                raise ValueError("remote_ai_camera_missing_observation")
            observation = AICameraObservation(**observation_payload)
            frame_rgb = None
            if _camera_observation_ready(observation):
                frame_rgb = _decode_rgb_frame(
                    _decode_remote_frame_bundle_png(payload.get("frame_png_base64"))
                )
        except Exception as exc:
            error_code = _remote_error_code(exc)
            logger.warning("Remote AI camera frame bundle failed with %s.", error_code)
            logger.debug("Remote AI camera frame bundle exception details.", exc_info=True)
            observation = AICameraObservation(
                observed_at=0.0,
                camera_online=False,
                camera_ready=False,
                camera_ai_ready=False,
                camera_error=error_code,
            )
            frame_rgb = None
        captured_at = observation.last_camera_frame_at or observation.observed_at
        if payload is not None:
            captured_at = _coerce_captured_at(
                payload.get("captured_at"),
                fallback=captured_at,
            )
        debug_details = _coerce_debug_details(None if payload is None else payload.get("debug_details"))
        if payload is not None and payload.get("cache_state") is not None:
            debug_details = dict(debug_details or {})
            debug_details["cache_state"] = str(payload.get("cache_state") or "")
        model = observation.model
        if payload is not None and payload.get("model") is not None:
            model = str(payload.get("model")).strip() or observation.model
        return observation, captured_at, model, debug_details, frame_rgb

    def _snapshot_from_observation(
        self,
        observation: AICameraObservation,
        *,
        captured_at: float,
        model: str | None,
    ) -> ProactiveVisionSnapshot:
        """Map one processed observation onto the social provider contract."""

        social = self._mapper._to_social_observation(observation)
        return ProactiveVisionSnapshot(
            observation=social,
            response_text=self._mapper._response_text(observation),
            captured_at=captured_at,
            image=None,
            source_device=self.config.source_device,
            input_format=self.config.input_format,
            response_id=None,
            request_id=None,
            model=model or observation.model,
        )


def _decode_rgb_frame(payload: bytes) -> Any:
    """Decode one remote PNG snapshot into an RGB ndarray-like frame."""

    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on runtime environment.
        raise RuntimeError("remote_ai_camera_frame_decoder_unavailable") from exc

    try:
        with Image.open(BytesIO(payload)) as image:
            return np.asarray(image.convert("RGB"))
    except Exception as exc:
        raise RuntimeError("remote_ai_camera_frame_decode_failed") from exc


def _camera_observation_ready(observation: AICameraObservation) -> bool:
    """Return whether the remote helper delivered one healthy camera observation."""

    return bool(
        observation.camera_online
        and observation.camera_ready
        and observation.camera_ai_ready
        and not str(observation.camera_error or "").strip()
    )


def _coerce_debug_details(value: object) -> dict[str, object] | None:
    """Return one shallow-copy debug payload when the response carries a mapping."""

    if not isinstance(value, dict):
        return None
    return dict(value)


def _coerce_captured_at(value: object, *, fallback: float) -> float:
    """Return one finite capture timestamp or fall back to the provided value."""

    try:
        captured_at = float(cast(Any, value))
    except (TypeError, ValueError):
        return fallback
    return captured_at if captured_at == captured_at else fallback


def _remote_frame_gesture_model_name(model: str | None) -> str:
    """Normalize the hot-path model name for main-Pi-side remote-frame gesture runs."""

    if model == "local-imx500+mediapipe-live-gesture+pose-fallback":
        return "remote-imx500-detection+local-mediapipe-live-gesture+pose-fallback"
    return "remote-imx500-detection+local-mediapipe-live-gesture"


def _decode_remote_frame_bundle_png(value: object) -> bytes:
    """Decode one base64 PNG field carried by the helper bundle endpoint."""

    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("remote_ai_camera_missing_frame_bundle")
    try:
        payload = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("remote_ai_camera_invalid_frame_bundle") from exc
    if not payload:
        raise RuntimeError("remote_ai_camera_invalid_frame_bundle")
    return payload


def _merge_debug_details(
    local_debug: dict[str, object] | None,
    *,
    remote_debug: dict[str, object] | None = None,
    transport_mode: str,
    remote_route: str,
) -> dict[str, object]:
    """Merge local/main-Pi debug facts with helper-side transport provenance."""

    payload = dict(local_debug or {})
    payload["transport_mode"] = transport_mode
    payload["remote_route"] = remote_route
    if remote_debug:
        payload["remote_debug"] = dict(remote_debug)
    return payload


def _append_provider_stage_ms(
    debug_details: dict[str, object] | None,
    **stage_values: float | None,
) -> dict[str, object]:
    """Attach bounded provider-side stage timings to the current debug payload."""

    payload = dict(debug_details or {})
    provider_stage_ms = {
        key: round(float(value), 3)
        for key, value in stage_values.items()
        if value is not None
    }
    if provider_stage_ms:
        payload["provider_stage_ms"] = provider_stage_ms
    return payload


def _elapsed_ms(started_ns: int) -> float:
    """Return one bounded monotonic elapsed duration in milliseconds."""

    return round((time.monotonic_ns() - started_ns) / 1_000_000.0, 3)


def _bounded_dimension(value: object, *, minimum: int, maximum: int, fallback: int) -> int:
    """Clamp one optional frame dimension into a sane bounded transport range."""

    try:
        number = int(cast(Any, value))
    except (TypeError, ValueError):
        return fallback
    if number < minimum:
        return minimum
    if number > maximum:
        return maximum
    return number


def _remote_error_code(exc: Exception) -> str:
    """Map remote-transport failures to stable camera error codes."""

    if isinstance(exc, HTTPError):
        return f"remote_ai_camera_http_{exc.code}"
    if isinstance(exc, json.JSONDecodeError):
        return "remote_ai_camera_invalid_json"
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return "remote_ai_camera_timeout"
    if isinstance(exc, URLError):
        if isinstance(exc.reason, socket.timeout):
            return "remote_ai_camera_timeout"
        return "remote_ai_camera_unreachable"
    if isinstance(exc, ValueError):
        return "remote_ai_camera_invalid_payload"
    if isinstance(exc, RuntimeError):
        return str(exc)
    return "remote_ai_camera_provider_failed"


__all__ = [
    "RemoteAICameraObservationProvider",
    "RemoteAICameraProviderConfig",
    "RemoteFrameAICameraObservationProvider",
]
