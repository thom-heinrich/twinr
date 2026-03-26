"""Provide bounded continuous AI-Deck still-camera observations for Twinr.

This provider bridges the Bitcraze AI-Deck WiFi still-camera path into the
existing OpenAI-backed proactive vision classifier. It is intentionally
conservative: no local pose/gesture hot path is claimed, so the provider only
owns the slower inspect-style observation lane.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.camera import V4L2StillCamera
from twinr.providers.openai import OpenAIBackend

from .observers import OpenAIVisionObservationProvider, ProactiveVisionSnapshot


_AIDECK_DEVICE_SCHEME = "aideck://"


@dataclass(frozen=True, slots=True)
class AIDeckOpenAIVisionProviderConfig:
    """Store the bounded AI-Deck camera device contract for proactive vision."""

    camera_device: str

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "AIDeckOpenAIVisionProviderConfig":
        """Build one validated AI-Deck provider config from ``TwinrConfig``."""

        camera_device = str(getattr(config, "camera_device", "") or "").strip()
        if not camera_device.lower().startswith(_AIDECK_DEVICE_SCHEME):
            raise ValueError("AI-Deck proactive vision requires TWINR_CAMERA_DEVICE to use aideck://host[:port].")
        return cls(camera_device=camera_device)


class AIDeckOpenAIVisionObservationProvider:
    """Classify AI-Deck still frames continuously through the OpenAI vision path."""

    supports_attention_refresh = False
    supports_gesture_refresh = False

    def __init__(
        self,
        *,
        observer: Any,
        config: AIDeckOpenAIVisionProviderConfig | None = None,
    ) -> None:
        """Wrap one nested still-camera vision observer."""

        self.observer = observer
        self.config = config

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        backend: OpenAIBackend,
        camera: V4L2StillCamera | None = None,
        camera_lock: Lock | None = None,
    ) -> "AIDeckOpenAIVisionObservationProvider":
        """Build one AI-Deck OpenAI provider directly from ``TwinrConfig``."""

        provider_config = AIDeckOpenAIVisionProviderConfig.from_config(config)
        resolved_camera = camera or V4L2StillCamera.from_config(config)
        resolved_device = str(getattr(resolved_camera, "device", provider_config.camera_device) or "").strip()
        if not resolved_device.lower().startswith(_AIDECK_DEVICE_SCHEME):
            raise ValueError("AI-Deck proactive vision requires an aideck:// camera adapter.")
        return cls(
            observer=OpenAIVisionObservationProvider(
                backend=backend,
                camera=resolved_camera,
                camera_lock=camera_lock,
            ),
            config=provider_config,
        )

    def observe(self) -> ProactiveVisionSnapshot:
        """Capture one bounded AI-Deck frame and classify it for proactive use."""

        return self.observer.observe()

    def close(self) -> None:
        """Close the nested camera backend when it exposes a close hook."""

        nested_camera = getattr(self.observer, "camera", None)
        close = getattr(nested_camera, "close", None)
        if callable(close):
            close()
