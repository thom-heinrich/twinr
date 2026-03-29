"""Compatibility tombstone for retired helper-Pi proactive camera providers.

The productive Twinr runtime is single-Pi only. The historical helper-Pi
`remote_proxy` / `remote_frame` implementations were archived under
`__legacy__/src/twinr/proactive/social/remote_camera_provider.py` and are no
longer part of the supported runtime surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from twinr.agent.base_agent.config import TwinrConfig


_LEGACY_REMOTE_PROVIDER_ERROR = (
    "Legacy helper-Pi proactive vision providers are no longer supported. "
    "Twinr now requires local proactive vision on the main Pi."
)


@dataclass(frozen=True, slots=True)
class RemoteAICameraProviderConfig:
    """Fail closed when retired helper-Pi provider config is instantiated."""

    base_url: str = ""

    def __post_init__(self) -> None:
        raise ValueError(_LEGACY_REMOTE_PROVIDER_ERROR)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "RemoteAICameraProviderConfig":
        """Keep the historic factory surface but fail closed immediately."""

        del config
        raise ValueError(_LEGACY_REMOTE_PROVIDER_ERROR)


class RemoteAICameraObservationProvider:
    """Retired helper-Pi JSON provider kept only as a fail-closed shim."""

    supports_attention_refresh = False
    supports_gesture_refresh = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise ValueError(_LEGACY_REMOTE_PROVIDER_ERROR)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "RemoteAICameraObservationProvider":
        del cls, config
        raise ValueError(_LEGACY_REMOTE_PROVIDER_ERROR)


class RemoteFrameAICameraObservationProvider(RemoteAICameraObservationProvider):
    """Retired helper-Pi frame-bundle provider kept only as a fail-closed shim."""


__all__ = [
    "RemoteAICameraObservationProvider",
    "RemoteAICameraProviderConfig",
    "RemoteFrameAICameraObservationProvider",
]
