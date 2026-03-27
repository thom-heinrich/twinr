"""Build shared dotenv/process-env context for config loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .constants import DEFAULT_VOICE_ACTIVATION_PHRASES
from .parsing import (
    _derive_camera_host_mode,
    _derive_proactive_vision_provider,
    _derive_snapshot_proxy_url,
    _parse_csv_strings,
    _parse_optional_url,
    _read_dotenv,
)


class ConfigValueGetter(Protocol):
    """Represent ``get_value`` with an optional default argument."""

    def __call__(self, name: str, default: str | None = None) -> str | None: ...


@dataclass(frozen=True, slots=True)
class ConfigLoadContext:
    """Carry shared derived values across the domain-specific loaders."""

    project_root: Path
    default_remote_runtime_check_mode: str
    get_value: ConfigValueGetter
    voice_activation_phrases: tuple[str, ...]
    camera_host_mode: str
    effective_second_pi_base_url: str | None
    camera_device: str
    camera_proxy_snapshot_url: str | None
    proactive_vision_provider: str
    drone_base_url: str | None


def build_config_load_context(env_path: str | Path = ".env") -> ConfigLoadContext:
    """Read dotenv + process env and derive the shared config-loading context."""

    path = Path(env_path)
    file_values = _read_dotenv(path)
    project_root = path.parent.resolve()
    default_remote_runtime_check_mode = (
        "watchdog_artifact" if project_root == Path("/twinr") else "direct"
    )

    def get_value(name: str, default: str | None = None) -> str | None:
        if name in os.environ:
            return os.environ[name]
        return file_values.get(name, default)

    voice_activation_phrases = _parse_csv_strings(
        get_value("TWINR_VOICE_ACTIVATION_PHRASES"),
        DEFAULT_VOICE_ACTIVATION_PHRASES,
    )
    raw_camera_host_mode = get_value("TWINR_CAMERA_HOST_MODE")
    camera_second_pi_base_url = _parse_optional_url(
        get_value("TWINR_CAMERA_SECOND_PI_BASE_URL"),
        strip_trailing_slash=True,
    )
    proactive_remote_camera_base_url = _parse_optional_url(
        get_value("TWINR_PROACTIVE_REMOTE_CAMERA_BASE_URL"),
        strip_trailing_slash=True,
    )
    camera_proxy_snapshot_url = _parse_optional_url(
        get_value("TWINR_CAMERA_PROXY_SNAPSHOT_URL"),
    )
    camera_device = get_value("TWINR_CAMERA_DEVICE", "/dev/video0") or "/dev/video0"
    camera_host_mode = _derive_camera_host_mode(
        raw_value=raw_camera_host_mode,
        camera_second_pi_base_url=camera_second_pi_base_url,
        proactive_remote_camera_base_url=proactive_remote_camera_base_url,
        camera_proxy_snapshot_url=camera_proxy_snapshot_url,
    )
    effective_second_pi_base_url = (
        proactive_remote_camera_base_url or camera_second_pi_base_url
    )
    if camera_proxy_snapshot_url is None and camera_host_mode == "second_pi":
        camera_proxy_snapshot_url = _derive_snapshot_proxy_url(
            effective_second_pi_base_url
        )
    proactive_vision_provider = _derive_proactive_vision_provider(
        get_value("TWINR_PROACTIVE_VISION_PROVIDER"),
        camera_host_mode=camera_host_mode,
        proactive_remote_camera_base_url=effective_second_pi_base_url,
        camera_device=camera_device,
    )
    drone_base_url = _parse_optional_url(
        get_value("TWINR_DRONE_BASE_URL"),
        strip_trailing_slash=True,
    )

    return ConfigLoadContext(
        project_root=project_root,
        default_remote_runtime_check_mode=default_remote_runtime_check_mode,
        get_value=get_value,
        voice_activation_phrases=voice_activation_phrases,
        camera_host_mode=camera_host_mode,
        effective_second_pi_base_url=effective_second_pi_base_url,
        camera_device=camera_device,
        camera_proxy_snapshot_url=camera_proxy_snapshot_url,
        proactive_vision_provider=proactive_vision_provider,
        drone_base_url=drone_base_url,
    )
