"""Orchestrate the canonical TwinrConfig env-loading flow."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .context import build_config_load_context
from .load_channels import load_channels_config
from .load_hardware_display import load_hardware_display_config
from .load_providers import load_provider_config
from .load_runtime_memory import load_runtime_memory_config
from .load_turn_streaming import load_turn_streaming_config
from .load_vision_proactive import load_vision_proactive_config

if TYPE_CHECKING:
    from .schema import TwinrConfig


def load_twinr_config(
    cls: type["TwinrConfig"], env_path: str | Path = ".env"
) -> "TwinrConfig":
    """Build a config snapshot from dotenv files and process env."""

    context = build_config_load_context(env_path)
    settings: dict[str, Any] = {}
    for loader in (
        load_provider_config,
        load_turn_streaming_config,
        load_channels_config,
        load_vision_proactive_config,
        load_runtime_memory_config,
        load_hardware_display_config,
    ):
        settings.update(loader(context))
    return cls(**settings)
