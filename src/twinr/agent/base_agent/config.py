"""Define the canonical environment-backed runtime configuration for Twinr.

Import ``TwinrConfig`` and the legacy helper symbols from this module. The
implementation now lives in focused sibling modules so parsing, normalization,
and env loading can evolve independently without changing callers.
"""

##REFACTOR: 2026-03-27##
from __future__ import annotations

from ._config.constants import (
    DEFAULT_BUTTON_PROBE_LINES,
    SUPPORTED_DISPLAY_DRIVERS,
    GPIO_DISPLAY_DRIVERS,
    SUPPORTED_DISPLAY_LAYOUTS,
    DEFAULT_OPENAI_MAIN_MODEL,
)
from ._config.parsing import (
    _read_dotenv,
    _parse_bool,
    _parse_optional_bool,
    _parse_optional_int,
    _parse_optional_text,
    _parse_optional_url,
    _parse_camera_host_mode,
    _derive_camera_host_mode,
    _derive_snapshot_proxy_url,
    _uses_aideck_camera_device,
    _derive_proactive_vision_provider,
    _normalize_model_setting,
    _parse_float,
    _parse_clamped_float,
    _parse_csv_ints,
    _parse_csv_strings,
    _parse_csv_mapping,
    _parse_local_semantic_router_mode,
    _parse_attention_servo_driver,
    _parse_attention_servo_control_mode,
    _default_display_poll_interval_s,
)
from ._config.schema import TwinrConfig

TwinrConfig.__module__ = __name__

__all__ = [
    "DEFAULT_BUTTON_PROBE_LINES",
    "SUPPORTED_DISPLAY_DRIVERS",
    "GPIO_DISPLAY_DRIVERS",
    "SUPPORTED_DISPLAY_LAYOUTS",
    "DEFAULT_OPENAI_MAIN_MODEL",
    "_read_dotenv",
    "_parse_bool",
    "_parse_optional_bool",
    "_parse_optional_int",
    "_parse_optional_text",
    "_parse_optional_url",
    "_parse_camera_host_mode",
    "_derive_camera_host_mode",
    "_derive_snapshot_proxy_url",
    "_uses_aideck_camera_device",
    "_derive_proactive_vision_provider",
    "_normalize_model_setting",
    "_parse_float",
    "_parse_clamped_float",
    "_parse_csv_ints",
    "_parse_csv_strings",
    "_parse_csv_mapping",
    "_parse_local_semantic_router_mode",
    "_parse_attention_servo_driver",
    "_parse_attention_servo_control_mode",
    "_default_display_poll_interval_s",
    "TwinrConfig",
]
