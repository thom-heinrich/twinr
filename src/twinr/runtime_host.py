"""Host and launch-policy helpers for authoritative Pi runtimes."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig


RUNTIME_SUPERVISOR_ENV_KEY = "TWINR_RUNTIME_SUPERVISOR_ACTIVE"


def env_flag(name: str) -> bool:
    """Interpret one conventional environment flag as a boolean."""

    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def uses_pi_runtime_root(env_file: str | Path) -> bool:
    """Return whether the provided env file targets the Pi acceptance checkout."""

    env_path = Path(env_file).resolve()
    pi_root = Path("/twinr").resolve()
    return pi_root in env_path.parents or env_path == pi_root / ".env"


def is_raspberry_pi_host() -> bool:
    """Detect whether the current machine reports itself as a Raspberry Pi."""

    model_path = Path("/proc/device-tree/model")
    try:
        return "Raspberry Pi" in model_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def should_enable_display_surface(
    config: TwinrConfig,
    env_file: str | Path,
    *,
    suppress_when_supervisor_env: bool = False,
    uses_pi_runtime_root_fn: Callable[[str | Path], bool] = uses_pi_runtime_root,
    is_raspberry_pi_host_fn: Callable[[], bool] = is_raspberry_pi_host,
) -> bool:
    """Return whether the authoritative visible Twinr display should run."""

    if suppress_when_supervisor_env and env_flag(RUNTIME_SUPERVISOR_ENV_KEY):
        return False
    if not uses_pi_runtime_root_fn(env_file):
        return False
    explicit_setting = getattr(config, "display_companion_enabled", None)
    if explicit_setting is not None:
        return bool(explicit_setting)
    if (
        str(getattr(config, "display_driver", "") or "").strip().lower() == "hdmi_wayland"
        and bool(getattr(config, "voice_orchestrator_enabled", False))
    ):
        return False
    return is_raspberry_pi_host_fn()


__all__ = [
    "RUNTIME_SUPERVISOR_ENV_KEY",
    "env_flag",
    "is_raspberry_pi_host",
    "should_enable_display_surface",
    "uses_pi_runtime_root",
]
