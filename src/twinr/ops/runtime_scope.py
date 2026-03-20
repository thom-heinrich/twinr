"""Build scoped runtime configs for non-primary Twinr processes.

Auxiliary loops such as external text channels must not reuse the primary
display/runtime snapshot path. Otherwise a crashing side process can overwrite
the face and status that the main Pi runtime is showing.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig

_RUNTIME_SCOPE_DIRNAME = "runtime-scopes"


def build_scoped_runtime_config(
    config: TwinrConfig,
    *,
    scope_name: str,
    restore_runtime_state_on_startup: bool | None = None,
) -> TwinrConfig:
    """Return a config clone whose runtime snapshot is isolated per scope."""

    scoped_path = resolve_scoped_runtime_state_path(config, scope_name=scope_name)
    return replace(
        config,
        runtime_state_path=str(scoped_path),
        restore_runtime_state_on_startup=(
            config.restore_runtime_state_on_startup
            if restore_runtime_state_on_startup is None
            else bool(restore_runtime_state_on_startup)
        ),
    )


def resolve_scoped_runtime_state_path(config: TwinrConfig, *, scope_name: str) -> Path:
    """Resolve the scoped runtime snapshot path below the primary state folder."""

    project_root = Path(config.project_root).expanduser().resolve(strict=False)
    primary_path = Path(config.runtime_state_path).expanduser()
    if not primary_path.is_absolute():
        primary_path = project_root / primary_path
    safe_scope = _normalize_scope_name(scope_name)
    return (primary_path.parent / _RUNTIME_SCOPE_DIRNAME / safe_scope / primary_path.name).resolve(strict=False)


def _normalize_scope_name(value: str) -> str:
    parts = [
        "".join(character.lower() if character.isalnum() else "-" for character in chunk).strip("-")
        for chunk in str(value or "").strip().split("/")
    ]
    normalized = "-".join(part for part in parts if part)
    if normalized:
        return normalized
    raise ValueError("scope_name must contain at least one alphanumeric character")
