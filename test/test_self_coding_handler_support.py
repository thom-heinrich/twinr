"""Focused tests for self-coding handler runtime support helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from twinr.agent.tools.handlers.self_coding_support import ensure_self_coding_runtime


def _owner(tmp_path: Path) -> SimpleNamespace:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = SimpleNamespace(
        project_root=str(project_root),
        automation_store_path=str(project_root / "state" / "automations.json"),
        local_timezone_name="Europe/Berlin",
        automation_max_entries=32,
    )
    return SimpleNamespace(config=config)


def test_ensure_self_coding_runtime_reuses_cached_helpers_without_config_drift(tmp_path: Path) -> None:
    owner = _owner(tmp_path)

    first = ensure_self_coding_runtime(owner)
    second = ensure_self_coding_runtime(owner)

    assert second["store"] is first["store"]
    assert second["automation_store"] is first["automation_store"]
    assert second["activation_service"] is first["activation_service"]
    assert second["flow"] is first["flow"]


def test_ensure_self_coding_runtime_rebuilds_runtime_when_public_config_changes(tmp_path: Path) -> None:
    owner = _owner(tmp_path)

    first = ensure_self_coding_runtime(owner)
    owner.config.automation_max_entries = 64
    second = ensure_self_coding_runtime(owner)

    assert second["automation_store"] is not first["automation_store"]
    assert second["activation_service"] is not first["activation_service"]
    assert second["health_service"] is not first["health_service"]
