"""Derive benchmark auth/state context for official WebArena Verified tasks.

This module stays intentionally narrow. It only reads official task metadata,
resolves storage-state paths, and converts supported official environment auth
settings into generic Twinr browser-context metadata. Any active login/state
materialization lives in a separate benchmark bootstrap module so Twinr's
runtime browser code stays free of benchmark-only account logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any
import json

from webarena_verified.api import WebArenaVerified
from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
from webarena_verified.types.task import WebArenaVerifiedTask


@dataclass(frozen=True, slots=True)
class WebArenaVerifiedAuthContext:
    """Describe optional auth/state bootstrap inputs for one benchmark task."""

    require_login: bool
    storage_state_path: str | None = None
    extra_http_headers: dict[str, str] | None = None


def load_raw_task_metadata(*, task_id: int) -> dict[str, Any]:
    """Load one task entry from the shipped raw WebArena task dataset."""

    dataset_root = resources.files("webarena_verified").joinpath("assets", "dataset")
    raw_path = dataset_root.joinpath("test.raw.json")
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    for item in payload:
        if int(item.get("task_id")) == int(task_id):
            return dict(item)
    raise ValueError(f"Unknown raw WebArena task id: {task_id}")


def derive_task_auth_context(
    *,
    task: WebArenaVerifiedTask,
    config: WebArenaVerifiedConfig,
    benchmark: WebArenaVerified | None = None,
    auth_state_root: Path | None = None,
) -> WebArenaVerifiedAuthContext:
    """Return generic Twinr browser bootstrap context for one official task."""

    raw_task = load_raw_task_metadata(task_id=int(task.task_id))
    require_login = bool(raw_task.get("require_login", False))
    storage_state_path = resolve_storage_state_path(
        raw_path=str(raw_task.get("storage_state") or "").strip(),
        auth_state_root=auth_state_root,
    )
    environment = lookup_environment(config=config, site=task.sites[0])
    extra_http_headers = _derive_header_login_headers(
        environment=environment,
        site_name=str(task.sites[0].value),
        benchmark=benchmark,
    )
    return WebArenaVerifiedAuthContext(
        require_login=require_login,
        storage_state_path=storage_state_path,
        extra_http_headers=extra_http_headers or None,
    )


def lookup_environment(*, config: WebArenaVerifiedConfig, site: Any) -> EnvironmentConfig | None:
    """Resolve one site mapping from the official config object."""

    if config.environments is None:
        return None
    for key in (site, getattr(site, "value", None), getattr(site, "url_name_template", None)):
        if key is None:
            continue
        try:
            environment = config.environments.get(key)
        except TypeError:
            environment = None
        if environment is not None:
            return environment
    return None


def resolve_storage_state_path(*, raw_path: str, auth_state_root: Path | None) -> str | None:
    """Resolve one official raw storage-state reference into an existing path."""

    cleaned = str(raw_path or "").strip()
    if not cleaned or auth_state_root is None:
        return None
    candidate = (Path(auth_state_root).expanduser().resolve() / cleaned).resolve()
    if candidate.is_file():
        return str(candidate)
    fallback = (Path(auth_state_root).expanduser().resolve() / Path(cleaned).name).resolve()
    if fallback.is_file():
        return str(fallback)
    return None


def _derive_header_login_headers(
    *,
    environment: EnvironmentConfig | None,
    site_name: str,
    benchmark: WebArenaVerified | None,
) -> dict[str, str]:
    """Convert official header-login settings into generic browser headers."""

    if environment is None or environment.credentials is None or benchmark is None:
        return {}
    header_name = benchmark.get_custom_auth_header_name(site_name)
    username = str(environment.credentials.get("username") or "").strip()
    if not header_name or not username:
        return {}
    if not bool(environment.use_header_login):
        return {}
    return {str(header_name): username}
