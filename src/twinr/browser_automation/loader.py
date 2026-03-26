"""Load one optional local browser automation driver from the repo workspace.

Twinr keeps the stable request/response boundary in ``src/twinr`` while the
actual browser stack stays in the gitignored repo-root ``browser_automation/``
folder. This loader is the only bridge between those two worlds.
"""

from __future__ import annotations

from hashlib import blake2s
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path, PurePosixPath
import sys
from threading import RLock
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, cast

from twinr.browser_automation.contracts import BrowserAutomationAvailability, BrowserAutomationDriver

if TYPE_CHECKING:
    from twinr.agent.base_agent.config import TwinrConfig


DEFAULT_BROWSER_AUTOMATION_WORKSPACE = "browser_automation"
DEFAULT_BROWSER_AUTOMATION_ENTRY_MODULE = "adapter.py"
BROWSER_AUTOMATION_FACTORY_NAME = "create_browser_automation_driver"

_LOAD_LOCK = RLock()


class BrowserAutomationLoadError(RuntimeError):
    """Raised when the optional browser automation workspace cannot be loaded safely."""


class BrowserAutomationUnavailableError(BrowserAutomationLoadError):
    """Raised when the optional browser automation workspace is disabled or missing."""


def _safe_relative_path(value: object, *, field_name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise BrowserAutomationLoadError(f"{field_name} must be a non-empty relative path")
    path = PurePosixPath(value.strip())
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise BrowserAutomationLoadError(f"{field_name} must stay within the repo root")
    return path


def resolve_browser_automation_workspace(*, config: TwinrConfig, project_root: str | Path) -> Path:
    """Resolve the configured local browser automation workspace under the repo root."""

    root_path = Path(project_root).expanduser().resolve()
    relative = _safe_relative_path(
        getattr(config, "browser_automation_workspace_path", DEFAULT_BROWSER_AUTOMATION_WORKSPACE),
        field_name="browser_automation_workspace_path",
    )
    workspace = (root_path / Path(relative)).resolve()
    try:
        workspace.relative_to(root_path)
    except ValueError as exc:
        raise BrowserAutomationLoadError("browser_automation_workspace_path escapes the repo root") from exc
    return workspace


def resolve_browser_automation_entry_path(*, config: TwinrConfig, project_root: str | Path) -> Path:
    """Resolve the configured browser automation entry module inside the workspace."""

    workspace = resolve_browser_automation_workspace(config=config, project_root=project_root)
    entry_relative = _safe_relative_path(
        getattr(config, "browser_automation_entry_module", DEFAULT_BROWSER_AUTOMATION_ENTRY_MODULE),
        field_name="browser_automation_entry_module",
    )
    if entry_relative.suffix != ".py":
        raise BrowserAutomationLoadError("browser_automation_entry_module must point to a .py file")
    entry_path = (workspace / Path(entry_relative)).resolve()
    try:
        entry_path.relative_to(workspace)
    except ValueError as exc:
        raise BrowserAutomationLoadError("browser_automation_entry_module escapes the workspace") from exc
    return entry_path


def probe_browser_automation(*, config: TwinrConfig, project_root: str | Path) -> BrowserAutomationAvailability:
    """Return a small availability snapshot without importing the local backend."""

    enabled = bool(getattr(config, "browser_automation_enabled", False))
    workspace_path: Path | None = None
    entry_path: Path | None = None
    try:
        workspace_path = resolve_browser_automation_workspace(config=config, project_root=project_root)
        entry_path = resolve_browser_automation_entry_path(config=config, project_root=project_root)
    except BrowserAutomationLoadError as exc:
        return BrowserAutomationAvailability(
            enabled=enabled,
            available=False,
            reason=str(exc),
            workspace_path=str(workspace_path) if workspace_path is not None else None,
            entry_module=str(entry_path) if entry_path is not None else None,
        )

    if not enabled:
        return BrowserAutomationAvailability(
            enabled=False,
            available=False,
            reason="Browser automation is disabled by config.",
            workspace_path=str(workspace_path),
            entry_module=str(entry_path),
        )
    if not workspace_path.exists():
        return BrowserAutomationAvailability(
            enabled=True,
            available=False,
            reason="Browser automation workspace does not exist.",
            workspace_path=str(workspace_path),
            entry_module=str(entry_path),
        )
    if not workspace_path.is_dir():
        return BrowserAutomationAvailability(
            enabled=True,
            available=False,
            reason="Browser automation workspace is not a directory.",
            workspace_path=str(workspace_path),
            entry_module=str(entry_path),
        )
    if not entry_path.exists():
        return BrowserAutomationAvailability(
            enabled=True,
            available=False,
            reason="Browser automation entry module does not exist.",
            workspace_path=str(workspace_path),
            entry_module=str(entry_path),
        )
    if not entry_path.is_file():
        return BrowserAutomationAvailability(
            enabled=True,
            available=False,
            reason="Browser automation entry module is not a file.",
            workspace_path=str(workspace_path),
            entry_module=str(entry_path),
        )
    return BrowserAutomationAvailability(
        enabled=True,
        available=True,
        reason="Browser automation workspace is available.",
        workspace_path=str(workspace_path),
        entry_module=str(entry_path),
    )


def _module_name_for_entry(entry_path: Path) -> str:
    digest = blake2s(str(entry_path).encode("utf-8", "strict"), digest_size=8).hexdigest()
    return f"twinr_browser_automation_local_{digest}"


def _load_module(entry_path: Path) -> ModuleType:
    module_name = _module_name_for_entry(entry_path)
    spec = spec_from_file_location(module_name, entry_path)
    if spec is None or spec.loader is None:
        raise BrowserAutomationLoadError(f"Could not create import spec for {entry_path.name}")
    module = module_from_spec(spec)
    with _LOAD_LOCK:
        previous_module = sys.modules.get(module_name)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module
            raise BrowserAutomationLoadError(f"Failed to import {entry_path.name}") from exc
    return module


def _validated_factory(factory_obj: object, *, entry_name: str) -> Callable[..., Any]:
    if not callable(factory_obj):
        raise BrowserAutomationLoadError(
            f"{entry_name} must export {BROWSER_AUTOMATION_FACTORY_NAME}(*, config, project_root)"
        )
    return cast(Callable[..., Any], factory_obj)


def load_browser_automation_driver(*, config: TwinrConfig, project_root: str | Path) -> BrowserAutomationDriver:
    """Load and validate the local browser automation driver.

    The local entry module must expose
    ``create_browser_automation_driver(*, config, project_root)`` and return an
    object implementing :class:`BrowserAutomationDriver`.
    """

    availability = probe_browser_automation(config=config, project_root=project_root)
    if not availability.available:
        raise BrowserAutomationUnavailableError(availability.reason or "Browser automation is unavailable.")
    entry_path = resolve_browser_automation_entry_path(config=config, project_root=project_root)
    module = _load_module(entry_path)
    factory: Callable[..., object] = _validated_factory(
        getattr(module, BROWSER_AUTOMATION_FACTORY_NAME, None),
        entry_name=entry_path.name,
    )
    try:
        # The adapter export is validated above; pylint cannot follow this dynamic import edge.
        driver = factory(  # pylint: disable=not-callable
            config=config,
            project_root=Path(project_root).expanduser().resolve(),
        )
    except BrowserAutomationLoadError:
        raise
    except Exception as exc:
        raise BrowserAutomationLoadError(
            f"{BROWSER_AUTOMATION_FACTORY_NAME} failed for {entry_path.name}"
        ) from exc
    if not isinstance(driver, BrowserAutomationDriver):
        raise BrowserAutomationLoadError(
            f"{BROWSER_AUTOMATION_FACTORY_NAME} must return a BrowserAutomationDriver"
        )
    return driver


__all__ = [
    "BROWSER_AUTOMATION_FACTORY_NAME",
    "DEFAULT_BROWSER_AUTOMATION_ENTRY_MODULE",
    "DEFAULT_BROWSER_AUTOMATION_WORKSPACE",
    "BrowserAutomationLoadError",
    "BrowserAutomationUnavailableError",
    "load_browser_automation_driver",
    "probe_browser_automation",
    "resolve_browser_automation_entry_path",
    "resolve_browser_automation_workspace",
]
