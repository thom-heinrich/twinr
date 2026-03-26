"""Expose the stable Twinr browser-automation boundary.

This package intentionally owns only the versioned contracts plus the loader
that reaches into the repo-root ``browser_automation/`` workspace. Concrete
browser stacks stay optional and local-only until Twinr adopts one.
"""

from twinr.browser_automation.contracts import (
    BROWSER_AUTOMATION_STATUSES,
    BrowserAutomationArtifact,
    BrowserAutomationAvailability,
    BrowserAutomationDriver,
    BrowserAutomationRequest,
    BrowserAutomationResult,
)
from twinr.browser_automation.loader import (
    BROWSER_AUTOMATION_FACTORY_NAME,
    DEFAULT_BROWSER_AUTOMATION_ENTRY_MODULE,
    DEFAULT_BROWSER_AUTOMATION_WORKSPACE,
    BrowserAutomationLoadError,
    BrowserAutomationUnavailableError,
    load_browser_automation_driver,
    probe_browser_automation,
    resolve_browser_automation_entry_path,
    resolve_browser_automation_workspace,
)

__all__ = [
    "BROWSER_AUTOMATION_FACTORY_NAME",
    "BROWSER_AUTOMATION_STATUSES",
    "BrowserAutomationArtifact",
    "BrowserAutomationAvailability",
    "BrowserAutomationDriver",
    "BrowserAutomationLoadError",
    "BrowserAutomationRequest",
    "BrowserAutomationResult",
    "BrowserAutomationUnavailableError",
    "DEFAULT_BROWSER_AUTOMATION_ENTRY_MODULE",
    "DEFAULT_BROWSER_AUTOMATION_WORKSPACE",
    "load_browser_automation_driver",
    "probe_browser_automation",
    "resolve_browser_automation_entry_path",
    "resolve_browser_automation_workspace",
]
