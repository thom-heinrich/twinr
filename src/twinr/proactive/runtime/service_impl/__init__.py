"""Refactored implementation package for proactive runtime service wiring.

This package holds the real implementation behind
``twinr.proactive.runtime.service``. Import the public API from the legacy
module path; the wrapper remains stable and delegates here.
"""

from twinr.proactive.runtime.service_impl.builder import (
    BuildDefaultProactiveMonitorDependencies,
    build_default_proactive_monitor,
)
from twinr.proactive.runtime.service_impl.coordinator import (
    ProactiveCoordinator,
    ProactiveTickResult,
)
from twinr.proactive.runtime.service_impl.monitor import ProactiveMonitorService

__all__ = [
    "BuildDefaultProactiveMonitorDependencies",
    "ProactiveCoordinator",
    "ProactiveMonitorService",
    "ProactiveTickResult",
    "build_default_proactive_monitor",
]
