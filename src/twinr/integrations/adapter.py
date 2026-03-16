"""Define the minimal adapter contract for Twinr integrations.

Adapters expose a manifest and a synchronous ``execute()`` method that turns an
``IntegrationRequest`` into an ``IntegrationResult``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from twinr.integrations.models import IntegrationManifest, IntegrationRequest, IntegrationResult


@runtime_checkable
class IntegrationAdapter(Protocol):
    """Describe the synchronous interface implemented by integration adapters."""

    manifest: IntegrationManifest

    def execute(self, request: IntegrationRequest) -> IntegrationResult:
        """Execute one integration request and return a normalized result."""

        ...


@dataclass(slots=True)
class CallableIntegrationAdapter:
    """Wrap a plain callable in the ``IntegrationAdapter`` protocol."""

    manifest: IntegrationManifest
    handler: Callable[[IntegrationRequest], IntegrationResult]

    def execute(self, request: IntegrationRequest) -> IntegrationResult:
        """Delegate request handling to the wrapped callable."""

        return self.handler(request)
