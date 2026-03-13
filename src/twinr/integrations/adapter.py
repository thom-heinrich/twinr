from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from twinr.integrations.models import IntegrationManifest, IntegrationRequest, IntegrationResult


@runtime_checkable
class IntegrationAdapter(Protocol):
    manifest: IntegrationManifest

    def execute(self, request: IntegrationRequest) -> IntegrationResult:
        ...


@dataclass(slots=True)
class CallableIntegrationAdapter:
    manifest: IntegrationManifest
    handler: Callable[[IntegrationRequest], IntegrationResult]

    def execute(self, request: IntegrationRequest) -> IntegrationResult:
        return self.handler(request)
