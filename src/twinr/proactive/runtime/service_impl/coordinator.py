"""Public proactive coordinator class composed from focused mixins.

The wrapper module ``twinr.proactive.runtime.service`` re-exports this class
unchanged; no caller migration is required.
"""

# mypy: ignore-errors

from twinr.proactive.runtime.service_impl.coordinator_core import (
    ProactiveCoordinatorCoreMixin,
    ProactiveTickResult,
)
from twinr.proactive.runtime.service_impl.coordinator_display import ProactiveCoordinatorDisplayMixin
from twinr.proactive.runtime.service_impl.coordinator_observation import ProactiveCoordinatorObservationMixin
from twinr.proactive.runtime.service_impl.coordinator_perception import ProactiveCoordinatorPerceptionMixin


class ProactiveCoordinator(
    ProactiveCoordinatorDisplayMixin,
    ProactiveCoordinatorObservationMixin,
    ProactiveCoordinatorPerceptionMixin,
    ProactiveCoordinatorCoreMixin,
):
    """Coordinate one proactive monitor tick across sensors and policies.

    The coordinator owns the runtime-facing orchestration for PIR, vision,
    ambient audio, presence sessions, trigger review, and automation
    observation export. Lower-level trigger scoring and voice matching remain
    in sibling packages.
    """


__all__ = ["ProactiveCoordinator", "ProactiveTickResult"]
