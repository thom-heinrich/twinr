"""Build the deterministic MVP capability view for the Adaptive Skill Engine."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .contracts import CapabilityAvailability, CapabilityDefinition
from .status import CapabilityRiskClass, CapabilityStatus

if TYPE_CHECKING:
    from twinr.agent.base_agent.config import TwinrConfig
    from twinr.integrations.runtime import ManagedIntegrationsRuntime

ManagedIntegrationFactory = Callable[..., "ManagedIntegrationsRuntime"]

_DEFAULT_CAPABILITY_DEFINITIONS: tuple[CapabilityDefinition, ...] = (
    CapabilityDefinition(
        capability_id="camera",
        module_name="camera",
        summary="Observe semantic camera-side presence and visibility signals.",
        risk_class=CapabilityRiskClass.MODERATE,
    ),
    CapabilityDefinition(
        capability_id="pir",
        module_name="pir",
        summary="Observe passive infrared motion signals from the configured sensor.",
        risk_class=CapabilityRiskClass.LOW,
    ),
    CapabilityDefinition(
        capability_id="speaker",
        module_name="speaker",
        summary="Speak short voice-first output through Twinr's bounded TTS path.",
        risk_class=CapabilityRiskClass.HIGH,
    ),
    CapabilityDefinition(
        capability_id="llm_call",
        module_name="llm_call",
        summary="Run bounded reasoning and extraction calls through Twinr's provider layer.",
        risk_class=CapabilityRiskClass.HIGH,
    ),
    CapabilityDefinition(
        capability_id="memory",
        module_name="memory",
        summary="Read and write bounded self-coding memory state for skills and jobs.",
        risk_class=CapabilityRiskClass.MODERATE,
    ),
    CapabilityDefinition(
        capability_id="scheduler",
        module_name="scheduler",
        summary="Schedule bounded follow-ups and recurring checks for learned skills.",
        risk_class=CapabilityRiskClass.MODERATE,
    ),
    CapabilityDefinition(
        capability_id="rules",
        module_name="rules",
        summary="Compose bounded trigger and condition logic for automation-first skills.",
        risk_class=CapabilityRiskClass.LOW,
    ),
    CapabilityDefinition(
        capability_id="safety",
        module_name="safety",
        summary="Read Twinr safety context such as night mode and self-disable paths.",
        risk_class=CapabilityRiskClass.HIGH,
    ),
    CapabilityDefinition(
        capability_id="email",
        module_name="email",
        summary="Read and prepare email through the managed email mailbox integration.",
        risk_class=CapabilityRiskClass.HIGH,
        requires_configuration=True,
        integration_id="email_mailbox",
    ),
    CapabilityDefinition(
        capability_id="calendar",
        module_name="calendar",
        summary="Read agenda and upcoming events through the managed calendar integration.",
        risk_class=CapabilityRiskClass.MODERATE,
        requires_configuration=True,
        integration_id="calendar_agenda",
    ),
)


class SelfCodingCapabilityRegistry:
    """Resolve curated ASE capabilities to their current runtime readiness."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        env_path: str | Path | None = None,
        definitions: tuple[CapabilityDefinition, ...] | None = None,
        integration_runtime_factory: ManagedIntegrationFactory | None = None,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve(strict=False)
        self.env_path = None if env_path is None else Path(env_path).expanduser().resolve(strict=False)
        self._definitions = tuple(definitions or _DEFAULT_CAPABILITY_DEFINITIONS)
        self._integration_runtime_factory = integration_runtime_factory
        self._assert_unique_capability_ids()

    @classmethod
    def from_config(
        cls,
        config: "TwinrConfig",
        *,
        env_path: str | Path | None = None,
        definitions: tuple[CapabilityDefinition, ...] | None = None,
        integration_runtime_factory: ManagedIntegrationFactory | None = None,
    ) -> "SelfCodingCapabilityRegistry":
        project_root = getattr(config, "project_root", ".")
        resolved_env_path = env_path
        if resolved_env_path is None:
            resolved_env_path = Path(project_root).expanduser().resolve(strict=False) / ".env"
        return cls(
            project_root=project_root,
            env_path=resolved_env_path,
            definitions=definitions,
            integration_runtime_factory=integration_runtime_factory,
        )

    def definitions(self) -> tuple[CapabilityDefinition, ...]:
        """Return the curated capability definitions for the current registry."""

        return self._definitions

    def definition_for(self, capability_id: str) -> CapabilityDefinition | None:
        """Return one capability definition by id."""

        for definition in self._definitions:
            if definition.capability_id == capability_id:
                return definition
        return None

    def availability_snapshot(self) -> tuple[CapabilityAvailability, ...]:
        """Return the current readiness snapshot for all curated capabilities."""

        integration_runtime, integration_error = self._load_managed_integrations()
        readiness_by_id = {}
        if integration_runtime is not None:
            readiness_by_id = {
                item.integration_id: item
                for item in getattr(integration_runtime, "readiness", ())
            }

        snapshot: list[CapabilityAvailability] = []
        for definition in self._definitions:
            if definition.integration_id is None:
                snapshot.append(
                    CapabilityAvailability(
                        capability_id=definition.capability_id,
                        status=CapabilityStatus.READY,
                        detail="Built-in Twinr runtime capability is available for ASE.",
                        metadata={"source": "builtin"},
                    )
                )
                continue

            if integration_error is not None:
                snapshot.append(
                    CapabilityAvailability(
                        capability_id=definition.capability_id,
                        status=CapabilityStatus.BLOCKED,
                        detail=integration_error,
                        integration_id=definition.integration_id,
                        metadata={"source": "managed_integration", "error": integration_error},
                    )
                )
                continue

            readiness = readiness_by_id.get(definition.integration_id)
            if readiness is None:
                snapshot.append(
                    CapabilityAvailability(
                        capability_id=definition.capability_id,
                        status=CapabilityStatus.UNCONFIGURED,
                        detail="Managed integration is not configured yet.",
                        integration_id=definition.integration_id,
                        metadata={"source": "managed_integration", "readiness_status": "missing"},
                    )
                )
                continue

            readiness_status = str(getattr(readiness, "status", "") or "").strip().lower()
            readiness_detail = (
                str(getattr(readiness, "detail", "") or "").strip()
                or str(getattr(readiness, "summary", "") or "").strip()
                or "Managed integration status is unknown."
            )
            status = CapabilityStatus.READY if readiness_status == "ok" else CapabilityStatus.UNCONFIGURED
            snapshot.append(
                CapabilityAvailability(
                    capability_id=definition.capability_id,
                    status=status,
                    detail=readiness_detail,
                    integration_id=definition.integration_id,
                    metadata={"source": "managed_integration", "readiness_status": readiness_status or "unknown"},
                )
            )

        return tuple(snapshot)

    def availability_for(self, capability_id: str) -> CapabilityAvailability | None:
        """Return the current readiness record for one capability id."""

        for record in self.availability_snapshot():
            if record.capability_id == capability_id:
                return record
        return None

    def configured_capability_ids(self) -> tuple[str, ...]:
        """Return the ids of capabilities that are currently ready for ASE."""

        return tuple(record.capability_id for record in self.availability_snapshot() if record.configured)

    def _assert_unique_capability_ids(self) -> None:
        seen: set[str] = set()
        for definition in self._definitions:
            if definition.capability_id in seen:
                raise ValueError(f"Duplicate capability id: {definition.capability_id}")
            seen.add(definition.capability_id)

    def _load_managed_integrations(self) -> tuple["ManagedIntegrationsRuntime" | None, str | None]:
        runtime_factory = self._integration_runtime_factory
        if runtime_factory is None:
            from twinr.integrations.runtime import build_managed_integrations

            runtime_factory = build_managed_integrations
        try:
            try:
                runtime = runtime_factory(self.project_root, env_path=self.env_path)
            except TypeError:
                runtime = runtime_factory(self.project_root, self.env_path)
        except Exception as exc:
            return None, f"Managed integration readiness could not be loaded: {type(exc).__name__}: {exc}"
        return runtime, None
