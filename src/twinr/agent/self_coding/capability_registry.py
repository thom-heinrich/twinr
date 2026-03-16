"""Build the deterministic MVP capability view for the Adaptive Skill Engine."""

from __future__ import annotations

import inspect
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

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

_SAFE_MANAGED_INTEGRATION_LOAD_ERROR = "Managed integration readiness could not be loaded."
_INVALID_MANAGED_READINESS_ERROR = "Managed integration readiness data is invalid."
_INVALID_PROJECT_ROOT_ERROR = "Project root path is invalid."
_INVALID_ENV_PATH_ERROR = "Managed integration configuration path is invalid."

_READY_READINESS_STATUSES = frozenset({"ok", "ready", "available", "healthy"})
_UNCONFIGURED_READINESS_STATUSES = frozenset({"missing", "unconfigured", "not_configured", "not-configured"})


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
        raw_project_root = Path(project_root).expanduser()
        # AUDIT-FIX(#2): Resolve the project root once and validate the env path relative to it instead of trusting process-CWD-relative input.
        self.project_root = raw_project_root.resolve(strict=False)

        raw_env_path = None if env_path is None else Path(env_path).expanduser()
        if raw_env_path is not None and not raw_env_path.is_absolute():
            # AUDIT-FIX(#2): Treat relative env paths as project-root-relative so callers cannot escape the Twinr deployment tree via cwd quirks.
            raw_env_path = self.project_root / raw_env_path
        self.env_path = None if raw_env_path is None else raw_env_path.resolve(strict=False)

        # AUDIT-FIX(#2): Record configuration path violations so managed integrations degrade safely instead of reading arbitrary files or symlinks.
        self._configuration_error = self._validate_configuration_paths(
            resolved_project_root=self.project_root,
            raw_env_path=raw_env_path,
            resolved_env_path=self.env_path,
        )
        # AUDIT-FIX(#7): Respect intentionally empty custom definition sets instead of silently restoring defaults.
        self._definitions = tuple(definitions) if definitions is not None else _DEFAULT_CAPABILITY_DEFINITIONS
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
        readiness_by_id: dict[str, Any] = {}
        if integration_runtime is not None and integration_error is None:
            # AUDIT-FIX(#3): Validate readiness payload structure so malformed runtime data blocks integrations instead of crashing the registry.
            readiness_by_id, readiness_error = self._readiness_by_integration_id(integration_runtime)
            if readiness_error is not None:
                integration_error = readiness_error
                readiness_by_id = {}

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
                        metadata={
                            "source": "managed_integration",
                            "error_code": self._managed_integration_error_code(integration_error),
                        },
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

            readiness_status = self._normalize_text(self._readiness_field(readiness, "status")).lower()
            readiness_detail = self._readiness_detail(readiness, readiness_status)
            # AUDIT-FIX(#4): Distinguish configuration gaps from operational failures so callers receive the right remediation path.
            status = self._capability_status_from_readiness(readiness_status)
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
        if self._configuration_error is not None:
            return None, self._configuration_error

        runtime_factory = self._integration_runtime_factory
        if runtime_factory is None:
            try:
                # AUDIT-FIX(#5): Import failures in the default runtime path must degrade to BLOCKED instead of crashing the caller.
                from twinr.integrations.runtime import build_managed_integrations
            except Exception as exc:
                return None, self._safe_managed_integrations_error(exc)

            runtime_factory = build_managed_integrations
        try:
            runtime = self._invoke_runtime_factory(runtime_factory)
        except Exception as exc:
            # AUDIT-FIX(#1): Return sanitized availability detail so secrets and filesystem paths do not leak through API/UI surfaces.
            return None, self._safe_managed_integrations_error(exc)
        return runtime, None

    def _invoke_runtime_factory(self, runtime_factory: ManagedIntegrationFactory) -> "ManagedIntegrationsRuntime":
        supports_env_path_keyword = self._factory_accepts_env_path_keyword(runtime_factory)
        if supports_env_path_keyword is False:
            return runtime_factory(self.project_root, self.env_path)
        try:
            return runtime_factory(self.project_root, env_path=self.env_path)
        except TypeError as exc:
            # AUDIT-FIX(#6): Only retry with positional env_path when the callable truly rejects the keyword, not when the factory body raised its own TypeError.
            if supports_env_path_keyword is True or not self._is_env_path_signature_error(exc):
                raise
            return runtime_factory(self.project_root, self.env_path)

    @staticmethod
    def _factory_accepts_env_path_keyword(runtime_factory: ManagedIntegrationFactory) -> bool | None:
        # AUDIT-FIX(#6): Inspect the callable signature first so a real TypeError inside the factory is not misdiagnosed as a calling-convention mismatch.
        try:
            signature = inspect.signature(runtime_factory)
        except (TypeError, ValueError):
            return None

        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return True
        return "env_path" in signature.parameters

    @staticmethod
    def _is_env_path_signature_error(exc: TypeError) -> bool:
        message = str(exc)
        keyword_error_fragments = (
            "unexpected keyword argument",
            "positional-only arguments passed as keyword",
            "takes no keyword arguments",
            "does not take keyword arguments",
        )
        if "env_path" not in message or not any(fragment in message for fragment in keyword_error_fragments):
            return False

        extracted_traceback = traceback.extract_tb(exc.__traceback__)
        return len(extracted_traceback) <= 1

    # AUDIT-FIX(#3): Normalize readiness payloads from mappings or objects and reject malformed or duplicate integration ids deterministically.
    def _readiness_by_integration_id(
        self,
        integration_runtime: "ManagedIntegrationsRuntime",
    ) -> tuple[dict[str, Any], str | None]:
        readiness = getattr(integration_runtime, "readiness", ())
        if readiness is None:
            return {}, None

        readiness_by_id: dict[str, Any] = {}
        if isinstance(readiness, Mapping):
            items = readiness.items()
            for integration_id, item in items:
                normalized_integration_id = self._normalize_text(integration_id)
                if not normalized_integration_id or normalized_integration_id in readiness_by_id:
                    return {}, _INVALID_MANAGED_READINESS_ERROR
                readiness_by_id[normalized_integration_id] = item
            return readiness_by_id, None

        try:
            iterator = iter(readiness)
        except TypeError:
            return {}, _INVALID_MANAGED_READINESS_ERROR

        try:
            for item in iterator:
                normalized_integration_id = self._normalize_text(self._readiness_field(item, "integration_id"))
                if not normalized_integration_id or normalized_integration_id in readiness_by_id:
                    return {}, _INVALID_MANAGED_READINESS_ERROR
                readiness_by_id[normalized_integration_id] = item
        except Exception:
            return {}, _INVALID_MANAGED_READINESS_ERROR

        return readiness_by_id, None

    @staticmethod
    def _readiness_field(readiness: Any, field_name: str) -> Any:
        if isinstance(readiness, Mapping):
            return readiness.get(field_name)
        return getattr(readiness, field_name, None)

    def _readiness_detail(self, readiness: Any, readiness_status: str) -> str:
        detail = (
            self._normalize_text(self._readiness_field(readiness, "detail"))
            or self._normalize_text(self._readiness_field(readiness, "summary"))
        )
        if detail:
            return detail
        if readiness_status in _READY_READINESS_STATUSES:
            return "Managed integration is ready."
        if readiness_status in _UNCONFIGURED_READINESS_STATUSES:
            return "Managed integration is not configured yet."
        return "Managed integration is currently unavailable."

    @staticmethod
    def _capability_status_from_readiness(readiness_status: str) -> CapabilityStatus:
        if readiness_status in _READY_READINESS_STATUSES:
            return CapabilityStatus.READY
        if readiness_status in _UNCONFIGURED_READINESS_STATUSES:
            return CapabilityStatus.UNCONFIGURED
        return CapabilityStatus.BLOCKED

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    # AUDIT-FIX(#2): Reject env-path traversal, existing symlink targets, and non-file configuration paths before handing them to the runtime factory.
    def _validate_configuration_paths(
        self,
        *,
        resolved_project_root: Path,
        raw_env_path: Path | None,
        resolved_env_path: Path | None,
    ) -> str | None:
        if resolved_project_root.exists() and not resolved_project_root.is_dir():
            return _INVALID_PROJECT_ROOT_ERROR

        if raw_env_path is None or resolved_env_path is None:
            return None

        try:
            if raw_env_path.exists() and raw_env_path.is_symlink():
                return _INVALID_ENV_PATH_ERROR
        except OSError:
            return _INVALID_ENV_PATH_ERROR

        if not self._path_is_within_root(resolved_env_path, resolved_project_root):
            return _INVALID_ENV_PATH_ERROR

        if resolved_env_path.exists() and not resolved_env_path.is_file():
            return _INVALID_ENV_PATH_ERROR

        return None

    @staticmethod
    def _path_is_within_root(candidate: Path, root: Path) -> bool:
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        return True

    @staticmethod
    def _safe_managed_integrations_error(exc: BaseException) -> str:
        # AUDIT-FIX(#1): Collapse raw exceptions into coarse safe messages so availability detail stays non-sensitive and senior-safe.
        if isinstance(exc, PermissionError):
            return "Managed integration configuration cannot be accessed."
        if isinstance(exc, FileNotFoundError):
            return "Managed integration configuration file is missing."
        if isinstance(exc, ModuleNotFoundError):
            return "Managed integrations runtime is unavailable on this device."
        if isinstance(exc, TimeoutError):
            return "Managed integration readiness timed out."
        return _SAFE_MANAGED_INTEGRATION_LOAD_ERROR

    @staticmethod
    def _managed_integration_error_code(detail: str) -> str:
        if detail in {_INVALID_PROJECT_ROOT_ERROR, _INVALID_ENV_PATH_ERROR}:
            return "invalid_configuration"
        if detail == _INVALID_MANAGED_READINESS_ERROR:
            return "invalid_readiness"
        if detail == "Managed integrations runtime is unavailable on this device.":
            return "runtime_unavailable"
        if detail == "Managed integration readiness timed out.":
            return "timeout"
        if detail == "Managed integration configuration cannot be accessed.":
            return "configuration_unreadable"
        if detail == "Managed integration configuration file is missing.":
            return "configuration_missing"
        return "load_failed"