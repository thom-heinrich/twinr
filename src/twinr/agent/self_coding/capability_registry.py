"""Build the deterministic MVP capability view for the Adaptive Skill Engine."""

from __future__ import annotations

import inspect
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .contracts import CapabilityAvailability, CapabilityDefinition
from .modules import MODULE_LIBRARY, SelfCodingModuleSpec, module_spec_for
from .status import CapabilityStatus

if TYPE_CHECKING:
    from twinr.agent.base_agent.config import TwinrConfig
    from twinr.integrations.runtime import ManagedIntegrationsRuntime

ManagedIntegrationFactory = Callable[..., "ManagedIntegrationsRuntime"]


# AUDIT-FIX(#4): Freeze the imported module library immediately so later mutations cannot change the registry behind our back.
_DEFAULT_MODULE_SPECS: tuple[SelfCodingModuleSpec, ...] = tuple(MODULE_LIBRARY)


def _build_capability_definitions(
    module_specs: tuple[SelfCodingModuleSpec, ...],
) -> tuple[CapabilityDefinition, ...]:
    return tuple(spec.capability_definition() for spec in module_specs)


try:
    # AUDIT-FIX(#4): Build default definitions behind a guarded import-time boundary so one bad module spec does not crash module import.
    _DEFAULT_CAPABILITY_DEFINITIONS: tuple[CapabilityDefinition, ...] = _build_capability_definitions(
        _DEFAULT_MODULE_SPECS
    )
    _DEFAULT_DEFINITION_ERROR: str | None = None
except Exception:
    _DEFAULT_CAPABILITY_DEFINITIONS = ()
    _DEFAULT_DEFINITION_ERROR = "Capability definitions could not be loaded."

_SAFE_MANAGED_INTEGRATION_LOAD_ERROR = "Managed integration readiness could not be loaded."
_INVALID_MANAGED_READINESS_ERROR = "Managed integration readiness data is invalid."
_INVALID_MANAGED_RUNTIME_ERROR = "Managed integrations runtime is invalid."
_INVALID_PROJECT_ROOT_ERROR = "Project root path is invalid."
_INVALID_ENV_PATH_ERROR = "Managed integration configuration path is invalid."
_CAPABILITY_DEFINITION_LOAD_ERROR = "Capability definitions could not be loaded."

_READY_READINESS_STATUSES = frozenset({"ok", "ready", "available", "healthy"})
_UNCONFIGURED_READINESS_STATUSES = frozenset({"missing", "unconfigured", "not_configured", "not-configured"})
_SUPPORTED_MODULE_SPEC_LOOKUP_FIELDS = ("capability_id", "module_name", "module_id", "name")


class SelfCodingCapabilityRegistry:
    """Resolve curated ASE capabilities to their current runtime readiness."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        env_path: str | Path | None = None,
        definitions: tuple[CapabilityDefinition, ...] | None = None,
        module_specs: tuple[SelfCodingModuleSpec, ...] | None = None,
        integration_runtime_factory: ManagedIntegrationFactory | None = None,
    ) -> None:
        raw_project_root, project_root_error = self._coerce_path_input(
            project_root,
            invalid_error=_INVALID_PROJECT_ROOT_ERROR,
            fallback=Path("."),
        )
        # AUDIT-FIX(#1): Bad project_root values must degrade to a blocked registry instead of exploding during Path() / resolve().
        self.project_root = self._safe_resolve_path(raw_project_root or Path("."), fallback=Path("."))

        raw_env_path, env_path_error = self._coerce_path_input(
            env_path,
            invalid_error=_INVALID_ENV_PATH_ERROR,
            allow_none=True,
        )
        if raw_env_path is not None and not raw_env_path.is_absolute():
            # AUDIT-FIX(#1): Resolve relative env paths against the validated project root, not the process cwd.
            raw_env_path = self.project_root / raw_env_path
        self.env_path = None if raw_env_path is None else self._safe_resolve_path(
            raw_env_path,
            fallback=self.project_root / ".env",
        )

        # AUDIT-FIX(#1): Keep invalid path inputs in a safe blocked state instead of letting filesystem edge cases crash later.
        self._configuration_error = (
            project_root_error
            or env_path_error
            or self._validate_configuration_paths(
                resolved_project_root=self.project_root,
                raw_env_path=raw_env_path,
                resolved_env_path=self.env_path,
            )
        )
        self._module_specs = tuple(module_specs) if module_specs is not None else _DEFAULT_MODULE_SPECS
        self._uses_default_module_specs = module_specs is None
        self._assert_unique_module_ids()

        self._definition_error: str | None = None
        if definitions is not None:
            self._definitions = tuple(definitions)
        elif self._uses_default_module_specs:
            # AUDIT-FIX(#4): Preserve the curated default definitions when they loaded cleanly, but remember guarded-load failures for later blocked snapshots.
            self._definitions = _DEFAULT_CAPABILITY_DEFINITIONS
            self._definition_error = _DEFAULT_DEFINITION_ERROR
        else:
            # AUDIT-FIX(#5): Derive default definitions from the active custom module specs so the registry cannot mix custom modules with global definitions.
            self._definitions, self._definition_error = self._definitions_for_module_specs(self._module_specs)

        self._integration_runtime_factory = integration_runtime_factory
        self._assert_unique_capability_ids()

    @classmethod
    def from_config(
        cls,
        config: "TwinrConfig",
        *,
        env_path: str | Path | None = None,
        definitions: tuple[CapabilityDefinition, ...] | None = None,
        module_specs: tuple[SelfCodingModuleSpec, ...] | None = None,
        integration_runtime_factory: ManagedIntegrationFactory | None = None,
    ) -> "SelfCodingCapabilityRegistry":
        project_root = getattr(config, "project_root", ".")
        resolved_env_path = env_path
        if resolved_env_path is None:
            # AUDIT-FIX(#1): Let __init__ resolve the default env path relative to project_root so invalid config values cannot crash from_config.
            resolved_env_path = ".env"
        return cls(
            project_root=project_root,
            env_path=resolved_env_path,
            definitions=definitions,
            module_specs=module_specs,
            integration_runtime_factory=integration_runtime_factory,
        )

    def definitions(self) -> tuple[CapabilityDefinition, ...]:
        """Return the curated capability definitions for the current registry."""

        return self._definitions

    def definition_for(self, capability_id: str) -> CapabilityDefinition | None:
        """Return one capability definition by id."""

        normalized_capability_id = self._normalize_identifier(capability_id)
        for definition in self._definitions:
            # AUDIT-FIX(#7): Normalize lookup ids so stray whitespace does not create silent misses for callers or tests.
            if self._normalize_identifier(definition.capability_id) == normalized_capability_id:
                return definition
        return None

    def module_specs(self) -> tuple[SelfCodingModuleSpec, ...]:
        """Return the curated module specs behind the registry."""

        return self._module_specs

    def module_spec_for(self, capability_id: str) -> SelfCodingModuleSpec | None:
        """Return one curated module spec by capability id or module name."""

        normalized_capability_id = self._normalize_identifier(capability_id)
        for spec in self._module_specs:
            # AUDIT-FIX(#5): Search the active registry module specs first so custom registries do not accidentally leak back to the global module library.
            if self._module_spec_matches(spec, normalized_capability_id):
                return spec
        if self._uses_default_module_specs:
            return module_spec_for(capability_id)
        return None

    def availability_snapshot(self) -> tuple[CapabilityAvailability, ...]:
        """Return the current readiness snapshot for all curated capabilities."""

        if self._definition_error is not None:
            return self._blocked_snapshot_from_module_specs(self._definition_error)

        integration_runtime, integration_error = self._load_managed_integrations()
        readiness_by_id: dict[str, Any] = {}
        if integration_runtime is not None and integration_error is None:
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

            # AUDIT-FIX(#7): Normalize integration ids before lookup so harmless formatting drift does not show capabilities as unconfigured.
            readiness = readiness_by_id.get(self._normalize_identifier(definition.integration_id))
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

            readiness_status = self._normalize_status(self._readiness_field(readiness, "status"))
            readiness_detail = self._readiness_detail(readiness, readiness_status)
            snapshot.append(
                CapabilityAvailability(
                    capability_id=definition.capability_id,
                    status=self._capability_status_from_readiness(readiness_status),
                    detail=readiness_detail,
                    integration_id=definition.integration_id,
                    metadata={"source": "managed_integration", "readiness_status": readiness_status or "unknown"},
                )
            )

        return tuple(snapshot)

    def availability_for(self, capability_id: str) -> CapabilityAvailability | None:
        """Return the current readiness record for one capability id."""

        normalized_capability_id = self._normalize_identifier(capability_id)
        for record in self.availability_snapshot():
            # AUDIT-FIX(#7): Normalize lookup ids for snapshot reads too, otherwise whitespace-only mismatches silently fail.
            if self._normalize_identifier(record.capability_id) == normalized_capability_id:
                return record
        return None

    def configured_capability_ids(self) -> tuple[str, ...]:
        """Return the ids of capabilities that are currently ready for ASE."""

        return tuple(record.capability_id for record in self.availability_snapshot() if record.configured)

    def _assert_unique_capability_ids(self) -> None:
        seen: set[str] = set()
        for definition in self._definitions:
            normalized_capability_id = self._normalize_identifier(definition.capability_id)
            if not normalized_capability_id:
                raise ValueError("Capability id cannot be blank.")
            if normalized_capability_id in seen:
                raise ValueError(f"Duplicate capability id: {normalized_capability_id}")
            seen.add(normalized_capability_id)

    def _assert_unique_module_ids(self) -> None:
        seen: set[str] = set()
        for spec in self._module_specs:
            normalized_capability_id = self._normalize_identifier(getattr(spec, "capability_id", None))
            if not normalized_capability_id:
                raise ValueError("Module capability id cannot be blank.")
            if normalized_capability_id in seen:
                raise ValueError(f"Duplicate module capability id: {normalized_capability_id}")
            seen.add(normalized_capability_id)

    def _load_managed_integrations(self) -> tuple["ManagedIntegrationsRuntime" | None, str | None]:
        if self._configuration_error is not None:
            return None, self._configuration_error

        runtime_factory = self._integration_runtime_factory
        if runtime_factory is None:
            try:
                from twinr.integrations.runtime import build_managed_integrations
            except Exception as exc:
                return None, self._safe_managed_integrations_error(exc)

            runtime_factory = build_managed_integrations
        try:
            runtime = self._invoke_runtime_factory(runtime_factory)
        except Exception as exc:
            return None, self._safe_managed_integrations_error(exc)
        if runtime is None:
            # AUDIT-FIX(#6): A factory that returns None violated the runtime contract; report BLOCKED instead of pretending everything is merely unconfigured.
            return None, _INVALID_MANAGED_RUNTIME_ERROR
        return runtime, None

    def _invoke_runtime_factory(self, runtime_factory: ManagedIntegrationFactory) -> "ManagedIntegrationsRuntime":
        supports_env_path_keyword = self._factory_accepts_env_path_keyword(runtime_factory)
        if supports_env_path_keyword is False:
            return runtime_factory(self.project_root, self.env_path)
        try:
            return runtime_factory(self.project_root, env_path=self.env_path)
        except TypeError as exc:
            if supports_env_path_keyword is True or not self._is_env_path_signature_error(exc):
                raise
            return runtime_factory(self.project_root, self.env_path)

    @staticmethod
    def _factory_accepts_env_path_keyword(runtime_factory: ManagedIntegrationFactory) -> bool | None:
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

    def _readiness_by_integration_id(
        self,
        integration_runtime: "ManagedIntegrationsRuntime",
    ) -> tuple[dict[str, Any], str | None]:
        try:
            # AUDIT-FIX(#2): Treat readiness as untrusted runtime data: attribute access itself may raise or be missing.
            readiness = getattr(integration_runtime, "readiness")
        except Exception:
            return {}, _INVALID_MANAGED_READINESS_ERROR
        if readiness is None:
            return {}, None

        readiness_by_id: dict[str, Any] = {}
        if isinstance(readiness, Mapping):
            try:
                # AUDIT-FIX(#2): Snapshot mapping items up front so concurrent mutation cannot raise mid-iteration.
                items = tuple(readiness.items())
            except Exception:
                return {}, _INVALID_MANAGED_READINESS_ERROR
            for integration_id, item in items:
                normalized_integration_id = self._normalize_identifier(integration_id)
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
                normalized_integration_id = self._normalize_identifier(
                    self._readiness_field(item, "integration_id")
                )
                if not normalized_integration_id or normalized_integration_id in readiness_by_id:
                    return {}, _INVALID_MANAGED_READINESS_ERROR
                readiness_by_id[normalized_integration_id] = item
        except Exception:
            return {}, _INVALID_MANAGED_READINESS_ERROR

        return readiness_by_id, None

    @staticmethod
    def _readiness_field(readiness: Any, field_name: str) -> Any:
        try:
            # AUDIT-FIX(#2): Guard field extraction because custom mapping / property accessors may raise arbitrary exceptions.
            if isinstance(readiness, Mapping):
                return readiness.get(field_name)
            return getattr(readiness, field_name, None)
        except Exception:
            return None

    def _readiness_detail(self, readiness: Any, readiness_status: str) -> str:
        del readiness
        # AUDIT-FIX(#3): Never surface raw runtime detail/summary directly because they may contain secrets, filesystem paths, or stack traces.
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
    def _normalize_text(value: Any, *, max_length: int | None = None) -> str:
        if value is None:
            return ""
        try:
            text = str(value).strip()
        except Exception:
            return ""
        if max_length is not None and len(text) > max_length:
            return ""
        return text

    @classmethod
    def _normalize_identifier(cls, value: Any) -> str:
        return cls._normalize_text(value, max_length=255)

    @classmethod
    def _normalize_status(cls, value: Any) -> str:
        return cls._normalize_text(value, max_length=64).lower()

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
            # AUDIT-FIX(#1): Reject any existing or broken symlink outright instead of letting the runtime chase attacker-controlled targets.
            if raw_env_path.is_symlink():
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
        if detail == _INVALID_MANAGED_RUNTIME_ERROR:
            return "invalid_runtime"
        if detail == _CAPABILITY_DEFINITION_LOAD_ERROR:
            return "invalid_definitions"
        if detail == "Managed integrations runtime is unavailable on this device.":
            return "runtime_unavailable"
        if detail == "Managed integration readiness timed out.":
            return "timeout"
        if detail == "Managed integration configuration cannot be accessed.":
            return "configuration_unreadable"
        if detail == "Managed integration configuration file is missing.":
            return "configuration_missing"
        return "load_failed"

    @staticmethod
    def _coerce_path_input(
        value: str | Path | None,
        *,
        invalid_error: str,
        allow_none: bool = False,
        fallback: Path | None = None,
    ) -> tuple[Path | None, str | None]:
        if value is None:
            if allow_none:
                return None, None
            return fallback, invalid_error
        try:
            return Path(value).expanduser(), None
        except (TypeError, ValueError, RuntimeError):
            return fallback, invalid_error

    @staticmethod
    def _safe_resolve_path(path: Path, *, fallback: Path) -> Path:
        try:
            return path.resolve(strict=False)
        except (OSError, RuntimeError):
            return fallback.resolve(strict=False)

    @staticmethod
    def _definitions_for_module_specs(
        module_specs: tuple[SelfCodingModuleSpec, ...],
    ) -> tuple[tuple[CapabilityDefinition, ...], str | None]:
        try:
            return _build_capability_definitions(module_specs), None
        except Exception:
            return (), _CAPABILITY_DEFINITION_LOAD_ERROR

    def _blocked_snapshot_from_module_specs(self, detail: str) -> tuple[CapabilityAvailability, ...]:
        snapshot: list[CapabilityAvailability] = []
        for index, spec in enumerate(self._module_specs, start=1):
            capability_id = self._normalize_identifier(getattr(spec, "capability_id", None)) or f"unknown_capability_{index}"
            snapshot.append(
                CapabilityAvailability(
                    capability_id=capability_id,
                    status=CapabilityStatus.BLOCKED,
                    detail=detail,
                    metadata={"source": "module_spec", "error_code": self._managed_integration_error_code(detail)},
                )
            )
        return tuple(snapshot)

    def _module_spec_matches(self, spec: SelfCodingModuleSpec, capability_id: str) -> bool:
        if not capability_id:
            return False
        for field_name in _SUPPORTED_MODULE_SPEC_LOOKUP_FIELDS:
            if self._normalize_identifier(getattr(spec, field_name, None)) == capability_id:
                return True
        return False