# CHANGELOG: 2026-03-30
# BUG-1: Fixed a verifier-backend lifecycle leak. The old code created a second OpenAI backend for
#        `verification_stt` but did not keep or close that backend on success, which can leak HTTP
#        clients / sockets across bundle rebuilds and long-running Pi deployments.
# BUG-2: Fixed brittle verifier-config cloning. The old code used `dataclasses.replace()` directly,
#        which breaks if `TwinrConfig` is migrated to a Pydantic/BaseSettings-style config object.
# BUG-3: Fixed truthy-string handling for `streaming_transcript_verifier_enabled`; values like
#        `"false"` no longer enable costly verifier startup by accident.
# BUG-4: Fixed blank verifier-model handling. Blank / whitespace verifier model values now fall back
#        to the primary OpenAI STT model instead of failing later in provider initialization.
# SEC-1: Hardened error rendering so malformed provider values are only echoed when they match a safe
#        provider-token pattern; short secret-like strings are now redacted instead of being logged.
# IMP-1: Added explicit provider capability metadata and selection objects so downstream orchestration
#        can reason about tool-calling / realtime / redaction constraints instead of hard-coding vendors.
# IMP-2: Added managed bundle shutdown (`close()`, context-manager support) backed by `ExitStack`,
#        which is the modern Python pattern for resource-safe dynamic factories.
# IMP-3: Added lazy verifier initialization by default for Pi-class devices. This preserves drop-in
#        semantics while avoiding a second OpenAI client until the verifier is actually used.
# BREAKING: `verification_stt` may now be lazily materialized on first use when
#           `streaming_transcript_verifier_enabled` is true, unless
#           `streaming_transcript_verifier_lazy_init` is set to false.

"""Assemble Twinr's configured runtime provider bundle.

This module validates the top-level provider-selection fields on
``TwinrConfig``, constructs the selected STT, LLM, and TTS providers, and
rolls back any resources created here if bootstrap fails. Provider-specific
transport and SDK logic stays in the sibling provider packages.

2026 notes
----------
This version upgrades the bundle factory in three ways:

1. Explicit resource ownership:
   Resources created here stay owned here and are closed explicitly via
   ``StreamingProviderBundle.close()`` rather than relying on garbage
   collection.
2. Capability metadata:
   The returned bundle now exposes resolved provider names and a capability
   matrix that downstream routing code can inspect.
3. Edge-aware verifier behavior:
   The optional transcript verifier is lazily initialized by default to avoid
   paying memory/socket overhead on Raspberry Pi-class devices until it is
   actually used.
"""

from __future__ import annotations

import copy
from contextlib import ExitStack
from dataclasses import dataclass, field, is_dataclass, replace
import logging
import re
from threading import RLock
from typing import Callable, Final, Mapping, TypeVar

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import (
    CombinedSpeechAgentProvider,
    ProviderBundle,
    SpeechToTextProvider,
)
from twinr.providers.deepgram import DeepgramSpeechToTextProvider
from twinr.providers.groq import GroqAgentTextProvider, GroqToolCallingAgentProvider
from twinr.providers.openai import OpenAIBackend, OpenAIProviderBundle

_SAFE_PROVIDER_VALUE_RE: Final[re.Pattern[str]] = re.compile(
    r"^[a-z0-9][a-z0-9._-]{0,31}$"
)
_TRUE_STRINGS: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_FALSE_STRINGS: Final[frozenset[str]] = frozenset({"0", "false", "no", "off", ""})
_MISSING: Final[object] = object()

_STT_CAPABILITIES: Final[dict[str, dict[str, bool]]] = {
    "openai": {
        "realtime_transcription": True,
        "context_aware_oob_transcription": True,
        "pii_redaction": False,
    },
    "deepgram": {
        "realtime_transcription": True,
        "dynamic_reconfiguration": True,
        "pii_redaction": True,
    },
}
_LLM_CAPABILITIES: Final[dict[str, dict[str, bool]]] = {
    "openai": {
        "tool_calling": True,
        "structured_outputs": True,
        "native_speech_to_speech": True,
    },
    "groq": {
        "tool_calling": True,
        "structured_outputs": True,
        "server_side_tools": True,
        "structured_outputs_with_tools": False,
    },
}
_TTS_CAPABILITIES: Final[dict[str, dict[str, bool]]] = {
    "openai": {
        "streaming_audio": True,
        "styleable_voice": True,
    },
}

_SUPPORTED_STT_PROVIDERS: Final[tuple[str, ...]] = tuple(_STT_CAPABILITIES)
_SUPPORTED_LLM_PROVIDERS: Final[tuple[str, ...]] = tuple(_LLM_CAPABILITIES)
_SUPPORTED_TTS_PROVIDERS: Final[tuple[str, ...]] = tuple(_TTS_CAPABILITIES)

_T = TypeVar("_T")
_LOGGER = logging.getLogger(__name__)


class ProviderConfigurationError(RuntimeError):
    """Signal a deterministic provider-bootstrap failure."""


@dataclass(frozen=True)
class ProviderSelection:
    """Normalized provider names chosen for this runtime bundle."""

    stt: str
    llm: str
    tts: str


@dataclass(frozen=True)
class ProviderCapabilityMatrix:
    """Resolved provider capabilities for the active bundle.

    The capability names are intentionally generic so downstream policy and
    routing code can reason about behavior instead of vendor strings.
    """

    stt: Mapping[str, bool]
    llm: Mapping[str, bool]
    tts: Mapping[str, bool]

    def supports(self, capability: str) -> bool:
        """Return whether any active provider advertises ``capability``."""
        return bool(
            self.stt.get(capability, False)
            or self.llm.get(capability, False)
            or self.tts.get(capability, False)
        )


@dataclass
class StreamingProviderBundle(ProviderBundle):
    """Extend ``ProviderBundle`` with Twinr-specific support infrastructure.

    Attributes:
        print_backend: Combined speech/text facade used for printing and
            support flows that need coordinated STT, agent, and TTS access.
        support_backend: Shared OpenAI backend reused for support and fallback
            provider paths.
        selection: Resolved provider names for STT, LLM, and TTS.
        capabilities: Capability matrix for the resolved providers.
        verification_stt: Optional second-pass STT provider used to verify
            short or low-confidence streaming turns.
    """

    print_backend: CombinedSpeechAgentProvider
    support_backend: OpenAIBackend
    selection: ProviderSelection
    capabilities: ProviderCapabilityMatrix
    verification_stt: SpeechToTextProvider | None = None
    _resource_stack: ExitStack | None = field(default=None, repr=False, compare=False)
    _close_lock: RLock = field(
        default_factory=RLock, init=False, repr=False, compare=False
    )

    def close(self) -> None:
        """Close resources created by the bundle factory."""
        stack: ExitStack | None
        with self._close_lock:
            stack = self._resource_stack
            self._resource_stack = None
        if stack is not None:
            stack.close()

    def __enter__(self) -> "StreamingProviderBundle":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class _LazyManagedSpeechToTextProvider:
    """Lazily build and own an STT provider.

    This keeps optional verifier infrastructure off the hot startup path on
    constrained devices while still supporting deterministic cleanup.
    """

    def __init__(
        self,
        *,
        name: str,
        factory: Callable[[], tuple[SpeechToTextProvider, tuple[object, ...]]],
    ) -> None:
        self._name = name
        self._factory = factory
        self._provider: SpeechToTextProvider | None = None
        self._owned_resources: tuple[object, ...] = ()
        self._lock = RLock()
        self._closed = False

    def _materialize(self) -> SpeechToTextProvider:
        with self._lock:
            if self._provider is not None:
                return self._provider
            if self._closed:
                raise ProviderConfigurationError(
                    f"{self._name} cannot be initialized after close()"
                )
            provider, owned_resources = self._factory()
            self._provider = _require_component(provider, name=self._name)
            self._owned_resources = tuple(owned_resources)
            return self._provider

    def close(self) -> None:
        with self._lock:
            provider = self._provider
            owned_resources = self._owned_resources
            self._provider = None
            self._owned_resources = ()
            self._closed = True

        closed_ids: set[int] = set()
        if provider is not None:
            closed_ids.add(id(provider))
            _close_if_possible(provider)
        for resource in reversed(owned_resources):
            resource_id = id(resource)
            if resource_id in closed_ids:
                continue
            closed_ids.add(resource_id)
            _close_if_possible(resource)

    def __getattr__(self, name: str) -> object:
        return getattr(self._materialize(), name)

    def __repr__(self) -> str:
        state = "materialized" if self._provider is not None else "lazy"
        return f"{type(self).__name__}(name={self._name!r}, state={state})"


def _get_config_attr(
    config: TwinrConfig,
    attr_name: str,
    *,
    default: object = _MISSING,
) -> object:
    value = getattr(config, attr_name, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise ProviderConfigurationError(
                f"TwinrConfig is missing required attribute {attr_name!r}"
            )
        return default
    return value


def _normalize_optional_text(value: object, *, attr_name: str) -> str | None:
    """Normalize an optional text config field.

    Returns ``None`` for unset / blank values.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProviderConfigurationError(
            f"{attr_name} must be a string or unset, got {type(value).__name__}"
        )
    normalized = value.strip()
    return normalized or None


def _coerce_bool(value: object, *, attr_name: str, default: bool = False) -> bool:
    """Parse a permissive bool field without accepting arbitrary truthy strings."""
    if value is _MISSING:
        return default
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ProviderConfigurationError(
            f"{attr_name} must be a bool-like value, got int {value!r}"
        )
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    raise ProviderConfigurationError(
        f"{attr_name} must be a bool-like value, got {type(value).__name__}"
    )


def _describe_config_value(value: object) -> str:
    """Render a bounded, secret-safe config description for error messages."""
    if isinstance(value, str):
        candidate = value.strip().lower()
        if not candidate:
            return "<blank>"
        if _SAFE_PROVIDER_VALUE_RE.fullmatch(candidate):
            return repr(candidate)
        return f"<redacted str len={len(value)}>"
    rendered = repr(value)
    if len(rendered) > 48:
        return f"<{type(value).__name__}>"
    return rendered


def _normalize_provider_name(
    value: object,
    *,
    env_key: str,
    default: str,
) -> str:
    """Normalize one configured provider name or fall back to the default."""
    if value is None:
        return default
    if not isinstance(value, str):
        raise ProviderConfigurationError(
            f"{env_key} must be a string or unset, got {type(value).__name__}"
        )
    normalized = value.strip().lower()
    return normalized or default


def _resolve_provider_name(
    config: TwinrConfig,
    *,
    attr_name: str,
    env_key: str,
    default: str,
    supported: tuple[str, ...],
) -> str:
    """Resolve and validate one provider selector from ``TwinrConfig``."""
    raw_value = _get_config_attr(config, attr_name)
    resolved = _normalize_provider_name(raw_value, env_key=env_key, default=default)
    if resolved not in supported:
        supported_values = ", ".join(supported)
        raise ProviderConfigurationError(
            f"Unsupported {env_key}: {_describe_config_value(raw_value)}. "
            f"Supported values: {supported_values}"
        )
    return resolved


def _resolve_bool_attr(
    config: TwinrConfig,
    *,
    attr_name: str,
    default: bool = False,
) -> bool:
    return _coerce_bool(
        _get_config_attr(config, attr_name, default=default),
        attr_name=attr_name,
        default=default,
    )


def _clone_config_with_overrides(
    config: TwinrConfig, **overrides: object
) -> TwinrConfig:
    """Clone config across dataclass- and pydantic-style config objects."""
    if is_dataclass(config):
        try:
            return replace(config, **overrides)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise ProviderConfigurationError(
                "Failed to clone TwinrConfig with dataclass overrides"
            ) from exc

    model_copy = getattr(config, "model_copy", None)
    if callable(model_copy):
        try:
            return model_copy(update=overrides, deep=False)
        except TypeError:
            if not overrides:
                try:
                    return model_copy(deep=False)
                except Exception as exc:  # pragma: no cover - defensive wrapper
                    raise ProviderConfigurationError(
                        "Failed to clone TwinrConfig with model_copy(deep=False)"
                    ) from exc
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise ProviderConfigurationError(
                "Failed to clone TwinrConfig with model_copy(update=...)"
            ) from exc

    copy_method = getattr(config, "copy", None)
    if callable(copy_method):
        if overrides:
            try:
                return copy_method(update=overrides)
            except TypeError:
                pass
            except Exception as exc:  # pragma: no cover - defensive wrapper
                raise ProviderConfigurationError(
                    "Failed to clone TwinrConfig with copy(update=...)"
                ) from exc
        else:
            try:
                return copy_method()
            except TypeError:
                pass
            except Exception as exc:  # pragma: no cover - defensive wrapper
                raise ProviderConfigurationError(
                    "Failed to clone TwinrConfig with copy()"
                ) from exc

    try:
        cloned = copy.copy(config)
    except Exception as exc:  # pragma: no cover - defensive wrapper
        raise ProviderConfigurationError(
            "Failed to clone TwinrConfig with copy.copy()"
        ) from exc

    for attr_name, value in overrides.items():
        try:
            setattr(cloned, attr_name, value)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise ProviderConfigurationError(
                f"Failed to set cloned TwinrConfig attribute {attr_name!r}"
            ) from exc
    return cloned


def _resolve_verifier_model(config: TwinrConfig) -> str | None:
    """Resolve the verifier STT model, falling back to the primary OpenAI STT model."""
    verifier_model = _normalize_optional_text(
        _get_config_attr(
            config,
            "streaming_transcript_verifier_model",
            default=None,
        ),
        attr_name="streaming_transcript_verifier_model",
    )
    if verifier_model is not None:
        return verifier_model

    primary_model = _normalize_optional_text(
        _get_config_attr(config, "openai_stt_model", default=None),
        attr_name="openai_stt_model",
    )
    return primary_model


def _build_openai_streaming_verification_provider(
    config: TwinrConfig,
) -> tuple[SpeechToTextProvider, tuple[object, ...]]:
    """Build the bounded OpenAI verifier STT used for short streaming turns.

    The live streaming path can receive short or low-confidence transcripts.
    This verifier is kept separate from the main support backend so verifier
    model overrides do not mutate the primary OpenAI runtime.
    """
    verifier_model = _resolve_verifier_model(config)
    verifier_config = _clone_config_with_overrides(
        config,
        **({"openai_stt_model": verifier_model} if verifier_model is not None else {}),
    )

    verifier_backend: OpenAIBackend | None = None
    try:
        verifier_backend = OpenAIBackend(config=verifier_config)
        verifier_bundle = OpenAIProviderBundle.from_backend(verifier_backend)
        verifier_stt = _require_component(
            verifier_bundle.stt,
            name="openai streaming transcript verifier",
        )
        return verifier_stt, (verifier_backend,)
    except Exception:
        _close_if_possible(verifier_backend)
        raise


def _build_capability_matrix(selection: ProviderSelection) -> ProviderCapabilityMatrix:
    """Resolve the capability matrix for the selected providers."""
    return ProviderCapabilityMatrix(
        stt=dict(_STT_CAPABILITIES[selection.stt]),
        llm=dict(_LLM_CAPABILITIES[selection.llm]),
        tts=dict(_TTS_CAPABILITIES[selection.tts]),
    )


def _require_component(component: _T | None, *, name: str) -> _T:
    """Require that a selected provider component is present."""
    if component is None:
        raise ProviderConfigurationError(
            f"Provider bundle is missing required component: {name}"
        )
    return component


def _close_if_possible(resource: object | None) -> None:
    """Close a created resource without masking the primary failure."""
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            _LOGGER.warning(
                "Provider-factory rollback failed to close %s.",
                type(resource).__name__,
                exc_info=True,
            )


def _register_owned_resource(
    stack: ExitStack,
    resource: object | None,
    *,
    seen_ids: set[int],
) -> None:
    """Register a factory-owned resource for cleanup exactly once."""
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if not callable(close):
        return
    resource_id = id(resource)
    if resource_id in seen_ids:
        return
    seen_ids.add(resource_id)
    stack.callback(_close_if_possible, resource)


def build_streaming_provider_bundle(
    config: TwinrConfig,
    *,
    support_backend: OpenAIBackend | None = None,
) -> StreamingProviderBundle:
    """Build the runtime provider bundle from Twinr's provider config.

    Validates the configured STT, LLM, and TTS provider names before any
    provider-specific work starts. Resources created inside this factory are
    attached to the returned bundle and closed again if bundle creation fails.

    Args:
        config: Runtime config with provider-selection fields and provider
            credentials.
        support_backend: Optional pre-built OpenAI backend to reuse for the
            support path instead of constructing one inside the factory.

    Returns:
        A fully wired ``StreamingProviderBundle`` for the active runtime.

    Raises:
        ProviderConfigurationError: If config is malformed, a provider name is
            unsupported, or a selected provider cannot be initialized safely.
    """
    if config is None:
        raise ProviderConfigurationError("config must not be None")

    selection = ProviderSelection(
        stt=_resolve_provider_name(
            config,
            attr_name="stt_provider",
            env_key="TWINR_STT_PROVIDER",
            default="openai",
            supported=_SUPPORTED_STT_PROVIDERS,
        ),
        llm=_resolve_provider_name(
            config,
            attr_name="llm_provider",
            env_key="TWINR_LLM_PROVIDER",
            default="openai",
            supported=_SUPPORTED_LLM_PROVIDERS,
        ),
        tts=_resolve_provider_name(
            config,
            attr_name="tts_provider",
            env_key="TWINR_TTS_PROVIDER",
            default="openai",
            supported=_SUPPORTED_TTS_PROVIDERS,
        ),
    )
    capabilities = _build_capability_matrix(selection)

    resource_stack = ExitStack()
    owned_resource_ids: set[int] = set()

    try:
        if support_backend is None:
            try:
                support = OpenAIBackend(config=config)
            except Exception as exc:
                raise ProviderConfigurationError(
                    "Failed to initialize OpenAI support backend"
                ) from exc
            _register_owned_resource(
                resource_stack,
                support,
                seen_ids=owned_resource_ids,
            )
        else:
            support = support_backend

        try:
            openai_bundle = OpenAIProviderBundle.from_backend(support)
        except Exception as exc:
            raise ProviderConfigurationError(
                "Failed to initialize OpenAI provider bundle"
            ) from exc

        openai_agent = _require_component(
            openai_bundle.agent,
            name="openai agent",
        )
        print_backend = _require_component(
            openai_bundle.combined,
            name="openai combined backend",
        )

        if selection.stt == "openai":
            stt = _require_component(
                openai_bundle.stt,
                name="openai stt",
            )
        elif selection.stt == "deepgram":
            try:
                stt = DeepgramSpeechToTextProvider(config=config)
            except Exception as exc:
                raise ProviderConfigurationError(
                    "Failed to initialize Deepgram STT provider"
                ) from exc
            _register_owned_resource(
                resource_stack,
                stt,
                seen_ids=owned_resource_ids,
            )
        else:
            raise ProviderConfigurationError(
                f"Unsupported TWINR_STT_PROVIDER: {_describe_config_value(selection.stt)}"
            )

        if selection.llm == "openai":
            agent = openai_agent
            tool_agent = _require_component(
                openai_bundle.tool_agent,
                name="openai tool agent",
            )
        elif selection.llm == "groq":
            try:
                groq_agent = GroqAgentTextProvider(
                    config=config,
                    support_provider=openai_agent,
                )
            except Exception as exc:
                raise ProviderConfigurationError(
                    "Failed to initialize Groq text agent"
                ) from exc
            _register_owned_resource(
                resource_stack,
                groq_agent,
                seen_ids=owned_resource_ids,
            )

            try:
                groq_tool_agent = GroqToolCallingAgentProvider(config=config)
            except Exception as exc:
                raise ProviderConfigurationError(
                    "Failed to initialize Groq tool-calling agent"
                ) from exc
            _register_owned_resource(
                resource_stack,
                groq_tool_agent,
                seen_ids=owned_resource_ids,
            )

            agent = groq_agent
            tool_agent = groq_tool_agent
        else:
            raise ProviderConfigurationError(
                f"Unsupported TWINR_LLM_PROVIDER: {_describe_config_value(selection.llm)}"
            )

        tool_agent = _require_component(
            tool_agent,
            name=f"{selection.llm} tool agent",
        )

        if selection.tts == "openai":
            tts = _require_component(
                openai_bundle.tts,
                name="openai tts",
            )
        else:
            raise ProviderConfigurationError(
                f"Unsupported TWINR_TTS_PROVIDER: {_describe_config_value(selection.tts)}"
            )

        verification_stt: SpeechToTextProvider | None = None
        if _resolve_bool_attr(
            config,
            attr_name="streaming_transcript_verifier_enabled",
            default=False,
        ):
            verifier_lazy = _resolve_bool_attr(
                config,
                attr_name="streaming_transcript_verifier_lazy_init",
                default=True,
            )
            if verifier_lazy:
                verifier_config_snapshot = _clone_config_with_overrides(config)
                lazy_verifier = _LazyManagedSpeechToTextProvider(
                    name="openai streaming transcript verifier",
                    factory=lambda: _build_openai_streaming_verification_provider(
                        verifier_config_snapshot
                    ),
                )
                verification_stt = lazy_verifier
                _register_owned_resource(
                    resource_stack,
                    lazy_verifier,
                    seen_ids=owned_resource_ids,
                )
            else:
                try:
                    verification_stt, verifier_resources = (
                        _build_openai_streaming_verification_provider(config)
                    )
                except Exception as exc:
                    raise ProviderConfigurationError(
                        "Failed to initialize streaming transcript verifier provider"
                    ) from exc
                for resource in verifier_resources:
                    _register_owned_resource(
                        resource_stack,
                        resource,
                        seen_ids=owned_resource_ids,
                    )

        bundle = StreamingProviderBundle(
            stt=stt,
            agent=agent,
            tts=tts,
            tool_agent=tool_agent,
            print_backend=print_backend,
            support_backend=support,
            selection=selection,
            capabilities=capabilities,
            verification_stt=verification_stt,
            _resource_stack=resource_stack.pop_all(),
        )
        _LOGGER.info(
            "Built streaming provider bundle (stt=%s, llm=%s, tts=%s, verifier=%s, verifier_lazy=%s).",
            selection.stt,
            selection.llm,
            selection.tts,
            verification_stt is not None,
            isinstance(verification_stt, _LazyManagedSpeechToTextProvider),
        )
        return bundle
    except ProviderConfigurationError:
        resource_stack.close()
        raise
    except Exception as exc:
        resource_stack.close()
        raise ProviderConfigurationError(
            "Failed to build streaming provider bundle"
        ) from exc
