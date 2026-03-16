"""Assemble Twinr's configured runtime provider bundle.

This module validates the top-level provider-selection fields on
``TwinrConfig``, constructs the selected STT, LLM, and TTS providers, and
rolls back any resources created here if bootstrap fails. Provider-specific
transport and SDK logic stays in the sibling provider packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeVar

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import CombinedSpeechAgentProvider, ProviderBundle
from twinr.providers.deepgram import DeepgramSpeechToTextProvider
from twinr.providers.groq import GroqAgentTextProvider, GroqToolCallingAgentProvider
from twinr.providers.openai import OpenAIBackend, OpenAIProviderBundle

_SUPPORTED_STT_PROVIDERS: Final[tuple[str, ...]] = ("openai", "deepgram")  # AUDIT-FIX(#2): Centralize supported provider names so validation is deterministic and defaults stay coherent.
_SUPPORTED_LLM_PROVIDERS: Final[tuple[str, ...]] = ("openai", "groq")  # AUDIT-FIX(#2): Centralize supported provider names so validation is deterministic and defaults stay coherent.
_SUPPORTED_TTS_PROVIDERS: Final[tuple[str, ...]] = ("openai",)  # AUDIT-FIX(#2): Centralize supported provider names so validation is deterministic and defaults stay coherent.
_MISSING: Final[object] = object()  # AUDIT-FIX(#1): Distinguish a malformed config object from an unset provider value.

_T = TypeVar("_T")


class ProviderConfigurationError(RuntimeError):
    """Signal a deterministic provider-bootstrap failure."""  # AUDIT-FIX(#1): Give callers a stable exception type for bootstrap/config failures.


@dataclass
class StreamingProviderBundle(ProviderBundle):
    """Extend ``ProviderBundle`` with shared support backends used by Twinr.

    Attributes:
        print_backend: Combined speech/text facade used for printing and
            support flows that need coordinated STT, agent, and TTS access.
        support_backend: Shared OpenAI backend reused for support and fallback
            provider paths.
    """

    print_backend: CombinedSpeechAgentProvider
    support_backend: OpenAIBackend


def _describe_config_value(value: object, *, max_length: int = 32) -> str:
    """Render a bounded config value description for error messages.

    Args:
        value: Raw config value to describe.
        max_length: Maximum number of characters to keep before falling back
            to a type-level placeholder. Defaults to 32.

    Returns:
        A short string that helps diagnose malformed config without dumping
        large or secret-like values into logs.
    """

    if isinstance(value, str):
        if len(value) > max_length:
            return f"<str len={len(value)}>"
        return repr(value)
    rendered = repr(value)
    if len(rendered) > max_length:
        return f"<{type(value).__name__}>"
    return rendered  # AUDIT-FIX(#6): Avoid dumping huge or secret-like garbage values into logs while keeping errors diagnosable.


def _normalize_provider_name(
    value: object,
    *,
    env_key: str,
    default: str,
) -> str:
    """Normalize one configured provider name or fall back to the default.

    Args:
        value: Raw config value for the provider selector.
        env_key: Environment-variable name used in error messages.
        default: Provider name to use when the config is unset or blank.

    Returns:
        A lower-cased provider name suitable for allow-list validation.

    Raises:
        ProviderConfigurationError: If ``value`` is set but is not a string.
    """

    if value is None:
        return default
    if not isinstance(value, str):
        raise ProviderConfigurationError(
            f"{env_key} must be a string or unset, got {type(value).__name__}"
        )  # AUDIT-FIX(#1): Replace accidental AttributeError paths with a deterministic config error.
    normalized = value.strip().lower()
    return normalized or default  # AUDIT-FIX(#2): Treat blank/whitespace values as unset so the documented default still applies.


def _resolve_provider_name(
    config: TwinrConfig,
    *,
    attr_name: str,
    env_key: str,
    default: str,
    supported: tuple[str, ...],
) -> str:
    """Resolve and validate one provider selector from ``TwinrConfig``.

    Args:
        config: Runtime config object that carries provider selection fields.
        attr_name: Attribute name to read from ``config``.
        env_key: Environment-variable name used in error messages.
        default: Provider name to use when the config value is unset or blank.
        supported: Allowed provider names for this selector.

    Returns:
        The normalized provider name after allow-list validation.

    Raises:
        ProviderConfigurationError: If the config object is malformed or the
            resolved provider name is unsupported.
    """

    raw_value = getattr(config, attr_name, _MISSING)
    if raw_value is _MISSING:
        raise ProviderConfigurationError(
            f"TwinrConfig is missing required attribute {attr_name!r}"
        )  # AUDIT-FIX(#1): Fail fast if the injected config object is malformed.

    resolved = _normalize_provider_name(raw_value, env_key=env_key, default=default)
    if resolved not in supported:
        supported_values = ", ".join(supported)
        raise ProviderConfigurationError(
            f"Unsupported {env_key}: {_describe_config_value(raw_value)}. "
            f"Supported values: {supported_values}"
        )  # AUDIT-FIX(#6): Sanitize unexpected config values and include the allow-list in the same error.
    return resolved


def _require_component(component: _T | None, *, name: str) -> _T:
    """Require that a selected provider component is present.

    Args:
        component: Component instance pulled from a provider bundle.
        name: Human-readable component name for the failure message.

    Returns:
        The non-``None`` component instance.

    Raises:
        ProviderConfigurationError: If the component is missing.
    """

    if component is None:
        raise ProviderConfigurationError(
            f"Provider bundle is missing required component: {name}"
        )  # AUDIT-FIX(#3): Detect incomplete provider bundles during bootstrap instead of during a live conversation.
    return component


def _close_if_possible(resource: object | None) -> None:
    """Close a created resource without masking the primary failure.

    Args:
        resource: Resource that may expose a ``close()`` method.
    """

    if resource is None:
        return
    close = getattr(resource, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass  # AUDIT-FIX(#5): Cleanup is best-effort and must never hide the original initialization failure.


def build_streaming_provider_bundle(
    config: TwinrConfig,
    *,
    support_backend: OpenAIBackend | None = None,
) -> StreamingProviderBundle:
    """Build the runtime provider bundle from Twinr's provider config.

    Validates the configured STT, LLM, and TTS provider names before any
    provider-specific work starts. Resources created inside this factory are
    closed again if a later bootstrap step fails.

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
        raise ProviderConfigurationError("config must not be None")  # AUDIT-FIX(#1): Replace a later AttributeError with a deterministic bootstrap failure.

    stt_name = _resolve_provider_name(
        config,
        attr_name="stt_provider",
        env_key="TWINR_STT_PROVIDER",
        default="openai",
        supported=_SUPPORTED_STT_PROVIDERS,
    )
    llm_name = _resolve_provider_name(
        config,
        attr_name="llm_provider",
        env_key="TWINR_LLM_PROVIDER",
        default="openai",
        supported=_SUPPORTED_LLM_PROVIDERS,
    )
    tts_name = _resolve_provider_name(
        config,
        attr_name="tts_provider",
        env_key="TWINR_TTS_PROVIDER",
        default="openai",
        supported=_SUPPORTED_TTS_PROVIDERS,
    )

    owned_resources: list[object] = []  # AUDIT-FIX(#5): Roll back only resources created in this factory if bootstrap aborts part-way.

    try:
        if support_backend is None:
            try:
                support = OpenAIBackend(config=config)
            except Exception as exc:
                raise ProviderConfigurationError(
                    "Failed to initialize OpenAI support backend"
                ) from exc  # AUDIT-FIX(#1): Wrap backend init errors with a stable, actionable failure type.
            owned_resources.append(support)
        else:
            support = support_backend  # AUDIT-FIX(#4): Respect caller-supplied backends even if they define falsey truthiness.

        try:
            openai_bundle = OpenAIProviderBundle.from_backend(support)
        except Exception as exc:
            raise ProviderConfigurationError(
                "Failed to initialize OpenAI provider bundle"
            ) from exc  # AUDIT-FIX(#1): Separate backend-creation failures from bundle-wiring failures.

        openai_agent = _require_component(
            openai_bundle.agent,
            name="openai agent",
        )  # AUDIT-FIX(#3): Validate OpenAI components before they are reused as fallbacks/support.
        print_backend = _require_component(
            openai_bundle.combined,
            name="openai combined backend",
        )  # AUDIT-FIX(#3): Fail fast if the print/support path is unavailable.

        if stt_name == "openai":
            stt = _require_component(
                openai_bundle.stt,
                name="openai stt",
            )  # AUDIT-FIX(#3): Validate the selected STT component eagerly.
        elif stt_name == "deepgram":
            try:
                stt = DeepgramSpeechToTextProvider(config=config)
            except Exception as exc:
                raise ProviderConfigurationError(
                    "Failed to initialize Deepgram STT provider"
                ) from exc  # AUDIT-FIX(#1): Preserve the root cause while making the failing stage explicit.
            owned_resources.append(stt)  # AUDIT-FIX(#5): Roll back Deepgram resources if a later init step fails.
        else:
            raise ProviderConfigurationError(
                f"Unsupported TWINR_STT_PROVIDER: {_describe_config_value(stt_name)}"
            )  # AUDIT-FIX(#2): Defensive branch; prior validation should make this unreachable.

        if llm_name == "openai":
            agent = openai_agent
            tool_agent = _require_component(
                openai_bundle.tool_agent,
                name="openai tool agent",
            )  # AUDIT-FIX(#3): Do not return a bundle that cannot perform required tool calls.
        elif llm_name == "groq":
            try:
                groq_agent = GroqAgentTextProvider(
                    config=config,
                    support_provider=openai_agent,
                )
            except Exception as exc:
                raise ProviderConfigurationError(
                    "Failed to initialize Groq text agent"
                ) from exc  # AUDIT-FIX(#1): Keep the failing stage explicit for operational recovery.
            owned_resources.append(groq_agent)  # AUDIT-FIX(#5): Roll back Groq text resources if a later init step fails.

            try:
                groq_tool_agent = GroqToolCallingAgentProvider(config=config)
            except Exception as exc:
                raise ProviderConfigurationError(
                    "Failed to initialize Groq tool-calling agent"
                ) from exc  # AUDIT-FIX(#1): Keep the failing stage explicit for operational recovery.
            owned_resources.append(groq_tool_agent)  # AUDIT-FIX(#5): Roll back Groq tool resources if a later init step fails.

            agent = groq_agent
            tool_agent = groq_tool_agent
        else:
            raise ProviderConfigurationError(
                f"Unsupported TWINR_LLM_PROVIDER: {_describe_config_value(llm_name)}"
            )  # AUDIT-FIX(#2): Defensive branch; prior validation should make this unreachable.

        tool_agent = _require_component(
            tool_agent,
            name=f"{llm_name} tool agent",
        )  # AUDIT-FIX(#3): Preserve the original non-None tool-agent guarantee across all providers.

        if tts_name == "openai":
            tts = _require_component(
                openai_bundle.tts,
                name="openai tts",
            )  # AUDIT-FIX(#3): Validate the selected TTS component eagerly.
        else:
            raise ProviderConfigurationError(
                f"Unsupported TWINR_TTS_PROVIDER: {_describe_config_value(tts_name)}"
            )  # AUDIT-FIX(#2): Defensive branch; prior validation should make this unreachable.

        return StreamingProviderBundle(
            stt=stt,
            agent=agent,
            tts=tts,
            tool_agent=tool_agent,
            print_backend=print_backend,
            support_backend=support,
        )
    except ProviderConfigurationError:
        for resource in reversed(owned_resources):
            _close_if_possible(resource)
        raise
    except Exception as exc:
        for resource in reversed(owned_resources):
            _close_if_possible(resource)
        raise ProviderConfigurationError(
            "Failed to build streaming provider bundle"
        ) from exc  # AUDIT-FIX(#1): Collapse stray bootstrap failures into the same stable error type expected by supervisors/tests.
