"""Expose Twinr ops APIs without importing the full runtime stack eagerly."""

from __future__ import annotations

from importlib import import_module


_EXPORTS: dict[str, tuple[str, str]] = {
    "DeviceFact": ("twinr.ops.devices", "DeviceFact"),
    "DeviceOverview": ("twinr.ops.devices", "DeviceOverview"),
    "DeviceStatus": ("twinr.ops.devices", "DeviceStatus"),
    "collect_device_overview": ("twinr.ops.devices", "collect_device_overview"),
    "ServiceHealth": ("twinr.ops.health", "ServiceHealth"),
    "TwinrSystemHealth": ("twinr.ops.health", "TwinrSystemHealth"),
    "collect_system_health": ("twinr.ops.health", "collect_system_health"),
    "ConfigCheck": ("twinr.ops.checks", "ConfigCheck"),
    "check_summary": ("twinr.ops.checks", "check_summary"),
    "run_config_checks": ("twinr.ops.checks", "run_config_checks"),
    "TwinrOpsEventStore": ("twinr.ops.events", "TwinrOpsEventStore"),
    "compact_text": ("twinr.ops.events", "compact_text"),
    "TwinrInstanceAlreadyRunningError": (
        "twinr.ops.locks",
        "TwinrInstanceAlreadyRunningError",
    ),
    "TwinrInstanceLock": ("twinr.ops.locks", "TwinrInstanceLock"),
    "loop_instance_lock": ("twinr.ops.locks", "loop_instance_lock"),
    "loop_lock_owner": ("twinr.ops.locks", "loop_lock_owner"),
    "loop_lock_path": ("twinr.ops.locks", "loop_lock_path"),
    "TwinrOpsPaths": ("twinr.ops.paths", "TwinrOpsPaths"),
    "resolve_ops_paths": ("twinr.ops.paths", "resolve_ops_paths"),
    "resolve_ops_paths_for_config": ("twinr.ops.paths", "resolve_ops_paths_for_config"),
    "RemoteMemoryWatchdog": ("twinr.ops.remote_memory_watchdog", "RemoteMemoryWatchdog"),
    "RemoteMemoryWatchdogSample": (
        "twinr.ops.remote_memory_watchdog",
        "RemoteMemoryWatchdogSample",
    ),
    "RemoteMemoryWatchdogSnapshot": (
        "twinr.ops.remote_memory_watchdog",
        "RemoteMemoryWatchdogSnapshot",
    ),
    "OpenAIEnvContractStatus": (
        "twinr.ops.openai_env_contract",
        "OpenAIEnvContractStatus",
    ),
    "check_openai_env_contract": (
        "twinr.ops.openai_env_contract",
        "check_openai_env_contract",
    ),
    "SelfTestResult": ("twinr.ops.self_test", "SelfTestResult"),
    "TwinrSelfTestRunner": ("twinr.ops.self_test", "TwinrSelfTestRunner"),
    "SupportBundleInfo": ("twinr.ops.support", "SupportBundleInfo"),
    "build_support_bundle": ("twinr.ops.support", "build_support_bundle"),
    "redact_env_values": ("twinr.ops.support", "redact_env_values"),
    "TokenUsage": ("twinr.ops.usage", "TokenUsage"),
    "TwinrUsageStore": ("twinr.ops.usage", "TwinrUsageStore"),
    "UsageRecord": ("twinr.ops.usage", "UsageRecord"),
    "UsageSummary": ("twinr.ops.usage", "UsageSummary"),
    "extract_model_name": ("twinr.ops.usage", "extract_model_name"),
    "extract_token_usage": ("twinr.ops.usage", "extract_token_usage"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
