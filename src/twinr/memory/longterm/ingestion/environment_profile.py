"""Expose the stable environment-profile API over the decomposed implementation.

This module intentionally stays thin. The production implementation now lives
under ``twinr.memory.longterm.ingestion.environment_profile_impl`` so the
historic import path remains stable while the compiler is split by
responsibility.
"""

from __future__ import annotations

##REFACTOR: 2026-03-27##

from . import environment_profile_impl as _impl

LongTermEnvironmentProfileCompiler = _impl.LongTermEnvironmentProfileCompiler
SmartHomeEnvironmentBaseline = _impl.SmartHomeEnvironmentBaseline
SmartHomeEnvironmentBaselineStat = _impl.SmartHomeEnvironmentBaselineStat
SmartHomeEnvironmentDayProfile = _impl.SmartHomeEnvironmentDayProfile
SmartHomeEnvironmentDeviation = _impl.SmartHomeEnvironmentDeviation
SmartHomeEnvironmentDeviationMetric = _impl.SmartHomeEnvironmentDeviationMetric
SmartHomeEnvironmentEpoch = _impl.SmartHomeEnvironmentEpoch
SmartHomeEnvironmentEvent = _impl.SmartHomeEnvironmentEvent
SmartHomeEnvironmentNode = _impl.SmartHomeEnvironmentNode

_PUBLIC_EXPORTS = {
    "LongTermEnvironmentProfileCompiler",
    "SmartHomeEnvironmentBaseline",
    "SmartHomeEnvironmentBaselineStat",
    "SmartHomeEnvironmentDayProfile",
    "SmartHomeEnvironmentDeviation",
    "SmartHomeEnvironmentDeviationMetric",
    "SmartHomeEnvironmentEpoch",
    "SmartHomeEnvironmentEvent",
    "SmartHomeEnvironmentNode",
}
_COMPAT_EXPORTS = tuple(name for name in _impl.__all__ if name not in _PUBLIC_EXPORTS)

globals().update({name: getattr(_impl, name) for name in _COMPAT_EXPORTS})

__all__ = [
    "LongTermEnvironmentProfileCompiler",
    "SmartHomeEnvironmentBaseline",
    "SmartHomeEnvironmentBaselineStat",
    "SmartHomeEnvironmentDayProfile",
    "SmartHomeEnvironmentDeviation",
    "SmartHomeEnvironmentDeviationMetric",
    "SmartHomeEnvironmentEpoch",
    "SmartHomeEnvironmentEvent",
    "SmartHomeEnvironmentNode",
]

for _name in tuple(_PUBLIC_EXPORTS) + _COMPAT_EXPORTS:
    _exported = globals()[_name]
    try:
        _exported.__module__ = __name__
    except (AttributeError, TypeError):
        continue
