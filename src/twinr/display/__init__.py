"""Expose Twinr's stable display package exports lazily.

Lazy resolution keeps internal helpers such as ``display.heartbeat`` import-safe
for ops modules that should not have to import the full rendering stack.
"""

from __future__ import annotations

from typing import Any

__all__ = ["TwinrStatusDisplayLoop", "WaveshareEPD4In2V2"]


def __getattr__(name: str) -> Any:
    """Resolve public display exports on demand."""

    if name == "TwinrStatusDisplayLoop":
        from twinr.display.service import TwinrStatusDisplayLoop

        return TwinrStatusDisplayLoop
    if name == "WaveshareEPD4In2V2":
        from twinr.display.waveshare_v2 import WaveshareEPD4In2V2

        return WaveshareEPD4In2V2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
