"""Expose Twinr's public e-paper display APIs.

Import display runtime entry points from this package root to keep callers
decoupled from the underlying module split.
"""

from twinr.display.service import TwinrStatusDisplayLoop
from twinr.display.waveshare_v2 import WaveshareEPD4In2V2

__all__ = ["TwinrStatusDisplayLoop", "WaveshareEPD4In2V2"]
