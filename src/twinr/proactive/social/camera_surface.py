"""Compatibility wrapper for the proactive camera stabilization surface.

##REFACTOR: 2026-03-27##
The runtime implementation now lives in ``camera_surface_impl``. Keep this
module path stable for callers, tests, and patch targets that still import
``twinr.proactive.social.camera_surface`` directly.
"""

from __future__ import annotations

from .camera_surface_impl import (
    ProactiveCameraSnapshot,
    ProactiveCameraSurfaceConfig,
    ProactiveCameraSurfaceImpl,
    ProactiveCameraSurfaceUpdate,
)


class ProactiveCameraSurface(ProactiveCameraSurfaceImpl):
    """Backward-compatible public surface class on the legacy module path."""


ProactiveCameraSnapshot.__module__ = __name__
ProactiveCameraSurfaceConfig.__module__ = __name__
ProactiveCameraSurfaceUpdate.__module__ = __name__

__all__ = [
    "ProactiveCameraSnapshot",
    "ProactiveCameraSurface",
    "ProactiveCameraSurfaceConfig",
    "ProactiveCameraSurfaceUpdate",
]
