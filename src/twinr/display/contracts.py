"""Shared contracts for Twinr display backends.

The status loop only needs a narrow adapter surface. Keeping that contract in
one place lets Twinr swap display transports cleanly without growing more
hardware conditionals inside the orchestration loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from twinr.display.face_cues import DisplayFaceCue


DisplayStateFields = tuple[tuple[str, str], ...]
DisplayLogSections = tuple[tuple[str, tuple[str, ...]], ...]


class TwinrDisplayAdapter(Protocol):
    """Describe the minimal behavior required by the display loop."""

    emit: Callable[[str], None] | None
    _last_rendered_status: str | None

    def show_test_pattern(self) -> None: ...

    def supports_idle_waiting_animation(self) -> bool: ...

    def show_status(
        self,
        status: str,
        *,
        headline: str | None = None,
        details: tuple[str, ...] = (),
        state_fields: DisplayStateFields = (),
        log_sections: DisplayLogSections = (),
        animation_frame: int = 0,
        face_cue: DisplayFaceCue | None = None,
    ) -> None: ...

    def close(self) -> None: ...
