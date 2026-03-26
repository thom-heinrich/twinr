"""Persist bounded continuous-servo runtime state for safe restarts.

Continuous-rotation servos have no absolute angle feedback, so Twinr needs one
small explicit state file when operators manually align a known reference pose.
This module keeps that persistence concern out of the higher follow controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile


def _bounded_heading_degrees(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        return 0.0
    try:
        checked_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(checked_value):
        return 0.0
    return max(-180.0, min(180.0, checked_value))


def _bounded_uncertainty_degrees(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        return 0.0
    try:
        checked_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(checked_value):
        return 0.0
    return max(0.0, min(180.0, checked_value))


def _optional_timestamp(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        checked_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(checked_value):
        return None
    return checked_value


def _bounded_pulse_width_us(value: object) -> int:
    if not isinstance(value, (int, float, str)):
        return 1500
    try:
        checked_value = int(value)
    except (TypeError, ValueError):
        return 1500
    return max(500, min(2500, checked_value))


def _bounded_duration_s(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        return 0.0
    try:
        checked_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(checked_value):
        return 0.0
    return max(0.0, min(600.0, checked_value))


@dataclass(frozen=True, slots=True)
class AttentionServoMovementSegment:
    """Persist one bounded continuous-servo movement slice away from zero."""

    pulse_width_us: int
    duration_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pulse_width_us", _bounded_pulse_width_us(self.pulse_width_us))
        object.__setattr__(self, "duration_s", _bounded_duration_s(self.duration_s))

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "AttentionServoMovementSegment":
        """Build one normalized movement slice from JSON-safe payload data."""

        return cls(
            pulse_width_us=_bounded_pulse_width_us(payload.get("pulse_width_us", 1500)),
            duration_s=_bounded_duration_s(payload.get("duration_s", 0.0)),
        )

    def to_payload(self) -> dict[str, object]:
        """Return one JSON-safe payload for persistence."""

        return {
            "pulse_width_us": int(self.pulse_width_us),
            "duration_s": round(float(self.duration_s), 6),
        }


def _normalized_movement_journal(value: object) -> tuple[AttentionServoMovementSegment, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    normalized_segments: list[AttentionServoMovementSegment] = []
    for raw_segment in list(value)[:256]:
        if isinstance(raw_segment, AttentionServoMovementSegment):
            normalized_segment = raw_segment
        elif isinstance(raw_segment, dict):
            normalized_segment = AttentionServoMovementSegment.from_payload(raw_segment)
        else:
            continue
        if normalized_segment.duration_s <= 0.0:
            continue
        normalized_segments.append(normalized_segment)
    return tuple(normalized_segments)


@dataclass(frozen=True, slots=True)
class AttentionServoRuntimeState:
    """Store one persisted virtual heading plus startup hold state."""

    heading_degrees: float = 0.0
    heading_uncertainty_degrees: float = 0.0
    movement_journal: tuple[AttentionServoMovementSegment, ...] = ()
    hold_until_armed: bool = False
    return_to_zero_requested: bool = False
    zero_reference_confirmed: bool = False
    updated_at: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "heading_degrees", _bounded_heading_degrees(self.heading_degrees))
        object.__setattr__(
            self,
            "heading_uncertainty_degrees",
            _bounded_uncertainty_degrees(self.heading_uncertainty_degrees),
        )
        object.__setattr__(self, "movement_journal", _normalized_movement_journal(self.movement_journal))
        object.__setattr__(self, "hold_until_armed", bool(self.hold_until_armed))
        object.__setattr__(self, "return_to_zero_requested", bool(self.return_to_zero_requested))
        object.__setattr__(self, "zero_reference_confirmed", bool(self.zero_reference_confirmed))
        object.__setattr__(self, "updated_at", _optional_timestamp(self.updated_at))

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "AttentionServoRuntimeState":
        """Build one normalized runtime state from JSON-safe payload data."""

        return cls(
            heading_degrees=_bounded_heading_degrees(payload.get("heading_degrees", 0.0)),
            heading_uncertainty_degrees=_bounded_uncertainty_degrees(
                payload.get("heading_uncertainty_degrees", 0.0)
            ),
            movement_journal=_normalized_movement_journal(payload.get("movement_journal", ())),
            hold_until_armed=bool(payload.get("hold_until_armed", False)),
            return_to_zero_requested=bool(payload.get("return_to_zero_requested", False)),
            zero_reference_confirmed=bool(payload.get("zero_reference_confirmed", False)),
            updated_at=_optional_timestamp(payload.get("updated_at")),
        )

    def to_payload(self) -> dict[str, object]:
        """Return one JSON-safe payload for persistence."""

        return {
            "heading_degrees": round(float(self.heading_degrees), 3),
            "heading_uncertainty_degrees": round(float(self.heading_uncertainty_degrees), 3),
            "movement_journal": [segment.to_payload() for segment in self.movement_journal],
            "hold_until_armed": bool(self.hold_until_armed),
            "return_to_zero_requested": bool(self.return_to_zero_requested),
            "zero_reference_confirmed": bool(self.zero_reference_confirmed),
            "updated_at": self.updated_at,
        }

    def hold_current_heading(self, *, updated_at: float | None = None) -> "AttentionServoRuntimeState":
        """Return one hold snapshot that preserves the current virtual heading."""

        return AttentionServoRuntimeState(
            heading_degrees=self.heading_degrees,
            heading_uncertainty_degrees=self.heading_uncertainty_degrees,
            movement_journal=self.movement_journal,
            hold_until_armed=True,
            return_to_zero_requested=False,
            zero_reference_confirmed=self.zero_reference_confirmed,
            updated_at=updated_at,
        )

    def adopt_current_as_zero(self, *, updated_at: float | None = None) -> "AttentionServoRuntimeState":
        """Return one hold snapshot that reanchors the current pose as zero."""

        return AttentionServoRuntimeState(
            heading_degrees=0.0,
            heading_uncertainty_degrees=0.0,
            movement_journal=(),
            hold_until_armed=True,
            return_to_zero_requested=False,
            zero_reference_confirmed=True,
            updated_at=updated_at,
        )

    def arm_follow(self, *, updated_at: float | None = None) -> "AttentionServoRuntimeState":
        """Return one runtime snapshot that resumes follow motion from this heading."""

        return AttentionServoRuntimeState(
            heading_degrees=self.heading_degrees,
            heading_uncertainty_degrees=self.heading_uncertainty_degrees,
            movement_journal=self.movement_journal,
            hold_until_armed=False,
            return_to_zero_requested=False,
            zero_reference_confirmed=self.zero_reference_confirmed,
            updated_at=updated_at,
        )

    def request_return_to_estimated_zero(self, *, updated_at: float | None = None) -> "AttentionServoRuntimeState":
        """Return one runtime snapshot that asks Twinr to drive back to estimated zero."""

        return AttentionServoRuntimeState(
            heading_degrees=self.heading_degrees,
            heading_uncertainty_degrees=self.heading_uncertainty_degrees,
            movement_journal=self.movement_journal,
            hold_until_armed=False,
            return_to_zero_requested=True,
            zero_reference_confirmed=self.zero_reference_confirmed,
            updated_at=updated_at,
        )


class AttentionServoStateStore:
    """Load and save one small JSON state file for the attention servo."""

    def __init__(self, path: str | Path) -> None:
        resolved_path = Path(path).expanduser().resolve(strict=False)
        if not str(resolved_path).strip():
            raise ValueError("attention servo state path must not be empty")
        self.path = resolved_path

    def load(self) -> AttentionServoRuntimeState | None:
        """Return the current state file content, or ``None`` when absent."""

        if not self.path.is_file():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"attention servo state file must contain a JSON object: {self.path}")
        return AttentionServoRuntimeState.from_payload(payload)

    def load_or_default(self) -> AttentionServoRuntimeState:
        """Return the saved runtime state, or one default snapshot when absent."""

        loaded_state = self.load()
        if loaded_state is None:
            return AttentionServoRuntimeState()
        return loaded_state

    def mtime_ns(self) -> int | None:
        """Return the current state-file mtime in nanoseconds when present."""

        try:
            return self.path.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def save(self, state: AttentionServoRuntimeState) -> None:
        """Persist one normalized state snapshot atomically."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state.to_payload(), sort_keys=True, indent=2) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self.path.parent),
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_file.write(payload)
            temp_path = Path(temp_file.name)
        temp_path.replace(self.path)
