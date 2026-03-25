"""Persist bounded continuous-servo runtime state for safe restarts.

Continuous-rotation servos have no absolute angle feedback, so Twinr needs one
small explicit state file when operators manually align a known reference pose.
This module keeps that persistence concern out of the higher follow controller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True, slots=True)
class AttentionServoRuntimeState:
    """Store one persisted virtual heading plus startup hold state."""

    heading_degrees: float = 0.0
    hold_until_armed: bool = False
    zero_reference_confirmed: bool = False
    updated_at: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "heading_degrees", _bounded_heading_degrees(self.heading_degrees))
        object.__setattr__(self, "hold_until_armed", bool(self.hold_until_armed))
        object.__setattr__(self, "zero_reference_confirmed", bool(self.zero_reference_confirmed))
        object.__setattr__(self, "updated_at", _optional_timestamp(self.updated_at))

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "AttentionServoRuntimeState":
        """Build one normalized runtime state from JSON-safe payload data."""

        return cls(
            heading_degrees=payload.get("heading_degrees", 0.0),
            hold_until_armed=bool(payload.get("hold_until_armed", False)),
            zero_reference_confirmed=bool(payload.get("zero_reference_confirmed", False)),
            updated_at=payload.get("updated_at"),
        )

    def to_payload(self) -> dict[str, object]:
        """Return one JSON-safe payload for persistence."""

        return asdict(self)


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
