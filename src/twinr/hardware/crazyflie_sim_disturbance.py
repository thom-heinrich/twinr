"""Typed physical disturbance schedules for CrazySim MuJoCo runs.

Twinr uses this module to describe real force/torque disturbances injected into
the CrazySim MuJoCo plant. The goal is not a second simulator. The goal is one
explicit, serializable schedule that can be launched through the existing
CrazySim lane and evaluated through the existing hover/replay acceptance tools.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal, Sequence


_VECTOR_LENGTH = 3
_DEFAULT_ZERO_VECTOR = (0.0, 0.0, 0.0)
_SUPPORTED_ACTIVATION_MODES = frozenset({"immediate", "after_airborne", "after_host_phase"})


Vector3 = tuple[float, float, float]
DisturbanceActivationMode = Literal["immediate", "after_airborne", "after_host_phase"]


def _coerce_float(value: object, *, field_name: str) -> float:
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"{field_name} must be numeric, got {type(value).__name__}")
    return float(value)


def _coerce_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"{field_name} must be an integer-like value, got {type(value).__name__}")
    return int(value)


def _coerce_vector3(
    value: Sequence[float | int] | None,
    *,
    field_name: str,
) -> Vector3:
    if value is None:
        return _DEFAULT_ZERO_VECTOR
    if len(value) != _VECTOR_LENGTH:
        raise ValueError(
            f"{field_name} must contain exactly {_VECTOR_LENGTH} numeric entries, "
            f"got {len(value)}"
        )
    try:
        return (
            float(value[0]),
            float(value[1]),
            float(value[2]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain numeric entries") from exc


@dataclass(frozen=True, slots=True)
class CrazySimDisturbancePulse:
    """Apply one bounded physical disturbance to one CrazySim agent."""

    name: str
    start_s: float
    duration_s: float
    target_agent: int = 0
    world_force_n: Vector3 = _DEFAULT_ZERO_VECTOR
    body_force_n: Vector3 = _DEFAULT_ZERO_VECTOR
    body_torque_nm: Vector3 = _DEFAULT_ZERO_VECTOR

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("disturbance pulse name must be non-empty")
        if int(self.target_agent) < 0:
            raise ValueError("disturbance target_agent must be >= 0")
        if float(self.start_s) < 0.0:
            raise ValueError("disturbance start_s must be >= 0")
        if float(self.duration_s) <= 0.0:
            raise ValueError("disturbance duration_s must be > 0")
        if not any(
            abs(component) > 0.0
            for component in (
                *self.world_force_n,
                *self.body_force_n,
                *self.body_torque_nm,
            )
        ):
            raise ValueError(
                "disturbance pulse must declare at least one non-zero world force, "
                "body force, or body torque"
            )

    @property
    def end_s(self) -> float:
        """Return the exclusive end time of this disturbance pulse."""

        return float(self.start_s) + float(self.duration_s)

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "CrazySimDisturbancePulse":
        """Build one validated pulse from JSON-compatible input."""

        name = str(payload.get("name") or "").strip()
        return cls(
            name=name,
            start_s=_coerce_float(payload.get("start_s", 0.0), field_name="start_s"),
            duration_s=_coerce_float(payload.get("duration_s", 0.0), field_name="duration_s"),
            target_agent=_coerce_int(payload.get("target_agent", 0), field_name="target_agent"),
            world_force_n=_coerce_vector3(
                payload.get("world_force_n"),  # type: ignore[arg-type]
                field_name="world_force_n",
            ),
            body_force_n=_coerce_vector3(
                payload.get("body_force_n"),  # type: ignore[arg-type]
                field_name="body_force_n",
            ),
            body_torque_nm=_coerce_vector3(
                payload.get("body_torque_nm"),  # type: ignore[arg-type]
                field_name="body_torque_nm",
            ),
        )

    def to_payload(self) -> dict[str, object]:
        """Return one JSON-compatible representation of this disturbance."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class CrazySimDisturbancePlan:
    """Describe one serializable MuJoCo disturbance schedule."""

    name: str
    activation_mode: DisturbanceActivationMode
    activation_height_m: float
    pulses: tuple[CrazySimDisturbancePulse, ...]
    activation_phase: str | None = None
    activation_status: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("disturbance plan name must be non-empty")
        if str(self.activation_mode) not in _SUPPORTED_ACTIVATION_MODES:
            raise ValueError(
                f"unsupported disturbance activation mode `{self.activation_mode}`; "
                f"choose one of: {', '.join(sorted(_SUPPORTED_ACTIVATION_MODES))}"
            )
        if self.activation_mode == "after_airborne" and float(self.activation_height_m) <= 0.0:
            raise ValueError(
                "after_airborne disturbance plans require activation_height_m > 0"
            )
        if self.activation_mode == "after_host_phase":
            if not str(self.activation_phase or "").strip():
                raise ValueError(
                    "after_host_phase disturbance plans require a non-empty activation_phase"
                )
            if not str(self.activation_status or "").strip():
                raise ValueError(
                    "after_host_phase disturbance plans require a non-empty activation_status"
                )
        if not self.pulses:
            raise ValueError("disturbance plan must contain at least one pulse")

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "CrazySimDisturbancePlan":
        """Build one validated disturbance plan from JSON-compatible input."""

        raw_pulses = payload.get("pulses")
        if not isinstance(raw_pulses, Sequence) or isinstance(raw_pulses, (str, bytes, bytearray)):
            raise ValueError("disturbance plan payload must contain a sequence field `pulses`")
        pulses = tuple(
            CrazySimDisturbancePulse.from_mapping(dict(item))
            for item in raw_pulses
        )
        return cls(
            name=str(payload.get("name") or "").strip(),
            activation_mode=str(payload.get("activation_mode") or "immediate"),  # type: ignore[arg-type]
            activation_height_m=_coerce_float(
                payload.get("activation_height_m", 0.08),
                field_name="activation_height_m",
            ),
            activation_phase=(
                None
                if payload.get("activation_phase") is None
                else str(payload.get("activation_phase"))
            ),
            activation_status=(
                None
                if payload.get("activation_status") is None
                else str(payload.get("activation_status"))
            ),
            description=None if payload.get("description") is None else str(payload.get("description")),
            pulses=pulses,
        )

    def to_payload(self) -> dict[str, object]:
        """Return one JSON-compatible representation of this disturbance plan."""

        payload = asdict(self)
        payload["pulses"] = tuple(pulse.to_payload() for pulse in self.pulses)
        return payload


def load_crazysim_disturbance_plan(path: Path) -> CrazySimDisturbancePlan:
    """Load one validated disturbance plan from a JSON file."""

    candidate = path.expanduser().resolve(strict=True)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("disturbance plan JSON must be an object")
    return CrazySimDisturbancePlan.from_mapping(payload)


def write_crazysim_disturbance_plan(
    path: Path,
    plan: CrazySimDisturbancePlan,
) -> Path:
    """Write one disturbance plan JSON file and return the resolved path."""

    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).absolute()
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(
        json.dumps(plan.to_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return candidate


@dataclass(frozen=True, slots=True)
class CrazySimDisturbanceRuntimeEvent:
    """Represent one runtime event emitted by the MuJoCo disturbance bridge."""

    kind: str
    plan_name: str
    agent_id: int
    sim_time_s: float
    z_m: float | None = None
    pulse_name: str | None = None
    elapsed_since_anchor_s: float | None = None
    host_phase: str | None = None
    host_status: str | None = None
    host_phase_elapsed_s: float | None = None
    world_force_n: Vector3 = _DEFAULT_ZERO_VECTOR
    body_force_n: Vector3 = _DEFAULT_ZERO_VECTOR
    body_torque_nm: Vector3 = _DEFAULT_ZERO_VECTOR

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "CrazySimDisturbanceRuntimeEvent":
        """Build one validated runtime event from JSON-compatible input."""

        return cls(
            kind=str(payload.get("kind") or "").strip(),
            plan_name=str(payload.get("plan_name") or "").strip(),
            agent_id=_coerce_int(payload.get("agent_id", 0), field_name="agent_id"),
            sim_time_s=_coerce_float(payload.get("sim_time_s", 0.0), field_name="sim_time_s"),
            z_m=(
                None
                if payload.get("z_m") is None
                else _coerce_float(payload.get("z_m"), field_name="z_m")
            ),
            pulse_name=None if payload.get("pulse_name") is None else str(payload.get("pulse_name")),
            elapsed_since_anchor_s=(
                None
                if payload.get("elapsed_since_anchor_s") is None
                else _coerce_float(
                    payload.get("elapsed_since_anchor_s"),
                    field_name="elapsed_since_anchor_s",
                )
            ),
            host_phase=None if payload.get("host_phase") is None else str(payload.get("host_phase")),
            host_status=None if payload.get("host_status") is None else str(payload.get("host_status")),
            host_phase_elapsed_s=(
                None
                if payload.get("host_phase_elapsed_s") is None
                else _coerce_float(
                    payload.get("host_phase_elapsed_s"),
                    field_name="host_phase_elapsed_s",
                )
            ),
            world_force_n=_coerce_vector3(
                payload.get("world_force_n"),  # type: ignore[arg-type]
                field_name="world_force_n",
            ),
            body_force_n=_coerce_vector3(
                payload.get("body_force_n"),  # type: ignore[arg-type]
                field_name="body_force_n",
            ),
            body_torque_nm=_coerce_vector3(
                payload.get("body_torque_nm"),  # type: ignore[arg-type]
                field_name="body_torque_nm",
            ),
        )

    def to_payload(self) -> dict[str, object]:
        """Return one JSON-compatible representation of this runtime event."""

        return asdict(self)


def load_crazysim_disturbance_runtime_events(
    path: Path,
) -> tuple[CrazySimDisturbanceRuntimeEvent, ...]:
    """Load the runtime event stream emitted by the MuJoCo disturbance bridge."""

    candidate = path.expanduser().resolve(strict=True)
    events: list[CrazySimDisturbanceRuntimeEvent] = []
    with candidate.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError("disturbance runtime event JSONL lines must be objects")
            events.append(CrazySimDisturbanceRuntimeEvent.from_mapping(payload))
    return tuple(events)
