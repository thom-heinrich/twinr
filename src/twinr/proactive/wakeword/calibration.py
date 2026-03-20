"""Manage wakeword calibration profiles and config application.

This module stores per-device wakeword tuning overrides, validates persisted
profile data, and applies the resulting overrides onto ``TwinrConfig`` without
changing unrelated wakeword settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import json

from twinr.agent.base_agent.config import TwinrConfig


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_optional_text_tuple(name: str, value: object | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, (list, tuple)):
        parts = tuple(str(part).strip() for part in value if str(part).strip())
    else:
        raise ValueError(f"{name} must be a list of strings or a comma-separated string.")
    return parts or None


def _normalize_probability(name: str, value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a float in [0.0, 1.0].")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float in [0.0, 1.0].") from exc
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be a float in [0.0, 1.0].")
    return normalized


def _normalize_positive_int(name: str, value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= 1.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= 1.") from exc
    if normalized < 1:
        raise ValueError(f"{name} must be an integer >= 1.")
    return normalized


def _normalize_backend(value: object | None, *, allow_disabled: bool) -> str | None:
    normalized = (_normalize_optional_text(value) or "").lower()
    if not normalized:
        return None
    allowed = {"openwakeword", "stt"}
    if allow_disabled:
        allowed.add("disabled")
    if normalized not in allowed:
        raise ValueError(f"Unsupported wakeword backend: {normalized}")
    return normalized


def _normalize_verifier_mode(value: object | None) -> str | None:
    normalized = (_normalize_optional_text(value) or "").lower()
    if not normalized:
        return None
    if normalized not in {"disabled", "ambiguity_only", "always"}:
        raise ValueError(f"Unsupported wakeword verifier mode: {normalized}")
    return normalized


def _resolve_profile_path(config: TwinrConfig, configured_path: str) -> Path:
    project_root = Path(config.project_root).expanduser().resolve(strict=False)
    raw_path = Path(configured_path).expanduser()
    candidate = raw_path if raw_path.is_absolute() else (project_root / raw_path)
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(project_root):
        raise ValueError("wakeword calibration paths must stay inside project_root.")
    return resolved


@dataclass(frozen=True, slots=True)
class WakewordCalibrationProfile:
    """Describe one persisted wakeword calibration profile.

    Optional fields override the corresponding wakeword settings on top of the
    base ``TwinrConfig``. Validation normalizes backend names,
    probability-like values, and timestamps at construction time.
    """

    primary_backend: str | None = None
    fallback_backend: str | None = None
    verifier_mode: str | None = None
    verifier_margin: float | None = None
    stt_phrases: tuple[str, ...] | None = None
    threshold: float | None = None
    vad_threshold: float | None = None
    patience_frames: int | None = None
    activation_samples: int | None = None
    deactivation_threshold: float | None = None
    notes: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "primary_backend",
            _normalize_backend(self.primary_backend, allow_disabled=False),
        )
        object.__setattr__(
            self,
            "fallback_backend",
            _normalize_backend(self.fallback_backend, allow_disabled=True),
        )
        object.__setattr__(self, "verifier_mode", _normalize_verifier_mode(self.verifier_mode))
        object.__setattr__(
            self,
            "verifier_margin",
            _normalize_probability("verifier_margin", self.verifier_margin),
        )
        object.__setattr__(
            self,
            "stt_phrases",
            _normalize_optional_text_tuple("stt_phrases", self.stt_phrases),
        )
        object.__setattr__(self, "threshold", _normalize_probability("threshold", self.threshold))
        object.__setattr__(
            self,
            "vad_threshold",
            _normalize_probability("vad_threshold", self.vad_threshold),
        )
        object.__setattr__(
            self,
            "deactivation_threshold",
            _normalize_probability("deactivation_threshold", self.deactivation_threshold),
        )
        object.__setattr__(
            self,
            "patience_frames",
            _normalize_positive_int("patience_frames", self.patience_frames),
        )
        object.__setattr__(
            self,
            "activation_samples",
            _normalize_positive_int("activation_samples", self.activation_samples),
        )
        object.__setattr__(self, "notes", _normalize_optional_text(self.notes))
        updated_at = _normalize_optional_text(self.updated_at)
        object.__setattr__(self, "updated_at", updated_at or _utc_now_iso_z())

    @classmethod
    def from_payload(cls, payload: dict[str, object] | None) -> "WakewordCalibrationProfile":
        """Build a validated profile from JSON-compatible payload data."""

        data = dict(payload or {})
        return cls(
            primary_backend=data.get("primary_backend"),
            fallback_backend=data.get("fallback_backend"),
            verifier_mode=data.get("verifier_mode"),
            verifier_margin=data.get("verifier_margin"),
            stt_phrases=data.get("stt_phrases"),
            threshold=data.get("threshold"),
            vad_threshold=data.get("vad_threshold"),
            patience_frames=data.get("patience_frames"),
            activation_samples=data.get("activation_samples"),
            deactivation_threshold=data.get("deactivation_threshold"),
            notes=data.get("notes"),
            updated_at=data.get("updated_at"),
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize the profile into a JSON-compatible mapping."""

        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


class WakewordCalibrationStore:
    """Persist wakeword calibration profiles on disk."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "WakewordCalibrationStore":
        """Build the primary calibration store from ``TwinrConfig``."""

        return cls(_resolve_profile_path(config, config.wakeword_calibration_profile_path))

    @classmethod
    def recommended_from_config(cls, config: TwinrConfig) -> "WakewordCalibrationStore":
        """Build the recommendation store from ``TwinrConfig``."""

        return cls(_resolve_profile_path(config, config.wakeword_calibration_recommended_path))

    def load(self) -> WakewordCalibrationProfile | None:
        """Load and validate one calibration profile from disk if present."""

        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("wakeword calibration profile must be a JSON object.")
        return WakewordCalibrationProfile.from_payload(payload)

    def save(self, profile: WakewordCalibrationProfile) -> WakewordCalibrationProfile:
        """Persist one calibration profile and refresh its update timestamp."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        updated = replace(profile, updated_at=_utc_now_iso_z())
        self.path.write_text(
            json.dumps(updated.to_payload(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return updated


def apply_wakeword_calibration(
    config: TwinrConfig,
    profile: WakewordCalibrationProfile | None,
) -> TwinrConfig:
    """Apply a calibration profile onto one base ``TwinrConfig``.

    Args:
        config: Base Twinr configuration to start from.
        profile: Optional wakeword calibration overrides. When ``None``, the
            original config is returned unchanged.

    Returns:
        A config copy whose wakeword settings reflect the supplied profile
        while preserving unrelated configuration fields.
    """

    if profile is None:
        return config
    primary_backend = profile.primary_backend or config.wakeword_primary_backend or config.wakeword_backend
    fallback_backend = profile.fallback_backend or config.wakeword_fallback_backend
    verifier_mode = profile.verifier_mode or config.wakeword_verifier_mode
    return replace(
        config,
        wakeword_backend=primary_backend,
        wakeword_primary_backend=primary_backend,
        wakeword_fallback_backend=fallback_backend,
        wakeword_verifier_mode=verifier_mode,
        wakeword_verifier_margin=(
            config.wakeword_verifier_margin
            if profile.verifier_margin is None
            else profile.verifier_margin
        ),
        wakeword_stt_phrases=(
            config.wakeword_stt_phrases
            if profile.stt_phrases is None
            else profile.stt_phrases
        ),
        wakeword_openwakeword_threshold=(
            config.wakeword_openwakeword_threshold
            if profile.threshold is None
            else profile.threshold
        ),
        wakeword_openwakeword_vad_threshold=(
            config.wakeword_openwakeword_vad_threshold
            if profile.vad_threshold is None
            else profile.vad_threshold
        ),
        wakeword_openwakeword_patience_frames=(
            config.wakeword_openwakeword_patience_frames
            if profile.patience_frames is None
            else profile.patience_frames
        ),
        wakeword_openwakeword_activation_samples=(
            config.wakeword_openwakeword_activation_samples
            if profile.activation_samples is None
            else profile.activation_samples
        ),
        wakeword_openwakeword_deactivation_threshold=(
            config.wakeword_openwakeword_deactivation_threshold
            if profile.deactivation_threshold is None
            else profile.deactivation_threshold
        ),
    )


__all__ = [
    "WakewordCalibrationProfile",
    "WakewordCalibrationStore",
    "apply_wakeword_calibration",
]
