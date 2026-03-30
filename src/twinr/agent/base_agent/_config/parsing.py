"""Parse primitive Twinr config values from environment-style inputs.

Purpose and boundaries:
- Convert dotenv/process-env strings into typed primitive values.
- Keep helper parsing deterministic and side-effect free.
- Avoid importing the heavyweight config dataclass to prevent cycles.
"""

from __future__ import annotations

from pathlib import Path


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read simple ``KEY=VALUE`` pairs from a dotenv-style file."""

    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _parse_bool(value: str | None, default: bool) -> bool:
    """Parse a Twinr boolean env value with a fallback default."""

    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def _parse_optional_bool(value: str | None) -> bool | None:
    """Parse an optional boolean env value or return ``None``."""

    if value is None or not value.strip():
        return None
    return _parse_bool(value, False)


def _parse_optional_int(value: str | None) -> int | None:
    """Parse an optional integer env value or return ``None``."""

    if value is None or not value.strip():
        return None
    return int(value)


def _parse_optional_float(value: str | float | int | None) -> float | None:
    """Parse an optional float env value or return ``None``."""

    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        return float(normalized)
    return float(value)


def _parse_optional_text(value: str | None) -> str | None:
    """Parse an optional text env value or return ``None``."""

    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _parse_optional_url(
    value: str | None, *, strip_trailing_slash: bool = False
) -> str | None:
    """Parse one optional URL-like value and normalize surrounding whitespace."""

    normalized = _parse_optional_text(value)
    if normalized is None:
        return None
    if strip_trailing_slash:
        normalized = normalized.rstrip("/")
    return normalized or None


def _parse_camera_host_mode(value: str | None, default: str = "onboard") -> str:
    """Parse the high-level camera host mode used for open-source wiring defaults."""

    normalized = (
        str(value or default or "onboard").strip().lower().replace("-", "_")
        or "onboard"
    )
    if normalized in {"main_pi", "local"}:
        normalized = "onboard"
    if normalized in {"helper_pi", "peer_pi", "remote"}:
        normalized = "second_pi"
    if normalized not in {"onboard", "second_pi"}:
        raise ValueError(f"Unsupported camera host mode: {value}")
    return normalized


def _derive_camera_host_mode(
    *,
    raw_value: str | None,
    camera_second_pi_base_url: str | None,
    proactive_remote_camera_base_url: str | None,
    camera_proxy_snapshot_url: str | None,
) -> str:
    """Resolve the effective high-level camera topology from explicit or legacy envs."""

    if raw_value is not None and raw_value.strip():
        return _parse_camera_host_mode(raw_value, default="onboard")
    if (
        camera_second_pi_base_url is not None
        or proactive_remote_camera_base_url is not None
        or camera_proxy_snapshot_url is not None
    ):
        return "second_pi"
    return "onboard"


def _derive_snapshot_proxy_url(base_url: str | None) -> str | None:
    """Derive the peer snapshot endpoint from the helper AI-camera base URL."""

    normalized = _parse_optional_url(base_url, strip_trailing_slash=True)
    if normalized is None:
        return None
    return f"{normalized}/snapshot.png"


def _uses_aideck_camera_device(value: str | None) -> bool:
    """Return whether the configured still camera uses the AI-Deck stream URI."""

    normalized = str(value or "").strip().lower()
    return normalized.startswith("aideck://")


def _derive_proactive_vision_provider(
    raw_value: str | None,
    *,
    camera_host_mode: str,
    proactive_remote_camera_base_url: str | None,
    camera_device: str | None,
) -> str:
    """Resolve the effective proactive camera provider from friendly topology config."""

    explicit_value = _parse_optional_text(raw_value)
    if explicit_value is not None:
        return explicit_value.strip().lower()
    del camera_host_mode, proactive_remote_camera_base_url
    if _uses_aideck_camera_device(camera_device):
        return "aideck_openai"
    return "local_first"


def _normalize_model_setting(value: object, *, fallback: str) -> str:
    """Return one non-blank model identifier, or the provided fallback."""

    normalized = str(value or "").strip()
    return normalized or fallback


def _parse_float(
    value: str | float | int | None,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Parse a float env value and clamp it to optional bounds."""

    if value is None:
        parsed = default
    elif isinstance(value, str):
        if not value.strip():
            parsed = default
        else:
            parsed = float(value)
    else:
        parsed = float(value)
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _parse_clamped_float(
    value: str | None,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Parse a float env value through the shared clamp-aware helper."""

    return _parse_float(value, default, minimum=minimum, maximum=maximum)


def _parse_csv_ints(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    """Parse a comma-separated integer list or return the default tuple."""

    if value is None or not value.strip():
        return default
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_csv_strings(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse a comma-separated string list or return the default tuple."""

    if value is None or not value.strip():
        return default
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    return parsed or default


def _parse_csv_mapping(
    value: str | None,
    default: tuple[tuple[str, str], ...] = (),
) -> tuple[tuple[str, str], ...]:
    """Parse one comma-separated ``key=value`` mapping into a normalized tuple."""

    if value is None or not value.strip():
        return default
    parsed: list[tuple[str, str]] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Expected KEY=VALUE entry, got {part!r}.")
        key, mapped_value = (piece.strip() for piece in part.split("=", 1))
        if not key or not mapped_value:
            raise ValueError(f"Expected non-empty KEY=VALUE entry, got {part!r}.")
        parsed.append((key, mapped_value))
    return tuple(parsed) or default


def _parse_local_semantic_router_mode(value: str | None, default: str = "off") -> str:
    """Parse the local semantic-router mode with validation."""

    normalized = str(value or default or "off").strip().lower() or "off"
    if normalized not in {"off", "shadow", "gated"}:
        raise ValueError(f"Unsupported local semantic router mode: {value}")
    return normalized


def _parse_attention_servo_driver(value: str | None, default: str = "auto") -> str:
    """Parse the configured attention-servo driver strategy with validation."""

    normalized = str(value or default or "auto").strip().lower() or "auto"
    if normalized == "maestro":
        normalized = "pololu_maestro"
    if normalized in {"peer_maestro", "peer_pololu"}:
        normalized = "peer_pololu_maestro"
    if normalized not in {
        "auto",
        "twinr_kernel",
        "sysfs_pwm",
        "pigpio",
        "lgpio_pwm",
        "lgpio",
        "pololu_maestro",
        "peer_pololu_maestro",
    }:
        raise ValueError(f"Unsupported attention-servo driver: {value}")
    return normalized


def _parse_attention_servo_control_mode(
    value: str | None, default: str = "position"
) -> str:
    """Parse the configured attention-servo control model with validation."""

    normalized = (
        str(value or default or "position").strip().lower().replace("-", "_")
        or "position"
    )
    if normalized in {"continuous", "continuous_servo"}:
        normalized = "continuous_rotation"
    if normalized not in {"position", "continuous_rotation"}:
        raise ValueError(f"Unsupported attention-servo control mode: {value}")
    return normalized


def _default_display_poll_interval_s(display_driver: str | None) -> float:
    """Return one backend-aware default display poll interval.

    HDMI surfaces need a much faster cue/render cadence than e-paper backends,
    otherwise face-follow and emoji acknowledgements feel delayed even when the
    upstream sensor path is already fast.
    """

    normalized = (display_driver or "").strip().lower()
    if normalized in {"hdmi_fbdev", "hdmi_wayland"}:
        return 0.12
    return 0.5
