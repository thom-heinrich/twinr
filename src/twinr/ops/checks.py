"""Run configuration checks for Twinr's operational surfaces.

This module inspects API keys, audio devices, printer, camera, GPIO, PIR, and
runtime-state paths and returns normalized check records for ops tooling.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from shutil import which
import subprocess

from twinr.agent.base_agent.config import TwinrConfig


_VALID_STATUSES = frozenset({"ok", "warn", "fail"})
_ALLOWED_PIR_BIASES = frozenset({"pull-up", "pull-down", "disable", "disabled", "as-is", "none"})
_PRINTER_CHECK_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class ConfigCheck:
    """Represent one normalized configuration audit result."""

    key: str
    label: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        # AUDIT-FIX(#8): Normalize statuses at the data boundary so downstream summaries stay stable.
        object.__setattr__(self, "status", _normalize_status(self.status))
        # AUDIT-FIX(#11): Sanitize free-text fields to avoid leaking raw control characters into UI/logs.
        object.__setattr__(self, "key", _sanitize_identifier(self.key))
        object.__setattr__(self, "label", _sanitize_detail(self.label, default="Unknown check"))
        object.__setattr__(self, "detail", _sanitize_detail(self.detail, default="No additional detail."))

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def run_config_checks(config: TwinrConfig) -> tuple[ConfigCheck, ...]:
    """Run the standard Twinr ops configuration checks.

    Args:
        config: Twinr runtime configuration to inspect.

    Returns:
        A tuple of normalized ``ConfigCheck`` results ordered for UI display.
    """

    # AUDIT-FIX(#2): Isolate each checker so one malformed config field cannot abort the whole audit run.
    checks: tuple[tuple[str, str, Callable[[TwinrConfig], ConfigCheck]], ...] = (
        ("openai_key", "OpenAI key", _openai_key_check),
        ("audio_input", "Audio input", _audio_input_check),
        ("proactive_audio_input", "Proactive audio input", _proactive_audio_input_check),
        ("audio_output", "Audio output", _audio_output_check),
        ("printer_queue", "Printer queue", _printer_check),
        ("camera", "Camera", _camera_check),
        ("gpio", "GPIO buttons", _gpio_check),
        ("display_gpio", "Display GPIO", _display_gpio_check),
        ("pir", "PIR motion sensor", _pir_check),
        ("runtime_state", "Runtime snapshot", _runtime_state_check),
    )
    return tuple(_run_check_safely(key, label, check_fn, config) for key, label, check_fn in checks)


def check_summary(checks: tuple[ConfigCheck, ...] | list[ConfigCheck]) -> dict[str, int]:
    """Count configuration checks by normalized status."""

    summary = {"ok": 0, "warn": 0, "fail": 0}
    for check in checks:
        # AUDIT-FIX(#8): Unknown or malformed statuses are counted as failures instead of creating surprise keys.
        status = _normalize_status(getattr(check, "status", None))
        summary[status] += 1
    return summary


def _openai_key_check(config: TwinrConfig) -> ConfigCheck:
    # AUDIT-FIX(#9): Reject whitespace-only keys so the health check cannot falsely report API readiness.
    api_key = _clean_config_text(getattr(config, "openai_api_key", ""))
    if api_key:
        return ConfigCheck("openai_key", "OpenAI key", "ok", "API key is configured.")
    return ConfigCheck("openai_key", "OpenAI key", "fail", "OPENAI_API_KEY is missing.")


def _audio_input_check(config: TwinrConfig) -> ConfigCheck:
    device = _clean_config_text(getattr(config, "audio_input_device", ""))
    if not device:
        return ConfigCheck("audio_input", "Audio input", "fail", "No input device is configured.")
    if which("arecord") is None:
        return ConfigCheck(
            "audio_input",
            "Audio input",
            "warn",
            f"Input device is set to `{_display_value(device)}`, but `arecord` is not on PATH.",
        )
    return ConfigCheck(
        "audio_input",
        "Audio input",
        "ok",
        f"Input device `{_display_value(device)}` is configured and `arecord` is available.",
    )


def _audio_output_check(config: TwinrConfig) -> ConfigCheck:
    device = _clean_config_text(getattr(config, "audio_output_device", ""))
    if not device:
        return ConfigCheck("audio_output", "Audio output", "fail", "No output device is configured.")
    if which("aplay") is None:
        return ConfigCheck(
            "audio_output",
            "Audio output",
            "warn",
            f"Output device is set to `{_display_value(device)}`, but `aplay` is not on PATH.",
        )
    return ConfigCheck(
        "audio_output",
        "Audio output",
        "ok",
        f"Output device `{_display_value(device)}` is configured and `aplay` is available.",
    )


def _proactive_audio_input_check(config: TwinrConfig) -> ConfigCheck:
    proactive_audio_enabled = _coerce_bool(getattr(config, "proactive_audio_enabled", False))
    if not proactive_audio_enabled:
        return ConfigCheck(
            "proactive_audio_input",
            "Proactive audio input",
            "ok",
            "Proactive background-audio sampling is disabled.",
        )

    device = _clean_config_text(getattr(config, "proactive_audio_input_device", ""))
    if device:
        if which("arecord") is None:
            return ConfigCheck(
                "proactive_audio_input",
                "Proactive audio input",
                "warn",
                f"Proactive mic `{_display_value(device)}` is configured, but `arecord` is not on PATH.",
            )
        return ConfigCheck(
            "proactive_audio_input",
            "Proactive audio input",
            "ok",
            f"Proactive mic `{_display_value(device)}` is configured for background sampling.",
        )

    primary_device = _clean_config_text(getattr(config, "audio_input_device", ""))
    if not primary_device:
        return ConfigCheck(
            "proactive_audio_input",
            "Proactive audio input",
            "fail",
            "Proactive background audio is enabled, but no input device is available.",
        )
    if which("arecord") is None:
        return ConfigCheck(
            "proactive_audio_input",
            "Proactive audio input",
            "warn",
            f"Proactive audio will reuse `{_display_value(primary_device)}`, but `arecord` is not on PATH.",
        )
    return ConfigCheck(
        "proactive_audio_input",
        "Proactive audio input",
        "ok",
        f"Proactive audio reuses the primary input device `{_display_value(primary_device)}`.",
    )


def _printer_check(config: TwinrConfig) -> ConfigCheck:
    queue = _clean_config_text(getattr(config, "printer_queue", ""))
    if not queue:
        return ConfigCheck("printer_queue", "Printer queue", "fail", "TWINR_PRINTER_QUEUE is missing.")
    if which("lp") is None:
        return ConfigCheck(
            "printer_queue",
            "Printer queue",
            "warn",
            f"Queue `{_display_value(queue)}` is configured, but the `lp` command is not available.",
        )
    if which("lpstat") is None:
        return ConfigCheck(
            "printer_queue",
            "Printer queue",
            "warn",
            # AUDIT-FIX(#10): Missing `lpstat` means the queue cannot be validated; that is not an `ok` state.
            f"Queue `{_display_value(queue)}` is configured, but queue validation via `lpstat` is unavailable.",
        )
    try:
        result = subprocess.run(
            ["lpstat", "-p", queue],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            text=True,
            # AUDIT-FIX(#10): Use a less brittle timeout for cold-started or heavily loaded Raspberry Pi / CUPS setups.
            timeout=_PRINTER_CHECK_TIMEOUT_SECONDS,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return ConfigCheck(
            "printer_queue",
            "Printer queue",
            "warn",
            f"Queue `{_display_value(queue)}` is configured, but `lpstat` could not validate it: {exc}",
        )
    if result.returncode == 0:
        return ConfigCheck(
            "printer_queue",
            "Printer queue",
            "ok",
            f"Queue `{_display_value(queue)}` is configured and visible to CUPS.",
        )
    stderr = result.stderr.strip() or result.stdout.strip() or "Queue is not visible to CUPS."
    return ConfigCheck("printer_queue", "Printer queue", "warn", stderr)


def _camera_check(config: TwinrConfig) -> ConfigCheck:
    raw_device = _clean_config_text(getattr(config, "camera_device", ""))
    ffmpeg_path = _clean_config_text(getattr(config, "camera_ffmpeg_path", ""))
    # AUDIT-FIX(#3): Reject blank camera paths explicitly so `Path('') == '.'` cannot produce a false green check.
    if not raw_device:
        return ConfigCheck("camera", "Camera", "fail", "No camera device is configured.")

    device = Path(raw_device).expanduser()
    ffmpeg_ok = _binary_or_path_exists(ffmpeg_path)
    if not device.exists():
        return ConfigCheck(
            "camera",
            "Camera",
            "warn",
            f"Camera device `{_display_value(device)}` does not exist on this machine.",
        )
    # AUDIT-FIX(#3): Require an actual character device; a directory or regular file is not a valid V4L2 camera endpoint.
    if not device.is_char_device():
        return ConfigCheck(
            "camera",
            "Camera",
            "fail",
            f"Camera device `{_display_value(device)}` exists, but it is not a character device.",
        )
    if ffmpeg_ok:
        return ConfigCheck(
            "camera",
            "Camera",
            "ok",
            "Camera device is present and ffmpeg is available.",
        )
    return ConfigCheck(
        "camera",
        "Camera",
        "warn",
        f"Camera device `{_display_value(device)}` exists, but ffmpeg `{_display_value(ffmpeg_path)}` is not available.",
    )


def _gpio_check(config: TwinrConfig) -> ConfigCheck:
    raw_gpios = getattr(config, "button_gpios", {})
    if not isinstance(raw_gpios, Mapping):
        return ConfigCheck(
            "gpio",
            "GPIO buttons",
            "fail",
            "Button GPIO mapping must be a dictionary-like object.",
        )

    gpios = dict(raw_gpios)
    chip = _clean_config_text(getattr(config, "gpio_chip", ""))
    required_keys = ("green", "yellow")
    missing = [key for key in required_keys if key not in gpios]
    valid_required_values = {key: gpios[key] for key in required_keys if key in gpios and _is_valid_gpio_number(gpios[key])}
    invalid = [key for key in required_keys if key in gpios and key not in valid_required_values]

    if not chip:
        # AUDIT-FIX(#4): GPIO diagnostics must fail fast when the chip path/name is absent because button reads will not work.
        return ConfigCheck("gpio", "GPIO buttons", "fail", "No GPIO chip is configured.")

    if missing:
        if gpios:
            return ConfigCheck(
                "gpio",
                "GPIO buttons",
                "warn",
                f"Only part of the button mapping is configured: {_format_mapping(gpios)}.",
            )
        return ConfigCheck("gpio", "GPIO buttons", "fail", "No green/yellow button GPIOs are configured.")

    if invalid:
        return ConfigCheck(
            "gpio",
            "GPIO buttons",
            "fail",
            f"Invalid GPIO number configured for: {', '.join(sorted(invalid))}.",
        )

    if valid_required_values["green"] == valid_required_values["yellow"]:
        # AUDIT-FIX(#4): TALK and PRINT must not share a line or the physical buttons become ambiguous.
        return ConfigCheck("gpio", "GPIO buttons", "fail", "Green and yellow buttons cannot share the same GPIO.")

    return ConfigCheck(
        "gpio",
        "GPIO buttons",
        "ok",
        f"Green={valid_required_values['green']} and yellow={valid_required_values['yellow']} are configured on `{_display_value(chip)}`.",
    )


def _display_gpio_check(config: TwinrConfig) -> ConfigCheck:
    display_pins = _raw_display_gpio_map(config)
    missing = [name for name, value in display_pins.items() if value is None]
    if missing:
        # AUDIT-FIX(#5): Explicitly validate required display lines instead of assuming `display_gpio_conflicts()` covers them.
        return ConfigCheck(
            "display_gpio",
            "Display GPIO",
            "fail",
            f"Missing display GPIO configuration for: {', '.join(missing)}.",
        )

    invalid = [name for name, value in display_pins.items() if not _is_valid_gpio_number(value)]
    if invalid:
        return ConfigCheck(
            "display_gpio",
            "Display GPIO",
            "fail",
            f"Invalid display GPIO number configured for: {', '.join(invalid)}.",
        )

    valid_display_pins = {name: int(value) for name, value in display_pins.items()}
    duplicate_groups = _duplicate_gpio_groups(valid_display_pins)
    if duplicate_groups:
        return ConfigCheck(
            "display_gpio",
            "Display GPIO",
            "fail",
            "; ".join(duplicate_groups),
        )

    button_conflicts = _gpio_map_collisions(valid_display_pins, _valid_button_gpio_map(config), other_label="button")
    if button_conflicts:
        return ConfigCheck(
            "display_gpio",
            "Display GPIO",
            "fail",
            "; ".join(button_conflicts),
        )

    pir_gpio = getattr(config, "pir_motion_gpio", None)
    if _is_valid_gpio_number(pir_gpio) and int(pir_gpio) in set(valid_display_pins.values()):
        return ConfigCheck(
            "display_gpio",
            "Display GPIO",
            "fail",
            f"Display GPIO collides with PIR GPIO {pir_gpio}.",
        )

    raw_conflicts = config.display_gpio_conflicts()
    if isinstance(raw_conflicts, str):
        conflicts = [raw_conflicts]
    elif isinstance(raw_conflicts, Iterable) and not isinstance(raw_conflicts, Mapping):
        conflicts = [str(conflict) for conflict in raw_conflicts if _clean_config_text(conflict)]
    else:
        conflicts = []
    if conflicts:
        return ConfigCheck(
            "display_gpio",
            "Display GPIO",
            "fail",
            "; ".join(conflicts),
        )
    return ConfigCheck(
        "display_gpio",
        "Display GPIO",
        "ok",
        (
            f"Display pins are configured as "
            f"CS={valid_display_pins['CS']}, DC={valid_display_pins['DC']}, "
            f"RESET={valid_display_pins['RESET']}, BUSY={valid_display_pins['BUSY']}."
        ),
    )


def _pir_check(config: TwinrConfig) -> ConfigCheck:
    pir_gpio = getattr(config, "pir_motion_gpio", None)
    if pir_gpio is None:
        return ConfigCheck(
            "pir",
            "PIR motion sensor",
            "warn",
            "TWINR_PIR_MOTION_GPIO is not configured.",
        )

    if not _is_valid_gpio_number(pir_gpio):
        # AUDIT-FIX(#6): Reject non-integer or negative GPIO values before wiring collisions are evaluated.
        return ConfigCheck("pir", "PIR motion sensor", "fail", "PIR GPIO must be a non-negative integer.")

    pir_gpio = int(pir_gpio)
    button_lines = set(_valid_button_gpio_map(config).values())
    if pir_gpio in button_lines:
        return ConfigCheck(
            "pir",
            "PIR motion sensor",
            "fail",
            f"PIR GPIO {pir_gpio} collides with a button input.",
        )

    display_lines = set(_valid_display_gpio_map(config).values())
    if pir_gpio in display_lines:
        return ConfigCheck(
            "pir",
            "PIR motion sensor",
            "fail",
            f"PIR GPIO {pir_gpio} collides with a display GPIO.",
        )

    raw_bias = _clean_config_text(getattr(config, "pir_bias", ""))
    normalized_bias = raw_bias.lower()
    if normalized_bias not in _ALLOWED_PIR_BIASES:
        # AUDIT-FIX(#6): Normalize and validate bias values so harmless casing/whitespace differences do not break checks.
        return ConfigCheck(
            "pir",
            "PIR motion sensor",
            "fail",
            f"PIR bias `{_display_value(raw_bias)}` is unsupported.",
        )
    active_high = _coerce_bool(getattr(config, "pir_active_high", False))
    return ConfigCheck(
        "pir",
        "PIR motion sensor",
        "ok",
        (
            f"PIR is configured on GPIO {pir_gpio} "
            f"(active_high={str(active_high).lower()}, bias={normalized_bias})."
        ),
    )


def _runtime_state_check(config: TwinrConfig) -> ConfigCheck:
    raw_path = _clean_config_text(getattr(config, "runtime_state_path", ""))
    # AUDIT-FIX(#1): Fail on blank paths so `Path('') == '.'` cannot silently point the runtime state at the current directory.
    if not raw_path:
        return ConfigCheck("runtime_state", "Runtime snapshot", "fail", "TWINR_RUNTIME_STATE_PATH is missing.")

    path = Path(raw_path).expanduser()
    parent = path.parent

    # AUDIT-FIX(#1): Reject symlink components to reduce TOCTOU and path-redirection risk for file-backed state.
    if _path_has_symlink_component(path):
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "fail",
            f"Snapshot path `{_display_value(path)}` must not use symlink components.",
        )

    if path.exists() and path.is_dir():
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "fail",
            f"Snapshot path `{_display_value(path)}` points to a directory, not a file.",
        )

    if path.exists() and not path.is_file():
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "fail",
            f"Snapshot path `{_display_value(path)}` is not a regular file.",
        )

    if parent.exists() and not parent.is_dir():
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "fail",
            f"Snapshot parent `{_display_value(parent)}` is not a directory.",
        )

    if not parent.exists():
        nearest_existing = _nearest_existing_ancestor(parent)
        if not nearest_existing.exists() or not nearest_existing.is_dir():
            return ConfigCheck(
                "runtime_state",
                "Runtime snapshot",
                "fail",
                f"Snapshot ancestor `{_display_value(nearest_existing)}` is not a writable directory.",
            )
        if not os.access(nearest_existing, os.W_OK | os.X_OK):
            return ConfigCheck(
                "runtime_state",
                "Runtime snapshot",
                "fail",
                f"Snapshot directory `{_display_value(parent)}` cannot be created from `{_display_value(nearest_existing)}`.",
            )
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "warn",
            f"Snapshot directory `{_display_value(parent)}` does not exist yet. It will be created on first write.",
        )

    if not os.access(parent, os.W_OK | os.X_OK):
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "fail",
            f"Snapshot directory `{_display_value(parent)}` is not writable.",
        )

    if path.exists() and not os.access(path, os.W_OK):
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "fail",
            f"Snapshot file `{_display_value(path)}` is not writable.",
        )

    return ConfigCheck(
        "runtime_state",
        "Runtime snapshot",
        "ok",
        f"Snapshot path is `{_display_value(path)}`.",
    )


def _binary_or_path_exists(value: str) -> bool:
    candidate = _clean_config_text(value)
    # AUDIT-FIX(#7): Empty strings, directories, and non-executable files must not count as available binaries.
    if not candidate:
        return False
    if which(candidate):
        return True
    path = Path(candidate).expanduser()
    return path.exists() and path.is_file() and os.access(path, os.X_OK)


def _run_check_safely(
    key: str,
    label: str,
    check_fn: Callable[[TwinrConfig], ConfigCheck],
    config: TwinrConfig,
) -> ConfigCheck:
    try:
        result = check_fn(config)
    except Exception as exc:
        return ConfigCheck(
            key,
            label,
            "fail",
            f"{label} check could not be completed because the configuration could not be read safely ({exc.__class__.__name__}).",
        )
    if not isinstance(result, ConfigCheck):
        return ConfigCheck(
            key,
            label,
            "fail",
            f"{label} check returned an invalid result type `{type(result).__name__}`.",
        )
    return result


def _normalize_status(value: object) -> str:
    text = _clean_config_text(value).lower()
    if text in _VALID_STATUSES:
        return text
    return "fail"


def _clean_config_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _sanitize_identifier(value: object) -> str:
    text = _clean_config_text(value)
    if not text:
        return "unknown"
    return _sanitize_detail(text, default="unknown").replace(" ", "_")


def _sanitize_detail(value: object, *, default: str) -> str:
    if value is None:
        return default
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = "".join(character if character.isprintable() else " " for character in text)
    text = " ".join(text.split())
    if not text:
        return default
    if len(text) > 400:
        return f"{text[:397]}..."
    return text


def _display_value(value: object, unset: str = "<unset>") -> str:
    text = _clean_config_text(value)
    if not text:
        return unset
    return _sanitize_detail(text, default=unset)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = _clean_config_text(value).lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return bool(value)


def _is_valid_gpio_number(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _raw_display_gpio_map(config: TwinrConfig) -> dict[str, object]:
    return {
        "CS": getattr(config, "display_cs_gpio", None),
        "DC": getattr(config, "display_dc_gpio", None),
        "RESET": getattr(config, "display_reset_gpio", None),
        "BUSY": getattr(config, "display_busy_gpio", None),
    }


def _valid_display_gpio_map(config: TwinrConfig) -> dict[str, int]:
    return {
        name: int(value)
        for name, value in _raw_display_gpio_map(config).items()
        if _is_valid_gpio_number(value)
    }


def _valid_button_gpio_map(config: TwinrConfig) -> dict[str, int]:
    raw_gpios = getattr(config, "button_gpios", {})
    if not isinstance(raw_gpios, Mapping):
        return {}
    return {
        str(key): int(value)
        for key, value in raw_gpios.items()
        if _is_valid_gpio_number(value)
    }


def _duplicate_gpio_groups(gpios: Mapping[str, int]) -> list[str]:
    by_gpio: dict[int, list[str]] = {}
    for name, gpio in gpios.items():
        by_gpio.setdefault(gpio, []).append(name)
    return [
        f"{'/'.join(sorted(names))} share GPIO {gpio}"
        for gpio, names in sorted(by_gpio.items())
        if len(names) > 1
    ]


def _gpio_map_collisions(
    gpios: Mapping[str, int],
    other_gpios: Mapping[str, int],
    *,
    other_label: str,
) -> list[str]:
    collisions: list[str] = []
    for name, gpio in gpios.items():
        for other_name, other_gpio in other_gpios.items():
            if gpio == other_gpio:
                collisions.append(f"{name} GPIO {gpio} collides with {other_label} `{_display_value(other_name)}`")
    return collisions


def _format_mapping(mapping: Mapping[object, object]) -> str:
    return ", ".join(
        f"{_display_value(key)}={_display_value(value)}"
        for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))
    )


def _nearest_existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _path_has_symlink_component(path: Path) -> bool:
    for candidate in (path, *path.parents):
        if candidate.exists() and candidate.is_symlink():
            return True
    return False
