from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import which
import subprocess

from twinr.agent.base_agent.config import TwinrConfig


@dataclass(frozen=True, slots=True)
class ConfigCheck:
    key: str
    label: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def run_config_checks(config: TwinrConfig) -> tuple[ConfigCheck, ...]:
    return (
        _openai_key_check(config),
        _audio_input_check(config),
        _proactive_audio_input_check(config),
        _audio_output_check(config),
        _printer_check(config),
        _camera_check(config),
        _gpio_check(config),
        _display_gpio_check(config),
        _pir_check(config),
        _runtime_state_check(config),
    )


def check_summary(checks: tuple[ConfigCheck, ...] | list[ConfigCheck]) -> dict[str, int]:
    summary = {"ok": 0, "warn": 0, "fail": 0}
    for check in checks:
        summary[check.status] = summary.get(check.status, 0) + 1
    return summary


def _openai_key_check(config: TwinrConfig) -> ConfigCheck:
    if config.openai_api_key:
        return ConfigCheck("openai_key", "OpenAI key", "ok", "API key is configured.")
    return ConfigCheck("openai_key", "OpenAI key", "fail", "OPENAI_API_KEY is missing.")


def _audio_input_check(config: TwinrConfig) -> ConfigCheck:
    if not config.audio_input_device.strip():
        return ConfigCheck("audio_input", "Audio input", "fail", "No input device is configured.")
    if which("arecord") is None:
        return ConfigCheck(
            "audio_input",
            "Audio input",
            "warn",
            f"Input device is set to `{config.audio_input_device}`, but `arecord` is not on PATH.",
        )
    return ConfigCheck(
        "audio_input",
        "Audio input",
        "ok",
        f"Input device `{config.audio_input_device}` is configured and `arecord` is available.",
    )


def _audio_output_check(config: TwinrConfig) -> ConfigCheck:
    if not config.audio_output_device.strip():
        return ConfigCheck("audio_output", "Audio output", "fail", "No output device is configured.")
    if which("aplay") is None:
        return ConfigCheck(
            "audio_output",
            "Audio output",
            "warn",
            f"Output device is set to `{config.audio_output_device}`, but `aplay` is not on PATH.",
        )
    return ConfigCheck(
        "audio_output",
        "Audio output",
        "ok",
        f"Output device `{config.audio_output_device}` is configured and `aplay` is available.",
    )


def _proactive_audio_input_check(config: TwinrConfig) -> ConfigCheck:
    if not config.proactive_audio_enabled:
        return ConfigCheck(
            "proactive_audio_input",
            "Proactive audio input",
            "ok",
            "Proactive background-audio sampling is disabled.",
        )

    device = (config.proactive_audio_input_device or "").strip()
    if device:
        if which("arecord") is None:
            return ConfigCheck(
                "proactive_audio_input",
                "Proactive audio input",
                "warn",
                f"Proactive mic `{device}` is configured, but `arecord` is not on PATH.",
            )
        return ConfigCheck(
            "proactive_audio_input",
            "Proactive audio input",
            "ok",
            f"Proactive mic `{device}` is configured for background sampling.",
        )

    primary_device = config.audio_input_device.strip()
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
            f"Proactive audio will reuse `{primary_device}`, but `arecord` is not on PATH.",
        )
    return ConfigCheck(
        "proactive_audio_input",
        "Proactive audio input",
        "ok",
        f"Proactive audio reuses the primary input device `{primary_device}`.",
    )


def _printer_check(config: TwinrConfig) -> ConfigCheck:
    queue = config.printer_queue.strip()
    if not queue:
        return ConfigCheck("printer_queue", "Printer queue", "fail", "TWINR_PRINTER_QUEUE is missing.")
    if which("lp") is None:
        return ConfigCheck(
            "printer_queue",
            "Printer queue",
            "warn",
            f"Queue `{queue}` is configured, but the `lp` command is not available.",
        )
    if which("lpstat") is None:
        return ConfigCheck(
            "printer_queue",
            "Printer queue",
            "ok",
            f"Queue `{queue}` is configured. Queue validation via `lpstat` is unavailable.",
        )
    try:
        result = subprocess.run(
            ["lpstat", "-p", queue],
            capture_output=True,
            check=False,
            text=True,
            timeout=2.0,
        )
    except Exception as exc:
        return ConfigCheck(
            "printer_queue",
            "Printer queue",
            "warn",
            f"Queue `{queue}` is configured, but `lpstat` could not validate it: {exc}",
        )
    if result.returncode == 0:
        return ConfigCheck(
            "printer_queue",
            "Printer queue",
            "ok",
            f"Queue `{queue}` is configured and visible to CUPS.",
        )
    stderr = result.stderr.strip() or result.stdout.strip() or "Queue is not visible to CUPS."
    return ConfigCheck("printer_queue", "Printer queue", "warn", stderr)


def _camera_check(config: TwinrConfig) -> ConfigCheck:
    device = Path(config.camera_device)
    ffmpeg_ok = _binary_or_path_exists(config.camera_ffmpeg_path)
    if device.exists() and ffmpeg_ok:
        return ConfigCheck(
            "camera",
            "Camera",
            "ok",
            f"Camera device `{device}` exists and ffmpeg is available.",
        )
    if not device.exists():
        return ConfigCheck(
            "camera",
            "Camera",
            "warn",
            f"Camera device `{device}` does not exist on this machine.",
        )
    return ConfigCheck(
        "camera",
        "Camera",
        "warn",
        f"Camera device `{device}` exists, but ffmpeg `{config.camera_ffmpeg_path}` is not available.",
    )


def _gpio_check(config: TwinrConfig) -> ConfigCheck:
    gpios = config.button_gpios
    if {"green", "yellow"}.issubset(gpios):
        return ConfigCheck(
            "gpio",
            "GPIO buttons",
            "ok",
            f"Green={gpios['green']} and yellow={gpios['yellow']} are configured on {config.gpio_chip}.",
        )
    if gpios:
        return ConfigCheck(
            "gpio",
            "GPIO buttons",
            "warn",
            f"Only part of the button mapping is configured: {gpios}.",
        )
    return ConfigCheck("gpio", "GPIO buttons", "fail", "No green/yellow button GPIOs are configured.")


def _display_gpio_check(config: TwinrConfig) -> ConfigCheck:
    conflicts = config.display_gpio_conflicts()
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
            f"CS={config.display_cs_gpio}, DC={config.display_dc_gpio}, "
            f"RESET={config.display_reset_gpio}, BUSY={config.display_busy_gpio}."
        ),
    )


def _pir_check(config: TwinrConfig) -> ConfigCheck:
    if config.pir_motion_gpio is None:
        return ConfigCheck(
            "pir",
            "PIR motion sensor",
            "warn",
            "TWINR_PIR_MOTION_GPIO is not configured.",
        )
    button_lines = set(config.button_gpios.values())
    if config.pir_motion_gpio in button_lines:
        return ConfigCheck(
            "pir",
            "PIR motion sensor",
            "fail",
            f"PIR GPIO {config.pir_motion_gpio} collides with a button input.",
        )
    if config.pir_bias not in {"pull-up", "pull-down", "disable", "disabled", "as-is", "none"}:
        return ConfigCheck(
            "pir",
            "PIR motion sensor",
            "fail",
            f"PIR bias `{config.pir_bias}` is unsupported.",
        )
    return ConfigCheck(
        "pir",
        "PIR motion sensor",
        "ok",
        (
            f"PIR is configured on GPIO {config.pir_motion_gpio} "
            f"(active_high={str(config.pir_active_high).lower()}, bias={config.pir_bias})."
        ),
    )


def _runtime_state_check(config: TwinrConfig) -> ConfigCheck:
    path = Path(config.runtime_state_path)
    parent = path.parent if path.parent != Path("") else Path(".")
    if not parent.exists():
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "warn",
            f"Snapshot directory `{parent}` does not exist yet. It will be created on first write.",
        )
    if parent.is_dir():
        return ConfigCheck(
            "runtime_state",
            "Runtime snapshot",
            "ok",
            f"Snapshot path is `{path}`.",
        )
    return ConfigCheck(
        "runtime_state",
        "Runtime snapshot",
        "fail",
        f"Snapshot parent `{parent}` is not a directory.",
    )


def _binary_or_path_exists(value: str) -> bool:
    if which(value):
        return True
    return Path(value).exists()
