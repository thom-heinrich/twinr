from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.language import language_display_name
from twinr.text_utils import folded_lookup_text

_MEMORY_CAPACITY_PRESETS: tuple[tuple[int, str, int, int], ...] = (
    (1, "compact", 12, 6),
    (2, "balanced", 20, 10),
    (3, "extended", 28, 12),
    (4, "high", 32, 14),
)
_NUMERIC_SETTING_SPECS = {
    "speech_pause_ms": {
        "label": "speech pause",
        "env_key": "TWINR_SPEECH_PAUSE_MS",
        "minimum": 700,
        "maximum": 2200,
        "step": 200,
        "type": "int",
        "unit": "ms",
    },
    "follow_up_timeout_s": {
        "label": "follow-up listening window",
        "env_key": "TWINR_CONVERSATION_FOLLOW_UP_TIMEOUT_S",
        "minimum": 2.0,
        "maximum": 8.0,
        "step": 1.0,
        "type": "float",
        "unit": "s",
    },
}
_SPEECH_SPEED_SPEC = {
    "label": "speech speed",
    "env_keys": ("OPENAI_TTS_SPEED", "OPENAI_REALTIME_SPEED"),
    "minimum": 0.75,
    "maximum": 1.15,
    "step": 0.1,
    "unit": "x",
}
_SPOKEN_VOICE_OPTIONS: tuple[dict[str, str], ...] = (
    {
        "name": "marin",
        "label": "Marin",
        "description": "warm, calm, and very natural",
        "tts_voice": "marin",
        "realtime_voice": "marin",
        "selection_hint": "warm, calm, natural, and a little lighter",
    },
    {
        "name": "cedar",
        "label": "Cedar",
        "description": "calm, slightly deeper, and steady",
        "tts_voice": "cedar",
        "realtime_voice": "cedar",
        "selection_hint": "calm, steady, deeper, and more masculine",
    },
    {
        "name": "sage",
        "label": "Sage",
        "description": "gentle, soft, and reassuring",
        "tts_voice": "sage",
        "realtime_voice": "sage",
        "selection_hint": "gentle, soft, reassuring, and neutral",
    },
    {
        "name": "alloy",
        "label": "Alloy",
        "description": "neutral, clear, and balanced",
        "tts_voice": "alloy",
        "realtime_voice": "alloy",
        "selection_hint": "neutral, clear, and balanced",
    },
    {
        "name": "coral",
        "label": "Coral",
        "description": "bright, friendly, and cheerful",
        "tts_voice": "coral",
        "realtime_voice": "coral",
        "selection_hint": "bright, friendly, cheerful, and a little lighter",
    },
    {
        "name": "echo",
        "label": "Echo",
        "description": "clear, direct, and slightly firmer",
        "tts_voice": "echo",
        "realtime_voice": "echo",
        "selection_hint": "clear, firmer, direct, and a little deeper",
    },
)
_SPOKEN_VOICE_BY_NAME = {
    option["name"]: option for option in _SPOKEN_VOICE_OPTIONS
}
@dataclass(frozen=True, slots=True)
class SimpleSettingUpdate:
    setting: str
    label: str
    summary: str
    env_updates: dict[str, str]
    changed: bool
    data: dict[str, object]


def supported_setting_names() -> tuple[str, ...]:
    return ("memory_capacity", "spoken_voice", "speech_speed", *tuple(_NUMERIC_SETTING_SPECS))


def supported_spoken_voices(language: str | None = None) -> tuple[str, ...]:
    del language
    return tuple(option["name"] for option in _SPOKEN_VOICE_OPTIONS)


def spoken_voice_options_context(*, language: str | None = None) -> str:
    del language
    return "; ".join(
        f"{option['name']} ({option['selection_hint']})"
        for option in _SPOKEN_VOICE_OPTIONS
    )


def spoken_voice_language_note(language: str | None) -> str:
    language_name = language_display_name(language)
    return (
        f"All supported Twinr spoken voices can speak {language_name}. "
        "OpenAI notes that the built-in voices are optimized for English, so accent quality can vary a bit in other languages."
    )


def adjustable_settings_context(config: TwinrConfig) -> str:
    level, label, max_turns, keep_recent = memory_capacity_level(config)
    current_voice = current_spoken_voice(config)
    current_speed = current_speech_speed(config)
    return (
        "Current bounded simple settings that you may adjust only after an explicit user request: "
        f"memory_capacity level {level}/4 ({label}; max turns {max_turns}, keep recent {keep_recent}); "
        f"spoken_voice {current_voice} (available: {spoken_voice_options_context(language=config.openai_realtime_language)}); "
        f"speech_speed {current_speed:.2f}x; "
        f"speech_pause_ms {config.speech_pause_ms}; "
        f"follow_up_timeout_s {config.conversation_follow_up_timeout_s:.1f}. "
        f"{spoken_voice_language_note(config.openai_realtime_language)} "
        "If the user asks which voices are available, answer from this list. "
        "When the user describes a desired voice, resolve it to one supported voice from this list and only write that supported voice name."
    )


def update_simple_setting(
    config: TwinrConfig,
    *,
    setting: str,
    action: str,
    value: float | int | str | None = None,
) -> SimpleSettingUpdate:
    normalized_setting = setting.strip().lower()
    normalized_action = action.strip().lower()
    if normalized_setting == "memory_capacity":
        return _update_memory_capacity(config, action=normalized_action, value=value)
    if normalized_setting == "spoken_voice":
        return _update_spoken_voice(config, action=normalized_action, value=value)
    if normalized_setting == "speech_speed":
        return _update_speech_speed(config, action=normalized_action, value=value)
    if normalized_setting in _NUMERIC_SETTING_SPECS:
        return _update_numeric_setting(config, setting=normalized_setting, action=normalized_action, value=value)
    raise ValueError(f"Unsupported simple setting: {setting}")


def write_env_updates(path: str | Path, updates: dict[str, str]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines = file_path.read_text(encoding="utf-8").splitlines() if file_path.exists() else []

    result: list[str] = []
    seen: set[str] = set()
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            result.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            if key not in seen:
                result.append(f"{key}={_quote_env_value(updates[key])}")
                seen.add(key)
            continue
        result.append(line)

    for key, value in updates.items():
        if key in seen:
            continue
        result.append(f"{key}={_quote_env_value(value)}")

    file_path.write_text("\n".join(result).rstrip() + "\n", encoding="utf-8")


def memory_capacity_level(config: TwinrConfig) -> tuple[int, str, int, int]:
    target = (config.memory_max_turns, config.memory_keep_recent)
    best = min(
        _MEMORY_CAPACITY_PRESETS,
        key=lambda preset: abs(preset[2] - target[0]) + abs(preset[3] - target[1]),
    )
    return best


def current_spoken_voice(config: TwinrConfig) -> str:
    normalized = (config.openai_realtime_voice or config.openai_tts_voice or "sage").strip().lower()
    if normalized in _SPOKEN_VOICE_BY_NAME:
        return normalized
    fallback = (config.openai_tts_voice or normalized or "sage").strip().lower()
    if fallback in _SPOKEN_VOICE_BY_NAME:
        return fallback
    return "sage"


def current_speech_speed(config: TwinrConfig) -> float:
    return round((float(config.openai_tts_speed) + float(config.openai_realtime_speed)) / 2.0, 2)


def _update_memory_capacity(
    config: TwinrConfig,
    *,
    action: str,
    value: float | int | str | None,
) -> SimpleSettingUpdate:
    current_level, _current_label, _preset_max_turns, _preset_keep_recent = memory_capacity_level(config)
    next_level = current_level
    if action == "increase":
        next_level = min(len(_MEMORY_CAPACITY_PRESETS), current_level + 1)
    elif action == "decrease":
        next_level = max(1, current_level - 1)
    elif action == "set":
        if value is None:
            raise ValueError("memory_capacity with action=set requires a level value")
        try:
            next_level = int(round(float(value)))
        except (TypeError, ValueError) as exc:
            raise ValueError("memory_capacity level must be a number between 1 and 4") from exc
        next_level = max(1, min(len(_MEMORY_CAPACITY_PRESETS), next_level))
    else:
        raise ValueError("memory_capacity action must be increase, decrease, or set")

    level, label, max_turns, keep_recent = _MEMORY_CAPACITY_PRESETS[next_level - 1]
    changed = (max_turns, keep_recent) != (config.memory_max_turns, config.memory_keep_recent)
    summary = (
        f"memory capacity {label} (level {level}/4, max turns {max_turns}, keep recent {keep_recent})"
    )
    return SimpleSettingUpdate(
        setting="memory_capacity",
        label="memory capacity",
        summary=summary,
        env_updates={
            "TWINR_MEMORY_MAX_TURNS": str(max_turns),
            "TWINR_MEMORY_KEEP_RECENT": str(keep_recent),
        },
        changed=changed,
        data={
            "level": level,
            "label": label,
            "memory_max_turns": max_turns,
            "memory_keep_recent": keep_recent,
            "previous_level": current_level,
            "previous_memory_max_turns": config.memory_max_turns,
            "previous_memory_keep_recent": config.memory_keep_recent,
        },
    )


def _update_spoken_voice(
    config: TwinrConfig,
    *,
    action: str,
    value: float | int | str | None,
) -> SimpleSettingUpdate:
    if action != "set":
        raise ValueError("spoken_voice action must be set")
    requested_voice = str(value or "").strip()
    if not requested_voice:
        raise ValueError(f"spoken_voice requires one of: {', '.join(supported_spoken_voices())}")
    option = _resolve_spoken_voice_option(requested_voice)
    if option is None:
        raise ValueError(f"spoken_voice must be one of: {', '.join(supported_spoken_voices())}")
    changed = (
        option["tts_voice"] != config.openai_tts_voice
        or option["realtime_voice"] != config.openai_realtime_voice
    )
    summary = f"spoken voice {option['label']} ({option['description']})"
    return SimpleSettingUpdate(
        setting="spoken_voice",
        label="spoken voice",
        summary=summary,
        env_updates={
            "OPENAI_TTS_VOICE": option["tts_voice"],
            "OPENAI_REALTIME_VOICE": option["realtime_voice"],
        },
        changed=changed,
        data={
            "voice": option["name"],
            "tts_voice": option["tts_voice"],
            "realtime_voice": option["realtime_voice"],
            "description": option["description"],
            "previous_tts_voice": config.openai_tts_voice,
            "previous_realtime_voice": config.openai_realtime_voice,
        },
    )


def _update_speech_speed(
    config: TwinrConfig,
    *,
    action: str,
    value: float | int | str | None,
) -> SimpleSettingUpdate:
    spec = _SPEECH_SPEED_SPEC
    current_value = current_speech_speed(config)
    if action == "increase":
        next_value = min(float(spec["maximum"]), current_value + float(spec["step"]))
    elif action == "decrease":
        next_value = max(float(spec["minimum"]), current_value - float(spec["step"]))
    elif action == "set":
        if value in {None, ""}:
            raise ValueError("speech_speed with action=set requires a value")
        try:
            next_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("speech_speed value must be numeric") from exc
        next_value = min(float(spec["maximum"]), max(float(spec["minimum"]), next_value))
    else:
        raise ValueError("speech_speed action must be increase, decrease, or set")

    rendered_value = round(next_value, 2)
    previous_tts = round(float(config.openai_tts_speed), 2)
    previous_realtime = round(float(config.openai_realtime_speed), 2)
    summary = f"speech speed {rendered_value:.2f}x"
    return SimpleSettingUpdate(
        setting="speech_speed",
        label=str(spec["label"]),
        summary=summary,
        env_updates={
            str(spec["env_keys"][0]): f"{rendered_value:.2f}",
            str(spec["env_keys"][1]): f"{rendered_value:.2f}",
        },
        changed=rendered_value != previous_tts or rendered_value != previous_realtime,
        data={
            "speech_speed": rendered_value,
            "previous_tts_speed": previous_tts,
            "previous_realtime_speed": previous_realtime,
            "minimum": float(spec["minimum"]),
            "maximum": float(spec["maximum"]),
            "unit": str(spec["unit"]),
        },
    )


def _resolve_spoken_voice_option(requested_voice: str) -> dict[str, str] | None:
    normalized = _normalize_lookup_text(requested_voice)
    if not normalized:
        return None
    for option in _SPOKEN_VOICE_OPTIONS:
        if normalized in {
            _normalize_lookup_text(option["name"]),
            _normalize_lookup_text(option["label"]),
        }:
            return option
    return None


def _normalize_lookup_text(value: str) -> str:
    return folded_lookup_text(value)


def _update_numeric_setting(
    config: TwinrConfig,
    *,
    setting: str,
    action: str,
    value: float | int | str | None,
) -> SimpleSettingUpdate:
    spec = _NUMERIC_SETTING_SPECS[setting]
    current_value = getattr(config, setting if setting != "follow_up_timeout_s" else "conversation_follow_up_timeout_s")
    minimum = spec["minimum"]
    maximum = spec["maximum"]
    step = spec["step"]
    if action == "increase":
        next_value = min(maximum, float(current_value) + float(step))
    elif action == "decrease":
        next_value = max(minimum, float(current_value) - float(step))
    elif action == "set":
        if value is None:
            raise ValueError(f"{setting} with action=set requires a value")
        try:
            next_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{setting} value must be numeric") from exc
        next_value = min(maximum, max(minimum, next_value))
    else:
        raise ValueError(f"{setting} action must be increase, decrease, or set")

    if spec["type"] == "int":
        rendered_value = int(round(next_value))
        previous_value = int(round(float(current_value)))
        env_value = str(rendered_value)
    else:
        rendered_value = round(next_value, 1)
        previous_value = round(float(current_value), 1)
        env_value = f"{rendered_value:.1f}"

    unit = spec["unit"]
    summary = f"{spec['label']} {rendered_value}{unit}"
    return SimpleSettingUpdate(
        setting=setting,
        label=str(spec["label"]),
        summary=summary,
        env_updates={str(spec["env_key"]): env_value},
        changed=rendered_value != previous_value,
        data={
            "value": rendered_value,
            "previous_value": previous_value,
            "minimum": minimum,
            "maximum": maximum,
            "unit": unit,
        },
    )


def _quote_env_value(value: str) -> str:
    normalized = value.strip()
    if normalized == "":
        return '""'
    if any(char.isspace() for char in normalized) or any(char in normalized for char in ['#', '"', "'"]):
        escaped = normalized.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return normalized
