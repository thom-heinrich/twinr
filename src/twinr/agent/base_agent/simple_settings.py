from __future__ import annotations

import math
import os
import re
import stat
import tempfile
import threading
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

# AUDIT-FIX(#1): Serialize local .env writes so concurrent async tasks cannot interleave read-modify-write cycles.
_ENV_UPDATE_LOCK = threading.RLock()
# AUDIT-FIX(#3): Reject invalid env keys up front to prevent malformed .env lines and variable injection.
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# AUDIT-FIX(#5): Fall back to a safe known preset when persisted memory settings are missing or malformed.
_DEFAULT_MEMORY_PRESET = _MEMORY_CAPACITY_PRESETS[1]
# AUDIT-FIX(#7): Ignore filler words when resolving descriptive voice requests.
_VOICE_MATCH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "bit",
        "clear",
        "for",
        "little",
        "more",
        "of",
        "slightly",
        "the",
        "to",
        "very",
        "voice",
        "with",
    }
)


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
    speech_pause_ms = current_numeric_setting(config, "speech_pause_ms")
    follow_up_timeout_s = current_numeric_setting(config, "follow_up_timeout_s")
    realtime_language = getattr(config, "openai_realtime_language", None)
    return (
        "Current bounded simple settings that you may adjust only after an explicit user request: "
        f"memory_capacity level {level}/4 ({label}; max turns {max_turns}, keep recent {keep_recent}); "
        f"spoken_voice {current_voice} (available: {spoken_voice_options_context(language=realtime_language)}); "
        f"speech_speed {current_speed:.2f}x; "
        f"speech_pause_ms {speech_pause_ms}; "
        f"follow_up_timeout_s {float(follow_up_timeout_s):.1f}. "
        f"{spoken_voice_language_note(realtime_language)} "
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
    if not updates:
        return  # AUDIT-FIX(#8): Empty update sets are a no-op and must not create or rewrite the env file.

    file_path = _validated_env_file_path(path)  # AUDIT-FIX(#2): Reject unsafe paths before touching the filesystem.
    validated_updates = _validated_env_updates(updates)  # AUDIT-FIX(#3): Reject malformed keys/values before serialization.
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with _ENV_UPDATE_LOCK:  # AUDIT-FIX(#1): Keep the read-modify-write cycle atomic within the single-process Twinr runtime.
        existing_lines = _read_existing_env_lines(file_path)
        result: list[str] = []
        seen: set[str] = set()
        for line in existing_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                result.append(line)
                continue
            key = stripped.split("=", 1)[0].strip()
            if key in validated_updates:
                if key not in seen:
                    result.append(f"{key}={_quote_env_value(validated_updates[key])}")
                    seen.add(key)
                continue
            result.append(line)

        for key, value in validated_updates.items():
            if key in seen:
                continue
            result.append(f"{key}={_quote_env_value(value)}")

        _atomic_write_text(
            file_path,
            "\n".join(result).rstrip() + "\n",
            encoding="utf-8",
        )


def memory_capacity_level(config: TwinrConfig) -> tuple[int, str, int, int]:
    default_level, _default_label, default_max_turns, default_keep_recent = _DEFAULT_MEMORY_PRESET
    del default_level
    target = (
        _coerce_config_int(
            getattr(config, "memory_max_turns", default_max_turns),
            default=default_max_turns,
        ),
        _coerce_config_int(
            getattr(config, "memory_keep_recent", default_keep_recent),
            default=default_keep_recent,
        ),
    )
    best = min(
        _MEMORY_CAPACITY_PRESETS,
        key=lambda preset: abs(preset[2] - target[0]) + abs(preset[3] - target[1]),
    )
    return best


def current_spoken_voice(config: TwinrConfig) -> str:
    normalized = _normalized_supported_voice_name(getattr(config, "openai_realtime_voice", None))
    if normalized is not None:
        return normalized
    fallback = _normalized_supported_voice_name(getattr(config, "openai_tts_voice", None))
    if fallback is not None:
        return fallback
    return "sage"


def current_speech_speed(config: TwinrConfig) -> float:
    minimum = float(_SPEECH_SPEED_SPEC["minimum"])
    maximum = float(_SPEECH_SPEED_SPEC["maximum"])
    tts_speed = _coerce_finite_float(
        getattr(config, "openai_tts_speed", 1.0),
        default=1.0,
        minimum=minimum,
        maximum=maximum,
    )
    realtime_speed = _coerce_finite_float(
        getattr(config, "openai_realtime_speed", 1.0),
        default=1.0,
        minimum=minimum,
        maximum=maximum,
    )
    return round((tts_speed + realtime_speed) / 2.0, 2)  # AUDIT-FIX(#4): Clamp malformed persisted numeric values instead of crashing.


def current_numeric_setting(config: TwinrConfig, setting: str) -> float | int:
    spec = _NUMERIC_SETTING_SPECS[setting]
    current_value = _current_numeric_setting_value(config, setting)
    if spec["type"] == "int":
        return int(round(current_value))  # AUDIT-FIX(#4): Return a safe numeric fallback when persisted config is malformed.
    return round(current_value, 1)  # AUDIT-FIX(#4): Return a safe numeric fallback when persisted config is malformed.


def _update_memory_capacity(
    config: TwinrConfig,
    *,
    action: str,
    value: float | int | str | None,
) -> SimpleSettingUpdate:
    current_level, _current_label, _preset_max_turns, _preset_keep_recent = memory_capacity_level(config)
    current_max_turns = _coerce_config_int(
        getattr(config, "memory_max_turns", _DEFAULT_MEMORY_PRESET[2]),
        default=_DEFAULT_MEMORY_PRESET[2],
    )
    current_keep_recent = _coerce_config_int(
        getattr(config, "memory_keep_recent", _DEFAULT_MEMORY_PRESET[3]),
        default=_DEFAULT_MEMORY_PRESET[3],
    )
    next_level = current_level
    if action == "increase":
        next_level = min(len(_MEMORY_CAPACITY_PRESETS), current_level + 1)
    elif action == "decrease":
        next_level = max(1, current_level - 1)
    elif action == "set":
        if value is None:
            raise ValueError("memory_capacity with action=set requires a level value")
        next_level = _parse_memory_capacity_level(value)  # AUDIT-FIX(#6): Reject ambiguous non-integer levels instead of silently banker-rounding them.
        next_level = max(1, min(len(_MEMORY_CAPACITY_PRESETS), next_level))
    else:
        raise ValueError("memory_capacity action must be increase, decrease, or set")

    level, label, max_turns, keep_recent = _MEMORY_CAPACITY_PRESETS[next_level - 1]
    changed = (max_turns, keep_recent) != (current_max_turns, current_keep_recent)
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
            "previous_memory_max_turns": current_max_turns,
            "previous_memory_keep_recent": current_keep_recent,
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
    current_tts_voice = _normalized_supported_voice_name(getattr(config, "openai_tts_voice", None)) or "sage"
    current_realtime_voice = _normalized_supported_voice_name(getattr(config, "openai_realtime_voice", None)) or current_tts_voice
    changed = (
        option["tts_voice"] != current_tts_voice
        or option["realtime_voice"] != current_realtime_voice
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
            "previous_tts_voice": current_tts_voice,
            "previous_realtime_voice": current_realtime_voice,
        },
    )


def _update_speech_speed(
    config: TwinrConfig,
    *,
    action: str,
    value: float | int | str | None,
) -> SimpleSettingUpdate:
    spec = _SPEECH_SPEED_SPEC
    minimum = float(spec["minimum"])
    maximum = float(spec["maximum"])
    step = float(spec["step"])
    current_value = current_speech_speed(config)
    if action == "increase":
        next_value = min(maximum, current_value + step)
    elif action == "decrease":
        next_value = max(minimum, current_value - step)
    elif action == "set":
        if value in {None, ""}:
            raise ValueError("speech_speed with action=set requires a value")
        next_value = _parse_requested_float(
            value,
            setting="speech_speed",
            minimum=minimum,
            maximum=maximum,
        )
    else:
        raise ValueError("speech_speed action must be increase, decrease, or set")

    rendered_value = round(next_value, 2)
    previous_tts = round(
        _coerce_finite_float(
            getattr(config, "openai_tts_speed", 1.0),
            default=1.0,
            minimum=minimum,
            maximum=maximum,
        ),
        2,
    )
    previous_realtime = round(
        _coerce_finite_float(
            getattr(config, "openai_realtime_speed", 1.0),
            default=1.0,
            minimum=minimum,
            maximum=maximum,
        ),
        2,
    )
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
            "minimum": minimum,
            "maximum": maximum,
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

    for option in _SPOKEN_VOICE_OPTIONS:
        option_blobs = (
            _normalize_lookup_text(option["description"]),
            _normalize_lookup_text(option["selection_hint"]),
        )
        if any(normalized in blob for blob in option_blobs):
            return option  # AUDIT-FIX(#7): Accept descriptive spoken-voice requests such as "gentle and reassuring".

    requested_tokens = _meaningful_lookup_tokens(normalized)
    if not requested_tokens:
        return None

    best_option: dict[str, str] | None = None
    best_score = 0
    for option in _SPOKEN_VOICE_OPTIONS:
        option_tokens = _voice_option_lookup_tokens(option)
        score = len(requested_tokens & option_tokens)
        if score > best_score:
            best_option = option
            best_score = score

    return best_option if best_score > 0 else None


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
    current_value = _current_numeric_setting_value(config, setting)
    minimum = float(spec["minimum"])
    maximum = float(spec["maximum"])
    step = float(spec["step"])
    if action == "increase":
        next_value = min(maximum, current_value + step)
    elif action == "decrease":
        next_value = max(minimum, current_value - step)
    elif action == "set":
        if value is None:
            raise ValueError(f"{setting} with action=set requires a value")
        next_value = _parse_requested_float(
            value,
            setting=setting,
            minimum=minimum,
            maximum=maximum,
        )
    else:
        raise ValueError(f"{setting} action must be increase, decrease, or set")

    if spec["type"] == "int":
        rendered_value = int(round(next_value))
        previous_value = int(round(current_value))
        env_value = str(rendered_value)
    else:
        rendered_value = round(next_value, 1)
        previous_value = round(current_value, 1)
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


def _current_numeric_setting_value(config: TwinrConfig, setting: str) -> float:
    spec = _NUMERIC_SETTING_SPECS[setting]
    attribute_name = setting if setting != "follow_up_timeout_s" else "conversation_follow_up_timeout_s"
    return _coerce_finite_float(
        getattr(config, attribute_name, spec["minimum"]),
        default=float(spec["minimum"]),
        minimum=float(spec["minimum"]),
        maximum=float(spec["maximum"]),
    )  # AUDIT-FIX(#4): Clamp malformed persisted numeric values instead of throwing on float() conversion.


def _parse_requested_float(
    value: float | int | str,
    *,
    setting: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{setting} value must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{setting} value must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{setting} value must be a finite number")  # AUDIT-FIX(#4): Reject NaN and infinity explicitly.
    return min(maximum, max(minimum, parsed))


def _parse_memory_capacity_level(value: float | int | str) -> int:
    if isinstance(value, bool):
        raise ValueError("memory_capacity level must be an integer between 1 and 4")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("memory_capacity level must be an integer between 1 and 4") from exc
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError("memory_capacity level must be an integer between 1 and 4")
    return int(parsed)


def _coerce_finite_float(
    value: object,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _coerce_config_int(
    value: object,
    *,
    default: int,
) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed) or not parsed.is_integer():
        return default
    return int(parsed)  # AUDIT-FIX(#5): Treat malformed persisted integer config as safe defaults instead of crashing the nearest-preset lookup.


def _normalized_supported_voice_name(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in _SPOKEN_VOICE_BY_NAME:
        return normalized
    return None


def _meaningful_lookup_tokens(value: str) -> set[str]:
    return {
        token
        for token in value.split()
        if len(token) > 2 and token not in _VOICE_MATCH_STOPWORDS
    }


def _voice_option_lookup_tokens(option: dict[str, str]) -> set[str]:
    return _meaningful_lookup_tokens(
        " ".join(
            (
                _normalize_lookup_text(option["name"]),
                _normalize_lookup_text(option["label"]),
                _normalize_lookup_text(option["description"]),
                _normalize_lookup_text(option["selection_hint"]),
            )
        )
    )


def _validated_env_file_path(path: str | Path) -> Path:
    raw_path = Path(path).expanduser()
    if raw_path.name == "":
        raise ValueError("Environment file path must point to a file")
    if any(part == ".." for part in raw_path.parts):
        raise ValueError("Environment file path must not contain parent-directory traversal")
    candidate = raw_path if raw_path.is_absolute() else Path.cwd() / raw_path
    _ensure_no_symlink_components(candidate.parent)
    try:
        existing_stat = candidate.lstat()
    except FileNotFoundError:
        return candidate
    if stat.S_ISLNK(existing_stat.st_mode):
        raise ValueError("Environment file path must not be a symlink")
    if not stat.S_ISREG(existing_stat.st_mode):
        raise ValueError("Environment file path must point to a regular file")
    return candidate  # AUDIT-FIX(#2): Reject symlinked or traversing paths before reading or replacing files.


def _ensure_no_symlink_components(path: Path) -> None:
    if str(path) == "":
        return
    if path.is_absolute():
        current = Path(path.anchor)
        parts = path.parts[1:]
    else:
        current = Path.cwd()
        parts = path.parts
    for part in parts:
        current /= part
        try:
            path_stat = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(path_stat.st_mode):
            raise ValueError(f"Environment file path contains a symlinked directory: {current}")


def _validated_env_updates(updates: dict[str, str]) -> dict[str, str]:
    validated: dict[str, str] = {}
    for key, value in updates.items():
        key_text = str(key)
        value_text = str(value)
        if not _ENV_KEY_RE.fullmatch(key_text):
            raise ValueError(f"Invalid environment variable name: {key_text}")
        if _contains_unsafe_env_control_chars(value_text):
            raise ValueError(f"Invalid control characters in environment value for {key_text}")
        validated[key_text] = value_text
    return validated


def _contains_unsafe_env_control_chars(value: str) -> bool:
    return any(char in value for char in ("\x00", "\n", "\r"))


def _read_existing_env_lines(file_path: Path) -> list[str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW  # AUDIT-FIX(#2): Refuse to follow a symlink swapped in between validation and read.
    try:
        file_descriptor = os.open(file_path, flags)
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise OSError(f"Could not safely open environment file for reading: {file_path}") from exc

    file_stat = os.fstat(file_descriptor)
    if not stat.S_ISREG(file_stat.st_mode):
        os.close(file_descriptor)
        raise OSError(f"Environment file path is not a regular file: {file_path}")

    with os.fdopen(file_descriptor, "r", encoding="utf-8") as handle:
        return handle.read().splitlines()


def _atomic_write_text(file_path: Path, content: str, *, encoding: str) -> None:
    existing_mode: int | None = None
    try:
        existing_mode = stat.S_IMODE(file_path.lstat().st_mode)
    except FileNotFoundError:
        existing_mode = None

    file_descriptor, temp_path_str = tempfile.mkstemp(
        prefix=f".{file_path.name}.",
        suffix=".tmp",
        dir=file_path.parent,
        text=True,
    )
    temp_path = Path(temp_path_str)
    try:
        if existing_mode is not None:
            os.fchmod(file_descriptor, existing_mode)
        with os.fdopen(file_descriptor, "w", encoding=encoding, newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, file_path)  # AUDIT-FIX(#1): Replace the destination atomically to avoid torn writes on restart or crash.
        _fsync_directory(file_path.parent)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        file_descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _quote_env_value(value: str) -> str:
    if value == "":
        return '""'
    if any(char.isspace() for char in value) or any(char in value for char in ['#', '"', "'"]):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'  # AUDIT-FIX(#9): Preserve leading and trailing spaces instead of stripping them away.
    return value