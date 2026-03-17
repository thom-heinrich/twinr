"""Expose bounded spoken-output primitives for future self_coding skills."""

from __future__ import annotations

import re
from typing import Final, NoReturn

from twinr.agent.self_coding.status import CapabilityRiskClass

from .base import SelfCodingModuleFunction, SelfCodingModuleSpec, runtime_unavailable

# AUDIT-FIX(#1): Enforce the "bounded spoken output" contract locally so malformed
# self-coding calls fail fast before any future runtime wiring or TTS path is touched.
_MAX_UTTERANCE_CHARS: Final[int] = 240
# AUDIT-FIX(#2): Bound symbolic sound identifiers and keep them as inert asset keys.
_MAX_SOUND_NAME_CHARS: Final[int] = 64
# AUDIT-FIX(#2): Restrict sound identifiers to safe symbolic names instead of arbitrary strings.
_SOUND_NAME_RE: Final[re.Pattern[str]] = re.compile(
    rf"^[A-Za-z0-9][A-Za-z0-9_.-]{{0,{_MAX_SOUND_NAME_CHARS - 1}}}$",
    re.ASCII,
)


# AUDIT-FIX(#1): Reject empty, multiline, overlong, and control-character payloads
# at the module boundary to keep future speech output bounded and senior-safe.
def _validate_single_utterance(value: str, *, param_name: str, max_chars: int) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{param_name} must be str, got {type(value).__name__}")

    stripped_value = value.strip()
    if not stripped_value:
        raise ValueError(f"{param_name} must not be empty")

    if len(stripped_value) > max_chars:
        raise ValueError(f"{param_name} must be at most {max_chars} characters")

    if any(separator in value for separator in ("\n", "\r", "\u2028", "\u2029")):
        raise ValueError(f"{param_name} must be a single line")

    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{param_name} must not contain control characters")


# AUDIT-FIX(#2): Treat sound names as safe symbolic identifiers, not free-form text
# or path-like input that could widen downstream asset-selection behavior.
def _validate_sound_name(sound_name: str) -> None:
    if not isinstance(sound_name, str):
        raise TypeError(f"sound_name must be str, got {type(sound_name).__name__}")

    normalized = sound_name.strip()
    if not normalized:
        raise ValueError("sound_name must not be empty")

    if len(normalized) > _MAX_SOUND_NAME_CHARS:
        raise ValueError(
            f"sound_name must be at most {_MAX_SOUND_NAME_CHARS} characters"
        )

    if "/" in normalized or "\\" in normalized or ".." in normalized:
        raise ValueError("sound_name must be a safe symbolic identifier, not a path")

    if not _SOUND_NAME_RE.fullmatch(normalized):
        raise ValueError(
            "sound_name may contain only letters, digits, '.', '_' and '-'"
        )


# AUDIT-FIX(#3): Fail closed even if runtime_unavailable() is softened, mocked, or
# regresses to a log-and-return helper; these builtins must never silently succeed.
def _raise_runtime_unavailable(capability_name: str) -> NoReturn:
    runtime_unavailable(capability_name)
    raise RuntimeError(
        f"{capability_name} unexpectedly returned from runtime_unavailable(); "
        "speaker builtins must fail closed outside the runtime."
    )


def say(text: str) -> None:
    """Speak one short, single-line utterance through Twinr's bounded TTS path."""

    # AUDIT-FIX(#1): Validate spoken text at the public boundary before dispatch.
    _validate_single_utterance(
        text,
        param_name="text",
        max_chars=_MAX_UTTERANCE_CHARS,
    )
    # AUDIT-FIX(#3): Keep the stub fail-closed outside the actual Twinr runtime.
    _raise_runtime_unavailable("speaker.say")


def play_sound(sound_name: str) -> None:
    """Play one named non-verbal sound cue using a safe symbolic sound identifier."""

    # AUDIT-FIX(#2): Validate the symbolic sound identifier before any runtime dispatch.
    _validate_sound_name(sound_name)
    # AUDIT-FIX(#3): Keep the stub fail-closed outside the actual Twinr runtime.
    _raise_runtime_unavailable("speaker.play_sound")


def ask_and_wait(question: str) -> str:
    """Ask one short, single-line question aloud and wait for one bounded spoken reply."""

    # AUDIT-FIX(#1): Validate spoken prompt text at the public boundary before dispatch.
    _validate_single_utterance(
        question,
        param_name="question",
        max_chars=_MAX_UTTERANCE_CHARS,
    )
    # AUDIT-FIX(#3): Keep the stub fail-closed outside the actual Twinr runtime.
    _raise_runtime_unavailable("speaker.ask_and_wait")


# AUDIT-FIX(#4): Make the same safety contract visible in metadata so self-coding
# callers know the brevity and formatting rules before generating code.
MODULE_SPEC = SelfCodingModuleSpec(
    capability_id="speaker",
    module_name="speaker",
    summary=(
        "Speak brief, single-line voice-first output through Twinr's bounded TTS path."
    ),
    risk_class=CapabilityRiskClass.HIGH,
    public_api=(
        SelfCodingModuleFunction(
            name="say",
            signature="say(text: str) -> None",
            summary=(
                "Speak one brief, single-line user-facing sentence "
                "(max 240 chars; no control characters)."
            ),
            effectful=True,
            tags=("effectful", "voice"),
        ),
        SelfCodingModuleFunction(
            name="play_sound",
            signature="play_sound(sound_name: str) -> None",
            summary=(
                "Play one pre-registered sound identifier using only letters, digits, "
                "'.', '_' or '-'."
            ),
            effectful=True,
            tags=("effectful", "voice"),
        ),
        SelfCodingModuleFunction(
            name="ask_and_wait",
            signature="ask_and_wait(question: str) -> str",
            summary=(
                "Speak one brief, single-line question "
                "(max 240 chars) and wait for one bounded spoken answer."
            ),
            returns="the recognized answer text, or an empty string if nothing usable was recognized",
            effectful=True,
            tags=("effectful", "voice", "dialogue"),
        ),
    ),
    tags=("output", "builtin", "voice"),
)

__all__ = ["MODULE_SPEC", "ask_and_wait", "play_sound", "say"]