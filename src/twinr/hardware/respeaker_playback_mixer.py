# CHANGELOG: 2026-03-29
# BUG-1: Reassert the XVF3800 Linux playback mixer at runtime so spoken replies
#        do not stay almost inaudible when the card drifts below 100 percent.

"""Keep ReSpeaker XVF3800 playback loud enough on Linux runtime startups.

XMOS documents a Linux-specific XVF3800 playback issue where the PCM mixer can
drop well below 100 percent and must be raised again before the loudspeaker is
audible. Twinr already normalizes this during Pi audio setup; this module
repeats the same bounded mixer repair when the runtime constructs its playback
backend so a drifted card does not leave spoken replies whisper-quiet.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import shutil
import subprocess

from twinr.hardware.respeaker_duplex_playback import config_targets_respeaker

_LOGGER = logging.getLogger(__name__)

_DEFAULT_TARGET_PERCENT = 100
_PLAYBACK_DEVICE_ALIASES = frozenset({"twinr_playback_hw", "twinr_playback_softvol"})
_ALSA_CARD_TOKEN_PREFIXES = frozenset(
    {
        "hw",
        "plughw",
        "sysdefault",
        "front",
        "rear",
        "center_lfe",
        "side",
        "surround40",
        "surround41",
        "surround50",
        "surround51",
        "surround71",
        "iec958",
        "dmix",
        "dsnoop",
        "asym",
        "plug",
        "softvol",
    }
)
_ALSA_CARD_LINE_RE = re.compile(r"^\s*(?P<index>\d+)\s+\[(?P<id>[^\]]+)\]:\s*(?P<desc>.*)$")
_PLAYBACK_CAPABILITY_MARKERS = (
    "playback channels:",
    "capabilities: pvolume",
    "capabilities: pswitch",
)


@dataclass(frozen=True, slots=True)
class RespeakerPlaybackMixerNormalizationResult:
    """Summarize one bounded runtime playback-mixer normalization attempt."""

    attempted: bool
    card_index: int | None
    normalized_controls: tuple[str, ...]
    stored_state: bool
    reason: str


def ensure_respeaker_playback_mixer(
    playback_device: str | None,
    *,
    target_percent: int = _DEFAULT_TARGET_PERCENT,
) -> RespeakerPlaybackMixerNormalizationResult:
    """Raise the active XVF3800 playback controls back to an audible level."""

    normalized_device = str(playback_device or "").strip()
    if not _device_targets_respeaker_playback(normalized_device):
        return RespeakerPlaybackMixerNormalizationResult(
            attempted=False,
            card_index=None,
            normalized_controls=(),
            stored_state=False,
            reason="non_respeaker_playback",
        )

    card_index = _resolve_respeaker_card_index(normalized_device)
    if card_index is None:
        return RespeakerPlaybackMixerNormalizationResult(
            attempted=False,
            card_index=None,
            normalized_controls=(),
            stored_state=False,
            reason="respeaker_card_not_found",
        )

    amixer_path = shutil.which("amixer")
    if not amixer_path:
        return RespeakerPlaybackMixerNormalizationResult(
            attempted=False,
            card_index=card_index,
            normalized_controls=(),
            stored_state=False,
            reason="amixer_missing",
        )

    normalized_percent = max(0, min(100, int(target_percent)))
    normalized_controls: list[str] = []
    for control_ref in _list_simple_mixer_controls(amixer_path=amixer_path, card_index=card_index):
        if not _control_has_playback_capability(
            amixer_path=amixer_path,
            card_index=card_index,
            control_ref=control_ref,
        ):
            continue
        if _set_control_percent(
            amixer_path=amixer_path,
            card_index=card_index,
            control_ref=control_ref,
            percent=normalized_percent,
        ):
            normalized_controls.append(control_ref)

    stored_state = False
    if normalized_controls:
        stored_state = _store_card_playback_state(card_index)
        _LOGGER.info(
            "ReSpeaker playback mixer normalized on card %s via %s (stored=%s).",
            card_index,
            ", ".join(normalized_controls),
            stored_state,
        )
    else:
        _LOGGER.warning(
            "ReSpeaker playback mixer normalization found no playback-capable controls on card %s.",
            card_index,
        )

    return RespeakerPlaybackMixerNormalizationResult(
        attempted=True,
        card_index=card_index,
        normalized_controls=tuple(normalized_controls),
        stored_state=stored_state,
        reason="ok" if normalized_controls else "no_playback_controls",
    )


def _device_targets_respeaker_playback(playback_device: str) -> bool:
    normalized = playback_device.strip().lower()
    if not normalized:
        return False
    if normalized in _PLAYBACK_DEVICE_ALIASES:
        return True
    return config_targets_respeaker(playback_device)


def _extract_alsa_card_token(playback_device: str) -> str | None:
    normalized = playback_device.strip().lower()
    if not normalized or ":" not in normalized:
        return None
    card_marker = "card="
    marker_index = normalized.find(card_marker)
    if marker_index >= 0:
        tail = normalized[marker_index + len(card_marker) :]
        token = tail.split(",", 1)[0].split(":", 1)[0].strip()
        return token or None
    prefix, tail = normalized.split(":", 1)
    if prefix not in _ALSA_CARD_TOKEN_PREFIXES:
        return None
    token = tail.split(",", 1)[0].strip()
    return token or None


def _read_alsa_cards() -> list[dict[str, str]]:
    try:
        lines = Path("/proc/asound/cards").read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []

    cards: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in lines:
        match = _ALSA_CARD_LINE_RE.match(line)
        if match:
            current = {
                "index": match.group("index").strip(),
                "id": match.group("id").strip().lower(),
                "text": " ".join(
                    part for part in (match.group("id"), match.group("desc")) if part
                ).strip().lower(),
            }
            cards.append(current)
            continue
        if current is not None and line[:1].isspace():
            current["text"] = f'{current["text"]} {line.strip().lower()}'.strip()
        else:
            current = None
    return cards


def _resolve_respeaker_card_index(playback_device: str) -> int | None:
    card_token = _extract_alsa_card_token(playback_device)
    cards = _read_alsa_cards()
    for card in cards:
        if not any(marker in card["text"] for marker in ("respeaker", "xvf3800", "array")):
            continue
        if card_token is None or card_token in {card["index"], card["id"]}:
            return int(card["index"])
    return None


def _run_text_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _list_simple_mixer_controls(*, amixer_path: str, card_index: int) -> tuple[str, ...]:
    completed = _run_text_command([amixer_path, "-c", str(card_index), "scontrols"])
    if completed.returncode != 0:
        return ()
    control_refs: list[str] = []
    prefix = "Simple mixer control '"
    for raw_line in (completed.stdout or "").splitlines():
        if not raw_line.startswith(prefix):
            continue
        control_line = raw_line[len(prefix) :]
        if "'," not in control_line:
            continue
        control_name, control_index = control_line.rsplit("',", 1)
        control_refs.append(f"{control_name},{control_index.strip()}")
    return tuple(control_refs)


def _control_has_playback_capability(
    *,
    amixer_path: str,
    card_index: int,
    control_ref: str,
) -> bool:
    completed = _run_text_command([amixer_path, "-c", str(card_index), "sget", control_ref])
    if completed.returncode != 0:
        return False
    output = (completed.stdout or "").lower()
    return any(marker in output for marker in _PLAYBACK_CAPABILITY_MARKERS)


def _set_control_percent(
    *,
    amixer_path: str,
    card_index: int,
    control_ref: str,
    percent: int,
) -> bool:
    completed = _run_text_command(
        [amixer_path, "-c", str(card_index), "sset", control_ref, f"{percent}%", "unmute"]
    )
    return completed.returncode == 0


def _store_card_playback_state(card_index: int) -> bool:
    alsactl_path = shutil.which("alsactl")
    if not alsactl_path:
        return False
    completed = _run_text_command([alsactl_path, "store", str(card_index)])
    return completed.returncode == 0


__all__ = [
    "RespeakerPlaybackMixerNormalizationResult",
    "ensure_respeaker_playback_mixer",
]
