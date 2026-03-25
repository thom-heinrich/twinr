"""Fail-closed guardrails for the production voice gateway.

Twinr's live voice gateway has one supported architecture: transcript-first
server-side activation detection over the remote thh1986 ASR path. These
checks stay separate from the websocket transport so server startup can reject
legacy or incomplete voice env before any live gateway process accepts audio.
"""

from __future__ import annotations

from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig


_RETIRED_VOICE_ENV_KEYS = (
    "TWINR_VOICE_ORCHESTRATOR_WAKE_STAGE1_MODE",
    "TWINR_WAKEWORD_ENABLED",
    "TWINR_WAKEWORD_BACKEND",
    "TWINR_WAKEWORD_PRIMARY_BACKEND",
    "TWINR_WAKEWORD_FALLBACK_BACKEND",
    "TWINR_WAKEWORD_VERIFIER_MODE",
)


def _read_env_values(env_file: str | None) -> dict[str, str]:
    """Return a minimal key/value view of one env file for contract checks."""

    if not env_file:
        return {}
    path = Path(env_file)
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _assert_no_retired_voice_env(env_file: str | None) -> None:
    """Raise when the voice-gateway env still declares retired voice knobs."""

    env_values = _read_env_values(env_file)
    for key in _RETIRED_VOICE_ENV_KEYS:
        if key in env_values:
            raise ValueError(
                f"{key} is retired and must be removed from the voice-gateway env; "
                "Twinr runs only the transcript-first remote ASR voice path."
            )


def assert_transcript_first_voice_gateway_contract(
    config: TwinrConfig,
    *,
    env_file: str | None = None,
) -> None:
    """Raise when the live voice gateway drifts away from the thh1986 ASR path."""

    _assert_no_retired_voice_env(env_file)
    remote_asr_url = str(getattr(config, "voice_orchestrator_remote_asr_url", "") or "").strip()
    if not remote_asr_url:
        raise ValueError(
            "Twinr voice gateway requires TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_URL; "
            "the thh1986 transcript-first path is the only supported live activation path."
        )


__all__ = ["assert_transcript_first_voice_gateway_contract"]
