"""Fail-closed policy for the production voice gateway.

Twinr's live voice gateway now has one supported architecture: transcript-first
server-side activation detection over the remote thh1986 ASR path. This module
keeps that product decision out of transport wiring so the websocket server can
reject stale launches before they reopen deleted local detector paths.
"""

from __future__ import annotations

from twinr.agent.base_agent.config import TwinrConfig


_REQUIRED_STAGE1_MODE = "remote_asr"


def assert_transcript_first_voice_gateway_contract(config: TwinrConfig) -> None:
    """Raise when the live voice gateway drifts away from the thh1986 ASR path."""

    stage1_mode = str(getattr(config, "voice_orchestrator_wake_stage1_mode", "") or "").strip().lower()
    if stage1_mode != _REQUIRED_STAGE1_MODE:
        raise ValueError(
            "Twinr voice gateway must run transcript-first remote_asr stage1 only; "
            "deleted local detector paths are forbidden for the live gateway."
        )
    remote_asr_url = str(getattr(config, "voice_orchestrator_remote_asr_url", "") or "").strip()
    if not remote_asr_url:
        raise ValueError(
            "Twinr voice gateway requires TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_URL; "
            "the thh1986 transcript-first path is the only supported live wake path."
        )
__all__ = ["assert_transcript_first_voice_gateway_contract"]
