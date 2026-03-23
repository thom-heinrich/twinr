"""Fail-closed policy for the production voice gateway.

Twinr's live voice gateway now has one supported architecture: transcript-first
server-side wake detection over the remote/local STT path. This module keeps
that product decision out of transport wiring so the websocket server can
reject stale rescue launches before they reopen the old backend/openwakeword
path.
"""

from __future__ import annotations

from twinr.agent.base_agent.config import TwinrConfig


_REQUIRED_STAGE1_MODE = "local_stt"


def assert_transcript_first_voice_gateway_contract(config: TwinrConfig) -> None:
    """Raise when the live voice gateway drifts away from the thh1986 STT path."""

    stage1_mode = str(getattr(config, "voice_orchestrator_wake_stage1_mode", "") or "").strip().lower()
    if stage1_mode != _REQUIRED_STAGE1_MODE:
        raise ValueError(
            "Twinr voice gateway must run transcript-first local_stt stage1 only; "
            "backend/openwakeword rescue paths are forbidden for the live gateway."
        )
    local_stt_url = str(getattr(config, "voice_orchestrator_local_stt_url", "") or "").strip()
    if not local_stt_url:
        raise ValueError(
            "Twinr voice gateway requires TWINR_VOICE_ORCHESTRATOR_LOCAL_STT_URL; "
            "the thh1986 transcript-first path is the only supported live wake path."
        )
    if bool(getattr(config, "wakeword_enabled", False)):
        raise ValueError(
            "Twinr voice gateway must keep generic wakeword backends disabled; "
            "the thh1986 transcript-first path is the only supported live wake path."
        )


__all__ = ["assert_transcript_first_voice_gateway_contract"]
