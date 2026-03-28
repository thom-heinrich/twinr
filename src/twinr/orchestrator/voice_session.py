"""Compatibility wrapper for the orchestrator voice session.

Import ``EdgeOrchestratorVoiceSession`` from this module. The implementation now
lives in focused sibling modules so runtime-state handling, observability,
transcription, and utterance buffering can evolve independently without
changing callers.
"""

##REFACTOR: 2026-03-27##
from __future__ import annotations

from twinr.orchestrator.voice_session_impl import EdgeOrchestratorVoiceSessionImpl


class EdgeOrchestratorVoiceSession(EdgeOrchestratorVoiceSessionImpl):
    """Preserve the legacy module import path for the voice session."""


__all__ = ["EdgeOrchestratorVoiceSession"]
