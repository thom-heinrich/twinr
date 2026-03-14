from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Event, Thread
from typing import Literal

WorkingFeedbackKind = Literal["processing", "answering", "printing"]


@dataclass(frozen=True, slots=True)
class WorkingFeedbackProfile:
    delay_ms: int
    pause_ms: int
    volume: float
    gap_ms: int
    patterns: tuple[tuple[tuple[int, int], ...], ...]


_DEFAULT_WORKING_FEEDBACK_PROFILES: dict[WorkingFeedbackKind, WorkingFeedbackProfile] = {
    "processing": WorkingFeedbackProfile(
        delay_ms=450,
        pause_ms=620,
        volume=0.12,
        gap_ms=28,
        patterns=(
            ((659, 52), (880, 38)),
            ((740, 44), (988, 34), (831, 40)),
            ((698, 46), (932, 36)),
            ((784, 42), (1046, 32), (932, 34)),
        ),
    ),
    "answering": WorkingFeedbackProfile(
        delay_ms=320,
        pause_ms=560,
        volume=0.13,
        gap_ms=24,
        patterns=(
            ((988, 42), (1318, 32)),
            ((1046, 38), (1396, 28), (1175, 32)),
            ((1175, 36), (1568, 26)),
            ((932, 40), (1244, 30), (1396, 28)),
        ),
    ),
    "printing": WorkingFeedbackProfile(
        delay_ms=180,
        pause_ms=500,
        volume=0.14,
        gap_ms=0,
        patterns=(
            ((988, 58),),
            ((1046, 54),),
            ((932, 60),),
            ((1108, 52),),
        ),
    ),
}


def _play_sequence(
    player,
    sequence: tuple[tuple[int, int], ...],
    *,
    volume: float,
    sample_rate: int,
    gap_ms: int,
) -> None:
    play_tone_sequence = getattr(player, "play_tone_sequence", None)
    if callable(play_tone_sequence):
        play_tone_sequence(
            sequence,
            volume=volume,
            sample_rate=sample_rate,
            gap_ms=gap_ms,
        )
        return
    for frequency_hz, duration_ms in sequence:
        player.play_tone(
            frequency_hz=frequency_hz,
            duration_ms=duration_ms,
            volume=volume,
            sample_rate=sample_rate,
        )


def start_working_feedback_loop(
    player,
    *,
    kind: WorkingFeedbackKind,
    sample_rate: int,
    emit: Callable[[str], None] | None = None,
    profiles: Mapping[WorkingFeedbackKind, WorkingFeedbackProfile] | None = None,
) -> Callable[[], None]:
    profile_map = profiles or _DEFAULT_WORKING_FEEDBACK_PROFILES
    profile = profile_map[kind]
    stop_event = Event()

    def worker() -> None:
        if stop_event.wait(max(0.0, profile.delay_ms / 1000.0)):
            return
        pattern_index = 0
        while not stop_event.is_set():
            sequence = profile.patterns[pattern_index % len(profile.patterns)]
            pattern_index += 1
            try:
                _play_sequence(
                    player,
                    sequence,
                    volume=profile.volume,
                    sample_rate=sample_rate,
                    gap_ms=profile.gap_ms,
                )
            except Exception as exc:
                if emit is not None:
                    emit(f"working_feedback_error={kind}:{exc}")
                return
            if stop_event.wait(max(0.05, profile.pause_ms / 1000.0)):
                return

    thread = Thread(target=worker, daemon=True)
    thread.start()

    def stop() -> None:
        stop_event.set()
        thread.join(timeout=1.0)

    return stop
