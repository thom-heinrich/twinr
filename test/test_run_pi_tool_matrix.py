"""Regression coverage for the live Pi tool-matrix harness helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
import unittest

from test.run_pi_tool_matrix import run_text_turn


@dataclass
class _FakeResponse:
    text: str = ""
    tool_calls: list[object] = None  # type: ignore[assignment]
    tool_results: list[object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tool_calls is None:
            self.tool_calls = []
        if self.tool_results is None:
            self.tool_results = []


class _FailingStreamingTurnLoop:
    def run(self, *args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        raise RuntimeError("primary failure")


class _FakeRuntime:
    def begin_listening(self, **kwargs):  # noqa: ANN003
        del kwargs

    def submit_transcript(self, prompt: str) -> None:
        del prompt

    def begin_answering(self) -> None:
        return None

    def finalize_agent_turn(self, text: str) -> str:
        return text

    def record_personality_tool_history(self, **kwargs):  # noqa: ANN003
        del kwargs

    def finish_speaking(self) -> None:
        return None

    def fail(self, message: str) -> None:
        raise RuntimeError(f"fail handler exploded: {message}")

    @property
    def status(self):  # noqa: D401
        class _Status:
            value = "waiting"

        return _Status()

    def tool_provider_conversation_context(self) -> list[object]:
        return []


class _FakeLoop:
    def __init__(self) -> None:
        self.emit = lambda _line: None
        self.runtime = _FakeRuntime()
        self.streaming_turn_loop = _FailingStreamingTurnLoop()
        self.config = type(
            "Config",
            (),
            {
                "openai_realtime_input_sample_rate": 24000,
                "openai_realtime_instructions": "",
            },
        )()
        self._current_turn_audio_pcm = None


class RunTextTurnTest(unittest.TestCase):
    def test_run_text_turn_surfaces_runtime_fail_errors(self) -> None:
        loop = _FakeLoop()

        with self.assertRaisesRegex(RuntimeError, "fail handler exploded: primary failure"):
            run_text_turn(cast(Any, loop), "Bitte sei ruhig.")


if __name__ == "__main__":
    unittest.main()
