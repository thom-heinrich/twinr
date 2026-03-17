from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.llm_json import request_structured_json_object


class _FakeResponsesAPI:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("No fake responses left for create()")
        return self._responses.pop(0)


class _FakeBackend:
    def __init__(self, responses: list[object]) -> None:
        self.responses_api = _FakeResponsesAPI(responses)
        self._client = SimpleNamespace(responses=self.responses_api)
        self.config = SimpleNamespace(default_model="gpt-5.2")

    def _build_response_request(
        self,
        prompt: str,
        *,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int | None = None,
    ) -> dict[str, object]:
        return {
            "model": model,
            "prompt": prompt,
            "instructions": instructions,
            "allow_web_search": allow_web_search,
            "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": max_output_tokens,
        }

    def _extract_output_text(self, response: object) -> str:
        return str(getattr(response, "output_text", "") or "")


class RequestStructuredJsonObjectTests(unittest.TestCase):
    def test_retries_with_higher_token_limit_after_max_output_truncation(self) -> None:
        backend = _FakeBackend(
            responses=[
                SimpleNamespace(
                    status="incomplete",
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                    output_parsed=None,
                    output_text='{"midterm_packets": [{"packet_id": "cut',
                ),
                SimpleNamespace(
                    status="completed",
                    incomplete_details=None,
                    output_parsed={"midterm_packets": []},
                    output_text="",
                ),
            ]
        )

        payload = request_structured_json_object(
            backend,
            prompt="prompt",
            instructions="instructions",
            schema_name="test_schema",
            schema={"type": "object", "properties": {"midterm_packets": {"type": "array"}}, "required": ["midterm_packets"]},
            max_output_tokens=900,
        )

        self.assertEqual(payload, {"midterm_packets": []})
        self.assertEqual(
            [call["max_output_tokens"] for call in backend.responses_api.calls],
            [900, 1800],
        )

    def test_falls_back_to_complete_json_text_when_output_parsed_is_missing(self) -> None:
        backend = _FakeBackend(
            responses=[
                SimpleNamespace(
                    status="completed",
                    incomplete_details=None,
                    output_parsed=None,
                    output_text='{"ok": true, "items": []}',
                )
            ]
        )

        payload = request_structured_json_object(
            backend,
            prompt="prompt",
            instructions="instructions",
            schema_name="test_schema",
            schema={"type": "object"},
            max_output_tokens=512,
        )

        self.assertEqual(payload, {"ok": True, "items": []})

    def test_raises_explicit_error_when_structured_output_remains_incomplete(self) -> None:
        backend = _FakeBackend(
            responses=[
                SimpleNamespace(
                    status="incomplete",
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                    output_parsed=None,
                    output_text='{"midterm_packets": [{"packet_id": "cut',
                ),
                SimpleNamespace(
                    status="incomplete",
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                    output_parsed=None,
                    output_text='{"midterm_packets": [{"packet_id": "still_cut',
                ),
                SimpleNamespace(
                    status="incomplete",
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                    output_parsed=None,
                    output_text='{"midterm_packets": [{"packet_id": "final_cut',
                ),
            ]
        )

        with self.assertRaises(ValueError) as ctx:
            request_structured_json_object(
                backend,
                prompt="prompt",
                instructions="instructions",
                schema_name="test_schema",
                schema={"type": "object"},
                max_output_tokens=512,
            )

        self.assertIn("status=incomplete", str(ctx.exception))
        self.assertIn("incomplete_reason=max_output_tokens", str(ctx.exception))
        self.assertIn(
            "attempted_output_token_limits=[512, 1024, 2048, 4096]",
            str(ctx.exception),
        )
        self.assertEqual(
            [call["max_output_tokens"] for call in backend.responses_api.calls],
            [512, 1024, 2048, 4096],
        )


if __name__ == "__main__":
    unittest.main()
