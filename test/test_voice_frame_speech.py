from __future__ import annotations

from types import SimpleNamespace

from twinr.agent.workflows.voice_frame_speech import EdgeVoiceFrameSpeechAnnotator


def test_edge_voice_frame_speech_annotator_keeps_live_room_audio_stateless() -> None:
    calls: list[dict[str, object]] = []

    def _fake_classifier(
        pcm_bytes: bytes | None,
        *,
        sample_rate: int | None,
        channels: int | None,
        stream_id: object | None = None,
        end_of_stream: bool = False,
    ) -> object:
        calls.append(
            {
                "pcm_len": 0 if pcm_bytes is None else len(pcm_bytes),
                "sample_rate": sample_rate,
                "channels": channels,
                "stream_id": stream_id,
                "end_of_stream": end_of_stream,
            }
        )
        return SimpleNamespace(speech_probability=0.42)

    annotator = EdgeVoiceFrameSpeechAnnotator(
        sample_rate=16_000,
        channels=1,
        classifier=_fake_classifier,
    )

    evidence = annotator.classify_frame(b"\x01\x00" * 1600)

    assert evidence.speech_probability == 0.42
    assert calls == [
        {
            "pcm_len": 3200,
            "sample_rate": 16_000,
            "channels": 1,
            "stream_id": None,
            "end_of_stream": False,
        }
    ]


def test_edge_voice_frame_speech_annotator_reset_is_noop_for_stateless_frames() -> None:
    annotator = EdgeVoiceFrameSpeechAnnotator(sample_rate=16_000, channels=1)

    annotator.reset()
