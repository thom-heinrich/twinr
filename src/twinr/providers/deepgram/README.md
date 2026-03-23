# deepgram

`deepgram` owns Twinr's Deepgram speech-to-text integration. It exposes the
provider that handles batch transcription and bounded live-stream sessions
against Deepgram's REST and websocket endpoints.

## Responsibility

`deepgram` owns:
- validate and send Deepgram REST transcription requests
- manage bounded websocket streaming sessions and endpoint callbacks
- normalize Deepgram payloads into Twinr STT contracts

Streaming note:
- `speech_final` endpoint callbacks are only early stop hints for Twinr's turn controller.
- `session.finalize()` still waits for a real final Deepgram segment before returning, unless the bounded finalize timeout expires first.

`deepgram` does **not** own:
- provider selection or fallback orchestration
- microphone capture, playback, or WAV generation
- LLM, tool-calling, or live-search behavior

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Public package export |
| [speech.py](./speech.py) | REST and websocket STT implementation |
| [component.yaml](./component.yaml) | Structured package metadata |
| [AGENTS.md](./AGENTS.md) | Local editing invariants |

## Usage

```python
from twinr.providers.deepgram import DeepgramSpeechToTextProvider

provider = DeepgramSpeechToTextProvider(config)
transcript = provider.transcribe(audio_bytes, content_type="audio/wav")
```

```python
session = provider.start_streaming_session(sample_rate=16000, channels=1)
session.send_pcm(pcm_chunk)
result = session.finalize()
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [provider factory](../factory.py)
- [hardware audio capture](../../hardware/audio.py)
