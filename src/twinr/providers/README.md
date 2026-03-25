# providers

`providers` owns Twinr's provider package root. It exposes the stable
`twinr.providers` import surface for runtime callers and assembles the active
`StreamingProviderBundle` from the configured STT, LLM, and TTS providers.

## Responsibility

`providers` owns:
- assemble the runtime provider bundle in [`factory.py`](./factory.py)
- expose package-root imports in [`__init__.py`](./__init__.py)
- define the boundary above [`deepgram/`](./deepgram/README.md), [`groq/`](./groq/README.md), and [`openai/`](./openai/README.md)
- attach the optional OpenAI streaming-transcript verifier to the runtime bundle when the fail-safe verifier path is enabled

`providers` does **not** own:
- provider-specific transport or SDK logic inside the sibling provider packages
- workflow orchestration in `src/twinr/agent/workflows`
- hardware capture, playback, or WAV handling in `src/twinr/hardware`

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Root exports and package-root imports |
| [factory.py](./factory.py) | Provider bundle assembly |
| [deepgram/](./deepgram/README.md) | Deepgram STT package |
| [groq/](./groq/README.md) | Groq text package |
| [openai/](./openai/README.md) | OpenAI provider package |
| [component.yaml](./component.yaml) | Structured package metadata |

## Usage

```python
from twinr.providers import OpenAIBackend, build_streaming_provider_bundle

backend = OpenAIBackend(config=config)
bundle = build_streaming_provider_bundle(config, support_backend=backend)

# bundle.verification_stt is the optional bounded second-pass STT verifier
# used by the streaming loop for suspicious short live transcripts.
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [deepgram](./deepgram/README.md)
- [groq](./groq/README.md)
- [openai](./openai/README.md)
