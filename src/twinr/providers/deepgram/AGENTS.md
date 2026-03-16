# AGENTS.md — /src/twinr/providers/deepgram

## Scope

This directory owns Deepgram-specific speech-to-text transport, payload
normalization, and bounded streaming-session behavior. Structural metadata
lives in [component.yaml](./component.yaml).

Out of scope:
- provider selection or fallback wiring in `src/twinr/providers/factory.py`
- microphone capture, playback, or WAV generation in `src/twinr/hardware/`
- LLM or tool-calling behavior in sibling provider packages

## Key files

- `__init__.py` — public export surface; treat changes as API-impacting
- `speech.py` — REST transcription and websocket session implementation
- `component.yaml` — package metadata, callers, and test map

## Invariants

- `DeepgramSpeechToTextProvider` remains the only public export from this package.
- Batch transcription must fail with explicit runtime errors; do not hide transport or decode failures behind empty transcripts.
- `transcribe_path()` must keep safe file handling: regular files only, no symlink traversal, deterministic descriptor cleanup.
- Streaming sessions must stay bounded: backpressure, finalize waits, and keepalive behavior must not become unbounded.
- Interim and endpoint callbacks must stay isolated from transport failures; callback exceptions must not tear down the stream.

## Verification

After any edit in this directory, run:

```bash
python3 -m compileall src/twinr/providers/deepgram
PYTHONPATH=src pytest test/test_deepgram_provider.py test/test_provider_factory.py -q
```

If `speech.py` changed in a way that affects streaming-session semantics, also run:

```bash
PYTHONPATH=src pytest test/test_streaming_runner.py -q
```

## Coupling

`speech.py` changes -> also check:
- `src/twinr/providers/factory.py`
- `src/twinr/agent/workflows/runner.py`
- `src/twinr/agent/workflows/realtime_runner.py`
- `src/twinr/agent/workflows/streaming_runner.py`
- `test/test_deepgram_provider.py`

`__init__.py` changes -> also check:
- `src/twinr/providers/factory.py`
- `test/test_provider_factory.py`

## Security

- Never log `DEEPGRAM_API_KEY` values or outbound authorization headers.
- Keep websocket and file-input validation strict; do not reintroduce implicit symlink/device reads or malformed URL acceptance.
- Keep outbound payload handling bounded; no unbounded queue growth or message-size disabling here.

## Output expectations

- Update docstrings when request parameters, callback behavior, or session-state semantics change.
- Keep [README.md](./README.md), [AGENTS.md](./AGENTS.md), and [component.yaml](./component.yaml) aligned when exports, callers, or verification commands change.
- Keep diffs focused on Deepgram STT behavior; broader provider-routing changes belong in `src/twinr/providers/factory.py`.
