# Twinr OpenAI Backend

Twinr currently starts with an OpenAI-first backend slice for:

- speech-to-text
- foundation model responses
- text-to-speech
- optional web search in the LLM step

## Recommended `.env` settings

```dotenv
OPENAI_MODEL=gpt-5.2
OPENAI_REASONING_EFFORT=medium
OPENAI_STT_MODEL=whisper-1
OPENAI_TTS_MODEL=gpt-4o-mini-tts
OPENAI_TTS_VOICE=marin
OPENAI_TTS_FORMAT=wav
OPENAI_TTS_INSTRUCTIONS="Speak in clear, warm, natural standard German with native German pronunciation. No English accent."
OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview
OPENAI_REALTIME_VOICE=sage
OPENAI_REALTIME_TRANSCRIPTION_MODEL=whisper-1
OPENAI_REALTIME_LANGUAGE=de
OPENAI_REALTIME_INPUT_SAMPLE_RATE=24000
OPENAI_REALTIME_INSTRUCTIONS="Speak in clear, warm, natural standard German with native German pronunciation. No English accent."
TWINR_CONVERSATION_FOLLOW_UP_ENABLED=true
TWINR_CONVERSATION_FOLLOW_UP_TIMEOUT_S=4.0
TWINR_AUDIO_BEEP_FREQUENCY_HZ=1175
TWINR_AUDIO_BEEP_DURATION_MS=180
TWINR_AUDIO_BEEP_VOLUME=0.9
TWINR_AUDIO_BEEP_SETTLE_MS=140
OPENAI_SEND_PROJECT_HEADER=false
TWINR_OPENAI_ENABLE_WEB_SEARCH=true
TWINR_OPENAI_WEB_SEARCH_CONTEXT_SIZE=medium
```

`OPENAI_SEND_PROJECT_HEADER=false` is recommended when the API key is already project-scoped (`sk-proj-...`). In that setup, sending an additional explicit project header can break audio requests even when LLM access works.

## Python entry points

- `src/twinr/providers/openai_backend.py`
- `src/twinr/__main__.py`

## Smoke examples

```bash
cd /twinr
PYTHONPATH=src python3 -m twinr --env-file /twinr/.env --openai-prompt "Reply with exactly OK."
PYTHONPATH=src python3 -m twinr --env-file /twinr/.env --openai-prompt "What happened in the world today?" --openai-web-search
PYTHONPATH=src python3 -m twinr --env-file /twinr/.env --tts-text "Hello from Twinr" --tts-output /tmp/twinr.wav
```

## Realtime voice loop

```bash
cd /twinr
source .venv/bin/activate
twinr --env-file /twinr/.env --run-realtime-loop
```

This path uses the Realtime API for direct audio input/output. It is the low-latency voice mode. The current implementation does not call the OpenAI `web_search` tool from inside the Realtime session.
When follow-up mode is enabled, Twinr beeps before listening and automatically opens a short follow-up listening window after each answer.
