# Twinr OpenAI Backend

Twinr currently starts with an OpenAI-first backend slice for:

- speech-to-text
- foundation model responses
- text-to-speech
- optional web search in the LLM step

## Recommended `.env` settings

```dotenv
OPENAI_MODEL=gpt-5.4-mini
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
OPENAI_VISION_DETAIL=auto
TWINR_CAMERA_DEVICE=/dev/video0
TWINR_CAMERA_WIDTH=640
TWINR_CAMERA_HEIGHT=480
TWINR_CAMERA_FRAMERATE=30
TWINR_CAMERA_INPUT_FORMAT=bayer_grbg8
TWINR_VISION_REFERENCE_IMAGE=/home/thh/reference-user.jpg
```

`OPENAI_SEND_PROJECT_HEADER=false` is recommended when the API key is already project-scoped (`sk-proj-...`). In that setup, sending an additional explicit project header can break audio requests even when LLM access works.

`OPENAI_MODEL` is Twinr's central non-coding text model. Search, supervisor routing, first-word replies, conversation closure, long-term-memory text compilation, and display reserve generation inherit it unless you set a dedicated override. Audio-specialized paths remain separate on purpose: `OPENAI_STT_MODEL`, `OPENAI_TTS_MODEL`, `OPENAI_REALTIME_MODEL`, `OPENAI_REALTIME_TRANSCRIPTION_MODEL`, and `TWINR_STREAMING_TRANSCRIPT_VERIFIER_MODEL`.

## Python entry points

- `src/twinr/providers/openai/backend.py`
- `src/twinr/__main__.py`

## Smoke examples

```bash
cd /twinr
PYTHONPATH=src ./.venv/bin/python -m twinr --env-file /twinr/.env --openai-prompt "Reply with exactly OK."
PYTHONPATH=src ./.venv/bin/python -m twinr --env-file /twinr/.env --openai-prompt "What happened in the world today?" --openai-web-search
PYTHONPATH=src ./.venv/bin/python -m twinr --env-file /twinr/.env --tts-text "Hello from Twinr" --tts-output /tmp/twinr.wav
PYTHONPATH=src ./.venv/bin/python -m twinr --env-file /twinr/.env --camera-capture-output /tmp/twinr-camera.png
PYTHONPATH=src ./.venv/bin/python -m twinr --env-file /twinr/.env --vision-prompt "Bild 1 ist das Live-Kamerabild. Bild 2 ist das Referenzfoto des Nutzers. Ist es dieselbe Person? Antworte nur mit JA oder NEIN." --vision-camera-capture --vision-save-capture /tmp/twinr-camera.png --vision-image /home/thh/reference-user.jpg
```

The vision flow can attach multiple images in one OpenAI Responses API request. Twinr labels the live camera frame separately from any user-supplied reference images so the model can compare them.
The active realtime and streaming loops can trigger this automatically for common visual requests and will include the configured reference image when a stored user portrait path is available.
