# Twinr Migration Plan

## Goal

Replace the current OpenAI Realtime voice path with a cheaper provider mix **without losing Twinr's existing tool capabilities**.

For the latency-first production plan that prioritizes first spoken word, reliable tool calls, German quality, and Pi stability, see `docs/LATENCY_PLAN.md`.

The migration target is:

- local VAD
- streaming or fast-turn STT
- text-first LLM agent loop with tool calls
- streamed TTS output

This keeps the current Twinr product model intact:

- green button starts a voice turn
- yellow button prints the latest answer
- reminders, automations, memory, search, camera inspection, settings changes, and profile tools continue to work

## Current State

Twinr currently uses:

- realtime voice session: `src/twinr/providers/openai/realtime.py`
- realtime hardware loop: `src/twinr/agent/workflows/realtime_runner.py`
- shared tool registry/schemas: `src/twinr/agent/tools/`
- text/search/print backend: `src/twinr/providers/openai/backend.py`

Today the expensive part is the always-audio-native OpenAI Realtime path configured via:

- `.env` → `OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview`

This path is fast and convenient, but it couples:

- audio transport
- STT
- LLM generation
- TTS generation
- tool-call orchestration

more tightly than Twinr needs.

## Recommended Target Stack

### Preferred target

- `Silero VAD` local
- `Deepgram Nova-3` streaming STT for German
- `Groq` streaming LLM
- `OpenAI gpt-4o-mini-tts` for final spoken voice

### Why this target

- removes the expensive OpenAI Realtime audio loop
- keeps a strong German voice
- preserves fast text/tool orchestration
- keeps the architecture provider-agnostic
- lets search remain separate from the speech stack

### Cost/quality note

If absolute cost minimization becomes more important than STT interactivity, the cheaper fallback is:

- `Silero VAD` local
- `Groq Whisper` after end-of-turn
- `Groq` streaming LLM
- `OpenAI gpt-4o-mini-tts`

That is cheaper than Deepgram streaming STT, but less conversational during the listening phase.

## Non-Goals

This migration should **not** change:

- green/yellow button semantics
- print flow
- reminder flow
- automation flow
- durable memory flow
- web UI operator model
- current display state model

This is a transport/provider migration, not a product redesign.

## Core Architectural Decision

The key separation is:

1. `Audio transport and turn detection`
2. `STT`
3. `LLM + tool loop`
4. `TTS`
5. `hardware playback`

Tool calls must stay attached to the **text agent loop**, not to a provider-specific speech session.

That means Twinr should treat:

- `speech-to-text`
- `LLM/tool orchestration`
- `text-to-speech`

as separate capabilities behind explicit interfaces.

## Target Runtime Shape

### Audio and turn layer

Responsibilities:

- button-triggered listen window
- VAD-based start/end-of-speech detection
- adaptive pause handling
- voice-activation/proactive gating
- microphone capture and playback locking

This remains under:

- `src/twinr/hardware/`
- `src/twinr/agent/workflows/`

### STT layer

Responsibilities:

- accept captured audio
- return:
  - partial transcript events when supported
  - final transcript
  - language / confidence metadata if available

Recommended locations:

- `src/twinr/provider/deepgram/`
- `src/twinr/provider/groq/`

### LLM agent layer

Responsibilities:

- consume transcript text
- load memory/personality/tool context
- stream response text
- emit tool calls
- receive tool results
- continue generation after tool execution

This should become the canonical home of Twinr intelligence.

Recommended locations:

- `src/twinr/agent/tools/`
- `src/twinr/agent/base_agent/`
- `src/twinr/provider/groq/`

### TTS layer

Responsibilities:

- accept generated text chunks
- return playable audio chunks
- stay independent from tool logic

Recommended locations:

- `src/twinr/provider/openai/` for `gpt-4o-mini-tts`
- optional fallback `src/twinr/provider/piper/`

## Tool Capability Preservation

Twinr must keep the current tool surface.

At minimum, the migrated path must preserve:

- `print_receipt`
- `search_live_info`
- `inspect_camera`
- `end_conversation`
- `schedule_reminder`
- `list_automations`
- `create_time_automation`
- `create_sensor_automation`
- `update_time_automation`
- `update_sensor_automation`
- `delete_automation`
- `remember_memory`
- `remember_contact`
- `lookup_contact`
- `remember_preference`
- `remember_plan`
- `update_user_profile`
- `update_personality`
- `update_simple_setting`
- `enroll_voice_profile`
- `get_voice_profile_status`
- `reset_voice_profile`

These tools are not conceptually tied to OpenAI Realtime. They are Twinr capabilities and should live behind a provider-independent executor.

## Search Strategy

Search does **not** need to be migrated at the same time as speech.

The lowest-risk path is:

- remove OpenAI Realtime for live audio
- keep the existing OpenAI web-search-backed implementation for `search_live_info`

That means Twinr can still use:

- `src/twinr/providers/openai/backend.py`

for:

- `web_search`
- print composition
- reminder phrasing
- proactive phrasing

while the main voice turn loop moves away from OpenAI Realtime.

This keeps scope bounded and avoids a large simultaneous provider rewrite.

## Recommended Repo Structure

Use the existing repo layout and extend it conservatively.

### Keep

- `src/twinr/agent/tools/`
- `src/twinr/agent/workflows/`
- `src/twinr/provider/openai/`
- `src/twinr/providers/openai/`
- `src/twinr/hardware/`

### Add

- `src/twinr/provider/deepgram/`
  - streaming STT implementation
- `src/twinr/provider/groq/`
  - LLM streaming implementation
  - optional fast-turn Whisper implementation
- `src/twinr/provider/piper/`
  - optional local/offline TTS fallback

### Refactor

- move the remaining tool handler execution out of `src/twinr/agent/workflows/realtime_runner.py`
- keep `src/twinr/agent/tools/runtime/registry.py` as the canonical tool registry
- add an executor under `src/twinr/agent/tools/runtime/` that is reusable from non-Realtime paths

Recommended new agent-side pieces:

- `src/twinr/agent/tools/runtime/executor.py`
- `src/twinr/agent/tools/handlers/`
  - `memory.py`
  - `printing.py`
  - `search.py`
  - `reminders.py`
  - `automations.py`
  - `camera.py`
  - `settings.py`
  - `voice_profile.py`

This keeps `realtime_runner.py` and the replacement voice runner focused on orchestration rather than business logic.

## Execution Model

The replacement loop should work like this:

1. user presses green
2. local VAD opens the capture window
3. STT provider streams or finalizes transcript text
4. Twinr starts a text-first LLM turn
5. LLM may emit tool calls
6. Twinr executes tool calls through the shared tool executor
7. LLM continues and streams answer text
8. text is chunked at sentence or clause boundaries
9. TTS provider synthesizes each chunk
10. audio is played immediately

This preserves fast feedback while avoiding provider-coupled speech sessions.

## Text Streaming Rule

Do **not** send every token directly into TTS.

Use a small text chunker that waits for:

- sentence endings
- strong punctuation
- or a bounded clause window

before sending chunks to TTS.

This is important because:

- it keeps prosody natural
- it avoids choppy audio
- it still gives low perceived latency

## Migration Phases

### Phase 1 — Finish separation of concerns

Goal:

- make tool execution provider-independent

Work:

- finish extracting tool handlers from `src/twinr/agent/workflows/realtime_runner.py`
- keep `src/twinr/agent/tools/` as the canonical registry/executor layer

Acceptance:

- current OpenAI Realtime path still works unchanged
- targeted tests pass locally
- same path passes on the Pi

### Phase 2 — Add explicit provider interfaces

Goal:

- define provider contracts for:
  - STT
  - LLM streaming/tool loop
  - TTS streaming

Work:

- add minimal provider protocols / contracts
- add adapters for:
  - Deepgram STT
  - Groq LLM
  - OpenAI TTS

Acceptance:

- contract-level tests pass locally
- import path works on the Pi

### Phase 3 — Build a new streaming voice runner

Goal:

- create a non-Realtime main voice loop

Work:

- add a new workflow runner under `src/twinr/agent/workflows/`
- reuse:
  - buttons
  - recorder
  - adaptive timing
  - playback
  - runtime
  - display updates

Do not replace the current Realtime loop immediately.

Acceptance:

- new runner can complete a green-button turn on the Pi
- print, reminder, memory, and search tools still work

### Phase 4 — Feature-flagged deployment

Goal:

- switch by config, not by branch

Recommended env shape:

- `TWINR_VOICE_RUNTIME=openai_realtime|streaming`
- `TWINR_STT_PROVIDER=openai|deepgram|groq`
- `TWINR_LLM_PROVIDER=openai|groq`
- `TWINR_TTS_PROVIDER=openai|piper`

Acceptance:

- same repo can boot both paths
- Pi can switch between them cleanly

### Phase 5 — Cut over on the Pi

Goal:

- make the new streaming path the default runtime

Acceptance:

- Pi-first validation passes for the full assistant
- OpenAI Realtime remains available only as an emergency fallback until confidence is high

## Pi Acceptance Checklist

The migration is not done until the Pi proves all of this:

### Core voice flow

- green button starts listening
- slow speech still works
- short mid-question pauses do not break the turn
- answer starts quickly and remains understandable
- `Danke`, `Pause`, `Stop` still end the follow-up loop

### Tool capabilities

- print via yellow button
- print via explicit spoken request
- search via spoken request
- reminder creation and due delivery
- automation create/list/update/delete
- durable memory writes
- durable memory recall
- profile/personality updates
- camera inspection tool
- simple bounded settings changes

### Proactive behavior

- proactive prompts still pause while an active conversation is running
- reminders and automations do not break the active conversation state model

### Operator surface

- web UI still shows memory, reminders, automations, and settings correctly
- display loop still reflects runtime state

## Risks

### Risk 1 — Streaming STT language mismatch

Deepgram `Flux` is not the right German default.

Mitigation:

- use a German-capable streaming STT model such as `Nova-3`
- keep provider choice explicit in config

### Risk 2 — TTS latency feels worse than Piper

OpenAI TTS will add network RTT and synthesis time.

Mitigation:

- chunk text at sentence/clause boundaries
- start TTS as soon as the first chunk is stable
- keep Piper as optional fallback

### Risk 3 — Tool loop regressions

The main migration risk is not audio. It is losing tool reliability while moving provider boundaries.

Mitigation:

- extract the tool executor first
- keep the tool surface unchanged
- test tool behavior directly before changing the voice path

### Risk 4 — Scope explosion

Trying to migrate STT, LLM, TTS, search, and all tools at once will slow delivery and increase instability.

Mitigation:

- keep OpenAI search initially
- migrate voice transport first
- replace search later only if there is a clear product reason

## Recommended First Increment

The safest first implementation target is:

1. finish tool executor extraction
2. add provider interfaces
3. add `Deepgram STT + Groq LLM + OpenAI TTS`
4. build a new `streaming` voice runner behind a feature flag
5. validate full tool parity on the Pi

Do **not** start by deleting `OpenAIRealtimeSession`.

First make the cheaper path real and Pi-validated. Only then demote the old path.

## Recommendation Summary

For Twinr, the best near-term migration target is:

- `Silero VAD` local
- `Deepgram Nova-3` streaming STT
- `Groq` streaming LLM
- `OpenAI gpt-4o-mini-tts`
- existing OpenAI web-search path kept for now

This gives the best balance of:

- lower cost than OpenAI Realtime
- much better voice quality than Piper-only
- preserved tool capabilities
- lower architectural coupling
- easier debugging and provider replacement later

## External References

These are the main external references behind this migration plan:

- OpenAI pricing: `https://openai.com/api/pricing/`
- OpenAI pricing docs: `https://developers.openai.com/api/docs/pricing`
- OpenAI TTS guide: `https://platform.openai.com/docs/guides/text-to-speech/quick-start.pls`
- Groq pricing: `https://groq.com/pricing`
- Groq speech-to-text docs: `https://console.groq.com/docs/speech-to-text`
- Deepgram pricing: `https://deepgram.com/pricing`
- Deepgram language guidance: `https://developers.deepgram.com/docs/language-detection`
- Silero VAD: `https://github.com/snakers4/silero-vad`
- Piper TTS: `https://github.com/rhasspy/piper`
