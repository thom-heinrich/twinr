# Twinr Low-Latency Voice Plan

## Objective

Build the next Twinr voice runtime for:

- minimal time to first spoken word
- reliable multi-turn tool calls
- strong German performance
- maximum Raspberry Pi stability

This plan is intentionally conservative. It optimizes latency **without** giving up the current Twinr capability surface or replacing stable parts just because they are theoretically faster.

## Product Constraints

These are non-negotiable:

- no hardcoded intent routing
- tools remain model-driven
- green button / yellow button semantics stay unchanged
- search, memory, reminders, automations, camera, printing, settings, and profile tools must keep working
- the Pi remains the runtime acceptance environment

## Decision

### Recommended production target

- local wake/beep + local audio device control
- local VAD / endpointing assistance
- `Deepgram Nova-3` streaming STT
- `OpenAI` text/tool loop
- low-latency streamed TTS

### Recommended order of change

1. replace STT first
2. optimize tool/text loop latency second
3. optimize TTS third
4. only then consider a runtime cutover

### What not to do now

- do not switch to a research speech-to-speech model
- do not switch tool calling to a less-proven provider just for token speed
- do not move multiple major layers at once

The current evidence says: **tool reliability is more valuable than raw model TTFT** for Twinr.

## Architecture Principles

### 1. Keep the agent loop text-first

Twinr's core intelligence should stay:

- transcript in
- tool loop over text
- response text out
- TTS only after text decisions

Reason:

- reliable tool calling
- easier debugging
- easier persistence and replay
- provider independence

### 2. Treat latency as a pipeline problem

The main latency budget is not one component. It is:

1. end-of-turn detection
2. STT finalization
3. tool/text loop
4. first TTS chunk
5. playback start

We should reduce all five, not chase a single model benchmark.

### 3. Separate production lane from experiment lane

- `current_realtime` remains the safety baseline
- `streaming_low_latency` is the optimization lane
- cutover happens only after Pi acceptance gates pass

## Target Runtime

## Stage A — Fast and stable listening

### A1. Streaming STT

Use `Deepgram Nova-3` streaming with:

- `interim_results=true`
- explicit German-capable model selection
- endpointing configured for spoken conversations, not dictation

Initial tuning target:

- endpointing window: `300–500 ms`
- final tuning based on Pi spoken tests, not laptop tests

### A2. Earlier turn-finalization without brittle cutoffs

Twinr should not wait for a large fixed silence timeout.

Instead:

- use STT partials + endpointing
- keep the existing pause-grace logic
- finalize only when both are plausible:
  - silence/endpointer says likely done
  - partial text has stabilized enough

This keeps slower senior speech usable without dragging every fast turn.

### A3. Pre-open the network path on button press

As soon as green is pressed:

- audio device is already owned
- STT websocket opens immediately
- no lazy connection after the user has already finished speaking

This is a cheap latency win and low risk.

## Stage B — Reliable low-latency tool loop

### B1. Keep OpenAI as the canonical tool/text lane

For the next production lane:

- keep `OpenAI` for tool calls
- keep the current provider-neutral tool executor
- do not reintroduce provider-specific speech-session tool logic

Reason:

- this is the current strongest evidence-backed path for Twinr's 24-tool surface
- it already passed the Pi matrix

### B2. Use prompt caching aggressively

Keep the prompt prefix stable so caching hits:

- system instructions
- personality base text
- tool schemas
- invariant style guidance

Dynamic items should be appended after the stable prefix:

- current transcript
- recent raw tail
- small tool-specific context
- active reminders/automations summaries

Target outcome:

- the expensive static prefix is cached
- per-turn work only pays for the dynamic tail

### B3. Keep tool-loop context narrow

Do not bloat the tool loop with broad hidden context.

The rule is:

- exact persisted facts that have an explicit tool must stay behind the tool
- hidden context may carry tone, subtext, and light personalization
- hidden context must not become a backdoor that suppresses explicit tools

This is now a proven Twinr failure mode and should stay fixed.

### B4. Preserve model-driven tool choice

Allowed:

- better instructions
- narrower context
- retries on malformed tool outputs
- better schemas

Not allowed:

- keyword routing
- Alexa-style hardcoded intent maps
- scripted forced tool triggers

## Stage C — Faster first spoken word

### C1. TTS must be truly streamed

The next latency win must come from the TTS side.

Twinr should:

- request audio as a stream, not as a completed file
- start playback on the first stable chunk
- avoid buffering the whole utterance before sound starts

### C2. Chunk earlier than full sentences

Do not wait for perfect sentence boundaries.

Use chunk boundaries such as:

- strong punctuation
- clause endings
- stable short acknowledgements

Examples of acceptable early chunks:

- `Ich schaue kurz nach.`
- `Einen Moment bitte.`
- `Morgen in Schwarzenbek ...`

This must still be model-driven text, not a scripted speech shell.

### C3. Keep German voice quality high

Primary acceptance criteria for TTS:

- clear standard German pronunciation
- no obvious English accent
- calm, trustworthy voice quality
- no robotic clipping on short streamed chunks

Primary path:

- keep current high-quality OpenAI voice lane while optimizing streaming behavior

Secondary evaluation track:

- benchmark `Deepgram Aura-2` German streaming TTS as an alternative

Do **not** switch to the fastest voice if it sounds synthetic or unstable.

## Stage D — Search-specific optimization

Search remains the slowest user-visible path. Treat it separately.

### D1. Search worker stays separate

Keep:

- a dedicated search worker
- smaller search prompt
- exact date/location grounding

Do not drag the full conversation context into search requests.

### D2. Search should acknowledge quickly

Search turns should start with a short first clause quickly, then continue with the result.

Acceptable pattern:

- `Ich schaue kurz online nach.`
- then the real answer once search returns

This is still model-generated. The system should only make this style easy and natural.

### D3. Optimize search only after basic turns are fast

Do not let search tuning block the full migration.

The order should be:

1. simple turns
2. memory/reminder/print turns
3. search turns

## Stage E — Runtime stability

### E1. One active production loop

The Pi must not run overlapping competing voice loops.

Rules:

- one production voice loop at a time
- bounded smoke loops for experiments
- explicit restore to baseline after experiments

### E2. Keep hard fallback path

Until cutover is complete:

- `current_realtime` remains the fallback lane
- switching back must be one config change, not a refactor

### E3. Avoid “shadow complexity”

Do not add:

- duplicate memory paths
- duplicate automation semantics
- provider-specific tool contracts

Latency work must reduce delay, not increase operational complexity.

## Acceptance Gates

The new lane is cutover-ready only when **all** of these pass on the Pi.

### Functional gates

- full tool matrix: `24/24` pass
- no hardcoded tool routing
- multi-turn memory and persistence stay correct
- reminders, printing, automations, and camera remain bounded and stable

### Spoken UX gates

- no obvious cutoffs for slower senior speech
- no repeated false “nothing heard” failures during normal pauses
- clean recovery after silence, printer, and reminder playback

### Latency gates

Suggested cutover targets:

- simple turn: `<= 3500 ms` to first spoken word
- memory/reminder/print turn: `<= 4500 ms`
- search turn: `<= 7000 ms` to first spoken word

And:

- median total-turn time must not regress materially against the current production lane
- if first-word latency improves but completion gets much worse, cutover is not accepted

### Stability gates

- bounded 10–15 turn spoken soak test on the Pi
- no loop crashes
- no audio-device lock failures
- no tool-discipline regressions after restart

## Implementation Phases

### Phase 1 — STT cut

Deliver:

- `Deepgram Nova-3` streaming STT in the live streaming lane
- interim results + endpointing
- green-button connection prewarm

Acceptance:

- spoken Pi tests show fewer long waits after user stop
- German transcript quality is at least as good as current path

### Phase 2 — tool/text latency cut

Deliver:

- prompt-cached OpenAI tool loop
- stable cached prefix
- narrow dynamic context

Acceptance:

- no tool regressions
- measurable drop in tool/text latency on Pi benchmarks

### Phase 3 — TTS first-word cut

Deliver:

- true streamed TTS output
- clause-level chunking
- playback starts on first stable chunk

Acceptance:

- first spoken word latency drops meaningfully on Pi
- no noticeable German voice degradation

### Phase 4 — spoken-loop acceptance

Deliver:

- real Pi spoken benchmark
- real button-driven acceptance
- cost/latency comparison against current production lane

Acceptance:

- cutover only if latency and stability targets both pass

## Recommended Cutover Rule

Use this exact rule:

- if the streaming lane is only cheaper but still noticeably slower, **do not cut over**
- if the streaming lane is cheaper **and** meets the spoken-latency gates on the Pi, cut over
- if search remains the only slow case, cut over only if search is clearly announced and still acceptable in practice

## Final Recommendation

The next best Twinr path is:

1. `Deepgram Nova-3` streaming STT
2. `OpenAI` text/tool loop with prompt caching
3. streamed high-quality TTS
4. Pi-first spoken acceptance

This path best fits the actual goal:

- minimal first-word latency
- reliable tool calls
- good German behavior
- maximum operational stability

It is the lowest-risk path that still has a realistic chance of beating the current experience where it matters.
